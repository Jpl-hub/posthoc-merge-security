"""Check a pinned NUFILT gate/fold relation on fixed CPU toy matrices.

Only isolated gate and folding methods execute; no optimizer, CLIP or dataset.
Usage on the existing server: python nufilt_fold_audit.py SOURCE_ROOT OUTPUT
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PIN = "4cd298fb7ea22bd4bcd343748fb4b8e3f9815324"
HASHES = {
    "fusion_bench/models/filter_lora.py":
        "1d0897cb65c6bbc10520dafb8294be1d281034d812fa7be6374bf9babd7828ad",
    "fusion_bench/method/nufilt/nufilt.py":
        "68381a159aa37b538f77225aa68f497272acb9f8d59f16600e4a494d7176af75",
}
root, output = map(Path, sys.argv[1:])
for name, expected in HASHES.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
parsed = ast.parse((root / next(iter(HASHES))).read_text())
classes = []
for node in parsed.body:
    if isinstance(node, ast.ClassDef) and node.name == "ResidualProjGate":
        classes.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "FilterLoRA":
        node.body = [f for f in node.body if isinstance(f, ast.FunctionDef)
                     and f.name in ("forward", "merge_to_base")]
        classes.append(node)
assert len(classes) == 2
namespace = {"torch": torch, "nn": nn, "Tensor": torch.Tensor}
exec(compile(ast.Module(body=classes, type_ignores=[]), "pinned_gate", "exec"),
     namespace)

X = np.array([[1, 0, 2], [0, 1, 3], [-1, 2, 1]], dtype=float)
T = np.array([[1, 2, 0], [0, -1, 1]], dtype=float)
W0 = np.array([[2, 0, 1], [1, 1, -1]], dtype=float)
b0 = np.array([1, -2], dtype=float)
U = np.array([[0], [1], [0]], dtype=float)
P = np.eye(3) - U @ U.T
A = np.array([[0, 1, 3]], dtype=float)
cases = {
    "zero_B": np.zeros((3, 1)),
    "symmetric_BA": A.T.copy(),
    "nonsymmetric_BA": np.array([[1], [2], [0]], dtype=float),
}
records = {}
with torch.no_grad():
    for name, B in cases.items():
        gate = namespace["ResidualProjGate"](d=3, r=1, U=torch.tensor(U)).double()
        gate.A.copy_(torch.tensor(A))
        gate.B.copy_(torch.tensor(B))
        module = namespace["FilterLoRA"].__new__(namespace["FilterLoRA"])
        nn.Module.__init__(module)
        module.base_model = nn.Linear(3, 2, bias=True).double()
        module.task_tv = nn.Linear(3, 2, bias=False).double()
        module.base_model.weight.copy_(torch.tensor(W0))
        module.base_model.bias.copy_(torch.tensor(b0))
        module.task_tv.weight.copy_(torch.tensor(T))
        module.gate = gate
        forward = module(torch.tensor(X)).numpy()
        folded_weight = module.merge_to_base().numpy()
        folded = X @ folded_weight.T + b0
        row_gate = P + B @ A
        independent_forward = X @ W0.T + b0 + X @ row_gate @ T.T
        independent_folded = X @ (W0 + T @ row_gate).T + b0
        predicted_gap = X @ (B @ A - (B @ A).T) @ T.T
        np.testing.assert_array_equal(forward, independent_forward)
        np.testing.assert_array_equal(folded, independent_folded)
        np.testing.assert_array_equal(forward - folded, predicted_gap)
        records[name] = {"forward": forward.tolist(), "folded": folded.tolist(),
                         "max_absolute_gap": float(np.abs(predicted_gap).max())}
assert records["zero_B"]["max_absolute_gap"] == 0
assert records["symmetric_BA"]["max_absolute_gap"] == 0
assert records["nonsymmetric_BA"]["max_absolute_gap"] > 0
result = {"status": "passed", "commit": PIN, "source_sha256": HASHES,
          "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "device": "CPU", "torch": torch.__version__, "numpy": np.__version__,
          "X": X.tolist(), "task_weight": T.tolist(), "base_weight": W0.tolist(),
          "base_bias": b0.tolist(), "U": U.tolist(), "A": A.tolist(),
          "B_cases": {k: v.tolist() for k, v in cases.items()}, "cases": records,
          "scope": "Isolated methods and NumPy algebra; not NUFILT reproduction",
          "optimizer_run": False, "new_CLIP_forwards": 0,
          "new_protected_weights": 0, "new_classification_results": 0}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "gaps": {
    k: v["max_absolute_gap"] for k, v in records.items()}}))
