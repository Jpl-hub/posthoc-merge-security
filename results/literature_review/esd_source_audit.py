"""Check ESD's local error identity on an existing cache; no model forward."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_grad_enabled(False)
out = Path("outputs/esd_source_audit")
protocol = json.loads((out / "protocol.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (out / "results.json").exists(), "Refuse to overwrite a completed audit"
source = Path(protocol["source_file"])
capsule = Path(protocol["input_capsule"])
assert sha(source) == protocol["source_sha256"]
assert sha(capsule) == protocol["input_capsule_sha256"]
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_compute_esd")
module = ast.Module(body=[function], type_ignores=[])
env = {"torch": torch}
exec(compile(module, str(source), "exec"), env)

cache = torch.load(capsule, map_location="cpu")
x = cache["native"].reshape(-1, 768).double()
update = cache["q"].double() - cache["q0"].double()
y = x @ update.T
n, d = y.shape
assert n > d, "Do not enter the source's short-sample basis-padding branch"
eigenvalues, basis = env["_compute_esd"](y)
assert eigenvalues.device.type == "cpu" and basis.device.type == "cpu"

# Independent NumPy input-response product and symmetric-eigenvalue calculation.
yn = x.numpy() @ (cache["q"].double().numpy() - cache["q0"].double().numpy()).T
np.testing.assert_allclose(yn, y.numpy(), rtol=1e-11, atol=1e-11)
gram = yn.T @ yn / (n - 1)
values_np = np.linalg.eigvalsh(gram)[::-1].copy()
np.testing.assert_allclose(values_np, eigenvalues.numpy(), rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(basis.T.numpy() @ basis.numpy(), np.eye(d),
                           rtol=0, atol=1e-12)

# Check every orthogonal component as an algebra identity, not a rank search.
component_mse = np.mean((yn @ basis.numpy()) ** 2, axis=0)
predicted_mse = eigenvalues.numpy() * (n - 1) / n
np.testing.assert_allclose(component_mse, predicted_mse, rtol=1e-10, atol=1e-12)
mean = yn.mean(axis=0)
centered = yn - mean
total_mse = float(np.mean(np.sum(yn * yn, axis=1)))
centered_mse = float(np.mean(np.sum(centered * centered, axis=1)))
mean_energy = float(mean @ mean)
np.testing.assert_allclose(total_mse, centered_mse + mean_energy, rtol=1e-12)

result = {
    "protocol": protocol,
    "source_function_lines": [function.lineno, function.end_lineno],
    "script_sha256": sha(__file__),
    "protocol_sha256": sha(out / "protocol.json"),
    "samples": n,
    "output_dimension": d,
    "raw_local_shift_mean_squared_norm": total_mse,
    "centered_local_shift_mean_squared_norm": centered_mse,
    "local_shift_mean_squared_norm": mean_energy,
    "mean_fraction_of_raw_energy": mean_energy / total_mse,
    "source_eigenvalue_sum": float(eigenvalues.sum()),
    "source_to_empirical_mse_factor": (n - 1) / n,
    "max_component_identity_abs_error": float(np.max(np.abs(component_mse - predicted_mse))),
    "independent_NumPy_product_and_eigenvalues": "passed",
    "mean_covariance_decomposition": "passed",
    "new_model_forwards": 0,
    "new_protected_weights": 0,
    "new_classification_results": 0,
    "candidate_rank_or_strength_search": False,
    "closed_Query_support_reopened": False,
    "scope": "ESD source/error-identity audit only; not a defense or full ESM reproduction",
    "completed": True,
}
(out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
