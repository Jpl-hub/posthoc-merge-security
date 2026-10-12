"""Independent CPU covariance-factor audit of existing middle-layer records."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

sys.path.insert(0, str(Path("experiments").resolve()))
from clip_benchmark import checkpoint_loader

torch.set_grad_enabled(False)
torch.set_num_threads(4)
output = Path("outputs/complement_update_audit")
record = json.loads((output / "results.json").read_text())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


for path, sha in record["provenance"]["input_sha256"].items():
    assert digest(path) == sha, path
assert digest(output.parent / "complement_update_audit.py") == record["provenance"]["script_sha256"]
root = Path("checkpoints/ViT-B-32")
for name, sha in record["provenance"]["checkpoint_sha256"].items():
    assert digest(root / name) == sha, name
load = checkpoint_loader(Path("external/secure-merge-attack"))
anchor = load(str(root / "zeroshot.pt")).cpu().eval()
expert = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
layer = expert.model.visual.transformer.resblocks[5].mlp.c_fc
w = layer.weight.double().numpy()
b = layer.bias.double().numpy()
w0 = anchor.model.visual.transformer.resblocks[5].mlp.c_fc.weight.double().numpy()
source = Path("outputs/middle_completion/artifacts")
moments = torch.load(source / "diagnostic_moments.pt", map_location="cpu")
factors = {}
for context, moment in moments.items():
    sigma = moment["covariance"].numpy()
    assert np.max(np.abs(sigma - sigma.T)) < 1e-10
    eigenvalues, vectors = eigh(sigma, driver="evd")
    assert eigenvalues.min() > -1e-10 * max(1, eigenvalues.max())
    factors[context] = vectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]


def factor_rms(matrix, factor, mean):
    return float(np.sqrt(np.square(matrix @ factor).sum() + np.square(mean).sum()))


checks = []
for name in ("defender", "attacker"):
    saved = torch.load(source / f"{name}.pt", map_location="cpu")
    u = saved["basis"]["u"].numpy()
    mu = saved["basis"]["mean"].numpy()
    c = np.eye(768) - u @ u.T
    matrices = {"generic_anchor_complement_rms": w0 @ c,
                "original_expert_complement_rms": w @ c,
                "task_update_complement_rms": (w - w0) @ c}
    actual_dw = saved["completed"]["weight"].double().numpy() - w
    actual_db = saved["completed"]["bias"].double().numpy() - b
    for context, moment in moments.items():
        row = next(r for r in record["responses"] if r["basis"] == name and r["context"] == context)
        mean = moment["mean"].numpy()
        values = {key: factor_rms(matrix, factors[context], matrix @ (mean - mu))
                  for key, matrix in matrices.items()}
        scale = 1.0 if context.endswith("/native") else 0.5
        values["actual_completion_preactivation_rms"] = scale * factor_rms(
            actual_dw, factors[context], actual_dw @ mean + actual_db)
        values["mathematical_completion_preactivation_rms"] = scale * values["task_update_complement_rms"]
        values["task_update_over_anchor_response"] = values["task_update_complement_rms"] / values["generic_anchor_complement_rms"]
        relative_errors = {key: abs(value - row[key]) / max(1e-12, abs(value))
                           for key, value in values.items()}
        assert max(relative_errors.values()) < 1e-8, (name, context, relative_errors)
        checks.append(dict(basis=name, context=context, maximum_relative_error=max(relative_errors.values())))
assert len(checks) == 8
validation = dict(passed=True, method="NumPy/SciPy covariance square-root factors and affine mean responses; independent of source Gram trace implementation.",
                  input_fingerprints_verified=True, checkpoint_fingerprints_verified=True,
                  result_sha256=digest(output / "results.json"), verifier_sha256=digest(__file__),
                  device="CPU only", new_model_forwards=0, new_protected_weights=0,
                  checks=checks)
(output / "independent_validation.json").write_text(json.dumps(validation, indent=2))
print(json.dumps(validation), flush=True)
