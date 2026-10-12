"""Independent NumPy SVD / direct-cleaned-query check on server-only inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.special import softmax
import torch

root = Path("outputs/key_contrast_audit")
result = json.loads((root / "results.json").read_text())
data = {k: v.numpy().astype(np.float64) for k, v in
        torch.load(root / "verification_inputs.pt", map_location="cpu").items()}
x = data["native"]
keys = x @ data["k"].T + data["bk"]
contrasts = keys - keys.mean(axis=1, keepdims=True)
flat = contrasts.reshape(-1, 64)
# SVD of the sample matrix, not eigendecomposition of its covariance.
_, singular, vh = scipy.linalg.svd(flat, full_matrices=False, lapack_driver="gesdd")
eigenvalues = singular**2 / len(flat)
assert np.allclose(eigenvalues[::-1], result["eigenvalues_ascending"], rtol=2e-7, atol=1e-14)
rank = int((eigenvalues > result["exact_rank_tolerance"]).sum())
assert rank == result["exact_numerical_rank"]
r = int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), result["protocol"]["approximate_retention"])) + 1
assert r == result["approximate_retained_dimension"]
c = np.eye(64) - vh[:r].T @ vh[:r]
clean_q = data["q"] - c @ (data["q"] - data["q0"])
clean_bq = data["bq"] - c @ (data["bq"] - data["bq0"])
checks = []
for row in result["responses"]:
    context, t = row["context"], row["coefficient"]
    h = data[context]
    # Form the completed query matrix first; do not use the audit's delta formula.
    if context == "native":
        q, bq, qc, bqc = data["q"], data["bq"], clean_q, clean_bq
        k, bk = data["k"], data["bk"]
    else:
        q = data["q0"] + t * (data["q"] - data["q0"])
        bq = data["bq0"] + t * (data["bq"] - data["bq0"])
        qc = data["q0"] + t * (clean_q - data["q0"])
        bqc = data["bq0"] + t * (clean_bq - data["bq0"])
        k = data["k0"] + t * (data["k"] - data["k0"])
        bk = data["bk0"] + t * (data["bk"] - data["bk0"])
    kvals = h @ k.T + bk
    kvals -= kvals.mean(axis=1, keepdims=True)
    scores = (h @ q.T + bq) @ kvals.swapaxes(-1, -2) / 8
    clean_scores = (h @ qc.T + bqc) @ kvals.swapaxes(-1, -2) / 8
    values = dict(key_complement_energy_fraction=float(np.sum((kvals@c)**2)/np.sum(kvals**2)),
                  cleanup_centered_score_rms=float(np.sqrt(np.mean((clean_scores-scores)**2))),
                  baseline_centered_score_rms=float(np.sqrt(np.mean(scores**2))),
                  cleanup_over_baseline_score=float(np.linalg.norm(clean_scores-scores)/np.linalg.norm(scores)),
                  attention_probability_rms_change=float(np.sqrt(np.mean((softmax(clean_scores,axis=-1)-softmax(scores,axis=-1))**2))))
    errors = {key: abs(value-row[key]) for key, value in values.items()}
    assert all(error < 2e-11 for error in errors.values()), errors
    checks.append(dict(context=context, recomputed=values, absolute_error=errors))
validation = dict(passed=True, method="Independent NumPy/SciPy sample-matrix SVD and direct completed-query scores",
                  exact_rank=rank, approximate_retained_dimension=r, checks=checks,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  verification_inputs_sha256=hashlib.sha256((root/"verification_inputs.pt").read_bytes()).hexdigest(),
                  scope="Cached training-input linear algebra only; no classification or real partner fusion validation",
                  new_model_forwards=0, new_protected_weights=0)
(root/"independent_validation.json").write_text(json.dumps(validation,indent=2))
print(json.dumps(validation),flush=True)
