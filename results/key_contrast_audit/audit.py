"""Frozen cached-input feasibility audit. No CUDA, forward, or protection weights."""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("experiments").resolve()))
from clip_benchmark import checkpoint_loader, digest

torch.set_num_threads(4)
torch.set_grad_enabled(False)
out = Path("outputs/key_contrast_audit")
protocol = json.loads((out / "protocol.json").read_text())
cache = Path("/root/autodl-tmp/research_archive/2026-10-12/3090/qk_complement/artifacts/calibration_inputs.pt")
inputs = torch.load(cache, map_location="cpu")
root = Path("checkpoints/ViT-B-32")
pins = json.loads(Path("configs/checkpoints.json").read_text())
hashes = {}
for name in ("zeroshot.pt", "MNIST/finetuned.pt"):
    hashes[name] = digest(root / name)
    expected = next(f["sha256"] for f in pins["files"] if f["path"] == "ViT-B-32/" + name)
    assert hashes[name] == expected
load = checkpoint_loader(Path("external/secure-merge-attack"))
anchor = load(str(root / "zeroshot.pt")).cpu().eval()
expert = load(str(root / "MNIST/finetuned.pt")).cpu().eval()

def qk(model):
    a = model.model.visual.transformer.resblocks[0].attn
    # One fixed 64-dimensional head, with query/key biases included.
    w, b = a.in_proj_weight.double(), a.in_proj_bias.double()
    return w[:64], b[:64], w[768:832], b[768:832]

q, bq, k, bk = qk(expert)
q0, bq0, k0, bk0 = qk(anchor)
x = inputs["native"].double()
assert x.shape == (256, 50, 768)
keys = x @ k.T + bk
contrasts = keys - keys.mean(dim=1, keepdim=True)
flat = contrasts.flatten(0, 1)
cov = flat.T @ flat / len(flat)
eigenvalues, basis = torch.linalg.eigh(cov)
tolerance = max(flat.shape) * torch.finfo(torch.float64).eps * eigenvalues[-1]
rank = int((eigenvalues > tolerance).sum())
descending = eigenvalues.flip(0).clamp_min(0)
r = int(torch.searchsorted(descending.cumsum(0) / descending.sum(), protocol["approximate_retention"])) + 1
p = basis[:, -r:] @ basis[:, -r:].T
c = torch.eye(64, dtype=torch.float64) - p
rows = []
for context, scale in (("native", 1.0), ("proxy", 0.5)):
    h = inputs[context].double()
    wc = q if context == "native" else q0 + scale * (q - q0)
    bc = bq if context == "native" else bq0 + scale * (bq - bq0)
    wk = k if context == "native" else k0 + scale * (k - k0)
    bbk = bk if context == "native" else bk0 + scale * (bk - bk0)
    qc = h @ wc.T + bc
    kc = h @ wk.T + bbk
    kc = kc - kc.mean(dim=1, keepdim=True)
    task_query_c = (h @ (q - q0).T + bq - bq0) @ c
    anchor_query_c = (h @ q0.T + bq0) @ c
    cleanup_scores = -scale * task_query_c @ kc.transpose(-1, -2) / math.sqrt(64)
    original_scores = qc @ kc.transpose(-1, -2) / math.sqrt(64)
    baseline_attention = original_scores.softmax(-1)
    cleaned_attention = (original_scores + cleanup_scores).softmax(-1)
    rows.append(dict(context=context, coefficient=scale,
        key_complement_energy_fraction=float((kc @ c).square().sum() / kc.square().sum()),
        task_query_complement_rms=float(task_query_c.square().sum(-1).mean().sqrt()),
        anchor_query_complement_rms=float(anchor_query_c.square().sum(-1).mean().sqrt()),
        task_over_anchor_query_response=float(task_query_c.norm() / anchor_query_c.norm()),
        cleanup_centered_score_rms=float(cleanup_scores.square().mean().sqrt()),
        baseline_centered_score_rms=float(original_scores.square().mean().sqrt()),
        cleanup_over_baseline_score=float(cleanup_scores.norm() / original_scores.norm()),
        attention_probability_rms_change=float((cleaned_attention - baseline_attention).square().mean().sqrt())))
result = dict(protocol=protocol, key_contrast_dimension=64, exact_numerical_rank=rank,
    exact_rank_tolerance=float(tolerance), eigenvalues_ascending=eigenvalues.tolist(),
    approximate_retained_dimension=r, approximate_complement_dimension=64-r,
    native_spectrum_tail_fraction=float(descending[r:].sum()/descending.sum()),
    responses=rows, new_model_forwards=0, new_protected_weights=0,
    provenance=dict(script_sha256=digest(__file__), protocol_sha256=digest(out/"protocol.json"),
        cache_sha256=digest(cache), checkpoint_sha256=hashes,
        source_cache_commit="8d28525", framework=torch.__version__),
    decision="No nonzero exact global key-contrast complement at this frozen head" if rank == 64 else "Finite-cache nullspace exists; unseen-data preservation and useful cleaning cost remain unproven",
    completed=True)
(out / "results.json").write_text(json.dumps(result, indent=2))
# Heavy verification capsule stays only on the server.
torch.save(dict(native=x, proxy=inputs["proxy"], q=q, bq=bq, k=k, bk=bk,
    q0=q0, bq0=bq0, k0=k0, bk0=bk0), out / "verification_inputs.pt")
print(json.dumps(result), flush=True)
