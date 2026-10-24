"""Read-only applicability check for the known even-plus-linear symmetry."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from _impl.io import digest
from _impl.models import initialize, load_checkpoint


initialize()
start = time.monotonic()
cfg = json.loads(Path("configs/construct.json").read_text())["targets"]["MNIST"]["basis"]
assert digest(cfg["path"]) == cfg["sha256"]
basis = torch.load(cfg["path"], map_location="cpu", weights_only=False)
model = load_checkpoint("MNIST/finetuned.pt")
state = model.state_dict()
prefix = "model.visual.transformer.resblocks.11."
key = prefix + "mlp.c_fc.weight"
V = next(v["v"].double().cuda() for v in basis.values() if v["source_key"] == key)
A = state[key].double().cuda()
B = state[prefix + "mlp.c_proj.weight"].double().cuda()
a = state[prefix + "mlp.c_fc.bias"].double().cuda()
gamma = state[prefix + "ln_2.weight"].double().cuda()
beta = state[prefix + "ln_2.bias"].double().cuda()
assert bool((gamma != 0).all())
G = V / gamma[:, None]
_, _, vh = torch.linalg.svd(G.sum(0, keepdim=True), full_matrices=True)
Qtask = torch.linalg.qr(G @ vh[1:].T, mode="reduced").Q
Qfull = torch.linalg.qr(torch.ones(768, 1, dtype=torch.float64, device="cuda"), mode="complete").Q[:, 1:]
btb = B.T @ B
rng = np.random.default_rng(2026102402)
records = []
for name, Q in [("task_LN_intersection", Qtask), ("full_LN", Qfull)]:
    R = A @ (gamma[:, None] * Q)
    gram = btb * (R @ R.T)
    scale = gram.diag().sqrt()
    assert bool((scale > 0).all())
    K = gram / (scale[:, None] * scale[None, :])
    eigen = torch.linalg.eigvalsh(K).cpu().numpy()
    lo, hi = float(eigen[0]), float(eigen[-1])
    kn = K.cpu().numpy()
    shift = lo / 2 if lo > 0 else 0.0
    shifted = kn - shift * np.eye(len(kn))
    try:
        chol = np.linalg.cholesky(shifted)
        chol_error = float(np.linalg.norm(chol @ chol.T - shifted) / np.linalg.norm(shifted))
        chol_min = float(np.diag(chol).min())
    except np.linalg.LinAlgError:
        chol_error, chol_min = None, None
    probes = rng.standard_normal((16, 3072))
    errors = []
    for probe in probes:
        c = torch.tensor(probe, dtype=torch.float64, device="cuda")
        direct = float((B @ (c[:, None] * R)).square().sum())
        predicted = float(c @ gram @ c)
        errors.append(abs(direct - predicted) / max(direct, 1e-30))
    u = torch.tensor(rng.standard_normal((16, Q.shape[1])), dtype=torch.float64, device="cuda")
    u *= 0.5 * np.sqrt(768) / u.norm(dim=1, keepdim=True)
    x = beta + u @ (gamma[:, None] * Q).T
    z = x @ A.T + a
    mask = torch.arange(3072, device="cuda") % 2 == 0
    delta = ((-z[:, mask]) * torch.sigmoid(-1.702 * z[:, mask]) - z[:, mask] * torch.sigmoid(1.702 * z[:, mask])) @ B[:, mask].T
    offset = B[:, mask] @ (A[mask] @ beta + a[mask])
    predicted = -(u @ R[mask].T) @ B[:, mask].T
    records.append(dict(domain=name, dimension=Q.shape[1], neurons=3072,
        centred_domain_error=float(Q.sum(0).abs().max()),
        correlation_min_eigenvalue=lo, correlation_max_eigenvalue=hi,
        smallest_ten_eigenvalues=eigen[:10].tolist(), numpy_cholesky_shift=shift,
        numpy_cholesky_min_diagonal=chol_min, numpy_cholesky_relative_error=chol_error,
        direct_Gram_max_relative_error=max(errors),
        QuickGELU_linear_identity_max_error=float((delta + offset - predicted).abs().max())))
result = dict(status="read_only_known_symmetry_applicability", layer=11, target="MNIST",
    basis_sha256=cfg["sha256"], checkpoint_sha256=digest("checkpoints/ViT-B-32/MNIST/finetuned.pt"),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records,
    real_images=0, classification_heads=0, partner_models=0, new_defense_points=0,
    optimized_masks=0, expert_updates=0, seconds=time.monotonic()-start,
    scope="Only collective whole-affine sign reversal with unchanged B and free output bias, on two declared continuous LN domains. No claim about approximate compensation, other numerical edits, cross-layer compensation, task accuracy or AGA.")
out = Path("outputs/2026-10-24_research/group_condition.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
