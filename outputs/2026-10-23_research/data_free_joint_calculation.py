"""One fixed full-width, image-free joint A/B feasibility calculation."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space, qr
from scipy.optimize import minimize

from _impl.io import check_files, digest, save_json
from _impl.models import initialize, load_checkpoint
from _impl.subspace import _minimax_quadratics

initialize()
started = time.monotonic()
protocol = "results/data_free_compensation/pilot_protocol.json"
cfg = json.loads(Path(protocol).read_text())
check_files(cfg["source_sha256"])
check_files({cfg["basis"]["path"]: cfg["basis"]["sha256"]})
out = Path(cfg["output"])
out.mkdir(parents=True, exist_ok=False)
expert = load_checkpoint("MNIST/finetuned.pt")
block = expert.model.visual.transformer.resblocks[cfg["layer"]]
basis = torch.load(cfg["basis"]["path"], map_location="cpu", weights_only=False)
prefix = f"model.visual.transformer.resblocks.{cfg['layer']}.mlp."
ba, bb = [
    next(v for v in basis.values() if v["source_key"] == prefix + k + ".weight")
    for k in ["c_fc", "c_proj"]
]


def device(value):
    return torch.as_tensor(value, dtype=torch.double, device="cuda")


A, B = [device(m.weight.detach().numpy()) for m in [block.mlp.c_fc, block.mlp.c_proj]]
a = device(block.mlp.c_fc.bias.detach().numpy())
gamma, beta = [p.detach().double().numpy() for p in [block.ln_2.weight, block.ln_2.bias]]
assert abs(gamma).min() > 1e-8
va, ua, vb = [ba["v"].numpy(), ba["u"].numpy(), bb["v"].numpy()]
g = va / gamma[:, None]
task, _ = qr(g @ null_space(g.sum(0, keepdims=True)), mode="economic")
right, _ = qr(np.column_stack([va, beta]), mode="economic")
free = device(null_space(vb.T))
left, right = device(ua), device(right)
radius = float(A.norm()) * cfg["relative_A_distance"]
regularizer = cfg["ridge"] / float(B.square().sum())


def project(value):
    value = value - left @ (left.T @ value)
    return value - (value @ right) @ right.T


def samples(seed, count):
    rng = np.random.default_rng(seed)
    values = []
    for domain in [task, None]:
        dimension = task.shape[1] if domain is not None else len(beta) - 1
        u = rng.normal(size=(count, dimension if domain is not None else len(beta)))
        if domain is None:
            u -= u.mean(1, keepdims=True)
        u *= (
            np.sqrt(len(beta)) * rng.random(count) ** (1 / dimension) / np.linalg.norm(u, axis=1)
        )[:, None]
        if domain is not None:
            u = u @ domain.T
        x = device(beta + gamma * u)
        z = x @ A.T + a
        y = (z * torch.sigmoid(1.702 * z)) @ B.T
        values.append((x, y, float(y.square().sum())))
    return values


source = samples(cfg["seed"], cfg["source_n"])
initial = project(A).cpu().numpy()
assert np.linalg.norm(initial) > 0
trace = []


def calculate(flat, gradient=True):
    z = project(device(flat.reshape(A.shape)))
    norm = z.norm()
    unit = z / norm
    Ap = A + radius * unit
    quadratic, cross, constant = [], [], []
    responses = []
    for x, y, energy in source:
        zp = x @ Ap.T + a
        hp = zp * torch.sigmoid(1.702 * zp)
        features, error = hp @ free, y - hp @ B.T
        quadratic.append((features.T @ features / energy).cpu().numpy())
        cross.append((error.T @ features / energy).cpu().numpy())
        constant.append(float(error.square().sum()) / energy)
        responses.append((hp, zp))
    correction, row = _minimax_quadratics(quadratic, cross, constant, regularizer)
    Bp = B + device(correction) @ free.T
    measured, derivative = [], torch.zeros_like(A)
    weight = row["first_distribution_weight"]
    for w, (x, y, energy), (hp, zp) in zip([weight, 1 - weight], source, responses):
        residual = hp @ Bp.T - y
        measured.append(float(residual.square().sum()) / energy)
        if gradient:
            sigmoid = torch.sigmoid(1.702 * zp)
            phi_prime = sigmoid + 1.702 * zp * sigmoid * (1 - sigmoid)
            derivative += 2 * w * ((residual @ Bp) * phi_prime).T @ x / energy
    error = max(abs(v - e) for v, e in zip(measured, row["normalized_squared_errors"]))
    assert error < 1e-8, error
    assert abs(row["primal_dual_gap"]) < 1e-8, row
    row.update(
        evaluation=len(trace),
        direct_loss_error=error,
        A_only_normalized_squared_errors=constant,
        relative_A_distance=float((Ap - A).norm() / A.norm()),
        relative_B_distance=float((Bp - B).norm() / B.norm()),
        seconds=time.monotonic() - started,
    )
    trace.append(row)
    save_json(out / "trace.json", trace)
    print("JOINT", json.dumps(row), flush=True)
    if not gradient:
        return Ap, Bp, row
    derivative = project(derivative)
    derivative = radius / norm * (derivative - (derivative * unit).sum() * unit)
    return row["objective"], derivative.cpu().numpy().ravel()


solution = minimize(
    calculate, initial.ravel(), jac=True, method="L-BFGS-B", options=cfg["optimizer"]
)
Ap, Bp, final = calculate(solution.x, gradient=False)
edits = {prefix + k + ".weight": v.float().cpu() for k, v in [("c_fc", Ap), ("c_proj", Bp)]}
torch.save(edits, out / "point.pt")
frozen = dict(
    point_sha256=digest(out / "point.pt"),
    protocol_sha256=digest(protocol),
    source_sha256=cfg["source_sha256"],
    before_confirmation=True,
    optimizer_success=bool(solution.success),
    optimizer_message=str(solution.message),
    iterations=int(solution.nit),
    function_evaluations=int(solution.nfev),
    final=final,
)
save_json(out / "before_confirmation.json", frozen)

# Independent synthetic vectors are drawn only after the sole final point is frozen.
Ap, Bp = [device(edits[prefix + k + ".weight"].numpy()) for k in ["c_fc", "c_proj"]]
confirmation = []
for x, y, energy in samples(cfg["confirmation_seed"], cfg["confirmation_n"]):
    zp = x @ Ap.T + a
    residual = (zp * torch.sigmoid(1.702 * zp)) @ Bp.T - y
    confirmation.append(float((residual.square().sum() / energy).sqrt()))
constraints = dict(
    A_right=float(((Ap - A) @ device(va)).norm() / (Ap - A).norm()),
    A_left=float((device(ua).T @ (Ap - A)).norm() / (Ap - A).norm()),
    B_right=float(((Bp - B) @ device(vb)).norm() / (Bp - B).norm()),
)
passed = max(confirmation) <= cfg["probe_relative_RMS_limit"]
passed &= final["relative_B_distance"] <= cfg["relative_B_bound"]
result = frozen | dict(
    status="completed",
    probe_passed=bool(passed),
    confirmation_relative_RMS=confirmation,
    task_constraints=constraints,
    full_width=int(A.shape[0]),
    real_images=0,
    model_image_forwards=0,
    teacher_training_updates=0,
    new_accuracy_results=0,
    seconds=time.monotonic() - started,
    limitation="One full-width block calculation on explicit synthetic LN probes. Not a whole-model utility or TA/AGA result. No claim of a new optimization algorithm.",
)
save_json(out / "results.json", result)
print("FINISHED", json.dumps(result), flush=True)
