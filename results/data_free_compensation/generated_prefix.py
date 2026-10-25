"""Generated-input evaluation of the existing fixed prefix-reconstruction method."""
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.metrics import logical_parameter_distance, task_change
from _impl.models import checked_models, initialize

def _fit_nonlinear_mlp_layer(A, B, a, ba, bb, X, target, phi, cfg):
    """Fixed task-complement change and one finite output reconstruction."""
    import torch.nn.functional as F

    ua, va, vb = ba["u"].cuda(), ba["v"].cuda(), bb["v"].cuda()
    columns = torch.cat([ua, A @ va, vb, a[:, None]], dim=1)
    columns = columns[:, columns.norm(dim=0) > 0]
    columns /= columns.norm(dim=0)
    C, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    C = C[:, singular > singular[0] * 1e-10]
    Ap = (2 * C @ (C.T @ A) - A).float()
    Bt = (2 * (B @ C) @ C.T - B).float()

    width = B.shape[1]
    gram = torch.zeros(width, width, dtype=torch.double, device="cuda")
    cross = torch.zeros(width, B.shape[0], dtype=torch.double, device="cuda")
    error_energy = torch.zeros((), dtype=torch.double, device="cuda")
    target_energy = error_energy.clone()
    for x, y in zip(X.flatten(0, 1).split(4096), target.flatten(0, 1).split(4096)):
        hp = phi(F.linear(x.cuda(), Ap, a.float()))
        y = y.cuda()
        error = (y - F.linear(hp, Bt)).double()
        error_energy += error.square().sum()
        target_energy += y.double().square().sum()
        hp = hp.double()
        z = hp - (hp @ vb) @ vb.T
        gram += z.T @ z
        cross += z.T @ error
    damping = cfg["damping"] * gram.diagonal().mean()
    if not float(damping) > 0:
        raise ValueError("No active complementary calibration directions")
    system = gram + damping * torch.eye(width, dtype=torch.double, device="cuda")
    delta = torch.linalg.solve(system, cross)
    solve_error = float((system @ delta - cross).norm() / cross.norm())
    if not solve_error < 1e-8:
        raise ValueError(f"Ridge solve residual: {solve_error}")
    delta -= vb @ (vb.T @ delta)
    fitted_error = error_energy - 2 * (delta * cross).sum() + (delta * (gram @ delta)).sum()
    Bp = (Bt.double() + delta.T).float()
    da, db = Ap.double() - A, Bp.double() - B
    distances = [float(da.norm() / A.norm()), float(db.norm() / B.norm())]
    details = dict(
        task_rank=va.shape[1],
        fixed_hidden_dimension=C.shape[1],
        relative_A_distance=distances[0],
        relative_B_distance=distances[1],
        logical_distance=sum(v * v for v in distances) ** 0.5,
        task_A_response_relative=float((da @ va).norm() / da.norm()),
        task_B_response_relative=float((db @ vb).norm() / db.norm()),
        ridge_damping=float(damping),
        ridge_solve_residual=solve_error,
        output_relative_RMS_before_fit=float((error_energy / target_energy).sqrt()),
        output_relative_RMS_after_ridge=float((fitted_error.clamp_min(0) / target_energy).sqrt()),
    )
    details["parameter_range_passed"] = all(
        d <= limit + 1e-6 for d, limit in zip(distances, cfg["matrix_bounds"])
    )
    return Ap.cpu(), Bp.cpu(), details


@torch.no_grad()
def main():
    initialize()
    start = time.monotonic()
    protocol = Path("results/data_free_compensation/generated_prefix_protocol.json")
    cfg = json.loads(protocol.read_text())
    source = Path(__file__).read_text()
    assert digest(__file__) == cfg["implementation_sha256"]
    fit = next(ast.get_source_segment(source, n) for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name == "_fit_nonlinear_mlp_layer")
    assert hashlib.sha256(fit.encode()).hexdigest() == cfg["algorithm_reference"]["helper_sha256"]
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(protocol.read_text())
    for key in ["source", "basis"]:
        assert digest(cfg[key]["path"]) == cfg[key]["sha256"]

    _, expert, head, pins = checked_models(cfg["target"])
    raw = torch.load(cfg["source"]["path"], map_location="cpu", weights_only=False)
    basis = torch.load(cfg["basis"]["path"], map_location="cpu", weights_only=False)
    entries = {v["source_key"]: v for v in basis.values()}
    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
    std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
    images = []
    for x in raw["final"].split(cfg["source"]["group_size"]):
        x = F.interpolate(x.cuda(), (224, 224), mode="bilinear", align_corners=False)
        views = torch.cat([torch.roll(x, tuple(s), (2, 3)) for s in cfg["source"]["views"]])
        images.append(((views - mean) / std).cpu())
    images = torch.cat(images)
    assert len(images) == cfg["source"]["source_views"]

    def forward(model):
        model.cuda().eval()
        head.cuda().eval()
        values = []
        for x in images.split(cfg["source"]["batch_size"]):
            values.append(head(model(x.cuda())).cpu())
        model.cpu()
        head.cpu()
        return torch.cat(values)

    teacher = {i: [] for i in cfg["layers"]}
    hooks = [expert.model.visual.transformer.resblocks[i].register_forward_hook(
        lambda _m, _a, y, i=i: teacher[i].append(y.detach().permute(1, 0, 2).cpu()))
        for i in cfg["layers"]]
    try:
        before = forward(expert)
    finally:
        for hook in hooks:
            hook.remove()
    source_error = float((before - raw["cache"]["native"]["scores"]).abs().max())
    assert source_error <= 5e-4
    assert torch.equal(before.argmax(1), raw["teacher_class"])
    published = copy.deepcopy(expert).eval()
    state = expert.state_dict()
    rows, edits = [], {}
    image_forwards = len(images)

    for i in cfg["layers"]:
        block = published.model.visual.transformer.resblocks[i]
        inputs, residuals = [], []
        hooks = [
            block.mlp.c_fc.register_forward_pre_hook(
                lambda _m, a: inputs.append(a[0].detach().permute(1, 0, 2).cpu())),
            block.ln_2.register_forward_pre_hook(
                lambda _m, a: residuals.append(a[0].detach().permute(1, 0, 2).cpu())),
        ]
        try:
            forward(published)
            image_forwards += len(images)
        finally:
            for hook in hooks:
                hook.remove()
        p = f"model.visual.transformer.resblocks.{i}.mlp."
        A, B = [state[p + n + ".weight"].double().cuda() for n in ["c_fc", "c_proj"]]
        a = state[p + "c_fc.bias"].double().cuda()
        X = torch.cat(inputs)
        target = torch.cat(teacher.pop(i)) - torch.cat(residuals) - state[p + "c_proj.bias"]
        Ap, Bp, row = _fit_nonlinear_mlp_layer(
            A, B, a, entries[p + "c_fc.weight"], entries[p + "c_proj.weight"],
            X, target, block.mlp.gelu, cfg)
        row["layer"] = i
        rows.append(row)
        changed = {p + "c_fc.weight": Ap, p + "c_proj.weight": Bp}
        for key, value in changed.items():
            published.state_dict()[key].copy_(value)
        edits.update(changed)
        print("PREFIX_LAYER", i, json.dumps(row), flush=True)
        if not row["parameter_range_passed"]:
            torch.save(edits, out / "partial_edits.pt")
            result = dict(status="closed_parameter_range_failed", per_layer=rows,
                          image_forwards=image_forwards, source=source,
                          source_sha256=digest(__file__), real_construction_images=0)
            (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            return

    torch.save(edits, out / "point.pt")
    sha = digest(out / "point.pt")
    print("POINT_FROZEN", sha, flush=True)
    after = forward(published)
    image_forwards += len(images)
    torch.save(dict(before=before, after=after, teacher_class=before.argmax(1)),
               out / "source_scores.pt")
    source_stats = task_change(before.double().numpy(), after.double().numpy())
    result = dict(
        status="unique_generated_prefix_point_frozen",
        point_sha256=sha, source_sha256=digest(__file__), source=source,
        protocol_sha256=digest(protocol), checkpoint_sha256=pins,
        source_native_replay_max_error=source_error, per_layer=rows,
        source_response=source_stats,
        generated_full_image_forwards=image_forwards,
        generated_input_base_count=160, generated_views=800,
        source_scores_sha256=digest(out / "source_scores.pt"),
        parameter_distance=logical_parameter_distance(edits, state),
        real_construction_images=0, teacher_training_updates=0,
        optimizer_steps=0, linear_solves=len(rows), new_AGA_calls=0,
        seconds=time.monotonic() - start,
        limits=cfg["comparison_limits"])
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PREFIX_DONE", json.dumps({k: v for k, v in result.items()
                                    if k not in ["source", "per_layer", "checkpoint_sha256"]}),
          flush=True)


if __name__ == "__main__":
    main()
