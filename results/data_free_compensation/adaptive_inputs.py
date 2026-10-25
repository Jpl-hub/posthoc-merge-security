"""One Data-Free Defense2 construction with active synthetic inputs."""
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.metrics import logical_parameter_distance
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


def main():
    initialize()
    start = time.monotonic()
    protocol = Path("results/data_free_compensation/adaptive_inputs_protocol.json")
    cfg = json.loads(protocol.read_text())
    source = Path(__file__).read_text()
    assert digest(__file__) == cfg["implementation_sha256"]
    fit = next(ast.get_source_segment(source, n) for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name == "_fit_nonlinear_mlp_layer")
    assert hashlib.sha256(fit.encode()).hexdigest() == cfg["fit_helper"]["sha256"]
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(protocol.read_text())
    for key in ["source", "basis"]:
        assert digest(cfg[key]["path"]) == cfg[key]["sha256"]
    base, expert, head, pins = checked_models(cfg["target"])
    del base
    raw = torch.load(cfg["source"]["path"], map_location="cpu", weights_only=False)
    basis = torch.load(cfg["basis"]["path"], map_location="cpu", weights_only=False)
    entries = {v["source_key"]: v for v in basis.values()}
    latents = raw["final"].clone()
    del raw
    state = expert.state_dict()
    published = copy.deepcopy(expert).eval().requires_grad_(False)
    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
    std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
    counters = dict(generated_full_image_forwards=0, input_gradient_model_forwards=0,
                    pixel_optimizer_steps=0, linear_solves=0)

    def views(x):
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        return (torch.cat([torch.roll(x, tuple(s), (2, 3))
                          for s in cfg["source"]["views"]]) - mean) / std

    def scores(model, x):
        counters["generated_full_image_forwards"] += len(x)
        if torch.is_grad_enabled() and x.requires_grad:
            counters["input_gradient_model_forwards"] += len(x)
        return head(model(x))

    @torch.no_grad()
    def image_pool(pixels):
        return torch.cat([views(x.cuda()).cpu()
                          for x in pixels.split(cfg["source"]["group_size"])])

    @torch.no_grad()
    def forward_pool(model, images):
        model.cuda().eval()
        head.cuda().eval()
        result = torch.cat([scores(model, x.cuda()).cpu()
                            for x in images.split(cfg["source"]["batch_size"])])
        model.cpu()
        head.cpu()
        return result

    # Set the task-complement A changes before input search; B remains original.
    with torch.no_grad():
        for i in cfg["layers"]:
            p = f"model.visual.transformer.resblocks.{i}.mlp."
            A = state[p + "c_fc.weight"].double().cuda()
            a = state[p + "c_fc.bias"].double().cuda()
            ba, bb = [entries[p + n + ".weight"] for n in ["c_fc", "c_proj"]]
            columns = torch.cat([ba["u"].cuda(), A @ ba["v"].cuda(),
                                 bb["v"].cuda(), a[:, None]], dim=1)
            columns = columns[:, columns.norm(dim=0) > 0]
            columns /= columns.norm(dim=0)
            C, singular, _ = torch.linalg.svd(columns, full_matrices=False)
            C = C[:, singular > singular[0] * 1e-10]
            Ap = (2 * C @ (C.T @ A) - A).float().cpu()
            published.state_dict()[p + "c_fc.weight"].copy_(Ap)

    groups = [torch.nn.Parameter(x.clone().cuda())
              for x in latents.split(cfg["source"]["group_size"])]
    optimizers = [torch.optim.Adam([x], lr=cfg["source"]["pixel_lr"]) for x in groups]
    pixel_history = [latents.clone()]
    pools = [image_pool(latents)]
    rounds, edits = [], {}
    temperature = cfg["source"]["temperature"]

    for cycle in range(cfg["source"]["rounds"]):
        expert.cuda().eval()
        published.cuda().eval()
        head.cuda().eval()
        first_losses, last_losses, gradient_norms = [], [], []
        for group_index, (pixels, optimizer) in enumerate(zip(groups, optimizers)):
            for step in range(cfg["source"]["pixel_steps_per_round"]):
                x = views(pixels)
                teacher_logits = scores(expert, x)
                student_logits = scores(published, x)
                divergence = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                )
                if step == 0:
                    first_losses.append(float(divergence.detach()))
                optimizer.zero_grad(set_to_none=True)
                (-divergence).backward()
                assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
                gradient_norms.append(float(pixels.grad.norm()))
                optimizer.step()
                counters["pixel_optimizer_steps"] += 1
                with torch.no_grad():
                    pixels.clamp_(0, 1)
            with torch.no_grad():
                x = views(pixels)
                last_losses.append(float(F.kl_div(
                    F.log_softmax(scores(published, x) / temperature, dim=1),
                    F.softmax(scores(expert, x) / temperature, dim=1),
                    reduction="batchmean",
                )))
            print("PIXEL_GROUP", cycle + 1, group_index,
                  first_losses[-1], last_losses[-1], flush=True)
        assert all(p.grad is None for model in [expert, published, head]
                   for p in model.parameters())
        expert.cpu()
        published.cpu()
        head.cpu()
        latents = torch.cat([x.detach().cpu() for x in groups])
        pixel_history.append(latents.clone())
        torch.save(pixel_history, out / "inputs.pt")
        pools.append(image_pool(latents))
        images = torch.cat(pools)

        teacher = {i: [] for i in cfg["layers"]}
        hooks = [expert.model.visual.transformer.resblocks[i].register_forward_hook(
            lambda _m, _a, y, i=i: teacher[i].append(
                y.detach().permute(1, 0, 2).cpu())) for i in cfg["layers"]]
        try:
            before = forward_pool(expert, images)
        finally:
            for hook in hooks:
                hook.remove()

        rows = []
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
                forward_pool(published, images)
            finally:
                for hook in hooks:
                    hook.remove()
            p = f"model.visual.transformer.resblocks.{i}.mlp."
            A, B = [state[p + n + ".weight"].double().cuda()
                    for n in ["c_fc", "c_proj"]]
            a = state[p + "c_fc.bias"].double().cuda()
            X = torch.cat(inputs)
            target = torch.cat(teacher.pop(i)) - torch.cat(residuals) - state[p + "c_proj.bias"]
            with torch.no_grad():
                Ap, Bp, row = _fit_nonlinear_mlp_layer(
                    A, B, a, entries[p + "c_fc.weight"], entries[p + "c_proj.weight"],
                    X, target, block.mlp.gelu, cfg)
                assert torch.equal(Ap, published.state_dict()[p + "c_fc.weight"].cpu())
                changed = {p + "c_fc.weight": Ap, p + "c_proj.weight": Bp}
                for key, value in changed.items():
                    published.state_dict()[key].copy_(value)
                edits.update(changed)
            counters["linear_solves"] += 1
            row["layer"] = i
            rows.append(row)
            print("LAYER", cycle + 1, i, row["relative_B_distance"],
                  row["output_relative_RMS_after_ridge"], flush=True)
            if not row["parameter_range_passed"]:
                torch.save(edits, out / "partial_edits.pt")
                failure = dict(status="closed_parameter_range_failed", cycle=cycle + 1,
                               per_layer=rows, counters=counters, source=source,
                               source_sha256=digest(__file__), real_construction_images=0)
                (out / "results.json").write_text(json.dumps(failure, indent=2) + "\n")
                return
        after = forward_pool(published, images)
        with torch.no_grad():
            divergence = float(F.kl_div(
                F.log_softmax(after / temperature, dim=1),
                F.softmax(before / temperature, dim=1), reduction="batchmean"))
        row = dict(
            cycle=cycle + 1, source_views=len(images),
            critic_KL_before=sum(first_losses) / len(first_losses),
            critic_KL_after=sum(last_losses) / len(last_losses),
            input_gradient_norm_min=min(gradient_norms),
            input_gradient_norm_max=max(gradient_norms),
            source_teacher_prediction_changes=int((after.argmax(1) != before.argmax(1)).sum()),
            source_KL_after_fit=divergence,
            teacher_class_counts=torch.bincount(before.argmax(1), minlength=10).tolist(),
            per_layer=rows,
        )
        rounds.append(row)
        torch.save(dict(before=before, after=after), out / "source_scores.pt")
        (out / "progress.json").write_text(json.dumps(dict(rounds=rounds, counters=counters),
                                                        indent=2) + "\n")
        print("ROUND", json.dumps({k: v for k, v in row.items() if k != "per_layer"}),
              flush=True)
        del images, teacher, X, target, inputs, residuals

    torch.save(edits, out / "point.pt")
    sha = digest(out / "point.pt")
    result = dict(
        status="unique_adaptive_input_point_frozen", point_sha256=sha,
        source_sha256=digest(__file__), source=source, protocol_sha256=digest(protocol),
        checkpoint_sha256=pins, rounds=rounds, counters=counters,
        input_history_sha256=digest(out / "inputs.pt"),
        source_scores_sha256=digest(out / "source_scores.pt"),
        parameter_distance=logical_parameter_distance(edits, state),
        real_construction_images=0, teacher_training_updates=0,
        model_parameter_gradient_steps=0, new_AGA_calls=0,
        seconds=time.monotonic() - start, limits=cfg["novelty"])
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("POINT_FROZEN", sha, "D", result["parameter_distance"], flush=True)
    print("COUNTERS", json.dumps(counters), flush=True)


if __name__ == "__main__":
    main()
