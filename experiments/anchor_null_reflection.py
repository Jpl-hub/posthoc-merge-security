"""One coupled nonlinear MLP reflection inside the public anchor nullspace.

This is a mechanism screen, not an accuracy or security result. Explicitly
include public anchor-span deletion; no future partners or gradient fitting.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from critical_subspace import capture
from eight_task_benchmark import digest
from local_erasure import rescale
from neuron_projection import activation
from path_subspace import context, read, response
from scipy.linalg import eigh
from subspace_probe import save_json


@torch.no_grad()
def anchor_basis(mlp, rcond):
    parts = [mlp.c_fc.weight, mlp.c_fc.bias[:, None], mlp.c_proj.weight.T]
    matrix = torch.cat([x.double().cuda() / x.double().norm() for x in parts], dim=1)
    q, r = torch.linalg.qr(matrix, mode="complete")
    singular = torch.linalg.svdvals(r[: matrix.shape[1]])
    assert float(singular[-1] / singular[0]) > rcond
    span, null = q[:, : matrix.shape[1]], q[:, matrix.shape[1] :]
    leakage = float((matrix.T @ null).norm() / matrix.norm())
    assert leakage < 1e-10
    return (
        span,
        null,
        dict(
            anchor_rank=matrix.shape[1],
            null_dimension=null.shape[1],
            smallest_relative_singular=float(singular[-1] / singular[0]),
            null_relative_residual=leakage,
        ),
    )


@torch.no_grad()
def direction(native, base, captured, layer, cfg, null):
    w, b = native.c_fc.weight.double().cuda(), native.c_fc.bias.double().cuda()
    aw, ab = base.c_fc.weight.double().cuda(), base.c_fc.bias.double().cuda()
    tensors = {}
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
        x = captured[name][f"up:{layer}"].double().cuda()
        z = x @ ((1 - beta) * aw + beta * w).T + (1 - beta) * ab + beta * b
        tensors[name] = (z, activation(z))
    scales = [float(x.square().sum() / len(x)) for x in tensors["expert"]]
    grams = {}
    for name, pair in tensors.items():
        projected = [x @ null for x in pair]
        grams[name] = sum(
            x.T @ x / len(x) / scale for x, scale in zip(projected, scales, strict=True)
        )
    ge = grams["expert"]
    ridge = cfg["gram_ridge_ratio"] * float(ge.trace()) / len(ge)
    denominator = ge + ridge * torch.eye(len(ge), dtype=torch.float64, device="cuda")
    path = sum(grams[str(t)] for t in cfg["betas"]) / len(cfg["betas"])
    value, vectors = eigh(
        path.cpu().numpy(), denominator.cpu().numpy(), subset_by_index=[len(ge) - 1] * 2
    )
    coeff = torch.from_numpy(vectors[:, 0].copy()).cuda()
    v = null @ coeff
    v /= v.norm()
    v *= 1 if float(v[v.abs().argmax()]) >= 0 else -1
    coeff = null.T @ v
    eigen_residual = (path @ coeff - float(value[0]) * denominator @ coeff).norm()
    eigen_residual /= (path @ coeff).norm().clamp_min(1e-30)
    assert float(eigen_residual) < 1e-7
    evidence = dict(
        vector=v.cpu(),
        native_gram=ge.cpu(),
        path_gram=path.cpu(),
        ridge=ridge,
        eigenvalue=float(value[0]),
    )
    return (
        v,
        tensors,
        evidence,
        dict(
            generalized_eigenvalue=float(value[0]),
            eigen_residual=float(eigen_residual),
            normalized_native_energy=float(coeff @ ge @ coeff),
            normalized_path_energy=float(coeff @ path @ coeff),
            ridge_energy=ridge,
        ),
    )


@torch.no_grad()
def reflect(mlp, vector):
    result = copy.deepcopy(mlp).cpu().eval()
    v = vector.double().cuda()
    w, b, down = [x.double().cuda() for x in [mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight]]
    result.c_fc.weight.copy_((w - 2 * v[:, None] * (v @ w)[None, :]).float().cpu())
    result.c_fc.bias.copy_((b - 2 * v * (v @ b)).float().cpu())
    result.c_proj.weight.copy_((down - 2 * (down @ v)[:, None] * v[None, :]).float().cpu())
    return result


@torch.no_grad()
def public_span(mlp, span):
    result = copy.deepcopy(mlp).cpu().eval()
    p = span.double().cuda()
    w, b, down = [x.double().cuda() for x in [mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight]]
    result.c_fc.weight.copy_((p @ (p.T @ w)).float().cpu())
    result.c_fc.bias.copy_((p @ (p.T @ b)).float().cpu())
    result.c_proj.weight.copy_(((down @ p) @ p.T).float().cpu())
    return result


@torch.no_grad()
def local_check(native, base, tensors, v, changed, captured, layer, betas):
    rows = {}
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in betas]:
        z, h = tensors[name]
        down = ((1 - beta) * base.c_proj.weight + beta * native.c_proj.weight).double().cuda()
        a = z @ v
        delta = activation(z - 2 * a[:, None] * v[None, :]) - h
        expected = delta @ down.T - 2 * ((h + delta) @ v)[:, None] * (down @ v)[None, :]
        q = torch.eye(len(v), dtype=torch.float64, device="cuda") - 2 * v[:, None] * v[None, :]
        direct = activation(z @ q) @ (down @ q).T - h @ down.T
        absolute = float((expected - direct).abs().max())
        assert absolute < 1e-10
        x = captured[name][f"up:{layer}"].double().cuda()
        w = ((1 - beta) * base.c_fc.weight.double() + beta * changed.c_fc.weight.double()).cuda()
        b = ((1 - beta) * base.c_fc.bias.double() + beta * changed.c_fc.bias.double()).cuda()
        d = (
            (1 - beta) * base.c_proj.weight.double() + beta * changed.c_proj.weight.double()
        ).cuda()
        actual = activation(x @ w.T + b) @ d.T - h @ down.T
        rounding = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
        rows[name] = dict(
            local_output_relative_rms=float(actual.norm() / (h @ down.T).norm()),
            algebra_max_absolute_error=absolute,
            compiled_rounding_relative_error=rounding,
        )
    return rows


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    assert cfg["rank"] == 1
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = Path(cfg["parent"])
    parent_meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    captured_path = parent / "artifacts/inputs.pt"
    assert digest(captured_path) == parent_meta["raw_sha256"]["inputs.pt"]
    captured = torch.load(captured_path, map_location="cpu", weights_only=True)
    root, load, _, models = context(cfg)
    original, anchor = load(str(root / cfg["task"] / "finetuned.pt")), models["anchor"]
    checkpoints = {
        str(p.relative_to(root)): digest(p)
        for p in [root / "zeroshot.pt", root / cfg["task"] / "finetuned.pt"]
    }
    assert checkpoints == parent_meta["checkpoint_sha256"]
    release, cleaned_original, cleaned_release = [
        copy.deepcopy(original).cpu().eval() for _ in range(3)
    ]
    directions, rows = {}, []
    for layer in cfg["layers"]:
        native = original.model.visual.transformer.resblocks[layer].mlp
        base = anchor.model.visual.transformer.resblocks[layer].mlp
        span, null, anchor_check = anchor_basis(base, cfg["anchor_rank_rcond"])
        v, tensors, evidence, selection = direction(native, base, captured, layer, cfg, null)
        changed = reflect(native, v)
        release.model.visual.transformer.resblocks[layer].mlp = changed
        a, b = public_span(native, span), public_span(changed, span)
        cleaned_original.model.visual.transformer.resblocks[layer].mlp = a
        cleaned_release.model.visual.transformer.resblocks[layer].mlp = b
        deletion_error = max(
            float((a.state_dict()[k] - b.state_dict()[k]).abs().max()) for k in a.state_dict()
        )
        assert deletion_error < 2e-6
        checks = local_check(native, base, tensors, v, changed, captured, layer, cfg["betas"])
        row = dict(
            layer=layer,
            anchor=anchor_check,
            selection=selection,
            local=checks,
            public_deletion_state_max_absolute_error=deletion_error,
        )
        print("REFLECTION", row, flush=True)
        directions[str(layer)] = evidence
        rows.append(row)
    original_state = original.state_dict()
    raw = {}
    for name, model in [
        ("release", release),
        ("public_original", cleaned_original),
        ("public_protected", cleaned_release),
    ]:
        raw[name] = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if not torch.equal(v, original_state[k])
        }
        assert all(".mlp.c_fc." in k or k.endswith(".mlp.c_proj.weight") for k in raw[name])
        torch.save(raw[name], artifacts / f"{name}_edits.pt")
    torch.save(directions, artifacts / "directions.pt")
    save_json(out / "protocol.json", cfg)
    save_json(out / "construction_checks.json", rows)
    indices = read(parent / "training_indices.json")
    save_json(out / "training_indices.json", indices)
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    batches = {
        split: torch.stack([data.train_dataset[i][0] for i in indices[split]])
        for split in ["calibration", "confirmation"]
    }
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
        reference = original if beta == 1 else rescale(original, anchor, beta)
        observed, _ = capture(reference, batches["calibration"], cfg["layers"])
        for layer in cfg["layers"]:
            assert torch.equal(observed[f"up:{layer}"], captured[name][f"up:{layer}"])
    functions = {}
    for name, model in [
        ("release", release),
        ("public_original", cleaned_original),
        ("public_protected", cleaned_release),
    ]:
        functions[name] = {}
        for path, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
            before = original if beta == 1 else rescale(original, anchor, beta)
            after = model if beta == 1 else rescale(model, anchor, beta)
            functions[name][path] = {
                split: response(before, after, images) for split, images in batches.items()
            }
    save_json(out / "functions.json", functions)
    helpers = [
        "neuron_projection.py",
        "path_subspace.py",
        "local_erasure.py",
        "critical_subspace.py",
        "functional_contrast.py",
        "clip_benchmark.py",
        "eight_task_benchmark.py",
        "subspace_probe.py",
        "mechanisms.py",
        "operator_diagnostics.py",
        "nufilt_alignment.py",
        "fine_subspace.py",
        "joint_posthoc.py",
        "polynomial_probe.py",
        "score_recovery.py",
        "low_activity_posthoc.py",
    ]
    inputs = [
        args.config,
        str(parent / "metadata.json"),
        str(parent / "validation.json"),
        str(parent / "training_indices.json"),
        str(captured_path),
        cfg["fusion_config"],
        "configs/checkpoints.json",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            stage="construction_screen_only",
            defense_success_claimed=False,
            future_partner_loaded=False,
            labels_used=False,
            optimizer_steps=0,
            response_reconstruction_fits=0,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path("experiments") / n) for n in helpers},
            input_sha256={n: digest(n) for n in inputs},
            checkpoint_sha256=checkpoints,
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/anchor_null_reflection.json")
    parser.add_argument("--output", default="outputs/2026-10-11/anchor_null_reflection")
    run(parser.parse_args())
