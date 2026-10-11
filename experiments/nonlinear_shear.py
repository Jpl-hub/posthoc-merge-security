"""Target-only nilpotent MLP shear selected with a nonlinear output Gram.

Save one fixed release and its explicit public deletion attack. No labels,
future fusion partner, optimizer, gradient or response reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import time
from pathlib import Path

import torch
from anchor_null_reflection import direction
from critical_subspace import capture
from eight_task_benchmark import digest
from functional_contrast import features
from local_erasure import rescale
from neuron_projection import activation
from path_subspace import context, read
from scipy.linalg import eigh
from subspace_probe import save_json


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def activation_prime(z):
    p = torch.sigmoid(1.702 * z)
    return p + 1.702 * z * p * (1 - p)


@torch.no_grad()
def spaces(parts, rcond):
    c = torch.cat([p.double().cuda() / p.double().norm() for p in parts], dim=1)
    q, r = torch.linalg.qr(c, mode="complete")
    singular = torch.linalg.svdvals(r[: c.shape[1]])
    assert float(singular[-1] / singular[0]) > rcond
    span, null = q[:, : c.shape[1]], q[:, c.shape[1] :]
    error = float((c.T @ null).norm() / c.norm())
    assert error < 1e-10
    return (
        span,
        null,
        dict(
            rank=c.shape[1],
            null_dimension=null.shape[1],
            null_relative_error=error,
            smallest_relative_singular=float(singular[-1] / singular[0]),
        ),
    )


@torch.no_grad()
def output_grams(native, base, tensors, v, null, cfg):
    grams, raw, scales = {}, {}, {}
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
        z, h = tensors[name]
        down = (
            1 - beta
        ) * base.c_proj.weight.double().cuda() + beta * native.c_proj.weight.double().cuda()
        k = (z @ v)[:, None] * activation_prime(z) - (h @ v)[:, None]
        g = (down.T @ down) * (k.T @ k / len(k))
        scale = float((h @ down.T).square().sum() / len(h))
        projected = null.T @ g @ null / scale
        grams[name] = (projected + projected.T) / 2
        raw[name], scales[name] = g, scale
    native_gram = grams["expert"]
    ridge = cfg["gram_ridge_ratio"] * float(native_gram.trace()) / len(native_gram)
    den = native_gram + ridge * torch.eye(len(native_gram), dtype=torch.float64, device="cuda")
    path = sum(grams[str(t)] for t in cfg["betas"]) / len(cfg["betas"])
    values, vectors = eigh(
        path.cpu().numpy(), den.cpu().numpy(), subset_by_index=[len(den) - 1] * 2
    )
    coef = torch.from_numpy(vectors[:, 0].copy()).cuda()
    u = null @ coef
    u /= u.norm()
    u *= 1 if float(u[u.abs().argmax()]) >= 0 else -1
    coef = null.T @ u
    error = float(
        (path @ coef - float(values[0]) * den @ coef).norm() / (path @ coef).norm().clamp_min(1e-30)
    )
    assert error < 1e-7 and float(abs(u @ v)) < 1e-10
    proof = dict(
        u=u.cpu(),
        eigenvalue=float(values[0]),
        ridge=ridge,
        native_gram_sha256=tensor_digest(native_gram),
        path_gram_sha256=tensor_digest(path),
    )
    report = dict(
        generalized_eigenvalue=float(values[0]),
        eigen_residual=error,
        normalized_native_energy=float(coef @ native_gram @ coef),
        normalized_path_energy=float(coef @ path @ coef),
        ridge_energy=ridge,
    )
    return u, raw, scales, proof, report


@torch.no_grad()
def compile_shear(mlp, u, v):
    result = copy.deepcopy(mlp).cpu().eval()
    w, b, down = [p.double().cuda() for p in [mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight]]
    result.c_fc.weight.copy_((w + u[:, None] * (v @ w)[None, :]).float().cpu())
    result.c_fc.bias.copy_((b + u * (v @ b)).float().cpu())
    result.c_proj.weight.copy_((down - (down @ u)[:, None] * v[None, :]).float().cpu())
    return result


@torch.no_grad()
def public_cross_span(mlp, base, input_span, output_span):
    result = copy.deepcopy(mlp).cpu().eval()
    w, b, down = [p.double().cuda() for p in [mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight]]
    w0, b0, v0 = [p.double().cuda() for p in [base.c_fc.weight, base.c_fc.bias, base.c_proj.weight]]
    pw, pv = input_span.double().cuda(), output_span.double().cuda()
    result.c_fc.weight.copy_((w0 + pv @ (pv.T @ (w - w0))).float().cpu())
    result.c_fc.bias.copy_((b0 + pv @ (pv.T @ (b - b0))).float().cpu())
    result.c_proj.weight.copy_((v0 + ((down - v0) @ pw) @ pw.T).float().cpu())
    return result


@torch.no_grad()
def local_checks(native, base, changed, captured, tensors, layer, u, v, grams, scales, betas):
    identity = torch.eye(len(u), dtype=torch.float64, device="cuda")
    t, inverse = identity + u[:, None] * v[None, :], identity - u[:, None] * v[None, :]
    assert float((t @ inverse - identity).abs().max()) < 1e-10
    rows = {}
    second_bound = 1.702 * (0.5 + 1 / math.e)
    for name, beta in [("expert", 1.0)] + [(str(b), b) for b in betas]:
        z, h = tensors[name]
        down = (
            1 - beta
        ) * base.c_proj.weight.double().cuda() + beta * native.c_proj.weight.double().cuda()
        s, prime = z @ v, activation_prime(z)
        after = activation(z + s[:, None] * u[None, :])
        actual = (after - h) @ down.T - (after @ v)[:, None] * (down @ u)[None, :]
        direct = activation(z @ t.T) @ (down @ inverse).T - h @ down.T
        algebra_error = float((actual - direct).abs().max())
        assert algebra_error < 1e-10
        k = s[:, None] * prime - (h @ v)[:, None]
        linear = (k * u) @ down.T
        predicted = float(u @ grams[name] @ u)
        observed = float(linear.square().sum() / len(linear))
        gram_error = abs(predicted - observed) / max(observed, 1e-30)
        assert gram_error < 1e-8
        rem = (actual - linear).norm(dim=1)
        bound = (down.norm() + (down @ u).norm()) * (
            0.5 * second_bound * s.square() * u.square().norm()
        )
        bound += (down @ u).norm() * s.abs() * ((prime * u) @ v).abs()
        assert bool((rem <= bound + 1e-10).all())
        x = captured[name][f"up:{layer}"].double().cuda()
        w = (
            1 - beta
        ) * base.c_fc.weight.double().cuda() + beta * changed.c_fc.weight.double().cuda()
        b = (1 - beta) * base.c_fc.bias.double().cuda() + beta * changed.c_fc.bias.double().cuda()
        d = (
            1 - beta
        ) * base.c_proj.weight.double().cuda() + beta * changed.c_proj.weight.double().cuda()
        compiled = activation(x @ w.T + b) @ d.T - h @ down.T
        rows[name] = dict(
            local_relative_rms=float(
                compiled.square().sum().div(len(compiled)).div(scales[name]).sqrt()
            ),
            linear_relative_rms=(observed / scales[name]) ** 0.5,
            nonlinear_remainder_ratio=float(
                (actual - linear).norm() / actual.norm().clamp_min(1e-30)
            ),
            gram_relative_error=gram_error,
            algebra_max_absolute_error=algebra_error,
            taylor_bound_verified=True,
            max_taylor_bound=float(bound.max()),
            compiled_rounding_relative_error=float(
                (compiled - actual).norm() / actual.norm().clamp_min(1e-30)
            ),
        )
    return rows


@torch.no_grad()
def construct(original, anchor, captured, cfg):
    release, a, b = [copy.deepcopy(original).cpu().eval() for _ in range(3)]
    evidence, rows = {}, []
    for layer in cfg["layers"]:
        native = original.model.visual.transformer.resblocks[layer].mlp
        base = anchor.model.visual.transformer.resblocks[layer].mlp
        pw, nv, vspace = spaces(
            [base.c_fc.weight, base.c_fc.bias[:, None]], cfg["anchor_rank_rcond"]
        )
        v, tensors, ve, vcheck = direction(native, base, captured, layer, cfg, nv)
        joined, nu, uspace = spaces([base.c_proj.weight.T, v[:, None]], cfg["anchor_rank_rcond"])
        pv = joined[:, : base.c_proj.out_features]
        u, grams, scales, ue, ucheck = output_grams(native, base, tensors, v, nu, cfg)
        changed = compile_shear(native, u, v)
        release.model.visual.transformer.resblocks[layer].mlp = changed
        ca, cb = [public_cross_span(x, base, pw, pv) for x in [native, changed]]
        (
            a.model.visual.transformer.resblocks[layer].mlp,
            b.model.visual.transformer.resblocks[layer].mlp,
        ) = ca, cb
        deletion_error = max(
            float((ca.state_dict()[k] - cb.state_dict()[k]).abs().max()) for k in ca.state_dict()
        )
        assert deletion_error < 2e-6
        local = local_checks(
            native, base, changed, captured, tensors, layer, u, v, grams, scales, cfg["betas"]
        )
        ve = {k: value for k, value in ve.items() if not k.endswith("gram")}
        evidence[str(layer)] = dict(v=v.cpu(), **ue, v_selection=ve)
        row = dict(
            layer=layer,
            v_space=vspace,
            u_space=uspace,
            v_selection=vcheck,
            u_selection=ucheck,
            local=local,
            public_deletion_state_max_absolute_error=deletion_error,
        )
        rows.append(row)
        print("SHEAR", row, flush=True)
    return dict(release=release, public_original=a, public_protected=b), evidence, rows


@torch.no_grad()
def screen(original, anchor, models, head, batches, cfg):
    rows, tensors = {}, {}
    for path, beta in [("expert", 1.0)] + [(str(b), b) for b in cfg["betas"]]:
        before = original if beta == 1 else rescale(original, anchor, beta)
        before.cuda().eval()
        cached = {split: features(before, images) for split, images in batches.items()}
        before.cpu()
        for split, value in cached.items():
            tensors[f"original/{path}/{split}"] = value
        for name, model in models.items():
            rows.setdefault(name, {})[path] = {}
            after = model if beta == 1 else rescale(model, anchor, beta)
            after.cuda().eval()
            for split, images in batches.items():
                baseline, value = cached[split], features(after, images)
                weight = head.weight.detach().float().cpu()
                old, new = baseline @ weight.T, value @ weight.T
                rows[name][path][split] = dict(
                    actual_feature_rms=float(
                        (value - baseline).double().square().sum().div(len(images)).sqrt()
                    ),
                    target_head_changed_predictions=int(old.argmax(-1).ne(new.argmax(-1)).sum()),
                    target_head_score_rms=float(
                        (new - old).double().square().sum().div(len(images)).sqrt()
                    ),
                    images=len(images),
                    labels_used=False,
                )
                tensors[f"{name}/{path}/{split}"] = value
            after.cpu()
    return rows, tensors


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    assert cfg["rank"] == cfg["amplitude"] == 1
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = Path(cfg["parent"])
    meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    captured_path = parent / "artifacts/inputs.pt"
    assert digest(captured_path) == meta["raw_sha256"]["inputs.pt"]
    captured = torch.load(captured_path, map_location="cpu", weights_only=True)
    root, load, _, holders = context(cfg)
    original, anchor = (
        load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval(),
        holders["anchor"],
    )
    pins = {
        str(p.relative_to(root)): digest(p)
        for p in [root / "zeroshot.pt", root / cfg["task"] / "finetuned.pt"]
    }
    assert pins == meta["checkpoint_sha256"]
    models, proof, checks = construct(original, anchor, captured, cfg)
    state = original.state_dict()
    for name, model in models.items():
        edits = {
            k: v.clone() for k, v in model.state_dict().items() if not torch.equal(v, state[k])
        }
        assert all(".mlp.c_fc." in k or k.endswith(".mlp.c_proj.weight") for k in edits)
        torch.save(edits, artifacts / f"{name}_edits.pt")
    torch.save(proof, artifacts / "directions.pt")
    save_json(out / "protocol.json", cfg)
    save_json(out / "construction_checks.json", checks)
    indices = read(parent / "training_indices.json")
    save_json(out / "training_indices.json", indices)
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    batches = {
        split: torch.stack([data.train_dataset[i][0] for i in indices[split]])
        for split in ["calibration", "confirmation"]
    }
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
        model = original if beta == 1 else rescale(original, anchor, beta)
        observed, _ = capture(model, batches["calibration"], cfg["layers"])
        assert all(torch.equal(observed[k], value) for k, value in captured[name].items())
    head_path = root / f"head_{cfg['task']}.pt"
    pins[str(head_path.relative_to(root))] = digest(head_path)
    all_pins = {
        x["path"].split("/", 1)[1]: x["sha256"] for x in read("configs/checkpoints.json")["files"]
    }
    assert all(all_pins[k] == value for k, value in pins.items())
    values, feature_values = screen(original, anchor, models, load(str(head_path)), batches, cfg)
    torch.save(feature_values, artifacts / "features.pt")
    save_json(out / "functions.json", values)
    helpers = [
        "anchor_null_reflection.py",
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
            target_head_loaded_after_weight_freeze=True,
            optimizer_steps=0,
            response_reconstruction_fits=0,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path("experiments") / n) for n in helpers},
            input_sha256={n: digest(n) for n in inputs},
            checkpoint_sha256=pins,
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nonlinear_shear.json")
    parser.add_argument("--output", default="outputs/2026-10-11/nonlinear_shear")
    run(parser.parse_args())
