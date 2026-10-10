"""Test local MLP-to-Attention task-response relocation without gradient updates.

For teacher block output Y, the public MLP M0 requires a post-attention state U
satisfying U + M0(LN2(U)) = Y. Solve this activation equation numerically, then
fit the existing Attention output projection to U-X. The released MLP is public;
its original task parameters are not available to an FC/MLP reset.

A task-vector-complement Value perturbation is calibrated on the target alone.
Both activation inversion and linear fitting need independent utility checks.
The closed-form solver is existing work; this probe makes no novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import author_recover, merge_all
from joint_posthoc import capture_blocks, digest, flatten_tokens
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, replace_components
from polynomial_probe import ridge_update
from subspace_probe import make_loader, save_json


@torch.no_grad()
def capture_inputs(model, images, layer):
    block = model.model.visual.transformer.resblocks[layer]
    x, z = [], []
    handles = [
        block.register_forward_pre_hook(lambda _m, a: x.append(flatten_tokens(a[0]))),
        block.ln_1.register_forward_hook(lambda _m, _a, y: z.append(flatten_tokens(y))),
    ]
    model.cuda().eval()
    try:
        for batch in images.split(8):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return torch.cat(x), torch.cat(z)


@torch.no_grad()
def head_features(block, inputs, image_count):
    """Use PyTorch's unchanged attention kernel with an identity output projection."""
    tokens = len(inputs) // image_count
    assert tokens * image_count == len(inputs)
    normed = inputs.reshape(image_count, tokens, -1)
    block.cuda().eval()
    width = inputs.shape[-1]
    values = []
    for batch in normed.split(8):
        q = batch.permute(1, 0, 2).cuda()
        h, _ = F.multi_head_attention_forward(
            q,
            q,
            q,
            width,
            block.attn.num_heads,
            block.attn.in_proj_weight,
            block.attn.in_proj_bias,
            block.attn.bias_k,
            block.attn.bias_v,
            block.attn.add_zero_attn,
            0.0,
            torch.eye(width, device="cuda"),
            None,
            training=False,
            need_weights=False,
            attn_mask=None,
        )
        # Check that the extracted representation reconstructs the actual branch.
        actual = block.attention(q)
        rebuilt = F.linear(h, block.attn.out_proj.weight, block.attn.out_proj.bias)
        assert float((actual - rebuilt).norm() / actual.norm()) < 1e-6
        values.append(flatten_tokens(h))
    block.cpu()
    return torch.cat(values)


@torch.no_grad()
def value_perturbation(block, public, inputs, config, svd):
    width = inputs.shape[-1]
    x = inputs.double().cuda()
    mean = x.mean(0)
    centered = x - mean
    w = block.attn.in_proj_weight[2 * width :].double().cuda()
    b = block.attn.in_proj_bias[2 * width :].double().cuda()
    delta = w - public.attn.in_proj_weight[2 * width :].double().cuda()
    u, s, v = svd(delta, full_matrices=False)
    rank = (
        int(torch.searchsorted(s.square().cumsum(0) / s.square().sum(), config["task_energy"])) + 1
    )
    task = v[:, :rank]
    # Explicitly exclude the known LayerNorm affine-null direction.
    null = block.ln_1.weight.double().cuda().reciprocal()
    null = null - task @ (task.T @ null)
    span = (
        torch.cat([task, (null / null.norm())[:, None]], dim=1)
        if float(null.norm()) > 1e-8
        else task
    )
    q = torch.linalg.qr(span, mode="complete").Q[:, span.shape[1] :]
    if q.shape[1] < config["noise_rank"]:
        raise RuntimeError("Insufficient nontrivial task complement; fixed candidate stopped.")
    cov = (centered @ q).T @ (centered @ q) / len(x)
    _, vectors = torch.linalg.eigh(cov)
    incoming = q @ vectors[:, : config["noise_rank"]]
    noise = u[:, : config["noise_rank"]] @ incoming.T
    response = centered @ noise.T
    gain = config["own_value_response_ratio"] * (x @ w.T + b).norm() / response.norm()
    perturbation = gain * noise
    leakage = float((noise @ task).norm() / noise.norm())
    ln_null = block.ln_1.weight.double().cuda().reciprocal()
    affine_null_leakage = float((noise @ ln_null).norm() / (noise.norm() * ln_null.norm()))
    assert leakage < 1e-8 and affine_null_leakage < 1e-8
    block.attn.in_proj_weight[2 * width :].add_(perturbation.float().cpu())
    block.attn.in_proj_bias[2 * width :].sub_((perturbation @ mean).float().cpu())
    return dict(
        task_rank=rank,
        task_energy_retained=float(s[:rank].square().sum() / s.square().sum()),
        noise_rank=config["noise_rank"],
        task_span_leakage=leakage,
        affine_null_leakage=affine_null_leakage,
        gain=float(gain),
        value_weight_relative_edit=float(perturbation.norm() / w.norm()),
    ), dict(
        task_basis=task.cpu(),
        incoming=incoming.cpu(),
        perturbation=perturbation.cpu(),
        mean=mean.cpu(),
    )


@torch.no_grad()
def invert_mlp_residual(block, output, initial, config):
    """A bounded activation solve, not weight optimization or a convergence theorem."""
    block.cuda().eval()
    result, evidence = [], []
    for target, start in zip(output.split(512), initial.split(512), strict=True):
        y, current = target.cuda(), start.cuda()
        for step in range(config["inverse_iterations"]):
            residual = current + block.mlp(block.ln_2(current)) - y
            error = float((residual.norm(dim=1) / y.norm(dim=1).clamp_min(1e-8)).max())
            if error <= config["inverse_tolerance"]:
                break
            current -= config["inverse_damping"] * residual
        residual = current + block.mlp(block.ln_2(current)) - y
        maximum = float((residual.norm(dim=1) / y.norm(dim=1).clamp_min(1e-8)).max())
        evidence.append(dict(iterations=step + 1, maximum_relative_residual=maximum))
        result.append(current.cpu())
    block.cpu()
    converged = all(
        item["maximum_relative_residual"] <= config["inverse_tolerance"] for item in evidence
    )
    return torch.cat(result), dict(converged=converged, chunks=evidence)


@torch.no_grad()
def build(teacher, anchor, images, config, output, svd, *, repair=False):
    model = copy.deepcopy(teacher).cpu().eval()
    targets = capture_blocks(teacher, images, config["layers"])
    for layer in config["layers"]:
        block = model.model.visual.transformer.resblocks[layer]
        public = anchor.model.visual.transformer.resblocks[layer]
        if repair:
            block.attn.in_proj_weight.copy_(public.attn.in_proj_weight)
            block.attn.in_proj_bias.copy_(public.attn.in_proj_bias)
            block.attn.out_proj.load_state_dict(public.attn.out_proj.state_dict())
        else:
            block.mlp.load_state_dict(public.mlp.state_dict(), strict=True)
    records = []
    for layer in config["layers"]:
        block = model.model.visual.transformer.resblocks[layer]
        public = anchor.model.visual.transformer.resblocks[layer]
        x, z = capture_inputs(model, images, layer)
        if not repair:
            noise, geometry = value_perturbation(block, public, z, config, svd)
        else:
            noise, geometry = {}, {}
        h = head_features(block, z, len(images))
        output_projection = block.attn.out_proj
        initial = x + F.linear(h, output_projection.weight, output_projection.bias)
        target = targets[layer]["output"]
        desired, inversion = invert_mlp_residual(block, target, initial, config)
        if not inversion["converged"]:
            save_json(output / f"layer_{layer}_inverse_failure.json", inversion)
            raise RuntimeError(f"Residual inverse failed at layer {layer}; no fitting or retuning.")
        design = torch.cat([h.double(), torch.ones(len(h), 1, dtype=torch.float64)], dim=1).cuda()
        before = torch.cat(
            [output_projection.weight.T.double(), output_projection.bias[None].double()]
        ).cuda()
        wanted = (desired - x).double().cuda()
        update, fit = ridge_update(
            design, wanted - design @ before, config["ridge_ratio"], solver="primal"
        )
        output_projection.weight.add_(update[:-1].T.float().cpu())
        output_projection.bias.add_(update[-1].float().cpu())
        fit_error = float((design @ (before + update) - wanted).norm() / wanted.norm())
        torch.save(
            dict(
                design=design.cpu(),
                target=wanted.cpu(),
                before=before.cpu(),
                update=update.cpu(),
                inversion_target=target,
                inversion_solution=desired,
                block_input=x,
                **geometry,
            ),
            output / f"layer_{layer}_fit.pt",
        )
        realized = capture_blocks(model, images, [layer])[layer]["output"]
        records.append(
            dict(
                layer=layer,
                noise=noise,
                inversion=inversion,
                fit=fit,
                fitted_attention_relative_error=fit_error,
                calibration_block_relative_error=float((realized - target).norm() / target.norm()),
            )
        )
        save_json(output / "fitting.json", records)
        print(
            "TASK_RELOCATION_LAYER",
            layer,
            fit_error,
            records[-1]["calibration_block_relative_error"],
            flush=True,
        )
        del design, before, wanted, update
    return model, records


@torch.no_grad()
def run(args):
    config = json.loads(Path(args.config).read_bytes())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", config)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    package = types.ModuleType("dataset")
    package.__path__ = [str((Path(args.author_repo) / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    root = Path(args.checkpoints)
    task = config["task"]
    original = load(str(root / task / "finetuned.pt"))
    anchor = load(str(root / "zeroshot.pt"))
    head = load(str(root / f"head_{task}.pt"))
    parent = Path(config["support_parent"])
    choice = json.loads((parent / "selection.json").read_bytes())[task]
    decision = json.loads((parent / "decision.json").read_bytes())[task]
    assert choice["layers"] == config["layers"]
    assert all(
        decision[k]
        for k in [
            "selection_gate_passed",
            "confirmation_task_gate_passed",
            "necessary_support_gate_passed",
            "full_block_gate_passed",
        ]
    )
    dataset = get_dataset(task, original.val_preprocess, args.data).train_dataset
    prior = json.loads((parent / "training_indices.json").read_bytes())[task]
    excluded = set(prior["selection"]) | set(prior["confirmation"])
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    available = [i for i in order if i not in excluded]
    n, m, k = (
        config["calibration_samples"],
        config["confirmation_samples"],
        config["attack_samples"],
    )
    indices = dict(
        calibration=available[:n],
        confirmation=available[n : n + m],
        attack=available[n + m : n + m + k],
    )
    assert len(indices["attack"]) == k
    save_json(out / "training_indices.json", indices)
    images = torch.stack([dataset[i][0] for i in indices["calibration"]])
    svd = author_svd(
        args.nufilt_repo, json.loads(Path("configs/nufilt_alignment.json").read_bytes())
    )
    status = dict(
        completed=False,
        candidate_valid=False,
        partner_used_for_construction=False,
        optimizer_steps=0,
    )
    try:
        candidate, fitting = build(original, anchor, images, config, artifacts, svd)
    except RuntimeError as error:
        status.update(completed=True, construction_gate_passed=False, reason=str(error))
        save_json(out / "decision.json", status)
        raise
    status["construction_gate_passed"] = True
    release = artifacts / "release.pt"
    torch.save(candidate.state_dict(), release)
    confirmation = make_loader(dataset, indices["confirmation"])
    rows, predictions = [], {}

    def record(preset, group, model):
        p, y = [], []
        score = evaluate(
            model,
            head,
            confirmation,
            "cuda",
            prediction_sink=lambda a, b: (p.append(a.clone()), y.append(b.clone())),
        )
        prediction, labels = torch.cat(p), torch.cat(y)
        reference = predictions.get(
            ("unprotected", group), dict(prediction=prediction, labels=labels)
        )
        assert torch.equal(labels, reference["labels"])
        rows.append(
            dict(
                preset=preset,
                group=group,
                paired=paired_counts(reference["prediction"].eq(labels), prediction.eq(labels)),
                **score,
            )
        )
        predictions[preset, group] = dict(prediction=prediction, labels=labels)
        save_json(out / "results.json", rows)
        torch.save(predictions, artifacts / "predictions.pt")
        print("TASK_RELOCATION_RESULT", preset, group, score["correct"], score["n"], flush=True)
        return score

    baseline = record("unprotected", "standalone", original)
    utility = record("relocation", "standalone", candidate)
    loss = 100 * (baseline["top1"] - utility["top1"])
    status.update(utility_loss_pp=loss, utility_gate_passed=loss <= config["utility_budget_pp"])
    if status["utility_gate_passed"]:
        # Only after the release and utility gate are frozen do future partners load.
        frozen = digest(release)
        fusion = json.loads(Path("configs/eight_tasks.json").read_bytes())
        models = {name: load(str(root / name / "finetuned.pt")) for name in fusion["tasks"]}
        models["anchor"] = anchor
        author, _ = load_author(Path(args.author_repo) / "utils_attack_B32.py")
        attack_images = torch.stack([dataset[i][0] for i in indices["attack"]])
        for preset, source in [("unprotected", original), ("relocation", candidate)]:
            for group in [
                "direct",
                "pretrained_fc",
                "pretrained_mlp",
                "pretrained_attn",
                "pretrained_attn_mlp",
                "pretrained_block",
                "aga",
                "functional_qkv_rebuild",
            ]:
                if group == "direct":
                    recovered = source
                elif group == "aga":
                    recovered = author_recover(source, anchor, author)
                elif group == "functional_qkv_rebuild":
                    destination = artifacts / f"{preset}_rebuild"
                    destination.mkdir()
                    recovered, _ = build(
                        source, anchor, attack_images, config, destination, svd, repair=True
                    )
                else:
                    component = {
                        "pretrained_fc": "mlp.c_fc",
                        "pretrained_mlp": "mlp",
                        "pretrained_attn": "attn",
                        "pretrained_attn_mlp": "attn_mlp",
                        "pretrained_block": "block",
                    }[group]
                    recovered, _ = replace_components(source, anchor, config["layers"], component)
                trial = dict(models)
                trial[task] = recovered
                merged = merge_all(trial, root, config["coefficient"])
                record(preset, group, merged)
                del trial, merged
        assert digest(release) == frozen
        best = {
            preset: max(
                (r for r in rows if r["preset"] == preset and r["group"] != "standalone"),
                key=lambda r: r["top1"],
            )
            for preset in ["unprotected", "relocation"]
        }
        gap = 100 * (best["unprotected"]["top1"] - best["relocation"]["top1"])
        status.update(
            best_measured_attack=best,
            best_recovery_gap_pp=gap,
            candidate_valid=gap >= config["minimum_recovery_gap_pp"],
        )
    torch.save(predictions, artifacts / "predictions.pt")
    status["completed"] = True
    save_json(out / "decision.json", status)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "joint_posthoc.py",
                    "mechanisms.py",
                    "nufilt_alignment.py",
                    "operator_diagnostics.py",
                    "polynomial_probe.py",
                    "subspace_probe.py",
                ]
            },
            checkpoint_sha256={
                name: digest(root / name)
                for name in ["zeroshot.pt", f"{task}/finetuned.pt", f"head_{task}.pt"]
            },
            indices_sha256=digest(out / "training_indices.json"),
            release_sha256=digest(release),
            predictions_sha256=digest(artifacts / "predictions.pt"),
            fits=len(fitting),
            optimizer_steps=0,
            calibration_labels_used=False,
            task=task,
            data=args.data,
            author_repo=args.author_repo,
            support_selection_sha256=digest(parent / "selection.json"),
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    print("TASK_RELOCATION_COMPLETE", json.dumps(status), flush=True)


@torch.no_grad()
def root_diagnostic(args):
    """Audit two activation solvers without fitting weights or changing the candidate."""
    import numpy as np

    out = Path(args.output)
    config = json.loads((out / "protocol.json").read_text())
    assert config["task"] == "Cars" and config["layers"][0] == 9
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    original = load(str(Path(args.checkpoints) / "Cars/finetuned.pt"))
    anchor = load(str(Path(args.checkpoints) / "zeroshot.pt"))
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo) / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    indices = json.loads((out / "training_indices.json").read_text())["calibration"][:8]
    data = get_dataset("Cars", original.val_preprocess, args.data).train_dataset
    images = torch.stack([data[i][0] for i in indices])
    values = capture_blocks(original, images, [9], details=True)[9]
    y = values["output"].double().cuda()
    initial = values["post_attention"].double()
    block = original.model.visual.transformer.resblocks[9]
    mlp = copy.deepcopy(anchor.model.visual.transformer.resblocks[9].mlp).double().cuda().eval()
    gamma, beta = block.ln_2.weight.double().cuda(), block.ln_2.bias.double().cuda()
    eps = block.ln_2.eps
    # Existing SciPy solver, applied to all 400 token states.
    from scipy.optimize import NoConvergence, anderson

    normalizer = y.norm(dim=1).cpu().numpy()
    calls = 0

    def residual(value):
        nonlocal calls
        calls += 1
        x = torch.as_tensor(value, dtype=torch.float64, device="cuda")
        z = F.layer_norm(x, [x.shape[-1]], gamma, beta, eps)
        return (x + mlp(z) - y).cpu().numpy()

    def norm(value):
        return float(
            np.max(np.linalg.norm(np.asarray(value).reshape(len(y), -1), axis=1) / normalizer)
        )

    started = time.perf_counter()
    converged = True
    try:
        result = anderson(
            residual, initial.numpy(), alpha=-1.0, M=5, maxiter=64, f_tol=1e-5, tol_norm=norm
        )
    except NoConvergence as error:
        converged = False
        result = error.args[0]
    record = dict(
        images=8,
        tokens=len(y),
        solver="scipy.optimize.anderson",
        maxiter=64,
        memory=5,
        alpha=-1.0,
        converged=converged,
        maximum_relative_residual=norm(residual(result)),
        function_evaluations=calls,
        seconds=time.perf_counter() - started,
        weight_fits=0,
        optimizer_steps=0,
        source_sha256=digest(__file__),
        sample_indices=indices,
        scope="Activation equation only. No candidate or classification result.",
    )
    raw_problem = out / "artifacts/root_problem.pt"
    torch.save(
        dict(
            initial=initial,
            target=y.cpu(),
            anderson_solution=torch.as_tensor(result),
            gamma=gamma.cpu(),
            beta=beta.cpu(),
            epsilon=eps,
        ),
        raw_problem,
    )
    record["raw_problem_sha256"] = digest(raw_problem)
    save_json(out / "root_diagnostic.json", record)
    print("ANDERSON_ROOT_DIAGNOSTIC", json.dumps(record), flush=True)
    started = time.perf_counter()
    u = initial.reshape(8, 50, -1)[:, 0].cuda()
    y = y.reshape(8, 50, -1)[:, 0]
    fc, proj = mlp.c_fc, mlp.c_proj
    assert isinstance(mlp.ln, torch.nn.Identity)
    kind = type(mlp.gelu).__name__
    trace = []
    for step in range(16):
        c = u - u.mean(1, keepdim=True)
        scale = (c.square().mean(1, keepdim=True) + eps).sqrt()
        z = c / scale * gamma + beta
        pre = F.linear(z, fc.weight, fc.bias)
        if kind == "QuickGELU":
            sigmoid = torch.sigmoid(1.702 * pre)
            derivative = sigmoid + 1.702 * pre * sigmoid * (1 - sigmoid)
        elif kind == "GELU":
            derivative = (
                0.5 * (1 + torch.erf(pre / (2.0**0.5)))
                + pre * torch.exp(-0.5 * pre.square()) / (2 * np.pi) ** 0.5
            )
        else:
            raise RuntimeError(kind)
        residual_value = u + mlp(z) - y
        error = residual_value.norm(dim=1) / y.norm(dim=1)
        trace.append({"step": step, "maximum_relative_residual": float(error.max())})
        if float(error.max()) <= 1e-5:
            break
        # Analytic MLP-LayerNorm Jacobian; float32 matrix products, double linear solve.
        b = (
            (proj.weight.float()[None] * derivative.float()[:, None, :]) @ fc.weight.float()[None]
        ).double()
        b = b * gamma[None, None, :] / scale[:, None, :]
        j = (
            torch.eye(u.shape[1], device="cuda", dtype=torch.float64)[None]
            + b
            - b.sum(2, keepdim=True) / u.shape[1]
        )
        j -= ((b @ c[:, :, None]) @ c[:, None, :]) / (u.shape[1] * scale[:, :, None].square())
        du = torch.linalg.solve(j, -residual_value[:, :, None]).squeeze(-1)
        accepted = torch.zeros(len(u), device="cuda", dtype=torch.bool)
        best = u.clone()
        for power in range(12):
            trial = u + (2.0 ** (-power)) * du
            normalized = F.layer_norm(trial, [trial.shape[-1]], gamma, beta, eps)
            trial_error = (trial + mlp(normalized) - y).norm(dim=1) / y.norm(dim=1)
            good = (trial_error < error) & ~accepted
            best[good] = trial[good]
            accepted |= good
        u = best
        if not bool(accepted.all()):
            break
    z = F.layer_norm(u, [u.shape[-1]], gamma, beta, eps)
    final = (u + mlp(z) - y).norm(dim=1) / y.norm(dim=1)
    # Verify the analytic Jacobian, then measure conditioning at the stalled states.
    c = u - u.mean(1, keepdim=True)
    scale = (c.square().mean(1, keepdim=True) + eps).sqrt()
    z = c / scale * gamma + beta
    pre = F.linear(z, fc.weight, fc.bias)
    if kind == "QuickGELU":
        sigmoid = torch.sigmoid(1.702 * pre)
        derivative = sigmoid + 1.702 * pre * sigmoid * (1 - sigmoid)
    else:
        derivative = (
            0.5 * (1 + torch.erf(pre / (2.0**0.5)))
            + pre * torch.exp(-0.5 * pre.square()) / (2 * np.pi) ** 0.5
        )
    b = (
        (proj.weight.float()[None] * derivative.float()[:, None, :]) @ fc.weight.float()[None]
    ).double()
    b = b * gamma[None, None, :] / scale[:, None, :]
    j = (
        torch.eye(u.shape[1], device="cuda", dtype=torch.float64)[None]
        + b
        - b.sum(2, keepdim=True) / u.shape[1]
    )
    j -= ((b @ c[:, :, None]) @ c[:, None, :]) / (u.shape[1] * scale[:, :, None].square())
    sv = torch.linalg.svdvals(j)
    generator = torch.Generator().manual_seed(20261015)
    v = torch.randn(u.shape, generator=generator, dtype=torch.float64).cuda()
    v /= v.norm(dim=1, keepdim=True)

    def point_residual(x):
        return x + mlp(F.layer_norm(x, [x.shape[-1]], gamma, beta, eps)) - y

    fd = (point_residual(u + 1e-4 * v) - point_residual(u - 1e-4 * v)) / (2e-4)
    analytic = (j @ v[:, :, None]).squeeze(-1)
    jacobian_error = float((analytic - fd).norm() / fd.norm())
    assert jacobian_error < 1e-5, jacobian_error
    record = {
        "images": 8,
        "tokens": 8,
        "token_scope": "CLS only",
        "solver": "Batched analytic Newton with backtracking residual decrease",
        "maxiter": 16,
        "converged": bool((final <= 1e-5).all()),
        "maximum_relative_residual": float(final.max()),
        "per_token_residual": final.cpu().tolist(),
        "trace": trace,
        "minimum_singular_values": sv[:, -1].cpu().tolist(),
        "condition_numbers": (sv[:, 0] / sv[:, -1]).cpu().tolist(),
        "directional_jacobian_relative_error": jacobian_error,
        "seconds": time.perf_counter() - started,
        "weight_fits": 0,
        "optimizer_steps": 0,
        "scope": "Activation inverse diagnostic only. Analytic Jacobian uses input derivatives, no parameter gradients or weight updates.",
    }
    record.update(
        source_sha256=digest(__file__),
        sample_indices=indices,
        checkpoint_sha256={
            name: digest(Path(args.checkpoints) / name)
            for name in ["Cars/finetuned.pt", "zeroshot.pt"]
        },
    )
    raw_newton = out / "artifacts/newton_diagnostic.pt"
    torch.save(
        dict(
            solution=u.cpu(),
            target=y.cpu(),
            jacobian=j.cpu(),
            finite_difference=fd.cpu(),
            direction=v.cpu(),
            analytic_direction=analytic.cpu(),
        ),
        raw_newton,
    )
    record["raw_newton_sha256"] = digest(raw_newton)
    save_json(out / "newton_root_diagnostic.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-diagnostic", action="store_true")
    parser.add_argument("--config", default="configs/task_relocation_cars.json")
    parser.add_argument("--output", default="outputs/2026-10-10/task_relocation_cars")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--nufilt-repo", default="external/NUFILT")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    (root_diagnostic if args.root_diagnostic else run)(args)
