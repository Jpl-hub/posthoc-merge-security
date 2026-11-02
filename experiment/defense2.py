"""Defense 2: frozen-expert A/B construction on pinned synthetic inputs."""

# Preserve the three expressions from the executed source.
# ruff: noqa: E731

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.attacks import load_author
from _impl.io import digest
from _impl.metrics import logical_parameter_distance
from _impl.merge import merge
from _impl.models import apply, checked_models, initialize


def classification_violation(scores, labels):
    """Zero once another class matches or exceeds the teacher class."""
    own = scores.gather(1, labels[:, None]).flatten()
    other = scores.scatter(1, labels[:, None], -torch.inf).amax(1)
    return (own - other).clamp_min(0).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["MNIST", "SVHN", "DTD"], default="MNIST")
    parser.add_argument("--inputs", type=Path, default=Path("configs/defense2_inputs.json"))
    parser.add_argument("--calibration-aware", action="store_true")
    parser.add_argument("--task-reference", action="store_true")
    parser.add_argument("--input-bias", action="store_true")
    parser.add_argument("--noise-calibration", action="store_true")
    parser.add_argument("--finite-classification", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibration_aware and args.output is None:
        parser.error("--calibration-aware requires a separate --output directory")
    if args.task_reference and not args.calibration_aware:
        parser.error("--task-reference requires --calibration-aware")
    if args.input_bias and not args.task_reference:
        parser.error("--input-bias requires --task-reference and --calibration-aware")
    if args.noise_calibration and not args.input_bias:
        parser.error("--noise-calibration requires --input-bias")
    if args.finite_classification and not args.noise_calibration:
        parser.error("--finite-classification requires --noise-calibration")
    initialize()
    torch.set_default_dtype(torch.float64)
    start = time.perf_counter()
    inputs = json.loads(args.inputs.read_text())[args.target]
    root = Path(inputs["root"])
    out = args.output or root / "point"
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((root / "source/result.json").read_text())
    assert meta["source_arrays_sha256"] == inputs["source_sha256"]
    assert digest(root / "source/source.pt") == meta["source_arrays_sha256"]
    raw = torch.load(root / "source/source.pt", map_location="cpu", weights_only=False)
    (base, expert, head, pins) = checked_models(args.target)
    (s, bs) = (expert.state_dict(), base.state_dict())
    k = "model.visual.transformer.resblocks.11.mlp."
    to = lambda v: v.double().cuda()
    (A, A0, a, a0, B, B0, b, b0, gamma, beta) = [
        to(v)
        for v in [
            s[k + "c_fc.weight"],
            bs[k + "c_fc.weight"],
            s[k + "c_fc.bias"],
            bs[k + "c_fc.bias"],
            s[k + "c_proj.weight"],
            bs[k + "c_proj.weight"],
            s[k + "c_proj.bias"],
            bs[k + "c_proj.bias"],
            s["model.visual.transformer.resblocks.11.ln_2.weight"],
            s["model.visual.transformer.resblocks.11.ln_2.bias"],
        ]
    ]
    # Change A outside the selected task-SVD directions.
    D = A - A0
    (U, _, Vh) = torch.linalg.svd(D, full_matrices=False)
    (U, V) = (U[:, :32], Vh[:32].T)
    if args.input_bias:
        # Preserve affine preactivations at beta + span(V), while including
        # the input bias in the public incoming-weight correspondence.
        Q = torch.eye(A.shape[1], device=A.device) - V @ V.T
        muq = Q @ beta
        H = Q - muq[:, None] * muq[None, :] / (1 + muq @ muq)
        C = A0 - a0[:, None] * beta[None, :]
        A0r = C @ H
        K = A0r @ C.T
        rhs = -(A * A0).sum(1) - a * a0
    else:
        A0r = A0 - A0 @ V @ V.T
        K = A0r @ A0r.T
        rhs = -(A * A0).sum(1)
    G = -(U @ U.T) * K
    G.diagonal().add_(K.diagonal())
    lam = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(G))
    dA = lam * A0r
    dA -= U @ (U.T @ dA)
    Ap = (A + dA).float().double()
    ap = (a - dA @ beta).float().double() if args.input_bias else a
    del K, G
    (author, sha) = load_author(Path("external/secure-merge-attack/utils_attack_B32.py"))
    assert sha == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    pi = author["solve_permutation"](Ap.float().cpu(), A0.float().cpu()).cuda()
    assert sorted(pi.tolist()) == list(range(len(A)))
    Rorig = torch.cat([B, b[:, None]], 1)
    cache = {name: {key: to(v) for (key, v) in row.items()} for (name, row) in raw["cache"].items()}
    public_names = ["direct", "recovered"]
    if args.task_reference:
        # The second reference is the clean target itself, never another task.
        # Its direct half-publication prefix is exactly the native prefix.
        cache["task_direct"] = cache["native"]
        aligned = apply(expert, inputs["clean_AGA"], full=True)
        clean = merge(base, aligned, expert, .5).cuda().eval()
        v = clean.model.visual
        block = v.transformer.resblocks[11]
        rows = dict(x=[], prefix=[], final=[], scores=[])
        hooks = [
            block.mlp.c_fc.register_forward_pre_hook(lambda m, a: rows["x"].append(a[0][0].cpu())),
            block.ln_2.register_forward_pre_hook(lambda m, a: rows["prefix"].append(a[0][0].cpu())),
            v.ln_post.register_forward_pre_hook(lambda m, a: rows["final"].append(a[0].cpu())),
        ]
        norm = expert.val_preprocess.transforms[-1]
        mean = torch.tensor(norm.mean, device="cuda", dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(norm.std, device="cuda", dtype=torch.float32)[None, :, None, None]
        with torch.no_grad():
            for pixels in raw["final"].split(8):
                x = F.interpolate(pixels.cuda(), (224, 224), mode="bilinear", align_corners=False)
                images = (torch.cat([torch.roll(x, s, (2, 3)) for s in
                                    [(0, 0), (7, 0), (-7, 0), (0, 7), (0, -7)]]) - mean) / std
                rows["scores"].append((F.normalize(clean(images), dim=-1) @ head.weight.cuda().T).cpu())
        for hook in hooks:
            hook.remove()
        rows = {key: torch.cat(value) for key, value in rows.items()}
        torch.save(rows, out / "task_reference_cache.pt")
        cache["task_recovered"] = {key: to(value) for key, value in rows.items()}
        public_names += ["task_direct", "task_recovered"]
        del clean, aligned, block, v, rows
        torch.cuda.empty_cache()
    noise_cache = {}
    if args.noise_calibration:
        # Independent construction noise; the attack's noise remains evaluation-only.
        noise = torch.rand(64, 3, 32, 32, dtype=torch.float32,
                           generator=torch.Generator().manual_seed(2026110201))
        torch.save(noise, out / "construction_noise.pt")
        norm = expert.val_preprocess.transforms[-1]
        mean = torch.tensor(norm.mean, dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(norm.std, dtype=torch.float32)[None, :, None, None]
        images = (F.interpolate(noise, (224, 224), mode="bilinear", align_corners=False) - mean) / std
        aligned = apply(expert, inputs["clean_AGA"], full=True)
        for name, left, right in [("direct", expert, base), ("recovered", aligned, base),
                                  ("native", expert, expert), ("task_recovered", aligned, expert)]:
            clean = merge(base, left, right, .5) if name != "native" else copy.deepcopy(expert)
            clean = clean.cuda().eval()
            v = clean.model.visual
            rows = dict(x=[], prefix=[], final=[], scores=[])
            block = v.transformer.resblocks[11]
            hooks = [
                block.mlp.c_fc.register_forward_pre_hook(lambda m, a: rows["x"].append(a[0][0].cpu())),
                block.ln_2.register_forward_pre_hook(lambda m, a: rows["prefix"].append(a[0][0].cpu())),
                v.ln_post.register_forward_pre_hook(lambda m, a: rows["final"].append(a[0].cpu())),
            ]
            with torch.no_grad():
                for batch in images.split(8):
                    rows["scores"].append((F.normalize(clean(batch.cuda()), dim=-1) @ head.weight.cuda().T).cpu())
            for hook in hooks:
                hook.remove()
            noise_cache[name] = {key: to(torch.cat(value)) for key, value in rows.items()}
            del clean, block, v, rows
        noise_cache["task_direct"] = noise_cache["native"]
        torch.save({n: {k: v.cpu() for k, v in row.items()} for n, row in noise_cache.items()},
                   out / "construction_noise_cache.pt")
        del aligned
        torch.cuda.empty_cache()
    activate = lambda z: z * torch.sigmoid(1.702 * z)

    def augmented(x):
        return torch.cat([activate(x @ Ap.T + ap).T, torch.ones(1, len(x), device="cuda")], 0)

    # Fit B and its bias to frozen-teacher outputs on the two synthetic populations.
    G = torch.zeros(3073, 3073, device="cuda")
    T = torch.zeros_like(Rorig)
    constant = G.new_zeros(())
    groups = []
    source = Path(inputs["layer_inputs"]["path"])
    assert digest(source) == inputs["layer_inputs"]["sha256"]
    u = torch.load(source, map_location="cpu", weights_only=False)
    for name, count in [("task_layer", len(u)), ("generated_native", len(cache["native"]["x"]))]:
        gi = torch.zeros_like(G)
        ti = torch.zeros_like(T)
        yy = G.new_zeros(())
        ys = torch.zeros(768, device="cuda")
        for lo in range(0, count, 256):
            x = (
                beta + gamma * u[lo : lo + 256].cuda()
                if name == "task_layer"
                else cache["native"]["x"][lo : lo + 256]
            )
            H = augmented(x)
            Y = (
                activate(x @ A.T + a) @ B.T + b
                if name == "task_layer"
                else cache["native"]["final"][lo : lo + 256]
                - cache["native"]["prefix"][lo : lo + 256]
            )
            gi += H @ H.T
            ti += Y.T @ H.T
            yy += Y.square().sum()
            ys += Y.sum(0)
        variance = yy / count - ys.square().sum() / count**2
        assert variance > 0
        scale = 1 / (count * variance)
        G += gi * scale
        T += ti * scale
        constant += yy * scale
        groups.append(dict(population=name, n=count, variance=float(variance)))
    rho = 0.01 * G.trace() / len(G)
    if args.input_bias:
        rho = inputs["original_ridge"]  # Preserve this task's original source regularizer.
    G += rho * torch.eye(len(G), device="cuda")
    T += rho * Rorig
    constant += rho * Rorig.square().sum()
    L = torch.linalg.cholesky(G)
    RN = torch.cholesky_solve(T.T, L).T
    native = lambda R: float((R @ G * R).sum() - 2 * (R * T).sum() + constant)
    tau = native(Rorig)
    if args.input_bias:
        tau = inputs["original_native_limit"]  # Preserve this task's original bound.
    minimum = native(RN)
    radius2 = tau - minimum
    assert radius2 > 1e-12, (tau, minimum)
    # Build the public-base fusion and AGA contexts, including the full classifier.
    hidden = {}
    fixed = {}
    noise_hidden = {}
    noise_fixed = {}
    readouts = {}
    W = to(head.weight)
    assert float(head.bias.abs().max()) == 0
    eps = expert.model.visual.ln_post.eps
    for name in ["native"] + public_names:
        x = cache[name]["x"]
        if name == "native":
            H = activate(x @ Ap.T + ap).T
            fixed[name] = cache[name]["prefix"].T
            hidden[name] = torch.cat([H, torch.ones(1, H.shape[1], device="cuda")], 0)
        else:
            recovered_context = name.endswith("recovered")
            (AA, aa) = (Ap[pi], ap[pi]) if recovered_context else (Ap, ap)
            Ar, ar, Br, br = (A, a, B, b) if name.startswith("task_") else (A0, a0, B0, b0)
            H = activate(x @ ((AA + Ar) / 2).T + (aa + ar) / 2).T
            fixed[name] = cache[name]["prefix"].T + (Br @ H + br[:, None]) / 2
            if recovered_context:
                back = torch.empty_like(H)
                back[pi] = H
                H = back
            hidden[name] = torch.cat([H, torch.ones(1, H.shape[1], device="cuda")], 0) / 2
        names = ["model.visual.ln_post.weight", "model.visual.ln_post.bias", "model.visual.proj"]
        readouts[name] = [
            to(s[key] if name == "native" or name.startswith("task_") else (s[key] + bs[key]) / 2)
            for key in names
        ]
        if args.noise_calibration and name != "native":
            x = noise_cache[name]["x"]
            H = activate(x @ ((AA + Ar) / 2).T + (aa + ar) / 2).T
            noise_fixed[name] = noise_cache[name]["prefix"].T + (Br @ H + br[:, None]) / 2
            if recovered_context:
                back = torch.empty_like(H)
                back[pi] = H
                H = back
            noise_hidden[name] = torch.cat([H, torch.ones(1, H.shape[1], device="cuda")], 0) / 2

    def classify(r, params):
        (g, beta, P) = params
        z = F.layer_norm(r.T, (768,), g, beta, eps) @ P
        return F.normalize(z, dim=-1) @ W.T

    original_tail_error = max(
        (
            float(
                (classify(cache[name]["final"].T, readouts[name]) - cache[name]["scores"])
                .abs()
                .max()
            )
            for name in cache
        )
    )
    assert original_tail_error < 0.0005, original_tail_error
    noise_tail_error = None
    if args.noise_calibration:
        noise_tail_error = max(float((classify(row["final"].T, readouts[name]) -
                                      row["scores"]).abs().max()) for name, row in noise_cache.items())
        assert noise_tail_error < 0.0005, noise_tail_error

    teacher_class = raw["teacher_class"].long().cuda()
    joint_incoming_cost = A0 @ Ap.T + a0[:, None] * ap[None, :] if args.input_bias else None
    matching_calls = 1

    def calibrated(scores, reference=None):
        # Probability division, evaluated on the fixed synthetic source only.
        reference = scores if reference is None else reference
        log_prior = torch.logsumexp(F.log_softmax(reference, dim=1), dim=0) - math.log(len(reference))
        return scores - log_prior

    def public(R):
        nonlocal matching_calls
        scores = {
            name: classify(fixed[name] + R @ hidden[name], readouts[name])
            for name in public_names
        }
        noise_scores = {name: classify(noise_fixed[name] + R @ noise_hidden[name], readouts[name])
                        for name in public_names} if args.noise_calibration else {}
        if args.input_bias:
            # The exact joint MLP assignment is recomputed at every objective
            # evaluation, including backtracking. No stale order is optimized.
            from scipy.optimize import linear_sum_assignment

            with torch.no_grad():
                cost = joint_incoming_cost + B0.T @ R[:, :-1]
                order = torch.as_tensor(
                    linear_sum_assignment(cost.detach().cpu().numpy(), maximize=True)[1],
                    device=R.device,
                )
                matching_calls += 1
            for name, cached, Ar, ar, Br, br, readout in [
                ("joint_direct", cache["direct"], A0, a0, B0, b0, readouts["direct"]),
                ("joint_task", cache["native"], A, a, B, b, readouts["native"]),
            ]:
                h = activate(cached["x"] @ ((Ap[order] + Ar) / 2).T + (ap[order] + ar) / 2).T
                residual = cached["prefix"].T + (Br @ h + br[:, None]) / 2
                residual = residual + (R[:, :-1][:, order] @ h + R[:, -1, None]) / 2
                scores[name] = classify(residual, readout)
                if args.noise_calibration:
                    nc = noise_cache["direct" if name == "joint_direct" else "native"]
                    h = activate(nc["x"] @ ((Ap[order] + Ar) / 2).T + (ap[order] + ar) / 2).T
                    residual = nc["prefix"].T + (Br @ h + br[:, None]) / 2
                    residual = residual + (R[:, :-1][:, order] @ h + R[:, -1, None]) / 2
                    noise_scores[name] = classify(residual, readout)
        if args.calibration_aware:
            if args.finite_classification:
                violations = [
                    classification_violation(v, teacher_class)
                    for name, values in scores.items()
                    for v in [values, calibrated(values), calibrated(values, noise_scores[name])]
                ]
                return torch.stack(violations).mean(), scores, noise_scores
            # Reduce correct-class probability before and after calibration;
            # no fixed destination class, new images, or fitted test labels.
            correct = torch.cat(
                [
                    F.log_softmax(v, dim=1).gather(1, teacher_class[:, None]).flatten()
                    for name, values in scores.items()
                    for v in ([values, calibrated(values)] +
                              ([calibrated(values, noise_scores[name])] if args.noise_calibration else []))
                ]
            )
            return torch.logsumexp(correct, 0) - math.log(len(correct)), scores, noise_scores
        gaps = torch.cat([v[:, 1:] - v[:, :1] for v in scores.values()])
        return (torch.logsumexp(gaps.flatten(), 0) - math.log(gaps.numel()), scores, noise_scores)

    def source_accuracy(R):
        with torch.no_grad():
            _, scores, noise_scores = public(R)
            values = {
                "native": classify(fixed["native"] + R @ hidden["native"], readouts["native"])
            }
            values.update(scores)
            values.update({name + "_calibrated": calibrated(v) for name, v in scores.items()})
            if args.noise_calibration:
                values.update({name + "_noise_calibrated": calibrated(v, noise_scores[name])
                               for name, v in scores.items()})
            return {name: int(v.argmax(1).eq(teacher_class).sum()) for name, v in values.items()}

    R = RN.clone()
    initial_accuracy = source_accuracy(R) if args.calibration_aware else None
    trace = []
    stop = "iteration_limit"
    derivative = None
    first_feasible = None

    def native_preserved(value):
        predicted = classify(fixed["native"] + value @ hidden["native"], readouts["native"])
        return int(predicted.argmax(1).eq(teacher_class).sum()) >= initial_accuracy["native"]

    # Optimize B inside the native-output error constraint; A remains fixed.
    for iteration in range(500):
        R = R.detach().requires_grad_(True)
        (loss, scores, _) = public(R)
        assert torch.isfinite(loss)
        if args.finite_classification and float(loss) == 0:
            # Once the finite public requirement holds, decrease native error.
            # This feasible ray search is not a constrained-stationarity claim.
            R = R.detach()
            nv = native(R)
            if first_feasible is None:
                first_feasible = dict(iteration=iteration, native_quadratic=nv)
            if nv - minimum <= 1e-6 * (1 + abs(nv)):
                stop = "native_minimum_with_finite_classification"
                break
            step = 1.0
            accepted = False
            with torch.no_grad():
                for backtrack in range(24):
                    proposal = R + step * (RN - R)
                    value, _, _ = public(proposal)
                    if float(value) == 0 and native_preserved(proposal):
                        R = proposal
                        accepted = True
                        break
                    step *= 0.5
            if not accepted:
                stop = "finite_classification_native_ray_boundary"
                break
            row = dict(iteration=iteration + 1, phase="native_retention",
                       public_objective=float(value), native_quadratic=native(R),
                       native_limit=tau, step=step)
            trace.append(row)
            print(json.dumps(row), flush=True)
            continue
        grad = torch.autograd.grad(loss, R)[0]
        dual = torch.cholesky_solve(grad.T, L).T
        dn = float((dual * grad).sum())
        assert dn >= 0
        if dn < 1e-24:
            stop = "zero_public_gradient"
            R = R.detach()
            break
        oracle = RN - (radius2 / dn) ** 0.5 * dual
        assert native(oracle) <= tau + 1e-08
        direction = oracle - R.detach()
        gap = float((grad * (R.detach() - oracle)).sum())
        assert gap >= -1e-08
        if iteration == 0:
            v = direction * (Rorig.norm() / direction.norm())
            step = 1e-06
            fd = float(
                (public(R.detach() + step * v)[0] - public(R.detach() - step * v)[0]) / (2 * step)
            )
            ad = float((grad * v).sum())
            error = abs(fd - ad) / (1 + abs(fd))
            assert error < 1e-05, (fd, ad, error)
            derivative = dict(finite_difference=fd, autograd=ad, relative_error=error)
        if gap <= 1e-06 * (1 + abs(float(loss))):
            stop = "small_conditional_gradient_gap"
            R = R.detach()
            break
        step = 1.0
        accepted = False
        with torch.no_grad():
            for backtrack in range(24):
                proposal = R.detach() + step * direction
                (value, _, _) = public(proposal)
                if (float(value) <= float(loss) - 0.0001 * step * gap
                        and (not args.finite_classification or native_preserved(proposal))):
                    accepted = True
                    R = proposal
                    break
                step *= 0.5
            if not accepted:
                stop = "fixed_line_search_stalled"
                R = R.detach()
                break
            nv = native(R)
            assert nv <= tau + 1e-08
            row = dict(
                iteration=iteration + 1,
                public_objective=float(value),
                conditional_gradient_gap=gap,
                native_quadratic=nv,
                native_limit=tau,
                step=step,
                public_class0_before_step={
                    name: int(v.argmax(1).eq(0).sum()) for (name, v) in scores.items()
                },
            )
            trace.append(row)
            if iteration % 25 == 0:
                print(json.dumps(row), flush=True)
    R = R.detach()
    edits = {
        k + "c_fc.weight": Ap.float().cpu(),
        k + "c_proj.weight": R[:, :-1].float().cpu(),
        k + "c_proj.bias": R[:, -1].float().cpu(),
    }
    if args.input_bias:
        edits[k + "c_fc.bias"] = ap.float().cpu()
    assert all((torch.isfinite(v).all() for v in edits.values()))
    torch.save(edits, out / "point.pt")
    Rf = torch.cat([to(edits[k + "c_proj.weight"]), to(edits[k + "c_proj.bias"])[:, None]], 1)
    with torch.no_grad():
        (objective, public_scores, noise_scores) = public(Rf)
        native_scores = classify(fixed["native"] + Rf @ hidden["native"], readouts["native"])
        source_counts = dict(
            native_teacher_retained=int(
                (native_scores.argmax(1).cpu() == raw["teacher_class"]).sum()
            ),
            native_n=len(native_scores),
            public_class0={
                name: int(v.argmax(1).eq(0).sum()) for (name, v) in public_scores.items()
            },
        )
    final_accuracy = source_accuracy(Rf) if args.calibration_aware else None
    source_feasible = None
    if args.calibration_aware:
        torch.save(
            {
                "native": native_scores.cpu(),
                **{name: v.cpu() for name, v in public_scores.items()},
                "teacher_class": teacher_class.cpu(),
                **({"construction_noise": {n: v.cpu() for n, v in noise_scores.items()}}
                   if args.noise_calibration else {}),
            },
            out / "source_scores.pt",
        )
        checked_contexts = public_names + (["joint_direct", "joint_task"] if args.input_bias else [])
        source_feasible = (
            final_accuracy["native"] >= initial_accuracy["native"]
            and all(
                final_accuracy[n + "_calibrated"] < initial_accuracy[n + "_calibrated"]
                for n in checked_contexts
            )
            and (not args.task_reference or all(final_accuracy[n] < initial_accuracy[n] for n in checked_contexts))
            and (not args.noise_calibration or all(final_accuracy[n + "_noise_calibrated"] <
                                                   initial_accuracy[n + "_noise_calibrated"] for n in checked_contexts))
            and native(Rf) <= tau + 1e-5 * (1 + abs(tau))
        )
        if args.finite_classification:
            source_feasible = (
                final_accuracy["native"] >= initial_accuracy["native"]
                and all(final_accuracy[n + suffix] == 0 for n in checked_contexts
                        for suffix in ["", "_calibrated", "_noise_calibrated"])
                and float(objective) == 0
                and native(Rf) <= tau + 1e-5 * (1 + abs(tau))
            )
    result = dict(
        status="sole_native_ellipse_classification_frozen_before_images",
        target=args.target,
        input_config_sha256=digest(args.inputs),
        input_definition=inputs,
        point_sha256=digest(out / "point.pt"),
        source_sha256=digest(__file__),
        source=Path(__file__).read_text(),
        source_cache_sha256=meta["source_arrays_sha256"],
        source_counts=source_counts,
        calibration_aware=args.calibration_aware,
        task_reference=args.task_reference,
        input_bias=args.input_bias,
        noise_calibration=args.noise_calibration,
        finite_classification=args.finite_classification,
        first_finite_classification=first_feasible,
        construction_noise_seed=2026110201 if args.noise_calibration else None,
        construction_noise_forwards=256 if args.noise_calibration else 0,
        construction_noise_tail_maximum_error=noise_tail_error,
        public_contexts=public_names + (["joint_direct", "joint_task"] if args.input_bias else []),
        additional_clean_synthetic_forwards=len(teacher_class) if args.task_reference else 0,
        initial_source_accuracy=initial_accuracy,
        final_source_accuracy=final_accuracy,
        source_feasible=source_feasible,
        source_scores_sha256=digest(out / "source_scores.pt") if args.calibration_aware else None,
        public_objective=float(objective),
        native_limit=tau,
        native_minimum=minimum,
        native_final=native(R),
        native_final_FP32=native(Rf),
        native_groups=groups,
        ridge=float(rho),
        original_tail_maximum_error=original_tail_error,
        derivative_check=derivative,
        iterations=len(trace),
        stop_reason=stop,
        trace=trace,
        metrics=dict(
            A_relative=float((Ap - A).norm() / A.norm()),
            B_relative=float((Rf[:, :-1] - B).norm() / B.norm()),
            logical_D=logical_parameter_distance(edits, s),
            input_bias_relative=float((ap - a).norm() / a.norm()),
        ),
        seconds=time.perf_counter() - start,
        real_images=0,
        new_full_AGA_calls=0,
        matching_calls=matching_calls,
        original_expert_updates=0,
        synthetic_images_reused=len(raw["final"]),
        generated_views=len(teacher_class),
        synthetic_native_layer_inputs=len(u),
        scope="SharedB conditionalnativequadratic/completepublicclassifier construction. OrdinaryFW/ellipsoid/Choleskyarenotnewalgorithm. Finitegapnotglobalguarantee. NativeboundequalA-onlysourcevalue,notnaturalaccuracy. Head/classesexplicitallowed;no partner/test/closedB read.",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for (k, v) in result.items() if k not in ["source", "trace"]}), flush=True
    )


if __name__ == "__main__":
    main()
