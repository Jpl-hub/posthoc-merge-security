"""Defense 2: frozen MNIST A/B construction. See README for source caches and provenance."""

# Preserve the three expressions from the executed source.
# ruff: noqa: E731

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.attacks import load_author
from _impl.io import digest
from _impl.metrics import logical_parameter_distance
from _impl.models import checked_models, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-aware", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibration_aware and args.output is None:
        parser.error("--calibration-aware requires a separate --output directory")
    initialize()
    torch.set_default_dtype(torch.float64)
    start = time.perf_counter()
    root = Path("outputs/native_ellipse_classification_20261026")
    out = args.output or root / "point"
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((root / "source/result.json").read_text())
    assert digest(root / "source/source.pt") == meta["source_arrays_sha256"]
    raw = torch.load(root / "source/source.pt", map_location="cpu", weights_only=False)
    (base, expert, head, pins) = checked_models("MNIST")
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
    A0r = A0 - A0 @ V @ V.T
    K = A0r @ A0r.T
    G = -(U @ U.T) * K
    G.diagonal().add_(K.diagonal())
    lam = torch.cholesky_solve((-(A * A0).sum(1))[:, None], torch.linalg.cholesky(G))
    dA = lam * A0r
    dA -= U @ (U.T @ dA)
    Ap = (A + dA).float().double()
    del K, G
    (author, sha) = load_author(Path("external/secure-merge-attack/utils_attack_B32.py"))
    assert sha == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    pi = author["solve_permutation"](Ap.float().cpu(), A0.float().cpu()).cuda()
    assert sorted(pi.tolist()) == list(range(len(A)))
    Rorig = torch.cat([B, b[:, None]], 1)
    cache = {name: {key: to(v) for (key, v) in row.items()} for (name, row) in raw["cache"].items()}
    activate = lambda z: z * torch.sigmoid(1.702 * z)

    def augmented(x):
        return torch.cat([activate(x @ Ap.T + a).T, torch.ones(1, len(x), device="cuda")], 0)

    # Fit B and its bias to frozen-teacher outputs on the two synthetic populations.
    G = torch.zeros(3073, 3073, device="cuda")
    T = torch.zeros_like(Rorig)
    constant = G.new_zeros(())
    groups = []
    source = Path("outputs/task_shaped_compensation_20261026/layer_inputs.pt")
    assert digest(source) == "10938b057263f17afd6551b8c066b3216633c6e038e7ec4360c8e770e8c83dfe"
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
    G += rho * torch.eye(len(G), device="cuda")
    T += rho * Rorig
    constant += rho * Rorig.square().sum()
    L = torch.linalg.cholesky(G)
    RN = torch.cholesky_solve(T.T, L).T
    native = lambda R: float((R @ G * R).sum() - 2 * (R * T).sum() + constant)
    tau = native(Rorig)
    minimum = native(RN)
    radius2 = tau - minimum
    assert radius2 > 1e-12, (tau, minimum)
    # Build the public-base fusion and AGA contexts, including the full classifier.
    hidden = {}
    fixed = {}
    readouts = {}
    W = to(head.weight)
    assert float(head.bias.abs().max()) == 0
    eps = expert.model.visual.ln_post.eps
    for name in ["native", "direct", "recovered"]:
        x = cache[name]["x"]
        if name == "native":
            H = activate(x @ Ap.T + a).T
            fixed[name] = cache[name]["prefix"].T
            hidden[name] = torch.cat([H, torch.ones(1, H.shape[1], device="cuda")], 0)
        else:
            (AA, aa) = (Ap, a) if name == "direct" else (Ap[pi], a[pi])
            H = activate(x @ ((AA + A0) / 2).T + (aa + a0) / 2).T
            fixed[name] = cache[name]["prefix"].T + (B0 @ H + b0[:, None]) / 2
            if name == "recovered":
                back = torch.empty_like(H)
                back[pi] = H
                H = back
            hidden[name] = torch.cat([H, torch.ones(1, H.shape[1], device="cuda")], 0) / 2
        names = ["model.visual.ln_post.weight", "model.visual.ln_post.bias", "model.visual.proj"]
        readouts[name] = [
            to(s[key] if name == "native" else (s[key] + bs[key]) / 2) for key in names
        ]

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

    teacher_class = raw["teacher_class"].long().cuda()

    def calibrated(scores):
        # Probability division, evaluated on the fixed synthetic source only.
        log_prior = torch.logsumexp(F.log_softmax(scores, dim=1), dim=0) - math.log(len(scores))
        return scores - log_prior

    def public(R):
        scores = {
            name: classify(fixed[name] + R @ hidden[name], readouts[name])
            for name in ["direct", "recovered"]
        }
        if args.calibration_aware:
            # Reduce correct-class probability before and after calibration;
            # no fixed destination class, new images, or fitted test labels.
            correct = torch.cat(
                [
                    F.log_softmax(v, dim=1).gather(1, teacher_class[:, None]).flatten()
                    for values in scores.values()
                    for v in (values, calibrated(values))
                ]
            )
            return torch.logsumexp(correct, 0) - math.log(len(correct)), scores
        gaps = torch.cat([v[:, 1:] - v[:, :1] for v in scores.values()])
        return (torch.logsumexp(gaps.flatten(), 0) - math.log(gaps.numel()), scores)

    def source_accuracy(R):
        with torch.no_grad():
            _, scores = public(R)
            values = {
                "native": classify(fixed["native"] + R @ hidden["native"], readouts["native"])
            }
            values.update(scores)
            values.update({name + "_calibrated": calibrated(v) for name, v in scores.items()})
            return {name: int(v.argmax(1).eq(teacher_class).sum()) for name, v in values.items()}

    R = RN.clone()
    initial_accuracy = source_accuracy(R) if args.calibration_aware else None
    trace = []
    stop = "iteration_limit"
    derivative = None
    # Optimize B inside the native-output error constraint; A remains fixed.
    for iteration in range(500):
        R = R.detach().requires_grad_(True)
        (loss, scores) = public(R)
        assert torch.isfinite(loss)
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
                (value, _) = public(proposal)
                if float(value) <= float(loss) - 0.0001 * step * gap:
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
    assert all((torch.isfinite(v).all() for v in edits.values()))
    torch.save(edits, out / "point.pt")
    Rf = torch.cat([to(edits[k + "c_proj.weight"]), to(edits[k + "c_proj.bias"])[:, None]], 1)
    with torch.no_grad():
        (objective, public_scores) = public(Rf)
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
            },
            out / "source_scores.pt",
        )
        source_feasible = (
            final_accuracy["native"] >= initial_accuracy["native"]
            and all(
                final_accuracy[n + "_calibrated"] < initial_accuracy[n + "_calibrated"]
                for n in ["direct", "recovered"]
            )
            and native(Rf) <= tau + 1e-5 * (1 + abs(tau))
        )
    result = dict(
        status="sole_native_ellipse_classification_frozen_before_images",
        point_sha256=digest(out / "point.pt"),
        source_sha256=digest(__file__),
        source=Path(__file__).read_text(),
        source_cache_sha256=meta["source_arrays_sha256"],
        source_counts=source_counts,
        calibration_aware=args.calibration_aware,
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
        ),
        seconds=time.perf_counter() - start,
        real_images=0,
        new_full_AGA_calls=0,
        matching_calls=1,
        original_expert_updates=0,
        synthetic_images_reused=160,
        generated_views=800,
        synthetic_native_layer_inputs=16384,
        scope="SharedB conditionalnativequadratic/completepublicclassifier construction. OrdinaryFW/ellipsoid/Choleskyarenotnewalgorithm. Finitegapnotglobalguarantee. NativeboundequalA-onlysourcevalue,notnaturalaccuracy. Head/classesexplicitallowed;no partner/test/closedB read.",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for (k, v) in result.items() if k not in ["source", "trace"]}), flush=True
    )


if __name__ == "__main__":
    main()
