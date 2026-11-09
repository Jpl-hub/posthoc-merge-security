"""Analyze frozen NBP readouts on CPU; no images or parameter updates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _impl.io import digest, new_output, save_json
from _impl.models import initialize, load_checkpoint
from cfg import CHECKPOINTS, require_root


def readout(model):
    visual = model.model.visual
    return {"gain": visual.ln_post.weight.detach().float().clone(),
            "bias": visual.ln_post.bias.detach().float().clone(),
            "projection": visual.proj.detach().float().clone()}


def combine(base, models, coefficient):
    result = {}
    for key, value in base.items():
        delta = models[0][key] - value
        for model in models[1:]:
            delta = delta + (model[key] - value)
        result[key] = value + coefficient * delta
    return result


@torch.no_grad()
def minimum_bias_norm(output):
    """Bound the norm needed for the existing fixed-class proxy condition.

    A convex quadratic program, not a new defense or a recovery guarantee.
    Its nonnegative dual variables give a checkable lower bound on the optimum.
    No optimized tensor is saved or evaluated on images.
    """
    from scipy.optimize import minimize, nnls

    paths = ["results/phase3/bias_weight_only/eight_expert/evaluation.json",
             "results/phase3/bias_weight_only/eight_expert/extension_evaluation.json"]
    specs = {k: v for p in paths for k, v in json.loads(Path(p).read_text())["defenses"].items()
             if k.startswith("Defense1_")}
    out = new_output(output, {"source_sha256": digest(__file__),
        "input_protocol_sha256": {p: digest(p) for p in paths},
        "scope": "Same frozen class, public-base TA0.5, and safety2; norm analysis only. "
                 "No images, partners, new release or accuracy claim.",
        "solver": "SLSQP", "ftol": 1e-12, "maxiter": 1000,
        "rank_relative_tolerance": 1e-10, "active_tolerance": 1e-7})
    anchor = readout(load_checkpoint("zeroshot.pt"))
    rows = []
    for task in ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]:
        original = readout(load_checkpoint(f"{task}/finetuned.pt"))
        head = load_checkpoint(f"head_{task}.pt")
        weights = head.weight.double()
        proxy = {k: v.double() for k, v in combine(anchor, [original, anchor], .5).items()}
        projection = original["projection"].double().numpy()
        u, singular, _ = np.linalg.svd(projection, full_matrices=True)
        null = u[:, int((singular > singular[0]*1e-10).sum()):]
        ref = specs["Defense1_"+task]["release"]
        assert digest(ref["path"]) == ref["sha256"]
        released = torch.load(ref["path"], map_location="cpu", weights_only=False)["model.visual.ln_post.bias"]
        delta = released.double()-original["bias"].double()
        winner = int((delta @ proxy["projection"] @ weights.T).argmax())
        q = proxy["projection"] @ (weights[winner]-weights[torch.arange(len(weights)) != winner]).T
        normal = proxy["gain"][:, None]*q
        normal -= normal.mean(0, keepdim=True)
        b = (2*(math.sqrt(len(projection))*normal.norm(dim=0)-proxy["bias"]@q).clamp_min(0)).numpy()
        original_norm = float(delta.norm())
        scale = np.maximum(b, 1e-12)
        g = (.5*q.numpy().T@null)*original_norm/scale[:, None]
        rhs = b/scale
        x0 = null.T@delta.numpy()/original_norm
        assert np.all(g@x0 > 0)
        x0 *= max(1.0, float(np.max(rhs/(g@x0))))*(1+1e-8)
        fit = minimize(lambda x: .5*x@x, x0, jac=lambda x: x, method="SLSQP",
            constraints={"type": "ineq", "fun": lambda x: g@x-rhs, "jac": lambda x: g},
            options={"ftol": 1e-12, "maxiter": 1000})
        x, slack = fit.x, g@fit.x-rhs
        active = slack <= 1e-7
        multipliers, _ = nnls(g[active].T, x, maxiter=10000)
        dual_vector = g[active].T@multipliers
        lower = float(rhs[active]@multipliers-.5*(dual_vector@dual_vector))
        upper = float(.5*(x@x))
        gap = (upper-lower)/max(upper, 1e-30)
        candidate = null@x*original_norm
        row = dict(task=task, fixed_class=winner, original_update_norm=original_norm,
            minimum_norm_estimate=float(np.linalg.norm(candidate)),
            norm_ratio=float(np.linalg.norm(x)), solver_success=bool(fit.success),
            solver_message=str(fit.message), iterations=int(fit.nit),
            minimum_scaled_slack=float(slack.min()), relative_primal_dual_gap=gap,
            certified_within_tolerance=bool(slack.min() >= -1e-8 and -1e-8 <= gap <= 1e-7),
            native_kernel_relative_error=float(np.linalg.norm(candidate@projection)/np.linalg.norm(candidate)),
            release_sha256=ref["sha256"])
        rows.append(row)
        save_json(out/"results.json", dict(records=rows, completed=len(rows)==7,
            real_images=0, synthetic_images=0, new_publication_files=0, expert_updates=0))
        print("MINIMUM_BIAS_NORM", task, row["norm_ratio"], gap, row["certified_within_tolerance"], flush=True)


@torch.no_grad()
def class_bias_gain(output):
    """Measure surviving bias effects on centered class-score numerators.

    M=P(W-mean(W))^T. Restricting delta_bias to col(M) excludes both
    embedding-null and class-common changes removable without changing the
    native class decision. This measures an offset ratio, not accuracy.
    """
    out = new_output(output, {"source_sha256": digest(__file__),
        "scope": "Fixed public-base TA0.5; native class-score numerators, not normalized logits. "
                 "No partner, images, amplitude selection or new release.",
        "rank_relative_tolerance": 1e-10, "device": "cuda", "dtype": "float64"})
    base = load_checkpoint("zeroshot.pt")
    p0 = readout(base)["projection"].double().cuda()
    del base
    rows = []
    for task in ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]:
        expert = load_checkpoint(f"{task}/finetuned.pt")
        projection = readout(expert)["projection"].double().cuda()
        del expert
        head = load_checkpoint(f"head_{task}.pt")
        assert head.bias is None or torch.count_nonzero(head.bias) == 0
        weights = head.weight.double().cuda()
        centered = (weights - weights.mean(0)).T
        native = projection @ centered
        proxy = .25 * (projection + p0) @ centered
        u, singular, _ = torch.linalg.svd(native, full_matrices=False)
        keep = singular > singular[0] * 1e-10
        basis = u[:, keep]
        transfer = (basis / singular[keep]).T @ proxy
        maximum = float(torch.linalg.svdvals(transfer)[0])
        independent = np.linalg.pinv(native.cpu().numpy(), rcond=1e-10) @ proxy.cpu().numpy()
        numpy_maximum = float(np.linalg.svd(independent, compute_uv=False)[0])
        error = abs(maximum-numpy_maximum) / max(numpy_maximum, 1e-15)
        assert error < 1e-8, (task, error)
        projector = basis @ basis.T
        preservation_error = float((projector @ native-native).norm()/native.norm())
        rows.append(dict(task=task, classes=len(weights), class_rank=int(keep.sum()),
            maximum_proxy_to_native_class_offset_ratio=maximum,
            independent_numpy_relative_error=error,
            class_projection_preservation_error=preservation_error))
        save_json(out/"results.json", dict(records=rows, completed=len(rows)==7,
            real_images=0, synthetic_images=0, new_defense_weights=0, expert_updates=0))
        print("CLASS_OFFSET_GAIN", task, maximum, error, flush=True)


@torch.no_grad()
def projected_bias_gain(output):
    """Bound the public-proxy effect of bias noise retained by P-column cleanup.

    If v=delta_bias^T P and delta_bias lies in col(P), then its two-way
    TA0.5 offset is v (I + P^dagger P0)/4. No images or head are used.
    """
    out = new_output(output, {"source_sha256": digest(__file__),
        "scope": "Raw embedding operator norm for public-base TA0.5 only. "
                 "Not classification accuracy or a bound for unknown partners.",
        "device": "cuda", "dtype": "float64", "maximum_relative_formula_error": 1e-9})
    base = load_checkpoint("zeroshot.pt")
    p0 = readout(base)["projection"].double().cuda()
    del base
    rows = []
    for task in ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]:
        name = f"{task}/finetuned.pt"
        expert = load_checkpoint(name)
        p = readout(expert)["projection"].double().cuda()
        del expert
        u, singular, vh = torch.linalg.svd(p, full_matrices=False)
        assert torch.all(singular > singular[0]*1e-10)
        relative = vh.T @ ((u.T @ p0) / singular[:, None])
        q, r = np.linalg.qr(p.cpu().numpy(), mode="reduced")
        independent = np.linalg.solve(r, q.T @ p0.cpu().numpy())
        error = float(np.linalg.norm(relative.cpu().numpy()-independent) / np.linalg.norm(independent))
        assert error < 1e-9, (task, error)
        transfer = .25*(torch.eye(p.shape[1], dtype=torch.double, device="cuda")+relative)
        values = torch.linalg.svdvals(transfer)
        rows.append(dict(task=task, checkpoint_sha256=digest(CHECKPOINTS/name),
            base_sha256=digest(CHECKPOINTS/"zeroshot.pt"),
            maximum_proxy_to_native_embedding_offset_ratio=float(values[0]),
            minimum_proxy_to_native_embedding_offset_ratio=float(values[-1]),
            independent_numpy_relative_error=error))
        save_json(out / "results.json", {"records": rows, "completed": len(rows) == 7,
            "real_image_forwards": 0, "synthetic_image_forwards": 0,
            "expert_updates": 0, "new_defense_weights": 0})
        print(task, float(values[0]), error, flush=True)


@torch.no_grad()
def tail_equivalence(output):
    """Check affine tail equivalence with a fixed prefix, not a new defense.

    For centered normalized z, the embedding is z A + c with
    A = C diag(gain) P and c = bias^T P. Equal A,c give the same
    embedding for every such z. When A has full column rank, gain=1,
    projection=A and bias=A(A^T A)^-1 c realize that same map.
    """
    paths = ["results/phase3/bias_weight_only/eight_expert/evaluation.json",
             "results/phase3/bias_weight_only/eight_expert/extension_evaluation.json"]
    specs = {k: v for p in paths for k, v in json.loads(Path(p).read_text())["defenses"].items()
             if k.startswith("Defense1_")}
    out = new_output(output, {"source_sha256": digest(__file__),
        "input_protocol_sha256": {p: digest(p) for p in paths},
        "scope": "Fixed prefix and exact preservation of the full final embedding for all centered normalized states. "
                 "No image data, partner, head, new publication or accuracy measurement. "
                 "This does not exclude approximate task-preserving changes or changes to earlier blocks.",
        "dtype": "float64", "rank_relative_tolerance": 1e-10,
        "device": "cuda", "maximum_relative_formula_error": 1e-9,
        "probes": "16 fixed DCT hidden vectors, scales1e-3--1e3; algebra checks, not images"})
    records = []
    for task in ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]:
        name = f"{task}/finetuned.pt"
        model = load_checkpoint(name)
        original = {k: v.double().cuda() for k, v in readout(model).items()}
        eps = model.model.visual.ln_post.eps
        del model
        gain, bias, projection = (original[k] for k in ["gain", "bias", "projection"])
        a = gain[:, None] * projection
        a = a - a.mean(0, keepdim=True)
        u, singular, vh = torch.linalg.svd(a, full_matrices=False)
        rank = int((singular > singular[0]*1e-10).sum())
        assert rank == a.shape[1], (task, rank)
        c = bias @ projection
        canonical_bias = u @ ((vh @ c) / singular)
        # Independent CPU QR solve, explicit centering matrix and native LayerNorm.
        p_np, g_np, b_np = [v.cpu().numpy() for v in [projection, gain, bias]]
        center = np.eye(len(g_np)) - np.ones((len(g_np), len(g_np)))/len(g_np)
        a_np = center @ (g_np[:, None]*p_np)
        q_np, r_np = np.linalg.qr(a_np, mode="reduced")
        independent_bias = q_np @ np.linalg.solve(r_np.T, b_np @ p_np)
        error = float(np.linalg.norm(independent_bias-canonical_bias.cpu().numpy()) /
                      max(np.linalg.norm(independent_bias), 1e-15))
        assert error < 1e-9, (task, error)
        coordinate = torch.arange(len(gain), dtype=torch.double, device="cuda") + .5
        frequencies = torch.arange(1, 17, dtype=torch.double, device="cuda")[:, None]
        hidden = torch.cos(math.pi * frequencies * coordinate / len(gain))
        hidden *= torch.logspace(-3, 3, 16, dtype=torch.double, device="cuda")[:, None]
        native = torch.nn.functional.layer_norm(hidden, (len(gain),), gain, bias, eps) @ projection
        folded = torch.nn.functional.layer_norm(hidden, (len(gain),), torch.ones_like(gain), canonical_bias, eps) @ a
        feature_error = float((native-folded).norm()/native.norm())
        assert feature_error < 1e-9, (task, feature_error)
        ref = specs["Defense1_"+task]["release"]
        assert digest(ref["path"]) == ref["sha256"]
        edits = torch.load(ref["path"], map_location="cpu", weights_only=False)
        assert list(edits) == ["model.visual.ln_post.bias"]
        released_bias = edits["model.visual.ln_post.bias"].double().cuda()
        canonical_released_bias = u @ ((vh @ (released_bias @ projection)) / singular)
        record = dict(task=task, checkpoint_sha256=digest(CHECKPOINTS/name),
            release_sha256=ref["sha256"], centered_map_rank=rank,
            minimum_singular_value=float(singular[-1]), condition_number=float(singular[0]/singular[-1]),
            independent_numpy_relative_error=error, hidden_vector_relative_embedding_error=feature_error,
            original_update_norm=float((released_bias-bias).norm()),
            folded_bias_difference_norm=float((canonical_released_bias-canonical_bias).norm()),
            fp32_release_embedding_residual_norm=float(((released_bias-bias) @ projection).norm()))
        records.append(record)
        save_json(out / "results.json", {"records": records, "completed": len(records) == 7,
            "real_image_forwards": 0, "synthetic_image_forwards": 0,
            "expert_updates": 0, "new_defense_weights": 0})
        print(task, rank, error, feature_error, flush=True)


@torch.no_grad()
def direction_analysis(output):
    """Compare all class choices using weights only; never select a release."""
    from _impl.models import checked_models

    paths = ["results/phase3/bias_weight_only/eight_expert/evaluation.json",
             "results/phase3/bias_weight_only/eight_expert/extension_evaluation.json"]
    specs = {k: v for p in paths for k, v in json.loads(Path(p).read_text())["defenses"].items()
             if k.startswith("Defense1_")}
    out = new_output(output, {"source_sha256": digest(__file__),
        "input_protocol_sha256": {p: digest(p) for p in paths},
        "scope": "All class indices, two fixed direction formulas, proxy coefficient0.5, "
                 "SVD tolerance1e-10 and safety2. Weight-only analysis, no publication or selection."})
    rows, tasks, precision = [], [], []
    for task in ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]:
        base, expert, head, pins = checked_models(task)
        original, anchor = readout(expert), readout(base)
        proxy = {k: v.double() for k, v in combine(anchor, [original, anchor], .5).items()}
        projection = original["projection"].double()
        left, singular, _ = torch.linalg.svd(projection, full_matrices=True)
        rank = int((singular > singular[0] * 1e-10).sum())
        null = left[:, rank:]
        weights = head.weight.double()
        centered = weights - weights.mean(0)
        h = null.T @ proxy["projection"] @ centered.T
        u, values, vh = torch.linalg.svd(h, full_matrices=False)
        keep = values > values[0] * 1e-10
        desired = torch.eye(len(weights), dtype=torch.double) - 1 / len(weights)
        coefficients = u[:, keep] @ ((vh[keep] @ desired) / values[keep, None])
        directions = {"Least_squares": null @ coefficients,
            "Class_projection": null @ (null.T @ ((anchor["projection"].double()+projection)/2) @ centered.T)}
        requested = int(torch.randint(len(weights), (1,), generator=torch.Generator().manual_seed(20261014)))
        reference = np.linalg.lstsq(h.numpy().T, desired[:, requested].numpy(), rcond=1e-10)[0]
        relative_error = float(np.linalg.norm(reference-coefficients[:, requested].numpy()) /
                               np.linalg.norm(reference))
        assert relative_error < 1e-7
        feature = proxy["gain"][:, None] * (proxy["projection"] @ weights.T)
        feature -= feature.mean(0, keepdim=True)
        gram = feature.T @ feature
        squared = gram.diag()[:, None] + gram.diag()[None, :] - 2*gram
        worst = math.sqrt(len(projection)) * squared.clamp_min(0).sqrt()
        bias_scores = proxy["bias"] @ proxy["projection"] @ weights.T
        ref = specs["Defense1_"+task]["release"]
        assert digest(ref["path"]) == ref["sha256"]
        bias = original["bias"].double()
        released = torch.load(ref["path"], map_location="cpu", weights_only=False)["model.visual.ln_post.bias"]
        delta = released.double()-bias
        for method, directions_matrix in directions.items():
            scores = directions_matrix.T @ proxy["projection"] @ weights.T
            for index in range(len(weights)):
                direction = directions_matrix[:, index]
                winner = int(scores[index].argmax())
                mask = torch.arange(len(weights)) != winner
                gaps = scores[index, winner]-scores[index, mask]
                valid = bool(torch.all(gaps > 1e-10))
                magnitude = None
                if valid:
                    numerator = (worst[winner, mask] - bias_scores[winner] + bias_scores[mask]).clamp_min(0)
                    magnitude = 2*float((numerator / (.5*gaps)).max())*float(direction.norm())
                rows.append(dict(task=task, method=method, requested_class=index,
                    largest_offset_class=winner, positive_gaps=valid,
                    minimum_unit_direction_gap=float(gaps.min()/direction.norm()),
                    sufficient_update_norm=magnitude,
                    fit_residual=float((h.T@coefficients[:, index]-desired[:, index]).norm()) if method=="Least_squares" else None,
                    cosine_between_directions=float(torch.nn.functional.cosine_similarity(
                        directions["Least_squares"][:, index], directions["Class_projection"][:, index], dim=0))))
        head_norm = weights.norm(dim=1)
        tasks.append(dict(task=task, checkpoint_sha256=pins, publication=ref,
            classes=len(weights), projection_rank=rank, class_map_rank=int(keep.sum()),
            class_map_condition=float(values[keep][0]/values[keep][-1]),
            original_requested_class=requested, numpy_least_squares_relative_error=relative_error,
            frozen_update_norm=float(delta.norm()), native_bias_norm=float(bias.norm()),
            delta_to_native_bias_norm=float(delta.norm()/bias.norm()) if bias.norm()>0 else None,
            maximum_released_bias=float(released.abs().max()),
            head_norm_range=[float(head_norm.min()),float(head_norm.max())],
            maximum_head_bias=0.0 if head.bias is None else float(head.bias.abs().max())))
        for name, dtype in [("FP32",torch.float32),("FP16",torch.float16),("BF16",torch.bfloat16)]:
            changed = released.to(dtype).double()
            clean = original["bias"].to(dtype).double()
            rounded_projection = original["projection"].to(dtype).double()
            finite = bool(torch.isfinite(changed).all())
            precision.append(dict(task=task, storage_dtype=name, finite_bias=finite,
                nonfinite_bias_values=int((~torch.isfinite(changed)).sum()),
                rounding_only_embedding_residual=float(((changed-clean)@rounded_projection).norm()) if finite else None))
        print("ALL_CLASS_WEIGHT_ANALYSIS",task,len(weights),flush=True)
    save_json(out/"results.json",dict(tasks=tasks,rows=rows,precision=precision,
        scope="Weight-only analysis of every class; no test accuracy, deployment forward or parameter selection.",
        real_images=0,synthetic_images=0,expert_updates=0,new_publications=0))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--directions", action="store_true", help="Analyze every class choice and storage rounding")
    mode.add_argument("--tail-equivalence", action="store_true", help="Check the affine final readout with a fixed prefix")
    mode.add_argument("--projection-gain", action="store_true", help="Bound bias offsets retained by projection cleanup")
    mode.add_argument("--class-gain", action="store_true", help="Measure surviving class-score offset amplification")
    mode.add_argument("--minimum-norm", action="store_true", help="Bound the norm needed for the fixed-class proxy condition")
    args = parser.parse_args()
    require_root()
    initialize()
    torch.set_num_threads(2)
    if args.minimum_norm:
        minimum_bias_norm(args.output)
        return
    if args.class_gain:
        class_bias_gain(args.output)
        return
    if args.projection_gain:
        projected_bias_gain(args.output)
        return
    if args.tail_equivalence:
        tail_equivalence(args.output)
        return
    if args.directions:
        direction_analysis(args.output)
        return
    protocol_paths = ["results/phase3/bias_weight_only/eight_expert/evaluation.json",
                      "results/phase3/bias_weight_only/eight_expert/extension_evaluation.json"]
    protocols = [json.loads(Path(p).read_text()) for p in protocol_paths]
    specs = {k: v for p in protocols for k, v in p["defenses"].items() if k.startswith("Defense1_")}
    order = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
    anchor = readout(load_checkpoint("zeroshot.pt"))
    experts = {task: readout(load_checkpoint(f"{task}/finetuned.pt")) for task in order}
    out = new_output(args.output, {"source_sha256": digest(__file__),
        "input_protocol_sha256": {p: digest(p) for p in protocol_paths},
        "scope": "Real-arithmetic bounds for the stored FP32 readout tensors. Partner weights are used only for analysis of already frozen releases, never construction or selection."})
    rows, native = [], []
    maximum_error = 0.0
    for task in [t for t in order if t != "EuroSAT"]:
        original = experts[task]
        ref = specs["Defense1_" + task]["release"]
        assert digest(ref["path"]) == ref["sha256"]
        edits = torch.load(ref["path"], map_location="cpu", weights_only=False)
        assert list(edits) == ["model.visual.ln_post.bias"]
        released = original | {"bias": edits["model.visual.ln_post.bias"].float()}
        delta = released["bias"].double() - original["bias"].double()
        native.append({"task": task, "release_sha256": ref["sha256"],
                       "bias_delta_norm": float(delta.norm()),
                       "native_embedding_residual_norm": float((delta @ original["projection"].double()).norm())})
        head = load_checkpoint(f"head_{task}.pt")
        assert head.bias is None or torch.count_nonzero(head.bias) == 0
        weights = head.weight.double()
        peer = "SVHN" if task == "MNIST" else "MNIST"
        for scenario, names, coefficient in [("Public-base proxy", [task, "Base"], 0.5),
                ("Two experts", [task, peer], 0.5), ("Eight experts", order, 0.2)]:
            for protected in [False, True]:
                models = [anchor if name == "Base" else released if name == task and protected else experts[name]
                          for name in names]
                state = {k: v.double() for k, v in combine(anchor, models, coefficient).items()}
                gain, bias, projection = (state[k] for k in ["gain", "bias", "projection"])
                winner = int((bias @ projection @ weights.T).argmax())
                differences = weights[winner] - weights
                q = projection @ differences.T
                normal = gain[:, None] * q
                normal = normal - normal.mean(0, keepdim=True)
                lower = bias @ q - math.sqrt(len(gain)) * normal.norm(dim=0)
                q_np = differences.numpy() @ projection.numpy().T
                a_np = q_np * gain.numpy()[None]
                independent = q_np @ bias.numpy() - math.sqrt(len(gain)) * np.linalg.norm(a_np - a_np.mean(1, keepdims=True), axis=1)
                np.testing.assert_allclose(lower.numpy(), independent, rtol=1e-9, atol=1e-7)
                maximum_error = max(maximum_error, float(np.max(np.abs(lower.numpy() - independent))))
                mask = torch.arange(len(weights)) != winner
                minimum = float(lower[mask].min())
                row = {"task": task, "scenario": scenario, "protected": protected,
                       "constant_class_candidate": winner, "minimum_margin_lower_bound": minimum,
                       "constant_prediction_certified_real_arithmetic": minimum > 0,
                       "embedding_shift_norm": float((coefficient * delta @ projection).norm()) if protected else 0.0,
                       "per_class_margin_lower_bounds": lower.tolist()}
                rows.append(row)
                print(task, scenario, protected, minimum, flush=True)
    save_json(out / "results.json", {"native": native, "rows": rows,
        "independent_numpy_formula_check": True, "maximum_formula_difference": maximum_error,
        "real_image_forwards": 0, "synthetic_image_forwards": 0,
        "expert_updates": 0, "new_defense_weights": 0})


if __name__ == "__main__":
    main()
