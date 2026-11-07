"""Analyze frozen NBP readouts on CPU; no images or parameter updates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _impl.io import digest, new_output, save_json
from _impl.models import initialize, load_checkpoint
from cfg import require_root


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
                    fit_residual=float((scores[index]-desired[:, index]).norm()) if method=="Least_squares" else None,
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
    parser.add_argument("--directions", action="store_true", help="Analyze every class choice and storage rounding")
    args = parser.parse_args()
    require_root()
    initialize()
    torch.set_num_threads(2)
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
