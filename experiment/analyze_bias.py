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
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_root()
    initialize()
    torch.set_num_threads(2)
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
