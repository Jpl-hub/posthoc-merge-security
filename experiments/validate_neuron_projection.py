"""Replay task-only neuron projection and check its exact local algebra."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from critical_subspace import capture
from eight_task_benchmark import digest
from local_erasure import rescale
from neuron_projection import activation, construct, pair_error
from path_subspace import context, read, response
from subspace_probe import save_json


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and meta["stage"] == "construction_screen_only"
    assert not meta["defense_success_claimed"]
    assert cfg == read(args.config) and meta["config_sha256"] == digest(args.config)
    assert meta["source_sha256"] == digest("experiments/neuron_projection.py")
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, _, models = context(cfg)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    original = load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval()
    anchor = models["anchor"]
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    indices = read(out / "training_indices.json")
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    images = torch.stack([data.train_dataset[i][0] for i in indices["calibration"]])
    confirmation = torch.stack([data.train_dataset[i][0] for i in indices["confirmation"]])
    saved = torch.load(out / "artifacts/inputs.pt", map_location="cpu", weights_only=True)
    for name, beta in [("expert", 1.0)] + [(str(beta), beta) for beta in cfg["betas"]]:
        reference = original if beta == 1 else rescale(original, anchor, beta)
        value, _ = capture(reference, images, cfg["layers"])
        assert all(torch.equal(tensor, value[key]) for key, tensor in saved[name].items())
    release, edits, selected, scores, checks = construct(original, anchor, saved, cfg)
    compiled = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    assert edits.keys() == compiled.keys() and all(
        torch.equal(tensor, compiled[key]) for key, tensor in edits.items()
    )
    assert selected == read(out / "selected.json") and scores == read(out / "pair_scores.json")
    assert checks == read(out / "construction_checks.json")
    local_checks = []
    for layer in cfg["layers"]:
        native = original.model.visual.transformer.resblocks[layer].mlp.cuda().double()
        base = anchor.model.visual.transformer.resblocks[layer].mlp.cuda().double()
        changed = release.model.visual.transformer.resblocks[layer].mlp.cuda().double()
        pairs = selected[str(layer)]
        used = [index for pair in pairs for index in pair]
        assert len(used) == len(set(used))
        untouched = [index for index in range(native.c_fc.out_features) if index not in used]
        assert torch.equal(native.c_fc.weight[untouched], changed.c_fc.weight[untouched])
        assert torch.equal(native.c_fc.bias[untouched], changed.c_fc.bias[untouched])
        assert torch.equal(native.c_proj.weight[:, untouched], changed.c_proj.weight[:, untouched])
        assert torch.equal(native.c_proj.bias, changed.c_proj.bias)
        maximum = 0.0
        for pair in pairs:
            i, j = pair
            assert torch.equal(changed.c_fc.weight[i], changed.c_fc.weight[j])
            assert torch.equal(changed.c_fc.bias[i], changed.c_fc.bias[j])
            assert torch.equal(changed.c_proj.weight[:, i], changed.c_proj.weight[:, j])
            for name, beta in [("expert", 1.0)] + [(str(beta), beta) for beta in cfg["betas"]]:
                x = saved[name][f"up:{layer}"][:16].double().cuda()
                coefficients, columns, _ = pair_error(native, base, x, beta, pair)
                w = (1 - beta) * base.c_fc.weight[pair] + beta * native.c_fc.weight[pair]
                b = (1 - beta) * base.c_fc.bias[pair] + beta * native.c_fc.bias[pair]
                v = (1 - beta) * base.c_proj.weight[:, pair] + beta * native.c_proj.weight[:, pair]
                delta_w = native.c_fc.weight[pair].mean(0) - native.c_fc.weight[pair]
                delta_b = native.c_fc.bias[pair].mean() - native.c_fc.bias[pair]
                delta_v = (
                    native.c_proj.weight[:, pair].mean(1, keepdim=True)
                    - native.c_proj.weight[:, pair]
                )
                old = activation(x @ w.T + b) @ v.T
                new = (
                    activation(x @ (w + beta * delta_w).T + b + beta * delta_b)
                    @ (v + beta * delta_v).T
                )
                error = float((new - old - coefficients @ columns.T).abs().max())
                assert error < 1e-10
                maximum = max(maximum, error)
        local_checks.append(
            dict(
                layer=layer,
                pairs=len(pairs),
                maximum_pair_algebra_error=maximum,
                unpaired_tensors_unchanged=True,
                tied_units_verified=True,
            )
        )
        native.cpu().float()
        base.cpu().float()
        changed.cpu().float()
    functions = read(out / "functions.json")
    for name, beta in [("expert", 1.0)] + [(str(beta), beta) for beta in cfg["betas"]]:
        before = original if beta == 1 else rescale(original, anchor, beta)
        after = release if beta == 1 else rescale(release, anchor, beta)
        for split, batch in [("calibration", images), ("confirmation", confirmation)]:
            assert response(before, after, batch) == functions[name][split]
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            stage="construction_screen_only",
            elapsed_seconds=time.perf_counter() - start,
            compiled_edits_replayed=True,
            all_shortlisted_scores_and_selection_replayed=True,
            calibration_inputs_replayed=True,
            full_accuracy_or_attack_results_claimed=False,
            local_checks=local_checks,
            source_sha256=digest(__file__),
            metadata_sha256=digest(out / "metadata.json"),
            functions_sha256=digest(out / "functions.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neuron_projection.json")
    parser.add_argument("--output", default="outputs/2026-10-11/neuron_projection")
    validate(parser.parse_args())
