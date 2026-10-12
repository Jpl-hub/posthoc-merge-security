"""Task-dependent QuickGELU regions: a necessary-condition diagnostic.

No protected release, optimizer, response reconstruction, future partner,
official test or safety claim. Reuses the author model/data/head and target
profile selection. Affine activation hooks are diagnostic controls only.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from eight_task_benchmark import digest
from functional_contrast import features
from local_erasure import rescale
from operator_diagnostics import paired_counts
from path_subspace import context, read
from query_partner_cost import runtime
from subspace_probe import save_json


@torch.no_grad()
def capture(model, images, layers, batch_size):
    bounds, handles = {}, []
    model.cuda().eval()
    last_layer = len(model.model.visual.transformer.resblocks) - 1

    def hook(layer, module, args, output):
        z = output.detach()
        if layer == last_layer:
            z = z[:1]
        z = z.reshape(-1, z.shape[-1]).cpu()
        if layer not in bounds:
            bounds[layer] = dict(minimum=z.amin(0), maximum=z.amax(0), witness=z[:8].clone())
        else:
            bounds[layer]["minimum"] = torch.minimum(bounds[layer]["minimum"], z.amin(0))
            bounds[layer]["maximum"] = torch.maximum(bounds[layer]["maximum"], z.amax(0))

    for layer in layers:
        module = model.model.visual.transformer.resblocks[layer].mlp.c_fc
        handles.append(
            module.register_forward_hook(lambda m, a, y, index=layer: hook(index, m, a, y))
        )
    try:
        for batch in images.split(batch_size):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return bounds


def selection(bounds, threshold):
    return {
        layer: dict(
            positive=value["minimum"].ge(threshold).nonzero().flatten(),
            negative=value["maximum"].le(-threshold).nonzero().flatten(),
        )
        for layer, value in bounds.items()
    }


@torch.no_grad()
def affine_features(model, images, selected):
    handles = []

    def hook(layer, module, args, output):
        out = output.clone()
        positive, negative = selected[layer]["positive"], selected[layer]["negative"]
        out[..., positive] = args[0][..., positive]
        out[..., negative] = 0
        return out

    for layer in selected:
        module = model.model.visual.transformer.resblocks[layer].mlp.gelu
        handles.append(
            module.register_forward_hook(lambda m, a, y, index=layer: hook(index, m, a, y))
        )
    try:
        model.cuda().eval()
        result = features(model, images)
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return result


@torch.no_grad()
def support_reset(expert, anchor, selected):
    result = copy.deepcopy(expert).cpu().eval()
    for layer, choice in selected.items():
        indexes = torch.cat([choice["positive"], choice["negative"]])
        target = result.model.visual.transformer.resblocks[layer].mlp
        source = anchor.model.visual.transformer.resblocks[layer].mlp
        target.c_fc.weight[indexes] = source.c_fc.weight[indexes]
        target.c_fc.bias[indexes] = source.c_fc.bias[indexes]
        target.c_proj.weight[:, indexes] = source.c_proj.weight[:, indexes]
    return result


def response(before, after, head, labels):
    a = before @ head.weight.detach().cpu().T + head.bias.detach().cpu()
    b = after @ head.weight.detach().cpu().T + head.bias.detach().cpu()
    return dict(
        n=len(labels),
        original_correct=int(a.argmax(1).eq(labels).sum()),
        changed_correct=int(b.argmax(1).eq(labels).sum()),
        paired=paired_counts(a.argmax(1).eq(labels), b.argmax(1).eq(labels)),
        normalized_feature_delta_rms=float(
            (after.double() - before.double()).square().sum(1).mean().sqrt()
        ),
    )


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    runtime()
    started = time.perf_counter()
    root, load, _, holders = context(cfg)
    anchor = holders["anchor"]
    parent = Path(cfg["profile_parent"])
    profiles, old_indices = read(parent / "selection.json"), read(parent / "training_indices.json")
    from dataset.registry import get_dataset

    threshold = (
        math.log((1 - cfg["relative_activation_tail"]) / cfg["relative_activation_tail"])
        / cfg["quickgelu_slope"]
    )
    rows, raw = {}, {}
    for task in cfg["tasks"]:
        assert not profiles[task]["partner_used"]
        layers = profiles[task]["layers"]
        expert = load(str(root / task / "finetuned.pt")).cpu().eval()
        head = load(str(root / f"head_{task}.pt")).cpu().eval()
        assert head.normalize and torch.count_nonzero(head.bias) == 0
        data = get_dataset(task, expert.val_preprocess, "data", batch_size=128, num_workers=2)
        indexes = dict(
            calibration=old_indices[task]["selection"][: cfg["calibration_samples"]],
            confirmation=old_indices[task]["confirmation"][: cfg["confirmation_samples"]],
        )
        assert not set(indexes["calibration"]).intersection(indexes["confirmation"])
        images, labels = {}, {}
        for split, idx in indexes.items():
            pairs = [data.train_dataset[i] for i in idx]
            images[split] = torch.stack([x for x, _ in pairs])
            labels[split] = torch.tensor([int(y) for _, y in pairs])
        bounds = capture(expert, images["calibration"], layers, cfg["batch_size"])
        chosen = selection(bounds, threshold)
        reset = support_reset(expert, anchor, chosen)
        task_raw = dict(
            indexes=indexes, bounds=bounds, selected=chosen, labels=labels, responses={}
        )
        task_rows = dict(layers=layers, regions={}, responses={})
        for layer, group in chosen.items():
            task_rows["regions"][str(layer)] = {name: len(idx) for name, idx in group.items()}
        for name, model in [("native", expert)] + [
            (f"beta_{beta}", rescale(expert, anchor, beta)) for beta in cfg["betas"]
        ]:
            for split, x in images.items():
                key = f"{name}/{split}"
                measured = capture(model, x, layers, cfg["batch_size"])
                validity = {}
                for layer, group in chosen.items():
                    validity[str(layer)] = dict(
                        positive_violations=int(
                            measured[layer]["minimum"][group["positive"]].lt(threshold).sum()
                        ),
                        negative_violations=int(
                            measured[layer]["maximum"][group["negative"]].gt(-threshold).sum()
                        ),
                    )
                before = features(model.cuda().eval(), x)
                model.cpu()
                after = affine_features(model, x, chosen)
                task_rows["responses"][key] = dict(
                    response(before, after, head, labels[split]), validity=validity
                )
                task_raw["responses"][key] = dict(bounds=measured, original=before, affine=after)
                if name == "native":
                    reset_features = features(reset.cuda().eval(), x)
                    reset.cpu()
                    task_rows["responses"][key]["support_reset"] = response(
                        before, reset_features, head, labels[split]
                    )
                    task_raw["responses"][key]["support_reset"] = reset_features
        rows[task], raw[task] = task_rows, task_raw
        print("REGIONS", task, task_rows, flush=True)
    torch.save(raw, out / "artifacts/regions.pt")
    save_json(out / "protocol.json", cfg)
    save_json(
        out / "results.json",
        dict(
            threshold=threshold,
            tasks=rows,
            protected_candidates=0,
            official_test_access=False,
            future_partner_access=False,
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - started,
            source_sha256=digest(Path(__file__)),
            config_sha256=digest(args.config),
            raw_sha256=digest(out / "artifacts/regions.pt"),
        ),
    )


@torch.no_grad()
def validate(args):
    cfg, out = read(args.config), Path(args.output)
    meta, results = read(out / "metadata.json"), read(out / "results.json")
    assert meta["completed"] and meta["source_sha256"] == digest(Path(__file__))
    assert meta["config_sha256"] == digest(args.config)
    assert meta["raw_sha256"] == digest(out / "artifacts/regions.pt")
    runtime()
    root, load, _, holders = context(cfg)
    anchor = holders["anchor"]
    raw = torch.load(out / "artifacts/regions.pt", map_location="cpu")
    from dataset.registry import get_dataset

    checks = 0
    for task, saved in raw.items():
        expert = load(str(root / task / "finetuned.pt")).cpu().eval()
        head = load(str(root / f"head_{task}.pt")).cpu().eval()
        data = get_dataset(task, expert.val_preprocess, "data", batch_size=128, num_workers=2)
        layers = results["tasks"][task]["layers"]
        images = {
            split: torch.stack([data.train_dataset[i][0] for i in indexes])
            for split, indexes in saved["indexes"].items()
        }
        for split, indexes in saved["indexes"].items():
            assert torch.equal(
                saved["labels"][split],
                torch.tensor([int(data.train_dataset[index][1]) for index in indexes]),
            )
        bounds = capture(expert, images["calibration"], layers, cfg["batch_size"])
        chosen = selection(bounds, results["threshold"])
        for layer in layers:
            assert torch.equal(bounds[layer]["minimum"], saved["bounds"][layer]["minimum"])
            assert torch.equal(bounds[layer]["maximum"], saved["bounds"][layer]["maximum"])
            for group in ("positive", "negative"):
                assert torch.equal(chosen[layer][group], saved["selected"][layer][group])
                values = saved["bounds"][layer][
                    "minimum" if group == "positive" else "maximum"
                ].numpy()
                expected = np.flatnonzero(
                    values >= results["threshold"]
                    if group == "positive"
                    else values <= -results["threshold"]
                )
                assert np.array_equal(expected, chosen[layer][group].numpy())
            z = bounds[layer]["witness"].double()
            module = expert.model.visual.transformer.resblocks[layer].mlp.gelu
            numpy_g = z.numpy() / (1 + np.exp(-cfg["quickgelu_slope"] * z.numpy()))
            assert np.max(np.abs(numpy_g - module(z).numpy())) < 1e-12
            high, low = z >= results["threshold"], z <= -results["threshold"]
            assert np.all(
                np.abs(numpy_g[high] - z.numpy()[high])
                <= cfg["relative_activation_tail"] * np.abs(z.numpy()[high]) + 1e-12
            )
            assert np.all(
                np.abs(numpy_g[low])
                <= cfg["relative_activation_tail"] * np.abs(z.numpy()[low]) + 1e-12
            )
        reset = support_reset(expert, anchor, chosen)
        for name, model in [("native", expert)] + [
            (f"beta_{beta}", rescale(expert, anchor, beta)) for beta in cfg["betas"]
        ]:
            for split, x in images.items():
                key = f"{name}/{split}"
                measured = capture(model, x, layers, cfg["batch_size"])
                for layer in layers:
                    for bound in ("minimum", "maximum"):
                        assert torch.equal(
                            measured[layer][bound], saved["responses"][key]["bounds"][layer][bound]
                        )
                before = features(model.cuda().eval(), x)
                model.cpu()
                after = affine_features(model, x, chosen)
                assert torch.equal(before, saved["responses"][key]["original"])
                assert torch.equal(after, saved["responses"][key]["affine"])
                recorded = results["tasks"][task]["responses"][key]
                actual = response(before, after, head, saved["labels"][split])
                assert all(actual[k] == recorded[k] for k in actual)
                if name == "native":
                    value = features(reset.cuda().eval(), x)
                    reset.cpu()
                    assert torch.equal(value, saved["responses"][key]["support_reset"])
                    assert (
                        response(before, value, head, saved["labels"][split])
                        == recorded["support_reset"]
                    )
                checks += 1
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            replayed_training_responses=checks,
            numpy_region_selection=True,
            numpy_quickgelu_bound=True,
            results_sha256=digest(out / "results.json"),
            raw_sha256=meta["raw_sha256"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/activation_regions.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    validate(arguments) if arguments.validate else run(arguments)
