"""Frozen routing candidate: matched TA/author AGA/public Key restoration.

Use the shared evaluation/validation runner and verified prior baselines.
No additional candidate, fitting, known partner construction or query attack.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from eight_task_benchmark import digest
from linear_completion_eval import run, validate
from path_subspace import context, read


def inputs(cfg):
    parent = Path(cfg["parent"])
    meta = read(parent / "metadata.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    assert digest(parent / "artifacts/release_edits.pt") == cfg["release_sha256"]
    for name, pin in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == pin
    info = read(parent / "construction.json")
    assert info["layer"] == cfg["layer"] and info["head"] == cfg["head"]
    baseline = Path(cfg["baseline_parent"])
    bm = read(baseline / "metadata.json")
    assert bm["completed"] and read(baseline / "validation.json")["status"] == "passed"
    for name, pin in bm["raw_sha256"].items():
        assert digest(baseline / "artifacts" / name) == pin
    root, load, fusion, models = context(cfg, all_tasks=True)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    assert bm["author_sha256"] == read("configs/sources.json")["AGA"]["sha256"]
    assert read(baseline / "protocol.json")["coefficient"] == cfg["coefficient"]
    for name, pin in bm["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    release = copy.deepcopy(models[cfg["task"]]).cpu().eval()
    edits = torch.load(parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    from dataset.registry import get_dataset

    heads = {t: load(str(root / f"head_{t}.pt")).cpu().eval() for t in cfg["evaluation_tasks"]}
    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in heads
    }
    return root, fusion, models, release, heads, data


def reset(model, anchor, cfg):
    result = copy.deepcopy(model).cpu().eval()
    current = result.model.visual.transformer.resblocks[cfg["layer"]].attn
    public = anchor.model.visual.transformer.resblocks[cfg["layer"]].attn
    section = slice(768 + cfg["head"] * 64, 768 + (cfg["head"] + 1) * 64)
    current.in_proj_weight[section].copy_(public.in_proj_weight[section])
    return result


def states(teacher, aga, anchor, cfg):
    return dict(
        direct=teacher,
        aga=aga,
        reset=reset(teacher, anchor, cfg),
        aga_reset=reset(aga, anchor, cfg),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/value_conditioned_routing_eval.json")
    parser.add_argument("--output", default="outputs/2026-10-12/value_conditioned_routing_eval")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate(args, input_loader=inputs, state_builder=states)
    else:
        run(
            args,
            input_loader=inputs,
            state_builder=states,
            parent_presets={"original": "original", "protected": "attention_only"},
        )
