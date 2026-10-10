"""Measure the public replacement floor for a declared local edit support.

This is a mechanism exclusion diagnostic on unprotected experts, not an attack
success claim on a released protected model. Reuse author data/evaluation/TA.
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
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import digest, merge_all
from operator_diagnostics import paired_counts
from subspace_probe import make_loader, save_json


@torch.no_grad()
def reset_support(original, anchor, layers):
    """Replace V/O/MLP; check every Q/K slice, norm and outside tensor unchanged."""
    model = copy.deepcopy(original).cpu().eval()
    before, base, state = original.state_dict(), anchor.state_dict(), model.state_dict()
    slices = {}
    for layer in layers:
        prefix = f"model.visual.transformer.resblocks.{layer}."
        for key in state:
            suffix = key.removeprefix(prefix)
            if not key.startswith(prefix):
                continue
            if suffix.startswith(("mlp.", "attn.out_proj.")):
                state[key].copy_(base[key])
                slices[key] = [0, state[key].shape[0]]
            elif suffix in ("attn.in_proj_weight", "attn.in_proj_bias"):
                width = state[key].shape[0] // 3
                state[key][2 * width :].copy_(base[key][2 * width :])
                slices[key] = [2 * width, 3 * width]
    assert len(slices) == 24
    for key, value in state.items():
        if key not in slices:
            assert torch.equal(value, before[key])
        else:
            start, end = slices[key]
            assert torch.equal(value[start:end], base[key][start:end])
            assert torch.equal(value[:start], before[key][:start])
            assert torch.equal(value[end:], before[key][end:])
    return model, slices


@torch.no_grad()
def run(args):
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    assert config["task"] == "Cars" and config["layers"] == [9, 10, 11]
    parent = Path(config["parent"])
    parent_meta = json.loads((parent / "metadata.json").read_bytes())
    raw_path = parent / "artifacts/predictions.pt"
    assert digest(raw_path) == parent_meta["predictions_sha256"]
    prior = torch.load(raw_path, map_location="cpu", weights_only=True)
    prior_rows = json.loads((parent / "results.json").read_bytes())
    indices = json.loads((parent / "training_indices.json").read_bytes())["Cars"]["confirmation"]
    assert len(indices) == 2048
    save_json(out / "protocol.json", config)
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    assert fusion["coefficient"] == config["coefficient"]
    models = {t: load(str(root / t / "finetuned.pt")) for t in fusion["tasks"]}
    models["anchor"] = load(str(root / "zeroshot.pt"))
    task, anchor = models["Cars"], models["anchor"]
    head = load(str(root / "head_Cars.pt"))
    reset, support = reset_support(task, anchor, config["layers"])
    save_json(out / "support.json", support)
    reset_file = artifacts / "public_reset.pt"
    torch.save(reset.state_dict(), reset_file)
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo) / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset
    from torch.utils.data import DataLoader, TensorDataset

    data = get_dataset("Cars", task.val_preprocess, args.data).train_dataset
    batches = list(make_loader(data, indices))
    loader = DataLoader(
        TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
        batch_size=128,
        shuffle=False,
        pin_memory=True,
    )
    rows, raw = [], {}
    for preset, own in [("original", task), ("public_reset", reset)]:
        combined = merge_all(dict(models, Cars=own), root, config["coefficient"])
        for group, model in [("standalone", own), ("fusion", combined)]:
            predictions, labels = [], []
            score = evaluate(
                model,
                head,
                loader,
                "cuda",
                prediction_sink=lambda p, y: (
                    predictions.append(p.clone()),
                    labels.append(y.clone()),
                ),
            )
            p, y = torch.cat(predictions), torch.cat(labels)
            baseline = next(
                r
                for r in prior_rows
                if r["task"] == "Cars"
                and r["split"] == "confirmation"
                and r["group"] == group
                and r["component"] == "none"
            )
            reference = prior[baseline["id"]]
            assert torch.equal(y, reference["labels"])
            if preset == "original":
                assert torch.equal(p, reference["prediction"])
            row = dict(
                preset=preset,
                group=group,
                original_replay_exact=bool(preset == "original"),
                parent_control_id=baseline["id"],
                paired=paired_counts(reference["prediction"].eq(y), p.eq(y)),
                **score,
            )
            rows.append(row)
            raw[preset, group] = dict(prediction=p, labels=y)
            print("PUBLIC_RESET", preset, group, score["correct"], score["n"], flush=True)
        del combined
    save_json(out / "results.json", rows)
    raw_file = artifacts / "predictions.pt"
    torch.save(raw, raw_file)
    loss = next(
        r["paired"]["loss_pp"]
        for r in rows
        if r["preset"] == "public_reset" and r["group"] == "fusion"
    )
    save_json(
        out / "decision.json",
        dict(
            new_defense=False,
            proposed_support_can_meet_gap=loss >= config["minimum_gap_pp"],
            maximum_drop_from_original_direct_pp=loss,
            reason="Complete actual-support replacement removes any edits within this support. This is a conditional floor, not protected-model accuracy.",
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            predictions_sha256=digest(raw_file),
            reset_sha256=digest(reset_file),
            parent_predictions_sha256=digest(raw_path),
            parent_source_sha256=parent_meta["source_sha256"],
            checkpoint_sha256=parent_meta["checkpoint_sha256"],
            helper_sha256={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            elapsed_seconds=time.perf_counter() - start,
            fits=0,
            optimizer_steps=0,
            fresh_data=False,
            test_usage="none",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/relocation_public_reset.json")
    parser.add_argument("--output", default="outputs/2026-10-10/relocation_public_reset")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
