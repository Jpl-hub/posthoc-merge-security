"""Measure the expert-discard floor before claiming any anti-merging effect."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import digest, merge_all
from operator_diagnostics import paired_counts
from subspace_probe import save_json


@torch.no_grad()
def run(args):
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    assert config["coefficient"] == fusion["coefficient"] == 0.2
    assert config["presets"] == ["all_eight", *[f"omit_{t}" for t in config["tasks"]]]
    save_json(out / "protocol.json", config)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    models = {name: load(str(root / name / "finetuned.pt")) for name in fusion["tasks"]}
    models["anchor"] = load(str(root / "zeroshot.pt"))
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset
    from torch.utils.data import DataLoader, TensorDataset

    loaders, heads = {}, {}
    for task in config["tasks"]:
        data = get_dataset(
            task, models["anchor"].val_preprocess, args.data, batch_size=128, num_workers=2
        )
        assert len(data.test_dataset) == fusion["expected_test_counts"][task]
        batches = list(data.test_loader)
        loaders[task] = DataLoader(
            TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
            batch_size=128,
            pin_memory=True,
        )
        heads[task] = load(str(root / f"head_{task}.pt"))
        print("INPUT_CACHE", task, len(data.test_dataset), flush=True)
    parent = Path(config["parent"])
    parent_meta = json.loads((parent / "metadata.json").read_bytes())
    assert (
        digest(parent / "artifacts/predictions.pt") == parent_meta["raw_sha256"]["predictions.pt"]
    )
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows, raw = [], {}
    for preset in config["presets"]:
        assert time.perf_counter() - started < config["time_limit_seconds"]
        chosen = dict(models)
        omitted = None if preset == "all_eight" else preset.removeprefix("omit_")
        if omitted:
            chosen[omitted] = models["anchor"]
        combined = merge_all(chosen, root, config["coefficient"])
        for task in config["tasks"]:
            predictions, labels = [], []
            score = evaluate(
                combined,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
            )
            p, y = torch.cat(predictions), torch.cat(labels)
            reference = prior[f"original/direct/fusion/{task}"]
            assert torch.equal(y, reference["labels"])
            if omitted is None:
                assert torch.equal(p, reference["prediction"])
            key = f"{preset}/{task}"
            raw[key] = dict(prediction=p, labels=y)
            rows.append(
                dict(
                    id=key,
                    preset=preset,
                    omitted_expert=omitted,
                    dataset=task,
                    group="fusion",
                    actual_nonzero_experts=7 if omitted else 8,
                    paired=paired_counts(reference["prediction"].eq(y), p.eq(y)),
                    **score,
                )
            )
            torch.save(raw, artifacts / "predictions.pt")
            save_json(out / "results.json", rows)
            print("OMISSION", key, score["correct"], score["n"], flush=True)
        del combined
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            fusion_config_sha256=digest(config["fusion_config"]),
            parent_predictions_sha256=parent_meta["raw_sha256"]["predictions.pt"],
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            checkpoint_sha256=parent_meta["checkpoint_sha256"],
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            elapsed_seconds=time.perf_counter() - started,
            optimizer_steps=0,
            response_fits=0,
            calibration_samples=0,
            protected_model_evaluated=False,
            original_official_predictions_replay_exact=True,
            coefficient_renormalized=False,
            test_usage="Fixed baseline only; no candidate selection or tuning",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expert_omission.json")
    parser.add_argument("--output", default="outputs/2026-10-11/expert_omission")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
