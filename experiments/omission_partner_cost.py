"""Measure the known shared-head SVHN cost of frozen omission-score recovery.

No new fit, query, defense, coefficient selection or release weight is created.
All three student states are reproduced from the validated parent artifacts.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import context, read
from subspace_probe import make_loader, save_json


@torch.no_grad()
def run(args):
    parent = Path(args.parent)
    cfg, meta = read(parent / "protocol.json"), read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert not (parent / "svhn_results.json").exists()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg, all_tasks=True)
    from dataset.registry import get_dataset

    head = load(str(root / "head_SVHN.pt"))
    target_head = load(str(root / "head_MNIST.pt"))
    assert torch.equal(head.weight, target_head.weight) and torch.equal(head.bias, target_head.bias)
    data = get_dataset(
        "SVHN", models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2
    )
    loader = cache(data)
    truth = torch.as_tensor(data.test_dataset.labels, dtype=torch.long)
    assert len(truth) == fusion["expected_test_counts"]["SVHN"] == 26032
    student = merge_all(dict(models, **{cfg["task"]: models["anchor"]}), root, cfg["coefficient"])
    rows, raw = [], {}
    for preset in ("omitted", "original_query", "protected_query"):
        fitted = copy.deepcopy(student).cpu().eval()
        if preset != "omitted":
            path = parent / f"artifacts/{preset}_readout.pt"
            assert digest(path) == meta["raw_sha256"][path.name]
            saved = torch.load(path, map_location="cpu", weights_only=True)
            fitted.model.visual.proj.copy_(saved["compiled_omitted_projection"])
        assert all(
            torch.equal(v, student.state_dict()[k])
            for k, v in fitted.state_dict().items()
            if k != "model.visual.proj"
        )
        p, y = [], []
        score = evaluate(
            fitted, head, loader, "cuda", prediction_sink=lambda a, b: (p.append(a), y.append(b))
        )
        value = dict(prediction=torch.cat(p), labels=torch.cat(y))
        assert torch.equal(value["labels"], truth)
        assert score["correct"] == int(value["prediction"].eq(truth).sum())
        raw[preset] = value
        rows.append(dict(preset=preset, dataset="SVHN", **score))
        prefix = []
        evaluate(
            fitted,
            head,
            make_loader(data.test_dataset, list(range(128))),
            "cuda",
            prediction_sink=lambda a, b: prefix.append(a),
        )
        assert torch.equal(torch.cat(prefix), value["prediction"][:128])
        print("SVHN_COST", preset, score["correct"], score["n"], flush=True)
        del fitted
    for row in rows:
        value = raw[row["preset"]]
        row["change_from_omission"] = paired_counts(
            raw["omitted"]["prediction"].eq(truth), value["prediction"].eq(truth)
        )
    file = parent / "artifacts/svhn_predictions.pt"
    torch.save(raw, file)
    save_json(parent / "svhn_results.json", rows)
    helpers = [
        "clip_benchmark.py",
        "eight_task_benchmark.py",
        "local_erasure.py",
        "operator_diagnostics.py",
        "path_subspace.py",
        "subspace_probe.py",
    ]
    save_json(
        parent / "svhn_validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            actual_full_evaluations=3,
            no_new_fit_or_query=True,
            same_digit_head=True,
            all_full_labels_and_counts_checked=True,
            all_three_prefixes_replayed=True,
            source_sha256=digest(__file__),
            helper_sha256={n: digest(Path(__file__).with_name(n)) for n in helpers},
            parent_metadata_sha256=digest(parent / "metadata.json"),
            predictions_sha256=digest(file),
            head_sha256=digest(root / "head_SVHN.pt"),
            results_sha256=digest(parent / "svhn_results.json"),
            other_four_tasks_unmeasured=True,
        ),
    )
    print("SVHN_COST_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="outputs/2026-10-11/omission_recovery")
    run(parser.parse_args())
