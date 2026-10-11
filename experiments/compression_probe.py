"""Challenge one response-compressed expert with an existing native-score attack.

The control was selected after inspecting response_cleanup tests. It is a
standard compression baseline, not a new defense, independent confirmatory
experiment, or irreversible-information guarantee. Query images are disjoint
from the compression calibration; all expert controls use identical queries.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import reset
from eight_task_benchmark import digest, merge_all
from local_erasure import cache, sites
from operator_diagnostics import paired_counts
from path_subspace import context, read
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


def setup(cfg):
    parent = Path(cfg["parent"])
    assert read(parent / "validation.json")["status"] == "passed"
    meta = read(parent / "metadata.json")
    file = parent / "artifacts/margin_mnist_original_cleanup.pt"
    assert digest(file) == meta["raw_sha256"][file.name]
    root, load, fusion, models = context(cfg, all_tasks=True)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    compressed = copy.deepcopy(models[cfg["task"]]).cpu().eval()
    saved = torch.load(file, map_location="cpu", weights_only=True)
    for key, value in saved["edits"].items():
        compressed.state_dict()[key].copy_(value)
    selection = read(Path(cfg["profile_parent"]) / "training_indices.json")[cfg["task"]][
        "selection"
    ]
    ids = selection[cfg["query_offset"] : cfg["query_offset"] + cfg["samples"]]
    assert len(ids) == cfg["samples"] == len(set(ids))
    calibration = read(parent / "training_indices.json")["margin_mnist"]["calibration"]
    assert not set(ids).intersection(calibration)
    images = torch.stack([data[cfg["task"]].train_dataset[i][0] for i in ids])
    inverse, inverse_checks = head_inverse(
        heads[cfg["task"]], [h for t, h in heads.items() if t != cfg["task"]]
    )
    return root, fusion, models, data, heads, compressed, images, ids, inverse, inverse_checks


@torch.no_grad()
def run(args):
    cfg = read(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, fusion, models, data, heads, compressed, images, ids, inverse, inverse_checks = setup(cfg)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    loaders = {t: cache(data[t]) for t in cfg["evaluation_tasks"]}
    raw, rows, fits = {}, [], {}
    for preset, release in (("original", models[cfg["task"]]), ("compressed", compressed)):
        student = merge_all(dict(models, **{cfg["task"]: release}), root, cfg["coefficient"])
        restored, fits[preset], evidence = recover(
            release, student, heads[cfg["task"]], inverse, images, cfg["ridge_ratio"]
        )
        torch.save(evidence, out / f"artifacts/{preset}_readout.pt")
        merged = merge_all(dict(models, **{cfg["task"]: restored}), root, cfg["coefficient"])
        expected = student.model.visual.proj.double() + evidence["projection_delta"]
        assert float((merged.model.visual.proj.double() - expected).norm() / expected.norm()) < 1e-6
        for task in cfg["evaluation_tasks"]:
            p, y = [], []
            score = evaluate(
                merged,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            key = f"{preset}/{task}"
            raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
            rows.append(dict(id=key, preset=preset, dataset=task, **score))
            print("RESULT", key, score["correct"], score["n"], flush=True)
        del student, merged, restored
    for row in rows:
        value, control = raw[row["id"]], raw[f"original/{row['dataset']}"]
        row["matched"] = paired_counts(
            control["prediction"].eq(control["labels"]), value["prediction"].eq(value["labels"])
        )
    # Complete declared support replacement compiles to the same state, independent of queries.
    reset_states = [
        reset(m, models["anchor"], sites(cfg["layers"]), "public_support").state_dict()
        for m in (models[cfg["task"]], compressed)
    ]
    assert all(torch.equal(v, reset_states[1][k]) for k, v in reset_states[0].items())
    save_json(
        out / "support_reset.json",
        dict(
            full_states_identical=True,
            layers=cfg["layers"],
            calibration_queries=0,
            accuracy_reused_from="outputs/2026-10-11/path_subspace",
            target_correct=8723,
            target_n=10000,
        ),
    )
    save_json(out / "results.json", rows)
    save_json(out / "fitting.json", fits)
    save_json(
        out / "query_indices.json",
        dict(indices=ids, compression_calibration_overlap=0, inverse_checks=inverse_checks),
    )
    torch.save(raw, out / "artifacts/predictions.pt")
    helpers = [
        "clip_benchmark.py",
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "local_erasure.py",
        "operator_diagnostics.py",
        "path_subspace.py",
        "polynomial_probe.py",
        "score_recovery.py",
        "subspace_probe.py",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - started,
            actual_full_evaluations=len(rows),
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path(__file__).with_name(n)) for n in helpers},
            parent_metadata_sha256=digest(Path(cfg["parent"]) / "metadata.json"),
            raw_sha256={p.name: digest(p) for p in (out / "artifacts").iterdir()},
            checkpoint_sha256={
                n: digest(root / n)
                for n in [
                    "zeroshot.pt",
                    *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
                    *[f"head_{t}.pt" for t in cfg["evaluation_tasks"]],
                ]
            },
            calibration_labels_used=False,
            teacher_hidden_features_used=False,
            defender_gradient_steps=0,
            selected_after_test_inspection=True,
            novelty_claim=False,
            new_defense_validated=False,
        ),
    )
    print("COMPLETE", len(rows), flush=True)


@torch.no_grad()
def validate(args):
    out = Path(args.output)
    cfg, meta, rows = [read(out / n) for n in ("protocol.json", "metadata.json", "results.json")]
    assert meta["completed"] and meta["actual_full_evaluations"] == len(rows) == 6
    assert (
        digest(__file__) == meta["source_sha256"] and digest(args.config) == meta["config_sha256"]
    )
    for n, h in meta["helper_sha256"].items():
        assert digest(Path(__file__).with_name(n)) == h
    for n, h in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / n) == h
    assert digest(Path(cfg["parent"]) / "metadata.json") == meta["parent_metadata_sha256"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, fusion, models, data, heads, compressed, images, ids, inverse, _ = setup(cfg)
    for n, h in meta["checkpoint_sha256"].items():
        assert digest(root / n) == h
    assert ids == read(out / "query_indices.json")["indices"]
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    for row in rows:
        task = row["dataset"]
        truth = torch.as_tensor(
            [y for _, y in data[task].test_dataset._samples]
            if task == "Cars"
            else data[task].test_dataset.targets,
            dtype=torch.long,
        )
        value, control = raw[row["id"]], raw[f"original/{task}"]
        assert torch.equal(value["labels"], truth)
        correct = int(value["prediction"].eq(truth).sum())
        assert (
            correct == row["correct"]
            and row["n"] == len(truth) == fusion["expected_test_counts"][task]
            and row["top1"] == correct / len(truth)
        )
        assert row["matched"] == paired_counts(
            control["prediction"].eq(truth), value["prediction"].eq(truth)
        )
    for preset, release in (("original", models[cfg["task"]]), ("compressed", compressed)):
        student = merge_all(dict(models, **{cfg["task"]: release}), root, cfg["coefficient"])
        restored, fit, evidence = recover(
            release, student, heads[cfg["task"]], inverse, images, cfg["ridge_ratio"]
        )
        assert fit == read(out / "fitting.json")[preset]
        saved = torch.load(
            out / f"artifacts/{preset}_readout.pt", map_location="cpu", weights_only=True
        )
        assert saved.keys() == evidence.keys() and all(
            torch.equal(v, saved[k]) for k, v in evidence.items()
        )
        merged = merge_all(dict(models, **{cfg["task"]: restored}), root, cfg["coefficient"])
        for task in cfg["evaluation_tasks"]:
            p, y = [], []
            evaluate(
                merged,
                heads[task],
                make_loader(data[task].test_dataset, list(range(128))),
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            assert torch.equal(torch.cat(p), raw[f"{preset}/{task}"]["prediction"][:128])
        del student, restored, merged
    a, b = [
        reset(m, models["anchor"], sites(cfg["layers"]), "public_support").state_dict()
        for m in (models[cfg["task"]], compressed)
    ]
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    parent = read("outputs/2026-10-11/path_subspace/results.json")
    reference = next(r for r in parent if r["id"] == "MNIST/original/support_reset/fusion/MNIST")
    assert reference["correct"] == read(out / "support_reset.json")["target_correct"]
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            official_labels_checked=True,
            all_six_prefixes_replayed=True,
            both_256_query_fits_replayed_exact=True,
            support_reset_states_identical=True,
            results_sha256=digest(out / "results.json"),
            source_sha256=digest(__file__),
            new_defense_validated=False,
        ),
    )
    print("VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compression_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-11/compression_probe")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate(args) if args.validate else run(args)
