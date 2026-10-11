"""Omit the entire target expert, then reconstruct native scores with 256 queries.

An extra functional recovery diagnostic, separate from paper AGA. The student
contains seven expert task vectors and a zero target vector. Only its final
projection is fitted; no target-release tensor enters the merged state.
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
from response_cleanup import load_release
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


def setup(cfg):
    root, load, fusion, models = context(cfg, all_tasks=True)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    teacher = load_release(cfg["teacher"], models[cfg["task"]])
    parent = Path(cfg["teacher_parent"])
    assert read(parent / "validation.json")["raw_artifacts_checked"]
    teacher_check = Path(cfg["teacher_evaluation"])
    assert read(teacher_check / "validation.json")["status"] == "passed"
    assert (
        digest(cfg["teacher"]["release"]) == read(teacher_check / "metadata.json")["release_sha256"]
    )
    selection = read(Path(cfg["profile_parent"]) / "training_indices.json")[cfg["task"]][
        "selection"
    ]
    ids = selection[cfg["query_offset"] : cfg["query_offset"] + cfg["samples"]]
    defender = read(parent / "training_indices.json")[cfg["task"]]["calibration"]
    assert len(ids) == cfg["samples"] == len(set(ids)) and not set(ids).intersection(defender)
    images = torch.stack([data[cfg["task"]].train_dataset[i][0] for i in ids])
    inverse, checks = head_inverse(
        heads[cfg["task"]], [h for t, h in heads.items() if t != cfg["task"]]
    )
    student = merge_all(dict(models, **{cfg["task"]: models["anchor"]}), root, cfg["coefficient"])
    return root, fusion, models, data, heads, teacher, images, ids, inverse, checks, student


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
    root, fusion, models, data, heads, teacher, images, ids, inverse, checks, student = setup(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    loaders = {t: cache(data[t]) for t in cfg["evaluation_tasks"]}
    raw, rows, fits, states = {}, [], {}, {}
    for preset, release in (
        ("omitted", None),
        ("original_query", models[cfg["task"]]),
        ("protected_query", teacher),
    ):
        if release is None:
            fitted = student
        else:
            unused, fits[preset], evidence = recover(
                release, student, heads[cfg["task"]], inverse, images, cfg["ridge_ratio"]
            )
            del unused
            fitted = copy.deepcopy(student).cpu().eval()
            fitted.model.visual.proj.copy_(
                (student.model.visual.proj.double() + evidence["projection_delta"]).float()
            )
            assert all(
                torch.equal(v, student.state_dict()[k])
                for k, v in fitted.state_dict().items()
                if k != "model.visual.proj"
            )
            evidence["compiled_omitted_projection"] = fitted.model.visual.proj.detach().clone()
            torch.save(evidence, out / f"artifacts/{preset}_readout.pt")
            states[preset] = True
        for task in cfg["evaluation_tasks"]:
            p, y = [], []
            score = evaluate(
                fitted,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            key = f"{preset}/{task}"
            raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
            rows.append(dict(id=key, preset=preset, dataset=task, **score))
            print("RESULT", key, score["correct"], score["n"], flush=True)
        del fitted
    for row in rows:
        value = raw[row["id"]]
        base = raw[f"omitted/{row['dataset']}"]
        row["change_from_omission"] = paired_counts(
            base["prediction"].eq(base["labels"]), value["prediction"].eq(value["labels"])
        )
        if row["preset"] != "omitted":
            control = raw[f"original_query/{row['dataset']}"]
            row["matched"] = paired_counts(
                control["prediction"].eq(control["labels"]), value["prediction"].eq(value["labels"])
            )
    partners = {
        f"{p}/{t}": int(raw[f"{p}/{t}"]["prediction"].ne(raw[f"omitted/{t}"]["prediction"]).sum())
        for p in ("original_query", "protected_query")
        for t in cfg["evaluation_tasks"]
        if t != cfg["task"]
    }
    save_json(out / "results.json", rows)
    save_json(out / "fitting.json", fits)
    save_json(
        out / "query_indices.json",
        dict(indices=ids, defender_calibration_overlap=0, head_inverse_checks=checks),
    )
    save_json(
        out / "state_checks.json",
        dict(
            target_task_vector_zero=True,
            all_nonprojection_tensors_equal_to_seven_expert_student=states,
            partner_prediction_changes=partners,
            release_tensors_merged=False,
            teacher_outputs_used=True,
        ),
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
        "response_cleanup.py",
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
            teacher_state_sha256=digest(cfg["teacher"]["release"]),
            raw_sha256={p.name: digest(p) for p in (out / "artifacts").iterdir()},
            checkpoint_sha256={
                n: digest(root / n)
                for n in [
                    "zeroshot.pt",
                    *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
                    *[f"head_{t}.pt" for t in cfg["evaluation_tasks"]],
                ]
            },
            target_parameters_used_in_student=False,
            calibration_labels_used=False,
            teacher_hidden_features_used=False,
            optimizer_steps=0,
            paper_aga_result=False,
            novelty_claim=False,
            new_defense_validated=False,
        ),
    )
    print("COMPLETE", len(rows), flush=True)


@torch.no_grad()
def validate(args):
    out = Path(args.output)
    cfg, meta, rows = [read(out / n) for n in ("protocol.json", "metadata.json", "results.json")]
    assert meta["completed"] and len(rows) == meta["actual_full_evaluations"] == 9
    assert (
        digest(__file__) == meta["source_sha256"] and digest(args.config) == meta["config_sha256"]
    )
    for n, h in meta["helper_sha256"].items():
        assert digest(Path(__file__).with_name(n)) == h
    for n, h in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / n) == h
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, fusion, models, data, heads, teacher, images, ids, inverse, _, student = setup(cfg)
    assert ids == read(out / "query_indices.json")["indices"]
    for n, h in meta["checkpoint_sha256"].items():
        assert digest(root / n) == h
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(raw) == len(rows) == len({r["id"] for r in rows})
    for r in rows:
        t = r["dataset"]
        truth = torch.as_tensor(
            [y for _, y in data[t].test_dataset._samples]
            if t == "Cars"
            else data[t].test_dataset.targets,
            dtype=torch.long,
        )
        v = raw[r["id"]]
        assert torch.equal(v["labels"], truth)
        correct = int(v["prediction"].eq(truth).sum())
        assert (
            correct == r["correct"]
            and r["n"] == len(truth) == fusion["expected_test_counts"][t]
            and r["top1"] == correct / len(truth)
        )
    for preset, release in (
        ("omitted", None),
        ("original_query", models[cfg["task"]]),
        ("protected_query", teacher),
    ):
        fitted = copy.deepcopy(student).cpu().eval()
        if release is not None:
            unused, fit, e = recover(
                release, student, heads[cfg["task"]], inverse, images, cfg["ridge_ratio"]
            )
            del unused
            saved = torch.load(
                out / f"artifacts/{preset}_readout.pt", map_location="cpu", weights_only=True
            )
            assert fit == read(out / "fitting.json")[preset] and all(
                torch.equal(v, saved[k]) for k, v in e.items()
            )
            fitted.model.visual.proj.copy_(
                (student.model.visual.proj.double() + e["projection_delta"]).float()
            )
            assert torch.equal(fitted.model.visual.proj, saved["compiled_omitted_projection"])
        assert all(
            torch.equal(v, student.state_dict()[k])
            for k, v in fitted.state_dict().items()
            if k != "model.visual.proj"
        )
        for task in cfg["evaluation_tasks"]:
            p, y = [], []
            evaluate(
                fitted,
                heads[task],
                make_loader(data[task].test_dataset, list(range(128))),
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            assert torch.equal(torch.cat(p), raw[f"{preset}/{task}"]["prediction"][:128])
        del fitted
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            official_labels_checked=True,
            all_nine_prefixes_replayed=True,
            both_query_fits_replayed_exact=True,
            target_release_tensors_absent_from_student=True,
            results_sha256=digest(out / "results.json"),
            source_sha256=digest(__file__),
            new_defense_validated=False,
        ),
    )
    print("VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/omission_recovery.json")
    parser.add_argument("--output", default="outputs/2026-10-11/omission_recovery")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate(args) if args.validate else run(args)
