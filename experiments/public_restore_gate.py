"""Test one target-dependent region against public reset and fixed recovery.

An unprotected teacher control, not a release or a defense success. Reuse the
existing fixed-student attack to screen a region before constructing weights.
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
from local_release_audit import compile_query
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read
from polynomial_probe import ridge_update
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


def inputs(cfg):
    root, load, fusion, models = context(cfg, all_tasks=True)
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    for name, sha in expected.items():
        assert digest(root / name) == sha
    parent = Path(cfg["baseline_parent"])
    meta = read(parent / "metadata.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    assert meta["checkpoint_sha256"] == expected
    assert digest(parent / "artifacts/predictions.pt") == meta["raw_sha256"]["predictions.pt"]
    assert meta["source_manifest"]["experiments/eight_task_benchmark.py"] == digest(
        "experiments/eight_task_benchmark.py"
    )
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    target = cfg["target"]
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    selection = read(Path(cfg["profile_parent"]) / "selection.json")[target]
    assert selection["layers"] == cfg["layers"] and not selection["partner_used"]
    ids = read(Path(cfg["profile_parent"]) / "training_indices.json")[target]["selection"][
        : cfg["attacker_samples"]
    ]
    assert len(ids) == cfg["attacker_samples"]
    images = torch.stack([data[target].train_dataset[i][0] for i in ids])
    return root, fusion, models, heads, data, ids, images, prior


@torch.no_grad()
def states(cfg, models, root, heads, images):
    target = cfg["target"]
    erased, keys = replace_components(models[target], models["anchor"], cfg["layers"], "attn_mlp")
    student = merge_all(dict(models, **{target: erased}), root, cfg["coefficient"])
    restored, fit, evidence = compile_query(models[target], erased, student, heads, images, cfg)
    queried = merge_all(dict(models, **{target: restored}), root, cfg["coefficient"])
    actual = queried.model.visual.proj.double() - student.model.visual.proj.double()
    error = float(
        (actual - evidence["projection_delta"]).norm() / evidence["projection_delta"].norm()
    )
    assert error < 1e-6, error
    assert all(
        torch.equal(v, student.state_dict()[k])
        for k, v in queried.state_dict().items()
        if k != "model.visual.proj"
    )
    fit.update(
        compiled_projection_relative_error=error,
        region_keys=keys,
        whole_attention_including_qk_replaced=True,
    )
    return (
        {
            "reset/standalone": erased,
            "reset/fusion": student,
            "reset_query/standalone": restored,
            "reset_query/fusion": queried,
        },
        fit,
        evidence,
    )


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, fusion, models, heads, data, ids, images, prior = inputs(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    save_json(out / "attack_indices.json", ids)
    target = cfg["target"]
    built, fit, evidence = states(cfg, models, root, heads, images)
    torch.save(evidence, artifacts / "query_fit.pt")
    save_json(out / "fitting.json", fit)
    loaders = {t: cache(d) for t, d in data.items()}
    raw, rows = {}, []
    for task in cfg["evaluation_tasks"]:
        key = f"original/fusion/{task}"
        source = f"original/direct/fusion/{task}"
        raw[key] = prior[source]
        labels = truth(data[task], task)
        assert torch.equal(labels, raw[key]["labels"])
        correct = int(raw[key]["prediction"].eq(labels).sum())
        rows.append(
            dict(
                key=key,
                preset="original",
                group="fusion",
                dataset=task,
                correct=correct,
                n=len(labels),
                top1=correct / len(labels),
                reused=True,
                parent_key=source,
            )
        )
    built = {"original/standalone": models[target], **built}
    for case, model in built.items():
        preset, group = case.split("/")
        for task in [target] if group == "standalone" else cfg["evaluation_tasks"]:
            p, y = [], []
            key = f"{case}/{task}"
            score = evaluate(
                model,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
            rows.append(
                dict(key=key, preset=preset, group=group, dataset=task, reused=False, **score)
            )
            save_json(out / "results.json", rows)
            torch.save(raw, artifacts / "predictions.pt")
            print("PUBLIC_RESTORE_GATE", key, score["correct"], score["n"], flush=True)
    for row in rows:
        labels = raw[row["key"]]["labels"]
        reference = raw[f"original/{row['group']}/{row['dataset']}"]
        row["original_change"] = paired_counts(
            reference["prediction"].eq(labels), raw[row["key"]]["prediction"].eq(labels)
        )
    save_json(out / "results.json", rows)
    assert len(rows) == 15 and sum(not r["reused"] for r in rows) == 11
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            fresh_full_evaluations=11,
            reused_baselines=4,
            new_defense=False,
            teacher_is_unprotected=True,
            teacher_unique_query_inputs=256,
            query_fits=1,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            parent_prediction_sha256=read(Path(cfg["baseline_parent"]) / "metadata.json")[
                "raw_sha256"
            ]["predictions.pt"],
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            dataset_source_sha256={
                str(p): digest(p)
                for p in sorted(Path("external/secure-merge-attack/datasets").glob("*.py"))
            },
            target_truth_or_teacher_hidden_used=False,
            eight_task_mean=False,
        ),
    )


@torch.no_grad()
def validate(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for p, sha in meta["source_manifest"].items():
        assert digest(p) == sha, p
    for n, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / n) == sha
    root, fusion, models, heads, data, ids, _images, prior = inputs(cfg)
    assert ids == read(out / "attack_indices.json")
    target = cfg["target"]
    evidence = torch.load(out / "artifacts/query_fit.pt", map_location="cpu", weights_only=True)
    update, fit = ridge_update(
        evidence["design"].double().cuda(),
        evidence["residual"].double().cuda(),
        cfg["ridge_ratio"],
        solver="primal",
    )
    error = float((update.cpu() - evidence["update"]).norm() / evidence["update"].norm())
    assert error < 1e-6, error
    erased, _ = replace_components(models[target], models["anchor"], cfg["layers"], "attn_mlp")
    restored = copy.deepcopy(erased).cpu().eval()
    restored.model.visual.proj.copy_(evidence["restored_projection"])
    built = {
        "original/standalone": models[target],
        "original/fusion": merge_all(models, root, 0.2),
        "reset/standalone": erased,
        "reset/fusion": merge_all(dict(models, **{target: erased}), root, 0.2),
        "reset_query/standalone": restored,
        "reset_query/fusion": merge_all(dict(models, **{target: restored}), root, 0.2),
    }
    actual = (
        built["reset_query/fusion"].model.visual.proj.double()
        - built["reset/fusion"].model.visual.proj.double()
    )
    assert (
        float((actual - evidence["projection_delta"]).norm() / evidence["projection_delta"].norm())
        < 1e-6
    )
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 15
    for row in rows:
        task = row["dataset"]
        v = raw[row["key"]]
        labels = truth(data[task], task)
        assert torch.equal(v["labels"], labels)
        assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
        assert row["correct"] == int(v["prediction"].eq(labels).sum()) and row["top1"] == row[
            "correct"
        ] / len(labels)
        reference = raw[f"original/{row['group']}/{task}"]
        assert row["original_change"] == paired_counts(
            reference["prediction"].eq(labels), v["prediction"].eq(labels)
        )
        if row["reused"]:
            assert torch.equal(v["prediction"], prior[row["parent_key"]]["prediction"])
        sink = []
        evaluate(
            built[f"{row['preset']}/{row['group']}"],
            heads[task],
            make_loader(data[task].test_dataset, list(range(128)), 128),
            "cuda",
            prediction_sink=lambda p, _y: sink.append(p),
        )
        assert torch.equal(torch.cat(sink), v["prediction"][:128])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            counts_pairs_and_official_prefixes_checked=15,
            independent_primal_ridge_relative_error=error,
            full_baseline_arrays_reused_and_checked=4,
            public_full_attention_mlp_reset_compilation_checked=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            new_defense=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/public_restore_gate.json")
    parser.add_argument("--output", default="outputs/2026-10-11/public_restore_gate_cars")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
