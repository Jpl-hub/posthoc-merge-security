"""Audit omitted partner costs of two already frozen DTD score-recovery attacks.

No new defense, fitting, queries, coefficient search or candidate selection.
Seven official tests are available; EuroSAT enters head geometry only.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from head_collapse import truth as subset_truth
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import context, read
from subspace_probe import make_loader, save_json


def truth(data, task):
    if task == "GTSRB":
        return torch.tensor([label for _, label in data.test_dataset._samples], dtype=torch.long)
    return subset_truth(data, task)


def contrasts(head):
    """Class-score differences; common class shifts cannot change argmax."""
    c = head.weight.detach().double().cpu()
    assert torch.count_nonzero(head.bias) == 0 and head.normalize
    return c[1:] - c[:1]


@torch.no_grad()
def geometry(heads, target, groups, rcond):
    rows, evidence = [], {}
    c = contrasts(heads[target])
    reference = float(torch.linalg.svdvals(c)[0])
    for name, tasks in groups.items():
        b = torch.cat([contrasts(heads[t]) for t in tasks])
        _, s, vh = torch.linalg.svd(b, full_matrices=False)
        rank = int(s.gt(s[0] * rcond).sum())
        basis = vh[:rank]
        null = torch.eye(b.shape[1], dtype=torch.float64) - basis.T @ basis
        projected = torch.linalg.svdvals(c @ null)
        target_rank = int(projected.gt(reference * rcond).sum())
        error = float((null @ null - null).norm())
        assert error < 1e-10
        rows.append(
            dict(
                group=name,
                partners=tasks,
                contrast_rows=len(b),
                rank=rank,
                null_dimension=b.shape[1] - rank,
                target_contrast_rank=target_rank,
                largest_singular=float(s[0]),
                smallest_singular=float(s[-1]),
                relative_cutoff=rcond,
                null_idempotence_error=error,
                all_input_score_preservation_only=True,
                classification_or_query_impossibility_claim=False,
            )
        )
        evidence[name] = dict(singular_values=s, projected_target_singular_values=projected)
    return rows, evidence


@torch.no_grad()
def holders(cfg):
    root, load, fusion, originals = context(cfg, all_tasks=True)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    parent = Path(cfg["parent"])
    meta = read(parent / "metadata.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    assert expected == meta["checkpoint_sha256"]
    for name, pin in expected.items():
        assert digest(root / name) == pin
    raw_names = (
        [f"{cfg['target']}_edits.pt"]
        + [f"{cfg['target']}_{preset}_readout.pt" for preset in ["original", "protected"]]
        + ["predictions.pt"]
    )
    for name in raw_names:
        assert digest(parent / "artifacts" / name) == meta["raw_sha256"][name]
    target = cfg["target"]
    for preset in ["original", "protected"]:
        fit = read(parent / "fitting.json")[f"{target}/{preset}"]
        assert fit["calibration_samples"] == 256
        assert (
            not fit["calibration_ground_truth_used"]
            and not fit["teacher_intermediate_features_used"]
        )
    release = copy.deepcopy(originals[target]).cpu().eval()
    edits = torch.load(
        parent / "artifacts" / f"{target}_edits.pt", map_location="cpu", weights_only=True
    )
    for name, value in edits.items():
        release.state_dict()[name].copy_(value)
    states, checks, deltas = {}, [], {}
    for preset, model in [("original", originals[target]), ("protected", release)]:
        states[preset + "_direct"] = merge_all(
            dict(originals, **{target: model}), root, cfg["coefficient"]
        )
        evidence = torch.load(
            parent / "artifacts" / f"{target}_{preset}_readout.pt",
            map_location="cpu",
            weights_only=True,
        )
        attacked = copy.deepcopy(model).cpu().eval()
        attacked.model.visual.proj.copy_(evidence["restored_projection"])
        merged = merge_all(dict(originals, **{target: attacked}), root, cfg["coefficient"])
        actual = (
            merged.model.visual.proj.double()
            - states[preset + "_direct"].model.visual.proj.double()
        )
        error = float(
            (actual - evidence["projection_delta"]).norm() / evidence["projection_delta"].norm()
        )
        assert error < 1e-6
        assert all(
            torch.equal(v, states[preset + "_direct"].state_dict()[k])
            for k, v in merged.state_dict().items()
            if k != "model.visual.proj"
        )
        states[preset + "_query"] = merged
        deltas[preset] = evidence["projection_delta"]
        checks.append(
            dict(
                preset=preset,
                compiled_projection_relative_error=error,
                teacher_queries_reused=256,
                new_queries=0,
                new_fits=0,
            )
        )
    heads = {t: load(str(root / f"head_{t}.pt")) for t in fusion["tasks"]}
    return root, load, fusion, heads, states, checks, deltas


def runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, load, fusion, heads, states, checks, deltas = holders(cfg)
    geoms, spectra = geometry(heads, cfg["target"], cfg["head_groups"], cfg["rcond"])
    save_json(out / "head_geometry.json", geoms)
    torch.save(spectra, artifacts / "head_spectra.pt")
    leak = {
        preset: {
            task: dict(
                contrast_update_frobenius=float((delta @ contrasts(head).T).norm()),
                contrast_update_over_projection_norm=float(
                    (delta @ contrasts(head).T).norm() / delta.norm()
                ),
            )
            for task, head in heads.items()
        }
        for preset, delta in deltas.items()
    }
    save_json(out / "head_leakage.json", leak)
    save_json(out / "compilation_checks.json", checks)
    prior = torch.load(
        Path(cfg["parent"]) / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    from dataset.registry import get_dataset

    raw, rows, prior_checks = {}, [], []
    for task in cfg["evaluation_tasks"]:
        data = get_dataset(
            task, states["original_direct"].val_preprocess, "data", batch_size=128, num_workers=2
        )
        loader, labels = cache(data), truth(data, task)
        for name, model in states.items():
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            p, y = [], []
            score = evaluate(
                model,
                heads[task],
                loader,
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            key = f"{name}/{task}"
            value = dict(prediction=torch.cat(p), labels=torch.cat(y))
            assert torch.equal(value["labels"], labels)
            assert score["correct"] == int(value["prediction"].eq(labels).sum())
            assert score["n"] == fusion["expected_test_counts"][task]
            raw[key] = value
            row = dict(key=key, preset=name, dataset=task, reused=False, **score)
            if name.endswith("query"):
                source = f"{name.replace('query', 'direct')}/{task}"
                row.update(
                    same_preset_direct_reference=source,
                    same_preset_direct_change=paired_counts(
                        raw[source]["prediction"].eq(labels), value["prediction"].eq(labels)
                    ),
                )
            if name.startswith("protected"):
                source = f"{name.replace('protected', 'original')}/{task}"
                row.update(
                    same_attack_original_reference=source,
                    same_attack_original_change=paired_counts(
                        raw[source]["prediction"].eq(labels), value["prediction"].eq(labels)
                    ),
                )
            if task in cfg["previously_evaluated_tasks"]:
                preset, attack = name.split("_")
                source = f"{cfg['target']}/{preset}/{'readout' if attack == 'query' else 'direct'}/fusion/{task}"
                assert torch.equal(value["labels"], prior[source]["labels"])
                assert torch.equal(value["prediction"], prior[source]["prediction"])
                prior_checks.append(
                    dict(key=key, parent_key=source, all_predictions_identical=True)
                )
            rows.append(row)
            print("PARTNER_COST", key, score["correct"], score["n"], flush=True)
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        del data, loader
    assert len(rows) == 28 and len(prior_checks) == 12
    save_json(out / "parent_prediction_checks.json", prior_checks)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            full_evaluations=28,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            parent_input_sha256={
                str(Path(cfg["parent"]) / n): digest(Path(cfg["parent"]) / n)
                for n in [
                    "metadata.json",
                    "validation.json",
                    "protocol.json",
                    "fitting.json",
                    "artifacts/DTD_edits.pt",
                    "artifacts/DTD_original_readout.pt",
                    "artifacts/DTD_protected_readout.pt",
                    "artifacts/predictions.pt",
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            dataset_source_sha256={
                str(p): digest(p)
                for p in sorted(Path("external/secure-merge-attack/datasets").glob("*.py"))
            },
            new_defense_generated=False,
            new_queries=0,
            new_fits=0,
            eurosat_accuracy_evaluated=False,
            all_eight_task_performance_claim=False,
            source_manifest=read(args.manifest)["source_manifest"],
        ),
    )


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and cfg == read(args.config)
    for name, pin in meta["source_manifest"].items():
        assert digest(name) == pin
    for name, pin in meta["parent_input_sha256"].items():
        assert digest(name) == pin
    for name, pin in meta["dataset_source_sha256"].items():
        assert digest(name) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    runtime()
    root, load, fusion, heads, states, checks, deltas = holders(cfg)
    assert checks == read(out / "compilation_checks.json")
    for name, pin in meta["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    geom, evidence = geometry(heads, cfg["target"], cfg["head_groups"], cfg["rcond"])
    assert geom == read(out / "head_geometry.json")
    saved = torch.load(out / "artifacts/head_spectra.pt", map_location="cpu", weights_only=True)
    for name, record in evidence.items():
        assert all(torch.equal(v, saved[name][k]) for k, v in record.items())
    leakage = read(out / "head_leakage.json")
    for preset, delta in deltas.items():
        for task, head in heads.items():
            update = delta @ head.weight.double().T
            centered = update[:, 1:] - update[:, :1]
            expected = leakage[preset][task]["contrast_update_frobenius"]
            assert abs(float(centered.norm()) - expected) < 1e-8 * max(expected, 1)
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 28
    prior = torch.load(
        Path(cfg["parent"]) / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    from dataset.registry import get_dataset

    count = 0
    for task in cfg["evaluation_tasks"]:
        data = get_dataset(
            task, states["original_direct"].val_preprocess, "data", batch_size=128, num_workers=2
        )
        labels = truth(data, task)
        loader = make_loader(data.test_dataset, list(range(128)), 128)
        for row in [r for r in rows if r["dataset"] == task]:
            value = raw[row["key"]]
            assert torch.equal(value["labels"], labels)
            assert row["correct"] == int(value["prediction"].eq(labels).sum())
            assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
            assert row["top1"] == row["correct"] / row["n"]
            for reference, change in [
                ("same_preset_direct_reference", "same_preset_direct_change"),
                ("same_attack_original_reference", "same_attack_original_change"),
            ]:
                if reference in row:
                    assert row[change] == paired_counts(
                        raw[row[reference]]["prediction"].eq(labels), value["prediction"].eq(labels)
                    )
            sink = []
            evaluate(
                states[row["preset"]],
                heads[task],
                loader,
                "cuda",
                prediction_sink=lambda p, _y: sink.append(p),
            )
            assert torch.equal(torch.cat(sink), value["prediction"][:128])
            count += 1
        del data, loader
    for item in read(out / "parent_prediction_checks.json"):
        assert torch.equal(raw[item["key"]]["prediction"], prior[item["parent_key"]]["prediction"])
    assert count == 28
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            compilation_replayed=True,
            public_head_geometry_replayed=True,
            contrast_leakage_independently_checked=True,
            full_count_and_pair_checks=28,
            official_prediction_prefixes_replayed=28,
            previous_full_prediction_arrays_identical=12,
            no_new_defense_or_query_fit=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/query_partner_cost.json")
    parser.add_argument("--output", default="outputs/2026-10-11/query_partner_cost")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
