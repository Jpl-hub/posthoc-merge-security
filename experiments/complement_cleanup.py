"""Matched public reset and calibration PCA cleanup of one frozen release.

No defense refit, gradient updates, release-generator oracle, teacher response
queries or test-based attack selection. PCA uses other public training images.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import evaluate
from critical_subspace import parameters
from eight_task_benchmark import digest, merge_all
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import context, read
from query_partner_cost import runtime, truth
from signed_complement import aliases
from subspace_probe import collect_moments, make_loader, save_json


@torch.no_grad()
def cleanup(model, anchor, sites, moments=None, energy=0.99):
    result = copy.deepcopy(model)
    records = []
    for site, alias in zip(sites, aliases(sites), strict=True):
        _, w, b = parameters(result.state_dict(), site)
        _, w0, b0 = parameters(anchor.state_dict(), site)
        if moments is None:
            w.copy_(w0)
            b.copy_(b0)
            continue
        m = moments[alias]
        values, vectors = torch.linalg.eigh(m["covariance"])
        values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
        rank = int(torch.searchsorted(values.cumsum(0) / values.sum(), energy)) + 1
        p = vectors[:, :rank] @ vectors[:, :rank].T
        removed = (w.double() - w0.double()) @ (torch.eye(len(p), dtype=torch.float64) - p)
        w.copy_((w.double() - removed).float())
        b.copy_((b.double() + removed @ m["mean"]).float())
        records.append(
            dict(
                site=site,
                rank=rank,
                retained_energy=float(values[:rank].sum() / values.sum()),
                removed_update_norm=float(removed.norm()),
            )
        )
    changed = {k for site in sites for k in parameters(model.state_dict(), site)[0]}
    assert all(
        torch.equal(v, model.state_dict()[k])
        for k, v in result.state_dict().items()
        if k not in changed
    )
    for site in sites:
        if site.startswith("v:"):
            keys, weight, _ = parameters(result.state_dict(), site)
            for key in keys:
                assert torch.equal(
                    result.state_dict()[key][: 2 * weight.shape[1]],
                    model.state_dict()[key][: 2 * weight.shape[1]],
                )
    return result, records


@torch.no_grad()
def run(args):
    cfg, parent, out = read(args.config), Path(args.parent), Path(args.output)
    assert read(parent / "validation.json")["status"] == "passed"
    assert read(parent / "metadata.json")["completed"]
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    started = time.perf_counter()
    save_json(out / "protocol.json", cfg)
    release_cfg = read(parent / "protocol.json")
    root, load, fusion, models = context(release_cfg, all_tasks=True)
    target = release_cfg["task"]
    expert, anchor = models[target], models["anchor"]
    release = copy.deepcopy(expert)
    release_path = parent / "artifacts/release_edits.pt"
    assert digest(release_path) == read(parent / "release_freeze.json")["sha256"]
    for key, value in torch.load(release_path, map_location="cpu", weights_only=True).items():
        release.state_dict()[key].copy_(value)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in release_cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in data}
    indices = read(Path(cfg["profile_parent"]) / "training_indices.json")[target]["selection"][
        : cfg["attacker_samples"]
    ]
    defender = read(parent / "training_indices.json")[target]
    assert not set(indices).intersection(defender["calibration"] + defender["confirmation"])
    save_json(
        out / "attack_indices.json",
        dict(
            indices=indices,
            ground_truth_used=False,
            disjoint_from_defender=True,
            external_queries=0,
        ),
    )
    tests = {t: cache(d) for t, d in data.items()}
    rows, predictions, controls, evidence, geometry = [], {}, {}, {}, {}
    shared = None
    for preset, model in [("original", expert), ("protected", release)]:
        moments = collect_moments(
            model, make_loader(data[target].train_dataset, indices), aliases(release_cfg["sites"])
        )
        evidence[preset] = moments
        for attack in ("pca", "public_reset"):
            processed, rec = cleanup(
                model,
                anchor,
                release_cfg["sites"],
                moments if attack == "pca" else None,
                cfg["pca_energy"],
            )
            geometry[f"{preset}/{attack}"] = rec
            if preset == "protected" and attack == "public_reset":
                assert shared is not None
                assert all(
                    torch.equal(v, shared.state_dict()[k])
                    for k, v in processed.state_dict().items()
                )
            merged = merge_all(
                dict(models, **{target: processed}), root, release_cfg["coefficient"]
            )
            for group, tested, tasks in [
                ("native", processed, [target]),
                ("fusion", merged, list(data)),
            ]:
                for task in tasks:
                    key = f"{preset}/{attack}/{group}/{task}"
                    original_key = key.replace("protected/", "original/", 1)
                    alias = preset == "protected" and attack == "public_reset"
                    if alias:
                        predictions[key] = predictions[original_key]
                        value = dict(
                            controls[original_key], evaluation_seconds=None, peak_gpu_bytes=None
                        )
                    else:
                        sink = []
                        value = evaluate(
                            tested,
                            heads[task],
                            tests[task],
                            "cuda",
                            prediction_sink=lambda p, _y: sink.append(p),
                        )
                        predictions[key] = torch.cat(sink)
                        controls[key] = value
                    row = dict(
                        key=key,
                        preset=preset,
                        attack=attack,
                        group=group,
                        task=task,
                        alias=alias,
                        **value,
                    )
                    if preset == "protected":
                        label = truth(data[task], task)
                        row["paired"] = paired_counts(
                            predictions[original_key].eq(label), predictions[key].eq(label)
                        )
                    rows.append(row)
                    print("RECOVERY", row, flush=True)
            if preset == "original" and attack == "public_reset":
                shared = processed
            del merged, processed
    torch.save(
        dict(
            moments=evidence,
            predictions=predictions,
            labels={t: truth(d, t) for t, d in data.items()},
        ),
        artifacts / "recovery.pt",
    )
    save_json(out / "results.json", rows)
    save_json(out / "geometry.json", geometry)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - started,
            parent_release_sha256=digest(release_path),
            source_sha256=digest(__file__),
            source_config_sha256=digest(args.config),
            raw_sha256=digest(artifacts / "recovery.pt"),
            fresh_full_evaluations=sum(not r["alias"] for r in rows),
            aliases=sum(r["alias"] for r in rows),
        ),
    )


@torch.no_grad()
def validate(args):
    runtime()
    parent, out = Path(args.parent), Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and digest(out / "artifacts/recovery.pt") == meta["raw_sha256"]
    raw = torch.load(out / "artifacts/recovery.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    for row in rows:
        y = raw["labels"][row["task"]].numpy()
        pred = raw["predictions"][row["key"]].numpy()
        assert len(y) == row["n"] and int((pred == y).sum()) == row["correct"]
        if row["preset"] == "protected":
            original = raw["predictions"][row["key"].replace("protected/", "original/", 1)].numpy()
            assert int(((original == y) & (pred != y)).sum()) == row["paired"]["lost"]
            assert int(((original != y) & (pred == y)).sum()) == row["paired"]["gained"]
    release_cfg = read(parent / "protocol.json")
    root, load, fusion, models = context(release_cfg, all_tasks=True)
    for name, pin in read(parent / "metadata.json")["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    target = release_cfg["task"]
    expert, anchor = models[target], models["anchor"]
    release = copy.deepcopy(expert)
    release_path = parent / "artifacts/release_edits.pt"
    assert digest(release_path) == meta["parent_release_sha256"]
    for key, value in torch.load(release_path, map_location="cpu", weights_only=True).items():
        release.state_dict()[key].copy_(value)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in release_cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in data}
    max_formula_error, prefixes = 0.0, 0
    for preset, model in [("original", expert), ("protected", release)]:
        for attack in ("pca", "public_reset"):
            moments = raw["moments"][preset] if attack == "pca" else None
            processed, _ = cleanup(model, anchor, release_cfg["sites"], moments, cfg["pca_energy"])
            if moments is not None:
                for site, alias in zip(
                    release_cfg["sites"], aliases(release_cfg["sites"]), strict=True
                ):
                    moment = moments[alias]
                    eig, u = np.linalg.eigh(moment["covariance"].numpy())
                    eig, u = np.maximum(eig[::-1], 0), u[:, ::-1]
                    rank = int(np.searchsorted(np.cumsum(eig) / eig.sum(), cfg["pca_energy"])) + 1
                    p = u[:, :rank] @ u[:, :rank].T
                    _, w, b = parameters(model.state_dict(), site)
                    _, w0, _ = parameters(anchor.state_dict(), site)
                    _, actual_w, actual_b = parameters(processed.state_dict(), site)
                    delta = w.double().numpy() - w0.double().numpy()
                    expected_w = (w0.double().numpy() + delta @ p).astype(np.float32)
                    expected_b = (
                        b.double().numpy() + delta @ (np.eye(len(p)) - p) @ moment["mean"].numpy()
                    ).astype(np.float32)
                    error = max(
                        float(np.max(np.abs(expected_w - actual_w.numpy()))),
                        float(np.max(np.abs(expected_b - actual_b.numpy()))),
                    )
                    assert error < 2e-6, (preset, site, error)
                    max_formula_error = max(max_formula_error, error)
            merged = merge_all(
                dict(models, **{target: processed}), root, release_cfg["coefficient"]
            )
            for group, tested, tasks in [
                ("native", processed, [target]),
                ("fusion", merged, list(data)),
            ]:
                for task in tasks:
                    pred = []
                    evaluate(
                        tested,
                        heads[task],
                        make_loader(data[task].test_dataset, list(range(32))),
                        "cuda",
                        prediction_sink=lambda p, _y: pred.append(p),
                    )
                    assert torch.equal(
                        torch.cat(pred),
                        raw["predictions"][f"{preset}/{attack}/{group}/{task}"][:32],
                    )
                    prefixes += 1
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            independent_numpy_formula_max_error=max_formula_error,
            count_rows=len(rows),
            fresh_prediction_prefixes=prefixes,
            results_sha256=digest(out / "results.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/complement_cleanup.json")
    parser.add_argument("--parent", default="outputs/2026-10-12/signed_complement")
    parser.add_argument("--output", default="outputs/2026-10-12/signed_complement_cleanup")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate(args) if args.validate else run(args)
