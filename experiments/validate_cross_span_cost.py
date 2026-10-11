"""Verify public deletion algebra, official counts and all evaluation prefixes."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from cross_span_cost import edited_state, holders
from eight_task_benchmark import digest, merge_all
from head_collapse import truth
from nonlinear_shear import spaces
from operator_diagnostics import paired_counts
from path_subspace import read
from subspace_probe import make_loader, save_json


@torch.no_grad()
def validate(args):
    started, out = time.perf_counter(), Path(args.output)
    cfg, meta, rows = [read(out / n) for n in ["protocol.json", "metadata.json", "results.json"]]
    assert cfg == read(args.config) and meta["config_sha256"] == digest(args.config)
    assert meta["completed"] and meta["source_sha256"] == digest("experiments/cross_span_cost.py")
    assert meta["actual_full_evaluations"] == len(rows) == 21 and meta["reused_rows"] == 0
    assert not meta["defense_efficacy_claimed"] and not meta["eight_task_mean_reported"]
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for kind in ["input_sha256", "external_sha256"]:
        for name, expected in meta[kind].items():
            assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models, states, geometry, gap = holders(cfg)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    assert len(meta["checkpoint_sha256"]) == 17
    observed = read(out / "geometry.json")
    assert geometry == observed["cases"]
    assert gap == observed["protected_original_deleted_state_max_absolute_error"]
    assert not observed["bit_exact_identity_assumed"]
    release = copy.deepcopy(models["DTD"])
    for k, value in torch.load(
        Path(cfg["shear_parent"]) / "artifacts/release_edits.pt",
        map_location="cpu",
        weights_only=True,
    ).items():
        release.state_dict()[k].copy_(value)
    direct_checks = []
    for (name, task), model in states.items():
        saved = torch.load(
            out / f"artifacts/{name}_{task}.pt", map_location="cpu", weights_only=True
        )
        current = edited_state(models[task], model)
        assert saved.keys() == current.keys() and all(
            torch.equal(v, saved[k]) for k, v in current.items()
        )
        source = release if name == "deleted_protected" else models[task]
        allowed = set()
        for layer in cfg["cases"][task]:
            base, src, found = [
                m.model.visual.transformer.resblocks[layer].mlp
                for m in [models["anchor"], source, model]
            ]
            pw, _, _ = spaces([base.c_fc.weight, base.c_fc.bias[:, None]], 1e-10)
            pv, _, _ = spaces([base.c_proj.weight.T], 1e-10)
            pw, pv = pw @ pw.T, pv @ pv.T
            a, b, c = [
                p.double().cuda() for p in [base.c_fc.weight, base.c_fc.bias, base.c_proj.weight]
            ]
            expected = [
                a + pv @ (src.c_fc.weight.double().cuda() - a),
                b + pv @ (src.c_fc.bias.double().cuda() - b),
                c + (src.c_proj.weight.double().cuda() - c) @ pw,
            ]
            errors = [
                float((value.double().cuda() - wanted).abs().max())
                for value, wanted in zip(
                    [found.c_fc.weight, found.c_fc.bias, found.c_proj.weight], expected, strict=True
                )
            ]
            assert max(errors) < 1e-6
            assert torch.equal(found.c_proj.bias, src.c_proj.bias)
            assert max(float((p @ p - p).abs().max()) for p in [pw, pv]) < 1e-10
            allowed.update(
                f"model.visual.transformer.resblocks.{layer}.mlp.{n}"
                for n in ["c_fc.weight", "c_fc.bias", "c_proj.weight"]
            )
            direct_checks.append(
                dict(
                    preset=name,
                    task=task,
                    layer=layer,
                    direct_projection_errors=errors,
                    projector_idempotence_verified=True,
                )
            )
        assert set(current) <= allowed
        assert all(
            torch.equal(v, source.state_dict()[k])
            for k, v in model.state_dict().items()
            if k not in allowed
        )
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["DTD"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    labels = {t: truth(d, t) for t, d in data.items()}
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    expected_keys = {f"original/{t}/standalone/{t}" for t in cfg["cases"]}
    expected_keys.update(f"original/all/fusion/{t}" for t in cfg["evaluation_tasks"])
    for name, task in states:
        expected_keys.add(f"{name}/{task}/standalone/{task}")
        expected_keys.update(f"{name}/{task}/fusion/{t}" for t in cfg["evaluation_tasks"])
    assert set(raw) == expected_keys == {r["key"] for r in rows} and len(expected_keys) == 21
    for row in rows:
        value, task = raw[row["key"]], row["dataset"]
        assert not row["reused"] and torch.equal(value["labels"], labels[task])
        assert row["n"] == len(labels[task]) == fusion["expected_test_counts"][task]
        assert row["correct"] == int(value["prediction"].eq(labels[task]).sum())
        assert row["top1"] == row["correct"] / row["n"]
        if row["preset"] == "original":
            continue
        target = row["target"] if row["scope"] == "standalone" else "all"
        key = f"original/{target}/{row['scope']}/{task}"
        assert row["matched_reference"] == key
        assert row["matched_change"] == paired_counts(
            raw[key]["prediction"].eq(labels[task]), value["prediction"].eq(labels[task])
        )
        if row["preset"] == "deleted_protected":
            original = raw[f"deleted_original/DTD/{row['scope']}/{task}"]
            assert row["release_equivalence_changed_predictions"] == int(
                original["prediction"].ne(value["prediction"]).sum()
            )
    replay = {("original", "all", "fusion"): merge_all(models, root, cfg["coefficient"])}
    for task in cfg["cases"]:
        replay["original", task, "standalone"] = models[task]
    for (name, task), model in states.items():
        replay[name, task, "standalone"] = model
        replay[name, task, "fusion"] = merge_all(
            dict(models, **{task: model}), root, cfg["coefficient"]
        )
    prefixes = []
    for row in rows:
        predictions = []
        evaluate(
            replay[row["preset"], row["target"], row["scope"]],
            heads[row["dataset"]],
            make_loader(data[row["dataset"]].test_dataset, list(range(128))),
            "cuda",
            prediction_sink=lambda p, _y: predictions.append(p),
        )
        assert torch.equal(torch.cat(predictions), raw[row["key"]]["prediction"][:128])
        prefixes.append(row["key"])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            source_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            all_twenty_one_official_counts_recomputed=True,
            all_matched_paired_counts_recomputed=True,
            all_three_edited_states_replayed=True,
            all_twenty_one_prefixes_replayed=prefixes,
            direct_checks=direct_checks,
            published_release_deletion_equivalence_recomputed=True,
            no_defense_efficacy_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_span_cost.json")
    parser.add_argument("--output", default="outputs/2026-10-11/cross_span_cost")
    validate(parser.parse_args())
