"""Verify official counts and replay matched posthoc recovery states."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import author_recover, digest, merge_all
from head_collapse import identical, truth
from mechanisms import load_author
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read
from posthoc_benchmark import changed_tensors, duplicate_groups, public_unfold, reset_groups
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta, rows = [
        read(out / name) for name in ["protocol.json", "metadata.json", "results.json"]
    ]
    assert cfg == read(args.config) and meta["config_sha256"] == digest(args.config)
    assert meta["completed"] and meta["source_sha256"] == digest("experiments/posthoc_benchmark.py")
    assert meta["new_full_evaluations"] == sum(not row["reused"] for row in rows) == 45
    assert meta["reused_metric_rows"] == sum(row["reused"] for row in rows) == 9
    assert meta["report_rows"] == len(rows) == 54 and not meta["eight_task_mean_reported"]
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    parent = Path(cfg["parent"])
    assert (
        digest(parent / "artifacts/release_edits.pt")
        == meta["release_sha256"]
        == cfg["release_sha256"]
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg, all_tasks=True)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    task, anchor = cfg["task"], models["anchor"]
    original = models[task]
    release = copy.deepcopy(original).cpu().eval()
    edits = torch.load(parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    state = release.state_dict()
    for key, value in edits.items():
        state[key].copy_(value)
    assert all(
        torch.equal(value, original.state_dict()[key])
        for key, value in state.items()
        if key not in edits
    )
    unfolded, groups = public_unfold(release, anchor)
    ordinary, ordinary_groups = public_unfold(original, anchor)
    identical(original, ordinary)
    assert not ordinary_groups and groups == read(out / "observable_groups.json")["protected"]
    assert duplicate_groups(release) == groups
    saved_unfold = torch.load(
        out / "artifacts/public_unfold_edits.pt", map_location="cpu", weights_only=True
    )
    computed = changed_tensors(release, unfolded)
    assert computed.keys() == saved_unfold.keys() and all(
        torch.equal(v, saved_unfold[k]) for k, v in computed.items()
    )
    unfold_checks = []
    for layer, clusters in groups.items():
        src = release.model.visual.transformer.resblocks[int(layer)].mlp
        base = anchor.model.visual.transformer.resblocks[int(layer)].mlp
        current = unfolded.model.visual.transformer.resblocks[int(layer)].mlp
        for group in clusters:
            w, b, v = (
                base.c_fc.weight[group].double(),
                base.c_fc.bias[group].double(),
                base.c_proj.weight[:, group].double(),
            )
            assert torch.equal(
                current.c_fc.weight[group],
                (src.c_fc.weight[group].double() + w - w.mean(0)).float(),
            )
            assert torch.equal(
                current.c_fc.bias[group], (src.c_fc.bias[group].double() + b - b.mean()).float()
            )
            assert torch.equal(
                current.c_proj.weight[:, group],
                (src.c_proj.weight[:, group].double() + v - v.mean(1, keepdim=True)).float(),
            )
        used = {i for group in clusters for i in group}
        unchanged = [i for i in range(src.c_fc.out_features) if i not in used]
        assert torch.equal(current.c_fc.weight[unchanged], src.c_fc.weight[unchanged])
        assert torch.equal(current.c_fc.bias[unchanged], src.c_fc.bias[unchanged])
        assert torch.equal(current.c_proj.weight[:, unchanged], src.c_proj.weight[:, unchanged])
        assert torch.equal(current.c_proj.bias, src.c_proj.bias)
        unfold_checks.append(
            dict(layer=int(layer), groups=len(clusters), public_complement_formula_verified=True)
        )
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, original.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    labels = {t: truth(value, t) for t, value in data.items()}
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(raw) == len(rows) == len({row["key"] for row in rows})
    for row in rows:
        value = raw[row["key"]]
        assert torch.equal(value["labels"], labels[row["dataset"]])
        assert row["n"] == len(value["labels"]) == fusion["expected_test_counts"][row["dataset"]]
        assert row["correct"] == int(value["prediction"].eq(value["labels"]).sum())
        assert row["top1"] == row["correct"] / row["n"]
        if row["reused"]:
            source = raw[row["source_key"]]
            assert all(torch.equal(v, source[k]) for k, v in value.items())
        reference = f"original/{row['attack']}/{row['scope']}/{row['dataset']}"
        if row["preset"] == "protected" and reference in raw:
            assert row["matched_change"] == paired_counts(
                raw[reference]["prediction"].eq(value["labels"]),
                value["prediction"].eq(value["labels"]),
            )
    author, author_hash = load_author(
        "external/secure-merge-attack/utils_attack_B32.py", include_wrappers=True
    )
    assert (
        author_hash
        == meta["author_aga_sha256"]
        == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    )
    states = {
        ("original", "none", "standalone"): original,
        ("protected", "none", "standalone"): release,
        ("protected", "public_unfold", "standalone"): unfolded,
    }
    query_indices = read(out / "query_indices.json")
    assert (
        query_indices
        == read(Path(cfg["query_parent"]) / "training_indices.json")[task]["selection"][
            slice(*cfg["query_slice"])
        ]
    )
    training = read(parent / "training_indices.json")
    assert len(query_indices) == 256 and not set(query_indices).intersection(
        training["calibration"] + training["confirmation"]
    )
    query_images = torch.stack([data[task].train_dataset[i][0] for i in query_indices])
    inverse, geometry = head_inverse(heads[task], [heads[t] for t in cfg["preserved_query_heads"]])
    assert geometry == read(out / "query_fits.json")["geometry"]
    for preset, teacher in [("original", original), ("protected", release)]:
        student = merge_all(dict(models, **{task: teacher}), root, cfg["coefficient"])
        states[preset, "direct", "fusion"] = student
        restored = author_recover(teacher, anchor, author)
        saved = torch.load(
            out / f"artifacts/{preset}_aga_edits.pt", map_location="cpu", weights_only=True
        )
        actual = changed_tensors(teacher, restored)
        assert actual.keys() == saved.keys() and all(
            torch.equal(v, saved[k]) for k, v in actual.items()
        )
        states[preset, "aga", "fusion"] = merge_all(
            dict(models, **{task: restored}), root, cfg["coefficient"]
        )
        unused, fit, evidence = recover(
            teacher, student, heads[task], inverse, query_images, cfg["ridge_ratio"]
        )
        del unused
        saved = torch.load(
            out / f"artifacts/{preset}_query.pt", map_location="cpu", weights_only=True
        )
        assert fit == read(out / "query_fits.json")["fits"][preset]
        assert all(torch.equal(value, saved[key]) for key, value in evidence.items())
        queried = copy.deepcopy(student).cpu().eval()
        projection = (student.model.visual.proj.double() + evidence["projection_delta"]).float()
        assert torch.equal(projection, saved["compiled_student_projection"])
        queried.model.visual.proj.copy_(projection)
        states[preset, "readout", "fusion"] = queried
    states["protected", "public_unfold", "fusion"] = merge_all(
        dict(models, **{task: unfolded}), root, cfg["coefficient"]
    )
    restored = author_recover(unfolded, anchor, author)
    saved = torch.load(out / "artifacts/unfold_aga_edits.pt", map_location="cpu", weights_only=True)
    actual = changed_tensors(unfolded, restored)
    assert actual.keys() == saved.keys() and all(
        torch.equal(v, saved[k]) for k, v in actual.items()
    )
    states["protected", "unfold_aga", "fusion"] = merge_all(
        dict(models, **{task: restored}), root, cfg["coefficient"]
    )
    for attack in ["support_reset", "reset_mlp"]:
        if attack == "support_reset":
            a, b = [reset_groups(value, anchor, groups) for value in [original, release]]
        else:
            a, b = [
                replace_components(value, anchor, list(range(12)), "mlp")[0]
                for value in [original, release]
            ]
        identical(a, b)
        states["identical_reset", attack, "standalone"] = a
        states["identical_reset", attack, "fusion"] = merge_all(
            dict(models, **{task: a}), root, cfg["coefficient"]
        )
    prefixes = []
    for row in rows:
        if row["reused"]:
            continue
        p = []
        evaluate(
            states[row["preset"], row["attack"], row["scope"]],
            heads[row["dataset"]],
            make_loader(data[row["dataset"]].test_dataset, list(range(128))),
            "cuda",
            prediction_sink=lambda a, _b: p.append(a),
        )
        assert torch.equal(torch.cat(p), raw[row["key"]]["prediction"][:128])
        prefixes.append(row["key"])
    partner_checks = {}
    for preset in ["original", "protected"]:
        partner_checks[preset] = {
            t: dict(
                changed_predictions=int(
                    raw[f"{preset}/readout/fusion/{t}"]["prediction"]
                    .ne(raw[f"{preset}/direct/fusion/{t}"]["prediction"])
                    .sum()
                ),
                direct_correct=next(
                    row["correct"] for row in rows if row["key"] == f"{preset}/direct/fusion/{t}"
                ),
                query_correct=next(
                    row["correct"] for row in rows if row["key"] == f"{preset}/readout/fusion/{t}"
                ),
            )
            for t in cfg["evaluation_tasks"]
            if t != task
        }
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            official_full_metric_counts_verified=len(rows),
            actual_model_prefixes_replayed=len(prefixes),
            explicit_reused_rows_verified=meta["reused_metric_rows"],
            three_author_aga_states_replayed=True,
            two_query_fits_replayed=True,
            public_unfold_checks=unfold_checks,
            query_partner_checks=partner_checks,
            replacement_states_identical=True,
            source_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neuron_projection_benchmark.json")
    parser.add_argument("--output", default="outputs/2026-10-11/neuron_projection_benchmark")
    validate(parser.parse_args())
