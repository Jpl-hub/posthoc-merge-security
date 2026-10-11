"""Matched real tests of a frozen posthoc release and public recovery.

Reuse official TA/AGA and existing score recovery. Duplicate-neuron unfolding
uses only the accessible release and public anchor. No defender reconstruction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import author_recover, digest, merge_all
from head_collapse import identical, truth
from local_erasure import cache
from mechanisms import load_author
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read
from score_recovery import head_inverse, recover
from subspace_probe import save_json


@torch.no_grad()
def duplicate_groups(reference):
    """Infer groups from published tensors; private construction records unused."""
    groups = {}
    for layer, block in enumerate(reference.model.visual.transformer.resblocks):
        up, down = block.mlp.c_fc, block.mlp.c_proj
        buckets = {}
        for neuron in range(up.out_features):
            values = [up.weight[neuron], up.bias[neuron : neuron + 1], down.weight[:, neuron]]
            h = hashlib.sha256()
            for value in values:
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            buckets.setdefault(h.digest(), []).append(neuron)
        found = [group for group in buckets.values() if len(group) > 1]
        for group in found:
            for neuron in group[1:]:
                assert torch.equal(up.weight[neuron], up.weight[group[0]])
                assert torch.equal(up.bias[neuron], up.bias[group[0]])
                assert torch.equal(down.weight[:, neuron], down.weight[:, group[0]])
        if found:
            groups[str(layer)] = sorted(found)
    return groups


@torch.no_grad()
def public_unfold(reference, anchor):
    groups = duplicate_groups(reference)
    result = copy.deepcopy(reference).cpu().eval()
    for layer, clusters in groups.items():
        mlp = result.model.visual.transformer.resblocks[int(layer)].mlp
        base = anchor.model.visual.transformer.resblocks[int(layer)].mlp
        for group in clusters:
            w0, b0, v0 = (
                base.c_fc.weight[group].double(),
                base.c_fc.bias[group].double(),
                base.c_proj.weight[:, group].double(),
            )
            mlp.c_fc.weight[group] = (mlp.c_fc.weight[group].double() + w0 - w0.mean(0)).float()
            mlp.c_fc.bias[group] = (mlp.c_fc.bias[group].double() + b0 - b0.mean()).float()
            mlp.c_proj.weight[:, group] = (
                mlp.c_proj.weight[:, group].double() + v0 - v0.mean(1, keepdim=True)
            ).float()
    return result, groups


@torch.no_grad()
def reset_groups(reference, anchor, groups):
    result = copy.deepcopy(reference).cpu().eval()
    for layer, clusters in groups.items():
        mlp = result.model.visual.transformer.resblocks[int(layer)].mlp
        base = anchor.model.visual.transformer.resblocks[int(layer)].mlp
        for group in clusters:
            mlp.c_fc.weight[group] = base.c_fc.weight[group]
            mlp.c_fc.bias[group] = base.c_fc.bias[group]
            mlp.c_proj.weight[:, group] = base.c_proj.weight[:, group]
    return result


def changed_tensors(reference, current):
    return {
        key: value.clone()
        for key, value in current.state_dict().items()
        if not torch.equal(value, reference.state_dict()[key])
    }


def source_manifest(cfg, args, parent):
    helpers = [
        "clip_benchmark.py",
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "head_collapse.py",
        "local_erasure.py",
        "mechanisms.py",
        "nufilt_alignment.py",
        "operator_diagnostics.py",
        "path_subspace.py",
        "score_recovery.py",
        "polynomial_probe.py",
        "subspace_probe.py",
        "functional_contrast.py",
        "joint_posthoc.py",
        "low_activity_posthoc.py",
        "fine_subspace.py",
    ]
    inputs = [
        cfg["fusion_config"],
        "configs/checkpoints.json",
        "configs/sources.json",
        str(parent / "metadata.json"),
        str(parent / "validation.json"),
        str(parent / "release_manifest.json"),
        str(parent / "training_indices.json"),
        str(parent / "selected.json"),
        str(Path(cfg["query_parent"]) / "training_indices.json"),
    ]
    return dict(
        source_sha256=digest(__file__),
        config_sha256=digest(args.config),
        helper_sha256={name: digest(Path("experiments") / name) for name in helpers},
        input_sha256={name: digest(name) for name in inputs},
    )


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    parent = Path(cfg["parent"])
    assert read(parent / "validation.json")["status"] == "passed"
    artifact = parent / "artifacts/release_edits.pt"
    assert (
        digest(artifact)
        == cfg["release_sha256"]
        == read(parent / "release_manifest.json")["sha256"]
    )
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg, all_tasks=True)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    pins = {
        item["path"].split("/", 1)[1]: item["sha256"]
        for item in read("configs/checkpoints.json")["files"]
    }
    checkpoint_hashes = {
        str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*.pt"))
    }
    assert checkpoint_hashes == pins
    task, anchor = cfg["task"], models["anchor"]
    original = models[task]
    release = copy.deepcopy(original).cpu().eval()
    edits = torch.load(artifact, map_location="cpu", weights_only=True)
    state = release.state_dict()
    for key, value in edits.items():
        state[key].copy_(value)
    assert changed_tensors(original, release).keys() == edits.keys()
    protected_unfold, groups = public_unfold(release, anchor)
    original_unfold, original_groups = public_unfold(original, anchor)
    assert not original_groups
    identical(original, original_unfold)
    expected = read(parent / "selected.json")
    assert groups == {key: sorted(value) for key, value in expected.items() if value}
    save_json(
        out / "observable_groups.json",
        dict(
            protected=groups,
            original=original_groups,
            group_selection_uses_private_construction=False,
        ),
    )
    torch.save(changed_tensors(release, protected_unfold), artifacts / "public_unfold_edits.pt")
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    assert torch.equal(heads["MNIST"].weight, heads["SVHN"].weight)
    assert all(torch.count_nonzero(head.bias) == 0 and head.normalize for head in heads.values())
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, original.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    loaders = {t: cache(value) for t, value in data.items()}
    labels = {t: truth(value, t) for t, value in data.items()}
    queries = read(Path(cfg["query_parent"]) / "training_indices.json")[task]["selection"][
        slice(*cfg["query_slice"])
    ]
    indices = read(parent / "training_indices.json")
    assert len(queries) == 256 and not set(queries).intersection(
        indices["calibration"] + indices["confirmation"]
    )
    save_json(out / "query_indices.json", queries)
    query_images = torch.stack([data[task].train_dataset[i][0] for i in queries])
    author, author_hash = load_author(
        "external/secure-merge-attack/utils_attack_B32.py", include_wrappers=True
    )
    assert author_hash == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    inverse, geometry = head_inverse(heads[task], [heads[t] for t in cfg["preserved_query_heads"]])
    rows, raw, fits, equality = [], {}, {}, {}

    def measure(model, preset, attack, scope, datasets):
        for t in datasets:
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            p, y = [], []
            score = evaluate(
                model,
                heads[t],
                loaders[t],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            key = f"{preset}/{attack}/{scope}/{t}"
            value = dict(prediction=torch.cat(p), labels=torch.cat(y))
            assert torch.equal(value["labels"], labels[t])
            assert score["n"] == fusion["expected_test_counts"][t]
            assert score["correct"] == int(value["prediction"].eq(value["labels"]).sum())
            raw[key] = value
            rows.append(
                dict(
                    key=key,
                    preset=preset,
                    attack=attack,
                    scope=scope,
                    dataset=t,
                    reused=False,
                    **score,
                )
            )
            print("MEASURE", key, score["correct"], score["n"], flush=True)

    def alias(source, preset, attack, scope, datasets):
        for t in datasets:
            reference = f"original/{source}/{scope}/{t}"
            key = f"{preset}/{attack}/{scope}/{t}"
            row = next(row for row in rows if row["key"] == reference)
            raw[key] = raw[reference]
            rows.append(
                dict(
                    key=key,
                    preset=preset,
                    attack=attack,
                    scope=scope,
                    dataset=t,
                    reused=True,
                    source_key=reference,
                    correct=row["correct"],
                    n=row["n"],
                    top1=row["top1"],
                )
            )

    for preset, teacher in [("original", original), ("protected", release)]:
        measure(teacher, preset, "none", "standalone", [task])
        student = merge_all(dict(models, **{task: teacher}), root, cfg["coefficient"])
        measure(student, preset, "direct", "fusion", cfg["evaluation_tasks"])
        restored = author_recover(teacher, anchor, author)
        torch.save(changed_tensors(teacher, restored), artifacts / f"{preset}_aga_edits.pt")
        combined = merge_all(dict(models, **{task: restored}), root, cfg["coefficient"])
        measure(combined, preset, "aga", "fusion", cfg["evaluation_tasks"])
        del restored, combined
        unused, fits[preset], evidence = recover(
            teacher, student, heads[task], inverse, query_images, cfg["ridge_ratio"]
        )
        del unused
        query_model = copy.deepcopy(student).cpu().eval()
        query_model.model.visual.proj.copy_(
            (student.model.visual.proj.double() + evidence["projection_delta"]).float()
        )
        evidence["compiled_student_projection"] = query_model.model.visual.proj.detach().clone()
        torch.save(evidence, artifacts / f"{preset}_query.pt")
        measure(query_model, preset, "readout", "fusion", cfg["evaluation_tasks"])
        del student, query_model
    alias("none", "original", "public_unfold", "standalone", [task])
    alias("direct", "original", "public_unfold", "fusion", cfg["evaluation_tasks"])
    measure(protected_unfold, "protected", "public_unfold", "standalone", [task])
    combined = merge_all(dict(models, **{task: protected_unfold}), root, cfg["coefficient"])
    measure(combined, "protected", "public_unfold", "fusion", cfg["evaluation_tasks"])
    del combined
    restored = author_recover(protected_unfold, anchor, author)
    torch.save(changed_tensors(protected_unfold, restored), artifacts / "unfold_aga_edits.pt")
    combined = merge_all(dict(models, **{task: restored}), root, cfg["coefficient"])
    alias("aga", "original", "unfold_aga", "fusion", cfg["evaluation_tasks"])
    measure(combined, "protected", "unfold_aga", "fusion", cfg["evaluation_tasks"])
    del restored, combined
    for attack in ["support_reset", "reset_mlp"]:
        if attack == "support_reset":
            a, b = [reset_groups(value, anchor, groups) for value in [original, release]]
        else:
            a, b = [
                replace_components(value, anchor, list(range(12)), "mlp")[0]
                for value in [original, release]
            ]
        identical(a, b)
        equality[attack] = dict(
            original_and_protected_states_identical=True,
            measured_once=True,
            preset_mapping={"original": "identical_reset", "protected": "identical_reset"},
        )
        measure(a, "identical_reset", attack, "standalone", [task])
        combined = merge_all(dict(models, **{task: a}), root, cfg["coefficient"])
        measure(combined, "identical_reset", attack, "fusion", cfg["evaluation_tasks"])
        del a, b, combined
    for row in rows:
        reference = f"original/{row['attack']}/{row['scope']}/{row['dataset']}"
        if row["preset"] == "protected" and reference in raw:
            row["matched_change"] = paired_counts(
                raw[reference]["prediction"].eq(raw[reference]["labels"]),
                raw[row["key"]]["prediction"].eq(raw[row["key"]]["labels"]),
            )
    save_json(out / "results.json", rows)
    save_json(out / "query_fits.json", dict(geometry=geometry, fits=fits))
    save_json(out / "replacement_checks.json", equality)
    torch.save(raw, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            new_full_evaluations=sum(not row["reused"] for row in rows),
            reused_metric_rows=sum(row["reused"] for row in rows),
            report_rows=len(rows),
            release_sha256=cfg["release_sha256"],
            author_aga_sha256=author_hash,
            checkpoint_sha256=checkpoint_hashes,
            eight_task_mean_reported=False,
            raw_sha256={
                path.name: digest(path) for path in sorted(artifacts.iterdir()) if path.is_file()
            },
            **source_manifest(cfg, args, parent),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neuron_projection_benchmark.json")
    parser.add_argument("--output", default="outputs/2026-10-11/neuron_projection_benchmark")
    run(parser.parse_args())
