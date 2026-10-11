"""Measure public deletion costs of anchor-stabilizing MLP shear families."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from head_collapse import truth
from local_erasure import cache
from nonlinear_shear import public_cross_span, spaces
from operator_diagnostics import paired_counts
from path_subspace import context, read
from subspace_probe import save_json


@torch.no_grad()
def public_delete(reference, anchor, layers, rcond=1e-10):
    result = copy.deepcopy(reference).cpu().eval()
    checks = []
    for layer in layers:
        base = anchor.model.visual.transformer.resblocks[layer].mlp
        pw, _, a = spaces([base.c_fc.weight, base.c_fc.bias[:, None]], rcond)
        pv, _, b = spaces([base.c_proj.weight.T], rcond)
        mlp = reference.model.visual.transformer.resblocks[layer].mlp
        result.model.visual.transformer.resblocks[layer].mlp = public_cross_span(mlp, base, pw, pv)
        checks.append(dict(layer=layer, input_span=a, output_span=b))
    return result, checks


def edited_state(reference, model):
    return {
        k: v.clone()
        for k, v in model.state_dict().items()
        if not torch.equal(v, reference.state_dict()[k])
    }


@torch.no_grad()
def holders(cfg):
    root, load, fusion, models = context(cfg, all_tasks=True)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    parent = Path(cfg["shear_parent"])
    meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    artifact = parent / "artifacts/release_edits.pt"
    assert digest(artifact) == cfg["release_sha256"] == meta["raw_sha256"]["release_edits.pt"]
    release = copy.deepcopy(models["DTD"])
    for key, value in torch.load(artifact, map_location="cpu", weights_only=True).items():
        release.state_dict()[key].copy_(value)
    states, geometry = {}, {}
    for task, layers in cfg["cases"].items():
        a, check = public_delete(models[task], models["anchor"], layers)
        states["deleted_original", task] = a
        geometry[task] = check
    a, check = public_delete(release, models["anchor"], cfg["cases"]["DTD"])
    assert check == geometry["DTD"]
    expected = torch.load(
        parent / "artifacts/public_protected_edits.pt", map_location="cpu", weights_only=True
    )
    actual = edited_state(models["DTD"], a)
    assert actual.keys() == expected.keys() and all(
        torch.equal(v, expected[k]) for k, v in actual.items()
    )
    states["deleted_protected", "DTD"] = a
    state_gap = max(
        float((a.state_dict()[k] - states["deleted_original", "DTD"].state_dict()[k]).abs().max())
        for k in a.state_dict()
    )
    assert state_gap < 2e-6
    return root, load, fusion, models, states, geometry, state_gap


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models, states, geometry, gap = holders(cfg)
    expected = {
        x["path"].split("/", 1)[1]: x["sha256"] for x in read("configs/checkpoints.json")["files"]
    }
    pins = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))}
    assert pins == expected and len(pins) == 17
    save_json(out / "protocol.json", cfg)
    save_json(
        out / "geometry.json",
        dict(
            cases=geometry,
            protected_original_deleted_state_max_absolute_error=gap,
            bit_exact_identity_assumed=False,
        ),
    )
    for (name, task), model in states.items():
        torch.save(edited_state(models[task], model), artifacts / f"{name}_{task}.pt")
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["DTD"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    loaders, labels = (
        {t: cache(d) for t, d in data.items()},
        {t: truth(d, t) for t, d in data.items()},
    )
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    raw, rows = {}, []

    def measure(model, preset, target, scope, tasks):
        for task in tasks:
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            p, y = [], []
            score = evaluate(
                model,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            value = dict(prediction=torch.cat(p), labels=torch.cat(y))
            assert torch.equal(value["labels"], labels[task])
            assert score["n"] == fusion["expected_test_counts"][task]
            assert score["correct"] == int(value["prediction"].eq(value["labels"]).sum())
            key = f"{preset}/{target}/{scope}/{task}"
            raw[key] = value
            rows.append(
                dict(
                    key=key,
                    preset=preset,
                    target=target,
                    scope=scope,
                    dataset=task,
                    reused=False,
                    **score,
                )
            )
            print("COST", key, score["correct"], score["n"], flush=True)

    for task in cfg["cases"]:
        measure(models[task], "original", task, "standalone", [task])
    measure(
        merge_all(models, root, cfg["coefficient"]),
        "original",
        "all",
        "fusion",
        cfg["evaluation_tasks"],
    )
    for (name, task), model in states.items():
        measure(model, name, task, "standalone", [task])
        measure(
            merge_all(dict(models, **{task: model}), root, cfg["coefficient"]),
            name,
            task,
            "fusion",
            cfg["evaluation_tasks"],
        )
    assert len(rows) == 21
    for row in rows:
        if row["preset"] == "original":
            continue
        task, scope, dataset = row["target"], row["scope"], row["dataset"]
        source = f"original/{task if scope == 'standalone' else 'all'}/{scope}/{dataset}"
        value, before = raw[row["key"]], raw[source]
        row["matched_reference"] = source
        row["matched_change"] = paired_counts(
            before["prediction"].eq(before["labels"]), value["prediction"].eq(value["labels"])
        )
        if row["preset"] == "deleted_protected":
            original = raw[f"deleted_original/DTD/{scope}/{dataset}"]
            row["release_equivalence_changed_predictions"] = int(
                original["prediction"].ne(value["prediction"]).sum()
            )
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    helpers = [
        "nonlinear_shear.py",
        "anchor_null_reflection.py",
        "neuron_projection.py",
        "path_subspace.py",
        "local_erasure.py",
        "critical_subspace.py",
        "functional_contrast.py",
        "clip_benchmark.py",
        "eight_task_benchmark.py",
        "subspace_probe.py",
        "mechanisms.py",
        "operator_diagnostics.py",
        "nufilt_alignment.py",
        "fine_subspace.py",
        "joint_posthoc.py",
        "polynomial_probe.py",
        "score_recovery.py",
        "low_activity_posthoc.py",
        "head_collapse.py",
    ]
    parent = Path(cfg["shear_parent"])
    inputs = [
        args.config,
        cfg["fusion_config"],
        "configs/checkpoints.json",
        "configs/sources.json",
    ] + [
        str(parent / n)
        for n in [
            "metadata.json",
            "validation.json",
            "artifacts/release_edits.pt",
            "artifacts/public_protected_edits.pt",
        ]
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            actual_full_evaluations=21,
            reused_rows=0,
            defense_efficacy_claimed=False,
            eight_task_mean_reported=False,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path("experiments") / n) for n in helpers},
            input_sha256={n: digest(n) for n in inputs},
            checkpoint_sha256=pins,
            external_sha256={
                str(p): digest(p)
                for p in sorted(Path("external/secure-merge-attack").rglob("*.py"))
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_span_cost.json")
    parser.add_argument("--output", default="outputs/2026-10-11/cross_span_cost")
    run(parser.parse_args())
