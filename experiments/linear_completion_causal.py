"""Separate a frozen release's functional edit from parameter-mixing error.

These oracle interventions use deleted source weights. They are neither attacks,
new releases nor implementations of a functional merging paper. Only the final
CLS MLP output changes; all four branches have identical upstream computation.
"""

from __future__ import annotations

import argparse
import copy
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from functional_contrast import features
from linear_completion import even
from linear_completion_eval import inputs
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import read
from query_partner_cost import runtime, truth
from subspace_probe import save_json


def load_inputs(cfg):
    base = read(cfg["base_config"])
    assert base["evaluation_tasks"] == ["DTD", "MNIST"]
    assert base["task"] == "DTD" and base["layer"] == 11
    root, fusion, models, release, heads, data = inputs(base)
    parent = Path(cfg["evaluation_parent"])
    meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    for name, pin in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == pin
    raw = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    geometry = torch.load(
        Path(base["parent"]) / "artifacts/construction.pt", map_location="cpu", weights_only=True
    )["geometry"]
    source = models[base["task"]].model.visual.transformer.resblocks[base["layer"]].mlp
    released = release.model.visual.transformer.resblocks[base["layer"]].mlp
    frozen = {"original": models[base["task"]], "protected": release}
    states = {}
    for preset, model in frozen.items():
        attacked = copy.deepcopy(model).cpu().eval()
        edits = torch.load(
            parent / "artifacts" / f"{preset}_aga_edits.pt", map_location="cpu", weights_only=True
        )
        for key, value in edits.items():
            attacked.state_dict()[key].copy_(value)
        states[preset] = dict(direct=model, aga=attacked)
    return base, root, fusion, models, states, source, released, geometry, heads, data, raw


def merged_pair(models, states, attack, root, base):
    pair = {
        p: merge_all(dict(models, **{base["task"]: states[p][attack]}), root, base["coefficient"])
        for p in ("original", "protected")
    }
    prefix = f"model.visual.transformer.resblocks.{base['layer']}.mlp."
    a, b = pair["original"].state_dict(), pair["protected"].state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(v, b[k]) for k, v in a.items() if not k.startswith(prefix))
    return pair


def even_delta(source, geometry):
    drop = geometry["drop"]
    a = source.c_fc.weight[drop].double().cuda()
    b = source.c_fc.bias[drop].double().cuda()
    v = source.c_proj.weight[:, drop].double().cuda()
    mean = geometry["even_mean"][drop].double().cuda()

    def delta(x):
        return -F.linear(even(F.linear(x.double(), a, b)) - mean, v).to(x.dtype)

    return delta


def dense_delta(source, released):
    # Independent reference: evaluate the two complete frozen MLPs, rather
    # than reusing the even decomposition or a neuron-selection mask.
    before = copy.deepcopy(source).double().cuda().eval()
    after = copy.deepcopy(released).double().cuda().eval()

    def delta(x):
        return (after(x.double()) - before(x.double())).to(x.dtype)

    return delta


@contextmanager
def intervention(model, layer, kind, delta, coefficient):
    if kind in ("original", "protected"):
        yield
        return
    assert kind in ("functional_only", "mixing_residual")
    sign = 1 if kind == "functional_only" else -1

    def hook(_module, args, output):
        value = output.clone()
        # No later block reads patches: CLS is sufficient for this final MLP.
        value[0] += sign * coefficient * delta(args[0][0])
        return value

    module = model.model.visual.transformer.resblocks[layer].mlp
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def local_algebra(pair, images, base, delta, full_delta):
    records = {}
    for name, model in pair.items():
        xs, ys = [], []
        block = model.model.visual.transformer.resblocks[base["layer"]]
        handle = block.mlp.register_forward_hook(
            lambda _m, a, y: (xs.append(a[0][0].detach().cpu()), ys.append(y[0].detach().cpu()))
            and None
        )
        try:
            features(model.cuda().eval(), images)
        finally:
            handle.remove()
            model.cpu()
        records[name] = dict(x=torch.cat(xs), y=torch.cat(ys))
    x = records["original"]["x"]
    assert torch.equal(x, records["protected"]["x"])
    df = delta(x.cuda()).cpu().double()
    oracle = full_delta(x.cuda()).cpu().double()
    error = float((df - oracle).abs().max())
    assert error < 2e-5
    y0 = records["original"]["y"].double()
    yr = records["protected"]["y"].double()
    functional = base["coefficient"] * df
    mixed = yr - y0
    residual = mixed - functional
    numerator = (functional * residual).sum()
    return dict(
        source_formula_vs_dense_delta_max_absolute_error=error,
        upstream_cls_inputs_exactly_equal=True,
        reconstruction_max_absolute_error=float((functional + residual - mixed).abs().max()),
        functional_delta_rms=float(functional.square().mean().sqrt()),
        mixing_residual_rms=float(residual.square().mean().sqrt()),
        total_mlp_delta_rms=float(mixed.square().mean().sqrt()),
        functional_residual_cosine=float(numerator / (functional.norm() * residual.norm())),
        images=len(images),
    )


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    started = time.perf_counter()
    base, root, fusion, models, states, source, release, geometry, heads, data, parent_raw = (
        load_inputs(cfg)
    )
    save_json(out / "protocol.json", cfg)
    loaders = {t: cache(d) for t, d in data.items()}
    formula, dense = even_delta(source, geometry), dense_delta(source, release)
    rows, raw, algebra = [], {}, {}
    for attack in ("direct", "aga"):
        pair = merged_pair(models, states, attack, root, base)
        for task in base["evaluation_tasks"]:
            images = torch.stack(
                [data[task].test_dataset[i][0] for i in range(cfg["prefix_images"])]
            )
            algebra[f"{attack}/{task}"] = local_algebra(pair, images, base, formula, dense)
            for kind in ("original", "protected", "functional_only", "mixing_residual"):
                key = f"{attack}/{kind}/{task}"
                if kind in ("original", "protected"):
                    previous = f"{kind}/{attack}/fusion/{task}"
                    raw[key] = parent_raw[previous]
                    result = dict(
                        correct=int(raw[key]["prediction"].eq(raw[key]["labels"]).sum()),
                        n=len(raw[key]["labels"]),
                        reused_parent_row=previous,
                    )
                    result["top1"] = result["correct"] / result["n"]
                else:
                    model = pair["original" if kind == "functional_only" else "protected"]
                    predictions, labels = [], []
                    with intervention(model, base["layer"], kind, formula, base["coefficient"]):
                        result = evaluate(
                            model,
                            heads[task],
                            loaders[task],
                            "cuda",
                            prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
                        )
                    raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(labels))
                assert result["n"] == fusion["expected_test_counts"][task]
                assert torch.equal(raw[key]["labels"], truth(data[task], task))
                baseline = raw[f"{attack}/original/{task}"]
                row = dict(
                    key=key,
                    attack=attack,
                    kind=kind,
                    task=task,
                    **result,
                    paired=paired_counts(
                        baseline["prediction"].eq(baseline["labels"]),
                        raw[key]["prediction"].eq(raw[key]["labels"]),
                    ),
                )
                rows.append(row)
                print("COMPLETION_CAUSAL", key, result["correct"], result["n"], flush=True)
                save_json(out / "results.json", rows)
        del pair
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "algebra.json", algebra)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - started,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            release_sha256=base["release_sha256"],
            fresh_full_tests=8,
            reused_parent_rows=8,
            new_releases=0,
            gradient_steps=0,
            fitting_steps=0,
            oracle_deleted_source_parameters_used=True,
            functional_merging_algorithm_claim=False,
            defense_success_claim=False,
            exact_table6_reproduction=False,
            eight_task_mean_reported=False,
        ),
    )


@torch.no_grad()
def validate(args):
    cfg, out = read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for name, pin in meta["source_manifest"].items():
        assert digest(name) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    base, root, _, models, states, source, release, geometry, heads, data, parent_raw = load_inputs(
        cfg
    )
    for item in read("configs/checkpoints.json")["files"]:
        assert digest(root / item["path"].split("/", 1)[1]) == item["sha256"]
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = {r["key"]: r for r in read(out / "results.json")}
    formula, dense = even_delta(source, geometry), dense_delta(source, release)
    algebra = read(out / "algebra.json")
    checks = 0
    for attack in ("direct", "aga"):
        pair = merged_pair(models, states, attack, root, base)
        for task in base["evaluation_tasks"]:
            images = torch.stack(
                [data[task].test_dataset[i][0] for i in range(cfg["prefix_images"])]
            )
            assert local_algebra(pair, images, base, formula, dense) == algebra[f"{attack}/{task}"]
            for kind in ("original", "protected", "functional_only", "mixing_residual"):
                key = f"{attack}/{kind}/{task}"
                value, row = raw[key], rows[key]
                y = truth(data[task], task)
                assert torch.equal(value["labels"], y)
                assert row["correct"] == int(value["prediction"].eq(y).sum())
                assert row["n"] == len(y) and row["top1"] == row["correct"] / row["n"]
                baseline = raw[f"{attack}/original/{task}"]
                assert row["paired"] == paired_counts(
                    baseline["prediction"].eq(y), value["prediction"].eq(y)
                )
                if kind in ("original", "protected"):
                    previous = parent_raw[row["reused_parent_row"]]
                    assert torch.equal(previous["prediction"], value["prediction"])
                model = pair["original" if kind in ("original", "functional_only") else "protected"]
                # Fresh predictions use full source/release MLP evaluation.
                with intervention(model, base["layer"], kind, dense, base["coefficient"]):
                    f = features(model.cuda().eval(), images)
                model.cpu()
                predictions = (f @ heads[task].weight.T + heads[task].bias).argmax(1)
                assert torch.equal(predictions, value["prediction"][: len(images)])
                checks += 1
        del pair
    assert checks == 16
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            complete_counts_labels_pairs=True,
            fresh_dense_oracle_prediction_prefixes=checks,
            algebra_fresh_replays=4,
            hashes_source_checkpoints_release_parent_raw=True,
            new_releases=0,
            recovery_or_defense_claim=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/linear_completion_causal.json")
    parser.add_argument("--output", default="outputs/2026-10-12/linear_completion_causal")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    (validate if options.validate else run)(options)
