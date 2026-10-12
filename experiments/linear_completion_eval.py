"""Matched TA/author-AGA/public-reset tests of a frozen linear completion.

Never reconstruct or alter the release. Reuse upstream merger, AGA, data,
evaluation and exact-state alias checks. Two evaluated heads, not Table 6.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import author_recover, digest, merge_all
from functional_contrast import features
from local_erasure import cache
from local_release_audit import state_equal
from mechanisms import load_author
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read
from query_partner_cost import runtime, truth
from subspace_probe import save_json


def inputs(cfg):
    parent = Path(cfg["parent"])
    meta = read(parent / "metadata.json")
    pcfg = read(parent / "protocol.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    for name, pin in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == pin
    assert cfg["task"] == pcfg["task"] and cfg["layer"] == pcfg["layer"]
    assert digest(parent / "artifacts/release_edits.pt") == cfg["release_sha256"]
    root, load, fusion, models = context(cfg, all_tasks=True)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    release = copy.deepcopy(models[cfg["task"]]).cpu().eval()
    edits = torch.load(parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    heads = {t: load(str(root / f"head_{t}.pt")).cpu().eval() for t in cfg["evaluation_tasks"]}
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    return root, fusion, models, release, heads, data


def reset(model, anchor, cfg):
    return replace_components(model, anchor, [cfg["layer"]], "mlp")[0]


def save_changes(model, original, path):
    torch.save(
        {
            k: v.clone()
            for k, v in model.state_dict().items()
            if not torch.equal(v, original.state_dict()[k])
        },
        path,
    )


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    start = time.perf_counter()
    root, fusion, models, release, heads, data = inputs(cfg)
    target, anchor = cfg["task"], models["anchor"]
    original = models[target]
    author, sha = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert sha == read("configs/sources.json")["AGA"]["sha256"]
    save_json(out / "protocol.json", cfg)
    raw, rows, attack_times, aliases, original_states = {}, [], {}, {}, {}
    parent_raw = torch.load(
        Path(cfg["parent"]) / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    parent_rows = {r["key"]: r for r in read(Path(cfg["parent"]) / "results.json")}
    loaders = {t: cache(d) for t, d in data.items()}
    for preset, teacher in [("original", original), ("protected", release)]:
        t0 = time.perf_counter()
        aga = author_recover(teacher, anchor, author)
        attack_times[preset] = time.perf_counter() - t0
        save_changes(aga, teacher, artifacts / f"{preset}_aga_edits.pt")
        states = dict(
            direct=teacher,
            aga=aga,
            reset=reset(teacher, anchor, cfg),
            aga_reset=reset(aga, anchor, cfg),
        )
        for attack, model in states.items():
            if preset == "original":
                original_states[attack] = model
            equivalent = preset == "protected" and attack in {"reset", "aga_reset"}
            if equivalent:
                assert state_equal(model, original_states[attack])
            merged = None if equivalent else merge_all(dict(models, **{target: model}), root, 0.2)
            settings = [("standalone", target)] + [("fusion", t) for t in cfg["evaluation_tasks"]]
            for group, task in settings:
                key = f"{preset}/{attack}/{group}/{task}"
                if equivalent:
                    previous = f"original/{attack}/{group}/{task}"
                    raw[key] = raw[previous]
                    row = dict(
                        next(r for r in rows if r["key"] == previous),
                        key=key,
                        preset=preset,
                        reused_exact_state=previous,
                    )
                    aliases[key] = previous
                elif attack == "direct" and group == "standalone":
                    source = f"{preset}/official_test"
                    raw[key] = parent_raw[source]
                    row = dict(
                        parent_rows[source],
                        key=key,
                        preset=preset,
                        attack=attack,
                        group=group,
                        dataset=task,
                        reused_parent_row=source,
                    )
                    row.pop("paired", None)
                else:
                    predictions, labels = [], []
                    tested = model if group == "standalone" else merged
                    result = evaluate(
                        tested,
                        heads[task],
                        loaders[task],
                        "cuda",
                        prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
                    )
                    raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(labels))
                    assert result["n"] == fusion["expected_test_counts"][task]
                    assert torch.equal(raw[key]["labels"], truth(data[task], task))
                    row = dict(
                        key=key, preset=preset, attack=attack, group=group, dataset=task, **result
                    )
                if preset == "protected":
                    baseline = raw[f"original/{attack}/{group}/{task}"]
                    y = baseline["labels"]
                    row["paired_matched_attack"] = paired_counts(
                        baseline["prediction"].eq(y), raw[key]["prediction"].eq(y)
                    )
                rows.append(row)
                print("COMPLETION_EVAL", key, row["correct"], row["n"], flush=True)
                save_json(out / "results.json", rows)
                torch.save(raw, artifacts / "predictions.pt")
            del merged
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - start,
            author_aga_sha256=sha,
            author_aga_seconds=attack_times,
            release_sha256=cfg["release_sha256"],
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            checkpoint_sha256={
                r["path"].split("/", 1)[1]: r["sha256"]
                for r in read("configs/checkpoints.json")["files"]
            },
            evaluated_heads=cfg["evaluation_tasks"],
            eight_task_mean_reported=False,
            exact_table6_reproduction=False,
            fresh_full_tests=16,
            native_parent_rows=2,
            exact_state_alias_rows=len(aliases),
            reset_aliases=aliases,
            attack_gradient_updates=0,
            attack_calibration_samples=0,
            attack_teacher_queries=0,
            reset_support_public_by_protocol=True,
            defense_success_claim=False,
        ),
    )


@torch.no_grad()
def validate(args):
    cfg, out = read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for path, pin in meta["source_manifest"].items():
        assert digest(path) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    root, _, models, release, heads, data = inputs(cfg)
    for path, pin in meta["checkpoint_sha256"].items():
        assert digest(root / path) == pin
    predictions = torch.load(
        out / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    rows = {r["key"]: r for r in read(out / "results.json")}
    originals, checks = {}, 0
    author, sha = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert sha == meta["author_aga_sha256"]
    for preset, teacher in [("original", models[cfg["task"]]), ("protected", release)]:
        aga = copy.deepcopy(teacher).cpu().eval()
        changed = torch.load(
            out / "artifacts" / f"{preset}_aga_edits.pt", map_location="cpu", weights_only=True
        )
        for key, value in changed.items():
            aga.state_dict()[key].copy_(value)
        replay = author_recover(teacher, models["anchor"], author)
        assert state_equal(aga, replay)
        del replay
        states = dict(
            direct=teacher,
            aga=aga,
            reset=reset(teacher, models["anchor"], cfg),
            aga_reset=reset(aga, models["anchor"], cfg),
        )
        for attack, model in states.items():
            if preset == "original":
                originals[attack] = model
            elif attack in {"reset", "aga_reset"}:
                assert state_equal(model, originals[attack])
            merged = merge_all(dict(models, **{cfg["task"]: model}), root, cfg["coefficient"])
            for group, task in [("standalone", cfg["task"])] + [
                ("fusion", t) for t in cfg["evaluation_tasks"]
            ]:
                key = f"{preset}/{attack}/{group}/{task}"
                raw, row = predictions[key], rows[key]
                labels = truth(data[task], task)
                assert torch.equal(raw["labels"], labels)
                assert row["correct"] == int(raw["prediction"].eq(labels).sum())
                assert row["n"] == len(labels) and row["top1"] == row["correct"] / row["n"]
                x = torch.stack([data[task].test_dataset[i][0] for i in range(32)])
                tested = model if group == "standalone" else merged
                f = features(tested.cuda().eval(), x)
                tested.cpu()
                observed = (f @ heads[task].weight.T + heads[task].bias).argmax(1)
                assert torch.equal(observed, raw["prediction"][:32])
                if preset == "protected":
                    before = predictions[f"original/{attack}/{group}/{task}"]
                    assert row["paired_matched_attack"] == paired_counts(
                        before["prediction"].eq(labels), raw["prediction"].eq(labels)
                    )
                if key in meta["reset_aliases"]:
                    baseline = predictions[meta["reset_aliases"][key]]
                    assert torch.equal(raw["prediction"], baseline["prediction"])
                checks += 1
            del merged
    assert checks == 24
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            prediction_prefix_checks=checks,
            counts_labels_and_pairs=True,
            reset_and_aga_then_reset_exact_state_aliases=True,
            source_checkpoint_release_and_raw_hashes=True,
            original_author_aga_fresh_exact_state_replays=2,
            defense_success_claim=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/linear_completion_eval.json")
    parser.add_argument("--output", default="outputs/2026-10-12/linear_completion_eval")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    (validate if options.validate else run)(options)
