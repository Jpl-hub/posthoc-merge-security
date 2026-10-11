"""Replay public response cleanup, author AGA and full-test prediction evidence."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate, merge
from critical_subspace import parameters
from eight_task_benchmark import author_recover, digest, merge_all
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_subspace import context, read
from response_cleanup import baseline, cleanup, compare_function, get_indices, load_release
from subspace_probe import make_loader, save_json


def truth_for(task, dataset):
    if task == "Cars":
        labels = [label for _, label in dataset._samples]
    elif task == "SVHN":
        labels = dataset.labels
    else:
        labels = dataset.targets
    return torch.as_tensor(labels, dtype=torch.long)


@torch.no_grad()
def validate(args):
    out = Path(args.output)
    cfg, meta, rows = [read(out / p) for p in ("protocol.json", "metadata.json", "results.json")]
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert meta["completed"] and len(rows) == meta["rows"] == meta["actual_full_evaluations"] == 44
    assert meta["source_sha256"] == digest("experiments/response_cleanup.py")
    assert read(args.config) == cfg and meta["config_sha256"] == digest(args.config)
    for name, value in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == value
    for name, value in meta["input_sha256"].items():
        assert digest(name) == value
    for name, value in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == value
    root, load, fusion, models = context(cfg, all_tasks=True)
    for name, value in meta["checkpoint_sha256"].items():
        assert digest(root / name) == value
    from dataset.registry import get_dataset

    tasks = sorted({t for c in cfg["cases"] for t in c["evaluation_tasks"]})
    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in tasks
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in tasks}
    artifact = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    raw, prior = artifact["new"], artifact["reused_baselines"]
    assert len(raw) == len(prior) == len(rows) == len({r["id"] for r in rows})
    holders = {}
    cases = {c["name"]: c for c in cfg["cases"]}
    labels = {t: truth_for(t, data[t].test_dataset) for t in tasks}
    for row in rows:
        key, task = row["id"], row["dataset"]
        value, parent = raw[key], prior[key]
        assert torch.equal(value["labels"], labels[task])
        assert row["n"] == len(labels[task]) == fusion["expected_test_counts"][task]
        correct = int(value["prediction"].eq(labels[task]).sum())
        assert row["correct"] == correct and row["top1"] == correct / row["n"]
        reference, saved = baseline(
            cases[row["case"]], row["preset"], row["attack"], row["group"], task, holders
        )
        assert reference == row["baseline"]
        assert all(torch.equal(parent[k], saved[k]) for k in parent)
        assert row["change_from_parent"] == paired_counts(
            parent["prediction"].eq(labels[task]), value["prediction"].eq(labels[task])
        )
        control = raw[key.replace("/protected/", "/original/")]
        assert row["matched"] == paired_counts(
            control["prediction"].eq(labels[task]), value["prediction"].eq(labels[task])
        )
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == meta["original_author_aga_sha256"]
    geometry, functions, indices = [
        read(out / p) for p in ("geometry.json", "functions.json", "training_indices.json")
    ]
    checks, prefixes = [], 0
    for case in cfg["cases"]:
        target, name = case["task"], case["name"]
        assert indices[name] == get_indices(cfg, case)
        images = {
            k: torch.stack([data[target].train_dataset[i][0] for i in indices[name][k]])
            for k in ("calibration", "confirmation")
        }
        original = models[target]
        published = load_release(case, original)
        for preset, model in (("original", original), ("protected", published)):
            saved = torch.load(
                out / f"artifacts/{name}_{preset}_cleanup.pt", map_location="cpu", weights_only=True
            )
            clean, replayed_rows, replayed_evidence = cleanup(
                model, models["anchor"], images["calibration"], case, cfg
            )
            assert replayed_rows == geometry[f"{name}/{preset}"]
            assert replayed_evidence["context_checks"] == saved["evidence"]["context_checks"]
            for site, tensors in replayed_evidence["sites"].items():
                assert all(
                    torch.equal(v, saved["evidence"]["sites"][site][k]) for k, v in tensors.items()
                )
            changed = {
                k: v
                for k, v in clean.state_dict().items()
                if not torch.equal(v, model.state_dict()[k])
            }
            assert changed.keys() == saved["edits"].keys()
            assert all(torch.equal(v, saved["edits"][k]) for k, v in changed.items())
            # Compile the saved release independently of the replayed constructor.
            compiled = copy.deepcopy(model).cpu().eval()
            for key, value in saved["edits"].items():
                compiled.state_dict()[key].copy_(value)
            assert all(
                torch.equal(v, compiled.state_dict()[k]) for k, v in clean.state_dict().items()
            )
            # Bias compensation is tested from saved means without the solver.
            for site, evidence in saved["evidence"]["sites"].items():
                _, before_w, before_b = parameters(model.state_dict(), site)
                _, after_w, after_b = parameters(compiled.state_dict(), site)
                residual = (
                    (after_w.double() - before_w.double()) @ evidence["mean"]
                    + after_b.double()
                    - before_b.double()
                )
                expected_mean_error = next(
                    r["compiled_mean_max_absolute_error"]
                    for r in replayed_rows
                    if r["site"] == site
                )
                assert abs(float(residual.abs().max()) - expected_mean_error) < 1e-7
            for split in images:
                assert (
                    compare_function(model, clean, heads[target], images[split])
                    == functions[f"{name}/{preset}/{split}"]
                )
            for attack in cfg["attacks"]:
                restored = (
                    compiled
                    if attack == "cleanup"
                    else author_recover(compiled, models["anchor"], author)
                )
                merged = (
                    merge(models["anchor"], restored, models["SVHN"], case["coefficient"])
                    if case["fusion"] == "two_task"
                    else merge_all(dict(models, **{target: restored}), root, case["coefficient"])
                )
                requests = [
                    ("standalone", target, restored),
                    *[("fusion", t, merged) for t in case["evaluation_tasks"]],
                ]
                for group, task, evaluated in requests:
                    key = f"{name}/{preset}/{attack}/{group}/{task}"
                    p, y = [], []
                    evaluate(
                        evaluated,
                        heads[task],
                        make_loader(data[task].test_dataset, list(range(128))),
                        "cuda",
                        prediction_sink=lambda a, b: (p.append(a), y.append(b)),
                    )
                    assert torch.equal(torch.cat(p), raw[key]["prediction"][:128]), key
                    assert torch.equal(torch.cat(y), raw[key]["labels"][:128]), key
                    prefixes += 1
                del restored, merged
            checks.append(
                dict(
                    case=name,
                    preset=preset,
                    full_construct_replay_exact=True,
                    compiled_state_exact=True,
                    mean_compensation_checked=True,
                    calibration_and_confirmation_replayed=True,
                    sites=len(replayed_rows),
                )
            )
            del clean, compiled
        del published
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            rows=len(rows),
            raw_artifacts_checked=True,
            official_labels_checked=True,
            all_44_parent_comparisons_exact=True,
            all_six_cleanup_models_replayed_exact=True,
            all_64_local_solvers_replayed=True,
            prefix_replays=prefixes,
            checks=checks,
            elapsed_seconds=time.perf_counter() - started,
            validator_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
            frozen_run_source_sha256=meta["source_sha256"],
            new_defense_validated=False,
        ),
    )
    print("VALIDATION_COMPLETE", prefixes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/response_cleanup.json")
    parser.add_argument("--output", default="outputs/2026-10-11/response_cleanup")
    validate(parser.parse_args())
