"""Fixed input-subspace recovery where two public digit heads coincide.

Extra adaptive attack on immutable releases. No defense or novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import digest, merge_all
from local_erasure import cache
from operator_diagnostics import paired_counts, query_readout
from polynomial_probe import ridge_update
from score_recovery import head_inverse
from subspace_probe import save_json


@torch.no_grad()
def constrained_update(target_design, partner_design, residual, inverse, ratio, tolerance):
    _, singular, vectors = torch.linalg.svd(partner_design, full_matrices=False)
    rank = int((singular > singular[0] * tolerance).sum())
    rows = vectors[:rank]
    null = torch.eye(target_design.shape[1], dtype=torch.float64, device="cuda") - rows.T @ rows
    design = target_design @ null
    update, fit = ridge_update(design, residual, ratio)
    update = null @ update
    delta = update @ inverse
    fit.update(
        input_rank=rank,
        null_dimension=null.shape[0] - rank,
        target_projected_energy_fraction=float(
            design.square().sum() / target_design.square().sum()
        ),
        partner_design_null_relative_error=float(
            (partner_design @ null).norm() / partner_design.norm()
        ),
        partner_score_update_relative_error=float(
            (partner_design @ update).norm() / residual.norm()
        ),
        projected_normal_equation_relative_error=float(
            (null @ (design.T @ (design @ update - residual) + fit["ridge"] * update)).norm()
            / (design.T @ residual).norm()
        ),
    )
    return delta, update, null, fit


@torch.no_grad()
def fit_projection(release, student, head, inverse, target_images, partner_images, cfg):
    c = head.weight.double().cuda()
    release.cuda().eval()
    scores = torch.cat([release(batch.cuda()).double() @ c.T for batch in target_images.split(32)])
    release.cpu()
    target_predictions, hm = query_readout(student, head, target_images, capture_features=True)
    partner_predictions, hs = query_readout(student, head, partner_images, capture_features=True)
    p = student.model.visual.proj.double().cuda()
    residual = scores - hm @ p @ c.T
    delta, update, null, fit = constrained_update(
        hm, hs, residual, inverse, cfg["ridge_ratio"], cfg["rank_relative_tolerance"]
    )
    fit.update(
        target_samples=len(hm),
        public_partner_samples=len(hs),
        target_calibration_prediction_agreement=int(
            (hm @ (p + delta) @ c.T).argmax(-1).eq(scores.argmax(-1)).sum()
        ),
        partner_calibration_prediction_changes=int(
            (hs @ (p + delta) @ c.T).argmax(-1).cpu().ne(partner_predictions).sum()
        ),
    )
    restored = copy.deepcopy(release).cpu().eval()
    restored.model.visual.proj.copy_(
        (release.model.visual.proj.double().cuda() + delta / cfg["coefficient"]).float().cpu()
    )
    evidence = {
        "teacher_scores": scores.cpu(),
        "target_design": hm.cpu(),
        "partner_design": hs.cpu(),
        "target_student_predictions": target_predictions,
        "partner_student_predictions": partner_predictions,
        "initial_projection": p.cpu(),
        "residual": residual.cpu(),
        "update": update.cpu(),
        "null": null.cpu(),
        "inverse": inverse.cpu(),
        "projection_delta": delta.cpu(),
        "restored_projection": restored.model.visual.proj.detach().clone(),
    }
    return restored, fit, evidence


def load_context(cfg):
    root = Path("checkpoints/ViT-B-32")
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    fusion = json.loads(Path("configs/eight_tasks.json").read_bytes())
    models = {t: load(str(root / t / "finetuned.pt")) for t in fusion["tasks"]}
    models["anchor"] = load(str(root / "zeroshot.pt"))
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    assert cfg["coefficient"] == fusion["coefficient"]
    assert torch.equal(heads["MNIST"].weight, heads["SVHN"].weight)
    assert torch.equal(heads["MNIST"].bias, heads["SVHN"].bias)
    assert all(torch.count_nonzero(h.bias) == 0 and h.normalize for h in heads.values())
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path("external/secure-merge-attack/datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    assert all(len(data[t].test_dataset) == fusion["expected_test_counts"][t] for t in data)
    return root, models, heads, data


def released_model(original, preset, parent):
    release = copy.deepcopy(original).cpu().eval()
    if preset == "erased":
        edits = torch.load(
            parent / "artifacts/MNIST_edits.pt", map_location="cpu", weights_only=True
        )
        state = release.state_dict()
        for key, value in edits.items():
            state[key].copy_(value)
    return release


@torch.no_grad()
def run(args):
    cfg = json.loads(Path(args.config).read_bytes())
    out, parent, prior = Path(args.output), Path(cfg["release_parent"]), Path(cfg["score_parent"])
    start = time.perf_counter()
    for directory in (parent, prior):
        assert json.loads((directory / "validation.json").read_bytes())["status"] == "passed"
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    root, models, heads, data = load_context(cfg)
    pinned = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    assert all(digest(Path("checkpoints") / entry["path"]) == entry["sha256"] for entry in pinned)
    for directory in (parent, prior):
        meta = json.loads((directory / "metadata.json").read_bytes())
        assert all(digest(directory / "artifacts" / n) == v for n, v in meta["raw_sha256"].items())
    target_indices = json.loads((parent / "attack_indices.json").read_bytes())["MNIST"]
    partner_indices = torch.randperm(
        len(data["SVHN"].train_dataset), generator=torch.Generator().manual_seed(cfg["input_seed"])
    ).tolist()[: cfg["input_samples"]]
    assert len(target_indices) == len(partner_indices) == cfg["input_samples"]
    save_json(out / "attack_indices.json", dict(MNIST=target_indices, SVHN=partner_indices))
    target_images = torch.stack([data["MNIST"].train_dataset[i][0] for i in target_indices])
    partner_images = torch.stack([data["SVHN"].train_dataset[i][0] for i in partner_indices])
    loaders = {t: cache(d) for t, d in data.items()}
    controls = torch.load(
        parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    old = torch.load(prior / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    inverse, geometry = head_inverse(heads["MNIST"], [heads[t] for t in cfg["preserve_head_tasks"]])
    geometry.update(shared_digit_head=True, universal_svhn_null_implies_zero_target_update=True)
    save_json(out / "geometry.json", geometry)
    raw, rows, fits = {}, [], {}
    for preset in ("original", "erased"):
        release = released_model(models["MNIST"], preset, parent)
        student = merge_all(dict(models, MNIST=release), root, cfg["coefficient"])
        restored, fit, evidence = fit_projection(
            release, student, heads["MNIST"], inverse, target_images, partner_images, cfg
        )
        fits[preset] = fit
        torch.save(evidence, artifacts / f"{preset}_fit.pt")
        save_json(out / "fitting.json", fits)
        old_release = copy.deepcopy(release)
        old_fit = torch.load(
            prior / f"artifacts/MNIST_{preset}.pt", map_location="cpu", weights_only=True
        )
        old_release.model.visual.proj.copy_(old_fit["restored_projection"])
        for method, candidate in (
            ("direct", release),
            ("head_only", old_release),
            ("input_null", restored),
        ):
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            fused = merge_all(dict(models, MNIST=candidate), root, cfg["coefficient"])
            for task in cfg["evaluation_tasks"]:
                key = f"{preset}/{method}/{task}"
                reused = task != "SVHN" and method != "input_null"
                if reused:
                    old_key = (
                        f"MNIST/{preset}/direct/fusion/{task}"
                        if method == "direct"
                        else f"MNIST/{preset}/{task}"
                    )
                    raw[key] = (controls if method == "direct" else old)[old_key]
                    score = dict(
                        correct=int(raw[key]["prediction"].eq(raw[key]["labels"]).sum()),
                        n=len(raw[key]["labels"]),
                    )
                    score["top1"] = score["correct"] / score["n"]
                else:
                    pred, labels = [], []
                    score = evaluate(
                        fused,
                        heads[task],
                        loaders[task],
                        "cuda",
                        prediction_sink=lambda p, y: (pred.append(p), labels.append(y)),
                    )
                    raw[key] = dict(prediction=torch.cat(pred), labels=torch.cat(labels))
                baseline = raw[f"{preset}/direct/{task}"]
                assert torch.equal(raw[key]["labels"], baseline["labels"])
                rows.append(
                    dict(
                        id=key,
                        preset=preset,
                        method=method,
                        dataset=task,
                        reused_from_parent=bool(reused),
                        **score,
                        changed_from_direct=int(
                            raw[key]["prediction"].ne(baseline["prediction"]).sum()
                        ),
                        paired_to_direct=paired_counts(
                            baseline["prediction"].eq(baseline["labels"]),
                            raw[key]["prediction"].eq(raw[key]["labels"]),
                        ),
                    )
                )
                save_json(out / "results.json", rows)
                torch.save(raw, artifacts / "predictions.pt")
                print("RESULT", key, score["correct"], score["n"], flush=True)
            del fused
        del release, student, restored, old_release
    helpers = [
        "clip_benchmark.py",
        "eight_task_benchmark.py",
        "local_erasure.py",
        "operator_diagnostics.py",
        "polynomial_probe.py",
        "score_recovery.py",
        "subspace_probe.py",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path(__file__).with_name(n)) for n in helpers},
            fusion_config_sha256=digest("configs/eight_tasks.json"),
            checkpoint_sha256={e["path"]: e["sha256"] for e in pinned},
            parent_sha256={
                str(d / n): digest(d / n)
                for d in (parent, prior)
                for n in ("metadata.json", "validation.json", "attack_indices.json")
                if (d / n).exists()
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            elapsed_seconds=time.perf_counter() - start,
            rows=len(rows),
            fits=len(fits),
            actual_full_evaluations=sum(not r["reused_from_parent"] for r in rows),
            defender_changed=False,
            calibration_truth_used=False,
            optimizer_steps=0,
            input_preservation_is_calibration_only=True,
            all_seven_partners_tested=False,
            full_sloa=False,
        ),
    )


@torch.no_grad()
def validate(args):
    """Check official counts, independent fit equations and parameter replay."""
    from subspace_probe import make_loader

    out = Path(args.output)

    def read(name):
        return json.loads((out / name).read_bytes())

    cfg, meta = read("protocol.json"), read("metadata.json")
    assert meta["completed"] and meta["rows"] == 24 and meta["fits"] == 2
    assert meta["actual_full_evaluations"] == 12 and not meta["defender_changed"]
    assert not meta["calibration_truth_used"] and meta["optimizer_steps"] == 0
    assert meta["input_preservation_is_calibration_only"]
    assert not meta["all_seven_partners_tested"] and not meta["full_sloa"]
    assert read("execution.json")["exit_code"] == 0 and not read("execution.json")["timed_out"]
    assert digest(__file__) == meta["source_sha256"]
    assert digest(args.config) == meta["config_sha256"]
    assert digest("configs/eight_tasks.json") == meta["fusion_config_sha256"]
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(Path("checkpoints") / name) == expected
    for name, expected in meta["parent_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    root, models, heads, data = load_context(cfg)
    parent, prior = Path(cfg["release_parent"]), Path(cfg["score_parent"])
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    controls = torch.load(
        parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    old = torch.load(prior / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    labels = {}
    for task, dataset in data.items():
        test = dataset.test_dataset
        labels[task] = (
            torch.tensor([y for _, y in test._samples])
            if hasattr(test, "_samples")
            else torch.as_tensor(test.labels if hasattr(test, "labels") else test.targets)
        )
    rows = read("results.json")
    expected_ids = {
        f"{p}/{m}/{t}"
        for p in ("original", "erased")
        for m in ("direct", "head_only", "input_null")
        for t in cfg["evaluation_tasks"]
    }
    assert set(raw) == {r["id"] for r in rows} == expected_ids
    for row in rows:
        preset, method, task = row["preset"], row["method"], row["dataset"]
        current, baseline = raw[row["id"]], raw[f"{preset}/direct/{task}"]
        y = labels[task]
        assert torch.equal(current["labels"], y)
        assert current["prediction"].dtype == torch.int64 and current["prediction"].shape == y.shape
        assert current["prediction"].min() >= 0 and current["prediction"].max() < len(
            heads[task].weight
        )
        assert row["correct"] == int(current["prediction"].eq(y).sum())
        assert row["n"] == len(y) and row["top1"] == row["correct"] / len(y)
        assert row["changed_from_direct"] == int(
            current["prediction"].ne(baseline["prediction"]).sum()
        )
        assert row["paired_to_direct"] == paired_counts(
            baseline["prediction"].eq(y), current["prediction"].eq(y)
        )
        if row["reused_from_parent"]:
            old_key = (
                f"MNIST/{preset}/direct/fusion/{task}"
                if method == "direct"
                else f"MNIST/{preset}/{task}"
            )
            stored = (controls if method == "direct" else old)[old_key]
            assert all(torch.equal(current[k], stored[k]) for k in current)
    indices = read("attack_indices.json")
    assert indices["MNIST"] == json.loads((parent / "attack_indices.json").read_bytes())["MNIST"]
    assert (
        indices["SVHN"]
        == torch.randperm(
            len(data["SVHN"].train_dataset),
            generator=torch.Generator().manual_seed(cfg["input_seed"]),
        ).tolist()[: cfg["input_samples"]]
    )
    images = {
        t: torch.stack([data[t].train_dataset[i][0] for i in selected])
        for t, selected in indices.items()
    }
    inverse, _ = head_inverse(heads["MNIST"], [heads[t] for t in cfg["preserve_head_tasks"]])
    c = heads["MNIST"].weight.double()
    b = torch.cat([heads[t].weight.double() for t in cfg["preserve_head_tasks"]])
    checks, prefixes = [], 0
    for preset in ("original", "erased"):
        release = released_model(models["MNIST"], preset, parent)
        student = merge_all(dict(models, MNIST=release), root, cfg["coefficient"])
        restored, fit, evidence = fit_projection(
            release, student, heads["MNIST"], inverse, images["MNIST"], images["SVHN"], cfg
        )
        saved = torch.load(
            out / f"artifacts/{preset}_fit.pt", map_location="cpu", weights_only=True
        )
        assert fit == read("fitting.json")[preset]
        assert all(torch.equal(value, saved[key]) for key, value in evidence.items())
        h, s, a, u, r, delta = [
            saved[k]
            for k in (
                "target_design",
                "partner_design",
                "null",
                "update",
                "residual",
                "projection_delta",
            )
        ]
        normal = a @ (h.T @ (h @ u - r) + fit["ridge"] * u)
        normal_error = float(normal.norm() / (a @ h.T @ r).norm())
        assert normal_error < 1e-8
        assert float((a @ a - a).norm()) < 1e-8 and torch.equal(a, a.T)
        assert float((s @ u).norm() / r.norm()) < 1e-8
        assert float((delta @ c.T - u).norm() / u.norm()) < 1e-8
        assert float((delta @ b.T).norm() / u.norm()) < 1e-8
        compiled = merge_all(dict(models, MNIST=restored), root, cfg["coefficient"])
        before, after = student.state_dict(), compiled.state_dict()
        key = "model.visual.proj"
        assert all(torch.equal(v, after[k]) for k, v in before.items() if k != key)
        actual = after[key].double() - before[key].double()
        assert float((actual - delta).norm() / delta.norm()) < 1e-6
        old_release = copy.deepcopy(release)
        old_fit = torch.load(
            prior / f"artifacts/MNIST_{preset}.pt", map_location="cpu", weights_only=True
        )
        old_release.model.visual.proj.copy_(old_fit["restored_projection"])
        for method, candidate in (
            ("direct", release),
            ("head_only", old_release),
            ("input_null", restored),
        ):
            fused = merge_all(dict(models, MNIST=candidate), root, cfg["coefficient"])
            for task in cfg["evaluation_tasks"]:
                sink = []
                evaluate(
                    fused,
                    heads[task],
                    make_loader(data[task].test_dataset, list(range(128))),
                    "cuda",
                    prediction_sink=lambda p, _y: sink.append(p),
                )
                assert torch.equal(
                    torch.cat(sink), raw[f"{preset}/{method}/{task}"]["prediction"][:128]
                )
                prefixes += 1
            del fused
        for task in cfg["preserve_head_tasks"]:
            assert torch.equal(
                raw[f"{preset}/input_null/{task}"]["prediction"],
                raw[f"{preset}/direct/{task}"]["prediction"],
            )
        checks.append(
            dict(
                preset=preset,
                projected_normal_error=normal_error,
                fp32_partner_calibration_score_error=float((s @ actual @ c.T).norm() / r.norm()),
                fp32_compilation_error=float((actual - delta).norm() / delta.norm()),
            )
        )
        del release, student, restored, compiled, old_release
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            rows=len(rows),
            prefix_replays=prefixes,
            original_and_erased_fits_replayed=True,
            official_labels_checked=True,
            two_universal_partner_prediction_arrays_checked=4,
            algebra=checks,
            validator_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/input_score_recovery.json")
    parser.add_argument("--output", default="outputs/2026-10-11/input_score_recovery")
    parser.add_argument("--validate", action="store_true")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = parser.parse_args()
    (validate if args.validate else run)(args)
