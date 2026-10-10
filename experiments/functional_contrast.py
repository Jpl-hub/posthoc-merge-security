"""A bounded forward-only probe of cross-module functional cancellation.

This reuses layerwise task-vector rescaling, scipy's generalized eigenproblem,
the original Task Arithmetic merger, data registry, and original author AGA.
It is a mechanism test, not a novelty claim or proof of anti-merging security.
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
import torch.nn.functional as F
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import author_recover, digest, merge_all
from mechanisms import load_author
from operator_diagnostics import paired_counts, replace_components
from scipy.linalg import eigh
from subspace_probe import save_json


def groups(config, state):
    result = []
    for layer in config["layers"]:
        for component in config["components"]:
            prefix = f"model.visual.transformer.resblocks.{layer}.{component}."
            keys = [key for key in state if key.startswith(prefix)]
            assert len(keys) == 4
            result.append(dict(layer=layer, component=component, keys=keys))
    return result


@torch.no_grad()
def features(model, images):
    return torch.cat([F.normalize(model(batch.cuda()), dim=-1).cpu() for batch in images.split(32)])


@torch.no_grad()
def derivatives(reference, teacher, anchor, images, support, step):
    """Central finite differences of final normalized features; no weight fitting."""
    model = copy.deepcopy(reference).cuda().eval()
    state = model.state_dict()
    target, base = teacher.state_dict(), anchor.state_dict()
    columns = []
    for item in support:
        before = {k: state[k].clone() for k in item["keys"]}
        delta = {k: (target[k] - base[k]).cuda() for k in item["keys"]}
        values = []
        for sign in (1, -1):
            for key in item["keys"]:
                state[key].copy_(before[key] + sign * step * delta[key])
            values.append(features(model, images))
        columns.append((values[0] - values[1]) / (2 * step))
        for key in item["keys"]:
            state[key].copy_(before[key])
    model.cpu()
    return torch.stack(columns, dim=-1)


@torch.no_grad()
def construct(teacher, anchor, support, coefficients):
    model = copy.deepcopy(teacher).cpu().eval()
    state, original, base = model.state_dict(), teacher.state_dict(), anchor.state_dict()
    edited = set()
    for item, coefficient in zip(support, coefficients, strict=True):
        for key in item["keys"]:
            state[key].copy_(original[key] + coefficient * (original[key] - base[key]))
            edited.add(key)
    assert all(torch.equal(value, original[k]) for k, value in state.items() if k not in edited)
    assert all(torch.isfinite(value).all() for value in state.values())
    return model


def calibration_indices(config, length):
    excluded = set()
    for name in config["exclusions"]:
        value = json.loads(Path(name).read_bytes())
        if config["task"] in value:
            value = value[config["task"]]
        for indices in value.values():
            if isinstance(indices, list):
                excluded.update(indices)
    order = torch.randperm(length, generator=torch.Generator().manual_seed(config["seed"]))
    selected = [int(i) for i in order if int(i) not in excluded][: config["calibration_samples"]]
    assert len(selected) == config["calibration_samples"] and not excluded.intersection(selected)
    return selected, len(excluded)


@torch.no_grad()
def run(args):
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    save_json(out / "protocol.json", config)
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    assert config["coefficient"] == fusion["coefficient"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    task = config["task"]
    original = load(str(root / task / "finetuned.pt")).cpu().eval()
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    support = groups(config, original.state_dict())
    save_json(out / "support.json", support)
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    data = get_dataset(task, original.val_preprocess, args.data, batch_size=128, num_workers=2)
    indices, excluded = calibration_indices(config, len(data.train_dataset))
    save_json(out / "training_indices.json", dict(calibration=indices, excluded_count=excluded))
    # Labels are never read by the constructor, eigenproblem or amplitude rule.
    images = torch.stack([data.train_dataset[i][0] for i in indices])
    dt = derivatives(original, original, anchor, images, support, config["finite_difference_step"])
    d0 = derivatives(anchor, original, anchor, images, support, config["finite_difference_step"])
    flat_t, flat_0 = dt.double().flatten(0, 1), d0.double().flatten(0, 1)
    gt, g0 = flat_t.T @ flat_t / len(images), flat_0.T @ flat_0 / len(images)
    rho = config["regularizer_trace_fraction"] * float(gt.trace()) / len(support) + 1e-12
    eigenvalues, eigenvectors = eigh(g0.numpy(), (gt + rho * torch.eye(len(support))).numpy())
    direction = torch.from_numpy(eigenvectors[:, -1].copy())
    direction /= direction.abs().max()
    direction *= 1 if direction[direction.abs().argmax()] >= 0 else -1
    target_rms = float((direction @ gt @ direction).clamp_min(1e-24).sqrt())
    coefficients, records = {}, []
    for index, budget in enumerate(config["feature_rms_budgets"]):
        amplitude = min(config["maximum_absolute_coefficient"], budget / target_rms)
        name = f"budget_{index}"
        coefficients[name] = amplitude * direction
        records.append(dict(preset=name, budget=budget, amplitude=amplitude))
    torch.save(
        dict(
            images=images,
            target_derivatives=dt,
            anchor_derivatives=d0,
            target_gram=gt,
            anchor_gram=g0,
            direction=direction,
            coefficients=coefficients,
        ),
        artifacts / "geometry.pt",
    )
    save_json(
        out / "construction.json",
        dict(
            rho=rho,
            maximum_generalized_eigenvalue=float(eigenvalues[-1]),
            target_direction_feature_rms=target_rms,
            points=records,
            labels_used=False,
            future_partners_used=False,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    releases = {
        name: construct(original, anchor, support, value) for name, value in coefficients.items()
    }
    for name, model in releases.items():
        torch.save(model.state_dict(), artifacts / f"{name}.pt")
        pred_t, pred_0 = [], []
        for reference, sink in [(original, pred_t), (anchor, pred_0)]:
            reference.cuda().eval()
            before = features(reference, images)
            changed = construct(reference.cpu(), anchor, support, torch.zeros(len(support)))
            # Apply the target's actual edits at both reference points.
            state, target, base = changed.state_dict(), original.state_dict(), anchor.state_dict()
            for item, value in zip(support, coefficients[name], strict=True):
                for key in item["keys"]:
                    state[key].add_(value * (target[key] - base[key]))
            changed.cuda().eval()
            after = features(changed, images)
            changed.cpu()
            sink.append(float(((after - before).square().sum() / len(images)).sqrt()))
        records[int(name[-1])].update(
            actual_target_calibration_feature_rms=pred_t[0],
            actual_anchor_calibration_feature_rms=pred_0[0],
        )
    save_json(
        out / "construction.json",
        dict(
            rho=rho,
            maximum_generalized_eigenvalue=float(eigenvalues[-1]),
            target_direction_feature_rms=target_rms,
            points=records,
            labels_used=False,
            future_partners_used=False,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    # All releases and their hashes are frozen before reading other experts or tests.
    save_json(
        out / "release_manifest.json", {name: digest(artifacts / f"{name}.pt") for name in releases}
    )
    print("CONSTRUCTION_FROZEN", json.dumps(records), flush=True)
    models = {t: load(str(root / t / "finetuned.pt")) for t in fusion["tasks"]}
    models[task], models["anchor"] = original, anchor
    heads = {t: load(str(root / f"head_{t}.pt")) for t in config["evaluation_tasks"]}
    loaders = {}
    for name in config["evaluation_tasks"]:
        dataset = get_dataset(
            name, original.val_preprocess, args.data, batch_size=128, num_workers=2
        )
        assert len(dataset.test_dataset) == fusion["expected_test_counts"][name]
        # Identical deterministic author preprocessing, materialized once in
        # server RAM. No test images are persisted to disk or copied locally.
        from torch.utils.data import DataLoader, TensorDataset

        image_batches, label_batches = [], []
        for batch, labels in dataset.test_loader:
            image_batches.append(batch)
            label_batches.append(labels)
        loaders[name] = DataLoader(
            TensorDataset(torch.cat(image_batches), torch.cat(label_batches)),
            batch_size=128,
            shuffle=False,
            pin_memory=True,
        )
        del image_batches, label_batches
        print("OFFICIAL_INPUTS_CACHED", name, len(dataset.test_dataset), flush=True)
    author, source_hash = load_author(
        Path(args.author_repo) / "utils_attack_B32.py", include_wrappers=True
    )
    assert source_hash == json.loads(Path("configs/sources.json").read_bytes())["AGA"]["sha256"]
    rows, raw = [], {}

    def measure(preset, attack, group, model, dataset):
        p, y = [], []
        score = evaluate(
            model,
            heads[dataset],
            loaders[dataset],
            "cuda",
            prediction_sink=lambda a, b: (p.append(a), y.append(b)),
        )
        key = f"{preset}/{attack}/{group}/{dataset}"
        raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
        rows.append(
            dict(id=key, preset=preset, attack=attack, group=group, dataset=dataset, **score)
        )
        # Preserve every real prediction even if a later stage fails or times out.
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        print("RESULT", key, score["correct"], score["n"], flush=True)

    for preset, model in {"original": original, **releases}.items():
        for attack in config["attacks"]:
            assert time.perf_counter() - start < config["time_limit_seconds"], (
                "Frozen compute budget exhausted"
            )
            if attack == "direct":
                attacked = model
            elif attack == "aga":
                attacked = author_recover(model, anchor, author)
                torch.save(attacked.state_dict(), artifacts / f"{preset}_aga.pt")
            else:
                component = {
                    "reset_attention": "attn",
                    "reset_mlp": "mlp",
                    "reset_all": "attn_mlp",
                }[attack]
                attacked, _ = replace_components(model, anchor, config["layers"], component)
            if attack == "reset_all":
                reference, _ = replace_components(original, anchor, config["layers"], "attn_mlp")
                assert all(
                    torch.equal(value, reference.state_dict()[k])
                    for k, value in attacked.state_dict().items()
                )
            measure(preset, attack, "standalone", attacked, task)
            combined = merge_all(dict(models, **{task: attacked}), root, config["coefficient"])
            for dataset in config["evaluation_tasks"]:
                measure(preset, attack, "fusion", combined, dataset)
            del combined
    for row in rows:
        reference = raw[f"original/{row['attack']}/{row['group']}/{row['dataset']}"]
        value = raw[row["id"]]
        assert torch.equal(value["labels"], reference["labels"])
        row["paired"] = paired_counts(
            reference["prediction"].eq(reference["labels"]), value["prediction"].eq(value["labels"])
        )
    save_json(out / "results.json", rows)
    torch.save(raw, artifacts / "predictions.pt")
    summary = []
    original_direct = next(r for r in rows if r["id"] == f"original/direct/fusion/{task}")
    for name in releases:
        candidates = [
            r
            for r in rows
            if r["preset"] == name and r["group"] == "fusion" and r["dataset"] == task
        ]
        best = max(candidates, key=lambda r: r["top1"])
        utility = next(r for r in rows if r["id"] == f"{name}/direct/standalone/{task}")
        utility_loss = utility["paired"]["loss_pp"]
        summary.append(
            dict(
                preset=name,
                utility_loss_pp=utility_loss,
                best_recovery_attack=best["attack"],
                best_recovered_top1=best["top1"],
                gap_from_unprotected_direct_pp=100 * (original_direct["top1"] - best["top1"]),
                utility_reference_met=utility_loss <= config["utility_reference_pp"],
            )
        )
    save_json(
        out / "decision.json",
        dict(
            points=summary,
            novel_defense_claim=False,
            interpretation="Observed tradeoff only; security includes all tested bypasses.",
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            helper_sha256={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "mechanisms.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            checkpoint_sha256={
                p: digest(root / p)
                for p in [
                    "zeroshot.pt",
                    *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
                    *[f"head_{t}.pt" for t in config["evaluation_tasks"]],
                ]
            },
            elapsed_seconds=time.perf_counter() - start,
            rows=len(rows),
            optimizer_steps=0,
            calibration_labels_used=False,
            original_author_aga_sha256=source_hash,
            future_partners_in_constructor=False,
            official_test_selection=False,
            full_eight_task_average=False,
            official_inputs_cached="Unchanged author preprocessing; float32 server RAM only",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/functional_contrast_cars.json")
    parser.add_argument("--output", default="outputs/2026-10-10/functional_contrast_cars")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
