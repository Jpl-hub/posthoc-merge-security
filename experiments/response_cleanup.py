"""Anchor-nearest linear cleanup under a fixed public calibration response budget.

This is a bounded recovery diagnostic, not a new defense or a novelty claim.
Closed-form Tikhonov filtering and mean compensation use only the accessible
release, public anchor and unlabeled training inputs. Existing TA/AGA are reused.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate, merge
from critical_subspace import capture, parameters
from eight_task_benchmark import author_recover, digest, merge_all
from functional_contrast import features
from local_erasure import cache
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_subspace import context, read
from subspace_probe import save_json


@torch.no_grad()
def solve_cleanup(weight, bias, anchor_weight, inputs, relative_budget):
    """Minimum ||D||_F with ||(D-delta)Xc^T||_F bounded; bias fixes the mean."""
    x = inputs.double().cuda()
    mean = x.mean(0)
    x -= mean
    covariance = x.T @ x / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    negative = float(values.min())
    assert negative >= -1e-10 * float(values.max()), negative
    cutoff = float(values.max()) * 1e-12
    values = torch.where(values > cutoff, values, 0)
    original = weight.double().cuda()
    base = anchor_weight.double().cuda()
    delta = original - base
    coordinates = delta @ vectors
    powers = coordinates.square().sum(0) * values
    native_power = float(((original @ vectors).square().sum(0) * values).sum())
    budget = relative_budget**2 * native_power
    reset_power = float(powers.sum())
    if reset_power <= budget:
        multiplier = None
        factors = torch.zeros_like(values)
    else:

        def error(value):
            removed = torch.where(values > 0, value / (values + value), 1)
            return float((powers * removed.square()).sum())

        lower, upper = 0.0, float(values.max())
        while error(upper) <= budget:
            upper *= 2
        for _ in range(64):
            middle = (lower + upper) / 2
            if error(middle) <= budget:
                lower = middle
            else:
                upper = middle
        multiplier = lower
        factors = torch.where(values > 0, values / (values + multiplier), 0)
    cleaned_delta = (coordinates * factors) @ vectors.T
    compiled_weight = (base + cleaned_delta).float()
    compiled_bias = (bias.double().cuda() + (original - compiled_weight.double()) @ mean).float()
    residual = compiled_weight.double() - original
    actual_power = float(((residual @ vectors).square().sum(0) * values).sum())
    predicted_power = float((powers * (1 - factors).square()).sum())
    mean_error = float(
        (residual @ mean + compiled_bias.double() - bias.double().cuda()).abs().max()
    )
    stationarity = 0.0
    if multiplier is not None:
        lhs = cleaned_delta @ covariance + multiplier * cleaned_delta
        rhs = delta @ covariance
        stationarity = float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-30))
        assert stationarity < 1e-7, stationarity
    assert actual_power <= budget * 1.0001 + 1e-10
    row = dict(
        tokens=len(x),
        input_width=x.shape[1],
        covariance_numerical_rank=int(values.gt(0).sum()),
        covariance_cutoff=cutoff,
        smallest_raw_eigenvalue=negative,
        response_budget_ratio=relative_budget,
        centered_native_response_rms=native_power**0.5,
        full_anchor_reset_response_ratio=(reset_power / native_power) ** 0.5,
        predicted_response_ratio=(predicted_power / native_power) ** 0.5,
        compiled_response_ratio=(actual_power / native_power) ** 0.5,
        compiled_mean_max_absolute_error=mean_error,
        multiplier=multiplier,
        full_weight_reset=multiplier is None,
        update_norm_before=float(delta.norm()),
        update_norm_after=float(cleaned_delta.norm()),
        removed_update_energy_fraction=1
        - float(cleaned_delta.square().sum() / delta.square().sum()),
        stationarity_relative_error=stationarity,
    )
    evidence = dict(mean=mean.cpu(), eigenvalues=values.cpu(), filter_factors=factors.cpu())
    return compiled_weight.cpu(), compiled_bias.cpu(), row, evidence


def load_release(case, original):
    result = copy.deepcopy(original).cpu().eval()
    path = Path(case["release"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if case["format"] == "mlp_blocks":
        for layer, state in saved.items():
            result.model.visual.transformer.resblocks[layer].mlp.load_state_dict(state, strict=True)
    else:
        assert case["format"] == "edited_state"
        state = result.state_dict()
        for key, value in saved.items():
            state[key].copy_(value)
    return result


def get_indices(cfg, case):
    parent = Path(case["indices_parent"])
    if case["format"] == "mlp_blocks":
        values = read(parent / "training_indices.json")[case["task"]]
        indices = values["attack"][: cfg["samples"]]
        confirmation = values["development"][: cfg["confirmation_samples"]]
        defender = set(values["calibration"])
    else:
        indices = read(parent / "attack_indices.json")[case["task"]][: cfg["samples"]]
        values = read(parent / "training_indices.json")[case["task"]]
        confirmation = values["confirmation"]
        defender = set(values["calibration"])
    assert len(indices) == cfg["samples"] and len(indices) == len(set(indices))
    assert not set(indices).intersection(confirmation)
    assert not set(indices).intersection(defender)
    return dict(calibration=indices, confirmation=confirmation, defender_overlap=0)


@torch.no_grad()
def cleanup(model, anchor, images, case, cfg):
    inputs, checks = capture(model, images, case["layers"])
    result = copy.deepcopy(model).cpu().eval()
    rows, evidence, edited = [], {}, set()
    for layer in case["layers"]:
        for kind in case["kinds"]:
            site = f"{kind}:{layer}"
            keys, weight, bias = parameters(result.state_dict(), site)
            _, base, _ = parameters(anchor.state_dict(), site)
            w, b, row, tensors = solve_cleanup(
                weight, bias, base, inputs.pop(site), cfg["relative_response_budget"]
            )
            weight.copy_(w)
            bias.copy_(b)
            rows.append(dict(site=site, **row))
            evidence[site] = tensors
            edited.update(keys)
            print("CLEANUP", site, row["removed_update_energy_fraction"], flush=True)
    state, before = result.state_dict(), model.state_dict()
    assert all(torch.equal(v, before[k]) for k, v in state.items() if k not in edited)
    for layer in case["layers"]:
        key = f"model.visual.transformer.resblocks.{layer}.attn.in_proj_"
        for suffix in ("weight", "bias"):
            assert torch.equal(state[key + suffix][:1536], before[key + suffix][:1536])
    return result, rows, dict(sites=evidence, context_checks=checks)


@torch.no_grad()
def compare_function(before, after, head, images):
    values = []
    for model in (before, after):
        model.cuda().eval()
        values.append(features(model, images))
        model.cpu()
    a, b = values
    weight = head.weight.detach().float().cpu()
    return dict(
        images=len(images),
        normalized_feature_rms=float((b - a).double().square().sum().div(len(images)).sqrt()),
        native_prediction_changes=int(
            (a @ weight.T).argmax(-1).ne((b @ weight.T).argmax(-1)).sum()
        ),
        labels_used=False,
    )


def baseline(case, preset, attack, group, task, holders):
    parent = Path(case["baseline_direct"] if attack == "cleanup" else case["baseline_aga"])
    if str(parent) not in holders:
        meta = read(parent / "metadata.json")
        validation = read(parent / "validation.json")
        assert meta["completed"] and (
            validation.get("status") == "passed" or validation.get("raw_artifacts_checked")
        )
        holders[str(parent)] = (
            read(parent / "results.json"),
            torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True),
        )
    rows, raw = holders[str(parent)]
    if case["format"] == "mlp_blocks":
        name = "unprotected" if preset == "original" else "compensated_100"
        stage = "direct" if attack == "cleanup" else "aga_selected"
        stage += "_standalone" if group == "standalone" else ""
        key = (name, stage, task)
        matches = [
            r for r in rows if r["preset"] == name and r["group"] == stage and r["dataset"] == task
        ]
    else:
        stage = "direct" if attack == "cleanup" else "aga"
        key = f"{case['task']}/{preset}/{stage}/{group}/{task}"
        matches = [r for r in rows if r["id"] == key]
    assert len(matches) == 1
    return dict(
        source=str(parent),
        source_id=str(key),
        correct=matches[0]["correct"],
        n=matches[0]["n"],
        source_predictions_sha256=digest(parent / "artifacts/predictions.pt"),
    ), raw[key]


@torch.no_grad()
def run(args):
    cfg = read(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg, all_tasks=True)
    from dataset.registry import get_dataset

    anchor = models["anchor"]
    tasks = sorted({t for c in cfg["cases"] for t in c["evaluation_tasks"]})
    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in tasks
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in tasks}
    loaders = {t: cache(data[t]) for t in tasks}
    assert all(len(data[t].test_dataset) == fusion["expected_test_counts"][t] for t in tasks)
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == read("configs/sources.json")["AGA"]["sha256"]
    rows, raw, geometry, functions, sampling, manifests, controls, holders = (
        [],
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for case in cfg["cases"]:
        name, target = case["name"], case["task"]
        original = models[target]
        published = load_release(case, original)
        sampling[name] = get_indices(cfg, case)
        images = {
            k: torch.stack([data[target].train_dataset[i][0] for i in sampling[name][k]])
            for k in ("calibration", "confirmation")
        }
        for preset, model in (("original", original), ("protected", published)):
            clean, geometry[f"{name}/{preset}"], evidence = cleanup(
                model, anchor, images["calibration"], case, cfg
            )
            key = f"{name}_{preset}"
            changed = {
                k: v.clone()
                for k, v in clean.state_dict().items()
                if not torch.equal(v, model.state_dict()[k])
            }
            torch.save(dict(edits=changed, evidence=evidence), artifacts / f"{key}_cleanup.pt")
            manifests[key] = dict(
                edited_keys=list(changed), sha256=digest(artifacts / f"{key}_cleanup.pt")
            )
            for split in images:
                functions[f"{name}/{preset}/{split}"] = compare_function(
                    model, clean, heads[target], images[split]
                )
            for attack in cfg["attacks"]:
                restored = clean if attack == "cleanup" else author_recover(clean, anchor, author)
                if case["fusion"] == "two_task":
                    merged = merge(anchor, restored, models["SVHN"], case["coefficient"])
                else:
                    assert case["coefficient"] == fusion["coefficient"] == 0.2
                    merged = merge_all(
                        dict(models, **{target: restored}), root, case["coefficient"]
                    )
                requests = [
                    ("standalone", target, restored),
                    *[("fusion", t, merged) for t in case["evaluation_tasks"]],
                ]
                for group, task, evaluated in requests:
                    assert time.perf_counter() - started < cfg["time_limit_seconds"]
                    id_ = f"{name}/{preset}/{attack}/{group}/{task}"
                    p, y = [], []
                    score = evaluate(
                        evaluated,
                        heads[task],
                        loaders[task],
                        "cuda",
                        prediction_sink=lambda a, b: (p.append(a), y.append(b)),
                    )
                    value = dict(prediction=torch.cat(p), labels=torch.cat(y))
                    reference, parent = baseline(case, preset, attack, group, task, holders)
                    assert torch.equal(parent["labels"], value["labels"])
                    raw[id_] = value
                    controls[id_] = parent
                    rows.append(
                        dict(
                            id=id_,
                            case=name,
                            target=target,
                            preset=preset,
                            attack=attack,
                            group=group,
                            dataset=task,
                            **score,
                            baseline=reference,
                            change_from_parent=paired_counts(
                                parent["prediction"].eq(parent["labels"]),
                                value["prediction"].eq(value["labels"]),
                            ),
                        )
                    )
                    torch.save(
                        dict(new=raw, reused_baselines=controls), artifacts / "predictions.pt"
                    )
                    save_json(out / "results.json", rows)
                    print("RESULT", id_, score["correct"], score["n"], flush=True)
                del merged, restored
            del clean
        del published
        save_json(out / "geometry.json", geometry)
        save_json(out / "functions.json", functions)
        save_json(out / "training_indices.json", sampling)
    for row in rows:
        value = raw[row["id"]]
        control = raw[row["id"].replace("/protected/", "/original/")]
        row["matched"] = paired_counts(
            control["prediction"].eq(control["labels"]), value["prediction"].eq(value["labels"])
        )
    save_json(out / "results.json", rows)
    save_json(out / "cleanup_manifest.json", manifests)
    helper_names = [
        "clip_benchmark.py",
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "functional_contrast.py",
        "joint_posthoc.py",
        "local_erasure.py",
        "mechanisms.py",
        "operator_diagnostics.py",
        "path_subspace.py",
        "subspace_probe.py",
    ]
    inputs = {Path(c[k]) for c in cfg["cases"] for k in ("release",)}
    inputs.update(
        Path(c[k]) / n
        for c in cfg["cases"]
        for k in ("baseline_direct", "baseline_aga", "indices_parent")
        for n in ("metadata.json", "validation.json", "training_indices.json")
        if (Path(c[k]) / n).exists()
    )
    checkpoints = [
        "zeroshot.pt",
        *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
        *[f"head_{t}.pt" for t in tasks],
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - started,
            rows=len(rows),
            actual_full_evaluations=len(rows),
            reused_parent_comparisons=len(controls),
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path(__file__).with_name(n)) for n in helper_names},
            input_sha256={str(p): digest(p) for p in inputs},
            checkpoint_sha256={n: digest(root / n) for n in checkpoints},
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            original_author_aga_sha256=author_hash,
            calibration_labels_used=False,
            original_protected_expert_used_by_cleanup=False,
            future_partner_used_by_cleanup=False,
            optimizer_steps=0,
            novelty_claim=False,
            new_defense_validated=False,
        ),
    )
    print("COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/response_cleanup.json")
    parser.add_argument("--output", default="outputs/2026-10-11/response_cleanup")
    run(parser.parse_args())
