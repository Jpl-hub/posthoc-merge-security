"""Fine covariance-contrast directions with replacement preflight before defense.

Reuse pinned NUFILT SVD, scipy eigensolvers and author CLIP/TA/AGA. Local
activation contrast is a necessary proxy, not a guarantee about merged models.
No gradients, response fitting, future expert access during construction, or
novelty claim. Raw activations, directions and predictions stay on the server.
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
from eight_task_benchmark import author_recover, digest, merge_all
from functional_contrast import features
from joint_posthoc import flatten_tokens
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, replace_components
from scipy.linalg import eigh
from subspace_probe import make_loader, save_json


def fresh_indices(config, task, length):
    excluded = set()
    for name in config["exclusions"]:
        value = json.loads(Path(name).read_bytes())
        if task in value:
            value = value[task]
        elif task != config["task"] or "calibration" not in value:
            continue
        for indices in value.values():
            if isinstance(indices, list):
                excluded.update(indices)
    if task == "MNIST":
        excluded.update(json.loads(Path(config["mnist_exclusions"]).read_bytes()))
    order = torch.randperm(length, generator=torch.Generator().manual_seed(config["seed"])).tolist()
    available = [i for i in order if i not in excluded]
    n, m = config["calibration_samples"], config["confirmation_samples"]
    assert len(available) >= n + m, (task, len(available))
    return dict(
        calibration=available[:n], confirmation=available[n : n + m], excluded=len(excluded)
    )


@torch.no_grad()
def capture(model, images, layers):
    raw, handles = {}, []
    for layer in layers:
        block = model.model.visual.transformer.resblocks[layer]
        for name, module in [("qkv", block.ln_1), ("fc", block.ln_2)]:
            key = f"{name}:{layer}"
            raw[key] = []
            handles.append(
                module.register_forward_hook(
                    lambda _m, _a, value, k=key: raw[k].append(flatten_tokens(value))
                )
            )
        key = f"down:{layer}"
        raw[key] = []
        handles.append(
            block.mlp.c_proj.register_forward_pre_hook(
                lambda _m, args, k=key: raw[k].append(flatten_tokens(args[0]))
            )
        )
    model.cuda().eval()
    try:
        for batch in images.split(16):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return {key: torch.cat(value) for key, value in raw.items()}


def matrix_keys(site):
    kind, layer = site.split(":")
    prefix = f"model.visual.transformer.resblocks.{layer}."
    suffix = {"qkv": "attn.in_proj_", "fc": "mlp.c_fc.", "down": "mlp.c_proj."}[kind]
    return prefix + suffix + "weight", prefix + suffix + "bias"


@torch.no_grad()
def fit_direction(site, teacher, anchor, values, config, svd):
    kind, layer_text = site.split(":")
    layer = int(layer_text)
    weight_key, _ = matrix_keys(site)
    delta = (
        teacher.state_dict()[weight_key].double() - anchor.state_dict()[weight_key].double()
    ).cuda()
    x = values[config["task"]].double().cuda()
    mean = x.mean(0)
    centered = {name: value.double().cuda() - mean for name, value in values.items()}
    dimension = x.shape[1]
    if kind == "down":
        own_variance = centered[config["task"]].square().mean(0)
        regularizer = config["regularizer_trace_fraction"] * own_variance.mean()
        contrasts = torch.stack(
            [
                z.square().mean(0) / (own_variance + regularizer)
                for name, z in centered.items()
                if name != config["task"]
            ]
        )
        order = torch.argsort(contrasts.min(0).values, descending=True, stable=True)
        neurons = order[: config["sparse_neurons"]]
        basis = torch.eye(dimension, dtype=torch.float64, device="cuda")[:, neurons]
        rank, retained = None, None
    else:
        left, singular, vectors = svd(delta, full_matrices=False)
        rank = (
            int(
                torch.searchsorted(
                    singular.square().cumsum(0) / singular.square().sum(),
                    config["task_vector_energy"],
                )
            )
            + 1
        )
        retained = float(singular[:rank].square().sum() / singular.square().sum())
        span = vectors[:, :rank]
        block = teacher.model.visual.transformer.resblocks[layer]
        norm = block.ln_1 if kind == "qkv" else block.ln_2
        null = norm.weight.double().cuda().reciprocal()
        residual = null - span @ (span.T @ null)
        if float(residual.norm()) > 1e-8:
            span = torch.cat([span, (residual / residual.norm())[:, None]], dim=1)
        basis = torch.linalg.qr(span, mode="complete").Q[:, span.shape[1] :]
        neurons = None
    assert basis.shape[1] > 0, site
    projected = {name: value @ basis for name, value in centered.items()}
    covariances = {name: z.T @ z / len(z) for name, z in projected.items()}
    target = covariances[config["task"]]
    other = sum(c for name, c in covariances.items() if name != config["task"]) / (len(values) - 1)
    rho = config["regularizer_trace_fraction"] * float(target.trace()) / len(target) + 1e-12
    eigenvalues, eigenvectors = eigh(
        other.cpu().numpy(),
        (target + rho * torch.eye(len(target), dtype=target.dtype, device="cuda")).cpu().numpy(),
    )
    vector = torch.from_numpy(eigenvectors[:, -1].copy()).cuda()
    incoming = basis @ vector
    incoming /= incoming.norm()
    incoming *= 1 if incoming[incoming.abs().argmax()] >= 0 else -1
    outgoing = delta @ incoming if kind == "down" else left[:, 0]
    outgoing /= outgoing.norm()
    record = dict(
        site=site,
        task_rank=rank,
        task_energy_retained=retained,
        complement_dimension=basis.shape[1],
        rho=rho,
        maximum_regularized_contrast=float(eigenvalues[-1]),
        neurons=neurons.cpu().tolist() if neurons is not None else None,
    )
    if kind != "down":
        record["task_span_leakage"] = float((vectors[:, :rank].T @ incoming).norm())
        record["affine_null_leakage"] = float(abs(incoming @ null) / null.norm())
        assert record["task_span_leakage"] < 1e-8 and record["affine_null_leakage"] < 1e-8
    evidence = dict(
        mean=mean.cpu(),
        incoming=incoming.cpu(),
        outgoing=outgoing.cpu(),
        basis=basis.cpu(),
        projected_covariances={k: v.cpu() for k, v in covariances.items()},
    )
    return record, evidence


def assess(direction, values, task):
    v, mean = direction["incoming"].double(), direction["mean"].double()
    powers = {name: float(((z.double() - mean) @ v).square().mean()) for name, z in values.items()}
    ratios = {
        name: (power / max(powers[task], 1e-30)) ** 0.5
        for name, power in powers.items()
        if name != task
    }
    return dict(powers=powers, rms_ratios=ratios)


def support_for(records):
    result = {}
    for row in records:
        weight, bias = matrix_keys(row["site"])
        result[weight] = row["neurons"]  # None means complete matrix.
        result[bias] = None
    return result


@torch.no_grad()
def reset(model, anchor, support):
    result = copy.deepcopy(model).cpu().eval()
    state, base = result.state_dict(), anchor.state_dict()
    for key, columns in support.items():
        if columns is None:
            state[key].copy_(base[key])
        else:
            state[key][:, columns] = base[key][:, columns]
    return result


@torch.no_grad()
def perturb(original, directions, amplitude):
    model = copy.deepcopy(original).cpu().eval()
    state = model.state_dict()
    for site, direction in directions.items():
        weight, bias = matrix_keys(site)
        u, v, mean = direction["outgoing"], direction["incoming"], direction["mean"]
        change = amplitude * original.state_dict()[weight].double().norm() * u[:, None] @ v[None, :]
        state[weight].add_(change.float())
        state[bias].sub_((change @ mean).float())
    return model


@torch.no_grad()
def run(args):
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    save_json(out / "protocol.json", config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    task = config["task"]
    original, anchor = load(str(root / task / "finetuned.pt")), load(str(root / "zeroshot.pt"))
    svd = author_svd(args.nufilt_repo, json.loads(Path(config["nufilt_config"]).read_bytes()))
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset
    from torch.utils.data import DataLoader, TensorDataset

    datasets = {
        name: get_dataset(name, original.val_preprocess, args.data, batch_size=128, num_workers=2)
        for name in config["public_distributions"]
    }
    indices = {
        name: fresh_indices(config, name, len(data.train_dataset))
        for name, data in datasets.items()
    }
    save_json(out / "training_indices.json", indices)
    inputs, raw = {}, {}
    for split in ("calibration", "confirmation"):
        for name, data in datasets.items():
            assert len(data.test_dataset) == fusion["expected_test_counts"][name]
            images = torch.stack([data.train_dataset[i][0] for i in indices[name][split]])
            inputs[split, name] = images
            raw[split, name] = capture(original, images, config["layers"])
            print("CAPTURE", split, name, flush=True)
    records, directions = [], {}
    for site in raw["calibration", task]:
        values = {name: raw["calibration", name][site] for name in datasets}
        record, direction = fit_direction(site, original, anchor, values, config, svd)
        record["calibration"] = assess(direction, values, task)
        record["confirmation"] = assess(
            direction, {name: raw["confirmation", name][site] for name in datasets}, task
        )
        shift = record["confirmation"]["powers"][task] / record["calibration"]["powers"][task]
        record["target_power_shift"] = shift
        record["contrast_passed"] = (
            min(record["confirmation"]["rms_ratios"].values())
            >= config["minimum_confirmed_rms_ratio"]
            and 1 / config["maximum_target_power_shift"]
            <= shift
            <= config["maximum_target_power_shift"]
        )
        records.append(record)
        directions[site] = direction
        print(
            "DIRECTION",
            site,
            record["confirmation"]["rms_ratios"],
            record["contrast_passed"],
            flush=True,
        )
    torch.save(raw, artifacts / "activations.pt")
    torch.save(directions, artifacts / "directions.pt")
    save_json(out / "directions.json", records)
    eligible = [r for r in records if r["contrast_passed"]]
    supports = {
        kind: support_for([r for r in records if r["site"].startswith(kind + ":")])
        for kind in ("qkv", "fc", "down")
    }
    supports["eligible_union"] = support_for(eligible)
    save_json(out / "supports.json", supports)
    save_json(
        out / "direction_freeze.json",
        dict(
            directions_sha256=digest(artifacts / "directions.pt"),
            selected_sites=[r["site"] for r in eligible],
            future_partner_checkpoints_loaded=False,
            labels_used=False,
        ),
    )
    # Predeclared floor checks start only after target-only directions freeze.
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    models = {name: load(str(root / name / "finetuned.pt")) for name in fusion["tasks"]}
    models[task], models["anchor"] = original, anchor
    head = load(str(root / f"head_{task}.pt"))
    parent = Path(config["replacement_parent"])
    old_indices = json.loads((parent / "training_indices.json").read_bytes())[task]["confirmation"]
    batches = list(make_loader(datasets[task].train_dataset, old_indices))
    loader = DataLoader(
        TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
        batch_size=128,
        pin_memory=True,
    )
    rows, predictions = [], {}

    def measure(preset, group, model, dataset, data_loader, own_head, stage):
        p, y = [], []
        score = evaluate(
            model,
            own_head,
            data_loader,
            "cuda",
            prediction_sink=lambda a, b: (p.append(a), y.append(b)),
        )
        key = f"{stage}/{preset}/{group}/{dataset}"
        predictions[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
        rows.append(dict(id=key, stage=stage, preset=preset, group=group, dataset=dataset, **score))
        torch.save(predictions, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        print("MEASURE", key, score["correct"], score["n"], flush=True)

    floors = {}
    for preset, support in {"original": {}, **supports}.items():
        own = reset(original, anchor, support)
        measure(preset, "standalone", own, task, loader, head, "replacement_preflight")
        combined = merge_all(dict(models, **{task: own}), root, config["coefficient"])
        measure(preset, "fusion", combined, task, loader, head, "replacement_preflight")
        floors[preset] = rows[-1]["top1"]
        del own, combined
    gap = 100 * (floors["original"] - floors["eligible_union"])
    decision = dict(
        eligible_sites=[r["site"] for r in eligible],
        support_recovery_floor=floors["eligible_union"],
        maximum_possible_drop_pp=gap,
        candidate_constructed=False,
        novelty_claim=False,
        reason="No confirmed direction"
        if not eligible
        else "Exact-support replacement excludes the declared 5 pp recovery target"
        if gap < config["minimum_recovery_gap_pp"]
        else "Necessary preflight passed; finite candidate still needs validation",
    )
    if eligible and gap >= config["minimum_recovery_gap_pp"]:
        selected = {r["site"]: directions[r["site"]] for r in eligible}
        images = inputs["calibration", task]
        step = config["candidate_finite_difference_step"]
        plus, minus = perturb(original, selected, step), perturb(original, selected, -step)
        derivative = (
            features(plus.cuda().eval(), images) - features(minus.cuda().eval(), images)
        ) / (2 * step)
        plus.cpu()
        minus.cpu()
        rms = float((derivative.square().sum() / len(images)).sqrt())
        amplitude = min(
            config["candidate_maximum_weight_ratio"],
            config["candidate_feature_rms_budget"] / max(rms, 1e-12),
        )
        release = perturb(original, selected, amplitude)
        torch.save(release.state_dict(), artifacts / "release.pt")
        save_json(
            out / "candidate_freeze.json",
            dict(
                amplitude=amplitude,
                derivative_feature_rms=rms,
                release_sha256=digest(artifacts / "release.pt"),
                test_results_used=False,
                future_partner_weights_used=False,
            ),
        )
        decision["candidate_constructed"] = True
        author, author_hash = load_author(
            Path(args.author_repo) / "utils_attack_B32.py", include_wrappers=True
        )
        assert author_hash == json.loads(Path("configs/sources.json").read_bytes())["AGA"]["sha256"]
        loaders, heads = {}, {}
        for name, data in datasets.items():
            batches = list(data.test_loader)
            loaders[name] = DataLoader(
                TensorDataset(
                    torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])
                ),
                batch_size=128,
                pin_memory=True,
            )
            heads[name] = load(str(root / f"head_{name}.pt"))
        for preset, model in [("original", original), ("protected", release)]:
            for attack in ["direct", "aga", "reset_attention", "reset_mlp", "reset_support"]:
                assert time.perf_counter() - started < config["time_limit_seconds"]
                if attack == "direct":
                    attacked = model
                elif attack == "aga":
                    attacked = author_recover(model, anchor, author)
                    torch.save(attacked.state_dict(), artifacts / f"{preset}_aga.pt")
                elif attack == "reset_support":
                    attacked = reset(model, anchor, supports["eligible_union"])
                    control = reset(original, anchor, supports["eligible_union"])
                    assert all(
                        torch.equal(v, control.state_dict()[k])
                        for k, v in attacked.state_dict().items()
                    )
                else:
                    attacked, _ = replace_components(
                        model,
                        anchor,
                        config["layers"],
                        "attn" if attack == "reset_attention" else "mlp",
                    )
                label = f"{preset}_{attack}"
                measure(
                    label, "standalone", attacked, task, loaders[task], heads[task], "official_test"
                )
                combined = merge_all(dict(models, **{task: attacked}), root, config["coefficient"])
                for name in config["evaluation_tasks"]:
                    measure(
                        label, "fusion", combined, name, loaders[name], heads[name], "official_test"
                    )
                del combined
    for row in rows:
        preset = (
            "original"
            if row["stage"] == "replacement_preflight"
            else "original_" + row["preset"].split("_", 1)[1]
        )
        ref = predictions[f"{row['stage']}/{preset}/{row['group']}/{row['dataset']}"]
        current = predictions[row["id"]]
        assert torch.equal(ref["labels"], current["labels"])
        row["paired"] = paired_counts(
            ref["prediction"].eq(ref["labels"]), current["prediction"].eq(current["labels"])
        )
    save_json(out / "results.json", rows)
    if decision["candidate_constructed"]:
        lookup = {row["id"]: row for row in rows}
        best = max(
            (
                r
                for r in rows
                if r["stage"] == "official_test"
                and r["preset"].startswith("protected_")
                and r["group"] == "fusion"
                and r["dataset"] == task
            ),
            key=lambda r: r["top1"],
        )
        original_stand = lookup[f"official_test/original_direct/standalone/{task}"]
        protected_stand = lookup[f"official_test/protected_direct/standalone/{task}"]
        original_merge = lookup[f"official_test/original_direct/fusion/{task}"]
        decision.update(
            utility_loss_pp=100 * (original_stand["top1"] - protected_stand["top1"]),
            best_measured_recovery=best["preset"],
            best_recovered_top1=best["top1"],
            gap_from_original_direct_pp=100 * (original_merge["top1"] - best["top1"]),
            reason="Finite candidate evaluated; all performance and attack outcomes reported",
        )
    save_json(out / "decision.json", decision)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "functional_contrast.py",
                    "joint_posthoc.py",
                    "mechanisms.py",
                    "nufilt_alignment.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            checkpoint_sha256={
                p: digest(root / p)
                for p in [
                    "zeroshot.pt",
                    *[f"{name}/finetuned.pt" for name in fusion["tasks"]],
                    *[f"head_{name}.pt" for name in config["evaluation_tasks"]],
                ]
            },
            optimizer_steps=0,
            response_fits=0,
            calibration_labels_used=False,
            direction_partner_checkpoints_used=False,
            calibration_distributions=config["public_distributions"],
            all_tokens=50,
            floor_split="Previously inspected Cars training confirmation, not official test",
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fine_subspace_cars.json")
    parser.add_argument("--output", default="outputs/2026-10-11/fine_subspace_cars")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--nufilt-repo", default="external/NUFILT")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
