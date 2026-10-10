"""Test a NUFILT-proxy complement perturbation and explicit recovery counterexamples.

Frozen protocol; no optimizer or fine-tuning. The low-rank additive construction
is a mechanism hypothesis, not an original algorithm or validated protection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from bounded_complement import angular_cleanup
from clip_benchmark import checkpoint_loader, evaluate, merge, recover
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import adjust_comparisons, paired_counts
from polynomial_probe import capture, reconstruct
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST, SVHN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def construct_basis(original, anchor, merged, images, layer, config, svd):
    """Restrict the variance-ratio pencil to the task-vector right complement."""
    block = original.model.visual.transformer.resblocks[layer].mlp
    reference = anchor.model.visual.transformer.resblocks[layer].mlp
    delta = (block.c_fc.weight.double() - reference.c_fc.weight.double()).cuda()
    left, singular, right = svd(delta, full_matrices=False)
    complement = right[:, config["task_rank"] :]
    own, mixed = capture(original, images, layer), capture(merged, images, layer)
    z, y = own @ complement, mixed @ complement
    a, b = z.T @ z / len(z), y.T @ y / len(y)
    ridge = config["ridge_ratio"] * a.trace() / len(a)
    values, vectors = torch.linalg.eigh(a)
    whiten = (vectors * (values.clamp_min(0) + ridge).rsqrt()) @ vectors.T
    contrast = whiten @ b @ whiten
    ratios, directions = torch.linalg.eigh((contrast + contrast.T) / 2)
    rank = config["perturbation_rank"]
    incoming = torch.linalg.qr(complement @ whiten @ directions[:, -rank:], mode="reduced").Q
    outgoing = left[:, :rank]
    perturbation = outgoing @ incoming.T
    task_leakage = (perturbation @ right[:, : config["task_rank"]]).norm() / perturbation.norm()
    assert float(task_leakage) < 1e-8
    w, bias = block.c_fc.weight.double().cuda(), block.c_fc.bias.double().cuda()
    own_norm = (own @ w.T + bias).norm()
    response = (own @ perturbation.T).norm()
    return perturbation.cpu(), dict(
        layer=layer,
        task_proxy_rank=config["task_rank"],
        perturbation_rank=rank,
        task_increment_energy_retained=float(
            singular[: config["task_rank"]].square().sum() / singular.square().sum()
        ),
        task_span_leakage=float(task_leakage),
        generalized_ratio_max=float(ratios[-1]),
        unit_expert_response=float(response),
        expert_preactivation_norm=float(own_norm),
        unit_merged_response=float((mixed @ perturbation.T).norm()),
        merged_preactivation_norm=float((mixed @ w.T + bias).norm()),
        ridge=float(ridge),
        calibration_tokens=len(own),
    )


@torch.no_grad()
def pretrained_fc(released, anchor, layers):
    """Architecture-known attacker tries a declared joint FC replacement."""
    result = copy.deepcopy(released).cpu().eval()
    for layer in layers:
        result.model.visual.transformer.resblocks[layer].mlp.c_fc.load_state_dict(
            anchor.model.visual.transformer.resblocks[layer].mlp.c_fc.state_dict()
        )
    return result


@torch.no_grad()
def spectral_cleanup(released, anchor, layers, rank, svd):
    """Subtract leading released-update components; no defender basis or original."""
    result = copy.deepcopy(released).cpu().eval()
    for layer in layers:
        local = result.model.visual.transformer.resblocks[layer].mlp.c_fc
        base = anchor.model.visual.transformer.resblocks[layer].mlp.c_fc
        delta = (local.weight.double() - base.weight.double()).cuda()
        u, s, v = svd(delta, full_matrices=False)
        local.weight.sub_(((u[:, :rank] * s[:rank]) @ v[:, :rank].T).float().cpu())
    return result


@torch.no_grad()
def calibrated_cleanup(released, anchor, images, layers):
    """Preserve released update on 99.99%-energy input span, use anchor elsewhere."""
    result = copy.deepcopy(released).cpu().eval()
    evidence = []
    for layer in layers:
        x = capture(released, images, layer)
        values, vectors = torch.linalg.eigh(x.T @ x / len(x))
        values = values.clamp_min(0).flip(0)
        rank = int(torch.searchsorted(values.cumsum(0) / values.sum(), 0.9999)) + 1
        u = vectors[:, -rank:]
        local = result.model.visual.transformer.resblocks[layer].mlp.c_fc
        base = anchor.model.visual.transformer.resblocks[layer].mlp.c_fc
        delta = (local.weight.double() - base.weight.double()).cuda()
        local.weight.copy_((base.weight.double().cuda() + (delta @ u) @ u.T).float().cpu())
        evidence.append(dict(layer=layer, rank=rank, input_tokens=len(x), retained_energy=0.9999))
    return result, evidence


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    protocol = Path(args.config).read_bytes()
    config = json.loads(protocol)
    (out / "protocol.json").write_bytes(protocol)
    index_file = Path(config["parent"]) / "training_indices.json"
    index_bytes = index_file.read_bytes()
    indices = json.loads(index_bytes)
    (out / "training_indices.json").write_bytes(index_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    nufilt = json.loads(Path("configs/nufilt_alignment.json").read_bytes())
    svd = author_svd(args.nufilt_repo, nufilt)
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "SVHN/finetuned.pt"))
    second = load(str(root / "MNIST/finetuned.pt"))
    coefficient = config["coefficient"]
    start = time.perf_counter()
    original_merge = merge(anchor, original, second, coefficient)
    datasets = dict(
        MNIST=MNIST(args.data, train=True, transform=original.val_preprocess),
        SVHN=SVHN(args.data, split="train", transform=original.val_preprocess),
    )
    heads = {t: load(str(root / f"head_{t}.pt")) for t in datasets}
    loaders = {t: make_loader(d, indices[t]["development"]) for t, d in datasets.items()}
    for task in datasets:
        assert not set(indices[task]["calibration"]).intersection(indices[task]["attack"])
        assert not set(indices[task]["calibration"]).intersection(indices[task]["development"])
    images = torch.stack(
        [datasets["SVHN"][i][0] for i in indices["SVHN"]["calibration"][: config["samples"]]]
    )
    attack_images = torch.stack(
        [datasets["SVHN"][i][0] for i in indices["SVHN"]["attack"][: config["attack_samples"]]]
    )
    rows, predictions, controls = [], {}, {}

    def record(preset, group, model):
        scores = {}
        for task, loader in loaders.items():
            if (group == "standalone" or group.endswith("_standalone")) and task != "SVHN":
                continue
            p, y = [], []

            def sink(pred, target):
                p.append(pred)
                y.append(target)

            result = evaluate(model, heads[task], loader, "cuda", prediction_sink=sink)
            pred, labels = torch.cat(p), torch.cat(y)
            correct = pred == labels
            if preset == "unprotected":
                controls[group, task] = correct
            result.update(
                preset=preset,
                group=group,
                dataset=task,
                split="development",
                paired=paired_counts(controls[group, task], correct),
            )
            rows.append(result)
            predictions[preset, group, task] = dict(prediction=pred, labels=labels)
            scores[task] = result["top1"]
            save_json(out / "results.json", rows)
            print(
                "INTERFERENCE_RESULT",
                preset,
                group,
                task,
                result["correct"],
                result["n"],
                flush=True,
            )
        return scores

    baseline_own = record("unprotected", "standalone", original)["SVHN"]
    baseline_merge = record("unprotected", "direct", original_merge)["SVHN"]
    bases, diagnostics = {}, []
    for layer in config["layers"]:
        bases[layer], evidence = construct_basis(
            original, anchor, original_merge, images, layer, config, svd
        )
        diagnostics.append(evidence)
        print("INTERFERENCE_BASIS", layer, evidence["generalized_ratio_max"], flush=True)
    torch.save(bases, artifacts / "perturbation_directions.pt")
    save_json(out / "geometry.json", diagnostics)
    gates, edits = [], []
    attack_names = [
        "angular",
        "pretrained_fc",
        "spectral",
        "calibrated",
        "aga",
        "reconstruction_1",
        "reconstruction_2",
    ]
    author = None
    measured_attacks = set()

    def attack_record(preset, attack, model):
        scores = record(preset, attack + "_standalone", model)
        # Attack standalone keeps the same task mask as the direct standalone.
        scores.update(record(preset, attack, merge(anchor, model, second, coefficient)))
        return scores

    def attacks(model, name):
        nonlocal author
        result, audit = angular_cleanup(model, anchor, 0.9)
        save_json(out / f"{name}_angular.json", audit)
        yield "angular", result
        yield "pretrained_fc", pretrained_fc(model, anchor, config["layers"])
        yield (
            "spectral",
            spectral_cleanup(model, anchor, config["layers"], config["perturbation_rank"], svd),
        )
        result, audit = calibrated_cleanup(model, anchor, attack_images, config["layers"])
        save_json(out / f"{name}_calibrated.json", audit)
        yield "calibrated", result
        if author is None:
            author, _ = load_author(Path(args.author_repo) / "utils_attack_B32.py")
        recovery = out / f"{name}_aga"
        recovery.mkdir()
        yield "aga", recover(model, anchor, author, recovery)
        restored = copy.deepcopy(model)
        for step in range(2):
            context = model if step == 0 else merge(anchor, restored, second, coefficient)
            updated = copy.deepcopy(restored)
            fitting = []
            for layer in config["layers"]:
                fitted, evidence = reconstruct(
                    model, anchor, attack_images, layer, 1e-4, solver="auto", input_model=context
                )
                updated.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                    fitted.model.visual.transformer.resblocks[layer].mlp.state_dict()
                )
                fitting.append(dict(layer=layer, **evidence))
            restored = updated
            save_json(out / f"{name}_reconstruction_{step + 1}.json", fitting)
            yield f"reconstruction_{step + 1}", restored

    for preset in config["presets"]:
        name = preset["name"]
        candidate = copy.deepcopy(original).cpu().eval()
        for layer, geometry in zip(config["layers"], diagnostics):
            scale = (
                preset["own_response_ratio"]
                * geometry["expert_preactivation_norm"]
                / geometry["unit_expert_response"]
            )
            local = candidate.model.visual.transformer.resblocks[layer].mlp.c_fc
            perturbation = scale * bases[layer]
            local.weight.add_(perturbation.float())
            original_w = original.model.visual.transformer.resblocks[layer].mlp.c_fc.weight.double()
            cosine = float(
                (
                    local.weight.double()
                    * anchor.model.visual.transformer.resblocks[layer].mlp.c_fc.weight.double()
                ).sum()
                / (
                    local.weight.double().norm()
                    * anchor.model.visual.transformer.resblocks[layer]
                    .mlp.c_fc.weight.double()
                    .norm()
                )
            )
            edits.append(
                dict(
                    preset=name,
                    layer=layer,
                    scale=scale,
                    own_response_ratio=preset["own_response_ratio"],
                    merged_response_ratio=coefficient
                    * scale
                    * geometry["unit_merged_response"]
                    / geometry["merged_preactivation_norm"],
                    perturbation_norm=float(perturbation.norm()),
                    weight_norm_ratio=float(local.weight.double().norm() / original_w.norm()),
                    anchor_cosine=cosine,
                )
            )
        if config.get("output_compensation", False):
            fitting = []
            # Match the original function on calibration inputs while retaining E.
            # This is explicit closed-form parameter fitting, not extra fine-tuning.
            for layer in config["layers"]:
                fitted, evidence = reconstruct(
                    original,
                    anchor,
                    images,
                    layer,
                    config["compensation_ridge_ratio"],
                    solver="auto",
                    feature_model=candidate,
                    input_model=original,
                )
                candidate.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                    fitted.model.visual.transformer.resblocks[layer].mlp.state_dict()
                )
                fitting.append(dict(layer=layer, **evidence))
            save_json(out / f"{name}_defender_compensation.json", fitting)
        torch.save(
            {
                layer: candidate.model.visual.transformer.resblocks[layer].mlp.state_dict()
                for layer in config["layers"]
            },
            artifacts / f"{name}.pt",
        )
        own = record(name, "standalone", candidate)["SVHN"]
        direct = record(name, "direct", merge(anchor, candidate, second, coefficient))["SVHN"]
        passed = (
            100 * (baseline_own - own) <= config["utility_budget_pp"]
            and 100 * (baseline_merge - direct) >= config["minimum_protected_task_drop_pp"]
        )
        gates.append(
            dict(
                preset=name,
                utility_loss_pp=100 * (baseline_own - own),
                direct_loss_pp=100 * (baseline_merge - direct),
                passed=passed,
            )
        )
        save_json(out / "gates.json", gates)
        save_json(out / "edits.json", edits)
        if not passed and not config.get("force_attack_diagnostic", False):
            continue
        if not measured_attacks:
            for attack, restored in attacks(original, "unprotected"):
                attack_record("unprotected", attack, restored)
                measured_attacks.add(attack)
        assert measured_attacks == set(attack_names)
        for attack, restored in attacks(candidate, name):
            attack_record(name, attack, restored)
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    torch.save(predictions, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            gpu=torch.cuda.get_device_name(),
            source_sha256=digest(__file__),
            config_sha256=hashlib.sha256(protocol).hexdigest(),
            indices_sha256=digest(index_file),
            predictions_sha256=digest(artifacts / "predictions.pt"),
            nufilt_commit=nufilt["upstream_commit"],
            nufilt_svd_sha256=nufilt["upstream_svd_sha256"],
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / "SVHN/finetuned.pt",
                    root / "MNIST/finetuned.pt",
                ]
            },
            helper_sha256={
                p: digest(Path(__file__).with_name(p))
                for p in [
                    "clip_benchmark.py",
                    "polynomial_probe.py",
                    "bounded_complement.py",
                    "nufilt_alignment.py",
                    "operator_diagnostics.py",
                ]
            },
            new_defense_validated=False,
            test_usage="none",
            seed_count=1,
            calibration_labels_used=False,
            optimizer_steps=0,
            defender_parameter_fitting=(
                "closed-form output compensation on 256 unlabeled calibration images"
                if config.get("output_compensation", False)
                else "none"
            ),
        ),
    )
    print("INTERFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/subspace_interference.json")
    parser.add_argument("--output", default="outputs/2026-10-10/subspace_interference")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--nufilt-repo", default="external/NUFILT")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
