"""Synthetic-calibration low-activity perturbation: closed form, no training.

Screen one declared candidate, then attack it with both author AGA and adaptive
low-activity projection. No test samples are used to construct the perturbation.
Acceptance budget: standalone MNIST accuracy may fall by at most 0.5 percentage
points from the measured original 99.69%; the budget is not a security guarantee.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import checkpoint_loader, evaluate, merge, protect, recover
from mechanisms import load_author, relative
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def synthetic_images(preprocess, count: int, seed: int):
    """Procedural strokes at 28x28; no downloaded images, labels or task examples."""
    rng = np.random.default_rng(seed)
    values = []
    for i in range(count):
        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < 0.5:
                points = [tuple(map(int, rng.integers(3, 25, 2))) for _ in range(3)]
                draw.line(points, fill=int(rng.integers(180, 256)), width=int(rng.integers(1, 4)))
            else:
                x, y = map(int, rng.integers(3, 12, 2))
                draw.arc(
                    (x, y, x + int(rng.integers(8, 15)), y + int(rng.integers(8, 15))),
                    int(rng.integers(0, 90)),
                    int(rng.integers(190, 360)),
                    fill=int(rng.integers(180, 256)),
                    width=int(rng.integers(1, 4)),
                )
        values.append(preprocess(image))
    return torch.stack(values)


@torch.no_grad()
def collect_basis(model, calibration, layers, rank):
    """Centered covariance eigenspaces of public synthetic activations."""
    captured = {layer: [] for layer in layers}
    handles = []
    model.cuda().eval()
    for layer in layers:
        module = model.model.visual.transformer.resblocks[layer].ln_1
        handles.append(
            module.register_forward_hook(
                lambda _m, _a, output, index=layer: captured[index].append(
                    output.detach().reshape(-1, output.shape[-1]).cpu()
                )
            )
        )
    for batch in calibration.split(8):
        model(batch.cuda())
    for handle in handles:
        handle.remove()
    model.cpu()
    bases, stats = {}, []
    for layer in layers:
        x = torch.cat(captured[layer]).double().cuda()
        center = x.mean(0)
        x -= center
        covariance = x.T @ x / len(x)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        bases[layer] = (center.cpu(), eigenvectors[:, :rank].cpu())
        stats.append(
            dict(
                layer=layer,
                samples=len(x),
                rank=rank,
                low_variance_fraction=float(eigenvalues[:rank].sum() / eigenvalues.sum()),
                low_eigenvalue_max=float(eigenvalues[rank - 1]),
            )
        )
    return bases, stats


@torch.no_grad()
def perturb(model, bases, strength, seed):
    torch.manual_seed(seed)
    result = copy.deepcopy(model).cpu().eval()
    for layer, (center, basis) in bases.items():
        module = result.model.visual.transformer.resblocks[layer].attn
        u = torch.randn(module.in_proj_weight.shape[0], basis.shape[1], dtype=torch.float64)
        change = u @ basis.T
        change *= strength * module.in_proj_weight.double().norm() / change.norm()
        module.in_proj_weight.add_(change.float())
        module.in_proj_bias.sub_((change @ center).float())
    return result


@torch.no_grad()
def erase_low_activity(model, bases):
    """Adaptive input projection, with mean compensation, before alignment."""
    result = copy.deepcopy(model).cpu().eval()
    for layer, (center, basis) in bases.items():
        module = result.model.visual.transformer.resblocks[layer].attn
        removed = (module.in_proj_weight.double() @ basis) @ basis.T
        module.in_proj_weight.sub_(removed.float())
        module.in_proj_bias.add_((removed @ center).float())
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    author, source_hash = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    head = load(str(root / "head_MNIST.pt"))
    calibration = synthetic_images(original.val_preprocess, args.samples, args.seed)
    overall = time.perf_counter()
    bases, statistics = collect_basis(original, calibration, args.layers, args.rank)
    (output / "covariance.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    modified = perturb(original, bases, args.strength, args.seed + 100)
    candidate = protect(modified, args.seed)
    construction_seconds = time.perf_counter() - overall
    dataset = MNIST(args.data, train=False, download=False, transform=original.val_preprocess)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    x, _ = next(iter(loader))
    x = x[:16].cuda()

    @torch.no_grad()
    def features(model):
        value = model.cuda().eval()(x).cpu()
        model.cpu()
        return value

    y = features(original)
    rows = []
    common = dict(
        stage="exploration",
        experiment="synthetic_low_activity",
        scope="real_MNIST_full_test",
        dataset="MNIST",
        seed=args.seed,
        strength=args.strength,
        rank=args.rank,
        layers=args.layers,
        calibration_samples=args.samples,
        calibration_source="procedural_unlabelled_strokes",
        training="none",
        author_sha256=source_hash,
    )
    metric = evaluate(candidate, head, loader, "cuda")
    accepted = metric["top1"] >= 0.9969 - 0.005
    rows.append(
        dict(
            common,
            group="low_activity_standalone",
            coefficient=None,
            feature_error=relative(features(candidate), y),
            within_utility_budget=accepted,
            **metric,
        )
    )
    (output / "low_activity_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("LOW_ACTIVITY", json.dumps(rows[-1]), flush=True)
    # Even rejected candidates receive attacks, so negative evidence is retained.
    rawdir = output / "author_attack"
    rawdir.mkdir(exist_ok=True)
    attack_start = time.perf_counter()
    recovered = recover(candidate, anchor, author, rawdir)
    author_attack_seconds = time.perf_counter() - attack_start
    attack_start = time.perf_counter()
    # Attacker recomputes directions from the released model using the same public
    # procedural distribution, without knowing the defender's fitted eigenvectors.
    attack_bases, attack_stats = collect_basis(candidate, calibration, args.layers, args.rank)
    attack_model = erase_low_activity(candidate, attack_bases)
    attack_anchor = erase_low_activity(anchor, attack_bases)
    attack_second = erase_low_activity(second, attack_bases)
    attack_original = erase_low_activity(original, attack_bases)
    adaptivedir = output / "adaptive_attack"
    adaptivedir.mkdir(exist_ok=True)
    adaptive = recover(attack_model, attack_anchor, author, adaptivedir)
    adaptive_attack_seconds = time.perf_counter() - attack_start
    for group, model in [
        ("low_activity_AGA", recovered),
        ("low_activity_projection_AGA", adaptive),
    ]:
        rows.append(
            dict(
                common,
                group=group,
                coefficient=None,
                **evaluate(model, head, loader, "cuda"),
                feature_error=relative(features(model), y),
            )
        )
        print("LOW_ACTIVITY", json.dumps(rows[-1]), flush=True)
    for group, model, base, other in [
        ("low_activity_direct_merge", candidate, anchor, second),
        ("low_activity_AGA_merge", recovered, anchor, second),
        ("low_activity_projection_AGA_merge", adaptive, attack_anchor, attack_second),
        ("low_activity_projected_baseline_merge", attack_original, attack_anchor, attack_second),
    ]:
        merged = merge(base, model, other, 0.5)
        rows.append(
            dict(common, group=group, coefficient=0.5, **evaluate(merged, head, loader, "cuda"))
        )
        print("LOW_ACTIVITY", json.dumps(rows[-1]), flush=True)
        del merged
    (output / "low_activity_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata = dict(
        total_seconds=time.perf_counter() - overall,
        within_utility_budget=accepted,
        allowed_accuracy_drop_pp=0.5,
        baseline_accuracy=0.9969,
        construction_seconds=construction_seconds,
        author_attack_seconds=author_attack_seconds,
        adaptive_attack_seconds=adaptive_attack_seconds,
        rank=args.rank,
        layers=args.layers,
        training="none",
        calibration="synthetic_only",
        attack_covariance=attack_stats,
        merge_implementation="author_TaskVector_difference_and_sum",
        classifier_implementation="author_ImageClassifier_and_get_logits",
    )
    (output / "low_activity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("LOW_ACTIVITY_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/low_activity")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 9, 10, 11])
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261009)
    run(parser.parse_args())
