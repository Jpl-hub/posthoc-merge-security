"""CLIP ViT-B/32 post-hoc protection and author AGA, using official checkpoints.

No optimization or training. Test splits are used only for measurement. Task
Arithmetic coefficients are declared before evaluation, never fitted on tests.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import types
from pathlib import Path

import torch
from mechanisms import PREFIX, load_author, protect_attention, relative, state_error
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN


def checkpoint_loader(author_repo: Path):
    """Register the upstream pickle's legacy src.modeling name without editing it."""
    sys.path.insert(0, str(author_repo.resolve()))
    import modeling
    from utils import load_model_checkpoint

    package = types.ModuleType("src")
    package.__path__ = []
    sys.modules.setdefault("src", package)
    sys.modules.setdefault("src.modeling", modeling)
    return load_model_checkpoint


def layer_attention(state: dict[str, torch.Tensor], layer: int):
    prefix = f"model.visual.transformer.resblocks.{layer}.attn."
    return {
        PREFIX + key.removeprefix(prefix): value.cuda()
        for key, value in state.items()
        if key.startswith(prefix)
    }


@torch.no_grad()
def protect(model, seed: int):
    torch.manual_seed(seed)
    protected = copy.deepcopy(model).cpu().eval()
    state = protected.state_dict()
    for layer, block in enumerate(protected.model.visual.transformer.resblocks):
        transformed = protect_attention(layer_attention(state, layer), 12)
        prefix = f"model.visual.transformer.resblocks.{layer}.attn."
        for key, value in transformed.items():
            state[prefix + key.removeprefix(PREFIX)] = value.cpu()
        permutation = torch.randperm(block.mlp.c_fc.weight.shape[0])
        state[f"model.visual.transformer.resblocks.{layer}.mlp.c_fc.weight"] = (
            block.mlp.c_fc.weight[permutation].clone()
        )
        state[f"model.visual.transformer.resblocks.{layer}.mlp.c_fc.bias"] = block.mlp.c_fc.bias[
            permutation
        ].clone()
        state[f"model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight"] = (
            block.mlp.c_proj.weight[:, permutation].clone()
        )
    protected.load_state_dict(state, strict=True)
    return protected


@torch.no_grad()
def recover(model, anchor, author, output: Path):
    recovered = copy.deepcopy(model).cpu().eval()
    state = recovered.state_dict()
    reference = anchor.state_dict()
    records = []
    for layer, block in enumerate(recovered.model.visual.transformer.resblocks):
        t0 = time.perf_counter()
        new = author["attack_attention_headwise_precise"](
            state, reference, layer, num_heads=12, device="cuda"
        )
        prefix = f"model.visual.transformer.resblocks.{layer}.attn."
        expected = {
            prefix + k
            for k in [
                "in_proj_weight",
                "in_proj_bias",
                "out_proj.weight",
                "out_proj.bias",
            ]
        }
        assert set(new) == expected, (layer, expected, set(new))
        assert all(torch.isfinite(v).all() for v in new.values()), layer
        state.update({key: value.cpu() for key, value in new.items()})
        mlp_prefix = f"model.visual.transformer.resblocks.{layer}.mlp."
        permutation = author["solve_permutation"](
            state[mlp_prefix + "c_fc.weight"], reference[mlp_prefix + "c_fc.weight"]
        )
        assert permutation is not None
        for key in ["c_fc.weight", "c_fc.bias"]:
            state[mlp_prefix + key] = state[mlp_prefix + key][permutation]
        state[mlp_prefix + "c_proj.weight"] = state[mlp_prefix + "c_proj.weight"][:, permutation]
        records.append({"layer": layer, "seconds": time.perf_counter() - t0})
        print("RECOVERED_LAYER", layer, records[-1]["seconds"], flush=True)
    recovered.load_state_dict(state, strict=True)
    (output / "clip_recovery_layers.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return recovered


@torch.no_grad()
def merge(anchor, left, right, coefficient: float):
    result = copy.deepcopy(anchor).cpu().eval()
    sa, sl, sr = anchor.state_dict(), left.state_dict(), right.state_dict()
    merged = {}
    for key, value in sa.items():
        merged[key] = (
            value + coefficient * (sl[key] - value + sr[key] - value)
            if torch.is_floating_point(value)
            else value.clone()
        )
    result.load_state_dict(merged, strict=True)
    return result


@torch.no_grad()
def evaluate(model, head, loader, device: str):
    model.to(device).eval()
    head.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    count, correct = 0, 0
    for images, labels in loader:
        features = model(images.to(device, non_blocking=True))
        logits = head(features)
        prediction = logits.argmax(-1).cpu()
        correct += int((prediction == labels).sum())
        count += labels.numel()
    torch.cuda.synchronize()
    result = dict(
        correct=correct,
        n=count,
        top1=correct / count,
        evaluation_seconds=time.perf_counter() - start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    model.cpu()
    head.cpu()
    return result


def run(args):
    root = Path(args.checkpoints)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    author, source_hash = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    load = checkpoint_loader(Path(args.author_repo))
    start = time.perf_counter()
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    heads = {name: load(str(root / f"head_{name}.pt")) for name in args.datasets}
    loaders = {}
    for name in args.datasets:
        dataset = (
            MNIST(args.data, train=False, download=True, transform=original.val_preprocess)
            if name == "MNIST"
            else SVHN(
                args.data,
                split="test",
                download=True,
                transform=original.val_preprocess,
            )
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    x, _ = next(iter(loaders["MNIST"]))
    x = x[:16].cuda()
    with torch.no_grad():
        y = original.cuda().eval()(x).cpu()
    original.cpu()
    t0 = time.perf_counter()
    protected = protect(original, args.seed)
    protection_seconds = time.perf_counter() - t0
    with torch.no_grad():
        yp = protected.cuda().eval()(x).cpu()
    protected.cpu()
    output_error = relative(yp, y)
    assert output_error < 1e-4, f"Protection is not function preserving: {output_error}"
    t0 = time.perf_counter()
    recovered = recover(protected, anchor, author, output)
    recovery_seconds = time.perf_counter() - t0
    with torch.no_grad():
        yr = recovered.cuda().eval()(x).cpu()
    recovered.cpu()
    recovery_output_error = relative(yr, y)
    protected_weight_error = state_error(protected.state_dict(), original.state_dict())
    recovery_weight_error = state_error(recovered.state_dict(), original.state_dict())
    torch.save(protected.state_dict(), artifacts / "mnist_protected_state.pt")
    torch.save(recovered.state_dict(), artifacts / "mnist_recovered_state.pt")
    rows = []

    def save():
        (output / "clip_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (output / "clip_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    groups = [
        ("pretrained", anchor),
        ("A_unprotected", original),
        ("B_protected", protected),
        ("C_AGA", recovered),
        ("second_task_unprotected", second),
    ]
    for group, model in groups:
        for dataset_name in args.datasets:
            metrics = evaluate(model, heads[dataset_name], loaders[dataset_name], args.device)
            row = dict(
                stage="P2",
                experiment="CLIP_ViT_B32",
                scope="real_full_test",
                group=group,
                dataset=dataset_name,
                seed=args.seed,
                coefficient=None,
                **metrics,
            )
            if group in ["B_protected", "C_AGA"]:
                row.update(
                    protected_output_error=output_error,
                    recovered_output_error=recovery_output_error,
                    protected_weight_error=protected_weight_error,
                    recovered_weight_error=recovery_weight_error,
                )
            rows.append(row)
            save()
            print("BENCHMARK", json.dumps(row), flush=True)
    for coefficient in args.coefficients:
        for group, left in [
            ("D_unprotected_merge", original),
            ("E_protected_merge", protected),
            ("F_AGA_merge", recovered),
        ]:
            model = merge(anchor, left, second, coefficient)
            for dataset_name in args.datasets:
                metrics = evaluate(model, heads[dataset_name], loaders[dataset_name], args.device)
                row = dict(
                    stage="P2",
                    experiment="CLIP_ViT_B32",
                    scope="real_full_test",
                    group=group,
                    dataset=dataset_name,
                    seed=args.seed,
                    coefficient=coefficient,
                    **metrics,
                )
                rows.append(row)
                save()
                print("BENCHMARK", json.dumps(row), flush=True)
            del model
    metadata = dict(
        seed=args.seed,
        checkpoints=str(root),
        datasets=args.datasets,
        coefficients=args.coefficients,
        batch_size=args.batch_size,
        author_sha256=source_hash,
        protection_seconds=protection_seconds,
        recovery_seconds=recovery_seconds,
        total_seconds=time.perf_counter() - start,
        protected_output_error=output_error,
        recovered_output_error=recovery_output_error,
        protected_weight_error=protected_weight_error,
        recovered_weight_error=recovery_weight_error,
        calibration="none",
        training="none",
        model="official_TaskArithmetic_ViT-B-32",
        feature_equivalence_samples=16,
        test_usage="evaluation_only",
        tf32=False,
    )
    (output / "clip_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("CLIP_BENCHMARK_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets", nargs="+", choices=["MNIST", "SVHN"], default=["MNIST", "SVHN"]
    )
    parser.add_argument("--coefficients", nargs="+", type=float, default=[0.2, 0.5])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261009)
    run(parser.parse_args())
