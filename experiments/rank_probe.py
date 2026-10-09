"""Test irreversible Attention rank loss and pretrained completion; no training.

Rank truncation is a known compression operator, not a claimed new algorithm.
This probe tests whether changing AGA's subspace invariant protects merging.
All presets are fixed in a config before any test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge, recover
from coupled_alignment import align, load_alignment
from low_activity_posthoc import synthetic_images
from mechanisms import load_author
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def activation_moments(model, images, layers):
    """Second moments of affine V outputs, including the existing bias."""
    captured = {i: [] for i in layers}
    handles = []
    for i in layers:
        normalization = model.model.visual.transformer.resblocks[i].ln_1
        handles.append(
            normalization.register_forward_hook(
                lambda _m, _a, value, layer=i: captured[layer].append(
                    value.detach().reshape(-1, value.shape[-1]).cpu()
                )
            )
        )
    model.cuda().eval()
    for batch in images.split(8):
        model(batch.cuda())
    model.cpu()
    for handle in handles:
        handle.remove()
    moments = {}
    for i in layers:
        x = torch.cat(captured[i]).double().cuda()
        module = model.model.visual.transformer.resblocks[i].attn
        w = module.in_proj_weight[1536:].double().cuda()
        b = module.in_proj_bias[1536:].double().cuda()
        y = (x @ w.T + b).reshape(-1, 12, 64)
        moments[i] = [y[:, h].T @ y[:, h] / len(y) for h in range(12)]
    return moments


@torch.no_grad()
def truncate(model, moments, preset, layers, seed):
    result = copy.deepcopy(model).cpu().eval()
    generator = torch.Generator().manual_seed(seed)
    records = []
    for i in layers:
        module = result.model.visual.transformer.resblocks[i].attn
        for h in range(12):
            s = slice(1536 + 64 * h, 1536 + 64 * (h + 1))
            w, b = module.in_proj_weight[s], module.in_proj_bias[s]
            augmented = torch.cat([w.double(), b.double()[:, None]], dim=1).cuda()
            if preset["method"] == "dropout":
                mask = torch.rand(w.shape, generator=generator) >= preset["drop_rate"]
                w.mul_(mask)
                records.append(
                    dict(layer=i, head=h, removed_fraction=float((~mask).float().mean()))
                )
                continue
            moment = moments[i][h] if preset["method"] == "activation" else augmented @ augmented.T
            values, vectors = torch.linalg.eigh(moment)
            u = vectors[:, -preset["rank"] :]
            p = u @ u.T
            reduced = (p @ augmented).cpu()
            w.copy_(reduced[:, :-1])
            b.copy_(reduced[:, -1])
            sv = torch.linalg.svdvals(reduced.cuda())
            effective_rank = int((sv > sv[0] * 1e-5).sum())
            assert effective_rank <= preset["rank"], (i, h, effective_rank)
            records.append(
                dict(
                    layer=i,
                    head=h,
                    effective_rank=effective_rank,
                    removed_moment_fraction=float(values[: -preset["rank"]].sum() / values.sum()),
                )
            )
    return result, records


@torch.no_grad()
def complete(model, anchor, layers):
    """W0 attack: infer the output span and fill its complement from pretrained.

    Uses released weights only; never receives the defender's projection or seed.
    The same numerical rule is applied to the unprotected control.
    """
    result = copy.deepcopy(model).cpu().eval()
    for i in layers:
        local = result.model.visual.transformer.resblocks[i].attn
        reference = anchor.model.visual.transformer.resblocks[i].attn
        for h in range(12):
            s = slice(1536 + 64 * h, 1536 + 64 * (h + 1))
            a = torch.cat(
                [local.in_proj_weight[s].double(), local.in_proj_bias[s].double()[:, None]], dim=1
            ).cuda()
            base = torch.cat(
                [reference.in_proj_weight[s].double(), reference.in_proj_bias[s].double()[:, None]],
                dim=1,
            ).cuda()
            u, values, _ = torch.linalg.svd(a, full_matrices=False)
            active = u[:, values > values[0] * 1e-5]
            filled = (a + base - active @ (active.T @ base)).cpu()
            local.in_proj_weight[s].copy_(filled[:, :-1])
            local.in_proj_bias[s].copy_(filled[:, -1])
    return result


def run(args):
    output = Path(args.output)
    # An existing run must never be silently replaced.
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    (output / "protocol.json").write_bytes(config_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    functions, paired_sha = load_alignment(Path(args.paired_source))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    heads = {n: load(str(root / f"head_{n}.pt")) for n in ["MNIST", "SVHN"]}
    loaders = {
        "MNIST": DataLoader(
            MNIST(args.data, train=False, transform=original.val_preprocess),
            batch_size=128,
            num_workers=2,
            pin_memory=True,
        ),
        "SVHN": DataLoader(
            SVHN(args.data, split="test", transform=original.val_preprocess),
            batch_size=128,
            num_workers=2,
            pin_memory=True,
        ),
    }
    images = synthetic_images(
        original.val_preprocess, config["calibration_samples"], config["seed"]
    )
    moments = activation_moments(original, images, config["layers"])
    rows = []

    def record(preset, group, model):
        for dataset, loader in loaders.items():
            row = dict(
                experiment="attention_rank_loss",
                preset=preset["name"],
                method=preset["method"],
                rank=preset.get("rank"),
                drop_rate=preset.get("drop_rate"),
                group=group,
                dataset=dataset,
                seed=config["seed"],
                coefficient=None if group == "standalone" else config["coefficient"],
                calibration="procedural_unlabelled_strokes"
                if preset["method"] == "activation"
                else "none",
                calibration_samples=config["calibration_samples"]
                if preset["method"] == "activation"
                else 0,
                training="none",
                scope="real_full_test",
                **evaluate(model, heads[dataset], loader, "cuda"),
            )
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print("RANK_RESULT", json.dumps(row), flush=True)

    control = dict(name="unprotected", method="none")
    record(control, "standalone", original)
    aga_dir = output / "control_AGA"
    aga_dir.mkdir()
    au = recover(original, anchor, author, aga_dir)
    pu = align(original, anchor, functions)
    cu = complete(original, anchor, config["layers"])
    for group, left in [("direct", original), ("AGA", au), ("paired", pu), ("completion", cu)]:
        merged = merge(anchor, left, second, config["coefficient"])
        record(control, group, merged)
        del merged
    del au, pu, cu
    for preset in config["presets"]:
        candidate, details = truncate(original, moments, preset, config["layers"], config["seed"])
        run_dir = output / preset["name"]
        run_dir.mkdir()
        (run_dir / "compression.json").write_text(json.dumps(details, indent=2))
        record(preset, "standalone", candidate)
        direct = merge(anchor, candidate, second, config["coefficient"])
        record(preset, "direct", direct)
        del direct
        attack_dir = run_dir / "AGA"
        attack_dir.mkdir()
        recovered = recover(candidate, anchor, author, attack_dir)
        paired = align(candidate, anchor, functions)
        completed = complete(candidate, anchor, config["layers"])
        for group, left in [("AGA", recovered), ("paired", paired), ("completion", completed)]:
            merged = merge(anchor, left, second, config["coefficient"])
            record(preset, group, merged)
            del merged
        del candidate, recovered, paired, completed
    metadata = dict(
        completed=True,
        total_seconds=time.perf_counter() - start,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        author_sha256=author_sha,
        paired_source_sha256=paired_sha,
        training="none",
        calibration=config["calibration"],
        utility_budget_pp=config["utility_budget_pp"],
        selection=config["selection"],
        attacks=["direct", "author_AGA", "original_paired_alignment", "pretrained_completion"],
        completion_visibility="whitebox_without_data_or_defender_projection",
        merge_implementation="author_TaskVector_difference_and_sum",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("RANK_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rank_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-09/rank_probe")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument(
        "--paired-source", default="external/RotationSymmetry/vit-fusion/otfusion/pi_utils.py"
    )
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
