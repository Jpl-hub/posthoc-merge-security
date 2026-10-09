"""Explore data-free LayerNorm-null parameter changes and a canonicalization attack.

For y = gamma * z + beta, sum(z)=0, r=1/gamma satisfies r^T(y-beta)=0.
Thus W'=W+u r^T, b'=b-u(r^T beta) changes a following Linear's parameters
without changing its values on LayerNorm outputs. No fitting or training.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge, protect, recover
from mechanisms import load_author, relative, state_error
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


@torch.no_grad()
def null_change(model, strength: float, seed: int):
    torch.manual_seed(seed)
    result = copy.deepcopy(model).cpu().eval()
    for block in result.model.visual.transformer.resblocks:
        pairs = [
            (block.ln_1, block.attn.in_proj_weight, block.attn.in_proj_bias),
            (block.ln_2, block.mlp.c_fc.weight, block.mlp.c_fc.bias),
        ]
        for normalization, weight, bias in pairs:
            gamma = normalization.weight.double()
            assert float(gamma.abs().min()) > 1e-8
            r = gamma.reciprocal()
            r /= r.norm()
            u = torch.randn(weight.shape[0], dtype=torch.float64)
            u *= strength * weight.double().norm() / u.norm()
            shift = u[:, None] * r[None, :]
            weight.add_(shift.float())
            bias.sub_((u * (r @ normalization.bias.double())).float())
    return result


@torch.no_grad()
def canonicalize(model):
    """Erase the known affine-null direction; preserves every valid LN input."""
    result = copy.deepcopy(model).cpu().eval()
    for block in result.model.visual.transformer.resblocks:
        pairs = [
            (block.ln_1, block.attn.in_proj_weight, block.attn.in_proj_bias),
            (block.ln_2, block.mlp.c_fc.weight, block.mlp.c_fc.bias),
        ]
        for normalization, weight, bias in pairs:
            r = normalization.weight.double().reciprocal()
            r /= r.norm()
            component = weight.double() @ r
            weight.sub_((component[:, None] * r[None, :]).float())
            bias.add_((component * (r @ normalization.bias.double())).float())
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    author, source_hash = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    head = load(str(root / "head_MNIST.pt"))
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
    overall_start = time.perf_counter()
    baseline = protect(original, args.seed)
    canonical_anchor = canonicalize(anchor)
    canonical_second = canonicalize(second)
    canonical_original = canonicalize(original)
    for strength in args.strengths:
        candidate = null_change(baseline, strength, args.seed + 100)
        null_original = null_change(original, strength, args.seed + 100)
        protected_error = relative(features(candidate), y)
        assert protected_error < 1e-4, (strength, protected_error)
        sub = output / f"strength_{strength:g}"
        sub.mkdir(exist_ok=True)
        recovered = recover(candidate, anchor, author, sub)
        canonical_candidate = canonicalize(candidate)
        canonical_error = relative(features(canonical_candidate), y)
        assert canonical_error < 1e-4, (strength, canonical_error)
        adaptive = recover(canonical_candidate, canonical_anchor, author, sub)
        common = dict(
            stage="exploration",
            experiment="layernorm_affine_null",
            scope="real_MNIST_full_test",
            seed=args.seed,
            strength=strength,
            n=10000,
            protected_output_error=protected_error,
            canonical_output_error=canonical_error,
            author_sha256=source_hash,
        )
        groups = [
            ("LN_null_only", null_original),
            ("LN_null_plus_basis", candidate),
            ("LN_null_AGA", recovered),
            ("LN_null_canonical_AGA", adaptive),
        ]
        for group, model in groups:
            row = dict(
                common,
                group=group,
                coefficient=None,
                **evaluate(model, head, loader, "cuda"),
                feature_error=relative(features(model), y),
                weight_error=state_error(model.state_dict(), original.state_dict()),
            )
            rows.append(row)
            (output / "layernorm_results.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            print("LN_BENCHMARK", json.dumps(row), flush=True)
        for group, model, base, other in [
            ("LN_null_direct_merge", candidate, anchor, second),
            ("LN_null_AGA_merge", recovered, anchor, second),
            ("LN_null_canonical_AGA_merge", adaptive, canonical_anchor, canonical_second),
            ("canonical_unprotected_merge", canonical_original, canonical_anchor, canonical_second),
        ]:
            merged = merge(base, model, other, 0.5)
            row = dict(
                common, group=group, coefficient=0.5, **evaluate(merged, head, loader, "cuda")
            )
            rows.append(row)
            (output / "layernorm_results.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            print("LN_BENCHMARK", json.dumps(row), flush=True)
            del merged
    metadata = dict(
        total_seconds=time.perf_counter() - overall_start,
        strengths=args.strengths,
        training="none",
        calibration="none",
        note="Exploratory candidate with explicit adaptive canonicalization control",
    )
    (output / "layernorm_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("LN_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/layernorm")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--seed", type=int, default=20261009)
    run(parser.parse_args())
