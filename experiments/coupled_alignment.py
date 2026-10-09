"""Data-free paired Attention/FFN matching using pristine RotationSymmetry code."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import checkpoint_loader, evaluate, merge
from mechanisms import relative
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN


def load_alignment(path: Path):
    source = path.read_bytes()
    expected = json.loads(Path("configs/sources.json").read_text())["RotationSymmetry"]["sha256"]
    digest = hashlib.sha256(source).hexdigest()
    assert digest == expected, (digest, expected)
    tree = ast.parse(source.decode("utf-8"))
    names = {"Chunk_SVD_QK", "Chunk_SVD_VO", "Permute_IO"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in tree.body} == names
    env = {"torch": torch, "np": np, "linear_sum_assignment": linear_sum_assignment}
    exec(compile(tree, str(path), "exec"), env)
    return env, digest


@torch.no_grad()
def align(model, anchor, functions):
    aligned = copy.deepcopy(model).cpu().eval()
    for block, reference in zip(
        aligned.model.visual.transformer.resblocks,
        anchor.model.visual.transformer.resblocks,
        strict=True,
    ):

        def attention_params(b):
            q, k, v = b.attn.in_proj_weight.detach().double().cuda().chunk(3)
            bq, bk, bv = b.attn.in_proj_bias.detach().double().cuda().chunk(3)
            return {
                "W_Q": q,
                "W_K": k,
                "W_V": v,
                "B_Q": bq,
                "B_K": bk,
                "B_V": bv,
                "W_O": b.attn.out_proj.weight.detach().double().cuda(),
                "B_O": b.attn.out_proj.bias.detach().double().cuda(),
            }

        local, target = attention_params(block), attention_params(reference)
        qk = functions["Chunk_SVD_QK"](
            local, target, num_attention_heads=12, attention_head_size=64, use_scaling=False
        )
        vo = functions["Chunk_SVD_VO"](
            local, target, num_attention_heads=12, attention_head_size=64, use_scaling=False
        )
        block.attn.in_proj_weight.copy_(torch.cat([qk["W_Q"], qk["W_K"], vo["W_V"]]).cpu())
        block.attn.in_proj_bias.copy_(torch.cat([qk["B_Q"], qk["B_K"], vo["B_V"]]).cpu())
        block.attn.out_proj.weight.copy_(vo["W_O"].cpu())
        block.attn.out_proj.bias.copy_(vo["B_O"].cpu())

        def mlp_params(b):
            return {
                "W_I": b.mlp.c_fc.weight.detach().double().cuda(),
                "B_I": b.mlp.c_fc.bias.detach().double().cuda(),
                "W_O_FFN": b.mlp.c_proj.weight.detach().double().cuda(),
                "B_O_FFN": b.mlp.c_proj.bias.detach().double().cuda(),
            }

        ff = functions["Permute_IO"](mlp_params(block), mlp_params(reference))
        block.mlp.c_fc.weight.copy_(ff["W_I"].cpu())
        block.mlp.c_fc.bias.copy_(ff["B_I"].cpu())
        block.mlp.c_proj.weight.copy_(ff["W_O_FFN"].cpu())
        block.mlp.c_proj.bias.copy_(ff["B_O_FFN"].cpu())
    return aligned


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    functions, digest = load_alignment(Path(args.source))
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    protected = copy.deepcopy(original).cpu().eval()
    protected.load_state_dict(torch.load(args.protected_state, map_location="cpu"), strict=True)
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
    heads = {name: load(str(root / f"head_{name}.pt")) for name in loaders}
    alignment_start = time.perf_counter()
    ap, au, av = [align(model, anchor, functions) for model in [protected, original, second]]
    alignment_seconds = time.perf_counter() - alignment_start
    x = next(iter(loaders["MNIST"]))[0][:16].cuda()

    @torch.no_grad()
    def features(model):
        values = model.cuda().eval()(x).cpu()
        model.cpu()
        return values

    feature_errors = {
        name: relative(features(after), features(before))
        for name, before, after in [
            ("protected", protected, ap),
            ("unprotected", original, au),
            ("second", second, av),
        ]
    }
    assert max(feature_errors.values()) < 1e-4, feature_errors
    rows = []
    groups = [
        ("paired_protected_standalone", lambda: ap),
        ("paired_unprotected_standalone", lambda: au),
        ("paired_protected_only_merge", lambda: merge(anchor, ap, second, args.coefficient)),
        ("paired_unprotected_only_merge", lambda: merge(anchor, au, second, args.coefficient)),
        ("paired_all_protected_merge", lambda: merge(anchor, ap, av, args.coefficient)),
        ("paired_all_unprotected_merge", lambda: merge(anchor, au, av, args.coefficient)),
    ]
    for name, make_model in groups:
        model = make_model()
        for dataset, loader in loaders.items():
            row = {
                "stage": "P2",
                "experiment": "coupled_alignment",
                "scope": "real_full_test",
                "group": name,
                "dataset": dataset,
                "seed": 20261009,
                "coefficient": None if "standalone" in name else args.coefficient,
                "protection": "basis",
                "training": "none",
                "calibration": "none",
                **evaluate(model, heads[dataset], loader, "cuda"),
            }
            rows.append(row)
            (output / "coupled_results.json").write_text(json.dumps(rows, indent=2))
            print("PAIRED_EVAL", json.dumps(row), flush=True)
        del model
    metadata = {
        "source_sha256": digest,
        "source": args.source,
        "functions": ["Chunk_SVD_QK", "Chunk_SVD_VO", "Permute_IO"],
        "solver_dtype": "float64",
        "model_dtype": "float32",
        "feature_errors": feature_errors,
        "alignment_seconds": alignment_seconds,
        "total_seconds": time.perf_counter() - start,
        "coefficient": args.coefficient,
        "attention_heads": 12,
        "head_size": 64,
        "scaling": False,
        "head_matching": False,
        "protection": "basis",
        "data": "none for alignment; official full tests for evaluation only",
        "training": "none",
        "merge_implementation": "author_TaskVector_difference_and_sum",
    }
    (output / "coupled_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("PAIRED_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="external/RotationSymmetry/vit-fusion/otfusion/pi_utils.py"
    )
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    parser.add_argument(
        "--protected-state",
        default="outputs/2026-10-09/artifacts/mnist_protected_state.pt",
    )
    parser.add_argument("--output", default="outputs/2026-10-09/coupled")
    parser.add_argument("--coefficient", type=float, default=0.5)
    run(parser.parse_args())
