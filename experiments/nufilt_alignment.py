"""Check NUFILT's task-vector proxy against actual expert activation moments.

Reuse the pinned author's SVD and existing checkpoint/activation collectors.
This is a geometric diagnostic, not a NUFILT reproduction or a new defense.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Tuple

import torch
from clip_benchmark import checkpoint_loader
from subspace_probe import collect_moments, make_loader, save_json, site_parameters
from torchvision.datasets import MNIST, SVHN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_svd(repo, config):
    """Load the two unchanged pure functions without importing FusionBench."""
    repo = Path(repo)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert commit == config["upstream_commit"]
    source = repo / "fusion_bench/method/nufilt/utils.py"
    assert digest(source) == config["upstream_svd_sha256"]
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_svd", "svd"}]
    assert len(tree.body) == 2
    namespace = dict(torch=torch, Tensor=torch.Tensor, Tuple=Tuple)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["svd"]


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    (out / "protocol.json").write_bytes(Path(args.config).read_bytes())
    parent = Path(config["parent"])
    index_path = parent / "training_indices.json"
    frozen = json.loads(index_path.read_bytes())
    assert config["test_usage"] == "none"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    svd = author_svd(args.nufilt_repo, config)
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    sites = [f"{target}:{layer}" for layer in range(config["layers"]) for target in config["targets"]]
    rows, captured = [], {}
    start = time.perf_counter()
    for task in config["tasks"]:
        expert = load(str(root / task / "finetuned.pt")).cpu().eval()
        dataset = (
            MNIST(args.data, train=True, transform=expert.val_preprocess)
            if task == "MNIST"
            else SVHN(args.data, split="train", transform=expert.val_preprocess)
        )
        indices = frozen[task]["calibration"]
        assert len(indices) == len(set(indices)) == config["calibration_samples"]
        moments = collect_moments(expert, make_loader(dataset, indices, batch_size=32), sites)
        captured[task] = moments
        for site in sites:
            weight, _ = site_parameters(expert, site)
            base, _ = site_parameters(anchor, site)
            delta = (weight.double() - base.double()).cuda()
            left, singular, vectors = svd(delta, full_matrices=False)
            reconstruction_error = float(((left * singular) @ vectors.T - delta).norm() / delta.norm())
            assert reconstruction_error < 1e-10
            for view in config["moments"]:
                moment = moments[site]
                matrix = moment["covariance"].cuda()
                if view == "uncentered":
                    mean = moment["mean"].cuda()
                    matrix = matrix + mean[:, None] @ mean[None, :]
                values, basis = torch.linalg.eigh(matrix)
                assert float(values.min()) >= -1e-10 * float(values.max())
                values, basis = values.flip(0).clamp_min(0), basis.flip(1)
                fractions = values.cumsum(0) / values.sum()
                trace = matrix.trace()
                stats = dict(
                    task=task, site=site, moment=view, tokens=moment["tokens"],
                    data_ranks_at_energy={str(e): int(torch.searchsorted(fractions, e)) + 1 for e in [0.99, 0.9999]},
                    data_energy_top16=float(fractions[15]),
                    task_delta_norm=float(delta.norm()),
                    svd_relative_reconstruction_error=reconstruction_error,
                )
                for rank in config["task_vector_ranks"]:
                    proxy = vectors[:, :rank]
                    orthogonality_error = float((proxy.T @ proxy - torch.eye(rank, device="cuda", dtype=proxy.dtype)).norm())
                    assert orthogonality_error < 1e-10
                    outside = float((trace - (proxy.T @ matrix @ proxy).trace()) / trace)
                    assert -1e-10 <= outside <= 1 + 1e-10
                    affinities = {
                        str(r): float((proxy.T @ basis[:, :r]).square().sum() / r)
                        for r in config["data_ranks"]
                    }
                    assert all(r <= rank for r in config["data_ranks"])
                    rows.append(dict(
                        **stats, task_vector_rank=rank,
                        affinity_by_data_rank=affinities,
                        activation_energy_outside_proxy_fraction=outside,
                        task_vector_energy_retained=float(singular[:rank].square().sum() / singular.square().sum()),
                        proxy_orthogonality_error=orthogonality_error,
                    ))
            print(json.dumps(dict(task=task, site=site, completed_rows=len(rows))), flush=True)
        del expert
    torch.save(captured, artifacts / "activation_moments.pt")
    provenance = dict(
        source_sha256=digest(__file__), config_sha256=digest(args.config),
        collector_sha256=digest(Path(__file__).with_name("subspace_probe.py")),
        checkpoint_loader_sha256=digest(Path(__file__).with_name("clip_benchmark.py")),
        upstream_commit=config["upstream_commit"], upstream_svd_sha256=config["upstream_svd_sha256"],
        parent_indices_sha256=digest(index_path),
        checkpoint_sha256={str(path.relative_to(root)): digest(path) for path in [root / "zeroshot.pt", *[root / task / "finetuned.pt" for task in config["tasks"]]]},
        activation_moments_sha256=digest(artifacts / "activation_moments.pt"),
    )
    result = dict(protocol=config, provenance=provenance, seconds=time.perf_counter() - start, rows=rows,
                  accuracy_evaluations=0, model_edits=0, optimization_steps=0)
    save_json(out / "results.json", result)
    expected = len(config["tasks"]) * len(sites) * len(config["moments"]) * len(config["task_vector_ranks"])
    assert len(rows) == expected
    assert all(row["tokens"] == config["calibration_samples"] * 50 for row in rows)
    assert all(0 <= a <= 1 + 1e-10 for row in rows for a in row["affinity_by_data_rank"].values())
    assert all(0 <= row["task_vector_energy_retained"] <= 1 + 1e-10 for row in rows)
    save_json(out / "validation.json", dict(
        passed=True, expected_rows=expected, actual_rows=len(rows),
        checks=["pinned author commit and source hash", "SVD reconstruction and orthogonality", "positive semidefinite moments", "bounded energy and affinity", "frozen calibration sample and token counts", "complete geometric grid"],
        results_sha256=digest(out / "results.json"), provenance=provenance,
        accuracy_evaluations=0, official_test_usage=False,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ["config", "output", "author-repo", "nufilt-repo", "checkpoints", "data"]:
        parser.add_argument(f"--{argument}", required=True)
    run(parser.parse_args())
