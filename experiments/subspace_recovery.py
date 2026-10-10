"""Check task-information cost of erasing a declared input subspace.

No protected model is constructed. The exact-direction preset has oracle access;
the covariance preset uses public calibration responses but is not validated on
a protected release. Both are mechanism diagnostics, not security successes.
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
from eight_task_benchmark import digest, merge_all
from fine_subspace import matrix_keys
from operator_diagnostics import paired_counts
from subspace_probe import make_loader, save_json


@torch.no_grad()
def projected_reset(original, anchor, projections):
    model = copy.deepcopy(original).cpu().eval()
    state, own, base = model.state_dict(), original.state_dict(), anchor.state_dict()
    for site, evidence in projections.items():
        weight, bias = matrix_keys(site)
        basis, mean = evidence["basis"].double(), evidence["mean"].double()
        delta = own[weight].double() - base[weight].double()
        removed = (delta @ basis) @ basis.T
        state[weight].copy_((own[weight].double() - removed).float())
        state[bias].copy_((own[bias].double() + removed @ mean).float())
    return model


@torch.no_grad()
def run(args):
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config = json.loads(Path(args.config).read_bytes())
    parent = Path(config["parent"])
    protocol = json.loads((parent / "protocol.json").read_bytes())
    parent_meta = json.loads((parent / "metadata.json").read_bytes())
    assert json.loads((parent / "validation.json").read_bytes())["status"] == "passed"
    assert config["task"] == protocol["task"] == "Cars"
    assert config["coefficient"] == protocol["coefficient"]
    for name in ("directions.pt", "activations.pt", "predictions.pt"):
        assert digest(parent / "artifacts" / name) == parent_meta["raw_sha256"][name]
    save_json(out / "protocol.json", config)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    original, anchor = load(str(root / "Cars/finetuned.pt")), load(str(root / "zeroshot.pt"))
    directions = torch.load(
        parent / "artifacts/directions.pt", map_location="cpu", weights_only=True
    )
    activations = torch.load(
        parent / "artifacts/activations.pt", map_location="cpu", weights_only=True
    )
    selected = json.loads((parent / "decision.json").read_bytes())["eligible_sites"]
    oracle, public, records = {}, {}, []
    for site in selected:
        weight, _ = matrix_keys(site)
        direction = directions[site]
        mean, v = direction["mean"].double(), direction["incoming"].double()
        x = activations["calibration", "Cars"][site].double().cuda() - mean.cuda()
        width = x.shape[1]
        if site.startswith("down:"):
            record = next(
                r
                for r in json.loads((parent / "directions.json").read_bytes())
                if r["site"] == site
            )
            support = torch.eye(width, dtype=torch.float64)[:, record["neurons"]].cuda()
        else:
            support = torch.eye(width, dtype=torch.float64, device="cuda")
        z = x @ support
        covariance = z.T @ z / len(z)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0)
        fraction = eigenvalues.flip(0).cumsum(0) / eigenvalues.sum()
        rank = int(torch.searchsorted(fraction, config["retained_input_covariance_energy"])) + 1
        basis = (support @ eigenvectors[:, : len(eigenvalues) - rank]).cpu()
        oracle[site] = dict(basis=v[:, None], mean=mean)
        public[site] = dict(basis=basis, mean=mean)
        delta = original.state_dict()[weight].double() - anchor.state_dict()[weight].double()
        task_energy = float(delta.square().sum())
        removed = delta @ v
        contrast_power = float((x @ v.cuda()).square().mean())
        records.append(
            dict(
                site=site,
                oracle_task_weight_energy_fraction=float(removed.square().sum()) / task_energy,
                oracle_local_output_rms=float(removed.norm()) * contrast_power**0.5,
                public_retained_rank=rank,
                public_tail_dimension=basis.shape[1],
                public_retained_input_energy=float(fraction[rank - 1]),
                public_task_weight_energy_fraction=float((delta @ basis).square().sum())
                / task_energy,
                public_noise_energy_removed_fraction=float((basis.T @ v).square().sum()),
            )
        )
        print("PROJECTION", records[-1], flush=True)
        del x, z, covariance, support
    del activations
    projections = dict(direction_oracle=oracle, public_covariance=public)
    projection_file = artifacts / "projections.pt"
    torch.save(projections, projection_file)
    save_json(out / "geometry.json", records)
    save_json(
        out / "projection_freeze.json",
        dict(
            projections_sha256=digest(projection_file),
            selected_sites=selected,
            future_partner_checkpoints_loaded=False,
            public_projection_uses_defender_vectors=False,
            labels_used=False,
        ),
    )
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    models = {name: load(str(root / name / "finetuned.pt")) for name in fusion["tasks"]}
    models.update(Cars=original, anchor=anchor)
    head = load(str(root / "head_Cars.pt"))
    parent_raw = torch.load(
        parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    indices = json.loads(
        (Path(protocol["replacement_parent"]) / "training_indices.json").read_bytes()
    )["Cars"]["confirmation"]
    save_json(out / "training_indices.json", indices)
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset
    from torch.utils.data import DataLoader, TensorDataset

    data = get_dataset("Cars", original.val_preprocess, args.data).train_dataset
    batches = list(make_loader(data, indices))
    loader = DataLoader(
        TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
        batch_size=128,
        pin_memory=True,
    )
    rows, raw = [], {}
    for preset in config["presets"]:
        assert time.perf_counter() - started < config["time_limit_seconds"]
        own = (
            original
            if preset == "original"
            else projected_reset(original, anchor, projections[preset])
        )
        combined = merge_all(dict(models, Cars=own), root, config["coefficient"])
        for group, model in [("standalone", own), ("fusion", combined)]:
            predictions, labels = [], []
            score = evaluate(
                model,
                head,
                loader,
                "cuda",
                prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
            )
            p, y = torch.cat(predictions), torch.cat(labels)
            reference = parent_raw[f"replacement_preflight/original/{group}/Cars"]
            assert torch.equal(y, reference["labels"])
            if preset == "original":
                assert torch.equal(p, reference["prediction"])
            raw[f"{preset}/{group}"] = dict(prediction=p, labels=y)
            rows.append(
                dict(
                    id=f"{preset}/{group}",
                    preset=preset,
                    group=group,
                    dataset="Cars",
                    paired=paired_counts(reference["prediction"].eq(y), p.eq(y)),
                    **score,
                )
            )
            print("RECOVERY_COST", rows[-1]["id"], score["correct"], score["n"], flush=True)
        del combined
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            parent_metadata_sha256=digest(parent / "metadata.json"),
            parent_raw_sha256=parent_meta["raw_sha256"],
            checkpoint_sha256=parent_meta["checkpoint_sha256"],
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "fine_subspace.py",
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            elapsed_seconds=time.perf_counter() - started,
            optimizer_steps=0,
            response_fits=0,
            protected_model_evaluated=False,
            test_usage="none",
            oracle_access_explicit=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/subspace_recovery_cars.json")
    parser.add_argument("--output", default="outputs/2026-10-11/subspace_recovery_cars")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
