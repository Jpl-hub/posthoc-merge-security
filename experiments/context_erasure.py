"""Test task-delta deletion selected by expert versus merged input covariance.

This is a pair-conditioned mechanism probe, not a new defense claim. Projection
loses components conditional on its selected basis; that is not a security proof.
Reuse the existing author merger/evaluator and functional reconstruction attack.
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
from operator_diagnostics import adjust_comparisons, paired_counts
from polynomial_probe import capture, reconstruct
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST, SVHN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def directions(own, merged, ridge_ratio):
    """Generalized variance ratio, then Euclidean orthonormalization for deletion."""
    a, b = own.T @ own / len(own), merged.T @ merged / len(merged)
    ridge = ridge_ratio * a.trace() / len(a)
    values, vectors = torch.linalg.eigh(a)
    whitening = (vectors * (values.clamp_min(0) + ridge).rsqrt()) @ vectors.T
    contrast = whitening @ b @ whitening
    ratios, basis = torch.linalg.eigh((contrast + contrast.T) / 2)
    return dict(
        own=vectors,
        contrast=whitening @ basis.flip(1),
        ratios=ratios.flip(0),
        ridge=float(ridge),
    )


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    (out / "protocol.json").write_bytes(config_bytes)
    index_path = Path(config["parent"]) / "training_indices.json"
    indices = json.loads(index_path.read_bytes())
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "SVHN/finetuned.pt"))
    second = load(str(root / "MNIST/finetuned.pt"))
    coefficient = config["coefficient"]
    baseline_merge = merge(anchor, original, second, coefficient)
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    datasets = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    loaders = {
        task: make_loader(dataset, indices[task]["development"])
        for task, dataset in datasets.items()
    }
    cal = indices["SVHN"]["calibration"][: config["samples"]]
    assert not set(cal).intersection(indices["SVHN"]["development"])
    assert not set(cal).intersection(indices["SVHN"]["attack"])
    images = torch.stack([datasets["SVHN"][i][0] for i in cal])
    rows, predictions, controls, bases, hidden, edit_records = [], {}, {}, {}, {}, []

    def record(name, group, model):
        scores = {}
        for task, loader in loaders.items():
            if group == "standalone" and task != "SVHN":
                continue
            chunks, labels = [], []

            def sink(prediction, target):
                chunks.append(prediction)
                labels.append(target)

            result = evaluate(model, heads[task], loader, "cuda", prediction_sink=sink)
            pred, target = torch.cat(chunks), torch.cat(labels)
            correct = pred == target
            if name == "unprotected":
                controls[group, task] = correct
            result.update(
                preset=name,
                group=group,
                dataset=task,
                split="development",
                paired=paired_counts(
                    controls["standalone" if group == "standalone" else "direct", task], correct
                ),
            )
            rows.append(result)
            predictions[name, group, task] = dict(prediction=pred, labels=target)
            scores[task] = result["top1"]
            save_json(out / "results.json", rows)
            print("ERASURE_RESULT", name, group, task, result["correct"], result["n"], flush=True)
        return scores

    standalone_base = record("unprotected", "standalone", original)["SVHN"]
    merge_base = record("unprotected", "direct", baseline_merge)["SVHN"]
    for layer in config["layers"]:
        mlp = copy.deepcopy(original.model.visual.transformer.resblocks[layer].mlp).double().cuda()
        own = mlp.gelu(mlp.c_fc(capture(original, images, layer)))
        mixed = mlp.gelu(mlp.c_fc(capture(baseline_merge, images, layer)))
        hidden[layer] = own.cpu(), mixed.cpu()
        bases[layer] = directions(own, mixed, config["ridge_ratio"])
        print("ERASURE_BASIS", layer, float(bases[layer]["ratios"][0]), flush=True)
        del mlp, own, mixed
    torch.save(hidden, artifacts / "hidden_inputs.pt")
    torch.save(bases, artifacts / "directions.pt")
    gates = []
    author = None
    for preset in config["presets"]:
        candidate = copy.deepcopy(original).cpu().eval()
        name, rank = preset["name"], preset["rank"]
        for layer in config["layers"]:
            basis = bases[layer][preset["rule"]][:, :rank]
            u = torch.linalg.qr(basis, mode="reduced").Q
            assert torch.allclose(u.T @ u, torch.eye(rank, device="cuda", dtype=u.dtype), atol=1e-8)
            original_w = (
                original.model.visual.transformer.resblocks[layer].mlp.c_proj.weight.double().cuda()
            )
            anchor_w = (
                anchor.model.visual.transformer.resblocks[layer].mlp.c_proj.weight.double().cuda()
            )
            erased = ((original_w - anchor_w) @ u) @ u.T
            weight = candidate.model.visual.transformer.resblocks[layer].mlp.c_proj.weight
            weight.copy_((original_w - erased).float().cpu())
            residual = ((weight.double().cuda() - anchor_w) @ u).norm()
            assert float(residual / (original_w - anchor_w).norm().clamp_min(1e-12)) < 1e-5
            response = {}
            for context, h in zip(["expert", "merged"], hidden[layer]):
                h = h.cuda()
                response[context] = float(
                    (h @ erased.T).norm() / (h @ original_w.T).norm().clamp_min(1e-12)
                )
            edit_records.append(
                dict(
                    preset=name,
                    layer=layer,
                    rank=rank,
                    response=response,
                    erased_norm=float(erased.norm()),
                    projection_residual=float(residual),
                )
            )
        torch.save(
            {
                layer: candidate.model.visual.transformer.resblocks[layer].mlp.state_dict()
                for layer in config["layers"]
            },
            artifacts / f"{name}.pt",
        )
        own_score = record(name, "standalone", candidate)["SVHN"]
        direct = record(name, "direct", merge(anchor, candidate, second, coefficient))["SVHN"]
        passed = (standalone_base - own_score) * 100 <= config["utility_budget_pp"] and (
            merge_base - direct
        ) * 100 >= config["minimum_protected_task_drop_pp"]
        gates.append(
            dict(
                preset=name,
                utility_loss_pp=100 * (standalone_base - own_score),
                direct_loss_pp=100 * (merge_base - direct),
                passed=passed,
            )
        )
        if not passed:
            continue
        cleaned, audit = angular_cleanup(candidate, anchor, 0.9)
        save_json(out / f"{name}_cleanup.json", audit)
        record(name, "cleanup", merge(anchor, cleaned, second, coefficient))
        if author is None:
            author, _ = load_author(Path(args.author_repo) / "utils_attack_B32.py")
        recovery_out = out / f"{name}_aga"
        recovery_out.mkdir()
        restored = recover(candidate, anchor, author, recovery_out)
        record(name, "aga", merge(anchor, restored, second, coefficient))
        attack_images = torch.stack(
            [datasets["SVHN"][i][0] for i in indices["SVHN"]["attack"][:256]]
        )
        restored = copy.deepcopy(candidate)
        for step in range(2):
            context = candidate if step == 0 else merge(anchor, restored, second, coefficient)
            updated = copy.deepcopy(restored)
            fitting = []
            for layer in config["layers"]:
                local, evidence = reconstruct(
                    candidate,
                    anchor,
                    attack_images,
                    layer,
                    1e-4,
                    solver="auto",
                    input_model=context,
                )
                updated.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                    local.model.visual.transformer.resblocks[layer].mlp.state_dict()
                )
                fitting.append(dict(layer=layer, **evidence))
            restored = updated
            save_json(out / f"{name}_reconstruction_{step + 1}.json", fitting)
            record(name, f"reconstruction_{step + 1}", merge(anchor, restored, second, coefficient))
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    save_json(out / "edits.json", edit_records)
    save_json(out / "gates.json", gates)
    torch.save(predictions, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            gpu=torch.cuda.get_device_name(),
            source_sha256=digest(__file__),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            indices_sha256=digest(index_path),
            checkpoint_sha256={
                str(path.relative_to(root)): digest(path)
                for path in [
                    root / "zeroshot.pt",
                    root / "SVHN/finetuned.pt",
                    root / "MNIST/finetuned.pt",
                ]
            },
            helpers_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "polynomial_probe.py",
                    "operator_diagnostics.py",
                    "bounded_complement.py",
                ]
            },
            predictions_sha256=digest(artifacts / "predictions.pt"),
            training="no optimizer or fine-tuning; covariance calibration uses partner checkpoint",
            new_defense_validated=False,
            test_usage="none",
            seed_count=1,
        ),
    )
    print("ERASURE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/context_erasure.json")
    parser.add_argument("--output", default="outputs/2026-10-10/context_erasure")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
