"""Replay a rejected joint-transfer release; diagnose errors without refitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from joint_posthoc import capture_blocks
from operator_diagnostics import paired_counts
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def run(args):
    parent, out = Path(args.parent), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    config = json.loads((parent / "protocol.json").read_bytes())
    indices = json.loads((parent / "training_indices.json").read_bytes())
    fitting = json.loads((parent / "fitting.json").read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.checkpoints)
    load = checkpoint_loader(Path(args.author_repo))
    original = load(str(root / "MNIST/finetuned.pt"))
    head = load(str(root / "head_MNIST.pt"))
    release = copy.deepcopy(original).cpu().eval()
    state = torch.load(parent / "artifacts/release.pt", map_location="cpu", weights_only=True)
    for layer, block_state in state.items():
        block = release.model.visual.transformer.resblocks[layer]
        current = block.state_dict()
        current.update(block_state)
        block.load_state_dict(current, strict=True)
    dataset = MNIST(args.data, train=True, transform=original.val_preprocess)
    subsets = {
        "calibration": indices["calibration"],
        "confirmation_prefix_256": indices["confirmation"]["MNIST"][:256],
    }
    assert len(subsets["calibration"]) == 256
    assert not set(subsets["calibration"]) & set(subsets["confirmation_prefix_256"])
    save_json(
        out / "protocol.json",
        dict(
            parent=args.parent,
            subsets=subsets,
            fitting="none; replay frozen parameters",
            purpose="Check calibration utility and replay fitted block targets; no candidate selection.",
            confirmation="First 256 images from the already evaluated training confirmation; not a new test.",
            scope="No causal separation of compression and branch transfer; no representability floor claim.",
        ),
    )
    start = time.perf_counter()
    rows, predictions = [], {}
    for subset, selected in subsets.items():
        for group, model in [("unprotected", original), ("joint_transfer", release)]:
            chunks, labels = [], []

            def sink(pred, target):
                chunks.append(pred.clone())
                labels.append(target.clone())

            score = evaluate(
                model, head, make_loader(dataset, selected), "cuda", prediction_sink=sink
            )
            pred, target = torch.cat(chunks), torch.cat(labels)
            pair = None
            if group == "joint_transfer":
                ref = predictions[subset, "unprotected"]
                assert torch.equal(target, ref["labels"])
                pair = paired_counts(ref["prediction"].eq(target), pred.eq(target))
            predictions[subset, group] = dict(prediction=pred, labels=target)
            rows.append(dict(subset=subset, group=group, paired=pair, **score))
    save_json(out / "results.json", rows)
    raw = out / "artifacts/predictions.pt"
    torch.save(predictions, raw)
    layers = []
    for subset, selected in subsets.items():
        images = torch.stack([dataset[i][0] for i in selected])
        targets = capture_blocks(original, images, config["layers"])
        current = capture_blocks(release, images, config["layers"], details=True)
        for layer in config["layers"]:
            item, teacher = current[layer], targets[layer]["output"].double()
            mlp = (
                copy.deepcopy(release.model.visual.transformer.resblocks[layer].mlp).double().cuda()
            )
            hidden = mlp.gelu(mlp.c_fc(item["mlp_input"].double().cuda()))
            design = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
            target = teacher.cuda() - item["post_attention"].double().cuda()
            fitted = torch.cat([mlp.c_proj.weight.T, mlp.c_proj.bias[None]], dim=0)
            error = design @ fitted - target
            replay_error = float(error.norm() / target.norm())
            recorded = fitting[layer]["residual_target_relative_error"]
            if subset == "calibration":
                assert abs(replay_error - recorded) < 2e-5 * max(recorded, 1e-6)
            layers.append(
                dict(
                    subset=subset,
                    layer=layer,
                    residual_target_relative_error=replay_error,
                    original_block_relative_error=float(error.norm() / teacher.cuda().norm()),
                    actual_block_relative_error=float(
                        (item["output"].double() - teacher).norm() / teacher.norm()
                    ),
                    residual_target_to_teacher_norm=float(target.norm() / teacher.cuda().norm()),
                    recorded_calibration_residual=recorded,
                )
            )
            del mlp, hidden, design, target, fitted, error
        del images, targets, current
    save_json(out / "layer_errors.json", layers)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            source_sha256=digest(__file__),
            helpers_sha256={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "joint_posthoc.py",
                    "clip_benchmark.py",
                    "subspace_probe.py",
                    "operator_diagnostics.py",
                ]
            },
            parent_release_sha256=digest(parent / "artifacts/release.pt"),
            parent_protocol_sha256=digest(parent / "protocol.json"),
            parent_indices_sha256=digest(parent / "training_indices.json"),
            predictions_sha256=digest(raw),
            checkpoint_sha256={
                p: digest(root / p) for p in ["MNIST/finetuned.pt", "head_MNIST.pt"]
            },
            no_parameters_fitted_or_selected=True,
            calibrated_block_targets_replayed=True,
        ),
    )
    print("JOINT_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="outputs/2026-10-10/joint_transfer_mnist")
    parser.add_argument("--output", default="outputs/2026-10-10/joint_transfer_diagnostics")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
