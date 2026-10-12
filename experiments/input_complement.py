"""One exact grayscale-complement patch edit, with full AGA and public reset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from eight_task_benchmark import author_recover
from mechanisms import load_author
from subspace_probe import make_loader, save_json, split_training
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def color_basis(model):
    normalization = next(
        t for t in model.val_preprocess.transforms if hasattr(t, "mean") and hasattr(t, "std")
    )
    mean, std = [
        torch.tensor(v, dtype=torch.float64) for v in [normalization.mean, normalization.std]
    ]
    a, b = std.reciprocal(), mean / std
    v = torch.linalg.cross(a, b)
    v /= v.norm()
    assert abs(float(v @ a)) < 1e-12 and abs(float(v @ b)) < 1e-12
    return dict(mean=mean, std=std, v=v, projector=v[:, None] @ v[None, :])


@torch.no_grad()
def construct(original):
    basis = color_basis(original)
    weight = original.model.visual.conv1.weight.double()
    delta = 100 * torch.einsum("ocij,cd->odij", weight, basis["projector"])
    released = copy.deepcopy(original).cpu().eval()
    released.model.visual.conv1.weight.add_(delta.float())
    return released, dict(**basis, delta=delta, multiplier=100)


@torch.no_grad()
def public_reset(model, anchor, projector):
    """Public preprocessing determines the complement; no defender calibration."""
    result = copy.deepcopy(model).cpu().eval()
    current = result.model.visual.conv1.weight.double()
    base = anchor.model.visual.conv1.weight.double()
    removed = torch.einsum("ocij,cd->odij", current - base, projector)
    result.model.visual.conv1.weight.copy_((current - removed).float())
    return result


@torch.no_grad()
def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(
        construction="Target MNIST checkpoint and its public RGB preprocessing only; no future expert, input or test fitting.",
        edit="conv1 only; Delta W = 100 W C, C annihilates grayscale slope and preprocessing offset. One fixed multiplier, no scan.",
        calibration="256 MNIST training images verify the grayscale and FP32 preservation conditions; no fitting.",
        seed=20261012,
        coefficient=0.5,
        attack="Full original author attention then MLP wrappers, all 12 layers, on both experts.",
        public_reset="Replace the analytically known input complement by public anchor, then full AGA. Apply identically to unprotected control.",
        boundary="Exact input complement cannot interfere with protected grayscale task under linear parameter merging. Measure colored partner separately; this is not protected-task collapse or general recovery resistance.",
        scope="Two-task pilot. Prior MNIST and SVHN tests observed; not blind evaluation, novelty evidence or Table 6 reproduction.",
        stop="If standalone loss exceeds 2 pp, stop before fusion; do not tune this candidate.",
    )
    save_json(output / "protocol.json", protocol)
    pins = json.loads(Path("configs/sources.json").read_bytes())
    author, author_sha = load_author(
        Path(args.author_repo) / "utils_attack_B32.py", include_wrappers=True
    )
    assert author_sha == pins["AGA"]["sha256"]
    entries = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    root = Path(args.checkpoints)
    checkpoint_hashes = {}

    def verify(names):
        for name in names:
            entry = next(e for e in entries if e["path"] == "ViT-B-32/" + name)
            checkpoint_hashes[name] = digest(root / name)
            assert checkpoint_hashes[name] == entry["sha256"]

    verify(["zeroshot.pt", "MNIST/finetuned.pt", "head_MNIST.pt"])
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    original = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
    head_m = load(str(root / "head_MNIST.pt"))
    released, edit = construct(original)
    training = MNIST(args.data, train=True, transform=original.val_preprocess)
    indices = split_training(training, dict(calibration=256), protocol["seed"])
    save_json(output / "training_indices.json", indices)
    values = torch.stack([training[i][0] for i in indices["calibration"]]).double()
    actual_delta = (
        released.model.visual.conv1.weight.double() - original.model.visual.conv1.weight.double()
    )
    response = torch.nn.functional.conv2d(values, actual_delta, stride=32)
    original_response = torch.nn.functional.conv2d(
        values, original.model.visual.conv1.weight.double(), stride=32
    )
    fraction = float(response.square().mean().sqrt() / original_response.square().mean().sqrt())
    assert fraction < 1e-3, fraction
    save_json(
        output / "construction.json",
        dict(
            relative_calibration_patch_change=fraction,
            multiplier=100,
            projector_rank=1,
            patch_input_dimension=3072,
            preserved_linear_dimension=2048,
            exact_scope="Per-pixel span of grayscale slope and RGB normalization offset; conservative superset of grayscale inputs.",
        ),
    )
    torch.save(edit, output / "artifacts/release.pt")
    save_json(
        output / "release_freeze.json",
        dict(
            release_sha256=digest(output / "artifacts/release.pt"),
            checkpoint_sha256=checkpoint_hashes.copy(),
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            author_sha256=author_sha,
            protocol_sha256=digest(output / "protocol.json"),
            partner_loaded=False,
            artifact_type="Edit package; rebuild FP32 release from pinned MNIST checkpoint.",
        ),
    )
    rows, predictions = [], {}
    loader_m = make_loader(MNIST(args.data, train=False, transform=original.val_preprocess))

    def record(model, head, loader, preset, stage, dataset):
        batches = []
        metrics = evaluate(model, head, loader, "cuda", lambda p, y: batches.append((p, y)))
        pred, labels = [torch.cat([pair[i] for pair in batches]) for i in range(2)]
        key = preset + "/" + stage + "/" + dataset
        predictions[key] = dict(prediction=pred, target=labels)
        row = dict(preset=preset, stage=stage, dataset=dataset, **metrics)
        if preset == "released":
            control = predictions["original/" + stage + "/" + dataset]
            assert torch.equal(labels, control["target"])
            row["paired"] = paired_counts(control["prediction"].eq(labels), pred.eq(labels))
        rows.append(row)
        save_json(output / "results.json", rows)
        torch.save(predictions, output / "artifacts/predictions.pt")
        print("INPUT_RESULT", json.dumps(row), flush=True)
        return metrics

    native = {
        name: record(model, head_m, loader_m, name, "standalone", "MNIST")
        for name, model in [("original", original), ("released", released)]
    }
    loss = 100 * (native["original"]["top1"] - native["released"]["top1"])
    if loss <= 2:
        verify(["SVHN/finetuned.pt", "head_SVHN.pt"])
        partner = load(str(root / "SVHN/finetuned.pt")).cpu().eval()
        head_s = load(str(root / "head_SVHN.pt"))
        loader_s = make_loader(SVHN(args.data, split="test", transform=original.val_preprocess))
        recovered_partner = author_recover(partner, anchor, author)
        for name, model in [("original", original), ("released", released)]:
            for stage in ["direct", "AGA", "public_reset_AGA"]:
                if stage == "direct":
                    tested = merge(anchor, model, partner, protocol["coefficient"])
                else:
                    source = (
                        model if stage == "AGA" else public_reset(model, anchor, edit["projector"])
                    )
                    attacked = author_recover(source, anchor, author)
                    assert torch.equal(
                        attacked.model.visual.conv1.weight, source.model.visual.conv1.weight
                    )
                    record(attacked, head_m, loader_m, name, stage + "_standalone", "MNIST")
                    tested = merge(anchor, attacked, recovered_partner, protocol["coefficient"])
                    del attacked
                record(tested, head_m, loader_m, name, stage, "MNIST")
                record(tested, head_s, loader_s, name, stage, "SVHN")
                del tested
    save_json(
        output / "execution.json",
        dict(
            completed=True,
            standalone_loss_pp=loss,
            fusion_run=loss <= 2,
            new_candidates=1,
            optimizer_steps=0,
            parameter_search=False,
            gpu=torch.cuda.get_device_name(),
            checkpoint_sha256=checkpoint_hashes,
            predictions_sha256=digest(output / "artifacts/predictions.pt"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
