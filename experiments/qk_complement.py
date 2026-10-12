"""One fixed complement-only Q/K candidate; no optimizer or parameter sweep."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from eight_task_benchmark import author_recover
from mechanisms import load_author
from scipy.linalg import eigh
from subspace_probe import fit_basis, make_loader, save_json, split_training
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def inputs(model, dataset, indices):
    values = []
    handle = model.model.visual.transformer.resblocks[0].ln_1.register_forward_hook(
        lambda _m, _a, y: values.append(y.permute(1, 0, 2).detach().cpu())
    )
    try:
        model.cuda().eval()
        for images, _ in make_loader(dataset, indices, batch_size=32):
            model(images.cuda(non_blocking=True))
    finally:
        handle.remove()
        model.cpu()
    return torch.cat(values).double().cuda()


def within_covariance(x):
    centered = (x - x.mean(1, keepdim=True)).reshape(-1, x.shape[-1])
    return centered.T @ centered / len(centered)


def qk(x, attention):
    weight, bias = attention.in_proj_weight.double().cuda(), attention.in_proj_bias.double().cuda()
    q, k, _ = torch.nn.functional.linear(x, weight, bias).chunk(3, -1)
    return [y.reshape(len(x), x.shape[1], 12, 64).permute(0, 2, 1, 3) for y in [q, k]]


def centered_scores(q, k):
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(64)
    return scores - scores.mean(-1, keepdim=True)


@torch.no_grad()
def construct(original, x, xp, score_budget):
    flat = x.reshape(-1, 768)
    mean = flat.mean(0)
    covariance = (flat - mean).T @ (flat - mean) / len(flat)
    basis, stats = fit_basis(
        dict(mean=mean.cpu(), covariance=covariance.cpu(), tokens=len(flat)), 0.9999
    )
    p = basis["u"].cuda()
    null = original.model.visual.transformer.resblocks[0].ln_1.weight.double().cuda().reciprocal()
    null /= null.norm()
    assert float((p.T @ null).norm()) < 1e-5
    complete, _ = torch.linalg.qr(torch.cat([p, null[:, None]], dim=1), mode="complete")
    outside = complete[:, p.shape[1] + 1 :]
    cm, cp = within_covariance(x), within_covariance(xp)
    denominator = outside.T @ cm @ outside
    ridge = float(denominator.trace() / len(denominator)) * 1e-6
    assert ridge > 0
    a = (outside.T @ cp @ outside).cpu().numpy()
    b = denominator.cpu().numpy() + ridge * np.eye(len(denominator))
    eigenvalue, vector = eigh(a, b, subset_by_index=[len(a) - 1, len(a) - 1])
    v = outside @ torch.from_numpy(vector[:, 0]).cuda()
    v /= v.norm()
    attention = original.model.visual.transformer.resblocks[0].attn
    q, k = qk(x, attention)
    u = q[:, :, 0].mean(0)
    u /= u.norm(dim=-1, keepdim=True)
    g = (x - mean) @ v
    du = g[:, None, :, None] * u[None, :, None, :]
    base = centered_scores(q, k)
    linear = (du @ k.transpose(-1, -2) - q @ du.transpose(-1, -2)) / math.sqrt(64)
    quadratic = -(du @ du.transpose(-1, -2)) / math.sqrt(64)
    linear -= linear.mean(-1, keepdim=True)
    quadratic -= quadratic.mean(-1, keepdim=True)

    def rms(y):
        return float(y.square().mean().sqrt())

    linear_scale, quadratic_scale = rms(linear), rms(quadratic)
    target = score_budget * rms(base)
    alpha = 2 * target / (linear_scale + math.sqrt(linear_scale**2 + 4 * quadratic_scale * target))
    assert math.isfinite(alpha) and alpha > 0
    delta = alpha * u.reshape(768, 1) @ v[None, :]
    shift = delta @ mean
    released = copy.deepcopy(original).cpu().eval()
    edited = released.model.visual.transformer.resblocks[0].attn
    edited.in_proj_weight[:768].add_(delta.float().cpu())
    edited.in_proj_weight[768:1536].sub_(delta.float().cpu())
    edited.in_proj_bias[:768].sub_(shift.float().cpu())
    edited.in_proj_bias[768:1536].add_(shift.float().cpu())
    qr, kr = qk(x, edited)
    measured = rms(centered_scores(qr, kr) - base) / rms(base)
    dq = (
        edited.in_proj_weight[:768].double().cuda() - attention.in_proj_weight[:768].double().cuda()
    )
    leakage = float((dq @ p).norm() / dq.norm())
    stats.update(
        alpha=alpha,
        score_budget=score_budget,
        actual_calibration_score_change=measured,
        input_span_leakage=leakage,
        within_token_ratio=float(eigenvalue[0]),
        direction_generalized_eigen_residual=float(
            np.linalg.norm(a @ vector[:, 0] - eigenvalue[0] * b @ vector[:, 0])
            / np.linalg.norm(a @ vector[:, 0])
        ),
        ridge=ridge,
        linear_rms=linear_scale,
        quadratic_rms=quadratic_scale,
        original_score_rms=rms(base),
    )
    assert leakage < 1e-5 and measured <= score_budget * 1.02
    return (
        released,
        dict(
            mean=mean.cpu(),
            p=p.cpu(),
            v=v.cpu(),
            u=u.cpu(),
            delta=delta.cpu(),
            shift=shift.cpu(),
            alpha=alpha,
        ),
        stats,
    )


@torch.no_grad()
def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(
        layer=0,
        beta_proxy=0.5,
        coefficient=0.5,
        retained_energy=0.9999,
        seed=20261012,
        calibration=256,
        confirmation=256,
        score_budget=0.05,
        construction="Target MNIST expert, public anchor and unlabelled target training data only.",
        direction="Within-image token contrast in the target input complement; exact LayerNorm null excluded.",
        amplitude="One analytic bound on centered native score perturbation; no search or test fitting.",
        attack="Author attention wrapper then MLP wrapper, all layers, applied to both experts.",
        scope="Two-task pilot; earlier official tests and SVHN results have been observed; no Table 6 claim.",
        stop="Do not tune this candidate. If standalone loss exceeds 2 pp, stop before fusion.",
    )
    save_json(output / "protocol.json", protocol)
    pins = json.loads(Path("configs/sources.json").read_bytes())
    author, author_sha = load_author(
        Path(args.author_repo) / "utils_attack_B32.py", include_wrappers=True
    )
    assert author_sha == pins["AGA"]["sha256"]
    root = Path(args.checkpoints)
    entries = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    checkpoint_hashes = {}
    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "head_MNIST.pt"]:
        entry = next(e for e in entries if e["path"] == "ViT-B-32/" + name)
        checkpoint_hashes[name] = digest(root / name)
        assert checkpoint_hashes[name] == entry["sha256"]
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    original = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
    head_m = load(str(root / "head_MNIST.pt"))
    training = MNIST(args.data, train=True, transform=original.val_preprocess)
    splits = split_training(training, dict(calibration=256, confirmation=256), protocol["seed"])
    save_json(output / "training_indices.json", splits)
    x = inputs(original, training, splits["calibration"])
    proxy = merge(anchor, original, anchor, protocol["beta_proxy"])
    xp = inputs(proxy, training, splits["calibration"])
    released, edit, stats = construct(original, x, xp, protocol["score_budget"])
    torch.save(edit, output / "artifacts/release.pt")
    torch.save(
        dict(native=x.float().cpu(), proxy=xp.float().cpu()),
        output / "artifacts/calibration_inputs.pt",
    )
    save_json(output / "construction.json", stats)
    save_json(
        output / "release_freeze.json",
        dict(
            artifact_type="Edit package; FP32 release rebuilt from pinned target checkpoint.",
            release_sha256=digest(output / "artifacts/release.pt"),
            checkpoint_sha256=checkpoint_hashes,
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            author_sha256=author_sha,
            protocol_sha256=digest(output / "protocol.json"),
            partner_loaded=False,
        ),
    )
    del x, xp, proxy
    torch.cuda.empty_cache()
    rows, predictions = [], {}
    loader_m = make_loader(MNIST(args.data, train=False, transform=original.val_preprocess))
    loader_c = make_loader(training, splits["confirmation"])

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
        print("QK_RESULT", json.dumps(row), flush=True)
        return metrics

    native = {}
    for name, model in [("original", original), ("released", released)]:
        record(model, head_m, loader_c, name, "confirmation", "MNIST")
        native[name] = record(model, head_m, loader_m, name, "standalone", "MNIST")
    loss = 100 * (native["original"]["top1"] - native["released"]["top1"])
    if loss <= 2:
        # Future expert is loaded only after the actual release and its fingerprint exist.
        for name in ["SVHN/finetuned.pt", "head_SVHN.pt"]:
            entry = next(e for e in entries if e["path"] == "ViT-B-32/" + name)
            checkpoint_hashes[name] = digest(root / name)
            assert checkpoint_hashes[name] == entry["sha256"]
        partner = load(str(root / "SVHN/finetuned.pt")).cpu().eval()
        head_s = load(str(root / "head_SVHN.pt"))
        loader_s = make_loader(SVHN(args.data, split="test", transform=original.val_preprocess))
        recovered_partner = author_recover(partner, anchor, author)
        for name, model in [("original", original), ("released", released)]:
            for stage in ["direct", "AGA"]:
                if stage == "direct":
                    tested = merge(anchor, model, partner, protocol["coefficient"])
                else:
                    attacked = author_recover(model, anchor, author)
                    record(attacked, head_m, loader_m, name, "AGA_standalone", "MNIST")
                    tested = merge(anchor, attacked, recovered_partner, protocol["coefficient"])
                    del attacked
                record(tested, head_m, loader_m, name, stage, "MNIST")
                record(tested, head_s, loader_s, name, stage, "SVHN")
                del tested
    torch.save(predictions, output / "artifacts/predictions.pt")
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
            predictions_sha256=digest(output / "artifacts/predictions.pt"),
            checkpoint_sha256=checkpoint_hashes,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
