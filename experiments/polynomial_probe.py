"""Whole-FFN quadratic emulator and a closed-form reconstruction attack.

Public synthetic inputs only; no gradients or fine-tuning. Architecture mismatch
is never counted as security: evaluate shared-format TA and original-format LS.
This is not a full MergeBarrier or PLeaS reproduction.
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
from low_activity_posthoc import synthetic_images
from mechanisms import load_author, relative
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN


class QuadraticMLP(nn.Module):
    def __init__(self, center, basis, scale, coefficient):
        super().__init__()
        self.register_buffer("center", center.float())
        self.register_buffer("basis", basis.float())
        self.register_buffer("scale", scale.float())
        indices = torch.triu_indices(basis.shape[1], basis.shape[1])
        self.register_buffer("index_i", indices[0])
        self.register_buffer("index_j", indices[1])
        self.coefficient = nn.Parameter(coefficient.float(), requires_grad=False)

    def design(self, x):
        z = ((x - self.center) @ self.basis) / self.scale
        return torch.cat(
            [torch.ones_like(z[..., :1]), z, z[..., self.index_i] * z[..., self.index_j]], dim=-1
        )

    def forward(self, x):
        return self.design(x) @ self.coefficient


@torch.no_grad()
def capture(model, images, layer):
    values = []
    handle = model.model.visual.transformer.resblocks[layer].ln_2.register_forward_hook(
        lambda _m, _a, value: values.append(value.detach().reshape(-1, value.shape[-1]).cpu())
    )
    model.cuda().eval()
    for batch in images.split(8):
        model(batch.cuda())
    handle.remove()
    model.cpu()
    return torch.cat(values).double().cuda()


@torch.no_grad()
def feature_basis(anchor, images, layer, rank):
    x = capture(anchor, images, layer)
    center = x.mean(0)
    centered = x - center
    eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered / len(x))
    basis = eigenvectors[:, -rank:]
    scale = eigenvalues[-rank:].clamp_min(1e-8).sqrt()
    return center, basis, scale


@torch.no_grad()
def convert(model, images, layer, basis, ridge_ratio):
    x = capture(model, images, layer)
    mlp = copy.deepcopy(model.model.visual.transformer.resblocks[layer].mlp).double().cuda().eval()
    target = mlp(x)
    center, directions, scale = basis
    width = 1 + len(scale) + len(scale) * (len(scale) + 1) // 2
    module = (
        QuadraticMLP(center.cpu(), directions.cpu(), scale.cpu(), torch.zeros(width, x.shape[-1]))
        .double()
        .cuda()
    )
    phi = module.design(x)
    gram = phi.T @ phi
    ridge = ridge_ratio * gram.trace() / len(gram)
    coefficient = torch.linalg.solve(
        gram + ridge * torch.eye(len(gram), device="cuda", dtype=torch.float64), phi.T @ target
    )
    module.coefficient.copy_(coefficient)
    fit_error = relative(module(x), target)
    result = copy.deepcopy(model).cpu().eval()
    original_parameters = sum(
        p.numel() for p in result.model.visual.transformer.resblocks[layer].mlp.parameters()
    )
    result.model.visual.transformer.resblocks[layer].mlp = module.float().cpu().eval()
    del mlp
    evidence = dict(
        calibration_tokens=len(x),
        features=width,
        ridge=float(ridge),
        fit_feature_error=fit_error,
        original_ffn_parameters=original_parameters,
        released_ffn_parameters=sum(p.numel() for p in module.parameters()),
        released_ffn_bytes=sum(v.numel() * v.element_size() for v in module.state_dict().values()),
    )
    return result, evidence


@torch.no_grad()
def ridge_update(design, residual_target, ridge_ratio, solver="auto"):
    """Solve the existing anchor-centered ridge update; shared by block probes."""
    if solver == "auto":
        solver = "dual" if len(design) <= design.shape[1] else "primal"
    assert solver in {"dual", "primal"}
    if solver == "dual":
        gram = design @ design.T
        ridge = ridge_ratio * gram.trace() / len(gram)
        update = design.T @ torch.linalg.solve(
            gram + ridge * torch.eye(len(gram), device="cuda", dtype=torch.float64),
            residual_target,
        )
    else:
        gram = design.T @ design
        # Preserve the original dual objective and its mean row-energy scaling.
        ridge = ridge_ratio * design.square().sum() / len(design)
        update = torch.linalg.solve(
            gram + ridge * torch.eye(len(gram), device="cuda", dtype=torch.float64),
            design.T @ residual_target,
        )
    normal_residual = design.T @ (design @ update - residual_target) + ridge * update
    normal_error = float(
        normal_residual.norm() / (design.T @ residual_target).norm().clamp_min(1e-30)
    )
    assert normal_error < 1e-8, normal_error
    return update, dict(
        ridge=float(ridge),
        solver=solver,
        design_features=design.shape[1],
        normal_equation_relative_residual=normal_error,
    )


@torch.no_grad()
def reconstruct(
    released,
    anchor,
    images,
    layer,
    ridge_ratio,
    solver="dual",
    *,
    feature_model=None,
    input_model=None,
):
    """Fix pretrained W1 and fit W2 to released outputs; no original weights.

    Center the solution at pretrained W2, including a bias feature. The default
    dual solver is preserved; auto uses the smaller equivalent ridge system.
    The same attack is applied to the unprotected control. Optional feature/input
    models support explicitly labelled unprotected-control diagnostics only.
    """
    x = capture(released if input_model is None else input_model, images, layer)
    teacher = (
        copy.deepcopy(released.model.visual.transformer.resblocks[layer].mlp).double().cuda().eval()
    )
    target = teacher(x)
    student = (
        copy.deepcopy(anchor.model.visual.transformer.resblocks[layer].mlp).double().cuda().eval()
    )
    if feature_model is not None:
        student.c_fc.load_state_dict(
            feature_model.model.visual.transformer.resblocks[layer].mlp.c_fc.state_dict(),
            strict=True,
        )
    hidden = student.gelu(student.c_fc(x))
    design = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
    initial = torch.cat([student.c_proj.weight.T, student.c_proj.bias[None, :]], dim=0)
    update, evidence = ridge_update(design, target - design @ initial, ridge_ratio, solver)
    fitted = initial + update
    student.c_proj.weight.copy_(fitted[:-1].T)
    student.c_proj.bias.copy_(fitted[-1])
    residual = relative(student(x), target)
    result = copy.deepcopy(released).cpu().eval()
    result.model.visual.transformer.resblocks[layer].mlp = student.float().cpu().eval()
    del teacher
    return result, dict(
        calibration_tokens=len(x),
        fit_feature_error=residual,
        **evidence,
        first_layer="pretrained_fixed" if feature_model is None else "released_control_fixed",
        second_layer="closed_form_anchor_centered_ridge",
    )


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    layers = config.get("layers", [config.get("layer", 11)])
    assert layers and len(set(layers)) == len(layers)
    (output / "protocol.json").write_bytes(config_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    author, source_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
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
    images = synthetic_images(original.val_preprocess, config["defender_samples"], config["seed"])
    attack_images = synthetic_images(
        original.val_preprocess, config["attacker_samples"], config["seed"] + 1
    )
    conversion_start = time.perf_counter()
    bases = {layer: feature_basis(anchor, images, layer, config["latent_rank"]) for layer in layers}
    candidates = []
    evidence = {}
    for name, model in [("anchor", anchor), ("protected", original), ("other", second)]:
        converted = model
        details = {}
        for layer in layers:
            converted, info = convert(converted, images, layer, bases[layer], config["ridge_ratio"])
            details[str(layer)] = info
        candidates.append(converted)
        evidence[name] = info if len(layers) == 1 else details
    conversion_seconds = time.perf_counter() - conversion_start
    poly_anchor, protected, poly_second = candidates
    rows = []

    def record(group, model, coefficient=None):
        for dataset, loader in loaders.items():
            row = dict(
                experiment="whole_ffn_quadratic",
                group=group,
                dataset=dataset,
                scope="real_full_test",
                seed=config["seed"],
                coefficient=coefficient,
                layer=layers[0] if len(layers) == 1 else None,
                layers=layers,
                rank=config["latent_rank"],
                defender_calibration_samples=config["defender_samples"],
                attacker_calibration_samples=config["attacker_samples"],
                calibration="procedural_unlabelled_strokes",
                fitting="closed_form_ridge",
                training="no_gradient_or_optimizer",
                **evaluate(model, heads[dataset], loader, "cuda"),
            )
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print("POLY_RESULT", json.dumps(row), flush=True)

    record("unprotected_standalone", original)
    record("protected_standalone", protected)
    record("converted_other_standalone", poly_second)
    attack_dir = output / "AGA_compatible_layers"
    attack_dir.mkdir()
    aga = recover(protected, poly_anchor, author, attack_dir, skip_mlp_layers=layers)
    reconstruction_seconds = {}
    reconstructed = []
    for name, released in [
        ("protected_reconstruction", protected),
        ("control_reconstruction", original),
    ]:
        reconstruction_start = time.perf_counter()
        result = released
        details = {}
        for layer in layers:
            result, info = reconstruct(result, anchor, attack_images, layer, config["ridge_ratio"])
            details[str(layer)] = info
        evidence[name] = info if len(layers) == 1 else details
        reconstruction_seconds[name] = time.perf_counter() - reconstruction_start
        reconstructed.append(result)
    restored, control = reconstructed
    record("reconstructed_protected_standalone", restored)
    record("reconstructed_control_standalone", control)
    variants = [
        ("original_direct_merge", anchor, original, second),
        ("shared_format_merge", poly_anchor, protected, poly_second),
        ("shared_format_AGA_merge", poly_anchor, aga, poly_second),
        ("reconstructed_protected_merge", anchor, restored, second),
        ("reconstructed_control_merge", anchor, control, second),
    ]
    for group, base, left, right in variants:
        combined = merge(base, left, right, config["coefficient"])
        record(group, combined, config["coefficient"])
        del combined
    metadata = dict(
        completed=True,
        total_seconds=time.perf_counter() - start,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        author_sha256=source_sha,
        protocol=config,
        evidence=evidence,
        original_weights_retained_in_released_ffn=False,
        aga_mlp_skipped_layers=layers,
        conversion_seconds=conversion_seconds,
        reconstruction_seconds=reconstruction_seconds,
        defender_calibration_seed=config["seed"],
        attacker_calibration_seed=config["seed"] + 1,
        conversion_order="sequential; each block fitted on current model activations",
        reconstruction_source="released_model_pretrained_and_independent_public_synthetic_images",
        merge_implementation="author_TaskVector_difference_and_sum",
        architecture_mismatch_counted_as_security=False,
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("POLY_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/polynomial_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-09/polynomial_probe")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
