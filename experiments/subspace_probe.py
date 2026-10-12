"""Centered task subspaces and complement-only parameter edits.

Delta W = R(I-UU^T), Delta b = -Delta W mu. Preserving the estimated
subspace does not guarantee unseen classification accuracy or attack resistance.
Historical searches are retained in Git; this module contains shared math only.
"""

from __future__ import annotations

import copy
import itertools
import json

import torch
from torch.utils.data import DataLoader, Subset


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_loader(dataset, indices=None, batch_size=128):
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)


def split_training(dataset, sizes, seed):
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    result = {}
    offset = 0
    for name, count in sizes.items():
        result[name] = order[offset : offset + count]
        offset += count
    assert offset <= len(dataset)
    assert len(set(itertools.chain.from_iterable(result.values()))) == offset
    return result


def site_parameters(model, site):
    target, layer_text = site.split(":")
    block = model.model.visual.transformer.resblocks[int(layer_text)]
    if target == "V":
        return block.attn.in_proj_weight[1536:], block.attn.in_proj_bias[1536:]
    return block.mlp.c_fc.weight, block.mlp.c_fc.bias


@torch.no_grad()
def collect_moments(model, loader, sites, token_index=None):
    """Streaming centered moments of actual linear inputs, including all tokens."""
    sums, grams, counts = {}, {}, {}
    handles = []

    def capture(site, value):
        if token_index is not None:
            # CLIP normalization hooks return sequence, batch, channels.
            assert value.ndim == 3
            value = value[token_index]
        x = value.detach().reshape(-1, value.shape[-1]).double()
        if site not in sums:
            sums[site] = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
            grams[site] = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
            counts[site] = 0
        sums[site].add_(x.sum(0))
        grams[site].add_(x.T @ x)
        counts[site] += len(x)

    for site in sites:
        target, layer_text = site.split(":")
        block = model.model.visual.transformer.resblocks[int(layer_text)]
        normalization = block.ln_1 if target == "V" else block.ln_2
        handles.append(
            normalization.register_forward_hook(
                lambda _module, _args, value, name=site: capture(name, value)
            )
        )
    try:
        model.cuda().eval()
        for images, _ in loader:
            model(images.cuda(non_blocking=True))
    finally:
        model.cpu()
        for handle in handles:
            handle.remove()
    result = {}
    for site in sites:
        mean = sums[site] / counts[site]
        covariance = grams[site] / counts[site] - mean[:, None] @ mean[None, :]
        result[site] = dict(
            mean=mean.cpu(),
            covariance=((covariance + covariance.T) / 2).cpu(),
            tokens=counts[site],
        )
    return result


@torch.no_grad()
def fit_basis(moment, energy):
    values, vectors = torch.linalg.eigh(moment["covariance"].cuda())
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    fractions = values.cumsum(0) / values.sum()
    rank = int(torch.searchsorted(fractions, energy).item()) + 1
    assert 0 < rank < len(values), (rank, energy)
    curve = {str(r): float(fractions[r - 1]) for r in [16, 32, 64, 128, 256, 512, 768]}
    thresholds = {
        str(e): int(torch.searchsorted(fractions, e).item()) + 1 for e in [0.9, 0.99, 0.999, 0.9999]
    }
    basis = dict(
        mean=moment["mean"],
        u=vectors[:, :rank].cpu(),
        rank=rank,
        complement_rank=len(values) - rank,
    )
    stats = dict(
        rank=rank,
        complement_rank=len(values) - rank,
        ranks_at_energy=thresholds,
        explained_energy=curve,
        numerical_rank=int((values > values[0] * 1e-8).sum()),
        retained_fraction=float(fractions[rank - 1]),
        tokens=moment["tokens"],
    )
    return basis, stats


@torch.no_grad()
def complement_power(moment, basis):
    u = basis["u"]
    center_shift = moment["mean"] - basis["mean"]
    covariance = moment["covariance"] + center_shift[:, None] @ center_shift[None, :]
    total = covariance.trace()
    outside = total - (u.T @ covariance @ u).trace()
    return dict(total=float(total), outside=float(outside), fraction=float(outside / total))


@torch.no_grad()
def perturb(model, site, basis, strength, seed):
    """Delta W = scaled R(I-UU^T), Delta b = -Delta W mu.

    Strength is ||Delta W||_F / ||W||_F, not a claimed function-distance bound.
    """
    result = copy.deepcopy(model).cpu().eval()
    weight, bias = site_parameters(result, site)
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(weight.shape, dtype=torch.float64, generator=generator).cuda()
    u = basis["u"].cuda()
    delta = random_matrix - (random_matrix @ u) @ u.T
    delta *= strength * weight.double().norm().cuda() / delta.norm()
    orthogonality_error = float((delta @ u).norm() / delta.norm())
    assert orthogonality_error < 1e-10, orthogonality_error
    shift = delta @ basis["mean"].cuda()
    assert torch.isfinite(delta).all() and torch.isfinite(shift).all()
    weight.add_(delta.float().cpu())
    bias.sub_(shift.float().cpu())
    # Deployment uses float32; also check its actual main-space leakage.
    old_weight, _ = site_parameters(model, site)
    actual = (weight.double() - old_weight.double()).cuda()
    actual_leakage = float((actual @ u).norm() / actual.norm())
    assert actual_leakage < 1e-5, actual_leakage
    return result, dict(
        relative_parameter_change=float(actual.norm() / old_weight.double().norm().cuda()),
        main_space_relative_leakage_float64=orthogonality_error,
        main_space_relative_leakage_float32=actual_leakage,
        bias_change_norm=float(shift.norm()),
    )


@torch.no_grad()
def clean_complement(model, anchor, site, basis, mode):
    """Independent-data white-box attacks; never receive defender PCA or seed."""
    result = copy.deepcopy(model).cpu().eval()
    weight, bias = site_parameters(result, site)
    base, _ = site_parameters(anchor, site)
    difference = weight.double() if mode == "projection" else weight.double() - base.double()
    u = basis["u"]
    removed = difference - (difference @ u) @ u.T
    weight.sub_(removed.float())
    bias.add_((removed @ basis["mean"]).float())
    return result
