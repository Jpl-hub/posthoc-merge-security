"""One Key edit selected by attention Value-conditioned response, not input tails.

The analytical derivative includes softmax and the existing Value/output maps.
Reuse target capture, generalized eigensolver and native utility validation.
A mechanism probe, not an established new defense or security guarantee.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from joint_posthoc import capture_blocks
from local_erasure import rescale
from operator_attention_ablation import load_inputs, run, validate
from scipy.linalg import eigh
from subspace_probe import save_json


def weighted_sum(a, values):
    assert a.ndim == 2 and values.ndim == 3 and a.shape == values.shape[:2]
    return a[:, None, :].bmm(values).squeeze(1)


@torch.no_grad()
def responses(block, x):
    w, b = block.attn.in_proj_weight.double().cuda(), block.attn.in_proj_bias.double().cuda()
    q, k, v = torch.nn.functional.linear(x.double().cuda(), w, b).chunk(3, -1)
    n, tokens, _ = x.shape
    q, k, v = [t.reshape(n, tokens, 12, 64).transpose(1, 2) for t in (q, k, v)]
    a = ((q[:, :, :1] / 8) @ k.transpose(-1, -2)).softmax(-1)
    return q, k, v, a


@torch.no_grad()
def derivative(block, x, basis, u, head, scale):
    q, k, values, maps = responses(block, x)
    a = maps[:, head, 0]
    v = values[:, head]
    inputs = x.double().cuda()
    xc = inputs - weighted_sum(a, inputs)[:, None, :]
    vc = v - weighted_sum(a, v)[:, None, :]
    t = (q[:, head, 0] @ u) * scale / 8
    cross = vc.transpose(1, 2) @ (a[:, :, None] * (xc @ basis))
    o = block.attn.out_proj.weight[:, head * 64 : (head + 1) * 64].double().cuda()
    j = (o @ cross) * t[:, None, None]
    gram = j.transpose(1, 2).bmm(j).mean(0)
    energy = float((weighted_sum(a, v) @ o.T).square().sum(-1).mean())
    return gram, j, (q, k, values, maps), energy


@torch.no_grad()
def build_inputs(cfg, out):
    pcfg, bundle, _old_release, _old_edits = load_inputs(cfg)
    original, anchor = bundle[3]["original"], bundle[3]["anchor"]
    images = bundle[-1][: cfg["calibration_samples"]]
    layer = cfg["layer"]
    x = capture_blocks(original, images, [layer], details=True)[layer]["attention_input"]
    x = x.reshape(len(images), 50, 768).double().cuda()
    block = original.model.visual.transformer.resblocks[layer]
    q, _k, values, maps = responses(block, x)
    contributions = []
    for h in range(12):
        z = weighted_sum(maps[:, h, 0], values[:, h])
        o = block.attn.out_proj.weight[:, h * 64 : (h + 1) * 64].double().cuda()
        contributions.append(float((z @ o.T).square().sum(-1).mean()))
    head = int(np.argmax(contributions))
    u = q[:, head, 0].mean(0)
    u /= u.norm()
    centered = x.flatten(0, 1) - x.flatten(0, 1).mean(0)
    covariance = centered.T @ centered / len(centered)
    spectrum, vectors = torch.linalg.eigh(covariance)
    spectrum = spectrum.clamp_min(0).flip(0)
    vectors = vectors.flip(1)
    rank = int(torch.searchsorted(spectrum.cumsum(0) / spectrum.sum(), cfg["input_energy"])) + 1
    basis = vectors[:, :rank]
    native, j, raw, energy = derivative(block, x, basis, u, head, 1)
    proxy = rescale(original, anchor, cfg["beta"])
    px = capture_blocks(proxy, images, [layer], details=True)[layer]["attention_input"]
    px = px.reshape(len(images), 50, 768).double().cuda()
    proxy_gram, _pj, _pr, _pe = derivative(
        proxy.model.visual.transformer.resblocks[layer], px, basis, u, head, cfg["beta"]
    )
    rho = cfg["ridge_trace_fraction"] * float(native.trace()) / rank
    denominator = native.cpu().numpy() + rho * np.eye(rank)
    eigenvalues, eigenvectors = eigh(proxy_gram.cpu().numpy(), denominator)
    coefficients = torch.from_numpy(eigenvectors[:, -1].copy()).cuda()
    coefficients /= coefficients.norm()
    v = basis @ coefficients
    if v[v.abs().argmax()] < 0:
        v = -v
        coefficients = -coefficients
    qgain = raw[0][:, head, 0] @ u / 8
    scores = qgain[:, None] * (x @ v)
    scores -= scores.mean(1, keepdim=True)
    predicted_rms = float((coefficients @ native @ coefficients).sqrt())
    gain = min(
        cfg["local_relative_response_budget"] * energy**0.5 / predicted_rms,
        cfg["maximum_centered_logit_change"] / float(scores.abs().max()),
    )
    update = gain * u[:, None] @ v[None, :]
    release = copy.deepcopy(original).cpu().eval()
    key = f"model.visual.transformer.resblocks.{layer}.attn.in_proj_weight"
    weight = release.state_dict()[key]
    weight[768 + head * 64 : 768 + (head + 1) * 64].add_(update.float().cpu())
    edits = {key: weight.clone()}
    assert all(
        torch.equal(value, release.state_dict()[name])
        for name, value in original.state_dict().items()
        if name != key
    )
    save_json(out / "constructor_indices.json", bundle[6]["calibration"][: len(images)])
    proof = dict(
        basis=basis.cpu(),
        u=u.cpu(),
        v=v.cpu(),
        coefficients=coefficients.cpu(),
        native=native.cpu(),
        proxy=proxy_gram.cpu(),
        covariance=covariance.cpu(),
        gain=gain,
        rho=rho,
        head=head,
        layer=layer,
        inputs=x.cpu(),
        proxy_inputs=px.cpu(),
    )
    torch.save(proof, out / "artifacts/geometry.pt")
    changed = copy.deepcopy(block).cpu().eval()
    changed.attn.in_proj_weight.copy_(weight)
    after = responses(changed, x)[3][:, head, 0]
    before = maps[:, head, 0]
    output = block.attn.out_proj.weight[:, head * 64 : (head + 1) * 64].double().cuda()
    actual = weighted_sum(after - before, values[:, head]) @ output.T
    first = gain * (j @ coefficients)
    save_json(
        out / "construction.json",
        dict(
            head=head,
            layer=layer,
            input_rank=rank,
            retained_input_energy=float(spectrum[:rank].sum() / spectrum.sum()),
            direction_input_energy=float(v @ covariance @ v),
            gain=gain,
            regularized_local_proxy_ratio=float(eigenvalues[-1]),
            actual_native_attention_relative_rms=float(
                actual.norm() / (weighted_sum(before, values[:, head]) @ output.T).norm()
            ),
            first_order_native_relative_rms=float(
                first.norm() / (weighted_sum(before, values[:, head]) @ output.T).norm()
            ),
            first_order_remainder_relative_to_actual=float((actual - first).norm() / actual.norm()),
            edited_inside_retained_input_span=True,
            input_span_projection_preserves_entire_edit=True,
            no_fusion_or_security_claim=True,
        ),
    )
    pcfg = dict(pcfg, calibration_samples=len(images))
    return pcfg, bundle, release, edits


def frozen_inputs(cfg, out):
    pcfg, bundle, _old, _old_edits = load_inputs(cfg)
    release = copy.deepcopy(bundle[3]["original"]).cpu().eval()
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for k, v in edits.items():
        release.state_dict()[k].copy_(v)
    return pcfg, bundle, release, edits


def reconstruct(original, pcfg, p):
    u, v, basis, c = (p[k].numpy() for k in ("u", "v", "basis", "coefficients"))
    assert np.linalg.norm(basis @ c - v) < 1e-10
    assert np.linalg.norm(basis.T @ basis - np.eye(len(c))) < 1e-9
    a, b = p["proxy"].numpy(), p["native"].numpy() + p["rho"] * np.eye(len(c))
    ev, vec = eigh(a, b)
    ratio = float((c @ a @ c) / (c @ b @ c))
    assert abs(ratio - ev[-1]) < max(1e-9, abs(ev[-1]) * 1e-8)
    residual = float(np.linalg.norm(a @ c - ratio * b @ c) / max(np.linalg.norm(a @ c), 1e-30))
    assert residual < 1e-8
    layer, head = p["layer"], p["head"]
    key = f"model.visual.transformer.resblocks.{layer}.attn.in_proj_weight"
    old = original.state_dict()[key].numpy().copy()
    section = slice(768 + head * 64, 768 + (head + 1) * 64)
    old[section] += np.asarray(p["gain"] * u[:, None] @ v[None, :], dtype=np.float32)
    assert (
        np.linalg.norm((u[:, None] @ v[None, :]) @ basis @ basis.T - u[:, None] @ v[None, :]) < 1e-8
    )
    source = original.model.visual.transformer.resblocks[layer].attn
    x = p["inputs"].numpy()[:8]
    w, bias = (
        source.in_proj_weight.detach().numpy().astype(np.float64),
        source.in_proj_bias.detach().numpy().astype(np.float64),
    )
    response = x @ w.T + bias
    q, k, value = np.split(response, 3, axis=-1)
    q, k, value = (t[:, :, head * 64 : (head + 1) * 64] for t in (q, k, value))
    scores = np.einsum("nd,ntd->nt", q[:, 0], k) / 8

    def softmax(a):
        a = a - a.max(1, keepdims=True)
        a = np.exp(a)
        return a / a.sum(1, keepdims=True)

    a = softmax(scores)
    t = (q[:, 0] @ u)[:, None] * (x @ v) / 8
    centered_value = value - np.einsum("nt,ntd->nd", a, value)[:, None]
    output = (
        source.out_proj.weight[:, head * 64 : (head + 1) * 64].detach().numpy().astype(np.float64)
    )
    analytical = np.einsum("nt,ntd->nd", a * t, centered_value) @ output.T
    eps = 1e-4
    finite = (
        np.einsum(
            "nt,ntd->nd", (softmax(scores + eps * t) - softmax(scores - eps * t)) / (2 * eps), value
        )
        @ output.T
    )
    fd_error = float(np.linalg.norm(finite - analytical) / np.linalg.norm(analytical))
    assert fd_error < 1e-6
    return {key: torch.from_numpy(old)}, [
        dict(
            numpy_generalized_eigen_relative_residual=residual,
            numpy_top_ratio=ratio,
            numpy_softmax_value_derivative_relative_error=fd_error,
            edit_survives_input_span_projection=True,
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/value_conditioned_routing.json")
    parser.add_argument("--output", default="outputs/2026-10-12/value_conditioned_routing")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    if args.validate:
        validate(
            args,
            loader=lambda c: frozen_inputs(c, out),
            reconstruct=reconstruct,
            geometry_path=out / "artifacts/geometry.pt",
        )
    else:
        run(args, loader=lambda c: build_inputs(c, out))
