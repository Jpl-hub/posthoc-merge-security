"""Context/output-weighted VO recoding; keep QK and nonlinear MLP parameters.

Reuse the existing weighted-rank primitive, original attention extraction,
pinned NUFILT SVD, and native utility/validation runner. This is a calibrated
mechanism test, not a new compression algorithm or established defense.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from head_collapse import attention, factor
from joint_posthoc import capture_blocks
from nufilt_alignment import author_svd
from operator_attention_ablation import load_inputs, run, validate
from path_subspace import read
from subspace_probe import save_json
from task_relocation import head_features


@torch.no_grad()
def build_inputs(cfg, out):
    pcfg, bundle, _old_release, _old_edits = load_inputs(cfg)
    original, images = bundle[3]["original"], bundle[-1]
    captured = capture_blocks(original, images, pcfg["layers"], details=True)
    svd = author_svd("external/NUFILT", read(cfg["nufilt_config"]))
    release = copy.deepcopy(original).cpu().eval()
    geometry, rows = {}, []
    for layer in pcfg["layers"]:
        source = original.model.visual.transformer.resblocks[layer]
        target = release.model.visual.transformer.resblocks[layer]
        x = captured[layer]["attention_input"].reshape(len(images), -1, 768).cuda()
        maps, _v, _contexts = attention(source, x)
        geometry[str(layer)] = {}
        for head in range(source.attn.num_heads):
            # A_h X is the actual attended input of this head. Averaging bias
            # commutes with attention because each softmax row sums to one.
            virtual = maps[:, head].double() @ x.double()
            if layer == pcfg["layers"][-1]:
                virtual = virtual[:, :1]
            b, d, evidence, proof = factor(
                source, virtual, (head,), cfg["rank"], cfg["covariance_cutoff"], svd
            )
            rank, dim, width = cfg["rank"], source.attn.head_dim, source.attn.embed_dim
            section = slice(head * dim, (head + 1) * dim)
            value_rows = slice(2 * width + section.start, 2 * width + section.stop)
            target.attn.in_proj_weight[value_rows].zero_()
            target.attn.in_proj_bias[value_rows].zero_()
            active = slice(value_rows.start, value_rows.start + rank)
            target.attn.in_proj_weight[active].copy_(b[:, :-1].float().cpu())
            target.attn.in_proj_bias[active].copy_(b[:, -1].float().cpu())
            target.attn.out_proj.weight[:, section].zero_()
            target.attn.out_proj.weight[:, section.start : section.start + rank].copy_(
                d.float().cpu()
            )
            geometry[str(layer)][str(head)] = proof
            rows.append(
                dict(layer=layer, head=head, relevant_tokens_per_image=virtual.shape[1], **evidence)
            )
        del x, maps, _v, _contexts
    old = original.state_dict()
    edits = {k: v.clone() for k, v in release.state_dict().items() if not torch.equal(v, old[k])}
    for layer in pcfg["layers"]:
        block = release.model.visual.transformer.resblocks[layer]
        source = original.model.visual.transformer.resblocks[layer]
        assert torch.equal(block.attn.in_proj_weight[:1536], source.attn.in_proj_weight[:1536])
        assert torch.equal(block.attn.in_proj_bias[:1536], source.attn.in_proj_bias[:1536])
    assert all(torch.equal(v, release.state_dict()[k]) for k, v in old.items() if k not in edits)
    assert len(edits) == 9 and all(".attn." in k for k in edits)
    torch.save(geometry, out / "artifacts/geometry.pt")
    save_json(out / "construction.json", rows)
    return pcfg, bundle, release, edits


def frozen_inputs(cfg, out):
    pcfg, bundle, _old_release, _old_edits = load_inputs(cfg)
    release = copy.deepcopy(bundle[3]["original"]).cpu().eval()
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    return pcfg, bundle, release, edits


@torch.no_grad()
def reconstruct_weighted(original, pcfg, geometry, cfg, images):
    captured = capture_blocks(original, images, pcfg["layers"], details=True)
    rebuilt, evidence = {}, []
    for layer in pcfg["layers"]:
        source = original.model.visual.transformer.resblocks[layer]
        z = head_features(source, captured[layer]["attention_input"], len(images)).reshape(
            len(images), -1, 768
        )
        w = source.attn.in_proj_weight.detach().numpy().copy()
        bias = source.attn.in_proj_bias.detach().numpy().copy()
        output = source.attn.out_proj.weight.detach().numpy().copy()
        for head in range(source.attn.num_heads):
            proof = geometry[str(layer)][str(head)]
            c, o, g = (
                proof[k].numpy().astype(np.float64) for k in ("value", "output", "covariance")
            )
            b, d = (proof[k].numpy().astype(np.float64) for k in ("b", "d"))
            section = slice(head * 64, (head + 1) * 64)
            response = z[:, :, section]
            if layer == pcfg["layers"][-1]:
                response = response[:, :1]
            response = response.double().flatten(0, 1)
            fresh = (response.T @ response / len(response)).numpy()
            covariance_error = float(np.linalg.norm(fresh - g) / np.linalg.norm(g))
            assert covariance_error < 3e-6
            eig, vectors = np.linalg.eigh(g)
            keep = eig > cfg["covariance_cutoff"] * eig.max()
            eig = np.where(keep, eig, 0)
            root = (vectors * np.sqrt(eig)) @ vectors.T
            inverse = (vectors * np.where(keep, 1 / np.sqrt(np.maximum(eig, 1e-30)), 0)) @ vectors.T
            u, s, vh = np.linalg.svd(o @ root, full_matrices=False)
            rank = cfg["rank"]
            operator = (u[:, :rank] * s[:rank]) @ vh[:rank] @ inverse
            wanted = operator @ c
            error = float(np.linalg.norm(d @ b - wanted) / np.linalg.norm(wanted))
            assert error < 1e-8
            original_operator = np.c_[
                source.attn.in_proj_weight[1536 + head * 64 : 1536 + (head + 1) * 64]
                .detach()
                .numpy(),
                source.attn.in_proj_bias[1536 + head * 64 : 1536 + (head + 1) * 64]
                .detach()
                .numpy(),
            ]
            assert np.array_equal(original_operator.astype(np.float64), c)
            assert np.array_equal(
                source.attn.out_proj.weight[:, section].detach().numpy().astype(np.float64), o
            )
            value_rows = slice(1536 + head * 64, 1536 + (head + 1) * 64)
            w[value_rows] = 0
            bias[value_rows] = 0
            output[:, section] = 0
            w[value_rows.start : value_rows.start + rank] = b[:, :-1]
            bias[value_rows.start : value_rows.start + rank] = b[:, -1]
            output[:, section.start : section.start + rank] = d
            residual = o - operator
            declared = float(np.linalg.norm(residual @ root) ** 2)
            optimum = float((s[rank:] ** 2).sum())
            assert abs(declared - optimum) < max(1e-8, optimum * 1e-8)
            evidence.append(
                dict(
                    layer=layer,
                    head=head,
                    fresh_native_kernel_covariance_relative_error=covariance_error,
                    numpy_effective_vo_operator_relative_error=error,
                    weighted_mse=declared,
                    optimal_tail_mse=optimum,
                )
            )
        prefix = f"model.visual.transformer.resblocks.{layer}.attn."
        rebuilt[prefix + "in_proj_weight"] = torch.from_numpy(w)
        rebuilt[prefix + "in_proj_bias"] = torch.from_numpy(bias)
        rebuilt[prefix + "out_proj.weight"] = torch.from_numpy(output)
    return rebuilt, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weighted_attention.json")
    parser.add_argument("--output", default="outputs/2026-10-12/weighted_attention")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    if args.validate:
        cfg = read(args.config)
        _pcfg, bundle, _release, _edits = frozen_inputs(cfg, out)
        validate(
            args,
            loader=lambda c: frozen_inputs(c, out),
            reconstruct=lambda o, p, g: reconstruct_weighted(o, p, g, cfg, bundle[-1]),
            geometry_path=out / "artifacts/geometry.pt",
        )
    else:
        run(args, loader=lambda c: build_inputs(c, out))
