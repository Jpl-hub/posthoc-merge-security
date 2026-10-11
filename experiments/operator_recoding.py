"""One target-only lossy QK/VO/neuron recoding on a screened region.

QK uses a data-weighted bilinear rank approximation, not independent response
PCA. Reuse block capture, neuron PCA, ridge solve and official evaluation.
Compression or factor recoding alone is not a security or novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest
from joint_lossy_recoding import inputs, principal_frame
from joint_posthoc import capture_blocks
from local_erasure import cache
from operator_diagnostics import paired_counts, replace_components
from path_subspace import read
from polynomial_probe import ridge_update
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


def moment_roots(values, floor):
    moment = values.T @ values / len(values)
    eigenvalues, vectors = torch.linalg.eigh(moment)
    shift = floor * eigenvalues.mean()
    spectrum = (eigenvalues + shift).clamp_min(shift)
    root = (vectors * spectrum.sqrt()) @ vectors.T
    inverse = (vectors * spectrum.rsqrt()) @ vectors.T
    return root, inverse, float(shift)


def qk_factors(q, k, rank, floor):
    """Minimize the ridge-stabilized independent-token bilinear error metric."""
    rq, iq, sq = moment_roots(q, floor)
    rk, ik, sk = moment_roots(k, floor)
    u, singular, vh = torch.linalg.svd(rq @ rk)
    scale = singular[:rank].sqrt()
    a, b = iq @ (u[:, :rank] * scale), ik @ (vh[:rank].T * scale)
    return (
        a,
        b,
        dict(
            q_moment_ridge=sq,
            k_moment_ridge=sk,
            metric_relative_tail=float(singular[rank:].norm() / singular.norm()),
            rank=rank,
            dimension=q.shape[1],
        ),
    )


def qk_error(q, k, a, b, images):
    tokens, dim = len(q) // images, q.shape[1]
    q, k = q.reshape(images, tokens, dim), k.reshape(images, tokens, dim)
    original = q @ k.transpose(1, 2) / dim**0.5
    encoded = (q @ a) @ (k @ b).transpose(1, 2) / dim**0.5
    return dict(
        calibration_score_relative_error=float((encoded - original).norm() / original.norm()),
        calibration_softmax_relative_error=float(
            (encoded.softmax(-1) - original.softmax(-1)).norm() / original.softmax(-1).norm()
        ),
    )


@torch.no_grad()
def build(original, images, cfg, frames=None, solver="auto"):
    candidate = copy.deepcopy(original).cpu().eval()
    teacher = capture_blocks(original, images, cfg["layers"], details=True)
    saved, rows = {}, []
    for layer in cfg["layers"]:
        block = candidate.model.visual.transformer.resblocks[layer]
        source = original.model.visual.transformer.resblocks[layer]
        width, dim = source.attn.embed_dim, source.attn.head_dim
        x = teacher[layer]["attention_input"].double().cuda()
        response = torch.nn.functional.linear(
            x, source.attn.in_proj_weight.double().cuda(), source.attn.in_proj_bias.double().cuda()
        )
        attention, maps, projectors = [], [], []
        for head in range(source.attn.num_heads):
            section = slice(head * dim, (head + 1) * dim)
            q = response[:, section]
            k = response[:, width + section.start : width + section.stop]
            value = response[:, 2 * width + section.start : 2 * width + section.stop]
            if frames is None:
                a, b, evidence = qk_factors(q, k, cfg["qk_rank"], cfg["moment_ridge"])
                frame, _ = principal_frame(value, cfg["value_rank_per_head"])
                projector = frame @ frame.T
            else:
                a, b = (v.cuda() for v in frames[str(layer)]["qk_maps"][head])
                projector = frames[str(layer)]["value_projectors"][head].cuda()
                evidence = dict(rank=a.shape[1], dimension=dim)
            evidence.update(qk_error(q, k, a, b, len(images)))
            assert float((projector @ projector - projector).norm()) < 1e-9
            for offset, factor in [(0, a), (width, b)]:
                region = slice(offset + section.start, offset + section.stop)
                w = factor.T @ source.attn.in_proj_weight[region].double().cuda()
                bias = factor.T @ source.attn.in_proj_bias[region].double().cuda()
                block.attn.in_proj_weight[region].zero_()
                block.attn.in_proj_bias[region].zero_()
                active = slice(region.start, region.start + cfg["qk_rank"])
                block.attn.in_proj_weight[active].copy_(w.float().cpu())
                block.attn.in_proj_bias[active].copy_(bias.float().cpu())
            region = slice(2 * width + section.start, 2 * width + section.stop)
            block.attn.in_proj_weight[region].copy_(
                (projector @ source.attn.in_proj_weight[region].double().cuda()).float().cpu()
            )
            block.attn.in_proj_bias[region].copy_(
                (projector @ source.attn.in_proj_bias[region].double().cuda()).float().cpu()
            )
            block.attn.out_proj.weight[:, section].copy_(
                (source.attn.out_proj.weight[:, section].double().cuda() @ projector).float().cpu()
            )
            evidence["head"] = head
            evidence["removed_value_energy_fraction"] = float(
                (value - value @ projector).square().sum() / value.square().sum()
            )
            attention.append(evidence)
            maps.append(torch.stack([a.cpu(), b.cpu()]))
            projectors.append(projector.cpu())
        del x, response
        mlp = copy.deepcopy(source.mlp).double().cuda().eval()
        x = teacher[layer]["mlp_input"].double().cuda()
        preactivation = mlp.c_fc(x)
        hidden = mlp.gelu(preactivation)
        if frames is None:
            frame, neural = principal_frame(hidden, cfg["active_neurons"])
        else:
            frame = frames[str(layer)]["neuron_frame"].cuda()
            neural = dict(rank=frame.shape[1], dimension=frame.shape[0])
        assert float((frame.T @ frame - torch.eye(frame.shape[1], device="cuda")).norm()) < 1e-8
        projected = hidden @ frame
        neural.update(
            removed_hidden_energy_fraction=float(
                (hidden - projected @ frame.T).square().sum() / hidden.square().sum()
            ),
            nonlinear_commutation_relative_error=float(
                (mlp.gelu(preactivation @ frame) - projected).norm() / projected.norm()
            ),
        )
        affine = torch.cat([mlp.c_fc.weight, mlp.c_fc.bias[:, None]], dim=1)
        code = frame.T @ affine
        active = cfg["active_neurons"]
        block.mlp.c_fc.weight.zero_()
        block.mlp.c_fc.bias.zero_()
        block.mlp.c_fc.weight[:active].copy_(code[:, :-1].float().cpu())
        block.mlp.c_fc.bias[:active].copy_(code[:, -1].float().cpu())
        initial = torch.cat([(mlp.c_proj.weight @ frame).T, mlp.c_proj.bias[None, :]], dim=0)
        del mlp, x, preactivation, hidden, projected, affine
        current = capture_blocks(candidate, images, [layer], details=True)[layer]
        x = current["mlp_input"].double().cuda()
        encoded = torch.nn.functional.linear(
            x,
            block.mlp.c_fc.weight[:active].double().cuda(),
            block.mlp.c_fc.bias[:active].double().cuda(),
        )
        phi = block.mlp.gelu(encoded)
        design = torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)
        target = (
            teacher[layer]["output"].double().cuda() - current["post_attention"].double().cuda()
        )
        update, fit = ridge_update(
            design, target - design @ initial, cfg["ridge_ratio"], solver=solver
        )
        fitted = initial + update
        block.mlp.c_proj.weight.zero_()
        block.mlp.c_proj.weight[:, :active].copy_(fitted[:-1].T.float().cpu())
        block.mlp.c_proj.bias.copy_(fitted[-1].float().cpu())
        error = design @ fitted - target
        rows.append(
            dict(
                layer=layer,
                attention=attention,
                neurons=neural,
                original_block_relative_error=float(
                    error.norm() / teacher[layer]["output"].double().cuda().norm()
                ),
                **fit,
            )
        )
        saved[str(layer)] = dict(
            qk_maps=torch.stack(maps),
            value_projectors=torch.stack(projectors),
            neuron_frame=frame.cpu(),
            fitted_decoder=fitted.cpu(),
        )
        print("OPERATOR_RECODE_LAYER", layer, rows[-1]["original_block_relative_error"], flush=True)
        del x, encoded, phi, design, target, update, fitted, error, current, frame, initial
    return candidate, saved, rows


def preflight():
    generator = torch.Generator().manual_seed(89)
    q, k = (torch.randn(70, 8, generator=generator, dtype=torch.float64) for _ in range(2))
    a, b, metric = qk_factors(q, k, 4, 1e-8)
    rq, _, _ = moment_roots(q, 1e-8)
    rk, _, _ = moment_roots(k, 1e-8)
    residual = rq @ (torch.eye(8, dtype=torch.float64) - a @ b.T) @ rk
    expected = torch.linalg.svdvals(rq @ rk)[4:].norm()
    error = float(abs(residual.norm() - expected) / expected)
    assert error < 1e-12
    padded_q, padded_k = torch.zeros_like(q), torch.zeros_like(k)
    padded_q[:, :4], padded_k[:, :4] = q @ a, k @ b
    compilation = float(((padded_q @ padded_k.T) - q @ a @ b.T @ k.T).norm())
    assert compilation < 1e-12
    return dict(
        weighted_best_rank_residual_relative_error=error,
        padded_qk_compilation_absolute_error=compilation,
        unchanged_attention_denominator=8**0.5,
        independent_token_metric_not_actual_attention_bound=True,
        **metric,
    )


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    save_json(out / "preflight.json", preflight())
    root, _, fusion, models, head, data, indices, images = inputs(cfg)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    save_json(out / "training_indices.json", indices)
    original = models["original"]
    release, frames, fitting = build(original, images, cfg)
    old = original.state_dict()
    edits = {k: v.clone() for k, v in release.state_dict().items() if not torch.equal(v, old[k])}
    public, support = replace_components(original, models["anchor"], cfg["layers"], "attn_mlp")
    erased, _ = replace_components(release, models["anchor"], cfg["layers"], "attn_mlp")
    assert all(torch.equal(v, erased.state_dict()[k]) for k, v in public.state_dict().items())
    assert set(edits) <= set(support) and len(edits) > 0
    del public, erased
    torch.save(edits, artifacts / "release_edits.pt")
    torch.save(frames, artifacts / "frames.pt")
    save_json(out / "fitting.json", fitting)
    save_json(
        out / "release_freeze.json",
        dict(
            modified_keys=sorted(edits),
            release_edits_sha256=digest(artifacts / "release_edits.pt"),
            frames_sha256=digest(artifacts / "frames.pt"),
            public_full_support_reset_states_identical=True,
            construction_uses_future_partner=False,
            source_manifest=read(args.manifest)["source_manifest"],
        ),
    )
    raw, rows = {}, []
    for split, loader in [
        ("confirmation", make_loader(data.train_dataset, indices["confirmation"], 128)),
        ("official_test", cache(data)),
    ]:
        for preset, model in [("original", original), ("release", release)]:
            p, y = [], []
            score = evaluate(
                model, head, loader, "cuda", prediction_sink=lambda a, b: (p.append(a), y.append(b))
            )
            key = f"{preset}/{split}"
            raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
            row = dict(key=key, preset=preset, split=split, **score)
            if preset == "release":
                reference = raw[f"original/{split}"]
                row["paired"] = paired_counts(
                    reference["prediction"].eq(reference["labels"]),
                    raw[key]["prediction"].eq(raw[key]["labels"]),
                )
            rows.append(row)
            print("OPERATOR_RECODE_UTILITY", key, score["correct"], score["n"], flush=True)
        save_json(out / "results.json", rows)
        torch.save(raw, artifacts / "predictions.pt")
    assert rows[-2]["n"] == fusion["expected_test_counts"][cfg["target"]]
    parent = Path(cfg["baseline_parent"])
    meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert digest(parent / "artifacts/predictions.pt") == meta["raw_sha256"]["predictions.pt"]
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)[
        f"original/standalone/{cfg['target']}"
    ]
    assert all(torch.equal(prior[k], raw["original/official_test"][k]) for k in prior)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            full_evaluations=2,
            confirmation_evaluations=2,
            closed_form_fits=3,
            extra_gradient_steps=0,
            calibration_ground_truth_used=False,
            historical_region_selection_used_target_labels=True,
            construction_uses_future_partner=False,
            original_full_prediction_array_identical=True,
            defense_success_claim=False,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / cfg["target"] / "finetuned.pt",
                    root / f"head_{cfg['target']}.pt",
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


@torch.no_grad()
def validate(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for p, sha in meta["source_manifest"].items():
        assert digest(p) == sha, p
    for name, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == sha
    _, _, _, models, head, data, indices, images = inputs(cfg)
    assert indices == read(out / "training_indices.json")
    frames = torch.load(out / "artifacts/frames.pt", map_location="cpu", weights_only=True)
    replay, _, fitting = build(models["original"], images, cfg, frames=frames, solver="primal")
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    release = copy.deepcopy(models["original"]).cpu().eval()
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    difference = max(
        float((v - replay.state_dict()[k]).abs().max()) for k, v in release.state_dict().items()
    )
    assert difference < 5e-6, difference
    assert all(r["normal_equation_relative_residual"] < 1e-8 for r in fitting)
    a, _ = replace_components(models["original"], models["anchor"], cfg["layers"], "attn_mlp")
    b, _ = replace_components(release, models["anchor"], cfg["layers"], "attn_mlp")
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 4
    for row in rows:
        v = raw[row["key"]]
        official = row["split"] == "official_test"
        labels = (
            truth(data, cfg["target"])
            if official
            else torch.tensor([data.train_dataset[i][1] for i in indices["confirmation"]])
        )
        assert torch.equal(labels, v["labels"])
        assert row["n"] == len(labels) and row["correct"] == int(v["prediction"].eq(labels).sum())
        assert row["top1"] == row["correct"] / row["n"]
        if row["preset"] == "release":
            reference = raw[f"original/{row['split']}"]
            assert row["paired"] == paired_counts(
                reference["prediction"].eq(labels), v["prediction"].eq(labels)
            )
        sink = []
        dataset = data.test_dataset if official else data.train_dataset
        ids = list(range(128)) if official else indices["confirmation"][:128]
        evaluate(
            models["original"] if row["preset"] == "original" else release,
            head,
            make_loader(dataset, ids, 128),
            "cuda",
            prediction_sink=lambda p, _y: sink.append(p),
        )
        assert torch.equal(torch.cat(sink), v["prediction"][:128])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            three_primal_fits_recomputed_from_frozen_frames=True,
            compiled_weight_max_absolute_error=difference,
            counts_pairs_and_prefixes_checked=4,
            public_full_support_state_identity_checked=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            defense_success_claim=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/operator_recoding.json")
    parser.add_argument("--output", default="outputs/2026-10-11/operator_recoding_cars")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
