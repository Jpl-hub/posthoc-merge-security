"""One calibrated local lossy Attention/MLP recoding and utility screen.

Reuse block capture, ridge solve and author evaluation. Rank reduction is not
a security proof; privileged frame completion is labelled separately.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import reset
from eight_task_benchmark import digest
from joint_posthoc import capture_blocks
from local_erasure import cache, sites
from operator_diagnostics import paired_counts
from path_subspace import context, read
from polynomial_probe import ridge_update
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


@torch.no_grad()
def principal_frame(values, rank):
    moment = values.T @ values / len(values)
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    frame = eigenvectors[:, -rank:]
    columns = torch.arange(rank, device=frame.device)
    signs = frame[frame.abs().argmax(0), columns].sign()
    frame = frame * signs[None, :]
    total = eigenvalues.sum().clamp_min(1e-30)
    return frame, dict(
        dimension=values.shape[1],
        rank=rank,
        retained_uncentered_energy=float(eigenvalues[-rank:].sum() / total),
        orthogonality_error=float((frame.T @ frame - torch.eye(rank, device="cuda")).norm()),
    )


@torch.no_grad()
def build(original, images, cfg, supplied_frames=None):
    candidate = copy.deepcopy(original).cpu().eval()
    teacher = capture_blocks(original, images, cfg["layers"], details=True)
    frames, fitting = {}, []
    for layer in cfg["layers"]:
        block = candidate.model.visual.transformer.resblocks[layer]
        source = original.model.visual.transformer.resblocks[layer]
        width, head_dim = source.attn.embed_dim, source.attn.head_dim
        x = teacher[layer]["attention_input"].double().cuda()
        value = torch.nn.functional.linear(
            x,
            source.attn.in_proj_weight[2 * width :].double().cuda(),
            source.attn.in_proj_bias[2 * width :].double().cuda(),
        )
        projectors, attention_evidence = [], []
        for head in range(source.attn.num_heads):
            section = slice(head * head_dim, (head + 1) * head_dim)
            if supplied_frames is None:
                frame, _ = principal_frame(value[:, section], cfg["value_rank_per_head"])
                projector = frame @ frame.T
            else:
                projector = supplied_frames[str(layer)]["value_projectors"][head].cuda()
            error = float((projector @ projector - projector).norm())
            assert error < 1e-9
            residual = value[:, section] @ (torch.eye(head_dim, device="cuda") - projector)
            attention_evidence.append(
                dict(
                    head=head,
                    rank=int(torch.linalg.matrix_rank(projector, atol=1e-8, rtol=0)),
                    projector_error=error,
                    removed_value_energy_fraction=float(
                        residual.square().sum() / value[:, section].square().sum()
                    ),
                )
            )
            rows = slice(2 * width + head * head_dim, 2 * width + (head + 1) * head_dim)
            block.attn.in_proj_weight[rows].copy_(
                (projector @ source.attn.in_proj_weight[rows].double().cuda()).float().cpu()
            )
            block.attn.in_proj_bias[rows].copy_(
                (projector @ source.attn.in_proj_bias[rows].double().cuda()).float().cpu()
            )
            block.attn.out_proj.weight[:, section].copy_(
                (source.attn.out_proj.weight[:, section].double().cuda() @ projector).float().cpu()
            )
            projectors.append(projector.cpu())
        del x, value
        x = teacher[layer]["mlp_input"].double().cuda()
        source_mlp = copy.deepcopy(source.mlp).double().cuda().eval()
        preactivation = source_mlp.c_fc(x)
        hidden = source_mlp.gelu(preactivation)
        if supplied_frames is None:
            frame, neural_evidence = principal_frame(hidden, cfg["active_neurons"])
        else:
            frame = supplied_frames[str(layer)]["neuron_frame"].cuda()
            neural_evidence = dict(rank=frame.shape[1], dimension=frame.shape[0])
        orthogonality = float((frame.T @ frame - torch.eye(frame.shape[1], device="cuda")).norm())
        assert orthogonality < 1e-8
        projected = hidden @ frame
        neural_evidence.update(
            orthogonality_error=orthogonality,
            removed_hidden_energy_fraction=float(
                (hidden - projected @ frame.T).square().sum() / hidden.square().sum()
            ),
            nonlinear_commutation_relative_error=float(
                (source_mlp.gelu(preactivation @ frame) - projected).norm()
                / projected.norm().clamp_min(1e-30)
            ),
        )
        affine = torch.cat([source_mlp.c_fc.weight, source_mlp.c_fc.bias[:, None]], dim=1)
        code = frame.T @ affine
        active = cfg["active_neurons"]
        block.mlp.c_fc.weight.zero_()
        block.mlp.c_fc.bias.zero_()
        block.mlp.c_fc.weight[:active].copy_(code[:, :-1].float().cpu())
        block.mlp.c_fc.bias[:active].copy_(code[:, -1].float().cpu())
        initial = torch.cat(
            [(source_mlp.c_proj.weight @ frame).T, source_mlp.c_proj.bias[None, :]], dim=0
        )
        del source_mlp, x, hidden, projected, preactivation, affine
        current = capture_blocks(candidate, images, [layer], details=True)[layer]
        x = current["mlp_input"].double().cuda()
        # Use compiled float weights for fitting, matching the deployed inputs.
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
        update, solve = ridge_update(design, target - design @ initial, cfg["ridge_ratio"])
        fitted = initial + update
        block.mlp.c_proj.weight.zero_()
        block.mlp.c_proj.weight[:, :active].copy_(fitted[:-1].T.float().cpu())
        block.mlp.c_proj.bias.copy_(fitted[-1].float().cpu())
        residual = design @ fitted - target
        fitting.append(
            dict(
                layer=layer,
                attention=attention_evidence,
                neurons=neural_evidence,
                residual_target_relative_error=float(residual.norm() / target.norm()),
                original_block_relative_error=float(
                    residual.norm() / teacher[layer]["output"].double().cuda().norm()
                ),
                **solve,
            )
        )
        frames[str(layer)] = dict(
            value_projectors=torch.stack(projectors),
            neuron_frame=frame.cpu(),
            initial_decoder=initial.cpu(),
            fitted_decoder=fitted.cpu(),
        )
        print("JOINT_RECODE_LAYER", layer, fitting[-1]["original_block_relative_error"], flush=True)
        del x, encoded, phi, design, target, update, fitted, residual, frame, initial, current
    return candidate, frames, fitting


@torch.no_grad()
def oracle_complete(release, anchor, cfg, frames):
    result = copy.deepcopy(release).cpu().eval()
    for layer in cfg["layers"]:
        block = result.model.visual.transformer.resblocks[layer]
        public = anchor.model.visual.transformer.resblocks[layer]
        width, dim = block.attn.embed_dim, block.attn.head_dim
        for head, projector in enumerate(frames[str(layer)]["value_projectors"]):
            complement = torch.eye(dim, dtype=torch.float64) - projector
            rows = slice(2 * width + head * dim, 2 * width + (head + 1) * dim)
            cols = slice(head * dim, (head + 1) * dim)
            block.attn.in_proj_weight[rows].add_(
                (complement @ public.attn.in_proj_weight[rows].double()).float()
            )
            block.attn.in_proj_bias[rows].add_(
                (complement @ public.attn.in_proj_bias[rows].double()).float()
            )
            block.attn.out_proj.weight[:, cols].add_(
                (public.attn.out_proj.weight[:, cols].double() @ complement).float()
            )
        frame = frames[str(layer)]["neuron_frame"]
        active = cfg["active_neurons"]
        code = torch.cat(
            [block.mlp.c_fc.weight[:active], block.mlp.c_fc.bias[:active, None]], dim=1
        ).double()
        old = torch.cat([public.mlp.c_fc.weight, public.mlp.c_fc.bias[:, None]], dim=1).double()
        completed = frame @ code + old - frame @ (frame.T @ old)
        decoder = block.mlp.c_proj.weight[:, :active].double()
        public_decoder = public.mlp.c_proj.weight.double()
        down = decoder @ frame.T + public_decoder - (public_decoder @ frame) @ frame.T
        block.mlp.c_fc.weight.copy_(completed[:, :-1].float())
        block.mlp.c_fc.bias.copy_(completed[:, -1].float())
        block.mlp.c_proj.weight.copy_(down.float())
    return result


def inputs(cfg):
    root, load, fusion, models = context(cfg)
    target = cfg["target"]
    models["original"] = load(str(root / target / "finetuned.pt")).cpu().eval()
    head = load(str(root / f"head_{target}.pt"))
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    for name in ("zeroshot.pt", f"{target}/finetuned.pt", f"head_{target}.pt"):
        assert digest(root / name) == expected[name], name
    from dataset.registry import get_dataset

    data = get_dataset(
        target, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2
    )
    prior = read(Path(cfg["profile_parent"]) / "training_indices.json")[target]
    indices = dict(
        calibration=prior["selection"][: cfg["calibration_samples"]],
        confirmation=prior["confirmation"],
    )
    assert len(indices["calibration"]) == cfg["calibration_samples"]
    assert not set(indices["calibration"]) & set(indices["confirmation"])
    images = torch.stack([data.train_dataset[i][0] for i in indices["calibration"]])
    return root, load, fusion, models, head, data, indices, images


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, _, fusion, models, head, data, indices, images = inputs(cfg)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    save_json(out / "training_indices.json", indices)
    release, frames, fitting = build(models["original"], images, cfg)
    old = models["original"].state_dict()
    edits = {k: v.clone() for k, v in release.state_dict().items() if not torch.equal(v, old[k])}
    allowed = {
        f"model.visual.transformer.resblocks.{layer}.{suffix}"
        for layer in cfg["layers"]
        for suffix in [
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ]
    }
    assert set(edits) <= allowed and len(edits) > 0
    for layer in cfg["layers"]:
        for suffix in ("weight", "bias"):
            key = f"model.visual.transformer.resblocks.{layer}.attn.in_proj_{suffix}"
            assert torch.equal(release.state_dict()[key][:1536], old[key][:1536])
    torch.save(edits, artifacts / "release_edits.pt")
    torch.save(frames, artifacts / "frames.pt")
    save_json(out / "fitting.json", fitting)
    models["release"] = release
    models["oracle_frame_completion"] = oracle_complete(release, models["anchor"], cfg, frames)
    erased = reset(release, models["anchor"], sites(cfg["layers"]), "public_support")
    control = reset(models["original"], models["anchor"], sites(cfg["layers"]), "public_support")
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in erased.state_dict().items())
    del erased, control
    # Save the release before evaluation. No partner expert has been loaded.
    save_json(
        out / "release_freeze.json",
        dict(
            release_edits_sha256=digest(artifacts / "release_edits.pt"),
            frames_sha256=digest(artifacts / "frames.pt"),
            modified_keys=sorted(edits),
            construction_uses_future_partner=False,
            public_support_reset_states_identical=True,
            source_manifest=read(args.manifest)["source_manifest"],
        ),
    )
    raw, rows = {}, []
    for split, loader in [
        ("confirmation", make_loader(data.train_dataset, indices["confirmation"], 128)),
        ("official_test", cache(data)),
    ]:
        for preset in ("original", "release", "oracle_frame_completion"):
            chunks, labels = [], []
            score = evaluate(
                models[preset],
                head,
                loader,
                "cuda",
                prediction_sink=lambda p, y: (chunks.append(p), labels.append(y)),
            )
            key = f"{preset}/{split}"
            raw[key] = dict(prediction=torch.cat(chunks), labels=torch.cat(labels))
            row = dict(
                key=key,
                preset=preset,
                split=split,
                oracle_frame_information=preset.startswith("oracle"),
                **score,
            )
            if preset != "original":
                reference = raw[f"original/{split}"]
                row["original_change"] = paired_counts(
                    reference["prediction"].eq(reference["labels"]),
                    raw[key]["prediction"].eq(raw[key]["labels"]),
                )
                row["prediction_disagreement"] = int(
                    raw[key]["prediction"].ne(reference["prediction"]).sum()
                )
            rows.append(row)
            print("JOINT_RECODE_UTILITY", key, score["correct"], score["n"], flush=True)
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
    assert rows[-3]["n"] == fusion["expected_test_counts"][cfg["target"]]
    parent = Path(cfg["baseline_parent"])
    parent_meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert (
        digest(parent / "artifacts/predictions.pt") == parent_meta["raw_sha256"]["predictions.pt"]
    )
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    reference = prior[f"{cfg['target']}/original/direct/standalone/{cfg['target']}"]
    assert all(
        torch.equal(v, raw["original/official_test"][k])
        for k, v in reference.items()
        if k in {"prediction", "labels"}
    )
    save_json(
        out / "baseline_check.json",
        dict(
            original_full_prediction_array_identical=True,
            parent_prediction_sha256=parent_meta["raw_sha256"]["predictions.pt"],
            conditional_support_reset_baseline_only=True,
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            full_evaluations=3,
            confirmation_evaluations=3,
            closed_form_fits=3,
            extra_gradient_steps=0,
            calibration_ground_truth_used=False,
            construction_uses_future_partner=False,
            oracle_completion_not_public=True,
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
            dataset_source_sha256={
                str(p): digest(p)
                for p in sorted(Path("external/secure-merge-attack/datasets").glob("*.py"))
            },
        ),
    )


@torch.no_grad()
def validate(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for path, sha in meta["source_manifest"].items():
        assert digest(path) == sha, path
    for name, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == sha
    root, _, _, models, head, data, indices, images = inputs(cfg)
    assert indices == read(out / "training_indices.json")
    for name, sha in meta["checkpoint_sha256"].items():
        assert digest(root / name) == sha
    frames = torch.load(out / "artifacts/frames.pt", map_location="cpu", weights_only=True)
    replay, _, fitting = build(models["original"], images, cfg, supplied_frames=frames)
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    release = copy.deepcopy(models["original"]).cpu().eval()
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    difference = max(
        float((value - replay.state_dict()[key]).abs().max())
        for key, value in release.state_dict().items()
    )
    assert difference < 5e-6, difference
    assert all(r["normal_equation_relative_residual"] < 1e-8 for r in fitting)
    models["release"] = release
    models["oracle_frame_completion"] = oracle_complete(release, models["anchor"], cfg, frames)
    erased = reset(release, models["anchor"], sites(cfg["layers"]), "public_support")
    control = reset(models["original"], models["anchor"], sites(cfg["layers"]), "public_support")
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in erased.state_dict().items())
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 6
    for row in rows:
        value = raw[row["key"]]
        expected = (
            truth(data, cfg["target"])
            if row["split"] == "official_test"
            else torch.tensor([data.train_dataset[i][1] for i in indices["confirmation"]])
        )
        assert torch.equal(value["labels"], expected)
        assert row["n"] == len(expected) and row["correct"] == int(
            value["prediction"].eq(expected).sum()
        )
        assert row["top1"] == row["correct"] / row["n"]
        if row["preset"] != "original":
            reference = raw[f"original/{row['split']}"]
            assert row["original_change"] == paired_counts(
                reference["prediction"].eq(expected), value["prediction"].eq(expected)
            )
        dataset = data.test_dataset if row["split"] == "official_test" else data.train_dataset
        ids = list(range(128)) if row["split"] == "official_test" else indices["confirmation"][:128]
        sink = []
        evaluate(
            models[row["preset"]],
            head,
            make_loader(dataset, ids, 128),
            "cuda",
            prediction_sink=lambda p, _y: sink.append(p),
        )
        assert torch.equal(torch.cat(sink), value["prediction"][:128])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            three_closed_form_fits_replayed=True,
            compiled_weight_max_absolute_error=difference,
            six_counts_and_prediction_prefixes_checked=True,
            public_support_state_identity_checked=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            no_defense_success_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_lossy_recoding.json")
    parser.add_argument("--output", default="outputs/2026-10-11/joint_lossy_recoding")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
