"""Fixed Attention-only ablation of failed Cars operator recoding.

The parent Attention tensors and calibration geometry are immutable. Restore
every MLP and LayerNorm to the original expert. No new rank choice, decoder
fit, gradient step, future partner, or method-novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest
from functional_contrast import features
from joint_lossy_recoding import inputs
from operator_diagnostics import paired_counts
from path_subspace import read
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


def load_inputs(cfg):
    parent = Path(cfg["parent"])
    pcfg = read(parent / "protocol.json")
    meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert pcfg["target"] == "Cars" and pcfg["layers"] == [9, 10, 11]
    for key, pin in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / key) == pin
    result = inputs(pcfg)
    original = result[3]["original"]
    parent_edits = torch.load(
        parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True
    )
    edits = {k: v for k, v in parent_edits.items() if ".attn." in k}
    expected = {
        f"model.visual.transformer.resblocks.{layer}.attn.{name}"
        for layer in pcfg["layers"]
        for name in ("in_proj_weight", "in_proj_bias", "out_proj.weight")
    }
    assert set(edits) == expected
    release = copy.deepcopy(original).cpu().eval()
    for k, v in edits.items():
        release.state_dict()[k].copy_(v)
    assert all(
        torch.equal(v, release.state_dict()[k])
        for k, v in original.state_dict().items()
        if k not in edits
    )
    assert all(torch.isfinite(v).all() for v in release.state_dict().values())
    return pcfg, result, release, edits


def reconstruct_attention(original, pcfg, frames):
    """Independent FP64 NumPy reconstruction from the frozen QK/VO geometry."""
    result = {}
    errors = []
    for layer in pcfg["layers"]:
        source = original.model.visual.transformer.resblocks[layer].attn
        incoming = source.in_proj_weight.detach().numpy().astype(np.float64)
        bias = source.in_proj_bias.detach().numpy().astype(np.float64)
        outgoing = source.out_proj.weight.detach().numpy().astype(np.float64)
        new_in, new_bias, new_out = incoming.copy(), bias.copy(), outgoing.copy()
        width, dim = source.embed_dim, source.head_dim
        for head in range(source.num_heads):
            section = slice(head * dim, (head + 1) * dim)
            a, b = frames[str(layer)]["qk_maps"][head].numpy()
            p = frames[str(layer)]["value_projectors"][head].numpy()
            assert a.shape == b.shape == (dim, pcfg["qk_rank"])
            idempotence = float(np.linalg.norm(p @ p - p))
            assert idempotence < 1e-9
            for offset, transform in ((0, a), (width, b)):
                rows = slice(offset + section.start, offset + section.stop)
                active = slice(rows.start, rows.start + pcfg["qk_rank"])
                new_in[rows] = 0
                new_bias[rows] = 0
                new_in[active] = transform.T @ incoming[rows]
                new_bias[active] = transform.T @ bias[rows]
            value_rows = slice(2 * width + section.start, 2 * width + section.stop)
            new_in[value_rows] = p @ incoming[value_rows]
            new_bias[value_rows] = p @ bias[value_rows]
            new_out[:, section] = outgoing[:, section] @ p
            # Check the actual composed bilinear and value-output operators,
            # including Q/K affine terms, rather than singular-value labels.
            q = np.c_[incoming[section], bias[section]]
            key_rows = slice(width + section.start, width + section.stop)
            k = np.c_[incoming[key_rows], bias[key_rows]]
            qn = np.c_[new_in[section], new_bias[section]]
            kn = np.c_[new_in[key_rows], new_bias[key_rows]]
            score_error = float(np.max(np.abs(qn.T @ kn - q.T @ a @ b.T @ k)))
            vo_error = float(
                np.max(
                    np.abs(
                        new_out[:, section] @ new_in[value_rows]
                        - outgoing[:, section] @ p @ incoming[value_rows]
                    )
                )
            )
            assert score_error < 1e-10 and vo_error < 1e-10
            errors.append(
                dict(
                    layer=layer,
                    head=head,
                    qk_absolute_error=score_error,
                    vo_absolute_error=vo_error,
                    projector_error=idempotence,
                )
            )
        prefix = f"model.visual.transformer.resblocks.{layer}.attn."
        result[prefix + "in_proj_weight"] = torch.from_numpy(new_in.astype(np.float32))
        result[prefix + "in_proj_bias"] = torch.from_numpy(new_bias.astype(np.float32))
        result[prefix + "out_proj.weight"] = torch.from_numpy(new_out.astype(np.float32))
    return result, errors


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    start = time.perf_counter()
    pcfg, bundle, release, edits = load_inputs(cfg)
    _root, _load, fusion, models, head, data, indices, _images = bundle
    # The subset release is frozen before confirmation or test evaluation.
    torch.save(edits, artifacts / "release_edits.pt")
    save_json(out / "protocol.json", cfg)
    save_json(out / "training_indices.json", indices)
    save_json(
        out / "release_freeze.json",
        dict(
            release_sha256=digest(artifacts / "release_edits.pt"),
            parent_release_sha256=digest(Path(cfg["parent"]) / "artifacts/release_edits.pt"),
            edited_keys=sorted(edits),
            all_original_mlp_and_layernorm_tensors_preserved=True,
            additional_calibration_or_fitting=False,
            future_partner_used=False,
            gradient_steps=0,
            response_fits=0,
        ),
    )
    original = models["original"]
    raw, rows = {}, []
    for preset, model in (("original", original), ("attention_only", release)):
        for split, loader in (
            ("confirmation", make_loader(data.train_dataset, indices["confirmation"])),
            ("official_test", data.test_loader),
        ):
            predictions, labels = [], []
            result = evaluate(
                model,
                head,
                loader,
                "cuda",
                prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
            )
            key = f"{preset}/{split}"
            raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(labels))
            if split == "official_test":
                assert result["n"] == fusion["expected_test_counts"][pcfg["target"]]
                assert torch.equal(raw[key]["labels"], truth(data, pcfg["target"]))
            row = dict(key=key, **result)
            if preset == "attention_only":
                base = raw[f"original/{split}"]
                row["paired"] = paired_counts(
                    base["prediction"].eq(base["labels"]),
                    raw[key]["prediction"].eq(raw[key]["labels"]),
                )
            rows.append(row)
            print("ATTENTION_ABLATION", key, result["correct"], result["n"], flush=True)
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    losses = {r["key"].split("/")[-1]: r["paired"]["loss_pp"] for r in rows if "paired" in r}
    eligible = all(v <= cfg["exploratory_native_loss_limit_pp"] for v in losses.values())
    save_json(
        out / "decision.json",
        dict(
            native_losses_pp=losses,
            exploratory_utility_screen_passed=eligible,
            utility_goal_half_pp_passed=losses["official_test"] <= 0.5,
            fusion_or_recovery_measured=False,
            next_action="Freeze a separate real fusion/AGA/public restoration protocol"
            if eligible
            else "Stop this fixed ablation; no rank, strength or decoder-fit rescue",
            defense_success_claim=False,
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - start,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            parent_unchanged=True,
            attention_constructed_from_target_calibration_images=pcfg["calibration_samples"],
            additional_target_labels_used=False,
            future_partners_in_construction=False,
            fresh_full_tests=2,
            fresh_confirmation_tests=2,
            novelty_or_security_claim=False,
        ),
    )


@torch.no_grad()
def validate(args):
    cfg, out = read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for name, pin in meta["source_manifest"].items():
        assert digest(name) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    pcfg, bundle, release, edits = load_inputs(cfg)
    _root, _load, _fusion, models, head, data, indices, _images = bundle
    frames = torch.load(
        Path(cfg["parent"]) / "artifacts/frames.pt", map_location="cpu", weights_only=True
    )
    rebuilt, algebra = reconstruct_attention(models["original"], pcfg, frames)
    error = max(float((edits[k] - v).abs().max()) for k, v in rebuilt.items())
    assert error < 2e-6
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = {r["key"]: r for r in read(out / "results.json")}
    checks = 0
    for preset, model in (("original", models["original"]), ("attention_only", release)):
        for split, dataset, ids in (
            ("confirmation", data.train_dataset, indices["confirmation"]),
            ("official_test", data.test_dataset, list(range(len(data.test_dataset)))),
        ):
            key = f"{preset}/{split}"
            value, row = raw[key], rows[key]
            y = torch.tensor([int(dataset[i][1]) for i in ids])
            assert torch.equal(y, value["labels"])
            assert row["correct"] == int(value["prediction"].eq(y).sum())
            assert row["n"] == len(y) and row["top1"] == row["correct"] / row["n"]
            images = torch.stack([dataset[i][0] for i in ids[:32]])
            f = features(model.cuda().eval(), images)
            model.cpu()
            p = (f @ head.weight.T + head.bias).argmax(1)
            assert torch.equal(p, value["prediction"][:32])
            if preset == "attention_only":
                before = raw[f"original/{split}"]
                assert row["paired"] == paired_counts(
                    before["prediction"].eq(y), value["prediction"].eq(y)
                )
            checks += 1
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            numpy_attention_compilation_max_absolute_error=error,
            composed_qk_and_vo_checks=algebra,
            original_mlp_ln_and_other_tensors_unchanged=True,
            full_labels_counts_pairs_and_fresh_prediction_prefixes=checks,
            source_parent_checkpoint_and_release_hashes_checked=True,
            no_fusion_or_defense_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/operator_attention_ablation.json")
    parser.add_argument("--output", default="outputs/2026-10-12/operator_attention_ablation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    (validate if options.validate else run)(options)
