"""One stronger white-box recovery control on the Cars public-reset region.

Reconstruct three public-basis MLP decoders from observable block outputs.
No release is proposed. This tests a recovery cost, not defense performance.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from joint_posthoc import capture_blocks
from local_erasure import cache
from operator_diagnostics import paired_counts, replace_components
from path_subspace import read
from polynomial_probe import ridge_update
from public_restore_gate import inputs
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


def cholesky_check(design, target, ratio):
    gram = design.T @ design
    ridge = ratio * design.square().sum() / len(design)
    lhs = gram + ridge * torch.eye(len(gram), device="cuda", dtype=torch.float64)
    factor = torch.linalg.cholesky(lhs)
    rhs = design.T @ target
    update = torch.cholesky_solve(rhs, factor)
    error = float((lhs @ update - rhs).norm() / rhs.norm())
    assert error < 1e-8
    return update, dict(ridge=float(ridge), normal_equation_relative_residual=error)


def reset_region(teacher, anchor, cfg):
    component = cfg.get("reset_component", "attn_mlp")
    assert component in {"attn_mlp", "block"}
    return replace_components(teacher, anchor, cfg["layers"], component)


@torch.no_grad()
def build(teacher, anchor, images, cfg, independent=False):
    restored, keys = reset_region(teacher, anchor, cfg)
    target_blocks = capture_blocks(teacher, images, cfg["layers"])
    rows = []
    for layer in cfg["layers"]:
        current = capture_blocks(restored, images, [layer], details=True)[layer]
        mlp = restored.model.visual.transformer.resblocks[layer].mlp
        x = current["mlp_input"].double().cuda()
        source = copy.deepcopy(mlp).double().cuda().eval()
        phi = source.gelu(source.c_fc(x))
        design = torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)
        target = (
            target_blocks[layer]["output"].double().cuda()
            - current["post_attention"].double().cuda()
        )
        initial = torch.cat([source.c_proj.weight.T, source.c_proj.bias[None, :]], dim=0)
        residual = target - design @ initial
        solve = cholesky_check if independent else ridge_update
        update, fit = solve(design, residual, cfg["ridge_ratio"])
        fit["design_features"] = design.shape[1]
        fitted = initial + update
        mlp.c_proj.weight.copy_(fitted[:-1].T.float().cpu())
        mlp.c_proj.bias.copy_(fitted[-1].float().cpu())
        rows.append(
            dict(
                layer=layer,
                teacher_tokens=len(design),
                original_block_relative_error=float(
                    (design @ fitted - target).norm()
                    / target_blocks[layer]["output"].double().cuda().norm()
                ),
                **fit,
            )
        )
        print("BLOCK_RESTORE_LAYER", layer, rows[-1]["original_block_relative_error"], flush=True)
        del source, x, phi, design, target, initial, residual, update, fitted, current
    return restored, rows, keys


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, fusion, models, heads, data, ids, images, prior = inputs(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    save_json(out / "attack_indices.json", ids)
    target = cfg["target"]
    restored, fitting, keys = build(models[target], models["anchor"], images, cfg)
    erased, _ = reset_region(models[target], models["anchor"], cfg)
    edits = {
        k: v.clone()
        for k, v in restored.state_dict().items()
        if not torch.equal(v, erased.state_dict()[k])
    }
    assert set(edits) <= {k for k in keys if ".mlp.c_proj." in k} and len(edits) == 6
    torch.save(edits, artifacts / "restore_edits.pt")
    save_json(out / "fitting.json", fitting)
    loaders = {t: cache(d) for t, d in data.items()}
    raw, rows = {}, []
    reuse_reset = cfg.get("reset_component", "attn_mlp") == "attn_mlp"
    for preset in ["original", "reset"] if reuse_reset else ["original"]:
        for group, tasks in [("standalone", [target]), ("fusion", cfg["evaluation_tasks"])]:
            for task in tasks:
                key = f"{preset}/{group}/{task}"
                raw[key] = prior[key]
                labels = truth(data[task], task)
                assert torch.equal(labels, raw[key]["labels"])
                correct = int(raw[key]["prediction"].eq(labels).sum())
                rows.append(
                    dict(
                        key=key,
                        preset=preset,
                        group=group,
                        dataset=task,
                        correct=correct,
                        n=len(labels),
                        top1=correct / len(labels),
                        reused=True,
                    )
                )
    cases = (
        {"block_restore": restored} if reuse_reset else {"reset": erased, "block_restore": restored}
    )
    for preset, native in cases.items():
        merged = merge_all(dict(models, **{target: native}), root, cfg["coefficient"])
        for group, model, tasks in [
            ("standalone", native, [target]),
            ("fusion", merged, cfg["evaluation_tasks"]),
        ]:
            for task in tasks:
                p, y = [], []
                score = evaluate(
                    model,
                    heads[task],
                    loaders[task],
                    "cuda",
                    prediction_sink=lambda a, b: (p.append(a), y.append(b)),
                )
                key = f"{preset}/{group}/{task}"
                raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
                rows.append(
                    dict(key=key, preset=preset, group=group, dataset=task, reused=False, **score)
                )
                save_json(out / "results.json", rows)
                torch.save(raw, artifacts / "predictions.pt")
                print("BLOCK_RESTORE_GATE", key, score["correct"], score["n"], flush=True)
    for row in rows:
        labels = raw[row["key"]]["labels"]
        reference = raw[f"original/{row['group']}/{row['dataset']}"]
        row["original_change"] = paired_counts(
            reference["prediction"].eq(labels), raw[row["key"]]["prediction"].eq(labels)
        )
    save_json(out / "results.json", rows)
    assert len(rows) == len(raw) == 15
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            fresh_full_evaluations=sum(not r["reused"] for r in rows),
            reused_baselines=sum(r["reused"] for r in rows),
            layernorm_also_publicly_reset=not reuse_reset,
            new_defense=False,
            teacher_is_unprotected=True,
            teacher_unique_query_inputs=len(images),
            teacher_block_output_scalar_values=len(images) * 50 * 768 * len(cfg["layers"]),
            teacher_hidden_features_used=True,
            teacher_ground_truth_used=False,
            closed_form_fits=3,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            eight_task_mean=False,
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
    root, fusion, models, heads, data, ids, images, prior = inputs(cfg)
    assert ids == read(out / "attack_indices.json")
    target = cfg["target"]
    replay, fitting, _ = build(models[target], models["anchor"], images, cfg, independent=True)
    erased, _ = reset_region(models[target], models["anchor"], cfg)
    restored = copy.deepcopy(erased).cpu().eval()
    edits = torch.load(out / "artifacts/restore_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        restored.state_dict()[key].copy_(value)
    error = max(
        float((v - replay.state_dict()[k]).abs().max()) for k, v in restored.state_dict().items()
    )
    assert error < 5e-5, error
    built = {
        "original/standalone": models[target],
        "reset/standalone": erased,
        "block_restore/standalone": restored,
    }
    for preset in ["original", "reset", "block_restore"]:
        built[f"{preset}/fusion"] = merge_all(
            dict(models, **{target: built[f"{preset}/standalone"]}), root, 0.2
        )
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 15
    for row in rows:
        task = row["dataset"]
        v, labels = raw[row["key"]], truth(data[task], task)
        assert torch.equal(v["labels"], labels)
        assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
        assert row["correct"] == int(v["prediction"].eq(labels).sum()) and row["top1"] == row[
            "correct"
        ] / len(labels)
        reference = raw[f"original/{row['group']}/{task}"]
        assert row["original_change"] == paired_counts(
            reference["prediction"].eq(labels), v["prediction"].eq(labels)
        )
        if row["reused"]:
            assert torch.equal(v["prediction"], prior[row["key"]]["prediction"])
        sink = []
        evaluate(
            built[f"{row['preset']}/{row['group']}"],
            heads[task],
            make_loader(data[task].test_dataset, list(range(128)), 128),
            "cuda",
            prediction_sink=lambda p, _y: sink.append(p),
        )
        assert torch.equal(torch.cat(sink), v["prediction"][:128])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            independent_cholesky_fits=3,
            compiled_weight_max_absolute_error=error,
            normal_equation_residuals=[r["normal_equation_relative_residual"] for r in fitting],
            counts_pairs_and_prefixes_checked=15,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            new_defense=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/block_restore_gate.json")
    parser.add_argument("--output", default="outputs/2026-10-11/block_restore_gate_cars")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
