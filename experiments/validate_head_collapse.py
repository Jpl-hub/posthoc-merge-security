"""Replay joint-head release, matched recovery states and official predictions."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import capture
from eight_task_benchmark import author_recover, digest, merge_all
from head_collapse import construct, factor, identical, reset_heads, truth
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read, response
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta, rows = [read(out / p) for p in ("protocol.json", "metadata.json", "results.json")]
    assert meta["completed"] and len(rows) == meta["new_full_evaluations"] == 36
    assert meta["source_sha256"] == digest("experiments/head_collapse.py")
    assert read(args.config) == cfg and meta["config_sha256"] == digest(args.config)
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg, all_tasks=True)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    task, selected = cfg["task"], read(out / "selected.json")
    original, anchor = models[task], models["anchor"]
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    indices = read(out / "training_indices.json")
    assert not set(indices["queries"]).intersection(
        indices["calibration"] + indices["confirmation"]
    )
    images = torch.stack([data[task].train_dataset[i][0] for i in indices["calibration"]])
    captured, _ = capture(original, images, cfg["layers"])
    saved = torch.load(out / "artifacts/construction.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(v, saved["inputs"][k]) for k, v in captured.items())
    svd = author_svd("external/NUFILT", read(cfg["nufilt_config"]))
    release, edits, replay_selected, candidates, proof = construct(original, captured, cfg, svd)
    assert replay_selected == selected and candidates == read(out / "candidates.json")
    compiled = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    assert edits.keys() == compiled.keys() and all(
        torch.equal(v, compiled[k]) for k, v in edits.items()
    )
    optimum_checks = []
    for item in selected:
        layer = item["layer"]
        record = saved["factors"][str(layer)]
        assert all(torch.equal(v, proof[str(layer)][k]) for k, v in record.items())
        block = original.model.visual.transformer.resblocks[layer]
        x = captured[f"v:{layer}"].reshape(-1, 50, block.attn.embed_dim)
        b, d, geometry, _ = factor(
            block, x, item["pair"], cfg["rank"], cfg["covariance_cutoff"], svd
        )
        augmented = torch.cat(
            [
                x.double().cuda().flatten(0, 1),
                torch.ones(x.shape[0] * x.shape[1], 1, dtype=torch.float64, device="cuda"),
            ],
            1,
        )
        native = augmented @ record["value"].cuda().T @ record["output"].cuda().T
        compressed = augmented @ b.T @ d.T
        actual = float((native - compressed).square().sum() / len(native))
        assert abs(actual - geometry["actual_token_compression_mse"]) <= max(1e-8, actual * 1e-8)
        changed = release.model.visual.transformer.resblocks[layer].attn
        indexes = [slice(h * 64, (h + 1) * 64) for h in item["pair"]]
        for offset in (0, 768, 1536):
            aa, bb = [slice(offset + i.start, offset + i.stop) for i in indexes]
            assert torch.equal(changed.in_proj_weight[aa], changed.in_proj_weight[bb])
            assert torch.equal(changed.in_proj_bias[aa], changed.in_proj_bias[bb])
        assert torch.equal(
            changed.out_proj.weight[:, indexes[0]], changed.out_proj.weight[:, indexes[1]]
        )
        old_attention, new_attention = [], []
        for batch in x.split(16):
            seq = batch.cuda().transpose(0, 1)
            block.attn.cuda().eval()
            old_attention.append(
                block.attn(seq, seq, seq, need_weights=False)[0].transpose(0, 1).cpu()
            )
            block.attn.cpu()
            changed.cuda().eval()
            new_attention.append(
                changed(seq, seq, seq, need_weights=False)[0].transpose(0, 1).cpu()
            )
            changed.cpu()
        old_attention, new_attention = (
            torch.cat(old_attention).double(),
            torch.cat(new_attention).double(),
        )
        actual_relative = float((new_attention - old_attention).norm() / old_attention.norm())
        assert abs(actual_relative - item["local_relative_rms"]) < 2e-6
        optimum_checks.append(
            dict(
                layer=layer,
                actual_token_mse=actual,
                optimal_tail_mse=geometry["optimal_tail_mse"],
                compiled_attention_relative_rms=actual_relative,
                selected_relative_rms=item["local_relative_rms"],
                original_joint_rank=item["original_operator_rank"],
                compressed_rank=item["compressed_factor_rank"],
            )
        )
    confirmation = torch.stack([data[task].train_dataset[i][0] for i in indices["confirmation"]])
    assert response(original, release, confirmation) == read(out / "functions.json")["confirmation"]
    del saved, captured
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(rows) == len(raw) == len({r["key"] for r in rows})
    for row in rows:
        value = raw[row["key"]]
        labels = truth(data[row["dataset"]], row["dataset"])
        assert torch.equal(value["labels"], labels)
        assert row["n"] == len(labels) == fusion["expected_test_counts"][row["dataset"]]
        assert row["correct"] == int(value["prediction"].eq(labels).sum())
        assert abs(row["top1"] - row["correct"] / row["n"]) < 1e-12
        reference = f"original/{row['attack']}/{row['scope']}/{row['dataset']}"
        if row["preset"] == "protected" and reference in raw:
            assert row["matched_change"] == paired_counts(
                raw[reference]["prediction"].eq(labels), value["prediction"].eq(labels)
            )
    author, author_hash = load_author(
        "external/secure-merge-attack/utils_attack_B32.py", include_wrappers=True
    )
    assert author_hash == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    states = {("original", "none"): original, ("protected", "none"): release}
    inverse, geometry = head_inverse(heads[task], [heads[t] for t in cfg["preserved_query_heads"]])
    fit_data = read(out / "query_fits.json")
    assert geometry == fit_data["geometry"]
    query_images = torch.stack([data[task].train_dataset[i][0] for i in indices["queries"]])
    for preset, teacher in [("original", original), ("protected", release)]:
        student = merge_all(dict(models, **{task: teacher}), root, cfg["coefficient"])
        states[preset, "direct"] = student
        restored = author_recover(teacher, anchor, author)
        saved_aga = torch.load(
            out / f"artifacts/{preset}_aga_edits.pt", map_location="cpu", weights_only=True
        )
        changed = {
            k: v
            for k, v in restored.state_dict().items()
            if not torch.equal(v, teacher.state_dict()[k])
        }
        assert changed.keys() == saved_aga.keys() and all(
            torch.equal(v, saved_aga[k]) for k, v in changed.items()
        )
        states[preset, "aga"] = merge_all(
            dict(models, **{task: restored}), root, cfg["coefficient"]
        )
        unused, fit, replay = recover(
            teacher, student, heads[task], inverse, query_images, cfg["ridge_ratio"]
        )
        del unused, restored
        evidence = torch.load(
            out / f"artifacts/{preset}_query.pt", map_location="cpu", weights_only=True
        )
        assert fit == fit_data["fits"][preset]
        assert all(torch.equal(v, evidence[k]) for k, v in replay.items())
        fitted = copy.deepcopy(student).cpu().eval()
        projection = (student.model.visual.proj.double() + replay["projection_delta"]).float()
        assert torch.equal(projection, evidence["compiled_student_projection"])
        fitted.model.visual.proj.copy_(projection)
        assert all(
            torch.equal(v, student.state_dict()[k])
            for k, v in fitted.state_dict().items()
            if k != "model.visual.proj"
        )
        states[preset, "readout"] = fitted
    for attack in ("support_reset", "reset_attention"):
        if attack == "support_reset":
            a, b = [reset_heads(t, anchor, selected) for t in (original, release)]
        else:
            a, b = [
                replace_components(t, anchor, list(range(12)), "attn")[0]
                for t in (original, release)
            ]
        identical(a, b)
        states["identical_reset", attack, "standalone"] = a
        states["identical_reset", attack, "fusion"] = merge_all(
            dict(models, **{task: a}), root, cfg["coefficient"]
        )
    prefixes, partner_checks = [], {}
    for row in rows:
        key = (
            (row["preset"], row["attack"], row["scope"])
            if row["preset"] == "identical_reset"
            else (row["preset"], row["attack"])
        )
        predictions = []
        evaluate(
            states[key],
            heads[row["dataset"]],
            make_loader(data[row["dataset"]].test_dataset, list(range(128))),
            "cuda",
            prediction_sink=lambda a, _b: predictions.append(a),
        )
        assert torch.equal(torch.cat(predictions), raw[row["key"]]["prediction"][:128])
        prefixes.append(row["key"])
    for preset in ("original", "protected"):
        for t in cfg["evaluation_tasks"]:
            before, after = [
                raw[f"{preset}/{attack}/fusion/{t}"] for attack in ("direct", "readout")
            ]
            partner_checks[f"{preset}/{t}"] = dict(
                predictions_changed=int(before["prediction"].ne(after["prediction"]).sum()),
                change=paired_counts(
                    before["prediction"].eq(before["labels"]),
                    after["prediction"].eq(after["labels"]),
                ),
            )
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            validator_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
            full_metric_rows=len(rows),
            prefix_replays=len(prefixes),
            reconstructed_query_fits=2,
            construction_inputs_exact=True,
            all_396_structural_scores_replayed=True,
            selected_factors_exact=True,
            compiled_release_exact=True,
            all_unedited_tensors_exact=True,
            author_aga_replays=2,
            author_aga_sha256=author_hash,
            optimality_checks=optimum_checks,
            query_partner_checks=partner_checks,
            eight_task_mean_reported=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_collapse.json")
    parser.add_argument("--output", default="outputs/2026-10-11/head_collapse")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
