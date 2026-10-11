"""Replay frozen margin solves, released edits, attacks and official predictions."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import reset
from eight_task_benchmark import author_recover, digest, merge_all
from functional_contrast import features
from local_erasure import rescale, sites
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_margin import check_parent, solve_margin
from path_subspace import apply, context, edits_for, read
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


@torch.no_grad()
def calibration_linearity(release, head, images, geometry, saved):
    release.cuda().eval()
    after = features(release, images).double()
    release.cpu()
    before = saved["baseline"]["expert"].double()
    predicted = geometry["derivatives"]["expert"].double() @ saved["coefficients"]
    weight = head.weight.detach().double().cpu()
    original_scores, actual_scores = before @ weight.T, after @ weight.T
    linear_scores = (before + predicted) @ weight.T
    winner = original_scores.argmax(-1)
    mask = torch.nn.functional.one_hot(winner, num_classes=len(weight)).bool()

    def margins(scores):
        return (scores.gather(1, winner[:, None]) - scores)[~mask]

    original = margins(original_scores)
    return dict(
        n=len(images),
        predicted_feature_rms=float(predicted.square().sum().div(len(images)).sqrt()),
        actual_feature_rms=float((after - before).square().sum().div(len(images)).sqrt()),
        feature_prediction_relative_error=float(
            (after - before - predicted).norm() / (after - before).norm()
        ),
        maximum_predicted_relative_margin_change=float(
            ((margins(linear_scores) - original) / original).abs().max()
        ),
        maximum_actual_relative_margin_change=float(
            ((margins(actual_scores) - original) / original).abs().max()
        ),
        predicted_pseudo_prediction_changes=int(linear_scores.argmax(-1).ne(winner).sum()),
        actual_pseudo_prediction_changes=int(actual_scores.argmax(-1).ne(winner).sum()),
        scope="Calibration diagnostic of the frozen release; no new construction or amplitude revision",
    )


@torch.no_grad()
def check_partner_changes(
    target, task, models, root, cfg, direct, recovered, head, dataset, raw, evidence
):
    changed = torch.where(direct.ne(recovered))[0]
    before = merge_all(models, root, cfg["coefficient"])
    protected = dict(models)
    release = protected[target]
    restored = copy.deepcopy(release).cpu().eval()
    restored.model.visual.proj.copy_(evidence["restored_projection"])
    protected[target] = restored
    after = merge_all(protected, root, cfg["coefficient"])
    delta = after.model.visual.proj.double() - before.model.visual.proj.double()
    weight = head.weight.detach().double().cpu()
    ideal_error = float((evidence["projection_delta"] @ weight.T).norm())
    assert ideal_error < 1e-8
    assert all(
        torch.equal(v, before.state_dict()[k])
        for k, v in after.state_dict().items()
        if k != "model.visual.proj"
    )
    examples = []
    for block in torch.unique(changed // 128).tolist():
        ids = list(range(block * 128, min((block + 1) * 128, len(dataset))))
        logits = []
        for model, expected in [(before, direct), (after, recovered)]:
            values = []
            evaluate(model, head, make_loader(dataset, ids), "cuda", logit_sink=values.append)
            scores = torch.cat(values)
            assert torch.equal(scores.argmax(-1), expected[ids])
            logits.append(scores)
        for index in changed[(changed // 128).eq(block)].tolist():
            local = index - block * 128
            old, new = int(direct[index]), int(recovered[index])
            a, b = [float(value[local, old] - value[local, new]) for value in logits]
            assert a >= 0 and b <= 0
            examples.append(
                dict(
                    index=index,
                    before_class=old,
                    after_class=new,
                    ground_truth=int(
                        raw[f"{target}/protected/direct/fusion/{task}"]["labels"][index]
                    ),
                    before_pair_margin=a,
                    after_pair_margin=b,
                )
            )
    return dict(
        changed_predictions=len(changed),
        ideal_unnormalized_partner_score_update_norm=ideal_error,
        compiled_unnormalized_partner_score_update_matrix_norm=float((delta @ weight.T).norm()),
        changed_examples_replayed_in_original_batch=examples,
        scope="Exact FP64 null-space algebra does not imply byte-identical float32 compiled predictions",
    )


@torch.no_grad()
def validate(args):
    out = Path(args.output)
    cfg, meta, rows = [
        read(out / name) for name in ("protocol.json", "metadata.json", "results.json")
    ]
    started = time.perf_counter()
    parent, prior = check_parent(cfg)
    assert meta["completed"] and digest("experiments/path_margin.py") == meta["source_sha256"]
    assert digest(args.config) == meta["config_sha256"] and read(args.config) == cfg
    assert digest(parent / "metadata.json") == meta["parent_metadata_sha256"]
    assert digest(parent / "results.json") == meta["parent_results_sha256"]
    assert digest(parent / "training_indices.json") == meta["parent_indices_sha256"]
    for name, value in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == value
    for name, value in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == value
    root, load, fusion, models = context(cfg, all_tasks=True)
    from dataset.registry import get_dataset

    heads = {t: load(str(root / f"head_{t}.pt")).cpu().eval() for t in cfg["evaluation_tasks"]}
    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    indices = read(out / "training_indices.json")
    assert indices == read(parent / "training_indices.json")
    queries = read(out / "attack_indices.json")
    assert queries == read(parent / "attack_indices.json")
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    controls = torch.load(
        parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    assert len(rows) == len(raw) == 68 and meta["actual_full_evaluations"] == 26
    assert len(set(r["id"] for r in rows)) == len(rows)
    for row in rows:
        value = raw[row["id"]]
        dataset = data[row["dataset"]].test_dataset
        truth = torch.as_tensor(
            [y for _, y in dataset._samples] if row["dataset"] == "Cars" else dataset.targets,
            dtype=torch.long,
        )
        assert torch.equal(value["labels"], truth)
        assert len(truth) == fusion["expected_test_counts"][row["dataset"]] == row["n"]
        correct = int(value["prediction"].eq(truth).sum())
        assert correct == row["correct"] and row["top1"] == correct / len(truth)
        original_key = row["id"].replace("/protected/", "/original/")
        original = raw[original_key]
        assert row["matched"] == paired_counts(
            original["prediction"].eq(truth), value["prediction"].eq(truth)
        )
        if row["reused"]:
            assert all(torch.equal(value[k], controls[original_key][k]) for k in value)
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == meta["original_author_aga_sha256"] == prior["original_author_aga_sha256"]
    checks, prefixes, partner_changes, linearity, compilation = [], 0, {}, {}, {}
    for target in cfg["tasks"]:
        geometry = torch.load(
            parent / f"artifacts/{target}_geometry.pt", map_location="cpu", weights_only=True
        )
        saved = torch.load(
            out / f"artifacts/{target}_construction.pt", map_location="cpu", weights_only=True
        )
        x = torch.stack([data[target].train_dataset[i][0] for i in indices[target]["calibration"]])
        for name, beta in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
            model = (
                models[target]
                if name == "expert"
                else rescale(models[target], models["anchor"], beta)
            )
            model.cuda().eval()
            actual = features(model, x)
            model.cpu()
            assert torch.equal(actual, saved["baseline"][name])
            if name != "expert":
                del model
        coefficients, info = solve_margin(
            geometry, saved["baseline"], heads[target], models[target], cfg
        )
        assert (
            torch.equal(coefficients, saved["coefficients"])
            and info == read(out / "construction.json")[target]
        )
        assert (
            info["maximum_predicted_expert_relative_margin_change"]
            <= cfg["maximum_relative_margin_change"] + 1e-12
        )
        assert (
            info["maximum_matrix_relative_change"] <= cfg["maximum_matrix_relative_change"] + 1e-12
        )
        release = apply(models[target], edits_for(geometry["basis"], coefficients))
        edits = torch.load(
            out / f"artifacts/{target}_edits.pt", map_location="cpu", weights_only=True
        )
        expected = {
            k: v
            for k, v in release.state_dict().items()
            if not torch.equal(v, models[target].state_dict()[k])
        }
        assert edits.keys() == expected.keys() and all(
            torch.equal(v, expected[k]) for k, v in edits.items()
        )
        linearity[target] = calibration_linearity(release, heads[target], x, geometry, saved)
        original_reset = reset(
            models[target], models["anchor"], sites(cfg["layers"][target]), "public_support"
        )
        protected_reset = reset(
            release, models["anchor"], sites(cfg["layers"][target]), "public_support"
        )
        assert all(
            torch.equal(v, original_reset.state_dict()[k])
            for k, v in protected_reset.state_dict().items()
        )
        del original_reset, protected_reset, geometry
        cases = {f"beta_{beta}": rescale(release, models["anchor"], beta) for beta in cfg["betas"]}
        cases.update(direct=release, aga=author_recover(release, models["anchor"], author))
        evidence = torch.load(
            out / f"artifacts/{target}_readout.pt", map_location="cpu", weights_only=True
        )
        student = merge_all(dict(models, **{target: release}), root, cfg["coefficient"])
        inverse, _ = head_inverse(heads[target], [h for t, h in heads.items() if t != target])
        q_images = torch.stack([data[target].train_dataset[i][0] for i in queries[target]])
        restored, fit, replay = recover(
            release, student, heads[target], inverse, q_images, cfg["ridge_ratio"]
        )
        assert fit == read(out / "fitting.json")[target]
        assert evidence.keys() == replay.keys() and all(
            torch.equal(v, replay[k]) for k, v in evidence.items()
        )
        assert all(
            torch.equal(v, release.state_dict()[k])
            for k, v in restored.state_dict().items()
            if k != "model.visual.proj"
        )
        cases["readout"] = restored
        del student
        for attack, model in cases.items():
            relevant = [
                r
                for r in rows
                if r["target"] == target
                and r["preset"] == "protected"
                and r["attack"] == attack
                and not r["reused"]
            ]
            merged = (
                None
                if attack.startswith("beta_")
                else merge_all(dict(models, **{target: model}), root, cfg["coefficient"])
            )
            for row in relevant:
                sink, labels = [], []
                current = model if row["group"] != "fusion" else merged
                evaluate(
                    current,
                    heads[row["dataset"]],
                    make_loader(data[row["dataset"]].test_dataset, list(range(128))),
                    "cuda",
                    prediction_sink=lambda p, y: (sink.append(p), labels.append(y)),
                )
                assert torch.equal(torch.cat(sink), raw[row["id"]]["prediction"][:128])
                assert torch.equal(torch.cat(labels), raw[row["id"]]["labels"][:128])
                prefixes += 1
            del merged
        for task in cfg["evaluation_tasks"]:
            if task == target:
                continue
            direct = raw[f"{target}/protected/direct/fusion/{task}"]["prediction"]
            recovered = raw[f"{target}/protected/readout/fusion/{task}"]["prediction"]
            changes = int(direct.ne(recovered).sum())
            partner_changes[f"{target}/{task}"] = changes
            compilation[f"{target}/{task}"] = check_partner_changes(
                target,
                task,
                dict(models, **{target: release}),
                root,
                cfg,
                direct,
                recovered,
                heads[task],
                data[task].test_dataset,
                raw,
                evidence,
            )
        checks.append(
            dict(
                target=target,
                baseline_features_byte_exact=True,
                signed_solve_byte_exact=True,
                released_edits_byte_exact=True,
                support_reset_entire_state_identical=True,
                readout_all_evidence_byte_exact=True,
            )
        )
        print("VALIDATED", target, flush=True)
        del cases, release, restored
    assert prefixes == 26
    save_json(out / "linearity.json", linearity)
    save_json(out / "compilation.json", compilation)
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            validator_sha256=digest(__file__),
            checked_rows=len(rows),
            new_evaluation_prefixes_replayed=prefixes,
            reused_rows_checked=meta["reused_metric_rows"],
            source_sha256=meta["source_sha256"],
            results_sha256=digest(out / "results.json"),
            partner_full_prediction_changes=partner_changes,
            checks=checks,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/path_margin.json")
    parser.add_argument("--output", default="outputs/2026-10-11/path_margin")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    validate(args)
