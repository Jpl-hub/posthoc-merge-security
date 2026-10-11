"""Fixed hard-label prototype recovery on an already frozen posthoc release.

Reuse a pinned author's class-mean primitive and the existing public-reset
student, merger, evaluation and partner-preserving output projection inverse.
No protection, optimizer, continuous teacher response or search is introduced.
"""

from __future__ import annotations

import argparse
import ast
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from local_erasure import cache
from local_release_audit import branch_reset, load_inputs, state_equal
from operator_diagnostics import paired_counts, query_readout
from path_subspace import read
from query_partner_cost import runtime, truth
from score_recovery import head_inverse
from subspace_probe import make_loader, save_json


def author_classifier(cfg):
    """Load only the pinned standalone class; no author training imports run."""
    source = Path(cfg["author_source"])
    assert digest(source) == cfg["author_sha256"]
    assert read(source.parents[1] / "source.json")["commit"] == cfg["author_commit"]
    tree = ast.parse(source.read_text())
    selected = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cfg["prototype_class"]
    ]
    assert len(selected) == 1 and cfg["prototype_class"] == "PrototypeClassifier"
    namespace = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[cfg["prototype_class"]]


def audited_parent(cfg):
    parent = Path(cfg["audit_parent"])
    meta = read(parent / "metadata.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    for name, sha in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == sha
    assert read(parent / "protocol.json")["parent"] == cfg["parent"]
    return meta, torch.load(parent / "artifacts/predictions.pt", weights_only=True)


@torch.no_grad()
def compile_prototypes(reset_expert, student, head, inverse, prototypes, coefficient):
    p, c = student.model.visual.proj.double().cuda(), head.weight.double().cuda()
    scale = c.norm(dim=1).median()
    desired = prototypes.double().cuda() * scale
    delta = p @ (desired - c).T @ inverse
    identity_error = float(((p + delta) @ c.T - p @ desired.T).norm())
    assert identity_error < 1e-7
    restored = copy.deepcopy(reset_expert).cpu().eval()
    restored.model.visual.proj.copy_(
        (reset_expert.model.visual.proj.double().cuda() + delta / coefficient).float().cpu()
    )
    assert all(
        torch.equal(v, reset_expert.state_dict()[k])
        for k, v in restored.state_dict().items()
        if k != "model.visual.proj"
    )
    return restored, delta.cpu(), float(scale), identity_error


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    root, fusion, models, release, heads, data, indices, images = load_inputs(cfg)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    assert all(torch.count_nonzero(h.bias) == 0 and h.normalize for h in heads.values())
    parent_meta, control = audited_parent(cfg)
    assert indices == read(Path(cfg["audit_parent"]) / "attack_indices.json")
    save_json(out / "attack_indices.json", indices)
    target = cfg["target"]
    reset_expert = branch_reset(models[target], models["anchor"], cfg, "support_reset")
    assert state_equal(reset_expert, branch_reset(release, models["anchor"], cfg, "support_reset"))
    student = merge_all(dict(models, **{target: reset_expert}), root, cfg["coefficient"])
    inverse, geometry = head_inverse(
        heads[target], [heads[t] for t in cfg["preserved_query_heads"]]
    )
    _, design = query_readout(student, heads[target], images, capture_features=True)
    # Student features, not teacher hidden responses, enter the class means.
    features = F.normalize(design @ student.model.visual.proj.double().cuda(), dim=1)
    classifier = author_classifier(cfg)
    labels, prototypes, counts, restored, evidence, checks = {}, {}, {}, {}, {}, {}
    for preset, teacher in [("original", models[target]), ("protected", release)]:
        pseudo, _ = query_readout(teacher, heads[target], images)
        labels[preset] = pseudo
        counts[preset] = torch.bincount(pseudo, minlength=len(heads[target].weight)).tolist()
        assert min(counts[preset]) > 0, "Missing queried class; no test-based completion allowed"
        primitive = classifier(len(heads[target].weight), features.shape[1], device="cuda")
        primitive.update(features, pseudo.cuda())
        prototypes[preset] = primitive.get_prototypes().cpu()
        restored[preset], delta, scale, error = compile_prototypes(
            reset_expert, student, heads[target], inverse, prototypes[preset], cfg["coefficient"]
        )
        evidence[preset] = dict(
            teacher_predictions=pseudo,
            prototypes=prototypes[preset],
            projection_delta=delta,
            restored_projection=restored[preset].model.visual.proj.detach().clone(),
        )
        checks[preset] = dict(class_counts=counts[preset], scale=scale, ideal_readout_error=error)
    identical = state_equal(restored["original"], restored["protected"])
    assert not torch.equal(labels["original"], labels["protected"]) or identical
    certificate = dict(
        design=design.cpu(), features=features.cpu(), inverse=inverse.cpu(), **evidence
    )
    torch.save(certificate, artifacts / "certificate.pt")
    save_json(
        out / "fitting.json",
        dict(
            geometry=geometry,
            checks=checks,
            teacher_label_disagreements=int(labels["original"].ne(labels["protected"]).sum()),
            compiled_states_identical=identical,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    del design, features, inverse
    raw, rows, unique = {}, [], 0
    loaders = {t: cache(d) for t, d in data.items()}
    for preset in ("original", "protected"):
        if preset == "protected" and identical:
            for previous in [r for r in rows if r["preset"] == "original"]:
                key = previous["key"].replace("original/", "protected/", 1)
                raw[key] = raw[previous["key"]]
                rows.append(dict(previous, key=key, preset=preset, alias_of=previous["key"]))
            continue
        merged = merge_all(dict(models, **{target: restored[preset]}), root, cfg["coefficient"])
        delta = evidence[preset]["projection_delta"]
        actual = merged.model.visual.proj.double() - student.model.visual.proj.double()
        compilation_error = float((actual - delta).norm() / delta.norm())
        assert compilation_error < 1e-6
        assert all(
            torch.equal(v, student.state_dict()[k])
            for k, v in merged.state_dict().items()
            if k != "model.visual.proj"
        )
        checks[preset]["compiled_relative_error"] = compilation_error
        checks[preset]["ideal_partner_errors"] = {
            t: float((delta @ heads[t].weight.double().T).norm())
            for t in cfg["preserved_query_heads"]
        }
        assert max(checks[preset]["ideal_partner_errors"].values()) < 1e-8
        for group, task, model in [("standalone", target, restored[preset])] + [
            ("fusion", t, merged) for t in cfg["evaluation_tasks"]
        ]:
            key = f"{preset}/support_reset_prototype/{group}/{task}"
            predicted, ground_truth = [], []
            score = evaluate(
                model,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda p, y: (predicted.append(p), ground_truth.append(y)),
            )
            raw[key] = dict(prediction=torch.cat(predicted), labels=torch.cat(ground_truth))
            rows.append(
                dict(
                    key=key,
                    preset=preset,
                    attack="support_reset_prototype",
                    group=group,
                    dataset=task,
                    **score,
                )
            )
            unique += 1
            save_json(out / "results.json", rows)
            print("PROTOTYPE_RECOVERY", key, score["correct"], score["n"], flush=True)
        del merged
    for row in rows:
        value, task, preset, group = raw[row["key"]], row["dataset"], row["preset"], row["group"]
        y = value["labels"]
        for name, reference in [
            ("direct_change", f"{preset}/direct/{group}/{task}"),
            ("reset_change", f"{preset}/support_reset/{group}/{task}"),
            ("score_recovery_change", f"{preset}/support_reset_query/{group}/{task}"),
        ]:
            assert torch.equal(control[reference]["labels"], y)
            row[name] = paired_counts(
                control[reference]["prediction"].eq(y), value["prediction"].eq(y)
            )
        matched = raw[row["key"].replace(preset + "/", "original/", 1)]
        row["matched_original_change"] = paired_counts(
            matched["prediction"].eq(y), value["prediction"].eq(y)
        )
    save_json(out / "results.json", rows)
    save_json(out / "compilation_checks.json", checks)
    torch.save(raw, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            audit_parent_raw_sha256=parent_meta["raw_sha256"],
            full_evaluations=unique,
            reported_rows=len(rows),
            aliased_rows=len(rows) - unique,
            teacher_queries_per_control=len(images),
            teacher_access="hard argmax only",
            teacher_ground_truth_or_hidden_features_used=False,
            continuous_teacher_scores_used=False,
            class_mean_computations=2,
            ridge_fits=0,
            optimizer_steps=0,
            source_class=cfg["prototype_class"],
            new_defense=False,
            defense_success_claim=False,
            eurosat_evaluated=False,
            eight_task_mean=False,
            teacher_label_disagreements=int(labels["original"].ne(labels["protected"]).sum()),
            compiled_states_identical=identical,
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
    for path, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / path) == sha
    root, fusion, models, release, heads, data, indices, images = load_inputs(cfg)
    assert indices == read(out / "attack_indices.json")
    _, control = audited_parent(cfg)
    certificate = torch.load(out / "artifacts/certificate.pt", weights_only=True)
    raw = torch.load(out / "artifacts/predictions.pt", weights_only=True)
    rows = read(out / "results.json")
    reset_expert = branch_reset(models[cfg["target"]], models["anchor"], cfg, "support_reset")
    assert state_equal(reset_expert, branch_reset(release, models["anchor"], cfg, "support_reset"))
    student = merge_all(dict(models, **{cfg["target"]: reset_expert}), root, cfg["coefficient"])
    _, design = query_readout(student, heads[cfg["target"]], images, capture_features=True)
    assert torch.equal(design.cpu(), certificate["design"])
    features = F.normalize(design @ student.model.visual.proj.double().cuda(), dim=1).cpu()
    assert torch.equal(features, certificate["features"])
    c = heads[cfg["target"]].weight.double()
    other = torch.cat([heads[t].weight.double() for t in cfg["preserved_query_heads"]])
    # Independent QR null projector and Cholesky solve, rather than SVD/solve.
    q, _ = torch.linalg.qr(other.T, mode="reduced")
    projected = c @ (torch.eye(c.shape[1], dtype=torch.double) - q @ q.T)
    inverse = torch.cholesky_solve(projected, torch.linalg.cholesky(projected @ projected.T))
    inverse_error = float((inverse - certificate["inverse"]).norm() / inverse.norm())
    assert inverse_error < 1e-8
    prototype_errors, prefixes = {}, 0
    for preset in ("original", "protected"):
        evidence = certificate[preset]
        labels = evidence["teacher_predictions"]
        # Independent index aggregation, FP64 means and normalization.
        sums = torch.zeros(len(c), features.shape[1], dtype=torch.double)
        sums.index_add_(0, labels, features)
        means = sums / torch.bincount(labels, minlength=len(c))[:, None]
        prototypes = F.normalize(means, dim=1)
        prototype_errors[preset] = float((prototypes - evidence["prototypes"]).abs().max())
        assert prototype_errors[preset] < 1e-6
        # Recompute compilation from stored author's prototypes to separate rounding effects.
        delta = (
            student.model.visual.proj.double()
            @ (evidence["prototypes"].double() * c.norm(dim=1).median() - c).T
            @ inverse
        )
        assert float((delta - evidence["projection_delta"]).norm() / delta.norm()) < 1e-8
        restored = copy.deepcopy(reset_expert)
        restored.model.visual.proj.copy_(evidence["restored_projection"])
        expected = reset_expert.model.visual.proj.double() + delta / cfg["coefficient"]
        assert torch.allclose(expected.float(), restored.model.visual.proj, atol=2e-6, rtol=1e-6)
        merged = merge_all(dict(models, **{cfg["target"]: restored}), root, cfg["coefficient"])
        for row in [r for r in rows if r["preset"] == preset]:
            value, task, group = raw[row["key"]], row["dataset"], row["group"]
            y = truth(data[task], task)
            assert torch.equal(y, value["labels"])
            assert row["n"] == len(y) == fusion["expected_test_counts"][task]
            assert row["correct"] == int(value["prediction"].eq(y).sum())
            assert row["top1"] == row["correct"] / len(y)
            for name, attack in [
                ("direct_change", "direct"),
                ("reset_change", "support_reset"),
                ("score_recovery_change", "support_reset_query"),
            ]:
                reference = control[f"{preset}/{attack}/{group}/{task}"]
                assert row[name] == paired_counts(
                    reference["prediction"].eq(y), value["prediction"].eq(y)
                )
            matched = raw[row["key"].replace(preset + "/", "original/", 1)]
            assert row["matched_original_change"] == paired_counts(
                matched["prediction"].eq(y), value["prediction"].eq(y)
            )
            model = restored if group == "standalone" else merged
            sink = []
            evaluate(
                model,
                heads[task],
                make_loader(data[task].test_dataset, list(range(128)), 128),
                "cuda",
                prediction_sink=lambda p, _y: sink.append(p),
            )
            assert torch.equal(torch.cat(sink), value["prediction"][:128])
            if "alias_of" in row:
                assert meta["compiled_states_identical"]
                assert torch.equal(
                    certificate["original"]["restored_projection"], evidence["restored_projection"]
                )
                assert torch.equal(value["prediction"], raw[row["alias_of"]]["prediction"])
            prefixes += 1
        del merged, restored
    assert len(rows) == len(raw) == 10
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            seconds=time.perf_counter() - start,
            all_full_counts_and_comparisons=len(rows),
            official_prefixes_replayed=prefixes,
            independent_index_aggregation_max_errors=prototype_errors,
            independent_qr_cholesky_inverse_relative_error=inverse_error,
            fixed_student_features_replayed=True,
            no_new_teacher_queries=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            no_defense_success_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prototype_recovery_gate.json")
    parser.add_argument("--output", default="outputs/2026-10-11/prototype_recovery_gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    (validate if arguments.validate else run)(arguments)
