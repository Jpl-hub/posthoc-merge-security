"""Matched recovery audit for an already frozen local posthoc release.

Original AGA, public branch/support reset, and reset-then-functional recovery.
The query student is fixed independently of the teacher's release encoding.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import reset
from eight_task_benchmark import author_recover, digest, merge_all
from local_erasure import cache, sites
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_subspace import context, read
from polynomial_probe import ridge_update
from query_partner_cost import runtime, truth
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


def state_equal(left, right):
    return all(
        torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items()
    )


def load_inputs(cfg):
    parent = Path(cfg["parent"])
    meta, pcfg = read(parent / "metadata.json"), read(parent / "protocol.json")
    assert meta["completed"] and read(parent / "validation.json")["status"] == "passed"
    assert cfg["target"] == pcfg["target"] and cfg["layers"] == pcfg["layers"]
    for name, sha in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == sha
    root, load, fusion, models = context(cfg, all_tasks=True)
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    for name, sha in expected.items():
        assert digest(root / name) == sha
    release = copy.deepcopy(models[cfg["target"]]).cpu().eval()
    edits = torch.load(parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    indices = read(parent / "training_indices.json")["calibration"][: cfg["attacker_samples"]]
    assert len(indices) == cfg["attacker_samples"]
    images = torch.stack([data[cfg["target"]].train_dataset[i][0] for i in indices])
    return root, fusion, models, release, heads, data, indices, images


@torch.no_grad()
def branch_reset(model, anchor, cfg, attack):
    region = sites(cfg["layers"])
    if attack == "mlp_reset":
        region = {k: v for k, v in region.items() if k.split(":")[0] in {"up", "down"}}
    elif attack == "value_output_reset":
        region = {k: v for k, v in region.items() if k.split(":")[0] in {"v", "out"}}
    else:
        assert attack == "support_reset"
    return reset(model, anchor, region, "public_support")


@torch.no_grad()
def compile_query(teacher, reset_expert, student, heads, images, cfg):
    inverse, geometry = head_inverse(
        heads[cfg["target"]], [heads[t] for t in cfg["preserved_query_heads"]]
    )
    recovered, fit, evidence = recover(
        teacher, student, heads[cfg["target"]], inverse, images, cfg["ridge_ratio"]
    )
    result = copy.deepcopy(reset_expert).cpu().eval()
    result.model.visual.proj.copy_(recovered.model.visual.proj)
    # The output projection is the only change to the public-reset expert.
    assert all(
        torch.equal(v, reset_expert.state_dict()[k])
        for k, v in result.state_dict().items()
        if k != "model.visual.proj"
    )
    fit.update(
        geometry,
        public_support_reset_before_fit=True,
        fixed_student_independent_of_teacher_encoding=True,
    )
    return result, fit, evidence


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, fusion, models, release, heads, data, indices, images = load_inputs(cfg)
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    save_json(out / "attack_indices.json", indices)
    author, sha = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert sha == read("configs/sources.json")["AGA"]["sha256"]
    target = cfg["target"]
    original = models[target]
    anchor = models["anchor"]
    reset_control = branch_reset(original, anchor, cfg, "support_reset")
    reset_release = branch_reset(release, anchor, cfg, "support_reset")
    assert state_equal(reset_control, reset_release)
    student = merge_all(dict(models, **{target: reset_control}), root, 0.2)
    protected_student = merge_all(dict(models, **{target: reset_release}), root, 0.2)
    assert state_equal(student, protected_student)
    del protected_student, reset_release
    loaders = {t: cache(d) for t, d in data.items()}
    raw, rows, fits, mapping = {}, {}, {}, {}
    rows = []

    def measure(preset, attack, group, model, task):
        key = f"{preset}/{attack}/{group}/{task}"
        p, y = [], []
        score = evaluate(
            model,
            heads[task],
            loaders[task],
            "cuda",
            prediction_sink=lambda a, b: (p.append(a), y.append(b)),
        )
        raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
        rows.append(dict(key=key, preset=preset, attack=attack, group=group, dataset=task, **score))
        save_json(out / "results.json", rows)
        torch.save(raw, artifacts / "predictions.pt")
        print("LOCAL_RELEASE_AUDIT", key, score["correct"], score["n"], flush=True)

    for preset, teacher in [("original", original), ("protected", release)]:
        for attack in cfg["attacks"]:
            if attack == "direct":
                restored = teacher
            elif attack == "aga":
                restored = author_recover(teacher, anchor, author)
                before = teacher.state_dict()
                torch.save(
                    {
                        k: v.clone()
                        for k, v in restored.state_dict().items()
                        if not torch.equal(v, before[k])
                    },
                    artifacts / f"{preset}_aga_edits.pt",
                )
            elif attack == "support_reset_query":
                restored, fit, evidence = compile_query(
                    teacher, reset_control, student, heads, images, cfg
                )
                fits[preset] = fit
                torch.save(evidence, artifacts / f"{preset}_query.pt")
                save_json(out / "fitting.json", fits)
            else:
                restored = branch_reset(teacher, anchor, cfg, attack)
            merged = merge_all(dict(models, **{target: restored}), root, 0.2)
            if attack == "support_reset_query":
                actual = merged.model.visual.proj.double() - student.model.visual.proj.double()
                error = float(
                    (actual - evidence["projection_delta"]).norm()
                    / evidence["projection_delta"].norm()
                )
                assert error < 1e-6, error
                assert all(
                    torch.equal(v, student.state_dict()[k])
                    for k, v in merged.state_dict().items()
                    if k != "model.visual.proj"
                )
                mapping[preset] = dict(
                    compiled_projection_relative_error=error,
                    ideal_partner_errors={
                        t: float((evidence["projection_delta"] @ heads[t].weight.double().T).norm())
                        for t in cfg["preserved_query_heads"]
                    },
                    actual_partner_errors={
                        t: float((actual @ heads[t].weight.double().T).norm())
                        for t in cfg["preserved_query_heads"]
                    },
                )
                assert max(mapping[preset]["ideal_partner_errors"].values()) < 1e-8
            measure(preset, attack, "standalone", restored, target)
            for task in cfg["evaluation_tasks"]:
                measure(preset, attack, "fusion", merged, task)
            del merged, restored
    for row in rows:
        value = raw[row["key"]]
        labels = value["labels"]
        direct = f"{row['preset']}/direct/{row['group']}/{row['dataset']}"
        control = f"original/{row['attack']}/{row['group']}/{row['dataset']}"
        row["same_preset_direct_change"] = paired_counts(
            raw[direct]["prediction"].eq(labels), value["prediction"].eq(labels)
        )
        row["matched_original_change"] = paired_counts(
            raw[control]["prediction"].eq(labels), value["prediction"].eq(labels)
        )
    save_json(out / "results.json", rows)
    save_json(out / "compilation_checks.json", mapping)
    assert len(rows) == len(cfg["attacks"]) * 2 * (len(cfg["evaluation_tasks"]) + 1)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            full_evaluations=len(rows),
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            parent_raw_sha256=read(Path(cfg["parent"]) / "metadata.json")["raw_sha256"],
            author_normalized_sha256=sha,
            fixed_reset_students_identical=True,
            teacher_queries_per_control=len(images),
            query_fits=2,
            teacher_ground_truth_or_hidden_features_used=False,
            evaluation_tasks=cfg["evaluation_tasks"],
            eurosat_evaluated=False,
            eight_task_mean=False,
            dataset_source_sha256={
                str(p): digest(p)
                for p in sorted(Path("external/secure-merge-attack/datasets").glob("*.py"))
            },
            defense_success_claim=False,
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
    for n, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / n) == sha
    root, fusion, models, release, heads, data, indices, _images = load_inputs(cfg)
    assert indices == read(out / "attack_indices.json")
    target = cfg["target"]
    anchor = models["anchor"]
    original = models[target]
    reset_control = branch_reset(original, anchor, cfg, "support_reset")
    assert state_equal(reset_control, branch_reset(release, anchor, cfg, "support_reset"))
    student = merge_all(dict(models, **{target: reset_control}), root, 0.2)
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 60
    fits_checked = 0
    for preset, teacher in [("original", original), ("protected", release)]:
        for attack in cfg["attacks"]:
            if attack == "direct":
                restored = teacher
            elif attack == "aga":
                restored = copy.deepcopy(teacher).cpu().eval()
                edits = torch.load(
                    out / "artifacts" / f"{preset}_aga_edits.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for k, v in edits.items():
                    restored.state_dict()[k].copy_(v)
            elif attack == "support_reset_query":
                evidence = torch.load(
                    out / "artifacts" / f"{preset}_query.pt", map_location="cpu", weights_only=True
                )
                h = evidence["design"].double().cuda()
                residual = evidence["residual"].double().cuda()
                update, fit = ridge_update(h, residual, cfg["ridge_ratio"], solver="primal")
                error = float(
                    (update.cpu() - evidence["update"]).norm() / evidence["update"].norm()
                )
                assert error < 1e-6, error
                delta = update.cpu() @ evidence["inverse"]
                assert float((delta - evidence["projection_delta"]).norm() / delta.norm()) < 1e-6
                restored = copy.deepcopy(reset_control).cpu().eval()
                restored.model.visual.proj.copy_(evidence["restored_projection"])
                del h, residual, update
                fits_checked += 1
            else:
                restored = branch_reset(teacher, anchor, cfg, attack)
            merged = merge_all(dict(models, **{target: restored}), root, 0.2)
            if attack == "support_reset_query":
                expected = student.model.visual.proj.double() + evidence["projection_delta"]
                assert (
                    float((merged.model.visual.proj.double() - expected).norm() / expected.norm())
                    < 1e-6
                )
            for row in [r for r in rows if r["preset"] == preset and r["attack"] == attack]:
                task = row["dataset"]
                value = raw[row["key"]]
                labels = truth(data[task], task)
                assert torch.equal(value["labels"], labels)
                assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
                assert row["correct"] == int(value["prediction"].eq(labels).sum())
                assert row["top1"] == row["correct"] / row["n"]
                for field, reference in [
                    ("same_preset_direct_change", f"{preset}/direct/{row['group']}/{task}"),
                    ("matched_original_change", f"original/{attack}/{row['group']}/{task}"),
                ]:
                    assert row[field] == paired_counts(
                        raw[reference]["prediction"].eq(labels), value["prediction"].eq(labels)
                    )
                model = restored if row["group"] == "standalone" else merged
                sink = []
                evaluate(
                    model,
                    heads[task],
                    make_loader(data[task].test_dataset, list(range(128)), 128),
                    "cuda",
                    prediction_sink=lambda p, _y: sink.append(p),
                )
                assert torch.equal(torch.cat(sink), value["prediction"][:128])
            del merged, restored
    for task in cfg["evaluation_tasks"]:
        key = f"support_reset/fusion/{task}"
        assert torch.equal(
            raw[f"original/{key}"]["prediction"], raw[f"protected/{key}"]["prediction"]
        )
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            all_full_counts_and_paired_checks=60,
            official_prefixes_replayed=60,
            independent_primal_query_fits=fits_checked,
            fixed_public_reset_state_identity_checked=True,
            aga_frozen_outputs_replayed=True,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            no_defense_success_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_release_audit.json")
    parser.add_argument("--output", default="outputs/2026-10-11/local_release_audit")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
