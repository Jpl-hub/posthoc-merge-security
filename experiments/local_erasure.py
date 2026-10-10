"""Full tradeoff audit for one deterministic task-update deletion per expert.

Masking is an existing mechanism. This experiment tests utility and multiple
recoveries without acceptance thresholds, fine-tuning or future-partner use
during release construction. Heavy artifacts stay on the server.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from critical_subspace import parameters, reset
from eight_task_benchmark import author_recover, digest, merge_all
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, query_readout, replace_components
from polynomial_probe import ridge_update
from subspace_probe import save_json


def sites(layers):
    return {f"{kind}:{layer}": {} for layer in layers for kind in ("v", "out", "up", "down")}


@torch.no_grad()
def rescale(model, anchor, scale):
    result = copy.deepcopy(model).cpu().eval()
    own, base, state = model.state_dict(), anchor.state_dict(), result.state_dict()
    for key, value in state.items():
        if value.is_floating_point():
            value.copy_(base[key] + scale * (own[key] - base[key]))
    return result


@torch.no_grad()
def spectral_cleanup(model, anchor, svd):
    result = copy.deepcopy(model).cpu().eval()
    rows = []
    for site in sites(range(12)):
        _, weight, _ = parameters(result.state_dict(), site)
        _, base, _ = parameters(anchor.state_dict(), site)
        delta = (weight.double() - base.double()).cuda()
        energy = float(delta.square().sum())
        if energy == 0:
            rows.append(dict(site=site, skipped_zero=True))
            continue
        _, singular, vectors = svd(delta, full_matrices=False)
        v = vectors[:, 0]
        removed = (delta @ v[:, None]) @ v[None, :]
        weight.copy_((weight.double() - removed.cpu()).float())
        rows.append(
            dict(
                site=site,
                skipped_zero=False,
                removed_update_energy_fraction=float(singular[0].square()) / energy,
            )
        )
    return result, rows


@torch.no_grad()
def reconstruct_projection(release, student, head, images, ratio, coefficient):
    """No original target weights or teacher hidden features enter this attack."""
    pseudo, _ = query_readout(release, head, images)
    _, design = query_readout(student, head, images, capture_features=True)
    classes = torch.nn.functional.one_hot(pseudo.cuda(), len(head.weight)).double()
    public_head = head.weight.double().cuda()
    assert torch.count_nonzero(head.bias) == 0
    gram = public_head @ public_head.T
    inverse = torch.linalg.solve(gram, public_head)
    identity_error = float(
        (public_head @ inverse.T - torch.eye(len(public_head), device="cuda")).norm()
    )
    assert identity_error < 1e-8
    projection = student.model.visual.proj.double().cuda()
    residual = classes - design @ projection @ public_head.T
    update, fit = ridge_update(design, residual, ratio)
    delta = update @ inverse
    mapping_error = float((delta @ public_head.T - update).norm() / update.norm().clamp_min(1e-30))
    assert mapping_error < 1e-8
    restored = copy.deepcopy(release).cpu().eval()
    restored.model.visual.proj.copy_(
        (release.model.visual.proj.double().cuda() + delta / coefficient).float().cpu()
    )
    fit.update(
        samples=len(images),
        classes=len(public_head),
        pseudo_label_fit_correct=int(
            (design @ (projection + delta) @ public_head.T).argmax(-1).cpu().eq(pseudo).sum()
        ),
        head_right_inverse_error=identity_error,
        effective_update_mapping_error=mapping_error,
        calibration_ground_truth_used=False,
        teacher_hidden_features_used=False,
    )
    evidence = dict(
        pseudo_labels=pseudo,
        design=design.cpu(),
        residual=residual.cpu(),
        update=update.cpu(),
        projection_delta=delta.cpu(),
        restored_projection=restored.model.visual.proj.detach().clone(),
    )
    return restored, fit, evidence


def cache(dataset):
    from torch.utils.data import DataLoader, TensorDataset

    batches = list(dataset.test_loader)
    return DataLoader(
        TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
        batch_size=128,
        pin_memory=True,
    )


@torch.no_grad()
def run(args):
    config = json.loads(Path(args.config).read_bytes())
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    assert config["coefficient"] == fusion["coefficient"] == 0.2
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", config)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt"))
    originals, releases, manifests = {}, {}, {}
    selection = json.loads((Path(config["profile_parent"]) / "selection.json").read_bytes())
    for task in config["tasks"]:
        assert selection[task]["layers"] == config["layers"][task]
        assert not selection[task]["partner_used"]
        original = load(str(root / task / "finetuned.pt")).cpu().eval()
        released = reset(original, anchor, sites(config["layers"][task]), "public_support")
        own, published = original.state_dict(), released.state_dict()
        edits = {
            key: value.clone()
            for key, value in published.items()
            if not torch.equal(own[key], value)
        }
        torch.save(edits, artifacts / f"{task}_edits.pt")
        manifests[task] = dict(
            edited_keys=list(edits), edits_sha256=digest(artifacts / f"{task}_edits.pt")
        )
        originals[task], releases[task] = original, released
    save_json(
        out / "release_freeze.json",
        dict(models=manifests, future_partners_loaded=False, optimizer_steps=0, constructor_fits=0),
    )
    models = {task: load(str(root / task / "finetuned.pt")) for task in fusion["tasks"]}
    models.update(originals)
    models["anchor"] = anchor
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    loaders, data, heads = {}, {}, {}
    for task in config["evaluation_tasks"]:
        data[task] = get_dataset(
            task, anchor.val_preprocess, args.data, batch_size=128, num_workers=2
        )
        assert len(data[task].test_dataset) == fusion["expected_test_counts"][task]
        loaders[task] = cache(data[task])
        heads[task] = load(str(root / f"head_{task}.pt"))
        print("OFFICIAL_CACHE", task, len(data[task].test_dataset), flush=True)
    indices = json.loads((Path(config["profile_parent"]) / "training_indices.json").read_bytes())
    attack_indices = {
        task: indices[task]["selection"][: config["attacker_samples"]] for task in config["tasks"]
    }
    save_json(out / "attack_indices.json", attack_indices)
    attack_images = {
        task: torch.stack([data[task].train_dataset[i][0] for i in attack_indices[task]])
        for task in config["tasks"]
    }
    author, author_hash = load_author(
        Path(args.author_repo) / "utils_attack_B32.py", include_wrappers=True
    )
    assert author_hash == json.loads(Path("configs/sources.json").read_bytes())["AGA"]["sha256"]
    svd_config = json.loads(Path("configs/nufilt_alignment.json").read_bytes())
    svd = author_svd(args.nufilt_repo, svd_config)
    rows, raw, fits, spectral, projection_checks = [], {}, {}, {}, []
    actual_evaluations = 0

    def measure(target, preset, attack, group, model, task):
        nonlocal actual_evaluations
        key = f"{target}/{preset}/{attack}/{group}/{task}"
        reused = f"{config['tasks'][0]}/original/direct/fusion/{task}"
        reference = None
        if (
            target != config["tasks"][0]
            and preset == "original"
            and attack == "direct"
            and group == "fusion"
        ):
            item = raw[reused]
            score = {
                k: next(r for r in rows if r["id"] == reused)[k]
                for k in ["correct", "n", "top1", "evaluation_seconds", "peak_gpu_bytes"]
            }
            reference = reused
        else:
            p, y = [], []
            score = evaluate(
                model,
                heads[task],
                loaders[task],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            item = dict(prediction=torch.cat(p), labels=torch.cat(y))
            actual_evaluations += 1
        raw[key] = item
        rows.append(
            dict(
                id=key,
                target=target,
                preset=preset,
                attack=attack,
                group=group,
                dataset=task,
                reused_from=reference,
                **score,
            )
        )
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        print("RESULT", key, score["correct"], score["n"], flush=True)

    for target in config["tasks"]:
        for preset, release in [("original", originals[target]), ("erased", releases[target])]:
            for attack in config["attacks"]:
                assert time.perf_counter() - started < config["time_limit_seconds"]
                if attack == "direct":
                    restored = release
                elif attack == "aga":
                    restored = author_recover(release, anchor, author)
                elif attack in ("reset_attention", "reset_mlp"):
                    restored, _ = replace_components(
                        release, anchor, range(12), "attn" if attack == "reset_attention" else "mlp"
                    )
                elif attack == "spectral":
                    restored, evidence = spectral_cleanup(release, anchor, svd)
                    spectral[f"{target}/{preset}"] = evidence
                    save_json(out / "spectral.json", spectral)
                elif attack == "target_weight_one":
                    restored = rescale(release, anchor, 1 / config["coefficient"])
                else:
                    assert attack == "readout"
                    student = merge_all(
                        dict(models, **{target: release}), root, config["coefficient"]
                    )
                    restored, fit, evidence = reconstruct_projection(
                        release,
                        student,
                        heads[target],
                        attack_images[target],
                        config["readout_ridge_ratio"],
                        config["coefficient"],
                    )
                    fits[f"{target}/{preset}"] = fit
                    torch.save(evidence, artifacts / f"{target}_{preset}_readout.pt")
                    save_json(out / "fitting.json", fits)
                    expected = student.model.visual.proj.double() + evidence["projection_delta"]
                    del student
                if attack not in ("target_weight_one", "readout"):
                    measure(target, preset, attack, "standalone", restored, target)
                combined = merge_all(
                    dict(models, **{target: restored}), root, config["coefficient"]
                )
                if attack == "readout":
                    error = float(
                        (combined.model.visual.proj.double() - expected).norm() / expected.norm()
                    )
                    assert error < 1e-6
                    projection_checks.append(
                        dict(target=target, preset=preset, merged_projection_relative_error=error)
                    )
                for task in config["evaluation_tasks"]:
                    measure(target, preset, attack, "fusion", combined, task)
                del combined, restored
    for row in rows:
        item = raw[row["id"]]
        control = raw[f"{row['target']}/original/{row['attack']}/{row['group']}/{row['dataset']}"]
        assert torch.equal(item["labels"], control["labels"])
        row["matched"] = paired_counts(
            control["prediction"].eq(control["labels"]), item["prediction"].eq(item["labels"])
        )
    save_json(out / "results.json", rows)
    summary = []
    for target in config["tasks"]:
        points = [
            r
            for r in rows
            if r["target"] == target
            and r["preset"] == "erased"
            and r["group"] == "fusion"
            and r["dataset"] == target
        ]
        best = max(points, key=lambda r: r["top1"])
        controls = [
            r
            for r in rows
            if r["target"] == target
            and r["preset"] == "original"
            and r["group"] == "fusion"
            and r["dataset"] == target
        ]
        control_best = max(controls, key=lambda r: r["top1"])
        utility = next(r for r in rows if r["id"] == f"{target}/erased/direct/standalone/{target}")
        summary.append(
            dict(
                target=target,
                utility_loss_pp=utility["matched"]["loss_pp"],
                best_tested_recovery=best["attack"],
                protected_best_accuracy=best["top1"],
                unprotected_best_accuracy=control_best["top1"],
                best_to_best_gap_pp=100 * (control_best["top1"] - best["top1"]),
                selected_after_results=False,
            )
        )
    save_json(
        out / "decision.json",
        dict(
            points=summary,
            accepted_by_threshold=False,
            threshold_used=False,
            new_algorithm_claim=False,
            full_sloa_tested=False,
            protection_claim="Observed finite-budget tradeoff only; inspect partner losses and all matched controls",
        ),
    )
    paths = [
        "zeroshot.pt",
        *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
        *[f"head_{t}.pt" for t in config["evaluation_tasks"]],
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            fusion_config_sha256=digest(config["fusion_config"]),
            source_helpers_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "critical_subspace.py",
                    "eight_task_benchmark.py",
                    "fine_subspace.py",
                    "joint_posthoc.py",
                    "low_activity_posthoc.py",
                    "mechanisms.py",
                    "nufilt_alignment.py",
                    "operator_diagnostics.py",
                    "polynomial_probe.py",
                    "subspace_probe.py",
                ]
            },
            checkpoint_sha256={name: digest(root / name) for name in paths},
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            profile_selection_sha256=digest(Path(config["profile_parent"]) / "selection.json"),
            profile_indices_sha256=digest(Path(config["profile_parent"]) / "training_indices.json"),
            original_author_aga_sha256=author_hash,
            author_helpers_sha256={
                name: digest(Path(args.author_repo) / name)
                for name in ["task_vectors.py", "modeling.py", "utils.py"]
            },
            upstream_svd=svd_config,
            merged_projection_checks=projection_checks,
            elapsed_seconds=time.perf_counter() - started,
            reported_rows=len(rows),
            actual_full_test_evaluations=actual_evaluations,
            defender_optimizer_steps=0,
            defender_fits=0,
            constructor_future_partner_access=False,
            calibration_truth_for_attack=False,
            official_tests_used_for_selection=False,
            full_eight_task_average=False,
            calibration_reused=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_erasure.json")
    parser.add_argument("--output", default="outputs/2026-10-11/local_erasure")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--nufilt-repo", default="external/NUFILT")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
