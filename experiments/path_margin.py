"""One signed, class-margin-aware edit using frozen path-subspace geometry.

This is a calibration-dependent analytic parameter edit, not a data-free symmetry.
No gradient/backward call, iterative weight training, new FD search or partner use.
Public support reset remains an exact counterexample for this additive family.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import reset
from eight_task_benchmark import author_recover, digest, merge_all
from functional_contrast import features
from local_erasure import cache, rescale, sites
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_subspace import apply, context, edits_for, read
from score_recovery import head_inverse, recover
from subspace_probe import save_json


def margin_geometry(geometry, baseline, head):
    """Teacher pseudo-labels, every expert rival, nearest path rival; no GT labels."""
    weight = head.weight.detach().double().cpu()
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    scores = {name: value.double() @ weight.T for name, value in baseline.items()}
    teacher = scores["expert"].argmax(-1)
    n, classes = scores["expert"].shape
    rivals = torch.arange(classes).expand(n, -1)
    rivals = rivals[rivals.ne(teacher[:, None])].reshape(n, classes - 1)
    rows = torch.arange(n)[:, None]
    contrasts = weight[teacher[:, None]] - weight[rivals]
    own_margin = scores["expert"][rows, teacher[:, None]] - scores["expert"][rows, rivals]
    assert float(own_margin.min()) > 1e-8
    own = torch.einsum("nkf,nfd->nkd", contrasts, geometry["derivatives"]["expert"].double())
    relative = (own / own_margin[..., None]).flatten(0, 1)
    proxy, detail = [], {}
    for name, derivative in geometry["derivatives"].items():
        if name == "expert":
            continue
        masked = scores[name].clone()
        masked.scatter_(1, teacher[:, None], -torch.inf)
        rival = masked.argmax(-1)
        margin = scores[name][torch.arange(n), teacher] - masked.max(-1).values
        eligible = margin > 1e-8
        assert int(eligible.sum()) > 0
        contrast = weight[teacher] - weight[rival]
        change = torch.einsum("nf,nfd->nd", contrast, derivative.double())
        proxy.append((change[eligible] / margin[eligible, None]).mean(0))
        detail[name] = dict(rival=rival, margin=margin, eligible=eligible, derivative=change)
    return relative, torch.stack(proxy).mean(0), teacher, detail


def solve_margin(geometry, baseline, head, expert, cfg):
    a, g, teacher, paths = margin_geometry(geometry, baseline, head)
    h = a.T @ a / len(a)
    rho = cfg["regularizer_trace_fraction"] * float(h.trace()) / len(h) + 1e-12
    q = h + rho * torch.eye(len(h), dtype=torch.float64)
    raw = torch.linalg.solve(q, -g)
    direction = raw / raw.norm()
    assert float(g @ direction) < 0
    own_relative = float((a @ direction).abs().max())
    edits = edits_for(geometry["basis"], direction)
    from critical_subspace import parameters

    weight_relative = max(
        float(item["weight"].norm() / parameters(expert.state_dict(), site)[1].double().norm())
        for site, item in edits.items()
    )
    scale = min(
        cfg["maximum_relative_margin_change"] / own_relative,
        cfg["maximum_matrix_relative_change"] / weight_relative,
    )
    coefficients = scale * direction
    residual = float((q @ raw + g).norm() / g.norm())
    assert residual < 1e-8
    info = dict(
        rho=rho,
        solve_relative_residual=residual,
        coefficient_l2=float(coefficients.norm()),
        maximum_predicted_expert_relative_margin_change=float((a @ coefficients).abs().max()),
        maximum_matrix_relative_change=scale * weight_relative,
        signed_proxy_objective_change=float(g @ coefficients),
        pseudo_label_count=len(teacher),
        calibration_ground_truth_used=False,
        future_partners_used=False,
        backward_calls=0,
        iterative_weight_updates=0,
        analytic_parameter_solves=1,
        predicted_paths={
            name: dict(
                eligible=int(item["eligible"].sum()),
                mean_relative_margin_change=float(
                    ((item["derivative"] @ coefficients) / item["margin"])[item["eligible"]].mean()
                ),
                predicted_pseudo_agreement=int(
                    ((item["margin"] + item["derivative"] @ coefficients) > 0).sum()
                ),
            )
            for name, item in paths.items()
        },
    )
    return coefficients, info


@torch.no_grad()
def sample_response(reference, changed, head, images):
    reference.cuda().eval()
    before = features(reference, images)
    reference.cpu()
    changed.cuda().eval()
    after = features(changed, images)
    changed.cpu()
    weight = head.weight.detach().double().cpu()
    old, new = before.double() @ weight.T, after.double() @ weight.T
    labels = old.argmax(-1)
    mask = torch.nn.functional.one_hot(labels, num_classes=old.shape[1]).bool()
    old_margin = (
        old.gather(1, labels[:, None]).squeeze(1) - old.masked_fill(mask, -torch.inf).max(-1).values
    )
    new_margin = (
        new.gather(1, labels[:, None]).squeeze(1) - new.masked_fill(mask, -torch.inf).max(-1).values
    )
    return dict(
        n=len(images),
        pseudo_prediction_changes=int(old.argmax(-1).ne(new.argmax(-1)).sum()),
        mean_old_margin=float(old_margin.mean()),
        mean_new_margin_to_old_winner=float(new_margin.mean()),
        mean_margin_change=float((new_margin - old_margin).mean()),
        feature_rms=float(
            (after.double() - before.double()).square().sum().div(len(images)).sqrt()
        ),
    )


def check_parent(cfg):
    parent = Path(cfg["parent"])
    metadata = read(parent / "metadata.json")
    assert metadata["completed"] and read(parent / "validation.json")["status"] == "passed"
    for name, value in metadata["checkpoint_sha256"].items():
        assert digest(Path("checkpoints/ViT-B-32") / name) == value
    for name, value in metadata["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == value
    return parent, metadata


@torch.no_grad()
def run(args):
    cfg = read(args.config)
    parent, prior = check_parent(cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    started = time.perf_counter()
    save_json(out / "protocol.json", cfg)
    root, load, fusion, models = context(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    from dataset.registry import get_dataset

    anchor = models["anchor"]
    indices = read(parent / "training_indices.json")
    save_json(out / "training_indices.json", indices)
    data, images, releases, constructions, responses, freeze = {}, {}, {}, {}, {}, {}
    heads = {}
    for target in cfg["tasks"]:
        # Only the current target's expert, public head, anchor and train images enter here.
        expert = load(str(root / target / "finetuned.pt")).cpu().eval()
        heads[target] = load(str(root / f"head_{target}.pt")).cpu().eval()
        data[target] = get_dataset(
            target, anchor.val_preprocess, "data", batch_size=128, num_workers=2
        )
        images[target] = {
            split: torch.stack([data[target].train_dataset[i][0] for i in indices[target][split]])
            for split in ("calibration", "confirmation")
        }
        geometry = torch.load(
            parent / f"artifacts/{target}_geometry.pt", map_location="cpu", weights_only=True
        )
        baseline = {}
        for name, beta in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
            reference = expert if name == "expert" else rescale(expert, anchor, beta)
            reference.cuda().eval()
            baseline[name] = features(reference, images[target]["calibration"])
            reference.cpu()
            if name != "expert":
                del reference
        coefficients, constructions[target] = solve_margin(
            geometry, baseline, heads[target], expert, cfg
        )
        edits = edits_for(geometry["basis"], coefficients)
        release = apply(expert, edits)
        releases[target] = release
        state = {
            k: v.clone()
            for k, v in release.state_dict().items()
            if not torch.equal(v, expert.state_dict()[k])
        }
        torch.save(state, artifacts / f"{target}_edits.pt")
        torch.save(
            dict(baseline=baseline, coefficients=coefficients),
            artifacts / f"{target}_construction.pt",
        )
        original_reset = reset(expert, anchor, sites(cfg["layers"][target]), "public_support")
        protected_reset = reset(release, anchor, sites(cfg["layers"][target]), "public_support")
        assert all(
            torch.equal(v, original_reset.state_dict()[k])
            for k, v in protected_reset.state_dict().items()
        )
        freeze[target] = dict(
            edits_sha256=digest(artifacts / f"{target}_edits.pt"),
            construction_sha256=digest(artifacts / f"{target}_construction.pt"),
            parent_geometry_sha256=prior["raw_sha256"][f"{target}_geometry.pt"],
            public_support_reset_entire_state_identical=True,
            edited_keys=list(state),
        )
        del original_reset, protected_reset
        print("RELEASE_FROZEN", target, constructions[target], flush=True)
        for name, beta in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
            reference = expert if name == "expert" else rescale(expert, anchor, beta)
            changed = release if name == "expert" else apply(reference, edits, beta)
            for split in ("calibration", "confirmation"):
                responses[f"{target}/{name}/{split}"] = sample_response(
                    reference, changed, heads[target], images[target][split]
                )
            if name != "expert":
                del reference, changed
        del expert, geometry
    save_json(out / "construction.json", constructions)
    save_json(
        out / "release_freeze.json",
        dict(models=freeze, future_partners_loaded=False, target_head_used=True),
    )
    save_json(out / "responses.json", responses)
    # No candidate revision below this point. Partners/test labels enter only now.
    for task in fusion["tasks"]:
        models[task] = load(str(root / task / "finetuned.pt")).cpu().eval()
    for task in cfg["evaluation_tasks"]:
        if task not in heads:
            heads[task] = load(str(root / f"head_{task}.pt")).cpu().eval()
        if task not in data:
            data[task] = get_dataset(
                task, anchor.val_preprocess, "data", batch_size=128, num_workers=2
            )
    loaders = {task: cache(data[task]) for task in cfg["evaluation_tasks"]}
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == prior["original_author_aga_sha256"]
    control = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    parent_rows = {row["id"]: row for row in read(parent / "results.json")}
    attack_indices = read(parent / "attack_indices.json")
    save_json(out / "attack_indices.json", attack_indices)
    rows, raw, fits, mappings = [], {}, {}, []

    def measure(target, attack, group, model, task):
        assert time.perf_counter() - started < cfg["time_limit_seconds"]
        key = f"{target}/protected/{attack}/{group}/{task}"
        p, y = [], []
        score = evaluate(
            model,
            heads[task],
            loaders[task],
            "cuda",
            prediction_sink=lambda a, b: (p.append(a), y.append(b)),
        )
        raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
        rows.append(
            dict(
                id=key,
                target=target,
                preset="protected",
                attack=attack,
                group=group,
                dataset=task,
                reused=False,
                **score,
            )
        )
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        print("RESULT", key, score["correct"], score["n"], flush=True)

    for target in cfg["tasks"]:
        release = releases[target]
        for beta in cfg["betas"]:
            proxy = rescale(release, anchor, beta)
            measure(target, f"beta_{beta}", "proxy", proxy, target)
            del proxy
        for attack in cfg["attacks"]:
            if attack == "direct":
                restored = release
            elif attack == "aga":
                restored = author_recover(release, anchor, author)
            else:
                assert attack == "readout"
                query_images = torch.stack(
                    [data[target].train_dataset[i][0] for i in attack_indices[target]]
                )
                inverse, _ = head_inverse(
                    heads[target], [h for t, h in heads.items() if t != target]
                )
                student = merge_all(dict(models, **{target: release}), root, cfg["coefficient"])
                restored, fits[target], evidence = recover(
                    release, student, heads[target], inverse, query_images, cfg["ridge_ratio"]
                )
                expected = student.model.visual.proj.double() + evidence["projection_delta"]
                torch.save(evidence, artifacts / f"{target}_readout.pt")
                del student
            if attack != "readout":
                measure(target, attack, "standalone", restored, target)
            merged = merge_all(dict(models, **{target: restored}), root, cfg["coefficient"])
            if attack == "direct":
                baseline = merge_all(models, root, cfg["coefficient"])
                for split in ("calibration", "confirmation"):
                    responses[f"{target}/real_fusion/{split}"] = sample_response(
                        baseline, merged, heads[target], images[target][split]
                    )
                save_json(out / "responses.json", responses)
                del baseline
            if attack == "readout":
                error = float(
                    (merged.model.visual.proj.double() - expected).norm() / expected.norm()
                )
                assert error < 1e-6
                mappings.append(dict(target=target, relative_error=error))
            for task in cfg["evaluation_tasks"]:
                measure(target, attack, "fusion", merged, task)
            del restored, merged
    # Reuse byte-verified parent controls, not rerun identical checkpoints.
    keys = [row["id"].replace("/protected/", "/original/") for row in rows]
    keys += [
        f"{t}/original/support_reset/{g}/{d}"
        for t in cfg["tasks"]
        for g, ds in [("standalone", [t]), ("fusion", cfg["evaluation_tasks"])]
        for d in ds
    ]
    for key in keys:
        raw[key] = control[key]
        rows.append(dict(parent_rows[key], reused=True, reused_parent=cfg["parent"]))
        if "/support_reset/" in key:
            protected_key = key.replace("/original/", "/protected/")
            raw[protected_key] = control[key]
            rows.append(
                dict(
                    parent_rows[key],
                    id=protected_key,
                    preset="protected",
                    reused=True,
                    reused_parent=cfg["parent"],
                    reuse_reason="entire reset state byte-identical to original reset",
                )
            )
    for row in rows:
        value = raw[row["id"]]
        original = raw[row["id"].replace("/protected/", "/original/")]
        assert torch.equal(value["labels"], original["labels"])
        row["matched"] = paired_counts(
            original["prediction"].eq(original["labels"]), value["prediction"].eq(value["labels"])
        )
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    save_json(out / "fitting.json", fits)
    save_json(out / "projection_checks.json", mappings)
    helpers = ["path_subspace.py", *prior["helper_sha256"].keys()]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - started,
            rows=len(rows),
            actual_full_evaluations=sum(not r["reused"] for r in rows),
            reused_metric_rows=sum(r["reused"] for r in rows),
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={name: digest(Path(__file__).with_name(name)) for name in helpers},
            checkpoint_sha256=prior["checkpoint_sha256"],
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            parent_metadata_sha256=digest(parent / "metadata.json"),
            parent_results_sha256=digest(parent / "results.json"),
            parent_indices_sha256=digest(parent / "training_indices.json"),
            parent_predictions_sha256=prior["raw_sha256"]["predictions.pt"],
            original_author_aga_sha256=author_hash,
            constructor_candidates=len(cfg["tasks"]),
            calibration_ground_truth_used=False,
            target_pseudo_labels_used=True,
            future_partners_used=False,
            backward_calls=0,
            iterative_weight_updates=0,
            analytic_parameter_solves=len(cfg["tasks"]),
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
    run(args)
