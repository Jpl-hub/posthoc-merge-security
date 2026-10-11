"""One fixed score-recovery attack with public partner response constraints.

Reuses target queries from a failed frozen release. This is an attack audit,
not a new protection, full SLOA, or a parameter/coefficient search.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest
from local_erasure import cache
from operator_diagnostics import paired_counts, query_readout
from path_subspace import read
from query_partner_cost import contrasts, holders, runtime, truth
from scipy.linalg import solve_sylvester
from subspace_probe import make_loader, save_json


def verify(pins):
    for name, pin in pins.items():
        assert digest(name) == pin, name


@torch.no_grad()
def output_null(heads, cfg):
    b = torch.cat([contrasts(heads[t]) for t in cfg["exact_heads"]])
    _, s, vh = torch.linalg.svd(b, full_matrices=True)
    rank = int(s.gt(s[0] * cfg["rcond"]).sum())
    q = vh[rank:].T.contiguous()
    assert rank == 204 and q.shape == (512, 308)
    assert float((b @ q).norm()) < 1e-9
    assert float((q.T @ q - torch.eye(308)).norm()) < 1e-10
    return q


@torch.no_grad()
def solve(h, scores, p, c_raw, q, partners, cfg, independent=False):
    """Strictly convex quadratic, reduced to a weighted Sylvester equation."""
    scale = c_raw.norm() / len(c_raw) ** 0.5
    c = c_raw / scale
    residual = (scores[:, 1:] - scores[:, :1]) / scale - h @ p @ c.T
    u, s, vh = torch.linalg.svd(c @ q, full_matrices=False)
    assert len(s) == 46 and float(s[-1] / s[0]) > cfg["rcond"]
    gt = h.T @ h / len(h)
    d0 = sum(x.T @ x / (len(x) * q.shape[1]) for x in partners.values())
    d0 *= cfg["partner_weight"] / len(partners)
    ridge = cfg["ridge_ratio"] * d0.trace() / len(d0)
    assert float(ridge) > 0
    d = d0 + ridge * torch.eye(len(d0), dtype=torch.float64)
    chol = torch.linalg.cholesky(d)
    lower = torch.linalg.solve_triangular
    a = lower(chol, gt, upper=False)
    a = lower(chol, a.T, upper=False).T
    a = (a + a.T) / 2
    rhs = h.T @ residual @ u * s / (len(h) * len(c))
    rhs_white = lower(chol, rhs, upper=False)
    k = s.square() / len(c)
    if independent:
        eigenvalues, eigenvectors = torch.linalg.eigh(a)
        y = eigenvectors @ ((eigenvectors.T @ rhs_white) / (eigenvalues[:, None] * k + 1))
    else:
        y = torch.from_numpy(
            solve_sylvester(a.numpy(), np.diag((1 / k).numpy()), (rhs_white / k).numpy())
        ).contiguous()
    z = lower(chol.T, y, upper=True)
    x = z @ vh
    delta = x @ q.T
    normal_residual = gt @ z * k + d @ z - rhs
    relative = float(normal_residual.norm() / rhs.norm())
    assert relative < 1e-7 and torch.isfinite(delta).all()
    target_error = h @ delta @ c.T - residual
    partner_error = {t: float((value @ delta).square().mean()) for t, value in partners.items()}
    target_loss = float(target_error.square().mean())
    partner_loss = (
        cfg["partner_weight"]
        * sum(float((value @ x).square().mean()) for value in partners.values())
        / len(partners)
    )
    regularizer = float(ridge * x.square().sum())
    fit = dict(
        target_samples=len(h),
        target_score_contrasts=len(c),
        output_null_dimension=q.shape[1],
        teacher_score_scale=float(scale),
        target_loss_before=float(residual.square().mean()),
        target_loss_after=target_loss,
        partner_output_mse=partner_error,
        partner_loss=partner_loss,
        ridge_coefficient=float(ridge),
        regularizer=regularizer,
        objective_after=target_loss + partner_loss + regularizer,
        normal_equation_relative_residual=relative,
        projection_delta_norm=float(delta.norm()),
    )
    # Class 0 has score difference zero; include it for the real argmax.
    diffs = h @ (p + delta) @ c.T
    reconstructed = torch.cat([torch.zeros(len(h), 1, dtype=torch.float64), diffs], dim=1)
    fit["calibration_teacher_argmax_agreement"] = int(
        reconstructed.argmax(-1).eq(scores.argmax(-1)).sum()
    )
    return delta, fit


def load_parents(cfg):
    parent = Path(cfg["parent"])
    cost = Path(cfg["cost_parent"])
    assert read(cost / "validation.json")["status"] == "passed"
    meta = read(cost / "metadata.json")
    verify(meta["parent_input_sha256"])
    verify(meta["dataset_source_sha256"])
    for name, pin in meta["raw_sha256"].items():
        assert digest(cost / "artifacts" / name) == pin
    cost_cfg = read(cost / "protocol.json")
    assert cost_cfg["parent"] == cfg["parent"] and cost_cfg["target"] == cfg["target"]
    root, load, fusion, heads, states, checks, _ = holders(cost_cfg)
    states = {k: v for k, v in states.items() if k.endswith("direct")}
    raw = torch.load(cost / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    evidence = {
        preset: torch.load(
            parent / "artifacts" / f"DTD_{preset}_readout.pt",
            map_location="cpu",
            weights_only=True,
        )
        for preset in ["original", "protected"]
    }
    return root, fusion, heads, states, raw, evidence, meta


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    root, fusion, heads, states, prior, teachers, parent_meta = load_parents(cfg)
    q = output_null(heads, cfg)
    from dataset.registry import get_dataset

    indices, partner_features = {}, {p: {} for p in teachers}
    for position, task in enumerate(cfg["calibration_partners"]):
        data = get_dataset(
            task, states["original_direct"].val_preprocess, "data", batch_size=128, num_workers=2
        )
        ids = torch.randperm(
            len(data.train_dataset), generator=torch.Generator().manual_seed(cfg["seed"] + position)
        )[: cfg["partner_samples"]].tolist()
        indices[task] = dict(indices=ids, train_count=len(data.train_dataset), labels_used=False)
        images = torch.stack([data.train_dataset[i][0] for i in ids])
        for preset in teachers:
            _, h = query_readout(states[preset + "_direct"], heads[task], images, True)
            partner_features[preset][task] = h.cpu()
        del images, data
    save_json(out / "calibration_indices.json", indices)
    fits, compilation, fit_evidence = {}, [], {}
    c = contrasts(heads[cfg["target"]])
    for preset, teacher in teachers.items():
        direct = states[preset + "_direct"]
        p = direct.model.visual.proj.detach().double().cpu()
        h, scores = teacher["design"].double().cpu(), teacher["teacher_scores"].double().cpu()
        assert h.shape == (256, 768) and scores.shape == (256, 47)
        delta, fit = solve(h, scores, p, c, q, partner_features[preset], cfg)
        model = copy.deepcopy(direct).cpu().eval()
        model.model.visual.proj.copy_((p + delta).float())
        actual = model.model.visual.proj.double() - p
        error = float((actual - delta).norm() / delta.norm())
        assert error < 1e-6
        assert all(
            torch.equal(v, direct.state_dict()[k])
            for k, v in model.state_dict().items()
            if k != "model.visual.proj"
        )
        null_errors = {
            t: float((actual @ contrasts(heads[t]).T).norm()) for t in cfg["exact_heads"]
        }
        ideal_null_errors = {
            t: float((delta @ contrasts(heads[t]).T).norm()) for t in cfg["exact_heads"]
        }
        assert max(ideal_null_errors.values()) < 1e-8
        states[preset + "_conditioned"] = model
        fits[preset] = fit
        compilation.append(
            dict(
                preset=preset,
                relative_float32_error=error,
                actual_contrast_errors=null_errors,
                ideal_contrast_errors=ideal_null_errors,
                only_projection_modified=True,
            )
        )
        fit_evidence[preset] = dict(
            design=h,
            teacher_scores=scores,
            projection=p,
            contrasts=c,
            output_null=q,
            partner_features=partner_features[preset],
            projection_delta=delta,
            compiled_projection=model.model.visual.proj.detach().clone(),
        )
    torch.save(fit_evidence, artifacts / "fits.pt")
    save_json(out / "fitting.json", fits)
    save_json(out / "compilation_checks.json", compilation)
    rows, raw = [], {}
    old_rows = {r["key"]: r for r in read(Path(cfg["cost_parent"]) / "results.json")}
    for task in cfg["evaluation_tasks"]:
        data = get_dataset(
            task, states["original_direct"].val_preprocess, "data", batch_size=128, num_workers=2
        )
        loader, labels = cache(data), truth(data, task)
        for preset in teachers:
            baseline = f"{preset}_direct/{task}"
            value = prior[baseline]
            assert torch.equal(value["labels"], labels)
            raw[baseline] = value
            rows.append(dict(old_rows[baseline], reused=True, reused_from=cfg["cost_parent"]))
            name = preset + "_conditioned"
            key = name + "/" + task
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            ps, ys = [], []
            metric = evaluate(
                states[name],
                heads[task],
                loader,
                "cuda",
                prediction_sink=lambda p, y: (ps.append(p), ys.append(y)),
            )
            value = dict(prediction=torch.cat(ps), labels=torch.cat(ys))
            assert torch.equal(value["labels"], labels)
            assert metric["correct"] == int(value["prediction"].eq(labels).sum())
            assert metric["n"] == fusion["expected_test_counts"][task]
            raw[key] = value
            row = dict(
                key=key,
                preset=name,
                dataset=task,
                reused=False,
                **metric,
                direct_reference=baseline,
                direct_change=paired_counts(
                    prior[baseline]["prediction"].eq(labels), value["prediction"].eq(labels)
                ),
                unconditioned_reference=f"{preset}_query/{task}",
                unconditioned_change=paired_counts(
                    prior[f"{preset}_query/{task}"]["prediction"].eq(labels),
                    value["prediction"].eq(labels),
                ),
            )
            if preset == "protected":
                reference = f"original_conditioned/{task}"
                row.update(
                    original_reference=reference,
                    original_change=paired_counts(
                        raw[reference]["prediction"].eq(labels), value["prediction"].eq(labels)
                    ),
                )
            rows.append(row)
            print("CONDITIONED_RECOVERY", key, metric["correct"], metric["n"], flush=True)
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        del data, loader
    assert len(rows) == 28 and sum(not r["reused"] for r in rows) == 14
    pins = dict(parent_meta["parent_input_sha256"])
    pins.update(
        {
            str(Path(cfg["cost_parent"]) / name): digest(Path(cfg["cost_parent"]) / name)
            for name in [
                "protocol.json",
                "metadata.json",
                "validation.json",
                "results.json",
                "artifacts/predictions.pt",
            ]
        }
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            fresh_full_evaluations=14,
            reused_full_evaluations=14,
            new_teacher_queries=0,
            reused_teacher_queries_per_preset=256,
            partner_train_samples_per_task=cfg["partner_samples"],
            new_fixed_attacker_fits=2,
            defender_fits=0,
            labels_used_for_fitting=False,
            coefficient_search=False,
            full_sloa=False,
            new_defense=False,
            scipy_version=scipy.__version__,
            torch_version=str(torch.__version__),
            checkpoint_sha256=parent_meta["checkpoint_sha256"],
            parent_input_sha256=pins,
            dataset_source_sha256=parent_meta["dataset_source_sha256"],
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            source_manifest=read(args.manifest)["source_manifest"],
            all_eight_task_performance_claim=False,
            eurosat_evaluated=False,
        ),
    )


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and cfg == read(args.config)
    for group in ["source_manifest", "parent_input_sha256", "dataset_source_sha256"]:
        verify(meta[group])
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    runtime()
    root, fusion, heads, states, prior, teachers, _ = load_parents(cfg)
    for name, pin in meta["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    fitted = torch.load(out / "artifacts/fits.pt", map_location="cpu", weights_only=True)
    q = output_null(heads, cfg)
    from dataset.registry import get_dataset

    for preset, evidence in fitted.items():
        assert torch.equal(q, evidence["output_null"])
        assert torch.equal(evidence["design"], teachers[preset]["design"].cpu())
        assert torch.equal(evidence["teacher_scores"], teachers[preset]["teacher_scores"].cpu())
        delta, _ = solve(
            evidence["design"],
            evidence["teacher_scores"],
            evidence["projection"],
            evidence["contrasts"],
            q,
            evidence["partner_features"],
            cfg,
            independent=True,
        )
        error = float((delta - evidence["projection_delta"]).norm() / delta.norm())
        assert error < 1e-6
        p = states[preset + "_direct"].model.visual.proj.detach().double()
        assert torch.equal(p, evidence["projection"])
        assert torch.equal(
            (p + evidence["projection_delta"]).float(), evidence["compiled_projection"]
        )
        compiled = copy.deepcopy(states[preset + "_direct"])
        compiled.model.visual.proj.copy_(evidence["compiled_projection"])
        states[preset + "_conditioned"] = compiled
    for task, record in read(out / "calibration_indices.json").items():
        data = get_dataset(task, states["original_direct"].val_preprocess, "data", 128, 2)
        images = torch.stack([data.train_dataset[i][0] for i in record["indices"][:32]])
        assert len(data.train_dataset) == record["train_count"]
        for preset, evidence in fitted.items():
            _, h = query_readout(states[preset + "_direct"], heads[task], images, True)
            assert torch.equal(h.cpu(), evidence["partner_features"][task][:32])
        del images, data
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 28
    prefixes = 0
    for task in cfg["evaluation_tasks"]:
        data = get_dataset(task, states["original_direct"].val_preprocess, "data", 128, 2)
        labels = truth(data, task)
        loader = make_loader(data.test_dataset, list(range(128)), 128)
        for row in [r for r in rows if r["dataset"] == task]:
            value = raw[row["key"]]
            assert torch.equal(value["labels"], labels)
            assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
            assert row["correct"] == int(value["prediction"].eq(labels).sum())
            assert row["top1"] == row["correct"] / len(labels)
            if row["reused"]:
                assert torch.equal(value["prediction"], prior[row["key"]]["prediction"])
            else:
                for ref, change, source in [
                    ("direct_reference", "direct_change", prior),
                    ("unconditioned_reference", "unconditioned_change", prior),
                    ("original_reference", "original_change", raw),
                ]:
                    if ref in row:
                        assert row[change] == paired_counts(
                            source[row[ref]]["prediction"].eq(labels),
                            value["prediction"].eq(labels),
                        )
            ps = []
            evaluate(
                states[row["preset"]],
                heads[task],
                loader,
                "cuda",
                prediction_sink=lambda p, _y: ps.append(p),
            )
            assert torch.equal(torch.cat(ps), value["prediction"][:128])
            prefixes += 1
        del data, loader
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            independent_eigendecomposition_fits=2,
            partner_feature_prefixes_replayed=6,
            counts_and_pairs_checked=28,
            official_prediction_prefixes_replayed=prefixes,
            reused_full_prediction_arrays_identical=14,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/partner_conditioned_recovery.json")
    parser.add_argument("--output", default="outputs/2026-10-11/partner_conditioned_recovery")
    parser.add_argument(
        "--manifest",
        default="/root/autodl-tmp/migration_20261010/partner_conditioned_manifest.json",
    )
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    (validate if options.validate else run)(options)
