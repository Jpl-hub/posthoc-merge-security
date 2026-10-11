"""Known-partner complement directions and a uniform public replacement gate.

Check the cheapest erasure before evaluating a rank-one defense. Representative
edits validate algebra only; no utility-preserving protected release is claimed.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge
from eight_task_benchmark import digest
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import read
from query_partner_cost import runtime
from subspace_probe import collect_moments, fit_basis, make_loader, save_json, site_parameters
from torchvision.datasets import MNIST, SVHN


def inputs(cfg):
    root = Path("checkpoints/ViT-B-32")
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    pins = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    names = [
        "zeroshot.pt",
        "MNIST/finetuned.pt",
        "SVHN/finetuned.pt",
        "head_MNIST.pt",
        "head_SVHN.pt",
    ]
    for name in names:
        assert digest(root / name) == pins[name]
    anchor, teacher, partner = [load(str(root / p)).cpu().eval() for p in names[:3]]
    heads = {t: load(str(root / f"head_{t}.pt")) for t in ["MNIST", "SVHN"]}
    data = {
        "MNIST": MNIST("data", train=True, transform=teacher.val_preprocess),
        "SVHN": SVHN("data", split="train", transform=teacher.val_preprocess),
    }
    ids = read(Path(cfg["indices_parent"]) / "training_indices.json")
    assert all(len(ids[t]["calibration"]) == cfg["calibration_samples"] for t in data)
    moments = {
        task: collect_moments(
            teacher, make_loader(dataset, ids[task]["calibration"], 32), cfg["sites"]
        )
        for task, dataset in data.items()
    }
    tests = {
        "MNIST": MNIST("data", train=False, transform=teacher.val_preprocess),
        "SVHN": SVHN("data", split="test", transform=teacher.val_preprocess),
    }
    return anchor, teacher, partner, heads, tests, moments, {p: pins[p] for p in names}, ids


def public_reset(model, anchor, site):
    result = copy.deepcopy(model).cpu().eval()
    w, b = site_parameters(result, site)
    a, c = site_parameters(anchor, site)
    w.copy_(a)
    b.copy_(c)
    return result


@torch.no_grad()
def direction(teacher, partner, anchor, own, other, site, cfg):
    primary, stats = fit_basis(own, cfg["target_energy"])
    secondary, partner_stats = fit_basis(other, cfg["partner_energy"])
    span = primary["u"].double().cuda()
    block = teacher.model.visual.transformer.resblocks[int(site.split(":")[1])]
    norm = block.ln_1 if site.startswith("V:") else block.ln_2
    null = norm.weight.double().cuda().reciprocal()
    residual = null - span @ (span.T @ null)
    if float(residual.norm()) > 1e-10:
        span = torch.cat([span, (residual / residual.norm())[:, None]], dim=1)
    complement = torch.linalg.qr(span, mode="complete").Q[:, span.shape[1] :]
    projected = complement.T @ secondary["u"].double().cuda()
    gram = projected @ projected.T
    values, vectors = torch.linalg.eigh(gram)
    local = vectors[:, -1]
    v = complement @ local
    v *= 1 if v[v.abs().argmax()] >= 0 else -1
    ws, _ = site_parameters(partner, site)
    w0, _ = site_parameters(anchor, site)
    u = torch.linalg.svd((ws.double() - w0.double()).cuda(), full_matrices=False).U[:, 0]
    u *= 1 if u[u.abs().argmax()] >= 0 else -1
    w, b = site_parameters(teacher, site)
    alpha = cfg["representative_parameter_ratio"] * float(w.double().norm())
    noise = alpha * u[:, None] @ v[None, :]
    mean = own["mean"].double().cuda()
    release = copy.deepcopy(teacher).cpu().eval()
    wr, br = site_parameters(release, site)
    wr.add_(noise.float().cpu())
    br.sub_((noise @ mean).float().cpu())
    original_reset = public_reset(teacher, anchor, site)
    release_reset = public_reset(release, anchor, site)
    assert all(
        torch.equal(val, release_reset.state_dict()[k])
        for k, val in original_reset.state_dict().items()
    )
    p = primary["u"].double().cuda()
    d = w.double().cuda() - w0.double().cuda()
    # Exact-basis cleanup is a privileged algebra check, not the measured attack.
    qd = d - (d @ p) @ p.T
    released_difference = d + noise
    removed = released_difference - (released_difference @ p) @ p.T
    cleanup_error = float(
        (w.double().cuda() + noise - removed - (w.double().cuda() - qd)).abs().max()
    )
    bias_error = float(
        (b.double().cuda() - noise @ mean + removed @ mean - (b.double().cuda() + qd @ mean))
        .abs()
        .max()
    )
    assert cleanup_error < 1e-9 and bias_error < 1e-9
    cm = own["covariance"].double().cuda()
    shift = other["mean"].double().cuda() - mean
    cs = other["covariance"].double().cuda() + shift[:, None] @ shift[None, :]
    actual = wr.double().cuda() - w.double().cuda()
    row = dict(
        site=site,
        target_rank=primary["rank"],
        partner_rank=secondary["rank"],
        complement_without_affine_null=complement.shape[1],
        top_overlap_eigenvalue=float(values[-1]),
        eigen_equation_error=float((gram @ local - values[-1] * local).norm()),
        target_span_leakage=float((p.T @ v).norm()),
        affine_null_leakage=float(abs(v @ null) / null.norm()),
        target_centered_direction_variance=float(v @ cm @ v),
        partner_direction_second_moment_about_target_mean=float(v @ cs @ v),
        representative_parameter_ratio=float(actual.norm() / w.double().cuda().norm()),
        representative_target_linear_mse=alpha**2 * float(v @ cm @ v),
        representative_partner_linear_mse=alpha**2 * float(v @ cs @ v),
        exact_basis_cleanup_weight_error=cleanup_error,
        exact_basis_cleanup_bias_error=bias_error,
        full_public_reset_bitwise_identical=True,
        protected_utility_or_security_evaluated=False,
    )
    assert row["target_span_leakage"] < 1e-10 and row["affine_null_leakage"] < 1e-10
    evidence = dict(
        primary=primary,
        secondary=secondary,
        own=own,
        other=other,
        incoming=v.cpu(),
        outgoing=u.cpu(),
        alpha=alpha,
        representative_weight=wr.clone(),
        representative_bias=br.clone(),
        target_stats=stats,
        partner_stats=partner_stats,
    )
    return row, evidence


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    save_json(out / "protocol.json", cfg)
    anchor, teacher, partner, heads, tests, moments, pins, ids = inputs(cfg)
    save_json(out / "training_indices.json", ids)
    diagnostics, evidence = [], {}
    for site in cfg["sites"]:
        row, frame = direction(
            teacher, partner, anchor, moments["MNIST"][site], moments["SVHN"][site], site, cfg
        )
        diagnostics.append(row)
        evidence[site] = frame
    save_json(out / "geometry.json", diagnostics)
    torch.save(evidence, artifacts / "directions_and_representative_edits.pt")
    loaders = {
        task: cache(SimpleNamespace(test_loader=make_loader(data))) for task, data in tests.items()
    }
    raw, rows = {}, []
    for preset in ["original"] + cfg["sites"]:
        model = teacher if preset == "original" else public_reset(teacher, anchor, preset)
        combined = merge(anchor, model, partner, cfg["coefficient"])
        for group, target, task_list in [
            ("standalone", model, ["MNIST"]),
            ("fusion", combined, ["MNIST", "SVHN"]),
        ]:
            for task in task_list:
                predictions, labels = [], []
                metrics = evaluate(
                    target,
                    heads[task],
                    loaders[task],
                    "cuda",
                    prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
                )
                key = f"{preset}/{group}/{task}"
                raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(labels))
                reference = raw[f"original/{group}/{task}"]
                rows.append(
                    dict(
                        key=key,
                        preset=preset,
                        group=group,
                        dataset=task,
                        original_change=paired_counts(
                            reference["prediction"].eq(reference["labels"]),
                            raw[key]["prediction"].eq(raw[key]["labels"]),
                        ),
                        **metrics,
                    )
                )
                save_json(out / "results.json", rows)
                torch.save(raw, artifacts / "predictions.pt")
                print("PARTNER_NULLSPACE_GATE", key, metrics["correct"], metrics["n"], flush=True)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            full_evaluations=len(rows),
            representative_edits_only=True,
            utility_preserving_release_claim=False,
            new_defense=False,
            construction_uses_partner_inputs_and_checkpoint=True,
            checkpoint_sha256=pins,
            source_manifest=read(args.manifest)["source_manifest"],
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            config_sha256=digest(args.config),
            original_author_aga_run=False,
            amplitude_scan=False,
        ),
    )


@torch.no_grad()
def validate(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for p, pin in meta["source_manifest"].items():
        assert digest(p) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    anchor, teacher, partner, heads, tests, moments, pins, ids = inputs(cfg)
    assert pins == meta["checkpoint_sha256"] and ids == read(out / "training_indices.json")
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    saved = torch.load(
        out / "artifacts/directions_and_representative_edits.pt",
        map_location="cpu",
        weights_only=True,
    )
    errors = []
    for site in cfg["sites"]:
        row, frame = direction(
            teacher, partner, anchor, moments["MNIST"][site], moments["SVHN"][site], site, cfg
        )
        error = max(
            float((frame[k] - saved[site][k]).abs().max())
            for k in ["incoming", "outgoing", "representative_weight", "representative_bias"]
        )
        assert error < 5e-5
        assert row["full_public_reset_bitwise_identical"]
        errors.append(error)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 9
    for row in rows:
        task, v = row["dataset"], raw[row["key"]]
        labels = torch.as_tensor(tests[task].targets if task == "MNIST" else tests[task].labels)
        assert torch.equal(labels, v["labels"])
        assert row["n"] == len(labels) and row["correct"] == int(v["prediction"].eq(labels).sum())
        assert row["top1"] == row["correct"] / len(labels)
        reference = raw[f"original/{row['group']}/{task}"]
        assert row["original_change"] == paired_counts(
            reference["prediction"].eq(labels), v["prediction"].eq(labels)
        )
        model = (
            teacher if row["preset"] == "original" else public_reset(teacher, anchor, row["preset"])
        )
        if row["group"] == "fusion":
            model = merge(anchor, model, partner, cfg["coefficient"])
        sink = []
        evaluate(
            model,
            heads[task],
            make_loader(tests[task], list(range(128))),
            "cuda",
            prediction_sink=lambda p, _y: sink.append(p),
        )
        assert torch.equal(torch.cat(sink), v["prediction"][:128])
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            fresh_moment_direction_and_edit_replay_max_errors=errors,
            counts_labels_pairs_and_prefixes_checked=9,
            exact_support_reset_identity_checks=2,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            protected_model_accuracy_not_measured=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/partner_nullspace_gate.json")
    parser.add_argument("--output", default="outputs/2026-10-11/partner_nullspace_gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
