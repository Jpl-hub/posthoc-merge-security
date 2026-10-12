"""Independent geometry and real-path audit of a frozen complement release.

No fitting, new candidate, official-test access or coefficient changes. Actual
fusion enters only after release construction and full evaluation have ended.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from critical_subspace import parameters
from eight_task_benchmark import digest, merge_all
from functional_contrast import features
from local_erasure import rescale
from path_subspace import context, finite_differences, read
from query_partner_cost import runtime
from scipy.linalg import eigh
from signed_complement import aliases, margin_columns, margin_spec, response
from subspace_probe import collect_moments, make_loader, save_json


def transfer_statistics(reference, protected, proxy_reference, proxy_protected, labels, weight):
    """Separate feature transport, rival changes and baseline margin on fixed rows."""
    a, b, pa, pb = [
        x.double().numpy() for x in (reference, protected, proxy_reference, proxy_protected)
    ]
    y, w = labels.numpy(), weight.double().numpy()
    n, row = len(y), np.arange(len(y))

    def margin_spec_numpy(f):
        scores = f @ w.T
        competitors = scores.copy()
        competitors[row, y] = -np.inf
        rival = competitors.argmax(1)
        margin = scores[row, y] - scores[row, rival]
        return rival, margin, scores.argmax(1) == y

    rival, margin, correct = margin_spec_numpy(a)
    prival, pmargin, pcorrect = margin_spec_numpy(pa)
    after_correct = margin_spec_numpy(b)[2]
    pafter_correct = margin_spec_numpy(pb)[2]
    shared = correct & pcorrect
    assert shared.any()
    h, ph = w[y] - w[rival], w[y] - w[prival]
    delta, pdelta = b - a, pb - pa
    change = np.einsum("nf,nf->n", h, delta)
    pchange = np.einsum("nf,nf->n", ph, pdelta)
    # Exact decomposition uses unnormalized logits; no unequal denominators.
    feature_term = np.einsum("nf,nf->n", h, delta - pdelta)
    rival_term = np.einsum("nf,nf->n", h - ph, pdelta)
    decomposition_error = float(np.max(np.abs(change - pchange - feature_term - rival_term)))
    assert decomposition_error < 1e-10

    def cosine(x, z):
        return float(np.sum(x * z) / max(np.linalg.norm(x) * np.linalg.norm(z), 1e-30))

    def crossing(m, dm, valid, final):
        predicted = valid & (m + dm < 0)
        lost = valid & ~final
        return dict(
            reference_correct=int(valid.sum()),
            actual_lost=int(lost.sum()),
            fixed_original_rival_crossings=int(predicted.sum()),
            actual_losses_not_crossing_original_rival=int((lost & ~predicted).sum()),
            sufficient_crossings_without_actual_loss=int((predicted & ~lost).sum()),
        )

    correlation = float(np.corrcoef(change[shared], pchange[shared])[0, 1])
    assert np.isfinite(correlation)
    return dict(
        n=n,
        shared_reference_correct=int(shared.sum()),
        original_rival_agreement_fraction=float(np.mean(rival[shared] == prival[shared])),
        feature_delta_cosine=cosine(delta[shared], pdelta[shared]),
        feature_delta_norm_ratio=float(
            np.linalg.norm(delta[shared]) / np.linalg.norm(pdelta[shared])
        ),
        signed_margin_change_correlation=correlation,
        actual_to_proxy_original_margin_median_ratio=float(
            np.median(margin[shared] / pmargin[shared])
        ),
        actual_mean_logit_margin_change=float(change[shared].mean()),
        proxy_mean_logit_margin_change=float(pchange[shared].mean()),
        difference_feature_transport_term=float(feature_term[shared].mean()),
        difference_original_rival_term=float(rival_term[shared].mean()),
        decomposition_max_error=decomposition_error,
        actual_crossings=crossing(margin, change, correct, after_correct),
        proxy_crossings=crossing(pmargin, pchange, pcorrect, pafter_correct),
    )


@torch.no_grad()
def transfer_audit(args):
    """Replay confirmation only; leave the release and all existing results intact."""
    runtime()
    out = Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and read(out / "validation.json")["status"] == "passed"
    target = out / "proxy_transfer.json"
    assert not target.exists() if not args.validate_transfer else target.exists()
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    saved = torch.load(out / "artifacts/construction.pt", map_location="cpu", weights_only=True)
    raw = torch.load(out / "artifacts/responses.pt", map_location="cpu", weights_only=True)
    root, load, _, models = context(cfg, all_tasks=True)
    head = load(str(root / f"head_{cfg['task']}.pt")).cpu().eval()
    release = copy.deepcopy(models[cfg["task"]])
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    from dataset.registry import get_dataset

    data = get_dataset(
        cfg["task"], models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2
    )
    indices = read(out / "training_indices.json")[cfg["task"]]
    assert not set(indices["confirmation"]).intersection(indices["calibration"])
    used = indices["confirmation"][:32] if args.validate_transfer else indices["confirmation"]
    x = torch.stack([data.train_dataset[i][0] for i in used])
    labels = saved["labels"]["confirmation"]
    observed = {}
    for name, inputs in [
        ("reference", models),
        ("protected", dict(models, **{cfg["task"]: release})),
    ]:
        merged = merge_all(inputs, root, cfg["coefficient"])
        merged.cuda()
        observed[name] = features(merged, x)
        del merged
    artifact = out / "artifacts/proxy_transfer.pt"
    if args.validate_transfer:
        report = read(target)
        assert digest(artifact) == report["raw_sha256"]
        observed_saved = torch.load(artifact, map_location="cpu", weights_only=True)
        assert all(torch.equal(observed[k], observed_saved[k][:32]) for k in observed)
        observed = observed_saved
    actual_response = response(observed["reference"], observed["protected"], labels, head)
    assert (
        actual_response == read(out / "mechanism_audit.json")["actual_fusion_confirmation_response"]
    )
    results = {}
    for beta in cfg["betas"]:
        proxy = raw[f"beta_{beta}/confirmation"]
        results[str(beta)] = transfer_statistics(
            observed["reference"],
            observed["protected"],
            proxy["before"],
            proxy["after"],
            labels,
            head.weight,
        )
    if args.validate_transfer:
        assert results == report["results"]
        save_json(
            out / "proxy_transfer_validation.json",
            dict(
                status="passed",
                numpy_decomposition_recomputed=True,
                fresh_fusion_feature_prefixes=2,
                prefix_images=32,
                results_sha256=digest(target),
                raw_sha256=digest(artifact),
            ),
        )
    else:
        assert not artifact.exists()
        torch.save(observed, artifact)
        save_json(
            target,
            dict(
                source_sha256=digest(__file__),
                task=cfg["task"],
                results=results,
                raw_sha256=digest(artifact),
                parent_metadata_sha256=digest(out / "metadata.json"),
                confirmation_indices_sha256=digest(out / "training_indices.json"),
                release_unchanged=True,
                new_candidates=0,
                test_images_used=False,
                partner_access_only_after_release=True,
                classification_response=actual_response,
            ),
        )


@torch.no_grad()
def run(args):
    runtime()
    out = Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and read(out / "validation.json")["status"] == "passed"
    assert not (out / "mechanism_audit.json").exists()
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    saved = torch.load(out / "artifacts/construction.pt", map_location="cpu", weights_only=True)
    raw = torch.load(out / "artifacts/responses.pt", map_location="cpu", weights_only=True)
    geometry = []
    for item in saved["records"]:
        site, v = item["site"], item["incoming"].numpy()
        ev = saved["geometry"][site]
        a, b = ev["a"].numpy(), ev["b"].numpy()
        rho = cfg["covariance_ridge_fraction"] * np.trace(a) / len(a) + 1e-12
        values, vectors = eigh(b, a + rho * np.eye(len(a)))
        incoming = ev["complement"].numpy() @ vectors[:, -1]
        incoming /= np.linalg.norm(incoming)
        overlap = abs(float(incoming @ v))
        assert overlap > 1 - 1e-8, site
        powers = {}
        for name, moments in saved["moments"].items():
            alias = site.replace("v:", "V:").replace("up:", "FC:")
            m = moments[alias]
            centered = float(v @ m["covariance"].numpy() @ v)
            shift = float(v @ (m["mean"].numpy() - item["mean"].numpy())) ** 2
            powers[name] = dict(
                centered_power=centered,
                mean_shift_power=shift,
                mean_shift_fraction=shift / (centered + shift),
            )
        geometry.append(
            dict(
                site=site,
                independent_direction_overlap=overlap,
                independent_eigenvalue=float(values[-1]),
                powers=powers,
            )
        )
    root, load, fusion, models = context(cfg, all_tasks=True)
    expert, anchor = models[cfg["task"]], models["anchor"]
    head = load(str(root / f"head_{cfg['task']}.pt")).cpu().eval()
    release = copy.deepcopy(expert)
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], anchor.val_preprocess, "data", batch_size=128, num_workers=2)
    indices = read(out / "training_indices.json")[cfg["task"]]
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    calibration32 = torch.stack([data.train_dataset[i][0] for i in indices["calibration"][:32]])
    derivative_checks = []
    for name in saved["derivatives"]:
        factor = 1.0 if name == "expert" else float(name.split("_")[1])
        model = expert if name == "expert" else rescale(expert, anchor, factor)
        derivative = finite_differences(
            model, calibration32, saved["records"], cfg["finite_difference_step"], factor
        )
        expected = saved["derivatives"][name][:32]
        error = float((derivative - expected).abs().max())
        assert error == 0.0, (name, error)
        spec = {k: v[:32] for k, v in saved["specs"][name].items()}
        columns = margin_columns(derivative, head, saved["labels"]["calibration"][:32], spec)
        assert float((columns - saved["columns"][name][:32]).abs().max()) < 1e-8
        derivative_checks.append(
            dict(point=name, replayed_columns=derivative.shape[-1], max_absolute_error=error)
        )
        if name != "expert":
            del model
    retained_checks = []
    for item in saved["records"]:
        site = item["site"]
        _, we, be = parameters(expert.state_dict(), site)
        _, wp, bp = parameters(release.state_dict(), site)
        dw, db = wp.double() - we.double(), bp.double() - be.double()
        p = saved["geometry"][site]["retained"]
        retained_checks.append(
            dict(
                site=site,
                actual_weight_relative_change=float(dw.norm() / we.double().norm()),
                retained_space_relative_leakage=float((dw @ p).norm() / dw.norm().clamp_min(1e-30)),
                mean_compensation_max_error=float((db + dw @ item["mean"]).abs().max()),
            )
        )
    reference = merge_all(models, root, cfg["coefficient"])
    protected = merge_all(dict(models, **{cfg["task"]: release}), root, cfg["coefficient"])
    x = torch.stack([data.train_dataset[i][0] for i in indices["confirmation"]])
    labels = saved["labels"]["confirmation"]
    reference.cuda()
    before = features(reference, x)
    reference.cpu()
    protected.cuda()
    after = features(protected, x)
    protected.cpu()
    actual_response = response(before, after, labels, head)
    actual_moments = collect_moments(
        reference, make_loader(data.train_dataset, indices["confirmation"]), aliases(cfg["sites"])
    )
    incoming = []
    for item in saved["records"]:
        alias = item["site"].replace("v:", "V:").replace("up:", "FC:")
        v, m = item["incoming"], actual_moments[alias]
        shift = m["mean"] - item["mean"]
        centered, centroid = float(v @ m["covariance"] @ v), float(v @ shift) ** 2
        incoming.append(
            dict(
                site=item["site"],
                centered_power=centered,
                mean_shift_power=centroid,
                total_power=centered + centroid,
                native_calibration_power=geometry[len(incoming)]["powers"]["expert"][
                    "centered_power"
                ],
            )
        )
    linear_checks = []
    for name, derivative in saved["derivatives"].items():
        actual = raw[f"{name}/calibration"]["after"] - raw[f"{name}/calibration"]["before"]
        predicted = derivative.double() @ saved["coefficients"]
        baseline = saved["baseline"][name]
        assert torch.equal(baseline, raw[f"{name}/calibration"]["before"])
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual.double().flatten(), predicted.flatten(), dim=0
            )
        )
        linear_checks.append(
            dict(
                point=name,
                feature_delta_prediction_cosine=cosine,
                feature_delta_prediction_relative_error=float(
                    (actual.double() - predicted).norm() / actual.double().norm().clamp_min(1e-30)
                ),
            )
        )
    save_json(
        out / "mechanism_audit.json",
        dict(
            source_sha256=digest(__file__),
            geometry=geometry,
            derivative_checks=derivative_checks,
            fp32_retained_checks=retained_checks,
            linear_checks=linear_checks,
            actual_fusion_confirmation_response=actual_response,
            actual_fusion_inputs=incoming,
            confirmation_baseline_correct={
                key: int(
                    margin_spec(value["before"].double() @ head.weight.double().T, value["labels"])[
                        "correct"
                    ].sum()
                )
                for key, value in raw.items()
                if key.endswith("confirmation")
            },
            coefficients_unchanged=True,
            new_candidates=0,
            response_fits=0,
            actual_fusion_used_only_after_freeze=True,
            test_images_used=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/2026-10-12/signed_complement")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--proxy-transfer", action="store_true")
    mode.add_argument("--validate-transfer", action="store_true")
    args = parser.parse_args()
    transfer_audit(args) if args.proxy_transfer or args.validate_transfer else run(args)
