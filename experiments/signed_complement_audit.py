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
    run(parser.parse_args())
