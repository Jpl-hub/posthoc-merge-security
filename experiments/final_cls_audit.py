"""Independent final-token covariance and measured response audit of a fixed release.

No new candidate, fitting or full accuracy measurements. Training inputs only.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from eight_task_benchmark import digest, merge_all
from functional_contrast import features
from path_subspace import context, read
from query_partner_cost import runtime
from scipy.linalg import eigh
from subspace_probe import collect_moments, fit_basis, make_loader, save_json, site_parameters
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def capture(model, images):
    incoming, outgoing = [], []
    module = model.model.visual.transformer.resblocks[11].mlp.c_fc
    before = module.register_forward_pre_hook(
        lambda _m, args: incoming.append(args[0][0].detach().cpu())
    )
    after = module.register_forward_hook(
        lambda _m, _args, value: outgoing.append(value[0].detach().cpu())
    )
    model.cuda().eval()
    try:
        value = features(model, images)
    finally:
        before.remove()
        after.remove()
        model.cpu()
    return torch.cat(incoming), torch.cat(outgoing), value


@torch.no_grad()
def run(args):
    runtime()
    out, cfg = Path(args.output), read(args.config)
    assert cfg["sites"] == ["FC:11"] and cfg["token_scope"] == "final_cls"
    assert read(out / "validation.json")["status"] == "passed"
    meta = read(out / "metadata.json")
    for path, sha in meta["source_manifest"].items():
        assert digest(path) == sha, path
    root, load, fusion, models = context(cfg, all_tasks=True)
    original = models["MNIST"]
    protected = copy.deepcopy(original).cpu().eval()
    edits = torch.load(out / "artifacts/decoy_edits.pt", weights_only=True)
    assert digest(out / "artifacts/decoy_edits.pt") == meta["raw_sha256"]["decoy_edits.pt"]
    protected.load_state_dict(dict(protected.state_dict(), **edits))
    train = MNIST("data", train=True, transform=original.val_preprocess, download=False)
    indices = read(out / "training_indices.json")
    calibration = torch.stack([train[i][0] for i in indices["calibration"]])
    confirmation = torch.stack([train[i][0] for i in indices["confirmation"]])
    x, _, _ = capture(original, calibration)
    xn = x.double().numpy()
    mean = xn.mean(0)
    centered = xn - mean
    covariance = centered.T @ centered / len(xn)
    cert = torch.load(out / "artifacts/certificate.pt", weights_only=True)["FC:11"]
    assert np.allclose(mean, cert["mean"].numpy(), atol=1e-10, rtol=0)
    covariance_error = float(
        np.linalg.norm(covariance - cert["covariance"].numpy()) / np.linalg.norm(covariance)
    )
    assert covariance_error < 1e-10
    values, vectors = eigh(covariance)
    spectrum = np.maximum(values[::-1], 0)
    rank = int(np.searchsorted(spectrum.cumsum() / spectrum.sum(), cfg["task_energy"])) + 1
    p = vectors[:, -rank:]
    null = original.model.visual.transformer.resblocks[11].ln_2.weight.double().reciprocal().numpy()
    null /= np.linalg.norm(null)
    residual = null - p @ (p.T @ null)
    if np.linalg.norm(residual) > 1e-9:
        p = np.column_stack([p, residual / np.linalg.norm(residual)])
    saved_p = cert["p"].numpy()
    projector_error = float(np.linalg.norm(p @ p.T - saved_p @ saved_p.T))
    assert projector_error < 1e-6
    # Independent dependency intervention: replace every final patch MLP output
    # by a large constant; preserve only CLS. Earlier patch outputs are untouched.
    original.cuda().eval()
    baseline = features(original, confirmation[16:32])

    def replace_patches(_m, _args, value):
        result = value.clone()
        result[1:] = 1000000
        return result

    handle = original.model.visual.transformer.resblocks[11].mlp.register_forward_hook(
        replace_patches
    )
    try:
        intervened = features(original, confirmation[16:32])
    finally:
        handle.remove()
        original.cpu()
    assert torch.equal(baseline, intervened)
    delta = (
        site_parameters(protected, "FC:11")[0].double()
        - site_parameters(original, "FC:11")[0].double()
    )
    delta_bias = (
        site_parameters(protected, "FC:11")[1].double()
        - site_parameters(original, "FC:11")[1].double()
    )
    responses = []
    merged = merge_all(models, root, cfg["coefficient"])
    protected_merge = merge_all(dict(models, MNIST=protected), root, cfg["coefficient"])
    partner = SVHN("data", split="train", transform=original.val_preprocess, download=False)
    partner_images = torch.stack([partner[i][0] for i in range(256)])
    for name, reference, changed, images, factor in [
        ("native/MNIST", original, protected, confirmation, 1.0),
        ("fusion/MNIST", merged, protected_merge, confirmation, cfg["coefficient"]),
        ("fusion/SVHN", merged, protected_merge, partner_images, cfg["coefficient"]),
    ]:
        incoming, pre, before = capture(reference, images)
        changed_incoming, post, after = capture(changed, images)
        assert torch.equal(incoming, changed_incoming)
        predicted = factor * (incoming.double() @ delta.T + delta_bias)
        actual = post.double() - pre.double()
        response_error = float((predicted - actual).norm() / actual.norm())
        assert response_error < 1e-4
        responses.append(
            dict(
                scope=name,
                training_images=len(images),
                incoming_cls_bitwise_equal=True,
                preactivation_difference_relative_rms=float(actual.norm() / pre.double().norm()),
                predicted_preactivation_relative_error=response_error,
                normalized_feature_difference_rms=float(
                    (after - before).double().norm() / len(images) ** 0.5
                ),
            )
        )
    save_json(
        out / "causal_audit.json",
        dict(
            status="passed",
            audit_source_sha256=digest(__file__),
            independent_covariance_relative_error=covariance_error,
            independent_pca_rank=rank,
            independent_projector_frobenius_error=projector_error,
            large_final_patch_intervention_bitwise_equal=True,
            confirmation_images_disjoint_from_initial_causal_check=True,
            responses=responses,
            diagnostic_partner_training_images=256,
            partners_used_only_after_release=True,
            no_candidate_changed=True,
            no_new_accuracy_measurements=True,
        ),
    )


@torch.no_grad()
def compare_tokens(args):
    """Compare token definitions at the same layer and on identical images."""
    runtime()
    out, cfg = Path(args.output), read(args.config)
    assert read(out / "validation.json")["status"] == "passed"
    root, load, _fusion, _models = context(cfg)
    model = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
    data = MNIST("data", train=True, transform=model.val_preprocess, download=False)
    indices = read(out / "training_indices.json")["calibration"]
    moment = collect_moments(model, make_loader(data, indices, 32), ["FC:11"])["FC:11"]
    _basis, stats = fit_basis(moment, cfg["task_energy"])
    construction = read(out / "construction.json")[0]
    save_json(
        out / "token_comparison.json",
        dict(
            status="measured",
            layer=11,
            identical_training_images=len(indices),
            retained_energy=cfg["task_energy"],
            all_tokens=stats,
            cls_pca_rank=construction["rank"],
            cls_tokens=construction["tokens"],
            cls_complement_excluding_ln_null=construction["actual_complement_rank"],
            audit_source_sha256=digest(__file__),
            no_candidate_changed=True,
            scope="Geometry comparison; not a rank search, accuracy measurement or utility guarantee",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-comparison", action="store_true")
    arguments = parser.parse_args()
    (compare_tokens if arguments.token_comparison else run)(arguments)
