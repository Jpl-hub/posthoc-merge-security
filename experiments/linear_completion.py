"""One target-only QuickGELU linear completion, with a utility screen.

Keep original nonlinear units chosen by exact last-block margin counterfactuals.
Replace the remaining even responses by calibration means; preserve their
linear operator with antipodal neuron pairs. No optimizer or response fitting.
This lossy standard-width construction is neither a novelty nor security claim.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from clip_benchmark import evaluate
from eight_task_benchmark import digest
from functional_contrast import features
from operator_diagnostics import paired_counts
from path_subspace import context, read
from query_partner_cost import runtime, truth
from signed_complement import margin_spec
from subspace_probe import make_loader, save_json


def even(z):
    return z * (torch.sigmoid(1.702 * z) - 0.5)


@torch.no_grad()
def capture(model, images, layer):
    block = model.model.visual.transformer.resblocks[layer]
    assert layer == len(model.model.visual.transformer.resblocks) - 1
    z, residual = [], []
    handles = [
        block.mlp.c_fc.register_forward_hook(lambda _m, _a, y: z.append(y[0].cpu())),
        block.register_forward_hook(lambda _m, _a, y: residual.append(y[0].cpu())),
    ]
    try:
        model.cuda().eval()
        for batch in images.split(16):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return torch.cat(z), torch.cat(residual)


@torch.no_grad()
def select(model, head, images, labels, cfg):
    z, residual = capture(model, images, cfg["layer"])
    visual = model.model.visual
    ln = copy.deepcopy(visual.ln_post).double().cuda()
    projection = visual.proj.double().cuda()
    weight = head.weight.double().cuda()
    assert head.normalize and torch.count_nonzero(head.bias) == 0

    def scores(value):
        return F.normalize(ln(value) @ projection, dim=-1) @ weight.T

    residual = residual.double().cuda()
    e = even(z.double().cuda())
    mean = e.mean(0)
    baseline = scores(residual).cpu()
    spec = margin_spec(baseline, labels)
    native = model.model.visual.transformer.resblocks[cfg["layer"]].mlp
    outgoing = native.c_proj.weight.double().cuda()
    columns = []
    for start in range(0, e.shape[1], cfg["counterfactual_batch"]):
        stop = min(start + cfg["counterfactual_batch"], e.shape[1])
        # Each column deletes one centered even response. The actual tail,
        # including LayerNorm and feature normalization, is evaluated exactly.
        change = (e[:, start:stop] - mean[start:stop]).T[:, :, None]
        change = change * outgoing[:, start:stop].T[:, None, :]
        value = scores(residual[None, :, :] - change).cpu()
        row = torch.arange(len(labels))
        margins = value[:, row, labels] - value[:, row, spec["rival"]]
        columns.append(((margins - spec["margin"]) / spec["scale"]).T)
    changes = torch.cat(columns, dim=1)
    importance = changes[spec["correct"]].square().mean(0)
    order = torch.argsort(importance, descending=True, stable=True)
    keep = order[: cfg["retained_nonlinear_units"]].sort().values
    drop = order[cfg["retained_nonlinear_units"] :].sort().values
    return dict(
        z=z,
        residual=residual.cpu(),
        even_mean=mean.cpu(),
        baseline=baseline,
        labels=labels,
        spec=spec,
        margin_changes=changes,
        importance=importance,
        keep=keep,
        drop=drop,
    )


@torch.no_grad()
def compile_mlp(original, geometry, cfg):
    release = copy.deepcopy(original).cpu().eval()
    source = original.model.visual.transformer.resblocks[cfg["layer"]].mlp
    target = release.model.visual.transformer.resblocks[cfg["layer"]].mlp
    keep, drop = geometry["keep"], geometry["drop"]
    a, b = source.c_fc.weight.double().cuda(), source.c_fc.bias.double().cuda()
    v = source.c_proj.weight.double().cuda()
    mu = geometry["even_mean"].cuda()
    linear = 0.5 * v[:, drop] @ a[drop]
    left, singular, right = torch.linalg.svd(linear, full_matrices=False)
    signs = right[torch.arange(len(right), device="cuda"), right.abs().argmax(1)].sign()
    left, right = left * signs[None, :], right * signs[:, None]
    incoming = singular.sqrt()[:, None] * right
    outgoing = left * singular.sqrt()[None, :]
    active, rank = len(keep), len(singular)
    assert active + 2 * rank == source.c_fc.out_features
    target.c_fc.weight[:active].copy_(source.c_fc.weight[keep])
    target.c_fc.bias[:active].copy_(source.c_fc.bias[keep])
    target.c_proj.weight[:, :active].copy_(source.c_proj.weight[:, keep])
    target.c_fc.weight[active : active + rank].copy_(incoming.float().cpu())
    target.c_fc.weight[active + rank :].copy_(-incoming.float().cpu())
    target.c_fc.bias[active:].zero_()
    target.c_proj.weight[:, active : active + rank].copy_(outgoing.float().cpu())
    target.c_proj.weight[:, active + rank :].copy_(-outgoing.float().cpu())
    bias = source.c_proj.bias.double().cuda() + v[:, drop] @ (0.5 * b[drop] + mu[drop])
    target.c_proj.bias.copy_(bias.float().cpu())
    matrix_error = float((outgoing @ incoming - linear).norm() / linear.norm())
    assert matrix_error < 1e-10
    return release, dict(
        linear=linear.cpu(), incoming=incoming.cpu(), outgoing=outgoing.cpu(), bias=bias.cpu()
    )


@torch.no_grad()
def reference_features(original, images, geometry, cfg):
    model = copy.deepcopy(original).cuda().eval()
    drop, mean = geometry["drop"].cuda(), geometry["even_mean"].cuda()

    def hook(_module, args, output):
        value = output.clone()
        value[..., drop] = 0.5 * args[0][..., drop] + mean[drop].to(value.dtype)
        return value

    handle = model.model.visual.transformer.resblocks[cfg["layer"]].mlp.gelu.register_forward_hook(
        hook
    )
    try:
        return features(model, images)
    finally:
        handle.remove()
        model.cpu()


def inputs(cfg):
    root, load, fusion, _holders = context(cfg)
    original = load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval()
    head = load(str(root / f"head_{cfg['task']}.pt")).cpu().eval()
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    saved = read(Path(cfg["profile_parent"]) / "training_indices.json")[cfg["task"]]
    indices = dict(
        calibration=saved["selection"][: cfg["calibration_samples"]],
        confirmation=saved["confirmation"][: cfg["confirmation_samples"]],
    )
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    return root, fusion, original, head, data, indices


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    started = time.perf_counter()
    root, fusion, original, head, data, indices = inputs(cfg)
    pairs = [data.train_dataset[i] for i in indices["calibration"]]
    images, labels = torch.stack([x for x, _ in pairs]), torch.tensor([int(y) for _, y in pairs])
    geometry = select(original, head, images, labels, cfg)
    release, algebra = compile_mlp(original, geometry, cfg)
    edits = {
        k: v.clone()
        for k, v in release.state_dict().items()
        if not torch.equal(v, original.state_dict()[k])
    }
    prefix = f"model.visual.transformer.resblocks.{cfg['layer']}.mlp."
    assert set(edits) == {
        prefix + s for s in ["c_fc.weight", "c_fc.bias", "c_proj.weight", "c_proj.bias"]
    }
    assert all(torch.isfinite(value).all() for value in edits.values())
    torch.save(edits, artifacts / "release_edits.pt")
    torch.save(dict(geometry=geometry, algebra=algebra), artifacts / "construction.pt")
    save_json(out / "protocol.json", cfg)
    save_json(out / "training_indices.json", indices)
    save_json(
        out / "release_freeze.json",
        dict(
            release_sha256=digest(artifacts / "release_edits.pt"),
            gradient_steps=0,
            response_fits=0,
            future_partners_used=False,
            calibration_labels_used=True,
            retained=len(geometry["keep"]),
            dropped=len(geometry["drop"]),
        ),
    )
    # No test or confirmation information is read until the compiled release is frozen.
    confirmation = torch.stack([data.train_dataset[i][0] for i in indices["confirmation"]])
    equivalence = {}
    for name, x in [("calibration", images), ("confirmation", confirmation)]:
        observed = features(release.cuda().eval(), x)
        release.cpu()
        expected = reference_features(original, x, geometry, cfg)
        error = float((observed - expected).abs().max())
        assert error < cfg["compiled_feature_max_absolute_error"]
        equivalence[name] = error
    raw, rows = {}, []
    official_labels = truth(data, cfg["task"])
    for name, model in [("original", original), ("protected", release)]:
        for split, loader in [
            ("confirmation", make_loader(data.train_dataset, indices["confirmation"])),
            ("official_test", data.test_loader),
        ]:
            predictions, targets = [], []
            result = evaluate(
                model,
                head,
                loader,
                "cuda",
                prediction_sink=lambda p, y: (predictions.append(p), targets.append(y)),
            )
            key = f"{name}/{split}"
            raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(targets))
            if split == "official_test":
                assert result["n"] == fusion["expected_test_counts"][cfg["task"]]
                assert torch.equal(raw[key]["labels"], official_labels)
            row = dict(key=key, **result)
            if name == "protected":
                reference = raw[f"original/{split}"]
                row["paired"] = paired_counts(
                    reference["prediction"].eq(reference["labels"]),
                    raw[key]["prediction"].eq(raw[key]["labels"]),
                )
            rows.append(row)
            print("LINEAR_COMPLETION", key, result["correct"], result["n"], flush=True)
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - started,
            compiled_equivalence=equivalence,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / cfg["task"] / "finetuned.pt",
                    root / f"head_{cfg['task']}.pt",
                ]
            },
            full_evaluations=2,
            confirmation_evaluations=2,
            protected_candidates=1,
            future_partners_accessed=False,
            extra_gradient_steps=0,
            response_fits=0,
            defense_success_claim=False,
        ),
    )


@torch.no_grad()
def validate(args):
    cfg, out = read(args.config), Path(args.output)
    runtime()
    meta = read(out / "metadata.json")
    assert meta["completed"] and meta["config_sha256"] == digest(args.config)
    for path, pin in meta["source_manifest"].items():
        assert digest(path) == pin
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    root, _, original, head, data, indices = inputs(cfg)
    for name, pin in meta["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    assert indices == read(out / "training_indices.json")
    saved = torch.load(out / "artifacts/construction.pt", map_location="cpu", weights_only=True)
    g, alg = saved["geometry"], saved["algebra"]
    # Independent activation and classification-tail computation, rather than
    # only recounting stored scores. Check both sides of the selection boundary.
    z = g["z"].double().numpy()
    sig = 1 / (1 + np.exp(-1.702 * z))
    e = z * (sig - 0.5)
    assert np.allclose(e.mean(0), g["even_mean"].numpy(), atol=1e-12, rtol=1e-10)
    parity_error = float(np.max(np.abs(z * sig + z / (1 + np.exp(1.702 * z)) - z)))
    assert parity_error < 1e-12
    visual = original.model.visual
    residual = g["residual"].numpy()
    gamma, beta = visual.ln_post.weight.numpy(), visual.ln_post.bias.numpy()
    projection, h = visual.proj.double().numpy(), head.weight.double().numpy()

    def scores(r):
        centered = r - np.mean(r, axis=-1, keepdims=True)
        normed = centered / np.sqrt(
            np.mean(centered**2, axis=-1, keepdims=True) + visual.ln_post.eps
        )
        f = (normed * gamma + beta) @ projection
        return (f / np.linalg.norm(f, axis=-1, keepdims=True)) @ h.T

    native_scores = scores(residual)
    assert np.allclose(native_scores, g["baseline"].numpy(), atol=1e-10, rtol=1e-8)
    order = np.argsort(-g["importance"].numpy(), kind="stable")
    n_keep = cfg["retained_nonlinear_units"]
    probes = order[[0, 1, n_keep - 2, n_keep - 1, n_keep, n_keep + 1, -2, -1]]
    source = original.model.visual.transformer.resblocks[cfg["layer"]].mlp
    outgoing = source.c_proj.weight.double().numpy()
    labels, rivals = g["labels"].numpy(), g["spec"]["rival"].numpy()
    row = np.arange(len(labels))
    for unit in probes:
        deleted = residual - (e[:, unit] - e[:, unit].mean())[:, None] * outgoing[:, unit]
        s = scores(deleted)
        margin = s[row, labels] - s[row, rivals]
        change = (margin - g["spec"]["margin"].numpy()) / g["spec"]["scale"].numpy()
        assert np.allclose(change, g["margin_changes"].numpy()[:, unit], atol=1e-9, rtol=1e-6)
    changes, correct = g["margin_changes"].numpy(), g["spec"]["correct"].numpy()
    importance = np.mean(changes[correct] ** 2, axis=0)
    assert np.allclose(importance, g["importance"].numpy(), atol=1e-12, rtol=1e-10)
    selected = np.sort(np.argsort(-importance, kind="stable")[: cfg["retained_nonlinear_units"]])
    assert np.array_equal(selected, g["keep"].numpy())
    drop = g["drop"].numpy()
    a, v, b = source.c_fc.weight.numpy(), source.c_proj.weight.numpy(), source.c_fc.bias.numpy()
    linear = 0.5 * v[:, drop].astype(np.float64) @ a[drop].astype(np.float64)
    expected_bias = source.c_proj.bias.numpy() + v[:, drop].astype(np.float64) @ (
        0.5 * b[drop].astype(np.float64) + g["even_mean"].numpy()[drop]
    )
    assert np.allclose(linear, alg["linear"].numpy(), atol=1e-11, rtol=1e-9)
    assert np.allclose(expected_bias, alg["bias"].numpy(), atol=1e-11, rtol=1e-9)
    operator_error = float(
        np.linalg.norm(alg["outgoing"].numpy() @ alg["incoming"].numpy() - linear)
        / np.linalg.norm(linear)
    )
    assert operator_error < 1e-10
    release = copy.deepcopy(original).cpu().eval()
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        release.state_dict()[key].copy_(value)
    target = release.model.visual.transformer.resblocks[cfg["layer"]].mlp
    active, rank = len(g["keep"]), source.c_fc.in_features
    assert torch.equal(target.c_fc.weight[:active], source.c_fc.weight[g["keep"]])
    assert torch.equal(target.c_proj.weight[:, :active], source.c_proj.weight[:, g["keep"]])
    assert torch.equal(
        target.c_fc.weight[active : active + rank], -target.c_fc.weight[active + rank :]
    )
    assert torch.equal(
        target.c_proj.weight[:, active : active + rank], -target.c_proj.weight[:, active + rank :]
    )
    assert torch.count_nonzero(target.c_fc.bias[active:]) == 0
    predictions = torch.load(
        out / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    checks = 0
    for row in read(out / "results.json"):
        name, split = row["key"].split("/")
        model = original if name == "original" else release
        record = predictions[row["key"]]
        labels = (
            truth(data, cfg["task"])
            if split == "official_test"
            else torch.tensor([int(data.train_dataset[i][1]) for i in indices["confirmation"]])
        )
        assert torch.equal(labels, record["labels"])
        assert row["correct"] == int(record["prediction"].eq(labels).sum())
        dataset = data.test_dataset if split == "official_test" else data.train_dataset
        indexes = list(range(32)) if split == "official_test" else indices["confirmation"][:32]
        x = torch.stack([dataset[i][0] for i in indexes])
        f = features(model.cuda().eval(), x)
        model.cpu()
        p = (f @ head.weight.T + head.bias).argmax(1)
        assert torch.equal(p, record["prediction"][:32])
        if name == "protected":
            expected = reference_features(original, x, g, cfg)
            assert float((f - expected).abs().max()) < cfg["compiled_feature_max_absolute_error"]
            before = predictions[f"original/{split}"]["prediction"].eq(labels)
            assert row["paired"] == paired_counts(before, record["prediction"].eq(labels))
        checks += 1
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            prefix_checks=checks,
            source_checkpoint_raw_fingerprints=True,
            numpy_selection_bias_and_linear_operator=True,
            linear_operator_relative_error=operator_error,
            numpy_even_means_and_eight_counterfactual_columns=True,
            quickgelu_parity_error=parity_error,
            raw_prediction_counts_and_official_labels=True,
            compiled_reference_equivalence=True,
            defense_success_claim=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/linear_completion.json")
    parser.add_argument("--output", default="outputs/2026-10-12/linear_completion")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    (validate if options.validate else run)(options)
