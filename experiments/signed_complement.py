"""Target-only complement probe with a signed classification-margin objective.

One bounded construction; no gradient updates, response reconstruction or
partner access before release. PCA, NUFILT SVD, finite differences and the
author's original merger/AGA are reused. This is not a novelty claim.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import evaluate
from critical_subspace import parameters
from eight_task_benchmark import author_recover, digest, merge_all
from fine_subspace import fresh_indices
from functional_contrast import features
from local_erasure import cache, rescale
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts
from path_subspace import apply, context, edits_for, finite_differences, read
from query_partner_cost import runtime, truth
from scipy.linalg import eigh
from subspace_probe import collect_moments, make_loader, save_json


def aliases(sites):
    return [s.replace("v:", "V:").replace("up:", "FC:") for s in sites]


@torch.no_grad()
def geometry(expert, anchor, moments, cfg, svd):
    records, descriptions, evidence = [], [], {}
    for site, alias in zip(cfg["sites"], aliases(cfg["sites"]), strict=True):
        own = moments["expert"][alias]
        covariance, mean = own["covariance"], own["mean"]
        values, vectors = torch.linalg.eigh(covariance)
        values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
        rank = int(torch.searchsorted(values.cumsum(0) / values.sum(), cfg["task_energy"])) + 1
        retained = vectors[:, :rank]
        kind, layer = site.split(":")
        block = expert.model.visual.transformer.resblocks[int(layer)]
        ln = block.ln_1 if kind == "v" else block.ln_2
        null = ln.weight.double().cpu().reciprocal()
        null /= null.norm()
        residual = null - retained @ (retained.T @ null)
        span = retained
        if float(residual.norm()) > 1e-8:
            span = torch.cat([span, (residual / residual.norm())[:, None]], dim=1)
        complement = torch.linalg.qr(span, mode="complete").Q[:, span.shape[1] :]
        assert complement.shape[1] > 0, site
        a = complement.T @ covariance @ complement
        proxy_covariances = []
        for beta in cfg["betas"]:
            moment = moments[f"beta_{beta}"][alias]
            shift = moment["mean"] - mean
            proxy_covariances.append(moment["covariance"] + shift[:, None] @ shift[None, :])
        b = complement.T @ (sum(proxy_covariances) / len(proxy_covariances)) @ complement
        rho = cfg["covariance_ridge_fraction"] * float(a.trace()) / len(a) + 1e-12
        eigenvalues, eigenvectors = eigh(b.numpy(), (a + rho * torch.eye(len(a))).numpy())
        v = complement @ torch.from_numpy(eigenvectors[:, -1].copy())
        v /= v.norm()
        v *= 1 if v[v.abs().argmax()] >= 0 else -1
        _, we, _ = parameters(expert.state_dict(), site)
        _, w0, _ = parameters(anchor.state_dict(), site)
        left, singular, _ = svd((we.double() - w0.double()).cuda(), full_matrices=False)
        u = left[:, 0].cpu()
        u *= 1 if u[u.abs().argmax()] >= 0 else -1
        outgoing = u * we.double().norm()
        records.append(dict(site=site, mean=mean, incoming=v, outgoing=outgoing))
        description = dict(
            site=site,
            input_rank=rank,
            preserved_rank=span.shape[1],
            complement_dimension=complement.shape[1],
            rho=rho,
            leading_regularized_input_contrast=float(eigenvalues[-1]),
            retained_input_leakage=float((retained.T @ v).norm()),
            affine_null_leakage=float(abs(null @ v)),
            task_left_leading_energy=float(singular[0].square() / singular.square().sum()),
            incoming_power={
                name: float(
                    v
                    @ (
                        m[alias]["covariance"]
                        + (m[alias]["mean"] - mean)[:, None] @ (m[alias]["mean"] - mean)[None, :]
                    )
                    @ v
                )
                for name, m in moments.items()
            },
        )
        assert description["retained_input_leakage"] < 1e-8
        assert description["affine_null_leakage"] < 1e-8
        descriptions.append(description)
        evidence[site] = dict(complement=complement, retained=retained, a=a, b=b, null=null)
        print("GEOMETRY", description, flush=True)
    return records, descriptions, evidence


def logits(f, head):
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    return f.double() @ head.weight.double().cpu().T


def margin_spec(scores, labels):
    competitors = scores.clone()
    competitors[torch.arange(len(labels)), labels] = -torch.inf
    rival = competitors.argmax(1)
    margin = scores[torch.arange(len(labels)), labels] - scores[torch.arange(len(labels)), rival]
    correct = margin > 0
    assert correct.any()
    median = margin[correct].median()
    scale = margin.clamp_min(0.1 * median)
    return dict(rival=rival, margin=margin, correct=correct, scale=scale)


def margin_columns(derivative, head, labels, spec):
    contrasts = head.weight.double().cpu()[labels] - head.weight.double().cpu()[spec["rival"]]
    return torch.einsum("nfd,nf->nd", derivative.double(), contrasts) / spec["scale"][:, None]


def solve(columns, specs, cfg):
    own = columns["expert"][specs["expert"]["correct"]]
    gram = own.T @ own / len(own)
    g = sum(
        columns[f"beta_{b}"][specs[f"beta_{b}"]["correct"]].mean(0) for b in cfg["betas"]
    ) / len(cfg["betas"])
    rho = cfg["margin_ridge_fraction"] * float(gram.trace()) / len(gram) + 1e-12
    metric = gram + rho * torch.eye(len(gram), dtype=torch.float64)
    c = -torch.linalg.solve(metric, g)
    assert c.norm() > 1e-12
    c /= c.norm()
    amplitude = min(
        cfg["native_relative_margin_rms_budget"] / float((c @ metric @ c).sqrt()),
        cfg["maximum_matrix_relative_change"] / float(c.abs().max()),
    )
    c *= amplitude
    return (
        c,
        dict(
            coefficients=c.tolist(),
            rho=rho,
            maximum_matrix_relative_change=float(c.abs().max()),
            predicted_native_regularized_margin_rms=float((c @ metric @ c).sqrt()),
            predicted_proxy_signed_relative_margin_change=float(g @ c),
            predicted_by_point={
                name: float((j @ c)[specs[name]["correct"]].mean()) for name, j in columns.items()
            },
            labels_used=True,
            future_partners_used=False,
            gradient_updates=0,
            optimizer_steps=0,
            response_fits=0,
        ),
        dict(gram=gram, g=g, metric=metric),
    )


def response(before, after, labels, head, spec=None):
    a, b = logits(before, head), logits(after, head)
    spec = margin_spec(a, labels) if spec is None else spec
    fixed_after = b[torch.arange(len(labels)), labels] - b[torch.arange(len(labels)), spec["rival"]]
    changes = (fixed_after - spec["margin"]) / spec["scale"]
    correct_before, correct_after = a.argmax(1).eq(labels), b.argmax(1).eq(labels)
    return dict(
        n=len(labels),
        original_correct=int(correct_before.sum()),
        protected_correct=int(correct_after.sum()),
        paired=paired_counts(correct_before, correct_after),
        signed_relative_margin_change=float(changes[spec["correct"]].mean()),
        relative_margin_change_rms=float(changes[spec["correct"]].square().mean().sqrt()),
        normalized_feature_delta_rms=float(
            (after.double() - before.double()).square().sum(1).mean().sqrt()
        ),
    )


@torch.no_grad()
def run(args):
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    started = time.perf_counter()
    save_json(out / "protocol.json", cfg)
    root, load, fusion, holders = context(cfg)
    anchor = holders["anchor"]
    expert = load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval()
    head = load(str(root / f"head_{cfg['task']}.pt")).cpu().eval()
    profile = read(Path(cfg["profile_parent"]) / "selection.json")[cfg["task"]]
    assert not profile["partner_used"]
    assert sorted({int(s.split(":")[1]) for s in cfg["sites"]}) == profile["layers"]
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], expert.val_preprocess, "data", batch_size=128, num_workers=2)
    indices = fresh_indices(cfg, cfg["task"], len(data.train_dataset))
    save_json(out / "training_indices.json", {cfg["task"]: indices})
    images, labels = {}, {}
    for split in ("calibration", "confirmation"):
        rows = [data.train_dataset[i] for i in indices[split]]
        images[split] = torch.stack([r[0] for r in rows])
        labels[split] = torch.tensor([int(r[1]) for r in rows])
    points = {"expert": expert}
    for beta in cfg["betas"]:
        points[f"beta_{beta}"] = rescale(expert, anchor, beta)
    moments = {
        name: collect_moments(
            model, make_loader(data.train_dataset, indices["calibration"]), aliases(cfg["sites"])
        )
        for name, model in points.items()
    }
    svd = author_svd("external/NUFILT", read(cfg["nufilt_config"]))
    records, descriptions, geo = geometry(expert, anchor, moments, cfg, svd)
    derivatives, baseline, specs, columns = {}, {}, {}, {}
    for name, model in points.items():
        factor = 1.0 if name == "expert" else float(name.split("_")[1])
        model.cuda()
        baseline[name] = features(model, images["calibration"])
        model.cpu()
        specs[name] = margin_spec(logits(baseline[name], head), labels["calibration"])
        derivatives[name] = finite_differences(
            model, images["calibration"], records, cfg["finite_difference_step"], factor
        )
        columns[name] = margin_columns(derivatives[name], head, labels["calibration"], specs[name])
        print("DERIVATIVES", name, flush=True)
    coefficients, construction, system = solve(columns, specs, cfg)
    edits = edits_for(records, coefficients)
    release = apply(expert, edits)
    actual_edits = {
        k: v.clone()
        for k, v in release.state_dict().items()
        if not torch.equal(v, expert.state_dict()[k])
    }
    torch.save(actual_edits, artifacts / "release_edits.pt")
    torch.save(
        dict(
            records=records,
            moments=moments,
            geometry=geo,
            derivatives=derivatives,
            baseline=baseline,
            specs=specs,
            columns=columns,
            coefficients=coefficients,
            system=system,
            labels=labels,
        ),
        artifacts / "construction.pt",
    )
    save_json(out / "geometry.json", descriptions)
    save_json(out / "construction.json", construction)
    save_json(
        out / "release_freeze.json",
        dict(
            sha256=digest(artifacts / "release_edits.pt"),
            edited_keys=list(actual_edits),
            future_partner_loaded=False,
            test_data_used=False,
            calibration_labels_used=True,
        ),
    )
    responses, raw = {}, {}
    for name, model in points.items():
        factor = 1.0 if name == "expert" else float(name.split("_")[1])
        changed = release if name == "expert" else apply(model, edits, factor)
        for split in images:
            model.cuda()
            before = features(model, images[split])
            model.cpu()
            changed.cuda()
            after = features(changed, images[split])
            changed.cpu()
            key = f"{name}/{split}"
            raw[key] = dict(before=before, after=after, labels=labels[split])
            responses[key] = response(before, after, labels[split], head)
            print("RESPONSE", key, responses[key], flush=True)
        if name != "expert":
            del changed
    save_json(out / "responses.json", responses)
    torch.save(raw, artifacts / "responses.pt")
    del derivatives, moments, geo, points, holders
    # Candidate frozen. No proxy-confirmation or official test adjusts it.
    originals = {t: load(str(root / t / "finetuned.pt")).cpu().eval() for t in fusion["tasks"]}
    heads = {t: load(str(root / f"head_{t}.pt")).cpu().eval() for t in cfg["evaluation_tasks"]}
    datasets = {cfg["task"]: data}
    loaders = {}
    for t in cfg["evaluation_tasks"]:
        if t not in datasets:
            datasets[t] = get_dataset(
                t, anchor.val_preprocess, "data", batch_size=128, num_workers=2
            )
        assert len(datasets[t].test_dataset) == fusion["expected_test_counts"][t]
        loaders[t] = cache(datasets[t])
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == read("configs/sources.json")["AGA"]["sha256"]
    rows, predictions, mappings = [], {}, []
    for preset, model in [("original", expert), ("protected", release)]:
        for attack in ("direct", "aga"):
            processed = copy.deepcopy(model)
            if attack == "aga":
                processed = author_recover(processed, anchor, author)
                mappings.append(
                    dict(
                        preset=preset,
                        changed_fields=sum(
                            not torch.equal(v, model.state_dict()[k])
                            for k, v in processed.state_dict().items()
                        ),
                        author_entry_points=[
                            "universal_attention_attack_params",
                            "attack_vit_mlp_only",
                        ],
                    )
                )
            merged = merge_all(
                dict(originals, **{cfg["task"]: processed}), root, cfg["coefficient"]
            )
            for group, measured, tasks in [
                ("native", processed, [cfg["task"]]),
                ("fusion", merged, cfg["evaluation_tasks"]),
            ]:
                for task in tasks:
                    sink = []
                    value = evaluate(
                        measured,
                        heads[task],
                        loaders[task],
                        "cuda",
                        prediction_sink=lambda p, _y: sink.append(p),
                    )
                    key = f"{preset}/{attack}/{group}/{task}"
                    predictions[key] = torch.cat(sink)
                    correct = predictions[key].eq(truth(datasets[task], task))
                    row = dict(
                        key=key, preset=preset, attack=attack, group=group, task=task, **value
                    )
                    if preset == "protected":
                        ref = predictions[key.replace("protected/", "original/", 1)]
                        row["paired"] = paired_counts(ref.eq(truth(datasets[task], task)), correct)
                    rows.append(row)
                    print("MEASURE", row, flush=True)
            del merged, processed
    torch.save(
        dict(predictions=predictions, labels={t: truth(datasets[t], t) for t in datasets}),
        artifacts / "predictions.pt",
    )
    save_json(out / "results.json", rows)
    save_json(out / "mappings.json", mappings)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            seconds=time.perf_counter() - started,
            author_sha256=author_hash,
            checkpoint_sha256={
                name: digest(root / name)
                for name in [
                    "zeroshot.pt",
                    *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
                    *[f"head_{t}.pt" for t in cfg["evaluation_tasks"]],
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


@torch.no_grad()
def validate(args):
    """Independent NumPy solve/counts and fresh GPU prediction prefixes."""
    runtime()
    out = Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"]
    for name, pin in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == pin
    saved = torch.load(out / "artifacts/construction.pt", map_location="cpu", weights_only=True)
    j = saved["columns"]["expert"].numpy()[saved["specs"]["expert"]["correct"].numpy()]
    gram = j.T @ j / len(j)
    rho = cfg["margin_ridge_fraction"] * np.trace(gram) / len(gram) + 1e-12
    g = np.mean(
        [
            saved["columns"][f"beta_{b}"]
            .numpy()[saved["specs"][f"beta_{b}"]["correct"].numpy()]
            .mean(0)
            for b in cfg["betas"]
        ],
        axis=0,
    )
    c = -np.linalg.solve(gram + rho * np.eye(len(gram)), g)
    c /= np.linalg.norm(c)
    amplitude = min(
        cfg["native_relative_margin_rms_budget"]
        / np.sqrt(c @ (gram + rho * np.eye(len(gram))) @ c),
        cfg["maximum_matrix_relative_change"] / np.abs(c).max(),
    )
    coefficient_error = float(np.max(np.abs(c * amplitude - saved["coefficients"].numpy())))
    assert coefficient_error < 1e-8
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    for row in rows:
        pred, label = raw["predictions"][row["key"]].numpy(), raw["labels"][row["task"]].numpy()
        assert len(pred) == row["n"] and int((pred == label).sum()) == row["correct"]
        if row["preset"] == "protected":
            ref = (
                raw["predictions"][row["key"].replace("protected/", "original/", 1)].numpy()
                == label
            )
            correct = pred == label
            assert int((ref & ~correct).sum()) == row["paired"]["lost"]
            assert int((~ref & correct).sum()) == row["paired"]["gained"]
    responses = torch.load(out / "artifacts/responses.pt", map_location="cpu", weights_only=True)
    root, load, fusion, originals = context(cfg, all_tasks=True)
    for name, pin in meta["checkpoint_sha256"].items():
        assert digest(root / name) == pin
    expert, anchor = originals[cfg["task"]], originals["anchor"]
    release = copy.deepcopy(expert)
    state = release.state_dict()
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for key, value in edits.items():
        state[key].copy_(value)
    assert all(torch.equal(state[k], v) for k, v in expert.state_dict().items() if k not in edits)
    recomputed = apply(expert, edits_for(saved["records"], saved["coefficients"]))
    assert all(torch.equal(v, recomputed.state_dict()[k]) for k, v in state.items())
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    for key, value in responses.items():
        assert (
            response(value["before"], value["after"], value["labels"], heads[cfg["task"]])
            == read(out / "responses.json")[key]
        )
    from dataset.registry import get_dataset

    datasets = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    author, _ = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    prefixes = 0
    for preset, model in [("original", expert), ("protected", release)]:
        for attack in ("direct", "aga"):
            processed = copy.deepcopy(model)
            if attack == "aga":
                processed = author_recover(processed, anchor, author)
            merged = merge_all(
                dict(originals, **{cfg["task"]: processed}), root, cfg["coefficient"]
            )
            for group, tested, tasks in [
                ("native", processed, [cfg["task"]]),
                ("fusion", merged, cfg["evaluation_tasks"]),
            ]:
                for task in tasks:
                    sink = []
                    evaluate(
                        tested,
                        heads[task],
                        make_loader(datasets[task].test_dataset, list(range(32))),
                        "cuda",
                        prediction_sink=lambda p, _y: sink.append(p),
                    )
                    assert torch.equal(
                        torch.cat(sink),
                        raw["predictions"][f"{preset}/{attack}/{group}/{task}"][:32],
                    )
                    prefixes += 1
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            coefficient_numpy_error=coefficient_error,
            count_rows=len(rows),
            fresh_prediction_prefixes=prefixes,
            results_sha256=digest(out / "results.json"),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/signed_complement.json")
    parser.add_argument("--output", default="outputs/2026-10-12/signed_complement")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate(args) if args.validate else run(args)


if __name__ == "__main__":
    main()
