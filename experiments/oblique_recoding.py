"""One rank-deficient local Attention/MLP recoding and public completion screen.

No novelty or utility guarantee follows from the matrix rank. Reuse existing
author checkpoints/TA/evaluator; test neutral LN folding and matched recovery.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from eight_task_benchmark import digest, merge_all
from joint_posthoc import capture_blocks, flatten_tokens
from local_erasure import cache
from operator_diagnostics import paired_counts
from path_subspace import context, read
from query_partner_cost import runtime, truth
from subspace_probe import make_loader, save_json


def output_matrix(model, layers):
    return torch.cat(
        [
            module.weight.detach().double().cpu()
            for layer in layers
            for module in [
                model.model.visual.transformer.resblocks[layer].attn.out_proj,
                model.model.visual.transformer.resblocks[layer].mlp.c_proj,
            ]
        ],
        dim=1,
    )


@torch.no_grad()
def capture(model, images, layers):
    branch = {layer: {"attention": [], "mlp": []} for layer in layers}
    handles = []
    for layer in layers:
        block = model.model.visual.transformer.resblocks[layer]
        handles.append(
            block.attn.register_forward_hook(
                lambda _m, _a, value, layer=layer: branch[layer]["attention"].append(
                    flatten_tokens(value[0])
                )
            )
        )
        handles.append(
            block.mlp.register_forward_hook(
                lambda _m, _a, value, layer=layer: branch[layer]["mlp"].append(
                    flatten_tokens(value)
                )
            )
        )
    try:
        result = capture_blocks(model, images, layers, details=True)
    finally:
        for handle in handles:
            handle.remove()
    for layer in layers:
        result[layer].update({key: torch.cat(chunks) for key, chunks in branch[layer].items()})
    return result


@torch.no_grad()
def directions(teacher, anchor, values, layers):
    width = 768
    one = torch.ones(width, dtype=torch.float64) / width**0.5
    j = torch.eye(width, dtype=torch.float64) - one[:, None] @ one[None, :]
    task_gram = torch.zeros_like(j)
    for layer in layers:
        own = teacher.model.visual.transformer.resblocks[layer]
        base = anchor.model.visual.transformer.resblocks[layer]
        for a, b in [(own.attn.out_proj, base.attn.out_proj), (own.mlp.c_proj, base.mlp.c_proj)]:
            delta = a.weight.double().cpu() - b.weight.double().cpu()
            task_gram += delta @ delta.T / delta.square().sum()
    eigenvalues, eigenvectors = torch.linalg.eigh(j @ task_gram @ j)
    u = eigenvectors[:, -1]
    u -= (u @ one) * one
    u /= u.norm()
    u *= 1 if u[u.abs().argmax()] >= 0 else -1
    span = torch.linalg.qr(torch.stack([one, u], dim=1), mode="complete").Q[:, 2:]
    covariance = torch.zeros_like(j)
    tensors = {}
    for layer in layers:
        block = teacher.model.visual.transformer.resblocks[layer]
        local = values[layer]
        for name, norm in [("attention_input", block.ln_1), ("mlp_input", block.ln_2)]:
            z = (local[name].double() - norm.bias.double().cpu()) / norm.weight.double().cpu()
            tensors[f"{layer}/{name}"] = z
        for name in ["attention", "mlp"]:
            tensors[f"{layer}/{name}"] = local[name].double()
    for z in tensors.values():
        covariance += z.T @ z / z.square().sum()
    minimum, vectors = torch.linalg.eigh(span.T @ covariance @ span)
    v = span @ vectors[:, 0]
    v /= v.norm()
    v *= 1 if v[v.abs().argmax()] >= 0 else -1
    assert float(abs(v @ u)) < 1e-10 and float(abs(v @ one)) < 1e-10
    t = torch.eye(width, dtype=torch.float64) + (u - v)[:, None] @ v[None, :]
    assert float((t @ t - t).abs().max()) < 1e-10
    assert float((t @ (v - u)).norm()) < 1e-10
    assert float((t @ one - one).norm()) < 1e-10
    report = dict(
        rank=767,
        nonzero_singular_values="sqrt(2), 1 repeated766; one zero",
        alpha=1.0,
        task_output_eigenvalue=float(eigenvalues[-1]),
        selected_covariance_eigenvalue=float(minimum[0]),
        task_output_energy_fraction=float(u @ task_gram @ u / task_gram.trace()),
        calibration_response_fraction={
            name: float((z @ v).square().sum() / z.square().sum()) for name, z in tensors.items()
        },
    )
    return t, dict(u=u, v=v, covariance=covariance, task_gram=task_gram), report


@torch.no_grad()
def fold(teacher, layers, transform=None):
    result = copy.deepcopy(teacher).cpu().eval()
    t = torch.eye(768, dtype=torch.float64) if transform is None else transform
    for layer in layers:
        block = result.model.visual.transformer.resblocks[layer]
        for norm, module, attention in [
            (block.ln_1, block.attn, True),
            (block.ln_2, block.mlp.c_fc, False),
        ]:
            weight = module.in_proj_weight if attention else module.weight
            bias = module.in_proj_bias if attention else module.bias
            w = weight.double().clone()
            bias.copy_((bias.double() + w @ norm.bias.double()).float())
            weight.copy_(((w * norm.weight.double()[None, :]) @ t).float())
            norm.weight.fill_(1)
            norm.bias.zero_()
        for output in [block.attn.out_proj, block.mlp.c_proj]:
            output.weight.copy_((t @ output.weight.double()).float())
            output.bias.copy_((t @ output.bias.double()).float())
    return result


@torch.no_grad()
def canonical(model, anchor, layers):
    """Exact affine normalization change; no private teacher weights enter."""
    result = copy.deepcopy(model).cpu().eval()
    for layer in layers:
        block = result.model.visual.transformer.resblocks[layer]
        base = anchor.model.visual.transformer.resblocks[layer]
        for norm, reference, module, attention in [
            (block.ln_1, base.ln_1, block.attn, True),
            (block.ln_2, base.ln_2, block.mlp.c_fc, False),
        ]:
            weight = module.in_proj_weight if attention else module.weight
            bias = module.in_proj_bias if attention else module.bias
            w = weight.double().clone()
            new = (w * norm.weight.double()[None, :]) / reference.weight.double()[None, :]
            bias.copy_(
                (bias.double() + w @ norm.bias.double() - new @ reference.bias.double()).float()
            )
            weight.copy_(new.float())
            norm.load_state_dict(reference.state_dict())
    return result


@torch.no_grad()
def complete(model, anchor, layers, rcond):
    """Estimate a common transform from public anchor, then fill missing modes."""
    source, public = output_matrix(model, layers), output_matrix(anchor, layers)
    estimate = torch.linalg.solve(public @ public.T, public @ source.T).T
    left, singular, right = torch.linalg.svd(estimate, full_matrices=False)
    active = singular > singular[0] * rcond
    inverse = (right[active].T / singular[active]) @ left[:, active].T
    result = copy.deepcopy(model).cpu().eval()
    base_fold = fold(anchor, layers)
    for layer in layers:
        block = result.model.visual.transformer.resblocks[layer]
        base = base_fold.model.visual.transformer.resblocks[layer]
        for output, reference in [
            (block.attn.out_proj, base.attn.out_proj),
            (block.mlp.c_proj, base.mlp.c_proj),
        ]:
            w0, b0 = reference.weight.double(), reference.bias.double()
            output.weight.copy_((w0 + inverse @ (output.weight.double() - estimate @ w0)).float())
            output.bias.copy_((b0 + inverse @ (output.bias.double() - estimate @ b0)).float())
        for module, reference, attention in [
            (block.attn, base.attn, True),
            (block.mlp.c_fc, base.mlp.c_fc, False),
        ]:
            weight = module.in_proj_weight if attention else module.weight
            w0 = reference.in_proj_weight.double() if attention else reference.weight.double()
            weight.copy_((w0 + (weight.double() - w0 @ estimate) @ inverse).float())
    return result, dict(
        estimate=estimate,
        singular_values=singular,
        retained_rank=int(active.sum()),
        inverse=inverse,
    )


def load(cfg):
    root, loader, fusion, models = context(cfg)
    target = cfg["target"]
    models[target] = loader(str(root / target / "finetuned.pt")).cpu().eval()
    from dataset.registry import get_dataset

    data = get_dataset(
        target, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2
    )
    head = loader(str(root / f"head_{target}.pt"))
    indices = read(Path(cfg["indices_parent"]) / "attack_indices.json")
    assert len(indices) == 256 and len(set(indices)) == 256
    images = torch.stack([data.train_dataset[i][0] for i in indices])
    return root, loader, fusion, models, data, head, indices, images


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    runtime()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    root, loader, fusion, models, data, head, indices, images = load(cfg)
    target, layers = cfg["target"], cfg["layers"]
    original, anchor = models[target], models["anchor"]
    values = capture(original, images, layers)
    transform, proof, selection = directions(original, anchor, values, layers)
    folded, protected = fold(original, layers), fold(original, layers, transform)
    edits = {
        k: v.clone()
        for k, v in protected.state_dict().items()
        if not torch.equal(v, original.state_dict()[k])
    }
    assert all(
        any(k.startswith(f"model.visual.transformer.resblocks.{layer}.") for layer in layers)
        for k in edits
    )
    torch.save(edits, artifacts / "release_edits.pt")
    torch.save(dict(transform=transform, **proof), artifacts / "certificate.pt")
    save_json(out / "training_indices.json", dict(calibration=indices))
    save_json(out / "construction.json", selection)
    save_json(
        out / "release_freeze.json",
        dict(
            sha256=digest(artifacts / "release_edits.pt"),
            future_partner_checkpoints_loaded=False,
            calibration_labels_used=False,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    print("RELEASE_FROZEN", selection, flush=True)
    del values
    # The release is immutable before any future partner enters the evaluator.
    for task in fusion["tasks"]:
        if task not in models:
            models[task] = loader(str(root / task / "finetuned.pt")).cpu().eval()
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    assert {str(p.relative_to(root)): digest(p) for p in root.rglob("*.pt")} == expected
    parent = Path(cfg["baseline_parent"])
    prior_meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert prior_meta["checkpoint_sha256"] == expected
    assert digest(parent / "artifacts/predictions.pt") == prior_meta["raw_sha256"]["predictions.pt"]
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows, raw, attack_evidence = [], {}, {}
    test = cache(data)
    for group in ["standalone", "fusion"]:
        key = f"original/direct/{group}/{target}"
        raw[key] = prior[f"{target}/original/{group}/{target}"]
        y = truth(data, target)
        assert torch.equal(raw[key]["labels"], y)
        correct = int(raw[key]["prediction"].eq(y).sum())
        rows.append(
            dict(
                key=key,
                preset="original",
                attack="direct",
                group=group,
                dataset=target,
                correct=correct,
                n=len(y),
                top1=correct / len(y),
                reused=True,
            )
        )
    for preset, teacher in [("folded", folded), ("protected", protected)]:
        for attack in cfg["attacks"]:
            if attack == "direct":
                recovered = teacher
            elif attack == "canonical":
                recovered = canonical(teacher, anchor, layers)
            else:
                assert attack == "public_completion"
                recovered, evidence = complete(teacher, anchor, layers, cfg["inverse_rcond"])
                recovered = canonical(recovered, anchor, layers)
                attack_evidence[preset] = evidence
            merged = merge_all(dict(models, **{target: recovered}), root, cfg["coefficient"])
            for group, model in [("standalone", recovered), ("fusion", merged)]:
                p, y = [], []
                score = evaluate(
                    model,
                    head,
                    test,
                    "cuda",
                    prediction_sink=lambda a, b: (p.append(a), y.append(b)),
                )
                key = f"{preset}/{attack}/{group}/{target}"
                raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
                rows.append(
                    dict(
                        key=key,
                        preset=preset,
                        attack=attack,
                        group=group,
                        dataset=target,
                        reused=False,
                        **score,
                    )
                )
                save_json(out / "results.json", rows)
                print("OBLIQUE_SCREEN", key, score["correct"], score["n"], flush=True)
            del merged, recovered
    torch.save(attack_evidence, artifacts / "public_completion.pt")
    for row in rows:
        labels = raw[row["key"]]["labels"]
        base = raw[f"original/direct/{row['group']}/{target}"]
        row["original_change"] = paired_counts(
            base["prediction"].eq(labels), raw[row["key"]]["prediction"].eq(labels)
        )
        if row["preset"] == "protected":
            matched = raw[row["key"].replace("protected/", "folded/", 1)]
            row["matched_folded_change"] = paired_counts(
                matched["prediction"].eq(labels), raw[row["key"]]["prediction"].eq(labels)
            )
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    assert len(rows) == 14
    save_json(
        out / "completion.json",
        {
            name: dict(
                retained_rank=value["retained_rank"],
                smallest_singular=float(value["singular_values"][-1]),
                relative_transform_error=float(
                    (value["estimate"] - transform).norm() / transform.norm()
                )
                if name == "protected"
                else None,
            )
            for name, value in attack_evidence.items()
        },
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256=expected,
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            fresh_full_evaluations=12,
            reused_baselines=2,
            calibration_samples=256,
            evaluation_tasks=[target],
            partners_not_yet_evaluated=True,
            author_AGA_not_yet_evaluated=True,
            eight_task_mean=False,
            defense_success_claim=False,
        ),
    )


@torch.no_grad()
def validate(args):
    """Replay full counts/prefixes and independently check projection/completion."""
    from functional_contrast import features
    from scipy.linalg import eigh

    start, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(args.config), read(out / "metadata.json")
    assert meta["completed"] and cfg == read(out / "protocol.json")
    assert digest(args.config) == meta["config_sha256"]
    for path, sha in meta["source_manifest"].items():
        assert digest(path) == sha, path
    for name, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == sha, name
    runtime()
    root, loader, fusion, models, data, head, indices, images = load(cfg)
    assert read(out / "training_indices.json")["calibration"] == indices
    for name, sha in meta["checkpoint_sha256"].items():
        assert digest(root / name) == sha, name
    target, layers = cfg["target"], cfg["layers"]
    original, anchor = models[target], models["anchor"]
    proof = torch.load(out / "artifacts/certificate.pt", map_location="cpu", weights_only=True)
    u, v, transform = proof["u"], proof["v"], proof["transform"]
    one = torch.ones_like(u) / len(u) ** 0.5
    j = torch.eye(len(u), dtype=u.dtype) - torch.outer(one, one)
    _, eu = eigh((j @ proof["task_gram"] @ j).numpy(), subset_by_index=[767, 767])
    assert abs(float(u @ torch.from_numpy(eu[:, 0].copy()))) > 1 - 1e-9
    q = torch.linalg.qr(torch.stack([one, u], dim=1), mode="complete").Q[:, 2:]
    minimum, ev = eigh((q.T @ proof["covariance"] @ q).numpy(), subset_by_index=[0, 0])
    independent_v = q @ torch.from_numpy(ev[:, 0].copy())
    assert abs(float(v @ independent_v)) > 1 - 1e-7
    independent_t = j + torch.outer(one, one) + torch.outer(u - v, v)
    assert torch.allclose(transform, independent_t, atol=1e-12, rtol=0)
    spectrum = torch.linalg.svdvals(transform)
    assert float(spectrum[-1]) < 1e-12
    assert torch.allclose(spectrum[1:-1], torch.ones(766, dtype=spectrum.dtype), atol=1e-12)
    assert abs(float(spectrum[0]) - 2**0.5) < 1e-12
    folded, protected = fold(original, layers), copy.deepcopy(original).cpu().eval()
    edits = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    assert digest(out / "artifacts/release_edits.pt") == read(out / "release_freeze.json")["sha256"]
    protected.load_state_dict(dict(protected.state_dict(), **edits))
    reconstructed = fold(original, layers, transform)
    assert all(
        torch.equal(w, reconstructed.state_dict()[k]) for k, w in protected.state_dict().items()
    )
    assert all(
        torch.equal(w, original.state_dict()[k])
        for k, w in protected.state_dict().items()
        if k not in edits
    )
    del reconstructed
    # Exact affine controls are checked on forward outputs, not merely weights.
    forward_errors = {}
    samples = images[:8]
    reference = features(original.cuda().eval(), samples)
    original.cpu()
    value = features(folded.cuda().eval(), samples)
    folded.cpu()
    forward_errors["neutral_fold"] = float((value - reference).abs().max())
    assert forward_errors["neutral_fold"] < 1e-5
    for name, model in [("folded", folded), ("protected", protected)]:
        reference = features(model.cuda().eval(), samples)
        model.cpu()
        reparameterized = canonical(model, anchor, layers)
        value = features(reparameterized.cuda().eval(), samples)
        reparameterized.cpu()
        forward_errors[name + "_canonical"] = float((value - reference).abs().max())
        assert forward_errors[name + "_canonical"] < 1e-5
        del reparameterized
    for task in fusion["tasks"]:
        if task not in models:
            models[task] = loader(str(root / task / "finetuned.pt")).cpu().eval()
    parent = Path(cfg["baseline_parent"])
    parent_meta = read(parent / "metadata.json")
    assert read(parent / "validation.json")["status"] == "passed"
    assert (
        digest(parent / "artifacts/predictions.pt") == parent_meta["raw_sha256"]["predictions.pt"]
    )
    prior = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    evidence = torch.load(
        out / "artifacts/public_completion.pt", map_location="cpu", weights_only=True
    )
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 14
    completion_errors, checked = {}, 0
    labels = truth(data, target)
    prefix = make_loader(data.test_dataset, list(range(128)), 128)
    for preset, model in [("original", original), ("folded", folded), ("protected", protected)]:
        for attack in ["direct"] if preset == "original" else cfg["attacks"]:
            if attack == "direct":
                recovered = model
            elif attack == "canonical":
                recovered = canonical(model, anchor, layers)
            else:
                # Cholesky solve checks the estimator independently of torch.solve.
                public, source = output_matrix(anchor, layers), output_matrix(model, layers)
                chol = torch.linalg.cholesky(public @ public.T)
                estimate = torch.cholesky_solve(public @ source.T, chol).T
                saved = evidence[preset]
                error = float((estimate - saved["estimate"]).norm() / estimate.norm())
                assert error < 1e-10
                active = (
                    saved["singular_values"] > saved["singular_values"][0] * cfg["inverse_rcond"]
                )
                assert int(active.sum()) == saved["retained_rank"]
                residual = estimate @ saved["inverse"] @ estimate - estimate
                assert float(residual.norm() / estimate.norm()) < cfg["inverse_rcond"]
                recovered, replay = complete(model, anchor, layers, cfg["inverse_rcond"])
                assert all(
                    torch.equal(value, saved[key])
                    if isinstance(value, torch.Tensor)
                    else value == saved[key]
                    for key, value in replay.items()
                )
                recovered = canonical(recovered, anchor, layers)
                completion_errors[preset] = error
            merged = merge_all(dict(models, **{target: recovered}), root, cfg["coefficient"])
            for group, state in [("standalone", recovered), ("fusion", merged)]:
                key = f"{preset}/{attack}/{group}/{target}"
                row = next(row for row in rows if row["key"] == key)
                value = raw[key]
                assert torch.equal(value["labels"], labels)
                assert row["n"] == len(labels) == fusion["expected_test_counts"][target]
                assert row["correct"] == int(value["prediction"].eq(labels).sum())
                assert row["top1"] == row["correct"] / row["n"]
                if row["reused"]:
                    assert torch.equal(
                        value["prediction"],
                        prior[f"{target}/original/{group}/{target}"]["prediction"],
                    )
                assert row["original_change"] == paired_counts(
                    raw[f"original/direct/{group}/{target}"]["prediction"].eq(labels),
                    value["prediction"].eq(labels),
                )
                if preset == "protected":
                    assert row["matched_folded_change"] == paired_counts(
                        raw[key.replace("protected/", "folded/", 1)]["prediction"].eq(labels),
                        value["prediction"].eq(labels),
                    )
                sink = []
                evaluate(state, head, prefix, "cuda", prediction_sink=lambda p, _y: sink.append(p))
                assert torch.equal(torch.cat(sink), value["prediction"][:128]), key
                checked += 1
            del recovered, merged
    assert checked == 14
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            full_counts_pairs_and_official_prefixes=checked,
            released_weights_reconstructed=True,
            rank_and_directions_independent_scipy_checked=True,
            minimum_covariance_eigenvalue=float(minimum[0]),
            affine_forward_max_errors=forward_errors,
            independent_cholesky_completion_errors=completion_errors,
            results_sha256=digest(out / "results.json"),
            metadata_sha256=digest(out / "metadata.json"),
            defense_success_claim=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/oblique_recoding.json")
    parser.add_argument("--output", default="outputs/2026-10-12/oblique_recoding")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    parsed = parser.parse_args()
    (validate if parsed.validate else run)(parsed)
