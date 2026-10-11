"""Lossy joint-head representation, with matched public and functional recovery.

One fixed target-only construction. Existing covariance SVD, TA, AGA and score
recovery are reused. No novelty or accuracy-preservation guarantee is assumed.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path

import torch
from clip_benchmark import evaluate
from critical_subspace import capture
from eight_task_benchmark import author_recover, digest, merge_all
from local_erasure import cache
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, replace_components
from path_subspace import context, read
from score_recovery import head_inverse, recover
from subspace_probe import save_json


@torch.no_grad()
def attention(block, inputs):
    x = inputs.cuda()
    n, tokens, width = x.shape
    heads, dim = block.attn.num_heads, width // block.attn.num_heads
    assert getattr(block, "attn_mask", None) is None
    weights, bias = block.attn.in_proj_weight.cuda(), block.attn.in_proj_bias.cuda()
    qkv = torch.nn.functional.linear(x, weights, bias)
    q, k, v = [a.reshape(n, tokens, heads, dim).transpose(1, 2) for a in qkv.chunk(3, -1)]
    maps = ((q / dim**0.5) @ k.transpose(-1, -2)).softmax(-1)
    contexts = (maps @ v).transpose(1, 2).reshape(n, tokens, width)
    return maps, v, contexts


@torch.no_grad()
def factor(block, inputs, pair, rank, cutoff, svd):
    """Optimal rank-r joint VO on the declared truncated value covariance."""
    width, dim = block.attn.embed_dim, block.attn.head_dim
    indexes = torch.cat([torch.arange(h * dim, (h + 1) * dim) for h in pair]).cuda()
    c = torch.cat(
        [
            block.attn.in_proj_weight[2 * width :].double().cuda()[indexes],
            block.attn.in_proj_bias[2 * width :].double().cuda()[indexes, None],
        ],
        1,
    )
    o = block.attn.out_proj.weight.double().cuda()[:, indexes]
    x = inputs.double().cuda().flatten(0, 1)
    x = torch.cat([x, torch.ones(len(x), 1, device="cuda", dtype=torch.float64)], 1)
    z = x @ c.T
    raw_covariance = z.T @ z / len(z)
    values, vectors = torch.linalg.eigh(raw_covariance)
    raw_min, raw_max = float(values.min()), float(values.max())
    assert raw_min >= -1e-9 * raw_max
    keep = values > cutoff * values.max()
    values = torch.where(keep, values, 0)
    half = (vectors * values.sqrt()) @ vectors.T
    invhalf = (vectors * torch.where(keep, values.clamp_min(1e-30).rsqrt(), 0)) @ vectors.T
    u, singular, right = svd(o @ half, full_matrices=False)
    b = right[:, :rank].T @ invhalf @ c
    d = u[:, :rank] * singular[:rank]
    error = o - d @ right[:, :rank].T @ invhalf
    covariance_error = float(((error @ raw_covariance) * error).sum())
    declared_error = float((error @ half).square().sum())
    optimum = float(singular[rank:].square().sum())
    assert abs(declared_error - optimum) <= max(1e-8, optimum * 1e-8)
    return (
        b,
        d,
        dict(
            covariance_rank=int(keep.sum()),
            minimum_raw_eigenvalue=raw_min,
            covariance_cutoff=cutoff * raw_max,
            discarded_covariance_energy=float(torch.trace(raw_covariance) - values.sum()),
            declared_compression_mse=declared_error,
            optimal_tail_mse=optimum,
            actual_token_compression_mse=covariance_error,
            original_joint_rank_upper_bound=len(c),
            compressed_rank_cap=rank,
        ),
        dict(value=c.cpu(), output=o.cpu(), covariance=raw_covariance.cpu(), b=b.cpu(), d=d.cpu()),
    )


@torch.no_grad()
def construct(original, inputs, cfg, svd):
    release = copy.deepcopy(original).cpu().eval()
    selected, candidates, evidence = [], [], {}
    for layer in cfg["layers"]:
        block = original.model.visual.transformer.resblocks[layer]
        x = inputs[f"v:{layer}"].reshape(-1, 50, block.attn.embed_dim)
        maps, values, contexts = attention(block, x)
        error = float((contexts.cpu().flatten(0, 1) - inputs[f"out:{layer}"]).abs().max())
        assert error < 2e-5, (layer, error)
        output = torch.nn.functional.linear(
            contexts, block.attn.out_proj.weight.cuda(), block.attn.out_proj.bias.cuda()
        ).double()
        native_energy = float(output.square().sum())
        contributions = [
            (maps[:, h] @ values[:, h])
            @ block.attn.out_proj.weight[:, h * 64 : (h + 1) * 64].cuda().T
            for h in range(block.attn.num_heads)
        ]
        best, saved = None, None
        for pair in itertools.combinations(range(block.attn.num_heads), 2):
            b, d, geometry, proof = factor(
                block, x, pair, cfg["rank"], cfg["covariance_cutoff"], svd
            )
            old = sum(contributions[h] for h in pair)
            reduced = torch.nn.functional.linear(x.cuda(), b[:, :-1].float(), b[:, -1].float())
            for donor in pair:
                changed = (maps[:, donor] @ reduced) @ d.float().T
                relative = float(((changed - old).double().square().sum() / native_energy).sqrt())
                row = dict(layer=layer, pair=list(pair), donor=donor, local_relative_rms=relative)
                candidates.append(row)
                if best is None or relative < best["local_relative_rms"]:
                    best = dict(row, **geometry, context_max_absolute_error=error)
                    saved = proof
        assert best is not None and saved is not None
        best["original_operator_rank"] = int(
            torch.linalg.matrix_rank(saved["output"].cuda() @ saved["value"].cuda())
        )
        best["compressed_factor_rank"] = int(torch.linalg.matrix_rank(saved["b"].cuda()))
        selected.append(best)
        evidence[str(layer)] = saved
        compiled = release.model.visual.transformer.resblocks[layer].attn
        dim, width, donor = compiled.head_dim, compiled.embed_dim, best["donor"]
        source = slice(donor * dim, (donor + 1) * dim)
        for h in best["pair"]:
            target = slice(h * dim, (h + 1) * dim)
            for offset in (0, width):
                compiled.in_proj_weight[offset + h * dim : offset + (h + 1) * dim].copy_(
                    block.attn.in_proj_weight[offset + source.start : offset + source.stop]
                )
                compiled.in_proj_bias[offset + h * dim : offset + (h + 1) * dim].copy_(
                    block.attn.in_proj_bias[offset + source.start : offset + source.stop]
                )
            compiled.in_proj_weight[2 * width + target.start : 2 * width + target.stop].copy_(
                (saved["b"][:, :-1] / 2**0.5).float()
            )
            compiled.in_proj_bias[2 * width + target.start : 2 * width + target.stop].copy_(
                (saved["b"][:, -1] / 2**0.5).float()
            )
            compiled.out_proj.weight[:, target].copy_((saved["d"] / 2**0.5).float())
        print("HEAD_PAIR", best, flush=True)
    changed = {
        k: v.clone()
        for k, v in release.state_dict().items()
        if not torch.equal(v, original.state_dict()[k])
    }
    assert all(".attn." in k for k in changed)
    assert all(torch.isfinite(v).all() for v in changed.values())
    return release, changed, selected, candidates, evidence


@torch.no_grad()
def reset_heads(reference, anchor, selected):
    result = copy.deepcopy(reference).cpu().eval()
    for item in selected:
        block = result.model.visual.transformer.resblocks[item["layer"]].attn
        base = anchor.model.visual.transformer.resblocks[item["layer"]].attn
        width, dim = block.embed_dim, block.head_dim
        for h in item["pair"]:
            indexes = slice(h * dim, (h + 1) * dim)
            for offset in (0, width, 2 * width):
                target = slice(offset + indexes.start, offset + indexes.stop)
                block.in_proj_weight[target].copy_(base.in_proj_weight[target])
                block.in_proj_bias[target].copy_(base.in_proj_bias[target])
            block.out_proj.weight[:, indexes].copy_(base.out_proj.weight[:, indexes])
    return result


def identical(a, b):
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def truth(data, task):
    dataset = data.test_dataset
    values = (
        dataset.labels
        if task == "SVHN"
        else ([label for _, label in dataset._samples] if task == "Cars" else dataset.targets)
    )
    return torch.as_tensor(values, dtype=torch.long)


@torch.no_grad()
def run(args):
    start = time.perf_counter()
    cfg, out = read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, fusion, models = context(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    from dataset.registry import get_dataset

    assert read(Path(cfg["indices_parent"]) / "validation.json")["status"] == "passed"
    assert (
        cfg["layers"] == read(Path(cfg["query_parent"]) / "selection.json")[cfg["task"]]["layers"]
    )
    task = cfg["task"]
    original = load(str(root / task / "finetuned.pt")).cpu().eval()
    data = {task: get_dataset(task, original.val_preprocess, "data", batch_size=128, num_workers=2)}
    indices = read(Path(cfg["indices_parent"]) / "training_indices.json")[task]
    queries = read(Path(cfg["query_parent"]) / "training_indices.json")[task]["selection"]
    queries = queries[slice(*cfg["query_slice"])]
    assert (
        len(indices["calibration"]) == 64
        and len(indices["confirmation"]) == 128
        and len(queries) == 256
    )
    assert not set(queries).intersection(indices["calibration"] + indices["confirmation"])
    images = torch.stack([data[task].train_dataset[i][0] for i in indices["calibration"]])
    inputs, context_checks = capture(original, images, cfg["layers"])
    svd = author_svd("external/NUFILT", read(cfg["nufilt_config"]))
    release, changed, selected, candidates, evidence = construct(original, inputs, cfg, svd)
    torch.save(changed, artifacts / "release_edits.pt")
    torch.save(dict(inputs=inputs, factors=evidence), artifacts / "construction.pt")
    save_json(out / "selected.json", selected)
    save_json(out / "candidates.json", candidates)
    save_json(out / "context_checks.json", context_checks)
    save_json(out / "training_indices.json", dict(**indices, queries=queries))
    save_json(
        out / "release_manifest.json",
        dict(
            edited_keys=list(changed),
            sha256=digest(artifacts / "release_edits.pt"),
            future_partner_loaded=False,
            calibration_labels_used=False,
            optimizer_steps=0,
            defender_response_fits=0,
        ),
    )
    print("RELEASE_FROZEN", digest(artifacts / "release_edits.pt"), flush=True)
    del inputs
    _, _, _, models = context(cfg, all_tasks=True)
    identical(original, models[task])
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    assert torch.equal(heads[task].weight, heads["SVHN"].weight)
    for t in cfg["evaluation_tasks"]:
        if t not in data:
            data[t] = get_dataset(t, original.val_preprocess, "data", batch_size=128, num_workers=2)
    loaders = {t: cache(d) for t, d in data.items()}
    ground_truth = {t: truth(d, t) for t, d in data.items()}
    rows, raw, functions = [], {}, {}

    def measure(model, preset, attack, scope, datasets):
        for t in datasets:
            p, y = [], []
            score = evaluate(
                model,
                heads[t],
                loaders[t],
                "cuda",
                prediction_sink=lambda a, b: (p.append(a), y.append(b)),
            )
            key = f"{preset}/{attack}/{scope}/{t}"
            value = dict(prediction=torch.cat(p), labels=torch.cat(y))
            assert len(value["labels"]) == fusion["expected_test_counts"][t]
            assert torch.equal(value["labels"], ground_truth[t])
            assert score["correct"] == int(value["prediction"].eq(value["labels"]).sum())
            raw[key] = value
            rows.append(
                dict(key=key, preset=preset, attack=attack, scope=scope, dataset=t, **score)
            )
            print("MEASURE", key, score["correct"], score["n"], flush=True)

    confirmation = torch.stack([data[task].train_dataset[i][0] for i in indices["confirmation"]])
    from path_subspace import response

    functions["confirmation"] = response(original, release, confirmation)
    measure(original, "original", "none", "standalone", [task])
    measure(release, "protected", "none", "standalone", [task])
    author, author_hash = load_author(
        "external/secure-merge-attack/utils_attack_B32.py", include_wrappers=True
    )
    assert author_hash == "4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778"
    inverse, inverse_geometry = head_inverse(
        heads[task], [heads[t] for t in cfg["preserved_query_heads"]]
    )
    query_images = torch.stack([data[task].train_dataset[i][0] for i in queries])
    fits = {}
    for preset, teacher in [("original", original), ("protected", release)]:
        student = merge_all(dict(models, **{task: teacher}), root, cfg["coefficient"])
        measure(student, preset, "direct", "fusion", cfg["evaluation_tasks"])
        restored = author_recover(teacher, models["anchor"], author)
        edits = {
            k: v.clone()
            for k, v in restored.state_dict().items()
            if not torch.equal(v, teacher.state_dict()[k])
        }
        torch.save(edits, artifacts / f"{preset}_aga_edits.pt")
        merged = merge_all(dict(models, **{task: restored}), root, cfg["coefficient"])
        measure(merged, preset, "aga", "fusion", cfg["evaluation_tasks"])
        del restored, merged
        unused, fits[preset], evidence = recover(
            teacher, student, heads[task], inverse, query_images, cfg["ridge_ratio"]
        )
        del unused
        query_model = copy.deepcopy(student).cpu().eval()
        query_model.model.visual.proj.copy_(
            (student.model.visual.proj.double() + evidence["projection_delta"]).float()
        )
        evidence["compiled_student_projection"] = query_model.model.visual.proj.detach().clone()
        torch.save(evidence, artifacts / f"{preset}_query.pt")
        measure(query_model, preset, "readout", "fusion", cfg["evaluation_tasks"])
        del query_model, student
    equality = {}
    for attack in ("support_reset", "reset_attention"):
        if attack == "support_reset":
            a, b = [reset_heads(t, models["anchor"], selected) for t in (original, release)]
        else:
            a, b = [
                replace_components(t, models["anchor"], list(range(12)), "attn")[0]
                for t in (original, release)
            ]
        identical(a, b)
        equality[attack] = dict(original_and_protected_states_identical=True, measured_once=True)
        measure(a, "identical_reset", attack, "standalone", [task])
        student = merge_all(dict(models, **{task: a}), root, cfg["coefficient"])
        measure(student, "identical_reset", attack, "fusion", cfg["evaluation_tasks"])
        del a, b, student
    for row in rows:
        reference = f"original/{row['attack']}/{row['scope']}/{row['dataset']}"
        if row["preset"] == "protected" and reference in raw:
            row["matched_change"] = paired_counts(
                raw[reference]["prediction"].eq(raw[reference]["labels"]),
                raw[row["key"]]["prediction"].eq(raw[row["key"]]["labels"]),
            )
    save_json(out / "results.json", rows)
    save_json(out / "functions.json", functions)
    save_json(out / "query_fits.json", dict(geometry=inverse_geometry, fits=fits))
    save_json(out / "replacement_checks.json", equality)
    torch.save(raw, artifacts / "predictions.pt")
    helpers = [
        "clip_benchmark.py",
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "local_erasure.py",
        "mechanisms.py",
        "nufilt_alignment.py",
        "operator_diagnostics.py",
        "path_subspace.py",
        "score_recovery.py",
        "polynomial_probe.py",
        "subspace_probe.py",
        "functional_contrast.py",
        "joint_posthoc.py",
        "low_activity_posthoc.py",
        "fine_subspace.py",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            new_full_evaluations=len(rows),
            coefficient=cfg["coefficient"],
            source_sha256=digest(Path(__file__)),
            config_sha256=digest(Path(args.config)),
            helper_sha256={name: digest(Path("experiments") / name) for name in helpers},
            input_sha256={
                name: digest(name)
                for name in [
                    cfg["fusion_config"],
                    cfg["nufilt_config"],
                    "configs/checkpoints.json",
                    "configs/sources.json",
                    str(Path(cfg["indices_parent"]) / "training_indices.json"),
                    str(Path(cfg["query_parent"]) / "training_indices.json"),
                    str(Path(cfg["query_parent"]) / "selection.json"),
                ]
            },
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))
            },
            raw_sha256={p.name: digest(p) for p in sorted(artifacts.iterdir()) if p.is_file()},
            construction_frozen_before_partner_loading=True,
            author_aga_sha256=author_hash,
            eight_task_mean_reported=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_collapse.json")
    parser.add_argument("--output", default="outputs/2026-10-11/head_collapse")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
