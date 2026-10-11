"""One joint fine-subspace posthoc candidate per target, using beta-path sensitivity.

No defender gradients, optimizer, labels, response fitting or future partner use.
Finite differences, pinned NUFILT SVD, scipy eigensolver and author TA/AGA are reused.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from critical_subspace import capture, parameters, reset
from eight_task_benchmark import author_recover, digest, merge_all
from fine_subspace import fresh_indices
from functional_contrast import features
from local_erasure import cache, rescale, sites
from mechanisms import load_author
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts, replace_components
from scipy.linalg import eigh
from score_recovery import head_inverse, recover
from subspace_probe import make_loader, save_json


def read(path):
    return json.loads(Path(path).read_bytes())


def sampling_config(cfg, task):
    return dict(
        cfg,
        task=task,
        confirmation_samples=cfg.get("confirmation_samples_by_task", {}).get(
            task, cfg["confirmation_samples"]
        ),
    )


@torch.no_grad()
def basis(expert, anchor, inputs, layers, rank, svd):
    records, rows = [], []
    for layer in layers:
        for kind in ("v", "out", "up", "down"):
            site = f"{kind}:{layer}"
            _, we, _ = parameters(expert.state_dict(), site)
            _, w0, _ = parameters(anchor.state_dict(), site)
            delta = (we.double() - w0.double()).cuda()
            _, singular, vectors = svd(delta, full_matrices=False)
            mean = inputs[site].double().mean(0)
            for index in range(rank):
                v = vectors[:, index].clone()
                leakage = None
                if kind in ("v", "up"):
                    block = expert.model.visual.transformer.resblocks[layer]
                    ln = block.ln_1 if kind == "v" else block.ln_2
                    null = ln.weight.double().cuda().reciprocal()
                    null /= null.norm()
                    v -= (v @ null) * null
                    assert float(v.norm()) > 1e-8
                    v /= v.norm()
                    leakage = float(abs(v @ null))
                    assert leakage < 1e-10
                v *= 1 if v[v.abs().argmax()] >= 0 else -1
                outgoing = delta @ v
                records.append(
                    dict(site=site, incoming=v.cpu(), outgoing=outgoing.cpu(), mean=mean)
                )
                rows.append(
                    dict(
                        site=site,
                        index=index,
                        singular_value=float(singular[index]),
                        task_energy_fraction=float(
                            singular[index].square() / singular.square().sum()
                        ),
                        affine_null_leakage=leakage,
                    )
                )
            print("BASIS", site, rank, flush=True)
    return records, rows


@torch.no_grad()
def edits_for(records, coefficients):
    edits = {}
    for item, coefficient in zip(records, coefficients, strict=True):
        site = item["site"]
        update = coefficient * item["outgoing"][:, None] @ item["incoming"][None, :]
        if site not in edits:
            edits[site] = dict(
                weight=torch.zeros_like(update),
                bias=torch.zeros(update.shape[0], dtype=torch.float64),
            )
        edits[site]["weight"] += update
        edits[site]["bias"] -= update @ item["mean"]
    return edits


@torch.no_grad()
def apply(reference, edits, scale=1.0):
    model = copy.deepcopy(reference).cpu().eval()
    state = model.state_dict()
    edited = set()
    for site, item in edits.items():
        keys, weight, bias = parameters(state, site)
        weight.copy_((weight.double() + scale * item["weight"]).float())
        bias.copy_((bias.double() + scale * item["bias"]).float())
        edited.update(keys)
    before = reference.state_dict()
    assert all(torch.equal(v, before[k]) for k, v in state.items() if k not in edited)
    for site in edits:
        if site.startswith("v:"):
            keys, weight, _ = parameters(state, site)
            for key in keys:
                assert torch.equal(
                    state[key][: 2 * weight.shape[1]], before[key][: 2 * weight.shape[1]]
                )
    assert all(torch.isfinite(v).all() for v in state.values())
    return model


@torch.no_grad()
def finite_differences(reference, images, records, step, factor):
    model = copy.deepcopy(reference).cuda().eval()
    result = []
    for index, item in enumerate(records):
        _, weight, bias = parameters(model.state_dict(), item["site"])
        before_w, before_b = weight.clone(), bias.clone()
        update = (item["outgoing"][:, None] @ item["incoming"][None, :]).cuda()
        compensation = (-(update @ item["mean"].cuda())).cuda()
        values = []
        for sign in (1, -1):
            weight.copy_((before_w.double() + sign * step * factor * update).float())
            bias.copy_((before_b.double() + sign * step * factor * compensation).float())
            values.append(features(model, images))
        result.append((values[0] - values[1]) / (2 * step))
        weight.copy_(before_w)
        bias.copy_(before_b)
        if (index + 1) % 12 == 0:
            print("FD", factor, index + 1, len(records), flush=True)
    model.cpu()
    return torch.stack(result, dim=-1)


def gram(derivative):
    flat = derivative.double().flatten(0, 1)
    return flat.T @ flat / len(derivative)


def solve(grams, expert, records, cfg):
    own = grams["expert"]
    proxy = sum(grams[f"beta_{beta}"] for beta in cfg["betas"]) / len(cfg["betas"])
    rho = cfg["regularizer_trace_fraction"] * float(own.trace()) / len(own) + 1e-12
    denominator = own + rho * torch.eye(len(own), dtype=torch.float64)
    values, vectors = eigh(proxy.numpy(), denominator.numpy())
    direction = torch.from_numpy(vectors[:, -1].copy())
    direction /= direction.norm()
    direction *= 1 if direction[direction.abs().argmax()] >= 0 else -1
    own_rms = float((direction @ own @ direction).clamp_min(1e-24).sqrt())
    edits = edits_for(records, direction)
    relative = max(
        float(item["weight"].norm() / parameters(expert.state_dict(), site)[1].double().norm())
        for site, item in edits.items()
    )
    amplitude = min(
        cfg["feature_rms_budget"] / own_rms, cfg["maximum_matrix_relative_change"] / relative
    )
    coefficients = amplitude * direction
    error = float(
        (proxy @ direction - values[-1] * denominator @ direction).norm()
        / (proxy @ direction).norm().clamp_min(1e-30)
    )
    assert error < 1e-8
    return coefficients, dict(
        rho=rho,
        largest_eigenvalue=float(values[-1]),
        eigen_relative_residual=error,
        amplitude=amplitude,
        coefficient_l2=float(coefficients.norm()),
        weight_cap_active=cfg["maximum_matrix_relative_change"] / relative
        <= cfg["feature_rms_budget"] / own_rms,
        maximum_matrix_relative_change=amplitude * relative,
        predicted_feature_rms={
            name: float((coefficients @ g @ coefficients).clamp_min(0).sqrt())
            for name, g in grams.items()
        },
        labels_used=False,
        future_partners_used=False,
        optimizer_steps=0,
        response_fits=0,
    )


@torch.no_grad()
def response(reference, changed, images, prediction=None):
    reference.cuda().eval()
    before = features(reference, images)
    reference.cpu()
    changed.cuda().eval()
    after = features(changed, images)
    changed.cpu()
    delta = after - before
    value = dict(actual_feature_rms=float(delta.double().square().sum().div(len(images)).sqrt()))
    if prediction is not None:
        expected = prediction.double()
        value.update(
            predicted_feature_rms=float(expected.square().sum().div(len(images)).sqrt()),
            prediction_relative_error=float(
                (delta.double() - expected).norm() / delta.double().norm().clamp_min(1e-30)
            ),
            prediction_cosine=float(
                torch.nn.functional.cosine_similarity(
                    delta.double().flatten(), expected.flatten(), dim=0
                )
            ),
        )
    return value


def context(cfg, all_tasks=False):
    root = Path("checkpoints/ViT-B-32")
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    fusion = read(cfg["fusion_config"])
    names = fusion["tasks"] if all_tasks else []
    models = {task: load(str(root / task / "finetuned.pt")).cpu().eval() for task in names}
    models["anchor"] = load(str(root / "zeroshot.pt")).cpu().eval()
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path("external/secure-merge-attack/datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    return root, load, fusion, models


@torch.no_grad()
def run(args):
    cfg = read(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    started = time.perf_counter()
    root, load, fusion, models = context(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    from dataset.registry import get_dataset

    anchor = models["anchor"]
    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["tasks"]
    }
    indices, construction, responses, manifests, releases, images = {}, {}, {}, {}, {}, {}
    reused = {}
    svd = author_svd("external/NUFILT", read(cfg["nufilt_config"]))
    selection = read(Path(cfg["profile_parent"]) / "selection.json")
    for task in cfg["tasks"]:
        assert (
            cfg["layers"][task] == selection[task]["layers"] and not selection[task]["partner_used"]
        )
        original = load(str(root / task / "finetuned.pt")).cpu().eval()
        indices[task] = fresh_indices(
            sampling_config(cfg, task), task, len(data[task].train_dataset)
        )
        images[task] = {
            name: torch.stack([data[task].train_dataset[i][0] for i in indices[task][name]])
            for name in ("calibration", "confirmation")
        }
        x = images[task]["calibration"]
        parent = Path(args.reuse_construction) if args.reuse_construction else None
        if parent is not None and (parent / f"artifacts/{task}_geometry.pt").exists():
            prior = read(parent / "protocol.json")
            for key in cfg:
                if key != "confirmation_samples_by_task":
                    assert cfg[key] == prior[key]
            assert task not in cfg["confirmation_samples_by_task"]
            assert indices[task] == read(parent / "training_indices.json")[task]
            saved = torch.load(
                parent / f"artifacts/{task}_geometry.pt", map_location="cpu", weights_only=True
            )
            records, derivatives = saved["basis"], saved["derivatives"]
            descriptions = read(parent / f"{task}_basis.json")
            checks = read(parent / f"{task}_context_checks.json")
            reused[task] = dict(
                parent=str(parent),
                parent_source_commit=read(parent / "launch.json")["source_commit"],
                geometry_sha256=digest(parent / f"artifacts/{task}_geometry.pt"),
                edits_sha256=digest(parent / f"artifacts/{task}_edits.pt"),
            )
        else:
            inputs, checks = capture(original, x, cfg["layers"][task])
            records, descriptions = basis(
                original, anchor, inputs, cfg["layers"][task], cfg["rank_per_matrix"], svd
            )
            del inputs
            derivatives = {}
            for name, factor in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
                reference = original if name == "expert" else rescale(original, anchor, factor)
                derivatives[name] = finite_differences(
                    reference, x, records, cfg["finite_difference_step"], factor
                )
                if name != "expert":
                    del reference
        grams = {name: gram(value) for name, value in derivatives.items()}
        coefficients, construction[task] = solve(grams, original, records, cfg)
        release = apply(original, edits_for(records, coefficients))
        releases[task] = release
        edited = {
            k: v.clone()
            for k, v in release.state_dict().items()
            if not torch.equal(v, original.state_dict()[k])
        }
        if task in reused:
            prior_edits = torch.load(
                parent / f"artifacts/{task}_edits.pt", map_location="cpu", weights_only=True
            )
            assert (
                all(torch.equal(v, prior_edits[k]) for k, v in edited.items())
                and edited.keys() == prior_edits.keys()
            )
            assert construction[task] == read(parent / "construction.json")[task]
        torch.save(edited, artifacts / f"{task}_edits.pt")
        torch.save(
            dict(basis=records, derivatives=derivatives, grams=grams, coefficients=coefficients),
            artifacts / f"{task}_geometry.pt",
        )
        save_json(out / f"{task}_basis.json", descriptions)
        save_json(out / f"{task}_context_checks.json", checks)
        manifests[task] = dict(
            edited_keys=list(edited), edits_sha256=digest(artifacts / f"{task}_edits.pt")
        )
        print("RELEASE_FROZEN", task, json.dumps(construction[task]), flush=True)
        for name, factor in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
            reference = original if name == "expert" else rescale(original, anchor, factor)
            changed = (
                release
                if name == "expert"
                else apply(reference, edits_for(records, coefficients), factor)
            )
            for split in ("calibration", "confirmation"):
                linear = (
                    derivatives[name].double() @ coefficients if split == "calibration" else None
                )
                responses[f"{task}/{name}/{split}"] = response(
                    reference, changed, images[task][split], linear
                )
            if name != "expert":
                del reference, changed
        del derivatives, grams, records, coefficients
        save_json(out / "construction.json", construction)
        save_json(out / "training_indices.json", indices)
        save_json(out / "responses.json", responses)
        del original, release
    save_json(
        out / "release_freeze.json",
        dict(
            models=manifests,
            future_partners_loaded=False,
            labels_used=False,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    # Construction and amplitude end here. Partners, heads and test data enter only below.
    for task in fusion["tasks"]:
        if task not in models:
            models[task] = load(str(root / task / "finetuned.pt")).cpu().eval()
    heads = {task: load(str(root / f"head_{task}.pt")) for task in cfg["evaluation_tasks"]}
    loaders = {}
    for task in cfg["evaluation_tasks"]:
        if task not in data:
            data[task] = get_dataset(
                task, anchor.val_preprocess, "data", batch_size=128, num_workers=2
            )
        assert len(data[task].test_dataset) == fusion["expected_test_counts"][task]
        loaders[task] = cache(data[task])
    author, author_hash = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    assert author_hash == read("configs/sources.json")["AGA"]["sha256"]
    attack_indices = {
        t: read(Path(cfg["profile_parent"]) / "training_indices.json")[t]["selection"][
            : cfg["attacker_samples"]
        ]
        for t in cfg["tasks"]
    }
    save_json(out / "attack_indices.json", attack_indices)
    raw, rows, fits, mappings = {}, [], {}, []

    def measure(target, preset, attack, group, model, task):
        assert time.perf_counter() - started < cfg["time_limit_seconds"]
        key = f"{target}/{preset}/{attack}/{group}/{task}"
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
                preset=preset,
                attack=attack,
                group=group,
                dataset=task,
                **score,
            )
        )
        torch.save(raw, artifacts / "predictions.pt")
        save_json(out / "results.json", rows)
        print("RESULT", key, score["correct"], score["n"], flush=True)

    for target in cfg["tasks"]:
        attack_images = torch.stack(
            [data[target].train_dataset[i][0] for i in attack_indices[target]]
        )
        inverse, _ = head_inverse(heads[target], [h for t, h in heads.items() if t != target])
        for preset, release in [("original", models[target]), ("protected", releases[target])]:
            for beta in cfg["betas"]:
                proxy = rescale(release, anchor, beta)
                measure(target, preset, f"beta_{beta}", "proxy", proxy, target)
                del proxy
            for attack in cfg["attacks"]:
                if attack == "direct":
                    restored = release
                elif attack == "aga":
                    restored = author_recover(release, anchor, author)
                elif attack == "support_reset":
                    restored = reset(
                        release, anchor, sites(cfg["layers"][target]), "public_support"
                    )
                elif attack in ("reset_attention", "reset_mlp"):
                    restored, _ = replace_components(
                        release, anchor, range(12), "attn" if attack == "reset_attention" else "mlp"
                    )
                else:
                    assert attack == "readout"
                    student = merge_all(dict(models, **{target: release}), root, cfg["coefficient"])
                    restored, fits[f"{target}/{preset}"], evidence = recover(
                        release, student, heads[target], inverse, attack_images, cfg["ridge_ratio"]
                    )
                    expected = student.model.visual.proj.double() + evidence["projection_delta"]
                    torch.save(evidence, artifacts / f"{target}_{preset}_readout.pt")
                    save_json(out / "fitting.json", fits)
                    del student
                if attack != "readout":
                    measure(target, preset, attack, "standalone", restored, target)
                merged = merge_all(dict(models, **{target: restored}), root, cfg["coefficient"])
                if attack == "direct":
                    baseline = merge_all(models, root, cfg["coefficient"])
                    for split in ("calibration", "confirmation"):
                        responses[f"{target}/{preset}/real_fusion/{split}"] = response(
                            baseline, merged, images[target][split]
                        )
                    save_json(out / "responses.json", responses)
                    del baseline
                if attack == "readout":
                    error = float(
                        (merged.model.visual.proj.double() - expected).norm() / expected.norm()
                    )
                    assert error < 1e-6
                    mappings.append(dict(target=target, preset=preset, relative_error=error))
                for task in cfg["evaluation_tasks"]:
                    measure(target, preset, attack, "fusion", merged, task)
                del merged, restored
    for row in rows:
        item = raw[row["id"]]
        control = raw[f"{row['target']}/original/{row['attack']}/{row['group']}/{row['dataset']}"]
        assert torch.equal(item["labels"], control["labels"])
        row["matched"] = paired_counts(
            control["prediction"].eq(control["labels"]), item["prediction"].eq(item["labels"])
        )
    save_json(out / "results.json", rows)
    save_json(out / "projection_checks.json", mappings)
    paths = [
        "zeroshot.pt",
        *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
        *[f"head_{t}.pt" for t in cfg["evaluation_tasks"]],
    ]
    helpers = [
        "clip_benchmark.py",
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "fine_subspace.py",
        "functional_contrast.py",
        "joint_posthoc.py",
        "local_erasure.py",
        "low_activity_posthoc.py",
        "mechanisms.py",
        "nufilt_alignment.py",
        "operator_diagnostics.py",
        "polynomial_probe.py",
        "score_recovery.py",
        "subspace_probe.py",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - started,
            rows=len(rows),
            actual_full_evaluations=len(rows),
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={n: digest(Path(__file__).with_name(n)) for n in helpers},
            fusion_config_sha256=digest(cfg["fusion_config"]),
            checkpoint_sha256={n: digest(root / n) for n in paths},
            input_sha256={
                str(p): digest(p)
                for p in [
                    Path(cfg["nufilt_config"]),
                    Path("configs/checkpoints.json"),
                    Path("configs/sources.json"),
                    Path(cfg["profile_parent"]) / "selection.json",
                    Path(cfg["profile_parent"]) / "training_indices.json",
                    Path(cfg["mnist_exclusions"]),
                    *[Path(p) for p in cfg["exclusions"]],
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            original_author_aga_sha256=author_hash,
            constructor_candidates=2,
            source_labels_used=False,
            optimizer_steps=0,
            constructor_response_fits=0,
            future_partners_used=False,
            reused_construction=reused,
        ),
    )


@torch.no_grad()
def validate(args):
    out = Path(args.output)
    cfg, meta, rows = [
        read(out / name) for name in ("protocol.json", "metadata.json", "results.json")
    ]
    assert meta["completed"] and digest(__file__) == meta["source_sha256"]
    assert (
        digest(args.config) == meta["config_sha256"]
        and digest(cfg["fusion_config"]) == meta["fusion_config_sha256"]
    )
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    root, load, fusion, models = context(cfg, True)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(rows) == len(raw) == 100
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        item = raw[row["id"]]
        labels = torch.tensor(data[row["dataset"]].test_dataset.targets, dtype=torch.long)
        assert torch.equal(item["labels"], labels)
        assert row["n"] == fusion["expected_test_counts"][row["dataset"]] == len(labels)
        correct = int(item["prediction"].eq(labels).sum())
        assert row["correct"] == correct and row["top1"] == correct / len(labels)
        control = raw[f"{row['target']}/original/{row['attack']}/{row['group']}/{row['dataset']}"]
        assert row["matched"] == paired_counts(
            control["prediction"].eq(labels), item["prediction"].eq(labels)
        )
    author, _ = load_author(
        Path("external/secure-merge-attack/utils_attack_B32.py"), include_wrappers=True
    )
    indices = read(out / "training_indices.json")
    attack_indices = read(out / "attack_indices.json")
    checks, prefixes, fd_columns = [], 0, 0
    for target in cfg["tasks"]:
        geometry = torch.load(
            out / f"artifacts/{target}_geometry.pt", map_location="cpu", weights_only=True
        )
        records, derivatives, grams = [geometry[n] for n in ("basis", "derivatives", "grams")]
        assert len(records) == 48 and all(
            torch.equal(gram(d), grams[n]) for n, d in derivatives.items()
        )
        coeff, construction = solve(grams, models[target], records, cfg)
        assert (
            torch.equal(coeff, geometry["coefficients"])
            and construction == read(out / "construction.json")[target]
        )
        predicted = apply(models[target], edits_for(records, coeff))
        edits = torch.load(
            out / f"artifacts/{target}_edits.pt", map_location="cpu", weights_only=True
        )
        expected = {
            k: v
            for k, v in predicted.state_dict().items()
            if not torch.equal(v, models[target].state_dict()[k])
        }
        assert edits.keys() == expected.keys() and all(
            torch.equal(v, expected[k]) for k, v in edits.items()
        )
        assert not set(indices[target]["calibration"]) & set(indices[target]["confirmation"])
        recalculated = fresh_indices(
            sampling_config(cfg, target), target, len(data[target].train_dataset)
        )
        assert recalculated == indices[target]
        x = torch.stack([data[target].train_dataset[i][0] for i in indices[target]["calibration"]])
        for name, factor in [("expert", 1.0), *[(f"beta_{b}", b) for b in cfg["betas"]]]:
            reference = (
                models[target]
                if name == "expert"
                else rescale(models[target], models["anchor"], factor)
            )
            subset = [0, 17, 31, 47]
            replay = finite_differences(
                reference, x, [records[i] for i in subset], cfg["finite_difference_step"], factor
            )
            assert torch.equal(replay, derivatives[name][:, :, subset])
            fd_columns += len(subset)
            if name != "expert":
                del reference
        attack_images = torch.stack(
            [data[target].train_dataset[i][0] for i in attack_indices[target]]
        )
        inverse, _ = head_inverse(heads[target], [h for t, h in heads.items() if t != target])
        for preset, release in [("original", models[target]), ("protected", predicted)]:
            for beta in cfg["betas"]:
                candidate = rescale(release, models["anchor"], beta)
                key = f"{target}/{preset}/beta_{beta}/proxy/{target}"
                sink = []
                evaluate(
                    candidate,
                    heads[target],
                    make_loader(data[target].test_dataset, list(range(128))),
                    "cuda",
                    prediction_sink=lambda p, _y: sink.append(p),
                )
                assert torch.equal(torch.cat(sink), raw[key]["prediction"][:128])
                prefixes += 1
                del candidate
            for attack in cfg["attacks"]:
                if attack == "direct":
                    restored = release
                elif attack == "aga":
                    restored = author_recover(release, models["anchor"], author)
                elif attack == "support_reset":
                    restored = reset(
                        release, models["anchor"], sites(cfg["layers"][target]), "public_support"
                    )
                elif attack in ("reset_attention", "reset_mlp"):
                    restored, _ = replace_components(
                        release,
                        models["anchor"],
                        range(12),
                        "attn" if attack == "reset_attention" else "mlp",
                    )
                else:
                    student = merge_all(dict(models, **{target: release}), root, cfg["coefficient"])
                    restored, fit, evidence = recover(
                        release, student, heads[target], inverse, attack_images, cfg["ridge_ratio"]
                    )
                    assert fit == read(out / "fitting.json")[f"{target}/{preset}"]
                    saved = torch.load(
                        out / f"artifacts/{target}_{preset}_readout.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    assert all(torch.equal(v, saved[k]) for k, v in evidence.items())
                    del student
                cases = []
                if attack != "readout":
                    cases.append(("standalone", restored, target))
                combined = merge_all(dict(models, **{target: restored}), root, cfg["coefficient"])
                cases.extend(("fusion", combined, t) for t in cfg["evaluation_tasks"])
                for group, model, task in cases:
                    sink = []
                    evaluate(
                        model,
                        heads[task],
                        make_loader(data[task].test_dataset, list(range(128))),
                        "cuda",
                        prediction_sink=lambda p, _y: sink.append(p),
                    )
                    key = f"{target}/{preset}/{attack}/{group}/{task}"
                    assert torch.equal(torch.cat(sink), raw[key]["prediction"][:128])
                    prefixes += 1
                del cases, combined, restored
        for task in cfg["evaluation_tasks"]:
            a = raw[f"{target}/original/support_reset/fusion/{task}"]["prediction"]
            b = raw[f"{target}/protected/support_reset/fusion/{task}"]["prediction"]
            assert torch.equal(a, b)
        checks.append(
            dict(
                target=target,
                candidate_reconstructed_exactly=True,
                eigenproblem_replayed=True,
                support_reset_identical=True,
            )
        )
        del geometry, predicted
    assert prefixes == len(rows)
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            rows=len(rows),
            official_labels_checked=True,
            prefix_replays=prefixes,
            finite_difference_columns_replayed=fd_columns,
            all_four_readout_fits_replayed=True,
            checks=checks,
            validator_sha256=digest(__file__),
            results_sha256=digest(out / "results.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_subspace.json")
    parser.add_argument("--output", default="outputs/2026-10-11/path_subspace")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--reuse-construction",
        help="Reuse an immutable partial constructor with exactly matching settings and indices",
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = parser.parse_args()
    (validate if args.validate else run)(args)
