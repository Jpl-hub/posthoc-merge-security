"""Task-complement edits with an explicit original-AGA row matching objective.

One fixed construction; no gradients, response fitting, partner access or sweep.
PCA tails need not be task-null. Utilities and merging effects are measured.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import evaluate
from eight_task_benchmark import author_recover, digest, merge_all
from fine_subspace import fresh_indices
from functional_contrast import features
from local_erasure import cache
from mechanisms import load_author
from operator_diagnostics import paired_counts
from path_subspace import context, read
from query_partner_cost import runtime, truth
from subspace_probe import collect_moments, fit_basis, make_loader, save_json, site_parameters
from torchvision.datasets import MNIST


def public_reset(model, anchor, sites):
    result = copy.deepcopy(model).cpu().eval()
    for site in sites:
        w, b = site_parameters(result, site)
        w0, b0 = site_parameters(anchor, site)
        w.copy_(w0)
        b.copy_(b0)
    return result


@torch.no_grad()
def construct(original, anchor, moments, cfg):
    decoy, coherent = [copy.deepcopy(original).cpu().eval() for _ in range(2)]
    certificates, rows = {}, []
    for site in cfg["sites"]:
        basis, stats = fit_basis(moments[site], cfg["task_energy"])
        p = basis["u"].double().cuda()
        block = original.model.visual.transformer.resblocks[int(site.split(":")[1])]
        null = block.ln_2.weight.double().cuda().reciprocal()
        null /= null.norm()
        residual = null - p @ (p.T @ null)
        if float(residual.norm()) > 1e-9:
            p = torch.cat([p, (residual / residual.norm())[:, None]], dim=1)
        q = torch.eye(p.shape[0], dtype=torch.float64, device="cuda") - p @ p.T
        w, _ = site_parameters(original, site)
        a, _ = site_parameters(anchor, site)
        w, a = w.double().cuda(), a.double().cuda()
        # Match the author's epsilon-regularized cosine row normalization.
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        n = len(a)
        assert n % 2 == 0
        permutation = (torch.arange(n, device="cuda") + n // 2) % n
        z = a[permutation] @ q
        constant = w @ a.T
        slope = z @ a.T
        idx = torch.arange(n, device="cuda")
        intercept = constant[idx, permutation, None] - constant
        denominator = slope[idx, permutation, None] - slope
        margin = cfg["row_margin_fraction"] * w.norm(dim=1)
        required = margin[:, None] - intercept
        denominator[idx, permutation] = 1
        required[idx, permutation] = 0
        positive = denominator > cfg["slope_tolerance"]
        negative = denominator < -cfg["slope_tolerance"]
        lower = torch.where(positive, required / denominator, 0).amax(1).clamp_min(0)
        upper = torch.where(negative, required / denominator, torch.inf).amin(1)
        feasible = lower < upper
        feasible &= ~((~positive & ~negative) & (required > 0)).any(1)
        # Keep whole disjoint swap cycles so selected destinations remain bijective.
        feasible &= feasible[permutation]
        amplitude = lower * (1 + cfg["relative_margin_slack"]) + 1e-10
        feasible &= amplitude < upper
        feasible &= feasible[permutation]
        amplitude = torch.where(feasible, amplitude, 0)
        mean = moments[site]["mean"].double().cuda()
        delta = amplitude[:, None] * z
        control_delta = amplitude[:, None] * (a @ q)
        evidence = dict(
            p=p.cpu(),
            q=q.cpu(),
            mean=mean.cpu(),
            covariance=moments[site]["covariance"],
            permutation=permutation.cpu(),
            amplitude=amplitude.cpu(),
            feasible=feasible.cpu(),
            lower=lower.cpu(),
            upper=upper.cpu(),
            decoy_delta=delta.cpu(),
            coherent_delta=control_delta.cpu(),
            margin=margin.cpu(),
        )
        for model, update in [(decoy, delta), (coherent, control_delta)]:
            weight, bias = site_parameters(model, site)
            weight.copy_((weight.double().cuda() + update).float().cpu())
            bias.copy_((bias.double().cuda() - update @ mean).float().cpu())
        score = (w + delta) @ a.T
        difference = score[idx, permutation, None] - score
        difference[idx, permutation] = torch.inf
        guaranteed = (difference.amin(1) >= margin * (1 - 1e-6)) | ~feasible
        assert bool(guaranteed.all())
        covariance = moments[site]["covariance"].double().cuda()
        response_energy = float((w @ covariance * w).sum())
        row = dict(
            site=site,
            **stats,
            protected_span_rank=p.shape[1],
            actual_complement_rank=int(round(float(q.trace()))),
            selected_rows=int(feasible.sum()),
            hidden_rows=n,
            minimum_selected_margin=float(difference.amin(1)[feasible].min())
            if bool(feasible.any())
            else None,
            amplitude_min=float(amplitude[feasible].min()) if bool(feasible.any()) else None,
            amplitude_max=float(amplitude.max()),
            decoy_weight_relative_change=float(delta.norm() / w.norm()),
            coherent_weight_relative_change=float(control_delta.norm() / w.norm()),
            calibration_preactivation_relative_rms=float(
                ((delta @ covariance * delta).sum().clamp_min(0) / response_energy).sqrt()
            ),
            task_span_delta_error=float((delta @ p).norm() / delta.norm().clamp_min(1e-30)),
            exact_ln_null_component=float((delta @ null).norm() / delta.norm().clamp_min(1e-30)),
        )
        assert row["task_span_delta_error"] < 1e-9
        assert row["exact_ln_null_component"] < 1e-9
        certificates[site] = evidence
        rows.append(row)
        print("CONSTRUCT", row, flush=True)
    return decoy, coherent, certificates, rows


@torch.no_grad()
def matching(model, anchor, author, sites, certificate):
    evidence, report = {}, []
    for site in sites:
        w, _ = site_parameters(model, site)
        a, _ = site_parameters(anchor, site)
        restore = author["solve_permutation"](w, a, site).cpu()
        assignment = torch.argsort(restore)
        desired, selected = certificate[site]["permutation"], certificate[site]["feasible"]
        evidence[site] = restore
        report.append(
            dict(
                site=site,
                changed_rows=int(restore.ne(torch.arange(len(restore))).sum()),
                intended_decoy_rows=int(selected.sum()),
                intended_decoy_matches=int(assignment[selected].eq(desired[selected]).sum()),
            )
        )
    return evidence, report


@torch.no_grad()
def run(args):
    runtime()
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    root, loader, fusion, models = context(cfg)
    original = loader(str(root / "MNIST/finetuned.pt")).cpu().eval()
    anchor = models["anchor"]
    train = MNIST("data", train=True, download=False, transform=original.val_preprocess)
    indices = fresh_indices(cfg, "MNIST", len(train))
    save_json(out / "training_indices.json", indices)
    moments = collect_moments(
        original,
        make_loader(train, indices["calibration"], 32),
        cfg["sites"],
        token_index=0 if cfg.get("token_scope") == "final_cls" else None,
    )
    decoy, coherent, certificate, construction = construct(original, anchor, moments, cfg)
    releases = {"original": original, "coherent": coherent, "decoy": decoy}
    freeze = {}
    for name in ("decoy", "coherent"):
        edits = {
            k: w.clone()
            for k, w in releases[name].state_dict().items()
            if not torch.equal(w, original.state_dict()[k])
        }
        allowed = {
            f"model.visual.transformer.resblocks.{s.split(':')[1]}.mlp.c_fc.{part}"
            for s in cfg["sites"]
            for part in ("weight", "bias")
        }
        assert set(edits) <= allowed
        if cfg.get("setup_parent"):
            prior_edits = torch.load(
                Path(cfg["setup_parent"]) / f"artifacts/{name}_edits.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert edits.keys() == prior_edits.keys()
            assert all(torch.equal(w, prior_edits[k]) for k, w in edits.items())
        torch.save(edits, artifacts / f"{name}_edits.pt")
        freeze[name] = dict(sha256=digest(artifacts / f"{name}_edits.pt"), edited_keys=list(edits))
    torch.save(certificate, artifacts / "certificate.pt")
    save_json(out / "construction.json", construction)
    save_json(
        out / "release_freeze.json",
        dict(
            releases=freeze,
            future_partner_weights_loaded=False,
            future_partner_data_used=False,
            labels_used=False,
            optimizer_steps=0,
            response_fits=0,
            setup_parent=cfg.get("setup_parent"),
            same_release_as_setup_failure=bool(cfg.get("setup_parent")),
        ),
    )
    author, author_sha = load_author("external/secure-merge-attack/utils_attack_B32.py", True)
    assert author_sha == read("configs/sources.json")["AGA"]["sha256"]
    diagnostics, permutations = {}, {}
    confirm = torch.stack([train[i][0] for i in indices["confirmation"]])
    original.cuda()
    base_features = features(original, confirm)
    if cfg.get("token_scope") == "final_cls":
        assert cfg["sites"] == ["FC:11"]
        assert len(original.model.visual.transformer.resblocks) == 12
        same_batch_reference = features(original, confirm[:16])

        def zero_final_patches(_module, _args, value):
            modified = value.clone()
            modified[1:] = 0
            return modified

        handle = original.model.visual.transformer.resblocks[11].mlp.register_forward_hook(
            zero_final_patches
        )
        try:
            erased = features(original, confirm[:16])
        finally:
            handle.remove()
        assert torch.equal(erased, same_batch_reference)
        diagnostics["causal_check"] = dict(
            final_mlp_patch_outputs_zeroed=True,
            feature_bitwise_equal=True,
            batch_size=16,
            unmatched_batch_max_error=float((erased - base_features[:16]).abs().max()),
            scope="16 disjoint training confirmation images; structural claim limited to final MLP",
        )
    original.cpu()
    for name, model in releases.items():
        permutations[name], report = matching(model, anchor, author, cfg["sites"], certificate)
        model.cuda()
        values = features(model, confirm)
        model.cpu()
        diagnostics[name] = dict(
            matching=report,
            confirmation_feature_rms=float(
                (values - base_features).double().square().sum().div(len(confirm)).sqrt()
            ),
        )
        # MLP-only AGA must preserve the nonlinear standalone computation.
        permuted = author["attack_vit_mlp_only"](copy.deepcopy(model), anchor).cpu().eval()
        permuted.cuda()
        replay = features(permuted, confirm[:16])
        permuted.cpu()
        diagnostics[name]["mlp_only_feature_max_error"] = float((replay - values[:16]).abs().max())
        diagnostics[name]["mlp_only_fp32_equivalence_within_2e_5"] = (
            diagnostics[name]["mlp_only_feature_max_error"] < 2e-5
        )
        del permuted
    torch.save(permutations, artifacts / "assignments.pt")
    save_json(out / "diagnostics.json", diagnostics)
    # All release parameters are fixed before loading partners, heads or test sets.
    for task in fusion["tasks"]:
        models[task] = loader(str(root / task / "finetuned.pt")).cpu().eval()
    expected = {
        r["path"].split("/", 1)[1]: r["sha256"] for r in read("configs/checkpoints.json")["files"]
    }
    assert {str(p.relative_to(root)): digest(p) for p in root.rglob("*.pt")} == expected
    assert fusion["coefficient"] == cfg["coefficient"] == 0.2
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: loader(str(root / f"head_{t}.pt")) for t in data}
    tests = {t: cache(d) for t, d in data.items()}
    assert all(len(d.test_dataset) == fusion["expected_test_counts"][t] for t, d in data.items())
    states, rows, raw = {}, [], {}
    prior = None
    if cfg.get("baseline_parent"):
        parent = Path(cfg["baseline_parent"])
        prior_meta = read(parent / "metadata.json")
        assert read(parent / "validation.json")["status"] == "passed"
        assert prior_meta["checkpoint_sha256"] == expected
        assert (
            digest(parent / "artifacts/predictions.pt")
            == prior_meta["raw_sha256"]["predictions.pt"]
        )
        prior = torch.load(
            parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
        )
    for name, model in releases.items():
        for attack in ("direct", "aga"):
            states[f"{name}/{attack}"] = (
                model if attack == "direct" else author_recover(model, anchor, author)
            )
    # Public reset is checked only after the core candidate has actually been built.
    resets = {name: public_reset(model, anchor, cfg["sites"]) for name, model in releases.items()}
    assert all(
        all(torch.equal(w, resets["original"].state_dict()[k]) for k, w in m.state_dict().items())
        for m in resets.values()
    )
    states["shared/public_reset"] = resets["original"]
    for key, model in states.items():
        merged = merge_all(dict(models, MNIST=model), root, cfg["coefficient"])
        for group, task, measured in [
            ("standalone", "MNIST", model),
            *[("fusion", t, merged) for t in cfg["evaluation_tasks"]],
        ]:
            pred, labels = [], []
            name = f"{key}/{group}/{task}"
            reused = prior is not None and key.startswith("original/")
            if reused:
                raw[name] = prior[name]
                y = truth(data[task], task)
                assert torch.equal(raw[name]["labels"], y)
                correct = int(raw[name]["prediction"].eq(y).sum())
                metric = dict(correct=correct, n=len(y), top1=correct / len(y))
            else:
                metric = evaluate(
                    measured,
                    heads[task],
                    tests[task],
                    "cuda",
                    prediction_sink=lambda p, y: (pred.append(p), labels.append(y)),
                )
                raw[name] = dict(prediction=torch.cat(pred), labels=torch.cat(labels))
            rows.append(
                dict(
                    key=name,
                    preset=key.split("/")[0],
                    attack=key.split("/")[1],
                    group=group,
                    dataset=task,
                    reused=reused,
                    baseline_parent=cfg.get("baseline_parent") if reused else None,
                    **metric,
                )
            )
            save_json(out / "results.json", rows)
            print("COMPLEMENT_MATCHING", name, metric["correct"], metric["n"], flush=True)
        del merged
    for row in rows:
        attack = row["attack"] if row["attack"] != "public_reset" else "direct"
        base = raw[f"original/{attack}/{row['group']}/{row['dataset']}"]
        item = raw[row["key"]]
        row["matched_original_change"] = paired_counts(
            base["prediction"].eq(item["labels"]), item["prediction"].eq(item["labels"])
        )
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            source_manifest=read(args.manifest)["source_manifest"],
            config_sha256=digest(args.config),
            checkpoint_sha256=expected,
            author_sha256=author_sha,
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            calibration_samples=len(indices["calibration"]),
            confirmation_samples=len(confirm),
            fresh_full_evaluations=sum(not r["reused"] for r in rows),
            reused_baselines=sum(r["reused"] for r in rows),
            evaluation_tasks=cfg["evaluation_tasks"],
            eight_experts_participate=True,
            eight_task_mean=False,
            defense_success_claim=False,
            broader_adaptive_attacks_unrun=True,
        ),
    )


@torch.no_grad()
def validate(args):
    runtime()
    cfg, out = read(args.config), Path(args.output)
    meta = read(out / "metadata.json")
    assert meta["completed"] and cfg == read(out / "protocol.json")
    assert digest(args.config) == meta["config_sha256"]
    for path, sha in meta["source_manifest"].items():
        assert digest(path) == sha, path
    for name, sha in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == sha, name
    root, loader, fusion, models = context(cfg, all_tasks=True)
    for name, sha in meta["checkpoint_sha256"].items():
        assert digest(root / name) == sha, name
    original, anchor = models["MNIST"], models["anchor"]
    releases = {"original": original}
    certificate = torch.load(out / "artifacts/certificate.pt", weights_only=True)
    assignment = torch.load(out / "artifacts/assignments.pt", weights_only=True)
    from scipy.optimize import linear_sum_assignment

    algebra = []
    for name in ("decoy", "coherent"):
        model = copy.deepcopy(original).cpu().eval()
        edits = torch.load(out / f"artifacts/{name}_edits.pt", weights_only=True)
        model.load_state_dict(dict(model.state_dict(), **edits))
        for site, cert in certificate.items():
            w, b = site_parameters(original, site)
            wr, br = site_parameters(model, site)
            update = cert[f"{name}_delta"]
            assert torch.equal(wr, (w.double() + update).float())
            assert torch.allclose(
                br, (b.double() - update @ cert["mean"]).float(), atol=2e-7, rtol=0
            )
            actual = wr.double() - w.double()
            span_error = float((actual @ cert["p"]).norm() / actual.norm().clamp_min(1e-30))
            assert span_error < 2e-6
            assert torch.allclose(cert["q"] @ cert["q"], cert["q"], atol=1e-10, rtol=0)
            a, _ = site_parameters(anchor, site)
            wp, wb = wr.numpy(), a.numpy()
            wp = wp / (np.linalg.norm(wp, axis=1, keepdims=True) + 1e-8)
            wb = wb / (np.linalg.norm(wb, axis=1, keepdims=True) + 1e-8)
            cost = -(wp @ wb.T)
            ri, ci = linear_sum_assignment(cost)
            independent = np.zeros(len(ri), dtype=np.int64)
            independent[ci] = ri
            assert np.array_equal(independent, assignment[name][site].numpy())
            selected_matches = None
            if name == "decoy":
                selected = cert["feasible"].numpy()
                preferred = np.argsort(independent)[selected]
                selected_matches = int(np.sum(preferred == cert["permutation"].numpy()[selected]))
                if selected.all():
                    assert selected_matches == len(selected)
                # Independent FP64 bound calculation, without the construction solver.
                normalized = a.double().numpy()
                normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-8
                pi = cert["permutation"].numpy()
                z = normalized[pi] @ cert["q"].numpy()
                constant = w.double().numpy() @ normalized.T
                slope = z @ normalized.T
                ir = np.arange(len(pi))
                coefficient = slope[ir, pi, None] - slope
                required = cert["margin"].numpy()[:, None] - (constant[ir, pi, None] - constant)
                coefficient[ir, pi], required[ir, pi] = 1, 0
                denominator = np.where(coefficient > cfg["slope_tolerance"], coefficient, np.inf)
                lower = np.maximum((required / denominator).max(1), 0)
                expected_amplitude = lower * (1 + cfg["relative_margin_slack"]) + 1e-10
                assert np.allclose(
                    cert["amplitude"].numpy()[selected],
                    expected_amplitude[selected],
                    atol=1e-8,
                    rtol=1e-8,
                )
            algebra.append(
                dict(
                    preset=name,
                    site=site,
                    fp32_span_relative_error=span_error,
                    assignment_independently_replayed=True,
                    intended_decoy_matches=selected_matches,
                )
            )
        releases[name] = model
    author, sha = load_author("external/secure-merge-attack/utils_attack_B32.py", True)
    assert sha == meta["author_sha256"]
    states = {
        f"{name}/{attack}": model if attack == "direct" else author_recover(model, anchor, author)
        for name, model in releases.items()
        for attack in ("direct", "aga")
    }
    states["shared/public_reset"] = public_reset(original, anchor, cfg["sites"])
    for name, model in releases.items():
        reset = public_reset(model, anchor, cfg["sites"])
        assert all(
            torch.equal(w, states["shared/public_reset"].state_dict()[k])
            for k, w in reset.state_dict().items()
        )
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, anchor.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    heads = {t: loader(str(root / f"head_{t}.pt")) for t in data}
    raw = torch.load(out / "artifacts/predictions.pt", weights_only=True)
    rows = read(out / "results.json")
    prefixes = 0
    for key, model in states.items():
        merged = merge_all(dict(models, MNIST=model), root, cfg["coefficient"])
        for row in [r for r in rows if r["key"].startswith(key + "/")]:
            item = raw[row["key"]]
            labels = truth(data[row["dataset"]], row["dataset"])
            assert torch.equal(item["labels"], labels)
            assert row["correct"] == int(item["prediction"].eq(labels).sum())
            assert row["n"] == len(labels) and row["top1"] == row["correct"] / len(labels)
            attack = row["attack"] if row["attack"] != "public_reset" else "direct"
            base = raw[f"original/{attack}/{row['group']}/{row['dataset']}"]
            assert row["matched_original_change"] == paired_counts(
                base["prediction"].eq(labels), item["prediction"].eq(labels)
            )
            pred = []
            evaluate(
                model if row["group"] == "standalone" else merged,
                heads[row["dataset"]],
                make_loader(data[row["dataset"]].test_dataset, list(range(32)), 32),
                "cuda",
                prediction_sink=lambda p, _y: pred.append(p),
            )
            assert torch.equal(torch.cat(pred), item["prediction"][:32]), row["key"]
            prefixes += 1
        del merged
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            rows=len(rows),
            full_counts_and_official_labels_checked=True,
            prefix_replays=prefixes,
            release_reconstructed=True,
            algebra=algebra,
            public_reset_identical=True,
            source_and_raw_fingerprints_checked=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    (validate if arguments.validate else run)(arguments)
