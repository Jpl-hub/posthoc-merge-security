"""Test bounded complement edits against public-anchor cleanup before final tests.

Reuses the parent calibration/development split and the author evaluation stack.
An unsuccessful development gate produces no official-test measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, merge, recover
from mechanisms import load_author
from subspace_diagnostics import replace_norm_outliers, visual_linear_weights
from subspace_probe import (
    clean_complement,
    collect_moments,
    complement_power,
    evaluate,
    fit_basis,
    make_loader,
    save_json,
    site_parameters,
)
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def angular_cleanup(model, anchor, threshold):
    """Scan public matrices only; no data, defender sites, PCA or expert weights."""
    result = copy.deepcopy(model).cpu().eval()
    state, base = result.state_dict(), anchor.state_dict()
    inspected, replaced = [], []
    for key in visual_linear_weights(result):
        value = state[key]
        w, a = value.double(), base[key].double()
        cosine = float((w * a).sum() / (w.norm() * a.norm()).clamp_min(1e-12))
        inspected.append(dict(parameter=key, cosine=cosine))
        if cosine >= threshold:
            continue
        state[key] = base[key].clone()
        bias_key = key.removesuffix("weight") + "bias"
        if bias_key in state:
            state[bias_key] = base[bias_key].clone()
        replaced.append(dict(parameter=key, cosine=cosine, bias_replaced=bias_key in state))
    result.load_state_dict(state, strict=True)
    return result, dict(inspected=inspected, replaced=replaced)


@torch.no_grad()
def bounded_edit(original, sites, bases, scale):
    result = copy.deepcopy(original).cpu().eval()
    checks = []
    for site in sites:
        w, b = site_parameters(result, site)
        u = bases[site]["u"].cuda()
        old = w.double().cuda()
        complement = old - (old @ u) @ u.T
        delta = -scale * complement
        w.add_(delta.float().cpu())
        b.sub_((delta @ bases[site]["mean"].cuda()).float().cpu())
        actual = w.double().cuda() - old
        ratio = float(w.double().norm() / old.norm().cpu())
        leakage = float((actual @ u).norm() / actual.norm().clamp_min(1e-12))
        assert leakage < 1e-5 and torch.isfinite(w).all() and torch.isfinite(b).all()
        assert ratio <= 1 + 1e-6
        if scale == 2:
            assert abs(ratio - 1) < 1e-6
        checks.append(dict(site=site, weight_norm_ratio=ratio, main_space_leakage=leakage))
    return result, checks


@torch.no_grad()
def replace_site(original, anchor, site):
    model = copy.deepcopy(original).cpu().eval()
    w, b = site_parameters(model, site)
    aw, ab = site_parameters(anchor, site)
    w.copy_(aw)
    b.copy_(ab)
    return model


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    protocol_bytes = Path(args.config).read_bytes()
    config = json.loads(protocol_bytes)
    protected_task = config.get("protected_task", "MNIST")
    assert protected_task in {"MNIST", "SVHN"}
    other_task = "SVHN" if protected_task == "MNIST" else "MNIST"
    (out / "protocol.json").write_bytes(protocol_bytes)
    parent = Path(config["parent"])
    parent_bytes = (parent / "results.json").read_bytes()
    index_bytes = (parent / "training_indices.json").read_bytes()
    indices = json.loads(index_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / f"{protected_task}/finetuned.pt"))
    second = load(str(root / f"{other_task}/finetuned.pt"))
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    training = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    development = {
        task: make_loader(dataset, indices[task]["development"])
        for task, dataset in training.items()
    }
    calibration = make_loader(
        training[protected_task], indices[protected_task]["calibration"], batch_size=32
    )
    attack_loader = make_loader(
        training[protected_task], indices[protected_task]["attack"], batch_size=32
    )
    all_sites = [f"{target}:{layer}" for layer in range(12) for target in ["V", "FC"]]
    rows = [
        dict(**row, origin="parent_measurement")
        for row in json.loads(parent_bytes)
        if row["split"] == "development"
        and row["preset"] == "unprotected"
        and (row["group"] == "direct" or row["dataset"] == protected_task)
    ]
    reused_measurements = len(rows)
    assert reused_measurements == (3 if protected_task == "MNIST" else 2)
    scans, profile, screen = {}, [], []

    def record(preset, group, model, loaders=development, split="development"):
        accuracy = {}
        for task, loader in loaders.items():
            if group.endswith("standalone") and task != protected_task:
                continue
            row = dict(
                preset=preset,
                group=group,
                dataset=task,
                split=split,
                origin="new_evaluation",
                coefficient=None if group.endswith("standalone") else config["coefficient"],
                **evaluate(model, heads[task], loader, "cuda"),
            )
            accuracy[task] = row["top1"]
            rows.append(row)
            save_json(out / "results.json", rows)
            print("BOUNDED_RESULT", json.dumps(row), flush=True)
        return accuracy

    baseline = {r["dataset"]: r["top1"] for r in rows if r["group"] == "direct"}
    original_utility = (
        next(r["top1"] for r in rows if r["group"] == "standalone")
        if protected_task == "MNIST"
        else record("unprotected", "standalone", original)[protected_task]
    )
    control, scans["unprotected"] = angular_cleanup(
        original, anchor, config["angular_cosine_threshold"]
    )
    assert not scans["unprotected"]["replaced"]
    assert all(
        torch.equal(value, control.state_dict()[key])
        for key, value in original.state_dict().items()
    )
    del control
    record("pretrained", "standalone", anchor)
    record("pretrained", "direct", merge(anchor, anchor, second, config["coefficient"]))
    for site in all_sites:
        changed = replace_site(original, anchor, site)
        utility = record(f"replace_{site}", "standalone", changed)[protected_task]
        combined = record(
            f"replace_{site}", "direct", merge(anchor, changed, second, config["coefficient"])
        )
        profile.append(
            dict(
                site=site,
                kind="single_projection_matrix",
                standalone_loss_pp=100 * (original_utility - utility),
                merged_MNIST=combined["MNIST"],
                merged_SVHN=combined["SVHN"],
            )
        )
        save_json(out / "replacement_profile.json", profile)
        del changed
    for layer in range(12):
        changed = copy.deepcopy(original).cpu().eval()
        changed.model.visual.transformer.resblocks[layer].load_state_dict(
            anchor.model.visual.transformer.resblocks[layer].state_dict(), strict=True
        )
        utility = record(f"replace_block_{layer}", "standalone", changed)[protected_task]
        profile.append(
            dict(
                site=f"block:{layer}",
                kind="entire_block",
                standalone_loss_pp=100 * (original_utility - utility),
            )
        )
        save_json(out / "replacement_profile.json", profile)
        del changed
    moments = collect_moments(original, calibration, all_sites)
    basis_by_energy, stats = {}, {}
    for energy in config["retained_energies"]:
        basis_by_energy[str(energy)], stats[str(energy)] = {}, {}
        for site in all_sites:
            basis, diagnostic = fit_basis(moments[site], energy)
            basis_by_energy[str(energy)][site] = basis
            stats[str(energy)][site] = diagnostic
    save_json(out / "calibration_subspaces.json", stats)
    torch.save(basis_by_energy, artifacts / "defender_bases.pt")
    selected = None
    for energy, (site_group, sites), scale in itertools.product(
        config["retained_energies"], config["site_groups"].items(), config["edit_scales"]
    ):
        name = f"{site_group}_e{energy:g}_s{scale:g}"
        changed, check = bounded_edit(original, sites, basis_by_energy[str(energy)], scale)
        utility = record(name, "standalone", changed)[protected_task]
        direct = record(name, "direct", merge(anchor, changed, second, config["coefficient"]))
        cleaned, scans[name] = angular_cleanup(changed, anchor, config["angular_cosine_threshold"])
        cleaned_utility = record(name, "angular_standalone", cleaned)[protected_task]
        attacked = record(name, "angular", merge(anchor, cleaned, second, config["coefficient"]))
        norm_cleaned, norm_scan = replace_norm_outliers(
            changed, anchor, config["norm_ratio_threshold"]
        )
        utility_loss = 100 * (original_utility - utility)
        drops = {
            group: {
                protected_task: 100 * (baseline[protected_task] - acc[protected_task]),
                "mean": 50 * (sum(baseline.values()) - sum(acc.values())),
            }
            for group, acc in [("direct", direct), ("angular", attacked)]
        }
        feasible = utility_loss <= config["utility_budget_pp"] + 1e-9
        passes = feasible and all(
            d[protected_task] >= config["minimum_protected_task_drop_pp"] - 1e-9
            and d["mean"] >= config["minimum_mean_drop_pp"] - 1e-9
            for d in drops.values()
        )
        item = dict(
            name=name,
            energy=energy,
            site_group=site_group,
            scale=scale,
            standalone_loss_pp=utility_loss,
            utility_feasible=feasible,
            recovery_gate_passed=passes,
            mean_direct=sum(direct.values()) / 2,
            mean_angular=sum(attacked.values()) / 2,
            worst_case_mean=max(sum(direct.values()), sum(attacked.values())) / 2,
            cleaned_standalone=cleaned_utility,
            drops_pp=drops,
            checks=check,
            norm_scan=norm_scan,
        )
        screen.append(item)
        save_json(out / "development_screen.json", screen)
        save_json(out / "angular_scans.json", scans)
        if passes and (
            selected is None
            or (item["worst_case_mean"], utility_loss)
            < (selected["worst_case_mean"], selected["standalone_loss_pp"])
        ):
            selected = item
            torch.save(changed.state_dict(), artifacts / "selected_state.pt")
        del changed, cleaned, norm_cleaned
    selection = dict(
        selected=selected,
        utility_feasible_count=sum(s["utility_feasible"] for s in screen),
        recovery_gate_passed_count=sum(s["recovery_gate_passed"] for s in screen),
        test_usage="none_before_selection",
        test_run=selected is not None,
    )
    save_json(out / "selection.json", selection)
    if selected:
        testing = {
            "MNIST": make_loader(MNIST(args.data, train=False, transform=original.val_preprocess)),
            "SVHN": make_loader(SVHN(args.data, split="test", transform=original.val_preprocess)),
        }
        released = copy.deepcopy(original).cpu().eval()
        released.load_state_dict(torch.load(artifacts / "selected_state.pt"), strict=True)
        for name, model in [("unprotected", original), (selected["name"], released)]:
            record(name, "standalone", model, testing, "official_test")
            record(
                name,
                "direct",
                merge(anchor, model, second, config["coefficient"]),
                testing,
                "official_test",
            )
            angular, _ = angular_cleanup(model, anchor, config["angular_cosine_threshold"])
            attack_dir = out / f"{name}_AGA"
            attack_dir.mkdir()
            aga = recover(model, anchor, author, attack_dir)
            attacked_moments = collect_moments(model, attack_loader, all_sites)
            completion = copy.deepcopy(model).cpu().eval()
            for site in all_sites:
                basis, _ = fit_basis(attacked_moments[site], selected["energy"])
                completion = clean_complement(completion, anchor, site, basis, "completion")
            for group, changed in [("angular", angular), ("AGA", aga), ("completion", completion)]:
                record(name, f"{group}_standalone", changed, testing, "official_test")
                record(
                    name,
                    group,
                    merge(anchor, changed, second, config["coefficient"]),
                    testing,
                    "official_test",
                )
            del angular, aga, completion
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            config_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
            experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_results_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            parent_indices_sha256=hashlib.sha256(index_bytes).hexdigest(),
            author_sha256=author_sha,
            training="none",
            reused_measurements=reused_measurements,
            protected_task=protected_task,
            new_evaluations=sum(r["origin"] == "new_evaluation" for r in rows),
            official_test_rows=sum(r["split"] == "official_test" for r in rows),
        ),
    )
    print("BOUNDED_COMPLETE", flush=True)


def run_scope_audit(args):
    """Compare corrected attack endpoints with recorded replacement decisions."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.checkpoints)
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt"))
    config = json.loads(Path(args.config).read_bytes())
    parent = Path(config["parent"]).parent
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, controls = [], {}
    start = time.perf_counter()
    for task, directory in [("MNIST", "bounded_complement"), ("SVHN", "bounded_svhn")]:
        source = parent / directory
        protocol = json.loads((source / "protocol.json").read_bytes())
        recorded = json.loads((source / "angular_scans.json").read_bytes())
        screen = json.loads((source / "development_screen.json").read_bytes())
        bases = torch.load(source / "artifacts/defender_bases.pt")
        original = load(str(root / f"{task}/finetuned.pt"))
        _, control_scan = angular_cleanup(original, anchor, protocol["angular_cosine_threshold"])
        assert not control_scan["replaced"]
        controls[task] = control_scan
        for preset in screen:
            model, _ = bounded_edit(
                original,
                protocol["site_groups"][preset["site_group"]],
                bases[str(preset["energy"])],
                preset["scale"],
            )
            expected = copy.deepcopy(model).cpu().eval()
            state, reference = expected.state_dict(), anchor.state_dict()
            for item in recorded[preset["name"]]["replaced"]:
                key = item["parameter"]
                state[key] = reference[key].clone()
                bias = key.removesuffix("weight") + "bias"
                if bias in state:
                    state[bias] = reference[bias].clone()
            expected.load_state_dict(state, strict=True)
            corrected, scan = angular_cleanup(model, anchor, protocol["angular_cosine_threshold"])
            same_set = {r["parameter"] for r in scan["replaced"]} == {
                r["parameter"] for r in recorded[preset["name"]]["replaced"]
            }
            same_state = all(
                torch.equal(value, corrected.state_dict()[key])
                for key, value in expected.state_dict().items()
            )
            norm_corrected, norm_scan = replace_norm_outliers(
                model, anchor, protocol["norm_ratio_threshold"]
            )
            same_norm = norm_scan["replaced"] == preset["norm_scan"]["replaced"]
            assert same_set and same_state and same_norm
            assert all(
                torch.equal(value, norm_corrected.state_dict()[key])
                for key, value in model.state_dict().items()
            )
            rows.append(
                dict(
                    protected_task=task,
                    preset=preset["name"],
                    inspected=len(scan["inspected"]),
                    angular_replacement_set_equal=same_set,
                    angular_state_equal=same_state,
                    norm_replacement_set_equal=same_norm,
                    final_projection=next(
                        r for r in scan["inspected"] if r["parameter"] == "model.visual.proj"
                    ),
                )
            )
            save_json(out / "results.json", rows)
            del model, expected, corrected, norm_corrected
    save_json(out / "controls.json", controls)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            accuracy_evaluations=0,
            training="none",
            test_usage="none",
            candidates=len(rows),
            old_scope="48 visual block matrices plus unused text embedding; final visual projection omitted",
            corrected_scope="49 active visual matrices; text embedding excluded; final visual projection included",
            result="Recorded candidate cleanup endpoints unchanged under corrected scope",
            experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scanner_source_sha256=hashlib.sha256(
                Path(__file__).with_name("subspace_diagnostics.py").read_bytes()
            ).hexdigest(),
        ),
    )
    print("SCOPE_AUDIT_COMPLETE", flush=True)


def measure_distribution_shift(args):
    """Compare original/merged inputs in each task's own frozen PCA basis."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    (out / "protocol.json").write_bytes(config_bytes)
    parent = Path(config["parent"])
    index_bytes = (parent / "training_indices.json").read_bytes()
    indices = json.loads(index_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    checkpoints = Path(args.checkpoints)
    anchor = load(str(checkpoints / "zeroshot.pt"))
    experts = {task: load(str(checkpoints / f"{task}/finetuned.pt")) for task in ["MNIST", "SVHN"]}
    combined = merge(anchor, experts["MNIST"], experts["SVHN"], config["coefficient"])
    sites = [f"{target}:{layer}" for layer in range(12) for target in ["V", "FC"]]
    sources, rows = {}, []
    for task in ["MNIST", "SVHN"]:
        source = parent.parent / ("bounded_complement" if task == "MNIST" else "bounded_svhn")
        source_bytes = (source / "metadata.json").read_bytes()
        source_meta = json.loads(source_bytes)
        assert source_meta["completed"]
        bases = torch.load(source / "artifacts/defender_bases.pt")
        dataset = (
            MNIST(args.data, train=True, transform=experts[task].val_preprocess)
            if task == "MNIST"
            else SVHN(args.data, split="train", transform=experts[task].val_preprocess)
        )
        loader = make_loader(dataset, indices[task]["calibration"], batch_size=32)
        original_moments = collect_moments(experts[task], loader, sites)
        merged_moments = collect_moments(combined, loader, sites)
        for energy, by_site in bases.items():
            for site, basis in by_site.items():
                rows.append(
                    dict(
                        protected_task=task,
                        site=site,
                        retained_energy=float(energy),
                        rank=basis["rank"],
                        tokens=original_moments[site]["tokens"],
                        expert=complement_power(original_moments[site], basis),
                        unprotected_merge=complement_power(merged_moments[site], basis),
                    )
                )
        sources[task] = dict(
            directory=str(source),
            source_metadata_sha256=hashlib.sha256(source_bytes).hexdigest(),
            source_config_sha256=source_meta["config_sha256"],
            basis_sha256=hashlib.sha256(
                (source / "artifacts/defender_bases.pt").read_bytes()
            ).hexdigest(),
        )
    save_json(out / "results.json", rows)
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            training="none",
            test_usage="none",
            accuracy_evaluations=0,
            samples_per_task=256,
            covariance_center="same frozen defender mean",
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            parent_indices_sha256=hashlib.sha256(index_bytes).hexdigest(),
            experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sources=sources,
        ),
    )
    print("DISTRIBUTION_SHIFT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bounded_complement.json")
    parser.add_argument("--output", default="outputs/2026-10-09/post_meeting/bounded_complement")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scope-audit", action="store_true", help="Audit corrected visual scan.")
    mode.add_argument(
        "--distribution-shift", action="store_true", help="Diagnose both frozen task subspaces."
    )
    args = parser.parse_args()
    (
        run_scope_audit
        if args.scope_audit
        else measure_distribution_shift
        if args.distribution_shift
        else run
    )(args)
