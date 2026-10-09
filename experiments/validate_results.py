"""Scientific acceptance checks over actual machine-generated outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def validate(path: Path):
    rows = json.loads((path / "mechanism_results.json").read_text())
    assert len(rows) == 56, len(rows)
    mlp = [r for r in rows if r["experiment"] == "mlp_permutation"]
    attention = [r for r in rows if r["experiment"] == "attention_qk_vo"]
    assert max(r["protected_output_error"] for r in mlp + attention) < 1e-5
    assert max(r["protected_score_error"] for r in attention) < 1e-5
    assert all(r["permutation_accuracy"] == 1 for r in mlp)
    assert all(r["recovered_weight_error"] == 0 for r in mlp)
    p1 = [r for r in rows if r["experiment"] == "column_space_invariance"]
    assert max(r["ideal_projection_error"] for r in p1) < 1e-8
    assert max(r["error_bound_ratio"] for r in p1) <= 1 + 1e-8
    p3 = [r for r in rows if r["stage"] == "P3"]
    assert all(r["permutation_accuracy"] == 1 for r in p3)
    assert all(r["protected_output_error"] < 1e-5 for r in p3)
    assert all(r["adaptive_output_error"] < r["recovered_output_error"] / 100 for r in p3)
    counterexample = next(r for r in rows if r["experiment"] == "cosine_margin_counterexample")
    assert counterexample["theorem_condition"]
    assert counterexample["cosine_recovery_accuracy"] == 0
    assert counterexample["euclidean_recovery_accuracy"] == 1
    checks = [
        "P0_function_preservation",
        "MLP_exact_recovery",
        "P1_projection_identity",
        "P1_error_bound",
        "P3_adaptive_matching",
        "cosine_vs_euclidean_counterexample",
    ]
    if (path / "clip_metadata.json").exists():
        meta = json.loads((path / "clip_metadata.json").read_text())
        real = json.loads((path / "clip_results.json").read_text())
        assert meta["protected_output_error"] < 1e-4
        expected_n = {"MNIST": 10000, "SVHN": 26032}
        expected_rows = 34 if meta.get("attack_all") else 22
        assert len(real) == expected_rows, len(real)
        assert all(r["n"] == expected_n[r["dataset"]] for r in real)
        assert all(r["top1"] == r["correct"] / r["n"] for r in real)
        for dataset in meta["datasets"]:
            a = next(r for r in real if r["group"] == "A_unprotected" and r["dataset"] == dataset)
            b = next(r for r in real if r["group"] == "B_protected" and r["dataset"] == dataset)
            assert a["correct"] == b["correct"], (a, b)
        checks += [
            "CLIP_feature_preservation",
            "CLIP_full_test_counts",
            "CLIP_accuracy_arithmetic",
            "CLIP_standalone_accuracy_preservation",
        ]
        if meta.get("attack_all"):
            for group in ["G_AGA_all_unprotected_merge", "H_AGA_all_protected_merge"]:
                subset = [r for r in real if r["group"] == group]
                assert len(subset) == 4
                assert {r["dataset"] for r in subset} == set(expected_n)
                assert {r["coefficient"] for r in subset} == set(meta["coefficients"])
            checks.append("CLIP_defense_agnostic_complete")
    if (path / "layernorm/layernorm_metadata.json").exists():
        meta = json.loads((path / "layernorm/layernorm_metadata.json").read_text())
        ln = json.loads((path / "layernorm/layernorm_results.json").read_text())
        assert len(ln) == 8 * len(meta["strengths"])
        assert all(r["n"] == 10000 and r["top1"] == r["correct"] / r["n"] for r in ln)
        assert max(r["protected_output_error"] for r in ln) < 1e-4
        assert max(r["canonical_output_error"] for r in ln) < 1e-4
        for row in ln:
            if row["group"] in ["LN_null_only", "LN_null_plus_basis"]:
                assert row["correct"] == 9969
        checks += ["LN_function_preservation", "LN_full_test_counts", "LN_utility_preservation"]
    if (path / "low_activity/low_activity_metadata.json").exists():
        meta = json.loads((path / "low_activity/low_activity_metadata.json").read_text())
        low = json.loads((path / "low_activity/low_activity_results.json").read_text())
        assert len(low) == 7
        assert all(r["n"] == 10000 and r["top1"] == r["correct"] / r["n"] for r in low)
        assert all(r["training"] == "none" for r in low)
        assert all(r["calibration_source"] == "procedural_unlabelled_strokes" for r in low)
        standalone = next(r for r in low if r["group"] == "low_activity_standalone")
        accepted = standalone["top1"] >= meta.get("baseline_accuracy", 0.9969) - 0.005
        assert standalone["within_utility_budget"] == meta["within_utility_budget"] == accepted
        checks += [
            "low_activity_full_test_counts",
            "low_activity_budget_record",
            "synthetic_calibration_provenance",
        ]
    if (path / "params/clip_metadata.json").exists():
        meta = json.loads((path / "params/clip_metadata.json").read_text())
        params = json.loads((path / "params/clip_results.json").read_text())
        source = json.loads(Path("configs/sources.json").read_text())["PaRaMS"]
        assert meta["protection_source_sha256"] == source["sha256"]
        assert meta["training"] == "none" and meta["calibration"] == "none"
        assert meta["protected_output_error"] < 1e-4
        assert len(params) == 34
        expected_n = {"MNIST": 10000, "SVHN": 26032}
        assert all(r["n"] == expected_n[r["dataset"]] for r in params)
        assert all(r["top1"] == r["correct"] / r["n"] for r in params)
        for dataset in expected_n:
            standalone = {
                r["group"]: r["correct"]
                for r in params
                if r["coefficient"] is None and r["dataset"] == dataset
            }
            assert standalone["A_unprotected"] == standalone["B_protected"]
        for group in ["G_AGA_all_unprotected_merge", "H_AGA_all_protected_merge"]:
            subset = [r for r in params if r["group"] == group]
            assert len(subset) == 4
            assert {r["dataset"] for r in subset} == set(expected_n)
            assert {r["coefficient"] for r in subset} == set(meta["coefficients"])
        checks += [
            "PaRaMS_source_and_provenance",
            "PaRaMS_counts_and_utility",
            "PaRaMS_blind_control",
        ]
    if (path / "coupled/coupled_metadata.json").exists():
        meta = json.loads((path / "coupled/coupled_metadata.json").read_text())
        paired = json.loads((path / "coupled/coupled_results.json").read_text())
        assert len(paired) == 12
        assert max(meta["feature_errors"].values()) < 1e-4
        source = json.loads(Path("configs/sources.json").read_text())["RotationSymmetry"]
        assert meta["source_sha256"] == source["sha256"]
        assert all(r["training"] == "none" and r["calibration"] == "none" for r in paired)
        assert all(r["n"] == {"MNIST": 10000, "SVHN": 26032}[r["dataset"]] for r in paired)
        assert all(r["top1"] == r["correct"] / r["n"] for r in paired)
        for dataset in ["MNIST", "SVHN"]:
            a = next(
                r
                for r in paired
                if r["group"] == "paired_unprotected_standalone" and r["dataset"] == dataset
            )
            b = next(
                r
                for r in paired
                if r["group"] == "paired_protected_standalone" and r["dataset"] == dataset
            )
            baseline = next(
                r for r in real if r["group"] == "A_unprotected" and r["dataset"] == dataset
            )
            assert a["correct"] == b["correct"] == baseline["correct"]
        checks += ["paired_source", "paired_function_preservation", "paired_counts_and_utility"]
    source = json.loads(Path("configs/sources.json").read_text())
    expected_n = {"MNIST": 10000, "SVHN": 26032}
    if (path / "rank_probe/metadata.json").exists():
        folder = path / "rank_probe"
        meta = json.loads((folder / "metadata.json").read_text())
        rank = json.loads((folder / "results.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        assert meta["completed"] and len(rank) == 50
        assert (
            meta["config_sha256"]
            == hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest()
        )
        assert (
            meta["config_sha256"]
            == hashlib.sha256(Path("configs/rank_probe.json").read_bytes()).hexdigest()
        )
        assert meta["author_sha256"] == source["AGA"]["sha256"]
        assert meta["paired_source_sha256"] == source["RotationSymmetry"]["sha256"]
        assert all(
            r["n"] == expected_n[r["dataset"]] and r["top1"] == r["correct"] / r["n"] for r in rank
        )
        for preset in [dict(name="unprotected"), *protocol["presets"]]:
            subset = [r for r in rank if r["preset"] == preset["name"]]
            assert {(r["group"], r["dataset"]) for r in subset} == {
                (g, d)
                for g in ["standalone", "direct", "AGA", "paired", "completion"]
                for d in expected_n
            }
            if preset.get("rank"):
                detail = json.loads((folder / preset["name"] / "compression.json").read_text())
                assert len(detail) == 48
                assert {(r["layer"], r["head"]) for r in detail} == {
                    (layer, head) for layer in protocol["layers"] for head in range(12)
                }
                assert all(r["effective_rank"] <= preset["rank"] for r in detail)
        for dataset in expected_n:
            control = next(
                r
                for r in rank
                if r["preset"] == "unprotected"
                and r["group"] == "direct"
                and r["dataset"] == dataset
            )
            baseline = next(
                r
                for r in real
                if r["group"] == "D_unprotected_merge"
                and r["coefficient"] == 0.5
                and r["dataset"] == dataset
            )
            assert control["correct"] == baseline["correct"]
        assert all(r["training"] == "none" for r in rank)
        assert all((r["calibration_samples"] == 32) == (r["method"] == "activation") for r in rank)
        checks += [
            "rank_protocol_and_sources",
            "rank_full_counts_and_controls",
            "rank_actual_operator_ranks",
            "rank_calibration_labels",
        ]
    for name in ["polynomial_probe", "polynomial_multilayer"]:
        folder = path / name
        if not (folder / "metadata.json").exists():
            continue
        meta = json.loads((folder / "metadata.json").read_text())
        poly = json.loads((folder / "results.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        layers = protocol.get("layers", [protocol.get("layer", 11)])
        assert meta["completed"] and len(poly) == 20
        assert (
            meta["config_sha256"]
            == hashlib.sha256((folder / "protocol.json").read_bytes()).hexdigest()
        )
        assert (
            meta["config_sha256"]
            == hashlib.sha256(Path(f"configs/{name}.json").read_bytes()).hexdigest()
        )
        assert meta["author_sha256"] == source["AGA"]["sha256"]
        assert not meta["original_weights_retained_in_released_ffn"]
        assert not meta["architecture_mismatch_counted_as_security"]
        assert meta["aga_mlp_skipped_layers"] == layers
        assert all(
            r["n"] == expected_n[r["dataset"]] and r["top1"] == r["correct"] / r["n"] for r in poly
        )
        groups = {
            "unprotected_standalone",
            "protected_standalone",
            "converted_other_standalone",
            "reconstructed_protected_standalone",
            "reconstructed_control_standalone",
            "original_direct_merge",
            "shared_format_merge",
            "shared_format_AGA_merge",
            "reconstructed_protected_merge",
            "reconstructed_control_merge",
        }
        assert {(r["group"], r["dataset"]) for r in poly} == {
            (g, d) for g in groups for d in expected_n
        }
        assert protocol["defender_samples"] == protocol["attacker_samples"] == 32
        if "attacker_calibration_seed" in meta:
            assert meta["attacker_calibration_seed"] == meta["defender_calibration_seed"] + 1
        assert all(
            r["fitting"] == "closed_form_ridge" and r["training"] == "no_gradient_or_optimizer"
            for r in poly
        )
        assert all(r["calibration"] == "procedural_unlabelled_strokes" for r in poly)
        for dataset in expected_n:
            control = next(
                r for r in poly if r["group"] == "original_direct_merge" and r["dataset"] == dataset
            )
            baseline = next(
                r
                for r in real
                if r["group"] == "D_unprotected_merge"
                and r["coefficient"] == 0.5
                and r["dataset"] == dataset
            )
            assert control["correct"] == baseline["correct"]
        checks += [
            f"{name}_protocol_and_source",
            f"{name}_counts_and_control",
            f"{name}_release_and_attack_scope",
            f"{name}_calibration_and_fitting",
        ]
    resource = path / "eight_task/pretrained_1828/pretrained_checks.json"
    if resource.exists():
        rows = json.loads(resource.read_text())
        assert len(rows) == 2
        assert all(r["n"] == {"DTD": 1880, "GTSRB": 12630}[r["dataset"]] for r in rows)
        assert all(r["top1"] == r["correct"] / r["n"] and not r["eight_task_result"] for r in rows)
        checks.append("eight_task_preparatory_data_and_head_checks")
    resisc_resource = path / "eight_task/pretrained_2117/pretrained_checks.json"
    if resisc_resource.exists():
        resource_rows = json.loads(resisc_resource.read_text())
        assert len(resource_rows) == 1
        row = resource_rows[0]
        assert row["dataset"] == "RESISC45" and row["n"] == 6300
        assert row["top1"] == row["correct"] / row["n"]
        assert not row["eight_task_result"] and row["task_count"] == 1
        assert row["data_provider"] == "author_dataset_registry"
        manifest = json.loads((path / "eight_task/data_2104/data_manifest.json").read_text())
        assert len(manifest) == 1 and manifest[0]["task"] == "RESISC45"
        item = manifest[0]
        assert item["status"] == "ready" and item["dataset_ready"]
        assert item["image_count"] == 31500 and item["test_count"] == 6300
        assert {s["split"]: s["count"] for s in item["splits"]} == {
            "train": 18900,
            "val": 6300,
            "test": 6300,
        }
        assert all(s["upstream_md5_verified"] for s in item["splits"])
        assert item["archive_sha256"] == (
            "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
        )
        checks.append("RESISC45_verified_archive_splits_and_full_resource_check")
    report = {"status": "passed", "checks": checks}
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_subspace(path: Path):
    protocol_bytes = (path / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    meta = json.loads((path / "metadata.json").read_text())
    rows = json.loads((path / "results.json").read_text())
    presets = json.loads((path / "development_screen.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    indices = json.loads((path / "training_indices.json").read_text())
    subspaces = json.loads((path / "subspaces.json").read_text())
    assert meta["completed"] and meta["training"] == "none"
    assert meta["config_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert (
        meta["data_indices_sha256"]
        == hashlib.sha256((path / "training_indices.json").read_bytes()).hexdigest()
    )
    for task, splits in indices.items():
        assert len(splits["calibration"]) == protocol["calibration_samples"]
        assert len(splits["attack"]) == protocol["attack_samples"]
        assert len(splits["development"]) == protocol["development_samples"][task]
        flattened = [i for values in splits.values() for i in values]
        assert len(flattened) == len(set(flattened))
        assert min(flattened) >= 0 and max(flattened) < {"MNIST": 60000, "SVHN": 73257}[task]
    assert len(subspaces) == 24
    for item in subspaces:
        for field in ["MNIST", "SVHN_on_MNIST_expert"]:
            stats = item[field]
            assert stats["rank"] + stats["complement_rank"] == 768
            assert stats["retained_fraction"] >= protocol["retained_energy"]
            assert stats["tokens"] == 50 * protocol["calibration_samples"]
    assert len(presets) == len(protocol["layers"]) * len(protocol["targets"]) * len(
        protocol["strengths"]
    )
    assert all(p["check"]["main_space_relative_leakage_float32"] < 1e-5 for p in presets)
    for row in rows:
        expected_n = (
            {"MNIST": 10000, "SVHN": 26032}
            if row["split"] == "official_test"
            else protocol["development_samples"]
        )
        assert row["n"] == expected_n[row["dataset"]]
        assert row["top1"] == row["correct"] / row["n"]
        assert 0 <= row["correct"] <= row["n"] and math.isfinite(row["top1"])
    development = [r for r in rows if r["split"] == "development"]
    assert len(development) == 3 * (len(presets) + 1)
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in development}
    assert len(lookup) == len(development)
    original_utility = lookup["unprotected", "standalone", "MNIST"]["top1"]
    for preset in presets:
        utility = lookup[preset["name"], "standalone", "MNIST"]["top1"]
        mean_merged = (
            sum(lookup[preset["name"], "direct", task]["top1"] for task in ["MNIST", "SVHN"]) / 2
        )
        assert abs(preset["utility_loss_pp"] - 100 * (original_utility - utility)) < 1e-10
        assert abs(preset["mean_merged_accuracy"] - mean_merged) < 1e-10
        assert preset["utility_feasible"] == (
            preset["utility_loss_pp"] <= protocol["utility_budget_pp"] + 1e-9
        )
    feasible = [p for p in presets if p["utility_feasible"]]
    expected = (
        min(feasible, key=lambda p: (p["mean_merged_accuracy"], p["strength"]))
        if feasible
        else None
    )
    assert selection["selected"] == expected
    assert selection["feasible_count"] == len(feasible)
    test = [r for r in rows if r["split"] == "official_test"]
    assert len(test) == (24 if expected else 0)
    if expected:
        for preset in ["unprotected", expected["name"]]:
            subset = [r for r in test if r["preset"] == preset]
            assert len(subset) == 12
            assert {r["group"] for r in subset} == {
                "standalone",
                "direct",
                "AGA_standalone",
                "AGA",
                "projection_standalone",
                "projection",
                "completion_standalone",
                "completion",
            }
    report = dict(
        status="passed",
        checks=[
            "protocol_and_index_hashes",
            "disjoint_training_subsets",
            "all_layer_token_counts_and_pca_energy",
            "preset_count_and_projection_arithmetic",
            "accuracy_counts_and_finiteness",
            "development_only_selection",
            "complete_tests_with_matched_attack_controls",
        ],
        development_rows=len(development),
        official_test_rows=len(test),
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_subspace_diagnostics(path: Path):
    protocol_bytes = (path / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    meta = json.loads((path / "metadata.json").read_text())
    parent = Path(protocol["parent"])
    parent_bytes = (parent / "results.json").read_bytes()
    parent_rows = json.loads(parent_bytes)
    rows = json.loads((path / "results.json").read_text())
    assert meta["completed"] and meta["training"] == "none"
    assert meta["config_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert meta["parent_results_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    reused = [r for r in rows if r["origin"] == "parent_measurement"]
    new = [r for r in rows if r["origin"] == "new_evaluation"]
    assert len(reused) == meta["reused_measurements"] == 12
    assert len(new) == meta["new_evaluations"] == 30
    parent_lookup = {
        (r["preset"], r["group"], r["dataset"]): r
        for r in parent_rows
        if r["split"] == "official_test"
    }
    for row in reused:
        source = parent_lookup[row["preset"], row["group"], row["dataset"]]
        assert all(row[key] == value for key, value in source.items())
    assert len({(r["contrast"], r["group"], r["dataset"]) for r in rows}) == len(rows)
    for row in rows:
        assert row["n"] == {"MNIST": 10000, "SVHN": 26032}[row["dataset"]]
        assert row["top1"] == row["correct"] / row["n"]
        assert 0 <= row["correct"] <= row["n"] and math.isfinite(row["top1"])
    decomp = json.loads((path / "decomposition.json").read_text())
    assert 0 <= decomp["null_edit_energy_fraction"] <= 1
    assert decomp["reconstruction_relative_error"] < 1e-6
    screen = json.loads((parent / "development_screen.json").read_text())
    expected_fc = min(
        [p for p in screen if p["site"].startswith("FC:") and p["utility_feasible"]],
        key=lambda p: (p["mean_merged_accuracy"], p["strength"]),
    )
    fc = json.loads((path / "fc_selection.json").read_text())
    assert fc["source"] == "parent_development_only" and fc["selected"] == expected_fc
    diagnostics = json.loads((path / "parameter_diagnostics.json").read_text())
    assert set(diagnostics) == {"V_full", *protocol["contrasts"]}
    for item in diagnostics.values():
        for stage in ["before_AGA", "after_AGA"]:
            assert math.isfinite(item[stage]["norm_over_original"])
            assert item[stage]["norm_over_original"] > 0
            assert all(
                math.isfinite(view["relative_linear_output_rms"])
                and view["relative_linear_output_rms"] >= 0
                for view in item[stage]["same_input_linear_outputs"].values()
            )
    for contrast in ["control", "FC_development_selected"]:
        for group in ["projection", "completion"]:
            assert len([r for r in rows if r["contrast"] == contrast and r["group"] == group]) == 2
    report = dict(
        status="passed",
        reused_measurements=len(reused),
        new_evaluations=len(new),
        checks=[
            "protocol_and_parent_provenance",
            "parent_metric_reuse_without_new_count",
            "complete_counts_and_accuracy_arithmetic",
            "additive_component_reconstruction",
            "FC_development_only_selection",
            "finite_local_parameter_diagnostics",
            "matched_FC_projection_and_completion_controls",
        ],
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_bounded_complement(path: Path):
    protocol_bytes = (path / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    protected_task = protocol.get("protected_task", "MNIST")
    meta = json.loads((path / "metadata.json").read_text())
    parent = Path(protocol["parent"])
    parent_bytes = (parent / "results.json").read_bytes()
    indices_bytes = (parent / "training_indices.json").read_bytes()
    indices = json.loads(indices_bytes)
    rows = json.loads((path / "results.json").read_text())
    screen = json.loads((path / "development_screen.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    scans = json.loads((path / "angular_scans.json").read_text())
    profile = json.loads((path / "replacement_profile.json").read_text())
    assert meta["completed"] and meta["training"] == "none"
    assert meta["config_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert meta["parent_results_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    assert meta["parent_indices_sha256"] == hashlib.sha256(indices_bytes).hexdigest()
    reused = [r for r in rows if r["origin"] == "parent_measurement"]
    new = [r for r in rows if r["origin"] == "new_evaluation"]
    assert len(reused) == meta["reused_measurements"] == (3 if protected_task == "MNIST" else 2)
    assert len(new) == meta["new_evaluations"]
    parent_lookup = {
        (r["preset"], r["group"], r["dataset"]): r
        for r in json.loads(parent_bytes)
        if r["split"] == "development"
    }
    for row in reused:
        source = parent_lookup[row["preset"], row["group"], row["dataset"]]
        assert all(row[key] == value for key, value in source.items())
    assert len({(r["split"], r["preset"], r["group"], r["dataset"]) for r in rows}) == len(rows)
    for row in rows:
        expected_n = (
            len(indices[row["dataset"]]["development"])
            if row["split"] == "development"
            else {"MNIST": 10000, "SVHN": 26032}[row["dataset"]]
        )
        assert row["n"] == expected_n
        assert 0 <= row["correct"] <= row["n"]
        assert row["top1"] == row["correct"] / row["n"] and math.isfinite(row["top1"])
    assert len(profile) == 36 and len({p["site"] for p in profile}) == 36
    assert sum(p["kind"] == "single_projection_matrix" for p in profile) == 24
    assert sum(p["kind"] == "entire_block" for p in profile) == 12
    assert len(screen) == len(protocol["site_groups"]) * len(protocol["edit_scales"]) * len(
        protocol["retained_energies"]
    )
    lookup = {
        (r["preset"], r["group"], r["dataset"]): r["top1"]
        for r in rows
        if r["split"] == "development"
    }
    baseline = {task: lookup["unprotected", "direct", task] for task in ["MNIST", "SVHN"]}
    original = lookup["unprotected", "standalone", protected_task]
    for item in profile:
        site = item["site"]
        preset = (
            f"replace_{site}"
            if item["kind"] == "single_projection_matrix"
            else f"replace_block_{site.split(':')[1]}"
        )
        loss = 100 * (original - lookup[preset, "standalone", protected_task])
        assert abs(loss - item["standalone_loss_pp"]) < 1e-10
        if item["kind"] == "single_projection_matrix":
            for task in ["MNIST", "SVHN"]:
                assert item[f"merged_{task}"] == lookup[preset, "direct", task]
    for preset in screen:
        name = preset["name"]
        loss = 100 * (original - lookup[name, "standalone", protected_task])
        assert abs(loss - preset["standalone_loss_pp"]) < 1e-10
        assert preset["worst_case_mean"] == max(preset["mean_direct"], preset["mean_angular"])
        assert preset["utility_feasible"] == (loss <= protocol["utility_budget_pp"] + 1e-9)
        for group in ["direct", "angular"]:
            acc = {task: lookup[name, group, task] for task in baseline}
            assert abs(preset[f"mean_{group}"] - sum(acc.values()) / 2) < 1e-10
            assert (
                abs(
                    preset["drops_pp"][group][protected_task]
                    - 100 * (baseline[protected_task] - acc[protected_task])
                )
                < 1e-10
            )
            assert (
                abs(
                    preset["drops_pp"][group]["mean"]
                    - 50 * (sum(baseline.values()) - sum(acc.values()))
                )
                < 1e-10
            )
        passed = preset["utility_feasible"] and all(
            d[protected_task] >= protocol["minimum_protected_task_drop_pp"] - 1e-9
            and d["mean"] >= protocol["minimum_mean_drop_pp"] - 1e-9
            for d in preset["drops_pp"].values()
        )
        assert preset["recovery_gate_passed"] == passed
        assert len(preset["checks"]) == len(protocol["site_groups"][preset["site_group"]])
        for item in preset["checks"]:
            assert item["main_space_leakage"] < 1e-5 and 0 < item["weight_norm_ratio"] <= 1 + 1e-6
            if preset["scale"] == 2:
                assert abs(item["weight_norm_ratio"] - 1) < 1e-6
    assert set(scans) == {"unprotected", *(s["name"] for s in screen)}
    for scan in scans.values():
        assert len(scan["inspected"]) == 49
        assert len({item["parameter"] for item in scan["inspected"]}) == 49
        assert all(
            math.isfinite(item["cosine"]) and -1 - 1e-8 <= item["cosine"] <= 1 + 1e-8
            for item in scan["inspected"]
        )
        expected = {
            item["parameter"]
            for item in scan["inspected"]
            if item["cosine"] < protocol["angular_cosine_threshold"]
        }
        assert {item["parameter"] for item in scan["replaced"]} == expected
    assert not scans["unprotected"]["replaced"]
    eligible = [s for s in screen if s["recovery_gate_passed"]]
    expected = (
        min(eligible, key=lambda s: (s["worst_case_mean"], s["standalone_loss_pp"]))
        if eligible
        else None
    )
    assert selection["selected"] == expected
    assert selection["utility_feasible_count"] == sum(s["utility_feasible"] for s in screen)
    assert selection["recovery_gate_passed_count"] == len(eligible)
    official = [r for r in rows if r["split"] == "official_test"]
    assert len(official) == meta["official_test_rows"] == (24 if expected else 0)
    assert (
        selection["test_run"] == bool(expected)
        and selection["test_usage"] == "none_before_selection"
    )
    report = dict(
        status="passed",
        reused_measurements=len(reused),
        new_evaluations=len(new),
        official_test_rows=len(official),
        checks=[
            "protocol_and_parent_provenance",
            "labelled_metric_reuse",
            "complete_counts_and_accuracy_arithmetic",
            "complete_local_replacement_profile",
            "bounded_edit_and_reflection_norm_checks",
            "blind_angular_scan_and_matched_control",
            "development_only_utility_and_recovery_gate",
            "official_test_only_after_gate",
        ],
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_distribution_shift(path: Path):
    config_bytes = (path / "protocol.json").read_bytes()
    config = json.loads(config_bytes)
    meta = json.loads((path / "metadata.json").read_text())
    rows = json.loads((path / "results.json").read_text())
    index_bytes = (Path(config["parent"]) / "training_indices.json").read_bytes()
    assert meta["completed"] and meta["training"] == meta["test_usage"] == "none"
    assert meta["accuracy_evaluations"] == 0 and meta["samples_per_task"] == 256
    assert meta["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert meta["parent_indices_sha256"] == hashlib.sha256(index_bytes).hexdigest()
    assert len(rows) == 2 * 24 * len(config["retained_energies"])
    assert len({(r["protected_task"], r["site"], r["retained_energy"]) for r in rows}) == len(rows)
    stats = {}
    for task, source in meta["sources"].items():
        source_path = Path(source["directory"])
        source_bytes = (source_path / "metadata.json").read_bytes()
        assert source["source_metadata_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert source["source_config_sha256"] == json.loads(source_bytes)["config_sha256"]
        assert len(source["basis_sha256"]) == 64
        stats[task] = json.loads((source_path / "calibration_subspaces.json").read_text())
    for row in rows:
        assert row["tokens"] == 50 * meta["samples_per_task"]
        stat = stats[row["protected_task"]][str(row["retained_energy"])][row["site"]]
        assert row["rank"] == stat["rank"]
        assert abs(row["expert"]["fraction"] - (1 - stat["retained_fraction"])) < 1e-8
        for group in ["expert", "unprotected_merge"]:
            power = row[group]
            assert all(math.isfinite(value) for value in power.values())
            assert power["total"] > 0 and 0 <= power["outside"] <= power["total"]
            assert abs(power["fraction"] - power["outside"] / power["total"]) < 1e-12
    report = dict(
        status="passed",
        diagnostic_rows=len(rows),
        accuracy_evaluations=0,
        checks=[
            "protocol_and_frozen_indices",
            "source_metadata_and_basis_provenance",
            "complete_task_site_energy_grid",
            "original_PCA_energy_consistency",
            "token_counts_and_finite_energy_arithmetic",
            "no_test_or_accuracy_measurements",
        ],
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_scope_audit(path: Path):
    rows = json.loads((path / "results.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    controls = json.loads((path / "controls.json").read_bytes())
    assert meta["completed"] and meta["accuracy_evaluations"] == 0
    assert meta["training"] == meta["test_usage"] == "none"
    assert len(rows) == meta["candidates"] == 24
    assert len({(r["protected_task"], r["preset"]) for r in rows}) == len(rows)
    assert all(
        r["inspected"] == 49
        and r["angular_replacement_set_equal"]
        and r["angular_state_equal"]
        and r["norm_replacement_set_equal"]
        for r in rows
    )
    for scan in controls.values():
        assert not scan["replaced"] and len(scan["inspected"]) == 49
        names = {r["parameter"] for r in scan["inspected"]}
        assert "model.visual.proj" in names and "model.token_embedding.weight" not in names
    assert all(
        r["final_projection"]["parameter"] == "model.visual.proj"
        and r["final_projection"]["cosine"] >= 0.9
        for r in rows
    )
    report = dict(
        status="passed",
        candidates=len(rows),
        accuracy_evaluations=0,
        checks=[
            "complete_candidate_grid",
            "corrected_visual_scope_and_controls",
            "unchanged_angular_and_norm_replacement_sets",
            "exact_endpoint_state_equality",
        ],
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_norm_cleanup(path: Path):
    protocol_bytes = (path / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    meta = json.loads((path / "metadata.json").read_text())
    parent_bytes = (Path(protocol["parent"]) / "results.json").read_bytes()
    parent_rows = json.loads(parent_bytes)
    rows = json.loads((path / "results.json").read_text())
    scans = json.loads((path / "scans.json").read_text())
    assert meta["completed"] and meta["training"] == meta["calibration"] == "none"
    assert meta["test_threshold_selection"] == "none"
    assert meta["config_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert meta["parent_results_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    reused = [r for r in rows if r["origin"] == "parent_measurement"]
    new = [r for r in rows if r["origin"] == "new_evaluation"]
    assert len(reused) == meta["reused_measurements"] == 3
    assert len(new) == meta["new_evaluations"] == 6
    parent_lookup = {
        (r["preset"], r["group"], r["dataset"]): r
        for r in parent_rows
        if r["split"] == "official_test"
    }
    for row in reused:
        source = parent_lookup[row["preset"], row["group"], row["dataset"]]
        assert all(row[key] == value for key, value in source.items())
    assert set(scans) == {"control", "V_full", "FC_development_selected"}
    assert not scans["control"]["replaced"]
    assert all(scan["inspected_matrices"] == 49 for scan in scans.values())
    assert math.isfinite(protocol["norm_ratio_threshold"]) and protocol["norm_ratio_threshold"] > 1
    for contrast, scan in scans.items():
        assert len({item["parameter"] for item in scan["replaced"]}) == len(scan["replaced"])
        for item in scan["replaced"]:
            assert math.isfinite(item["norm_ratio"]) and (
                item["norm_ratio"] > protocol["norm_ratio_threshold"]
            )
            assert isinstance(item["bias_replaced"], bool)
        subset = [r for r in rows if r["contrast"] == contrast]
        assert {(r["group"], r["dataset"]) for r in subset} == {
            ("standalone", "MNIST"),
            ("direct", "MNIST"),
            ("direct", "SVHN"),
        }
        assert len(subset) == 3
    for row in rows:
        assert row["n"] == {"MNIST": 10000, "SVHN": 26032}[row["dataset"]]
        assert 0 <= row["correct"] <= row["n"]
        assert row["top1"] == row["correct"] / row["n"] and math.isfinite(row["top1"])
        assert row["coefficient"] == (
            None if row["group"] == "standalone" else protocol["coefficient"]
        )
    report = dict(
        status="passed",
        reused_measurements=len(reused),
        new_evaluations=len(new),
        checks=[
            "protocol_and_parent_provenance",
            "data_free_attack_and_no_test_threshold_search",
            "unchanged_control_and_explicit_metric_reuse",
            "full_linear_scan_and_outlier_rule",
            "complete_matched_contrasts",
            "accuracy_counts_and_coefficient_arithmetic",
        ],
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--subspace", action="store_true", help="Validate the subspace probe.")
    mode.add_argument(
        "--subspace-diagnostics", action="store_true", help="Validate causal contrasts."
    )
    mode.add_argument("--norm-cleanup", action="store_true", help="Validate data-free cleanup.")
    mode.add_argument("--bounded-complement", action="store_true", help="Validate bounded edits.")
    mode.add_argument(
        "--distribution-shift", action="store_true", help="Validate input diagnostics."
    )
    mode.add_argument("--scope-audit", action="store_true", help="Validate visual scan correction.")
    args = parser.parse_args()
    runner = (
        validate_scope_audit
        if args.scope_audit
        else validate_distribution_shift
        if args.distribution_shift
        else validate_bounded_complement
        if args.bounded_complement
        else validate_norm_cleanup
        if args.norm_cleanup
        else validate_subspace_diagnostics
        if args.subspace_diagnostics
        else validate_subspace
        if args.subspace
        else validate
    )
    runner(Path(args.output))
