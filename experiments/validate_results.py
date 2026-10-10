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


def validate_readout_stability(path, write_report=True):
    """Recompute conservative fixed-readout certificates from saved influences."""
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    assert meta["completed"] and not meta["new_defense"]
    assert meta["new_models"] == meta["new_teacher_queries"] == meta["new_fits"] == 0
    assert not meta["fitting_labels_used"] and meta["test_usage"] == "none"
    assert meta["confirmation_predictions_exactly_match_parent"]
    assert (
        meta["config_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    )
    parent = Path(config["parent"])
    assert (
        meta["parent_metadata_sha256"]
        == hashlib.sha256((parent / "metadata.json").read_bytes()).hexdigest()
    )
    assert [r["changed_queries"] for r in rows] == config["budgets"] == [0, 1, 2, 4, 8, 16]
    assert all(r["n"] == meta["baseline_n"] == 4000 for r in rows)
    raw = path / "artifacts/certificate.pt"
    if raw.exists():
        import torch

        assert meta["certificate_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        saved = torch.load(raw, map_location="cpu", weights_only=True)
        influence, margins = saved["influence"], saved["margins"]
        assert influence.shape == (4000, 512) and margins.shape == (4000,)
        assert torch.isfinite(influence).all() and torch.isfinite(margins).all()
        correct = saved["prediction"] == saved["labels"]
        assert int(correct.sum()) == meta["baseline_correct"] == 3785
        largest = influence[:, :256].abs().topk(16, dim=1).values.cumsum(dim=1)
        for row in rows:
            k = row["changed_queries"]
            bound = torch.zeros_like(margins) if k == 0 else 2 * largest[:, k - 1]
            certified = margins > bound + 1e-10
            assert int(certified.sum()) == row["certified_unchanged"]
            assert int((certified & correct).sum()) == row["certified_correct"]
    for row in rows:
        assert row["certified_fraction"] == row["certified_unchanged"] / row["n"]
        assert row["lower_bound_accuracy"] == row["certified_correct"] / row["n"]
    assert rows[0]["certified_unchanged"] == 4000
    assert all(a["certified_unchanged"] >= b["certified_unchanged"] for a, b in zip(rows, rows[1:]))
    report = dict(
        status="passed",
        rows=len(rows),
        raw_influence_checked=raw.exists(),
        whole_model_certified=False,
        new_defense=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def validate_readout_recovery(path, write_report=True):
    """Check output-only recovery without treating training probes as final evidence."""
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    if "projection_sha256" in config:
        assert meta["completed"] and not meta["new_defense"]
        assert meta["new_fits"] == meta["optimizer_steps"] == 0
        assert not meta["fitting_labels_used"]
        assert meta["test_usage"] == config["test_usage"]
        assert (
            meta["config_sha256"]
            == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
        )
        parent = Path(config["parent"])
        prior = json.loads((parent / "metadata.json").read_bytes())
        assert (
            meta["parent_metadata_sha256"]
            == hashlib.sha256((parent / "metadata.json").read_bytes()).hexdigest()
        )
        assert meta["checkpoint_sha256"] == prior["checkpoint_sha256"]
        assert meta["projection_sha256"] == config["projection_sha256"]
        for preset, value in config["projection_sha256"].items():
            assert (
                hashlib.sha256(
                    (parent / f"artifacts/{preset}_projection.pt").read_bytes()
                ).hexdigest()
                == value
            )
        expected = {
            (preset, "test", group, task)
            for preset in config["presets"]
            for group, task in [("standalone", "MNIST"), ("direct", "MNIST"), ("direct", "SVHN")]
        }
        keyed = {(r["preset"], r["split"], r["group"], r["dataset"]): r for r in rows}
        assert set(keyed) == expected and len(rows) == meta["new_evaluations"] == 12
        raw = path / "artifacts/predictions.pt"
        saved = None
        if raw.exists():
            import torch
            from operator_diagnostics import paired_counts

            assert hashlib.sha256(raw.read_bytes()).hexdigest() == meta["predictions_sha256"]
            saved = torch.load(raw, map_location="cpu", weights_only=False)
            assert set(saved) == expected
        for key, row in keyed.items():
            assert row["n"] == {"MNIST": 10000, "SVHN": 26032}[row["dataset"]]
            assert row["top1"] == row["correct"] / row["n"]
            if saved is not None:
                pred, labels = saved[key]["prediction"], saved[key]["labels"]
                assert len(pred) == len(labels) == row["n"]
                assert int((pred == labels).sum()) == row["correct"]
                reference = saved["unprotected", *key[1:]]
                assert torch.equal(labels, reference["labels"])
                for field, value in paired_counts(
                    reference["prediction"] == labels, pred == labels
                ).items():
                    assert abs(row["paired"][field] - value) < 1e-10
                if key[0] == "readout_protected":
                    matched = saved["readout_unprotected", *key[1:]]
                    for field, value in paired_counts(
                        matched["prediction"] == labels, pred == labels
                    ).items():
                        assert abs(row["matched_recovery"][field] - value) < 1e-10
                    if prior["target_disagreement"] == 0:
                        assert torch.equal(pred, matched["prediction"])
        report = dict(
            status="passed",
            rows=12,
            raw_predictions_checked=raw.exists(),
            frozen_projection_hashes_checked=True,
            new_fits=0,
            new_defense=False,
            results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
        )
        if write_report:
            (path / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return
    indices = json.loads((path / "training_indices.json").read_bytes())
    fitting = json.loads((path / "fitting.json").read_bytes())
    assert meta["completed"] and not meta["new_defense"]
    assert meta["test_usage"] == config["test_usage"] == "none"
    assert not meta["fitting_labels_used"] and meta["optimizer_steps"] == 0
    assert meta["task_head_weights_exactly_equal"] and meta["task_head_biases_zero"]
    for filename, field in [
        ("protocol.json", "config_sha256"),
        ("training_indices.json", "indices_sha256"),
    ]:
        assert meta[field] == hashlib.sha256((path / filename).read_bytes()).hexdigest()
    parent = json.loads((Path(config["index_source"]) / "training_indices.json").read_bytes())
    confirmation = json.loads(
        (Path(config["confirmation_source"]) / "training_indices.json").read_bytes()
    )
    assert indices["confirmation"] == confirmation["confirmation"]
    for task in ["MNIST", "SVHN"]:
        assert indices["development"][task] == parent[task]["development"]
        assert indices["fitting"][task] == parent[task]["attack"][:256]
        assert len(set(indices["fitting"][task])) == 256
        assert not set(indices["fitting"][task]).intersection(indices["development"][task])
        assert not set(indices["fitting"][task]).intersection(parent[task]["calibration"])
        assert not set(indices["fitting"][task]).intersection(indices["confirmation"].get(task, []))
    presets = ["unprotected", "public_backbone", "readout_unprotected", "readout_protected"]
    scopes = [
        (s, g, t)
        for s in ["development", "confirmation"]
        for t in indices[s]
        for g in ["standalone", "direct"]
        if g == "direct" or t == "MNIST"
    ]
    expected = {(p, *scope) for p in presets for scope in scopes}
    keyed = {(r["preset"], r["split"], r["group"], r["dataset"]): r for r in rows}
    assert set(keyed) == expected and len(rows) == len(expected) == meta["new_evaluations"] == 20
    for row in rows:
        assert row["n"] == len(indices[row["split"]][row["dataset"]])
        assert row["top1"] == row["correct"] / row["n"]
        assert 0 <= row["correct"] <= row["n"]
    assert {f["preset"] for f in fitting} == {"readout_unprotected", "readout_protected"}
    for fit in fitting:
        assert fit["images"] == 512 and fit["features"] == 768
        assert fit["normal_equation_relative_residual"] < 1e-8
        assert fit["head_mapping_relative_error"] < 1e-8
        assert fit["merged_projection_relative_error"] < 1e-6
    raw, targets = path / "artifacts/predictions.pt", path / "artifacts/teacher_predictions.pt"
    if raw.exists():
        import torch
        from operator_diagnostics import paired_counts

        assert meta["predictions_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        assert (
            meta["teacher_predictions_sha256"] == hashlib.sha256(targets.read_bytes()).hexdigest()
        )
        teacher = torch.load(targets, map_location="cpu", weights_only=True)
        a, b = teacher["readout_unprotected"], teacher["readout_protected"]
        assert a.shape == b.shape == (512,) and torch.equal(a[256:], b[256:])
        assert int((a != b).sum()) == meta["target_disagreement"]
        assert math.isclose(meta["target_difference_frobenius"], (2 * int((a != b).sum())) ** 0.5)
        saved = torch.load(raw, map_location="cpu", weights_only=False)
        assert set(saved) == expected
        for key, row in keyed.items():
            pred, labels = saved[key]["prediction"], saved[key]["labels"]
            assert int((pred == labels).sum()) == row["correct"]
            assert len(pred) == len(labels) == row["n"]
            control = saved["unprotected", *key[1:]]
            assert torch.equal(labels, control["labels"])
            for field, value in paired_counts(
                control["prediction"] == labels, pred == labels
            ).items():
                assert abs(row["paired"][field] - value) < 1e-10
            if row["preset"] == "readout_protected":
                matched = saved["readout_unprotected", *key[1:]]
                for field, value in paired_counts(
                    matched["prediction"] == labels, pred == labels
                ).items():
                    assert abs(row["matched_recovery"][field] - value) < 1e-10
                if meta["target_disagreement"] == 0:
                    assert torch.equal(pred, matched["prediction"])
    report = dict(
        status="passed",
        rows=len(rows),
        raw_predictions_checked=raw.exists(),
        ground_truth_labels_used_for_fitting=False,
        test_usage="none",
        new_defense=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def validate_operator_diagnostics(path, write_report=True):
    meta = json.loads((path / "metadata.json").read_bytes())
    config_bytes = (path / "protocol.json").read_bytes()
    config = json.loads(config_bytes)
    indices_bytes = (path / "training_indices.json").read_bytes()
    indices = json.loads(indices_bytes)
    parent_bytes = (Path(config["parent"]) / "training_indices.json").read_bytes()
    parent = json.loads(parent_bytes)
    rows = json.loads((path / "results.json").read_bytes())
    profile = json.loads((path / "replacement_profile.json").read_bytes())
    fitting = json.loads((path / "fitting.json").read_bytes())
    assert meta["completed"] and not meta["new_defense"]
    assert meta["test_usage"] == config["test_usage"] == "none"
    assert meta["reused_measurements"] == 0 and meta["seed_count"] == 1
    assert meta["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert meta["data_indices_sha256"] == hashlib.sha256(indices_bytes).hexdigest()
    assert meta["parent_indices_sha256"] == hashlib.sha256(parent_bytes).hexdigest()
    task = config["protected_task"]
    reconstruction = config.get("reconstruction", True)
    for dataset in ["MNIST", "SVHN"]:
        assert indices["development"][dataset] == parent[dataset]["development"]
    assert indices["fitting"][task] == parent[task]["attack"][: config["attacker_samples"]]
    assert (
        indices["local_check"][task] == parent[task]["calibration"][: config["local_check_samples"]]
    )
    confirmation = indices["confirmation"][task]
    assert len(confirmation) == len(set(confirmation)) == config["confirmation_samples"]
    assert not set(confirmation).intersection(i for subset in parent[task].values() for i in subset)
    assert all(0 <= i < {"SVHN": 73257, "MNIST": 60000}[task] for i in confirmation)
    expected = {"unprotected"}
    if reconstruction:
        expected.add("reconstruct_joint")
        expected.update(f"reconstruct_L{layer}" for layer in config["layers"])
    expected.update(
        f"replace_L{layer}_{component}"
        for layer in config["layers"]
        for component in config["components"]
    )
    expected.update(f"joint_{component}" for component in config["joint_components"])
    development_groups = {("standalone", task), ("direct", "MNIST"), ("direct", "SVHN")}
    expected_keys = {(p, "development", g, d) for p in expected for g, d in development_groups}
    expected_keys.update(
        (p, "confirmation", g, task)
        for p in config["confirmation_presets"]
        for g in ["standalone", "direct"]
    )
    assert len(rows) == meta["new_evaluations"] == len(expected_keys)
    assert config["confirmation_presets"][0] == "unprotected"
    assert set(config["confirmation_presets"]) <= expected
    assert {(r["preset"], r["split"], r["group"], r["dataset"]) for r in rows} == expected_keys
    controls = {
        (r["split"], r["group"], r["dataset"]): r for r in rows if r["preset"] == "unprotected"
    }
    for row in rows:
        control = controls[row["split"], row["group"], row["dataset"]]
        n = len(indices[row["split"]][row["dataset"]])
        assert row["n"] == n and 0 <= row["correct"] <= n
        assert row["top1"] == row["correct"] / n
        assert row["coefficient"] == (
            None if row["group"] == "standalone" else config["coefficient"]
        )
        pair = row["paired"]
        assert all(
            isinstance(pair[k], int) and pair[k] >= 0
            for k in ["both_correct", "lost", "gained", "both_wrong"]
        )
        assert sum(pair[k] for k in ["both_correct", "lost", "gained", "both_wrong"]) == n
        assert pair["both_correct"] + pair["gained"] == row["correct"]
        assert pair["both_correct"] + pair["lost"] == control["correct"]
        assert pair["loss_pp"] == 100 * (pair["lost"] - pair["gained"]) / n
        assert 0 <= pair["exact_mcnemar_p"] <= 1
        if row["preset"] != "unprotected":
            assert pair["exact_mcnemar_p"] <= pair["holm_p"] <= 1
            assert pair["family_size"] == (
                len(expected) - 1
                if row["split"] == "development"
                else len(config["confirmation_presets"]) - 1
            )
    expected_profile = len(config["layers"]) * len(config["components"]) + len(
        config["joint_components"]
    )
    assert len(profile) == len({p["preset"] for p in profile}) == expected_profile
    for item in profile:
        assert item["replaced_keys"] and len(item["replaced_keys"]) == len(
            set(item["replaced_keys"])
        )
        assert all(
            any(
                key.startswith(f"model.visual.transformer.resblocks.{layer}.")
                for layer in item["layers"]
            )
            for key in item["replaced_keys"]
        )
        if item["component"] in {"norms", "attn_mlp"}:
            suffixes = (
                {"ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"}
                if item["component"] == "norms"
                else {
                    "attn.in_proj_weight",
                    "attn.in_proj_bias",
                    "attn.out_proj.weight",
                    "attn.out_proj.bias",
                    "mlp.c_fc.weight",
                    "mlp.c_fc.bias",
                    "mlp.c_proj.weight",
                    "mlp.c_proj.bias",
                }
            )
            assert set(item["replaced_keys"]) == {
                f"model.visual.transformer.resblocks.{layer}.{suffix}"
                for layer in item["layers"]
                for suffix in suffixes
            }
    assert {f["layer"] for f in fitting} == (set(config["layers"]) if reconstruction else set())
    assert len(fitting) == (len(config["layers"]) if reconstruction else 0)
    for fit in fitting:
        assert fit["fit"]["calibration_tokens"] == config["attacker_samples"] * 50
        assert fit["check"]["check_tokens"] == config["local_check_samples"] * 50
        assert fit["reconstruction_seconds"] > 0
        check = fit["check"]
        assert all(
            math.isfinite(v) and v >= 0 for v in check.values() if isinstance(v, (float, int))
        )
        assert math.isclose(
            check["relative_full_output_error"],
            check["output_error_norm"] / check["target_output_norm"],
            rel_tol=1e-9,
        )
        assert math.isclose(
            check["relative_task_update_error"],
            check["output_error_norm"] / check["task_update_output_norm"],
            rel_tol=1e-9,
        )
    predictions_path = path / "artifacts/predictions.pt"
    raw_predictions_checked = predictions_path.exists()
    if raw_predictions_checked:
        import torch

        assert (
            hashlib.sha256(predictions_path.read_bytes()).hexdigest() == meta["predictions_sha256"]
        )
        predictions = torch.load(predictions_path, map_location="cpu")
        assert len(predictions) == len(rows)
        for row in rows:
            key = (row["preset"], row["split"], row["group"], row["dataset"])
            evidence = predictions[key]
            control = predictions[("unprotected", *key[1:])]
            assert torch.equal(evidence["labels"], control["labels"])
            current = evidence["prediction"] == evidence["labels"]
            baseline = control["prediction"] == control["labels"]
            assert int(current.sum()) == row["correct"]
            assert int((baseline & ~current).sum()) == row["paired"]["lost"]
            assert int((~baseline & current).sum()) == row["paired"]["gained"]
    report = dict(
        status="passed",
        new_evaluations=len(rows),
        new_defense=False,
        raw_predictions_checked=raw_predictions_checked,
        checks=[
            "frozen_protocol_and_parent_provenance",
            "fitting_check_development_confirmation_disjointness",
            "complete_component_joint_and_reconstruction_contrasts",
            "accuracy_and_paired_contingency_arithmetic",
            "comparison_family_accounting",
            "local_full_output_and_task_update_error",
            "server_prediction_fingerprint_and_counts"
            if raw_predictions_checked
            else "raw_predictions_retained_on_server",
        ],
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_reconstruction_budget(path, write_report=True):
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    fitting = json.loads((path / "fitting.json").read_bytes())
    parent = Path(config["parent"])
    fit_parent = Path(config["fitting_source"])
    previous = json.loads((parent / "training_indices.json").read_bytes())
    frozen = json.loads((fit_parent / "training_indices.json").read_bytes())

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert meta["completed"] and not meta["new_defense"]
    assert meta["test_usage"] == config["test_usage"] == "none"
    assert not meta["fitting_labels_used"] and meta["optimizer_steps"] == 0
    assert meta["seed_count"] == 1 and meta["reused_measurements"] == 0
    assert meta["config_sha256"] == digest(path / "protocol.json")
    assert meta["indices_sha256"] == digest(path / "training_indices.json")
    assert meta["parent_indices_sha256"] == digest(parent / "training_indices.json")
    assert meta["fitting_parent_indices_sha256"] == digest(fit_parent / "training_indices.json")
    assert indices["confirmation"] == previous["confirmation"]
    assert indices["development"]["MNIST"] == previous["development"]["MNIST"]
    assert indices["local_check"]["SVHN"] == frozen["SVHN"]["calibration"][:32]
    for budget in config["budgets"]:
        fit = indices["fitting"][str(budget)]
        assert fit == frozen["SVHN"]["attack"][:budget] and len(set(fit)) == budget
        assert not set(fit).intersection(indices["confirmation"]["SVHN"])
        assert not set(fit).intersection(indices["local_check"]["SVHN"])
    groups = {
        ("confirmation", "standalone", "SVHN"),
        ("confirmation", "direct", "SVHN"),
        ("development", "direct", "MNIST"),
    }
    presets = {"unprotected", *[f"fit_{n}" for n in config["budgets"]]}
    expected = {(preset, *key) for preset in presets for key in groups}
    assert len(rows) == meta["new_evaluations"] == 12
    assert {(r["preset"], r["split"], r["group"], r["dataset"]) for r in rows} == expected
    controls = {
        (r["split"], r["group"], r["dataset"]): r for r in rows if r["preset"] == "unprotected"
    }
    for row in rows:
        key = (row["split"], row["group"], row["dataset"])
        control = controls[key]
        n = len(indices[row["split"]][row["dataset"]])
        assert row["n"] == n and 0 <= row["correct"] <= n
        assert row["top1"] == row["correct"] / n
        assert row["coefficient"] == (
            None if row["group"] == "standalone" else config["coefficient"]
        )
        pair = row["paired"]
        assert all(
            isinstance(pair[k], int) and pair[k] >= 0
            for k in ["both_correct", "lost", "gained", "both_wrong"]
        )
        assert sum(pair[k] for k in ["both_correct", "lost", "gained", "both_wrong"]) == n
        assert pair["both_correct"] + pair["gained"] == row["correct"]
        assert pair["both_correct"] + pair["lost"] == control["correct"]
        assert pair["loss_pp"] == 100 * (pair["lost"] - pair["gained"]) / n
        if row["preset"] != "unprotected":
            assert pair["exact_mcnemar_p"] <= pair["holm_p"] <= 1
            assert pair["family_size"] == 3
        if row["preset"] in {"unprotected", "fit_32"}:
            assert row["previous_prediction_exact_match"] is True
    assert len(fitting) == 12
    assert {(f["budget"], f["layer"]) for f in fitting} == {
        (n, layer) for n in config["budgets"] for layer in config["layers"]
    }
    for fit in fitting:
        assert fit["fit"]["calibration_tokens"] == 50 * fit["budget"]
        assert fit["fit"]["design_features"] == 3073
        assert fit["fit"]["solver"] == ("dual" if fit["budget"] == 32 else "primal")
        assert 0 <= fit["fit"]["normal_equation_relative_residual"] < 1e-8
        assert fit["check"]["check_tokens"] == 1600
    raw_checked = (path / "artifacts/predictions.pt").exists()
    if raw_checked:
        import torch
        from operator_diagnostics import signed_margin

        for name, expected_sha in meta["artifact_sha256"].items():
            assert digest(path / "artifacts" / name) == expected_sha
        evidence = torch.load(
            path / "artifacts/predictions.pt", map_location="cpu", weights_only=True
        )
        assert set(evidence) == expected
        old_path = parent / "artifacts/predictions.pt"
        assert digest(old_path) == meta["parent_predictions_sha256"]
        old_evidence = torch.load(old_path, map_location="cpu", weights_only=True)
        for row in rows:
            key = (row["preset"], row["split"], row["group"], row["dataset"])
            raw, control = evidence[key], evidence[("unprotected", *key[1:])]
            assert torch.equal(raw["labels"], control["labels"])
            assert torch.equal(raw["prediction"], raw["logits"].argmax(1))
            current = raw["prediction"] == raw["labels"]
            baseline = control["prediction"] == control["labels"]
            assert int(current.sum()) == row["correct"]
            assert int((baseline & ~current).sum()) == row["paired"]["lost"]
            assert int((~baseline & current).sum()) == row["paired"]["gained"]
            margin = signed_margin(raw["logits"], raw["labels"])
            assert float(margin.median()) == row["margins"]["median"]
            if row["preset"] in {"unprotected", "fit_32"}:
                old_name = "unprotected" if row["preset"] == "unprotected" else "reconstruct_joint"
                assert torch.equal(
                    raw["prediction"], old_evidence[(old_name, *key[1:])]["prediction"]
                )
    report = dict(
        status="passed",
        new_evaluations=len(rows),
        new_defense=False,
        raw_artifacts_checked=raw_checked,
        results_sha256=digest(path / "results.json"),
        checks=[
            "frozen nested fitting budgets and data fingerprints",
            "fitting/evaluation/check disjointness",
            "complete declared grid",
            "paired counts and comparison families",
            "ridge normal equations and unchanged objective",
            "exact previous control predictions",
            "server artifacts, labels, predictions and margins"
            if raw_checked
            else "raw artifacts retained on server",
        ],
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_reconstruction_context(path, write_report=True):
    """Verify declared feature/context controls, including raw server evidence."""
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    fitting = json.loads((path / "fitting.json").read_bytes())
    local = json.loads((path / "local_errors.json").read_bytes())
    shifts = json.loads((path / "input_shift.json").read_bytes())
    parent, fit_parent = Path(config["parent"]), Path(config["fitting_source"])
    previous = json.loads((parent / "training_indices.json").read_bytes())
    frozen = json.loads((fit_parent / "training_indices.json").read_bytes())

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert meta["completed"] and not meta["new_defense"]
    assert config["test_usage"] == meta["test_usage"] == "none"
    assert not meta["fitting_labels_used"] and meta["optimizer_steps"] == 0
    assert meta["seed_count"] == 1 and meta["reused_measurements"] == 0
    assert meta["config_sha256"] == digest(path / "protocol.json")
    assert meta["indices_sha256"] == digest(path / "training_indices.json")
    assert meta["parent_indices_sha256"] == digest(parent / "training_indices.json")
    assert meta["parent_results_sha256"] == digest(parent / "results.json")
    assert meta["fitting_parent_indices_sha256"] == digest(fit_parent / "training_indices.json")
    assert indices["confirmation"] == previous["confirmation"]
    assert indices["development"] == previous["development"]
    assert indices["fitting"]["SVHN"] == previous["fitting"]["256"]
    assert indices["fitting"]["SVHN"] == frozen["SVHN"]["attack"][:256]
    for task in ["MNIST", "SVHN"]:
        checks = indices["local_check"][task]
        assert checks == frozen[task]["calibration"][:32] and len(set(checks)) == 32
        assert not set(checks).intersection(indices["fitting"].get(task, []))
        assert not set(checks).intersection(indices["development"].get(task, []))
        assert not set(checks).intersection(indices["confirmation"].get(task, []))
    assert not set(indices["fitting"]["SVHN"]).intersection(indices["confirmation"]["SVHN"])
    groups = {
        ("confirmation", "standalone", "SVHN"),
        ("confirmation", "direct", "SVHN"),
        ("development", "direct", "MNIST"),
    }
    presets = {"unprotected", *config["conditions"]}
    expected = {(preset, *key) for preset in presets for key in groups}
    assert len(rows) == meta["new_evaluations"] == 3 * len(presets)
    assert {(r["preset"], r["split"], r["group"], r["dataset"]) for r in rows} == expected
    controls = {
        (r["split"], r["group"], r["dataset"]): r for r in rows if r["preset"] == "unprotected"
    }
    for row in rows:
        key = (row["split"], row["group"], row["dataset"])
        n, pair = len(indices[row["split"]][row["dataset"]]), row["paired"]
        assert row["n"] == n and row["top1"] == row["correct"] / n
        assert row["coefficient"] == (
            None if row["group"] == "standalone" else config["coefficient"]
        )
        assert sum(pair[k] for k in ["both_correct", "lost", "gained", "both_wrong"]) == n
        assert pair["both_correct"] + pair["gained"] == row["correct"]
        assert pair["both_correct"] + pair["lost"] == controls[key]["correct"]
        assert pair["loss_pp"] == 100 * (pair["lost"] - pair["gained"]) / n
        if row["preset"] != "unprotected":
            assert pair["family_size"] == len(config["conditions"])
            assert pair["exact_mcnemar_p"] <= pair["holm_p"] <= 1
        if row["preset"] in {"unprotected", "public_expert"}:
            assert row["previous_prediction_exact_match"] is True
    assert len(fitting) == len(config["conditions"]) * len(config["layers"])
    assert {(f["preset"], f["layer"]) for f in fitting} == {
        (p, layer) for p in config["conditions"] for layer in config["layers"]
    }
    assert (
        sum(f["new_fit"] for f in fitting)
        == meta["new_fits"]
        == (len(config["conditions"]) - 1) * len(config["layers"])
    )
    assert sum(not f["new_fit"] for f in fitting) == meta["reused_fits"] == 4
    for f in fitting:
        feature, context = f["preset"].split("_")
        assert f["feature_source"] == feature and f["input_source"] == context
        assert f["new_fit"] == (f["preset"] != "public_expert")
        assert f["fit"]["calibration_tokens"] == 12800
        assert f["fit"]["solver"] == "primal" and f["fit"]["design_features"] == 3073
        assert 0 <= f["fit"]["normal_equation_relative_residual"] < 1e-8
        assert f["fit"]["first_layer"] == (
            "pretrained_fixed" if feature == "public" else "released_control_fixed"
        )
    expected_local = {
        (p, layer, task, context)
        for p in config["conditions"]
        for layer in config["layers"]
        for task in ["MNIST", "SVHN"]
        for context in ["expert", "merged"]
    }
    assert len(local) == len(expected_local)
    assert {
        (r["preset"], r["layer"], r["dataset"], r["input_context"]) for r in local
    } == expected_local
    for r in local:
        assert r["tokens"] == 1600
        for operator in ["expert_operator", "merged_operator"]:
            m = r[operator]
            for denominator, value in [
                ("target_output_norm", "relative_full_output_error"),
                ("task_update_output_norm", "relative_task_update_error"),
            ]:
                assert math.isclose(
                    m[value],
                    m["output_error_norm"] / max(m[denominator], 1e-12),
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                )
    assert len(shifts) == 8
    if "public_selfmerged" in config["conditions"]:
        assert meta["query_budget"] == config["query_budget"]
        query = config["query_budget"]
        assert query["unique_images"] == 256 and query["cached_first_pass"]
        assert query["public_expert_total_target_tokens_per_layer"] == 12800
        assert query["public_selfmerged_total_target_tokens_per_layer"] == 25600
        assert query["new_second_pass_target_tokens_per_layer"] == 12800
    raw_checked = (path / "artifacts/predictions.pt").exists()
    second_pass_contrasts = []
    if raw_checked:
        import torch
        from operator_diagnostics import paired_counts

        for name, fingerprint in meta["artifact_sha256"].items():
            assert digest(path / "artifacts" / name) == fingerprint
        for name, fingerprint in meta["parent_artifact_sha256"].items():
            assert digest(parent / "artifacts" / name) == fingerprint
        evidence = torch.load(
            path / "artifacts/predictions.pt", map_location="cpu", weights_only=True
        )
        prior = torch.load(
            parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True
        )
        assert set(evidence) == expected
        for row in rows:
            key = (row["preset"], row["split"], row["group"], row["dataset"])
            current, baseline = evidence[key], evidence[("unprotected", *key[1:])]
            assert torch.equal(current["labels"], baseline["labels"])
            assert torch.equal(current["prediction"], current["logits"].argmax(1))
            correct = current["prediction"] == current["labels"]
            reference = baseline["prediction"] == baseline["labels"]
            assert int(correct.sum()) == row["correct"]
            for k, value in paired_counts(reference, correct).items():
                assert value == row["paired"][k]
            score_error = float(
                (current["logits"] - baseline["logits"]).norm() / baseline["logits"].norm()
            )
            assert score_error == row["relative_score_error"]
            if row["preset"] in {"unprotected", "public_expert"}:
                old_name = "unprotected" if row["preset"] == "unprotected" else "fit_256"
                assert torch.equal(current["prediction"], prior[(old_name, *key[1:])]["prediction"])
        inputs = torch.load(
            path / "artifacts/check_inputs.pt", map_location="cpu", weights_only=True
        )
        assert set(inputs) == {
            (task, layer, context)
            for task in ["MNIST", "SVHN"]
            for layer in config["layers"]
            for context in ["expert", "merged"]
        }
        for shift in shifts:
            x = inputs[(shift["dataset"], shift["layer"], "expert")]
            y = inputs[(shift["dataset"], shift["layer"], "merged")]
            assert x.shape == y.shape == (1600, 768)
            assert float((y - x).norm() / x.norm()) == shift["relative_input_shift"]
        if "public_selfmerged" in config["conditions"]:
            first_state = torch.load(
                parent / "artifacts/fitted_mlp_256.pt", map_location="cpu", weights_only=True
            )
            second_state = torch.load(
                path / "artifacts/fitted_mlp_public_selfmerged.pt",
                map_location="cpu",
                weights_only=True,
            )
            for layer in config["layers"]:
                for name in ["c_fc.weight", "c_fc.bias"]:
                    assert torch.equal(first_state[layer][name], second_state[layer][name])
            for key in sorted(groups):
                first, second = (
                    evidence[("public_expert", *key)],
                    evidence[("public_selfmerged", *key)],
                )
                counts = paired_counts(
                    first["prediction"] == first["labels"], second["prediction"] == second["labels"]
                )
                second_pass_contrasts.append(
                    dict(
                        split=key[0],
                        group=key[1],
                        dataset=key[2],
                        reference="public_expert",
                        candidate="public_selfmerged",
                        **{k: v for k, v in counts.items() if k != "exact_mcnemar_p"},
                        scope="descriptive comparison on previously inspected training subsets",
                    )
                )
    report = dict(
        status="passed",
        new_defense=False,
        new_evaluations=len(rows),
        local_diagnostic_rows=len(local),
        raw_artifacts_checked=raw_checked,
        results_sha256=digest(path / "results.json"),
        checks=[
            "frozen data and parent fingerprints",
            "complete declared feature/context contrasts",
            "declared image/query budgets and no label use",
            "paired counts and comparison families",
            "ridge normal equations",
            "local error and context-shift arithmetic",
            "exact previous baseline predictions",
            "server predictions and saved inputs"
            if raw_checked
            else "raw evidence retained on server",
        ],
    )
    if second_pass_contrasts:
        report["second_pass_vs_first"] = second_pass_contrasts
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_joint_transfer(path, write_report=True):
    """Recount raw predictions and verify the frozen joint-transfer gate."""
    import torch
    from operator_diagnostics import paired_counts

    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    decision = json.loads((path / "decision.json").read_bytes())
    fitting = json.loads((path / "fitting.json").read_bytes())
    assert meta["completed"] and not meta["new_defense_validated"]
    assert config["test_usage"] == meta["test_usage"] == "none"
    assert meta["optimizer_steps"] == 0 and meta["closed_form_defender_fits"] == 12
    assert not meta["original_protected_checkpoint_available_to_attack"]
    parent_file = Path(config["indices_parent"]) / "training_indices.json"
    old_file = Path(config["exclude_confirmation"])
    parent = json.loads(parent_file.read_bytes())
    old = json.loads(old_file.read_bytes())
    excluded = set(old["confirmation"]["MNIST"])
    for subset in parent["MNIST"].values():
        excluded.update(subset)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(config["seed"])).tolist()
    expected_indices = [i for i in order if i not in excluded][: config["confirmation_samples"]]
    assert indices["confirmation"]["MNIST"] == expected_indices
    assert indices["confirmation"]["SVHN"] == parent["SVHN"]["development"]
    assert indices["calibration"] == parent["MNIST"]["calibration"]
    for key, file in [
        ("config_sha256", path / "protocol.json"),
        ("indices_sha256", path / "training_indices.json"),
        ("parent_indices_sha256", parent_file),
        ("excluded_indices_sha256", old_file),
        ("predictions_sha256", path / "artifacts/predictions.pt"),
        ("release_sha256", path / "artifacts/release.pt"),
    ]:
        assert meta[key] == hashlib.sha256(file.read_bytes()).hexdigest()
    expected = {
        (p, g, t)
        for p in ["unprotected", "joint_transfer"]
        for stage in config["groups"]
        for g, t in [(stage + "_standalone", "MNIST"), (stage, "MNIST"), (stage, "SVHN")]
    }
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    saved = torch.load(path / "artifacts/predictions.pt", map_location="cpu", weights_only=False)
    assert set(lookup) == set(saved) == expected
    assert len(rows) == meta["evaluations"] == len(expected)
    for key, row in lookup.items():
        pred, target = saved[key]["prediction"], saved[key]["labels"]
        reference = saved["unprotected", *key[1:]]
        assert torch.equal(target, reference["labels"])
        assert row["n"] == len(pred) == len(indices["confirmation"][row["dataset"]])
        assert row["correct"] == int(pred.eq(target).sum())
        assert row["top1"] == row["correct"] / row["n"]
        assert row["paired"] == paired_counts(reference["prediction"].eq(target), pred.eq(target))
        assert ((pred >= 0) & (pred < 10)).all()
    loss = 100 * (
        lookup["unprotected", "direct_standalone", "MNIST"]["top1"]
        - lookup["joint_transfer", "direct_standalone", "MNIST"]["top1"]
    )
    gaps = {
        g: 100
        * (lookup["unprotected", g, "MNIST"]["top1"] - lookup["joint_transfer", g, "MNIST"]["top1"])
        for g in config["groups"]
    }
    assert math.isclose(decision["utility_loss_pp"], loss, abs_tol=1e-10)
    assert decision["matched_attack_gap_pp"] == gaps
    best = {
        p: max((lookup[p, g, "MNIST"] for g in config["groups"]), key=lambda r: r["top1"])
        for p in ["unprotected", "joint_transfer"]
    }
    gap = 100 * (best["unprotected"]["top1"] - best["joint_transfer"]["top1"])
    assert math.isclose(gap, decision["best_measured_attack_gap_pp"], abs_tol=1e-10)
    for p, row in best.items():
        assert decision["best_measured_attack"][p] == {
            k: row[k] for k in ["group", "correct", "n", "top1"]
        }
    assert decision["passed_initial_gate"] == (
        loss <= config["utility_budget_pp"] and gap >= config["minimum_attack_gap_pp"]
    )
    reset = "pretrained_attn_mlp_all"
    cost = 100 * (
        lookup["unprotected", "direct", "MNIST"]["top1"]
        - lookup["joint_transfer", reset, "MNIST"]["top1"]
    )
    assert math.isclose(cost, decision["complete_support_reset_cost_pp"], abs_tol=1e-10)
    assert meta["full_support_reset_parameters_and_predictions_exactly_equal"]
    for g, t in [(reset, "MNIST"), (reset, "SVHN"), (reset + "_standalone", "MNIST")]:
        assert torch.equal(
            saved["unprotected", g, t]["prediction"], saved["joint_transfer", g, t]["prediction"]
        )
    release = torch.load(path / "artifacts/release.pt", map_location="cpu", weights_only=True)
    assert set(release) == set(config["layers"]) == set(range(12))
    suffixes = {
        "attn.in_proj_weight",
        "attn.in_proj_bias",
        "attn.out_proj.weight",
        "attn.out_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    }
    for state in release.values():
        assert set(state) == suffixes and all(torch.isfinite(v).all() for v in state.values())
    support = json.loads((path / "release_support.json").read_bytes())["keys"]
    assert set(support) <= {
        f"model.visual.transformer.resblocks.{layer}.{suffix}"
        for layer in range(12)
        for suffix in suffixes
    }
    assert len(fitting) == 12 and [f["layer"] for f in fitting] == list(range(12))
    for fit in fitting:
        assert fit["ridge"] > 0 and fit["normal_equation_relative_residual"] < 1e-8
        for name in ["attention_input_projection", "mlp_input_projection"]:
            assert fit[name]["rank"] == config["representation_rank"]
            assert fit[name]["samples"] == config["calibration_samples"] * 50
            assert fit[name]["parameter_span_residual"] < 1e-8
    report = dict(
        status="passed",
        raw_predictions_checked=True,
        evaluations=len(rows),
        new_defense_validated=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
        checks=[
            "frozen_protocol_and_artifact_hashes",
            "fresh_disjoint_training_confirmation",
            "all_raw_counts_and_paired_tables",
            "complete_matched_controls",
            "best_attack_envelope_gate",
            "full_support_reset_prediction_identity",
            "finite_release_and_projection_fits",
        ],
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def validate_redundancy_erasure(path, write_report=True):
    """Validate the declared redundancy deletion and its raw paired outcomes."""
    import torch
    from operator_diagnostics import paired_counts

    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    parent_file = Path(config["indices_parent"]) / "training_indices.json"
    parent = json.loads(parent_file.read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    decision = json.loads((path / "decision.json").read_bytes())
    edits = json.loads((path / "edits.json").read_bytes())
    assert meta["completed"] and not meta["new_defense_validated"]
    assert meta["test_usage"] == config["test_usage"] == "none"
    assert (
        meta["new_perturbations"]
        == meta["new_fits"]
        == meta["optimizer_steps"]
        == meta["teacher_queries"]
        == 0
    )
    assert not meta["original_protected_task_checkpoint_available_to_attack"]
    for key, filename in [
        ("config_sha256", "protocol.json"),
        ("indices_sha256", "training_indices.json"),
    ]:
        assert meta[key] == hashlib.sha256((path / filename).read_bytes()).hexdigest()
    assert meta["parent_indices_sha256"] == hashlib.sha256(parent_file.read_bytes()).hexdigest()
    assert meta["frozen_release_sha256"] == config["frozen_release_sha256"]
    assert indices["MNIST"] == parent["confirmation"]["MNIST"] and len(indices["MNIST"]) == 4000
    assert indices["SVHN"] == parent["development"]["SVHN"] and len(indices["SVHN"]) == 1000
    presets = ["unprotected", "erasure_only", "frozen_release", "combined"]
    groups = config["groups"]
    expected = {
        (p, g, t)
        for p in presets
        for stage in groups
        for g, t in [(stage + "_standalone", "MNIST"), (stage, "MNIST"), (stage, "SVHN")]
    }
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    assert set(lookup) == expected and len(rows) == meta["new_evaluations"] == len(expected)
    raw = path / "artifacts/predictions.pt"
    assert (
        raw.exists() and meta["predictions_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    )
    saved = torch.load(raw, map_location="cpu", weights_only=False)
    assert set(saved) == expected
    for key, row in lookup.items():
        pred, labels = saved[key]["prediction"], saved[key]["labels"]
        reference = saved["unprotected", *key[1:]]
        assert torch.equal(labels, reference["labels"])
        assert row["n"] == len(labels) == len(indices[row["dataset"]])
        assert row["correct"] == int(pred.eq(labels).sum())
        assert row["top1"] == row["correct"] / row["n"]
        assert row["paired"] == paired_counts(reference["prediction"].eq(labels), pred.eq(labels))
    loss = 100 * (
        lookup["unprotected", "direct_standalone", "MNIST"]["top1"]
        - lookup["combined", "direct_standalone", "MNIST"]["top1"]
    )
    gaps = {
        g: 100
        * (lookup["unprotected", g, "MNIST"]["top1"] - lookup["combined", g, "MNIST"]["top1"])
        for g in groups
    }
    assert math.isclose(decision["utility_loss_pp"], loss, abs_tol=1e-10)
    assert decision["protected_task_gap_pp"] == gaps
    assert decision["passed_initial_gate"] == (
        loss <= config["utility_budget_pp"]
        and min(gaps.values()) >= config["minimum_attack_gap_pp"]
    )
    assert not decision["further_aga_or_reconstruction_run"]
    attn_suffix = {"in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"}
    assert set(edits["erased_attention_keys"]) == {
        f"model.visual.transformer.resblocks.{layer}.attn.{suffix}"
        for layer in range(12)
        for suffix in attn_suffix
    }
    assert len(edits["attack_edits"]) == len(presets) * 2
    for edit in edits["attack_edits"]:
        suffixes = {"c_fc.weight", "c_fc.bias"}
        layers = config["protected_layers"]
        if edit["group"] == "pretrained_mlp_all":
            suffixes |= {"c_proj.weight", "c_proj.bias"}
            layers = list(range(12))
        assert set(edit["keys"]) == {
            f"model.visual.transformer.resblocks.{layer}.mlp.{suffix}"
            for layer in layers
            for suffix in suffixes
        }
    for item, pair in zip(
        edits["full_mlp_reset_identities"],
        [("unprotected", "frozen_release"), ("erasure_only", "combined")],
        strict=True,
    ):
        assert (item["left"], item["right"]) == pair and item[
            "parameters_and_predictions_exactly_equal"
        ]
        for g, t in [
            ("pretrained_mlp_all_standalone", "MNIST"),
            ("pretrained_mlp_all", "MNIST"),
            ("pretrained_mlp_all", "SVHN"),
        ]:
            assert torch.equal(
                saved[pair[0], g, t]["prediction"], saved[pair[1], g, t]["prediction"]
            )
    report = dict(
        status="passed",
        raw_predictions_checked=True,
        new_evaluations=len(rows),
        new_defense_validated=False,
        checks=[
            "frozen_release_and_protocol_fingerprints",
            "exact_parent_indices",
            "complete_controls_and_replacements",
            "raw_correct_counts_and_paired_tables",
            "declared_stop_gate",
            "attention_and_mlp_edit_support",
            "full_mlp_reset_prediction_identity",
        ],
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def validate_context_erasure(path, write_report=True):
    """Check the finite deletion protocol, paired counts and server predictions."""
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    gates = json.loads((path / "gates.json").read_bytes())
    edits = json.loads((path / "edits.json").read_bytes())
    assert meta["completed"] and not meta["new_defense_validated"]
    assert config["test_usage"] == meta["test_usage"] == "none"
    assert (
        meta["config_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        meta["indices_sha256"]
        == hashlib.sha256((path / "training_indices.json").read_bytes()).hexdigest()
    )
    presets = {p["name"]: p for p in config["presets"]}
    assert {g["preset"] for g in gates} == set(presets)
    groups = {("standalone", "SVHN"), ("direct", "SVHN"), ("direct", "MNIST")}
    expected = {(p, g, t) for p in ["unprotected", *presets] for g, t in groups}
    for gate in gates:
        if gate["passed"]:
            expected.update(
                (gate["preset"], g, t)
                for g in ["cleanup", "aga", "reconstruction_1", "reconstruction_2"]
                for t in ["MNIST", "SVHN"]
            )
    keyed = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    assert len(keyed) == len(rows) and set(keyed) == expected
    for gate in gates:
        name = gate["preset"]
        losses = []
        for group in ["standalone", "direct"]:
            losses.append(
                100
                * (keyed["unprotected", group, "SVHN"]["top1"] - keyed[name, group, "SVHN"]["top1"])
            )
        assert abs(gate["utility_loss_pp"] - losses[0]) < 1e-10
        assert abs(gate["direct_loss_pp"] - losses[1]) < 1e-10
        assert gate["passed"] == (
            losses[0] <= config["utility_budget_pp"]
            and losses[1] >= config["minimum_protected_task_drop_pp"]
        )
    assert len(edits) == len(presets) * len(config["layers"])
    for edit in edits:
        assert edit["rank"] == presets[edit["preset"]]["rank"]
        assert edit["layer"] in config["layers"]
        assert edit["erased_norm"] >= 0 and edit["projection_residual"] >= 0
        assert all(math.isfinite(v) and v >= 0 for v in edit["response"].values())
    raw = path / "artifacts/predictions.pt"
    raw_checked = raw.exists()
    if raw_checked:
        import torch
        from operator_diagnostics import paired_counts

        assert meta["predictions_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        saved = torch.load(raw, map_location="cpu", weights_only=False)
        assert set(saved) == expected
        for key, row in keyed.items():
            item = saved[key]
            pred, labels = item["prediction"], item["labels"]
            group = "standalone" if row["group"] == "standalone" else "direct"
            reference = saved["unprotected", group, row["dataset"]]
            assert torch.equal(labels, reference["labels"])
            assert len(labels) == len(indices[row["dataset"]]["development"]) == row["n"]
            correct = pred == labels
            assert int(correct.sum()) == row["correct"]
            counts = paired_counts(reference["prediction"] == labels, correct)
            for name, value in counts.items():
                assert abs(row["paired"][name] - value) < 1e-10
            assert abs(row["top1"] - row["correct"] / row["n"]) < 1e-10
    report = dict(
        status="passed",
        raw_artifacts_checked=raw_checked,
        direct_gate_survivors=sum(g["passed"] for g in gates),
        new_defense_validated=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_subspace_interference(path, write_report=True):
    """Audit the finite perturbation grid and matched-control recovery results."""
    config = json.loads((path / "protocol.json").read_bytes())
    protected_task = config["protected_task"]
    meta = json.loads((path / "metadata.json").read_bytes())
    indices = json.loads((path / "training_indices.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    gates = json.loads((path / "gates.json").read_bytes())
    geometry = json.loads((path / "geometry.json").read_bytes())
    edits = json.loads((path / "edits.json").read_bytes())
    assert meta["completed"] and not meta["new_defense_validated"]
    assert meta["optimizer_steps"] == 0 and not meta["calibration_labels_used"]
    assert config["test_usage"] == meta["test_usage"] == "none"
    assert (
        meta["config_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        meta["indices_sha256"]
        == hashlib.sha256((path / "training_indices.json").read_bytes()).hexdigest()
    )
    presets = {p["name"]: p for p in config["presets"]}
    groups = {("standalone", protected_task), ("direct", "SVHN"), ("direct", "MNIST")}
    expected = {(p, g, t) for p in ["unprotected", *presets] for g, t in groups}
    attack_names = {
        "angular",
        "pretrained_fc",
        "spectral",
        "calibrated",
        "aga",
        "reconstruction_1",
        "reconstruction_2",
    }
    if config.get("aga_all", False):
        attack_names.add("aga_all")
    attacked = {
        g["preset"] for g in gates if g["passed"] or config.get("force_attack_diagnostic", False)
    }
    if attacked:
        for name in {"unprotected", *attacked}:
            expected.update((name, a, t) for a in attack_names for t in ["MNIST", "SVHN"])
            expected.update((name, a + "_standalone", protected_task) for a in attack_names)
    keyed = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    assert len(keyed) == len(rows) and set(keyed) == expected
    assert {g["preset"] for g in gates} == set(presets)
    for g in gates:
        name = g["preset"]
        own = 100 * (
            keyed["unprotected", "standalone", protected_task]["top1"]
            - keyed[name, "standalone", protected_task]["top1"]
        )
        direct = 100 * (
            keyed["unprotected", "direct", protected_task]["top1"]
            - keyed[name, "direct", protected_task]["top1"]
        )
        assert abs(own - g["utility_loss_pp"]) < 1e-10 and abs(direct - g["direct_loss_pp"]) < 1e-10
        assert g["passed"] == (
            own <= config["utility_budget_pp"]
            and direct >= config["minimum_protected_task_drop_pp"]
        )
    geom = {g["layer"]: g for g in geometry}
    assert set(geom) == set(config["layers"])
    assert len(edits) == len(presets) * len(geom)
    for e in edits:
        g = geom[e["layer"]]
        assert g["task_proxy_rank"] == config["task_rank"]
        assert g["perturbation_rank"] == config["perturbation_rank"]
        assert 0 <= g["task_span_leakage"] < 1e-8
        assert g["calibration_tokens"] == config["samples"] * 50
        response = presets[e["preset"]]["own_response_ratio"]
        assert e["own_response_ratio"] == response
        assert (
            abs(e["scale"] * g["unit_expert_response"] / g["expert_preactivation_norm"] - response)
            < 1e-10
        )
        assert (
            abs(
                config["coefficient"]
                * e["scale"]
                * g["unit_merged_response"]
                / g["merged_preactivation_norm"]
                - e["merged_response_ratio"]
            )
            < 1e-10
        )
        assert all(
            math.isfinite(e[k])
            for k in ["scale", "perturbation_norm", "weight_norm_ratio", "anchor_cosine"]
        )
    fitting_audit = []
    for name in sorted({"unprotected", *attacked} if attacked else set()):
        for step in [1, 2]:
            fit_path = path / f"{name}_reconstruction_{step}.json"
            evidence = json.loads(fit_path.read_bytes())
            assert {e["layer"] for e in evidence} == set(config["layers"])
            assert len(evidence) == len(config["layers"])
            for e in evidence:
                assert e["calibration_tokens"] == config["attack_samples"] * 50
                assert e["design_features"] == 3073 and e["first_layer"] == "pretrained_fixed"
                assert 0 < e["ridge"] < math.inf
                assert 0 <= e["normal_equation_relative_residual"] < 1e-8
                assert 0 <= e["fit_feature_error"] < math.inf
            fitting_audit.append(
                dict(
                    file=fit_path.name,
                    sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                    target_token_evaluations=sum(e["calibration_tokens"] for e in evidence),
                    maximum_normal_residual=max(
                        e["normal_equation_relative_residual"] for e in evidence
                    ),
                )
            )
    if config.get("output_compensation", False):
        for name in presets:
            fit_path = path / f"{name}_defender_compensation.json"
            evidence = json.loads(fit_path.read_bytes())
            assert {e["layer"] for e in evidence} == set(config["layers"])
            assert len(evidence) == len(config["layers"])
            for e in evidence:
                assert e["calibration_tokens"] == config["samples"] * 50
                assert e["first_layer"] == "released_control_fixed"
                assert e["design_features"] == 3073 and 0 < e["ridge"] < math.inf
                assert 0 <= e["normal_equation_relative_residual"] < 1e-8
                assert 0 <= e["fit_feature_error"] < math.inf
            fitting_audit.append(
                dict(
                    file=fit_path.name,
                    sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                    target_token_evaluations=sum(e["calibration_tokens"] for e in evidence),
                    maximum_normal_residual=max(
                        e["normal_equation_relative_residual"] for e in evidence
                    ),
                )
            )
    raw = path / "artifacts/predictions.pt"
    raw_checked = raw.exists()
    prediction_agreement = []
    if raw_checked:
        import torch
        from operator_diagnostics import paired_counts

        assert meta["predictions_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        saved = torch.load(raw, map_location="cpu", weights_only=False)
        assert set(saved) == expected
        for key, row in keyed.items():
            pred, labels = saved[key]["prediction"], saved[key]["labels"]
            reference = saved["unprotected", row["group"], row["dataset"]]
            assert torch.equal(labels, reference["labels"])
            assert len(labels) == len(indices[row["dataset"]]["development"]) == row["n"]
            correct = pred == labels
            assert int(correct.sum()) == row["correct"]
            assert abs(row["top1"] - row["correct"] / row["n"]) < 1e-10
            for k, value in paired_counts(reference["prediction"] == labels, correct).items():
                assert abs(row["paired"][k] - value) < 1e-10
            if row["preset"] != "unprotected" and row["group"] not in ["standalone", "direct"]:
                prediction_agreement.append(
                    dict(
                        preset=row["preset"],
                        group=row["group"],
                        dataset=row["dataset"],
                        differing_predictions=int((pred != reference["prediction"]).sum()),
                        n=len(labels),
                    )
                )
    report = dict(
        status="passed",
        raw_artifacts_checked=raw_checked,
        direct_gate_survivors=sum(g["passed"] for g in gates),
        attacked_presets=sorted(attacked),
        fitting_audit=fitting_audit,
        recovery_prediction_agreement=prediction_agreement,
        new_defense_validated=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_subspace_test(path: Path, write_report=True):
    """Check frozen-release full-test counts, labels, provenance and paired comparisons."""
    config = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    assert meta["completed"] and not meta["new_defense_validated"]
    assert meta["test_usage"] == config["test_usage"] == "fixed_full_test_evaluation"
    assert not meta["test_data_used_for_protection_or_reconstruction"]
    assert not meta["protection_recomputed"] and meta["optimizer_steps"] == 0
    assert not meta["calibration_labels_used"]
    assert (
        meta["config_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    )
    task = config["protected_task"]
    name = config["presets"][0]["name"]
    groups = config["recovery_groups"]
    assert groups and len(groups) == len(set(groups))
    assert set(groups) <= {
        "direct",
        "aga_all",
        "aga_selected",
        "pretrained_fc",
        "canonical_fc",
        "pretrained_mlp",
        "pretrained_block",
        "reconstruction_1",
    }
    if "aga_selected" in groups:
        assert (
            config["aga_selected_scope"]
            == "author AGA on the selected expert only; partner unchanged"
        )
        sources = json.loads(Path("configs/sources.json").read_bytes())
        assert meta["selected_aga_normalized_source_sha256"] == sources["AGA"]["sha256"]
        assert (
            meta["selected_aga_helper_sha256"]
            == hashlib.sha256(Path("experiments/eight_task_benchmark.py").read_bytes()).hexdigest()
        )
    expected = {(p, g, t) for p in ["unprotected", name] for g in groups for t in ["MNIST", "SVHN"]}
    expected.update((p, g + "_standalone", task) for p in ["unprotected", name] for g in groups)
    keyed = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    assert len(keyed) == len(rows) and set(keyed) == expected
    counts = {"MNIST": 10000, "SVHN": 26032}
    if "canonical_fc" in groups:
        for preset in ["unprotected", name]:
            audit_bytes = (path / f"{preset}_affine_audit.json").read_bytes()
            assert hashlib.sha256(audit_bytes).hexdigest() == meta["affine_audit_sha256"][preset]
            audit = json.loads(audit_bytes)
            assert (
                audit["binary_diagonal_gauge_state_exact_match"]
                and audit["calibration_images"] == 0
            )
            assert len(audit["operator_checks"]) == 2 * len(config["layers"])
            assert (
                max(r["effective_weight_relative_error"] for r in audit["operator_checks"]) < 1e-6
            )
            assert max(r["effective_bias_relative_error"] for r in audit["operator_checks"]) < 1e-6
    for row in rows:
        assert row["split"] == "test" and row["n"] == counts[row["dataset"]]
        assert 0 <= row["correct"] <= row["n"]
        assert abs(row["top1"] - row["correct"] / row["n"]) < 1e-12
    raw = path / "artifacts/predictions.pt"
    if raw.exists():
        import torch
        from operator_diagnostics import paired_counts

        assert meta["predictions_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
        saved = torch.load(raw, map_location="cpu", weights_only=False)
        assert set(saved) == expected
        for key, row in keyed.items():
            labels, pred = saved[key]["labels"], saved[key]["prediction"]
            assert (
                hashlib.sha256(labels.numpy().tobytes()).hexdigest()
                == meta["test_label_sha256"][row["dataset"]]
            )
            assert len(labels) == row["n"] and int((pred == labels).sum()) == row["correct"]
            control = saved["unprotected", row["group"], row["dataset"]]
            assert torch.equal(labels, control["labels"])
            for field, value in paired_counts(
                control["prediction"] == labels, pred == labels
            ).items():
                assert abs(row["paired"][field] - value) < 1e-10
            if row["group"].startswith(("pretrained_mlp", "pretrained_block")):
                assert torch.equal(pred, control["prediction"])
    support_checked = bool({"pretrained_mlp", "pretrained_block"} & set(groups))
    if support_checked:
        support_file = path / "support_audit.json"
        assert meta["support_reset_checked"]
        assert meta["support_audit_sha256"] == hashlib.sha256(support_file.read_bytes()).hexdigest()
        support = json.loads(support_file.read_bytes())
        assert support["all_release_edits_erased"]
        assert support["restored_state_exactly_matches_reset_control"]
        assert support["all_reset_predictions_exactly_match_control"]
        assert set(support["changed_release_keys"]) <= set(support["reset_keys"])
        assert support["calibration_images"] == 0
        assert not support["protected_original_available_to_attack"]
        if "pretrained_block" in groups:
            block = support["reset_groups"]["pretrained_block"]
            assert block["restored_state_exactly_matches_reset_control"]
            assert block["all_reset_predictions_exactly_match_control"]
            assert set(block["changed_release_keys"]) <= set(block["reset_keys"])
            assert len(block["reset_keys"]) == 12 * len(config["layers"])
    source = Path(config["source_directory"])
    source_checked = (source / "artifacts" / f"{name}.pt").is_file()
    if source_checked:
        assert (
            meta["release_sha256"]
            == hashlib.sha256((source / "artifacts" / f"{name}.pt").read_bytes()).hexdigest()
        )
        assert (
            meta["training_indices_sha256"]
            == hashlib.sha256((source / "training_indices.json").read_bytes()).hexdigest()
        )
    report = dict(
        status="passed",
        rows=len(rows),
        raw_artifacts_checked=raw.exists(),
        frozen_release_checked=source_checked,
        support_reset_checked=support_checked,
        new_defense_validated=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def validate_joint_diagnostics(path: Path, write_report=True):
    """Recount fixed-release calibration/confirmation predictions and fit replay."""
    import torch
    from operator_diagnostics import paired_counts

    meta = json.loads((path / "metadata.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    errors = json.loads((path / "layer_errors.json").read_bytes())
    parent = Path(protocol["parent"])
    indices = json.loads((parent / "training_indices.json").read_bytes())
    fitting = json.loads((parent / "fitting.json").read_bytes())
    assert meta["completed"] and meta["no_parameters_fitted_or_selected"]
    assert meta["calibrated_block_targets_replayed"]
    assert protocol["subsets"] == {
        "calibration": indices["calibration"],
        "confirmation_prefix_256": indices["confirmation"]["MNIST"][:256],
    }
    assert not set(protocol["subsets"]["calibration"]) & set(
        protocol["subsets"]["confirmation_prefix_256"]
    )
    for key, file in [
        ("parent_release_sha256", parent / "artifacts/release.pt"),
        ("parent_protocol_sha256", parent / "protocol.json"),
        ("parent_indices_sha256", parent / "training_indices.json"),
        ("predictions_sha256", path / "artifacts/predictions.pt"),
    ]:
        assert meta[key] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert (
        meta["source_sha256"]
        == hashlib.sha256(Path("experiments/joint_diagnostics.py").read_bytes()).hexdigest()
    )
    for name, value in meta["helpers_sha256"].items():
        assert value == hashlib.sha256((Path("experiments") / name).read_bytes()).hexdigest()
    raw = torch.load(path / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    expected = {(s, g) for s in protocol["subsets"] for g in ["unprotected", "joint_transfer"]}
    assert len(rows) == len(raw) == 4 and set(raw) == expected
    assert {(r["subset"], r["group"]) for r in rows} == expected
    for row in rows:
        item = raw[row["subset"], row["group"]]
        pred, labels = item["prediction"], item["labels"]
        assert pred.shape == labels.shape == (256,) and row["n"] == 256
        assert int(pred.eq(labels).sum()) == row["correct"]
        assert row["top1"] == row["correct"] / row["n"]
        reference = raw[row["subset"], "unprotected"]
        assert torch.equal(labels, reference["labels"])
        if row["group"] == "unprotected":
            assert row["paired"] is None
        else:
            assert row["paired"] == paired_counts(
                reference["prediction"].eq(labels), pred.eq(labels)
            )
    assert len(errors) == 24
    assert {(e["subset"], e["layer"]) for e in errors} == {
        (s, layer) for s in protocol["subsets"] for layer in range(12)
    }
    for row in errors:
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
        assert math.isclose(
            row["residual_target_relative_error"] * row["residual_target_to_teacher_norm"],
            row["original_block_relative_error"],
            rel_tol=1e-9,
        )
        assert (
            row["recorded_calibration_residual"]
            == fitting[row["layer"]]["residual_target_relative_error"]
        )
        if row["subset"] == "calibration":
            assert math.isclose(
                row["residual_target_relative_error"],
                row["recorded_calibration_residual"],
                rel_tol=2e-5,
            )
    last = next(e for e in errors if e["subset"] == "calibration" and e["layer"] == 11)
    original_check = json.loads((parent / "calibration_check.json").read_bytes())
    assert math.isclose(
        last["actual_block_relative_error"],
        original_check["final_block_relative_error"],
        rel_tol=1e-8,
    )
    report = dict(
        status="passed",
        rows=4,
        raw_predictions_checked=True,
        no_refit_or_new_candidate=True,
        confirmation_previously_inspected=True,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=[
            "exact frozen parent release",
            "subset indices and disjointness",
            "raw paired prediction recount",
            "calibrated block target replay",
            "finite relative errors and norm identities",
            "runner and helper source hashes",
        ],
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_eight_protection(path: Path, write_report=True):
    """Audit each protected expert against its matching author-AGA control."""
    import torch
    from operator_diagnostics import paired_counts

    config = json.loads(Path("configs/eight_tasks.json").read_bytes())
    protocol = json.loads((path / "protocol.json").read_bytes())
    meta = json.loads((path / "metadata.json").read_bytes())
    rows = json.loads((path / "results.json").read_bytes())
    audits = json.loads((path / "task_audits.json").read_bytes())
    hashes = json.loads((path / "artifact_hashes.json").read_bytes())
    sources = json.loads(Path("configs/sources.json").read_bytes())
    expected_hashes = {
        entry["path"].removeprefix("ViT-B-32/"): entry["sha256"]
        for entry in json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    }
    tasks = protocol["evaluation_tasks"]
    assert len(tasks) == len(set(tasks)) == 7
    assert set(tasks) == set(config["tasks"]) - {"EuroSAT"}
    assert meta["completed"] and not meta["table6_reproduction"]
    assert not meta["eight_task_average_reported"] and not protocol["table6_reproduction"]
    assert protocol["tasks"] == meta["merge_tasks"] == config["tasks"]
    assert protocol["coefficient"] == meta["coefficient"] == config["coefficient"] == 0.2
    assert protocol["missing_evaluation_task"] == "EuroSAT"
    assert protocol["protection"] == "official_PaRaMS" and protocol["params_max_iter"] == 1
    assert protocol["protection_scope"] == "one expert at a time"
    assert protocol["sources"] == sources
    assert meta["evaluation_tasks"] == tasks
    assert (
        meta["protocol_sha256"] == hashlib.sha256((path / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        meta["source_sha256"]
        == hashlib.sha256(Path("experiments/eight_task_benchmark.py").read_bytes()).hexdigest()
    )
    for name, value in meta["helpers_sha256"].items():
        assert value == hashlib.sha256((Path("experiments") / name).read_bytes()).hexdigest()
    assert meta["upstream_source_sha256"]["PaRaMS"] == sources["PaRaMS"]["sha256"]
    assert meta["upstream_normalized_source_sha256"]["AGA"] == sources["AGA"]["sha256"]
    required = (
        {"zeroshot.pt"}
        | {f"{t}/finetuned.pt" for t in config["tasks"]}
        | {f"head_{t}.pt" for t in tasks}
    )
    assert set(meta["checkpoint_sha256"]) == required
    for name, value in meta["checkpoint_sha256"].items():
        assert value == expected_hashes[name]
    references = {
        "unprotected_standalone": None,
        "unprotected_merge": None,
        "params_standalone": "unprotected_standalone",
        "params_merge": "unprotected_merge",
        "AGA_unprotected_standalone": "unprotected_standalone",
        "AGA_unprotected_merge": "unprotected_merge",
        "AGA_params_standalone": "AGA_unprotected_standalone",
        "AGA_params_merge": "AGA_unprotected_merge",
    }
    assert len(rows) == 56 and len(audits) == 7 and len(hashes) == 21
    assert [a["task"] for a in audits] == tasks
    assert len({(r["dataset"], r["group"]) for r in rows}) == 56
    evidence = []
    block_keys = {
        "attn.in_proj_weight",
        "attn.in_proj_bias",
        "attn.out_proj.weight",
        "attn.out_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    }
    for task, audit in zip(tasks, audits, strict=True):
        for suffix in ["params", "AGA_params", "predictions"]:
            raw = path / "artifacts" / f"{task}_{suffix}.pt"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == hashes[raw.name]
            saved = torch.load(raw, map_location="cpu", weights_only=True)
            if suffix != "predictions":
                assert set(saved) == set(range(12))
                assert all(set(block) == block_keys for block in saved.values())
                assert all(
                    torch.isfinite(v).all() for block in saved.values() for v in block.values()
                )
        assert set(saved) == set(references)
        task_rows = [r for r in rows if r["dataset"] == task]
        assert [r["group"] for r in task_rows] == list(references)
        target = saved["unprotected_standalone"]["labels"]
        for row in task_rows:
            group = row["group"]
            pred, labels = saved[group]["prediction"], saved[group]["labels"]
            assert row["stage"] == "eight_expert_protection_check"
            assert row["protected_task"] == task and row["merge_tasks"] == config["tasks"]
            assert row["coefficient"] == 0.2 and row["n"] == config["expected_test_counts"][task]
            assert pred.shape == labels.shape == (row["n"],) and torch.equal(labels, target)
            assert int(pred.eq(labels).sum()) == row["correct"]
            assert row["top1"] == row["correct"] / row["n"]
            assert row["reference_group"] == references[group]
            if references[group] is None:
                assert row["paired"] is None
            else:
                ref = saved[references[group]]["prediction"]
                assert row["paired"] == paired_counts(ref.eq(labels), pred.eq(labels))
        changes = int(
            saved["params_standalone"]["prediction"]
            .ne(saved["unprotected_standalone"]["prediction"])
            .sum()
        )
        assert audit["utility_prediction_changes"] == changes
        evidence.append(
            dict(
                task=task,
                n=len(target),
                utility_prediction_changes=changes,
                aga_merge_prediction_changes=int(
                    saved["AGA_params_merge"]["prediction"]
                    .ne(saved["AGA_unprotected_merge"]["prediction"])
                    .sum()
                ),
                aga_standalone_prediction_changes=int(
                    saved["AGA_params_standalone"]["prediction"]
                    .ne(saved["AGA_unprotected_standalone"]["prediction"])
                    .sum()
                ),
            )
        )
    report = dict(
        status="passed",
        rows=len(rows),
        protected_experts=len(tasks),
        examples_per_group=sum(config["expected_test_counts"][t] for t in tasks),
        raw_predictions_checked=True,
        artifact_hashes_checked=21,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
        matched_prediction_evidence=evidence,
        checks=[
            "fixed eight-expert fusion and coefficient",
            "seven full official test splits",
            "paired prediction counts",
            "raw artifact hashes and finite block states",
            "author source pins with explicit line-ending normalization",
            "frozen runner/helper source hashes",
            "all input checkpoint manifest hashes",
        ],
        full_eight_task_accuracy_reported=False,
        paper_table_reproduced=False,
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def validate_eight_task_resources(path: Path, write_report=True):
    """Audit test counts and saved predictions without claiming full reproduction."""
    config = json.loads(Path("configs/eight_tasks.json").read_bytes())
    checks = []
    for prefix in ["expert", "merged_resource"]:
        for result in sorted(path.glob(f"*/{prefix}_checks.json")):
            folder = result.parent
            rows = json.loads(result.read_bytes())
            meta = json.loads((folder / f"{prefix}_check_metadata.json").read_bytes())
            source = json.loads((folder / "source_validation.json").read_bytes())
            assert source["source_sha256"] == meta["source_sha256"]
            assert source["matches_committed_source"]
            assert meta["completed"] and not meta["eight_task_result"]
            assert [r["dataset"] for r in rows] == meta["tasks"]
            assert len(meta["tasks"]) == len(set(meta["tasks"]))
            raw = folder / f"{prefix}_predictions.pt"
            saved = None
            if raw.exists():
                import torch

                assert hashlib.sha256(raw.read_bytes()).hexdigest() == meta["predictions_sha256"]
                saved = torch.load(raw, map_location="cpu", weights_only=True)
                assert set(saved) == set(meta["tasks"])
            for row in rows:
                task = row["dataset"]
                assert row["stage"] == "resource_validation" and not row["eight_task_result"]
                assert row["n"] == config["expected_test_counts"][task]
                assert 0 <= row["correct"] <= row["n"]
                assert row["top1"] == row["correct"] / row["n"]
                if prefix == "merged_resource":
                    assert row["group"] == "eight_expert_resource_check"
                    assert row["merge_tasks"] == meta["merge_tasks"] == config["tasks"]
                    assert len(row["merge_tasks"]) == 8
                    assert row["coefficient"] == meta["coefficient"] == config["coefficient"]
                else:
                    assert row["group"] == "official_expert_check"
                if saved is not None:
                    pred, labels = saved[task]["prediction"], saved[task]["labels"]
                    assert pred.shape == labels.shape == (row["n"],)
                    assert int((pred == labels).sum()) == row["correct"]
            checks.append(
                dict(
                    directory=folder.name,
                    rows=len(rows),
                    raw_predictions_checked=saved is not None,
                    results_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                    source_commit=source["source_commit"],
                )
            )
    assert checks, "No completed resource evaluations"
    report = dict(
        status="passed",
        checks=checks,
        all_raw_predictions_checked=all(c["raw_predictions_checked"] for c in checks),
        full_eight_task_accuracy_reported=False,
        paper_table_reproduced=False,
    )
    if write_report:
        (path / "resource_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


def validate_class_recompile(path: Path):
    """Recount predictions and check the saved fit and class geometry independently."""
    import torch
    from operator_diagnostics import paired_counts

    torch.set_num_threads(4)

    def read(name):
        return json.loads((path / name).read_bytes())

    meta, config, decision = read("metadata.json"), read("protocol.json"), read("decision.json")
    rows, indices = read("results.json"), read("training_indices.json")
    assert meta["completed"] and meta["optimizer_steps"] == 0 and meta["closed_form_fits"] == 1
    assert not meta["calibration_labels_used"] and meta["test_usage"] == "none"
    for filename, expected in meta["artifacts_sha256"].items():
        assert hashlib.sha256((path / "artifacts" / filename).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256(Path("experiments/class_response_recompile.py").read_bytes()).hexdigest()
        == meta["source_sha256"]
    )
    assert (
        hashlib.sha256(Path("configs/class_response_recompile.json").read_bytes()).hexdigest()
        == meta["config_sha256"]
    )
    for filename, expected in meta["helpers_sha256"].items():
        assert hashlib.sha256((Path("experiments") / filename).read_bytes()).hexdigest() == expected
    calibration, confirmation = set(indices["calibration"]), set(indices["confirmation"]["MNIST"])
    assert len(calibration) == config["calibration_samples"]
    assert len(confirmation) == config["confirmation_samples"] and not calibration & confirmation
    parent = json.loads((Path(config["index_parent"]) / "training_indices.json").read_bytes())
    previous = set().union(*(set(v) for v in parent["MNIST"].values()))
    for folder in config["exclude_confirmations"]:
        previous.update(
            json.loads((Path(folder) / "training_indices.json").read_bytes())["confirmation"][
                "MNIST"
            ]
        )
    assert not (calibration | confirmation) & previous
    assert indices["confirmation"]["SVHN"] == parent["SVHN"]["development"]
    predictions = torch.load(
        path / "artifacts/predictions.pt", map_location="cpu", weights_only=False
    )
    assert len(predictions) == len(rows)
    for row in rows:
        item = predictions[row["preset"], row["group"], row["dataset"]]
        p, y = item["prediction"], item["labels"]
        ref = predictions["unprotected", row["group"], row["dataset"]]
        assert torch.equal(y, ref["labels"])
        assert row["n"] == len(y) == len(indices["confirmation"][row["dataset"]])
        assert row["correct"] == int(p.eq(y).sum()) and row["top1"] == row["correct"] / row["n"]
        assert row["paired"] == paired_counts(ref["prediction"].eq(y), p.eq(y))
    utility = {r["preset"]: r["top1"] for r in rows if r["group"] == "standalone"}
    loss = 100 * (utility["unprotected"] - utility["recompiled"])
    assert math.isclose(loss, decision["utility_loss_pp"], abs_tol=1e-12)
    assert decision["utility_passed"] == (loss <= config["utility_budget_pp"])
    assert len(rows) == (2 + 4 * len(config["groups"]) if decision["utility_passed"] else 2)
    if decision["utility_passed"]:
        best = {
            p: max(
                r["top1"]
                for r in rows
                if r["preset"] == p and r["group"] != "standalone" and r["dataset"] == "MNIST"
            )
            for p in utility
        }
        gap = 100 * (best["unprotected"] - best["recompiled"])
        assert math.isclose(gap, decision["best_measured_attack_gap_pp"], abs_tol=1e-12)
        assert decision["attack_gate_passed"] == (gap >= config["minimum_attack_gap_pp"])
        for task in indices["confirmation"]:
            a = predictions["unprotected", "pretrained_attn_mlp_all", task]["prediction"]
            b = predictions["recompiled", "pretrained_attn_mlp_all", task]["prediction"]
            assert torch.equal(a, b)
    else:
        assert read("attack_audits.json") == []
    assert not decision["new_defense_validated"] and not decision["functional_recovery_tested"]
    fit = torch.load(path / "artifacts/fitting.pt", map_location="cpu", weights_only=False)
    h, target, delta = fit["design"], fit["target_delta"], fit["update"]
    ridge = config["ridge_ratio"] * h.square().sum() / len(h)
    normal_error = float(
        (h.T @ (h @ delta - target) + ridge * delta).norm() / (h.T @ target).norm().clamp_min(1e-30)
    )
    assert normal_error < 1e-8

    def unit(z):
        center = z - z.mean(1, keepdim=True)
        return center / (center.square().mean(1, keepdim=True) + fit["epsilon"]).sqrt()

    response_error = float(
        (
            unit(fit["target"]) @ fit["class_directions"]
            - unit(fit["teacher"]) @ fit["class_directions"]
        ).norm()
        / (unit(fit["teacher"]) @ fit["class_directions"]).norm()
    )
    assert response_error < 1e-10
    for operation in [
        lambda z: z.mean(1),
        lambda z: (z - z.mean(1, keepdim=True)).square().mean(1),
    ]:
        assert torch.allclose(
            operation(fit["target"]), operation(fit["base"]), rtol=1e-10, atol=1e-10
        )
    release = torch.load(path / "artifacts/release.pt", map_location="cpu", weights_only=False)
    output = fit["before"].float() + delta.float()
    assert torch.equal(release[11]["mlp.c_proj.weight"], output[:-1].T)
    assert torch.equal(release[11]["mlp.c_proj.bias"], output[-1])
    fitted = fit["base"] + h @ delta
    target_error = float((fitted - fit["target"]).norm() / fit["target"].norm())
    class_error = float(
        ((unit(fitted) - unit(fit["teacher"])) @ fit["class_directions"]).norm()
        / (unit(fit["teacher"]) @ fit["class_directions"]).norm()
    )
    report = dict(
        status="passed",
        raw_prediction_rows_checked=len(rows),
        excluded_subsets_checked=True,
        normal_equation_relative_residual=normal_error,
        exact_target_response_relative_error=response_error,
        calibration_fitted_target_relative_error=target_error,
        calibration_fitted_class_difference_relative_error=class_error,
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts_sha256=meta["artifacts_sha256"],
        utility_gate_verified=True,
        new_defense_validated=False,
        functional_recovery_tested=False,
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def validate_task_support(path: Path):
    """Audit the MNIST-only selection, deletion certificate and public recovery floor."""
    import torch
    from operator_diagnostics import paired_counts

    def read(name):
        return json.loads((path / name).read_bytes())

    meta, config = read("metadata.json"), read("protocol.json")
    rows, selection = read("results.json"), read("selection.json")
    indices, trace, decision = (
        read("training_indices.json"),
        read("pruning_trace.json"),
        read("decision.json"),
    )
    assert (
        meta["completed"]
        and not meta["new_defense"]
        and meta["fits"] == meta["optimizer_steps"] == 0
    )
    assert meta["test_usage"] == "none" and meta["partner_loaded_after_selection"]
    for filename, key in [
        ("selection.json", "selection_sha256"),
        ("training_indices.json", "indices_sha256"),
        ("artifacts/predictions.pt", "predictions_sha256"),
    ]:
        assert hashlib.sha256((path / filename).read_bytes()).hexdigest() == meta[key]
    assert (
        hashlib.sha256(Path("experiments/operator_diagnostics.py").read_bytes()).hexdigest()
        == meta["source_sha256"]
    )
    assert (
        hashlib.sha256(Path("configs/task_support_mnist.json").read_bytes()).hexdigest()
        == meta["config_sha256"]
    )
    for name, sha in meta["helper_sha256"].items():
        assert hashlib.sha256((Path("experiments") / name).read_bytes()).hexdigest() == sha
    assert set(meta["selection_checkpoint_sha256"]) == {
        "zeroshot.pt",
        "MNIST/finetuned.pt",
        "head_MNIST.pt",
    }
    assert set(meta["evaluation_only_checkpoint_sha256"]) == {"SVHN/finetuned.pt"}
    parent = json.loads((Path(config["index_parent"]) / "training_indices.json").read_bytes())
    assert (
        indices["selection"]["MNIST"]
        == parent["MNIST"]["development"][: config["selection_samples"]]
    )
    excluded = set().union(*(set(v) for v in parent["MNIST"].values()))
    for folder in config["exclude_runs"]:
        saved = json.loads((Path(folder) / "training_indices.json").read_bytes())
        excluded.update(saved["confirmation"]["MNIST"])
        excluded.update(saved.get("calibration", []))
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(config["seed"])).tolist()
    expected = [i for i in order if i not in excluded][: config["confirmation_samples"]]
    assert indices["confirmation"]["MNIST"] == expected and not set(expected) & excluded
    raw = torch.load(path / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(raw) == len(rows)
    controls, by_id, supports = {}, {}, {}
    for row in rows:
        item = raw[row["id"]]
        p, y = item["prediction"], item["labels"]
        key = row["split"], row["group"]
        if not row["support"]:
            controls[key] = item
        ref = controls[key]
        assert torch.equal(y, ref["labels"])
        assert row["n"] == len(y) == len(indices[row["split"]]["MNIST"])
        assert row["correct"] == int(p.eq(y).sum()) and row["top1"] == row["correct"] / row["n"]
        assert row["paired"] == paired_counts(ref["prediction"].eq(y), p.eq(y))
        by_id[row["id"]] = row
        if row["split"] == "selection":
            supports[tuple(map(tuple, row["support"]))] = row
    baseline = supports[()]

    def loss(row):
        return 100 * (baseline["top1"] - row["top1"])

    modules = {(layer, kind) for layer in range(12) for kind in ["attn", "mlp"]}
    assert all((module,) in supports for module in modules)
    assert all(((layer, "attn"), (layer, "mlp")) in supports for layer in range(12))
    assert len(supports) == selection["selection_variants"] <= config["maximum_selection_variants"]
    active = set(modules)
    for step in trace:
        assert set(map(tuple, step["before"])) == active
        trial = active - {tuple(step["removed"])}
        row = by_id[step["measurement"]]
        assert set(map(tuple, step["trial"])) == set(map(tuple, row["support"])) == trial
        assert step["accepted"] == (loss(row) >= config["minimum_task_drop_pp"])
        if step["accepted"]:
            active = trial
    assert sorted(active) == list(map(tuple, selection["support"]))
    assert (
        selection["modules"] == len(active)
        and not selection["globally_minimum"]
        and not selection["partner_loaded"]
    )
    assert math.isclose(
        loss(supports[tuple(sorted(active))]), selection["standalone_loss_pp"], abs_tol=1e-12
    )
    if selection["inclusion_minimal"]:
        assert (
            not selection["budget_stopped"]
            and loss(supports[tuple(sorted(active))]) >= config["minimum_task_drop_pp"]
        )
        for certificate in selection["deletion_certificate"]:
            row = by_id[certificate["measurement"]]
            assert set(map(tuple, row["support"])) == active - {tuple(certificate["removed"])}
            assert math.isclose(loss(row), certificate["task_loss_pp"], abs_tol=1e-12)
            assert loss(row) < config["minimum_task_drop_pp"]
    fusion = [row for row in rows if row["split"] == "confirmation" and row["group"] == "fusion"]
    assert len(fusion) == 2
    gap = 100 * (fusion[0]["top1"] - fusion[1]["top1"])
    assert decision["public_support_reset_accuracy"] == fusion[1]["top1"]
    assert math.isclose(gap, decision["direct_control_minus_recovery_floor_pp"], abs_tol=1e-12)
    assert decision["necessary_support_gate_passed"] == (gap >= config["minimum_fusion_drop_pp"])
    assert not decision["new_defense"]
    report = dict(
        status="passed",
        raw_rows_checked=len(rows),
        selection_variants=len(supports),
        subset_exclusion_verified=True,
        deletion_trace_verified=True,
        inclusion_minimal=selection["inclusion_minimal"],
        global_minimality_claim=False,
        partner_used_for_selection=False,
        public_reset_floor_verified=True,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_defense=False,
    )
    (path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def validate_task_profile(path: Path):
    """Retally raw predictions, verify labels, fixed selection and public reset support."""
    import itertools
    import sys
    import types

    import torch
    from clip_benchmark import checkpoint_loader
    from operator_diagnostics import paired_counts

    def read(name):
        return json.loads((path / name).read_bytes())

    def sha(file):
        return hashlib.sha256(Path(file).read_bytes()).hexdigest()

    meta, config, rows = read("metadata.json"), read("protocol.json"), read("results.json")
    selection, indices, decisions = (
        read("selection.json"),
        read("training_indices.json"),
        read("decision.json"),
    )
    audit, freeze = read("reset_audit.json"), read("selection_freeze.json")
    assert meta["completed"] and not meta["new_defense"] and meta["test_usage"] == "none"
    assert meta["fits"] == meta["optimizer_steps"] == 0 and meta["partner_loaded_after_selection"]
    assert meta["source_sha256"] == sha("experiments/operator_diagnostics.py")
    assert meta["config_sha256"] == sha("configs/task_profile.json")
    for name, expected in meta["helper_sha256"].items():
        assert sha(Path("experiments") / name) == expected
    for name, expected in meta["upstream_sha256"].items():
        assert sha(Path(meta["author_repo"]) / name) == expected
    manifest = json.loads(Path("configs/checkpoints.json").read_bytes())
    manifest_hashes = {
        item["path"].removeprefix("ViT-B-32/"): item["sha256"] for item in manifest["files"]
    }
    for name, expected in meta["checkpoint_sha256"].items():
        assert sha(Path(meta["checkpoint_root"]) / name) == expected == manifest_hashes[name]
    assert sha("configs/checkpoints.json") == meta["manifest_sha256"]
    for filename, key in [
        ("selection.json", "selection_sha256"),
        ("training_indices.json", "indices_sha256"),
        ("artifacts/predictions.pt", "predictions_sha256"),
    ]:
        assert sha(path / filename) == meta[key]
    assert freeze == dict(
        selection_sha256=meta["selection_sha256"], rows_before_partner=132, partner_loaded=False
    )
    raw = torch.load(path / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    assert len(rows) == len(raw) == meta["row_count"] == 162
    excluded = set()
    for folder in config["mnist_exclusions"]:
        saved = json.loads((Path(folder) / "training_indices.json").read_bytes())
        if "MNIST" in saved:
            value = saved["MNIST"]
            excluded.update(
                itertools.chain.from_iterable(value.values()) if isinstance(value, dict) else value
            )
        for key in ["selection", "confirmation"]:
            if key in saved and "MNIST" in saved[key]:
                excluded.update(saved[key]["MNIST"])
        excluded.update(
            saved.get("calibration", []) if isinstance(saved.get("calibration", []), list) else []
        )
    assert set(read("excluded_mnist_indices.json")) == excluded
    checkpoint_loader(Path(meta["author_repo"]))
    package = types.ModuleType("dataset")
    package.__path__ = [str((Path(meta["author_repo"]) / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    expected_family = [("none", ())]
    expected_family += [(name, (i,)) for i in range(12) for name in ["attn", "mlp", "attn_mlp"]]
    expected_family += [("attn_mlp", tuple(w)) for w in config["windows"]]
    expected_family += [(name, tuple(range(12))) for name in ["attn", "mlp", "attn_mlp"]]
    for offset, task in enumerate(config["tasks"]):
        dataset = get_dataset(task, None, meta["data"]).train_dataset
        targets = getattr(dataset, "targets", None)
        if targets is None:
            targets = [target for _, target in dataset._samples]
        y_all = torch.as_tensor(targets, dtype=torch.int64)
        assert len(dataset) == meta["dataset_metadata"][task]["train_count"]
        assert (
            hashlib.sha256(y_all.numpy().tobytes()).hexdigest()
            == meta["dataset_metadata"][task]["labels_sha256"]
        )
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(config["seed"] + offset)
        ).tolist()
        allowed = [i for i in order if task != "MNIST" or i not in excluded]
        n, m = config["selection_samples"], config["confirmation_samples"][task]
        assert indices[task] == dict(selection=allowed[:n], confirmation=allowed[n : n + m])
        assert not set(indices[task]["selection"]) & set(indices[task]["confirmation"])
        profile = [r for r in rows if r["task"] == task and r["split"] == "selection"]
        assert [(r["component"], tuple(r["layers"])) for r in profile] == expected_family
        assert all(r["group"] == "standalone" for r in profile)
        candidates = [r for r in profile if r["component"] == "attn_mlp" and len(r["layers"]) <= 3]
        base = profile[0]
        for row in profile[1:]:
            assert abs(row["standalone_loss_pp"] - 100 * (base["top1"] - row["top1"])) < 1e-9
        eligible = [
            r for r in candidates if r["standalone_loss_pp"] >= config["minimum_task_drop_pp"]
        ]
        chosen = (
            min(eligible, key=lambda r: (len(r["layers"]), -r["standalone_loss_pp"], r["layers"]))
            if eligible
            else max(candidates, key=lambda r: (r["standalone_loss_pp"], [-i for i in r["layers"]]))
        )
        item = selection[task]
        assert item["measurement_id"] == chosen["id"] and item["layers"] == chosen["layers"]
        assert item["selection_gate_passed"] == bool(eligible) and not item["partner_used"]
        assert item["inputs"] == [
            f"{task}/finetuned.pt",
            "zeroshot.pt",
            f"head_{task}.pt",
            f"{task} training selection labels",
        ]
        for split, group in [
            ("selection", "standalone"),
            ("confirmation", "standalone"),
            ("confirmation", "fusion"),
        ]:
            values = [
                r for r in rows if (r["task"], r["split"], r["group"]) == (task, split, group)
            ]
            assert len(values) == (44 if split == "selection" else 5)
            reference = raw[next(r for r in values if r["component"] == "none")["id"]]
            expected_labels = y_all[indices[task][split]]
            for row in values:
                evidence = raw[row["id"]]
                assert torch.equal(evidence["labels"], expected_labels)
                assert evidence["prediction"].shape == expected_labels.shape
                correct = evidence["prediction"].eq(expected_labels)
                assert row["correct"] == int(correct.sum()) and row["n"] == len(expected_labels)
                assert row["top1"] == row["correct"] / row["n"]
                assert row["paired"] == paired_counts(
                    reference["prediction"].eq(expected_labels), correct
                )
        confirmation = [r for r in rows if r["task"] == task and r["split"] == "confirmation"]
        lookup = {(r["group"], r["component"]): r for r in confirmation}
        for component in ["attn", "mlp", "attn_mlp", "block"]:
            keys = audit[task][component]["keys"]
            assert (
                len(keys)
                == len(item["layers"])
                * {"attn": 4, "mlp": 4, "attn_mlp": 8, "block": 12}[component]
            )
            assert len(keys) == len(set(keys))
            assert all(
                any(
                    k.startswith(f"model.visual.transformer.resblocks.{i}.") for i in item["layers"]
                )
                for k in keys
            )
            assert (
                audit[task][component]["replaced_values_match_anchor"]
                and audit[task][component]["outside_support_matches_original"]
            )
            floor = decisions[task]["floors"][component]
            assert floor["fusion_accuracy"] == lookup["fusion", component]["top1"]
            assert (
                abs(
                    floor["standalone_loss_pp"]
                    - 100
                    * (
                        lookup["standalone", "none"]["top1"]
                        - lookup["standalone", component]["top1"]
                    )
                )
                < 1e-9
            )
            assert (
                abs(
                    floor["direct_minus_reset_pp"]
                    - 100 * (lookup["fusion", "none"]["top1"] - lookup["fusion", component]["top1"])
                )
                < 1e-9
            )
        dec = decisions[task]
        assert not dec["new_defense"] and dec["selection_gate_passed"] == bool(eligible)
        assert dec["confirmation_task_gate_passed"] == (
            dec["floors"]["attn_mlp"]["standalone_loss_pp"] >= config["minimum_task_drop_pp"]
        )
        assert dec["necessary_support_gate_passed"] == (
            dec["floors"]["attn_mlp"]["direct_minus_reset_pp"] >= config["minimum_fusion_drop_pp"]
        )
        assert dec["full_block_gate_passed"] == (
            dec["floors"]["block"]["direct_minus_reset_pp"] >= config["minimum_fusion_drop_pp"]
        )
    result = dict(
        status="passed",
        raw_rows_checked=len(rows),
        source_and_checkpoint_hashes_verified=True,
        dataset_labels_and_subset_exclusions_verified=True,
        frozen_target_only_selection_verified=True,
        reset_coverage_verified=True,
        new_defense=False,
        results_sha256=sha(path / "results.json"),
        validator_source_sha256=sha(__file__),
    )
    (path / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_task_relocation(path: Path):
    """Verify raw classification, erased MLP weights and the activation/linear solves."""
    import sys
    import types

    import torch
    from clip_benchmark import checkpoint_loader
    from operator_diagnostics import paired_counts

    def read(name):
        return json.loads((path / name).read_bytes())

    def sha(file):
        return hashlib.sha256(Path(file).read_bytes()).hexdigest()

    config, meta, decision = read("protocol.json"), read("metadata.json"), read("decision.json")
    assert meta["completed"] and decision["completed"] and decision["construction_gate_passed"]
    assert meta["optimizer_steps"] == 0 and not meta["calibration_labels_used"]
    assert meta["source_sha256"] == sha("experiments/task_relocation.py")
    assert meta["config_sha256"] == sha("configs/task_relocation_cars.json")
    assert config["test_usage"] == "none"
    for name, expected in meta["helper_sha256"].items():
        assert sha(Path("experiments") / name) == expected
    for filename, key in [
        ("training_indices.json", "indices_sha256"),
        ("artifacts/release.pt", "release_sha256"),
        ("artifacts/predictions.pt", "predictions_sha256"),
    ]:
        assert sha(path / filename) == meta[key]
    parent = Path(config["support_parent"])
    assert sha(parent / "selection.json") == meta["support_selection_sha256"]
    prior = json.loads((parent / "training_indices.json").read_bytes())[config["task"]]
    excluded = set(prior["selection"]) | set(prior["confirmation"])
    indices = read("training_indices.json")
    assert all(not set(selected) & excluded for selected in indices.values())
    assert all(
        not set(indices[a]) & set(indices[b])
        for a, b in [
            ("calibration", "confirmation"),
            ("calibration", "attack"),
            ("confirmation", "attack"),
        ]
    )
    assert len(indices["calibration"]) == config["calibration_samples"]
    assert len(indices["confirmation"]) == config["confirmation_samples"]
    assert len(indices["attack"]) == config["attack_samples"]
    load = checkpoint_loader(Path(meta["author_repo"]))
    root = Path("checkpoints/ViT-B-32")
    for name, expected in meta["checkpoint_sha256"].items():
        assert sha(root / name) == expected
    original = load(str(root / config["task"] / "finetuned.pt"))
    anchor = load(str(root / "zeroshot.pt"))
    release = torch.load(path / "artifacts/release.pt", map_location="cpu", weights_only=True)
    before, base = original.state_dict(), anchor.state_dict()
    support = [f"model.visual.transformer.resblocks.{i}." for i in config["layers"]]
    edited_keys = []
    for key, value in release.items():
        if not torch.equal(value, before[key]):
            edited_keys.append(key)
            assert any(key.startswith(prefix) for prefix in support)
            assert ".attn." in key or ".mlp." in key
        if any(key.startswith(prefix + "mlp.") for prefix in support):
            assert torch.equal(value, base[key])
    fitting = json.loads((path / "artifacts/fitting.json").read_bytes())
    assert len(fitting) == meta["fits"] == len(config["layers"])
    inverse_maximum = 0.0
    for row in fitting:
        layer = row["layer"]
        raw = torch.load(
            path / f"artifacts/layer_{layer}_fit.pt", map_location="cpu", weights_only=True
        )
        a, target, before_fit, delta = raw["design"], raw["target"], raw["before"], raw["update"]
        ridge = row["fit"]["ridge"]
        residual = a.T @ (a @ (before_fit + delta) - target) + ridge * delta
        relative = float(
            residual.norm() / (a.T @ (target - a @ before_fit)).norm().clamp_min(1e-30)
        )
        assert relative < 1e-8
        out = f"model.visual.transformer.resblocks.{layer}.attn.out_proj."
        expected = before_fit + delta
        assert torch.allclose(
            release[out + "weight"].double(), expected[:-1].T, rtol=1e-6, atol=1e-6
        )
        assert torch.allclose(release[out + "bias"].double(), expected[-1], rtol=1e-6, atol=1e-6)
        block = original.model.visual.transformer.resblocks[layer]
        block.mlp.load_state_dict(anchor.model.visual.transformer.resblocks[layer].mlp.state_dict())
        with torch.no_grad():
            u, y = raw["inversion_solution"], raw["inversion_target"]
            inverse_residual = u + block.mlp(block.ln_2(u)) - y
            worst = float((inverse_residual.norm(dim=1) / y.norm(dim=1).clamp_min(1e-8)).max())
        assert worst <= config["inverse_tolerance"] * 1.1
        inverse_maximum = max(inverse_maximum, worst)
        n, task = raw["perturbation"], raw["task_basis"]
        assert float((n @ task).norm() / n.norm()) < 1e-8
        assert row["noise"]["affine_null_leakage"] < 1e-8
    package = types.ModuleType("dataset")
    package.__path__ = [str((Path(meta["author_repo"]) / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    dataset = get_dataset(config["task"], None, meta["data"]).train_dataset
    labels = torch.tensor([target for _, target in dataset._samples])
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    available = [i for i in order if i not in excluded]
    n, m, k = (
        config["calibration_samples"],
        config["confirmation_samples"],
        config["attack_samples"],
    )
    assert indices == dict(
        calibration=available[:n],
        confirmation=available[n : n + m],
        attack=available[n + m : n + m + k],
    )
    predictions = torch.load(
        path / "artifacts/predictions.pt", map_location="cpu", weights_only=True
    )
    rows = read("results.json")
    assert len(rows) == len(predictions) == (18 if decision["utility_gate_passed"] else 2)
    for row in rows:
        evidence = predictions[row["preset"], row["group"]]
        expected_labels = labels[indices["confirmation"]]
        assert torch.equal(evidence["labels"], expected_labels)
        correct = evidence["prediction"].eq(expected_labels)
        assert row["correct"] == int(correct.sum()) and row["n"] == len(expected_labels)
        assert row["top1"] == row["correct"] / row["n"]
        ref = predictions["unprotected", row["group"]]["prediction"].eq(expected_labels)
        assert row["paired"] == paired_counts(ref, correct)
    lookup = {(r["preset"], r["group"]): r for r in rows}
    loss = 100 * (
        lookup["unprotected", "standalone"]["top1"] - lookup["relocation", "standalone"]["top1"]
    )
    assert abs(loss - decision["utility_loss_pp"]) < 1e-9
    assert decision["utility_gate_passed"] == (loss <= config["utility_budget_pp"])
    if not decision["utility_gate_passed"]:
        assert not decision["candidate_valid"]
    else:
        best = {
            preset: max(
                (r for r in rows if r["preset"] == preset and r["group"] != "standalone"),
                key=lambda r: r["top1"],
            )
            for preset in ["unprotected", "relocation"]
        }
        gap = 100 * (best["unprotected"]["top1"] - best["relocation"]["top1"])
        assert best == decision["best_measured_attack"]
        assert abs(gap - decision["best_recovery_gap_pp"]) < 1e-9
        assert decision["candidate_valid"] == (gap >= config["minimum_recovery_gap_pp"])
    result = dict(
        status="passed",
        raw_rows_checked=len(rows),
        fits_verified=len(fitting),
        maximum_inverse_relative_residual=inverse_maximum,
        public_mlp_state_verified=True,
        edited_keys=edited_keys,
        subset_exclusion_verified=True,
        utility_loss_pp=loss,
        results_sha256=sha(path / "results.json"),
        validator_source_sha256=sha(__file__),
    )
    (path / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--task-support", action="store_true")
    mode.add_argument("--task-profile", action="store_true")
    mode.add_argument("--task-relocation", action="store_true")
    mode.add_argument("--class-recompile", action="store_true")
    mode.add_argument("--eight-task-resources", action="store_true")
    mode.add_argument("--eight-protection", action="store_true")
    mode.add_argument("--readout-recovery", action="store_true")
    mode.add_argument("--readout-stability", action="store_true")
    mode.add_argument(
        "--subspace-test", action="store_true", help="Validate frozen-release full tests."
    )
    mode.add_argument(
        "--subspace-interference", action="store_true", help="Validate complement interference."
    )
    mode.add_argument("--context-erasure", action="store_true", help="Validate context deletion.")
    mode.add_argument("--redundancy-erasure", action="store_true")
    mode.add_argument("--joint-transfer", action="store_true")
    mode.add_argument("--joint-diagnostics", action="store_true")
    mode.add_argument("--subspace", action="store_true", help="Validate the subspace probe.")
    mode.add_argument(
        "--reconstruction-context",
        action="store_true",
        help="Validate fitting feature/context controls.",
    )
    mode.add_argument(
        "--subspace-diagnostics", action="store_true", help="Validate causal contrasts."
    )
    mode.add_argument("--norm-cleanup", action="store_true", help="Validate data-free cleanup.")
    mode.add_argument("--bounded-complement", action="store_true", help="Validate bounded edits.")
    mode.add_argument(
        "--distribution-shift", action="store_true", help="Validate input diagnostics."
    )
    mode.add_argument("--scope-audit", action="store_true", help="Validate visual scan correction.")
    mode.add_argument(
        "--reconstruction-budget",
        action="store_true",
        help="Validate fixed reconstruction budgets.",
    )
    mode.add_argument(
        "--operator-diagnostics",
        action="store_true",
        help="Validate component and reconstruction controls.",
    )
    args = parser.parse_args()
    runner = (
        validate_task_relocation
        if args.task_relocation
        else validate_task_profile
        if args.task_profile
        else validate_task_support
        if args.task_support
        else validate_class_recompile
        if args.class_recompile
        else validate_readout_stability
        if args.readout_stability
        else validate_readout_recovery
        if args.readout_recovery
        else validate_subspace_test
        if args.subspace_test
        else validate_eight_task_resources
        if args.eight_task_resources
        else validate_eight_protection
        if args.eight_protection
        else validate_subspace_interference
        if args.subspace_interference
        else validate_context_erasure
        if args.context_erasure
        else validate_redundancy_erasure
        if args.redundancy_erasure
        else validate_joint_transfer
        if args.joint_transfer
        else validate_joint_diagnostics
        if args.joint_diagnostics
        else validate_reconstruction_context
        if args.reconstruction_context
        else validate_reconstruction_budget
        if args.reconstruction_budget
        else validate_operator_diagnostics
        if args.operator_diagnostics
        else validate_scope_audit
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
