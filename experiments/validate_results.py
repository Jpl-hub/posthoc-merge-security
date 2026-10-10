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
    for task in ["MNIST", "SVHN"]:
        assert indices["development"][task] == parent[task]["development"]
    assert indices["fitting"]["SVHN"] == parent["SVHN"]["attack"][: config["attacker_samples"]]
    assert (
        indices["local_check"]["SVHN"]
        == parent["SVHN"]["calibration"][: config["local_check_samples"]]
    )
    confirmation = indices["confirmation"]["SVHN"]
    assert len(confirmation) == len(set(confirmation)) == config["confirmation_samples"]
    assert not set(confirmation).intersection(
        i for subset in parent["SVHN"].values() for i in subset
    )
    assert all(0 <= i < 73257 for i in confirmation)
    expected = {"unprotected", "reconstruct_joint"}
    expected.update(f"reconstruct_L{layer}" for layer in config["layers"])
    expected.update(
        f"replace_L{layer}_{component}"
        for layer in config["layers"]
        for component in config["components"]
    )
    expected.update(f"joint_{component}" for component in config["joint_components"])
    assert len(expected) == 34
    development_groups = {("standalone", "SVHN"), ("direct", "MNIST"), ("direct", "SVHN")}
    expected_keys = {(p, "development", g, d) for p in expected for g, d in development_groups}
    expected_keys.update(
        (p, "confirmation", g, "SVHN")
        for p in config["confirmation_presets"]
        for g in ["standalone", "direct"]
    )
    assert len(rows) == meta["new_evaluations"] == len(expected_keys) == 110
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
            assert pair["family_size"] == (33 if row["split"] == "development" else 3)
    assert len(profile) == 28 and len({p["preset"] for p in profile}) == 28
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
    assert {f["layer"] for f in fitting} == set(config["layers"]) and len(fitting) == 4
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
    assert groups == ["direct", "aga_all", "pretrained_fc", "reconstruction_1"]
    expected = {(p, g, t) for p in ["unprotected", name] for g in groups for t in ["MNIST", "SVHN"]}
    expected.update((p, g + "_standalone", task) for p in ["unprotected", name] for g in groups)
    keyed = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    assert len(keyed) == len(rows) and set(keyed) == expected
    counts = {"MNIST": 10000, "SVHN": 26032}
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
        new_defense_validated=False,
        results_sha256=hashlib.sha256((path / "results.json").read_bytes()).hexdigest(),
    )
    if write_report:
        (path / "validation.json").write_text(json.dumps(report, indent=2))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eight-task-resources", action="store_true")
    mode.add_argument(
        "--subspace-test", action="store_true", help="Validate frozen-release full tests."
    )
    mode.add_argument(
        "--subspace-interference", action="store_true", help="Validate complement interference."
    )
    mode.add_argument("--context-erasure", action="store_true", help="Validate context deletion.")
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
        validate_subspace_test
        if args.subspace_test
        else validate_eight_task_resources
        if args.eight_task_resources
        else validate_subspace_interference
        if args.subspace_interference
        else validate_context_erasure
        if args.context_erasure
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
