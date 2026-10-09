"""Scientific acceptance checks over actual machine-generated outputs."""

import argparse
import hashlib
import json
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
    report = {"status": "passed", "checks": checks}
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    validate(Path(parser.parse_args().output))
