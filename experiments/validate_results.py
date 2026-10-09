"""Scientific acceptance checks over actual machine-generated outputs."""

import argparse
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
    report = {"status": "passed", "checks": checks}
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    validate(Path(parser.parse_args().output))
