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
        assert len(real) == 22, len(real)
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
    report = {"status": "passed", "checks": checks}
    (path / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    validate(Path(parser.parse_args().output))
