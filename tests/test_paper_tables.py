"""Rebuilding a report must preserve every scientific input and both table formats."""

import hashlib
import json
from pathlib import Path

from _impl.report import defense_comparison, paper_tables


def test_current_comparison_is_reproducible_and_keeps_partner_results(tmp_path):
    root = Path("results/comparison")
    saved = json.loads((root / "sources.json").read_text())
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in saved["input_sha256"]}
    defense_comparison(tmp_path)
    for name in ["eight_expert", "attacks", "partners", "feature_distillation",
                 "defense2_versions", "defense2_version_partners", "defense2_eight_versions",
                 "defense2_eight_calibration", "defense2_eight_calibration_target",
                 "eight_calibration", "eight_calibration_target", "defense2_mergers", "prodistill_components",
                 "defense2_projection", "defense2_projection_partners"]:
        for suffix in ["md", "tex"]:
            assert (tmp_path / f"{name}.{suffix}").read_bytes() == (root / f"{name}.{suffix}").read_bytes()
    rebuilt = json.loads((tmp_path / "sources.json").read_text())
    assert rebuilt == saved
    assert before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    assert len(saved["eight_expert_rows"]) == 3 * 4 * 2 * 7
    assert len(saved["tables"]["attacks"]["rows"]) == len(saved["tables"]["partners"]["rows"]) == 12
    assert len(saved["tables"]["feature_distillation"]["rows"]) == 12
    assert len(saved["tables"]["defense2_versions"]["rows"]) == 6
    assert len(saved["tables"]["defense2_version_partners"]["rows"]) == 6
    assert len(saved["tables"]["defense2_eight_versions"]["rows"]) == 18
    assert len(saved["tables"]["defense2_eight_calibration"]["rows"]) == 12
    assert len(saved["tables"]["defense2_eight_calibration_target"]["rows"]) == 6
    assert len(saved["tables"]["eight_calibration"]["rows"]) == 24
    assert len(saved["tables"]["eight_calibration_target"]["rows"]) == 12
    assert len(saved["tables"]["defense2_mergers"]["rows"]) == 6
    assert len(saved["tables"]["defense2_projection"]["rows"]) == 6
    assert len(saved["tables"]["defense2_projection_partners"]["rows"]) == 6
    assert len(saved["tables"]["prodistill_components"]["rows"]) == 6
    for name, table in saved["tables"].items():
        markdown = (root / f"{name}.md").read_text()
        latex = (root / f"{name}.tex").read_text()
        for row in table["rows"]:
            assert "| " + " | ".join(row) + " |" in markdown
            assert " & ".join(row) + r" \\" in latex


def test_report_matches_published_tables_without_modifying_inputs(tmp_path):
    root = Path("results/paper_tables")
    meta = json.loads((root / "table_sources.json").read_text())
    paths = list(meta["input_sha256"]) + ["results/task_directed_pairing/table_provenance.json"]
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    out = tmp_path / "tables"
    paper_tables(None, out)
    for name in ["tables.md", "tables.tex", "table_data.json", "figure.tex"]:
        assert (out / name).read_bytes() == (root / name).read_bytes(), name
    assert before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    new = json.loads((out / "table_sources.json").read_text())
    assert new["input_sha256"] == meta["input_sha256"]
    assert new["selected_rows"] == meta["selected_rows"]
    assert new["tables"] == meta["tables"]
    keys = {t["key"] for t in json.loads((out / "table_data.json").read_text())}
    assert {
        "main",
        "three_tasks",
        "numerical_ablation",
        "numerical_reset",
        "numerical_eight",
    } <= keys
