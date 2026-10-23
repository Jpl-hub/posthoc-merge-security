"""Rebuilding a report must preserve every scientific input and both table formats."""

import hashlib
import json
from pathlib import Path

from _impl.report import paper_tables


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
