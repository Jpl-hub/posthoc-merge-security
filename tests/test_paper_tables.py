"""The public exporter preserves frozen evidence and reproduces published tables."""

import hashlib
import json
from pathlib import Path

from _impl.report import rank_cells, tables


def test_ranking_respects_metric_directions_and_ties():
    rows = [["None", "99.50", "90.00"], ["A", "99.50", "2.00"],
            ["B", "99.00", "2.00"]]
    tex, md = rank_cells(rows, {1: True, 2: False})
    assert tex == [["None", r"\textbf{99.50}", r"\underline{90.00}"],
                   ["A", r"\textbf{99.50}", r"\textbf{2.00}"],
                   ["B", r"\underline{99.00}", r"\textbf{2.00}"]]
    assert md[2][1] == "<u>99.00</u>"
    assert rows[0][1] == "99.50"
    equal = [["A", "2.00"], ["B", "2.00"]]
    assert rank_cells(equal, {1: False}) == (equal, equal)


def test_nbp_tables_reproduce_recorded_exports_without_mutating_inputs(tmp_path):
    published = Path('results/phase3/bias_weight_only/tables')
    manifest = json.loads((published / 'sources.json').read_text())
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in manifest['input_sha256']}
    tables(tmp_path)
    for name in manifest['tables']:
        for suffix in ['md', 'tex']:
            assert (tmp_path / f'{name}.{suffix}').read_bytes() == (published / f'{name}.{suffix}').read_bytes()
    assert before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    assert json.loads((tmp_path / 'sources.json').read_text()) == manifest
