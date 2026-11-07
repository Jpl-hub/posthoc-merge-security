"""The public exporter preserves frozen evidence and reproduces published tables."""

import hashlib
import json
from pathlib import Path

from _impl.report import tables


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
