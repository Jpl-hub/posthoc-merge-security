"""Recount cached rank-one predictions and target training shifts; no model forward."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
SOURCE = ROOT / 'outputs/feature_noise'
OUT = ROOT / 'outputs/class_noise'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(path, n):
    data = path.read_bytes()
    assert struct.unpack('>II', data[:8]) == (2049, n)
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


torch.set_num_threads(4)
metadata = json.loads((SOURCE / 'execution.json').read_text())
assert sha(SOURCE / 'artifacts/release.pt') == metadata['release_sha256']
assert sha(SOURCE / 'artifacts/predictions.pt') == metadata['predictions_sha256']
raw = torch.load(SOURCE / 'artifacts/predictions.pt', map_location='cpu')
test_y = labels(ROOT / 'data/MNIST/raw/t10k-labels-idx1-ubyte', 10000)
counts = json.loads((OUT / 'current_rank1_analysis.json').read_text())
assert counts['raw_sha256'] == metadata['predictions_sha256']
for record in counts['records']:
    stage = record['stage']
    pair = [raw[f'{name}/{stage}/MNIST'] for name in ['original', 'released']]
    for item in pair:
        np.testing.assert_array_equal(item['target'].numpy(), test_y)
    before, after = [item['prediction'].numpy() for item in pair]
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (test_y, after), 1)
    np.testing.assert_array_equal(confusion, record['released_confusion'])
    np.testing.assert_array_equal(confusion.sum(0), record['prediction_histogram'])
    for row in record['classes']:
        selected = test_y == row['class']
        assert row['n'] == int(selected.sum())
        assert row['original_correct'] == int((before[selected] == test_y[selected]).sum())
        assert row['released_correct'] == int((after[selected] == test_y[selected]).sum())

artifact = torch.load(SOURCE / 'artifacts/release.pt', map_location='cpu')
indices = json.loads((SOURCE / 'training_indices.json').read_text())['fit']
train_path = ROOT / 'data/MNIST/raw/train-labels-idx1-ubyte'
train_y = labels(train_path, 60000)[indices]
shift = (artifact['proxy_hidden'] - artifact['fit_hidden']).numpy()
basis, direction = [artifact[key].numpy() for key in ['basis', 'direction']]
centers = np.stack([shift[train_y == k].mean(0) for k in range(10)])
centers -= (centers @ basis) @ basis.T
weights = np.bincount(train_y, minlength=10) / len(train_y)
mean = weights @ centers
centered = centers - mean
singular = np.linalg.svd(np.sqrt(weights)[:, None] * centered, compute_uv=False)
diagnosis = json.loads((OUT / 'class_shift_analysis.json').read_text())
assert diagnosis['calibration_n'] == len(train_y)
assert diagnosis['native_label_sha256'] == sha(train_path)
assert diagnosis['source_release_sha256'] == metadata['release_sha256']
assert diagnosis['centered_rank_numerical'] == int((singular > singular[0] * 1e-10).sum())
for key, value in {
    'mean_shift_norm': np.linalg.norm(mean),
    'class_variation_rms': np.linalg.norm(singular),
    'centered_top1_energy_fraction': singular[0] ** 2 / np.sum(singular ** 2),
}.items():
    np.testing.assert_allclose(diagnosis[key], value, rtol=1e-9, atol=1e-10)
for row in diagnosis['classes']:
    k = row['class']
    assert row['n'] == int((train_y == k).sum())
    for key, value in {
        'shift_norm': np.linalg.norm(centers[k]),
        'old_direction_response_mean': centers[k] @ direction,
        'difference_from_global_shift_norm': np.linalg.norm(centered[k]),
    }.items():
        np.testing.assert_allclose(row[key], value, rtol=1e-9, atol=1e-10)

receipt = dict(status='passed', script_sha256=sha(Path(__file__)),
               raw_prediction_sha256=metadata['predictions_sha256'],
               source_release_sha256=metadata['release_sha256'],
               records_sha256={name: sha(OUT / name) for name in
                   ['current_rank1_analysis.json', 'class_shift_analysis.json']},
               model_forwards=0, new_candidates=0,
               limitation='Cached fit-only class directions; no held-out confirmation or defense claim.')
(OUT / 'analysis_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
