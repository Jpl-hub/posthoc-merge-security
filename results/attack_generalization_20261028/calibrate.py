"""Apply the already fixed output calibration to three frozen SVD-deletion controls."""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

from _impl.data import official_labels
from _impl.io import digest, save_json
from _impl.metrics import row_metrics
from _impl.merge import merge
from _impl.models import apply, checked_models, initialize, load_checkpoint

initialize()
start = time.perf_counter()
protocol = Path(sys.argv[1])
cfg = json.loads(protocol.read_text())
out = Path(cfg['output'])
out.mkdir(exist_ok=False)
save_json(out / 'protocol.json', cfg)
noise = torch.rand(cfg['input_n'], 3, 32, 32,
                   generator=torch.Generator().manual_seed(cfg['input_seed']))
torch.save(noise, out / 'noise.pt')
assert digest(cfg['reuse_calibration']['path']) == cfg['reuse_calibration']['sha256']
prior_calibration = torch.load(cfg['reuse_calibration']['path'], map_location='cpu', weights_only=False)
calibration = {k: v for k, v in prior_calibration.items() if k.startswith('DTD/Unprotected/')}
pins = {}
image_forwards = 0

# All calibration vectors are frozen before opening any saved benchmark scores.
for case in cfg['cases']:
    task, peer = case['target'], case['partner']
    base, target, head, pins[task] = checked_models(task)
    partner = load_checkpoint(peer + '/finetuned.pt')
    peer_head = load_checkpoint('head_' + peer + '.pt')
    aligned = apply(target, case['AGA_references'][task], full=True)
    partner_aligned = apply(partner, case['AGA_references'][peer], full=True)
    normalizer = target.val_preprocess.transforms[-1]
    mean = torch.tensor(normalizer.mean)[None, :, None, None]
    std = torch.tensor(normalizer.std)[None, :, None, None]
    inputs = (F.interpolate(noise, (224, 224), mode='bilinear', align_corners=False) - mean) / std
    heads = {task: head.cuda().eval(), peer: peer_head.cuda().eval()}
    for name, own, restored in [
        ('Unprotected', target, aligned),
        (case['name'], apply(target, case['release']), apply(aligned, case['AGA_edits'])),
    ]:
        if name == 'Unprotected':
            continue
        for scope, left, right in [('None', own, partner),
                                   ('Protected expert', restored, partner),
                                   ('All experts', restored, partner_aligned)]:
            model = merge(base, left, right, .5).cuda().eval()
            with torch.no_grad():
                features = torch.cat([model(x.cuda()).cpu() for x in inputs.split(8)])
                for dataset, classifier in heads.items():
                    scores = classifier(features.cuda()).cpu().double()
                    assert torch.isfinite(scores).all()
                    prior = torch.logsumexp(F.log_softmax(scores, dim=1), dim=0) - np.log(len(scores))
                    calibration[task + '/' + name + '/' + scope + '/' + dataset] = dict(scores=scores, log_prior=prior)
            image_forwards += len(inputs)
            print('NOISE_SCORED', task, name, scope, flush=True)
            del model
            gc.collect()
    del base, target, partner, aligned, partner_aligned, own, restored, left, right, head, peer_head, heads
    gc.collect()
torch.save(calibration, out / 'calibration.pt')
frozen_sha = digest(out / 'calibration.pt')
print('CALIBRATION_FROZEN', frozen_sha, flush=True)

rows, corrected = [], {}
max_prior_error = 0.
for case in cfg['cases']:
    assert digest(case['raw_scores']['path']) == case['raw_scores']['sha256']
    raw = torch.load(case['raw_scores']['path'], map_location='cpu', weights_only=False)
    for key, row in raw.items():
        name, merger, scope, task = key.split('/')
        if merger != 'TA':
            continue
        assert name in ['Unprotected', case['name']]
        assert row['indices'] == list(range(len(row['target'])))
        np.testing.assert_array_equal(row['target'].numpy(), official_labels(task))
        fitted = calibration[case['target'] + '/' + name + '/' + scope + '/' + task]
        scores = fitted['scores'].numpy()
        prior_np = logsumexp(scores - logsumexp(scores, axis=1, keepdims=True), axis=0) - np.log(len(scores))
        max_prior_error = max(max_prior_error, float(np.max(np.abs(prior_np - fitted['log_prior'].numpy()))))
        value = row['logits'].double() - fitted['log_prior']
        prediction = (row['logits'].numpy().astype(np.float64) - prior_np).argmax(1)
        np.testing.assert_array_equal(value.argmax(1).numpy(), prediction)
        metric = row_metrics(value, row['target'], row['logits'], len(prior_np))
        assert metric['correct'] == int((prediction == row['target'].numpy()).sum())
        full_key = case['target'] + '/' + key
        corrected[full_key] = dict(logits=value, target=row['target'], indices=row['indices'])
        rows.append(dict(key=full_key, **metric))
        print('CALIBRATED', full_key, metric['correct'], metric['n'], flush=True)

matched = []
for key, row in corrected.items():
    task, name, merger, scope, dataset = key.split('/')
    if name == 'Unprotected':
        continue
    control = corrected['/'.join([task, 'Unprotected', merger, scope, dataset])]
    np.testing.assert_array_equal(row['target'], control['target'])
    metric = row_metrics(row['logits'], row['target'], control['logits'], row['logits'].shape[1])
    matched.append(dict(key=key, **metric))
assert len(rows) == 12 * len(cfg['cases']) and len(matched) == 6 * len(cfg['cases']) and max_prior_error < 1e-10
assert digest(out / 'calibration.pt') == frozen_sha
torch.save(corrected, out / 'corrected_scores.pt')
save_json(out / 'results.json', dict(
    status='completed', torch_version=torch.__version__, GPU=torch.cuda.get_device_name(), rows=rows, matched_corrected_controls=matched,
    protocol_sha256=digest(protocol), source_sha256=digest(__file__),
    calibration_sha256=frozen_sha, corrected_scores_sha256=digest(out / 'corrected_scores.pt'),
    noise_sha256=digest(out / 'noise.pt'), checkpoint_sha256=pins,
    maximum_numpy_prior_error=max_prior_error, calibration_frozen_before_benchmark_read=True,
    generated_image_forwards=image_forwards, new_real_image_forwards=0,
    new_AGA_calls=0, new_defense_points=0, teacher_training_updates=0,
    seconds=time.perf_counter()-start,
))
