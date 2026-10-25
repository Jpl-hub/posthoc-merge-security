"""One frozen output-calibration check; no defense construction or real-image fitting."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.data import official_labels
from _impl.io import digest
from _impl.metrics import row_metrics
from _impl.merge import merge
from _impl.models import apply, checked_models, initialize, load_checkpoint

initialize()
start = time.perf_counter()
protocol = Path('results/data_free_compensation/readout_calibration_protocol.json')
cfg = json.loads(protocol.read_text())
out = Path(cfg['output'])
out.mkdir(exist_ok=False)
(out / 'protocol.json').write_text(json.dumps(cfg, indent=2) + '\n')
configs = [json.loads(Path(p).read_text()) for p in cfg['evaluation_protocols']]
base, target, head, pins = checked_models('MNIST')
partner = load_checkpoint('SVHN/finetuned.pt')
aligned = apply(target, configs[0]['AGA_references']['MNIST'], full=True)
partner_aligned = apply(partner, configs[0]['AGA_references']['SVHN'], full=True)
g = torch.Generator().manual_seed(cfg['input_seed'])
noise = torch.rand(cfg['input_n'], 3, 32, 32, generator=g)
normalizer = target.val_preprocess.transforms[-1]
mean = torch.tensor(normalizer.mean)[None, :, None, None]
std = torch.tensor(normalizer.std)[None, :, None, None]
inputs = (F.interpolate(noise, (224, 224), mode='bilinear', align_corners=False) - mean) / std
torch.save(noise, out / 'noise.pt')
head.cuda().eval().requires_grad_(False)


@torch.no_grad()
def content_free_scores(model):
    # Calibration receives only the attacker's own fused model, public head and noise.
    model.cuda().eval().requires_grad_(False)
    scores = torch.cat([head(model(x.cuda())).cpu() for x in inputs.split(8)])
    model.cpu()
    assert torch.isfinite(scores).all()
    return scores.double()


cases = [('Unprotected', target, aligned)]
for config in configs:
    name, spec = next(iter(config['defenses'].items()))
    cases.append((name, apply(target, spec['release']), apply(aligned, spec['AGA_edits'])))
calibration = {}
for name, own, restored in cases:
    for scope, left, right in [('None', own, partner),
                               ('Protected expert', restored, partner),
                               ('All experts', restored, partner_aligned)]:
        model = merge(base, left, right, .5)
        scores = content_free_scores(model)
        log_prior = torch.logsumexp(F.log_softmax(scores, dim=1), dim=0) - np.log(len(scores))
        # Same diagonal probability division as the fixed official calibration, in log space.
        calibration[name + '/' + scope] = dict(scores=scores, log_prior=log_prior)
        del model
        gc.collect()
torch.save(calibration, out / 'calibration.pt')
calibration_sha = digest(out / 'calibration.pt')
print('CALIBRATION_FROZEN', calibration_sha, flush=True)

# All corrections are fixed before any saved benchmark labels/scores are read.
rows, corrected = [], {}
maximum_prior_error = 0.
for record in cfg['raw_scores']:
    assert digest(record['path']) == record['sha256']
    raw = torch.load(record['path'], map_location='cpu', weights_only=False)
    for key, row in raw.items():
        name, merger, scope, task = key.split('/')
        if merger != 'TA' or (record['name'] != name):
            continue
        assert row['indices'] == list(range(len(row['target'])))
        np.testing.assert_array_equal(row['target'].numpy(), official_labels(task))
        fitted = calibration[name + '/' + scope]
        prior = fitted['log_prior']
        scores = fitted['scores'].numpy()
        exponents = np.exp(scores - scores.max(1, keepdims=True))
        numpy_prior = np.log((exponents / exponents.sum(1, keepdims=True)).mean(0))
        maximum_prior_error = max(maximum_prior_error, float(np.max(np.abs(numpy_prior - prior.numpy()))))
        value = row['logits'].double() - prior
        pred = (row['logits'].numpy().astype(np.float64) - numpy_prior).argmax(1)
        np.testing.assert_array_equal(value.argmax(1).numpy(), pred)
        metric = row_metrics(value, row['target'], row['logits'], 10)
        corrected[key] = dict(logits=value, target=row['target'], indices=row['indices'])
        rows.append(dict(key=key, **metric))
        print('CALIBRATED', key, metric['correct'], metric['n'], flush=True)
assert len(rows) == 18 and maximum_prior_error < 1e-10
assert digest(out / 'calibration.pt') == calibration_sha
torch.save(corrected, out / 'corrected_scores.pt')
source = Path(__file__).read_text()
result = dict(status='completed_fixed_content_free_calibration_check', rows=rows,
              source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              protocol_sha256=digest(protocol), calibration_sha256=calibration_sha,
              corrected_scores_sha256=digest(out / 'corrected_scores.pt'),
              noise_sha256=digest(out / 'noise.pt'), checkpoint_sha256=pins,
              maximum_independent_numpy_prior_error=maximum_prior_error,
              calibration_frozen_before_benchmark_read=True, new_real_image_forwards=0,
              generated_image_forwards=9*cfg['input_n'], new_AGA_calls=0,
              new_defense_points=0, teacher_training_updates=0, seconds=time.perf_counter()-start)
(out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print('DONE', result['seconds'], flush=True)
