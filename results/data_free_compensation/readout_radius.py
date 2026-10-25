"""One cached calculation of finite final-projection margin sensitivity."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
protocol_path = Path('results/data_free_compensation/readout_radius_protocol.json')
cfg = json.loads(protocol_path.read_text())
root = Path('outputs/frozen_classifier_20261025')
out = root / 'readout_radius'
out.mkdir(exist_ok=False)
point = root / 'class_coupling/point.pt'
cache = root / 'class_coupling/source.pt'
aligned = Path('outputs/feature_noise/artifacts/original_AGA.pt')
for p, sha in [(point, cfg['point_sha256']), (cache, cfg['cached_source_sha256']), (aligned, cfg['original_AGA_sha256'])]:
    assert digest(p) == sha, p
base, expert, head, pins = checked_models('MNIST')
sb, se = base.state_dict(), expert.state_dict()
edits = torch.load(point, map_location='cpu', weights_only=False)
sa = torch.load(aligned, map_location='cpu', weights_only=False)
prefix = torch.load(cache, map_location='cpu', weights_only=False)['prefix']
key = 'model.visual.transformer.resblocks.11.mlp.'
keys = [key + s for s in ['c_fc.weight', 'c_fc.bias', 'c_proj.weight', 'c_proj.bias']]
A, a, B, b = [se[k] for k in keys]
row = {v.numpy().tobytes(): i for i, v in enumerate(A)}
order = torch.tensor([row[v.numpy().tobytes()] for v in sa[keys[0]]])
assert len(set(order.tolist())) == len(order)
assert torch.equal(sa[keys[1]], a[order]) and torch.equal(sa[keys[2]], B[:, order])
mix = lambda x, y: x + .5 * (y - x)
v = 'model.visual.'
P = mix(sb[v+'proj'], se[v+'proj']).double().numpy()
gamma = mix(sb[v+'ln_post.weight'], se[v+'ln_post.weight']).double().numpy()
beta = mix(sb[v+'ln_post.bias'], se[v+'ln_post.bias']).double().numpy()
W = head.weight.double().numpy()
d = W[0] - W[1:]
pnorm = np.linalg.norm(P)
radius = np.linalg.norm(.5*(se[v+'proj']-sb[v+'proj']).double().numpy()) / pnorm
records, arrays = {}, {}
for name in cfg['states']:
    values = [edits.get(k, se[k]) for k in keys]
    if name == 'recovered':
        values = [values[0][order], values[1][order], values[2][:, order], values[3]]
    am, aa, bm, bb = [mix(sb[k], w).cuda() for k, w in zip(keys, values)]
    x = prefix[name]['x'].float().cuda()
    preactivation = F.linear(x, am, aa)
    r = prefix[name]['prefix'].float().cuda() + F.linear(preactivation*torch.sigmoid(1.702*preactivation), bm, bb)
    r = r.double().cpu().numpy()
    z = (r-r.mean(1, keepdims=True)) / np.sqrt(r.var(1, keepdims=True)+expert.model.visual.ln_post.eps)
    z = z*gamma+beta
    scores = z @ P @ W.T
    assert np.all(scores.argmax(1) == 0)
    margins = z @ P @ d.T
    dual = np.linalg.norm(z, axis=1)[:, None]*np.linalg.norm(d, axis=1)[None, :]
    critical = margins/(pnorm*dual)
    lower = margins-radius*pnorm*dual
    i, j = np.unravel_index(critical.argmin(), critical.shape)
    delta = -1.000001*margins[i, j]/(dual[i, j]**2)*np.outer(z[i], d[j])
    after = z @ (P+delta) @ W.T
    t = torch.tensor(z[i], dtype=torch.double, requires_grad=False)
    pp = torch.tensor(P, dtype=torch.double, requires_grad=True)
    contrast = torch.tensor(d[j], dtype=torch.double)
    grad = torch.autograd.grad(t @ pp @ contrast, pp)[0].numpy()
    gradient_error = float(np.abs(grad-np.outer(z[i], d[j])).max())
    support_error = float(abs(np.linalg.norm(grad)-dual[i, j]))
    identity_error = float(abs((after[i, 0]-after[i, j+1]) + .000001*margins[i, j]))
    assert gradient_error < 1e-10 and support_error < 1e-10 and identity_error < 1e-9
    assert after[i, j+1] > after[i, 0]
    records[name] = dict(n=len(z), own_update_relative_radius=float(radius), minimum_critical_radius=float(critical.min()), critical_radius_quantiles=np.quantile(critical.min(1), [0,.25,.5,.75,1]).tolist(), all_nine_positive_at_own_radius=int((lower.min(1)>0).sum()), witness_input=int(i), witness_competitor=int(j+1), witness_relative_norm=float(np.linalg.norm(delta)/pnorm), witness_class0_remaining=int((after.argmax(1)==0).sum()), witness_prediction_histogram=np.bincount(after.argmax(1), minlength=10).tolist(), analytic_gradient_error=gradient_error, dual_norm_error=support_error, finite_margin_identity_error=identity_error)
    arrays[name] = dict(margins=margins, critical_relative_radius=critical, own_radius_lower_bound=lower, witness_delta_P=delta, witness_predictions=after.argmax(1))
torch.save(arrays, out/'arrays.pt')
source = Path(__file__).read_text()
result = dict(status='completed_read_only', protocol=cfg, protocol_sha256=digest(protocol_path), checkpoint_sha256=pins, records=records, arrays_sha256=digest(out/'arrays.pt'), seconds=time.perf_counter()-start, real_images=0, new_image_forwards=0, new_full_AGA_calls=0, new_matching_calls=0, new_defense_points=0, source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest())
(out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'}, indent=2))
