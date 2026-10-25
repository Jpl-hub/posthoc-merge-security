"""One finite conditional solve for equal public class scores and native preservation."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
protocol = Path('results/data_free_compensation/neutral_compensation_protocol.json')
cfg = json.loads(protocol.read_text())
out = Path(cfg['output'])
out.mkdir(exist_ok=False)
(out / 'protocol.json').write_text(json.dumps(cfg, indent=2) + '\n')
for ref in cfg['inputs'].values():
    assert digest(ref['path']) == ref['sha256']


def svd(matrix):
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    threshold = torch.finfo(matrix.dtype).eps * max(matrix.shape) * s[0]
    keep = s > threshold
    return u[:, keep], s[keep], vh[keep], s, float(threshold)


def neutral_targets(residual, L, bias, eps):
    u, s, vh, spectrum, tolerance = svd(L)
    q = vh.T
    u0 = -q @ ((u.T @ bias) / s)
    d = len(u0)
    assert float(u0.square().sum()) < d
    centered = residual - residual.mean(0, keepdim=True)
    free = centered - q @ (q.T @ centered)
    sigma = ((free.square().sum(0) + d * eps) / (d - u0.square().sum())).sqrt()
    return q, q.T @ (u0[:, None] * sigma - centered), u0, spectrum, tolerance


# A fixed small independent solve checks the nonlinear LN identity and both constraints.
rng = np.random.default_rng(2026102511)
lt = rng.normal(size=(3, 8));lt -= lt.mean(1, keepdims=True)
bt = .01 * rng.normal(size=3)
rt = rng.normal(size=(8, 10))
qt, tt, ut, _, _ = neutral_targets(torch.tensor(rt), torch.tensor(lt), torch.tensor(bt), 1e-5)
hn = torch.tensor(rng.normal(size=(16, 4)))
kp = torch.tensor(rng.normal(size=(16, 10)))
un, _, _, _, _ = svd(hn)
ft = kp - un @ (un.T @ kp)
uf, sf, vf, _, _ = svd(ft)
mt = ((tt @ vf.T) / sf) @ uf.T
mt -= (mt @ un) @ un.T
numpy_m = np.linalg.lstsq(ft.numpy().T, tt.numpy().T, rcond=None)[0].T
np.testing.assert_allclose(mt.numpy(), numpy_m, atol=1e-10, rtol=1e-10)
new_rt = torch.tensor(rt) + qt @ mt @ kp
zc = new_rt - new_rt.mean(0, keepdim=True)
identity_error = float((torch.tensor(lt) @ zc / (zc.square().mean(0) + 1e-5).sqrt() + torch.tensor(bt)[:, None]).abs().max())
assert identity_error < 1e-10 and float((mt @ hn).abs().max()) < 1e-10

base, expert, head, pins = checked_models('MNIST')
raw = torch.load(cfg['inputs']['clean_source']['path'], map_location='cpu', weights_only=False)
geometry = torch.load(cfg['inputs']['geometry']['path'], map_location='cpu', weights_only=False)
basis = torch.load(cfg['inputs']['task_basis']['path'], map_location='cpu', weights_only=False)
clean_aga = torch.load(cfg['inputs']['clean_AGA']['path'], map_location='cpu', weights_only=False)
key = 'model.visual.transformer.resblocks.11.mlp.'
original, anchor = expert.state_dict(), base.state_dict()
to = lambda x: x.double().cuda()
A, order = to(geometry['A_prime_FP32']), geometry['author_assignment'].cuda()
B, B0, a, a0, b, b0 = [to(v) for v in [original[key+'c_proj.weight'], anchor[key+'c_proj.weight'], original[key+'c_fc.bias'], anchor[key+'c_fc.bias'], original[key+'c_proj.bias'], anchor[key+'c_proj.bias']]]
A0 = to(anchor[key+'c_fc.weight'])
cache = {n: {k: to(v) for k, v in row.items()} for n, row in raw['cache'].items()}
phi = lambda x: x * torch.sigmoid(1.702 * x)
hn = phi(F.linear(cache['native']['x'], A, a)).T
hd = phi(F.linear(cache['direct']['x'], (A0+A)/2, (a0+a)/2)).T
ha = phi(F.linear(cache['recovered']['x'], (A0+A[order])/2, (a0+a[order])/2)).T
back = torch.empty_like(ha);back[order] = ha
hidden = dict(native=hn, direct=hd/2, recovered=back/2)
fixed = dict(native=cache['native']['prefix'].T+B@hn+b[:, None],
             direct=cache['direct']['prefix'].T+(B+B0)@hd/2+(b+b0)[:, None]/2,
             recovered=cache['recovered']['prefix'].T+B@back/2+B0@ha/2+(b+b0)[:, None]/2)
public = merge(base, expert, base, .5)
for name in ['model.visual.ln_post.weight', 'model.visual.ln_post.bias', 'model.visual.proj']:
    assert torch.equal(clean_aga[name], original[name])
gamma, beta, P = [to(v) for v in [public.model.visual.ln_post.weight, public.model.visual.ln_post.bias, public.model.visual.proj]]
W = to(head.weight)
assert float(head.bias.abs().max()) == 0
D = W[1:] - W[0]
L = (D @ P.T) * gamma
L -= L.mean(1, keepdim=True)
offset = D @ P.T @ beta
K = torch.cat([hidden['direct'], hidden['recovered']], 1)
R = torch.cat([fixed['direct'], fixed['recovered']], 1)
Q, T, u0, head_spectrum, head_tolerance = neutral_targets(R, L, offset, expert.model.visual.ln_post.eps)
task = next(v for v in basis.values() if v['source_key'] == key+'c_fc.weight')['v']
V = torch.linalg.qr(to(task), mode='reduced').Q
task_hidden = phi(F.linear(torch.cat([V.T, -V.T]), A, a)).T
preserve = torch.cat([hn, task_hidden], 1)
U, _, _, native_spectrum, native_tolerance = svd(preserve)
free = K - U @ (U.T @ K)
UF, SF, VF, public_spectrum, public_tolerance = svd(free)
M = ((T @ VF.T) / SF) @ UF.T
M -= (M @ U) @ U.T
delta = Q @ M
native_error = float((delta @ preserve).norm() / (1 + (B @ preserve).norm()))
public_error = float((M @ K - T).norm() / (1 + T.norm()))
assert native_error < 1e-7 and public_error < 1e-7, (native_error, public_error)
assert torch.isfinite(delta).all()
torch.save(dict(delta_B=delta.cpu(), Q=Q.cpu(), M=M.cpu(), u0=u0.cpu(),
                target=T.cpu(), native_spectrum=native_spectrum.cpu(),
                public_spectrum=public_spectrum.cpu(), head_spectrum=head_spectrum.cpu()), out / 'solution.pt')
edits = {key+'c_fc.weight': A.float().cpu(), key+'c_proj.weight': (B+delta).float().cpu()}
torch.save(edits, out / 'point.pt')
point_sha = digest(out / 'point.pt')
print('POINT_FROZEN', point_sha, 'B_relative', float(delta.norm()/B.norm()), flush=True)
candidate = copy.deepcopy(expert)
candidate.load_state_dict(original | edits, strict=True)
aligned = copy.deepcopy(expert)
aligned.load_state_dict(clean_aga | {key+'c_fc.weight': edits[key+'c_fc.weight'][order.cpu()],
                                   key+'c_fc.bias': original[key+'c_fc.bias'][order.cpu()],
                                   key+'c_proj.weight': edits[key+'c_proj.weight'][:, order.cpu()]}, strict=True)
models = dict(native=candidate, direct=merge(base, candidate, base, .5), recovered=merge(base, aligned, base, .5))
norm = expert.val_preprocess.transforms[-1]
mean = torch.tensor(norm.mean, device='cuda')[None, :, None, None]
std = torch.tensor(norm.std, device='cuda')[None, :, None, None]
scores, metrics = {}, {}
with torch.no_grad():
    for name, model in models.items():
        model.cuda().eval().requires_grad_(False)
        values = []
        for i in range(0, 160, 8):
            x = F.interpolate(raw['final'][i:i+8].cuda(), (224, 224), mode='bilinear', align_corners=False)
            x = (torch.cat([torch.roll(x, shift, (2, 3)) for shift in [(0,0),(7,0),(-7,0),(0,7),(0,-7)]])-mean)/std
            values.append(head.cuda()(model(x)).cpu())
        value = torch.cat(values)
        assert torch.isfinite(value).all()
        scores[name] = value
        metrics[name] = dict(teacher_class_retained=int(value.argmax(1).eq(raw['teacher_class']).sum()),
                             maximum_class_score_spread=float((value.max(1).values-value.min(1).values).max()),
                             prediction_counts=torch.bincount(value.argmax(1), minlength=10).tolist())
        model.cpu()
torch.save(dict(scores=scores, teacher_class=raw['teacher_class'], requested=raw['requested']), out / 'source_scores.pt')
source = Path(__file__).read_text()
result = dict(status='sole_conditional_point_frozen_before_natural_images', point_sha256=point_sha,
              metrics=metrics, B_relative=float(delta.norm()/B.norm()),
              A_relative=cfg['A_relative'], source_native_constraint_error=native_error,
              source_public_constraint_error=public_error, u0_squared_norm=float(u0.square().sum()),
              head_rank=Q.shape[1], preserve_rank=U.shape[1], public_rank=len(SF),
              SVD_thresholds=[head_tolerance, native_tolerance, public_tolerance],
              independent_toy_LN_error=identity_error, solution_sha256=digest(out/'solution.pt'),
              source_arrays_sha256=digest(out/'source_scores.pt'), source=source,
              source_sha256=hashlib.sha256(source.encode()).hexdigest(), protocol_sha256=digest(protocol),
              checkpoint_sha256=pins, generated_full_image_forwards=2400, real_construction_images=0,
              teacher_training_updates=0, new_AGA_calls=0, optimizer_steps=0,
              seconds=time.perf_counter()-start, limits=cfg['limits'])
(out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'}, indent=2))
