"""Directional derivative through the fixed last-block ProDistill recovery.

No defense optimization. Reuse the first eleven recovered blocks and the same
64 synthetic inputs. Compare forward-mode differentiation with the pinned
author's coefficient wrapper and torch.optim.Adam at two finite-difference steps.
"""
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp

from _impl.io import digest, save_json
from _impl.models import apply, checked_models, initialize
from _impl.prodistill import author


def main():
    initialize()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cfg = json.loads(Path(__file__).with_name('derivative_protocol.json').read_text())
    spec = json.loads(Path(cfg['parent_protocol']).read_text())
    case = spec['cases'][0]
    out = Path(cfg['output'])
    out.mkdir(exist_ok=False)
    save_json(out / 'protocol.json', cfg | {'executed_source_sha256': digest(__file__)})
    start = time.monotonic()
    base, original, head, _ = checked_models('MNIST')
    pub = apply(original, case['releases']['Defense2'])
    ref = case['synthetic_source']
    assert digest(ref['path']) == ref['sha256']
    raw = torch.load(ref['path'], map_location='cpu', weights_only=False)
    pixels = raw['final'][ref['indices']].cuda()
    norm = original.val_preprocess.transforms[-1]
    images = F.interpolate(pixels, (224, 224), mode='bilinear', align_corners=False)
    images = (images - images.new_tensor(norm.mean)[None, :, None, None]) / images.new_tensor(norm.std)[None, :, None, None]
    del raw, pixels
    recovered = copy.deepcopy(original)
    assert digest(cfg['recovered_model']['path']) == cfg['recovered_model']['sha256']
    recovered.load_state_dict(torch.load(cfg['recovered_model']['path'], map_location='cpu', weights_only=False))

    def inputs(model):
        captured = []
        block = model.model.visual.transformer.resblocks[11]
        hook = block.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach().cpu()))
        with torch.no_grad():
            model.cuda()
            for batch in images.split(16):
                model(batch)
            model.cpu()
        hook.remove()
        return torch.cat(captured, 1).cuda().double()

    xm, xt, xb = inputs(recovered), inputs(pub), inputs(base)
    with torch.no_grad():
        original.cuda(); head.cuda()
        labels = torch.cat([head(original(x)).argmax(1) for x in images.split(16)])
        original.cpu(); head.cpu()
    block = copy.deepcopy(base.model.visual.transformer.resblocks[11]).cuda().double().eval()
    names = tuple(dict(block.named_parameters()))
    anchor = tuple(p.detach() for p in block.parameters())
    own = pub.model.visual.transformer.resblocks[11]
    clean = original.model.visual.transformer.resblocks[11]
    theta = tuple(p.detach().cuda().double() for p in own.parameters())
    direction = tuple(p - q.detach().cuda().double() for p, q in zip(theta, clean.parameters()))
    assert {name for name, v in zip(names, direction) if v.count_nonzero()} == {
        'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias'}
    delta = tuple(p - b for p, b in zip(theta, anchor))
    readout = copy.deepcopy(recovered.model.visual.ln_post).cuda().double()
    projection = recovered.model.visual.proj.detach().cuda().double()
    head = head.cuda().double()
    env = author(spec)
    del images, recovered, pub, original, base, own, clean

    def call(parameters, x):
        return functional_call(block, dict(zip(names, parameters)), (x,))

    teacher, dteacher = jvp(lambda p: call(p, xt), (theta,), (direction,))
    teacher, dteacher = teacher.detach(), dteacher.detach()
    with torch.no_grad():
        public = call(anchor, xb)

    def loss(alphas, changes, target):
        weights = tuple(b + a * d for b, a, d in zip(anchor, alphas, changes))
        student = call(weights, xm)
        # Same sum as two groups of 64 identical images; no mean-loss rescaling.
        return (student - target).square().sum() + (student - public).square().sum()

    def measure(weights):
        cls = call(weights, xm)[0]
        logits = head(readout(cls) @ projection)
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1).mean()
        ce = F.cross_entropy(logits, labels)
        return torch.stack((entropy, ce))

    alphas = tuple(torch.full_like(d, .3) for d in delta)
    zeros = lambda: tuple(torch.zeros_like(d) for d in delta)
    tangent, m, v, dm, dv = zeros(), zeros(), zeros(), zeros(), zeros()
    gradient = grad(loss, argnums=0)
    history = []
    for step in range(1, 101):
        g, dg = jvp(gradient, (alphas, delta, teacher), (tangent, direction, dteacher))
        next_values = [[] for _ in range(6)]
        with torch.no_grad():
            for a, da, mi, vi, dmi, dvi, gi, dgi in zip(alphas, tangent, m, v, dm, dv, g, dg):
                mi = .9 * mi + .1 * gi
                dmi = .9 * dmi + .1 * dgi
                vi = .999 * vi + .001 * gi.square()
                dvi = .999 * dvi + .002 * gi * dgi
                root = vi.sqrt()
                denom = root / (1 - .999 ** step) ** .5 + 1e-8
                # At zero gradient, the composite Adam update has derivative
                # dm/eps. Set the irrelevant sqrt tangent to zero to avoid 0/0.
                ddenom = torch.where(root > 0, dvi / (2 * root.clamp_min(1e-300)), 0) / (1 - .999 ** step) ** .5
                scale = .01 / (1 - .9 ** step)
                aa = a - scale * mi / denom
                dd = da - scale * (dmi / denom - mi * ddenom / denom.square())
                for dst, val in zip(next_values, (aa, dd, mi, vi, dmi, dvi)):
                    assert torch.isfinite(val).all()
                    dst.append(val)
        alphas, tangent, m, v, dm, dv = map(tuple, next_values)
        if step in (1, 2, 10, 25, 50, 75, 100):
            row = {'step': step, 'loss': float(loss(alphas, delta, teacher)), 'seconds': time.monotonic() - start}
            history.append(row)
            save_json(out / 'progress.json', row)
            print('DERIVATIVE', row, flush=True)

    weights = tuple(b + a * d for b, a, d in zip(anchor, alphas, delta))
    direct = tuple(a * q for a, q in zip(alphas, direction))
    total = tuple(q + da * d for q, da, d in zip(direct, tangent, delta))
    values, exact = jvp(measure, (weights,), (total,))
    _, fixed_alpha = jvp(measure, (weights,), (direct,))
    torch.save({'alphas': tuple(p.cpu() for p in alphas), 'tangents': tuple(p.cpu() for p in tangent),
                'values': values.cpu(), 'exact': exact.cpu(), 'fixed_alpha': fixed_alpha.cpu()}, out / 'derivative.pt')

    def reference(offset):
        target = copy.deepcopy(block)
        target.load_state_dict(dict(zip(names, (p + offset * q for p, q in zip(theta, direction)))))
        wrapper = env['MergedModel'](copy.deepcopy(block), [target, copy.deepcopy(block)], 'elementwise')
        opt = torch.optim.Adam([p for p in wrapper.parameters() if p.requires_grad], lr=.01, foreach=False)
        with torch.no_grad():
            wanted = target(xt)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            student = wrapper(xm)
            value = (student - wanted).square().sum() + (student - public).square().sum()
            value.backward()
            opt.step()
        params = tuple(b + wrapper.alphas[0][i].detach() * (p + offset * q - b)
                       for i, (b, p, q) in enumerate(zip(anchor, theta, direction)))
        with torch.no_grad():
            value = measure(params).detach()
        error = max(float((a - b).abs().max()) for a, b in zip(weights, params)) if offset == 0 else None
        print('AUTHOR_REFERENCE', offset, value.tolist(), error, time.monotonic() - start, flush=True)
        return value, error

    baseline, error = reference(0)
    checks = []
    for h in cfg['finite_difference_steps']:
        plus, _ = reference(h)
        minus, _ = reference(-h)
        fd = (plus - minus) / (2 * h)
        checks.append({'step': h, 'finite_difference': fd.tolist(), 'absolute_error': (fd - exact).abs().tolist()})
        save_json(out / 'finite_differences.json', checks)
    result = {'metrics': ['recovered_entropy', 'recovered_cross_entropy'], 'values': values.tolist(),
              'author_reference_values': baseline.tolist(), 'maximum_author_weight_error': error,
              'total_derivative': exact.tolist(), 'fixed_alpha_derivative': fixed_alpha.tolist(),
              'finite_differences': checks, 'history': history, 'seconds': time.monotonic() - start,
              'peak_GPU_bytes': torch.cuda.max_memory_allocated(), 'defense_updates': 0,
              'real_images': 0, 'new_defense_models': 0, 'source_sha256': digest(__file__)}
    result['passed'] = error < 1e-8 and all(abs(a - b) <= 1e-5 + .02 * abs(b)
        for a, b in zip(checks[-1]['finite_difference'], exact.tolist()))
    save_json(out / 'result.json', result)
    print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
