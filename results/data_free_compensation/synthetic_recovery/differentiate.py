"""Differentiate fixed last-block recovery; optionally construct one A/B release.

Both modes reuse the first eleven recovered blocks and 64 synthetic inputs.
The default protocol checks a derivative; construction requires its own protocol.
"""
import copy
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp

from _impl.io import digest, save_json
from _impl.models import apply, checked_models, initialize
from _impl.prodistill import author


def recover_gradient(loss, measure, anchor, theta, teacher_fn, steps=100):
    """Exact reverse differentiation; CPU checkpoints bound GPU memory."""
    delta = tuple(p - b for p, b in zip(theta, anchor))
    target = teacher_fn(theta).detach()
    state = (tuple(torch.full_like(p, .3) for p in theta),
             tuple(torch.zeros_like(p) for p in theta), tuple(torch.zeros_like(p) for p in theta))
    inner_grad = grad(loss)

    def update(state, changes, teacher, step):
        aa, mm, vv = state
        gg = inner_grad(aa, changes, teacher)
        m = tuple(.9 * x + .1 * g for x, g in zip(mm, gg))
        v = tuple(.999 * x + .001 * g.square() for x, g in zip(vv, gg))
        # The composite Adam derivative at zero gradient is finite. Avoid the
        # standalone sqrt(0) derivative, whose cotangent is zero in this update.
        root = tuple(torch.where(x > 0, torch.where(x > 0, x, torch.ones_like(x)).sqrt(), 0) for x in v)
        a = tuple(x - (.01 / (1 - .9 ** step)) * y /
                  (z / (1 - .999 ** step) ** .5 + 1e-8) for x, y, z in zip(aa, m, root))
        return a, m, v

    history = []
    for step in range(1, steps + 1):
        history.append(tuple(tuple(x.detach().cpu() for x in row) for row in state))
        state = tuple(tuple(x.detach() for x in row) for row in update(state, delta, target, step))
    weights = tuple(b + a * d for b, a, d in zip(anchor, state[0], delta))
    value, pull = vjp(lambda w: measure(w)[0], weights)
    gw, = pull(torch.ones_like(value))
    gd = tuple(g * a for g, a in zip(gw, state[0]))
    cot = (tuple(g * d for g, d in zip(gw, delta)),
           tuple(torch.zeros_like(g) for g in gw), tuple(torch.zeros_like(g) for g in gw))
    gt = torch.zeros_like(target)
    for step in range(steps, 0, -1):
        previous = tuple(tuple(x.cuda() for x in row) for row in history.pop())
        _, pull = vjp(lambda s, d, t: update(s, d, t, step), previous, delta, target)
        cot, dd, dt = pull(cot)
        cot = tuple(tuple(x.detach() for x in row) for row in cot)
        gd = tuple((a + b).detach() for a, b in zip(gd, dd))
        gt = (gt + dt).detach()
        del pull, previous, dd, dt
    _, pull = vjp(teacher_fn, theta)
    through_target, = pull(gt)
    return float(value), tuple((a + b).detach() for a, b in zip(gd, through_target)), weights


def main():
    initialize()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('derivative_protocol.json'))
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    assert digest(__file__) == cfg['source_sha256']
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
    native_cache = raw['cache']['native']
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
    original_parameters = tuple(p.detach().cuda().double() for p in clean.parameters())
    direction = tuple(p - q.detach().cuda().double() for p, q in zip(theta, clean.parameters()))
    assert {name for name, v in zip(names, direction) if v.count_nonzero()} == {
        'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias'}
    delta = tuple(p - b for p, b in zip(theta, anchor))
    readout = copy.deepcopy(recovered.model.visual.ln_post).cuda().double()
    projection = recovered.model.visual.proj.detach().cuda().double()
    native_readout = copy.deepcopy(pub.model.visual.ln_post).cuda().double()
    native_projection = pub.model.visual.proj.detach().cuda().double()
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

    if cfg.get('outer_steps'):
        # Only A/a/B/b are changed. The fixed prefix and other tensors remain the
        # original expert's, so their previously recovered blocks can be reused.
        iA, ia, iB, ib = [names.index(n) for n in ('mlp.c_fc.weight', 'mlp.c_fc.bias',
                                                 'mlp.c_proj.weight', 'mlp.c_proj.bias')]
        A, a, B, b = [original_parameters[i] for i in (iA, ia, iB, ib)]
        gamma, beta = [original_parameters[names.index('ln_2.' + n)] for n in ('weight', 'bias')]
        U, _, Vh = torch.linalg.svd(A - anchor[iA], full_matrices=False)
        U, V = U[:, :32], Vh[:32].T
        task = json.loads(Path('configs/defense2_inputs.json').read_text())['MNIST']
        assert digest(task['layer_inputs']['path']) == task['layer_inputs']['sha256']
        u = torch.load(task['layer_inputs']['path'], map_location='cpu', weights_only=False).cuda().double()
        xn = native_cache['x'].cuda().double()
        prefix = native_cache['prefix'].cuda().double()
        target_native = native_cache['final'].cuda().double() - prefix
        native_labels = native_cache['scores'].argmax(1).cuda()
        act = lambda x: x * torch.sigmoid(1.702 * x)
        xu = beta + gamma * u
        with torch.no_grad():
            yu = torch.cat([F.linear(act(F.linear(x, A, a)), B, b) for x in xu.split(256)])
        populations = [(xu, yu), (xn, target_native)]
        variances = [float(y.square().sum() / len(y) - y.mean(0).square().sum()) for x, y in populations]
        R0 = torch.cat((B, b[:, None]), 1)
        na, nr = float(A.norm()), float(R0.norm())
        rho, tau = task['original_ridge'], task['original_native_limit']

        def native_scores(p):
            h = prefix + F.linear(act(F.linear(xn, p[iA], p[ia])), p[iB], p[ib])
            return head(native_readout(h) @ native_projection)

        @torch.no_grad()
        def native_error(p):
            R = torch.cat((p[iB], p[ib][:, None]), 1)
            error = rho * (R - R0).square().sum()
            for (x, y), variance in zip(populations, variances):
                for xx, yy in zip(x.split(256), y.split(256)):
                    predicted = F.linear(act(F.linear(xx, p[iA], p[ia])), p[iB], p[ib])
                    error += (predicted - yy).square().sum() / (len(x) * variance)
            return float(error)

        @torch.no_grad()
        def compensate(p):
            G = torch.eye(3073, device='cuda', dtype=torch.float64) * rho
            T = rho * R0
            constant = rho * R0.square().sum()
            for (x, y), variance in zip(populations, variances):
                for xx, yy in zip(x.split(256), y.split(256)):
                    h = torch.cat((act(F.linear(xx, p[iA], p[ia])), xx.new_ones(len(xx), 1)), 1)
                    scale = 1 / (len(x) * variance)
                    G.addmm_(h.T, h, alpha=scale)
                    T.addmm_(yy.T, h, alpha=scale)
                    constant += scale * yy.square().sum()
            L = torch.linalg.cholesky(G)
            center = torch.cholesky_solve(T.T, L).T
            minimum = constant - (center * T).sum()
            radius = tau - minimum
            if radius <= 0:
                return None, {'native_minimum': float(minimum)}
            R = torch.cat((p[iB], p[ib][:, None]), 1)
            deviation = R - center
            energy = (deviation @ G * deviation).sum()
            factor = min(1., float((radius * (1 - 1e-10) / energy.clamp_min(1e-30)).sqrt()))
            R = center + factor * deviation
            p = list(p); p[iB], p[ib] = R[:, :-1], R[:, -1]
            score = native_scores(tuple(p))
            matches = int((score.argmax(1) == native_labels).sum())
            distanceA = float((p[iA] - A).norm() / na)
            distanceB = float((p[iB] - B).norm() / B.norm())
            feasible = matches == len(native_labels) and distanceA <= 2 and distanceB <= 4
            return (tuple(p) if feasible else None), dict(native_matches=matches,
                native_error=float(minimum + factor ** 2 * energy), native_limit=tau,
                A_relative=distanceA, B_relative=distanceB, compensation_factor=factor)

        @torch.no_grad()
        def recover_value(p):
            target = call(p, xt)
            changes = tuple(x - y for x, y in zip(p, anchor))
            aa = tuple(torch.full_like(x, .3).requires_grad_() for x in changes)
            opt = torch.optim.Adam(aa, lr=.01, foreach=False)
            for _ in range(100):
                opt.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    value = loss(aa, changes, target)
                    value.backward()
                opt.step()
            w = tuple(b + alpha * d for b, alpha, d in zip(anchor, aa, changes))
            return float(measure(w)[0])

        records = []
        initial_theta = theta
        with torch.no_grad():
            initial_matches = int((native_scores(theta).argmax(1) == native_labels).sum())
        initial_error = native_error(theta)
        save_json(out / 'initial.json', dict(native_matches=initial_matches, native_error=initial_error,
                  native_limit=tau, source_total=len(native_labels)))
        assert initial_matches == len(native_labels) and initial_error <= tau + 1e-8
        for outer in range(cfg['outer_steps']):
            value, g, weights = recover_gradient(loss, measure, anchor, theta, lambda p: call(p, xt))
            if outer == 0:
                expected = json.loads(Path(__file__).with_name('derivative_result.json').read_text())
                dotted = float(sum((gi * di).sum() for gi, di in zip(g, direction)))
                error = abs(dotted - expected['total_derivative'][0])
                save_json(out / 'gradient_check.json', dict(dot=dotted, reference=expected['total_derivative'][0], error=error))
                assert error <= 1e-5 * (1 + abs(dotted)), error
                print('GRADIENT_CHECK', dotted, error, flush=True)
            dA = na * (g[iA] - g[ia][:, None] * beta)
            dA = dA - U @ (U.T @ dA)
            dA = dA - (dA @ V) @ V.T
            dR = nr * torch.cat((g[iB], g[ib][:, None]), 1)
            length = (dA.square().sum() + dR.square().sum()).sqrt()
            assert length.isfinite() and length > 0
            dA, dR = na * dA / length, nr * dR / length
            accepted = False
            attempts = []
            for k in range(cfg['backtracking_steps']):
                step = cfg['step_size'] / 2 ** k
                p = list(theta)
                p[iA] = theta[iA] + step * dA
                p[ia] = theta[ia] - step * (dA @ beta)
                p[iB] = theta[iB] + step * dR[:, :-1]
                p[ib] = theta[ib] + step * dR[:, -1]
                p, info = compensate(tuple(p))
                info.update(step=step, feasible=p is not None)
                if p is not None:
                    proposed = recover_value(p)
                    info['recovered_entropy'] = proposed
                    if proposed > value + 1e-8:
                        theta, accepted = p, True
                attempts.append(info)
                if accepted:
                    break
            records.append(dict(iteration=outer + 1, initial_entropy=value, accepted=accepted,
                                attempts=attempts, seconds=time.monotonic() - start))
            save_json(out / 'progress.json', records)
            print('CONSTRUCTION', json.dumps(records[-1]), flush=True)
            if not accepted:
                break
        publication = {f'model.visual.transformer.resblocks.11.{names[i]}': theta[i].float().cpu() for i in (iA, ia, iB, ib)}
        torch.save(publication, out / 'point.pt')
        rounded = tuple(p.float().double() if i in (iA, ia, iB, ib) else p for i, p in enumerate(theta))
        with torch.no_grad():
            matches = int((native_scores(rounded).argmax(1) == native_labels).sum())
        rounded_error = native_error(rounded)
        distanceA = float((rounded[iA] - A).norm() / na)
        distanceB = float((rounded[iB] - B).norm() / B.norm())
        feasible = matches == len(native_labels) and rounded_error <= tau + 1e-8 and distanceA <= 2 and distanceB <= 4
        save_json(out / 'freeze.json', dict(path=str(out / 'point.pt'), sha256=digest(out / 'point.pt'),
            accepted_steps=sum(row['accepted'] for row in records), source_matches_after_rounding=matches,
            source_total=len(native_labels), initial_entropy=records[0]['initial_entropy'],
            final_entropy=recover_value(rounded), real_construction_images=0,
            native_error_after_rounding=rounded_error, native_limit=tau, source_feasible=feasible,
            A_relative=distanceA, B_relative=distanceB,
            coefficient_forward_updates=100 * (len(records) + sum(a['feasible'] for r in records for a in r['attempts']) + 1),
            reverse_update_steps=100 * len(records),
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            source_sha256=digest(__file__), seconds=time.monotonic() - start,
            changed_from_initial=any(not torch.equal(p, q) for p, q in zip(rounded, initial_theta))))
        print('FROZEN', (out / 'freeze.json').read_text(), flush=True)
        return

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
