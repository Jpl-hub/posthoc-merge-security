"""One fixed joint task-response direction; no optimizer or partner selection."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from angular_distance import signed_residual
from attention_cascade import input_vjp, parameters, suffix
from clip_benchmark import digest
from covariance_noise import dataset
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, initialize
from verify_cascade import suffix as numpy_suffix
from verify_levelset_audits import convert, raw_scores


@torch.no_grad()
def capture(model, training, indices):
    groups = {k: [] for k in ['H', 'z10', 'h', 'features']}
    visual = model.model.visual
    hooks = [
        visual.transformer.resblocks[10].mlp.c_proj.register_forward_pre_hook(
            lambda _m, a: groups['z10'].append(a[0].transpose(0, 1).detach().cpu())),
        visual.transformer.resblocks[11].ln_1.register_forward_pre_hook(
            lambda _m, a: groups['H'].append(a[0].transpose(0, 1).detach().cpu())),
        visual.ln_post.register_forward_pre_hook(
            lambda _m, a: groups['h'].append(a[0].detach().cpu())),
    ]
    try:
        model.cuda().eval()
        for images, _ in make_loader(training, indices, 64):
            groups['features'].append(model(images.cuda()).detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()
        model.cpu()
    return {k: torch.cat(v) for k, v in groups.items()}


def inputs(cache, p, probes, contexts, cfg):
    """Natural inputs and finite CLS activation probes, in fixed original order."""
    n = len(cache['h'])
    choices = [('natural', torch.arange(n), None)]
    for layer in range(2):
        choices.append((str(layer), contexts.repeat_interleave(2 * cfg['probe_rank']), probes[layer].repeat(len(contexts), 1)))
    for group, take, shifts in choices:
        for start in range(0, len(take), cfg['batch_size']):
            ix = take[start:start + cfg['batch_size']]
            H = cache['H'][ix].cuda().double()
            z10 = cache['z10'][ix].cuda().double()
            rounding = cache['h'][ix].cuda().double() - suffix(H, p)[0]
            offset = torch.zeros((len(ix), p['B'].shape[1]), device='cuda', dtype=torch.float64)
            if group == '0':
                shift = shifts[start:start + len(ix)]
                H[:, 0] += shift @ p['D'].T
                z10[:, 0] += shift
            elif group == '1':
                offset = shifts[start:start + len(ix)]
            h, context = suffix(H, p)
            h = h + rounding + offset @ p['B'].T
            yield group, dict(H=H, z10=z10, rounding=rounding, offset=offset, h=h, context=context)


def jacobian(batch, p, axis, complements):
    jh, jr, z11, teacher = raw_jacobians({'h': batch['h'], 'r': batch['context']['r']}, p)
    upstream = input_vjp(jr, batch['context'], p)
    upstream = torch.einsum('nctd,d->nct', upstream, axis)
    j10 = p['D'].norm() * torch.einsum('nct,ntk->nck', upstream, batch['z10'])
    j11 = p['B'].norm() * (jh @ axis)[:, :, None] * (z11 + batch['offset'])[:, None]
    combined = torch.cat([j10 @ complements[0], j11 @ complements[1]], -1)
    y = teacher.argmax(1)
    rows = torch.arange(len(y), device='cuda')
    gap = teacher[rows, y, None] - teacher
    keep = torch.ones_like(gap, dtype=torch.bool)
    keep[rows, y] = False
    assert (gap[keep] > 0).all(), 'Do not drop or redefine a tied teacher margin.'
    relative = (combined[rows, y, None] - combined) / gap.masked_fill(~keep, 1.)[:, :, None]
    return relative[keep].reshape(len(y), -1, combined.shape[-1]), teacher


def predict(batch, p, deltas):
    H = batch['H'] + (batch['z10'] @ deltas[0].T)
    q = p | {'B': p['B'] + deltas[1]}
    return batch['rounding'] + suffix(H, q)[0] + batch['offset'] @ q['B'].T


def metric(before, after):
    y = before.argmax(1)
    changed = after.argmax(1) != y
    return dict(n=len(y), changed=int(changed.sum()),
                worst_task_drift=float(signed_residual(before, after).abs().max()),
                changed_by_teacher_class=torch.bincount(y[changed], minlength=before.shape[1]).tolist())


@torch.no_grad()
def run(protocol):
    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', cfg)
    for path, pin in cfg['source_sha256'].items():
        assert digest(path) == pin, path
    assert digest(cfg['source_indices']) == cfg['source_indices_sha256']
    initialize()
    assert '3090' in torch.cuda.get_device_name()
    start = time.monotonic()
    anchor, expert, head, pins = checked_models('MNIST')
    ids = json.loads(Path(cfg['source_indices']).read_text())['fit']
    assert len(ids) == cfg['source_n'] and not set(ids) & set(cfg['excluded_witness_indices'])
    save_json(out / 'training_indices.json', dict(fit=ids, role='reused_construction_only_not_confirmation'))
    training = dataset('MNIST', 'data', expert.val_preprocess, True)
    cache = capture(expert, training, ids)
    torch.save(cache, out / 'cache.pt')
    p = on_device(parameters(expert, head))
    pa = parameters(anchor, head)
    readout = p['gamma'][:, None] * p['projection'] @ (p['head'] - p['head'].mean(0)).T
    readout -= readout.mean(0)
    axis = torch.linalg.svd(readout, full_matrices=False)[0][:, 0]
    axis *= torch.sign(axis[axis.abs().argmax()])
    bases, complements = [], []
    for key in ['D', 'B']:
        vh = torch.linalg.svd(p[key] - pa[key].cuda(), full_matrices=False)[2][:cfg['probe_rank']]
        bases.append(vh)
        complements.append(torch.linalg.qr(vh.T, mode='complete').Q[:, cfg['probe_rank']:])
    natural = []
    z11s = []
    for at in range(0, len(ids), cfg['batch_size']):
        ix = slice(at, at + cfg['batch_size'])
        natural.append(scores(cache['h'][ix].cuda().double(), p).cpu())
        z11s.append(suffix(cache['H'][ix].cuda().double(), p)[1]['z'].cpu())
    natural = torch.cat(natural)
    y = natural.argmax(1)
    contexts = torch.cat([torch.where(y == c)[0][:cfg['contexts_per_class']] for c in range(natural.shape[1])])
    assert len(contexts) == natural.shape[1] * cfg['contexts_per_class']
    activations = [cache['z10'][:, 0].double(), torch.cat(z11s)]
    probes = []
    for activation, basis in zip(activations, bases):
        rms = (activation.cuda() @ basis.T).square().mean(0).sqrt()
        probes.append((cfg['finite_step'] * torch.tensor([-1., 1.], device='cuda')[:, None, None] * rms[None, :, None] * basis[None]).flatten(0, 1))
    width = sum(c.shape[1] for c in complements)
    gram = torch.zeros((width, width), device='cuda', dtype=torch.float64)
    matrices = []
    derivative_checks = []
    for _, batch in inputs(cache, p, probes, contexts, cfg):
        j, teacher = jacobian(batch, p, axis, complements)
        if not matrices:
            for layer in range(2):
                direction = torch.zeros(width, device='cuda', dtype=torch.float64)
                start_index = 0 if layer == 0 else complements[0].shape[1]
                count = complements[layer].shape[1]
                direction[start_index:start_index+count] = torch.cos(torch.arange(count, device='cuda', dtype=torch.float64))
                direction /= direction.norm()
                split = complements[0].shape[1]
                vectors = [complements[0] @ direction[:split], complements[1] @ direction[split:]]
                test_deltas = [p[k].norm() * axis[:, None] * v[None] for k, v in zip(['D', 'B'], vectors)]
                step = cfg['derivative_step']
                plus = scores(predict(batch, p, [step*d for d in test_deltas]), p)
                minus = scores(predict(batch, p, [-step*d for d in test_deltas]), p)
                difference = (signed_residual(teacher, plus) - signed_residual(teacher, minus)) / (2*step)
                keep = torch.ones_like(difference, dtype=torch.bool)
                keep[torch.arange(len(teacher), device='cuda'), teacher.argmax(1)] = False
                measured = difference[keep].reshape(len(teacher), -1)
                analytic = j @ direction
                error = float((measured-analytic).norm() / analytic.norm().clamp_min(1e-30))
                assert error < cfg['derivative_relative_tolerance'], (layer, error)
                derivative_checks.append(dict(layer=10+layer, relative_error=error))
        j = j.flatten(0, 1)
        gram += j.T @ j
        matrices.append(j.cpu())
    design = torch.cat(matrices)
    gram /= len(design)
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) / 2)
    direction = eigenvectors[:, 0]
    direction *= torch.sign(direction[direction.abs().argmax()])
    rayleigh = float((design @ direction.cpu()).square().mean())
    assert abs(rayleigh - float(eigenvalues[0])) <= cfg['eigen_relative_tolerance'] * max(1., float(gram.abs().max()))
    split = complements[0].shape[1]
    coefficients = [complements[0] @ direction[:split], complements[1] @ direction[split:]]
    radius = cfg['parameter_budget'] * (1. - cfg['distance_reserve'])
    deltas = [(p[k] + radius * p[k].norm() * axis[:, None] * v[None]).float().double() - p[k] for k, v in zip(['D', 'B'], coefficients)]
    distance = float(sum((delta.norm() / p[k].norm()).square() for k, delta in zip(['D', 'B'], deltas)).sqrt())
    assert distance <= cfg['parameter_budget']
    before, predictions = [], {key: [] for key in ['joint', 'B10_only', 'B11_only']}
    maximum = 0.
    pn = convert(p)
    for _, batch in inputs(cache, p, probes, contexts, cfg):
        before.append(scores(batch['h'], p).cpu())
        for key, pair in [('joint', deltas), ('B10_only', [deltas[0], torch.zeros_like(deltas[1])]), ('B11_only', [torch.zeros_like(deltas[0]), deltas[1]])]:
            h = predict(batch, p, pair)
            predictions[key].append(scores(h, p).cpu())
            if key == 'joint':
                hn = batch['H'].cpu().numpy() + batch['z10'].cpu().numpy() @ pair[0].cpu().numpy().T
                qn = pn | {'B': pn['B'] + pair[1].cpu().numpy()}
                h_numpy = batch['rounding'].cpu().numpy() + numpy_suffix(hn, qn)[0] + batch['offset'].cpu().numpy() @ qn['B'].T
                error = float(abs(raw_scores(h_numpy, pn) - predictions[key][-1].numpy()).max())
                assert error < cfg['numpy_score_tolerance']
                maximum = max(maximum, error)
    before = torch.cat(before)
    predictions = {key: torch.cat(values) for key, values in predictions.items()}
    stats = {key: metric(before, value) for key, value in predictions.items()}
    passed = stats['joint']['worst_task_drift'] <= cfg['task_budget'] and stats['joint']['changed'] == 0
    state = expert.state_dict()
    edits = {f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight': (p[k] + delta).float().cpu() for layer, k, delta in zip([10, 11], ['D', 'B'], deltas)}
    protected = copy.deepcopy(expert)
    protected.load_state_dict(state | edits, strict=True)
    images = torch.stack([training[i][0] for i in ids[:16]]).cuda()
    protected.cuda(); head.cuda()
    actual = head(protected(images)).cpu().double()
    protected.cpu(); head.cpu()
    replay_error = float((actual - predictions['joint'][:16]).abs().max())
    assert replay_error < cfg['full_model_score_tolerance']
    torch.save(dict(edits=edits, axis=axis.cpu(), coefficients=[v.cpu() for v in coefficients], bases=[v.cpu() for v in bases], gram=gram.cpu(), eigenvalues=eigenvalues.cpu(), direction=direction.cpu(), predictions=predictions, before=before), out / 'diagnostic.pt')
    if passed:
        torch.save(edits, out / 'release.pt')
    report = dict(status='source_passed_freeze_before_new_confirmation' if passed else 'closed_fixed_direction_failed_finite_source_task', source_n=len(ids), task_inputs=len(before), metrics=stats, parameter_distance=distance,
        per_layer_distance=[float(delta.norm() / p[k].norm()) for k, delta in zip(['D', 'B'], deltas)],
        task_probe_leakage=[float((delta @ basis.T).norm() / delta.norm().clamp_min(1e-30)) for delta, basis in zip(deltas, bases)],
        linear_mean_square=rayleigh, minimum_eigenvalue=float(eigenvalues[0]), maximum_linear_relative_response=float((radius * design @ direction.cpu()).abs().max()),
        maximum_independent_NumPy_score_error=maximum, full_model_16_score_error=replay_error,
        analytic_full_tail_derivative_checks=derivative_checks,
        checkpoint_sha256=pins, source_sha256=cfg['source_sha256'], cache_sha256=digest(out / 'cache.pt'), diagnostic_sha256=digest(out / 'diagnostic.pt'),
        release_sha256=digest(out / 'release.pt') if passed else None, new_expert_updates=0, gradient_steps=0, new_solver_directions=1, source_images_replayed=len(ids)+16,
        partner_used=False, test_used=False, confirmation_used=False, novelty_claim=False, wall_seconds=time.monotonic()-start,
        limitation='A single source-only Rayleigh direction, not a solved nonlinear minimax, unknown-partner guarantee or validated defense. Source failure closes it without changing radius, output axis, layer, eigendirection or rank.')
    save_json(out / 'results.json', report)
    print(json.dumps(report), flush=True)


@torch.no_grad()
def confirm(protocol):
    from confirm_temperature_compensation import unused
    from ln_affine_release import observe

    cfg = json.loads(Path(protocol).read_text())
    for path, pin in cfg['source_sha256'].items():
        assert digest(path) == pin, path
    out = Path(cfg['output']).resolve()
    assert not (out / 'confirmation.pt').exists() and not (out / 'confirmation_split.json').exists()
    for name in ['release.pt', 'results.json', 'diagnostic.pt', 'independent_validation.json']:
        assert digest(out / name) == cfg['source_artifacts'][name], name
    assert json.loads((out / 'independent_validation.json').read_text())['status'] == 'passed'
    initialize()
    start = time.monotonic()
    _, expert, head, pins = checked_models('MNIST')
    p = on_device(parameters(expert, head))
    edits = torch.load(out / 'release.pt', map_location='cpu')
    original = expert.state_dict()
    distance = sum(float((v.double()-original[k].double()).norm().square()/original[k].double().norm().square()) for k, v in edits.items())**.5
    assert distance <= cfg['parameter_budget']
    deltas = [(edits[f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'].double()-original[f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'].double()).cuda() for layer in [10, 11]]
    point = torch.load(out / 'diagnostic.pt', map_location='cpu')
    bases = [v.cuda() for v in point['bases']]
    del point
    save_json(out / 'pre_confirmation_freeze.json', dict(protocol=cfg, release_sha256=cfg['source_artifacts']['release.pt'], weights_changed=False))
    training = dataset('MNIST', 'data', expert.val_preprocess, True)
    old_ids = json.loads((out / 'training_indices.json').read_text())['fit']
    split = unused(training, set(old_ids) | set(cfg['excluded_witness_indices']), cfg)
    save_json(out / 'confirmation_split.json', split)
    cache = capture(expert, training, split['indices'])
    natural = scores(cache['h'].cuda().double(), p).cpu()
    y = natural.argmax(1)
    contexts = torch.cat([torch.where(y == c)[0][:cfg['contexts_per_class']] for c in range(natural.shape[1])])
    assert len(contexts) == natural.shape[1] * cfg['contexts_per_class']
    z11 = torch.cat([suffix(H.cuda().double(), p)[1]['z'].cpu() for H in cache['H'].split(cfg['batch_size'])])
    probes = []
    for activation, basis in zip([cache['z10'][:, 0].double(), z11], bases):
        rms = (activation.cuda() @ basis.T).square().mean(0).sqrt()
        probes.append((cfg['finite_step'] * torch.tensor([-1., 1.], device='cuda')[:, None, None] * rms[None, :, None] * basis[None]).flatten(0, 1))
    before, after, maximum = [], [], 0.
    pn = convert(p)
    for _, batch in inputs(cache, p, probes, contexts, cfg):
        h = predict(batch, p, deltas)
        before.append(scores(batch['h'], p).cpu())
        after.append(scores(h, p).cpu())
        H = batch['H'].cpu().numpy() + batch['z10'].cpu().numpy() @ deltas[0].cpu().numpy().T
        qn = pn | {'B': pn['B'] + deltas[1].cpu().numpy()}
        hn = batch['rounding'].cpu().numpy() + numpy_suffix(H, qn)[0] + batch['offset'].cpu().numpy() @ qn['B'].T
        error = float(abs(raw_scores(hn, pn)-after[-1].numpy()).max())
        assert error < cfg['numpy_score_tolerance']
        maximum = max(maximum, error)
    before, after = torch.cat(before), torch.cat(after)
    released = copy.deepcopy(expert)
    released.load_state_dict(original | edits, strict=True)
    head.cuda()
    actual_before = head(cache['features'].cuda()).cpu().double()
    head.cpu()
    actual_after, _ = observe(released, head, training, split['indices'])
    control = max(float((actual_before-before[:len(y)]).abs().max()), float((actual_after-after[:len(y)]).abs().max()))
    assert control < cfg['full_model_score_tolerance']
    measured = metric(before, after)
    actual = metric(actual_before, actual_after)
    passed = all(v['worst_task_drift'] <= cfg['task_budget'] and v['changed'] == 0 for v in [measured, actual])
    labels = torch.tensor([int(training[i][1]) for i in split['indices']])
    torch.save(dict(cache=cache, before=before, after=after, actual_original=actual_before, actual_release=actual_after, labels=labels, indices=split['indices']), out / 'confirmation.pt')
    report = dict(status='passed' if passed else 'failed_fixed_candidate_closed', parameter_distance=distance, source=measured, full_model=actual,
                  full_model_score_control=control, maximum_independent_NumPy_score_error=maximum,
                  original_correct=int((actual_before.argmax(1)==labels).sum()), released_correct=int((actual_after.argmax(1)==labels).sum()),
                  release_sha256=cfg['source_artifacts']['release.pt'], raw_sha256=digest(out / 'confirmation.pt'), split_sha256=digest(out / 'confirmation_split.json'),
                  checkpoint_sha256=pins, source_sha256=digest(__file__), protocol_sha256=digest(protocol), new_indices=len(y), full_model_forward_inputs=2*len(y),
                  confirmation_redrawn=False, candidate_repaired=False, partner_used=False, test_used=False, wall_seconds=time.monotonic()-start)
    save_json(out / 'confirmation_validation.json', report)
    print(json.dumps(report), flush=True)


def distributed(protocol):
    """One source-conditioned direction across all MLP outputs, not an optimizer."""
    import torch.nn.functional as F
    from angular_distance import angular_scores

    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'protocol.json').exists(), 'Never overwrite a previous run.'
    save_json(out / 'protocol.json', cfg)
    for path, pin in cfg['source_sha256'].items():
        assert digest(path) == pin, path
    assert digest(cfg['source_indices']) == cfg['source_indices_sha256']
    initialize()
    start = time.monotonic()
    anchor, expert, head, pins = checked_models(cfg['target'])
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    ids = json.loads(Path(cfg['source_indices']).read_text())['fit']
    assert len(ids) == cfg['source_n']
    assert not set(ids) & set(cfg['excluded_witness_indices'])
    save_json(out / 'training_indices.json', dict(fit=ids, role='reused_construction_only'))
    original = {k: v.clone() for k, v in expert.state_dict().items()}
    environment = copy.deepcopy(expert)
    environment.load_state_dict({k: .5 * (v + anchor.state_dict()[k])
                                 for k, v in original.items()}, strict=True)
    training = dataset(cfg['target'], 'data', expert.val_preprocess, True)
    basis, spectra = [], []
    for i, block in enumerate(expert.model.visual.transformer.resblocks):
        w = block.mlp.c_proj.weight.double().cuda()
        w0 = anchor.model.visual.transformer.resblocks[i].mlp.c_proj.weight.double().cuda()
        u, s, vh = torch.linalg.svd(w - w0, full_matrices=False)
        r = cfg['probe_rank']
        basis.append(dict(u=u[:, :r].float(), v=vh[:r].float(), norm=float(w.norm())))
        spectra.append(dict(layer=i, retained_task_energy=float(s[:r].square().sum() / s.square().sum())))
    del anchor
    expert.cuda(); environment.cuda(); head.cuda()
    width = basis[0]['v'].shape[1]
    shape = (len(basis), cfg['probe_rank'], width)
    runtime = dict(coef=None, probe=None, collect=False)
    moment = [torch.zeros(cfg['probe_rank'], device='cuda', dtype=torch.float64) for _ in basis]
    counted = [0] * len(basis)
    handles = []

    def attach(model, scale):
        for layer, block in enumerate(model.model.visual.transformer.resblocks):
            b = basis[layer]
            def pre(_m, args, layer=layer, b=b):
                z = args[0]
                if runtime['collect']:
                    value = z[0].double() @ b['v'].double().T
                    moment[layer].add_(value.square().sum(0))
                    counted[layer] += len(value)
                if runtime['probe'] is not None and runtime['probe'][0] == layer:
                    z = z.clone()
                    z[0] += runtime['probe'][1]
                    return (z,)
            def post(_m, args, value, layer=layer, b=b):
                if runtime['coef'] is None:
                    return value
                z = args[0]
                perpendicular = z - (z @ b['v'].T) @ b['v']
                return value + scale * b['norm'] * (perpendicular @ runtime['coef'][layer].T) @ b['u'].T
            handles.extend([block.mlp.c_proj.register_forward_pre_hook(pre),
                            block.mlp.c_proj.register_forward_hook(post)])

    attach(expert, 1.)
    attach(environment, .5)
    forwards = 0
    def forward(model, x):
        nonlocal forwards
        forwards += len(x)
        feature = model(x.cuda().to(next(model.parameters()).dtype))
        return head(feature), feature

    def observe(model, indices):
        values, features, labels = [], [], []
        for x, y in make_loader(training, indices, cfg['batch_size']):
            s, f = forward(model, x)
            values.append(s.detach().cpu()); features.append(f.detach().cpu()); labels.append(y)
        return torch.cat(values), torch.cat(features), torch.cat(labels)

    def residual(before, after):
        before, after = before.double(), after.double()
        y = before.argmax(1)
        gap = before.gather(1, y[:, None]) - before
        keep = torch.ones_like(gap, dtype=torch.bool)
        keep[torch.arange(len(y)), y] = False
        assert (gap[keep] > 0).all(), 'Do not floor or remove tied gaps.'
        ds = after - before
        result = (ds.gather(1, y[:, None]) - ds) / gap.masked_fill(~keep, 1.)
        return result.masked_fill(~keep, 0.)

    def metrics(before, after):
        return dict(n=len(before), worst_task_drift=float(residual(before, after).abs().max()),
                    changed=int((before.argmax(1) != after.argmax(1)).sum()))

    runtime['collect'] = True
    before, before_features, labels = observe(expert, ids)
    runtime['collect'] = False
    assert counted == [len(ids)] * len(basis)
    rms = [(m / n).sqrt().float() for m, n in zip(moment, counted)]
    n = cfg['fit_n']
    x = torch.stack([training[i][0] for i in ids[:n]])
    teacher = before[:n]
    y = teacher.argmax(1)
    other = teacher.clone(); other[torch.arange(n), y] = -torch.inf
    gap = teacher.gather(1, y[:, None]).squeeze(1) - other.max(1).values
    core = torch.cat([torch.where(y == c)[0][gap[y == c].argsort()[:cfg['core_per_class']]]
                      for c in range(teacher.shape[1])])
    assert len(core) == cfg['core_per_class'] * teacher.shape[1]
    env_before, _, _ = observe(environment, ids[:cfg['environment_n']])
    torch.save(dict(before=before, features=before_features, labels=labels, indices=ids,
                    environment_before=env_before, core=core), out / 'source_baseline.pt')
    env_y = env_before.argmax(1).cuda()
    ref = angular_scores(env_before.cuda())
    ref_other = ref.clone(); ref_other[torch.arange(len(ref)), env_y] = -torch.inf
    ref_gap = ref.gather(1, env_y[:, None]).squeeze(1) - ref_other.max(1).values
    assert (ref_gap > 0).all()
    groups = y[:len(ref)].cuda()

    def objective(scores):
        value = angular_scores(scores)
        competitor = value.clone(); competitor[torch.arange(len(value)), env_y] = -torch.inf
        ratio = (value.gather(1, env_y[:, None]).squeeze(1) - competitor.max(1).values) / ref_gap
        damage = torch.sigmoid(-ratio / cfg['sample_temperature'])
        means = torch.stack([damage[groups == c].mean() for c in range(teacher.shape[1])])
        assert torch.isfinite(means).all(), 'Do not silently omit absent source classes.'
        tau = cfg['class_temperature']
        return -tau * (torch.logsumexp(-means / tau, 0) - np.log(len(means)))

    c0 = torch.zeros(shape, device='cuda', requires_grad=True)
    with torch.enable_grad():
        runtime['coef'] = c0
        current, _ = forward(environment, x[:len(ref)])
        value = objective(current)
        gradient = torch.autograd.grad(value, c0)[0].detach().reshape(-1)
    gradient_norm = float(gradient.norm())
    gradient /= gradient_norm
    rows, row_norms = [], []
    for pos in core.tolist():
        with torch.enable_grad():
            runtime['coef'] = c0
            score, _ = forward(expert, x[pos:pos + 1])
            yy = int(y[pos])
            targets = [k for k in range(score.shape[1]) if k != yy]
            for count, k in enumerate(targets):
                response = (score[0, yy] - score[0, k]) / (teacher[pos, yy] - teacher[pos, k]).cuda()
                g = torch.autograd.grad(response, c0, retain_graph=count + 1 < len(targets))[0]
                row = g.detach().reshape(-1).cpu()
                norm = row.double().norm()
                assert norm > 0
                row_norms.append(float(norm))
                rows.append(row / norm.float())
    design = torch.stack(rows)
    fd_scores, fd_objectives = [], []
    eps = cfg['derivative_step']
    for sign in [-1., 1.]:
        runtime['coef'] = sign * eps * gradient.reshape(shape)
        ss, _ = forward(expert, x[core])
        ee, _ = forward(environment, x[:len(ref)])
        fd_scores.append(ss.detach().cpu().double())
        fd_objectives.append(float(objective(ee)))
    measured = (fd_scores[1] - fd_scores[0]) / (2 * eps)
    core_teacher = teacher[core].double()
    cy = core_teacher.argmax(1)
    cg = core_teacher.gather(1, cy[:, None]) - core_teacher
    mask = torch.ones_like(cg, dtype=torch.bool)
    mask[torch.arange(len(cy)), cy] = False
    measured = ((measured.gather(1, cy[:, None]) - measured) / cg.masked_fill(~mask, 1.))[mask]
    measured /= torch.tensor(row_norms, dtype=torch.float64)
    native_derivative_error = float((measured - design.double() @ gradient.cpu().double()).abs().max())
    objective_derivative_error = abs((fd_objectives[1] - fd_objectives[0]) / (2 * eps) - gradient_norm) / gradient_norm
    numerical = dict(native_absolute_error=native_derivative_error,
                     public_FP32_step001_relative_error=objective_derivative_error,
                     public_FP32_gradient_norm=gradient_norm, step=eps, source_sha256=digest(__file__))
    if cfg.get('objective_derivative_check_precision') == 'FP64_control_only':
        environment.double(); head.double()
        # OpenAI CLIP's LayerNorm explicitly casts inputs to FP32. Bypass only
        # that cast for this declared FP64 reference, then restore its forward.
        norm_forwards = []
        for module in environment.modules():
            if isinstance(module, torch.nn.LayerNorm):
                norm_forwards.append((module, module.forward))
                module.forward = lambda value, m=module: F.layer_norm(
                    value, m.normalized_shape, m.weight, m.bias, m.eps)
        for b in basis:
            b['u'] = b['u'].double(); b['v'] = b['v'].double()
        with torch.enable_grad():
            scalar = torch.zeros((), device='cuda', dtype=torch.float64, requires_grad=True)
            runtime['coef'] = scalar * gradient.reshape(shape).double()
            control_score, _ = forward(environment, x[:len(ref)])
            control_value = objective(control_score)
            derivative64 = float(torch.autograd.grad(control_value, scalar)[0])
        differences = {}
        for step in [eps, cfg['FP64_derivative_step']]:
            values = []
            for sign in [-1., 1.]:
                runtime['coef'] = sign * step * gradient.reshape(shape).double()
                ss, _ = forward(environment, x[:len(ref)])
                values.append(float(objective(ss)))
            differences[str(step)] = abs((values[1] - values[0]) / (2 * step) - derivative64) / abs(derivative64)
        numeric_score_error = float((control_score.detach().float() - current.detach()).abs().max())
        gradient_control_error = abs(derivative64 - gradient_norm) / gradient_norm
        for module, previous in norm_forwards:
            module.forward = previous
        environment.float(); head.float()
        for b in basis:
            b['u'] = b['u'].float(); b['v'] = b['v'].float()
        runtime['coef'] = None
        numerical.update(public_FP64_derivative=derivative64,
            public_FP64_finite_difference_relative_errors=differences,
            public_FP32_vs_FP64_directional_gradient_relative_error=gradient_control_error,
            public_FP32_vs_FP64_score_error=numeric_score_error,
            scientific_construction_precision='FP32 unchanged; FP64 only checks derivatives')
        save_json(out / 'derivative_checks.json', numerical)
        assert numeric_score_error <= cfg['full_model_score_tolerance']
        assert gradient_control_error <= cfg['gradient_precision_relative_tolerance']
        objective_derivative_error = differences[str(cfg['FP64_derivative_step'])]
    else:
        save_json(out / 'derivative_checks.json', numerical)
    assert native_derivative_error <= cfg['derivative_absolute_tolerance']
    assert objective_derivative_error <= cfg['derivative_relative_tolerance']
    j = design.cuda().double()
    gram = j @ j.T
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) / 2)
    tolerance = torch.finfo(torch.float64).eps * max(design.shape) * eigenvalues.abs().max()
    good = eigenvalues > tolerance
    rhs = j @ gradient.double()
    dual = eigenvectors[:, good] @ ((eigenvectors[:, good].T @ rhs) / eigenvalues[good])
    direction = gradient.double() - j.T @ dual
    retained = float(direction.norm())
    null_ratio = float((j @ direction).norm() / rhs.norm())
    assert retained > 1e-8 and null_ratio <= cfg['projection_tolerance']
    direction = (direction / direction.norm()).float().reshape(shape)
    # Independent NumPy verifies the retained direction and all original rows.
    jn = design.numpy().astype(np.float64)
    gn, dn = gradient.cpu().numpy().astype(np.float64), direction.cpu().numpy().reshape(-1).astype(np.float64)
    normal_equation_error = float(np.max(np.abs((jn @ jn.T) - gram.cpu().numpy())))
    independent_null_ratio = float(np.linalg.norm(jn @ dn) / np.linalg.norm(jn @ gn))
    nd, nv = np.linalg.eigh(jn @ jn.T)
    use = nd > np.finfo(np.float64).eps * max(design.shape) * abs(nd).max()
    ndirection = gn - jn.T @ (nv[:, use] @ ((nv[:, use].T @ (jn @ gn)) / nd[use]))
    ndirection /= np.linalg.norm(ndirection)
    numpy_direction_error = float(abs(ndirection - dn).max())
    assert normal_equation_error < cfg['numpy_gram_tolerance']
    assert independent_null_ratio <= cfg['projection_tolerance']
    assert numpy_direction_error <= cfg['numpy_direction_tolerance']
    layer_responses = (j.reshape(len(j), len(basis), -1) *
                       direction.double().reshape(1, len(basis), -1)).sum(2)
    cancellation = float(layer_responses.sum(1).square().sum() / layer_responses.square().sum())
    del j, jn, gram, rows, c0
    runtime['coef'] = None
    torch.save(dict(jacobian=design, row_norms=row_norms, gradient=gradient.cpu(), direction=direction.cpu(), core=core,
                    basis=[{k: v.cpu() if torch.is_tensor(v) else v for k, v in b.items()} for b in basis],
                    source_indices=ids), out / 'direction.pt')
    # One predeclared ray; step selection uses only complete source task response.
    attempts = []
    attempt_scores = []
    selected = None
    for k in range(cfg['backtracking_steps']):
        radius = cfg['parameter_budget'] * (1 - cfg['distance_reserve']) * 2.**(-k)
        runtime['coef'] = radius * direction
        actual, _, _ = observe(expert, ids[:n])
        measured = metrics(teacher, actual)
        attempts.append(dict(radius=radius, **measured))
        attempt_scores.append(actual)
        print('DISTRIBUTED_SOURCE', attempts[-1], flush=True)
        if measured['worst_task_drift'] <= cfg['source_reserve'] and measured['changed'] == 0:
            selected = radius
            break
    save_json(out / 'attempts.json', attempts)
    torch.save(attempt_scores, out / 'attempt_scores.pt')
    if selected is None:
        save_json(out / 'results.json', dict(status='closed_no_source_feasible_point', attempts=attempts,
                  new_candidate=False, new_confirmation=False, TA_runs=0, AGA_runs=0))
        return
    runtime['coef'] = selected * direction
    hook_source, _, _ = observe(expert, ids[:n])
    env_after, _, _ = observe(environment, ids[:cfg['environment_n']])
    runtime['coef'] = selected * gradient.reshape(shape)
    uncompensated, _, _ = observe(expert, ids[:n])
    runtime['coef'] = None
    # Probe every selected task singular input direction at one source context/class.
    context = core.reshape(teacher.shape[1], cfg['core_per_class'])[:, 0]
    probe_images = x[context]
    def probes(model):
        values = []
        for layer, b in enumerate(basis):
            offsets = (cfg['finite_step'] * torch.tensor([-1., 1.], device='cuda')[:, None, None] *
                       rms[layer][None, :, None] * b['v'][None]).flatten(0, 1)
            for at in range(0, len(context) * len(offsets), cfg['batch_size']):
                take = torch.arange(at, min(at + cfg['batch_size'], len(context) * len(offsets)))
                runtime['probe'] = (layer, offsets[(take % len(offsets)).cuda()])
                s, _ = forward(model, probe_images[take // len(offsets)])
                values.append(s.detach().cpu())
            runtime['probe'] = None
        return torch.cat(values)
    probe_before = probes(expert)
    for h in handles:
        h.remove()
    handles.clear()
    released = copy.deepcopy(expert)
    edits, inside = {}, []
    for layer, b in enumerate(basis):
        c = selected * direction[layer].double()
        v = b['v'].double(); u = b['u'].double()
        delta = b['norm'] * u @ (c - (c @ v.T) @ v)
        key = f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'
        edits[key] = (original[key].cuda().double() + delta).float().cpu()
        real = edits[key].double() - original[key].double()
        inside.append(dict(layer=layer, relative_distance=float(real.norm() / original[key].double().norm()),
                           task_right_response_norm=float((real.cuda() @ v.T).norm())))
    distance = sum(t['relative_distance']**2 for t in inside)**.5
    assert distance <= cfg['parameter_budget']
    released.load_state_dict(original | edits, strict=True)
    from aga_coordinate_audit import permutation
    ref = cfg['original_AGA_reference']
    assert digest(ref['path']) == ref['sha256']
    recovered = torch.load(ref['path'], map_location='cpu')
    transported = []
    for layer in range(len(basis)):
        prefix = f'model.visual.transformer.resblocks.{layer}.mlp.'
        ak, bk = prefix + 'c_fc.weight', prefix + 'c_proj.weight'
        order = permutation(original[ak], recovered[ak])
        assert torch.equal(recovered[bk], original[bk][:, order])
        delta = edits[bk].double() - original[bk].double()
        assert torch.equal(recovered[bk].double() + delta[:, order], edits[bk][:, order].double())
        transported.append(dict(layer=layer, permutation_sha256=__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest(),
                                edit_norm_before=float(delta.norm()), edit_norm_after=float(delta[:, order].norm())))
    save_json(out / 'AGA_transport.json', dict(status='fixed_B_transport_verified_against_cached_author_baseline',
              layers=transported, reference=ref, new_author_AGA_calls=0,
              scope='B-only edits leave all A and Attention tensors unchanged. Exact permutation transport is not attack/fusion classification performance. Cached baseline loaded after the unique source direction and radius were fixed.'))
    released.cuda(); attach(released, 0.)
    after, after_features, actual_labels = observe(released, ids)
    probe_after = probes(released)
    np.testing.assert_array_equal(actual_labels.numpy(), labels.numpy())
    source_check, probe_check = metrics(before, after), metrics(probe_before, probe_after)
    hook_error = float((after[:n] - hook_source).abs().max())
    assert hook_error <= cfg['full_model_score_tolerance']
    # NumPy recomputes the complete gap ratios and classifier readout.
    head_weights = head.weight.detach().cpu().double().numpy()
    maximum_head_error = 0.
    for features, actual in [(before_features, before), (after_features, after)]:
        f = features.double().numpy()
        expected = (f / np.linalg.norm(f, axis=1, keepdims=True)) @ head_weights.T
        maximum_head_error = max(maximum_head_error, float(abs(expected - actual.numpy()).max()))
    assert maximum_head_error <= cfg['full_model_score_tolerance']
    source_r = residual(before, after).numpy()
    yy = before.numpy().argmax(1); ix = np.arange(len(yy))
    gap = before.double().numpy()[ix, yy, None] - before.double().numpy()
    gap[ix, yy] = 1.
    ds = after.double().numpy() - before.double().numpy()
    check = (ds[ix, yy, None] - ds) / gap; check[ix, yy] = 0.
    np.testing.assert_allclose(source_r, check, atol=1e-10, rtol=1e-10)
    eligible = max(source_check['worst_task_drift'], probe_check['worst_task_drift']) <= cfg['task_budget']
    eligible &= source_check['changed'] == 0 and probe_check['changed'] == 0
    torch.save(dict(edits=edits, original={k: original[k] for k in edits}), out / 'point.pt')
    torch.save(dict(before=before, after=after, labels=labels, source_indices=ids,
                    probe_before=probe_before, probe_after=probe_after, hook_source=hook_source,
                    uncompensated=uncompensated, environment_before=env_before,
                    environment_after=env_after), out / 'scores.pt')
    result = dict(status='source_feasible_requires_frozen_new_confirmation' if eligible else 'closed_complete_source_or_probe_failure',
        parameter_distance=distance, per_layer=inside, source=source_check, task_probes=probe_check,
        uncompensated_source=metrics(teacher, uncompensated), environment=metrics(env_before, env_after),
        selected_radius=selected, attempts=attempts, retained_public_gradient_fraction=retained,
        native_linear_cancellation_energy_ratio=cancellation, task_jacobian_rows=len(design),
        task_jacobian_rank=int(good.sum()), projection_null_ratio=null_ratio,
        independent_NumPy_null_ratio=independent_null_ratio, NumPy_gram_error=normal_equation_error,
        independent_NumPy_direction_error=numpy_direction_error, native_derivative_error=native_derivative_error,
        objective_derivative_relative_error=objective_derivative_error,
        NumPy_head_error=maximum_head_error, materialized_hook_score_error=hook_error,
        source_correct=int((before.argmax(1) == labels).sum()), release_correct=int((after.argmax(1) == labels).sum()),
        source_sha256=digest(__file__), protocol_sha256=digest(protocol), checkpoint_sha256=pins,
        point_sha256=digest(out / 'point.pt'), scores_sha256=digest(out / 'scores.pt'),
        direction_sha256=digest(out / 'direction.pt'), task_SVD_spectra=spectra,
        images_forwarded=forwards, source_gradient_calls=1+len(design), expert_updates=0,
        new_confirmation=False, new_candidate=bool(eligible), TA_runs=0, AGA_runs=0,
        partner_used=False, test_used=False, wall_seconds=time.monotonic()-start,
        limitation='One multilayer source-conditioned linear direction with full nonlinear source/probe gates. Public half-task counterfactual is not real fusion/AGA or an unknown-partner guarantee. Projection/SVD are standard tools, not novelty claims.')
    save_json(out / 'results.json', result)
    for h in handles:
        h.remove()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    with torch.no_grad():
        distributed(args.protocol) if args.distributed else confirm(args.protocol) if args.confirm else run(args.protocol)
