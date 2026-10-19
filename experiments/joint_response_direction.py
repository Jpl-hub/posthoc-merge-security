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
    groups = {k: [] for k in ['H', 'z10', 'h']}
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
            model(images.cuda())
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
    out = Path(cfg['output'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    with torch.no_grad():
        run(args.protocol)
