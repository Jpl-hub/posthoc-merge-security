"""One fixed finite Q/K-temperature transform with source-only B distillation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize

from angular_distance import angular_scores, signed_residual
from attention_cascade import norm, parameters, suffix
from boundary_game import risk_first
from classification_completion import amplitude_target, feature_space, raw_score_target, readout
from clip_benchmark import digest
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, gelu, initialize
from verify_covariance_noise import official_labels


def cpu(values):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in values.items()}


def thermal(h, p, gain, shift=None):
    n, tokens, width = h.shape
    heads, dim = p['heads'], width // p['heads']
    x = norm(h, p['gamma1'], p['beta1'], p['eps1'])
    if shift is not None:
        x = x.clone()
        x[:, 0] += shift
    wq, wk, wv = p['W'].chunk(3)
    bq, bk, bv = p['w'].chunk(3)
    q = (x[:, 0] @ wq.T + bq).reshape(n, heads, dim) * gain
    k = (x @ wk.T + bk).reshape(n, tokens, heads, dim).transpose(1, 2) * gain
    v = (x @ wv.T + bv).reshape(n, tokens, heads, dim).transpose(1, 2)
    prob = torch.softmax(torch.einsum('nhd,nhtd->nht', q, k) / dim**.5, -1)
    avg = torch.einsum('nht,nhtd->nhd', prob, v).flatten(1)
    r = h[:, 0] + avg @ p['O'].T + p['ob']
    z = gelu(norm(r, p['gamma2'], p['beta2'], p['eps2']) @ p['A'].T + p['a'])
    return r + z @ p['B'].T + p['b'], r, z, prob


@torch.no_grad()
def task_basis(cal, p, anchor, cfg):
    width = p['A'].shape[1]
    task = p['W'][:2*width] - anchor['W'][:2*width]
    u, s, vh = torch.linalg.svd(task, full_matrices=False)
    h = cal['native']['H'][:cfg['selection_n']].cuda().double()
    out, ctx = suffix(h, p)
    jh, jr, _, teacher = raw_jacobians({'h': out + cal['native']['rounding'][:len(h)].cuda(), 'r': ctx['r']}, p)
    n, heads, dim = len(h), p['heads'], width // p['heads']
    jo = (jr @ p['O']).reshape(n, len(p['head']), heads, dim)
    dp = ctx['prob'][:, None] * (torch.einsum('nchd,nhtd->ncht', jo, ctx['v']) - (jo*ctx['average'][:, None]).sum(-1)[..., None]) / dim**.5
    dq = torch.einsum('ncht,nhtd->nchd', dp, ctx['k']).flatten(2)
    xv = ctx['x'] @ vh.T
    rq = (dq @ u[:width]) * xv[:, 0, None]
    qmode = torch.einsum('nhd,hdi->nhi', ctx['q'], u[width:].reshape(heads, dim, -1))
    rk = torch.einsum('nchi,nhi->nci', torch.einsum('ncht,nti->nchi', dp, xv), qmode)
    response = rq + rk
    angular = angular_scores(teacher)
    length = (teacher-teacher.mean(1, keepdim=True)).norm(dim=1, keepdim=True)
    response = (response-response.mean(1, keepdim=True)-angular[:, :, None]*(angular[:, :, None]*response).sum(1, keepdim=True))/length[:, :, None]
    y = teacher.argmax(1)
    weight = 1./torch.bincount(y, minlength=len(p['head']))[y]/int(y.unique().numel())
    importance = (response.square().sum(1)*weight[:, None]).sum(0)
    selected = importance.argsort(descending=True, stable=True)[:cfg['probe_rank']]
    # Independent finite check of one selected unit task-matrix mode.
    i = int(selected[0]); e = cfg['derivative_step']; delta = u[:, i, None] @ vh[i, None]
    plus, minus = dict(p), dict(p)
    plus['W'] = p['W'].clone(); minus['W'] = p['W'].clone()
    plus['W'][:2*width] += e*delta; minus['W'][:2*width] -= e*delta
    measured = (angular_scores(scores(thermal(h, plus, 1.)[0]+cal['native']['rounding'][:len(h)].cuda(), p)) - angular_scores(scores(thermal(h, minus, 1.)[0]+cal['native']['rounding'][:len(h)].cuda(), p))) / (2*e)
    error = float((measured-response[:, :, i]).abs().max())
    assert error < cfg['jacobian_tolerance'], error
    return dict(u=u[:, selected], s=s[selected], vh=vh[selected], selected=selected), dict(functional_fraction=float(importance[selected].sum()/importance.sum()), task_mode_fd_error=error, selected=selected.cpu().tolist())


@torch.no_grad()
def paired_inputs(cal, p, basis, cfg):
    caches, indices, core = [], [], []
    for group, key in [('native', 'fit'), ('scale', 'scale')]:
        raw = {k:cal[group][k] for k in ['H', 'h', 'rounding']}
        data, ids, n = risk_first(raw, cal['indices'][key], cfg['contexts_per_class'], p)
        offset = sum(len(c['H']) for c in caches)
        core.extend(range(offset, offset+n)); caches.append(data); indices.extend(ids)
    cache = {k:torch.cat([c[k] for c in caches]) for k in caches[0]}
    return pack_inputs(cache, core, p, basis, cal['basis_B'], cfg), indices, cache


@torch.no_grad()
def pack_inputs(cache, core, p, basis, b_basis, cfg):
    n, width = len(cache['H']), p['A'].shape[1]
    v = basis['vh']; vb = b_basis['vh'].cuda(); chunks=[]; natural=[]; ent=[]
    for start in range(0, n, cfg['batch_size']):
        h = cache['H'][start:start+cfg['batch_size']].cuda().double()
        x = norm(h, p['gamma1'], p['beta1'], p['eps1'])[:, 0]
        before, _, z, _ = thermal(h, p, 1.)
        natural.append((x@v.T, z@vb.T))
    rmsq=torch.cat([x[0] for x in natural]).square().mean(0).sqrt()
    rmsb=torch.cat([x[1] for x in natural]).square().mean(0).sqrt()
    signs=torch.tensor([-1., 1.], device='cuda', dtype=torch.float64)
    sq=(cfg['finite_step']*signs[:, None, None]*rmsq[None, :, None]*v[None]).flatten(0, 1)
    sb=(cfg['finite_step']*signs[:, None, None]*rmsb[None, :, None]*vb[None]).flatten(0, 1)
    take=torch.cat([torch.arange(n), torch.tensor(core).repeat_interleave(len(sq)), torch.tensor(core).repeat_interleave(len(sb))])
    shifts=torch.cat([torch.zeros(n, width), sq.cpu().repeat(len(core), 1), torch.zeros(len(core)*len(sb), width)])
    offsets=torch.cat([torch.zeros(n, p['B'].shape[1]), torch.zeros(len(core)*len(sq), p['B'].shape[1]), sb.cpu().repeat(len(core), 1)])
    for start in range(0, len(take), cfg['batch_size']):
        rows=take[start:start+cfg['batch_size']]; h=cache['H'][rows].cuda().double()
        shift=shifts[start:start+len(rows)].cuda().double(); off=offsets[start:start+len(rows)].cuda().double()
        before, _, _, prob0=thermal(h, p, 1., shift)
        base, r, z, prob1=thermal(h, p, 1.+cfg['gain_delta'], shift)
        rounding=cache['rounding'][rows].cuda()
        before += rounding+off@p['B'].T; base += rounding+off@p['B'].T; z += off
        chunks.append(dict(teacher_hidden=before, base=base, z=z, r=r))
        if start<n:
            ent.append((-(prob0*prob0.clamp_min(1e-30).log()).sum(-1).mean(1), -(prob1*prob1.clamp_min(1e-30).log()).sum(-1).mean(1)))
    packed={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    packed['teacher']=scores(packed['teacher_hidden'], p)
    return packed | dict(natural_n=n, core=core, take=take, shifts=shifts, offsets=offsets,
                        entropy_original_mean=float(torch.cat([x[0] for x in ent]).mean()), entropy_changed_mean=float(torch.cat([x[1] for x in ent]).mean()))


def predict(data, p, q, k):
    return scores(data['base']+p['B'].norm()*(data['z']@k.T)@q.T, p)


def metric(data, p, q, k, cfg):
    after=predict(data, p, q, k); n=data['natural_n']; old=data['teacher'].argmax(1); pred=after.argmax(1)
    return dict(distance=float((2*cfg['gain_delta']**2+k.square().sum()).sqrt()), B_relative_distance=float(k.norm()),
                worst_margin_drift=float(signed_residual(data['teacher'], after).abs().max()), natural_n=n,
                natural_changes=int((pred[:n]!=old[:n]).sum()), changed_by_teacher_class=torch.bincount(old[:n][pred[:n]!=old[:n]], minlength=len(p['head'])).cpu().tolist())


@torch.no_grad()
def spectral_fit(data, p, space, cfg):
    if cfg.get('distillation_target')=='centered_raw_margin_band':
        _, target, info=amplitude_target(data['base'], data['teacher_hidden'], space, p, cfg['task_budget'])
    else:
        _, target, info=raw_score_target(data['base'], data['teacher_hidden'], space, p['eps'])
    teacher=data['teacher']; y=teacher.argmax(1); rows=torch.arange(len(y), device='cuda')
    gaps=teacher[rows, y, None]-teacher; gaps[rows, y]=torch.inf
    weight=gaps.min(1).values.reciprocal().square(); weight/=weight.mean()
    z=p['B'].norm()*data['z']; gram=z.T@(weight[:, None]*z); rhs=z.T@(weight[:, None]*target)
    eigen, u=torch.linalg.eigh(gram); floor=torch.finfo(eigen.dtype).eps*max(z.shape)*eigen[-1]
    eigen=eigen.clamp_min(0.); rhs=u.T@rhs; keep=eigen>floor
    radius=(cfg['parameter_budget']**2-2*cfg['gain_delta']**2)**.5
    def solution(lam):return torch.where(keep[:, None], rhs/(eigen[:, None]+lam).clamp_min(float(floor)), 0.)
    lam=0.; coef=solution(lam)
    if coef.norm()>radius:
        low=0.; high=float(torch.linalg.matrix_norm(rhs)/radius)
        for _ in range(cfg['spectral_bisection_steps']):
            mid=(low+high)/2
            if solution(mid).norm()>radius:low=mid
            else:high=mid
        lam=high; coef=solution(lam)
    k=(u@coef).T
    return k, dict(**info, gram_numerical_rank=int(keep.sum()), lagrange_multiplier=lam,
                   weighted_fit_relative_error=float(((z@k.T-target)*weight.sqrt()[:, None]).norm()/(target*weight.sqrt()[:, None]).norm().clamp_min(1e-20)))


@torch.no_grad()
def repair(data, p, q, k, cfg):
    history=[]; memory=[]; nb=p['B'].norm(); radius2=cfg['parameter_budget']**2-2*cfg['gain_delta']**2
    for iteration in range(cfg['correction_steps']):
        before=metric(data, p, q, k, cfg)
        if before['worst_margin_drift']<=cfg['task_budget']:break
        res=signed_residual(data['teacher'], predict(data, p, q, k)); violation=res.abs().max(1).values
        order=violation.argsort(descending=True, stable=True).cpu().tolist()
        for i in order[:4]:
            if i not in memory and i not in data['core'] and len(memory)<cfg['witness_cap']:memory.append(i)
        ix=torch.tensor(list(dict.fromkeys(data['core']+memory+[order[0]])), device='cuda')
        h=data['base'][ix]+nb*(data['z'][ix]@k.T)@q.T
        jh, _, _, ss=raw_jacobians(dict(h=h, r=data['r'][ix]), p)
        j=nb*((jh@q)[:, :, :, None]*data['z'][ix, None, None, :]).flatten(2)
        old=data['teacher'][ix]; y=old.argmax(1); rows=torch.arange(len(ix), device='cuda')
        keep=torch.ones_like(old, dtype=torch.bool); keep[rows, y]=False
        gap=(old[rows, y, None]-old).masked_fill(~keep, 1.)
        matrix=((j[rows, y, None]-j)/gap[:, :, None])[keep]; rr=res[ix][keep]
        u, s, vh=torch.linalg.svd(matrix, full_matrices=False); rank=int((s>torch.finfo(s.dtype).eps*max(matrix.shape)*s[0]).sum())
        a=(u[:, :rank]*s[None, :rank]).cpu().numpy(); v=vh[:rank]; r=rr.cpu().numpy(); center=(v@k.flatten()).cpu().numpy()
        null=max(float(k.square().sum())-float(center@center), 0.); remaining=radius2-null
        scale=np.maximum(np.linalg.norm(a, axis=1), 1e-12); bound=cfg['task_budget']*cfg['slab_fraction']
        linear=LinearConstraint(a/scale[:, None], (-bound-r)/scale, (bound-r)/scale)
        ball=dict(type='ineq', fun=lambda x:remaining-np.sum((center+x)**2), jac=lambda x:-2*(center+x))
        solved=minimize(lambda x:.5*float(x@x), np.zeros(rank), jac=lambda x:x, method='SLSQP', constraints=[linear, ball], options=dict(maxiter=cfg['slab_solver_steps'], ftol=1e-11))
        excess=max(float(np.abs(a@solved.x+r).max())-bound, float(np.sum((center+solved.x)**2))-remaining, 0.)
        accepted=None
        if np.isfinite(solved.x).all() and excess<=cfg['slab_linear_tolerance']:
            delta=(v.T@torch.tensor(solved.x, device='cuda')).reshape_as(k)
            for j0 in range(cfg['backtracking_steps']):
                trial=k+.5**j0*delta
                if trial.square().sum()>radius2:continue
                if metric(data, p, q, trial, cfg)['worst_margin_drift']<before['worst_margin_drift']-1e-12:accepted=trial;break
        history.append(dict(iteration=iteration+1, before=before, source_rows=ix.cpu().tolist(), solver_success=bool(solved.success), solver_message=solved.message, constraint_excess=excess, accepted=accepted is not None))
        if accepted is None:break
        k=accepted
    return k, history


@torch.no_grad()
def construct(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True, exist_ok=False); start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    assert digest(Path(cfg['cache_reference'])/'calibration.pt')==cfg['cache_sha256']
    cal=torch.load(Path(cfg['cache_reference'])/'calibration.pt', map_location='cpu')
    anchor, expert, head, hashes=checked_models(cfg['target']); p=on_device(cal['parameters']); pa=on_device(parameters(anchor, head))
    assert torch.equal(parameters(expert, head)['W'], cal['parameters']['W'])
    basis, selection=task_basis(cal, p, pa, cfg)
    space, geometry=(readout if cfg.get('distillation_target')=='centered_raw_margin_band' else feature_space)(p)
    data, ids, cache=paired_inputs(cal, p, basis, cfg); q=space['Q']
    k, fit=spectral_fit(data, p, space, cfg); initial=metric(data, p, q, k, cfg)
    print(json.dumps(dict(stage='finite_spectral_distillation', initial=initial, fit=fit)), flush=True)
    k, history=repair(data, p, q, k, cfg); selected=metric(data, p, q, k, cfg)
    passed=selected['worst_margin_drift']<=cfg['task_budget'] and selected['distance']<=cfg['parameter_budget']
    point=out/'training_point.pt'; torch.save(dict(k=k.cpu(), output_basis=q.cpu(), task_basis=cpu(basis), B_basis=cal['basis_B'], cfg=cfg), point); pin=digest(point)
    # These counterfactuals explain the fixed point; they never select it.
    n=data['natural_n']; base=scores(data['teacher_hidden'][:n], p)
    # Use the original (unheated) gate for the compensation-only intervention.
    unheated=[]
    for start0 in range(0, n, cfg['batch_size']):
        h=cache['H'][start0:start0+cfg['batch_size']].cuda().double(); unheated.append(thermal(h, p, 1.)[2])
    solo=scores(data['teacher_hidden'][:n]+p['B'].norm()*(torch.cat(unheated)@k.T)@q.T, p)
    labels=torch.tensor(official_labels(cfg['target'], True)[ids], device='cuda'); after=predict(data, p, q, k)
    predictions=dict(indices=ids, source_scores=data['teacher'].cpu(), released_scores=after.cpu(), B_only_scores=solo.cpu())
    torch.save(predictions, out/'predictions.pt')
    torch.save(dict(parameters=cpu(p), task_basis=cpu(basis), output_basis=q.cpu(), B_basis=cal['basis_B'], native=cpu(data), indices=ids, cfg=cfg), out/'diagnostic.pt')
    result=dict(target=cfg['target'], status='task_calibration_passed_needs_independent_validation' if passed else 'closed_source_task_constraint_failure',
                initial=initial, selected=selected, task_gate_passed=passed, gain=1.+cfg['gain_delta'], entropy_original_mean=data['entropy_original_mean'], entropy_changed_mean=data['entropy_changed_mean'],
                B_only_changed=int((solo.argmax(1)!=base.argmax(1)).sum()), B_only_by_class=torch.bincount(base.argmax(1)[solo.argmax(1)!=base.argmax(1)], minlength=len(p['head'])).cpu().tolist(),
                original_correct=int((base.argmax(1)==labels).sum()), released_correct=int((after[:n].argmax(1)==labels).sum()),
                selection=selection, feature_geometry=geometry, spectral_fit=fit, corrections=len(history),
                training_point_sha256=pin, predictions_sha256=digest(out/'predictions.pt'), diagnostic_sha256=digest(out/'diagnostic.pt'), checkpoint_sha256=hashes,
                source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']), task_budget=cfg['task_budget'], parameter_budget=cfg['parameter_budget'],
                proxy_used=False, partner_used=False, test_used=False, confirmation_used=False, original_expert_updates=0, new_release_weights=0,
                read_only_task_mode_jacobians=1, compensation_jacobians=len(history), wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                scope='One fixed gain. Neither source-midpoint nor public-base proxy classification is an acceptance gate; fresh confirmation and trueTA/authorAGA follow any independently verified task-feasible point.')
    save_json(out/'results.json', result);save_json(out/'history.json', history);save_json(out/'protocol.json', cfg)
    save_json(out/'training_indices.json', {'fit':ids, 'scale':[]});print(json.dumps(result), flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol', required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text()); cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin, path
    initialize(); construct(cfg)
