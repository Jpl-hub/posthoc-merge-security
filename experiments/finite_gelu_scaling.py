"""One source-only finite QuickGELU scaling game, with exact endpoint checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from angular_distance import signed_residual
from boundary_game import risk_first
from clip_benchmark import digest
from finite_compensation_transport import geometry, sigma
from mixed_compensation_audit import ln
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, gelu, initialize, parameters
from verify_covariance_noise import official_labels


def cpu(p):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in p.items()}


@torch.no_grad()
def pack(cache, core, p, basis, cfg):
    """B-input probes stay explicit: rescaling B also rescales their offsets."""
    x, h = cache['x'], cache['h']; z = gelu(x@p['A'].T+p['a'])
    base = h-z@p['B'].T; va, vb = basis['A']['vh'], basis['B']['vh']
    r = len(va); assert r == len(vb) == cfg['probe_rank']
    sign = torch.tensor([-1., 1.], device=x.device, dtype=x.dtype)
    ra = (x@va.T).square().mean(0).sqrt(); rb = (z@vb.T).square().mean(0).sqrt()
    sa = (cfg['finite_step']*sign[:, None, None]*ra[None, :, None]*va[None]).flatten(0, 1)
    sb = (cfg['finite_step']*sign[:, None, None]*rb[None, :, None]*vb[None]).flatten(0, 1)
    take = torch.cat([torch.arange(len(x), device=x.device), core.repeat_interleave(2*r), core.repeat_interleave(2*r)])
    xx = torch.cat([x, (x[core, None]+sa[None]).flatten(0, 1), x[core].repeat_interleave(2*r, 0)])
    offset = torch.cat([torch.zeros(len(x)+len(core)*2*r, p['B'].shape[1], device=x.device, dtype=x.dtype), sb.repeat(len(core), 1)])
    return dict(x=xx, base=base[take], offset=offset, natural_n=len(x), core=core,
                shifts_a=sa, shifts_b=sb)


@torch.no_grad()
def inputs(cfg, p, basis):
    boundary = torch.load(cfg['boundary_diagnostic'], map_location='cpu')
    old = torch.load(cfg['old_calibration'], map_location='cpu'); caches, ids, core = [], [], []
    for cache, which in [(boundary['screen'], boundary['split']['indices']), (old['scale'], old['indices']['scale'])]:
        c, order, n = risk_first(cache, which, cfg['contexts_per_class'], p)
        offset = sum(len(g['h']) for g in caches); core.extend(range(offset, offset+n)); caches.append(c); ids.extend(order)
    raw = {k: torch.cat([c[k] for c in caches]).cuda() for k in ['r', 'h']}
    cache = dict(x=ln(raw['r'], p), h=raw['h'])
    return pack(cache, torch.tensor(core, device='cuda'), p, basis, cfg), cache, ids


def gains(p, classes, cfg):
    alpha = p['A'].square().sum(1)/p['A'].square().sum()
    if p['a'].norm() > 0: alpha += p['a'].square()/p['a'].square().sum()
    beta = p['B'].square().sum(0)/p['B'].square().sum()
    def cost(s): return alpha*(s-1).square()+beta*(1/s-1).square()
    unit = cfg['parameter_budget']**2/classes
    low = torch.ones_like(alpha); high = 1+(unit/alpha).sqrt()
    assert bool((alpha > 0).all()) and bool((cost(high) >= unit).all())
    for _ in range(cfg['root_steps']):
        mid = (low+high)/2; feasible = cost(mid) <= unit
        low = torch.where(feasible, mid, low); high = torch.where(feasible, high, mid)
    assert bool((cost(low) <= unit).all())
    return low, cost(low)


@torch.no_grad()
def construct(cfg):
    start = time.monotonic(); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    for file, pin in cfg['inputs'].items(): assert digest(file) == pin, file
    source = torch.load(cfg['mixed_diagnostic'], map_location='cpu')['parameters']; p = on_device(source)
    anchor, expert, head, pins = checked_models(cfg['target'])
    exact = parameters(expert, head); base = parameters(anchor, head)
    for key, value in exact.items():
        if torch.is_tensor(value): assert torch.equal(value, source[key]), key
    basis = {site: on_device(b) for site, b in torch.load(cfg['basis_diagnostic'], map_location='cpu')['basis'].items()}
    provenance = {}
    for site, b in basis.items():
        task = exact[site]-base[site]; u, v = b['u'].cpu(), b['vh'].cpu(); sv = torch.diag(u.T@task@v.T)
        error = float((task@v.T-u*sv).norm()/(task@v.T).norm())
        assert error < 1e-8
        provenance[site] = dict(rank=len(v), input_dimension=v.shape[1], output_dimension=u.shape[0], task_SVD_equation_error=error)
    del anchor, expert, head
    packed, cache, ids = inputs(cfg, p, basis); pre = packed['x']@p['A'].T+p['a']; z = gelu(pre)
    h = packed['base']+(z+packed['offset'])@p['B'].T; teacher = scores(h, p); classes = min(teacher.shape[1], pre.shape[1])
    s, cost = gains(p, classes, cfg)
    end = (gelu(pre*s)+packed['offset'])/s-(z+packed['offset'])
    mid = .5*(1+1/s)*(gelu(.5*(1+s)*pre)+packed['offset'])-(z+packed['offset'])
    # Analytic readout Jacobian acts on the EXACT finite hidden change. It is
    # a generator constraint only; all task inputs are validated nonlinearly.
    core = packed['core']; jh, _, _, _ = raw_jacobians(dict(h=h[core], r=torch.zeros_like(h[core])), p)
    t = teacher[core]; y = t.argmax(1); rows = torch.arange(len(t), device='cuda')
    keep = torch.ones_like(t, dtype=torch.bool); keep[rows, y] = False
    gap = (t[rows, y, None]-t).masked_fill(~keep, 1.)
    native = (((jh[rows, y, None]-jh)/gap[:, :, None]@p['B'])*end[core, None, :])[keep].cpu().numpy()
    m, b = geometry(p); nfit = cfg['objective_n']; numer = h[:nfit]@m+sigma(h[:nfit], p)*b
    y = numer.argmax(1); row = torch.arange(nfit, device='cuda'); other = numer.clone(); other[row, y] = -torch.inf; k = other.argmax(1)
    gap = numer[row, y]-numer[row, k]; assert bool((gap > 0).all())
    direct = (m[:, y]-m[:, k]).T@p['B']
    variance_bound = (p['B']-p['B'].mean(0)).norm(dim=0)/p['B'].shape[0]**.5
    lower = (-mid[:nfit]*direct-mid[:nfit].abs()*variance_bound*(b[y]-b[k]).abs()[:, None])/gap[:, None]
    assert len(y.unique()) == classes
    harm = torch.stack([lower[y == c].mean(0) for c in range(classes)]).cpu().numpy()
    width = len(s); matrix = np.vstack([np.column_stack([native, np.zeros(len(native))]),
                                      np.column_stack([-harm, np.ones(classes)]), np.r_[np.ones(width), 0.][None]])
    low = np.r_[np.full(len(native), -cfg['generator_margin_fraction']), np.full(classes, -np.inf), -np.inf]
    high = np.r_[np.full(len(native), cfg['generator_margin_fraction']), np.zeros(classes), classes]
    objective = np.r_[-np.ones(width)/classes, -1.]
    solved = milp(objective, integrality=np.r_[np.ones(width), 0.], bounds=Bounds(np.zeros(width+1), np.r_[np.ones(width), np.inf]),
                  constraints=LinearConstraint(matrix, low, high), options=dict(time_limit=cfg['solver_seconds'], node_limit=cfg['solver_node_limit'], mip_rel_gap=cfg['solver_gap']))
    solver = dict(status=int(solved.status), message=solved.message, incumbent=solved.x is not None)
    mask = torch.zeros_like(s)
    if solved.x is not None:
        assert np.abs(solved.x[:-1]-np.round(solved.x[:-1])).max() < 1e-5
        mask = torch.tensor(np.round(solved.x[:-1]), device='cuda'); v = np.r_[mask.cpu().numpy(), solved.x[-1]]
        assert np.max(np.maximum(matrix@v-high, low-matrix@v)) < 1e-6
        solver.update(objective=float(solved.fun), dual_bound=float(solved.mip_dual_bound), relative_gap=float(solved.mip_gap), nodes=int(solved.mip_node_count), worst_class_lower_harm=float(solved.x[-1]))
    selected_gain = 1+mask*(s-1)
    # Freeze one selected transform BEFORE exact validation; never repair it
    # or select another mask after seeing task or real fusion predictions.
    torch.save(dict(gain=selected_gain.cpu(), mask=mask.cpu(), available_gain=s.cpu(), cfg=cfg), out/'point.pt')
    point_pin = digest(out/'point.pt')
    after = scores(h+(end*mask)@p['B'].T, p); n = packed['natural_n']
    midpoint_hidden = h[:n]+(mid[:n]*mask)@p['B'].T; midpoint = scores(midpoint_hidden, p)
    drift = float(signed_residual(teacher, after).abs().max()); distance = float((cost@mask).sqrt())
    source_change = after[:n].argmax(1) != teacher[:n].argmax(1); middle_change = midpoint.argmax(1) != teacher[:n].argmax(1)
    total_lower = lower@mask; actual_numer = midpoint_hidden[:nfit]@m+sigma(midpoint_hidden[:nfit], p)*b
    actual_harm = 1-(actual_numer[row, y]-actual_numer[row, k])/gap
    bound_error = float((total_lower-actual_harm).max()); assert bound_error < 1e-9
    labels = torch.tensor(official_labels(cfg['target'], True)[ids], device='cuda')
    passed = bool(mask.sum() > 0 and distance <= cfg['parameter_budget'] and drift <= cfg['task_budget'] and not source_change.any())
    torch.save(dict(source_scores=teacher.cpu(), released_scores=after.cpu(), midpoint_scores=midpoint.cpu(),
                    midpoint_lower=total_lower.cpu(), midpoint_actual_harm=actual_harm.cpu(), objective_winner=y.cpu(), objective_competitor=k.cpu(), indices=ids), out/'predictions.pt')
    torch.save(dict(parameters=cpu(p), basis={k: cpu(v) for k, v in basis.items()}, cache=cpu(cache), core=core.cpu(),
                    shifts_a=packed['shifts_a'].cpu(), shifts_b=packed['shifts_b'].cpu(), indices=ids, cfg=cfg), out/'diagnostic.pt')
    result = dict(status='passed_task_calibration' if passed else 'closed_no_task_feasible_scaling_point', target=cfg['target'], method='finite_QuickGELU_scaling_class_maxmin',
                  task_gate_passed=passed, parameter_distance=distance, worst_margin_drift=drift, natural_n=n, all_task_inputs=len(h), selected_neurons=int(mask.sum()),
                  selected_indices=mask.nonzero().flatten().cpu().tolist(), selected_gains=selected_gain[mask.bool()].cpu().tolist(),
                  original_correct=int((teacher[:n].argmax(1) == labels).sum()), released_correct=int((after[:n].argmax(1) == labels).sum()), natural_changes=int(source_change.sum()),
                  midpoint_changes=int(middle_change.sum()), midpoint_changed_by_teacher_class=torch.bincount(teacher[:n].argmax(1)[middle_change], minlength=classes).cpu().tolist(),
                  certified_fixed_pair_crossings=int((total_lower > 1).sum()), finite_lower_bound_violation=bound_error,
                  class_lower_harm=[float(total_lower[y == c].mean()) for c in range(classes)], solver=solver, task_SVD_provenance=provenance,
                  point_sha256=point_pin, diagnostic_sha256=digest(out/'diagnostic.pt'), predictions_sha256=digest(out/'predictions.pt'), checkpoint_sha256=pins,
                  source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']), wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  integer_program_calls=1, analytic_readout_jacobian_calls=1, reverse_gradient_calls=0, original_expert_updates=0, new_model_forwards=0,
                  new_release_weights=0, partner_used=False, test_used=False, confirmation_used=False, midpoint_or_proxy_prediction_gate=False,
                  limitation='One known scaling geometry with exact QuickGELU residuals. Class simplex adversary is not an unknown-peer adversary. Finite source-local midpoint model has explicit unbounded peer mismatch remainders. Scale canonicalization is a recovery risk; no claim of novelty or robust defense.')
    save_json(out/'results.json', result); save_json(out/'protocol.json', cfg); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); construct(cfg)
