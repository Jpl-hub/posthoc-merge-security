"""One source-only score-eversion / upstream-restoration feasibility attempt."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from classification_completion import readout, sigma
from clip_benchmark import digest
from finite_gelu_scaling import cpu, inputs
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import checked_models, gelu, initialize, parameters
from verify_covariance_noise import official_labels


def score_jacobian(h, p):
    c = h-h.mean(1, keepdim=True)
    sd = (c.square().mean(1, keepdim=True)+p['eps']).sqrt()
    t = (c/sd*p['gamma']+p['beta'])@p['projection']
    tn = t.norm(dim=1, keepdim=True); f = t/tn; s = f@p['head'].T
    j = (p['head'][None]-s[:, :, None]*f[:, None])/tn[:, None]
    j = (j@p['projection'].T)*p['gamma']
    j = (j-j.mean(-1, keepdim=True)-c[:, None]*(j*c[:, None]).mean(-1, keepdim=True)/sd[:, None].square())/sd[:, None]
    return j, s


def inverse_numerator(h, desired, space, eps):
    """Minimum class-plane change with exactly the prescribed finite numerator."""
    q, m, v, w, a = [space[k] for k in ['Q', 'M', 'v', 'w', 'a']]
    g = q.T@m
    projected = (desired-h@m)@g.T@torch.linalg.inv(g@g.T)
    star = h+projected@q.T
    c = star-star.mean(1, keepdim=True); wc = w-w.mean()
    b0 = (c*wc).mean(1, keepdim=True)
    c0 = c.square().mean(1, keepdim=True)+eps
    rho = c0/((b0.square()+a*c0).sqrt()+b0)
    target_q = projected-rho*v
    result = h+target_q@q.T
    assert float((sigma(result, eps)-rho).abs().max()) < 1e-10
    assert float((result@m+rho*space['b']-desired).abs().max()) < 1e-8
    return result, target_q


def ball_ridge(values, vectors, rhs, radius, cfg):
    """PSD spectral trust-region least squares; root is not a parameter scan."""
    tolerance = torch.finfo(values.dtype).eps*cfg['spectral_dimension']*values.abs().max()
    assert float(values.min()) > -100*tolerance
    d = values.clamp_min(0); good = d > tolerance
    rr = vectors.T@rhs
    def solve(mu):
        inv = 1/(d+mu) if mu > 0 else torch.where(good, 1/d.clamp_min(tolerance), 0.)
        return vectors@(rr*inv[:, None])
    point = solve(0.)
    if float(point.norm()) <= radius:
        return point, dict(mu=0., rank=int(good.sum()), radius=radius, boundary=False)
    low = 0.; high = max(float(d.max()), 1e-12)
    for _ in range(cfg['root_bracket_steps']):
        if float(solve(high).norm()) <= radius: break
        high *= 2
    assert float(solve(high).norm()) <= radius
    for _ in range(cfg['root_steps']):
        mid = .5*(low+high)
        if float(solve(mid).norm()) > radius: low = mid
        else: high = mid
    point = solve(high)
    return point, dict(mu=high, rank=int(good.sum()), radius=radius, boundary=True)


def stats(before, after, labels):
    y, pred = before.argmax(1), after.argmax(1); changed = y != pred
    records = []
    for c in range(before.shape[1]):
        take = y == c
        records.append(dict(teacher_class=c, n=int(take.sum()), changed=int(changed[take].sum()),
                            changed_fraction=float(changed[take].double().mean())))
    return dict(n=len(y), original_correct=int((y == labels).sum()), transformed_correct=int((pred == labels).sum()),
                changed=int(changed.sum()), per_teacher_class=records)


@torch.no_grad()
def construct(cfg):
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    source = torch.load(cfg['mixed_diagnostic'], map_location='cpu')['parameters']; p = on_device(source)
    anchor, expert, head, pins = checked_models(cfg['target'])
    exact, base = parameters(expert, head), parameters(anchor, head)
    for key, value in exact.items():
        if torch.is_tensor(value): assert torch.equal(value, source[key]), key
    basis = {k: on_device(v) for k, v in torch.load(cfg['basis_diagnostic'], map_location='cpu')['basis'].items()}
    provenance = {}
    for site, b in basis.items():
        task = exact[site]-base[site]; u, v = b['u'].cpu(), b['vh'].cpu(); sv = torch.diag(u.T@task@v.T)
        err = float((task@v.T-u*sv).norm()/(task@v.T).norm()); assert err < 1e-8
        provenance[site] = dict(rank=len(v), task_SVD_equation_error=err)
    del anchor, expert, head
    packed, cache, ids = inputs(cfg, p, basis); n = packed['natural_n']; fit_n = cfg['fit_n']
    labels = torch.tensor(official_labels(cfg['target'], True)[ids], device='cuda')
    pre = packed['x']@p['A'].T+p['a']; hidden = gelu(pre)+packed['offset']
    h = packed['base']+hidden@p['B'].T; teacher = scores(h, p)
    space, geometry = readout(p); q = space['Q']; va = basis['A']['vh']
    norm_a, norm_b = p['A'].norm(), p['B'].norm()
    radius_b = radius_a = cfg['parameter_budget']/(2**.5)
    numerator = h[:fit_n]@space['M']+sigma(h[:fit_n], p['eps'])*space['b']
    ideal, target_q = inverse_numerator(h[:fit_n], -numerator, space, p['eps'])
    ideal_scores = scores(ideal, p)
    assert bool((ideal_scores.argmax(1) != teacher[:fit_n].argmax(1)).all())
    y = teacher[:fit_n].argmax(1); count = torch.bincount(y, minlength=teacher.shape[1])
    assert bool((count > 0).all()); weights = 1./count[y].double()/len(count)
    zfit = hidden[:fit_n]; design = weights.sqrt()[:, None]*zfit
    target = weights.sqrt()[:, None]*target_q/norm_b
    gram = design.T@design; rhs = design.T@target
    eigen, vectors = torch.linalg.eigh(gram)
    kb_t, b_solver = ball_ridge(eigen, vectors, rhs, radius_b, cfg|dict(spectral_dimension=len(gram)))
    kb = kb_t.T; db = norm_b*q@kb; changed_b = p['B']+db
    b_score = scores(packed['base']+hidden@changed_b.T, p)
    b_stats = stats(teacher[:n], b_score[:n], labels)
    harm_classes = sum(v['changed_fraction'] >= cfg['minimum_class_flip_fraction'] for v in b_stats['per_teacher_class'])
    gate_b = harm_classes >= cfg['minimum_harmed_classes']
    # Fix the downstream point before any compensation or independent data.
    torch.save(dict(K=kb.cpu(), delta_B=db.cpu(), b_solver=b_solver, cfg=cfg), out/'downstream_point.pt')
    print(json.dumps(dict(stage='downstream_frozen', distance=float(kb.norm()), changed=b_stats['changed'],
                          harm_classes=harm_classes, compensation_eligible=gate_b)), flush=True)
    task_input = packed['x']@va.T; c = torch.zeros(p['A'].shape[0], len(va), device='cuda', dtype=torch.float64)
    def predict(cc, selected=None):
        sel = slice(None) if selected is None else selected
        u = pre[sel]+norm_a*task_input[sel]@cc.T
        return scores(packed['base'][sel]+(gelu(u)+packed['offset'][sel])@changed_b.T, p)
    def metrics(ss, cc):
        rr = signed_residual(teacher, ss)
        return dict(worst_margin_drift=float(rr.abs().max()), parameter_distance=float((kb.square().sum()+cc.square().sum()).sqrt()),
                    upstream_distance=float(cc.norm()), natural_changes=int((ss[:n].argmax(1) != teacher[:n].argmax(1)).sum()))
    current = b_score; history = []; steps = []; active = set(packed['core'].cpu().tolist()); fd_errors = []
    initial = metrics(current, c)
    if gate_b:
        for iteration in range(cfg['compensation_steps']):
            before = metrics(current, c)
            if before['worst_margin_drift'] <= cfg['task_budget']: break
            residual = signed_residual(teacher, current); winner = teacher.argmax(1)
            # The task opponent selects the worst input/probe/competitor per class.
            row_risk = residual.abs().max(1).values
            for cls in range(teacher.shape[1]):
                members = (winner == cls).nonzero().flatten()
                active.add(int(members[row_risk[members].argmax()]))
            selected = torch.tensor(sorted(active), device='cuda'); xx = task_input[selected]
            u = pre[selected]+norm_a*xx@c.T; prob = torch.sigmoid(1.702*u)
            slope = prob+1.702*u*prob*(1-prob)
            h1 = packed['base'][selected]+(gelu(u)+packed['offset'][selected])@changed_b.T
            jh, now = score_jacobian(h1, p); t = teacher[selected]; yy = t.argmax(1); rows = torch.arange(len(t), device='cuda')
            keep = torch.ones_like(t, dtype=torch.bool); keep[rows, yy] = False
            gap = (t[rows, yy, None]-t).masked_fill(~keep, 1.); assert float(gap[keep].min()) > 1e-8
            aj = norm_a*(jh@changed_b)*slope[:, None]
            aa = ((aj[rows, yy, None]-aj)/gap[:, :, None])[keep]
            ss = xx[:, None].expand(-1, t.shape[1], -1)[keep]
            r = signed_residual(t, now)[keep]
            # J is a row-wise Kronecker product; do not materialize 98k columns.
            jj = (aa@aa.T)*(ss@ss.T)
            d, v = torch.linalg.eigh(jj); tol = torch.finfo(d.dtype).eps*max(len(jj), c.numel())*d.abs().max()
            assert float(d.min()) > -100*tol
            positive = d > tol; spectral = d.clamp_min(0)
            rhs1 = r-(aa*(ss@c.T)).sum(1)
            coeff = v.T@rhs1
            def restore(mu):
                inv = 1/(spectral+mu) if mu > 0 else torch.where(positive, 1/spectral.clamp_min(tol), 0.)
                dual = v@(coeff*inv)
                return -aa.T@(dual[:, None]*ss), dual
            proposal, dual = restore(0.); mu = 0.
            if float(proposal.norm()) > radius_a:
                low = 0.; high = max(float(spectral.max()), 1e-12)
                for _ in range(cfg['root_bracket_steps']):
                    if float(restore(high)[0].norm()) <= radius_a: break
                    high *= 2
                assert float(restore(high)[0].norm()) <= radius_a
                for _ in range(cfg['root_steps']):
                    mid = .5*(low+high)
                    if float(restore(mid)[0].norm()) > radius_a: low = mid
                    else: high = mid
                mu = high; proposal, dual = restore(mu)
            direction = proposal-c
            if float(direction.norm()) > 0:
                unit = direction/direction.norm(); eps = cfg['derivative_step']
                measured = (signed_residual(t, predict(c+eps*unit, selected))-signed_residual(t, predict(c-eps*unit, selected)))/(2*eps)
                predicted = (aa*(ss@unit.T)).sum(1)
                err = float((predicted-measured[keep]).abs().max()); fd_errors.append(err); assert err < cfg['derivative_tolerance']
            trials = []; accepted = None
            for j in range(cfg['backtracking_steps']):
                fraction = .5**j; trial = c+fraction*direction; trial_scores = predict(trial); result = metrics(trial_scores, trial)
                trials.append(dict(fraction=fraction, **result))
                if result['parameter_distance'] <= cfg['parameter_budget']*(1+1e-12) and result['worst_margin_drift'] < before['worst_margin_drift']-1e-12:
                    accepted = (trial, trial_scores); break
            steps.append(dict(C=c.cpu(), selected=selected.cpu(), factor_a=aa.cpu(), factor_s=ss.cpu(), residual=r.cpu(),
                              linear_rhs=rhs1.cpu(), dual=dual.cpu(), proposal=proposal.cpu(), mu=mu))
            history.append(dict(iteration=iteration+1, active_inputs=len(selected), task_rows=len(r), response_rank=int(positive.sum()),
                                proposal_upstream_distance=float(proposal.norm()), mu=mu, before=before, trials=trials, accepted=accepted is not None))
            print(json.dumps(dict(stage='compensation', **history[-1])), flush=True)
            if accepted is None: break
            c, current = accepted
    final = metrics(current, c)
    task_passed = bool(gate_b and final['worst_margin_drift'] <= cfg['task_budget'] and final['parameter_distance'] <= cfg['parameter_budget'] and final['natural_changes'] == 0)
    da = norm_a*c@va
    observed_distance = float(((da.norm()/norm_a)**2+(db.norm()/norm_b)**2).sqrt())
    assert abs(observed_distance-final['parameter_distance']) < 1e-12
    status = 'passed_source_task_feasibility' if task_passed else ('closed_downstream_multiclass_damage_insufficient' if not gate_b else 'closed_upstream_compensation_infeasible_under_fixed_protocol')
    torch.save(dict(C=c.cpu(), K=kb.cpu(), delta_A=da.cpu(), delta_B=db.cpu(), cfg=cfg), out/'point.pt')
    torch.save(dict(parameters=cpu(p), basis={k: cpu(v) for k, v in basis.items()}, cache=cpu(cache), indices=ids,
                    core=packed['core'].cpu(), shifts_a=packed['shifts_a'].cpu(), shifts_b=packed['shifts_b'].cpu(),
                    space=cpu(space), target_q=target_q.cpu(), ideal_hidden=ideal.cpu(), b_solver=b_solver, steps=steps, cfg=cfg), out/'diagnostic.pt')
    torch.save(dict(teacher=teacher.cpu(), downstream=b_score.cpu(), joint=current.cpu(), indices=ids), out/'predictions.pt')
    report = dict(target=cfg['target'], method='bounded_classifier_eversion_then_upstream_nonlinear_task_retraction', status=status,
                  source_task_passed=task_passed, downstream_task_stats=b_stats, joint_task_stats=stats(teacher[:n], current[:n], labels),
                  downstream_harmed_classes=harm_classes, downstream_gate_passed=gate_b, initial=initial, final=final, geometry=geometry,
                  downstream_solver=b_solver, task_SVD_provenance=provenance, training_fit_n=fit_n, natural_n=n, all_task_inputs=len(teacher),
                  history=history, analytic_derivative_maximum_error=max(fd_errors, default=0.), original_expert_updates=0,
                  publication_correction_updates=sum(s['accepted'] for s in history), analytic_jacobian_calls=len(steps),
                  reverse_autograd_calls=0, downstream_linear_solve_calls=1, new_full_model_forwards=0, new_release_weights=0,
                  partner_used=False, test_used=False, confirmation_used=False, midpoint_or_proxy_used=False,
                  checkpoint_sha256=pins, source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']),
                  wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  limitation='A single constructive source-only feasibility point; finite restoration failure does not prove nonexistence. Source B crossings are a route-specific design goal, not a universal or sufficient fusion condition. No unknown-peer coverage theorem and no true attack evaluation unless all preservation checks pass.')
    for name in ['downstream_point', 'point', 'diagnostic', 'predictions']: report[name+'_sha256'] = digest(out/(name+'.pt'))
    save_json(out/'results.json', report); save_json(out/'training_indices.json', dict(fit=ids[:fit_n], scale=ids[fit_n:])); save_json(out/'protocol.json', cfg)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); construct(cfg)
