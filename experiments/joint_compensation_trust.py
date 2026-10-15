"""Joint source-only finite compensation with nonlinear trust-region acceptance."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from classification_completion import readout
from clip_benchmark import digest
from downstream_first_compensation import score_jacobian, stats
from finite_gelu_scaling import cpu, inputs
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import checked_models, gelu, initialize, parameters
from verify_covariance_noise import official_labels


def pair_norm(a, b): return (a.square().sum()+b.square().sum()).sqrt()
def pair_dot(a, b): return sum((x*y).sum() for x, y in zip(a, b))


def score_jvp(h, dh, p):
    c = h-h.mean(1, keepdim=True); dc = dh-dh.mean(1, keepdim=True)
    sd = (c.square().mean(1, keepdim=True)+p['eps']).sqrt()
    dt = ((dc-c*(c*dc).mean(1, keepdim=True)/sd.square())/sd*p['gamma'])@p['projection']
    t = (c/sd*p['gamma']+p['beta'])@p['projection']; tn = t.norm(dim=1, keepdim=True); f = t/tn
    return ((dt-f*(f*dt).sum(1, keepdim=True))/tn)@p['head'].T


class JointProblem:
    def __init__(self, cfg):
        self.cfg = cfg
        for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
        source = torch.load(cfg['mixed_diagnostic'], map_location='cpu')['parameters']; self.p = on_device(source)
        anchor, expert, head, self.pins = checked_models(cfg['target'])
        exact, base = parameters(expert, head), parameters(anchor, head)
        for key, value in exact.items():
            if torch.is_tensor(value): assert torch.equal(value, source[key]), key
        self.basis = {k: on_device(v) for k, v in torch.load(cfg['basis_diagnostic'], map_location='cpu')['basis'].items()}
        self.provenance = {}
        for site, b in self.basis.items():
            task = exact[site]-base[site]; u, v = b['u'].cpu(), b['vh'].cpu(); sv = torch.diag(u.T@task@v.T)
            error = float((task@v.T-u*sv).norm()/(task@v.T).norm()); assert error < 1e-8
            self.provenance[site] = dict(rank=len(v), task_SVD_equation_error=error)
        del anchor, expert, head
        self.packed, self.cache, self.ids = inputs(cfg, self.p, self.basis); self.n = self.packed['natural_n']
        self.labels = torch.tensor(official_labels(cfg['target'], True)[self.ids], device='cuda')
        self.space, self.geometry = readout(self.p); self.q = self.space['Q']; self.va = self.basis['A']['vh']
        self.na, self.nb = self.p['A'].norm(), self.p['B'].norm()
        self.t = self.packed['x']@self.va.T; self.pre0 = self.packed['x']@self.p['A'].T+self.p['a']
        self.z0 = gelu(self.pre0)+self.packed['offset']
        self.c0 = torch.zeros(self.p['A'].shape[0], len(self.va), device='cuda', dtype=torch.float64)
        self.k0 = torch.zeros(self.q.shape[1], self.p['B'].shape[1], device='cuda', dtype=torch.float64)
        self.teacher = self.evaluate(self.c0, self.k0)['scores'].detach()
        self.y = self.teacher.argmax(1); self.rows = torch.arange(len(self.y), device='cuda')
        others = self.teacher.clone(); others[self.rows, self.y] = -torch.inf
        self.top_gap = self.teacher[self.rows, self.y]-others.max(1).values; assert self.top_gap.min() > 1e-8
        self.classes = self.teacher.shape[1]; self.active = set(self.packed['core'].cpu().tolist())

    def evaluate(self, c, k):
        pre = self.pre0+self.na*self.t@c.T
        z = gelu(pre)+self.packed['offset']; b = self.p['B']+self.nb*self.q@k
        h = self.packed['base']+z@b.T
        return dict(pre=pre, z=z, B=b, h=h, scores=scores(h, self.p))

    def latent(self, k):
        count = self.cfg['fit_n']; b = self.p['B']+self.nb*self.q@k
        s = scores(self.packed['base'][:count]+self.z0[:count]@b.T, self.p)
        y = self.y[:count]; row = self.rows[:count]; other = s.clone(); other[row, y] = -torch.inf
        gap = (s[row, y]-other.max(1).values)/self.top_gap[:count]
        smooth = torch.sigmoid(-gap/self.cfg['sample_temperature'])
        per_class = torch.stack([smooth[y == cls].mean() for cls in range(self.classes)])
        tau = self.cfg['class_temperature']
        value = -tau*(torch.logsumexp(-per_class/tau, 0)-math.log(self.classes))
        return value, per_class, s

    def objective(self, c, k):
        return self.latent(k)[0]+self.cfg['distance_weight']*pair_norm(c, k).square()

    def metrics(self, state, c, k):
        err = signed_residual(self.teacher, state['scores'])
        latent, per_class, b_scores = self.latent(k)
        b = stats(self.teacher[:self.cfg['fit_n']], b_scores, self.labels[:self.cfg['fit_n']])
        return dict(parameter_distance=float(pair_norm(c, k)), worst_task_drift=float(err.abs().max()),
                    natural_changes=int((state['scores'][:self.n].argmax(1) != self.y[:self.n]).sum()),
                    latent_value=float(latent), per_class_smooth_damage=per_class.cpu().tolist(),
                    objective=float(latent+self.cfg['distance_weight']*pair_norm(c, k).square()),
                    B_only_changed=b['changed'], B_only_classes_changed=sum(r['changed'] > 0 for r in b['per_teacher_class']),
                    B_only_class_counts=b['per_teacher_class'])

    def factors(self, state):
        risk = signed_residual(self.teacher, state['scores']).abs().max(1).values
        mandatory = set(self.packed['core'].cpu().tolist())
        for cls in range(self.classes):
            members = (self.y == cls).nonzero().flatten()
            witness = int(members[risk[members].argmax()]); self.active.add(witness); mandatory.add(witness)
        if self.cfg.get('working_set_exchange', False):
            mandatory.update(getattr(self, 'rejected_witnesses', set()))
            pool = self.active | mandatory; cap = self.cfg['maximum_active_inputs']
            assert len(mandatory) <= cap
            other = sorted(pool-mandatory, key=lambda i: (-float(risk[i]), i))
            self.active = mandatory | set(other[:cap-len(mandatory)])
        selected = torch.tensor(sorted(self.active), device='cuda'); jh, now = score_jacobian(state['h'][selected], self.p)
        teacher = self.teacher[selected]; yy = self.y[selected]; row = torch.arange(len(selected), device='cuda')
        keep = torch.ones_like(teacher, dtype=torch.bool); keep[row, yy] = False
        gap = (teacher[row, yy, None]-teacher).masked_fill(~keep, 1.)
        pre = state['pre'][selected]; prob = torch.sigmoid(1.702*pre); slope = prob+1.702*pre*prob*(1-prob)
        ja = self.na*(jh@state['B'])*slope[:, None]; jb = self.nb*(jh@self.q)
        aa = ((ja[row, yy, None]-ja)/gap[:, :, None])[keep]
        bb = ((jb[row, yy, None]-jb)/gap[:, :, None])[keep]
        ss = self.t[selected, None].expand(-1, self.classes, -1)[keep]
        zz = state['z'][selected, None].expand(-1, self.classes, -1)[keep]
        residual = signed_residual(teacher, now)[keep]
        gram_a = (aa@aa.T)*(ss@ss.T); gram = gram_a+(bb@bb.T)*(zz@zz.T)
        d, v = torch.linalg.eigh(gram); tol = torch.finfo(d.dtype).eps*max(len(gram), self.c0.numel()+self.k0.numel())*d.abs().max()
        assert float(d.min()) > -100*tol
        good = d > tol; inv = torch.where(good, 1/d.clamp_min(tol), 0.)
        def solve(rhs): return v@((v.T@rhs)*inv)
        def jv(c, k): return (aa*(ss@c.T)).sum(1)+(bb*(zz@k.T)).sum(1)
        def jt(dual): return aa.T@(dual[:, None]*ss), bb.T@(dual[:, None]*zz)
        return dict(selected=selected, aa=aa, bb=bb, ss=ss, zz=zz, residual=residual, gram=gram, gram_a=gram_a,
                    rank=int(good.sum()), solve=solve, jv=jv, jt=jt)

    def jvp(self, state, dc, dk):
        prob = torch.sigmoid(1.702*state['pre']); slope = prob+1.702*state['pre']*prob*(1-prob)
        dz = self.na*(self.t@dc.T)*slope
        dh = dz@state['B'].T+state['z']@(self.nb*self.q@dk).T
        ds = score_jvp(state['h'], dh, self.p)
        gap = self.teacher[self.rows, self.y, None]-self.teacher; gap[self.rows, self.y] = 1.
        response = (ds[self.rows, self.y, None]-ds)/gap
        response[self.rows, self.y] = 0.
        return response


def positive_root(a, b, c):
    """Largest nonnegative root of a*t^2+2*b*t+c <= 0, c <= 0."""
    if a <= 1e-30: return 0.
    return max(0., (-b+math.sqrt(max(0., b*b-a*c)))/a)


@torch.no_grad()
def optimize(cfg):
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False); start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    p = JointProblem(cfg); c, k = p.c0.clone(), p.k0.clone(); state = p.evaluate(c, k)
    radius = cfg['trust_initial']; history = []; snapshots = []; accepted = 0; reason = 'fixed_trial_cap'; calls = 0; reachability = None
    initial = p.metrics(state, c, k)
    for trial in range(cfg['maximum_trials']):
        if not cfg.get('working_set_exchange', False) and len(p.active)+p.classes > cfg['maximum_active_inputs']:
            reason = 'fixed_task_witness_cap'; break
        f = p.factors(state); calls += 1
        with torch.enable_grad():
            leaf = k.detach().clone().requires_grad_(True)
            value = p.latent(leaf)[0]+cfg['distance_weight']*leaf.square().sum()
            gk = torch.autograd.grad(value, leaf)[0]
        gc = 2*cfg['distance_weight']*c
        dual_g = f['solve'](f['jv'](gc, gk)); projected = f['jt'](dual_g)
        tangent = (gc-projected[0], gk-projected[1])
        normal_dual = f['solve'](-f['residual']); normal = f['jt'](normal_dual)
        if reachability is None:
            response_b = f['jv'](torch.zeros_like(c), gk); da, va = torch.linalg.eigh(f['gram_a'])
            threshold = torch.finfo(da.dtype).eps*max(len(da), c.numel())*da.abs().max(); good = da > threshold
            projection = va[:, good]@(va[:, good].T@response_b)
            reachability = dict(active_inputs=len(f['selected']), task_rows=len(response_b), upstream_response_rank=int(good.sum()),
                                B_objective_response_norm=float(response_b.norm()),
                                A_unreachable_B_response_fraction=float((response_b-projection).norm()/response_b.norm().clamp_min(1e-30)),
                                joint_gradient_retained_fraction=float(pair_norm(*tangent)/pair_norm(gc, gk)))
        # Keep a source-normal correction and a joint tangent direction inside
        # ONE trust region. Neither branch is fixed before task feasibility.
        nn = float(pair_norm(*normal)); scale = min(1., cfg['normal_radius_fraction']*radius/max(nn, 1e-30))
        normal = tuple(scale*v for v in normal)
        if float(pair_norm(c, k)) >= cfg['parameter_budget']*.98:
            dual_r = f['solve'](f['jv'](c, k)); projected_r = f['jt'](dual_r)
            radial = (c-projected_r[0], k-projected_r[1]); rr = float(pair_dot(radial, radial))
            outward = float(pair_dot(tangent, radial))
            if rr > 1e-25 and outward > 0: tangent = tuple(v-r*outward/rr for v, r in zip(tangent, radial))
        tn = float(pair_norm(*tangent))
        if tn < 1e-16: reason = 'no_nonzero_joint_tangent'; break
        tangent = tuple(v/tn for v in tangent)
        step = positive_root(1., float(pair_dot(normal, tangent)), float(pair_dot(normal, normal))-radius**2)
        dc, dk = tuple(n+step*t for n, t in zip(normal, tangent))
        # Exact quadratic clipping against the original global distance ball.
        safe_radius = cfg['parameter_budget']*(1-1e-12)
        scale = min(1., positive_root(float(pair_norm(dc, dk)**2), float(pair_dot((c, k), (dc, dk))), float(pair_norm(c, k)**2)-safe_radius**2))
        dc, dk = scale*dc, scale*dk
        predicted_gain = float(pair_dot((gc, gk), (dc, dk)))
        current_metrics = p.metrics(state, c, k); old_residual = signed_residual(p.teacher, state['scores'])
        linear_delta = p.jvp(state, dc, dk); predicted_residual = old_residual+linear_delta
        proposal_c, proposal_k = c+dc, k+dk
        proposed = p.evaluate(proposal_c, proposal_k); new_metrics = p.metrics(proposed, proposal_c, proposal_k)
        actual_gain = new_metrics['objective']-current_metrics['objective']
        ratio = actual_gain/predicted_gain if predicted_gain > 1e-14 else None
        actual_residual = signed_residual(p.teacher, proposed['scores'])
        error = float((actual_residual-predicted_residual).abs().max())
        linear_worst = float(predicted_residual.abs().max())
        task_ok = new_metrics['worst_task_drift'] <= cfg['task_budget'] and new_metrics['natural_changes'] == 0
        accept = bool(task_ok and linear_worst <= cfg['task_budget'] and new_metrics['parameter_distance'] <= cfg['parameter_budget']
                      and ratio is not None and ratio >= cfg['acceptance_ratio'] and actual_gain > 1e-12)
        # Check the actual derivative once at every model, before accepting any
        # finite step. The finite trust comparison above remains the real gate.
        length = float(pair_norm(dc, dk)); derivative_error = objective_error = 0.
        if length > 1e-16:
            uc, uk = dc/length, dk/length; eps = cfg['derivative_step']
            plus = p.evaluate(c+eps*uc, k+eps*uk); minus = p.evaluate(c-eps*uc, k-eps*uk)
            measured = (signed_residual(p.teacher, plus['scores'])-signed_residual(p.teacher, minus['scores']))/(2*eps)
            derivative_error = float((measured-linear_delta/length).abs().max()); assert derivative_error < cfg['derivative_tolerance']
            measured_objective = float((p.objective(c+eps*uc, k+eps*uk)-p.objective(c-eps*uc, k-eps*uk))/(2*eps))
            objective_error = abs(measured_objective-float(pair_dot((gc, gk), (uc, uk)))); assert objective_error < cfg['objective_derivative_tolerance']
        snapshots.append(dict(C=c.cpu(), K=k.cpu(), dC=dc.cpu(), dK=dk.cpu(), selected=f['selected'].cpu(),
                              gradient_C=gc.cpu(), gradient_K=gk.cpu(), normal_dual=normal_dual.cpu(), tangent_dual=dual_g.cpu(),
                              proposed_scores=proposed['scores'].cpu(), predicted_residual=predicted_residual.cpu()))
        record = dict(trial=trial+1, trust_radius=radius, active_inputs=len(f['selected']), task_response_rank=f['rank'],
                      normal_norm_before_clipping=nn, step_norm=length, predicted_gain=predicted_gain, actual_gain=actual_gain,
                      agreement_ratio=ratio, predicted_worst_task=linear_worst, finite_task_model_error=error,
                      derivative_error=derivative_error, objective_derivative_error=objective_error,
                      accepted=accept, before=current_metrics, proposal=new_metrics)
        history.append(record); print(json.dumps(record), flush=True)
        if accept:
            c, k, state = proposal_c, proposal_k, proposed; accepted += 1
            if ratio >= cfg['expansion_ratio'] and error <= cfg['task_model_expansion_tolerance']:
                radius = min(cfg['trust_maximum'], cfg['trust_expansion']*radius)
            elif ratio < cfg['shrink_accepted_ratio']: radius *= cfg['trust_contraction']
        else:
            radius *= cfg['trust_contraction']
            # Add source-only witnesses of a rejected constraint/model error.
            risk = torch.maximum(actual_residual.abs().max(1).values, (actual_residual-predicted_residual).abs().max(1).values)
            rejected = set()
            for cls in range(p.classes):
                members = (p.y == cls).nonzero().flatten(); rejected.add(int(members[risk[members].argmax()]))
            p.active.update(rejected); p.rejected_witnesses = rejected
        if radius < cfg['trust_minimum']: reason = 'trust_radius_below_predeclared_floor'; break
        if accepted >= cfg['maximum_accepted_steps']: reason = 'fixed_accepted_step_cap'; break
    final = p.metrics(state, c, k); latent, _, b_scores = p.latent(k)
    natural_pass = final['worst_task_drift'] <= cfg['task_budget'] and final['natural_changes'] == 0 and final['parameter_distance'] <= cfg['parameter_budget']
    harmed = sum(r['changed_fraction'] >= cfg['signal_minimum_class_fraction'] for r in final['B_only_class_counts'])
    signal = harmed >= math.ceil(p.classes*cfg['signal_class_coverage']) and final['latent_value'] > initial['latent_value']
    eligible = bool(natural_pass and signal)
    torch.save(dict(C=c.cpu(), K=k.cpu(), release_deltas=dict(A=(p.na*c@p.va).cpu(), B=(p.nb*p.q@k).cpu()), cfg=cfg), out/'training_point.pt')
    torch.save(dict(parameters=cpu(p.p), basis={k: cpu(v) for k, v in p.basis.items()}, output_basis=p.q.cpu(), cache=cpu(p.cache),
                    indices=p.ids, core=p.packed['core'].cpu(), shifts_a=p.packed['shifts_a'].cpu(), shifts_b=p.packed['shifts_b'].cpu(),
                    snapshots=snapshots, cfg=cfg), out/'diagnostic.pt')
    torch.save(dict(teacher=p.teacher.cpu(), joint=state['scores'].cpu(), B_only_fit=b_scores.cpu(), indices=p.ids), out/'predictions.pt')
    result = dict(status='eligible_for_independent_confirmation' if eligible else 'closed_no_joint_finite_signal', target=cfg['target'],
                  method='joint_task_SVD_compensation_with_source_class_adversary_and_nonlinear_trust_acceptance',
                  initial=initial, final=final, task_calibration_passed=natural_pass, latent_multiclass_signal_passed=signal,
                  harmed_classes=harmed, confirmation_eligible=eligible, stop_reason=reason, trials=len(history), accepted_updates=accepted,
                  explicit_reverse_gradient_calls=calls, analytic_task_jacobian_calls=calls, task_JVP_calls=len(history),
                  independent_confirmation_run=False, real_TA_run=False, real_AGA_run=False, expert_parameter_updates=0,
                  new_model_forwards=0, new_release_weights=0, source_midpoint_or_proxy_used=False, partner_used=False, test_used=False,
                  reachability_at_origin=reachability, geometry=p.geometry, task_SVD_provenance=p.provenance,
                  training_point_sha256=digest(out/'training_point.pt'), diagnostic_sha256=digest(out/'diagnostic.pt'), predictions_sha256=digest(out/'predictions.pt'),
                  checkpoint_sha256=p.pins, source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']),
                  natural_n=p.n, all_task_inputs=len(p.teacher), selected_step=accepted,
                  wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  limitation='One frozen source-only joint constrained path. Task witnesses approximate local geometry; every accepted step passes the FULL finite task gate. Class signal is specific to this construction, not a universal defense gate or an unknown-peer coverage guarantee. No claim of a global min-max solution.')
    save_json(out/'results.json', result); save_json(out/'history.json', history); save_json(out/'protocol.json', cfg)
    save_json(out/'training_indices.json', dict(fit=p.ids[:cfg['fit_n']], scale=p.ids[cfg['fit_n']:]))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); optimize(cfg)
