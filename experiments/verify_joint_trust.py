"""Independent NumPy finite responses, gradients and every trust acceptance."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
from scipy.special import expit, logsumexp

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_downstream_first import convert, jacobian
from verify_levelset_audits import raw_scores, residual


def verify(cfg):
    start = time.monotonic(); out = Path(cfg['output']); r = json.loads((out/'results.json').read_text())
    for name in ['training_point', 'diagnostic', 'predictions']: assert digest(out/(name+'.pt')) == r[name+'_sha256']
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin
    d = convert(torch.load(out/'diagnostic.pt', map_location='cpu')); point = convert(torch.load(out/'training_point.pt', map_location='cpu'))
    saved = convert(torch.load(out/'predictions.pt', map_location='cpu')); hist = json.loads((out/'history.json').read_text())
    p, cache = d['parameters'], d['cache']; q, va = d['output_basis'], d['basis']['A']['vh']
    np.testing.assert_allclose(q.T@q, np.eye(len(q.T)), atol=1e-10); np.testing.assert_allclose(va@va.T, np.eye(len(va)), atol=1e-10)
    bd = torch.load(cfg['boundary_diagnostic'], map_location='cpu'); old = torch.load(cfg['old_calibration'], map_location='cpu')
    lookup = {i: (cc, j) for cc, ids in [(bd['screen'], bd['split']['indices']), (old['scale'], old['indices']['scale'])] for j, i in enumerate(ids)}
    rr = np.stack([lookup[i][0]['r'][lookup[i][1]].numpy() for i in d['indices']]); hh = np.stack([lookup[i][0]['h'][lookup[i][1]].numpy() for i in d['indices']])
    np.testing.assert_allclose(cache['x'], ln(rr, p['gamma2'], p['beta2'], p['eps2']), atol=1e-10, rtol=1e-10); np.testing.assert_array_equal(cache['h'], hh)
    x = cache['x']; pre = x@p['A'].T+p['a']; z = pre*expit(1.702*pre); base = hh-z@p['B'].T
    n, core = len(x), d['core']; sa, sb = d['shifts_a'], d['shifts_b']; nprobe = len(core)*len(sa)
    for val, vv, shift in [(x, va, sa), (z, d['basis']['B']['vh'], sb)]:
        rms = np.sqrt(np.mean((val@vv.T)**2, axis=0)); expected = (cfg['finite_step']*np.array([-1., 1.])[:, None, None]*rms[None, :, None]*vv[None]).reshape(shift.shape)
        np.testing.assert_allclose(shift, expected, atol=1e-10, rtol=1e-10)
    take = np.r_[np.arange(n), np.repeat(core, len(sa)), np.repeat(core, len(sb))]
    xx = x[take].copy(); offset = np.zeros((len(take), len(p['A'])))
    xx[n:n+nprobe] += np.tile(sa, (len(core), 1)); offset[n+nprobe:] = np.tile(sb, (len(core), 1))
    pre0 = xx@p['A'].T+p['a']; t = xx@va.T; zz0 = pre0*expit(1.702*pre0)+offset; bb = base[take]
    na, nb = np.linalg.norm(p['A']), np.linalg.norm(p['B']); zero_c, zero_k = np.zeros_like(point['C']), np.zeros_like(point['K'])
    def evaluate(c, k):
        pre = pre0+na*t@c.T; z = pre*expit(1.702*pre)+offset
        h = bb+z@p['B'].T+nb*(z@k.T)@q.T
        return dict(pre=pre, z=z, h=h, scores=raw_scores(h, p))
    original = evaluate(zero_c, zero_k); teacher = original['scores']; y = teacher.argmax(1); rows = np.arange(len(y)); other = teacher.copy(); other[rows, y] = -np.inf
    gap0 = teacher[rows, y]-other.max(1); classes = teacher.shape[1]; fit = cfg['fit_n']; labels = official_labels(cfg['target'], True)[d['indices']]
    np.testing.assert_allclose(teacher, saved['teacher'], atol=1e-9, rtol=1e-9)
    def latent(c, k, gradient=False):
        h = original['h'][:fit]+nb*(zz0[:fit]@k.T)@q.T; ss = raw_scores(h, p); yy, ri = y[:fit], np.arange(fit)
        other = ss.copy(); other[ri, yy] = -np.inf; comp = other.argmax(1)
        margin = (ss[ri, yy]-ss[ri, comp])/gap0[:fit]; prob = expit(-margin/cfg['sample_temperature'])
        count = np.bincount(yy, minlength=classes); ac = np.array([prob[yy == c0].mean() for c0 in range(classes)])
        exponent = -ac/cfg['class_temperature']; pi = np.exp(exponent-logsumexp(exponent))
        value = -cfg['class_temperature']*(logsumexp(exponent)-math.log(classes))
        objective = value+cfg['distance_weight']*(np.sum(c*c)+np.sum(k*k))
        result = dict(objective=objective, latent_value=value, per_class=ac, scores=ss, class_opponent=pi)
        if gradient:
            gk = np.zeros_like(k)
            weight = -pi[yy]*prob*(1-prob)/(count[yy]*cfg['sample_temperature']*gap0[:fit])
            for lo in range(0, fit, 256):
                hi = min(fit, lo+256); jh, _ = jacobian(h[lo:hi], p); rr0 = np.arange(hi-lo)
                hg = weight[lo:hi, None]*(jh[rr0, yy[lo:hi]]-jh[rr0, comp[lo:hi]])
                gk += nb*(hg@q).T@zz0[lo:hi]
            result['gradient_K'] = gk+2*cfg['distance_weight']*k
        return result
    def jvp(state, k, dc, dk):
        pr = expit(1.702*state['pre']); slope = pr+1.702*state['pre']*pr*(1-pr)
        dz = na*(t@dc.T)*slope
        dh = dz@p['B'].T+nb*(dz@k.T+state['z']@dk.T)@q.T
        hc = state['h']-state['h'].mean(1, keepdims=True); dhc = dh-dh.mean(1, keepdims=True)
        sd = np.sqrt(np.mean(hc*hc, axis=1, keepdims=True)+p['eps'])
        dt = ((dhc-hc*np.mean(hc*dhc, axis=1, keepdims=True)/sd**2)/sd*p['gamma'])@p['projection']
        feat = (hc/sd*p['gamma']+p['beta'])@p['projection']; length = np.linalg.norm(feat, axis=1, keepdims=True); unit = feat/length
        ds = ((dt-unit*np.sum(unit*dt, axis=1, keepdims=True))/length)@p['head'].T
        gaps = teacher[rows, y, None]-teacher; gaps[rows, y] = 1.
        result = (ds[rows, y, None]-ds)/gaps; result[rows, y] = 0.
        return result
    records = []; score_error = jvp_error = gradient_error = 0.; previous_c, previous_k = zero_c, zero_k
    assert len(hist) == len(d['snapshots']) == r['trials']
    for record, snap in zip(hist, d['snapshots']):
        c, k, dc, dk = [snap[key] for key in ['C', 'K', 'dC', 'dK']]
        np.testing.assert_allclose(c, previous_c, atol=1e-12); np.testing.assert_allclose(k, previous_k, atol=1e-12)
        before = evaluate(c, k); after = evaluate(c+dc, k+dk)
        error = float(np.abs(after['scores']-snap['proposed_scores']).max()); score_error = max(score_error, error); assert error < 1e-9
        obj0, obj1 = latent(c, k, True), latent(c+dc, k+dk)
        gk = obj0['gradient_K']; gc = 2*cfg['distance_weight']*c
        ge = float(np.linalg.norm(gk-snap['gradient_K'])/max(np.linalg.norm(gk), 1e-20)); gradient_error = max(gradient_error, ge); assert ge < 1e-7
        np.testing.assert_allclose(gc, snap['gradient_C'], atol=1e-12)
        pred = residual(teacher, before['scores'])+jvp(before, k, dc, dk)
        je = float(np.abs(pred-snap['predicted_residual']).max()); jvp_error = max(jvp_error, je); assert je < 1e-7
        actual = residual(teacher, after['scores']); worst, linear = float(np.abs(actual).max()), float(np.abs(pred).max())
        pg = float(np.sum(gc*dc)+np.sum(gk*dk)); ag = float(obj1['objective']-obj0['objective']); rho = ag/pg if pg > 1e-14 else None
        distance = float(np.hypot(np.linalg.norm(c+dc), np.linalg.norm(k+dk)))
        changed = int((after['scores'][:n].argmax(1) != y[:n]).sum())
        accepted = bool(worst <= .5 and linear <= .5 and distance <= .5 and changed == 0 and rho is not None and rho >= cfg['acceptance_ratio'] and ag > 1e-12)
        assert accepted == record['accepted']
        assert abs(worst-record['proposal']['worst_task_drift']) < 1e-6 and abs(linear-record['predicted_worst_task']) < 1e-6
        assert abs(pg-record['predicted_gain']) < 1e-8 and abs(ag-record['actual_gain']) < 1e-8
        if accepted: previous_c, previous_k = c+dc, k+dk
        records.append(dict(trial=record['trial'], accepted=accepted, finite_worst=worst, linear_worst=linear,
                            agreement_ratio=rho, parameter_distance=distance, gradient_relative_error=ge))
        if len(records) % 3 == 0: print(json.dumps(dict(verified_trials=len(records))), flush=True)
    np.testing.assert_allclose(previous_c, point['C'], atol=1e-12); np.testing.assert_allclose(previous_k, point['K'], atol=1e-12)
    c, k = point['C'], point['K']; final = evaluate(c, k); obj = latent(c, k)
    np.testing.assert_allclose(final['scores'], saved['joint'], atol=1e-9, rtol=1e-9); np.testing.assert_allclose(obj['scores'], saved['B_only_fit'], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(point['release_deltas']['A'], na*c@va, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(point['release_deltas']['B'], nb*q@k, atol=1e-10, rtol=1e-10)
    distance = float(np.hypot(np.linalg.norm(point['release_deltas']['A'])/na, np.linalg.norm(point['release_deltas']['B'])/nb))
    worst = float(np.abs(residual(teacher, final['scores'])).max()); changes = int((final['scores'][:n].argmax(1) != y[:n]).sum())
    by = obj['scores'].argmax(1); bchanged = by != y[:fit]; harmed = 0
    for entry in r['final']['B_only_class_counts']:
        take_class = y[:fit] == entry['teacher_class']; ch = int(bchanged[take_class].sum())
        assert int(take_class.sum()) == entry['n'] and ch == entry['changed']
        harmed += ch/int(take_class.sum()) >= cfg['signal_minimum_class_fraction']
    task_ok = worst <= .5 and changes == 0 and distance <= .5
    signal = harmed >= math.ceil(classes*cfg['signal_class_coverage']) and obj['latent_value'] > r['initial']['latent_value']
    assert task_ok == r['task_calibration_passed'] and bool(task_ok and signal) == r['confirmation_eligible']
    result = dict(status='passed_task_calibration' if r['confirmation_eligible'] else 'passed_closed_protocol_verification',
                  target=cfg['target'], confirmation_eligible=r['confirmation_eligible'], task_calibration_passed=task_ok,
                  natural_n=n, all_task_inputs=len(take), natural_original_correct=int((y[:n] == labels).sum()),
                  natural_joint_correct=int((final['scores'][:n].argmax(1) == labels).sum()), natural_changes=changes,
                  B_only_fit_changed=int(bchanged.sum()), harmed_classes=int(harmed), worst_task_drift=worst, parameter_distance=distance,
                  max_score_error=score_error, max_full_task_JVP_error=jvp_error, max_class_minmax_gradient_relative_error=gradient_error,
                  all_trial_acceptances_verified=True, records=records, point_sha256=r['training_point_sha256'], diagnostic_sha256=r['diagnostic_sha256'],
                  source_sha256=digest(__file__), wall_seconds=time.monotonic()-start, new_model_forwards=0, new_release_weights=0, partner_used=False, test_used=False,
                  scope='Independent NumPy original source inputs, SVD probes, actual matrices, full finite scores and Jacobian-vector responses at EVERY trial, analytic class-adversary gradients, true/linear trust ratios, budgets, class counts and selected point. No source proxy or fusion efficacy inference.')
    save_json(out/'independent_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    torch.set_num_threads(4); verify(json.loads(Path(args.protocol).read_text()))
