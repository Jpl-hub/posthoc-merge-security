"""Independent NumPy reconstruction of downstream-first finite feasibility."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import raw_scores, residual


def convert(value):
    if torch.is_tensor(value): return value.detach().cpu().numpy()
    if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
    if isinstance(value, list): return [convert(v) for v in value]
    return value


def jacobian(h, p):
    c = h-h.mean(1, keepdims=True); sd = np.sqrt(np.mean(c*c, axis=1, keepdims=True)+p['eps'])
    t = (c/sd*p['gamma']+p['beta'])@p['projection']; tn = np.linalg.norm(t, axis=1, keepdims=True)
    f = t/tn; scores = f@p['head'].T
    j = (p['head'][None]-scores[:, :, None]*f[:, None])/tn[:, None]
    j = (j@p['projection'].T)*p['gamma']
    j = (j-j.mean(-1, keepdims=True)-c[:, None]*np.mean(j*c[:, None], axis=-1, keepdims=True)/sd[:, None]**2)/sd[:, None]
    return j, scores


def verify(cfg):
    start = time.monotonic(); root = Path(cfg['output']); report = json.loads((root/'results.json').read_text())
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin
    for name in ['downstream_point', 'point', 'diagnostic', 'predictions']: assert digest(root/(name+'.pt')) == report[name+'_sha256']
    point = convert(torch.load(root/'point.pt', map_location='cpu')); d = convert(torch.load(root/'diagnostic.pt', map_location='cpu'))
    pred = convert(torch.load(root/'predictions.pt', map_location='cpu')); p = d['parameters']; cache = d['cache']
    boundary = torch.load(cfg['boundary_diagnostic'], map_location='cpu'); old = torch.load(cfg['old_calibration'], map_location='cpu')
    lookup = {i: (cc, j) for cc, ids in [(boundary['screen'], boundary['split']['indices']), (old['scale'], old['indices']['scale'])] for j, i in enumerate(ids)}
    rr = np.stack([lookup[i][0]['r'][lookup[i][1]].numpy() for i in d['indices']])
    hh = np.stack([lookup[i][0]['h'][lookup[i][1]].numpy() for i in d['indices']])
    np.testing.assert_allclose(cache['x'], ln(rr, p['gamma2'], p['beta2'], p['eps2']), atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(cache['h'], hh)
    x = cache['x']; u = x@p['A'].T+p['a']; z = u*expit(1.702*u); base = hh-z@p['B'].T
    qa, qb = d['basis']['A']['vh'], d['basis']['B']['vh']; core = d['core']; sa, sb = d['shifts_a'], d['shifts_b']
    for values, basis, shifts in [(x, qa, sa), (z, qb, sb)]:
        rms = np.sqrt(np.mean((values@basis.T)**2, axis=0))
        expected = (cfg['finite_step']*np.array([-1., 1.])[:, None, None]*rms[None, :, None]*basis[None]).reshape(shifts.shape)
        np.testing.assert_allclose(expected, shifts, atol=1e-10, rtol=1e-10)
    head = p['head']-p['head'].mean(0); m = p['gamma'][:, None]*(p['projection']@head.T); m -= m.mean(0)
    b = p['beta']@p['projection']@head.T; q = d['space']['Q']; c, k = point['C'], point['K']
    np.testing.assert_allclose(q.T@q, np.eye(q.shape[1]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(q@(q.T@m), m, atol=1e-10, rtol=1e-10)
    na, nb = np.linalg.norm(p['A']), np.linalg.norm(p['B']); da, db = na*c@qa, nb*q@k
    np.testing.assert_allclose(da, point['delta_A'], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(db, point['delta_B'], atol=1e-10, rtol=1e-10)
    distance = float(np.hypot(np.linalg.norm(da)/na, np.linalg.norm(db)/nb))
    assert abs(distance-report['final']['parameter_distance']) < 1e-12
    ns = cfg['fit_n']; sig = lambda h: np.sqrt(np.mean((h-h.mean(1, keepdims=True))**2, axis=1, keepdims=True)+p['eps'])
    numerator = hh[:ns]@m+sig(hh[:ns])*b; ideal = d['ideal_hidden']
    inverse_error = float(np.max(np.abs(ideal@m+sig(ideal)*b+numerator))); assert inverse_error < 1e-8
    np.testing.assert_allclose((ideal-hh[:ns])@q, d['target_q'], atol=1e-9, rtol=1e-9)
    yfit = raw_scores(hh[:ns], p).argmax(1); count = np.bincount(yfit, minlength=len(head)); weight = 1./count[yfit]/len(count)
    rhs = z[:ns].T@(weight[:, None]*d['target_q']/nb)
    mu = report['downstream_solver']['mu']
    normal = z[:ns].T@(weight[:, None]*(z[:ns]@k.T))+mu*k.T-rhs
    kkt_error = float(np.linalg.norm(normal)/np.linalg.norm(rhs)); assert kkt_error < 1e-8
    assert np.linalg.norm(k) <= .5/np.sqrt(2)*(1+1e-12)
    n = len(x); rank2 = len(sa); take = np.r_[np.arange(n), np.repeat(core, rank2), np.repeat(core, rank2)]
    def context(pos):
        rows = take[pos]; xx = x[rows].copy(); off = np.zeros((len(rows), p['A'].shape[0]))
        wa = (pos >= n) & (pos < n+len(core)*rank2); wb = pos >= n+len(core)*rank2
        xx[wa] += sa[(pos[wa]-n) % rank2]; off[wb] = sb[(pos[wb]-n-len(core)*rank2) % rank2]
        return xx, off, base[rows]
    def run(pos, cc, changed_b):
        xx, off, bb = context(pos); pre = xx@p['A'].T+p['a']+na*(xx@qa.T)@cc.T
        h = bb+(pre*expit(1.702*pre)+off)@changed_b.T
        return h, pre, xx, off
    before, downstream, joint = [], [], []; maximum_score_error = 0.
    for begin in range(0, len(take), cfg['verification_batch_size']):
        pos = np.arange(begin, min(len(take), begin+cfg['verification_batch_size'])); xx, off, bb = context(pos)
        pre = xx@p['A'].T+p['a']; zz = pre*expit(1.702*pre)+off
        s0 = raw_scores(bb+zz@p['B'].T, p); s1 = raw_scores(bb+zz@(p['B']+db).T, p)
        sj = raw_scores(run(pos, c, p['B']+db)[0], p)
        for name, values in [('teacher', s0), ('downstream', s1), ('joint', sj)]:
            maximum_score_error = max(maximum_score_error, float(np.abs(values-pred[name][pos]).max()))
            np.testing.assert_allclose(values, pred[name][pos], atol=1e-9, rtol=1e-9)
        before.append(s0); downstream.append(s1); joint.append(sj)
    before, downstream, joint = map(np.concatenate, [before, downstream, joint])
    worst = float(np.abs(residual(before, joint)).max()); assert abs(worst-report['final']['worst_margin_drift']) < 1e-5
    labels = official_labels(cfg['target'], True)[d['indices']]; source_y = before[:n].argmax(1)
    def count_stats(values, reference):
        y = values[:n].argmax(1); changed = y != source_y
        assert int(changed.sum()) == reference['changed']
        assert int((y == labels).sum()) == reference['transformed_correct']
        assert int((source_y == labels).sum()) == reference['original_correct']
        for entry in reference['per_teacher_class']:
            take_class = source_y == entry['teacher_class']
            assert int(take_class.sum()) == entry['n'] and int(changed[take_class].sum()) == entry['changed']
        return int(changed.sum())
    count_stats(downstream, report['downstream_task_stats']); changes = count_stats(joint, report['joint_task_stats'])
    bclasses = sum(v['changed_fraction'] >= cfg['minimum_class_flip_fraction'] for v in report['downstream_task_stats']['per_teacher_class'])
    bpass = bclasses >= cfg['minimum_harmed_classes']
    passed = bool(bpass and worst <= .5 and distance <= .5 and changes == 0)
    assert passed == report['source_task_passed']
    derivative_errors, restoration_kkt = [], []
    for step in d['steps']:
        pos, cc = step['selected'], step['C']; h1, pre, xx, off = run(pos, cc, p['B']+db)
        jh, now = jacobian(h1, p); teacher = before[pos]; yy = teacher.argmax(1); rows = np.arange(len(yy))
        keep = np.ones(teacher.shape, bool); keep[rows, yy] = False
        gaps = teacher[rows, yy, None]-teacher; gaps[~keep] = 1.
        pr = expit(1.702*pre); slope = pr+1.702*pre*pr*(1-pr)
        jj = na*(jh@(p['B']+db))*slope[:, None]
        aa = ((jj[rows, yy, None]-jj)/gaps[:, :, None])[keep]
        ss = np.broadcast_to((xx@qa.T)[:, None], (len(xx), teacher.shape[1], len(qa)))[keep]
        np.testing.assert_allclose(aa, step['factor_a'], atol=1e-7, rtol=1e-9)
        np.testing.assert_allclose(ss, step['factor_s'], atol=1e-10, rtol=1e-10)
        rr = residual(teacher, now)[keep]
        np.testing.assert_allclose(rr, step['residual'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(step['linear_rhs'], rr-np.sum(aa*(ss@cc.T), axis=1), atol=1e-8, rtol=1e-8)
        proposal = step['proposal']; linear = np.sum(aa*(ss@proposal.T), axis=1)+step['linear_rhs']
        gradient = aa.T@(linear[:, None]*ss)+step['mu']*proposal
        scale = np.linalg.norm(aa.T@(step['linear_rhs'][:, None]*ss))
        error = float(np.linalg.norm(gradient)/max(scale, 1e-20)); restoration_kkt.append(error); assert error < 1e-6
        np.testing.assert_allclose(proposal, -aa.T@(step['dual'][:, None]*ss), atol=1e-8, rtol=1e-8)
        direction = proposal-cc; length = np.linalg.norm(direction)
        if length > 0:
            direction /= length; eps = cfg['derivative_step']
            plus = raw_scores(run(pos, cc+eps*direction, p['B']+db)[0], p)
            minus = raw_scores(run(pos, cc-eps*direction, p['B']+db)[0], p)
            measured = (residual(teacher, plus)-residual(teacher, minus))/(2*eps)
            estimated = np.sum(aa*(ss@direction.T), axis=1)
            error = float(np.max(np.abs(measured[keep]-estimated))); derivative_errors.append(error); assert error < cfg['derivative_tolerance']
    result = dict(status='passed_source_feasibility_verification' if passed else 'passed_failure_verification', source_task_passed=passed,
                  natural_n=n, all_task_inputs=len(take), parameter_distance=distance, worst_margin_drift=worst, natural_prediction_changes=changes,
                  downstream_changed=report['downstream_task_stats']['changed'], downstream_harmed_classes=bclasses,
                  inverse_numerator_maximum_error=inverse_error, downstream_KKT_relative_error=kkt_error,
                  maximum_score_reconstruction_error=maximum_score_error, restoration_KKT_relative_error_maximum=max(restoration_kkt, default=0.),
                  finite_derivative_error_maximum=max(derivative_errors, default=0.), checked_correction_steps=len(d['steps']),
                  point_sha256=report['point_sha256'], diagnostic_sha256=report['diagnostic_sha256'], source_sha256=digest(__file__),
                  wall_seconds=time.monotonic()-start, new_full_model_forwards=0, new_release_weights=0, partner_used=False, test_used=False,
                  scope='Independent NumPy official source inputs, A/B task probes, exact matrices, all finite CLIP scores and class counts, spectral KKT, every restoration Jacobian and finite derivative. Failed fixed-protocol correction is not a nonexistence theorem.')
    save_json(root/'independent_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    torch.set_num_threads(4); verify(json.loads(Path(args.protocol).read_text()))
