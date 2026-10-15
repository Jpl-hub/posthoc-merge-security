"""Independent NumPy finite reconstruction of the sole scaling point."""

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
from verify_levelset_audits import convert, raw_scores, residual


def verify(cfg):
    start = time.monotonic(); root = Path(cfg['output']); report = json.loads((root/'results.json').read_text())
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin
    for name, key in [('point.pt', 'point_sha256'), ('diagnostic.pt', 'diagnostic_sha256'), ('predictions.pt', 'predictions_sha256')]:
        assert digest(root/name) == report[key]
    point = torch.load(root/'point.pt', map_location='cpu'); data = torch.load(root/'diagnostic.pt', map_location='cpu')
    original = torch.load(root/'predictions.pt', map_location='cpu'); p = convert(data['parameters']); cache = convert(data['cache'])
    gain = point['gain'].numpy(); mask = point['mask'].numpy(); core = data['core'].numpy(); n = len(cache['x'])
    assert np.all((mask == 0) | (mask == 1)) and np.all(gain >= 1) and mask.sum() <= len(p['head'])
    assert np.array_equal(gain > 1, mask.astype(bool))
    # Rebuild every cached source input from pinned original target caches.
    boundary = torch.load(cfg['boundary_diagnostic'], map_location='cpu'); old = torch.load(cfg['old_calibration'], map_location='cpu')
    lookup = {i: (cache0, j) for cache0, ids in [(boundary['screen'], boundary['split']['indices']), (old['scale'], old['indices']['scale'])] for j, i in enumerate(ids)}
    raw_r = np.stack([lookup[i][0]['r'][lookup[i][1]].numpy() for i in data['indices']])
    raw_h = np.stack([lookup[i][0]['h'][lookup[i][1]].numpy() for i in data['indices']])
    np.testing.assert_allclose(cache['x'], ln(raw_r, p['gamma2'], p['beta2'], p['eps2']), atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(cache['h'], raw_h)
    A = p['A']*gain[:, None]; a = p['a']*gain; B = p['B']/gain[None]
    distance = float(np.sqrt(sum(np.sum((changed-p[key])**2)/np.sum(p[key]**2) for key, changed in [('A', A), ('a', a), ('B', B)] if np.linalg.norm(p[key]) > 0)))
    assert abs(distance-report['parameter_distance']) < 1e-12
    sa, sb = data['shifts_a'].numpy(), data['shifts_b'].numpy(); rank2 = len(sa)
    pre = cache['x']@p['A'].T+p['a']; z = pre*expit(1.702*pre); base = cache['h']-z@p['B'].T
    for site, values, shifts in [('A', cache['x'], sa), ('B', z, sb)]:
        v = data['basis'][site]['vh'].numpy(); rms = np.sqrt(np.mean((values@v.T)**2, axis=0))
        expected = (cfg['finite_step']*np.array([-1., 1.])[:, None, None]*rms[None, :, None]*v[None]).reshape(shifts.shape)
        np.testing.assert_allclose(shifts, expected, atol=1e-10, rtol=1e-10)
    take = np.r_[np.arange(n), np.repeat(core, rank2), np.repeat(core, rank2)]
    before, after, middle = [], [], []; finite_error = 0.
    for start0 in range(0, len(take), cfg['verification_batch_size']):
        positions = np.arange(start0, min(len(take), start0+cfg['verification_batch_size'])); rows = take[positions]
        x = cache['x'][rows].copy(); off = np.zeros((len(rows), len(gain)))
        wa = (positions >= n) & (positions < n+len(core)*rank2); wb = positions >= n+len(core)*rank2
        x[wa] += sa[(positions[wa]-n) % rank2]; off[wb] = sb[(positions[wb]-n-len(core)*rank2) % rank2]
        u = x@p['A'].T+p['a']; h0 = base[rows]+(u*expit(1.702*u)+off)@p['B'].T
        u1 = x@A.T+a; h1 = base[rows]+(u1*expit(1.702*u1)+off)@B.T
        b0, b1 = raw_scores(h0, p), raw_scores(h1, p); before.append(b0); after.append(b1)
        np.testing.assert_allclose(b0, original['source_scores'][positions].numpy(), atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(b1, original['released_scores'][positions].numpy(), atol=1e-9, rtol=1e-9)
        if start0 < n:
            count = min(n-start0, len(rows)); am = .5*(A+p['A']); bm = .5*(B+p['B']); aa = .5*(a+p['a'])
            um = x[:count]@am.T+aa; hm = base[rows[:count]]+(um*expit(1.702*um))@bm.T
            middle.append(raw_scores(hm, p))
    before, after, middle = np.concatenate(before), np.concatenate(after), np.concatenate(middle)
    np.testing.assert_allclose(middle, original['midpoint_scores'].numpy(), atol=1e-9, rtol=1e-9)
    # Independently evaluate the certified FINITE ranking lower bound.
    head = p['head']-p['head'].mean(0); m = p['gamma'][:, None]*(p['projection']@head.T); m -= m.mean(0)
    b = p['beta']@p['projection']@head.T; h = cache['h'][:cfg['objective_n']]; sigma = lambda v: np.sqrt(np.mean((v-v.mean(1, keepdims=True))**2, axis=1, keepdims=True)+p['eps'])
    numer = h@m+sigma(h)*b; y = numer.argmax(1); other = numer.copy(); other[np.arange(len(y)), y] = -np.inf; k = other.argmax(1)
    gap = numer[np.arange(len(y)), y]-numer[np.arange(len(y)), k]; u = pre[:len(y)]
    d = .5*(1+1/gain)*(.5*(1+gain)*u*expit(1.702*.5*(1+gain)*u))-u*expit(1.702*u)
    sd_bound = np.linalg.norm(p['B']-p['B'].mean(0), axis=0)/np.sqrt(p['B'].shape[0])
    lower = ((-d*((m[:, y]-m[:, k]).T@p['B'])-np.abs(d)*sd_bound*np.abs(b[y]-b[k])[:, None])/gap[:, None]).sum(1)
    hm = h+d@p['B'].T; nm = hm@m+sigma(hm)*b
    actual = 1-(nm[np.arange(len(y)), y]-nm[np.arange(len(y)), k])/gap
    finite_error = float(np.max(lower-actual)); assert finite_error < 1e-9
    np.testing.assert_allclose(lower, original['midpoint_lower'].numpy(), atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(actual, original['midpoint_actual_harm'].numpy(), atol=1e-9, rtol=1e-9)
    drift = float(np.abs(residual(before, after)).max()); changed = after[:n].argmax(1) != before[:n].argmax(1)
    assert abs(drift-report['worst_margin_drift']) < 1e-6 and int(changed.sum()) == report['natural_changes']
    labels = official_labels(cfg['target'], True)[data['indices']]
    old_correct = int((before[:n].argmax(1) == labels).sum()); new_correct = int((after[:n].argmax(1) == labels).sum())
    assert old_correct == report['original_correct'] and new_correct == report['released_correct']
    passed = bool(mask.sum() > 0 and distance <= cfg['parameter_budget'] and drift <= cfg['task_budget'] and not changed.any())
    assert passed == report['task_gate_passed']
    result = dict(status='passed_task_calibration' if passed else 'passed_failure_verification', target=cfg['target'], all_task_inputs=len(before), natural_n=n,
                  parameter_distance=distance, worst_margin_drift=drift, natural_prediction_changes=int(changed.sum()), original_correct=old_correct, released_correct=new_correct,
                  finite_lower_bound_violation=finite_error, midpoint_changed=int((middle.argmax(1) != before[:n].argmax(1)).sum()),
                  point_sha256=report['point_sha256'], source_sha256=digest(__file__), wall_seconds=time.monotonic()-start, new_model_forwards=0,
                  new_release_weights=0, partner_used=False, test_used=False, confirmation_used=False,
                  scope='Independent NumPy actual A/a/B matrices, QuickGELU, every natural/SVD input, exact CLIP normalization, finite bound and official-label recount. No proxy/midpoint prediction gate.')
    save_json(root/'independent_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    torch.set_num_threads(4); verify(json.loads(Path(args.protocol).read_text()))
