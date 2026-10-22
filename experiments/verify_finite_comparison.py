"""Independent NumPy reconstruction of frozen finite response comparisons."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
from scipy.special import expit, softmax

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores


def suffix(H, p):
    n, length, width = H.shape; heads = p['heads']; d = width//heads
    x = ln(H, p['gamma1'], p['beta1'], p['eps1'])
    wq, wk, wv = np.split(p['W'], 3); bq, bk, bv = np.split(p['w'], 3)
    q = (x[:, 0]@wq.T+bq).reshape(n, heads, d)
    k = (x@wk.T+bk).reshape(n, length, heads, d).transpose(0, 2, 1, 3)
    v = (x@wv.T+bv).reshape(n, length, heads, d).transpose(0, 2, 1, 3)
    prob = softmax(np.einsum('nhd,nhtd->nht', q, k)/np.sqrt(d), axis=-1)
    avg = np.einsum('nht,nhtd->nhd', prob, v).reshape(n, width)
    r = H[:, 0]+avg@p['O'].T+p['ob']; x2 = ln(r, p['gamma2'], p['beta2'], p['eps2'])
    pre = x2@p['A'].T+p['a']; z = pre*expit(1.702*pre)
    return r+z@p['B'].T+p['b'], dict(z=z, r=r, x=x2)


def geometry(p):
    head = p['head']-p['head'].mean(0); m = p['gamma'][:, None]*(p['projection']@head.T)
    return m-m.mean(0), p['beta']@p['projection']@head.T


def sd(h, p): return np.sqrt(np.mean((h-h.mean(1, keepdims=True))**2, axis=1, keepdims=True)+p['eps'])


def N(h, p):
    m, b = geometry(p); return h@m+sd(h, p)*b


def states(H, h, p, q):
    old, c0 = suffix(H, p); up = dict(q); up['B'] = p['B']; hu, c1 = suffix(H, up); hj, _ = suffix(H, q)
    db = q['B']-p['B']; u = hu-old; v = c0['z']@db.T; it = (c1['z']-c0['z'])@db.T
    assert np.max(np.abs(hj-old-u-v-it)) < 1e-9
    return dict(h0=h, upper=h+u, B_only=h+v, joint=h+hj-old, U=u, V=v, I=it,
                z=c1['z'], x0=c0['x'], x1=c1['x'])


def lower_median(v): return float(np.sort(v.reshape(-1))[(v.size-1)//2])


def harm(n0, n1):
    y = n0.argmax(1); rows = np.arange(len(y)); other = n1.copy(); other[rows, y] = -np.inf; k = other.argmax(1)
    gap = n0[rows, y]-n0[rows, k]; assert np.all(gap > 0)
    values = 1-(n1[rows, y]-n1[rows, k])/gap
    np.testing.assert_array_equal(values > 1, n1.argmax(1) != y)
    return y, k, gap, values


def check_record(st, p, q, labels, expected, basis):
    n0 = N(st['h0'], p); y = n0.argmax(1); center0 = n0-n0.mean(1, keepdims=True); norm0 = np.linalg.norm(center0, axis=1, keepdims=True)
    for name in ['upper', 'B_only', 'joint']:
        n1 = N(st[name], p); pred = n1.argmax(1); changed = pred != y; r = expected['components'][name]
        assert r['changed'] == int(changed.sum()) and r['released_correct'] == int((pred == labels).sum())
        assert r['changed_by_class'] == np.bincount(y[changed], minlength=n0.shape[1]).tolist()
        yy, kk, gap, values = harm(n0, n1)
        assert abs(float(values.mean())-r['mean_margin_harm']) < 1e-7
        c1 = n1-n1.mean(1, keepdims=True); alpha = np.sum(c1*center0, axis=1)/np.sum(center0**2, axis=1)
        angle = np.linalg.norm(c1/np.linalg.norm(c1, axis=1, keepdims=True)-center0/norm0, axis=1)
        transverse = np.linalg.norm(c1-alpha[:, None]*center0, axis=1)/norm0[:, 0]
        for field, value in [('radial_scale_median', alpha), ('angular_change_median', angle), ('transverse_relative_median', transverse)]:
            assert abs(lower_median(value)-r[field]) < 1e-8, field
        np.testing.assert_array_equal(pred, raw_scores(st[name], p).argmax(1))
    db = q['B']-p['B']; projected = db@basis.T@basis; m, b = geometry(p)
    ratio = np.sum(projected**2)/np.sum(db**2)
    assert abs(ratio-expected['B_task_input_parameter_energy_ratio']) < 1e-8
    terms = dict(upstream=st['U']@m, original_gate_B=st['V']@m, finite_gate_B_cross=st['I']@m,
                 normalization=(sd(st['joint'], p)-sd(st['h0'], p))*b)
    assert np.max(np.abs(sum(terms.values())-(N(st['joint'], p)-n0))) < 1e-8
    for name, value in terms.items():
        a = -(value[np.arange(len(y)), yy]-value[np.arange(len(y)), kk])/gap
        assert abs(float(a.mean())-expected['mean_harm_terms'][name]) < 1e-7


def verify(root):
    start = time.monotonic(); root = Path(root); report = json.loads((root/'results.json').read_text()); cfg = json.loads((root/'protocol.json').read_text())
    assert digest(root/'diagnostic.pt') == report['diagnostic_sha256'] and digest(root/'source_only_results.json') == report['source_only_results_sha256']
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    raw = torch.load(root/'diagnostic.pt', map_location='cpu'); labels = official_labels(cfg['target'], True)[raw['indices']]
    np.testing.assert_array_equal(labels, raw['labels'].numpy())
    cal = torch.load(cfg['source_prefix'], map_location='cpu'); lookup = {v: i for i, v in enumerate(cal['indices']['fit'])}; select = [lookup[i] for i in raw['indices']]
    np.testing.assert_array_equal(raw['environments']['native']['H'].numpy(), cal['native']['H'][select].numpy())
    va, vb = raw['basis']['A']['vh'].numpy(), raw['basis']['B']['vh'].numpy(); perm = raw['AGA_permutation'].numpy()
    pn = convert(raw['environments']['native']['parameters']); boundary = torch.load(cfg['boundary_reference'], map_location='cpu')
    x = ln(boundary['screen']['r'].numpy(), pn['gamma2'], pn['beta2'], pn['eps2']); rms = np.sqrt(np.mean((x@va.T)**2, axis=0))
    np.testing.assert_allclose(rms, raw['rms'].numpy(), atol=1e-9, rtol=1e-9)
    own_a = torch.load(cfg['original_aga'], map_location='cpu'); np.testing.assert_array_equal(pn['A'][perm].astype(np.float32), own_a['model.visual.transformer.resblocks.11.mlp.c_fc.weight'].numpy())
    source_states, source_params, checks = {}, {}, []; max_error = 0.
    for env, e in raw['environments'].items():
        p = convert(e['parameters']); H = e['H'].numpy(); h = e['h'].numpy()
        for case in cfg['cases']:
            name = case['name']; item = raw['cases'][name][env]; q = dict(p)
            for key, value in item['edits'].items(): q[key] = p[key]+value.numpy()
            st = states(H, h, p, q); row = report['records'][env][name]
            check_record(st, p, q, labels, row, vb[:, perm] if env == 'real_AGA' else vb)
            if env == 'native':
                source_states[name], source_params[name] = st, q
                data = raw['source_probe_data'][name]; take = data['take'].numpy(); teacher = N(h, p).argmax(1)
                expected_take = np.concatenate([np.flatnonzero(teacher == c)[np.argsort(np.asarray(raw['indices'])[teacher == c], kind='stable')[:cfg['probe_images_per_class']]] for c in range(len(p['head']))])
                np.testing.assert_array_equal(take, expected_take)
                shifts = np.stack([-cfg['probe_amplitude']*rms[:, None]*va, cfg['probe_amplitude']*rms[:, None]*va], axis=1).reshape(-1, va.shape[1])
                np.testing.assert_allclose(shifts, data['shifts'].numpy(), atol=1e-9, rtol=1e-9)
                xx, zz = st['x0'][take], st['x1'][take]; u0 = xx@p['A'].T+p['a']; u1 = zz@q['A'].T+q['a']
                b0 = st['h0'][take]-(u0*expit(1.702*u0))@p['B'].T; b1 = st['joint'][take]-(u1*expit(1.702*u1))@q['B'].T
                aa, bb = [], []
                for i in range(0, len(take), cfg['probe_batch_images']):
                    sl = slice(i, i+cfg['probe_batch_images']); a = (xx[sl, None]+shifts[None]).reshape(-1, xx.shape[1]); b = (zz[sl, None]+shifts[None]).reshape(-1, zz.shape[1])
                    a = a@p['A'].T+p['a']; b = b@q['A'].T+q['a']
                    aa.append(N(np.repeat(b0[sl], len(shifts), axis=0)+(a*expit(1.702*a))@p['B'].T, p))
                    bb.append(N(np.repeat(b1[sl], len(shifts), axis=0)+(b*expit(1.702*b))@q['B'].T, p))
                aa = np.concatenate(aa).reshape(data['before'].shape); bb = np.concatenate(bb).reshape(data['after'].shape)
                np.testing.assert_allclose(aa, data['before'].numpy(), atol=1e-8, rtol=1e-8); np.testing.assert_allclose(bb, data['after'].numpy(), atol=1e-8, rtol=1e-8)
                yy = teacher[take]; stable = (aa.argmax(-1) == yy[:, None, None]).all(-1); changed = bb.argmax(-1) != yy[:, None, None]; both = changed.all(-1) & stable
                check = row['source_probes']; assert int(stable.sum()) == check['teacher_stable_pairs'] and int(both.sum()) == check['protected_both_signs_changed']
                assert [int(both[yy == c].sum()) for c in range(len(p['head']))] == check['both_sign_failures_by_class']
                comm = (bb-aa)-(N(st['joint'][take], p)-N(h[take], p))[:, None, None]
                np.testing.assert_allclose(comm, data['commutator'].numpy(), atol=1e-8, rtol=1e-8)
            else:
                es, eq = source_states[name], source_params[name]; db = eq['B']-pn['B']; ze = es['z']
                if env == 'real_AGA': db, ze = db[:, perm], ze[:, perm]
                me, be = geometry(pn); mf, bf = geometry(p); coef = cfg['coefficient']; err_b = q['B']-p['B']-coef*db
                terms = dict(source_residual=coef*(N(es['joint'], pn)-N(es['h0'], pn)),
                             upper_compensation_mismatch=st['U']@mf-coef*es['U']@me,
                             conditional_activation_shift=coef*(st['z']-ze)@db.T@me,
                             readout_shift=coef*st['z']@db.T@(mf-me), attack_merge_B_remainder=st['z']@err_b.T@mf,
                             normalization_mismatch=(sd(st['joint'], p)-sd(h, p))*bf-coef*(sd(es['joint'], pn)-sd(es['h0'], pn))*be)
                actual = N(st['joint'], p)-N(h, p); error = float(np.max(np.abs(sum(terms.values())-actual))); max_error = max(max_error, error); assert error < 1e-8
                y, k, gap, _ = harm(N(h, p), N(st['joint'], p)); rows = np.arange(len(y))
                for term, value in terms.items():
                    np.testing.assert_allclose(value, item['transport'][term].numpy(), atol=1e-8, rtol=1e-8)
                    harms = -(value[rows, y]-value[rows, k])/gap
                    assert abs(float(harms.mean())-row['mean_transport_harm'][term]) < 1e-7
                np.testing.assert_allclose(N(st['joint'], p), item['numerator'].numpy(), atol=1e-8, rtol=1e-8)
            checks.append(dict(environment=env, method=name, exact_finite_class_counts=True, normalized_score_order=True))
    result = dict(status='passed', source_sha256=digest(__file__), diagnostic_sha256=report['diagnostic_sha256'],
                  matched_methods=4, matched_environments=3, matched_train_n=len(labels),
                  all_source_probe_inputs=4*len(take)*2*len(va), finite_transport_maximum_error=max_error,
                  source_only_metadata_sha256=report['source_only_results_sha256'], checks=checks, wall_seconds=time.monotonic()-start,
                  new_candidates=0, new_release_weights=0, new_model_forwards=0, partner_used=True, test_used=False,
                  scope='Independent NumPy all-token attention, actual changed tensors, QuickGELU, LayerNorm, source signed probes, all12 matched records and finite six-term transport. Retrospective only; no new efficacy/coverage conclusion.')
    save_json(root/'independent_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True); args = ap.parse_args()
    torch.set_num_threads(4); verify(args.output)
