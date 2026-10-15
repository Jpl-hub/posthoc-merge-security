"""Independent NumPy replay of every finite environment trial and witness."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from subspace_probe import save_json
from verify_cascade import suffix
from verify_covariance_noise import official_labels
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores, residual


def gap(s, y, scale):
    other = s.copy(); other[np.arange(len(y)), y] = -np.inf
    return (s[np.arange(len(y)), y]-other.max(1))/scale


def verify(cfg):
    start = time.monotonic(); out = Path(cfg['output'])
    report = json.loads((out/'results.json').read_text())
    assert digest(out/'diagnostic.pt') == report['diagnostic_sha256']
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin
    cache = torch.load(cfg['cache'], map_location='cpu')
    data = convert(torch.load(out/'diagnostic.pt', map_location='cpu'))
    p, pp = convert(cache['parameters']), convert(cache['proxy_parameters'])
    selection = json.loads((out/'selection.json').read_text())
    labels = official_labels(cfg['target'], True)
    # Independently select without protected scores and check source states.
    sy = raw_scores(cache['native']['h'].numpy(), p).argmax(1)
    py = raw_scores(cache['proxy']['h'].numpy(), pp).argmax(1)
    lookup = {x:i for i,x in enumerate(cache['indices']['fit'])}; chosen = []
    for cls in range(len(p['head'])):
        rows = [i for i, idx in enumerate(cache['indices']['proxy'])
                if int(py[i]) == cls == int(sy[lookup[idx]]) == int(labels[idx])]
        chosen.extend(rows[:cfg['per_class']])
    assert chosen == selection['proxy_positions']
    np.testing.assert_array_equal(data['H'], cache['proxy']['H'][chosen].double().numpy())
    np.testing.assert_array_equal(data['source_H'], cache['native']['H'][selection['native_positions']].double().numpy())
    h, rounding = data['H'], data['rounding']; d = h-data['source_H']
    size = np.linalg.norm(d.reshape(len(h), -1), axis=1); unit = d/size[:,None,None]
    def evaluate(eta, pars): return raw_scores(suffix(h+eta, pars)[0]+rounding, pars)
    baseline = evaluate(np.zeros_like(h), pp)
    np.testing.assert_allclose(baseline, data['baseline'], atol=1e-9, rtol=1e-9)
    y = baseline.argmax(1); other = baseline.copy(); other[np.arange(len(y)), y] = -np.inf
    scale = baseline[np.arange(len(y)), y]-other.max(1)
    records = []; trials_verified = 0; score_error = 0.; fd_error = 0.
    for case, expected in zip(cfg['cases'], report['cases']):
        saved = data['cases'][case['name']]; delta = saved['delta']
        distance = sum(np.sum(value*value)/np.sum(p[key]*p[key]) for key,value in delta.items())**.5
        assert distance <= cfg['parameter_budget']
        modified = pp | {key:pp[key]+cfg['coefficient']*value for key,value in delta.items()}
        eta = np.zeros_like(h); initial = evaluate(eta, modified); initial_changed = initial.argmax(1) != y
        np.testing.assert_allclose(initial, saved['initial'], atol=1e-9, rtol=1e-9)
        for step_index, item in enumerate(saved['steps']):
            step = item['step']; before = gap(evaluate(eta, modified), y, scale)
            active = initial_changed & (before < cfg['restored_gap'])
            eps = cfg['derivative_step']
            fd = (gap(evaluate(eta+eps*step, modified), y, scale)-gap(evaluate(eta-eps*step, modified), y, scale))/(2*eps)
            err = float(np.abs(fd-item['derivative']).max()); fd_error = max(fd_error, err)
            assert err < cfg['derivative_tolerance']
            accepted = np.zeros(len(h), bool); alpha = np.ones(len(h))
            for trial in item['trials']:
                np.testing.assert_array_equal(alpha, trial['alpha'])
                proposal = eta+alpha[:,None,None]*step
                b, a = evaluate(proposal, pp), evaluate(proposal, modified)
                err = max(float(np.abs(b-trial['baseline']).max()), float(np.abs(a-trial['released']).max()))
                score_error = max(score_error, err); assert err < 1e-9
                drift = np.abs(residual(baseline, b)).max(1)
                actual = gap(a, y, scale)-before; pred = alpha*item['derivative']
                valid = (active & ~accepted & (pred > 1e-12) & (actual > 1e-12)
                         & (actual/np.maximum(pred, 1e-30) >= cfg['acceptance_ratio'])
                         & (drift <= cfg['baseline_budget']) & (b.argmax(1) == y)
                         & (np.linalg.norm(proposal.reshape(len(h), -1), axis=1) <= size))
                np.testing.assert_array_equal(valid, trial['accepted'])
                accepted |= valid; alpha = np.where(accepted, alpha, alpha*.5); trials_verified += 1
            update = alpha*accepted
            np.testing.assert_array_equal(update, item['applied_scale'])
            eta += update[:,None,None]*step
            print(json.dumps(dict(case=case['name'], step=step_index, trials_verified=trials_verified)), flush=True)
        np.testing.assert_allclose(eta, saved['eta'], atol=1e-11, rtol=1e-11)
        b, a = evaluate(eta, pp), evaluate(eta, modified)
        np.testing.assert_allclose(b, saved['final_baseline'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(a, saved['final_released'], atol=1e-9, rtol=1e-9)
        margin = gap(a, y, scale)
        restored = initial_changed & (margin >= cfg['restored_gap'])
        # The optional 0.01 inner stopping target has ULP-level ties. Preserve
        # its ORIGINAL Torch count; do not reinterpret a tie as an efficacy
        # threshold pass. Actual positive-margin class restoration is separate.
        near_target = initial_changed & (np.abs(margin-cfg['restored_gap']) <= 1e-9)
        lower = int((initial_changed & (margin > cfg['restored_gap']+1e-9)).sum())
        upper = int((initial_changed & (margin >= cfg['restored_gap']-1e-9)).sum())
        prediction_restored = initial_changed & (a.argmax(1) == y)
        drift = float(np.abs(residual(baseline, b)).max())
        transverse = float(np.max(np.abs(np.sum(eta*unit, axis=(1,2)))/size))
        rho = np.linalg.norm(eta.reshape(len(h), -1), axis=1)/size
        assert drift <= cfg['baseline_budget']+1e-10 and transverse < 1e-10 and rho.max() <= 1+1e-10
        assert int(initial_changed.sum()) == expected['initially_changed']
        assert lower <= expected['restored'] <= upper
        for cls, entry in enumerate(expected['per_class']):
            mask = y == cls
            lo = int((initial_changed & mask & (margin > cfg['restored_gap']+1e-9)).sum())
            hi = int((initial_changed & mask & (margin >= cfg['restored_gap']-1e-9)).sum())
            assert int(mask.sum()) == entry['n'] and lo <= entry['restored'] <= hi
        records.append(dict(case=case['name'], initially_changed=int(initial_changed.sum()),
            original_torch_stopping_target_count=expected['restored'], numpy_strict_stopping_target_count=int(restored.sum()),
            stopping_target_near_ties=int(near_target.sum()), stopping_target_nonambiguous_count_interval=[lower, upper],
            prediction_restored=int(prediction_restored.sum()),
            still_changed=int((a.argmax(1) != y).sum()), baseline_worst_drift=drift,
            restored_minimum_positive_margin=float(margin[prediction_restored].min()) if prediction_restored.any() else None,
            maximum_transverse_error=transverse, relative_norms=rho[prediction_restored].tolist(),
            per_class=[dict(label=c, n=int((y == c).sum()), initially_changed=int((initial_changed & (y == c)).sum()),
                            prediction_restored=int((prediction_restored & (y == c)).sum())) for c in range(len(p['head']))]))
    result = dict(status='passed', records=records, all_finite_trials_verified=trials_verified,
        maximum_numpy_score_error=score_error, maximum_numpy_directional_fd_error=fd_error,
        diagnostic_sha256=report['diagnostic_sha256'], source_sha256=digest(__file__),
        wall_seconds=time.monotonic()-start, new_full_model_forwards=0, new_release_weights=0,
        peer_used=False, test_used=False, confirmation_used=False,
        scope='Independent all-token NumPy attention, LN/QuickGELU/normalized scores, selection, every finite trial, accept/reject decisions, transverse geometry and actual class restoration. Interface counterexamples only; no unknown-partner coverage or new defense.')
    save_json(out/'independent_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True)
    args = ap.parse_args(); torch.set_num_threads(4)
    verify(json.loads(Path(args.protocol).read_text()))
