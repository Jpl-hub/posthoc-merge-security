"""Read-only source-data audit of a failed frozen nonlinear restoration."""

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
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores, residual


def audit(cfg):
    start = time.monotonic(); out = Path(cfg['output'])
    report = json.loads((out/'results.json').read_text()); validation = json.loads((out/'independent_validation.json').read_text())
    assert validation['status'] == 'passed_failure_verification'
    assert digest(out/'diagnostic.pt') == report['diagnostic_sha256']
    data = convert(torch.load(out/'diagnostic.pt', map_location='cpu'))
    pred = convert(torch.load(out/'predictions.pt', map_location='cpu')); p = data['parameters']
    c = convert(torch.load(out/'point.pt', map_location='cpu')); n = len(data['indices']); core = data['core']
    ns = len(core)*len(data['shifts_a']); err = np.abs(residual(pred['teacher'], pred['joint']))
    groups = []; winner = pred['teacher'].argmax(1); pairs = err.shape[1]-1
    for name, lo, hi in [('fit_natural', 0, cfg['fit_n']), ('scale_natural', cfg['fit_n'], n),
                         ('A_task_SVD', n, n+ns), ('B_task_SVD', n+ns, len(err))]:
        e = err[lo:hi]; changed = pred['teacher'][lo:hi].argmax(1) != pred['joint'][lo:hi].argmax(1)
        groups.append(dict(group=name, n=hi-lo, worst=float(e.max()), changed=int(changed.sum()),
                           violating_inputs=int((e.max(1) > cfg['task_budget']).sum()),
                           violating_pairs=int((e > cfg['task_budget']).sum()), total_competitor_pairs=(hi-lo)*pairs))
    flat = np.unravel_index(err.argmax(), err.shape); row, comp = map(int, flat)
    mapping = np.r_[np.arange(n), np.repeat(core, len(data['shifts_a'])), np.repeat(core, len(data['shifts_b']))]
    y = int(winner[row]); s0, s1 = pred['teacher'][row], pred['joint'][row]
    worst = dict(task_row=row, source_train_index=data['indices'][int(mapping[row])], teacher_class=y, competitor_class=comp,
                 original_gap=float(s0[y]-s0[comp]), joint_gap=float(s1[y]-s1[comp]), relative_drift=float(err[row, comp]),
                 natural_input=row < n)
    final_step = data['steps'][-1]; selected = final_step['selected']; rows = mapping[selected]
    x = data['cache']['x'][rows].copy(); off = np.zeros((len(x), p['A'].shape[0]))
    a = (selected >= n) & (selected < n+ns); b = selected >= n+ns
    x[a] += data['shifts_a'][(selected[a]-n) % len(data['shifts_a'])]
    off[b] = data['shifts_b'][(selected[b]-n-ns) % len(data['shifts_b'])]
    base_pre = data['cache']['x']@p['A'].T+p['a']; z = base_pre*expit(1.702*base_pre)
    base = data['cache']['h']-z@p['B'].T
    proposal = final_step['proposal']; va = data['basis']['A']['vh']
    u = x@p['A'].T+p['a']+np.linalg.norm(p['A'])*(x@va.T)@proposal.T
    h = base[rows]+(u*expit(1.702*u)+off)@(p['B']+c['delta_B']).T
    true_r = residual(pred['teacher'][selected], raw_scores(h, p))
    teacher_y = pred['teacher'][selected].argmax(1); keep = np.ones(true_r.shape, bool); keep[np.arange(len(keep)), teacher_y] = False
    linear = np.sum(final_step['factor_a']*(final_step['factor_s']@proposal.T), axis=1)+final_step['linear_rhs']
    nonlinear = true_r[keep]
    record = dict(linear_worst_on_active_pairs=float(np.abs(linear).max()), finite_worst_on_active_pairs=float(np.abs(nonlinear).max()),
                  finite_linear_disagreement=float(np.abs(nonlinear-linear).max()), upstream_proposal_distance=float(np.linalg.norm(proposal)),
                  preallocated_upstream_radius=cfg['parameter_budget']/np.sqrt(2), active_inputs=len(selected),
                  nonlinear_full_step_global_worst=report['history'][-1]['trials'][0]['worst_margin_drift'],
                  accepted_step_fraction=report['history'][-1]['trials'][-1]['fraction'],
                  linear_regularizer=final_step['mu'])
    result = dict(status='verified_source_only_restoration_failure_attribution', target=cfg['target'], groups=groups, worst_witness=worst,
                  final_step_linear_vs_finite=record, source_task_passed=False, new_release_weights=0, new_model_forwards=0,
                  new_parameter_updates=0, partner_used=False, test_used=False, diagnostic_sha256=report['diagnostic_sha256'],
                  source_sha256=digest(__file__), wall_seconds=time.monotonic()-start,
                  conclusion='The downstream branch has finite multiclass effect, but the fixed shared upstream repair fails its task gate. Local least-squares stationarity does not certify finite restoration. No claim that the admissible function class is globally infeasible, nor that more iterations or budget would solve transfer.')
    save_json(out/'failure_analysis.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    torch.set_num_threads(4); audit(json.loads(Path(args.protocol).read_text()))
