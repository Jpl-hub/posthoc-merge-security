"""Finite class-compensation necessary bound on frozen source data; no candidate."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def check(source, output):
    assert not output.exists()
    pin = '677354d85b6afab153c1f97254d109fedf12564d6324fba7e8560bdfd2d9711b'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pin
    assert '4090' in torch.cuda.get_device_name()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic(); data = torch.load(source, map_location='cpu', weights_only=False)
    p = {k: v.cuda().double() if torch.is_tensor(v) else v for k, v in data['parameters'].items()}
    states = {k: v.cuda().double() for k, v in data['states'].items()}
    h, star, z = [states[k] for k in ['h0', 'upper', 'z']]; n, d = h.shape
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; head = p['head']-p['head'].mean(0)
    m, bias = linear@head.T, offset@head.T
    def sigma(x): return ((x-x.mean(1, keepdim=True)).square().mean(1)+p['eps']).sqrt()
    def feature(x): return x@linear+sigma(x)[:, None]*offset
    before_feature, star_feature = feature(h), feature(star)
    before = before_feature/before_feature.norm(dim=1, keepdim=True)@p['head'].T
    y = before.argmax(1); k = before.shape[1]; rows = torch.arange(n, device='cuda')
    competitors = torch.arange(k, device='cuda').expand(n, k)[~torch.nn.functional.one_hot(y, k).bool()].reshape(n, k-1)
    mg = m[:, y].T[:, :, None]-m[:, competitors].permute(1, 0, 2)
    bg = bias[y, None]-bias[competitors]
    g = torch.einsum('nd,ndc->nc', h, mg)+sigma(h)[:, None]*bg
    assert (g > 0).all(), 'Tied teacher margin: report separately rather than drop an input.'
    u, singular, _ = torch.linalg.svd(torch.stack([g, bg], -1), full_matrices=False)
    rank_mask = singular > singular[:, :1]*1e-10
    def project(x):
        return x-torch.einsum('ncr,nr->nc', u, rank_mask*torch.einsum('ncr,nc->nr', u, x))
    q = project(torch.einsum('nd,ndc->nc', star, mg))
    counts = torch.bincount(y, minlength=k); w = q/g.square().sum(1, keepdim=True)/counts[y, None]
    if float(w.norm()) > 0: w /= w.norm()
    qspace, sv, _ = torch.linalg.svd(m, full_matrices=False); qspace = qspace[:, sv > sv[0]*1e-10]
    full_adjoint = torch.einsum('ndc,nc->nd', mg, w).T@z
    class_adjoint = qspace.T@full_adjoint
    outside = float((full_adjoint-qspace@class_adjoint).norm()/full_adjoint.norm().clamp_min(1e-30))
    assert outside < 1e-8
    delta = data['B_delta'].cuda().double()
    after = feature(states['joint']); gap_after = (after@head.T)[rows, y, None]-(after@head.T).gather(1, competitors)
    identity = project(gap_after)-q-project(torch.einsum('nd,ndc->nc', z@delta.T, mg))
    identity_error = float(identity.abs().max()); assert identity_error < 1e-7
    # Independent NumPy readout, including both normalizations, on every source input.
    hn, rn, cn, he = [x.cpu().numpy() for x in [h, linear, offset, p['head']]]
    sig = np.sqrt(np.mean((hn-hn.mean(1, keepdims=True))**2, axis=1)+p['eps'])
    f = hn@rn+sig[:, None]*cn; independent = (f/np.linalg.norm(f, axis=1, keepdims=True))@he.T
    score_error = float(np.abs(independent-before.cpu().numpy()).max()); assert score_error < 1e-8
    radius = math.sqrt(.5**2-data['O_distance']**2); bnorm = data['B_norm']
    lipschitz = float(torch.linalg.matrix_norm(linear, ord=2)+offset.norm()/math.sqrt(d))
    lhs = float((w*q).sum().abs()); basis = data['B_task_basis'].cuda().double()
    records = []
    for name, adjoint, activation in [('B_input32_output_class', class_adjoint@basis.T, z@basis.T),
                                      ('B_full_input_output_class', class_adjoint, z),
                                      ('B_full_input_full_output', full_adjoint, z)]:
        tau = (star_feature.norm(dim=1)+lipschitz*bnorm*radius*activation.norm(dim=1))/before_feature.norm(dim=1)
        allowance = float(.5*(tau*(w.abs()*g).sum(1)).sum())
        coefficient = bnorm*float(adjoint.norm()); capacity = radius*coefficient
        lower = max(0., lhs-allowance)/coefficient if coefficient else None
        records.append(dict(family=name, dual_lhs=lhs, task_tolerance_allowance=allowance,
            parameter_capacity=capacity, budget_upper_rhs=allowance+capacity,
            lower_distance_bound=lower, infeasible_certificate=lhs>allowance+capacity+1e-8*max(1., lhs)))
    result = dict(status='finite_source_necessary_bound_checked_not_new_defense', source_n=n,
        source_sha256=pin, parent_sha256=data['parent_sha256'], implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_upstream_distance=data['O_distance'], remaining_B_distance=radius, task_budget=.5,
        exact_finite_projection_identity_max_abs=identity_error, NumPy_normalized_score_error=score_error,
        output_class_space_rank=qspace.shape[1], full_output_extra_adjoint_fraction=outside,
        dual_witness='Finite residual, teacher-class balanced, divided by teacher numerator gap norm squared; no weight optimization.',
        rows=records, gradients=0, new_weights=0, images_loaded=0, partner_used=False, test_used=False,
        confirmation_used=False, old_counterexamples_not_repaired=True,
        conclusion='An infeasible certificate excludes only this fixed finite upstream change and stated family. Otherwise this conservative bound is inconclusive, not a feasible-point or fusion-effect guarantee. Free B output directions add no projected class-normal directions.',
        GPU=torch.cuda.get_device_name(), wall_seconds=time.monotonic()-start)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    args = parser.parse_args(); check(args.source, args.output)
