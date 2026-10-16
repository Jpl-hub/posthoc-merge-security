"""Independent NumPy finite readout, projection and dual-bound reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def verify(source, report, output):
    assert not output.exists()
    result = json.loads(report.read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['source_sha256']
    data = torch.load(source, map_location='cpu', weights_only=False)
    p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in data['parameters'].items()}
    h, star, z, joint = [data['states'][k].numpy() for k in ['h0', 'upper', 'z', 'joint']]
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; head = p['head']-p['head'].mean(0)
    m, b = linear@head.T, offset@head.T
    def feature(x):
        sigma = np.sqrt(np.mean((x-x.mean(1, keepdims=True))**2, axis=1)+p['eps'])
        return x@linear+sigma[:, None]*offset
    f = feature(h); y = (f@head.T).argmax(1); n, d = h.shape; count = np.bincount(y, minlength=len(head))
    competitors = np.array([[c for c in range(len(head)) if c != t] for t in y])
    mg = m[:, y].T[:, :, None]-m[:, competitors].transpose(1, 0, 2)
    bg = b[y, None]-b[competitors]; numerator = f@head.T
    gaps = numerator[np.arange(n), y, None]-np.take_along_axis(numerator, competitors, axis=1)
    spans = np.stack([gaps, bg], -1)
    projection = np.eye(len(head)-1)-spans@np.linalg.pinv(spans, rcond=1e-10)
    q = np.einsum('nij,nj->ni', projection, np.einsum('nd,ndc->nc', star, mg))
    w = q/(gaps*gaps).sum(1, keepdims=True)/count[y, None]
    if np.linalg.norm(w): w /= np.linalg.norm(w)
    adjoint = np.einsum('ndc,nc->nd', mg, w).T@z
    lhs = abs(np.sum(w*q)); radius = result['remaining_B_distance']; norm = data['B_norm']
    lipschitz = np.linalg.norm(linear, 2)+np.linalg.norm(offset)/np.sqrt(d)
    values, ratios = [], []
    for row in result['rows']:
        restricted = row['family'] == 'B_input32_output_class'
        act = z@data['B_task_basis'].numpy().T if restricted else z
        dual = adjoint@data['B_task_basis'].numpy().T if restricted else adjoint
        tau = (np.linalg.norm(feature(star), axis=1)+lipschitz*norm*radius*np.linalg.norm(act, axis=1))/np.linalg.norm(f, axis=1)
        allowance = .5*np.sum(tau*(abs(w)*gaps).sum(1)); capacity = radius*norm*np.linalg.norm(dual)
        expected = [lhs, allowance, capacity, allowance+capacity]
        actual = [row[k] for k in ['dual_lhs', 'task_tolerance_allowance', 'parameter_capacity', 'budget_upper_rhs']]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-8)
        assert row['infeasible_certificate'] == (lhs > allowance+capacity+1e-8*max(1., lhs))
        values.append(dict(family=row['family'], maximum_scalar_difference=float(np.max(abs(np.array(actual)-expected)))))
        if not restricted:
            true_ratio = np.linalg.norm(feature(joint), axis=1)/np.linalg.norm(f, axis=1)
            assert (tau+1e-10 >= true_ratio).all()
            ratios.append(dict(maximum_bound=float(tau.max()), maximum_frozen_actual_ratio=float(true_ratio.max())))
    bdelta = data['B_delta'].numpy()
    np.testing.assert_allclose(star+z@bdelta.T, joint, atol=1e-8, rtol=1e-8)
    validation = dict(status='passed', independent_NumPy_projection_and_bounds=True, source_n=n,
        scalar_checks=values, normalization_bound_looseness=ratios,
        source_sha256=result['source_sha256'], report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gradients=0, images=0, new_weights=0, scope='Frozen source diagnosis only; bounds remain inconclusive.')
    output.write_text(json.dumps(validation, indent=2)+'\n'); print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--report', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound_validation.json'))
    args = parser.parse_args(); verify(args.source, args.report, args.output)
