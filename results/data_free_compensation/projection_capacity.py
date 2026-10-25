"""Global margin capacity of a fixed readout, with unrestricted hidden direction."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from _impl.io import digest
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
base, expert, head, pins = checked_models('MNIST')
P0 = base.model.visual.proj.detach()
Pe = expert.model.visual.proj.detach()
P = (P0+.5*(Pe-P0)).double().numpy()
W = head.weight.double().numpy()
pnorm = np.linalg.norm(P)
rho = np.linalg.norm((.5*(Pe-P0)).double().numpy())/pnorm
out = Path('outputs/frozen_classifier_20261025/projection_capacity')
out.mkdir(exist_ok=False)
records, arrays = [], {}
for y in range(10):
    competitors = [j for j in range(10) if j != y]
    d = W[y]-W[competitors]
    A = P @ (d/np.linalg.norm(d, axis=1, keepdims=True)).T/pnorm
    G = A.T @ A
    best, weight = np.inf, None
    for n in range(1, 10):
        for face in itertools.combinations(range(9), n):
            idx = np.array(face)
            g = G[np.ix_(idx, idx)]
            v = np.linalg.solve(g, np.ones(n))
            p = v/v.sum()
            if p.min() < -1e-12:
                continue
            full = np.zeros(9)
            full[idx] = np.maximum(p, 0)
            full /= full.sum()
            value = float(full@G@full)
            if value < best:
                best, weight = value, full
    direction = A@weight
    dual = np.linalg.norm(direction)
    direction /= dual
    primal = float((direction@A).min())
    kkt_violation = max(0., float(best-(G@weight).min()))
    reference = minimize(lambda p: p@G@p, np.ones(9)/9, jac=lambda p: 2*G@p, bounds=[(0.,1.)]*9, constraints=[dict(type='eq', fun=lambda p:p.sum()-1, jac=lambda p:np.ones(9))], method='SLSQP', options=dict(maxiter=1000, ftol=1e-15))
    assert reference.success and abs(primal-dual) < 1e-10 and kkt_violation < 1e-10
    error = abs(np.sqrt(reference.fun)-dual)
    assert error < 1e-8
    records.append(dict(class_id=y, unrestricted_maximum_relative_P_radius=float(dual), own_update_relative_radius=float(rho), upper_bound_exceeds_own_scale=bool(dual>rho), primal_dual_gap=float(dual-primal), simplex_KKT_violation=kkt_violation, scipy_reference_error=float(error), active_competitors=[competitors[k] for k in np.flatnonzero(weight>1e-9)]))
    arrays[y] = dict(simplex_weight=weight, optimal_unrestricted_hidden_direction=direction, contrasts=d)
np.savez(out/'arrays.npz', **{f'class_{y}_{k}':v for y,row in arrays.items() for k,v in row.items()})
source = Path(__file__).read_text()
result = dict(status='completed_read_only_capacity_calculation', records=records, checkpoint_sha256=pins, seconds=time.perf_counter()-start, source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(), arrays_sha256=digest(out/'arrays.npz'), new_image_forwards=0, real_images=0, new_defense_points=0, new_AGA_calls=0, note='All10classes fixed in advance;511simplex faces per class. Exact capacity only for arbitrary hidden direction and Frobenius projection uncertainty at this fixed public mean readout. Relaxation upper bound for actual LN/MLP outputs, not an attainable defense, task preservation, private-partner guarantee, or class selection rule.')
(out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'}, indent=2))
