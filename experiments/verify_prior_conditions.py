"""Analytic falsification checks, not a defense or paper-result reproduction."""

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import time

import sympy as s


def run(output):
    start = time.monotonic(); out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    u = s.symbols('u', real=True)
    margin = 2-(1-u)**2; loss = s.log(1+s.exp(-margin))
    p = 1/(1+s.exp(margin))
    assert s.simplify(s.diff(loss, u, 2)-(2*p+4*(u-1)**2*p*(1-p))) == 0
    assert s.diff(loss, u).subs(u, 1) == 0
    # On [1/2,1], margin lies in [7/4,2]; hence the following bound is uniform.
    mu = 2/(1+math.exp(2)); lower = mu/8
    increase = float(loss.subs(u, Q(1,2))-loss.subs(u, 1))
    assert increase >= lower > 0
    assert margin.subs(u, Q(1,2)) == Q(7,4)
    # A normalized-feature readout realizes these logits without any class flip.
    coordinates = []
    for value in [Q(1,2), Q(3,4), Q(1)]:
        g = Q(2)-(1-value)**2; z = [float(g)/3, math.sqrt(1-(float(g)/3)**2)]
        assert abs(sum(v*v for v in z)-1) < 1e-14
        assert 3*z[0] > 0
        coordinates.append(dict(scale=str(value), margin=str(g), normalized_feature=z))
    x, y = s.symbols('x y', real=True)
    delta = s.prod(((y-i)/(-i))**2 for i in range(1,21))/100
    protected_gap = x+delta
    for i in range(1,21):
        assert delta.subs(y,i) == s.diff(delta,y).subs(y,i) == 0
        assert s.diff(protected_gap,x).subs(y,i) == 1
    before = -Q(1,200); after = before+Q(1,100)
    assert before < 0 < after
    result = dict(status='analytic_conditions_verified_no_new_defense',
        loss_not_accuracy=dict(positive_uniform_curvature_lower_bound=mu,
            endpoint_loss_derivative=0, loss_increase=increase, guaranteed_loss_lower_bound=lower,
            minimum_margin='7/4', classification_flips=0, rows=coordinates),
        finite_paths_not_global=dict(paths=20, entire_paths_value_and_normal_derivative_preserved=True,
            tangent_first_derivative_preserved=True, unseen_point=['-1/200','0'],
            original_gap=str(before), protected_gap=str(after), unseen_prediction_flipped=True,
            full_parameter_realization_claimed=False),
        wall_seconds=time.monotonic()-start, gradients=0, images_loaded=0, weights_loaded=0,
        original_methods_run=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Counterexamples to invalid implications, not counterexamples to the actual conditional loss theorems of Trap2; not CLIP publication models or budget-feasible defenses.')
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True)
    run(ap.parse_args().output)
