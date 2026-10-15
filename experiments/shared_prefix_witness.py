"""One analytic shared-prefix realizability projection; no new protection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from angular_distance import signed_residual
from attention_cascade import suffix
from clip_benchmark import digest
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from verify_cascade import suffix as numpy_suffix
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores, residual


@torch.no_grad()
def run(cfg):
    start = time.monotonic(); out = Path(cfg['output']); assert not (out/'shared_prefix.json').exists()
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    d = torch.load(out/'diagnostic.pt', map_location='cpu')
    cache = torch.load(cfg['cache'], map_location='cpu'); selection = json.loads((out/'selection.json').read_text())
    p, pp = on_device(cache['parameters']), on_device(cache['proxy_parameters'])
    h, target, dr = d['H'].cuda(), d['cases']['O_B']['eta'].cuda(), (d['H']-d['source_H']).cuda()
    t = cache['proxy']['t'][selection['proxy_positions']].cuda().double()
    u, v = cache['basis']['u'].cuda().double(), cache['basis']['vh'].cuda().double()
    assert float((u.T@u-torch.eye(u.shape[1], device='cuda')).abs().max()) < 1e-8
    assert float((v@v.T-torch.eye(len(v), device='cuda')).abs().max()) < 1e-8
    # A single B10 edit produces eta_i = T_i X U^T exactly in real arithmetic;
    # V is the fixed task-vector right basis and T_i = Z10_i V^T.
    flat_t = t.flatten(0,1); gram = flat_t.T@flat_t
    rhs = flat_t.T@(target@u).flatten(0,1)
    inverse = torch.linalg.pinv(gram, rtol=cfg['rtol']); free = inverse@rhs
    rows = torch.einsum('ntr,nts->nrs', t, dr@u)
    invrows = torch.einsum('rs,nsu->nru', inverse, rows)
    constraint = (rows*free).sum((1,2))
    dual_gram = torch.einsum('nrs,mrs->nm', rows, invrows)
    dual = torch.linalg.pinv(dual_gram, rtol=cfg['rtol'])@constraint
    x = free-torch.einsum('n,nrs->rs', dual, invrows)
    delta_d = u@x.T@v; unclipped = float(delta_d.norm()/p['D'].norm())
    factor = min(1., cfg['prefix_parameter_cap']/max(unclipped, 1e-30)); x *= factor
    eta = t@x@u.T; delta_d = u@x.T@v
    original = d['baseline'].cuda(); rounding = d['rounding'].cuda()
    modified = pp | {key:pp[key]+.5*value.cuda() for key,value in d['cases']['O_B']['delta'].items()}
    b = scores(suffix(h+eta, pp)[0]+rounding, pp)
    a = scores(suffix(h+eta, modified)[0]+rounding, modified)
    size = dr.flatten(1).norm(dim=1); rho = eta.flatten(1).norm(dim=1)/size
    transverse = (eta*dr).sum((1,2)).abs()/size.square()
    drift = signed_residual(original,b).abs().max(1).values
    y = original.argmax(1); initial_changed = d['cases']['O_B']['initial'].cuda().argmax(1) != y
    feasible = (drift <= .5) & (rho <= 1.) & (transverse < 1e-10) & (b.argmax(1) == y)
    restored = initial_changed & (a.argmax(1) == y)
    # Independent metric projection with a nullspace equality-constrained LS
    # construction, not the production dual-correction formula.
    tn, un, vn, dn, targetn = map(lambda a:a.cpu().numpy(), [t,u,v,dr,target])
    T = tn.reshape(-1,len(vn)); Y = (targetn@un).reshape(-1,len(vn))
    rr = np.einsum('ntr,nts->nrs', tn,dn@un).reshape(len(h),-1)
    _, singular, vh = np.linalg.svd(rr,full_matrices=True)
    rank = int((singular > cfg['rtol']*singular[0]).sum())
    null = vh[rank:].T
    G = np.kron(T.T@T,np.eye(len(vn))); vec = (T.T@Y).reshape(-1)
    independent = (null@np.linalg.lstsq(null.T@G@null,null.T@vec,rcond=cfg['rtol'])[0]).reshape(len(vn),len(vn))
    delta = un@independent.T@vn; nfactor = min(1., cfg['prefix_parameter_cap']/(np.linalg.norm(delta)/float(p['D'].norm())))
    independent *= nfactor
    coeff_error = float(np.max(np.abs(independent-x.cpu().numpy())))
    assert coeff_error < 1e-7
    etan = tn@independent@un.T
    pn, mn = convert(cache['proxy_parameters']), convert({k:val.cpu() if torch.is_tensor(val) else val for k,val in modified.items()})
    hn, roundn = d['H'].numpy(), d['rounding'].numpy()
    bn = raw_scores(numpy_suffix(hn+etan,pn)[0]+roundn,pn)
    an = raw_scores(numpy_suffix(hn+etan,mn)[0]+roundn,mn)
    score_error = max(float(np.max(np.abs(bn-b.cpu().numpy()))),float(np.max(np.abs(an-a.cpu().numpy()))))
    assert score_error < 1e-8
    np.testing.assert_array_equal(bn.argmax(1),b.argmax(1).cpu())
    np.testing.assert_array_equal(an.argmax(1),a.argmax(1).cpu())
    assert float(transverse.max()) < 1e-10
    torch.save(dict(X=x.cpu(), eta=eta.cpu(), baseline=b.cpu(), protected=a.cpu(), cfg=cfg),out/'shared_prefix.pt')
    result = dict(status='passed_shared_prefix_projection_verification',
        initially_changed=int(initial_changed.sum()), prediction_restored=int(restored.sum()),
        restored_in_conditional_set=int((restored & feasible).sum()),
        baseline_prediction_changes=int((b.argmax(1)!=y).sum()), baseline_worst_drift=float(drift.max()),
        all40_conditional_constraints_passed=bool(feasible.all()),
        maximum_transverse_error=float(transverse.max()), maximum_relative_state_norm=float(rho.max()),
        prefix_parameter_distance_before_clipping=unclipped,prefix_parameter_distance=float(delta_d.norm()/p['D'].norm()),
        relative_witness_projection_residual=float((target-eta).norm()/target.norm()),
        independent_nullspace_coefficient_error=coeff_error, independent_full_suffix_score_error=score_error,
        per_class=[dict(label=c,n=int((y==c).sum()),initially_changed=int((initial_changed&(y==c)).sum()),
                        restored=int((restored&(y==c)).sum()),feasible_restored=int((restored&feasible&(y==c)).sum())) for c in range(len(p['head']))],
        numerical_equality_rank=rank, source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']),
        projection_sha256=digest(out/'shared_prefix.pt'),wall_seconds=time.monotonic()-start,
        publication_optimization_steps=0,environment_gradient_calls=0,new_releases=0,
        unknown_partner_used=False,test_used=False,new_full_model_forwards=0,
        limitation='One fixed rank32 source-task-SVD B10 prefix edit, analytic equality-constrained projection only. Exact finite suffix scores are checked independently. Not a trained partner, no global reachability/impossibility statement or new attack/release. A failed projection does not rule out other realizable environments.')
    save_json(out/'shared_prefix.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    run(cfg)
