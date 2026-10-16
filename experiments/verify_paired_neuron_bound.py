"""Finite class-margin bound for the frozen source-only contrast family."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from clip_benchmark import digest
from verify_cross_module import ln
from verify_levelset_audits import raw_scores


def run(output):
    start=time.monotonic(); out=Path(output); report=json.loads((out/'output_fit.json').read_text())
    assert digest(out/'training_point.pt')==report['training_point_sha256']
    raw=torch.load(out/'training_point.pt',map_location='cpu',weights_only=False)
    def array(d):return {k:v.numpy() if torch.is_tensor(v) else v for k,v in d.items()}
    p,q,pp,qp,cache=[array(raw[k]) for k in ['p','q','pp','qp','proxy_cache']]
    Q,S,X=[raw[k].numpy() for k in ['Q','S','X']]
    x=ln(cache['r'],pp['gamma2'],pp['beta2'],pp['eps2'])
    def activate(par):
        a=x@par['A'].T+par['a'];return a*expit(1.702*a)
    z=activate(qp); hg=cache['h']+(z-activate(pp))@pp['B'].T
    np.testing.assert_allclose(X,.5*z@S,atol=1e-9,rtol=1e-9)
    contrast=pp['head']-pp['head'].mean(0); M=pp['gamma'][:,None]*(pp['projection']@contrast.T); M-=M.mean(0)
    b=pp['beta']@pp['projection']@contrast.T; d=M.shape[0]
    def numerator(h):return h@M+np.sqrt(((h-h.mean(1,keepdims=True))**2).mean(1,keepdims=True)+pp['eps'])*b
    original=numerator(cache['h']); y=original.argmax(1); coalesced=numerator(hg)
    gaps=coalesced[np.arange(len(y)),y,None]-coalesced
    normals=M[:,y].T[:,None,:]-M.T[None]; biases=b[y,None]-b[None]
    task_lipschitz=np.linalg.norm(normals@Q,axis=2)+abs(biases)/d**.5
    full_lipschitz=np.linalg.norm(normals,axis=2)+abs(biases)/d**.5
    base_cost=sum((np.linalg.norm(q[k]-p[k])/np.linalg.norm(p[k]))**2 for k in ['A','a'])
    radius=np.linalg.norm(p['B'])*np.sqrt(.25-base_cost)
    u=2.**-24; bn=np.linalg.norm(p['B']); eb=u*(bn+radius)
    # Actual public proxy: fl(anchor + .5*fl(release-anchor)). Include
    # rounding in BOTH original and released merges, not only the last sum.
    anchor_bound=(2*np.linalg.norm(pp['B'])+(1+2*u+u*u)*bn)/(1-4*u-u*u)
    def merge_error(norm):return u*((1+.5*u)*norm+(2+.5*u)*anchor_bound)
    em=.5*eb+merge_error(bn+radius+eb)+merge_error(bn)
    lower=gaps-radius*np.linalg.norm(X,axis=1)[:,None]*task_lipschitz-em*np.linalg.norm(z,axis=1)[:,None]*full_lipschitz
    lower[np.arange(len(y)),y]=np.inf; certified=lower.min(1)>0
    before=raw_scores(cache['h'],pp); actual=raw['proxy_after'].numpy()
    result=dict(status='finite_family_bound_computed_not_unknown_peer_certificate',n=len(y),
        certified_no_class_change_for_any_real_K_within_original_budget=int(certified.sum()),
        not_certified=int((~certified).sum()),minimum_lower_gap=float(lower.min()),
        activation_contrast_norm_median=float(np.median(np.linalg.norm(X,axis=1))),activation_contrast_norm_maximum=float(np.linalg.norm(X,axis=1).max()),
        coefficient_radius=radius,parameter_rounding_norm_bound=em,actual_candidate_changed=int((actual.argmax(1)!=before.argmax(1)).sum()),
        per_class=[dict(class_id=c,n=int((y==c).sum()),certified=int((certified&(y==c)).sum())) for c in range(len(b))],
        source_sha256=digest(__file__),training_point_sha256=report['training_point_sha256'],wall_seconds=time.monotonic()-start,
        images_loaded=0,new_parameters_selected=False,scope='Fixed public-base suffix, fixed coalescence/Q/S, complete LN numerator. Bounds parameter rounding, not every GPU arithmetic operation. Not an upper bound on unknown-partner or full-space defense efficacy; uncertified is not evidence of a feasible flip.')
    (out/'finite_family_bound.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
