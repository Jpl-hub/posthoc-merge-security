"""NumPy finite-difference checks of the frozen task boundary diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest,merge
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_cross_module import ln
from verify_probe_candidate import array,features


def verify(output):
    out=Path(output);report=json.loads((out/'results.json').read_text());file=out/'diagnostic.pt'
    assert digest(file)==report['diagnostic_sha256'];data=torch.load(file,map_location='cpu');cfg=data['cfg']
    anchor,expert,head,pins=checked_models(report['target']);assert pins==report['checkpoint_sha256']
    convert=lambda model:{k:array(v) if torch.is_tensor(v) else v for k,v in extended_parameters(model,head).items()}
    p=convert(expert);proxy=merge(anchor,expert,anchor,cfg['proxy_beta']);pp=convert(proxy);del proxy
    rank=cfg['probe_rank'];size=rank**2;u,v=[array(data['basis_O'][k]) for k in ['u','vh']];q=array(data['output_basis'])
    gradient=array(data['gradient']);random=np.random.default_rng(20261015).normal(size=len(gradient));random/=np.linalg.norm(random)
    def hidden(r,pars):
        pre=ln(r,pars['gamma2'],pars['beta2'],pars['eps2'])@pars['A'].T+pars['a']
        return pre*expit(1.702*pre)
    def predict(item,pars,theta,factor=1.):
        r,z,h=[array(item[k]) for k in ['r','o_input','h']]
        eo=np.linalg.norm(p['O'])*u@theta[:size].reshape(rank,rank)@v
        eb=np.linalg.norm(p['B'])*q@theta[size:].reshape(q.shape[1],p['B'].shape[1])
        rounding=h-r-hidden(r,pars)@pars['B'].T-pars['b'];rr=r+factor*z@eo.T
        final=rounding+rr+hidden(rr,pars)@(pars['B']+factor*eb).T+pars['b']
        return features(final,(proxy if pars is pp else expert).model.visual)@pars['head'].T
    # Keep the public proxy object for its frozen readout only.
    proxy=merge(anchor,expert,anchor,cfg['proxy_beta'])
    zero=np.zeros_like(gradient);screen=data['screen'];baseline=predict(screen,p,zero)
    y=baseline.argmax(1);other=baseline.copy();other[np.arange(len(y)),y]=-np.inf;gaps=baseline[np.arange(len(y)),y]-other.max(1)
    indices=data['split']['indices'];critical=[]
    for c in range(baseline.shape[1]):
        members=np.flatnonzero(y==c);ordered=members[np.lexsort((np.asarray(indices)[members],gaps[members]))]
        critical.extend(ordered[:cfg['contexts_per_class']].tolist())
    assert sorted(critical)==data['critical_offsets'] and len(indices)==len(set(indices))==cfg['screen_n']
    for path,pin in data['split']['excluded_metadata_sha256'].items():assert digest(path)==pin,path
    proxy_before=predict(data['proxy'],pp,zero);winner=proxy_before.argmax(1);rows=np.arange(len(winner));others=proxy_before.copy();others[rows,winner]=-np.inf;competitor=others.argmax(1)
    counts=np.bincount(winner,minlength=proxy_before.shape[1]);weights=1./counts[winner]/int((counts>0).sum())
    angular=lambda s:(s-s.mean(1,keepdims=True))/np.linalg.norm(s-s.mean(1,keepdims=True),axis=1,keepdims=True)
    a=angular(proxy_before)
    # torch.median uses the lower middle order statistic for an even sample.
    # Reproduce that frozen definition rather than NumPy's averaged median.
    unit=np.sort(a[rows,winner]-a[rows,competitor])[(len(a)-1)//2]
    def objective(theta):
        s=angular(predict(data['proxy'],pp,theta,cfg['proxy_beta']));others=s.copy();others[rows,winner]=-np.inf
        return -np.sum(weights*np.maximum(s[rows,winner]-others.max(1),0))/unit
    step=cfg['derivative_step'];slope=(objective(step*random)-objective(-step*random))/(2*step)
    assert abs(slope-gradient@random)<1e-6,(slope,float(gradient@random),float(unit))
    checks=[]
    old=torch.load(Path(cfg['cache_reference'])/'calibration.pt',map_location='cpu')
    for record in report['records']:
        name=record['group'];saved=data['groups'][name];matrix=array(saved['matrix']);direction=array(saved['direction'])
        offsets=data['critical_offsets'] if name=='boundary' else data['balanced_offsets']
        item={k:v[offsets] for k,v in screen.items()} if name!='old_contexts' else {k:torch.cat([old[g][k][:cfg['old_probe_contexts']] for g in ['native','scale']]) for k in screen}
        before=predict(item,p,zero);winners=before.argmax(1);r=np.arange(len(before));gap=before[r,winners,None]-before;keep=np.ones_like(gap,dtype=bool);keep[r,winners]=False
        delta=(predict(item,p,step*random)-predict(item,p,-step*random))/(2*step)
        j=((delta[r,winners,None]-delta)/np.maximum(gap,1e-12))[keep]
        error=float(np.abs(j-matrix@random).max());assert error<1e-6
        _,s,vh=np.linalg.svd(matrix,full_matrices=False);tol=np.finfo(np.float64).eps*max(matrix.shape)*s[0];rank0=int((s>tol).sum())
        projected=gradient-vh[:rank0].T@(vh[:rank0]@gradient);d=projected/np.linalg.norm(projected)
        np.testing.assert_allclose(d,direction,atol=1e-7,rtol=1e-7)
        central=(objective(step*direction)-objective(-step*direction))/(2*step);assert abs(central-record['proxy_unit_directional_derivative'])<1e-6
        checks.append({'group':name,'context_margin_jvp_fd_maximum_error':error,'projection_direction_maximum_error':float(np.abs(d-direction).max()),'proxy_directional_fd_error':float(abs(central-record['proxy_unit_directional_derivative']))})
    result={'status':'passed','source_sha256':digest(__file__),'diagnostic_sha256':digest(file),'checks':checks,
            'proxy_random_direction_fd_error':float(abs(slope-gradient@random)),'critical_context_selection_reproduced':True,
            'calibration_metadata_hashes_verified':True,'new_model_forwards':0,'new_weights':0,'test_used':False,'partner_used':False,
            'scope':'Independent NumPy nonlinear forward finite differences, null projection and teacher-only selection; not efficacy or full test evaluation.'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
