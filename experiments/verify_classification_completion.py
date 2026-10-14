"""Independent NumPy class-level targets on the original target training cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_probe_candidate import array, features
from verify_cross_module import ln


def target_numpy(h0,teacher,q,w,a,linear,bias,epsilon,alpha_target):
    se=np.sqrt(np.mean((teacher-teacher.mean(1,keepdims=True))**2,1,keepdims=True)+epsilon)
    te=teacher@linear+se*bias;tn=np.linalg.norm(te,axis=1,keepdims=True)
    aux=h0-(h0@q)@q.T;task=(teacher@q)@q.T+se*w
    ac=aux-aux.mean(1,keepdims=True);tc=task-task.mean(1,keepdims=True);wc=w-w.mean()
    b0=np.mean(ac**2,1,keepdims=True)+epsilon;t0=np.mean(tc**2,1,keepdims=True);cross=np.mean(tc*wc,1,keepdims=True)
    def family(c):
        # Direct positive quadratic root, independent of the stable Torch expression.
        rho=(np.sqrt(c*c*cross*cross+a*(b0+c*c*t0))-c*cross)/a
        h=aux+c*task-rho*w;t=h@linear+rho*bias
        return h,c*tn/np.linalg.norm(t,axis=1,keepdims=True),rho
    low=np.zeros_like(tn);high=np.ones_like(tn)
    for _ in range(16):
        mask=family(high)[1]<alpha_target
        if not mask.any():break
        high=np.where(mask,high*2,high)
    assert np.all(family(high)[1]>=alpha_target)
    for _ in range(48):
        mid=(low+high)/2;mask=family(mid)[1]<alpha_target
        low=np.where(mask,mid,low);high=np.where(mask,high,mid)
    h,alpha,rho=family((low+high)/2)
    return h,alpha,rho


def verify(output):
    out=Path(output);r=json.loads((out/'results.json').read_text());cfg=json.loads((out/'protocol.json').read_text())
    path=Path(cfg['cache_reference'])/'calibration.pt';assert digest(path)==r['calibration_sha256']
    assert digest(out/'readout.pt')==r['readout_sha256']
    cal=torch.load(path,map_location='cpu');space=torch.load(out/'readout.pt',map_location='cpu')
    _,expert,head,hashes=checked_models(r['target']);assert hashes==r['checkpoint_sha256']
    raw=extended_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in raw.items()}
    q,w=array(space['Q']),array(space['w']);a=1.-np.mean((w-w.mean())**2)
    linear=p['gamma'][:,None]*p['projection'];linear-=linear.mean(0);bias=p['beta']@p['projection']
    contrast=p['head']-p['head'].mean(0);m=linear@contrast.T;b=bias@contrast.T
    np.testing.assert_allclose(q@q.T@m,m,atol=1e-10)
    np.testing.assert_allclose(w@m,b,atol=1e-10)
    assert abs(a-r['geometry']['quadratic_a'])<1e-12
    bo=cal['basis_O'];s=array(bo['s']);eo=np.linalg.norm(p['O'])*array(bo['u'])@(-cfg['parameter_budget']*np.diag(s)/np.linalg.norm(s))@array(bo['vh'])
    def hidden(rr):
        pre=ln(rr,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a']
        return rr+(pre*expit(1.702*pre))@p['B'].T+p['b']
    rows=[]
    for group in ['native','scale']:
        data=cal[group];h=array(data['h']);rr=array(data['r'])
        h0=h-hidden(rr)+hidden(rr+array(data['o_input'])@eo.T)
        ideal,alpha,rho=target_numpy(h0,h,q,w,a,linear,bias,p['eps'],1.-cfg['margin_fraction']/2.)
        before=features(h,expert.model.visual)@p['head'].T;after=features(ideal,expert.model.visual)@p['head'].T
        centered=lambda x:x-x.mean(1,keepdims=True)
        error=float(np.max(np.abs(centered(after)-(1.-cfg['margin_fraction']/2.)*centered(before))))
        sigma=np.sqrt(np.mean((ideal-ideal.mean(1,keepdims=True))**2,1,keepdims=True)+p['eps'])
        assert error<1e-10 and np.max(np.abs(sigma-rho))<1e-10
        np.testing.assert_array_equal(before.argmax(1),after.argmax(1))
        rows.append({'group':group,'n':len(h),'centered_score_maximum_error':error,'amplitude_root_maximum_error':float(np.max(np.abs(alpha-(1.-cfg['margin_fraction']/2.)))),'variance_identity_error':float(np.max(np.abs(sigma-rho))),'ideal_predictions_changed':0})
    result={'status':'passed','source_sha256':digest(__file__),'calibration_sha256':digest(path),'readout_sha256':digest(out/'readout.pt'),'records':rows,'new_release_weights':0,'new_model_forwards':0,'test_used':False,'partner_used':False,'scope':'Ideal per-input targets only; no uniform publication transform verified'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


def audit_raw_failure(output):
    out=Path(output);cfg=json.loads(Path('configs/raw_score_level_set.json').read_text())
    path=Path(cfg['cache_reference'])/'calibration.pt';assert digest(path)==cfg['calibration_sha256'];cache=torch.load(path,map_location='cpu')
    _,expert,head,hashes=checked_models('MNIST');p0=extended_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in p0.items()}
    assert head.bias is None or bool(torch.all(head.bias==0))
    contrast=p['head']-p['head'].mean(0);uc,sc,_=np.linalg.svd(contrast.T,full_matrices=False);rank=int((sc>sc[0]*cfg['readout_relative_tolerance']).sum());uc=uc[:,:rank]
    linear=p['gamma'][:,None]*p['projection'];linear-=linear.mean(0);ul,sl,vl=np.linalg.svd(linear,full_matrices=False)
    s=(ul/sl[None,:])@(vl@uc);bias=p['beta']@p['projection'];cg=bias@uc;ca=bias-cg@uc.T;cs=s-s.mean(0)
    rows=[]
    for group in ['native','scale']:
        h=array(cache[group]['h']);sd=np.sqrt(np.mean((h-h.mean(1,keepdims=True))**2,1,keepdims=True)+p['eps']);t=h@linear+sd*bias
        a=(t/np.linalg.norm(t,axis=1,keepdims=True))@uc;mass=np.sum(a*a,1,keepdims=True);complement=np.sqrt(np.maximum(1.-mass,0));assert complement.min()>1e-8
        bound=(np.linalg.norm(a@cs.T,axis=1,keepdims=True)*np.linalg.norm(ca)/complement+np.linalg.norm(cg@cs.T))/np.sqrt(h.shape[1])
        rows.append({'group':group,'n':len(h),'contraction_bound_maximum':float(bound.max()),'contraction_bound_median':float(np.median(bound)),'teacher_class_feature_energy_fraction_median':float(np.median(mass))})
    assert max(row['contraction_bound_maximum'] for row in rows)>1.
    result={'status':'confirmed_failure_of_preregistered_sufficient_condition','source_sha256':digest(__file__),'calibration_sha256':digest(path),'checkpoint_sha256':hashes,'feature_map_condition':float(sl[0]/sl[-1]),'class_feature_rank':rank,'pure_feature_class_change_error':float(np.max(np.abs(s.T@linear-uc.T))),'records':rows,'new_weights':0,'new_model_forwards':0,'test_used':False,'partner_used':False,'limitation':'Failure of this contraction certificate is not a proof that every raw-score equation is insoluble.'}
    save_json(out/'failure_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--raw-failure',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.raw_failure:audit_raw_failure(args.output)
    else:verify(args.output)
