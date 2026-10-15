"""Independent NumPy finite/derivative verification of the hidden Hermite chart."""

import argparse
import json
import os
import time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores,residual


def norm_jet(x,dx,gamma,beta,eps):
    c=x-x.mean(1,keepdims=True);dc=dx-dx.mean(1,keepdims=True)
    sd=np.sqrt(np.mean(c*c,1,keepdims=True)+eps);ds=np.mean(c*dc,1,keepdims=True)/sd
    return c/sd*gamma+beta,(dc/sd-c*ds/(sd*sd))*gamma


def mlp_jet(r,dr,p):
    x,dx=norm_jet(r,dr,p['gamma2'],p['beta2'],p['eps2']);a=x@p['A'].T+p['a'];da=dx@p['A'].T
    sig=expit(1.702*a);z=a*sig;dz=(sig+1.702*a*sig*(1-sig))*da
    return z,dz


def score_jet(h,dh,p):
    a,da=norm_jet(h,dh,p['gamma'],p['beta'],p['eps']);f=a@p['projection'];df=da@p['projection']
    length=np.linalg.norm(f,axis=1,keepdims=True);unit=f/length;der=(df-unit*np.sum(unit*df,1,keepdims=True))/length
    return unit@p['head'].T,der@p['head'].T


def path_jet(end,t,p,do):
    raw={k:end[k][:,0]+t[:,None]*(end[k][:,1]-end[k][:,0]) for k in end}
    dr={k:end[k][:,1]-end[k][:,0] for k in end}
    z0,dz0=mlp_jet(raw['r'],dr['r'],p)
    teacher=raw['rounding']+raw['r']+z0@p['B'].T+p['b'];dteacher=dr['rounding']+dr['r']+dz0@p['B'].T
    r=raw['r']+raw['o_input']@do.T;rdot=dr['r']+dr['o_input']@do.T;z,dz=mlp_jet(r,rdot,p)
    h0=raw['rounding']+r+z@p['B'].T+p['b'];dh0=dr['rounding']+rdot+dz@p['B'].T
    return dict(z=z,dz=dz,teacher=teacher,dteacher=dteacher,h0=h0,dh0=dh0)


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'mechanism.json').read_text())
    assert cfg['completion_target']=='hidden' and digest(cfg['calibration_reference'])==cfg['calibration_sha256']==rep['calibration_sha256']
    assert digest(out/'mechanism.pt')==rep['mechanism_sha256']
    cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));saved=convert(torch.load(out/'mechanism.pt',map_location='cpu'))
    p=cal['parameters'];basis=cal['bases']['O'];end=cal['endpoints'];roots=cal['roots'];pairs=cal['pairs'];rows=np.arange(len(roots))
    # No held-out values are loaded: only registered historical index metadata
    # is used to establish disjointness of this new construction draw.
    split_path=Path(cfg['calibration_reference']).parent/'construction_split.json';split=json.loads(split_path.read_text());ids=set(cal['indices']);excluded=set()
    for path,pin in split['excluded_metadata_sha256'].items():
        assert digest(path)==pin,path
        if not path.endswith('.json'):continue
        record=json.loads(Path(path).read_text())
        if 'confirmation' in Path(path).name:excluded.update(record.get('indices',[]))
    assert not ids&excluded
    c=-cfg['mechanism_O_distance']*np.diag(basis['s'])/np.linalg.norm(basis['s']);do=np.linalg.norm(p['O'])*basis['u']@c@basis['vh']
    jet=path_jet(end,roots,p,do);nb=np.linalg.norm(p['B']);teacher,_=score_jet(jet['teacher'],jet['dteacher'],p)
    root_gap=float(np.abs(teacher[rows,pairs[:,0]]-teacher[rows,pairs[:,1]]).max());assert root_gap<cfg['root_tolerance']
    records=[];maxscore=0.;maxcoef=0.;maxslope=0.
    for item in rep['records']:
        name=item['method'];actual=saved['records'][name];k=actual['K'];np.testing.assert_allclose(actual['C'],c,atol=1e-13,rtol=1e-13)
        condition=None
        if name!='uncompensated':
            f=jet['z'] if name=='value_only' else np.r_[jet['z'],jet['dz']]
            target=(jet['teacher']-jet['h0'])/nb
            if name=='value_and_derivative':target=np.r_[target,(jet['dteacher']-jet['dh0'])/nb]
            scale=np.linalg.norm(f,axis=1,keepdims=True);f=f/scale;target=target/scale
            # Independent rank-revealing least squares retains ALL original
            # equations, including reversed copies of the same chord.
            expected=np.linalg.lstsq(f,target,rcond=None)[0].T;err=float(np.abs(expected-k).max());maxcoef=max(maxcoef,err)
            np.testing.assert_allclose(expected,k,atol=1e-8,rtol=1e-7)
            equality=float(np.abs(f@k.T-target).max());assert equality<cfg['linear_equality_tolerance']
            take=cfg.get('canonical_path_rows')
            if take is not None:
                take=take+[len(roots)+i for i in take] if name=='value_and_derivative' else take
                condition=float(np.linalg.cond(f[take]))
            else:condition=float(np.linalg.cond(f))
        after,der=score_jet(jet['h0']+nb*jet['z']@k.T,jet['dh0']+nb*jet['dz']@k.T,p)
        _,td=score_jet(jet['teacher'],jet['dteacher'],p);value=after-teacher;value-=value.mean(1,keepdims=True);slope=der-td;slope-=slope.mean(1,keepdims=True)
        err=float(np.abs(value-actual['root_value']).max());derr=float(np.abs(slope-actual['root_slope']).max());maxscore=max(maxscore,err);maxslope=max(maxslope,derr)
        assert err<1e-8 and derr<1e-7
        finite=[]
        for want in actual['finite']:
            t=roots+want['sign']*(2.**-want['exponent'])*np.minimum(roots,1-roots);j=path_jet(end,t,p,do)
            ts=raw_scores(j['teacher'],p);ss=raw_scores(j['h0']+nb*j['z']@k.T,p)
            gap=ts[rows,pairs[:,0]]-ts[rows,pairs[:,1]];change=(ss-ts)[rows,pairs[:,0]]-(ss-ts)[rows,pairs[:,1]]
            np.testing.assert_allclose(gap,want['pair_gap'],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(change,want['pair_gap_change'],atol=1e-9,rtol=1e-9)
            finite.append(float((np.abs(change)/np.abs(gap)).max()))
        raw=cal['cache'];z0,_=mlp_jet(raw['r'],np.zeros_like(raw['r']),p);rounding=raw['h']-raw['r']-z0@p['B'].T-p['b']
        r=raw['r']+raw['o_input']@do.T;z,_=mlp_jet(r,np.zeros_like(r),p)
        ss=raw_scores(rounding+r+z@(p['B']+nb*k).T+p['b'],p);before=raw_scores(raw['h'],p)
        error=float(np.abs(ss-actual['natural_scores']).max());maxscore=max(maxscore,error);assert error<1e-8
        worst=float(np.abs(residual(before,ss)).max());changed=int((before.argmax(1)!=ss.argmax(1)).sum())
        assert abs(worst-item['natural_worst_task'])<1e-5 and changed==item['natural_changed']
        distance=float(np.sqrt(np.sum(c*c)+np.sum(k*k)));assert abs(distance-item['parameter_distance'])<1e-8
        records.append(dict(method=name,distance=distance,natural_worst_task=worst,natural_changed=changed,root_score_error=float(np.abs(value).max()),root_derivative_error=float(np.abs(slope).max()),finite_pair_relative_errors=finite,constraint_condition=condition))
    near=max(records[-1]['finite_pair_relative_errors'][-2:]);far=max(records[-1]['finite_pair_relative_errors'][:2]);actual_support=bool(records[-1]['root_score_error']<cfg['score_tolerance'] and records[-1]['root_derivative_error']<cfg['derivative_tolerance'] and near<=far/cfg['minimum_relative_decay'] and records[-1]['constraint_condition']<cfg['maximum_condition'])
    assert actual_support==rep['mechanism_supported']
    result=dict(status='passed_mechanism' if actual_support else 'passed_closed_mechanism',records=records,teacher_root_gap_maximum=root_gap,
        maximum_numpy_score_error=maxscore,maximum_numpy_derivative_error=maxslope,maximum_numpy_coefficient_error=maxcoef,
        construction_disjoint_from_json_confirmations=True,excluded_confirmation_indices=len(excluded),construction_n=len(ids),
        mechanism_sha256=rep['mechanism_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        new_release_weights=0,confirmation_run=False,true_TA_run=False,AGA_run=False,test_used=False,peer_used=False,
        limitation='Exact hidden interpolation and analytic derivative replay on fixed construction paths. No transverse-input or unknown-peer guarantee; fixed mechanism seed is not a confirmed publication candidate.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();torch.set_num_threads(4);verify(json.loads(Path(args.protocol).read_text()))
