"""Independent bracketed-root and SVD least-squares replay of gap-jet publication."""

import argparse,json,math,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit,logsumexp
from clip_benchmark import digest
from subspace_probe import save_json
from verify_downstream_first import convert
from verify_hidden_hermite import path_jet,mlp_jet
from verify_levelset_audits import raw_scores,residual


def decode(theta,cal,cfg):
    from scipy.optimize import brentq
    from verify_hidden_hermite import score_jet
    p=cal['parameters'];basis=cal['bases']['O'];q=cal['space']['Q'];size=cfg['probe_rank']**2
    c=theta[:size].reshape(cfg['probe_rank'],cfg['probe_rank']);l=theta[size:].reshape(q.shape[1],p['B'].shape[1]);do=np.linalg.norm(p['O'])*basis['u']@c@basis['vh']
    jet=path_jet(cal['endpoints'],cal['roots'],p,do);nb=np.linalg.norm(p['B']);free=q@l
    hf=jet['h0']+nb*jet['z']@free.T;dhf=jet['dh0']+nb*jet['dz']@free.T
    linear=p['gamma'][:,None]*p['projection'];linear-=linear.mean(0);bias=p['beta']@p['projection']
    pairs=cal['pairs'];w=p['head'][pairs[:,0]]-p['head'][pairs[:,1]];normal=w@linear.T;dd=w@bias
    b=np.linalg.norm(normal,axis=1);v=normal/b[:,None];lower=b-np.abs(dd)/np.sqrt(hf.shape[1]);assert lower.min()>cfg['minimum_monotone_slope']
    # Independent bracketed scalar solve, not the Torch quadratic formula.
    shifts=[]
    for i in range(len(hf)):
        def numerator(t):
            h=hf[i]+t*v[i];return normal[i]@h+dd[i]*np.sqrt(np.mean((h-h.mean())**2)+p['eps'])
        radius=abs(numerator(0))/lower[i]+1
        shifts.append(brentq(numerator,-radius,radius,xtol=1e-12,rtol=1e-14))
    ht=hf+np.array(shifts)[:,None]*v;centered=ht-ht.mean(1,keepdims=True);sd=np.sqrt(np.mean(centered**2,1)+p['eps'])
    length=np.linalg.norm(ht@linear+sd[:,None]*bias,axis=1);nn=normal+dd[:,None]*centered/(ht.shape[1]*sd[:,None])
    teacher,td=score_jet(jet['teacher'],jet['dteacher'],p);ar=np.arange(len(pairs));teacher_slope=td[ar,pairs[:,0]]-td[ar,pairs[:,1]]
    multiplier=(teacher_slope-np.sum(nn*dhf,1)/length)/(np.sum(nn*v,1)/length);dht=dhf+multiplier[:,None]*v
    f=np.r_[jet['z'],jet['dz']];target=np.r_[ht-jet['h0'],dht-jet['dh0']]/nb
    scale=np.linalg.norm(f,axis=1,keepdims=True);f=f/scale;target=target/scale;take=cfg['canonical_path_rows'];take=take+[len(cal['roots'])+i for i in take]
    a,bt=f[take],target[take];correction=np.linalg.lstsq(a,bt-a@free.T,rcond=None)[0].T;k=free+correction
    ss,ds=score_jet(jet['h0']+nb*jet['z']@k.T,jet['dh0']+nb*jet['dz']@k.T,p)
    gap_error=float(np.abs(ss[ar,pairs[:,0]]-ss[ar,pairs[:,1]]-(teacher[ar,pairs[:,0]]-teacher[ar,pairs[:,1]])).max())
    slope_error=float(np.abs(ds[ar,pairs[:,0]]-ds[ar,pairs[:,1]]-teacher_slope).max())
    assert gap_error<cfg['score_tolerance'] and slope_error<cfg['slope_tolerance']
    return c,k,dict(F=f,target=target,condition=float(np.linalg.cond(a)),do=do,gap_error=gap_error,slope_error=slope_error,monotone_lower=float(lower.min()))


def forward(data,p,c,k,cal,factor=1.):
    basis=cal['bases']['O'];do=np.linalg.norm(cal['parameters']['O'])*basis['u']@c@basis['vh'];nb=np.linalg.norm(cal['parameters']['B'])
    r=data['r']+factor*data['o_input']@do.T;z,_=mlp_jet(r,np.zeros_like(r),p);z+=data['b_offset']
    return raw_scores(data['rounding']+r+z@p['B'].T+p['b']+factor*nb*z@k.T,p)


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);report=json.loads((out/'results.json').read_text());history=json.loads((out/'history.json').read_text())
    for name,pin in cfg['inputs'].items():assert digest(name)==pin
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256'] and digest(out/'training_point.pt')==report['training_point_sha256']
    d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'));cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));p=cal['parameters']
    old=convert(torch.load(cfg['environment_cache'],map_location='cpu'));pp=old['proxy_parameters'];data=d['data'];teacher=d['teacher'];current=np.zeros_like(d['theta'])
    c0,k0,_=decode(current,cal,cfg);np.testing.assert_allclose(forward(data,p,c0,k0,cal),teacher,atol=1e-8,rtol=1e-9)
    # Confirm natural input ordering and exact context construction metadata.
    raw=cal['cache'];rs=raw_scores(raw['h'],p);y=rs.argmax(1);other=rs.copy();other[np.arange(len(y)),y]=-np.inf;gap=rs[np.arange(len(y)),y]-other.max(1)
    core=[]
    for cls in range(len(p['head'])):
        members=np.flatnonzero(y==cls);core.extend(members[np.argsort(gap[members],kind='stable')[:cfg['contexts_per_class']]].tolist())
    order=core+[i for i in range(len(y)) if i not in set(core)]
    np.testing.assert_array_equal(data['r'][:len(y)],raw['r'][order]);np.testing.assert_array_equal(data['o_input'][:len(y)],raw['o_input'][order])
    def objective(theta):
        c,k,info=decode(theta,cal,cfg);values=[];envscores=[]
        for e in d['environments']:
            ts=e['teacher'];yy=ts.argmax(1);rows=np.arange(len(yy));other=ts.copy();other[rows,yy]=-np.inf;g=ts[rows,yy]-other.max(1)
            ss=forward(e['data'],pp,c,k,cal,.5);other=ss.copy();other[rows,yy]=-np.inf;m=(ss[rows,yy]-other.max(1))/g
            damage=expit(-m/cfg['margin_temperature']);values.extend(damage[yy==cls].mean() for cls in range(len(p['head'])));envscores.append(ss)
        tau=cfg['opponent_temperature'];v=-tau*(logsumexp(-np.array(values)/tau)-math.log(len(values)))+cfg['distance_weight']*(np.sum(c*c)+np.sum(k*k))
        return v,c,k,info,envscores
    oldvalue=objective(current)[0];records=[];maxscore=maxcoef=maxfd=maxj=0.
    for h,snap in zip(history,d['snapshots']):
        np.testing.assert_allclose(current,snap['theta'],atol=1e-12,rtol=1e-12);proposal=current+snap['step'];obj,c,k,info,env=objective(proposal)
        maxcoef=max(maxcoef,float(np.abs(k-snap['K']).max()));np.testing.assert_allclose(k,snap['K'],atol=1e-8,rtol=1e-7)
        ss=forward(data,p,c,k,cal);err=max([float(np.abs(ss-snap['source_scores']).max())]+[float(np.abs(a-b).max()) for a,b in zip(env,snap['environment_scores'])]);maxscore=max(maxscore,err);assert err<1e-8
        pred=float(snap['gradient']@snap['step']);actual=obj-oldvalue;eps=cfg['derivative_step'];plus=objective(current+eps*snap['step']);minus=objective(current-eps*snap['step']);fd=(plus[0]-minus[0])/(2*eps)
        maxfd=max(maxfd,abs(fd-pred));assert abs(fd-pred)<cfg['derivative_tolerance']
        rows=h['active_rows'];active={k:v[rows] for k,v in data.items()};t=teacher[rows];yy=t.argmax(1);mask=np.ones_like(t,dtype=bool);mask[np.arange(len(yy)),yy]=False
        j=(residual(t,forward(active,p,plus[1],plus[2],cal))-residual(t,forward(active,p,minus[1],minus[2],cal)))/(2*eps)
        je=float(np.abs(j[mask]-snap['active_Jstep']).max());maxj=max(maxj,je);assert je<cfg['derivative_tolerance']
        worst=float(np.abs(residual(teacher,ss)).max());natural=int((ss[:len(y)].argmax(1)!=teacher[:len(y)].argmax(1)).sum());dist=float(np.sqrt(np.sum(c*c)+np.sum(k*k)))
        equality=float(np.abs(info['F']@k.T-info['target']).max());linear=float(np.abs(snap['active_residual']+snap['active_Jstep']).max())
        accept=bool(dist<=.5 and worst<=cfg['construction_task_reserve'] and natural==0 and equality<cfg['linear_equality_tolerance'] and info['condition']<cfg['maximum_condition'] and info['gap_error']<cfg['score_tolerance'] and info['slope_error']<cfg['slope_tolerance'] and linear<=cfg['construction_task_reserve'] and actual>1e-12 and actual/pred>=cfg['acceptance_ratio'])
        assert accept==h['accepted'];assert abs(worst-h['proposal']['worst_task_drift'])<1e-5
        records.append(dict(trial=h['trial'],accepted=accept,distance=dist,worst_task=worst,hermite_error=equality,gap_error=info['gap_error'],slope_error=info['slope_error']))
        if accept:current=proposal;oldvalue=obj
        print(json.dumps(records[-1]),flush=True)
    np.testing.assert_allclose(current,d['theta'],atol=1e-12,rtol=1e-12);obj,c,k,info,env=objective(current);ss=forward(data,p,c,k,cal)
    np.testing.assert_allclose(ss,d['final_scores'],atol=1e-8,rtol=1e-9);counts=[]
    for e,s,want in zip(d['environments'],env,report['final']['environments']):
        yy=e['teacher'].argmax(1);changed=s.argmax(1)!=yy;assert int(changed.sum())==want['changed']
        for row in want['per_class']:assert int((changed&(yy==row['label'])).sum())==row['changed']
        counts.append(dict(name=e['name'],changed=int(changed.sum())))
    result=dict(status='passed_source_task_constraints' if report['confirmation_eligible'] else 'passed_closed_no_point',verified_trials=len(records),records=records,
        source_inputs=len(teacher),source_worst=float(np.abs(residual(teacher,ss)).max()),parameter_distance=float(np.sqrt(np.sum(c*c)+np.sum(k*k))),
        environments=counts,maximum_numpy_score_error=maxscore,maximum_numpy_coefficient_error=maxcoef,maximum_objective_fd_error=maxfd,maximum_task_Jstep_error=maxj,
        training_point_sha256=report['training_point_sha256'],diagnostic_sha256=report['diagnostic_sha256'],wall_seconds=time.monotonic()-start,source_sha256=digest(__file__),
        new_release_weights=0,independent_confirmation_passed=False,peer_used=False,test_used=False,original_expert_updates=0)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();torch.set_num_threads(4);verify(json.loads(Path(a.protocol).read_text()))
