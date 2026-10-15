"""Independent finite source/opponent replay of the robust publication path."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit,logsumexp,softmax

from clip_benchmark import digest
from subspace_probe import save_json
from verify_cross_module import ln
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores,residual


def pack_source(raw,p,point):
    r,o,h=[raw[k] for k in ['r','o_input','h']];v=point['basis_O']['vh'];vb=point['basis_B']['vh']
    def hidden(x):
        pre=ln(x,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];return pre*expit(1.702*pre)
    rounding=h-r-hidden(r)@p['B'].T-p['b'];n=point['protocol']['probe_contexts'];count=2*len(v)
    step=float(np.float32(point['protocol']['finite_step']))
    so=(step*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((o@v.T)**2,0))[None,:,None]*v[None]).reshape(count,-1)
    sb=(step*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((hidden(r)@vb.T)**2,0))[None,:,None]*vb[None]).reshape(count,-1)
    rr=np.concatenate([r,(r[:n,None]+(so@p['O'].T)[None]).reshape(-1,r.shape[1]),np.repeat(r[:n],count,axis=0)])
    oo=np.concatenate([o,(o[:n,None]+so[None]).reshape(-1,o.shape[1]),np.repeat(o[:n],count,axis=0)])
    cc=np.concatenate([rounding,np.repeat(rounding[:n],count,axis=0),np.repeat(rounding[:n],count,axis=0)])
    off=np.concatenate([np.zeros((len(r)+n*count,p['B'].shape[1])),np.tile(sb,(n,1))])
    return dict(r=rr,o_input=oo,rounding=cc,b_offset=off)


def forward(data,p,theta,point,factor=1.):
    r=len(point['basis_O']['vh']);size=r*r;q=point['output_basis']
    norm=point.get('_normalization',{'O':np.linalg.norm(p['O']),'B':np.linalg.norm(p['B'])})
    do=norm['O']*point['basis_O']['u']@theta[:size].reshape(r,r)@point['basis_O']['vh']
    db=norm['B']*q@theta[size:].reshape(len(q.T),p['B'].shape[1])
    rr=data['r']+factor*data['o_input']@do.T
    pre=ln(rr,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];z=pre*expit(1.702*pre)+data['b_offset']
    return raw_scores(data['rounding']+rr+z@(p['B']+factor*db).T+p['b'],p)


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);report=json.loads((out/'results.json').read_text());hist=json.loads((out/'history.json').read_text())
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256'] and digest(out/'training_point.pt')==report['training_point_sha256']
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    saved=convert(torch.load(out/'diagnostic.pt',map_location='cpu'));point=convert(torch.load(cfg['initial_point'],map_location='cpu'))
    cal=convert(torch.load(cfg['cache'],map_location='cpu'));p,pp=cal['parameters'],cal['proxy_parameters']
    point['_normalization']={'O':np.linalg.norm(p['O']),'B':np.linalg.norm(p['B'])}
    packs=[pack_source(point[g],p,point) for g in ['native','scale']];source={k:np.concatenate([g[k] for g in packs]) for k in packs[0]}
    natural=np.r_[np.arange(len(point['native']['r'])),len(packs[0]['r'])+np.arange(len(point['scale']['r']))]
    zero=np.zeros_like(saved['theta']);teacher=forward(source,p,zero,point)
    np.testing.assert_allclose(teacher,saved['source_teacher'],atol=1e-8,rtol=1e-9)
    # Verify both environment inputs by independent full attention. They are
    # common-prefix parameter changes, never independent per-image vectors.
    ev=convert(torch.load(Path(cfg['environment_root'])/'diagnostic.pt',map_location='cpu'))['theta'].reshape(cfg['probe_rank'],cfg['probe_rank'])
    for index,env in enumerate(saved['environments']):
        s=np.zeros_like(ev) if index==0 else ev;raw=cal['proxy'];basis=cal['basis']
        h=raw['H'].astype(np.float64)+np.linalg.norm(p['D'])*(raw['t']@s.T)@basis['u'].T
        n,t,d=h.shape;heads=pp['heads'];width=d//heads;x=ln(h,pp['gamma1'],pp['beta1'],pp['eps1'])
        wq,wk,wv=np.split(pp['W'],3);bq,bk,bv=np.split(pp['w'],3)
        query=(x[:,0]@wq.T+bq).reshape(n,heads,width)
        key=(x@wk.T+bk).reshape(n,t,heads,width).transpose(0,2,1,3);value=(x@wv.T+bv).reshape(n,t,heads,width).transpose(0,2,1,3)
        prob=softmax(np.einsum('nhd,nhtd->nht',query,key)/np.sqrt(width),axis=-1)
        o=np.einsum('nht,nhtd->nhd',prob,value).reshape(n,d);r=h[:,0]+o@pp['O'].T+pp['ob']
        np.testing.assert_allclose(r,env['data']['r'],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(o,env['data']['o_input'],atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(forward(env['data'],pp,zero,point,.5),env['data']['teacher'],atol=1e-9,rtol=1e-9)
    def objective(theta):
        scores=[];values=[]
        for env in saved['environments']:
            ss=forward(env['data'],pp,theta,point,.5);teacher=env['data']['teacher'];y=teacher.argmax(1);row=np.arange(len(y))
            other=teacher.copy();other[row,y]=-np.inf;gap=teacher[row,y]-other.max(1)
            other=ss.copy();other[row,y]=-np.inf;m=(ss[row,y]-other.max(1))/gap;damage=expit(-m/cfg['margin_temperature'])
            values.extend(damage[y==c].mean() for c in range(ss.shape[1]));scores.append(ss)
        a=np.array(values);tau=cfg['opponent_temperature'];value=-tau*(logsumexp(-a/tau)-math.log(len(a)))+cfg['distance_weight']*np.sum(theta*theta)
        return value,scores
    current=saved['initial_theta'];old=objective(current)[0];maxerr=fdmax=0.;records=[]
    for entry,snap in zip(hist,saved['snapshots']):
        np.testing.assert_allclose(current,snap['theta'],atol=1e-12,rtol=1e-12)
        proposed=current+snap['step'];ss=forward(source,p,proposed,point);obj,envs=objective(proposed)
        err=max([float(np.max(np.abs(ss-snap['source'])))]+[float(np.max(np.abs(a-b))) for a,b in zip(envs,snap['environment_scores'])]);maxerr=max(maxerr,err);assert err<1e-8
        pred=float(snap['gradient']@snap['step']);actual=obj-old;eps=cfg['derivative_step']
        fd=(objective(current+eps*snap['step'])[0]-objective(current-eps*snap['step'])[0])/(2*eps)
        fdmax=max(fdmax,abs(fd-pred));assert abs(fd-pred)<cfg['derivative_tolerance']
        rows=np.array(entry['active_rows']);active={k:v[rows] for k,v in source.items()};before=teacher[rows]
        rr=residual(before,forward(active,p,current,point));jp=(residual(before,forward(active,p,current+eps*snap['step'],point))-residual(before,forward(active,p,current-eps*snap['step'],point)))/(2*eps)
        keep=np.ones_like(before,dtype=bool);keep[np.arange(len(rows)),before.argmax(1)]=False
        np.testing.assert_allclose(rr[keep],snap['active_r'],atol=1e-7,rtol=1e-7);np.testing.assert_allclose(jp[keep],snap['active_Jstep'],atol=1e-5,rtol=1e-5)
        worst=float(np.abs(residual(teacher,ss)).max());changes=int((ss[natural].argmax(1)!=teacher[natural].argmax(1)).sum());dist=float(np.linalg.norm(proposed));linear=float(np.abs(snap['active_r']+snap['active_Jstep']).max())
        accept=bool(dist<=.5 and worst<=.5 and changes==0 and linear<=.5 and actual>1e-12 and actual/pred>=cfg['acceptance_ratio'])
        assert accept==entry['accepted'];assert abs(worst-entry['proposal']['worst_task_drift'])<1e-6
        if accept:current=proposed;old=obj
        records.append(dict(trial=entry['trial'],accepted=accept,source_worst=worst,distance=dist));print(json.dumps(records[-1]),flush=True)
    np.testing.assert_allclose(current,saved['theta'],atol=1e-12,rtol=1e-12)
    final=forward(source,p,current,point);np.testing.assert_allclose(final,saved['final_source'],atol=1e-8,rtol=1e-9)
    obj,envs=objective(current);counts=[]
    for expected,env,ss in zip(report['final']['environments'],saved['environments'],envs):
        y=env['data']['teacher'].argmax(1);ch=ss.argmax(1)!=y
        assert int(ch.sum())==expected['changed']
        for c,row in enumerate(expected['per_class']):assert int((ch&(y==c)).sum())==row['changed']
        counts.append(dict(name=env['name'],changed=int(ch.sum()),classes_changed=int(sum(np.any(ch&(y==c)) for c in range(ss.shape[1])))))
    signal=min(x['changed'] for x in counts)>=cfg['minimum_changed'] and min(x['classes_changed'] for x in counts)>=cfg['minimum_changed_classes'] and obj>report['initial']['objective']
    assert bool(signal)==report['confirmation_eligible'];assert np.linalg.norm(current)<=.5
    result=dict(status='passed_task_calibration' if signal else 'passed_closed_protocol',source_inputs=len(teacher),source_worst_task=float(np.abs(residual(teacher,final)).max()),
        parameter_distance=float(np.linalg.norm(current)),environments=counts,confirmation_eligible=bool(signal),verified_trials=len(records),records=records,
        maximum_numpy_score_error=maxerr,maximum_objective_fd_error=fdmax,all_trial_acceptances_verified=True,
        point_sha256=report['training_point_sha256'],diagnostic_sha256=report['diagnostic_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        peer_used=False,test_used=False,new_release_weights=0,new_full_model_forwards=0)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();torch.set_num_threads(4)
    verify(json.loads(Path(args.protocol).read_text()))
