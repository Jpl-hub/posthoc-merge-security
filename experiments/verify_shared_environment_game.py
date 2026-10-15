"""NumPy full finite replay of the common-prefix environment path."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit, logsumexp

from clip_benchmark import digest
from subspace_probe import save_json
from verify_cascade import packed_native, suffix
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores, residual


def forward(packed,p,theta,basis,nd):
    take=packed['take'];cache=packed['cache'];r=len(basis['vh']);s=theta.reshape(r,r)
    h=cache['H'][take].astype(np.float64).copy();t=cache['t'][take].astype(np.float64).copy()
    h[:,0]+=packed['shift']@packed['up_map'];t[:,0]+=packed['shift']
    h+=nd*(t@s.T)@basis['u'].T
    out,_=suffix(h,p)
    return raw_scores(out+cache['rounding'][take]+packed['offset']@p['B'].T,p)


def subset(packed,take):
    return {**packed,**{key:packed[key][take] for key in ['take','shift','offset']}}


def batches(packed,p,theta,basis,nd,size):
    return np.concatenate([forward(subset(packed,slice(i,i+size)),p,theta,basis,nd)
                           for i in range(0,len(packed['take']),size)])


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);report=json.loads((out/'results.json').read_text())
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    saved=convert(torch.load(out/'diagnostic.pt',map_location='cpu'))
    cal=torch.load(Path(cfg['cache_output'])/'calibration.pt',map_location='cpu')
    p,pp,basis=map(convert,[cal['parameters'],cal['proxy_parameters'],cal['basis']]);nd=np.linalg.norm(p['D'])
    # Legacy frozen probes form their +/- finite_step factor in float32,
    # before multiplying source RMS in double. Match those input semantics.
    pcfg=cfg|{'finite_step':float(np.float32(cfg['finite_step']))}
    native,natural_n,ids=packed_native(cal,p,basis,pcfg)
    cache=convert(cal['proxy']);n=len(cache['H']);rank=cfg['probe_rank']
    proxy=dict(cache=cache,up_map=basis['vh']@pp['D'].T,take=np.arange(n),shift=np.zeros((n,rank)),offset=np.zeros((n,p['B'].shape[1])))
    point=convert(torch.load(cfg['protected_point'],map_location='cpu'));v=point['basis_O']['vh'];r=len(v);th=point['theta'];q=point['output_basis']
    delta={'O':np.linalg.norm(p['O'])*point['basis_O']['u']@th[:r*r].reshape(r,r)@v,
           'B':np.linalg.norm(p['B'])*q@th[r*r:].reshape(len(q.T),p['B'].shape[1])}
    mod=pp|{key:pp[key]+cfg['coefficient']*value for key,value in delta.items()}
    zero=np.zeros(rank*rank);batch=cfg['verification_batch_size']
    nt=batches(native,p,zero,basis,nd,batch);bt=batches(proxy,pp,zero,basis,nd,batch)
    initial=batches(proxy,mod,zero,basis,nd,batch)
    for actual,want in [(nt,saved['source_teacher']),(bt,saved['reference_teacher']),(initial,saved['initial_protected'])]:
        np.testing.assert_allclose(actual,want,atol=1e-8,rtol=1e-9)
    y=bt.argmax(1);row=np.arange(len(y));other=bt.copy();other[row,y]=-np.inf;gap=bt[row,y]-other.max(1)
    initial_changed=initial.argmax(1)!=y
    def objective(a,with_gradient=False):
        other=a.copy();other[row,y]=-np.inf;k=other.argmax(1);m=(a[row,y]-other[row,k])/gap
        temp=cfg['margin_temperature'];loss=temp*np.logaddexp(0,-m/temp)
        counts=np.bincount(y,minlength=a.shape[1]);ac=np.array([loss[y==c].mean() for c in range(a.shape[1])]);tau=cfg['class_temperature']
        value=-tau*(logsumexp(ac/tau)-math.log(len(ac)))
        if not with_gradient:return value
        pi=np.exp(ac/tau-logsumexp(ac/tau));weights=pi[y]*expit(-m/temp)/(counts[y]*gap)
        g=np.zeros_like(a);g[row,y]=weights;g[row,k]-=weights
        return value,g
    current=zero.copy();old_value=objective(initial);history=json.loads((out/'history.json').read_text())
    score_error=fd_error=class_gradient_error=0.;records=[]
    assert len(history)==len(saved['snapshots'])==report['trials']
    for entry,snap in zip(history,saved['snapshots']):
        np.testing.assert_allclose(current,snap['theta'],atol=1e-12,rtol=1e-12)
        step=snap['step'];proposal=current+step
        source=batches(native,p,proposal,basis,nd,batch)
        baseline=batches(proxy,pp,proposal,basis,nd,batch)
        protected=batches(proxy,mod,proposal,basis,nd,batch)
        err=max(float(np.max(np.abs(actual-snap[key]))) for actual,key in [(source,'source'),(baseline,'baseline'),(protected,'protected')])
        score_error=max(score_error,err);assert err<1e-8
        before=batches(proxy,mod,current,basis,nd,batch);value,grad=objective(before,True)
        ge=float(np.linalg.norm(grad-snap['score_gradient'])/max(np.linalg.norm(grad),1e-30));class_gradient_error=max(class_gradient_error,ge);assert ge<1e-7
        pred=float(snap['gradient']@step);actual=float(objective(protected)-old_value)
        eps=cfg['derivative_step'];plus=objective(batches(proxy,mod,current+eps*step,basis,nd,batch));minus=objective(batches(proxy,mod,current-eps*step,basis,nd,batch))
        fd=(plus-minus)/(2*eps);fe=abs(fd-pred);fd_error=max(fd_error,fe);assert fe<cfg['derivative_tolerance']
        # Independently check each active linear prediction by actual NumPy
        # finite differences, with the same frozen competitor normalization.
        active_res=[];active_js=[]
        for where in entry['locations']:
            pack,pars,teacher=(native,p,nt) if where['group']=='source' else (proxy,pp,bt)
            rows=np.asarray(where['rows']);sel=subset(pack,rows);teacher=teacher[rows]
            now=batches(sel,pars,current,basis,nd,batch)
            plus=batches(sel,pars,current+eps*step,basis,nd,batch);minus=batches(sel,pars,current-eps*step,basis,nd,batch)
            rr=residual(teacher,now);js=(residual(teacher,plus)-residual(teacher,minus))/(2*eps)
            keep=np.ones_like(teacher,dtype=bool);keep[np.arange(len(rows)),teacher.argmax(1)]=False
            active_res.append(rr[keep]);active_js.append(js[keep])
        r,jstep=np.concatenate(active_res),np.concatenate(active_js)
        np.testing.assert_allclose(r,snap['active_residual'],atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(jstep,snap['active_Jstep'],atol=1e-5,rtol=1e-5)
        linear=float(np.max(np.abs(snap['active_residual']+snap['active_Jstep'])))
        worst=float(np.abs(residual(nt,source)).max());refworst=float(np.abs(residual(bt,baseline)).max());dist=float(np.linalg.norm(proposal))
        changes=int((source[:natural_n].argmax(1)!=nt[:natural_n].argmax(1)).sum());bchanges=int((baseline.argmax(1)!=y).sum());ratio=actual/pred
        ok=bool(dist<=cfg['parameter_budget'] and worst<=cfg['task_budget'] and refworst<=cfg['task_budget']
                and changes==0 and bchanges==0 and linear<=cfg['task_budget'] and actual>1e-12 and ratio>=cfg['acceptance_ratio'])
        assert ok==entry['accepted'],(entry['trial'],ok,entry['accepted'])
        assert abs(worst-entry['proposal']['source_worst_task'])<1e-6 and abs(refworst-entry['proposal']['reference_worst_task'])<1e-6
        if ok:current=proposal;old_value=objective(protected)
        records.append(dict(trial=entry['trial'],accepted=ok,distance=dist,source_worst=worst,reference_worst=refworst,agreement_ratio=ratio))
        print(json.dumps(records[-1]),flush=True)
    np.testing.assert_allclose(current,saved['theta'],atol=1e-12,rtol=1e-12)
    source=batches(native,p,current,basis,nd,batch);baseline=batches(proxy,pp,current,basis,nd,batch);protected=batches(proxy,mod,current,basis,nd,batch)
    for actual,key in [(source,'final_source'),(baseline,'final_baseline'),(protected,'final_protected')]:np.testing.assert_allclose(actual,saved[key],atol=1e-8,rtol=1e-9)
    restored=initial_changed&(protected.argmax(1)==y);classes=sum(np.any(restored&(y==c)) for c in range(protected.shape[1]))
    for c,item in enumerate(report['final']['per_class']):
        mask=y==c;assert int(mask.sum())==item['n'] and int((restored&mask).sum())==item['restored']
    signal=int(restored.sum())>=cfg['minimum_restored'] and classes>=cfg['minimum_restored_classes']
    assert bool(signal)==report['finite_multiclass_signal']
    result=dict(status='passed',source_all_task_inputs=len(nt),source_natural_n=natural_n,reference_n=n,
        initially_changed=int(initial_changed.sum()),prediction_restored=int(restored.sum()),restored_classes=int(classes),
        finite_multiclass_signal=bool(signal),parameter_distance=float(np.linalg.norm(current)),
        source_worst_task=float(np.abs(residual(nt,source)).max()),reference_worst_task=float(np.abs(residual(bt,baseline)).max()),
        all_trial_acceptances_verified=True,verified_trials=len(records),records=records,
        maximum_numpy_score_error=score_error,maximum_numpy_objective_fd_error=fd_error,
        maximum_class_game_gradient_relative_error=class_gradient_error,diagnostic_sha256=report['diagnostic_sha256'],
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        new_publication_weights=0,peer_used=False,test_used=False,new_full_model_forwards=0,
        scope='Independent NumPy all-token attention, full9728 source probes and512 reference rows at every proposal, finite task budgets, accept/reject decisions, class opponent, shared-gradient directional derivatives, all active linear responses and actual multiclass restoration. Not unknown-peer coverage or an efficacy result.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    torch.set_num_threads(4);verify(json.loads(Path(args.protocol).read_text()))
