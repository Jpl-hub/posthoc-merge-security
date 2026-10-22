"""One finite minimax projection/bias completion; frozen teacher, no image fitting loop."""
import argparse
import json
import time
from pathlib import Path

import clarabel
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from clip_benchmark import digest
from subspace_probe import save_json
from task_probe_distance import checked_models, initialize
from verify_covariance_noise import official_labels


def normalized_input(h, p):
    h=np.asarray(h);c=h-h.mean(1,keepdims=True)
    return c/np.sqrt(np.mean(c*c,axis=1,keepdims=True)+p['eps'])*p['gamma']+p['beta']


def feature_slabs(f, v, head, bound, epsilon):
    """Connected normalized-feature gap band containing zero; quadratic events."""
    scores=f@head.T;y=scores.argmax(1);rows=np.arange(len(f))
    keep=np.arange(len(head))[None,:]!=y[:,None]
    gaps=(scores[rows,y,None]-scores)[keep].reshape(len(f),-1)
    hv=head@v;slopes=(hv[y,None]-hv[None,:])[keep].reshape(len(f),-1)
    cross=f@v;lo=-bound.copy();hi=bound.copy();error=0.
    for factor in [1-epsilon,1+epsilon]:
        aa=slopes**2-factor**2*gaps**2
        bb=2*(gaps*slopes-factor**2*gaps**2*cross[:,None])
        cc=(1-factor**2)*gaps**2
        for i in range(len(f)):
            for a,b,c in zip(aa[i],bb[i],cc[i]):
                coefficients=[c,b,a] if abs(a)>1e-14*(abs(b)+abs(c)) else [c,b]
                for root in np.polynomial.polynomial.polyroots(coefficients):
                    if abs(root.imag)>1e-10*max(1.,abs(root.real)):continue
                    r=float(root.real)
                    error=max(error,abs(a*r*r+b*r+c)/(abs(a)*r*r+abs(b*r)+abs(c)))
                    if r>0:hi[i]=min(hi[i],r)
                    if r<0:lo[i]=max(lo[i],r)
    assert error<1e-6 and np.all(lo<0) and np.all(hi>0)
    worst=0.
    for weight in np.linspace(0,1,17):
        s=.99*((1-weight)*lo+weight*hi)
        vv=f+s[:,None]*v;sc=vv/np.linalg.norm(vv,axis=1,keepdims=True)@head.T
        ratios=((sc[rows,y,None]-sc-scores[rows,y,None]+scores)[keep].reshape(len(f),-1))/gaps
        worst=max(worst,float(np.abs(ratios).max()))
    assert worst<epsilon+1e-6,worst
    return .99*lo,.99*hi,dict(root_backward_error=error,fixed17_grid_R=worst,formal_interval_arithmetic=False)


@torch.no_grad()
def cached_scores(h, p, head):
    result=[];head.cuda().eval()
    g=torch.tensor(p['gamma'],device='cuda',dtype=torch.float32)
    b=torch.tensor(p['beta'],device='cuda',dtype=torch.float32)
    P=torch.tensor(p['projection'],device='cuda',dtype=torch.float32)
    try:
        for batch in h.split(64):result.append(head(F.layer_norm(batch.float().cuda(),(768,),g,b,p['eps'])@P).cpu().double())
    finally:head.cpu()
    return torch.cat(result).numpy()


def metrics(before, after, labels=None):
    y=before.argmax(1);rows=np.arange(len(y));keep=np.arange(before.shape[1])[None,:]!=y[:,None]
    gap=(before[rows,y,None]-before)[keep]
    assert np.all(gap>0)
    result=dict(n=len(y),changed=int(np.sum(after.argmax(1)!=y)),R=float(np.max(np.abs((after[rows,y,None]-after-before[rows,y,None]+before)[keep]/gap))))
    if labels is not None:result.update(before_accuracy=float(np.mean(y==labels)),after_accuracy=float(np.mean(after.argmax(1)==labels)))
    return result


@torch.no_grad()
def run(protocol):
    initialize();start=time.monotonic();cfg=json.loads(Path(protocol).read_text())
    assert clarabel.__version__==cfg['solver_version']
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    cal=torch.load(cfg['calibration'],map_location='cpu',weights_only=False)
    assert cal['indices']==json.loads(Path(cfg['source_draw']).read_text())['indices'][:len(cal['indices'])]
    own=cal['cfg'];p,pf=cal['parameters'],cal['proxy_parameters'];H=p['head'];P=p['projection'];beta=p['beta']
    anchor,expert,head,pins=checked_models(own['target']);assert pins==cal['checkpoint_sha256']
    assert head.normalize and torch.count_nonzero(head.bias)==0
    np.testing.assert_array_equal(P,expert.model.visual.proj.double().numpy())
    np.testing.assert_array_equal(H,head.weight.double().numpy())
    U,_,Vh=np.linalg.svd(P-anchor.model.visual.proj.double().numpy(),full_matrices=False)
    n=1/p['gamma'];n/=np.linalg.norm(n)
    Q,s,_=np.linalg.svd(np.column_stack([P,U[:,:own['task_rank']],n]),full_matrices=True)
    N=Q[:,np.sum(s>s[0]*1e-10):];r=N.shape[1]
    a=H[own['collapse_class']]-H.mean(0);v=a-Vh[:own['task_rank']].T@(Vh[:own['task_rank']]@a);v/=np.linalg.norm(v)
    w=P@np.linalg.solve(P.T@P,v)
    z=normalized_input(cal['h'].numpy(),p);zf=normalized_input(cal['h_proxy'].numpy(),pf);mu=z.mean(0);m=mu@N
    G=np.eye(r)/np.linalg.norm(P)**2+(np.linalg.norm(w)/np.linalg.norm(beta))**2*np.outer(m,m)
    eigen,axes=np.linalg.eigh(G);S=(axes/np.sqrt(eigen))@axes.T
    teacher=cal['original_scores'].argmax(1).numpy();take=[np.flatnonzero(teacher==i)[0] for i in range(len(H))]
    basis=U[:,:own['task_rank']];rms=np.sqrt(np.mean((z@basis)**2,axis=0))
    shifts=np.concatenate([-own['finite_step']*rms[:,None]*basis.T,own['finite_step']*rms[:,None]*basis.T])
    zp=np.repeat(z[take],len(shifts),axis=0)+np.tile(shifts,(len(H),1));allz=np.concatenate([z,zp])
    features=allz@P;norm=np.linalg.norm(features,axis=1);T=((allz-mu)@N@S)/norm[:,None]
    radius=own['parameter_budget']*(1-1e-6)
    lo,hi,control=feature_slabs(features/norm[:,None],v,H,radius*np.linalg.norm(T,axis=1),own['source_reserve'])
    peers=np.arange(len(H))!=own['collapse_class'];contrast=H[own['collapse_class']]-H[peers]
    ef=zf@pf['projection'];enorm=np.linalg.norm(ef,axis=1)
    B=ef@contrast.T/enorm[:,None]
    C=own['coefficient']*((zf@N)[:,None,:]*(contrast@v)[None,:,None]-(contrast@(w@pf['projection']))[None,:,None]*m)
    C=(C@S)/enorm[:,None,None];C=C.reshape(-1,r)
    Alin=np.column_stack([np.vstack([T,-T,-C]),np.r_[np.zeros(2*len(T)),np.ones(len(C))]])
    blin=np.r_[hi,-lo,B.ravel()]
    Asoc=np.zeros((r+1,r+1));Asoc[1:,:r]=-np.eye(r)
    A=sp.csc_matrix(np.vstack([Alin,Asoc]));b=np.r_[blin,radius,np.zeros(r)];objective=np.r_[np.zeros(r),-1.]
    feasible=np.r_[np.zeros(r),B.min()-1]
    assert np.min(blin-Alin@feasible)>0
    settings=clarabel.DefaultSettings();settings.verbose=True;settings.max_iter=cfg['max_iterations']
    settings.time_limit=cfg['time_limit'];settings.direct_solve_method=cfg['solver_backend'];settings.max_threads=4
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=cfg['solver_tolerance']
    solver=clarabel.DefaultSolver(sp.csc_matrix((r+1,r+1)),objective,A,b,[clarabel.NonnegativeConeT(len(blin)),clarabel.SecondOrderConeT(r+1)],settings)
    sol=solver.solve();x=np.asarray(sol.x);dual=np.asarray(sol.z);slack=b-A@x
    np.savez_compressed(out/'conic_problem.npz',data=A.data,indices=A.indices,indptr=A.indptr,shape=A.shape,b=b,
        objective=objective,primal=x,dual=dual,nonnegative_rows=len(blin),N=N,whitening=S,v=v,w=w,mean=mu)
    record=dict(status=str(sol.status),variables=r+1,linear_constraints=len(blin),second_order_cones=1,
        margin=float(x[-1]),primal_objective=float(sol.obj_val),dual_objective=float(sol.obj_val_dual),iterations=int(sol.iterations),
        solver_seconds=float(sol.solve_time),source_sha256=digest(__file__),protocol_sha256=digest(protocol),inputs=cfg['inputs'],
        slab_numerical_control=control,new_full_image_forwards=0,new_AGA_calls=0,expert_training_updates=0)
    if str(sol.status)!='Solved':
        record.update(status='closed_solver_not_solved',solver_status=str(sol.status),wall_seconds=time.monotonic()-start)
        save_json(out/'results.json',record);print(json.dumps(record),flush=True);return
    stationarity=float(np.max(np.abs(A.T@dual+objective)))
    dual_cone=max(float(-dual[:len(blin)].min()),float(np.linalg.norm(dual[len(blin)+1:])-dual[len(blin)]),0.)
    primal_cone=max(float(-slack[:len(blin)].min()),float(np.linalg.norm(slack[len(blin)+1:])-slack[len(blin)]),0.)
    gap=abs(float(objective@x+b@dual))/(1+abs(float(objective@x))+abs(float(b@dual)))
    assert max(stationarity,dual_cone,primal_cone,gap)<cfg['certificate_tolerance']
    q=N@(S@x[:r]);db=-float(mu@q)*w
    edits={'model.visual.proj':torch.tensor((P+q[:,None]*v).astype(np.float32)),
        'model.visual.ln_post.bias':torch.tensor((beta+db).astype(np.float32))}
    torch.save(dict(edits=edits,q=q,v=v,beta_delta=db,mean=mu,solver_margin=x[-1]),out/'point.pt')
    qp,qb=edits['model.visual.proj'].double().numpy(),edits['model.visual.ln_post.bias'].double().numpy()
    changed=p|dict(projection=qp,beta=qb);changed_proxy=dict(pf)
    for key,field in [('model.visual.proj','projection'),('model.visual.ln_post.bias','beta')]:
        base=anchor.state_dict()[key];value=edits[key]
        changed_proxy[field]=(base+own['coefficient']*((value-base)+torch.zeros_like(base))).double().numpy()
    before,after,eb,ea=[cached_scores(h,par,head) for h,par in [(cal['h'],p),(cal['h'],changed),(cal['h_proxy'],pf),(cal['h_proxy'],changed_proxy)]]
    controls=[float(np.max(np.abs(before-cal['original_scores'].numpy()))),float(np.max(np.abs(eb-cal['proxy_original_scores'].numpy())))]
    assert max(controls)<=own['score_tolerance']
    errors=[]
    for h,par,actual in [(cal['h'],p,before),(cal['h'],changed,after),(cal['h_proxy'],pf,eb),(cal['h_proxy'],changed_proxy,ea)]:
        f=normalized_input(h.numpy(),par)@par['projection'];expected=f/np.linalg.norm(f,axis=1,keepdims=True)@H.T
        errors.append(float(np.max(np.abs(expected-actual))))
    assert max(errors)<=own['score_tolerance'],errors
    labels=official_labels(own['target'],True)[cal['indices']];native=metrics(before,after,labels);environment=metrics(eb,ea,labels)
    def projected(zz,pp):
        f=zz@pp;return f/np.linalg.norm(f,axis=1,keepdims=True)@H.T
    probes=metrics(projected(zp,P),projected(zp+qb-beta,qp))
    distance=float(np.hypot(np.linalg.norm(qp-P)/np.linalg.norm(P),np.linalg.norm(qb-beta)/np.linalg.norm(beta)))
    feasible=max(native['R'],probes['R'])<=own['task_budget'] and native['changed']==probes['changed']==0 and distance<=own['parameter_budget']
    supported=feasible and x[-1]>=cfg['minimum_positive_margin'] and np.all(ea.argmax(1)==own['collapse_class'])
    torch.save(dict(native_after=torch.tensor(after),proxy_after=torch.tensor(ea),labels=torch.tensor(labels)),out/'scores.pt')
    record.update(status='source_supported_requires_independent_check_and_reserved_confirmation' if supported else 'closed_fixed_finite_minimax_source',
        source_feasible=feasible,mechanism_supported=supported,native=native,public_AGA_proxy=environment,task_probes=probes,
        parameter_distance=distance,point_sha256=digest(out/'point.pt'),scores_sha256=digest(out/'scores.pt'),conic_problem_sha256=digest(out/'conic_problem.npz'),
        stationarity_error=stationarity,dual_cone_error=dual_cone,primal_cone_error=primal_cone,relative_duality_gap=gap,
        NumPy_score_errors=errors,full_original_control_errors=controls,cached_prefix_readout_examples=4*len(labels),
        reserved_confirmation_images_read=0,new_release_weights=0,wall_seconds=time.monotonic()-start,
        scope='One standard convex minimax solve from legitimate target-training cache; same fixed position,class,task directions,and parameter range as the closed mean-direction control. No source-point direction/amplitude retry,partner,test or new confirmation.')
    save_json(out/'results.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True)
    run(parser.parse_args().protocol)
