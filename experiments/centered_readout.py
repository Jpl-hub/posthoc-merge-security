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
    supported=bool(feasible and x[-1]>=cfg['minimum_positive_margin'] and np.all(ea.argmax(1)==own['collapse_class']))
    torch.save(dict(native_after=torch.tensor(after),proxy_after=torch.tensor(ea),labels=torch.tensor(labels)),out/'scores.pt')
    record.update(status='source_supported_requires_independent_check_and_reserved_confirmation' if supported else 'closed_fixed_finite_minimax_source',
        source_feasible=feasible,mechanism_supported=supported,native=native,public_AGA_proxy=environment,task_probes=probes,
        parameter_distance=distance,point_sha256=digest(out/'point.pt'),scores_sha256=digest(out/'scores.pt'),conic_problem_sha256=digest(out/'conic_problem.npz'),
        stationarity_error=stationarity,dual_cone_error=dual_cone,primal_cone_error=primal_cone,relative_duality_gap=gap,
        NumPy_score_errors=errors,full_original_control_errors=controls,cached_prefix_readout_examples=4*len(labels),
        reserved_confirmation_images_read=0,new_release_weights=0,wall_seconds=time.monotonic()-start,
        scope='One standard convex minimax solve from legitimate target-training cache; same fixed position,class,task directions,and parameter range as the closed mean-direction control. No source-point direction/amplitude retry,partner,test or new confirmation.')
    save_json(out/'results.json',record);print(json.dumps(record),flush=True)


def audit(protocol):
    """Read-only NumPy check of the frozen cone, edits, scores and official labels."""
    start=time.monotonic();cfg=json.loads(Path(protocol).read_text())
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    assert digest(__file__)==cfg['source_sha256']
    out=Path(cfg['output']);assert not (out/'independent_validation.json').exists()
    original=json.loads((out/'protocol.json').read_text());raw=np.load(out/'conic_problem.npz')
    cal=torch.load(original['calibration'],map_location='cpu',weights_only=False)
    scores=torch.load(out/'scores.pt',map_location='cpu',weights_only=False)
    point=torch.load(out/'point.pt',map_location='cpu',weights_only=False)
    A=sp.csc_matrix((raw['data'],raw['indices'],raw['indptr']),shape=tuple(raw['shape']))
    x,d,b,c=[raw[k] for k in ['primal','dual','b','objective']];count=int(raw['nonnegative_rows']);slack=b-A@x
    stationarity=float(abs(A.T@d+c).max());primal=max(0.,float(-slack[:count].min()),float(np.linalg.norm(slack[count+1:])-slack[count]))
    dual=max(0.,float(-d[:count].min()),float(np.linalg.norm(d[count+1:])-d[count]))
    gap=float(abs(c@x+b@d)/(1+abs(c@x)+abs(b@d)))
    assert max(stationarity,primal,dual,gap)<original['certificate_tolerance']
    p,pf=cal['parameters'],cal['proxy_parameters'];P=p['projection'];H=p['head'];N=raw['N'];S=raw['whitening'];v=raw['v'];w=raw['w'];mu=raw['mean']
    q=N@(S@x[:-1]);np.testing.assert_allclose(q,point['q'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(point['beta_delta'],-(mu@q)*w,rtol=0,atol=1e-12)
    assert set(point['edits'])=={'model.visual.proj','model.visual.ln_post.bias'}
    qp,qb=[point['edits'][k].double().numpy() for k in ['model.visual.proj','model.visual.ln_post.bias']]
    np.testing.assert_array_equal(qp,(P+q[:,None]*v).astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(qb,(p['beta']-(mu@q)*w).astype(np.float32).astype(np.float64))
    ortho=float(max(abs(N.T@N-np.eye(N.shape[1])).max(),abs(P.T@N).max(),abs(N.T@(1/p['gamma'])).max(),abs(P.T@w-v).max()))
    assert ortho<1e-10
    def zvalue(h,par):
        a=h.numpy();a=a-a.mean(1,keepdims=True)
        return a/np.sqrt((a*a).mean(1,keepdims=True)+par['eps'])*par['gamma']+par['beta']
    def score(zz,pp):
        f=zz@pp;return f/np.sqrt((f*f).sum(1,keepdims=True))@H.T
    z=zvalue(cal['h'],p);zf=zvalue(cal['h_proxy'],pf)
    np.testing.assert_allclose(z.mean(0),mu,rtol=0,atol=1e-12)
    G=np.eye(N.shape[1])/np.linalg.norm(P)**2+(np.linalg.norm(w)/np.linalg.norm(p['beta']))**2*np.outer(mu@N,mu@N)
    whitening=float(abs(S@G@S-np.eye(len(G))).max());assert whitening<1e-10
    base,expert,head,pins=checked_models(cal['cfg']['target']);assert pins==cal['checkpoint_sha256']
    np.testing.assert_array_equal(P,expert.model.visual.proj.double().numpy())
    np.testing.assert_array_equal(p['beta'],expert.model.visual.ln_post.bias.double().numpy())
    np.testing.assert_array_equal(p['gamma'],expert.model.visual.ln_post.weight.double().numpy())
    np.testing.assert_array_equal(H,head.weight.double().numpy())
    assert cal['cfg']['coefficient']==.5
    U,_,Vh=np.linalg.svd(P-base.model.visual.proj.double().numpy(),full_matrices=False)
    rank=cal['cfg']['task_rank'];task_error=float(max(abs(U[:,:rank].T@q).max(),abs(Vh[:rank]@v).max()))
    assert task_error<1e-9
    yy=cal['original_scores'].argmax(1).numpy();take=[np.flatnonzero(yy==i)[0] for i in range(len(H))]
    basis=U[:,:rank];rms=np.sqrt(np.mean((z@basis)**2,axis=0));step=cal['cfg']['finite_step']
    shifts=np.concatenate([-step*rms[:,None]*basis.T,step*rms[:,None]*basis.T])
    zp=np.repeat(z[take],len(shifts),axis=0)+np.tile(shifts,(len(H),1));zz=np.concatenate([z,zp]);fn=np.linalg.norm(zz@P,axis=1)
    T=((zz-mu)@N@S)/fn[:,None];np.testing.assert_allclose(A[:len(T),:-1].toarray(),T,rtol=0,atol=1e-11)
    np.testing.assert_allclose(A[len(T):2*len(T),:-1].toarray(),-T,rtol=0,atol=1e-11)
    peers=np.arange(len(H))!=cal['cfg']['collapse_class'];contrast=H[cal['cfg']['collapse_class']]-H[peers];en=np.linalg.norm(zf@pf['projection'],axis=1)
    C=.5*((zf@N)[:,None,:]*(contrast@v)[None,:,None]-(contrast@(w@pf['projection']))[None,:,None]*(mu@N))
    C=(C@S)/en[:,None,None]
    np.testing.assert_allclose(A[2*len(T):count,:-1].toarray(),-C.reshape(-1,len(G)),rtol=0,atol=1e-11)
    np.testing.assert_allclose(b[2*len(T):count],((zf@pf['projection'])@contrast.T/en[:,None]).ravel(),rtol=0,atol=1e-10)
    changed_proxy=dict(pf)
    for key,field in [('model.visual.proj','projection'),('model.visual.ln_post.bias','beta')]:
        zero=base.state_dict()[key];changed_proxy[field]=(zero+.5*((point['edits'][key]-zero)+torch.zeros_like(zero))).double().numpy()
    zc=zf-pf['beta']+changed_proxy['beta']
    actual=[cal['original_scores'].numpy(),scores['native_after'].numpy(),cal['proxy_original_scores'].numpy(),scores['proxy_after'].numpy()]
    expected=[score(z,P),score(z+qb-p['beta'],qp),score(zf,pf['projection']),score(zc,changed_proxy['projection'])]
    errors=[float(abs(a-e).max()) for a,e in zip(actual,expected)];assert max(errors)<=cal['cfg']['score_tolerance']
    labels=official_labels(cal['cfg']['target'],True)[cal['indices']];np.testing.assert_array_equal(labels,scores['labels'].numpy())
    native=metrics(actual[0],actual[1],labels);proxy=metrics(actual[2],actual[3],labels);probes=metrics(score(zp,P),score(zp+qb-p['beta'],qp))
    distance=float(np.sqrt(np.sum((qp-P)**2)/np.sum(P**2)+np.sum((qb-p['beta'])**2)/np.sum(p['beta']**2)))
    finite_error=float(abs((z+point['beta_delta'])@(P+q[:,None]*v)-(z@P+((z-mu)@q)[:,None]*v)).max())
    assert finite_error<1e-9
    feasible=bool(max(native['R'],probes['R'])<=cal['cfg']['task_budget'] and native['changed']==probes['changed']==0 and distance<=cal['cfg']['parameter_budget'])
    supported=bool(feasible and x[-1]>=original['minimum_positive_margin'] and np.all(actual[3].argmax(1)==cal['cfg']['collapse_class']))
    classes=[]
    for j in range(len(H)):
        mask=labels==j;classes.append(dict(class_id=j,n=int(mask.sum()),native_correct=int(np.sum((actual[1].argmax(1)==labels)&mask)),proxy_before_correct=int(np.sum((actual[2].argmax(1)==labels)&mask)),proxy_after_correct=int(np.sum((actual[3].argmax(1)==labels)&mask))))
    record=dict(status='source_supported_requires_reserved_confirmation' if supported else 'closed_fixed_finite_minimax_source',mechanism_supported=supported,source_feasible=feasible,
        native=native,public_AGA_proxy=proxy,task_probes=probes,parameter_distance=distance,solver_margin=float(x[-1]),per_class=classes,
        point_sha256=digest(out/'point.pt'),scores_sha256=digest(out/'scores.pt'),conic_problem_sha256=digest(out/'conic_problem.npz'),
        source_preregistration=cfg['source_preregistration'],original_source_sha256=original['source_sha256']['experiments/centered_readout.py'],
        solver_seconds=cfg['solver_seconds'],original_run_total_seconds=None,source_full_image_forwards=0,cached_prefix_readouts=4096,new_AGA_calls=0,expert_training_updates=0,reserved_confirmation_images_read=0,
        serialization_failure='NumPy bool_ at final JSON write; solver,point and scores complete. No second solve or science change.',failure_log_sha256=cfg['inputs'][cfg['failure_log']])
    save_json(out/'results.json',record)
    validation=dict(status='passed_read_only_independent_NumPy_recount',NumPy_score_errors=errors,stationarity_error=stationarity,primal_cone_error=primal,dual_cone_error=dual,relative_duality_gap=gap,
        orthogonality_error=ortho,task_projection_error=task_error,whitening_error=whitening,finite_compensation_error=finite_error,
        labels_checked=len(labels),linear_operator_reconstructed=True,result_sha256=digest(out/'results.json'),source_sha256=digest(__file__),protocol_sha256=digest(protocol),
        new_point_fits=0,new_full_image_forwards=0,wall_seconds=time.monotonic()-start)
    save_json(out/'independent_validation.json',validation);print(json.dumps(record),flush=True)


@torch.no_grad()
def native(protocol):
    """Untouched training confirmation, then the complete task, for one frozen point."""
    import copy
    from covariance_noise import dataset
    from clip_benchmark import evaluate
    from subspace_probe import make_loader
    initialize();cfg=json.loads(Path(protocol).read_text())
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    check=json.loads(Path(cfg['source_result']).read_text());assert check['mechanism_supported']
    assert json.loads(Path(cfg['source_validation']).read_text())['status']=='passed_read_only_independent_NumPy_recount'
    point=torch.load(cfg['point'],map_location='cpu',weights_only=False);_,expert,head,pins=checked_models(cfg['target'])
    assert digest(cfg['point'])==check['point_sha256']
    released=copy.deepcopy(expert);released.load_state_dict(expert.state_dict()|point['edits'],strict=True)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    holdout=json.loads(Path(cfg['holdout']).read_text());draw=json.loads(Path(cfg['source_draw']).read_text())
    assert holdout['indices']==draw['indices'][1024:] and len(holdout['indices'])==512
    assert not set(holdout['indices']).intersection(draw['indices'][:1024])
    save_json(out/'before_images.json',dict(point_sha256=digest(cfg['point']),checkpoint_sha256=pins,indices_sha256=digest(cfg['holdout']),source_sha256=digest(__file__),protocol_sha256=digest(protocol)))
    rows=[];start=time.monotonic()
    for stage,training in [('confirmation',True),('full',False)]:
        data=dataset(cfg['target'],'data',expert.val_preprocess,training);ids=holdout['indices'] if training else list(range(len(data)))
        values=[];prefix=[]
        for model in [expert,released]:
            sink=[];hidden=[]
            hook=model.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:hidden.append(args[0].detach().cpu()))
            try:evaluate(model,head,make_loader(data,ids,cfg['batch_size']),'cuda',logit_sink=sink.append)
            finally:hook.remove()
            values.append(torch.cat(sink).double());prefix.append(torch.cat(hidden))
        torch.testing.assert_close(prefix[0],prefix[1],rtol=0,atol=0)
        before,after=[v.numpy() for v in values];labels=official_labels(cfg['target'],training)[ids]
        row=metrics(before,after,labels);row.update(stage=stage,passed=bool(metrics(before,after)['R']<=cfg['task_budget'] and np.array_equal(before.argmax(1),after.argmax(1))))
        torch.save(dict(indices=ids,labels=torch.tensor(labels),before=values[0],after=values[1],h=prefix[0]),out/(stage+'.pt'))
        row['raw_sha256']=digest(out/(stage+'.pt'));rows.append(row)
        save_json(out/'native.json',dict(status='passed_observed_native_gates' if all(r['passed'] for r in rows) else 'closed_native_function_gate_failed',rows=rows,
            image_forwards=sum(2*r['n'] for r in rows),seconds=time.monotonic()-start,parameter_distance=check['parameter_distance'],parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],
            point_sha256=digest(cfg['point']),new_point_fits=0,new_AGA_calls=0,expert_training_updates=0))
        print(json.dumps(row),flush=True)
        if not row['passed']:break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol');parser.add_argument('--audit');parser.add_argument('--native')
    args=parser.parse_args()
    if args.audit:audit(args.audit)
    elif args.native:native(args.native)
    else:run(args.protocol)
