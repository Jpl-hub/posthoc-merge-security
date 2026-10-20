"""Finite class-compensation necessary bound on frozen source data; no candidate."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def finite_line_band(h, axis, p, epsilon, caps):
    """Numerical zero-component of a finite, normalized all-class margin band.

    h+t*axis has feature numerator v0+t*v1+sqrt(S(t))*c after final LN.
    A margin hitting q therefore satisfies A_q(t)+sqrt(S(t))*B_q(t)=0.
    Every such event is a root of the quartic A_q**2-S*B_q**2.
    Retaining extraneous squared roots is conservative. Floating root finding
    is NOT an interval-arithmetic certificate; actual FP32 gates remain mandatory.
    """
    from numpy.polynomial import polynomial as poly
    L=p['gamma'][:,None]*p['projection'];L=L-L.mean(0)
    c=p['beta']@p['projection'];hc=h-h.mean(1,keepdims=True);uc=axis-axis.mean()
    scale=np.sqrt(np.sum(hc*hc,axis=1)+h.shape[1]*p['eps'])
    v0=h@L;v1=axis@L;S=np.stack([np.mean(hc*hc,axis=1)+p['eps'],
        2*np.mean(hc*uc,axis=1)*scale,np.full(len(h),np.mean(uc*uc))*scale**2],1)
    before=(v0+np.sqrt(S[:,0,None])*c);before=before/np.linalg.norm(before,axis=1,keepdims=True)@p['head'].T
    y=before.argmax(1);lo=-np.asarray(caps)/scale;hi=np.asarray(caps)/scale;max_residual=0.;events=0
    for i in range(len(h)):
        w=v1*scale[i];D=np.array([v0[i]@v0[i],2*(v0[i]@w),w@w])+np.dot(c,c)*S[i]
        E=2*np.array([v0[i]@c,w@c])
        for k in range(len(p['head'])):
            if k==y[i]:continue
            a=p['head'][y[i]]-p['head'][k];n=np.array([a@v0[i],a@w]);nb=float(a@c)
            gap=before[i,y[i]]-before[i,k];assert gap>0
            for q in [(1-epsilon)*gap,(1+epsilon)*gap]:
                A=poly.polyadd(poly.polymul(n,n),nb*nb*S[i])-q*q*D
                B=2*nb*n-q*q*E
                F=poly.polysub(poly.polymul(A,A),poly.polymul(S[i],poly.polymul(B,B)))
                F=poly.polytrim(F);assert np.isfinite(F).all() and np.any(F)
                roots=poly.polyroots(F/np.max(abs(F)))
                for root in roots:
                    residual=abs(poly.polyval(root,F))/max(float(poly.polyval(abs(root),abs(F))),1e-300)
                    max_residual=max(max_residual,float(residual));assert residual<1e-6
                    if abs(root.imag)<=1e-7*max(1.,abs(root.real)):
                        r=float(root.real);assert r!=0
                        if r<0:lo[i]=max(lo[i],r)
                        else:hi[i]=min(hi[i],r)
                        events+=1
    # Fixed inward reserve for numerical roots, not a changed task budget.
    lo*=.99;hi*=.99
    assert (lo<0).all() and (hi>0).all()
    grid=np.linspace(0,1,17);t=lo[:,None]*(1-grid)+hi[:,None]*grid
    sig=np.sqrt(S[:,0,None]+S[:,1,None]*t+S[:,2,None]*t*t)
    v=v0[:,None,:]+t[:,:,None]*scale[:,None,None]*v1+sig[:,:,None]*c
    sc=(v/np.linalg.norm(v,axis=-1,keepdims=True))@p['head'].T
    gaps=before[np.arange(len(h)),y,None]-before;gaps[np.arange(len(h)),y]=1.
    winning=sc[np.arange(len(h))[:,None],np.arange(len(grid))[None,:],y[:,None]]
    ratio=(winning[:,:,None]-sc-(before[np.arange(len(h)),y,None]-before)[:,None,:])/gaps[:,None,:]
    ratio[np.arange(len(h))[:,None],np.arange(len(grid))[None,:],y[:,None]]=0
    worst=float(abs(ratio).max());assert worst<=epsilon+1e-7
    return lo*scale,hi*scale,dict(root_backward_error_max=max_residual,retained_real_events=events,
        fixed17_grid_worst_R=worst,interval_n=len(h),formal_interval_certificate=False)


def rank_one_band(protocol):
    """One non-null B channel, finite task intervals, unchanged-AGA transport."""
    import clarabel
    from scipy import sparse
    from angular_distance import signed_residual
    from clip_benchmark import digest,merge
    from covariance_noise import dataset
    from mechanisms import load_author
    from probe_margin_distance import scores
    from subspace_probe import make_loader,save_json
    from task_probe_distance import checked_models,initialize,parameters
    from verify_covariance_noise import official_labels
    initialize();start=time.monotonic();cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output'])
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    assert clarabel.__version__==cfg['clarabel_version']
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    save_json(out/'protocol.json',cfg);anchor,expert,head,pins=checked_models(cfg['target'])
    assert digest(cfg['source_split'])==cfg['source_split_sha256']
    split=json.loads(Path(cfg['source_split']).read_text());ids=split['indices'];assert len(ids)==cfg['source_n']
    assert not set(ids)&set(cfg['forbidden_indices']);save_json(out/'source_split.json',split)
    data=dataset(cfg['target'],'data',expert.val_preprocess,True)
    @torch.no_grad()
    def capture(model,take):
        zz=[];hh=[];ss=[];v=model.model.visual
        handles=[v.transformer.resblocks[11].mlp.c_proj.register_forward_pre_hook(lambda _m,a:zz.append(a[0][0].detach().cpu())),
                 v.ln_post.register_forward_pre_hook(lambda _m,a:hh.append(a[0].detach().cpu()))]
        try:
            model.cuda().eval();head.cuda().eval()
            for images,_ in make_loader(data,take,cfg['batch_size']):ss.append(head(model(images.cuda())).detach().cpu())
        finally:
            for handle in handles:handle.remove()
            model.cpu();head.cpu()
        return torch.cat(zz).double(),torch.cat(hh).double(),torch.cat(ss).double()
    p=parameters(expert,head);p0=parameters(anchor,head);pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in p.items()}
    M=p['gamma'][:,None]*(p['projection']@(p['head']-p['head'].mean(0)).T);M-=M.mean(0)
    axis=torch.linalg.svd(M,full_matrices=False)[0][:,0];axis*=torch.sign(axis[axis.abs().argmax()])
    V=torch.linalg.svd((p['B']-p0['B']).cuda(),full_matrices=False)[2][:cfg['rank']].cpu()
    Z,H,before=capture(expert,ids);gold=torch.tensor(official_labels(cfg['target'],True)[ids])
    control=float(abs(scores(H,p)-before).max());assert control<=cfg['score_tolerance']
    y=before.argmax(1);contexts=torch.tensor([int(torch.where(y==c)[0][0]) for c in range(len(p['head']))])
    rms=(Z@V.T).square().mean(0).sqrt()
    shifts=(cfg['probe_step']*torch.tensor([-1.,1.],dtype=torch.float64)[:,None,None]*rms[None,:,None]*V[None]).flatten(0,1)
    Zp=(Z[contexts,None]+shifts[None]).flatten(0,1);Hp=(H[contexts,None]+shifts[None]@p['B'].T).flatten(0,1)
    allZ=torch.cat([Z,Zp]);allH=torch.cat([H,Hp]);nb=float(p['B'].norm())
    lower,upper,numeric=finite_line_band(allH.numpy(),axis.numpy(),pn,cfg['source_reserve'],
        cfg['parameter_budget']*nb*np.linalg.norm(allZ.numpy(),axis=1))
    save_json(out/'band_geometry.json',numeric);print(json.dumps(dict(stage='finite_band_completed',**numeric)),flush=True)
    author,pin=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'));assert pin==cfg['AGA_text_sha256']
    permutation=author['solve_permutation'](p['A'].float(),p0['A'].float());assert permutation is not None
    permutation=torch.as_tensor(permutation,dtype=torch.long)
    assert digest(cfg['AGA_reference']['path'])==cfg['AGA_reference']['sha256']
    import copy
    aligned=copy.deepcopy(expert);aligned.load_state_dict(torch.load(cfg['AGA_reference']['path'],map_location='cpu'),strict=True)
    np.testing.assert_array_equal(parameters(aligned,head)['B'].numpy(),p['B'][:,permutation].numpy())
    env=merge(anchor,aligned,anchor,.5);ep=parameters(env,head);en={k:v.numpy() if torch.is_tensor(v) else v for k,v in ep.items()}
    Ez,Eh,Es=capture(env,ids[:cfg['environment_n']]);del env,aligned
    env_control=float(abs(scores(Eh,ep)-Es).max());assert env_control<=cfg['score_tolerance']
    # Exact derivative of the normalized finite readout; proposals only.
    L=en['gamma'][:,None]*en['projection'];L-=L.mean(0);c=en['beta']@en['projection']
    h=Eh.numpy();hc=h-h.mean(1,keepdims=True);u=axis.numpy();uc=u-u.mean();sig=np.sqrt(np.mean(hc*hc,axis=1)+en['eps'])
    sd=np.mean(hc*uc,axis=1)/sig;v=h@L+sig[:,None]*c;dv=u@L+sd[:,None]*c
    vn=np.linalg.norm(v,axis=1);derivative=(dv/vn[:,None]-v*(np.sum(v*dv,axis=1)/vn**3)[:,None])@en['head'].T
    es=(v/vn[:,None])@en['head'].T;ey=es.argmax(1);other=es.copy();other[np.arange(len(es)),ey]=-np.inf;k=other.argmax(1)
    gap=es[np.arange(len(es)),ey]-es[np.arange(len(es)),k];dg=derivative[np.arange(len(es)),ey]-derivative[np.arange(len(es)),k]
    with torch.enable_grad():
        hh=Eh[:16].clone().requires_grad_(True);s=scores(hh,ep);rr=torch.arange(16)
        fun=((s[rr,torch.tensor(ey[:16])]-s[rr,torch.tensor(k[:16])])/torch.tensor(gap[:16])).sum()
        ad=torch.autograd.grad(fun,hh)[0]@axis
    derivative_error=float(abs(ad.numpy()-dg[:16]/gap[:16]).max());assert derivative_error<=1e-7
    transported=Ez[:,torch.argsort(permutation)].numpy()
    G=np.stack([np.mean((-.5*nb*dg[ey==j]/gap[ey==j])[:,None]*transported[ey==j],axis=0) for j in range(len(p['head']))])
    assert np.isfinite(G).all();gs=float(np.linalg.norm(G,axis=1).max());assert gs>0
    F=nb*allZ.numpy();scale=np.linalg.norm(F,axis=1);F/=scale[:,None];npar=F.shape[1]+1
    linear=sparse.vstack([sparse.hstack([sparse.csc_matrix(F),sparse.csc_matrix((len(F),1))]),
        sparse.hstack([sparse.csc_matrix(-F),sparse.csc_matrix((len(F),1))]),sparse.csc_matrix(np.c_[-G/gs,np.ones(len(G))])],format='csc')
    equality=sparse.csc_matrix(np.c_[V.numpy(),np.zeros(len(V))]);ball=sparse.vstack([sparse.csc_matrix((1,npar)),
        sparse.hstack([-sparse.eye(npar-1),sparse.csc_matrix((npar-1,1))])],format='csc')
    A=sparse.vstack([linear,equality,ball],format='csc');radius=cfg['parameter_budget']*(1-cfg['distance_reserve'])
    b=np.r_[upper/scale,-lower/scale,np.zeros(len(G)+len(V)),radius,np.zeros(npar-1)]
    cones=[clarabel.NonnegativeConeT(linear.shape[0]),clarabel.ZeroConeT(len(V)),clarabel.SecondOrderConeT(npar)]
    q=np.r_[np.zeros(npar-1),-1.];settings=clarabel.DefaultSettings();settings.verbose=False
    settings.max_iter=100;settings.time_limit=cfg['solver_seconds'];settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-8
    sol=clarabel.DefaultSolver(sparse.csc_matrix((npar,npar)),q,A,b,cones,settings).solve();status=str(sol.status)
    np.savez_compressed(out/'solver.npz',x=np.asarray(sol.x),z=np.asarray(sol.z),s=np.asarray(sol.s),b=b,q=q)
    sparse.save_npz(out/'constraints.npz',A)
    raw=dict(indices=ids,labels=gold,parameters=p,axis=axis,V=V,Z=Z,H=H,before=before,probe_Z=Zp,probe_H=Hp,
        lower=torch.from_numpy(lower),upper=torch.from_numpy(upper),environment_parameters=ep,environment_Z=Ez,
        environment_H=Eh,environment_before=Es,permutation=permutation,G=torch.from_numpy(G))
    result=dict(status='closed_solver_'+status,solver_status=status,solver_seconds=sol.solve_time,solver_iterations=sol.iterations,
        source_head_error=control,environment_head_error=env_control,derivative_error=derivative_error,source_n=len(ids),
        finite_probe_n=len(Zp),source_band=numeric,checkpoint_sha256=pins,parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],
        source_sha256=digest(__file__),protocol_sha256=digest(protocol),new_candidates=0,new_AGA_calls=0,new_TA_calls=0,
        original_expert_updates=0,partner_used=False,test_used=False,confirmation_used=False,formal_interval_certificate=False)
    if status=='Solved':
        w=torch.from_numpy(np.asarray(sol.x)[:-1]);delta=nb*axis[:,None]*w[None,:]
        edit=(p['B']+delta).float();actual=edit.double()-p['B'];D=float(actual.norm()/p['B'].norm())
        assert D<=cfg['parameter_budget']
        with torch.no_grad():expert.model.visual.transformer.resblocks[11].mlp.c_proj.weight.copy_(edit)
        _,_,after=capture(expert,ids);rho=float(signed_residual(before,after).abs().max())
        prototype=scores(H+Z@actual.T,p);material_error=float(abs(after-prototype).max());assert material_error<=cfg['score_tolerance']
        probe_before=scores(Hp,p);probe_after=scores(Hp+Zp@actual.T,p);probe_R=float(signed_residual(probe_before,probe_after).abs().max())
        environment_after=scores(Eh+.5*Ez@actual[:,permutation].T,ep)
        source_ok=bool(rho<=cfg['task_budget'] and probe_R<=cfg['task_budget'] and torch.equal(before.argmax(1),after.argmax(1)))
        result.update(status='source_feasible_requires_independent_validation' if source_ok else 'closed_finite_source_failure',
            source_R=rho,probe_R=probe_R,source_changed=int((before.argmax(1)!=after.argmax(1)).sum()),
            parameter_distance=D,task32_response_fraction=float((actual@V.T).norm()/actual.norm()),
            minimum_linear_environment_harm=float((G@w.numpy()).min()),environment_changed=int((Es.argmax(1)!=environment_after.argmax(1)).sum()),
            environment_original_correct=int((Es.argmax(1)==gold[:len(Es)]).sum()),environment_correct=int((environment_after.argmax(1)==gold[:len(Es)]).sum()),
            full_model_score_error=material_error,new_candidates=1,source_feasible=source_ok)
        raw.update(w=w,actual_delta=actual,edited_B=edit,after=after,probe_before=probe_before,probe_after=probe_after,environment_after=environment_after)
        if source_ok:torch.save({'model.visual.transformer.resblocks.11.mlp.c_proj.weight':edit},out/'provisional_weights.pt')
    torch.save(raw,out/'diagnostic.pt');result.update(raw_sha256=digest(out/'diagnostic.pt'),constraints_sha256=digest(out/'constraints.npz'),
        solver_sha256=digest(out/'solver.npz'),wall_seconds=time.monotonic()-start,
        scope='One new finite non-null-output rank-one B family. Source intervals use the actual LN variance and feature norm,not a local task Jacobian or repaired closed point. Standard quartics/SVD/SOCP are not novelty. A linear public-AGA-anchor objective is a proposal,not a finite damage or unknown-partner guarantee. No confirmation/full task/real peer TA/fullAGA yet.')
    save_json(out/'results.json',result);print(json.dumps(result),flush=True)


def structural_unit_check(protocol):
    """Necessary source cost of one LN-annihilated unit; not a new defense."""
    from clip_benchmark import digest
    from probe_margin_distance import scores
    from subspace_probe import save_json
    from task_probe_distance import checked_models,initialize,parameters
    initialize();start=time.monotonic();cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output'])
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    assert digest(cfg['original_source']['path'])==cfg['original_source']['sha256']
    out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    raw=torch.load(cfg['original_source']['path'],map_location='cpu')
    anchor,expert,head,pins=checked_models('MNIST');p=parameters(expert,head);p0=parameters(anchor,head)
    for key,value in p.items():
        assert torch.equal(value,raw['parameters'][key]) if torch.is_tensor(value) else value==raw['parameters'][key]
    Z,H,before=raw['Z'],raw['H'],raw['before'];assert len(Z)==4096
    assert not set(raw['indices'])&set(cfg['forbidden_indices'])
    block=expert.model.visual.transformer.resblocks[11];gamma=block.ln_2.weight.double();beta=block.ln_2.bias.double()
    assert float(gamma.abs().min())>1e-8
    n=gamma.reciprocal();n/=n.norm();c=beta@n
    bound=Z.abs().amax(0)*p['B'].norm(dim=0);j=int(bound.argmin())
    na,nb,ba=[float(p[k].norm()) for k in ['A','B','a']];assert ba>0
    aj=p['A'][j];bj=p['B'][:,j];bias=float(p['a'][j]);radius=.5*(1-1e-6)
    aa=2+(na*float(c)/ba)**2;bb=-2*float(aj@n)/na+2*na*float(c)*bias/ba**2
    cost=float(aj.square().sum())/na**2+bias*bias/ba**2
    basic=dict(unit=j,deletion_hidden_norm_upper=float(bound[j]),deletion_A_bias_cost=cost**.5,
        checkpoint_sha256=pins,source_sha256=digest(__file__),protocol_sha256=digest(protocol),new_images=0,new_peer_TA_calls=0,new_AGA_calls=0)
    if cost>=radius**2:
        save_json(out/'results.json',basic|dict(status='closed_deletion_parameter_cost',released_weight_files=0,wall_seconds=time.monotonic()-start));return
    eta=(-bb+math.sqrt(bb*bb-4*aa*(cost-radius**2)))/(2*aa);assert eta>0
    M=p['gamma'][:,None]*(p['projection']@(p['head']-p['head'].mean(0)).T);M-=M.mean(0)
    u=torch.linalg.svd(M,full_matrices=False)[0][:,0];u*=torch.sign(u[u.abs().argmax()])
    edits=dict(A_row=(eta*na*n).float(),a_value=torch.tensor(-eta*na*float(c),dtype=torch.float32),B_column=(bj+eta*nb*u).float())
    da=edits['A_row'].double()-aj;db=edits['B_column'].double()-bj;dc=float(edits['a_value'])-bias
    D=(float(da.square().sum())/na**2+float(db.square().sum())/nb**2+dc*dc/ba**2)**.5;assert D<=.5
    Va=torch.linalg.svd((p['A']-p0['A']).cuda(),full_matrices=False)[2][:32].cpu()
    Vb=torch.linalg.svd((p['B']-p0['B']).cuda(),full_matrices=False)[2][:32].cpu()
    numerator=edits['A_row'].double()*gamma;numerator-=numerator.mean()
    affine_offset=float(edits['A_row'].double()@beta+edits['a_value'].double())
    uniform_pre_bound=math.sqrt(len(n))*float(numerator.norm())+abs(affine_offset)
    control=float(abs(scores(H,p)-before).max());assert control<=cfg['score_tolerance']
    # Exact real-arithmetic native transformation is removal of the old unit.
    ideal=H-Z[:,j,None]*bj[None];after=scores(ideal,p)
    y=before.argmax(1);contexts=torch.tensor([int(torch.where(y==k)[0][0]) for k in range(10)])
    rms=(Z@Vb.T).square().mean(0).sqrt()
    shifts=(.01*torch.tensor([-1.,1.],dtype=torch.float64)[:,None,None]*rms[None,:,None]*Vb[None]).flatten(0,1)
    Bnew=p['B'].clone();Bnew[:,j]=edits['B_column'].double()
    hp=(H[contexts,None]+shifts[None]@p['B'].T).flatten(0,1)
    hq=(ideal[contexts,None]+shifts[None]@Bnew.T).flatten(0,1)
    pb,pa=scores(hp,p),scores(hq,p)
    def rho(a,b):
        yy=a.argmax(1);g=a.gather(1,yy[:,None])-a;mask=~torch.nn.functional.one_hot(yy,10).bool();assert (g[mask]>0).all()
        ds=b-a;v=(ds.gather(1,yy[:,None])-ds)/g.masked_fill(~mask,1);return float(v[mask].abs().max())
    native_R,probe_R=rho(before,after),rho(pb,pa);changed=int((y!=after.argmax(1)).sum())
    feasible=native_R<=.5 and probe_R<=.5 and changed==0
    torch.save(dict(unit=j,eta=eta,edits=edits,gamma2=gamma,beta2=beta,n=n,c=c,u=u,Va=Va,Vb=Vb,
        contexts=contexts,shifts=shifts,after=after,probe_before=pb,probe_after=pa),out/'necessary_check.pt')
    result=basic|dict(status='necessary_source_checks_passed_not_actual_release' if feasible else 'closed_necessary_task_check',necessary_source_passed=feasible,
        eta=eta,parameter_distance=D,per_tensor_distance=dict(A=float(da.norm())/na,a=abs(dc)/ba,B=float(db.norm())/nb),
        A_task32_inside_distance=float((da@Va.T).norm())/na,B_task32_inside_distance=float(db.norm())/nb*float(Vb[:,j].norm()),
        native_n=len(Z),native_ideal_R=native_R,native_changed=changed,B_probe_n=len(pb),B_probe_R=probe_R,source_head_control_error=control,
        real_arithmetic_quantized_pre_activation_bound=uniform_pre_bound,quantized_branch_hidden_error_bound=uniform_pre_bound*float(edits['B_column'].double().norm()),
        raw_sha256=digest(out/'necessary_check.pt'),parameter_points_computed=1,released_weight_files=0,original_expert_updates=0,
        wall_seconds=time.monotonic()-start,scope='Only necessary ideal deletion/native and unrestricted B-probe costs of one known-component structural control. The real-arithmetic rounding bound excludes actual FP32 LN/dot-product error. No materialized model,new confirmation,test,actual peer TA or fullAGA. No neuron/layer/strength/probe/rank/seed repair; not an original or effective Defense2.')
    save_json(out/'results.json',result);print(json.dumps(result),flush=True)


def replay_line_system(root):
    """Replay byte-pinned A,b,q with another direct factorization, no images."""
    import clarabel
    from scipy import sparse
    from clip_benchmark import digest
    from subspace_probe import save_json
    root=Path(root);cfg=json.loads((root/'protocol.json').read_text());old=json.loads((root/'results.json').read_text())
    protocol=json.loads((root/'backend_protocol.json').read_text());out=root/'backend_replay'
    assert not out.exists();out.mkdir();start=time.monotonic()
    for path,pin in protocol['source_sha256'].items():assert digest(path)==pin,path
    for file,key in [('constraints.npz','constraints_sha256'),('solver.npz','solver_sha256'),('diagnostic.pt','raw_sha256')]:
        assert digest(root/file)==old[key]==protocol['input_sha256'][file]
    assert old['solver_status']=='MaxTime' and protocol['backend']=='faer' and protocol['threads']==4
    assert json.loads((root/'independent_validation.json').read_text())['shared_constraint_reconstruction_error']<1e-8
    A=sparse.load_npz(root/'constraints.npz');saved=np.load(root/'solver.npz');b,q=saved['b'],saved['q']
    m=2*(cfg['source_n']+10*2*cfg['rank'])+10;n=A.shape[1]
    cones=[clarabel.NonnegativeConeT(m),clarabel.ZeroConeT(cfg['rank']),clarabel.SecondOrderConeT(n)]
    assert A.shape[0]==m+cfg['rank']+n
    s=clarabel.DefaultSettings();s.verbose=True;s.direct_solve_method='faer';s.max_threads=4
    s.max_iter=100;s.time_limit=cfg['solver_seconds'];s.tol_feas=s.tol_gap_abs=s.tol_gap_rel=1e-8
    save_json(out/'protocol.json',protocol);print(json.dumps(dict(stage='same_matrix_backend_replay',shape=A.shape,nnz=A.nnz)),flush=True)
    solver=clarabel.DefaultSolver(sparse.csc_matrix((n,n)),q,A,b,cones,s)
    sol=solver.solve();np.savez_compressed(out/'solver.npz',x=sol.x,z=sol.z,s=sol.s,b=b,q=q)
    report=dict(status=str(sol.status),iterations=sol.iterations,solver_seconds=sol.solve_time,wall_seconds=time.monotonic()-start,
        backend='faer',threads=4,same_problem_data=True,input_sha256=protocol['input_sha256'],
        solver_sha256=digest(out/'solver.npz'),source_sha256=digest(__file__),new_images=0,new_expert_updates=0,
        new_released_weights=0,new_confirmation_inputs=0,new_TA_calls=0,new_AGA_calls=0,
        scope='Explicit computational amendment before any weight proposal/confirmation/test score was seen. Original auto-QDLDL MaxTime remains failed. Exact serialized A,b,q,cones and tolerances unchanged; one backend replay,no backend sweep or scientific hyperparameter change. A Solved coefficient vector is not source-retention or defense evidence.')
    save_json(out/'results.json',report);print(json.dumps(report),flush=True)


def dense_line_system(root):
    """Dense reference of the SAME frozen cone problem, no source/model rerun."""
    import cvxopt
    from cvxopt import matrix,solvers
    from scipy import sparse
    from clip_benchmark import digest
    from subspace_probe import save_json
    root=Path(root);protocol=json.loads((root/'dense_protocol.json').read_text());out=root/'dense_replay'
    assert not out.exists();out.mkdir();start=time.monotonic()
    assert cvxopt.__version__==protocol['cvxopt_version']
    for path,pin in protocol['source_sha256'].items():assert digest(path)==pin,path
    for file,pin in protocol['input_sha256'].items():assert digest(root/file)==pin,file
    A=sparse.load_npz(root/'constraints.npz');old=np.load(root/'solver.npz');b,q=old['b'],old['q'];m=9482;p=32;n=A.shape[1]
    assert A.shape==(m+p+n,n)
    inequalities=np.r_[np.arange(m),np.arange(m+p,A.shape[0])];G=A[inequalities].toarray();E=A[m:m+p].toarray()
    # Exact row split; equality rows are retained rather than dropped/penalized.
    v=np.sin(np.arange(n,dtype=float));expected=A@v;checked=np.r_[G[:m]@v,E@v,G[m:]@v]
    assembly_error=float(abs(checked-expected).max());assert assembly_error<1e-9
    save_json(out/'protocol.json',protocol);print(json.dumps(dict(stage='dense_same_system',shape=A.shape,row_split_error=assembly_error)),flush=True)
    solvers.options.update(show_progress=True,maxiters=100,abstol=1e-8,reltol=1e-8,feastol=1e-8)
    sol=solvers.conelp(matrix(q),matrix(G),matrix(b[inequalities]),dict(l=m,q=[n],s=[]),matrix(E),matrix(b[m:m+p]),kktsolver='chol')
    if sol['x'] is not None:
        z=np.asarray(sol['z']).ravel();s=np.asarray(sol['s']).ravel();y=np.asarray(sol['y']).ravel()
        np.savez_compressed(out/'solver.npz',x=np.asarray(sol['x']).ravel(),z=np.r_[z[:m],y,z[m:]],
            s=np.r_[s[:m],np.zeros(p),s[m:]],b=b,q=q)
    report=dict(status=sol['status'],iterations=sol['iterations'],wall_seconds=time.monotonic()-start,
        row_split_error=assembly_error,primal_infeasibility=sol['primal infeasibility'],dual_infeasibility=sol['dual infeasibility'],
        gap=sol['gap'],relative_gap=sol['relative gap'],primal_objective=sol['primal objective'],dual_objective=sol['dual objective'],
        input_sha256=protocol['input_sha256'],cvxopt_version=cvxopt.__version__,source_sha256=digest(__file__),
        new_images=0,new_weights=0,new_confirmation_inputs=0,new_TA_calls=0,new_AGA_calls=0,
        scope='Explicit second computational amendment: dense Cholesky reference of byte-pinned A,b,q with only a row split into identical equality/nonnegative/SOC constraints. No rank/axis/source/scientific budget/tolerance change. Prior MaxTime and terminated sparse replay preserved. Not an effective defense or task-retention pass.')
    if (out/'solver.npz').exists():report['solver_sha256']=digest(out/'solver.npz')
    save_json(out/'results.json',report);print(json.dumps(report),flush=True)


@torch.no_grad()
def normalization_range(source, output):
    """Read the old linear cache; do not solve for or publish a new B."""
    assert not output.exists()
    pin='cd771806645c9c112328938e1149c2b35623dfd7a5fbf7057301dee361eb03fe'
    hsh=hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):hsh.update(block)
    assert hsh.hexdigest()==pin and '3090' in torch.cuda.get_device_name()
    torch.set_num_threads(4);start=time.monotonic();raw=torch.load(source,map_location='cpu')
    p={k:v.cuda() if torch.is_tensor(v) else v for k,v in raw['p'].items()}
    j,z,h,dual=[raw[k].cuda() for k in ['j','z','hidden','dual']];n,d=h.shape
    L=p['gamma'][:,None]*p['projection'];L=L-L.mean(0);c=p['beta']@p['projection']
    M=L@(p['head']-p['head'].mean(0)).T;uq,sv,_=torch.linalg.svd(M,full_matrices=False)
    Q=uq[:,sv>sv[0]*1e-10];assert Q.shape==(768,9)
    jc=(j@Q)@Q.T;jo=j-jc;hc=h-h.mean(1,keepdim=True)
    sigma=(hc.square().mean(1,keepdim=True)+p['eps']).sqrt();u=h@L+sigma*c
    modes=torch.stack([hc/(d*sigma),u@L.T],-1);modes=modes-Q@(Q.T@modes)
    axes,sing,_=torch.linalg.svd(modes,full_matrices=False)
    axes=axes*(sing>sing[:,:1]*1e-10)[:,None,:]
    span_error=float((jo-(jo@axes)@axes.transpose(1,2)).norm()/j.norm());assert span_error<1e-9
    eigen=torch.linalg.eigvalsh(j.flatten(0,1).T@j.flatten(0,1));trace=j.square().sum()
    weight=z.square().sum(1);outside_fraction=float(jo.square().sum()/trace)
    weighted_fraction=float((jo.square().sum((1,2))*weight).sum()/(j.square().sum((1,2))*weight).sum())
    # Independent NumPy derivative of the unnormalised-numerator representation.
    ln,cn,hn,he=[x.cpu().numpy() for x in [L,c,h,p['head']]]
    yn=raw['winner'].numpy();gaps=raw['gaps'].numpy();maximum=relative=0.
    for start_index in range(0,n,128):
        sl=slice(start_index,start_index+128);hh=hn[sl];cc=hh-hh.mean(1,keepdims=True)
        ss=np.sqrt((cc*cc).mean(1,keepdims=True)+p['eps']);uu=hh@ln+ss*cn;norm=np.linalg.norm(uu,axis=1,keepdims=True)
        y=yn[sl];comp=np.tile(np.arange(10),(len(y),1))[~np.eye(10,dtype=bool)[y]].reshape(len(y),9)
        a=he[y,None,:]-he[comp];au=np.einsum('ncf,nf->nc',a,uu);ac=a@cn;uc=uu@cn
        actual=(a@ln.T/norm[:,:,None]+(ac/norm-au*uc[:,None]/norm**3)[:,:,None]*(cc/(d*ss))[:,None,:]
                -au[:,:,None]/norm[:,:,None]**3*(uu@ln.T)[:,None,:])/gaps[sl,:,None]
        ref=raw['j'][sl].numpy();maximum=max(maximum,float(abs(actual-ref).max()))
        relative=max(relative,float(np.linalg.norm(actual-ref)/np.linalg.norm(ref)))
    assert relative<1e-9
    nb=float(raw['norm_B']);old=nb*torch.einsum('ncd,nc->nd',j,dual).T@z
    parts={'full_old_iterate':old,'class_plane_part':Q@(Q.T@old),'outside_part':old-Q@(Q.T@old)};rows={}
    for name,matrix in parts.items():
        response=nb*torch.einsum('ncd,nd->nc',j,z@matrix.T)
        rows[name]=dict(parameter_norm=float(matrix.norm()),linear_residual_max=float((response-raw['b'].cuda()).abs().max()))
        if name=='full_old_iterate':assert float((response-raw['image'].cuda()).norm()/raw['image'].cuda().norm())<1e-9
    qn=np.linalg.svd(M.cpu().numpy(),full_matrices=False)[0][:,:9]
    projector_error=float(abs(qn@qn.T-(Q@Q.T).cpu().numpy()).max());assert projector_error<1e-9
    result=dict(status='complete_normalization_range_independently_checked',source_n=n,static_class_rank=9,
        complete_gradient_global_rank=int((eigen>eigen[-1]*1e-10).sum()),
        complete_gradient_99_9_energy_rank=int((eigen.flip(0).cumsum(0)<.999*eigen.sum()).sum())+1,
        outside_static_class_gradient_energy_fraction=outside_fraction,activation_weighted_operator_trace_fraction=weighted_fraction,
        outside_rank_at_most_two_per_input_relative_error=span_error,NumPy_full_derivative_relative_error=relative,
        NumPy_full_derivative_max_abs_error=maximum,NumPy_projector_error=projector_error,
        old_unconverged_iterate_components=rows,old_iterate_is_not_feasible_B=True,
        raw_source_sha256=pin,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_images=0,new_model_forwards=0,new_solver_steps=0,new_gradients=0,new_releases=0,
        partner_or_test_used=False,closed_point_repaired=False,wall_seconds=time.monotonic()-start,
        scope='Range/cost diagnostic on an existing fixed4096 construction linearization only. Extra normalization directions do not establish finite feasibility, historical causation, novelty or fusion damage.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


@torch.no_grad()
def check(source, output):
    assert not output.exists()
    pin = '677354d85b6afab153c1f97254d109fedf12564d6324fba7e8560bdfd2d9711b'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pin
    assert '4090' in torch.cuda.get_device_name()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic(); data = torch.load(source, map_location='cpu', weights_only=False)
    p = {k: v.cuda().double() if torch.is_tensor(v) else v for k, v in data['parameters'].items()}
    states = {k: v.cuda().double() for k, v in data['states'].items()}
    h, star, z = [states[k] for k in ['h0', 'upper', 'z']]; n, d = h.shape
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; head = p['head']-p['head'].mean(0)
    m, bias = linear@head.T, offset@head.T
    def sigma(x): return ((x-x.mean(1, keepdim=True)).square().mean(1)+p['eps']).sqrt()
    def feature(x): return x@linear+sigma(x)[:, None]*offset
    before_feature, star_feature = feature(h), feature(star)
    before = before_feature/before_feature.norm(dim=1, keepdim=True)@p['head'].T
    y = before.argmax(1); k = before.shape[1]; rows = torch.arange(n, device='cuda')
    competitors = torch.arange(k, device='cuda').expand(n, k)[~torch.nn.functional.one_hot(y, k).bool()].reshape(n, k-1)
    mg = m[:, y].T[:, :, None]-m[:, competitors].permute(1, 0, 2)
    bg = bias[y, None]-bias[competitors]
    g = torch.einsum('nd,ndc->nc', h, mg)+sigma(h)[:, None]*bg
    assert (g > 0).all(), 'Tied teacher margin: report separately rather than drop an input.'
    u, singular, _ = torch.linalg.svd(torch.stack([g, bg], -1), full_matrices=False)
    rank_mask = singular > singular[:, :1]*1e-10
    def project(x):
        return x-torch.einsum('ncr,nr->nc', u, rank_mask*torch.einsum('ncr,nc->nr', u, x))
    q = project(torch.einsum('nd,ndc->nc', star, mg))
    counts = torch.bincount(y, minlength=k); w = q/g.square().sum(1, keepdim=True)/counts[y, None]
    if float(w.norm()) > 0: w /= w.norm()
    qspace, sv, _ = torch.linalg.svd(m, full_matrices=False); qspace = qspace[:, sv > sv[0]*1e-10]
    full_adjoint = torch.einsum('ndc,nc->nd', mg, w).T@z
    class_adjoint = qspace.T@full_adjoint
    outside = float((full_adjoint-qspace@class_adjoint).norm()/full_adjoint.norm().clamp_min(1e-30))
    assert outside < 1e-8
    delta = data['B_delta'].cuda().double()
    after = feature(states['joint']); gap_after = (after@head.T)[rows, y, None]-(after@head.T).gather(1, competitors)
    identity = project(gap_after)-q-project(torch.einsum('nd,ndc->nc', z@delta.T, mg))
    identity_error = float(identity.abs().max()); assert identity_error < 1e-7
    # Independent NumPy readout, including both normalizations, on every source input.
    hn, rn, cn, he = [x.cpu().numpy() for x in [h, linear, offset, p['head']]]
    sig = np.sqrt(np.mean((hn-hn.mean(1, keepdims=True))**2, axis=1)+p['eps'])
    f = hn@rn+sig[:, None]*cn; independent = (f/np.linalg.norm(f, axis=1, keepdims=True))@he.T
    score_error = float(np.abs(independent-before.cpu().numpy()).max()); assert score_error < 1e-8
    radius = math.sqrt(.5**2-data['O_distance']**2); bnorm = data['B_norm']
    lipschitz = float(torch.linalg.matrix_norm(linear, ord=2)+offset.norm()/math.sqrt(d))
    lhs = float((w*q).sum().abs()); basis = data['B_task_basis'].cuda().double()
    records = []
    for name, adjoint, activation in [('B_input32_output_class', class_adjoint@basis.T, z@basis.T),
                                      ('B_full_input_output_class', class_adjoint, z),
                                      ('B_full_input_full_output', full_adjoint, z)]:
        tau = (star_feature.norm(dim=1)+lipschitz*bnorm*radius*activation.norm(dim=1))/before_feature.norm(dim=1)
        allowance = float(.5*(tau*(w.abs()*g).sum(1)).sum())
        coefficient = bnorm*float(adjoint.norm()); capacity = radius*coefficient
        lower = max(0., lhs-allowance)/coefficient if coefficient else None
        records.append(dict(family=name, dual_lhs=lhs, task_tolerance_allowance=allowance,
            parameter_capacity=capacity, budget_upper_rhs=allowance+capacity,
            lower_distance_bound=lower, infeasible_certificate=lhs>allowance+capacity+1e-8*max(1., lhs)))
    result = dict(status='finite_source_necessary_bound_checked_not_new_defense', source_n=n,
        source_sha256=pin, parent_sha256=data['parent_sha256'], implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_upstream_distance=data['O_distance'], remaining_B_distance=radius, task_budget=.5,
        exact_finite_projection_identity_max_abs=identity_error, NumPy_normalized_score_error=score_error,
        output_class_space_rank=qspace.shape[1], full_output_extra_adjoint_fraction=outside,
        dual_witness='Finite residual, teacher-class balanced, divided by teacher numerator gap norm squared; no weight optimization.',
        rows=records, gradients=0, new_weights=0, images_loaded=0, partner_used=False, test_used=False,
        confirmation_used=False, old_counterexamples_not_repaired=True,
        conclusion='An infeasible certificate excludes only this fixed finite upstream change and stated family. Otherwise this conservative bound is inconclusive, not a feasible-point or fusion-effect guarantee. Free B output directions add no projected class-normal directions.',
        GPU=torch.cuda.get_device_name(), wall_seconds=time.monotonic()-start)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def projection_cone(protocol):
    """One finite prefix/readout feasibility test; standard SOCP, no new optimizer."""
    import copy
    import clarabel
    from scipy import sparse
    from angular_distance import angular_scores
    from clip_benchmark import digest
    from confirm_temperature_compensation import unused
    from covariance_noise import dataset
    from subspace_probe import make_loader,save_json
    from task_probe_distance import checked_models,initialize

    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()), 'Never overwrite a previous run.'
    save_json(out/'protocol.json',cfg)
    for p,pin in cfg['source_sha256'].items():assert digest(p)==pin,p
    assert clarabel.__version__==cfg['clarabel_version'];initialize();start=time.monotonic()
    anchor,expert,head,pins=checked_models(cfg['target'])
    assert head.normalize and torch.count_nonzero(head.bias)==0
    train=dataset(cfg['target'],'data',expert.val_preprocess,True)
    excluded=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    if cfg.get('reuse_source_split'):
        assert digest(cfg['reuse_source_split'])==cfg['reuse_source_split_sha256']
        split=json.loads(Path(cfg['reuse_source_split']).read_text())
        assert not set(split['indices'])&excluded
    else:
        split=unused(train,excluded,cfg|dict(confirmation_n=cfg['source_n'],confirmation_seed=cfg['construction_seed']))
    split.update(used_for_optimization=True,role='New source construction only; no confirmation.')
    ids=split['indices'];save_json(out/'source_split.json',split);save_json(out/'training_indices.json',dict(fit=ids))
    original={k:v.clone() for k,v in expert.state_dict().items()}
    B=expert.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double()
    task=B-anchor.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double()
    u,_,vh=torch.linalg.svd(task,full_matrices=False);u=u[:,:cfg['rank']].float().cuda();v=vh[:cfg['rank']].float().cuda()
    P=expert.model.visual.proj.double();P0=anchor.model.visual.proj.double()
    up=torch.linalg.svd(P-P0,full_matrices=False)[0][:,:cfg['rank']].numpy()
    weights=head.weight.double().numpy();qh,sv,_=np.linalg.svd((weights-weights.mean(0)).T,full_matrices=False)
    qh=qh[:,sv>sv[0]*1e-10];assert qh.shape[1]==len(weights)-1
    env=copy.deepcopy(expert);env.load_state_dict({k:.5*(w+anchor.state_dict()[k]) for k,w in original.items()});del anchor
    expert.cuda();env.cuda();head.cuda();runtime=dict(c=None);nb=float(B.norm());handles=[]
    def edit(_m,args,value):
        if runtime['c'] is None:return value
        z=args[0];z=z-(z@v.T)@v
        return value+.5*nb*((z@runtime['c'].T)@u.T)
    handles.append(env.model.visual.transformer.resblocks[0].mlp.c_proj.register_forward_hook(edit))
    def observe(model,indices,capture=False):
        raw=[];scores=[];labels=[]
        hook=model.model.visual.ln_post.register_forward_hook(lambda _m,_a,value:raw.append(value.detach().cpu())) if capture else None
        try:
            with torch.no_grad():
                for x,y in make_loader(train,indices,cfg['batch_size']):scores.append(head(model(x.cuda())).cpu());labels.append(y)
        finally:
            if hook:hook.remove()
        return torch.cat(scores),torch.cat(raw) if capture else None,torch.cat(labels)
    before,X,labels=observe(expert,ids[:cfg['fit_n']],True)
    reference,_,_=observe(env,ids[:cfg['environment_n']]);ref=angular_scores(reference.cuda());ey=ref.argmax(1)
    other=ref.clone();other[torch.arange(len(ref)),ey]=-torch.inf
    gaps=ref[torch.arange(len(ref)),ey]-other.max(1).values;assert (gaps>0).all()
    group=before[:len(ref)].argmax(1).cuda();assert len(group.unique())==len(weights)
    images=torch.stack([train[i][0] for i in ids[:len(ref)]]).cuda()
    def objective(scores):
        z=angular_scores(scores);others=z.clone();others[torch.arange(len(z)),ey]=-torch.inf
        ratio=(z[torch.arange(len(z)),ey]-others.max(1).values)/gaps
        harm=torch.sigmoid(-ratio/cfg['sample_temperature'])
        means=torch.stack([harm[group==k].mean() for k in range(len(weights))]);tau=cfg['class_temperature']
        return -tau*(torch.logsumexp(-means/tau,0)-math.log(len(weights)))
    with torch.enable_grad():
        c=torch.zeros((cfg['rank'],B.shape[1]),device='cuda',requires_grad=True);runtime['c']=c
        value=objective(head(env(images)));g=torch.autograd.grad(value,c)[0].detach()
    gn=float(g.norm());assert gn>0;direction=g/g.norm();step=cfg['derivative_objective_step']/max(1.,gn)
    fd=[]
    with torch.no_grad():
        for sign in [-1.,1.]:runtime['c']=sign*step*direction;fd.append(float(objective(head(env(images)))))
    error=abs((fd[1]-fd[0])/(2*step)-gn)/gn
    numeric=dict(FP32_finite_difference_relative_error=error,step=step,gradient_norm=gn)
    if cfg.get('derivative_reference')=='forward_reverse_AD':
        from torch.autograd import forward_ad
        from torch.nn.attention import sdpa_kernel,SDPBackend
        expected=json.loads(Path(cfg['initial_derivative_failure']).read_text())
        assert digest(cfg['initial_derivative_failure'])==cfg['initial_derivative_failure_sha256']
        assert gn==expected['gradient_norm'] and abs(error-expected['relative_error'])<1e-10
        previous=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH),forward_ad.dual_level():
                runtime['c']=forward_ad.make_dual(torch.zeros_like(direction),direction)
                dual_scores=head(env(images))
                primal,tangent=forward_ad.unpack_dual(objective(dual_scores))
                score_error=float((forward_ad.unpack_dual(dual_scores).primal.cpu()-reference).abs().max())
            # Match both differentiation directions to the same attention kernel.
            with torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
                scalar=torch.zeros((),device='cuda',requires_grad=True)
                runtime['c']=scalar*direction
                reverse_scores=head(env(images))
                reverse=float(torch.autograd.grad(objective(reverse_scores),scalar)[0])
            ad_error=abs(float(tangent)-reverse)/max(1.,abs(reverse))
        finally:
            torch.backends.mha.set_fastpath_enabled(previous)
        numeric.update(forward_directional_derivative=float(tangent),matched_reverse_directional_derivative=reverse,
            forward_reverse_relative_error=ad_error,matched_math_score_error=score_error,
            original_kernel_vs_math_gradient_relative_difference=abs(reverse-gn)/gn,
            initial_finite_difference_still_failed=error>cfg['derivative_tolerance'],reference='Direct forward/reverse AD; original finite difference retained, threshold not raised.')
        save_json(out/'derivative_check.json',numeric)
        assert ad_error<=cfg['AD_relative_tolerance']
        assert abs(reverse-gn)/gn<=cfg['derivative_tolerance']
        assert score_error<=cfg['full_model_score_tolerance']
    else:
        save_json(out/'derivative_check.json',numeric)
        assert error<=cfg['derivative_tolerance']
    for h in handles:h.remove()
    del env,c,g,images
    radius=cfg['parameter_budget']/math.sqrt(2)*(1-cfg['distance_reserve'])
    coeff=radius*direction.double();delta=nb*u.double()@(coeff-(coeff@v.double().T)@v.double())
    edited=(B.cuda()+delta).float().cpu();db=edited.double()-B.cpu();distance=float(db.norm()/B.cpu().norm())
    with torch.no_grad():expert.model.visual.transformer.resblocks[0].mlp.c_proj.weight.copy_(edited.cuda())
    prefix,star,_=observe(expert,ids[:cfg['fit_n']],True)
    vn=X.double().numpy()@P.numpy();norm=np.linalg.norm(vn,axis=1);f=vn/norm[:,None]
    teacher_error=float(abs(f@weights.T-before.numpy()).max())
    assert teacher_error<=cfg['full_model_score_tolerance']
    base=star.double().numpy()@P.numpy();baseq=base@qh
    perp=np.linalg.norm(base-baseq@qh.T,axis=1)
    Z=float(P.norm())*star.double().numpy()@up;npar=cfg['rank']*qh.shape[1]
    teacher=before.numpy().astype(np.float64);yy=teacher.argmax(1);gap=teacher[np.arange(len(yy)),yy,None]-teacher
    matrices=[];rhs=[];cones=[];eps=cfg['source_reserve'];linear=[];linear_rhs=[]
    for i in range(len(yy)):
        mapping=np.kron(Z[i:i+1],np.eye(qh.shape[1]))
        for k in range(len(weights)):
            if k==yy[i]:continue
            a=weights[yy[i]]-weights[k];gik=gap[i,k];assert gik>0
            scale=(1-eps)*gik
            mat=np.vstack([(a@qh@mapping)[None],scale*mapping,np.zeros((1,npar))])
            const=np.r_[a@base[i],scale*baseq[i],scale*perp[i]]
            matrices.append(sparse.csc_matrix(-mat));rhs.append(const);cones.append(clarabel.SecondOrderConeT(len(const)))
            upper=a-(1+eps)*gik*f[i]
            linear.append(upper@qh@mapping);linear_rhs.append(-upper@base[i])
        linear.append(-f[i]@qh@mapping);linear_rhs.append(f[i]@base[i]-cfg['positive_norm_fraction']*norm[i])
    matrices.append(sparse.csc_matrix(np.stack(linear)));rhs.append(np.asarray(linear_rhs));cones.append(clarabel.NonnegativeConeT(len(linear)))
    remaining=math.sqrt(cfg['parameter_budget']**2-distance**2)*(1-cfg['distance_reserve'])
    matrices.append(sparse.vstack([sparse.csc_matrix((1,npar)),-sparse.eye(npar)],format='csc'))
    rhs.append(np.r_[remaining,np.zeros(npar)]);cones.append(clarabel.SecondOrderConeT(npar+1))
    A=sparse.vstack(matrices,format='csc');b=np.concatenate(rhs);quadratic=sparse.eye(npar,format='csc')
    # A deterministic algebra check of the packed cone, not another model point.
    check=np.zeros(npar);check[0]=remaining/2
    value=base[0]+Z[0]@check.reshape(cfg['rank'],-1)@qh.T
    k=next(k for k in range(len(weights)) if k!=yy[0]);a=weights[yy[0]]-weights[k]
    scale=(1-eps)*gap[0,k];expected=np.r_[a@value,scale*(value@qh),scale*perp[0]]
    packed_error=float(abs((b-A@check)[:len(expected)]-expected).max())
    assert packed_error<1e-8
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=cfg['solver_iterations'];settings.time_limit=cfg['solver_seconds']
    settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=cfg['solver_tolerance']
    sol=clarabel.DefaultSolver(quadratic,np.zeros(npar),A,b,cones,settings).solve()
    np.savez_compressed(out/'cone_raw.npz',x=np.asarray(sol.x),z=np.asarray(sol.z),s=np.asarray(sol.s),b=b)
    sparse.save_npz(out/'cone_A.npz',A)
    torch.save(dict(indices=ids,before=before,original_X=X,prefix_X=star,prefix_scores=prefix,labels=labels,
        B_edit=edited,B_direction=direction.cpu(),P=P.cpu(),projection_basis=torch.from_numpy(up),head_basis=torch.from_numpy(qh),head=torch.from_numpy(weights)),out/'source_data.pt')
    status=str(sol.status);report=dict(status=status,scope='One fixed B0 prefix and32x9 shared final-projection finite cone. No old candidate,rank/layer/radius retries,confirmation,test,TA orAGA.',
        parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],source_cone_epsilon=eps,
        prefix_distance=distance,remaining_P_distance=remaining,solver_seconds=sol.solve_time,solver_iterations=sol.iterations,
        source_sha256=digest(__file__),checkpoint_sha256=pins,original_head_control_error=teacher_error,
        packed_cone_control_error=packed_error,new_candidates=0,new_released_weights=0,new_TA_calls=0,new_AGA_calls=0)
    if status=='Solved':
        correction=np.asarray(sol.x).reshape(cfg['rank'],-1)
        projection=(P.numpy()+float(P.norm())*up@correction@qh.T).astype(np.float32)
        after=star.numpy().astype(np.float64)@projection;after=(after/np.linalg.norm(after,axis=1,keepdims=True))@weights.T
        ds=after-teacher;ratio=(ds[np.arange(len(yy)),yy,None]-ds)/np.where(gap>0,gap,1);ratio[np.arange(len(yy)),yy]=0
        report.update(source_R=float(abs(ratio).max()),source_changed=int((after.argmax(1)!=yy).sum()),
            P_distance=float(np.linalg.norm(projection-P.numpy())/float(P.norm())),next_gate='4096 source,full task SVD probes,then genuinely new confirmation and complete task. No release claim yet.')
        torch.save(dict(B_edit=edited,P_edit=torch.from_numpy(projection),source_scores=torch.from_numpy(after)),out/'feasibility_point.pt')
    else:
        z=np.asarray(sol.z);certificate_scale=-float(b@z)
        report.update(certificate_b_dot_z=float(b@z),certificate_stationarity=float(abs(A.T@z).max()),
            certificate_scale=certificate_scale,next_gate='Closed fixed inner-cone experiment; no retry or candidate. Infeasibility of this sufficient cone is not infeasibility of the original task band.')
    report.update(wall_seconds=time.monotonic()-start,source_raw_sha256=digest(out/'source_data.pt'),cone_A_sha256=digest(out/'cone_A.npz'),cone_raw_sha256=digest(out/'cone_raw.npz'))
    save_json(out/'results.json',report);print(json.dumps(report),flush=True)


@torch.no_grad()
def finite_ln_envelope(protocol):
    """One finite O/B feasibility problem with both final-LN norm endpoints."""
    import copy
    import subprocess
    import sys
    import cvxopt
    from cvxopt import matrix, solvers
    from clip_benchmark import digest
    from covariance_noise import dataset
    from cross_module_distance import capture_attention, extended_parameters, normalized_input
    from subspace_probe import save_json
    from task_probe_distance import checked_models, gelu, initialize
    initialize(); start = time.monotonic()
    cfg = json.loads(Path(protocol).read_text()); out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False); save_json(out/'protocol.json', cfg)
    for path, pin in cfg['source_sha256'].items():
        assert digest(path) == pin, path
    assert cvxopt.__version__ == cfg['cvxopt_version']
    assert digest(cfg['source_split']) == cfg['source_split_sha256']
    ids = json.loads(Path(cfg['source_split']).read_text())['indices'][:cfg['source_n']]
    assert len(ids) == len(set(ids)) == cfg['source_n']
    assert not set(ids) & set(cfg['forbidden_indices'])
    anchor, expert, head, pins = checked_models(cfg['target'])
    p = extended_parameters(expert, head); base = extended_parameters(anchor, head)
    rank = cfg['rank']; vo = torch.linalg.svd(p['O']-base['O'], full_matrices=False).Vh[:rank]
    vb = torch.linalg.svd(p['B']-base['B'], full_matrices=False).Vh[:rank]
    choices = []
    for j in range(12):
        sl = slice(64*j, 64*(j+1)); block = p['O'][:, sl]
        q = torch.linalg.qr(block, mode='reduced').Q
        normal = base['O'][:, sl]-q@(q.T@base['O'][:, sl])
        _, sv, vh = torch.linalg.svd(vo[:, sl], full_matrices=True)
        null = vh[int((sv > sv[0]*1e-10).sum()):].T
        u, s, v = torch.linalg.svd(normal@null, full_matrices=False)
        choices.append((float(s[0]), u[:, 0], null@v[0]))
    chosen = int(np.argmax([x[0] for x in choices])); singular, u, v = choices[chosen]
    assert singular > 1e-10
    if u[int(u.abs().argmax())] < 0: u, v = -u, -v
    do = torch.zeros_like(p['O']); do[:, chosen*64:(chosen+1)*64] = -u[:, None]*v[None, :]
    radius = cfg['parameter_budget']*(1-cfg['distance_reserve'])
    edited_o = (p['O']+p['O'].norm()*radius/math.sqrt(2)*do).float()
    do = edited_o.double()-p['O']; od = float(do.norm()/p['O'].norm())
    assert od <= cfg['parameter_budget']/math.sqrt(2)
    o_inside = float((do@vo.T).norm()/p['O'].norm())
    rb = math.sqrt(radius**2-od**2)
    torch.save(dict(O=edited_o,head=chosen,vo=vo,vb=vb), out/'prefix_freeze.pt')
    training = dataset(cfg['target'], 'data', expert.val_preprocess, training=True)
    original = capture_attention(expert, training, ids)
    publication = copy.deepcopy(expert)
    publication.model.visual.transformer.resblocks[11].attn.out_proj.weight.copy_(edited_o)
    prefix = capture_attention(publication, training, ids)
    z = gelu(normalized_input(prefix['r'], p)@p['A'].T+p['a'])
    outside = z-(z@vb.T)@vb
    r = torch.linalg.svd(outside, full_matrices=False).Vh[:rank]
    assert float((r@vb.T).abs().max()) < 1e-8
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; heads = p['head']
    m = linear@(heads-heads.mean(0)).T
    U = torch.linalg.svd(m, full_matrices=False).U[:, :len(heads)-1]
    n, width = prefix['h'].shape; a = U.shape[1]
    assert a == len(heads)-1
    F = (p['B'].norm()*(z@r.T)).numpy(); U = U.numpy()
    L, c, T = linear.numpy(), offset.numpy(), heads.numpy()
    h0, he = prefix['h'].numpy(), original['h'].numpy()
    ce = he-he.mean(1, keepdims=True); se = np.sqrt(np.mean(ce*ce, axis=1)+p['eps'])
    ve = he@L+se[:, None]*c; fe = ve/np.linalg.norm(ve, axis=1, keepdims=True)
    before = head(original['actual'].float()).double().numpy()
    control = float(np.max(abs(fe@T.T-before))); assert control <= cfg['score_tolerance']
    prefix_sigma = np.sqrt(np.mean((h0-h0.mean(1, keepdims=True))**2, axis=1)+p['eps'])
    prefix_feature = h0@L+prefix_sigma[:, None]*c
    prefix_score = prefix_feature/np.linalg.norm(prefix_feature, axis=1, keepdims=True)@T.T
    prefix_control = float(abs(prefix_score-head(prefix['actual'].float()).double().numpy()).max())
    assert prefix_control <= cfg['score_tolerance']
    teacher = before.argmax(1); competitors = np.array([[j for j in range(len(T)) if j != y] for y in teacher])
    gaps = before[np.arange(n), teacher, None]-np.take_along_axis(before, competitors, axis=1)
    assert np.min(gaps) > 0
    prefix_actual = head(prefix['actual'].float()).double().numpy(); dprefix = prefix_actual-before
    prefix_R = float(abs((dprefix[np.arange(n),teacher,None]-np.take_along_axis(dprefix,competitors,axis=1))/gaps).max())
    UL = U.T@L; _, sv, vh = np.linalg.svd(np.vstack([UL, c]), full_matrices=False)
    W = vh[sv > sv[0]*1e-10]; cp = c@W.T; up = UL@W.T
    nc = a*rank; nt = nc+n; nv = nc+3*n+1
    lin, lb, soc, sb, qdims = [], [], [], [], []
    def inequality(coeff, bound):
        lin.append(coeff); lb.append(float(bound))
    def norm_bound(constant, coefficient, index):
        block = np.vstack([np.zeros(nv), -coefficient]); block[0, index] = -1
        soc.append(block); sb.append(np.r_[0., constant]); qdims.append(len(constant)+1)
    tmax, zmax = [], []
    algebra_error = 0.
    for i in range(n):
        D = np.zeros((a, nv))
        for j in range(a): D[j, j*rank:(j+1)*rank] = F[i]
        center = h0[i]-h0[i].mean(); center_u = center@U
        residual = float(center@center-center_u@center_u)
        assert residual > -1e-8
        norm_bound(np.r_[center_u, math.sqrt(max(0., residual)+width*p['eps'])]/math.sqrt(width),
                   np.vstack([D, np.zeros(nv)])/math.sqrt(width), nc+i)
        tm = math.sqrt(float(center@center)/width+p['eps'])+rb*np.linalg.norm(F[i])/math.sqrt(width)
        tmax.append(tm); row = np.zeros(nv); row[nc+i] = 1; inequality(row, tm)
        slo = se[i]+ce[i]@(h0[i]-he[i])/(width*se[i]); slope = ce[i]@U/(width*se[i])
        sigma_bound = max(abs(slo)+np.linalg.norm(slope)*rb*np.linalg.norm(F[i]), tm)
        zm = np.linalg.norm(h0[i]@L)+np.linalg.norm(UL, 2)*rb*np.linalg.norm(F[i])+np.linalg.norm(c)*sigma_bound
        zmax.append(zm)
        vbase = h0[i]@L; vp = vbase@W.T
        residual_v = float(vbase@vbase-vp@vp); assert residual_v > -1e-8
        for side in range(2):
            coef = up.T@D; const = vp.copy()
            if side == 0:
                const += slo*cp; coef += cp[:, None]*(slope@D)[None, :]
            else:
                coef[:, nc+i] += cp
            zi = nt+2*i+side
            norm_bound(np.r_[const, math.sqrt(max(0., residual_v))], np.vstack([coef, np.zeros(nv)]), zi)
            row = np.zeros(nv); row[zi] = 1; inequality(row, zm)
            fp = fe[i]@W.T; f0 = float(fe[i]@vbase)+float((slo if side == 0 else 0.)*(fe[i]@c))
            fc = fp@coef
            inequality(-fc, f0-.5*np.linalg.norm(ve[i]))
            for j, other in enumerate(competitors[i]):
                diff = T[teacher[i]]-T[other]; dp = diff@W.T
                n0 = float(diff@vbase)+float((slo if side == 0 else 0.)*(diff@c))
                nr = dp@coef; lower, upper = (1-cfg['source_reserve'])*gaps[i, j], (1+cfg['source_reserve'])*gaps[i, j]
                row = -nr.copy(); row[zi] += lower; inequality(row, n0)
                inequality(nr-upper*fc, upper*f0-n0)
            # Check reduced norm against the unreduced finite feature at a fixed nonzero coefficient.
            x = np.zeros(nv); x[:nc] = 1e-4*np.sin(np.arange(nc)); dh = U@(D@x)
            x[nc+i] = math.sqrt(np.mean((h0[i]+dh-(h0[i]+dh).mean())**2)+p['eps'])
            ss = slo+slope@(D@x) if side == 0 else x[nc+i]
            full = (h0[i]+dh)@L+ss*c
            algebra_error = max(algebra_error, abs(np.linalg.norm(full)-np.linalg.norm(np.r_[const+coef@x, math.sqrt(max(0., residual_v))])))
    norm_bound(np.zeros(nc), np.c_[np.eye(nc), np.zeros((nc, nv-nc))], nv-1)
    row = np.zeros(nv); row[-1] = 1; inequality(row, rb)
    G = np.vstack(lin+soc); b = np.r_[lb, np.concatenate(sb)]; objective = np.zeros(nv); objective[-1] = 1
    assert algebra_error <= 1e-8
    np.savez_compressed(out/'cone.npz', G=G, b=b, objective=objective, qdims=qdims, linear_rows=len(lin))
    torch.save(dict(ids=ids,labels=training.targets[ids],parameters=p,original=original,prefix=prefix,Z=z,R=r,
                    U=torch.from_numpy(U),F=F,W=W,edited_O=edited_o,budget_B=rb,tmax=tmax,zmax=zmax,teacher=teacher,gaps=gaps), out/'source.pt')
    save_json(out/'pre_solver.json',dict(source_raw_sha256=digest(out/'source.pt'),cone_sha256=digest(out/'cone.npz')))
    subprocess.run([sys.executable,'experiments/verify_finite_compensation_bound.py','--ln-envelope',str(out),'--assembly-only'],check=True)
    print(json.dumps(dict(stage='finite_LN_envelope_ready',shape=G.shape,selected_head=chosen,O_distance=od,assembly_error=algebra_error)), flush=True)
    solvers.options.update(show_progress=True, maxiters=100, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    solved = time.monotonic()
    solution = solvers.conelp(matrix(objective), matrix(G), matrix(b), dict(l=len(lin), q=qdims, s=[]), kktsolver='chol')
    saved = {key:np.asarray(solution[key]).ravel() for key in ['x','s','z'] if solution[key] is not None}
    np.savez_compressed(out/'solver.npz', **saved)
    report = dict(status='closed_'+solution['status'],solver_status=solution['status'],iterations=solution['iterations'],
                  solver_seconds=time.monotonic()-solved,source_n=n,selected_head=chosen,O_distance=od,B_budget=rb,
                  source_control_error=control,prefix_control_error=prefix_control,reduced_norm_error=algebra_error,checkpoint_sha256=pins,
                  O_task32_inside_distance=o_inside,prefix_R=prefix_R,prefix_changed=int((prefix_actual.argmax(1)!=teacher).sum()),
                  original_expert_updates=0,new_TA_calls=0,new_AGA_calls=0,source_sha256=digest(__file__),
                  full_model_image_instances=2*n,prefix_freeze_sha256=digest(out/'prefix_freeze.pt'),
                  protocol_sha256=digest(protocol),published_weight=False)
    if solution['status'] == 'optimal':
        K = torch.from_numpy(saved['x'][:nc].reshape(a, rank))
        eb = (p['B']+p['B'].norm()*torch.from_numpy(U)@K@r).float()
        bd = float((eb.double()-p['B']).norm()/p['B'].norm()); assert math.hypot(od, bd) <= cfg['parameter_budget']
        stem = 'model.visual.transformer.resblocks.11.'
        torch.save({stem+'attn.out_proj.weight':edited_o, stem+'mlp.c_proj.weight':eb}, out/'frozen_release.pt')
        publication.model.visual.transformer.resblocks[11].mlp.c_proj.weight.copy_(eb)
        actual = capture_attention(publication, training, ids)
        after = head(actual['actual'].float()).double().numpy(); dy = after-before
        relative = (dy[np.arange(n), teacher, None]-np.take_along_axis(dy, competitors, axis=1))/gaps
        worst = float(abs(relative).max())
        torch.save(dict(before=torch.from_numpy(before),after=torch.from_numpy(after),H=actual['h']),out/'source_scores.pt')
        truth = training.targets[ids].numpy(); old_pred, new_pred = teacher, after.argmax(1)
        report.update(status='source_passed_pending_probes_and_independent_audit' if worst <= cfg['task_budget'] else 'closed_actual_source_task_budget',
                      B_distance=bd,parameter_distance=math.hypot(od, bd),source_R=worst,
                      B_task32_inside_distance=float(((eb.double()-p['B'])@vb.T).norm()/p['B'].norm()),
                      source_changed=int((after.argmax(1)!=teacher).sum()),source_original_correct=int((teacher==training.targets[ids].numpy()).sum()),
                      source_after_correct=int((after.argmax(1)==training.targets[ids].numpy()).sum()),full_model_image_instances=3*n,
                      per_class=[dict(class_id=j,n=int((truth==j).sum()),before_correct=int(((old_pred==truth)&(truth==j)).sum()),
                                      after_correct=int(((new_pred==truth)&(truth==j)).sum()),lost=int(((old_pred==truth)&(new_pred!=truth)&(truth==j)).sum()),
                                      gained=int(((old_pred!=truth)&(new_pred==truth)&(truth==j)).sum())) for j in range(len(T))],
                      release_sha256=digest(out/'frozen_release.pt'))
    report.update(wall_seconds=time.monotonic()-start,source_raw_sha256=digest(out/'source.pt'),cone_sha256=digest(out/'cone.npz'),solver_sha256=digest(out/'solver.npz'))
    save_json(out/'results.json',report); print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    parser.add_argument('--normalization-range',action='store_true')
    parser.add_argument('--projection-cone',type=Path)
    parser.add_argument('--rank-one-band',type=Path)
    parser.add_argument('--replay-line-system',type=Path)
    parser.add_argument('--dense-line-system',type=Path)
    parser.add_argument('--structural-unit',type=Path)
    parser.add_argument('--ln-envelope',type=Path)
    args = parser.parse_args()
    if args.ln_envelope:finite_ln_envelope(args.ln_envelope)
    elif args.structural_unit:structural_unit_check(args.structural_unit)
    elif args.dense_line_system:dense_line_system(args.dense_line_system)
    elif args.replay_line_system:replay_line_system(args.replay_line_system)
    elif args.rank_one_band:rank_one_band(args.rank_one_band)
    elif args.projection_cone:projection_cone(args.projection_cone)
    else:(normalization_range if args.normalization_range else check)(args.source, args.output)
