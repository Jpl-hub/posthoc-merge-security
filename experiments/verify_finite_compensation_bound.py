"""Independent NumPy finite readout, projection and dual-bound reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def verify(source, report, output):
    assert not output.exists()
    result = json.loads(report.read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['source_sha256']
    data = torch.load(source, map_location='cpu', weights_only=False)
    p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in data['parameters'].items()}
    h, star, z, joint = [data['states'][k].numpy() for k in ['h0', 'upper', 'z', 'joint']]
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; head = p['head']-p['head'].mean(0)
    m, b = linear@head.T, offset@head.T
    def feature(x):
        sigma = np.sqrt(np.mean((x-x.mean(1, keepdims=True))**2, axis=1)+p['eps'])
        return x@linear+sigma[:, None]*offset
    f = feature(h); y = (f@head.T).argmax(1); n, d = h.shape; count = np.bincount(y, minlength=len(head))
    competitors = np.array([[c for c in range(len(head)) if c != t] for t in y])
    mg = m[:, y].T[:, :, None]-m[:, competitors].transpose(1, 0, 2)
    bg = b[y, None]-b[competitors]; numerator = f@head.T
    gaps = numerator[np.arange(n), y, None]-np.take_along_axis(numerator, competitors, axis=1)
    spans = np.stack([gaps, bg], -1)
    projection = np.eye(len(head)-1)-spans@np.linalg.pinv(spans, rcond=1e-10)
    q = np.einsum('nij,nj->ni', projection, np.einsum('nd,ndc->nc', star, mg))
    w = q/(gaps*gaps).sum(1, keepdims=True)/count[y, None]
    if np.linalg.norm(w): w /= np.linalg.norm(w)
    adjoint = np.einsum('ndc,nc->nd', mg, w).T@z
    lhs = abs(np.sum(w*q)); radius = result['remaining_B_distance']; norm = data['B_norm']
    lipschitz = np.linalg.norm(linear, 2)+np.linalg.norm(offset)/np.sqrt(d)
    values, ratios = [], []
    for row in result['rows']:
        restricted = row['family'] == 'B_input32_output_class'
        act = z@data['B_task_basis'].numpy().T if restricted else z
        dual = adjoint@data['B_task_basis'].numpy().T if restricted else adjoint
        tau = (np.linalg.norm(feature(star), axis=1)+lipschitz*norm*radius*np.linalg.norm(act, axis=1))/np.linalg.norm(f, axis=1)
        allowance = .5*np.sum(tau*(abs(w)*gaps).sum(1)); capacity = radius*norm*np.linalg.norm(dual)
        expected = [lhs, allowance, capacity, allowance+capacity]
        actual = [row[k] for k in ['dual_lhs', 'task_tolerance_allowance', 'parameter_capacity', 'budget_upper_rhs']]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-8)
        assert row['infeasible_certificate'] == (lhs > allowance+capacity+1e-8*max(1., lhs))
        values.append(dict(family=row['family'], maximum_scalar_difference=float(np.max(abs(np.array(actual)-expected)))))
        if not restricted:
            true_ratio = np.linalg.norm(feature(joint), axis=1)/np.linalg.norm(f, axis=1)
            assert (tau+1e-10 >= true_ratio).all()
            ratios.append(dict(maximum_bound=float(tau.max()), maximum_frozen_actual_ratio=float(true_ratio.max())))
    bdelta = data['B_delta'].numpy()
    np.testing.assert_allclose(star+z@bdelta.T, joint, atol=1e-8, rtol=1e-8)
    validation = dict(status='passed', independent_NumPy_projection_and_bounds=True, source_n=n,
        scalar_checks=values, normalization_bound_looseness=ratios,
        source_sha256=result['source_sha256'], report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gradients=0, images=0, new_weights=0, scope='Frozen source diagnosis only; bounds remain inconclusive.')
    output.write_text(json.dumps(validation, indent=2)+'\n'); print(json.dumps(validation, indent=2))


def verify_projection_cone(root):
    """Rebuild finite cones in NumPy; check a solution or an infeasibility ray."""
    import time
    from scipy import sparse
    from clip_benchmark import digest
    from subspace_probe import save_json
    from task_probe_distance import checked_models,initialize
    from verify_covariance_noise import official_labels
    initialize();start=time.monotonic();root=Path(root)
    assert not (root/'independent_validation.json').exists()
    cfg=json.loads((root/'protocol.json').read_text());meta=json.loads((root/'results.json').read_text())
    for file,key in [('source_data.pt','source_raw_sha256'),('cone_A.npz','cone_A_sha256'),('cone_raw.npz','cone_raw_sha256')]:assert digest(root/file)==meta[key]
    data=torch.load(root/'source_data.pt',map_location='cpu',weights_only=False)
    raw=np.load(root/'cone_raw.npz');A=sparse.load_npz(root/'cone_A.npz');b=raw['b']
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==meta['checkpoint_sha256']
    P=data['P'].numpy();np.testing.assert_array_equal(P,expert.model.visual.proj.double().numpy())
    H=data['head'].numpy();np.testing.assert_array_equal(H,head.weight.double().numpy())
    ids=data['indices'];assert ids==json.loads((root/'source_split.json').read_text())['indices']
    assert not set(ids)&set(cfg['extra_excluded_indices'])
    split=json.loads((root/'source_split.json').read_text());excluded=set(cfg['extra_excluded_indices'])
    boundary=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    static={}
    for row in boundary['records']:static.setdefault(row['path'],[]).append(row['group'])
    for path,pin in split['excluded_metadata_sha256'].items():
        assert digest(path)==pin,path
        if path in static:
            value=json.loads(Path(path).read_text()) if Path(path).suffix=='.json' else torch.load(path,map_location='cpu',weights_only=False)['indices']
            for group in static[path]:excluded.update(value[group])
        else:
            value=json.loads(Path(path).read_text())
            groups=(list(value.values()) if isinstance(value,dict) else [value]) if Path(path).name=='training_indices.json' else [value.get('indices',[])]
            for group in groups:
                if isinstance(group,list):excluded.update(int(x) for x in group)
    for path in cfg['extra_index_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    assert len(excluded)==split['excluded_count']
    pool=sorted(set(range(60000))-excluded)
    expected=np.random.default_rng(cfg['construction_seed']).choice(pool,cfg['source_n'],replace=False).tolist()
    assert expected==ids and not set(ids)&excluded
    np.testing.assert_array_equal(data['labels'].numpy(),official_labels(cfg['target'],True)[ids[:cfg['fit_n']]])
    U,Q=data['projection_basis'].numpy(),data['head_basis'].numpy()
    np.testing.assert_allclose(U.T@U,np.eye(len(U.T)),atol=1e-12)
    np.testing.assert_allclose(Q.T@Q,np.eye(len(Q.T)),atol=1e-12)
    teacher=data['before'].numpy().astype(float);old=data['original_X'].numpy().astype(float)@P
    f=old/np.linalg.norm(old,axis=1,keepdims=True);control=float(abs(f@H.T-teacher).max());assert control<=cfg['full_model_score_tolerance']
    base=data['prefix_X'].numpy().astype(float)@P;Z=np.linalg.norm(P)*data['prefix_X'].numpy().astype(float)@U
    y=teacher.argmax(1);gap=teacher[np.arange(len(y)),y,None]-teacher;npar=A.shape[1];q=Q.shape[1];eps=cfg['source_reserve']
    remaining=meta['remaining_P_distance'];errors=[]
    for c in [np.zeros(npar),np.arange(1,npar+1,dtype=float)*remaining/(2*np.linalg.norm(np.arange(1,npar+1)))]:
        v=base+Z@c.reshape(cfg['rank'],q)@Q.T;perp=base-(base@Q)@Q.T
        expected=[];upper=[]
        for i in range(len(y)):
            for k in range(len(H)):
                if k==y[i]:continue
                a=H[y[i]]-H[k];scale=(1-eps)*gap[i,k]
                expected.extend(np.r_[a@v[i],scale*(v[i]@Q),scale*np.linalg.norm(perp[i])])
                upper.append((1+eps)*gap[i,k]*(f[i]@v[i])-a@v[i])
            upper.append(f[i]@v[i]-.5*np.linalg.norm(old[i]))
        expected=np.r_[expected,upper,remaining,c]
        error=float(abs(expected-(b-A@c)).max());assert error<1e-7;errors.append(error)
    B=expert.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double().numpy()
    actual_D=float(np.linalg.norm(data['B_edit'].numpy()-B)/np.linalg.norm(B))
    assert abs(actual_D-meta['prefix_distance'])<1e-10
    B0=anchor.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double().numpy()
    vb=np.linalg.svd(B-B0,full_matrices=False)[2][:cfg['rank']]
    deltaB=data['B_edit'].numpy()-B
    complement_leak=float(np.linalg.norm(deltaB@vb.T)/np.linalg.norm(deltaB))
    pu=np.linalg.svd(P-anchor.model.visual.proj.double().numpy(),full_matrices=False)[0][:,:cfg['rank']]
    basis_error=float(abs(U@U.T-pu@pu.T).max());assert basis_error<1e-8
    prefix=data['prefix_scores'].numpy().astype(float);ds=prefix-teacher
    drift=(ds[np.arange(len(y)),y,None]-ds)/np.where(gap>0,gap,1);drift[np.arange(len(y)),y]=0
    truth=data['labels'].numpy();prediction=prefix.argmax(1)
    report=dict(status='independent_cone_assembly_checked',solver_status=meta['status'],assembly_errors=errors,
        original_head_error=control,prefix_D=actual_D,source_sampling_reproduced=True,source_n=len(ids),excluded_source_indices=len(excluded),
        prefix_right_task32_response_fraction=complement_leak,projection_task32_basis_error=basis_error,
        fit_n=len(y),original_fit_correct=int((y==truth).sum()),prefix_fit_correct=int((prediction==truth).sum()),
        prefix_fit_changed=int((prediction!=y).sum()),prefix_fit_R=float(abs(drift).max()),
        prefix_per_class=[dict(class_id=k,n=int((truth==k).sum()),lost=int(((y==truth)&(prediction!=truth)&(truth==k)).sum()),
            gained=int(((y!=truth)&(prediction==truth)&(truth==k)).sum())) for k in range(len(H))],
        new_model_forwards=0,new_candidates=0,new_TA_calls=0,new_AGA_calls=0,
        source_sha256=digest(__file__),input_result_sha256=digest(root/'results.json'),raw_sha256=meta['cone_raw_sha256'])
    if meta['status']=='PrimalInfeasible':
        z=raw['z'].copy();stationarity=float(abs(A.T@z).max());dot=float(b@z);at=0;violation=0.
        def project_soc(t):
            norm=np.linalg.norm(t[1:])
            if norm<=t[0]:p=t.copy()
            elif norm<=-t[0]:p=np.zeros_like(t)
            else:p=.5*(norm+t[0])*np.r_[1.,t[1:]/norm]
            p[0]+=1e-12*max(1.,np.linalg.norm(p))
            return p
        for _ in range(len(y)*(len(H)-1)):
            t=z[at:at+q+2];violation=max(violation,float(np.linalg.norm(t[1:])-t[0]));z[at:at+q+2]=project_soc(t);at+=q+2
        count=len(y)*len(H);violation=max(violation,float(-z[at:at+count].min()))
        z[at:at+count]=np.maximum(z[at:at+count],0.)+1e-12;at+=count
        t=z[at:];assert len(t)==npar+1;violation=max(violation,float(np.linalg.norm(t[1:])-t[0]));z[at:]=project_soc(t)
        # A compact variable ball gives a certificate even with nonzero A.T z:
        # feasible => b.T z = x.T A.T z + s.T z >= -r*||A.T z||.
        projected_dot=float(b@z);residual_bound=remaining*float(np.linalg.norm(A.T@z))
        margin=-projected_dot-residual_bound
        report.update(dual_certificate_b_dot_z=dot,dual_stationarity_max=stationarity,dual_cone_violation=max(0.,violation),
            projected_dual_b_dot_z=projected_dot,parameter_ball_residual_allowance=residual_bound,bounded_dual_certificate_margin=margin,
            infeasibility_certificate_verified=margin>1e-5*(1+abs(projected_dot)+residual_bound),
            scope='Numerical certificate with explicit stationarity residual allowance on the known parameter ball and projection into self-dual cones. Excludes only this fixed source prefix,288-variable readout family,and sufficient epsilon0.25 cone; not the original0.5 task band or all defenses. Not a formal interval-arithmetic certificate.')
        np.savez_compressed(root/'projected_dual_certificate.npz',z=z)
        assert report['infeasibility_certificate_verified']
    elif meta['status']=='Solved':
        point=torch.load(root/'feasibility_point.pt',map_location='cpu',weights_only=False)
        v=data['prefix_X'].numpy().astype(float)@point['P_edit'].numpy().astype(float)
        s=v/np.linalg.norm(v,axis=1,keepdims=True)@H.T;delta=s-teacher
        r=(delta[np.arange(len(y)),y,None]-delta)/np.where(gap>0,gap,1);r[np.arange(len(y)),y]=0
        np.testing.assert_allclose(s,point['source_scores'].numpy(),atol=1e-8)
        report.update(actual_source_R=float(abs(r).max()),actual_P_distance=float(np.linalg.norm(point['P_edit'].numpy()-P)/np.linalg.norm(P)),
            scope='512 source feasibility only;4096 source/task-SVD/fresh-confirmation/full-task/TA/AGA gates remain. Not a validated defense.')
        assert report['actual_source_R']<=cfg['task_budget']
        assert (actual_D**2+report['actual_P_distance']**2)**.5<=cfg['parameter_budget']
    else:
        report['scope']='Solver did not return a fully solved point or a full infeasibility certificate; retain numerical/limit status without a scientific feasibility claim.'
    report['wall_seconds']=time.monotonic()-start;save_json(root/'independent_validation.json',report);print(json.dumps(report),flush=True)


def verify_rank_one_band(root):
    """Independent full finite scores, shared constraints and original-batch replay."""
    import copy,time
    from scipy import sparse
    from clip_benchmark import digest
    from covariance_noise import dataset
    from subspace_probe import make_loader,save_json
    from task_probe_distance import checked_models,initialize
    from verify_covariance_noise import official_labels
    from verify_levelset_audits import raw_scores
    initialize();start=time.monotonic();root=Path(root);assert not (root/'independent_validation.json').exists()
    cfg=json.loads((root/'protocol.json').read_text());result=json.loads((root/'results.json').read_text())
    for file,key in [('diagnostic.pt','raw_sha256'),('solver.npz','solver_sha256'),('constraints.npz','constraints_sha256')]:assert digest(root/file)==result[key]
    d=torch.load(root/'diagnostic.pt',map_location='cpu');p={k:v.numpy() if torch.is_tensor(v) else v for k,v in d['parameters'].items()}
    ids=d['indices'];assert ids==json.loads(Path(cfg['source_split']).read_text())['indices']
    np.testing.assert_array_equal(d['labels'].numpy(),official_labels(cfg['target'],True)[ids])
    A=sparse.load_npz(root/'constraints.npz');sol=np.load(root/'solver.npz');x=sol['x'];b=sol['b']
    z=np.concatenate([d['Z'].numpy(),d['probe_Z'].numpy()]);h=np.concatenate([d['H'].numpy(),d['probe_H'].numpy()])
    u=d['axis'].numpy();V=d['V'].numpy();lo=d['lower'].numpy();hi=d['upper'].numpy();nb=np.linalg.norm(p['B'])
    def residual(t,s):
        y=t.argmax(1);gap=t[np.arange(len(t)),y,None]-t;gap[np.arange(len(t)),y]=1
        delta=s-t;r=(delta[np.arange(len(t)),y,None]-delta)/gap;r[np.arange(len(t)),y]=0;return abs(r).max()
    # Independently use explicit NumPy LayerNorm, projection and L2 normalization.
    teacher=raw_scores(h,p);maximum=0.
    for frac in np.linspace(0,1,17):
        t=(1-frac)*lo+frac*hi;maximum=max(maximum,float(residual(teacher,raw_scores(h+t[:,None]*u,p))))
    assert maximum<=cfg['source_reserve']+1e-7
    w=x[:-1];tau=nb*z@w;expected=np.r_[(hi-tau)/np.linalg.norm(nb*z,axis=1),(tau-lo)/np.linalg.norm(nb*z,axis=1),
        d['G'].numpy()@w/np.linalg.norm(d['G'].numpy(),axis=1).max()-x[-1],-V@w,
        cfg['parameter_budget']*(1-cfg['distance_reserve']),w]
    error=float(abs(expected-(b-A@x)).max());assert error<1e-8
    report=dict(status='passed_finite_source_audit',solver_status=result['solver_status'],shared_constraint_reconstruction_error=error,
        independently_recomputed17_grid_R=maximum,formal_interval_certificate=False,source_n=len(ids),
        raw_sha256=result['raw_sha256'],producer_sha256=result['source_sha256'],verifier_sha256=digest(__file__),
        source_R=None,new_release_weights=0,full_peer_TA_calls=0,full_AGA_calls=0)
    if result['solver_status']=='Solved':
        delta=d['actual_delta'].numpy();after=raw_scores(d['H'].numpy()+d['Z'].numpy()@delta.T,p)
        head_error=float(abs(after-d['after'].numpy()).max());assert head_error<=cfg['score_tolerance']
        np.testing.assert_array_equal(p['B']+delta,d['edited_B'].double().numpy())
        rho=float(residual(d['before'].numpy(),d['after'].numpy()));assert abs(rho-result['source_R'])<1e-8
        ep={k:v.numpy() if torch.is_tensor(v) else v for k,v in d['environment_parameters'].items()}
        env=raw_scores(d['environment_H'].numpy()+.5*d['environment_Z'].numpy()@delta[:,d['permutation'].numpy()].T,ep)
        env_error=float(abs(env-d['environment_after'].numpy()).max());assert env_error<1e-8
        report.update(source_R=rho,full_model_tail_error=head_error,environment_NumPy_error=env_error,
            parameter_distance=float(np.linalg.norm(delta)/nb),source_feasible=result['source_feasible'])
        # Replay the original first128 context, not isolated new-size minibatches.
        _,expert,head,pins=checked_models(cfg['target']);assert pins==result['checkpoint_sha256']
        train=dataset(cfg['target'],'data',expert.val_preprocess,True);edited=copy.deepcopy(expert)
        with torch.no_grad():edited.model.visual.transformer.resblocks[11].mlp.c_proj.weight.copy_(d['edited_B'])
        errors=[]
        for model,target in [(expert,d['before']),(edited,d['after'])]:
            model.cuda().eval();head.cuda().eval()
            with torch.no_grad():
                images,_=next(iter(make_loader(train,ids[:128],128)));ss=head(model(images.cuda())).cpu().double()
            model.cpu();head.cpu();errors.append(float(abs(ss-target[:128]).max()));assert errors[-1]<=cfg['score_tolerance']
        report.update(original128_context_replay_errors=errors,replay_image_instances=256)
        before=d['before'].argmax(1).numpy();pred=d['after'].argmax(1).numpy();gold=d['labels'].numpy()
        report['per_class']=[dict(class_id=c,n=int((gold==c).sum()),original_correct=int(((before==gold)&(gold==c)).sum()),
            correct=int(((pred==gold)&(gold==c)).sum()),lost=int(((before==gold)&(pred!=gold)&(gold==c)).sum()),
            gained=int(((before!=gold)&(pred==gold)&(gold==c)).sum())) for c in range(len(p['head']))]
    report.update(wall_seconds=time.monotonic()-start,
        scope='Finite source and numerical interval audit only; no interval-arithmetic/all-input theorem,novelty,fresh confirmation,full-task pass or actual peer-fusion protection claimed.')
    save_json(root/'independent_validation.json',report);print(json.dumps(report),flush=True)


def validate_line_point(root, protocol):
    """Materialize the sole optimal point, then gate full FP32 source/probes."""
    import copy,time
    from scipy import sparse
    from scipy.special import expit,logsumexp
    from clip_benchmark import digest
    from covariance_noise import dataset
    from probe_margin_distance import scores
    from subspace_probe import make_loader,save_json
    from task_probe_distance import checked_models,initialize,parameters
    from verify_covariance_noise import official_labels
    from verify_levelset_audits import raw_scores
    initialize();start=time.monotonic();root=Path(root);protocol=Path(protocol)
    gate=json.loads(protocol.read_text());cfg=json.loads((root/'protocol.json').read_text())
    assert digest(__file__)==gate['verifier_sha256']
    assert not (root/'source_validation.json').exists() and not (root/'provisional_weights.pt').exists()
    initial=json.loads((root/'results.json').read_text());dense=json.loads((root/'dense_replay/results.json').read_text())
    assert dense['status']=='optimal' and digest(root/'dense_replay/solver.npz')==dense['solver_sha256']
    for file,pin in dense['input_sha256'].items():assert digest(root/file)==pin==initial[{'diagnostic.pt':'raw_sha256','constraints.npz':'constraints_sha256','solver.npz':'solver_sha256'}[file]]
    d=torch.load(root/'diagnostic.pt',map_location='cpu');p={k:v.numpy() if torch.is_tensor(v) else v for k,v in d['parameters'].items()}
    raw=np.load(root/'dense_replay/solver.npz');old=np.load(root/'solver.npz');A=sparse.load_npz(root/'constraints.npz')
    for key in ['b','q']:np.testing.assert_array_equal(raw[key],old[key])
    x=raw['x'];w=x[:-1];slack=raw['b']-A@x;m=9482;r=cfg['parameter_budget']*(1-cfg['distance_reserve'])
    errors=dict(nonnegative_violation=max(0.,float(-slack[:m].min())),equality=float(abs(slack[m:m+32]).max()),
        ball_violation=max(0.,float(np.linalg.norm(w)-r)),primal_slack=float(abs(slack-raw['s']).max()))
    assert max(errors.values())<1e-8
    Z,H=d['Z'].numpy(),d['H'].numpy();V,u=d['V'].numpy(),d['axis'].numpy();nb=np.linalg.norm(p['B'])
    F=nb*np.concatenate([Z,d['probe_Z'].numpy()]);scale=np.linalg.norm(F,axis=1);F/=scale[:,None]
    G=d['G'].numpy();gs=np.linalg.norm(G,axis=1).max()
    expected=np.r_[(d['upper'].numpy()/scale-F@w),F@w-d['lower'].numpy()/scale,G@w/gs-x[-1],-V@w,r,w]
    errors['independent_shared_assembly']=float(abs(expected-slack).max());assert errors['independent_shared_assembly']<1e-8
    # A bounded-ball dual upper bound retains stationarity residual explicitly.
    dual=raw['z'];n=len(F);mass=np.maximum(dual[2*n:2*n+10],0).sum();assert mass>0
    lam=np.maximum(dual[:2*n],0)/mass;pi=np.maximum(dual[2*n:2*n+10],0)/mass;nu=dual[m:m+32]/mass
    station=G.T@pi/gs-F.T@(lam[:n]-lam[n:])-V.T@nu
    upper=float(d['upper'].numpy()/scale@lam[:n]-d['lower'].numpy()/scale@lam[n:]+r*np.linalg.norm(station))
    lower=float((G@w/gs).min());assert upper+1e-8>=lower
    edit=(p['B']+nb*u[:,None]*w[None,:]).astype(np.float32);delta=edit.astype(float)-p['B'];D=float(np.linalg.norm(delta)/nb)
    key='model.visual.transformer.resblocks.11.mlp.c_proj.weight'
    torch.save({key:torch.from_numpy(edit)},root/'provisional_weights.pt')
    freeze=dict(weight_sha256=digest(root/'provisional_weights.pt'),solver_sha256=dense['solver_sha256'],
        source_protocol_sha256=digest(protocol),verifier_sha256=digest(__file__),parameter_distance=D,
        coefficient_audit=errors,minimum_normalized_linear_utility=lower,bounded_dual_upper=upper,
        objective_gradient_scale=gs,source_point_scores_seen=False,new_weights=1,
        scope='One quantized coefficient solution frozen before any point classification; not a source pass or defense.')
    save_json(root/'point_freeze.json',freeze);print(json.dumps(dict(stage='sole_point_frozen',**freeze)),flush=True)
    assert D<=cfg['parameter_budget'];del A,F,raw,old
    ids=d['indices'];assert ids==json.loads((root/'source_split.json').read_text())['indices']
    gold=official_labels(cfg['target'],True)[ids];np.testing.assert_array_equal(gold,d['labels'].numpy())
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==initial['checkpoint_sha256']
    A0=parameters(anchor,head)['A'].numpy();del anchor
    actual=parameters(expert,head)
    for k,v in p.items():np.testing.assert_array_equal(v,actual[k].numpy() if torch.is_tensor(actual[k]) else actual[k])
    release=copy.deepcopy(expert)
    with torch.no_grad():release.state_dict()[key].copy_(torch.from_numpy(edit))
    assert [k for k,v in expert.state_dict().items() if not torch.equal(v,release.state_dict()[k])]==[key]
    train=dataset(cfg['target'],'data',expert.val_preprocess,True);before=d['before'].numpy()
    def observe(model,indices,capture=False):
        xx=[];ss=[];block=model.model.visual.transformer.resblocks[11]
        hook=block.mlp.c_fc.register_forward_pre_hook(lambda _m,a:xx.append(a[0][0].detach().cpu())) if capture else None
        try:
            model.cuda().eval();head.cuda().eval()
            with torch.no_grad():
                for images,_ in make_loader(train,indices,128):ss.append(head(model(images.cuda())).cpu().double())
        finally:
            if hook:hook.remove()
            model.cpu();head.cpu()
        return torch.cat(ss).numpy(),torch.cat(xx).double().numpy() if capture else None
    def drift(a,b):
        y=a.argmax(1);g=a[np.arange(len(a)),y,None]-a;mask=np.arange(a.shape[1])[None]!=y[:,None]
        assert (g[mask]>0).all();g[~mask]=1.;ds=b-a;v=(ds[np.arange(len(a)),y,None]-ds)/g;v[~mask]=0
        return float(abs(v).max())
    def metric(a,b,labels):
        y=a.argmax(1);pred=b.argmax(1);c=y==labels;q=pred==labels
        return dict(n=len(y),R=drift(a,b),original_correct=int(c.sum()),correct=int(q.sum()),changed=int((y!=pred).sum()),
            original_NLL=float((logsumexp(a,axis=1)-a[np.arange(len(a)),labels]).mean()),
            NLL=float((logsumexp(b,axis=1)-b[np.arange(len(b)),labels]).mean()),
            per_class=[dict(class_id=j,n=int((labels==j).sum()),lost=int((c&~q&(labels==j)).sum()),
                gained=int((~c&q&(labels==j)).sum())) for j in range(a.shape[1])])
    controls={};original128,X=observe(expert,ids[:128],True)
    controls['original128_replay']=float(abs(original128-before[:128]).max());assert controls['original128_replay']<=cfg['score_tolerance']
    after,_=observe(release,ids);replay,_=observe(release,ids[:128])
    controls['point128_replay']=float(abs(replay-after[:128]).max());assert controls['point128_replay']<=cfg['score_tolerance']
    pn={k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in p.items()}
    source_hidden=H+Z@delta.T;independent=raw_scores(source_hidden,p)
    controls['actual_FP32_vs_NumPy_tail']=float(abs(after-independent).max());assert controls['actual_FP32_vs_NumPy_tail']<=cfg['score_tolerance']
    controls['NumPy_vs_Torch_tail']=float(abs(independent-scores(torch.from_numpy(source_hidden),pn).numpy()).max());assert controls['NumPy_vs_Torch_tail']<1e-8
    B_before=raw_scores(d['probe_H'].numpy(),p);B_after=raw_scores(d['probe_H'].numpy()+d['probe_Z'].numpy()@delta.T,p)
    y=before.argmax(1);contexts=np.array([np.flatnonzero(y==j)[0] for j in range(len(p['head']))]);assert contexts.max()<128
    va=np.linalg.svd(p['A']-A0,full_matrices=False)[2][:32]
    rms=np.sqrt(np.mean((X@va.T)**2,axis=0));shifts=(.01*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*va[None]).reshape(64,-1)
    xx=X[contexts];pre=xx@p['A'].T+p['a'];trial=(xx[:,None]+shifts[None]).reshape(-1,X.shape[1])@p['A'].T+p['a']
    dz=trial*expit(1.702*trial)-(pre*expit(1.702*pre))[:,None,:].repeat(64,axis=1).reshape(trial.shape)
    zz=np.repeat(Z[contexts],64,axis=0)+dz;hh=np.repeat(H[contexts],64,axis=0)+dz@p['B'].T
    A_before=raw_scores(hh,p);A_after=raw_scores(hh+zz@delta.T,p)
    controls['A_probe_NumPy_vs_Torch']=float(abs(A_after-scores(torch.from_numpy(hh+zz@delta.T),pn).numpy()).max());assert controls['A_probe_NumPy_vs_Torch']<1e-8
    ep={k:v.numpy() if torch.is_tensor(v) else v for k,v in d['environment_parameters'].items()};perm=d['permutation'].numpy()
    env=raw_scores(d['environment_H'].numpy()+.5*d['environment_Z'].numpy()@delta[:,perm].T,ep)
    source=metric(before,after,gold);probe=dict(B_n=len(B_before),B_R=drift(B_before,B_after),A_n=len(A_before),A_R=drift(A_before,A_after))
    passed=source['R']<=.5 and source['changed']==0 and max(probe['B_R'],probe['A_R'])<=.5
    np.savez_compressed(root/'source_scores.npz',indices=ids,labels=gold,before=before,after=after,
        B_before=B_before,B_after=B_after,A_before=A_before,A_after=A_after,environment_before=d['environment_before'].numpy(),environment_after=env)
    report=dict(status='source_pass_requires_fresh_confirmation' if passed else 'closed_source_function_failure',source_passed=passed,
        point_freeze=freeze,source=source,task_probes=probe,controls=controls,task32_response_fraction=float(np.linalg.norm(delta@V.T)/np.linalg.norm(delta)),
        public_environment_source_only=metric(d['environment_before'].numpy(),env,gold[:len(env)]),checkpoint_sha256=pins,
        source_scores_sha256=digest(root/'source_scores.npz'),source_sha256=digest(__file__),protocol_sha256=digest(protocol),
        new_FP32_image_forward_instances=4096+256,wall_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        new_confirmation_inputs=0,new_test_inputs=0,new_peer_TA_calls=0,new_full_AGA_calls=0,
        scope='One source-selected point; real FP32 source function and finite latent task probes,not new confirmation/full-task retention or actual peer-fusion evidence. No parameter repair or point selection.')
    save_json(root/'source_validation.json',report);print(json.dumps(report),flush=True)


def line_holdout(root, protocol):
    """Genuinely new train confirmation or complete test gate, no selection."""
    import copy,time
    from clip_benchmark import digest
    from confirm_temperature_compensation import unused
    from covariance_noise import dataset
    from probe_margin_distance import scores
    from subspace_probe import make_loader,save_json
    from task_probe_distance import checked_models,initialize,parameters
    initialize();start=time.monotonic();root=Path(root);cfg=json.loads(Path(protocol).read_text());stage=cfg['stage']
    assert digest(__file__)==cfg['verifier_sha256'] and stage in ['confirmation','full_task']
    source=json.loads((root/'source_validation.json').read_text());assert source['source_passed']
    assert json.loads((root/'source_independent.json').read_text())['status']=='passed'
    assert digest(root/'provisional_weights.pt')==cfg['weight_sha256']==source['point_freeze']['weight_sha256']
    if stage=='full_task':
        assert json.loads((root/'confirmation_validation.json').read_text())['passed']
        assert json.loads((root/'confirmation_independent.json').read_text())['status']=='passed'
    assert not (root/(stage+'_scores.npz')).exists()
    anchor,expert,head,pins=checked_models('MNIST');del anchor
    assert pins==source['checkpoint_sha256'];p=parameters(expert,head)
    release=copy.deepcopy(expert);edit=torch.load(root/'provisional_weights.pt',map_location='cpu');key=next(iter(edit));assert len(edit)==1
    assert key=='model.visual.transformer.resblocks.11.mlp.c_proj.weight'
    release.state_dict()[key].copy_(edit[key]);delta=edit[key].double()-p['B']
    assert float(delta.norm()/p['B'].norm())<=.5
    training=stage=='confirmation';data=dataset('MNIST','data',expert.val_preprocess,training)
    if training:
        construction=json.loads((root/'source_split.json').read_text());exclude=set(construction['indices'])|{41538,27330,31374,46715}
        split=unused(data,exclude,dict(target='MNIST',confirmation_n=512,confirmation_seed=2026102002))
        assert split['excluded_count']>=construction['excluded_count']+len(construction['indices'])
        ids=split['indices'];assert len(ids)==512 and not set(ids)&exclude
        save_json(root/'confirmation_split.json',split)
    else:
        ids=None;assert len(data)==10000
    original=[];changed=[];labels=[];tail_error=0.;captured={}
    v=expert.model.visual
    handles=[v.transformer.resblocks[11].mlp.c_proj.register_forward_pre_hook(lambda _m,a:captured.update(Z=a[0][0].detach().cpu().double())),
             v.ln_post.register_forward_pre_hook(lambda _m,a:captured.update(H=a[0].detach().cpu().double()))]
    expert.cuda().eval();release.cuda().eval();head.cuda().eval()
    try:
        with torch.no_grad():
            for images,y in make_loader(data,ids,128):
                images=images.cuda();a=head(expert(images)).cpu().double();b=head(release(images)).cpu().double()
                expected=scores(captured['H']+captured['Z']@delta.T,p)
                tail_error=max(tail_error,float(abs(expected-b).max()));original.append(a);changed.append(b);labels.append(y)
    finally:
        for handle in handles:handle.remove()
        expert.cpu();release.cpu();head.cpu()
    a,b=torch.cat(original),torch.cat(changed);gold=torch.cat(labels);y=a.argmax(1);gap=a.gather(1,y[:,None])-a
    mask=~torch.nn.functional.one_hot(y,10).bool();assert (gap[mask]>0).all()
    ds=b-a;rho=(ds.gather(1,y[:,None])-ds)/gap.masked_fill(~mask,1);rho[~mask]=0
    worst=float(rho.abs().max());pred=b.argmax(1);assert tail_error<=.0005
    np.savez_compressed(root/(stage+'_scores.npz'),indices=np.arange(len(data)) if ids is None else ids,
        labels=gold.numpy(),before=a.numpy(),after=b.numpy())
    report=dict(stage=stage,passed=worst<=.5 and torch.equal(y,pred),n=len(y),R=worst,
        original_correct=int((y==gold).sum()),correct=int((pred==gold).sum()),changed=int((y!=pred).sum()),
        tail_score_error=tail_error,parameter_distance=float(delta.norm()/p['B'].norm()),
        per_class=[dict(class_id=j,n=int((gold==j).sum()),lost=int(((y==gold)&(pred!=gold)&(gold==j)).sum()),
            gained=int(((y!=gold)&(pred==gold)&(gold==j)).sum())) for j in range(10)],
        scores_sha256=digest(root/(stage+'_scores.npz')),weight_sha256=cfg['weight_sha256'],source_sha256=digest(__file__),
        protocol_sha256=digest(protocol),split_sha256=digest(root/'confirmation_split.json') if training else None,
        historical_test_not_first_blind=not training,new_image_forward_instances=2*len(y),wall_seconds=time.monotonic()-start,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_peer_TA_calls=0,new_AGA_calls=0,
        scope='Frozen sole point; no redraw,repair or threshold change. Full test is a historical evaluation,not first-blind data.')
    save_json(root/(stage+'_validation.json'),report);print(json.dumps(report),flush=True)


def audit_line_scores(root,stage):
    """Independent NumPy class, all-margin and index recount from saved scores."""
    from clip_benchmark import digest
    from subspace_probe import save_json
    from verify_covariance_noise import official_labels
    root=Path(root);name='source' if stage=='source' else stage
    out=root/(name+'_independent.json');assert not out.exists()
    result=json.loads((root/(name+'_validation.json')).read_text());file=root/(name+'_scores.npz')
    assert digest(file)==result['source_scores_sha256' if stage=='source' else 'scores_sha256']
    d=np.load(file);a,b,gold=d['before'],d['after'],d['labels'];ids=d['indices'].tolist()
    np.testing.assert_array_equal(gold,official_labels('MNIST',stage!='full_task')[ids])
    def drift(a,b):
        y=a.argmax(1);old=a[np.arange(len(a)),y,None]-a;new=b[np.arange(len(b)),y,None]-b
        mask=np.arange(10)[None]!=y[:,None];assert (old[mask]>0).all()
        return float(abs((new[mask]-old[mask])/old[mask]).max())
    actual=result['source'] if stage=='source' else result;y,pred=a.argmax(1),b.argmax(1)
    rho=drift(a,b);assert abs(rho-actual['R'])<1e-10
    assert int((pred==gold).sum())==actual['correct'] and int((y==gold).sum())==actual['original_correct']
    assert int((y!=pred).sum())==actual['changed']
    for row in actual['per_class']:
        mask=gold==row['class_id'];assert int(mask.sum())==row['n']
        assert int(((y==gold)&(pred!=gold)&mask).sum())==row['lost']
        assert int(((y!=gold)&(pred==gold)&mask).sum())==row['gained']
    probes={}
    if stage=='source':
        for key in ['A','B']:
            probes[key]=drift(d[key+'_before'],d[key+'_after']);assert abs(probes[key]-result['task_probes'][key+'_R'])<1e-10
        assert ids==json.loads((root/'source_split.json').read_text())['indices']
    elif stage=='confirmation':
        split=json.loads((root/'confirmation_split.json').read_text());assert ids==split['indices']
        assert not set(ids)&set(json.loads((root/'source_split.json').read_text())['indices'])
    else:assert ids==list(range(10000))
    report=dict(status='passed',stage=stage,n=len(ids),R=rho,probe_R=probes,all_class_counts_exact=True,
        score_sha256=digest(file),validation_sha256=digest(root/(name+'_validation.json')),source_sha256=digest(__file__),
        new_model_forwards=0,scope='Independent direct new-gap/old-gap NumPy recount; no denominator floor,parameter change or new data.')
    save_json(out,report);print(json.dumps(report),flush=True)


def verify_structural_unit(root):
    """Independent deletion, affine budget and non-manifold probe reconstruction."""
    from clip_benchmark import digest
    from subspace_probe import save_json
    from task_probe_distance import checked_models,initialize,parameters
    from verify_covariance_noise import official_labels
    from verify_levelset_audits import raw_scores
    initialize();root=Path(root);cfg=json.loads((root/'protocol.json').read_text());result=json.loads((root/'results.json').read_text())
    assert digest(cfg['original_source']['path'])==cfg['original_source']['sha256']
    raw=torch.load(cfg['original_source']['path'],map_location='cpu');p={k:v.numpy() if torch.is_tensor(v) else v for k,v in raw['parameters'].items()}
    Z,H,before=[raw[k].numpy() for k in ['Z','H','before']];j=int(np.argmin(np.max(abs(Z),axis=0)*np.linalg.norm(p['B'],axis=0)))
    assert j==result['unit'];np.testing.assert_array_equal(raw['labels'],official_labels('MNIST',True)[raw['indices']])
    if result['status']=='closed_deletion_parameter_cost':
        cost=(np.linalg.norm(p['A'][j])**2/np.linalg.norm(p['A'])**2+p['a'][j]**2/np.linalg.norm(p['a'])**2)**.5
        assert cost>=.5*(1-1e-6);check=dict(deletion_cost=float(cost))
    else:
        assert digest(root/'necessary_check.pt')==result['raw_sha256'];d=torch.load(root/'necessary_check.pt',map_location='cpu')
        gamma,beta=d['gamma2'].numpy(),d['beta2'].numpy();n=1/gamma;n/=np.linalg.norm(n);c=beta@n
        anchor,expert,head,pins=checked_models('MNIST');assert pins==result['checkpoint_sha256']
        p0={k:v.numpy() if torch.is_tensor(v) else v for k,v in parameters(anchor,head).items()}
        np.testing.assert_array_equal(gamma,expert.model.visual.transformer.resblocks[11].ln_2.weight.double().numpy())
        np.testing.assert_array_equal(beta,expert.model.visual.transformer.resblocks[11].ln_2.bias.double().numpy())
        np.testing.assert_allclose(n,d['n'].numpy(),atol=1e-12);assert abs(c-float(d['c']))<1e-12
        edits={k:v.numpy() for k,v in d['edits'].items()};na=np.linalg.norm(p['A']);nb=np.linalg.norm(p['B']);ba=np.linalg.norm(p['a']);eta=d['eta']
        M=p['gamma'][:,None]*(p['projection']@(p['head']-p['head'].mean(0)).T);M-=M.mean(0)
        u=np.linalg.svd(M,full_matrices=False)[0][:,0];u*=np.sign(u[np.argmax(abs(u))]);np.testing.assert_allclose(u,d['u'],atol=1e-10)
        for key in ['A','B']:
            basis=d['V'+key.lower()].numpy();v=np.linalg.svd(p[key]-p0[key],full_matrices=False)[2][:32]
            np.testing.assert_allclose(basis@basis.T,np.eye(32),atol=1e-10)
            assert np.linalg.norm(basis-basis@v.T@v)<1e-7
        aa=2+(na*c/ba)**2;bb=-2*(p['A'][j]@n)/na+2*na*c*p['a'][j]/ba**2
        cc=np.linalg.norm(p['A'][j])**2/na**2+p['a'][j]**2/ba**2-(.5*(1-1e-6))**2
        assert abs(aa*eta**2+bb*eta+cc)<1e-12
        np.testing.assert_allclose(edits['A_row'],(eta*na*n).astype(np.float32),atol=1e-7,rtol=0)
        assert abs(float(edits['a_value'])-float(np.float32(-eta*na*c)))<1e-7
        np.testing.assert_allclose(edits['B_column'],(p['B'][:,j]+eta*nb*d['u'].numpy()).astype(np.float32),atol=1e-7,rtol=0)
        da=edits['A_row'].astype(float)-p['A'][j];db=edits['B_column'].astype(float)-p['B'][:,j];dc=float(edits['a_value'])-p['a'][j]
        D=(np.linalg.norm(da)**2/na**2+np.linalg.norm(db)**2/nb**2+dc**2/ba**2)**.5;assert abs(D-result['parameter_distance'])<1e-10 and D<=.5
        linear=edits['A_row'].astype(float)*gamma;linear-=linear.mean();offset=edits['A_row'].astype(float)@beta+float(edits['a_value'])
        bound=np.sqrt(len(n))*np.linalg.norm(linear)+abs(offset);assert abs(bound-result['real_arithmetic_quantized_pre_activation_bound'])<1e-10
        ideal=H-Z[:,j,None]*p['B'][:,j];after=raw_scores(ideal,p);ctx=d['contexts'].numpy();shift=d['shifts'].numpy()
        expected_ctx=np.array([np.flatnonzero(before.argmax(1)==j)[0] for j in range(10)]);np.testing.assert_array_equal(ctx,expected_ctx)
        vb=d['Vb'].numpy();rms=np.sqrt(np.mean((Z@vb.T)**2,axis=0))
        expected_shift=(.01*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*vb[None]).reshape(64,-1)
        np.testing.assert_allclose(shift,expected_shift,atol=1e-10)
        B=p['B'].copy();B[:,j]=edits['B_column'];hp=(H[ctx,None]+shift[None]@p['B'].T).reshape(-1,H.shape[1]);hq=(ideal[ctx,None]+shift[None]@B.T).reshape(hp.shape)
        pb,pa=raw_scores(hp,p),raw_scores(hq,p)
        error=max(float(abs(after-d['after'].numpy()).max()),float(abs(pb-d['probe_before'].numpy()).max()),float(abs(pa-d['probe_after'].numpy()).max()));assert error<1e-8
        def rho(a,b):
            y=a.argmax(1);g=a[np.arange(len(a)),y,None]-a;mask=np.arange(10)[None]!=y[:,None];assert (g[mask]>0).all()
            ng=b[np.arange(len(b)),y,None]-b;return float(abs((ng[mask]-g[mask])/g[mask]).max())
        native,probe=rho(before,after),rho(pb,pa);assert abs(native-result['native_ideal_R'])<1e-9 and abs(probe-result['B_probe_R'])<1e-9
        assert int((before.argmax(1)!=after.argmax(1)).sum())==result['native_changed']
        assert result['necessary_source_passed']==(native<=.5 and probe<=.5 and result['native_changed']==0)
        check=dict(parameter_distance=D,NumPy_full_scores_error=error,native_ideal_R=native,B_probe_R=probe,affine_pre_activation_bound=float(bound))
    save_json(root/'independent_validation.json',dict(status='passed',**check,source_sha256=digest(__file__),result_sha256=digest(root/'results.json'),
        original_source_sha256=cfg['original_source']['sha256'],new_images=0,new_candidates=0,scope='Necessary cache-only mechanism check,not actual FP32 task retention,new defense or recovery/peer-fusion evidence.'))


def verify_line_closure(root):
    """Reproduce the exclusion draw and signed-gap diagnosis, no model loads."""
    from clip_benchmark import digest
    from subspace_probe import save_json
    root=Path(root);out=root/'closure_independent.json';assert not out.exists()
    report=json.loads((root/'closure_analysis.json').read_text());split=json.loads((root/'confirmation_split.json').read_text())
    source=json.loads((root/'source_split.json').read_text());groups={}
    for row in json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())['records']:
        groups.setdefault(row['path'],[]).append(row['group'])
    excluded=set(source['indices'])|{41538,27330,31374,46715}
    for name,pin in split['excluded_metadata_sha256'].items():
        path=Path(name);assert digest(path)==pin
        data=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        if name in groups:
            for group in groups[name]:excluded.update(data[group])
        else:
            values=(list(data.values()) if isinstance(data,dict) else [data]) if path.name=='training_indices.json' else [data.get('indices',[])]
            for value in values:
                if isinstance(value,list):excluded.update(int(x) for x in value)
    assert len(excluded)==split['excluded_count']==report['excluded_count']
    ids=np.random.default_rng(2026102002).choice(sorted(set(range(60000))-excluded),512,replace=False).tolist()
    assert ids==split['indices'] and not set(ids)&excluded
    errors={}
    for stage in ['source','confirmation']:
        d=np.load(root/(stage+'_scores.npz'));row=report['groups'][stage];a,b=d['before'],d['after'];y=a.argmax(1)
        mask=np.arange(10)[None]!=y[:,None];old=a[np.arange(len(y)),y,None]-a;new=b[np.arange(len(y)),y,None]-b
        assert (old[mask]>0).all();old[~mask]=1.;change=(new-old)/old;change[~mask]=0.;per_input=abs(change).max(1)
        assert digest(root/(stage+'_scores.npz'))==row['scores_sha256']
        np.testing.assert_allclose(list(row['sample_R_quantiles'].values()),np.quantile(per_input,[.5,.9,.99,1]),rtol=0,atol=1e-10)
        assert int((per_input>.5).sum())==row['inputs_above_budget']
        assert int((abs(change[mask])>.5).sum())==row['margin_pairs_above_budget']
        assert int((change[mask]<-.5).sum())==row['decreased_margin_pairs_above_budget']
        assert int((change[mask]>.5).sum())==row['increased_margin_pairs_above_budget']
        i,k=np.unravel_index(abs(change).argmax(),change.shape);w=row['witness']
        assert i==w['position'] and k==w['competitor'] and int(d['indices'][i])==w['train_index']
        error=max(abs(change[i,k]-w['signed_relative_change']),abs(old[i,k]-w['original_gap']),abs(new[i,k]-w['published_gap']))
        assert error<1e-10;errors[stage]=float(error)
    save_json(out,dict(status='passed',fresh_sampling_reproduced=True,excluded_count=len(excluded),
        signed_drift_distribution_and_witness_errors=errors,closure_sha256=digest(root/'closure_analysis.json'),
        source_sha256=digest(__file__),new_model_forwards=0,new_weight_solutions=0,
        scope='Read-only independent reproduction of saved-score diagnosis and fresh split; confirmation budget failure remains.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--report', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound_validation.json'))
    parser.add_argument('--projection-cone',type=Path)
    parser.add_argument('--rank-one-band',type=Path)
    parser.add_argument('--line-point',type=Path)
    parser.add_argument('--line-holdout',type=Path)
    parser.add_argument('--audit-line',type=Path)
    parser.add_argument('--line-closure',type=Path)
    parser.add_argument('--structural-unit',type=Path)
    parser.add_argument('--stage',choices=['source','confirmation','full_task'])
    parser.add_argument('--protocol',type=Path)
    args = parser.parse_args()
    if args.structural_unit:verify_structural_unit(args.structural_unit)
    elif args.line_closure:verify_line_closure(args.line_closure)
    elif args.audit_line:audit_line_scores(args.audit_line,args.stage)
    elif args.line_holdout:line_holdout(args.line_holdout,args.protocol)
    elif args.line_point:validate_line_point(args.line_point,args.protocol)
    elif args.rank_one_band:verify_rank_one_band(args.rank_one_band)
    elif args.projection_cone:verify_projection_cone(args.projection_cone)
    else:verify(args.source, args.report, args.output)
