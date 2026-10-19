"""Finite class-compensation necessary bound on frozen source data; no candidate."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


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
    save_json(out/'derivative_check.json',dict(relative_error=error,step=step,gradient_norm=gn))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    parser.add_argument('--normalization-range',action='store_true')
    parser.add_argument('--projection-cone',type=Path)
    args = parser.parse_args()
    if args.projection_cone:projection_cone(args.projection_cone)
    else:(normalization_range if args.normalization_range else check)(args.source, args.output)
