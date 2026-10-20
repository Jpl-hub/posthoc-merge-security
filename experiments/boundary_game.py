"""Bounded task-level-set publication optimization after a curvature diagnosis."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores,fresh_confirmation,signed_residual
from boundary_retraction import local_matrix
from clip_benchmark import digest
from cross_module_distance import CrossProblem,capture_attention
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json


def risk_first(cache,indices,count,params):
    s=scores(on_device(cache)['h'],params);winner=s.argmax(1);rows=torch.arange(len(s),device='cuda');others=s.clone();others[rows,winner]=-torch.inf
    gap=s[rows,winner]-others.max(1).values;selected=[]
    for c in range(s.shape[1]):
        members=(winner==c).nonzero().flatten();assert len(members)>=count
        selected.extend(members[gap[members].argsort(stable=True)[:count]].cpu().tolist())
    order=selected+[i for i in range(len(s)) if i not in set(selected)]
    return {k:v[order] for k,v in cache.items()},[indices[i] for i in order],len(selected)


def critic_matrix(p,core,theta):
    residual=p.residuals(theta);offset=int(residual.abs().argmax());witness=None
    for group in p.native:
        length=group['teacher'].numel()
        if offset<length:
            row=offset//group['teacher'].shape[1];witness={k:v[row:row+1] for k,v in group.items()};break
        offset-=length
    packed={k:torch.cat([core[k],witness[k]]) for k in core}
    matrix,r=local_matrix(p,packed,theta);u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    tol=torch.finfo(torch.float64).eps*max(matrix.shape)*s[0];rank=int((s>tol).sum())
    return u[:,:rank],s[:rank],vh[:rank],r


def source_search(p,theta,core,cfg):
    """Existing finite tangent/retraction solver, shared by frozen protocols."""
    theta=theta.detach().clone();initial=p.metrics(theta);calls=0;gradients=0;corrections=0;history=[];failures=[]
    limit=cfg['parameter_budget']*(1-cfg.get('distance_reserve',0.))
    def correct(trial):
        nonlocal calls,corrections
        for _ in range(cfg['inner_corrections']):
            before=p.metrics(trial)
            if before['native_critic']<=cfg['margin_fraction']:return trial,before
            if calls>=cfg['maximum_jacobian_calls']:return None
            u,s,vh,r=critic_matrix(p,core,trial);calls+=1;q=-vh.T@((u.T@r)/s)
            accepted=None
            for k in range(cfg['backtracking_steps']):
                proposed=trial+(.5**k)*q
                if float(proposed.norm())>limit*(1+1e-12):continue
                metrics=p.metrics(proposed)
                if metrics['native_critic']<before['native_critic']-1e-12:accepted=proposed;break
            if accepted is None:return None
            trial=accepted;corrections+=1
        metrics=p.metrics(trial)
        return (trial,metrics) if metrics['native_critic']<=cfg['margin_fraction'] else None
    if initial['native_critic']>cfg['margin_fraction']:
        feasible=correct(theta);assert feasible is not None,'Fixed new calibration gate failed at initialization; stop this protocol'
        theta=feasible[0]
    best=(p.metrics(theta)['objective'],theta.clone(),0)
    for iteration in range(cfg['optimization_steps']):
        if calls>=cfg['maximum_jacobian_calls']:failures.append('predeclared_jacobian_cost_cap');break
        leaf=theta.clone().requires_grad_(True);gradient=torch.autograd.grad(p.objective(leaf),leaf)[0];gradients+=1
        u,s,vh,r=critic_matrix(p,core,theta);calls+=1
        direction=gradient-vh.T@(vh@gradient)
        if float(theta.norm())>=cfg['parameter_budget']*cfg['distance_boundary_fraction']:
            radial=theta-vh.T@(vh@theta)
            if float(radial.norm())>1e-14:direction-=radial*(direction@radial)/radial.square().sum()
        if float(direction.norm())<=1e-14:failures.append('no_nonzero_local_task_tangent');break
        direction/=direction.norm();accepted=None;trials=[]
        for k in range(cfg['backtracking_steps']):
            step=cfg['normalized_step']*.5**k;trial=theta+step*direction
            if float(trial.norm())>limit*(1+1e-12):continue
            feasible=correct(trial)
            trials.append({'step':step,'feasible_after_correction':feasible is not None})
            if feasible is not None:accepted=feasible;break
        if accepted is None:failures.append('no_feasible_finite_update');break
        theta,metrics=accepted
        if hasattr(p,'source_trace'):p.source_trace.append(theta.detach().cpu())
        if metrics['objective']>best[0]:best=(metrics['objective'],theta.clone(),iteration+1)
        history.append({'iteration':iteration+1,**metrics,'trials':trials,'jacobian_calls':calls,'correction_updates':corrections})
        if (iteration+1)%8==0:print(json.dumps(history[-1]),flush=True)
    theta=best[1];selected=p.metrics(theta)
    return theta,initial,selected,best[2],calls,gradients,corrections,history,failures


def optimize(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['boundary_reference']);ref=Path(cfg['retraction_reference'])
    assert digest(root/'diagnostic.pt')==cfg['diagnostic_sha256'] and digest(ref/'diagnostic.pt')==cfg['retraction_sha256']
    validation=json.loads((ref/'independent_validation.json').read_text());assert validation['status']=='passed_local_nonlinear_correction'
    boundary=torch.load(root/'diagnostic.pt',map_location='cpu');retracted=torch.load(ref/'diagnostic.pt',map_location='cpu');p=CrossProblem(target,cfg)
    cache,indices,n=risk_first(boundary['screen'],boundary['split']['indices'],cfg['contexts_per_class'],p.p)
    scale,si,ns=risk_first(p.cal['scale'],p.cal['indices']['scale'],cfg['contexts_per_class'],p.p);assert n==ns
    p.cfg['probe_contexts']=n
    p.complete({'native':cache,'scale':scale,'proxy':boundary['proxy']},p.cal['basis_O'],p.cal['basis_B'],{'fit':indices,'scale':si,'proxy':boundary['split']['indices'][:cfg['proxy_n']]})
    p.stress=lambda s:-(p.weights*p.gap(angular_scores(s)).relu()).sum()/p.unit
    core={k:torch.cat([group[k][:n] for group in p.native]) for k in p.native[0]}
    theta,initial,selected,step,calls,gradients,corrections,history,failures=source_search(p,retracted['theta'].cuda().double(),core,cfg)
    confirmation,labels,split=fresh_confirmation(p,cfg,capture_function=capture_attention);packed=p.prepare(confirmation,p.p,True)
    with torch.no_grad():
        after=p.predict(packed,p.p,theta);rho=float(signed_residual(packed['teacher'],after).abs().max())
        before=packed['teacher'][:len(labels)].argmax(1).cpu();pred=after[:len(labels)].argmax(1).cpu()
        eo=p.na*p.basis['A']['u']@theta[:p.a_size].reshape(p.r,p.r)@p.basis['A']['vh'];eb=p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)
    file=out/'candidate.pt';torch.save({**p.cal,'theta':theta.cpu(),'release_deltas':{'O':eo.cpu(),'B':eb.cpu()},'protocol':cfg,'confirmation':confirmation,'confirmation_indices':split['indices'],'confirmation_labels':labels,'confirmation_original_predictions':before,'confirmation_candidate_predictions':pred},file)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in p.class_space.items()},out/'readout.pt')
    report={'target':target,'method':'task_boundary_tangent_with_nonlinear_minimum_norm_retraction','initial':initial,'selected':selected,'selected_step':step,
        'planned_optimization_steps':cfg['optimization_steps'],'accepted_updates':len(history),'explicit_reverse_gradient_calls':gradients,'analytic_task_jacobian_calls':calls,'nonlinear_correction_updates':corrections,'stop_reasons':failures,
        'confirmation':{'n':len(labels),'original_correct':int((before==labels).sum()),'candidate_correct':int((pred==labels).sum()),'changed_predictions':int((before!=pred).sum()),'worst_margin_drift':rho},'confirmation_passed':rho<=cfg['margin_fraction'],
        'candidate_sha256':digest(file),'checkpoint_sha256':p.hashes,'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'original_expert_updated':False,'test_used':False,'partner_used':False,'new_release_weights':0}
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg);save_json(out/'confirmation_split.json',split);save_json(out/'training_indices.json',p.cal['indices'])
    print(json.dumps(report),flush=True)


def recovery_feedback(target,cfg):
    """One source-only joint O/B path through AGA's variable recovery map.

    Reuses source_search unchanged. No closed weights, confirmation, test or
    partner enter this construction; the public environment is only a proxy.
    """
    import copy
    from clip_benchmark import merge
    from covariance_noise import dataset
    from cross_module_distance import extended_parameters,ridge_column_recovery
    from local_residual_transport import cpu,pack,forward
    from mechanisms import load_author
    from task_probe_distance import checked_models
    start=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',cfg);torch.cuda.reset_peak_memory_stats()
    assert '3090' in torch.cuda.get_device_name()
    for name in ['source_split','AGA_reference']:
        assert digest(cfg[name]['path'])==cfg[name]['sha256'],name
    split=json.loads(Path(cfg['source_split']['path']).read_text());ids=split['indices']
    assert len(ids)==cfg['source_n'] and not set(ids)&set(cfg['forbidden_indices'])
    save_json(out/'source_split.json',split)
    anchor,expert,head,pins=checked_models(target)
    p=CrossProblem.__new__(CrossProblem);p.target=target;p.cfg=cfg;p.radius=cfg['parameter_budget']
    p.r=cfg['probe_rank'];p.a_size=p.r**2;p.form='matrix';p.b_width=3072;p.b_rows=768;p.output_basis=None
    p.p=on_device(extended_parameters(expert,head));base=on_device(extended_parameters(anchor,head))
    author,pin=load_author(Path(cfg['author_path']));assert pin==cfg['author_sha256']
    p.mlp_permutation=author['solve_permutation'](p.p['A'].float(),base['A'].float())
    canonical=copy.deepcopy(expert);canonical.load_state_dict(torch.load(cfg['AGA_reference']['path'],map_location='cpu'))
    aligned=on_device(extended_parameters(canonical,head))
    for key in ['A','a']:assert torch.equal(aligned[key],p.p[key][p.mlp_permutation]),key
    assert torch.equal(aligned['B'],p.p['B'][:,p.mlp_permutation])
    environment=merge(anchor,canonical,anchor,cfg['proxy_beta']);p.pp=on_device(extended_parameters(environment,head))
    p.public_O=base['O'];p.recovered_O=ridge_column_recovery(p.p['O'],p.public_O).detach()
    data=dataset(target,'data',expert.val_preprocess,True)
    prior=None
    if cfg.get('source_calibration_reference'):
        ref=cfg['source_calibration_reference'];assert digest(ref['path'])==ref['sha256']
        prior=torch.load(ref['path'],map_location='cpu');old=prior['cache']
        for name,now in [('p',p.p),('base',base),('pp',p.pp)]:
            for key,value in prior[name].items():
                if torch.is_tensor(value):assert torch.equal(value,now[key].cpu()),(name,key)
        old_ids=old['indices']['fit']+old['indices']['scale'];where={v:i for i,v in enumerate(old_ids)}
        assert set(old_ids)==set(ids) and len(where)==len(ids)
        order=torch.tensor([where[i] for i in ids])
        native={k:torch.cat([old['native'][k],old['scale'][k]])[order] for k in old['native']}
        assert old['indices']['proxy']==ids[:cfg['proxy_n']];proxy=old['proxy']
    else:
        native=capture_attention(expert,data,ids);proxy=capture_attention(environment,data,ids[:cfg['proxy_n']])
    # All SVD/edit frames use original parameters and original source activations.
    u,sv,vo=torch.linalg.svd(p.p['O']-base['O'],full_matrices=False);vo=vo[:p.r].clone()
    vb=torch.linalg.svd(p.p['B']-base['B'],full_matrices=False)[2][:p.r].clone()
    H=native['o_input'].cuda();outside=H-(H@vo.T)@vo
    R=torch.linalg.svd(outside,full_matrices=False)[2][:p.r].clone()
    assert float((R@vo.T).abs().max())<1e-8
    p.task_probe_O=vo
    cache={};indices={};n=cfg['contexts_per_class']*len(p.p['head'])
    cut=len(ids)-cfg['scale_n']
    for name,sl in [('native',slice(None,cut)),('scale',slice(cut,None))]:
        part={k:v[sl] for k,v in native.items()};selected=ids[sl]
        cache[name],indices['fit' if name=='native' else 'scale'],count=risk_first(part,selected,cfg['contexts_per_class'],p.p)
        assert count==n
    cache['proxy']=proxy;indices['proxy']=ids[:cfg['proxy_n']]
    p.complete(cache,dict(u=u[:,:p.r].clone(),s=sv[:p.r].clone(),vh=R),dict(vh=vb),indices)
    p.stress=lambda s:-(p.weights*p.gap(angular_scores(s)).relu()).sum()/p.unit
    zero=torch.zeros(p.a_size+p.b_rows*p.b_width,device='cuda',dtype=torch.float64)
    core={k:torch.cat([group[k][:n] for group in p.native]) for k in p.native[0]}
    # The old floor is inactive: require strictly positive genuine teacher gaps.
    minimum_gap=min(float((g['teacher'].gather(1,g['teacher'].argmax(1,keepdim=True))-g['teacher'])
        [~torch.nn.functional.one_hot(g['teacher'].argmax(1),len(p.p['head'])).bool()].min()) for g in p.native)
    assert minimum_gap>1e-8
    controls={}
    head.cuda()
    for name,cache0,pars in [('native',native,p.p),('public_after_AGA',proxy,p.pp)]:
        wanted=head(cache0['actual'].cuda().float()).double()
        error=float((scores(cache0['h'].cuda(),pars)-wanted).abs().max())
        assert error<=cfg['score_tolerance'];controls[name]=error
    if prior:
        for field,key in [('basis_O','u'),('basis_O','vh'),('basis_B','vh')]:
            assert torch.allclose(p.cal[field][key].cpu(),prior['cache'][field][key],atol=1e-12,rtol=1e-10)
        (out/'calibration.pt').symlink_to(Path(cfg['source_calibration_reference']['path']).resolve())
    else:
        torch.save(dict(p=cpu(p.p),base=cpu(base),pp=cpu(p.pp),cache=p.cal,task_probe_O=vo.cpu(),
            permutation=p.mlp_permutation.cpu(),cfg=cfg),out/'calibration.pt')
    leaf=zero.clone().requires_grad_(True);g=torch.autograd.grad(p.objective(leaf),leaf)[0]
    direction=g/g.norm();eps=cfg['derivative_step']
    measured=(p.objective(zero+eps*direction)-p.objective(zero-eps*direction))/(2*eps)
    # At zero every teacher gap is positive. Check the derivative of that
    # actual active max branch: a fixed finite difference may cross a runner-up
    # tie even when autograd is correct. The optimization objective is unchanged.
    original_angular=angular_scores(p.proxy['teacher']);other=original_angular.clone()
    other[p.rows,p.winner]=-torch.inf;runner=other.argmax(1)
    values=[]
    for sign in [-1,1]:
        ss=angular_scores(p.predict(p.proxy,p.pp,sign*eps*direction,.5))
        values.append(-(p.weights*(ss[p.rows,p.winner]-ss[p.rows,runner])).sum()/p.unit)
    active_measured=(values[1]-values[0])/(2*eps)
    derivative_error=float(abs(active_measured-g@direction)/g.norm())
    gradient_check=dict(original_piecewise_measured=float(measured),active_branch_measured=float(active_measured),autograd=float(g@direction),relative_error=derivative_error,
        epsilon=eps,original_objective=float(p.objective(zero)))
    save_json(out/'gradient_check.json',gradient_check);print(json.dumps(dict(stage='gradient_check',**gradient_check)),flush=True)
    torch.save(dict(gradient=g.cpu(),direction=direction.cpu(),plus_scores=p.predict(p.proxy,p.pp,eps*direction,.5).detach().cpu(),
        minus_scores=p.predict(p.proxy,p.pp,-eps*direction,.5).detach().cpu()),out/'gradient_check.pt')
    assert derivative_error<=cfg['derivative_tolerance']
    matrix,residual=local_matrix(p,core,zero)
    y=core['teacher'].argmax(1);keep=~torch.nn.functional.one_hot(y,len(p.p['head'])).bool()
    finite=(signed_residual(core['teacher'],p.predict(core,p.p,eps*direction))-
        signed_residual(core['teacher'],p.predict(core,p.p,-eps*direction)))/(2*eps)
    jvp_error=float((matrix@direction-finite[keep]).abs().max());assert jvp_error<=cfg['derivative_tolerance']
    save_json(out/'initial_controls.json',dict(full_score_errors=controls,objective_gradient_fd_relative_error=derivative_error,
        task_jvp_fd_max_error=jvp_error,minimum_original_class_gap=minimum_gap))
    p.source_trace=[]
    theta,initial,selected,step,calls,gradients,corrections,history,failures=source_search(p,zero,core,cfg)
    deltaO=p.na*p.basis['A']['u']@theta[:p.a_size].reshape(p.r,p.r)@p.basis['A']['vh']
    deltaB=p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)
    q=p.p|dict(O=(p.p['O']+deltaO).float().double(),B=(p.p['B']+deltaB).float().double())
    official=[]
    for weight in [p.p['O'],q['O']]:
        official.append(torch.cat([weight[:,j:j+64].float()@author['solve_linear_transformation'](
            weight[:,j:j+64].float(),p.public_O[:,j:j+64].float()) for j in range(0,768,64)],1).double())
    for original,actual in [(p.p['O'],official[0]),(q['O'],official[1])]:
        err=float((ridge_column_recovery(original,p.public_O)-actual).abs().max())
        assert err<=cfg['author_column_tolerance']
    canonical.model.visual.transformer.resblocks[11].attn.out_proj.weight.data.copy_(official[1].cpu().float())
    canonical.model.visual.transformer.resblocks[11].mlp.c_proj.weight.data.copy_(q['B'][:,p.mlp_permutation].cpu().float())
    final_environment=merge(anchor,canonical,anchor,cfg['proxy_beta'])
    prototype=on_device(extended_parameters(final_environment,head))
    ppdata=pack(proxy,p.pp,vo,vb,cfg)
    # Only natural proxy rows enter the signal; task probes are native controls.
    es=forward(ppdata,p.pp,prototype)[:len(proxy['h'])];before=p.proxy['teacher'];ey=before.argmax(1)
    changed=es.argmax(1)!=ey;perclass=[dict(label=c,n=int((ey==c).sum()),changed=int((changed&(ey==c)).sum())) for c in range(len(p.p['head']))]
    proxy_cache=capture_attention(final_environment,data,ids[:cfg['proxy_n']])
    proxy_actual=head(proxy_cache['actual'].cuda().float()).double()
    proxy_error=float((proxy_actual-es).abs().max());assert proxy_error<=cfg['score_tolerance']
    native_scores=[];actual=[];original=[]
    for name in ['native','scale']:
        cc=cache[name];packed=pack(cc,p.p,vo,vb,cfg)
        native_scores.append(forward(packed,p.p,q));original.append(head(cc['actual'].cuda().float()).double())
    release=copy.deepcopy(expert)
    release.model.visual.transformer.resblocks[11].attn.out_proj.weight.data.copy_(q['O'].cpu().float())
    release.model.visual.transformer.resblocks[11].mlp.c_proj.weight.data.copy_(q['B'].cpu().float())
    for name in ['fit','scale']:
        cc=capture_attention(release,data,indices[name]);actual.append(head(cc['actual'].cuda().float()).double())
    actual=torch.cat(actual);original=torch.cat(original);natural_model=torch.cat([s[:len(cache[k]['h'])] for s,k in zip(native_scores,['native','scale'])])
    full_error=float((actual-natural_model).abs().max());assert full_error<=cfg['score_tolerance']
    natural_R=float(signed_residual(original,actual).abs().max())
    probe_R=max(float(signed_residual(group['teacher'],after).abs().max()) for group,after in zip(p.native,native_scores))
    distances={k:float((q[k]-p.p[k]).norm()/p.p[k].norm()) for k in ['O','B']}
    inside={k:float(((q[k]-p.p[k])@v.T).norm()/p.p[k].norm()) for k,v in [('O',vo),('B',vb)]}
    D=sum(v*v for v in distances.values())**.5
    feasible=D<=cfg['parameter_budget'] and max(natural_R,probe_R)<=cfg['task_budget']
    signal=sum(r['changed']>=max(1,.01*r['n']) for r in perclass)>=len(perclass)//2
    result=dict(status='source_feasible_requires_independent_audit' if feasible and signal else 'closed_recovery_feedback_source_check',
        initial=initial,selected=selected,selected_step=step,accepted_updates=len(history),explicit_gradient_calls=gradients,
        analytic_task_jacobian_calls=calls,normal_corrections=corrections,stop_reasons=failures,
        parameter_distance=D,distances=distances,task32_inside_distances=inside,natural_source_R=natural_R,finite_probe_R=probe_R,
        actual_source_changed=int((actual.argmax(1)!=original.argmax(1)).sum()),public_after_AGA_changes=int(changed.sum()),
        public_after_AGA_per_class=perclass,source_feasible=feasible,source_signal=signal,
        full_model_score_error=full_error,public_full_model_score_error=proxy_error,checkpoint_sha256=pins,official_column_calls=24,new_full_AGA_calls=0,
        original_expert_updates=0,partner_used=False,test_used=False,confirmation_used=False,raw_calibration_sha256=digest(out/'calibration.pt'),
        source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),wall_seconds=time.monotonic()-start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),new_full_model_image_instances=(len(ids)+len(proxy['h']))*(1 if prior else 2),
        limitation='A target/base source-only recovery-feedback experiment,not a new effective defense,global saddle solution or unknown-partner guarantee.')
    torch.save(dict(theta=theta.cpu(),q=cpu({k:q[k] for k in ['O','B']}),
        prototype=cpu({k:prototype[k] for k in ['O','B']}),official_O=[x.cpu() for x in official],original=original.cpu(),actual=actual.cpu(),
        source_scores=[x.cpu() for x in native_scores],environment_scores=es.cpu(),environment_actual=proxy_actual.cpu(),
        initial_gradient=g.detach().cpu(),source_trace=p.source_trace,cfg=cfg),out/'training_point.pt')
    result['raw_point_sha256']=digest(out/'training_point.pt')
    if feasible and signal:torch.save({'model.visual.transformer.resblocks.11.attn.out_proj.weight':q['O'].cpu().float(),
        'model.visual.transformer.resblocks.11.mlp.c_proj.weight':q['B'].cpu().float()},out/'provisional_weights.pt')
    save_json(out/'history.json',history);save_json(out/'training_indices.json',indices);save_json(out/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);parser.add_argument('--recovery-feedback',action='store_true');args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (recovery_feedback if args.recovery_feedback else optimize)(args.target,cfg)
