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
    theta=retracted['theta'].cuda().double();initial=p.metrics(theta);calls=0;gradients=0;corrections=0;history=[];failures=[]
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
                if float(proposed.norm())>cfg['parameter_budget']*(1+1e-12):continue
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
            if float(trial.norm())>cfg['parameter_budget']*(1+1e-12):continue
            feasible=correct(trial)
            trials.append({'step':step,'feasible_after_correction':feasible is not None})
            if feasible is not None:accepted=feasible;break
        if accepted is None:failures.append('no_feasible_finite_update');break
        theta,metrics=accepted
        if metrics['objective']>best[0]:best=(metrics['objective'],theta.clone(),iteration+1)
        history.append({'iteration':iteration+1,**metrics,'trials':trials,'jacobian_calls':calls,'correction_updates':corrections})
        if (iteration+1)%8==0:print(json.dumps(history[-1]),flush=True)
    theta=best[1];selected=p.metrics(theta)
    confirmation,labels,split=fresh_confirmation(p,cfg,capture_function=capture_attention);packed=p.prepare(confirmation,p.p,True)
    with torch.no_grad():
        after=p.predict(packed,p.p,theta);rho=float(signed_residual(packed['teacher'],after).abs().max())
        before=packed['teacher'][:len(labels)].argmax(1).cpu();pred=after[:len(labels)].argmax(1).cpu()
        eo=p.na*p.basis['A']['u']@theta[:p.a_size].reshape(p.r,p.r)@p.basis['A']['vh'];eb=p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)
    file=out/'candidate.pt';torch.save({**p.cal,'theta':theta.cpu(),'release_deltas':{'O':eo.cpu(),'B':eb.cpu()},'protocol':cfg,'confirmation':confirmation,'confirmation_indices':split['indices'],'confirmation_labels':labels,'confirmation_original_predictions':before,'confirmation_candidate_predictions':pred},file)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in p.class_space.items()},out/'readout.pt')
    report={'target':target,'method':'task_boundary_tangent_with_nonlinear_minimum_norm_retraction','initial':initial,'selected':selected,'selected_step':best[2],
        'planned_optimization_steps':cfg['optimization_steps'],'accepted_updates':len(history),'explicit_reverse_gradient_calls':gradients,'analytic_task_jacobian_calls':calls,'nonlinear_correction_updates':corrections,'stop_reasons':failures,
        'confirmation':{'n':len(labels),'original_correct':int((before==labels).sum()),'candidate_correct':int((pred==labels).sum()),'changed_predictions':int((before!=pred).sum()),'worst_margin_drift':rho},'confirmation_passed':rho<=cfg['margin_fraction'],
        'candidate_sha256':digest(file),'checkpoint_sha256':p.hashes,'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'original_expert_updated':False,'test_used':False,'partner_used':False,'new_release_weights':0}
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg);save_json(out/'confirmation_split.json',split);save_json(out/'training_indices.json',p.cal['indices'])
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;optimize(args.target,cfg)
