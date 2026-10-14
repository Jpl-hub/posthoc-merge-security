"""Compare conditional and free joint publication directions at one frozen point."""

import argparse
import json
import time
from pathlib import Path

import torch

from clip_benchmark import digest
from functional_distance import ClassCompletion
from cross_module_distance import CrossProblem,capture_attention
from angular_distance import angular_scores,fresh_confirmation,signed_residual
from subspace_probe import save_json


def diagnose(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['point_reference']);r=json.loads((root/'results.json').read_text());assert digest(root/'candidate.pt')==cfg['candidate_sha256']==r['candidate_sha256']
    point=torch.load(root/'candidate.pt',map_location='cpu');assert r['target']==target
    coupled=ClassCompletion(target,cfg);p=coupled.problem;theta=point['theta'].cuda().double().requires_grad_(True)
    assert r['checkpoint_sha256']==p.hashes and theta.numel()==p.a_size+p.b_rows*p.b_width
    if cfg.get('coverage_objective'):
        p.stress=lambda s:-(p.weights*p.gap(angular_scores(s)).relu()).sum()/p.unit
    co=theta[:p.a_size].detach().clone().requires_grad_(True);restricted=coupled.coefficients(co)
    assert torch.allclose(restricted,theta,atol=1e-9,rtol=1e-7),'The comparison point is not on the same conditional graph'
    def stress(t):return p.stress(p.predict(p.proxy,p.pp,t,cfg['proxy_beta']))
    free=torch.autograd.grad(stress(theta),theta)[0]
    fdirection=free-(free@theta)/theta.square().sum()*theta;fdirection=fdirection/fdirection.norm()
    rg=torch.autograd.grad(stress(restricted),co,retain_graph=True)[0];normal=torch.autograd.grad(restricted.square().sum(),co)[0]
    cd=rg-(rg@normal)/normal.square().sum().clamp_min(1e-20)*normal
    _,rdirection=torch.autograd.functional.jvp(coupled.coefficients,co,cd);rdirection=rdirection/rdirection.norm()
    baseline=p.metrics(theta);records=[];artifacts={};radius=theta.norm().detach()
    for name,direction in [('free_joint',fdirection),('conditional_graph',rdirection)]:
        residual,derivative=torch.autograd.functional.jvp(p.residuals,theta,direction)
        bounds=torch.full_like(derivative,torch.inf);pos,neg=derivative>0,derivative<0
        bounds[pos]=(cfg['margin_fraction']-residual[pos])/derivative[pos];bounds[neg]=(-cfg['margin_fraction']-residual[neg])/derivative[neg]
        with torch.no_grad():
            h=cfg['derivative_step'];measured=float((stress(theta+h*direction)-stress(theta-h*direction))/(2*h));slope=float(free@direction)
            error=abs(measured-slope)/max(abs(slope),1e-20);assert error<1e-3
            finite=[];saved=[]
            for k in range(cfg['backtracking_steps']):
                step=cfg['normalized_step']*.5**k;trial=theta+step*direction;trial=radius*trial/trial.norm()
                metrics=p.metrics(trial);finite.append({'step':step,**metrics,'feasible':metrics['native_critic']<=cfg['margin_fraction'] and metrics['parameter_distance']<=p.radius*(1+1e-12)});saved.append(trial.cpu())
        records.append({'direction':name,'unit_physical_direction_norm':float(direction.norm()),'distance_tangent_dot':float(direction@theta),'ordering_directional_derivative':slope,'central_difference':measured,'gradient_fd_relative_error':error,'linear_native_step_bound':float(bounds.min()),'finite_steps':finite})
        artifacts[name]={'direction':direction.detach().cpu(),'native_residual':residual.detach().cpu(),'native_directional_derivative':derivative.detach().cpu(),'finite_thetas':saved}
    file=out/'directions.pt';torch.save({'theta':theta.detach().cpu(),'cfg':cfg,'directions':artifacts},file);torch.cuda.synchronize()
    report={'target':target,'status':'mechanism_comparison_only','reference_point_sha256':cfg['candidate_sha256'],'publication_variables':theta.numel(),'conditional_outer_variables':p.a_size,'baseline':baseline,'records':records,'direction_cosine':float(fdirection@rdirection),'ridge':float(coupled.ridge),'checkpoint_sha256':p.hashes,'directions_sha256':digest(file),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'explicit_reverse_gradient_calls':3,'native_JVP_calls':2,'coefficient_JVP_calls':1,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'original_expert_updated':False,'new_release_weights':0,'future_partner_used':False,'test_used':False,'confirmation_used_for_direction_selection':False,'selection_reason':'Compare tangent parameterizations, not select using true fusion performance.','finite_path':'Common physical-distance sphere retraction; the conditional record is a graph tangent, not the curved graph path.','coverage_objective':cfg.get('coverage_objective',False)}
    save_json(out/'direction_comparison.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


def optimize(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['point_reference']);meta=json.loads((root/'results.json').read_text());assert digest(root/'candidate.pt')==cfg['candidate_sha256']==meta['candidate_sha256']
    point=torch.load(root/'candidate.pt',map_location='cpu');p=CrossProblem(target,cfg);assert p.hashes==meta['checkpoint_sha256'] and meta['target']==target
    p.stress=lambda s:-(p.weights*p.gap(angular_scores(s)).relu()).sum()/p.unit
    theta=point['theta'].cuda().double().requires_grad_(True);initial=p.metrics(theta);assert initial['native_critic']<=cfg['margin_fraction'] and initial['parameter_distance']<=p.radius
    radius=theta.norm().detach();best=(initial['objective'],theta.detach().clone(),0);dual=0.;history=[]
    for step in range(cfg['optimization_steps']):
        residual=p.residuals(theta);witness=int(residual.abs().argmax())
        objective=p.objective(theta)-dual*(residual[witness].abs()/cfg['margin_fraction']-1.)
        gradient=torch.autograd.grad(objective,theta)[0]
        gradient=gradient-(gradient@theta)/theta.square().sum()*theta
        with torch.no_grad():
            trial=theta+cfg['normalized_step']*gradient/gradient.norm().clamp_min(1e-20);trial=radius*trial/trial.norm();metrics=p.metrics(trial)
            dual=max(0.,dual+cfg['dual_step']*(metrics['native_critic']/cfg['margin_fraction']-1.));theta.copy_(trial)
            feasible=metrics['native_critic']<=cfg['margin_fraction'] and metrics['parameter_distance']<=p.radius*(1+1e-12)
            if feasible and metrics['objective']>best[0]:best=(metrics['objective'],trial.clone(),step+1)
        history.append({'step':step+1,'dual_multiplier':dual,'native_witness_index':witness,'feasible':feasible,**metrics})
        if (step+1)%8==0:print(json.dumps(history[-1]),flush=True)
    theta=best[1];final=p.metrics(theta);confirmation,labels,split=fresh_confirmation(p,cfg,capture_function=capture_attention);packed=p.prepare(confirmation,p.p,True)
    with torch.no_grad():
        after=p.predict(packed,p.p,theta);rho=float(signed_residual(packed['teacher'],after).abs().max())
        before=packed['teacher'][:len(labels)].argmax(1).cpu();pred=after[:len(labels)].argmax(1).cpu()
        eo=p.na*p.basis['A']['u']@theta[:p.a_size].reshape(p.r,p.r)@p.basis['A']['vh'];eb=p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)
    artifact=out/'candidate.pt';torch.save({**p.cal,'theta':theta.cpu(),'release_deltas':{'O':eo.cpu(),'B':eb.cpu()},'protocol':cfg,'confirmation':confirmation,'confirmation_indices':split['indices'],'confirmation_labels':labels,'confirmation_original_predictions':before,'confirmation_candidate_predictions':pred,'parent_diagnostic_sha256':p.parent_sha},artifact)
    readout=out/'readout.pt';torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in p.class_space.items()},readout)
    result={'target':target,'method':'free_joint_class_completion_bounded_coverage_primal_dual','publication_coefficients':len(theta),'optimization_variables':len(theta),'initial':initial,'selected':final,'selected_step':best[2],'optimizer_updates':cfg['optimization_steps'],'explicit_reverse_gradient_calls':cfg['optimization_steps'],'coefficient_jvp_calls':0,'conditional_linear_solve_calls':0,'dual_updates':cfg['optimization_steps'],'feasible_training_iterates':1+sum(row['feasible'] for row in history),'temporarily_infeasible_optimization_iterates':sum(not row['feasible'] for row in history),'confirmation':{'n':len(labels),'original_correct':int((before==labels).sum()),'candidate_correct':int((pred==labels).sum()),'changed_predictions':int((before!=pred).sum()),'worst_margin_drift':rho},'confirmation_passed':rho<=cfg['margin_fraction'],'candidate_sha256':digest(artifact),'reference_point_sha256':cfg['candidate_sha256'],'checkpoint_sha256':p.hashes,'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'readout_sha256':digest(readout),'original_expert_parameters_updated':False,'gold_labels_used_for_optimization':False,'future_partner_used':False,'test_used':False,'new_release_weights':0,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'global_saddle_optimum_claim':False}
    save_json(out/'results.json',result);save_json(out/'history.json',history);save_json(out/'confirmation_split.json',split);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get('mode')=='optimize':optimize(args.target,cfg)
    else:diagnose(args.target,cfg)
