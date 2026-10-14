"""Compare conditional and free joint publication directions at one frozen point."""

import argparse
import json
import time
from pathlib import Path

import torch

from clip_benchmark import digest
from functional_distance import ClassCompletion
from subspace_probe import save_json


def diagnose(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['point_reference']);r=json.loads((root/'results.json').read_text());assert digest(root/'candidate.pt')==cfg['candidate_sha256']==r['candidate_sha256']
    point=torch.load(root/'candidate.pt',map_location='cpu');assert r['target']==target
    coupled=ClassCompletion(target,cfg);p=coupled.problem;theta=point['theta'].cuda().double().requires_grad_(True)
    assert r['checkpoint_sha256']==p.hashes and theta.numel()==p.a_size+p.b_rows*p.b_width
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
    report={'target':target,'status':'mechanism_comparison_only','reference_point_sha256':cfg['candidate_sha256'],'publication_variables':theta.numel(),'conditional_outer_variables':p.a_size,'baseline':baseline,'records':records,'direction_cosine':float(fdirection@rdirection),'ridge':float(coupled.ridge),'checkpoint_sha256':p.hashes,'directions_sha256':digest(file),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'explicit_reverse_gradient_calls':3,'native_JVP_calls':2,'coefficient_JVP_calls':1,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'original_expert_updated':False,'new_release_weights':0,'future_partner_used':False,'test_used':False,'confirmation_used_for_direction_selection':False,'selection_reason':'Compare tangent parameterizations, not select using true fusion performance.'}
    save_json(out/'direction_comparison.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;diagnose(args.target,cfg)
