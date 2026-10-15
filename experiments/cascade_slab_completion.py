"""Task-margin slab projection on a fixed nonlinear publication branch."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import LinearConstraint,minimize

from cascade_completion import task_metrics
from cascade_game import Cascade
from clip_benchmark import digest
from subspace_probe import save_json


@torch.no_grad()
def slab(p,trial,cfg,counters):
    history=[]
    for _ in range(cfg['completion_corrections']):
        before=task_metrics(p,trial)
        if before['distance']>cfg['parameter_budget']:return None,history
        if before['worst_margin_drift']<=cfg['task_budget']:return trial,history
        if counters['jacobians']>=cfg['completion_jacobian_cap']:return None,history
        *_,r,packed,matrix=p.critic(trial);counters['jacobians']+=1;matrix=matrix[:,p.size:]
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False);rank=int((s>torch.finfo(s.dtype).eps*max(matrix.shape)*s[0]).sum())
        u,s,vh=u[:,:rank],s[:rank],vh[:rank];a=(u*s[None]).cpu().numpy();rr=r.cpu().numpy()
        k=trial[p.size:];center=(vh@k).cpu().numpy();null_square=max(float(k.square().sum())-float(center@center),0.)
        remaining=cfg['parameter_budget']**2-float(trial[:p.size].square().sum())-null_square
        if remaining<=0:return None,history
        row_norm=np.maximum(np.linalg.norm(a,axis=1),1e-12);bound=cfg['task_budget']*cfg['linear_slab_interior_fraction']
        linear=LinearConstraint(a/row_norm[:,None],(-bound-rr)/row_norm,(bound-rr)/row_norm)
        ball=dict(type='ineq',fun=lambda y:remaining-np.sum((center+y)**2),jac=lambda y:-2*(center+y))
        solved=minimize(lambda y:.5*float(y@y),np.zeros(rank),jac=lambda y:y,method='SLSQP',constraints=[linear,ball],
                        options=dict(maxiter=cfg['slab_solver_steps'],ftol=1e-11))
        error=max(float(np.abs(a@solved.x+rr).max())-bound,0.)
        ball_error=max(float(np.sum((center+solved.x)**2))-remaining,0.)
        if not np.isfinite(solved.x).all() or error>cfg['slab_linear_tolerance'] or ball_error>cfg['slab_linear_tolerance']:
            history.append(dict(before=before,solver_success=bool(solved.success),solver_message=solved.message,linear_constraint_excess=error,ball_excess=ball_error,accepted=False));return None,history
        delta=vh.T@torch.tensor(solved.x,device='cuda');accepted=None
        for fraction in [.5**k for k in range(cfg['backtracking_steps'])]:
            t=trial.clone();t[p.size:]+=fraction*delta
            if t.norm()>cfg['parameter_budget']:continue
            measured=task_metrics(p,t)
            if measured['worst_margin_drift']<before['worst_margin_drift']-1e-12:accepted=t;break
        history.append(dict(before=before,correction_norm=float(delta.norm()),solver_steps=int(solved.nit),solver_success=bool(solved.success),
                            linear_constraint_excess=error,ball_excess=ball_error,accepted=accepted is not None))
        if accepted is None:return None,history
        trial=accepted;counters['corrections']+=1
    return (trial if task_metrics(p,trial)['worst_margin_drift']<=cfg['task_budget'] else None),history


def construct(cfg,problem_class=Cascade):
    out=Path(cfg['completion_output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=problem_class(cfg)
    previous=json.loads((Path(cfg['equality_completion_reference'])/'results.json').read_text())
    assert previous['status']=='closed_source_continuation_no_multiclass_fusion_effect'
    ray=(-torch.diag(p.basis['s'])/p.basis['s'].norm()).flatten();theta=p.zero.clone();amplitude=0.;history=[];counters=dict(jacobians=0,corrections=0)
    step=cfg['parameter_budget']/cfg['continuation_steps'];reason='continuation_step_cap'
    first_step=0
    if cfg.get('resume_point'):
        assert digest(cfg['resume_point'])==cfg['resume_sha256']
        resumed=torch.load(cfg['resume_point'],map_location='cpu');theta=resumed['theta'].cuda();amplitude=resumed['upstream_amplitude'];first_step=round(amplitude/step)
        counters.update(cfg['prior_construction_counters'])
    for iteration in range(first_step,cfg['continuation_steps']):
        trial=theta.clone();proposal=amplitude+step;trial[:p.size]=proposal*ray
        accepted,correction_history=slab(p,trial,cfg,counters)
        history.append(dict(iteration=iteration+1,upstream_distance=proposal,accepted=accepted is not None,corrections=correction_history))
        if accepted is None:reason='no_source_feasible_margin_slab_completion';break
        theta=accepted;amplitude=proposal
        print(json.dumps(dict(iteration=iteration+1,upstream_distance=amplitude,**task_metrics(p,theta),**counters)),flush=True)
    file=out/'point.pt';torch.save(dict(theta=theta.cpu(),upstream_amplitude=amplitude,cfg=cfg),file)
    frozen_sha=digest(file);metrics=p.metrics(theta)
    signal=metrics['proxy_changed']>=cfg['minimum_finite_changes'] and sum(v>0 for v in metrics['changed_by_teacher_class'])>=cfg['minimum_changed_classes']
    report=dict(target=cfg['target'],status='finite_slab_cascade_completion' if signal else 'closed_slab_cascade_no_multiclass_fusion_effect',
                method='source_only_nonlinear_attention_cascade_margin_slab_completion',selected=metrics,upstream_distance=amplitude,
                downstream_distance=float(theta[p.size:].norm()),accepted_continuation_steps=first_step+sum(x['accepted'] for x in history),stop_reason=reason,
                finite_multiclass_gate_passed=signal,**counters,point_sha256=frozen_sha,source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                construction_proxy_used=False,first_proxy_observation_after_point_sha256=frozen_sha,
                parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],test_used=False,partner_used=False,confirmation_used=False,
                original_expert_updates=0,explicit_reverse_gradient_calls=0,new_release_weights=0,wall_seconds=time.monotonic()-start)
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',p.indices);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;construct(cfg)
