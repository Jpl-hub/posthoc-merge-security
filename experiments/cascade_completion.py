"""Source-only finite continuation with a locked upstream task-SVD ray."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from cascade_game import Cascade,cpu
from clip_benchmark import digest
from subspace_probe import save_json


@torch.no_grad()
def task_metrics(p,theta):
    scores=p.predict(p.native,p.p,theta);residual=signed_residual(p.native['teacher'],scores)
    return dict(distance=float(theta.norm()),worst_margin_drift=float(residual.abs().max()),
                natural_changed=int((scores[:p.natural_n].argmax(1)!=p.native['teacher'][:p.natural_n].argmax(1)).sum()))


@torch.no_grad()
def complete(p,trial,cfg,counters):
    history=[]
    for _ in range(cfg['completion_corrections']):
        before=task_metrics(p,trial)
        if before['distance']>cfg['parameter_budget']:return None,history
        if before['worst_margin_drift']<=cfg['task_budget']:return trial,history
        if counters['jacobians']>=cfg['completion_jacobian_cap']:return None,history
        # Keep the finite upstream map fixed: only the downstream compensator
        # is allowed to repair task responses. No proxy enters this solve.
        *_,r,packed,matrix=p.critic(trial);counters['jacobians']+=1
        matrix=matrix[:,p.size:];u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
        rank=int((s>torch.finfo(s.dtype).eps*max(matrix.shape)*s[0]).sum())
        delta=-vh[:rank].T@((u[:,:rank].T@r)/s[:rank]);accepted=None
        for k in range(cfg['backtracking_steps']):
            t=trial.clone();t[p.size:]+=(.5**k)*delta
            if t.norm()>cfg['parameter_budget']:continue
            measured=task_metrics(p,t)
            if measured['worst_margin_drift']<before['worst_margin_drift']-1e-12:accepted=t;break
        history.append(dict(before=before,correction_norm=float(delta.norm()),accepted=accepted is not None))
        if accepted is None:return None,history
        trial=accepted;counters['corrections']+=1
    return (trial if task_metrics(p,trial)['worst_margin_drift']<=cfg['task_budget'] else None),history


def construct(cfg):
    out=Path(cfg['completion_output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=Cascade(cfg)
    old=Path(cfg['diagnostic_output']);validation=json.loads((old/'independent_validation.json').read_text())
    assert validation['status']=='passed_closed_finite_construction'
    ray=(-torch.diag(p.basis['s'])/p.basis['s'].norm()).flatten();theta=p.zero.clone();amplitude=0.;history=[];counters=dict(jacobians=0,corrections=0)
    step=cfg['parameter_budget']/cfg['continuation_steps'];reason='continuation_step_cap'
    for iteration in range(cfg['continuation_steps']):
        trial=theta.clone();proposal=amplitude+step;trial[:p.size]=proposal*ray
        accepted,correction_history=complete(p,trial,cfg,counters)
        history.append(dict(iteration=iteration+1,upstream_distance=proposal,accepted=accepted is not None,corrections=correction_history))
        if accepted is None:reason='no_source_feasible_fixed_upstream_completion';break
        theta=accepted;amplitude=proposal
        row=dict(iteration=iteration+1,upstream_distance=amplitude,**task_metrics(p,theta),**counters);print(json.dumps(row),flush=True)
    # Selection is now final. Proxy responses are observed for the first time
    # after this source-only continuation, never used to pick an earlier step.
    file=out/'point.pt';torch.save(dict(theta=theta.cpu(),upstream_amplitude=amplitude,cfg=cfg),file)
    frozen_sha=digest(file);metrics=p.metrics(theta)
    finite_gate=metrics['proxy_changed']>=cfg['minimum_finite_changes'] and sum(v>0 for v in metrics['changed_by_teacher_class'])>=cfg['minimum_changed_classes']
    report=dict(target=cfg['target'],status='finite_fixed_upstream_completion' if finite_gate else 'closed_source_continuation_no_multiclass_fusion_effect',
                method='source_only_nonlinear_attention_cascade_completion',selected=metrics,upstream_distance=amplitude,
                downstream_distance=float(theta[p.size:].norm()),accepted_continuation_steps=sum(x['accepted'] for x in history),stop_reason=reason,
                finite_multiclass_gate_passed=finite_gate,**counters,point_sha256=frozen_sha,source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                construction_proxy_used=False,first_proxy_observation_after_point_sha256=frozen_sha,
                parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],test_used=False,partner_used=False,confirmation_used=False,
                original_expert_updates=0,explicit_reverse_gradient_calls=0,new_release_weights=0,wall_seconds=time.monotonic()-start)
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',p.indices)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;construct(cfg)
