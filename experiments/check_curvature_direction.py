"""One finite feasibility check of the frozen restricted max-min direction."""

import argparse
import json
import time
from pathlib import Path

import torch

from clip_benchmark import digest
from ffn_levelset import FFNProblem
from subspace_probe import save_json


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    source=Path(cfg['direction_reference'])/'diagnostic.pt';assert digest(source)==cfg['direction_sha256']
    raw=torch.load(source,map_location='cpu');p=FFNProblem(cfg['target'],cfg)
    direction=(raw['krylov_basis']@raw['direction_in_fixed_basis']).cuda().double();direction/=direction.norm()
    n=len(p.cache['native']['h']);teacher=p.native[0]['teacher'][:n];y=teacher.argmax(1);calls=0
    def metrics(theta):
        with torch.no_grad():
            mid=p.predict(p.native[0],p.p,theta,.5)[:n];changed=mid.argmax(1)!=y
            return {'distance':float(theta.norm()),'task_worst_margin_drift':float(p.residuals(theta).abs().max()),
                'natural_n':n,'natural_midpoint_changed':int(changed.sum()),
                'midpoint_changed_by_teacher_class':torch.bincount(y[changed],minlength=teacher.shape[1]).cpu().tolist()}
    finite=[]
    for k in range(cfg['backtracking_steps']):
        step=cfg['parameter_budget']*.5**k;finite.append({'step':step,**metrics(step*direction)})
    first=next((row for row in finite if row['task_worst_margin_drift']<=cfg['margin_fraction']),None)
    theta=None;history=[]
    if first:
        theta=min(2*first['step'],cfg['parameter_budget'])*direction
        for _ in range(cfg['correction_steps']):
            before=metrics(theta)
            if before['task_worst_margin_drift']<=cfg['margin_fraction']:break
            u,s,vh,r=p.critic(theta);calls+=1;delta=-vh.T@((u.T@r)/s);accepted=None
            for k in range(cfg['backtracking_steps']):
                proposed=theta+(.5**k)*delta
                if float(proposed.norm())>cfg['parameter_budget']:continue
                after=metrics(proposed)
                if after['task_worst_margin_drift']<before['task_worst_margin_drift']-1e-12:accepted=(proposed,after);break
            history.append({'before':before,'correction_norm':float(delta.norm()),'accepted':accepted is not None,'after':accepted[1] if accepted else None})
            if accepted is None:break
            theta=accepted[0]
    final=metrics(theta) if theta is not None else None
    torch.save({'theta':theta.cpu() if theta is not None else None,'direction':direction.cpu(),'cfg':cfg},out/'diagnostic.pt')
    result={'status':'frozen_max_min_curvature_finite_check','target':cfg['target'],'finite_steps':finite,'first_feasible':first,'correction_history':history,'final':final,
        'analytic_task_jacobian_calls':calls,'publication_optimization_steps':0,'diagnostic_correction_steps':len(history),
        'new_model_forwards':0,'new_proxy_models':0,'new_release_weights':0,'partner_used':False,'test_used':False,'confirmation_used':False,
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'diagnostic_sha256':digest(out/'diagnostic.pt'),
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'limitation':'One fixed class-balanced curvature direction; finite feasibility and classification counts only. No publication optimization, global saddle or universal inability claim.'}
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;run(cfg)
