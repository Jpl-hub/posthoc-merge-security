"""One source-only task-tangent curvature direction and finite correction test."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from clip_benchmark import digest
from ffn_levelset import FFNProblem
from subspace_probe import save_json


def lanczos(operator, seed, steps):
    basis=[];alphas=[];betas=[];v=seed/seed.norm();previous=torch.zeros_like(v);beta=torch.zeros((),device=v.device,dtype=v.dtype)
    for _ in range(steps):
        basis.append(v);w=operator(v)-beta*previous;alpha=v@w;w-=alpha*v
        # Full reorthogonalization keeps a small symmetric Ritz problem reliable.
        for _ in range(2):
            for q in basis:w-=(q@w)*q
        alphas.append(alpha);newbeta=w.norm()
        if float(newbeta)<1e-12:break
        betas.append(newbeta);previous,v=v,w/newbeta;beta=newbeta
    tri=torch.diag(torch.stack(alphas));n=len(alphas)
    if n>1:
        edge=torch.stack(betas[:n-1]);tri+=torch.diag(edge,1)+torch.diag(edge,-1)
    eig,vec=torch.linalg.eigh(tri);q=torch.stack(basis,1);direction=q@vec[:,-1]
    return direction/direction.norm(),eig,tri,q


def diagnose(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    p=FFNProblem(cfg['target'],cfg);zero=torch.zeros(p.a_size+p.b_rows*p.b_width,device='cuda',dtype=torch.float64)
    _,_,v,_=p.critic(zero);n=cfg['contexts_per_class']*p.p['head'].shape[0]
    core={k:torch.cat([group[k][:n] for group in p.native]) for k in p.native[0]}
    teacher=core['teacher'];winner=teacher.argmax(1);rows=torch.arange(len(winner),device='cuda')
    keep=torch.ones_like(teacher,dtype=torch.bool);keep[rows,winner]=False
    gap=(teacher[rows,winner,None]-teacher).masked_fill(~keep,1.)
    classes=len(p.p['head']);counts=torch.bincount(winner,minlength=classes);weights=1./counts[winner].double()/classes/(classes-1)
    def objective(theta):
        s=p.predict(core,p.p,theta);g=s[rows,winner,None]-s
        return (weights[:,None]*(g/gap).masked_fill(~keep,0)).sum()
    leaf=zero.clone().requires_grad_(True);value=objective(leaf);gradient=torch.autograd.grad(value,leaf,create_graph=True)[0]
    project=lambda d:d-v.T@(v@d)
    calls=0
    def op(direction):
        nonlocal calls
        direction=project(direction);hv=torch.autograd.grad(gradient,leaf,grad_outputs=direction,retain_graph=True)[0];calls+=1
        return project(hv).detach()
    seed=torch.zeros_like(zero);seed[:p.a_size]=torch.diag(p.basis['A']['s']/p.na).flatten();seed=project(seed)
    assert float(seed.norm())>1e-12,'Task-vector seed vanished in the finite task tangent; stop, do not search seeds.'
    direction,eigs,tri,krylov=lanczos(op,seed,cfg['lanczos_steps']);hd=op(direction)
    curvature=float(direction@hd);residual=float((hd-curvature*direction).norm())
    h=cfg['curvature_fd_step'];fd=float((objective(h*direction)-2*objective(zero)+objective(-h*direction))/h**2)
    assert abs(fd-curvature)/max(abs(curvature),1)<cfg['hessian_relative_tolerance']
    def metrics(theta):
        with torch.no_grad():
            native=p.predict(p.native[0],p.p,theta);before=p.native[0]['teacher'];half=p.predict(p.native[0],p.p,theta,.5)
            y=before.argmax(1);changed=half.argmax(1)!=y
            return {'distance':float(theta.norm()),'task_worst_margin_drift':float(p.residuals(theta).abs().max()),
                'source_n':len(y),'source_midpoint_changed':int(changed.sum()),
                'midpoint_changed_by_teacher_class':torch.bincount(y[changed],minlength=classes).cpu().tolist(),
                'core_average_relative_endpoint_margin':float(objective(theta)),
                'core_average_relative_midpoint_margin':float(objective(.5*theta))}
    finite=[]
    for k in range(cfg['backtracking_steps']):
        step=cfg['parameter_budget']*.5**k;finite.append({'step':step,**metrics(step*direction)})
    first=next((row for row in finite if row['task_worst_margin_drift']<=cfg['margin_fraction']),None)
    history=[];theta=None;jcalls=1
    if first is not None:
        theta=min(2*first['step'],cfg['parameter_budget'])*direction
        for _ in range(cfg['correction_steps']):
            before=metrics(theta)
            if before['task_worst_margin_drift']<=cfg['margin_fraction']:break
            u,s,vh,r=p.critic(theta);jcalls+=1;q=-vh.T@((u.T@r)/s);accepted=None
            for k in range(cfg['backtracking_steps']):
                trial=theta+(.5**k)*q
                if float(trial.norm())>cfg['parameter_budget']:continue
                m=metrics(trial)
                if m['task_worst_margin_drift']<before['task_worst_margin_drift']-1e-12:accepted=(trial,m);break
            history.append({'before':before,'correction_norm':float(q.norm()),'accepted':accepted is not None,'after':accepted[1] if accepted else None})
            if accepted is None:break
            theta=accepted[0]
    class_curvature=[]
    with torch.no_grad():
        before=p.predict(core,p.p,zero);plus=p.predict(core,p.p,h*direction);minus=p.predict(core,p.p,-h*direction)
        for cls in range(classes):
            member=winner==cls
            relp=1+signed_residual(before[member],plus[member]);relm=1+signed_residual(before[member],minus[member])
            local_keep=keep[member]
            # signed_residual is the relative margin change; the masked winner
            # term is excluded, keeping the identical objective denominator.
            d2=((relp+relm-2)/h**2)[local_keep].mean()
            class_curvature.append({'class':cls,'mean_relative_margin_curvature':float(d2)})
    final=metrics(theta) if theta is not None else None
    file=out/'diagnostic.pt';torch.save({'direction':direction.cpu(),'theta':theta.cpu() if theta is not None else None,
        'task_row_basis':v.cpu(),'projected_hessian_direction':hd.cpu(),'krylov_basis':krylov.cpu(),'tridiagonal':tri.cpu(),
        'core':{k:v.cpu() for k,v in core.items()},'native':p.cache['native'],'parameters':{k:v.cpu() if torch.is_tensor(v) else v for k,v in p.p.items()},
        'basis_A':{k:v.cpu() for k,v in p.basis['A'].items()},'output_basis':p.q.cpu(),'cfg':cfg},file)
    result={'status':'source_only_task_tangent_curvature_diagnosis','target':cfg['target'],'projected_gradient_norm':float(project(gradient.detach()).norm()),
        'task_tangent_rank':v.shape[0],'coefficient_dimension':len(zero),'lanczos_steps':tri.shape[0],'hessian_vector_calls':calls,
        'largest_Ritz_value':float(eigs[-1]),'direction_curvature':curvature,'Ritz_residual_norm':residual,
        'directional_curvature_finite_difference':fd,'class_curvature':class_curvature,'finite_steps':finite,'first_feasible':first,
        'correction_history':history,'final':final,'analytic_task_jacobian_calls':jcalls,
        'parameter_budget':cfg['parameter_budget'],'task_budget':cfg['margin_fraction'],'new_model_forwards':0,'new_release_weights':0,
        'publication_optimization_steps':0,'diagnostic_correction_steps':len(history),'proxy_models_used':0,'partner_used':False,'test_used':False,'confirmation_used':False,
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'diagnostic_sha256':digest(file),
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'limitation':'One deterministic task-seeded finite Krylov direction, not a global Hessian extremum, proof of infeasibility, full worst-class optimization, or unknown-fusion transfer guarantee.'}
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;diagnose(cfg)
