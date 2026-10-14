"""One target-only nonlinear task-level-set correction diagnosis."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores,signed_residual
from clip_benchmark import digest
from cross_module_distance import CrossProblem,normalized_input
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import gelu


def local_matrix(problem,packed,theta):
    p=problem;co=theta[:p.a_size].reshape(p.r,p.r);cb=theta[p.a_size:].reshape(p.b_rows,p.b_width)
    eb=p.nb*p.output_basis@cb
    rr=packed['r']+p.na*(packed['task_input']@co.T)@p.basis['A']['u'].T
    z=gelu(normalized_input(rr,p.p)@p.p['A'].T+p.p['a'])+packed['b_offset']
    h=packed['rounding']+rr+z@(p.p['B']+eb).T+p.p['b']
    pars=p.p|{'B':p.p['B']+eb};jh,jr,_,after=raw_jacobians({'r':rr,'h':h},pars)
    first=p.na*((jr@p.basis['A']['u'])[:,:,:,None]*packed['task_input'][:,None,None,:])
    second=p.nb*((jh@p.output_basis)[:,:,:,None]*z[:,None,None,:])
    j=torch.cat([first.flatten(2),second.flatten(2)],2)
    teacher=packed['teacher'];winner=teacher.argmax(1);rows=torch.arange(len(winner),device='cuda')
    keep=torch.ones_like(teacher,dtype=torch.bool);keep[rows,winner]=False
    gap=(teacher[rows,winner,None]-teacher).masked_fill(~keep,1.)
    assert float(gap[keep].min())>1e-8
    matrix=(j[rows,winner,None,:]-j)/gap[:,:,None]
    residual=signed_residual(teacher,after)
    return matrix[keep],residual[keep]


def diagnose(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['boundary_reference']);report=json.loads((root/'results.json').read_text());valid=json.loads((root/'independent_validation.json').read_text())
    assert valid['status']=='passed' and valid['diagnostic_sha256']==report['diagnostic_sha256']==digest(root/'diagnostic.pt')==cfg['diagnostic_sha256']
    data=torch.load(root/'diagnostic.pt',map_location='cpu');p=CrossProblem(target,cfg);assert p.hashes==report['checkpoint_sha256']
    critical=data['critical_offsets'];cache=data['screen'];risk={k:v[critical] for k,v in cache.items()};p.cfg['probe_contexts']=len(critical)
    p.native=[p.prepare(cache,p.p,False),p.prepare(risk,p.p,True)];core=p.prepare(risk,p.p,False)
    p.proxy=p.prepare(data['proxy'],p.pp,False);teacher=p.proxy['teacher'];p.winner=teacher.argmax(1);p.rows=torch.arange(len(teacher),device='cuda')
    counts=torch.bincount(p.winner,minlength=teacher.shape[1]);p.weights=1./counts[p.winner].double()/int((counts>0).sum());p.unit=p.gap(angular_scores(teacher)).median()
    p.stress=lambda s:-(p.weights*p.gap(angular_scores(s)).relu()).sum()/p.unit
    record=next(r for r in report['records'] if r['group']=='boundary');first=record['first_training_feasible_step'];assert first is not None
    start_step=2*first['step'];assert start_step<=cfg['parameter_budget']
    theta=start_step*data['groups']['boundary']['direction'].cuda().double();initial=p.metrics(theta);assert initial['native_critic']>cfg['margin_fraction']
    history=[];saved=[];jac_calls=0
    for iteration in range(cfg['correction_steps']):
        residual=p.residuals(theta);flat=int(residual.abs().argmax());critic=p.metrics(theta)['native_critic']
        offset=flat;witness=None
        for group in p.native:
            length=group['teacher'].numel()
            if offset<length:
                row=offset//group['teacher'].shape[1];witness={k:v[row:row+1] for k,v in group.items()};break
            offset-=length
        assert witness is not None
        packed={k:torch.cat([core[k],witness[k]]) for k in core}
        matrix,r=local_matrix(p,packed,theta);jac_calls+=1
        # Check the real finite response at this nonlinear point, not only at zero.
        direction=data['groups']['boundary']['direction'].cuda().double();h=cfg['derivative_step']
        measured=(signed_residual(packed['teacher'],p.predict(packed,p.p,theta+h*direction))-signed_residual(packed['teacher'],p.predict(packed,p.p,theta-h*direction)))/(2*h)
        winner=packed['teacher'].argmax(1);keep=torch.ones_like(packed['teacher'],dtype=torch.bool);keep[torch.arange(len(winner),device='cuda'),winner]=False
        error=float((matrix@direction-measured[keep]).abs().max());assert error<1e-5
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False);tol=torch.finfo(torch.float64).eps*max(matrix.shape)*s[0];rank=int((s>tol).sum())
        correction=-vh[:rank].T@((u[:,:rank].T@r)/s[:rank]);trials=[];accepted=None
        for k in range(cfg['backtracking_steps']):
            fraction=.5**k;trial=theta+fraction*correction
            if float(trial.norm())>cfg['parameter_budget']*(1+1e-12):
                trials.append({'fraction':fraction,'budget_passed':False});continue
            metrics=p.metrics(trial);trials.append({'fraction':fraction,'budget_passed':True,**metrics})
            if metrics['native_critic']<critic-1e-12:accepted=(trial,metrics);break
        saved.append({'theta':theta.cpu(),'matrix':matrix.cpu(),'residual':r.cpu(),'correction':correction.cpu(),'packed':{k:v.cpu() for k,v in packed.items()}})
        history.append({'iteration':iteration+1,'old_worst_drift':critic,'worst_flat_witness':flat,'numerical_rank':rank,'correction_norm':float(correction.norm()),'local_jvp_fd_error':error,'trials':trials,'accepted':accepted is not None})
        if accepted is None:break
        theta=accepted[0]
        if accepted[1]['native_critic']<=cfg['margin_fraction']:break
    final=p.metrics(theta);file=out/'diagnostic.pt';torch.save({'theta':theta.cpu(),'steps':saved,'cfg':cfg,'reference_sha256':cfg['diagnostic_sha256']},file)
    result={'target':target,'status':'nonlinear_retraction_mechanism_only','initial':initial,'final':final,'initial_step_rule':'First infeasible doubled ray step above the frozen training-feasible step; not selected by proxy or confirmation.',
        'initial_step':start_step,'history':history,'analytic_publication_jacobian_calls':jac_calls,'accepted_correction_updates':sum(r['accepted'] for r in history),
        'diagnostic_sha256':digest(file),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'checkpoint_sha256':p.hashes,
        'independent_confirmation_run':False,'new_release_weights':0,'new_model_forwards':0,'expert_updated':False,'test_used':False,'partner_used':False,
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'limitation':'Correction is on finite calibration responses; no global task preservation or true fusion claim.'}
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;diagnose(args.target,cfg)
