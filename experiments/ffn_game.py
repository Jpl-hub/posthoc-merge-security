"""One bounded FFN publication optimization, followed by a fresh confirmation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from angular_distance import signed_residual
from clip_benchmark import digest
from covariance_noise import dataset
from cross_module_distance import capture_attention
from ffn_levelset import FFNProblem
from subspace_probe import save_json


def confirmation(p,cfg):
    """Exclude registered historical target indices before the sole confirmation."""
    excluded=set(i for group in p.indices.values() for i in group);sources={}
    audit=json.loads(Path(cfg['index_audit']).read_text())
    for rows in [v for v in audit.values() if isinstance(v,list)]:
        for row in rows:
            path=Path(row['path']);group=row['group']
            if path.suffix=='.json':
                value=json.loads(path.read_text())[group]
            else:
                value=torch.load(path,map_location='cpu')['indices'][group]
            excluded.update(value);sources[str(path)]=digest(path)
    for path in cfg['extra_confirmation_exclusions']:
        value=json.loads(Path(path).read_text());excluded.update(value['indices']);sources[path]=digest(path)
    training=dataset(p.target,'data',p.expert.val_preprocess,True)
    pool=sorted(set(range(len(training)))-excluded);assert len(pool)>=cfg['confirmation_n']
    indices=np.random.default_rng(cfg['confirmation_seed']).choice(pool,cfg['confirmation_n'],replace=False).tolist()
    split={'indices':indices,'seed':cfg['confirmation_seed'],'split':'target_train','used_for_optimization':False,
        'excluded_count':len(excluded),'excluded_metadata_sha256':sources}
    return capture_attention(p.expert,training,indices),torch.tensor([training[i][1] for i in indices]),split


def optimize(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    root=Path(cfg['ffn_reference']);meta=json.loads((root/'results.json').read_text())
    valid=json.loads((root/'independent_validation.json').read_text());assert valid['status']=='passed'
    assert digest(root/'diagnostic.pt')==cfg['ffn_diagnostic_sha256']==meta['diagnostic_sha256']
    assert meta['final']['native_critic']<=cfg['margin_fraction']
    raw=torch.load(root/'diagnostic.pt',map_location='cpu');p=FFNProblem(target,cfg)
    theta=raw['theta'].cuda().double();initial=p.metrics(theta);assert initial['native_critic']<=cfg['margin_fraction']
    calls=0;gradients=0;corrections=0;history=[];reasons=[]
    def correct(trial):
        nonlocal calls,corrections
        for _ in range(cfg['inner_corrections']):
            before=p.metrics(trial)
            if before['native_critic']<=cfg['margin_fraction']:return trial,before
            if calls>=cfg['maximum_jacobian_calls']:return None
            u,s,vh,r=p.critic(trial);calls+=1;q=-vh.T@((u.T@r)/s);accepted=None
            for k in range(cfg['backtracking_steps']):
                proposed=trial+(.5**k)*q
                if float(proposed.norm())>cfg['parameter_budget']:continue
                metrics=p.metrics(proposed)
                if metrics['native_critic']<before['native_critic']-1e-12:accepted=proposed;break
            if accepted is None:return None
            trial=accepted;corrections+=1
        metrics=p.metrics(trial)
        return (trial,metrics) if metrics['native_critic']<=cfg['margin_fraction'] else None
    best=(initial['objective'],theta.clone(),0)
    for step in range(cfg['optimization_steps']):
        if calls>=cfg['maximum_jacobian_calls']:reasons.append('fixed_jacobian_cost_cap');break
        leaf=theta.clone().requires_grad_(True);g=torch.autograd.grad(p.objective(leaf),leaf)[0];gradients+=1
        _,_,v,_=p.critic(theta);calls+=1;d=g-v.T@(v@g)
        if float(d.norm())<1e-14:reasons.append('no_local_feasible_direction');break
        d/=d.norm();accepted=None
        for k in range(cfg['backtracking_steps']):
            trial=theta+cfg['normalized_step']*.5**k*d
            if float(trial.norm())>cfg['parameter_budget']:continue
            accepted=correct(trial)
            if accepted is not None:break
        if accepted is None:reasons.append('no_feasible_finite_update');break
        theta,metrics=accepted
        if metrics['objective']>best[0]:best=(metrics['objective'],theta.clone(),step+1)
        history.append({'step':step+1,**metrics,'task_jacobian_calls':calls,'correction_updates':corrections})
        if (step+1)%8==0:print(json.dumps(history[-1]),flush=True)
    theta=best[1];selected=p.metrics(theta)
    # The selected coefficient fingerprint is frozen before confirmation data are drawn.
    torch.save({'theta':theta.cpu(),'selected_step':best[2],'protocol':cfg},out/'training_point.pt')
    frozen_hash=digest(out/'training_point.pt')
    cache,labels,split=confirmation(p,cfg);packed=p.prepare(cache,p.p,True)
    with torch.no_grad():
        after=p.predict(packed,p.p,theta);rho=float(signed_residual(packed['teacher'],after).abs().max())
        before=packed['teacher'][:len(labels)].argmax(1).cpu();pred=after[:len(labels)].argmax(1).cpu()
        ea=p.na*p.ua@theta[:p.a_size].reshape(p.r,p.r)@p.va;eb=p.nb*p.q@theta[p.a_size:].reshape(p.b_rows,p.b_width)
    candidate={'native':p.cache['native'],'scale':p.cache['scale'],'proxy':p.cache['proxy'],'indices':p.indices,
        'basis_A':raw['basis']['A'],'basis_B':raw['basis']['B'],'output_basis':p.q.cpu(),'theta':theta.cpu(),
        'release_deltas':{'A':ea.cpu(),'B':eb.cpu()},'protocol':cfg,'confirmation':cache,'confirmation_indices':split['indices'],
        'confirmation_labels':labels,'confirmation_original_predictions':before,'confirmation_candidate_predictions':pred}
    torch.save(candidate,out/'candidate.pt')
    report={'target':target,'method':'FFN_task_functional_levelset_with_nonlinear_correction','initial':initial,'selected':selected,'selected_step':best[2],
        'planned_optimization_steps':cfg['optimization_steps'],'accepted_updates':len(history),'explicit_reverse_gradient_calls':gradients,
        'analytic_task_jacobian_calls':calls,'nonlinear_correction_updates':corrections,'stop_reasons':reasons,
        'training_point_sha256_before_confirmation':frozen_hash,'confirmation':{'n':len(labels),'original_correct':int((before==labels).sum()),
            'candidate_correct':int((pred==labels).sum()),'changed_predictions':int((before!=pred).sum()),'worst_margin_drift':rho},
        'confirmation_passed':rho<=cfg['margin_fraction'],'candidate_sha256':digest(out/'candidate.pt'),'checkpoint_sha256':p.hashes,
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'wall_seconds':time.monotonic()-start,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'original_expert_updated':False,'partner_used':False,'test_used':False,
        'new_release_weights':0,'scope':'Finite target-training task game, not a global robust saddle. No formal TA/AGA until independent confirmation and verification pass.'}
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg)
    save_json(out/'confirmation_split.json',split);save_json(out/'training_indices.json',p.indices);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    optimize(args.target,cfg)
