"""Target-only boundary coverage and local feasible response geometry."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from angular_distance import angular_scores, signed_residual
from clip_benchmark import digest, merge
from covariance_noise import dataset
from cross_module_distance import CrossProblem, capture_attention, normalized_input
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_probe_distance import gelu


def unused_training(problem, cfg):
    """Exclude all old target calibration/confirmation metadata before sampling."""
    excluded=set(i for group in problem.cal['indices'].values() for i in group)
    excluded.update(problem.prior_confirmation_indices);sources={}
    for folder in Path('outputs').iterdir():
        if not folder.is_dir():continue
        case=folder/'case.json';own=problem.target.lower() in folder.name.lower()
        if case.exists():own=json.loads(case.read_text()).get('target')==problem.target
        if not own:continue
        for name in ['training_indices.json','holdout_protocol.json','confirmation_split.json']:
            path=folder/name
            if not path.exists():continue
            data=json.loads(path.read_text())
            if name=='holdout_protocol.json' and data.get('split')!='train':continue
            groups=list(data.values()) if name=='training_indices.json' else [data.get('indices',[])]
            for group in groups:
                if isinstance(group,list):excluded.update(group)
            sources[str(path)]=digest(path)
    # Two historical generic output directories store target indices in tensors.
    # Their MNIST metadata supplies pins; only this target's indices are loaded.
    for root in [Path(cfg['margin_reference']),Path('outputs/phase2_projected_game_retraction')]:
        path=root/(problem.target+'_diagnostic.pt')
        if path.exists():
            old=torch.load(path,map_location='cpu')
            for values in old.get('indices',{}).values():excluded.update(values)
            sources[str(path)]=digest(path)
    training=dataset(problem.target,'data',problem.expert.val_preprocess,True)
    pool=sorted(set(range(len(training)))-excluded)
    assert len(pool)>=cfg['screen_n']+cfg['future_confirmation_reserve']
    indices=np.random.default_rng(cfg['seed']).choice(pool,cfg['screen_n'],replace=False).tolist()
    return training,{'indices':indices,'excluded_count':len(excluded),'excluded_metadata_sha256':sources,
                     'split':'target_train','role':'mechanism_calibration_not_confirmation','seed':cfg['seed']}


def raw_jacobians(item,p):
    """Normalized CLIP scores differentiated through the exact last block."""
    h=item['h'];c=h-h.mean(1,keepdim=True);sd=(c.square().mean(1,keepdim=True)+p['eps']).sqrt()
    t=(c/sd*p['gamma']+p['beta'])@p['projection'];tn=t.norm(dim=1,keepdim=True);f=t/tn;s=f@p['head'].T
    j=(p['head'][None,:,:]-s[:,:,None]*f[:,None,:])/tn[:,None,:]
    j=(j@p['projection'].T)*p['gamma']
    j=(j-j.mean(-1,keepdim=True)-c[:,None,:]*(j*c[:,None,:]).mean(-1,keepdim=True)/sd[:,None,:].square())/sd[:,None,:]
    r=item['r'];rc=r-r.mean(1,keepdim=True);rs=(rc.square().mean(1,keepdim=True)+p['eps2']).sqrt()
    pre=normalized_input(r,p)@p['A'].T+p['a'];prob=torch.sigmoid(1.702*pre)
    slope=prob+1.702*pre*prob*(1-prob);hidden=pre*prob
    back=((j@p['B'])*slope[:,None,:])@p['A']*p['gamma2']
    back=(back-back.mean(-1,keepdim=True)-rc[:,None,:]*(back*rc[:,None,:]).mean(-1,keepdim=True)/rs[:,None,:].square())/rs[:,None,:]
    return j,j+back,hidden,s


def publication_jacobian(item,parameters,problem):
    jh,jr,z,s=raw_jacobians(item,parameters)
    co=problem.na*((jr@problem.basis['A']['u'])[:,:,:,None]*(item['o_input']@problem.basis['A']['vh'].T)[:,None,None,:])
    kb=problem.nb*((jh@problem.output_basis)[:,:,:,None]*z[:,None,None,:])
    return torch.cat([co.flatten(2),kb.flatten(2)],2),s


def margin_jacobian(item,problem):
    j,s=publication_jacobian(item,problem.p,problem);y=s.argmax(1);rows=torch.arange(len(s),device=s.device)
    gaps=s[rows,y,None]-s;keep=torch.ones_like(gaps,dtype=torch.bool);keep[rows,y]=False
    assert float(gaps[keep].min())>1e-8,'Ambiguous teacher margin; stop instead of changing its definition'
    response=(j[rows,y,None,:]-j)/gaps.masked_fill(~keep,1.)[:,:,None]
    return response[keep],s


def distribution(scores):
    s=scores.detach().cpu().numpy();y=s.argmax(1);others=s.copy();others[np.arange(len(s)),y]=-np.inf
    gaps=s[np.arange(len(s)),y]-others.max(1);rows=[]
    for c in range(s.shape[1]):
        values=gaps[y==c]
        rows.append({'class':c,'n':len(values),'quantiles':dict(zip(['minimum','q01','q05','median'],np.quantile(values,[0,.01,.05,.5]).tolist())) if len(values) else None})
    return {'n':len(s),'per_class':rows},y,gaps


def modes(item,problem):
    _,jr,z,s=raw_jacobians(item,problem.p);y=s.argmax(1);rows=torch.arange(len(s),device=s.device)
    keep=torch.ones_like(s,dtype=torch.bool);keep[rows,y]=False
    gaps=(s[rows,y,None]-s).masked_fill(~keep,1.)
    j=(jr[rows,y,None,:]-jr)/gaps[:,:,None];j[~keep]=0.
    u,v=problem.full_o_modes
    energy=((j@u).square()*(item['o_input']@v.T)[:,None,:].square()).sum((0,1))
    selected=problem.cal['basis_O']['selected'].cuda()
    block=(j@problem.basis['A']['u']).square().sum(2)*(item['o_input']@problem.basis['A']['vh'].T).square().sum(1)[:,None]
    full=j.square().sum(2)*item['o_input'].square().sum(1)[:,None]
    b_fraction=(z@problem.basis['B']['vh'].T).square().sum()/z.square().sum()
    return {'selected_fraction_of_diagonal_task_modes':float(energy[selected].sum()/energy.sum()),
            'selected_block_fraction_of_full_O_margin_gradient':float(block.sum()/full.sum()),
            'selected_B_input_fraction_of_activation_energy':float(b_fraction),
            'largest_margin_response_mode':int(energy.argmax())}


def diagnose(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    p=CrossProblem(target,cfg);training,split=unused_training(p,cfg);save_json(out/'training_indices.json',split)
    cache=capture_attention(p.expert,training,split['indices'])
    item=on_device(cache);_,_,_,native_scores=raw_jacobians(item,p.p)
    stats,y,gaps=distribution(native_scores);rng=np.random.default_rng(cfg['context_seed']);critical=[];balanced=[]
    for c in range(native_scores.shape[1]):
        members=np.flatnonzero(y==c);assert len(members)>=cfg['contexts_per_class']
        ordered=members[np.lexsort((np.asarray(split['indices'])[members],gaps[members]))]
        critical.extend(ordered[:cfg['contexts_per_class']].tolist());balanced.extend(rng.choice(members,cfg['contexts_per_class'],replace=False).tolist())
    critical=sorted(critical);balanced=sorted(balanced)
    groups={'boundary':{k:v[critical] for k,v in item.items()},'class_balanced':{k:v[balanced] for k,v in item.items()}}
    groups['old_contexts']={k:torch.cat([on_device(p.cal[g])[k][:cfg['old_probe_contexts']] for g in ['native','scale']]) for k in cache}
    anchor=p.anchor.model.visual.transformer.resblocks[11].attn.out_proj.weight.detach().cpu().double()
    u,_,vh=torch.linalg.svd(p.p_cpu['O']-anchor,full_matrices=False);p.full_o_modes=(u.cuda(),vh.cuda())
    proxy=merge(p.anchor,p.expert,p.anchor,cfg['proxy_beta']);proxy_cache=capture_attention(proxy,training,split['indices'][:cfg['proxy_n']]);del proxy
    proxy_item=on_device(proxy_cache);pj,ps=publication_jacobian(proxy_item,p.pp,p)
    teacher=ps.detach();winner=teacher.argmax(1);rows=torch.arange(len(teacher),device='cuda');others=teacher.clone();others[rows,winner]=-torch.inf;competitor=others.argmax(1)
    counts=torch.bincount(winner,minlength=teacher.shape[1]);weight=1./counts[winner].double()/int((counts>0).sum())
    direction_scores=angular_scores(teacher);unit=(direction_scores[rows,winner]-direction_scores[rows,competitor]).median();assert unit>0
    ds=torch.zeros_like(teacher);ds[rows,winner]=-weight/unit;ds[rows,competitor]=weight/unit
    ds=(ds-(ds*direction_scores).sum(1,keepdim=True)*direction_scores)/(teacher-teacher.mean(1,keepdim=True)).norm(dim=1,keepdim=True);ds-=ds.mean(1,keepdim=True)
    gradient=cfg['proxy_beta']*(ds[:,:,None]*pj).sum((0,1));del pj
    p.proxy=p.prepare(proxy_cache,p.pp,False);p.winner=winner;p.rows=rows;p.weights=weight;p.unit=unit
    p.stress=lambda scores:-(p.weights*p.gap(angular_scores(scores)).relu()).sum()/p.unit
    p.cfg['probe_contexts']=len(critical)
    p.native=[p.prepare(cache,p.p,False),p.prepare({k:v[critical] for k,v in cache.items()},p.p,True)]
    zero=torch.zeros(p.a_size+p.b_rows*p.b_width,device='cuda',dtype=torch.float64)
    score_objective=lambda t:p.stress(p.predict(p.proxy,p.pp,t,cfg['proxy_beta']))
    leaf=zero.clone().requires_grad_(True);automatic=torch.autograd.grad(score_objective(leaf),leaf)[0]
    gradient_error=float((automatic-gradient).norm()/automatic.norm());assert gradient_error<1e-7
    records=[];saved={}
    for name,group in groups.items():
        matrix,s=margin_jacobian(group,p);_,singular,vectors=torch.linalg.svd(matrix,full_matrices=False)
        tolerance=torch.finfo(torch.float64).eps*max(matrix.shape)*singular[0];rank=int((singular>tolerance).sum());basis=vectors[:rank]
        projected=gradient-basis.T@(basis@gradient);direction=projected/projected.norm();null_response=float((matrix@direction).abs().max())
        derivative=torch.autograd.functional.jvp(p.residuals,zero,direction)[1]
        with torch.no_grad():
            h=cfg['derivative_step'];measured=float((score_objective(h*direction)-score_objective(-h*direction))/(2*h));slope=float(gradient@direction)
            assert abs(measured-slope)/max(abs(slope),1e-20)<1e-3
            finite=[{'step':cfg['parameter_budget']*.5**k,**p.metrics(cfg['parameter_budget']*.5**k*direction)} for k in range(cfg['backtracking_steps'])]
        first=next((row for row in finite if row['native_critic']<=cfg['margin_fraction']),None)
        group_stats,_,_=distribution(s)
        records.append({'group':name,'contexts':len(s),'numerical_response_rank':rank,'rank_tolerance':float(tolerance),
            'gradient_retained_fraction':float(projected.norm()/gradient.norm()),'gradient_task_row_space_cosine':float((basis@gradient).norm()/gradient.norm()),
            'task_null_response_maximum':null_response,'proxy_unit_directional_derivative':slope,'proxy_central_difference':measured,
            'full_training_directional_margin_maximum':float(derivative.abs().max()),'linear_training_step_bound':float(cfg['margin_fraction']/derivative.abs().max()),
            'task_modes':modes(group,p),'margin_distribution':group_stats,'finite_steps':finite,'first_training_feasible_step':first})
        saved[name]={'matrix':matrix.cpu(),'direction':direction.cpu(),'task_directional_response':derivative.cpu(),'scores':s.cpu()}
    file=out/'diagnostic.pt';torch.save({'screen':cache,'proxy':proxy_cache,'critical_offsets':critical,'balanced_offsets':balanced,'split':split,
        'groups':saved,'gradient':gradient.cpu(),'basis_O':p.cal['basis_O'],'basis_B':p.cal['basis_B'],'output_basis':p.output_basis.cpu(),'cfg':cfg},file)
    report={'target':target,'status':'task_boundary_and_local_geometry_diagnosis','screen_distribution':stats,'records':records,
            'checkpoint_sha256':p.hashes,'diagnostic_sha256':digest(file),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),
            'proxy_gradient_relative_error':gradient_error,'new_model_forward_instances':cfg['screen_n']+cfg['proxy_n'],
            'read_only_gradient_calls':1,'task_JVP_calls':len(groups),'expert_parameter_updates':0,'publication_parameter_updates':0,'new_release_weights':0,
            'previous_confirmation_used':False,'partner_used':False,'test_used':False,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'selection':'Lowest teacher top-two margin per predicted class; same fixed count for every class. No candidate or confirmation information.',
            'limitation':'Local null sensitivity of finite contexts is not uniform preservation, proxy transport, or a robust defense guarantee.'}
    save_json(out/'protocol.json',cfg);save_json(out/'results.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    diagnose(args.target,cfg)
