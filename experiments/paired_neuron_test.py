"""One fixed-width coalescence check; no new optimizer or expert training."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from classification_completion import readout
from clip_benchmark import digest
from cross_module_distance import normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import gelu, initialize
from verify_levelset_audits import raw_scores, residual


def cpu(d):
    return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in d.items()}


def predict(data, p):
    z = gelu(data['x']@p['A'].T+p['a'])+data['offset']
    return scores(data['base']+z@p['B'].T, p)


@torch.no_grad()
def run(cfg):
    start=time.monotonic(); out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=False)
    assert '3090' in torch.cuda.get_device_name(); torch.cuda.reset_peak_memory_stats()
    for path,pin in cfg['inputs'].items(): assert digest(path)==pin,path
    raw=torch.load(cfg['source_cache'],map_location='cpu',weights_only=False)
    basis=torch.load(cfg['task_basis'],map_location='cpu',weights_only=False)['basis']
    p=on_device(raw['p']); va=basis['A']['vh'].cuda().double(); vb=basis['B']['vh'].cuda().double()
    groups=raw['groups']; counts=[groups[k]['natural_n'] for k in ['fit','scale']]; n=sum(counts)
    r=torch.cat([groups[k]['r'][:groups[k]['natural_n']] for k in ['fit','scale']]).cuda().double()
    rounding=torch.cat([groups[k]['rounding'][:groups[k]['natural_n']] for k in ['fit','scale']]).cuda().double()
    x=normalized_input(r,p); pre=x@p['A'].T+p['a']; z=gelu(pre); base=r+p['b']+rounding
    teacher=scores(base+z@p['B'].T,p); y=teacher.argmax(1); other=teacher.clone(); other[torch.arange(n,device='cuda'),y]=-torch.inf
    gap=teacher[torch.arange(n,device='cuda'),y]-other.max(1).values; assert gap.min()>0
    weights=1/(torch.bincount(y[:counts[0]],minlength=teacher.shape[1])[y[:counts[0]]].double()*gap[:counts[0]].square())
    weights/=weights.sum(); fit=pre[:counts[0]]; gram=fit.T@(weights[:,None]*fit)
    cost=(gram.diag()[:,None]+gram.diag()[None]-2*gram).clamp_min(0)
    norms=p['B'].norm(dim=0); cost*=((norms[:,None]+norms[None])/2).square()
    cost.fill_diagonal_(torch.inf); cost=torch.triu(cost,1)+torch.tril(torch.full_like(cost,torch.inf))
    space,_=readout(p); Q=space['Q']; pair_n=len(vb)+Q.shape[1]; pairs=[]
    for _ in range(pair_n):
        flat=int(cost.argmin()); i,j=divmod(flat,len(cost)); pairs.append([i,j])
        cost[[i,j],:]=torch.inf; cost[:,[i,j]]=torch.inf
    pairs=torch.tensor(pairs,device='cuda'); C=torch.zeros((p['B'].shape[1],pair_n),device='cuda',dtype=torch.float64)
    C[pairs[:,0],torch.arange(pair_n,device='cuda')]=2**-.5; C[pairs[:,1],torch.arange(pair_n,device='cuda')]=-2**-.5
    _,sing,vh=torch.linalg.svd(vb@C,full_matrices=True); rank=int((sing>sing[0]*1e-10).sum()); S=C@vh[rank:].T
    assert S.shape[1]>=Q.shape[1] and (vb@S).abs().max()<1e-10
    q=dict(p); q['A']=p['A'].clone(); q['a']=p['a'].clone()
    for field in ['A','a']:
        mean=(p[field][pairs[:,0]]+p[field][pairs[:,1]])/2
        q[field][pairs[:,0]]=mean; q[field][pairs[:,1]]=mean
        q[field]=q[field].float().double()
    contexts=[]
    for c in range(teacher.shape[1]):
        ids=(y==c).nonzero().flatten(); assert len(ids)>=cfg['contexts_per_class']
        contexts.extend(ids[gap[ids].argsort(stable=True)[:cfg['contexts_per_class']]].tolist())
    ctx=torch.tensor(contexts,device='cuda'); shifts=[]
    for value,V in [(x[:counts[0]],va),(z[:counts[0]],vb)]:
        rms=(value@V.T).square().mean(0).sqrt()
        shifts.append((cfg['finite_step']*torch.tensor([-1.,1.],device='cuda',dtype=torch.float64)[:,None,None]*rms[None,:,None]*V[None]).flatten(0,1))
    sa,sb=shifts; m=len(sa); assert len(sb)==m
    data=dict(x=torch.cat([x,(x[ctx,None]+sa[None]).flatten(0,1),x[ctx].repeat_interleave(m,0)]),
        base=torch.cat([base,base[ctx].repeat_interleave(m,0),base[ctx].repeat_interleave(m,0)]),
        offset=torch.cat([torch.zeros_like(z),torch.zeros((len(ctx)*m,z.shape[1]),device='cuda'),sb.repeat(len(ctx),1)]))
    before=predict(data,p); after=predict(data,q)
    from angular_distance import signed_residual
    drift=signed_residual(before,after).abs().max(1).values
    distance=float(sum(((q[k]-p[k]).norm()/p[k].norm()).square() for k in ['A','a']).sqrt())
    zstar=gelu(x@q['A'].T+q['a']); natural_kernel=float((zstar@S).abs().max())
    check=dict(q); remaining=max(0.,cfg['parameter_budget']**2-distance**2)**.5*(1-1e-6)
    K=torch.eye(Q.shape[1],S.shape[1],device='cuda',dtype=torch.float64); K*=remaining*p['B'].norm()/K.norm()
    check['B']=(p['B']+Q@K@S.T).float().double(); identity=predict(data,check)
    identities=dict(natural_activation_kernel_max=natural_kernel,task_probe_kernel_max=float((vb@S).abs().max()),
        added_component_full_score_error=float((identity-after).abs().max()),
        added_component_readout_norm=float(((check['B']-p['B']).T@space['M']).norm()))
    worst=int(drift.argmax()); changed=before[:n].argmax(1)!=after[:n].argmax(1)
    feasible=distance<=cfg['parameter_budget'] and float(drift.max())<=cfg['task_budget'] and not changed.any()
    report=dict(status='coalescence_source_feasible_not_a_release' if feasible else 'closed_fixed_pair_coalescence_budget_failure',
        preregistered_commit=cfg['preregistered_commit'],source_feasible=bool(feasible),pair_count=pair_n,probe_rank=len(vb),
        contrast_free_dimension=S.shape[1],parameter_distance=distance,task_inputs=len(before),natural_n=n,
        worst_task_drift=float(drift.max()),natural_worst_task_drift=float(drift[:n].max()),worst_row=worst,
        source_changed=int(changed.sum()),source_changed_by_class=torch.bincount(y[changed],minlength=teacher.shape[1]).cpu().tolist(),
        identities=identities,source_indices=raw['indices'],source_expert_frozen=True,new_release_weights=0,
        future_partner_used=False,test_used=False,confirmation_used=False,gradients=0,full_model_forwards=0,
        next_action='one preregistered source-only output fit, then freeze and fresh confirmation' if feasible else 'stop; no pair/strength/rank/seed repair or output optimization',
        source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
        wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    torch.save(dict(p=cpu(p),q=cpu(q),check=cpu(check),data=cpu(data),pairs=pairs.cpu(),S=S.cpu(),Q=Q.cpu(),
        before=before.cpu(),after=after.cpu(),identity=identity.cpu(),cfg=cfg,indices=raw['indices']),out/'diagnostic.pt')
    report['diagnostic_sha256']=digest(out/'diagnostic.pt');save_json(out/'results.json',report);save_json(out/'protocol.json',cfg)
    print(json.dumps({k:v for k,v in report.items() if k!='source_indices'}),flush=True)


@torch.no_grad()
def fit_output(cfg):
    from classification_completion import sigma
    from clip_benchmark import merge
    from covariance_noise import dataset
    from cross_module_distance import capture_attention, extended_parameters
    from subspace_probe import make_loader
    from task_probe_distance import checked_models
    import copy
    start=time.monotonic(); out=Path(cfg['output']); assert not (out/'training_point.pt').exists()
    assert json.loads((out/'results.json').read_text())['source_feasible']
    assert json.loads((out/'independent_validation.json').read_text())['status'].startswith('passed_')
    raw=torch.load(out/'diagnostic.pt',map_location='cpu',weights_only=False)
    p,q=[on_device(raw[k]) for k in ['p','q']]; Q,S=[raw[k].cuda() for k in ['Q','S']]
    anchor,expert,head,pins=checked_models(cfg['target']); assert torch.equal(p['A'].cpu(),extended_parameters(expert,head)['A'])
    proxy=merge(anchor,expert,anchor,.5); pp=on_device(extended_parameters(proxy,head)); ids=raw['indices']['fit'][:cfg['proxy_n']]
    train=dataset(cfg['target'],'data',expert.val_preprocess,True); cache=on_device(capture_attention(proxy,train,ids))
    xp=normalized_input(cache['r'],pp); qp=dict(pp); qp['A']=((q['A']-p['A'])*.5+pp['A']).float().double()
    qp['a']=((q['a']-p['a'])*.5+pp['a']).float().double(); zp=gelu(xp@qp['A'].T+qp['a'])
    rounding=cache['h']-(cache['r']+gelu(xp@pp['A'].T+pp['a'])@pp['B'].T+pp['b'])
    hg=rounding+cache['r']+zp@pp['B'].T+pp['b']; space,_=readout(pp)
    G=Q.T@space['M']; v=torch.linalg.solve(G@G.T,G@space['b']); w=v@Q.T; wc=w-w.mean()
    a=1-wc.square().mean(); assert a>1e-8 and (v@G-space['b']).norm()<1e-8
    target=-(cache['h']@space['M']+sigma(cache['h'],pp['eps'])*space['b'])
    correction=torch.linalg.solve(G@G.T,G@(target-hg@space['M']).T).T
    star=hg+correction@Q.T; centered=star-star.mean(1,keepdim=True)
    b0=(centered*wc).mean(1,keepdim=True); c0=centered.square().mean(1,keepdim=True)+pp['eps']
    rho=c0/((b0.square()+a*c0).sqrt()+b0); ideal=star-rho*w
    error=float((ideal@space['M']+sigma(ideal,pp['eps'])*space['b']-target).abs().max()); assert error<1e-8
    X=.5*zp@S; Y=(ideal-hg)@Q; teacher=scores(cache['h'],pp); y=teacher.argmax(1)
    weight=1/torch.bincount(y,minlength=len(pp['head']))[y].double(); weight/=weight.sum()
    gram=X.T@(weight[:,None]*X); ridge=cfg['ridge_relative']*torch.linalg.eigvalsh(gram)[-1]; assert ridge>0
    kt=torch.linalg.solve(gram+ridge*torch.eye(len(gram),device='cuda',dtype=torch.float64),X.T@(weight[:,None]*Y)).T
    base_cost=sum(((q[k]-p[k]).norm()/p[k].norm()).square() for k in ['A','a'])
    budget=(cfg['parameter_budget']**2-base_cost).sqrt()*(1-1e-6)
    K=kt/kt.norm()*(p['B'].norm()*budget); q['B']=(p['B']+Q@K@S.T).float().double()
    distance=float((base_cost+((q['B']-p['B']).norm()/p['B'].norm()).square()).sqrt())
    data=on_device(raw['data']); after=predict(data,q); before=raw['before'].cuda(); from angular_distance import signed_residual
    drift=float(signed_residual(before,after).abs().max()); feasible=distance<=cfg['parameter_budget'] and drift<=cfg['task_budget']
    qp['B']=(pp['B']+.5*(q['B']-p['B'])).float().double(); hp=rounding+cache['r']+zp@qp['B'].T+pp['b']; proxy_after=scores(hp,qp)
    gap=teacher.gather(1,y[:,None])-teacher.masked_fill(torch.nn.functional.one_hot(y,len(pp['head'])).bool(),-torch.inf).max(1).values[:,None]
    other=proxy_after.masked_fill(torch.nn.functional.one_hot(y,len(pp['head'])).bool(),-torch.inf).max(1).values
    damage=1-(proxy_after.gather(1,y[:,None]).flatten()-other)/gap.flatten(); changed=proxy_after.argmax(1)!=y
    prefix='model.visual.transformer.resblocks.11.mlp.'; fields={'A':'c_fc.weight','a':'c_fc.bias','B':'c_proj.weight'}
    edits={prefix+key:q[k].cpu().float() for k,key in fields.items()}; torch.save(edits,out/'frozen_release.pt'); release_pin=digest(out/'frozen_release.pt')
    controls={}
    for name,model,base_par,new_par,wanted in [('source',expert,p,q,after[:len(ids)]),('public_base_interpolation',proxy,pp,qp,proxy_after)]:
        model=copy.deepcopy(model)
        for k,key in fields.items(): model.state_dict()[prefix+key].copy_(new_par[k].cpu().float())
        actual=[];model.cuda();head.cuda()
        for images,_ in make_loader(train,ids[:20],64):actual.append(head(model(images.cuda())).cpu())
        model.cpu();head.cpu();actual=torch.cat(actual).double(); err=float((actual-wanted[:len(actual)].cpu()).abs().max()); assert err<=cfg['score_tolerance']
        controls[name]=dict(n=len(actual),score_error=err)
    point=dict(p=cpu(p),q=cpu(q),Q=Q.cpu(),S=S.cpu(),K=K.cpu(),kt=kt.cpu(),X=X.cpu(),Y=Y.cpu(),weight=weight.cpu(),ridge=ridge.cpu(),
        data=raw['data'],before=raw['before'],after=after.cpu(),proxy_cache=cpu(cache),pp=cpu(pp),qp=cpu(qp),proxy_after=proxy_after.cpu(),
        ideal=ideal.cpu(),target=target.cpu(),indices=raw['indices'],cfg=cfg,checkpoint_sha256=pins)
    torch.save(point,out/'training_point.pt'); save_json(out/'training_indices.json',raw['indices']); save_json(out/'case.json',dict(target=cfg['target']))
    result=dict(status='source_frozen_pending_independent_confirmation' if feasible else 'closed_quantized_source_budget_failure',source_feasible=feasible,
        parameter_distance=distance,worst_task_drift=drift,source_changed=int((after[:sum(len(v) for v in raw['indices'].values())].argmax(1)!=before[:sum(len(v) for v in raw['indices'].values())].argmax(1)).sum()),
        ideal_numerator_error=error,quadratic_a=float(a),raw_direction_norm=float(kt.norm()),fit_relative_error=float((X@kt.T-Y).norm()/Y.norm()),
        gram_condition=float(torch.linalg.cond(gram+ridge*torch.eye(len(gram),device='cuda'))),proxy_changed=int(changed.sum()),proxy_n=len(ids),
        proxy_per_class=[dict(class_id=c,n=int(y.eq(c).sum()),changed=int((changed&y.eq(c)).sum()),maximum_signed_damage=float(damage[y==c].max())) for c in range(len(pp['head']))],
        source_point_reselected=False,source_coalescence_repaired=False,proxy_is_not_unknown_peer=True,proxy_changes_not_a_confirmation_gate=True,
        full_model_controls=controls,release_sha256=release_pin,training_point_sha256=digest(out/'training_point.pt'),source_sha256=digest(__file__),
        gradients=0,optimizer_steps=0,conditional_linear_solves=1,full_model_forwards=len(ids)+40,wall_seconds=time.monotonic()-start,
        future_partner_used=False,test_used=False,confirmation_used=False)
    save_json(out/'output_fit.json',result);print(json.dumps(result),flush=True)


def verify(output):
    out=Path(output); report=json.loads((out/'results.json').read_text()); assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu',weights_only=False)
    p,q,check=[{k:v.numpy() if torch.is_tensor(v) else v for k,v in raw[t].items()} for t in ['p','q','check']]
    data={k:v.numpy() for k,v in raw['data'].items()}; maximum=0
    for label,par in [('before',p),('after',q),('identity',check)]:
        pre=data['x']@par['A'].T+par['a']; zz=pre*expit(1.702*pre)+data['offset']; ss=raw_scores(data['base']+zz@par['B'].T,par)
        err=float(abs(ss-raw[label].numpy()).max()); maximum=max(maximum,err)
        np.testing.assert_allclose(ss,raw[label],atol=1e-8,rtol=1e-9)
    rho=float(abs(residual(raw['before'].numpy(),raw['after'].numpy())).max()); assert abs(rho-report['worst_task_drift'])<1e-8
    for i,j in raw['pairs'].tolist(): np.testing.assert_array_equal(q['A'][i],q['A'][j]); assert q['a'][i]==q['a'][j]
    basis=torch.load(raw['cfg']['task_basis'],map_location='cpu',weights_only=False)['basis']['B']['vh'].numpy()
    S=raw['S'].numpy(); np.testing.assert_allclose(S.T@S,np.eye(S.shape[1]),atol=1e-10); assert abs(basis@S).max()<1e-10
    result=dict(status='passed_independent_numpy_complete_finite_scores_and_kernel',numpy_score_max_error=maximum,
        worst_task_drift=rho,source_feasible=report['source_feasible'],new_release_weights=0,source_sha256=digest(__file__))
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol'); ap.add_argument('--verify'); ap.add_argument('--fit-output',action='store_true'); args=ap.parse_args()
    if args.verify: verify(args.verify)
    else:
        cfg=json.loads(Path(args.protocol).read_text()); cfg['protocol_path']=args.protocol
        initialize();(fit_output if args.fit_output else run)(cfg)
