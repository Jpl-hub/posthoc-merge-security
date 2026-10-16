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
    def cpu(d):return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in d.items()}
    torch.save(dict(p=cpu(p),q=cpu(q),check=cpu(check),data=cpu(data),pairs=pairs.cpu(),S=S.cpu(),Q=Q.cpu(),
        before=before.cpu(),after=after.cpu(),identity=identity.cpu(),cfg=cfg,indices=raw['indices']),out/'diagnostic.pt')
    report['diagnostic_sha256']=digest(out/'diagnostic.pt');save_json(out/'results.json',report);save_json(out/'protocol.json',cfg)
    print(json.dumps(report),flush=True)


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
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol'); ap.add_argument('--verify'); args=ap.parse_args()
    if args.verify: verify(args.verify)
    else:
        cfg=json.loads(Path(args.protocol).read_text()); cfg['protocol_path']=args.protocol
        initialize();run(cfg)
