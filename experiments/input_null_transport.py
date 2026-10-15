"""One preregistered input-conditioned kernel transport and finite feasibility ray."""
import argparse,copy,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from attention_cascade import suffix
from boundary_game import risk_first
from classification_completion import readout
from clip_benchmark import digest
from covariance_noise import dataset
from cross_module_distance import CrossProblem,extended_parameters,normalized_input
from null_bias_leakage import analyze_scores,slope
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader,save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models,gelu,initialize
from verify_covariance_noise import official_labels


def class_game(G):
    """Min-norm convex hull: enumerate at most1023 class faces, no model search."""
    H=G@G.T;H=H/max(np.abs(H).max(),1e-30);best=None
    for size in range(1,len(G)+1):
        for ids in itertools.combinations(range(len(G)),size):
            ix=np.array(ids);h=H[np.ix_(ix,ix)]
            k=np.block([[h,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(size),1.];sol=np.linalg.lstsq(k,rhs,rcond=1e-13)[0]
            if np.abs(k@sol-rhs).max()>1e-9 or sol[:-1].min()<-1e-10:continue
            pi=np.zeros(len(G));pi[ix]=np.maximum(sol[:-1],0);pi/=pi.sum();v=pi@G;obj=float(v@v)
            if best is None or obj<best[0]:best=(obj,pi,v)
    assert best is not None
    q=np.sqrt(best[0]);direction=best[2]/max(q,1e-30);primal=float((G@direction).min())
    assert q-primal<1e-7*max(1.,q),(q,primal)
    return best[1],direction,dict(dual_value=q,primal_value=primal,duality_gap=q-primal,faces=2**len(G)-1)


def finite(data,p,E,factor=1.):
    x=normalized_input(data['r'],p);z=x@p['A'].T+p['a'];shift=factor*(x@E.T)
    h=data['rounding']+data['r']+(gelu(z)+data.get('b_offset',0.))@p['B'].T+p['b']
    linear=.5*shift@p['B'].T
    even=(gelu(z+shift)-.5*(z+shift)-gelu(z)+.5*z)@p['B'].T
    dh=linear+even;direct=(gelu(z+shift)-gelu(z))@p['B'].T
    assert float((direct-dh).abs().max())<1e-9
    before=scores(h,p);after=scores(h+dh,p);report,parts=analyze_scores(before,after)
    report['hidden_split_error']=float((direct-dh).abs().max())
    return report,dict(before=before,after=after,h=h,dh=dh,linear=linear,**parts)


def attribution(group,p):
    sp,_=readout(p);M,b=sp['M'],sp['b'];h,dh=group['h'],group['dh']
    sd=lambda v:((v-v.mean(1,keepdim=True)).square().mean(1)+p['eps']).sqrt()
    N=h@M+sd(h)[:,None]*b;pieces=dict(linear=group['linear']@M,even=(dh-group['linear'])@M,variance=(sd(h+dh)-sd(h))[:,None]*b)
    y=group['before'].argmax(1);rows=torch.arange(len(y),device=h.device);mask=torch.ones_like(N,dtype=torch.bool);mask[rows,y]=False
    gap=N[rows,y,None]-N;harm={k:(v-v[rows,y,None])/gap.masked_fill(~mask,1.) for k,v in pieces.items()}
    joint=sum(harm.values());j=joint.masked_fill(~mask,-torch.inf).argmax(1);damage=joint[rows,j]
    assert torch.equal(damage>1,group['after'].argmax(1)!=y)
    err=float((sum(pieces.values())-((h+dh)@M+sd(h+dh)[:,None]*b-N)).abs().max());assert err<1e-9
    return dict(maximum_damage=float(damage.max()),median_damage=float(damage.median()),crossings=int((damage>1).sum()),identity_error=err,
        means_at_joint_worst_competitor={k:float(v[rows,j].mean()) for k,v in harm.items()},
        per_class=[dict(label=c,n=int((y==c).sum()),crossings=int(((damage>1)&(y==c)).sum()),maximum_damage=float(damage[y==c].max())) for c in range(len(p['head']))])


def cpu(d):return {k:(v.cpu() if torch.is_tensor(v) else v) for k,v in d.items()}


@torch.no_grad()
def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for name,pin in cfg['inputs'].items():assert digest(name)==pin,name
    cal=torch.load(cfg['calibration_reference'],map_location='cpu');env=torch.load(cfg['environment_cache'],map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);pcpu=extended_parameters(expert,head);p=on_device(pcpu);p0=on_device(extended_parameters(anchor,head))
    for k,v in cal['parameters'].items():
        if torch.is_tensor(v):assert torch.equal(v,pcpu[k]),k
    raw,ids,n=risk_first(cal['cache'],cal['indices'],cfg['contexts_per_class'],p);rawgpu=on_device(raw)
    x=normalized_input(rawgpu['r'],p);z=x@p['A'].T+p['a'];s=scores(rawgpu['h'],p);y=s.argmax(1);rows=torch.arange(len(s),device='cuda')
    selection=cfg['selection_n'];J,_,_,_=raw_jacobians({k:v[:selection] for k,v in rawgpu.items()},p)
    K=(J@p['B'])*slope(z[:selection])[:,None,:]
    # Class-balanced actual task-mode margin response, not singular-value ordering.
    ua,sa,va=torch.linalg.svd(p['A']-p0['A'],full_matrices=False)
    response=(K@ua)*(x[:selection]@va.T)[:,None,:]*sa[None,None,:]
    yy=y[:selection];rr=rows[:selection];gap=s[:selection][rr,yy,None]-s[:selection];mask=torch.ones_like(gap,dtype=torch.bool);mask[rr,yy]=False
    relative=(response[rr,yy,None]-response)/gap.masked_fill(~mask,1.)[:,:,None];relative[rr,yy]=0
    counts=torch.bincount(yy,minlength=len(p['head']));weights=1/counts[yy].double()/int((counts>0).sum())
    importance=(relative.square().sum(1)*weights[:,None]).sum(0);chosen=importance.argsort(descending=True,stable=True)[:cfg['probe_rank']];V=va[chosen]
    # Intersect the exact output kernel with original low-gap score-normal kernels.
    normals=K[:n][rows[:n],y[:n],None]-K[:n]
    normals=normals[torch.arange(len(p['head']),device='cuda')[None,:]!=y[:n,None]]
    W=torch.cat([p['B']/p['B'].norm(dim=1,keepdim=True),normals/normals.norm(dim=1,keepdim=True)])
    _,sw,vw=torch.linalg.svd(W,full_matrices=False);rank=int((sw>torch.finfo(sw.dtype).eps*max(W.shape)*sw[0]).sum());R=vw[:rank]
    space,_=readout(p);G=space['Q'].T@(p['B']-p0['B']);G-=G@R.T@R
    _,sg,vg=torch.linalg.svd(G,full_matrices=False);nout=int((sg>torch.finfo(sg.dtype).eps*max(G.shape)*sg[0]).sum());U=vg[:nout].T
    assert nout>0 and float((W@U).abs().max())<1e-9
    pe=on_device(env['proxy_parameters']);he,context=suffix(on_device(env['proxy'])['H'].double(),pe)
    edata=dict(r=context['r'],rounding=on_device(env['proxy'])['rounding'])
    espace,_=readout(pe);h=he+edata['rounding'];ss=scores(h,pe);ey=ss.argmax(1);er=torch.arange(len(h),device='cuda');other=ss.clone();other[er,ey]=-torch.inf;ej=other.argmax(1)
    sd=((h-h.mean(1,keepdim=True)).square().mean(1)+pe['eps']).sqrt();N=h@espace['M']+sd[:,None]*espace['b']
    left=-.5*cfg['coefficient']*p['A'].norm()*((espace['M'][:,ey]-espace['M'][:,ej]).T@pe['B']@U)/(N[er,ey]-N[er,ej])[:,None]
    right=normalized_input(edata['r'],pe)@V.T
    linear_rows=(left[:,:,None]*right[:,None,:]).flatten(1)
    class_rows=torch.stack([linear_rows[ey==c].mean(0) for c in range(len(p['head']))]).cpu().numpy()
    pi,direction,game=class_game(class_rows);C=torch.tensor(direction.reshape(nout,cfg['probe_rank']),device='cuda');unit=p['A'].norm()*U@C@V
    assert abs(float(unit.norm()/p['A'].norm())-1)<1e-8
    # Old task-SVD probes are construction probes, never independent confirmation.
    cp=CrossProblem.__new__(CrossProblem);cp.cfg=dict(probe_contexts=n,finite_step=cfg['finite_step'],full_B_completion=True)
    cp.r=cfg['probe_rank'];cp.basis={'A':on_device(cal['bases']['O']),'B':on_device(cal['bases']['B'])};cp.na=p['O'].norm();cp.nb=p['B'].norm();cp.a_size=cp.r**2;cp.b_rows=p['B'].shape[0];cp.b_width=p['B'].shape[1];cp.output_basis=None
    data=cp.prepare(raw,p,True)
    rms=(x@V.T).square().mean(0).sqrt();F=p['A']@V.T*rms[None];frames=[];shifted_x=[];teacher_hidden=[]
    for i in range(n):
        ju=K[i]@F;normal=ju[y[i],None]-ju;normal=normal[torch.arange(len(p['head']),device='cuda')!=y[i]]
        _,sv,frame=torch.linalg.svd(normal,full_matrices=True);nrank=int((sv>sv[0]*torch.finfo(sv.dtype).eps*max(normal.shape)).sum())
        q=cfg['neighborhood_radius']*torch.cat([frame,-frame]);dx=(q*rms)@V
        shifted_x.append(x[i]+dx);teacher_hidden.append(rawgpu['h'][i]+(gelu(z[i]+q@F.T)-gelu(z[i]))@p['B'].T)
        frames.append(dict(source_position=i,normal_rank=nrank,frame=frame.cpu()))
    nx=torch.cat(shifted_x);nh=torch.cat(teacher_hidden);nz=nx@p['A'].T+p['a'];nb=scores(nh,p)
    trials=[];accepted=None;last=None
    for step,alpha in enumerate(cfg['source_ray']):
        exact=alpha*unit;E=(p['A']+exact).float().double()-p['A'];D=float(E.norm()/p['A'].norm());assert D<=cfg['parameter_budget']
        sr,sp=finite(data,p,E);na=scores(nh+(gelu(nz+nx@E.T)-gelu(nz))@p['B'].T,p);nr,npieces=analyze_scores(nb,na)
        ok=bool(sr['worst_task_drift']<=cfg['source_reserve'] and nr['worst_task_drift']<=cfg['source_reserve'] and sr['changed']==0 and nr['changed']==0)
        trials.append(dict(step=step,alpha=alpha,parameter_distance=D,source=sr,neighborhood=nr,source_feasible=ok));last=(E,exact,sp,na,npieces)
        save_json(out/'source_ray.json',trials)
        if ok:accepted=step;break
    E,exact,sp,na,npieces=last
    # No reference finite outcome was consulted when choosing the sole source ray point.
    environment,ep=finite(edata,pe,E,cfg['coefficient']);source=trials[-1]['source'];neighborhood=trials[-1]['neighborhood']
    source_labels=torch.tensor(official_labels(cfg['target'],True)[ids],device='cuda');elabels=torch.tensor(official_labels(cfg['target'],True)[env['indices']['proxy']],device='cuda')
    for report,group,labels in [(source,sp,source_labels),(environment,ep,elabels)]:
        a,b=group['before'].argmax(1),group['after'].argmax(1);report.update(original_correct=int((a[:len(labels)]==labels).sum()),protected_correct=int((b[:len(labels)]==labels).sum()),natural_n=len(labels),lost=int(((a[:len(labels)]==labels)&(b[:len(labels)]!=labels)).sum()),gained=int(((a[:len(labels)]!=labels)&(b[:len(labels)]==labels)).sum()))
    support=bool(accepted is not None and game['primal_value']>cfg['minimum_game_value'] and environment['lost']-environment['gained']>=cfg['minimum_net_losses'] and environment['classes_changed']>=cfg['minimum_environment_classes'])
    checks={}
    if accepted is not None:
        released=copy.deepcopy(expert);released.state_dict()['model.visual.transformer.resblocks.11.mlp.c_fc.weight'].copy_((p['A']+E).float().cpu())
        training=dataset(cfg['target'],'data',expert.val_preprocess,True);take=ids[:cfg['full_replay_n']]
        for name,model,expected in [('original',expert,sp['before'][:len(take)].argmax(1)),('prototype',released,sp['after'][:len(take)].argmax(1))]:
            model.cuda().eval();head.cuda().eval();actual=[]
            for images,_ in make_loader(training,take,64):actual.append(head(model(images.cuda())).argmax(1).cpu())
            model.cpu();head.cpu();assert torch.equal(torch.cat(actual),expected.cpu()),name;checks[name]=dict(n=len(take),all_predictions_match=True)
    torch.save(dict(delta_exact=exact.cpu(),delta_FP32=E.cpu(),unit=unit.cpu(),U=U.cpu(),V=V.cpu(),C=C.cpu(),W_rows=W.cpu(),W_rank=rank,selection_importance=importance.cpu(),selection_modes=chosen.cpu(),class_rows=class_rows,dual_weights=pi,source=cpu(sp),environment=cpu(ep),frames=frames,rms=rms.cpu(),neighborhood_before=nb.cpu(),neighborhood_after=na.cpu(),source_indices=ids,environment_indices=env['indices']['proxy'],trials=trials,cfg=cfg),out/'diagnostic.pt')
    attrs=dict(source=attribution(sp,p),environment=attribution(ep,pe));save_json(out/'signed_margin_attribution.json',attrs)
    result=dict(status='mechanism_supported_requires_fresh_confirmation' if support else ('closed_source_finite_infeasible' if accepted is None else 'closed_input_conditioned_transport_no_finite_multiclass_signal'),mechanism_supported=support,new_best=False,new_release_weights=0,
        parameter_distance=trials[-1]['parameter_distance'],accepted_source_step=accepted,source_ray=trials,source=source,neighborhood=neighborhood,environment=environment,class_game=game,
        output_basis_dimension=nout,kernel_constraint_rank=rank,kernel_free_dimension=p['A'].shape[0]-rank,input_task_rank=cfg['probe_rank'],source_matrix_kernel_error=float((p['B']@E).norm()/E.norm()),
        exact_basis_kernel_error=float((p['B']@U).abs().max()),core_first_order_margin_error=float((normals@U).abs().max()),task_projected_leakage_singular_values=sg.cpu().tolist(),
        full_replay=checks,new_model_forwards=sum(v['n'] for v in checks.values()),gradient_parameter_updates=0,source_sha256=digest(__file__),diagnostic_sha256=digest(out/'diagnostic.pt'),checkpoint_sha256=pins,
        source_natural_n=len(ids),source_task_n=len(data['r']),neighborhood_n=len(nb),old_candidate_loaded=False,peer_loaded=False,test_loaded=False,confirmation_loaded=False,continuous_neighborhood_certificate=False,
        wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitation='One input-conditioned kernel chart; one class-conic direction; at most8 source-only feasibility steps, reserve.25 within unchanged task budget.5. All geometry uses legal construction data. Reference is one shared actual public-base interpolation, not an unknown-peer distribution. Finite samples are not continuous/neighborhood or unseen-data proof. No use of finite reference losses to select direction, step, rank or seed. No release/trueTA/AGA unless fresh independent confirmation passes.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',dict(fit=ids,proxy=env['indices']['proxy']));print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();cfg=json.loads(Path(args.protocol).read_text());initialize()
    for name,pin in cfg['source_sha256'].items():assert digest(name)==pin,name
    run(cfg)
