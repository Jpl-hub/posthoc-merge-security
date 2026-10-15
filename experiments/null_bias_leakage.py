"""One frozen source-only test of output-null bias transport, not a release search."""

import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from angular_distance import signed_residual
from attention_cascade import suffix
from boundary_game import risk_first
from classification_completion import readout
from clip_benchmark import digest
from covariance_noise import dataset
from cross_module_distance import CrossProblem,extended_parameters,final_hidden,normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader,save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models,gelu,initialize


def slope(z):
    s=torch.sigmoid(1.702*z);return s+1.702*z*s*(1-s)


def analyze_scores(before,after):
    n,c=before.shape;y=before.argmax(1);rows=torch.arange(n,device=before.device);mask=torch.ones_like(before,dtype=torch.bool);mask[rows,y]=False
    s=before-before.mean(1,keepdim=True);s1=after-after.mean(1,keepdim=True)
    alpha=(s*s1).sum(1)/s.square().sum(1);res=s1-alpha[:,None]*s
    gap=before[rows,y,None]-before;adverse=res-res[rows,y,None]-alpha[:,None]*gap
    crossed=(adverse.masked_fill(~mask,-torch.inf).max(1).values>0)
    changed=after.argmax(1)!=y;assert torch.equal(crossed,changed)
    identity=(after[rows,y,None]-after-(alpha[:,None]*gap+res[rows,y,None]-res)).abs().max()
    ratio=(after[rows,y,None]-after)/gap.masked_fill(~mask,1.)
    return dict(n=n,changed=int(changed.sum()),classes_changed=sum(bool((changed&(y==i)).any()) for i in range(c)),
        per_class=[dict(label=i,n=int((y==i).sum()),changed=int((changed&(y==i)).sum())) for i in range(c)],
        score_decomposition_error=float(identity),positive_scales=int((alpha>0).sum()),
        worst_task_drift=float(signed_residual(before,after).abs().max()),
        minimum_gap_ratio=float(ratio.masked_fill(~mask,torch.inf).min()),
        maximum_signed_crossing_excess=float(adverse.masked_fill(~mask,-torch.inf).max())),dict(alpha=alpha,residual=res,adverse=adverse,gap=gap)


def finite(data,p,d,factor=1.):
    r=data['r'];z=normalized_input(r,p)@p['A'].T+p['a'];old=gelu(z)+data.get('b_offset',0.)
    base=data['rounding']+r+old@p['B'].T+p['b']
    linear=(factor*.5*d)@p['B'].T
    even=(gelu(z+factor*d)-.5*(z+factor*d))-(gelu(z)-.5*z)
    dh=linear+even@p['B'].T
    direct=(gelu(z+factor*d)-gelu(z))@p['B'].T
    assert float((direct-dh).abs().max())<1e-10
    before=scores(base,p);after=scores(base+dh,p);report,parts=analyze_scores(before,after)
    report.update(linear_hidden_norm=float(linear.norm()),even_hidden_rms=float((even@p['B'].T).square().sum(1).mean().sqrt()),
        exact_hidden_split_error=float((direct-dh).abs().max()))
    return report,dict(before=before,after=after,h=base,dh=dh,linear=linear,even=even,**parts)


def raw_readout(p):
    m=p['gamma'][:,None]*p['projection'];m=m-m.mean(0)
    return m,p['beta']@p['projection']


def neighborhood_bound(h,p,F,radius,constants):
    """Uniform Euclidean-ball Hessian bound, not sampled curvature certification."""
    L,c=raw_readout(p);width=h.shape[-1];sd=((h-h.mean()).square().mean()+p['eps']).sqrt()
    tn=(h@L+sd*c).norm();ql,bn,fn=constants
    m1=1.5*bn*fn;m2=1.5*bn*fn.square()
    low_sigma=sd-m1*radius/math.sqrt(width);low_t=tn-ql*m1*radius
    if min(float(low_sigma),float(low_t))<=0:return None
    weights=p['head'][:,None]-p['head'][None,:];wn=weights.norm(dim=-1)
    grad=wn*ql/low_t
    hess=wn*(6*ql.square()/low_t.square()+c.norm()/(width*low_sigma*low_t))
    return grad*m1,hess*m1.square()+grad*m2


@torch.no_grad()
def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for name,pin in cfg['inputs'].items():assert digest(name)==pin,name
    cal=torch.load(cfg['calibration_reference'],map_location='cpu');env=torch.load(cfg['environment_cache'],map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);pcpu=extended_parameters(expert,head);p=on_device(pcpu);p0=on_device(extended_parameters(anchor,head))
    for name,v in cal['parameters'].items():
        if torch.is_tensor(v):assert torch.equal(v,pcpu[name]),name
    space,_=readout(p);uB,sB,vB=torch.linalg.svd(p['B'],full_matrices=False)
    assert (sB>torch.finfo(sB.dtype).eps*max(p['B'].shape)*sB[0]).all()
    task=p['B']-p0['B'];G=space['Q'].T@task
    G=G-(G@vB.T)@vB
    ug,sg,vg=torch.linalg.svd(G,full_matrices=False);direction=vg[0]
    if direction[direction.abs().argmax()]<0:direction=-direction
    d=cfg['parameter_budget']*(1-cfg['export_reserve'])*p['a'].norm()*direction
    effective=(p['a'].cpu()+d.cpu()).float().double().cuda()-p['a']
    assert float(effective.norm()/p['a'].norm())<=cfg['parameter_budget']
    raw,ids,n=risk_first(cal['cache'],cal['indices'],cfg['contexts_per_class'],p)
    cp=CrossProblem.__new__(CrossProblem);cp.cfg=dict(probe_contexts=n,finite_step=cfg['finite_step'],full_B_completion=True)
    cp.r=cfg['probe_rank'];cp.basis={'A':on_device(cal['bases']['O']),'B':on_device(cal['bases']['B'])};cp.na=p['O'].norm();cp.nb=p['B'].norm();cp.a_size=cp.r**2
    cp.b_rows=p['B'].shape[0];cp.b_width=p['B'].shape[1];cp.output_basis=None
    data=cp.prepare(raw,p,True);source,sp=finite(data,p,effective)
    # A task-SVD coordinate ball and its complete normal/tangent frame.
    ua,sa,va=torch.linalg.svd(p['A']-p0['A'],full_matrices=False);va=va[:cfg['probe_rank']]
    x=normalized_input(on_device(raw)['r'],p);rms=(x@va.T).square().mean(0).sqrt();F=(p['A']@va.T)*rms[None]
    normals=[];tangents=[];neighborhood=[];bounds=[];hstep=cfg['derivative_step'];maxfd=0.
    pp=p|{'a':p['a']+effective}
    L,cc=raw_readout(p);constants=(torch.linalg.matrix_norm(L,2)+cc.norm()/math.sqrt(len(p['b'])),
        torch.linalg.matrix_norm(p['B'],2),torch.linalg.matrix_norm(F,2))
    for i in range(n):
        ri=on_device(raw)['r'][i:i+1];hi=on_device(raw)['h'][i:i+1];jh,_,_,_=raw_jacobians(dict(r=ri,h=hi),p)
        pre=x[i]@p['A'].T+p['a'];der=(jh[0]@p['B'])*slope(pre)[None];ju=der@F
        y=int(sp['before'][i].argmax());normal=ju[y,None]-ju;normal=normal[torch.arange(len(normal),device='cuda')!=y]
        _,sv,frame=torch.linalg.svd(normal,full_matrices=True);rank=int((sv>torch.finfo(sv.dtype).eps*max(normal.shape)*sv[0]).sum())
        normals.append(rank);tangents.append(cfg['probe_rank']-rank)
        shifts=cfg['neighborhood_radius']*torch.cat([frame,-frame]);pre_shifts=shifts@F.T
        h=hi+((gelu(pre[None]+pre_shifts)-gelu(pre))@p['B'].T)
        hr=hi+((gelu(pre[None]+pre_shifts+effective)-gelu(pre))@p['B'].T)
        bs,rs=scores(h,p),scores(hr,p);nr,npieces=analyze_scores(bs,rs)
        nr.update(source_position=i,teacher_class=y,normal_rank=rank,tangent_dimension=cfg['probe_rank']-rank,
            normal_changes=int(((bs.argmax(1)!=rs.argmax(1))[:rank]).sum()+((bs.argmax(1)!=rs.argmax(1))[cfg['probe_rank']:cfg['probe_rank']+rank]).sum()))
        neighborhood.append(nr)
        hrel=hi+(gelu(pre+effective)-gelu(pre))@p['B'].T
        jhr,_,_,_=raw_jacobians(dict(r=ri,h=hrel),pp)
        jur=((jhr[0]@p['B'])*slope(pre+effective)[None])@F
        jgap=(ju[y,None]-ju);jerr=(jur[y,None]-jur)-jgap
        center=scores(hi,p)[0];released=scores(hrel,p)[0];gap=center[y]-center;err=(released[y]-released)-gap
        ref=neighborhood_bound(hi[0],p,F,cfg['neighborhood_radius'],constants);rel=neighborhood_bound(hrel[0],pp,F,cfg['neighborhood_radius'],constants)
        if ref is not None and rel is not None:
            rho=cfg['neighborhood_radius'];lower=gap-rho*jgap.norm(dim=1)-.5*rho*rho*ref[1][y]
            upper=err.abs()+rho*jerr.norm(dim=1)+.5*rho*rho*(ref[1][y]+rel[1][y]);mask=torch.arange(len(gap),device='cuda')!=y
            cert=(lower[mask]>0).all() and (upper[mask]<=cfg['task_budget']*lower[mask]).all()
            bounds.append(dict(source_position=i,certified=bool(cert),denominator_nonzero_proved=True,
                minimum_teacher_gap_lower=float(lower[mask].min()),maximum_error_upper=float(upper[mask].max())))
        else:bounds.append(dict(source_position=i,certified=False,denominator_nonzero_proved=False))
        q=frame[0];plus=hi+((gelu(pre+hstep*(F@q))-gelu(pre))@p['B'].T);minus=hi+((gelu(pre-hstep*(F@q))-gelu(pre))@p['B'].T)
        fd=(scores(plus,p)-scores(minus,p))[0]/(2*hstep);maxfd=max(maxfd,float((fd-ju@q).abs().max()))
    assert maxfd<cfg['derivative_tolerance']
    # This is one real public-base interpolation model, not stitched states.
    pe=on_device(env['proxy_parameters']);rawenv=on_device(env['proxy']);hout,ctx=suffix(rawenv['H'].double(),pe)
    edata=dict(r=ctx['r'],rounding=rawenv['rounding'],b_offset=torch.zeros(len(hout),p['A'].shape[0],device='cuda',dtype=torch.float64))
    environment,ep=finite(edata,pe,effective,cfg['coefficient'])
    np.testing.assert_allclose(scores(hout+rawenv['rounding'],pe).cpu(),((rawenv['actual']/rawenv['actual'].norm(dim=1,keepdim=True))@pe['head'].T).cpu(),atol=5e-4,rtol=0)
    # Class-adversary dual bound for the LINEAR leakage, ignoring nonlinear even/LN terms.
    lm=pe['gamma'][:,None]*(pe['projection']@(pe['head']-pe['head'].mean(0)).T);lm-=lm.mean(0)
    bias=pe['beta']@pe['projection']@(pe['head']-pe['head'].mean(0)).T
    sigma=((ep['h']-ep['h'].mean(1,keepdim=True)).square().mean(1)+pe['eps']).sqrt()
    numerator=ep['h']@lm+sigma[:,None]*bias;y=ep['before'].argmax(1);rows=torch.arange(len(y),device='cuda')
    temp=ep['before'].clone();temp[rows,y]=-torch.inf;j=temp.argmax(1)
    coeff=-(cfg['coefficient']*.5)*((lm[:,y]-lm[:,j]).T@pe['B'])/(numerator[rows,y]-numerator[rows,j])[:,None]
    coeff=coeff-(coeff@vB.T)@vB;class_rows=torch.stack([coeff[y==c].mean(0) for c in range(len(p['head']))]).cpu().numpy()
    C=class_rows.shape[0];A=np.r_[np.c_[class_rows.T,-np.ones(class_rows.shape[1])],np.c_[-class_rows.T,-np.ones(class_rows.shape[1])]]
    lp=linprog(np.r_[np.zeros(C),1.],A_ub=A,b_ub=np.zeros(len(A)),A_eq=np.r_[np.ones(C),0.][None],b_eq=[1.],bounds=[(0,None)]*(C+1),method='highs')
    assert lp.success
    weights=lp.x[:C];dual_norm=float(np.linalg.norm(weights@class_rows));dual_upper=cfg['parameter_budget']*float(p['a'].norm())*dual_norm
    eligible=bool(source['worst_task_drift']<=cfg['task_budget'] and max(v['worst_task_drift'] for v in neighborhood)<=cfg['task_budget']
        and source['changed']==0 and environment['changed']>=cfg['minimum_environment_changes'] and environment['classes_changed']>=cfg['minimum_environment_classes'])
    # Verify actual FP32 materialization, with source-only pictures.
    released=copy.deepcopy(expert);key='model.visual.transformer.resblocks.11.mlp.c_fc.bias';released.state_dict()[key].copy_((p['a']+effective).float().cpu())
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);take=ids[:cfg['full_replay_n']];checks={}
    for name,model,expected in [('original',expert,sp['before'][:len(take)].argmax(1)),('prototype',released,sp['after'][:len(take)].argmax(1))]:
        model.cuda().eval();head.cuda().eval();actual=[]
        for images,_ in make_loader(training,take,64):actual.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu();head.cpu();assert torch.equal(torch.cat(actual),expected.cpu()),name;checks[name]=dict(n=len(take),all_predictions_match=True)
    torch.save(dict(delta_exact=d.cpu(),delta_FP32=effective.cpu(),direction=direction.cpu(),class_rows=class_rows,dual_weights=weights,
        basis_A=va.cpu(),rms_A=rms.cpu(),source= {k:v.cpu() for k,v in sp.items()},environment={k:v.cpu() for k,v in ep.items()},
        neighborhood=neighborhood,bounds=bounds,source_indices=ids,environment_indices=env['indices']['proxy'],cfg=cfg),out/'diagnostic.pt')
    result=dict(status='mechanism_supported_requires_fresh_confirmation' if eligible else 'closed_output_null_bias_mechanism',
        mechanism_supported=eligible,new_release_weights=0,new_best=False,parameter_distance=float(effective.norm()/p['a'].norm()),
        exact_bias_output_kernel_error=float((p['B']@d).norm()/d.norm()),FP32_bias_output_kernel_error=float((p['B']@effective).norm()/effective.norm()),
        task_SVD_projected_leakage_singular_values=sg.cpu().tolist(),source=source,environment=environment,
        neighborhood_inputs=sum(v['n'] for v in neighborhood),neighborhood_worst=max(v['worst_task_drift'] for v in neighborhood),
        certified_neighborhoods=sum(v['certified'] for v in bounds),neighborhood_contexts=n,maximum_task_coordinate_JVP_error=maxfd,
        linear_class_adversary_dual_upper=dual_upper,dual_simplex_error=float(abs(weights.sum()-1)),dual_weights=weights.tolist(),
        source_sha256=digest(__file__),diagnostic_sha256=digest(out/'diagnostic.pt'),checkpoint_sha256=pins,
        source_natural_n=len(ids),source_task_inputs=source['n'],new_full_model_forwards=2*len(take),full_replay=checks,
        gradient_parameter_updates=0,original_expert_updates=0,old_candidate_loaded=False,confirmation_data_loaded=False,peer_loaded=False,test_loaded=False,
        wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitation='One maximal projected task-B output-null bias shift, fixed sign/budget. One actual public interpolation is only a conditional source mechanism environment, not future-peer coverage. The linear class dual excludes even/LN terms; source finite checks decide feasibility. Continuous coordinate-ball bound is sufficient only; failure to certify is not proof of a violation. No closed-route continuation or amplitude/seed/rank scan.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',dict(fit=ids,proxy=env['indices']['proxy']));print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();cfg=json.loads(Path(a.protocol).read_text());initialize()
    for name,pin in cfg['source_sha256'].items():assert digest(name)==pin,name
    run(cfg)
