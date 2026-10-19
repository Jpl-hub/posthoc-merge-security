"""Read-only classification differential of the fixed author AGA O-head map."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest, merge
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters, final_hidden
from local_residual_transport import cpu, forward, metric
from mechanisms import load_author
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, gelu, initialize
from verify_levelset_audits import convert, raw_scores


def logical_parameter_distance(original, edits):
    """Keep unchanged Q/K/V slices out of a modified projection's denominator."""
    parts={}
    for key,value in edits.items():
        before=original[key].double();after=value.double()
        packed=key.endswith(('in_proj_weight','in_proj_bias'))
        groups=zip(['Q','K','V'],before.chunk(3,0),after.chunk(3,0)) if packed else [(None,before,after)]
        for name,a,b in groups:
            if torch.equal(a,b):continue
            assert float(a.norm())>0, 'A zero reference needs an explicit absolute budget.'
            parts[key+(':'+name if name else '')]=float((b-a).norm()/a.norm())
    return sum(v*v for v in parts.values())**.5,parts


@torch.no_grad()
def run(protocol):
    cfg=json.loads(Path(protocol).read_text());start=time.monotonic();out=Path(cfg['output'])
    for p,pin in cfg['source_sha256'].items():assert digest(p)==pin,p
    for r in ['source','AGA_reference']:assert digest(cfg[r]['path'])==cfg[r]['sha256'],r
    assert '3090' in torch.cuda.get_device_name();out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    anchor,expert,head,pins=checked_models('MNIST')
    source=torch.load(cfg['source']['path'],map_location='cpu');ids=source['indices'];assert len(ids)==512
    recovered=copy.deepcopy(expert);recovered.load_state_dict(torch.load(cfg['AGA_reference']['path'],map_location='cpu'),strict=True)
    proxy=merge(anchor,recovered,anchor,.5);p=on_device(extended_parameters(proxy,head))
    train=dataset('MNIST','data',expert.val_preprocess,True);cache=capture_attention(proxy,train,ids);c=on_device(cache)
    _,jr,z,teacher=raw_jacobians(c,p);n=len(ids)
    data=dict(r=c['r'],H=c['o_input'],rounding=c['h']-final_hidden(c['r'],p),offset=torch.zeros_like(z))
    actual=c['actual']/c['actual'].norm(dim=1,keepdim=True)@p['head'].T
    control=float((actual-teacher).abs().max());assert control<=cfg['score_tolerance']
    key='model.visual.transformer.resblocks.11.attn.out_proj.weight'
    W,A=expert.state_dict()[key].double(),anchor.state_dict()[key].double()
    u,s,vh=torch.linalg.svd(W-A,full_matrices=False);author,_=load_author(Path(cfg['author_path']))
    winner=teacher.argmax(1);rows=torch.arange(n,device='cuda');other=teacher.clone();other[rows,winner]=-torch.inf
    competitor=other.argmax(1);gap=teacher[rows,winner]-teacher[rows,competitor];assert float(gap.min())>0
    counts=torch.bincount(winner,minlength=10);assert bool((counts>0).all());weight=1./counts[winner].double()/10
    derivatives={name:[] for name in ['naive','transported','finite_difference']};operator_errors=[];finite_operators={}
    baseline=[]
    for h in range(12):
        sl=slice(64*h,64*(h+1));w,a=W[:,sl],A[:,sl]
        T=torch.linalg.solve(w.T@w+1e-8*torch.eye(64),w.T@a)
        f=w.float()@author['solve_linear_transformation'](w.float(),a.float());baseline.append((w,a,T,f))
    for i in range(cfg['rank']):
        E=W.norm()*u[:,i,None]*vh[i,None];transport=[];plus=[];minus=[];eps=cfg['finite_step']
        for h,(w,a,T,f) in enumerate(baseline):
            e=E[:,h*64:(h+1)*64];G=w.T@w+1e-8*torch.eye(64)
            dt=torch.linalg.solve(G,e.T@a-(e.T@w+w.T@e)@T);d=e@T+w@dt;transport.append(d)
            wn,an,en=w.numpy(),a.numpy(),e.numpy();gn=wn.T@wn+1e-8*np.eye(64)
            tn=np.linalg.solve(gn,wn.T@an);dn=en@tn+wn@np.linalg.solve(gn,en.T@an-(en.T@wn+wn.T@en)@tn)
            operator_errors.append(float(abs(dn-d.numpy()).max()))
            for sign,store in [(1,plus),(-1,minus)]:
                x=(w+sign*eps*e).float();store.append((x@author['solve_linear_transformation'](x,a.float())-f).double())
        dO=torch.cat(transport,1).cuda();naive=.5*torch.einsum('nci,ij,nj->nc',jr,E.cuda(),c['o_input'])
        true=.5*torch.einsum('nci,ij,nj->nc',jr,dO,c['o_input']);finite=[]
        for parts in [plus,minus]:finite.append(forward(data,p,p|dict(O=p['O']+.5*torch.cat(parts,1).cuda())))
        fd=(finite[0]-finite[1])/(2*eps)
        if i in [0,cfg['rank']-1]:finite_operators[i]=[torch.cat(parts,1) for parts in [plus,minus]]
        for name,value in [('naive',naive),('transported',true),('finite_difference',fd)]:derivatives[name].append(value.cpu())
    raw={k:torch.stack(v) for k,v in derivatives.items()};a,b,f=[raw[k].numpy() for k in ['naive','transported','finite_difference']]
    y,k=winner.cpu().numpy(),competitor.cpu().numpy();ix=np.arange(n);g=gap.cpu().numpy();weights=weight.cpu().numpy()
    gradients={label:np.sum(-(d[:,ix,y]-d[:,ix,k])/g*weights,axis=1) for label,d in [('naive',a),('transported',b)]}
    g0,g1=gradients.values();cos=float(g0@g1/(np.linalg.norm(g0)*np.linalg.norm(g1)))
    fd_error=float(np.linalg.norm(f-b)/np.linalg.norm(b));operator_error=max(operator_errors)
    report=dict(status='passed' if fd_error<=cfg['derivative_relative_tolerance'] and operator_error<=1e-8 else 'failed_fixed_derivative_control',
        rank=cfg['rank'],task_singular_values=s[:cfg['rank']].tolist(),source_n=n,source_ids_role='Previously used source construction only; not independent confirmation or new construction.',
        gradient_cosine=cos,derivative_relative_difference=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
        author_finite_difference_relative_error=fd_error,NumPy_operator_derivative_error=operator_error,full_prefix_score_control=control,
        naive_gradient=g0.tolist(),transported_gradient=g1.tolist(),checkpoint_sha256=pins,
        gradients=0,analytic_derivative_modes=cfg['rank'],new_proxy_model_forward_instances=n,
        publication_weights=0,defense_optimizer_steps=0,partner_used=False,test_used=False,confirmation_used=False,
        source_sha256=digest(__file__),protocol_sha256=digest(protocol),wall_seconds=time.monotonic()-start,
        scope='Conditional O-only derivative of the smooth float64 ridge extension, checked against quantized author output at one finite step. The prefix is public-base/AGA-source TA0.5, not an unknown peer or real defense evaluation. No direction or candidate is selected.')
    torch.save(dict(derivatives=raw,finite_operators=finite_operators,finite_step=cfg['finite_step'],cache=cache,
        p=cpu(p),teacher=teacher.cpu(),indices=ids,report=report),out/'response.pt')
    report['raw_sha256']=digest(out/'response.pt');save_json(out/'results.json',report);print(json.dumps(report),flush=True)


def verify(folder):
    out=Path(folder);report=json.loads((out/'results.json').read_text());assert digest(out/'response.pt')==report['raw_sha256']
    raw=torch.load(out/'response.pt',map_location='cpu');p=convert(raw['p']);c=convert(raw['cache']);r=c['r'];cc=r-r.mean(1,keepdims=True)
    pre=(cc/np.sqrt((cc*cc).mean(1,keepdims=True)+p['eps2'])*p['gamma2']+p['beta2'])@p['A'].T+p['a']
    h=r+(pre*expit(1.702*pre))@p['B'].T+p['b'];rounding=c['h']-h
    np.testing.assert_allclose(raw_scores(h+rounding,p),raw['teacher'],atol=1e-8,rtol=1e-9)
    a,b,f=[raw['derivatives'][k].numpy() for k in ['naive','transported','finite_difference']]
    finite_error=0.
    for mode,operators in raw['finite_operators'].items():
        scores=[]
        for dO in operators:
            rp=r+.5*c['o_input']@dO.numpy().T;cp=rp-rp.mean(1,keepdims=True)
            z=(cp/np.sqrt((cp*cp).mean(1,keepdims=True)+p['eps2'])*p['gamma2']+p['beta2'])@p['A'].T+p['a']
            scores.append(raw_scores(rounding+rp+(z*expit(1.702*z))@p['B'].T+p['b'],p))
        fd=(scores[0]-scores[1])/(2*raw['finite_step']);finite_error=max(finite_error,float(abs(fd-f[mode]).max()))
    assert finite_error<1e-7
    error=float(np.linalg.norm(f-b)/np.linalg.norm(b));assert abs(error-report['author_finite_difference_relative_error'])<1e-10
    s=raw['teacher'].numpy();y=s.argmax(1);ix=np.arange(len(y));other=s.copy();other[ix,y]=-np.inf;k=other.argmax(1)
    gap=s[ix,y]-s[ix,k];w=1./np.bincount(y,minlength=10)[y]/10
    gradients=[np.sum(-(d[:,ix,y]-d[:,ix,k])/gap*w,axis=1) for d in [a,b]]
    for got,name in zip(gradients,['naive_gradient','transported_gradient']):np.testing.assert_allclose(got,report[name],atol=1e-10,rtol=1e-10)
    cosine=float(gradients[0]@gradients[1]/(np.linalg.norm(gradients[0])*np.linalg.norm(gradients[1])))
    assert abs(cosine-report['gradient_cosine'])<1e-10
    save_json(out/'independent_validation.json',dict(status='passed_recount',source_sha256=digest(__file__),raw_sha256=report['raw_sha256'],
        gradient_cosine=cosine,numpy_finite_tail_derivative_error=finite_error,author_derivative_control_status=report['status'],new_images=0,new_releases=0,new_gradients=0))


@torch.no_grad()
def functional_cost(protocol):
    """Bound the closed O point's linearized B cost, without creating a release."""
    from probe_margin_distance import scores
    cfg=json.loads(Path(protocol).read_text());start=time.monotonic();out=Path(cfg['output'])
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    assert digest(cfg['diagnostic']['path'])==cfg['diagnostic']['sha256']
    assert '3090' in torch.cuda.get_device_name();out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    old=torch.load(cfg['diagnostic']['path'],map_location='cpu');ids=old['indices']['fit'];assert len(ids)==4096
    _,expert,head,pins=checked_models('MNIST');train=dataset('MNIST','data',expert.val_preprocess,True)
    original=extended_parameters(expert,head)
    assert all(torch.equal(old['p'][k],original[k]) for k in ['O','A','B','head'])
    c=on_device(capture_attention(expert,train,ids));p=on_device(old['p']);teacher=scores(c['h'],p)
    control=float((teacher-c['actual']/c['actual'].norm(dim=1,keepdim=True)@p['head'].T).abs().max());assert control<=cfg['score_tolerance']
    r=c['r']+c['o_input']@(old['q']['O']-old['p']['O']).cuda().T
    h=c['h']-final_hidden(c['r'],p)+final_hidden(r,p);jh,_,z,current=raw_jacobians(dict(r=r,h=h),p)
    y=teacher.argmax(1);ix=torch.arange(len(y),device='cuda');gap=teacher[ix,y,None]-teacher
    mask=torch.ones_like(gap,dtype=torch.bool);mask[ix,y]=False;assert bool((gap[mask]>0).all());gap[ix,y]=1
    j=((jh[ix,y,None]-jh)/gap[:,:,None])[mask].reshape(len(y),9,768)
    b=(1-(current[ix,y,None]-current)/gap)[mask].reshape(len(y),9);nb=p['B'].norm()
    def adjoint(lam):return nb*torch.einsum('ncd,nc->nd',j,lam).T@z
    def kernel(lam):return nb*torch.einsum('ncd,nd->nc',j,z@adjoint(lam).T)
    diagonal=nb.square()*j.square().sum(2)*z.square().sum(1)[:,None];assert float(diagonal.min())>0
    dual=torch.zeros_like(b);res=b.clone();pre=res/diagonal;direction=pre.clone();rho=(res*pre).sum();steps=0
    for steps in range(1,cfg['cg_iterations']+1):
        value=kernel(direction);den=(direction*value).sum();assert float(den)>0
        step=rho/den;dual+=step*direction;res-=step*value
        if steps%20==0:res=b-kernel(dual)
        if float(res.norm()/b.norm())<=cfg['cg_relative_tolerance']:break
        pre=res/diagonal;next_rho=(res*pre).sum();direction=pre+(next_rho/rho)*direction;rho=next_rho
    image=kernel(dual);adj=adjoint(dual);cost=float(adj.norm());residual=image-b
    lower=max(0.,float((dual*b).sum()-cfg['task_budget']*dual.abs().sum()))/cost
    linear_feasible=float(residual.abs().max())<=cfg['task_budget']
    hdir=nb*z@(adj/cost).T;eps=cfg['finite_step'];plus,minus=scores(h+eps*hdir,p),scores(h-eps*hdir,p)
    ds=(plus-minus)/(2*eps);fd=((ds[ix,y,None]-ds)/gap)[mask].reshape(len(y),9)
    predicted=torch.einsum('ncd,nd->nc',j,hdir);fd_error=float((fd-predicted).norm()/predicted.norm())
    distance_O=float((old['q']['O']-old['p']['O']).norm()/old['p']['O'].norm())
    report=dict(status='passed_linear_cost_control' if fd_error<=cfg['derivative_relative_tolerance'] else 'failed_fixed_cost_control',
        n=len(ids),linear_constraints=b.numel(),PCG_iterations=steps,linear_residual_max=float(residual.abs().max()),
        linear_residual_relative=float(residual.norm()/b.norm()),linear_B_cost_lower_bound=lower,
        linear_B_cost_upper_bound=cost if linear_feasible else None,linear_solution_norm=cost,linear_band_feasible=linear_feasible,
        fixed_O_distance=distance_O,original_joint_budget=0.5,task_budget=cfg['task_budget'],
        linear_joint_cost_lower_bound=(distance_O**2+lower**2)**.5,
        linear_joint_cost_upper_bound=(distance_O**2+cost**2)**.5 if linear_feasible else None,
        finite_difference_relative_error=fd_error,full_prefix_score_control=control,source_sha256=digest(__file__),protocol_sha256=digest(protocol),checkpoint_sha256=pins,
        original_model_repeat_forwards=len(ids),new_training_images=0,reverse_gradients=0,analytic_jacobian_inputs=len(ids),new_releases=0,
        closed_point_repaired=False,confirmation_or_test_used=False,wall_seconds=time.monotonic()-start,
        scope='Bounds apply only to the full normalized-class-margin linearization at the frozen O-only point on its original4096 fit inputs. Not finite task feasibility, a nonlinear cost bound, a replacement B, or permission to reopen the closed fixed point.')
    torch.save(dict(j=j.cpu(),z=z.cpu(),dual=dual.cpu(),b=b.cpu(),image=image.cpu(),norm_B=nb.cpu(),hidden=h.cpu(),
        hidden_direction=hdir.cpu(),p=cpu(p),winner=y.cpu(),gaps=gap[mask].reshape(len(y),9).cpu(),linear_probe=predicted.cpu(),finite_step=eps),out/'linear_cost.pt')
    report['raw_sha256']=digest(out/'linear_cost.pt');save_json(out/'results.json',report);print(json.dumps(report),flush=True)


def verify_cost(folder):
    out=Path(folder);report=json.loads((out/'results.json').read_text());assert digest(out/'linear_cost.pt')==report['raw_sha256']
    r=torch.load(out/'linear_cost.pt',map_location='cpu');j,z,dual,b,image=[r[k].numpy() for k in ['j','z','dual','b','image']];nb=float(r['norm_B'])
    adj=nb*np.einsum('ncd,nc->nd',j,dual).T@z;actual=nb*np.einsum('ncd,nd->nc',j,z@adj.T)
    error=float(np.linalg.norm(actual-image)/np.linalg.norm(image));assert error<1e-9
    cost=float(np.linalg.norm(adj));lower=max(0.,float(np.sum(dual*b)-report['task_budget']*np.abs(dual).sum()))/cost
    assert abs(cost-report['linear_solution_norm'])<1e-8 and abs(lower-report['linear_B_cost_lower_bound'])<1e-8
    assert bool(abs(actual-b).max()<=report['task_budget'])==report['linear_band_feasible']
    p=convert(r['p']);h=r['hidden'].numpy();d=r['hidden_direction'].numpy();eps=r['finite_step'];y=r['winner'].numpy();ix=np.arange(len(y))
    ds=(raw_scores(h+eps*d,p)-raw_scores(h-eps*d,p))/(2*eps);mask=np.ones_like(ds,dtype=bool);mask[ix,y]=False
    fd=(ds[ix,y,None]-ds)[mask].reshape(len(y),9)/r['gaps'].numpy();wanted=r['linear_probe'].numpy()
    fd_error=float(np.linalg.norm(fd-wanted)/np.linalg.norm(wanted));assert abs(fd_error-report['finite_difference_relative_error'])<1e-7
    save_json(out/'independent_validation.json',dict(status='passed_linear_dual_recount',raw_sha256=report['raw_sha256'],
        source_sha256=digest(__file__),numpy_kernel_relative_error=error,numpy_linear_cost=cost,numpy_dual_lower_bound=lower,
        finite_difference_relative_error=fd_error,new_images=0,new_releases=0,closed_point_repaired=False))


@torch.no_grad()
def affine_audit(protocol):
    """Trace the frozen affine point through actual attention; never select a new point."""
    from ln_affine_release import observe
    from scipy.special import logsumexp, softmax
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);start=time.monotonic()
    validation=json.loads((out/'independent_validation.json').read_text());assert validation['status']=='passed'
    meta=json.loads((out/'execution.json').read_text());assert digest(out/'artifacts/predictions.pt')==meta['predictions_sha256']
    source=Path(cfg['prior_confirmation']).parent;assert not (out/'affine_mechanism.pt').exists()
    src=torch.load(source/'source.pt',map_location='cpu');_,expert,head,pins=checked_models('MNIST')
    ref=cfg['AGA_references']['MNIST'];assert digest(ref['path'])==ref['sha256']
    expert.load_state_dict(torch.load(ref['path'],map_location='cpu'),strict=True);changed=copy.deepcopy(expert)
    edits=torch.load(out/'artifacts/AGA_release_edits.pt',map_location='cpu');assert sorted(edits)==cfg['allowed_keys']
    for k,v in edits.items():changed.state_dict()[k].copy_(v)
    train=dataset('MNIST','data',expert.val_preprocess,True);maps=[];qs=[];ks=[];native=[];prefix=None;control=0.
    for model in [expert,changed]:
        score,x=observe(model,head,train,src['indices'],True);native.append(score)
        if prefix is None:prefix=x
        else:assert torch.equal(prefix,x),'Prefixes must be identical for the causal Q-only comparison.'
        block=model.model.visual.transformer.resblocks[11].cuda();allq=[];allk=[];allp=[]
        for part in x.split(64):
            norm=block.ln_1(part.cuda().transpose(0,1));q,k,_=torch.nn.functional.linear(norm,block.attn.in_proj_weight,block.attn.in_proj_bias).chunk(3,-1)
            q=q[0].reshape(-1,12,64).double();k=k.transpose(0,1).reshape(len(part),50,12,64).transpose(1,2).double()
            logits=torch.einsum('nhd,nhtd->nht',q,k)/8;prob=logits.softmax(-1)
            _,actual=block.attn(norm,norm,norm,need_weights=True,average_attn_weights=False)
            control=max(control,float((actual[:,:,0].double()-prob).abs().max()));allq.append(q.cpu());allk.append(k.cpu());allp.append(prob.cpu())
        block.cpu();qs.append(torch.cat(allq));ks.append(torch.cat(allk));maps.append(torch.cat(allp))
    assert control<=5e-6 and torch.equal(ks[0],ks[1])
    q0,q1=[q.numpy() for q in qs];k=ks[0].numpy();p0,p1=[p.numpy() for p in maps]
    logits=[np.einsum('nhd,nhtd->nht',q,k)/8 for q in [q0,q1]]
    error=max(float(abs(softmax(z,axis=-1)-p).max()) for z,p in zip(logits,[p0,p1]));assert error<1e-12
    delta=logits[1]-logits[0];tv=abs(p1-p0).sum(-1)/2;bound=np.tanh(np.ptp(delta,axis=-1)/4);assert np.all(tv<=bound+1e-12)
    full=torch.load(out/'artifacts/predictions.pt',map_location='cpu');classes=[];margins=[]
    for key,row in full.items():
        if not key.startswith('released/'):continue
        original=full[key.replace('released/','original/')];a,b=[v['logits'].double().numpy() for v in [original,row]];labels=row['target'].numpy();n=len(labels);ix=np.arange(n)
        logp=b-logsumexp(b,axis=1,keepdims=True);oldp=a-logsumexp(a,axis=1,keepdims=True)
        winner=a.argmax(1);other=a.copy();other[ix,winner]=-np.inf;runner=other.argmax(1);gap=a[ix,winner]-a[ix,runner]
        harm=1-(b[ix,winner]-b[ix,runner])/gap
        margins.append(dict(key=key,signed_margin_harm_quantiles=np.quantile(harm,[0,.5,.95,1]).tolist(),teacher_runner_crossings=int((harm>1).sum())))
        for c in range(10):
            mask=labels==c;pred0=a.argmax(1);pred1=b.argmax(1)
            classes.append(dict(key=key,class_id=c,n=int(mask.sum()),original_accuracy=float((pred0[mask]==labels[mask]).mean()),released_accuracy=float((pred1[mask]==labels[mask]).mean()),
                original_mean_NLL=float(-oldp[ix[mask],labels[mask]].mean()),released_mean_NLL=float(-logp[ix[mask],labels[mask]].mean()),
                lost=int(((pred0==labels)&(pred1!=labels)&mask).sum()),gained=int(((pred0!=labels)&(pred1==labels)&mask).sum())))
    torch.save(dict(q=qs,k=ks[0],attention=maps,source_scores=native,source_indices=src['indices']),out/'affine_mechanism.pt')
    report=dict(status='passed_numpy_attention_and_class_loss_recount',native_attention_control=control,numpy_attention_error=error,
        source_query_relative_change=float(np.linalg.norm(q1-q0)/np.linalg.norm(q0)),source_attention_TV_quantiles=np.quantile(tv,[0,.5,.95,1]).tolist(),
        source_attention_TV_maximum_by_head=tv.max(0).tolist(),softmax_TV_bound_valid=True,
        source_changed_predictions=int((native[0].argmax(1)!=native[1].argmax(1)).sum()),source_n=len(src['indices']),
        repeated_original_source_image_forwards=2*len(src['indices']),new_images=0,new_weights=0,gradients=0,
        full_class_loss=classes,full_signed_margin_harm=margins,wall_seconds=time.monotonic()-start,
        source_sha256=digest(__file__),raw_sha256=digest(out/'affine_mechanism.pt'),full_predictions_sha256=meta['predictions_sha256'],
        scope='Posthoc explanation of one frozen failed point: identical actual AGA prefixes, full native attention control, NumPy softmax/TV bound and existing full evaluation losses. Not direction selection or an unknown-partner guarantee.')
    save_json(out/'mechanism_and_class_loss.json',report);print(json.dumps({k:v for k,v in report.items() if not k.startswith('full_')}),flush=True)


@torch.no_grad()
def branch_norm_check(protocol):
    """A finite skip/branch norm identity, not a retry of the old readout reflection."""
    from probe_margin_distance import scores
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);start=time.monotonic()
    assert digest(__file__)==cfg['source_sha256'] and digest(cfg['source']['path'])==cfg['source']['sha256']
    out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    raw=torch.load(cfg['source']['path'],map_location='cpu');anchor,expert,head,pins=checked_models('MNIST')
    p=extended_parameters(expert,head);pg=on_device(p);block=expert.model.visual.transformer.resblocks[11].cuda();rs=[];zs=[]
    for part in raw['inputs'].split(64):
        x=part.cuda().transpose(0,1);v=block.ln_1(x);a=block.attn(v,v,v,need_weights=False)[0]
        r=x[0]+a[0];z=block.mlp.gelu(block.mlp.c_fc(block.ln_2(r)));rs.append(r.double());zs.append(z.double())
    block.cpu();r,z=torch.cat(rs),torch.cat(zs);s=r+pg['b'];m=z@pg['B'].T;h=s+m
    d=h.shape[1];readout=torch.cat([torch.ones((d,1),device='cuda'),pg['gamma'][:,None]*pg['projection']],1)
    u,sv,_=torch.linalg.svd(readout,full_matrices=True);rank=int((sv>sv[0]*1e-10).sum());kernel=u[:,rank:]
    skip=s@kernel;gram=skip.T@skip/len(s);eigenvalues,v=torch.linalg.eigh(gram);axes=kernel@v
    cumulative=4*(axes.T@pg['B']).square().sum(1).cumsum(0)/pg['B'].square().sum()
    k=int((cumulative<=(cfg['parameter_budget']*(1-1e-6))**2).sum());assert k>0
    N=axes[:,:k];ideal=pg['B']-2*N@(N.T@pg['B']);B=ideal.float().double()
    distance=float((B-pg['B']).norm()/pg['B'].norm());assert distance<=cfg['parameter_budget']
    before=scores(h,pg);full_control=float((before.cpu()-raw['before']).abs().max());assert full_control<=cfg['score_tolerance']
    groups={};sigma0=(h-h.mean(1,keepdim=True)).square().mean(1)+pg['eps'];identity_error=0.;np_error=0.
    pn=convert(p)
    for lam in [1.,.5]:
        hi=s+z@(pg['B']+lam*(B-pg['B'])).T;after=scores(hi,pg)
        exact=s+m-2*lam*(m@N)@N.T
        measured=(exact-exact.mean(1,keepdim=True)).square().mean(1)+pg['eps']-sigma0
        rhs=(-4*lam*((s@N)*(m@N)).sum(1)-4*lam*(1-lam)*(m@N).square().sum(1))/d
        error=float((measured-rhs).abs().max());identity_error=max(identity_error,error);assert error<1e-8
        ns=raw_scores(hi.cpu().numpy(),pn);np_error=max(np_error,float(abs(ns-after.cpu().numpy()).max()));assert np_error<1e-8
        groups[str(lam)]=dict(metric=metric(before,after,len(h)),
            relative_sigma_change_quantiles=torch.quantile(((sigma0+measured).sqrt()/sigma0.sqrt()-1),torch.tensor([0.,.5,1.],device='cuda',dtype=torch.float64)).cpu().tolist())
    report=dict(status='completed_read_only_mechanism_check',natural_n=len(h),reflection_rank=k,kernel_dimension=kernel.shape[1],parameter_distance=distance,
        source_task_feasible=groups['1.0']['metric']['worst_task_drift']<=cfg['task_budget'],groups=groups,finite_variance_identity_error=identity_error,
        NumPy_complete_score_error=np_error,native_prefix_score_control=full_control,skip_covariance_eigenvalues=eigenvalues.cpu().tolist(),
        kernel_readout_error=float((N.T@readout).abs().max()),expert_updates=0,gradients=0,new_images=0,
        prefix_block_replays=len(h),new_confirmations=0,new_released_models=0,source_sha256=digest(__file__),checkpoint_sha256=pins,
        scope='Same original construction inputs; ideal conditional branch reflection plus one quantized-B counterfactual, not a native release/confirmation or real fusion result. Midpoint is source self-chord only, not a universal defense gate.',wall_seconds=time.monotonic()-start)
    torch.save(dict(N=N.cpu(),r=r.cpu(),z=z.cpu(),p=p,groups=groups,source_indices=raw['indices']),out/'mechanism.pt')
    report['raw_sha256']=digest(out/'mechanism.pt');save_json(out/'results.json',report);print(json.dumps(report),flush=True)


def verify_branch_norm(folder):
    out=Path(folder);report=json.loads((out/'results.json').read_text());assert digest(out/'mechanism.pt')==report['raw_sha256']
    raw=torch.load(out/'mechanism.pt',map_location='cpu');N,r,z=[raw[k].numpy() for k in ['N','r','z']];p=convert(raw['p']);B=p['B']
    readout=np.column_stack([np.ones(768),p['gamma'][:,None]*p['projection']]);u,s,_=np.linalg.svd(readout,full_matrices=True);nk=u[:,int((s>s[0]*1e-10).sum()):]
    assert N.shape[1]==nk.shape[1]==report['reflection_rank']==255
    projector_error=float(abs(N@N.T-nk@nk.T).max());assert projector_error<1e-10
    q=(B-2*N@(N.T@B)).astype('float32').astype('float64');distance=float(np.linalg.norm(q-B)/np.linalg.norm(B));assert abs(distance-report['parameter_distance'])<1e-8
    skip=r+p['b'];m=z@B.T;h=skip+m;before=raw_scores(h,p);d=768;var=((h-h.mean(1,keepdims=True))**2).mean(1)+p['eps'];errors=[];sigma_error=0.
    for lam in [1.,.5]:
        after=raw_scores(skip+z@(B+lam*(q-B)).T,p);winner=before.argmax(1);ix=np.arange(len(h));gap=before[ix,winner,None]-before;gap[ix,winner]=1
        delta=after-before;rho=abs((delta[ix,winner,None]-delta)/np.maximum(gap,1e-12));rho[ix,winner]=0
        errors.append(abs(float(rho.max())-report['groups'][str(lam)]['metric']['worst_task_drift']))
        assert int((before.argmax(1)!=after.argmax(1)).sum())==report['groups'][str(lam)]['metric']['changed']
        exact=skip+m-2*lam*(m@N)@N.T;measured=((exact-exact.mean(1,keepdims=True))**2).mean(1)+p['eps']-var
        rhs=(-4*lam*((skip@N)*(m@N)).sum(1)-4*lam*(1-lam)*((m@N)**2).sum(1))/d
        sigma_error=max(sigma_error,float(abs(measured-rhs).max()))
    assert max(errors)<1e-6 and sigma_error<1e-8
    # Uniform-source dual bound for the specific proposed variance-contraction mechanism.
    sk,mb=skip@nk,m@nk;cross=(sk.T@mb+mb.T@sk)/(2*len(h));middle=2*cross+mb.T@mb/len(h)
    ev,em=[np.linalg.eigvalsh(x) for x in [cross,middle]]
    certificate=dict(source_count=len(h),source_endpoint_cross_eigenvalue_range=[float(ev[0]),float(ev[-1])],
        source_midpoint_contraction_eigenvalue_range=[float(em[0]),float(em[-1])],
        uniform_positive_midpoint_variance_contraction_excluded=bool(em[-1]<-1e-10),
        exact_source_variance_preservation_nonzero_reflection_excluded=bool(ev[-1]<-1e-10 or ev[0]>1e-10),
        scope='Uniform average is a legal simplex dual only for this scalar skip/branch variance mechanism in the fixed full visual-readout kernel. Not a classification-defense impossibility or a necessary gate for other methods.')
    # A class margin numerator is affine in sigma. If it stays nonnegative at
    # both ends, no interior contraction can change that teacher decision.
    linear=((h-h.mean(1,keepdims=True))*p['gamma'])@p['projection']@p['head'].T
    bias=p['beta']@p['projection']@p['head'].T;teacher=before.argmax(1);ix=np.arange(len(h))
    zero_gap=linear[ix,teacher,None]-linear;bias_gap=bias[teacher,None]-bias[None,:]
    can_contract=(zero_gap<0).any(1);can_any=((zero_gap<0)|(bias_gap<0)).any(1)
    mask=~np.eye(10,dtype=bool)[teacher]
    pt=raw['p'];ht=torch.from_numpy(h);at=((ht-ht.mean(1,keepdim=True))*pt['gamma'])@pt['projection']@pt['head'].T
    zero_control=float(abs(at.numpy()-linear).max());assert zero_control<1e-8
    envelope=dict(source_n=len(h),maximum_teacher_changes_under_any_pointwise_variance_contraction=int(can_contract.sum()),
        maximum_teacher_changes_under_any_positive_pointwise_variance=int(can_any.sum()),
        minimum_zero_sigma_teacher_margin=float(zero_gap[mask].min()),NumPy_torch_zero_sigma_numerator_error=zero_control,
        contraction_by_teacher_class=[dict(class_id=c,n=int((teacher==c).sum()),maximum_changed=int((can_contract&(teacher==c)).sum())) for c in range(10)],
        scope='Exact scalar-numerator decision envelope at these fixed source hidden states and readout. Allows independent arbitrary sigma per image, hence an upper bound for any shared reflection whose only score change is sigma. Not an accuracy bound after a changed peer/readout or AGA.')
    end=skip+z@q.T;endvar=((end-end.mean(1,keepdims=True))**2).mean(1)+p['eps'];upper_sigma=np.sqrt(np.maximum(var,endvar))
    linear_delta=((end-h-(end-h).mean(1,keepdims=True))*p['gamma'])@p['projection']@p['head'].T
    leakage=abs(linear_delta[ix,teacher,None]-linear_delta)
    chord_lower=np.minimum(zero_gap,zero_gap+upper_sigma[:,None]*bias_gap)-leakage
    chord=dict(all_fixed_source_teacher_predictions_certified=bool((chord_lower[mask]>0).all()),
        minimum_numerator_margin_lower_bound=float(chord_lower[mask].min()),maximum_quantized_readout_margin_leakage=float(leakage[mask].max()),
        scope='All real interpolation coefficients0..1 in the cached last-block B-only chord, by convexity of augmented centered norm and linear margin numerators. Not a bound on task-drift0.5, all FP32 arithmetic, new inputs, a changed prefix/readout, TA with a peer, or AGA.')
    save_json(out/'independent_validation.json',dict(status='passed_numpy_finite_norm_and_score_recount',raw_sha256=report['raw_sha256'],
        source_sha256=digest(__file__),NumPy_projector_error=projector_error,NumPy_task_drift_recount_error=max(errors),NumPy_variance_identity_error=sigma_error,
        geometric_rank=255,ideal_orthogonal_family_maximum_distance=float(2*np.linalg.norm(nk.T@B)/np.linalg.norm(B)),
        quantized_prototype_distance=distance,new_images=0,new_releases=0,source_task_feasibility_is_not_independent_confirmation=True,
        finite_variance_bound=certificate,scalar_decision_envelope=envelope,source_chord_certificate=chord))


@torch.no_grad()
def branch_proxy_bound(protocol):
    """Relax all kernel reflections at one fixed, legitimate public-base merger."""
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);start=time.monotonic()
    assert digest(__file__)==cfg['source_sha256'];out.mkdir(parents=True,exist_ok=False)
    for name in ['kernel','proxy','AGA_original']:assert digest(cfg[name]['path'])==cfg[name]['sha256'],name
    save_json(out/'protocol.json',cfg);kernel=torch.load(cfg['kernel']['path'],map_location='cpu')['N'].numpy()
    raw=torch.load(cfg['proxy']['path'],map_location='cpu');p=convert(raw['p']);cache=convert(raw['cache'])
    state=torch.load(cfg['AGA_original']['path'],map_location='cpu');B=state['model.visual.transformer.resblocks.11.mlp.c_proj.weight'].double().numpy()
    r=cache['r'];center=r-r.mean(1,keepdims=True);x=center/np.sqrt((center**2).mean(1,keepdims=True)+p['eps2'])*p['gamma2']+p['beta2']
    pre=x@p['A'].T+p['a'];z=pre*expit(1.702*pre);h=cache['h'];teacher=raw['teacher'].numpy();score=raw_scores(h,p)
    control=float(abs(score-teacher).max());assert control<1e-8
    assert kernel.shape==(768,255) and float(abs(kernel.T@kernel-np.eye(255)).max())<1e-10
    assert float(abs(kernel.sum(0)).max())<1e-10
    winner=score.argmax(1);ix=np.arange(len(h));hc=h-h.mean(1,keepdims=True)
    sigma=np.sqrt((hc*hc).mean(1)+p['eps']);w=(z@B.T)@kernel;v=hc@kernel
    radius=np.linalg.norm(w,axis=1)/2;center_norm=np.linalg.norm(v-w/2,axis=1)
    other=sigma*sigma-(v*v).sum(1)/768;assert float(other.min())>=p['eps']-1e-10
    low=np.sqrt(other+(center_norm-radius)**2/768);high=np.sqrt(other+(center_norm+radius)**2/768)
    functional=(p['gamma'][:,None]*p['projection'])@p['head'].T;basis=kernel.T@functional
    linear=hc@functional;bias=p['beta']@p['projection']@p['head'].T
    t=(basis[:,winner].T[:,None,:]-basis.T[None]);norm_t=np.linalg.norm(t,axis=-1)
    wt=w@basis;dot=wt[ix,winner,None]-wt;loss=(dot+np.linalg.norm(w,axis=1)[:,None]*norm_t)/2
    linear_gap=linear[ix,winner,None]-linear;bias_gap=bias[winner,None]-bias[None]
    lower=linear_gap-loss+bias_gap*np.where(bias_gap>=0,low[:,None],high[:,None])
    mask=~np.eye(10,dtype=bool)[winner];stable=np.all(np.where(mask,lower,np.inf)>0,axis=1)
    # Full-kernel and zero-projector controls; neither selects another candidate.
    checks=[]
    for delta in [np.zeros_like(w),w]:
        altered=h-delta@kernel.T;cs=altered-altered.mean(1,keepdims=True);sd=np.sqrt((cs*cs).mean(1)+p['eps'])
        numer=cs@functional+sd[:,None]*bias;gap=numer[ix,winner,None]-numer
        assert float((gap-lower)[mask].min())>=-1e-8
        assert np.all((sd>=low-1e-10)&(sd<=high+1e-10));checks.append(int((numer.argmax(1)!=winner).sum()))
    # Independently repeat the geometric norm interval in torch float64.
    tw,tv=[torch.from_numpy(a) for a in [w,v]];tr=tw.norm(dim=1)/2;tc=(tv-tw/2).norm(dim=1)
    tl=(torch.from_numpy(other)+(tc-tr).square()/768).sqrt().numpy();th=(torch.from_numpy(other)+(tc+tr).square()/768).sqrt().numpy()
    interval_error=max(float(abs(tl-low).max()),float(abs(th-high).max()));assert interval_error<1e-10
    report=dict(status='passed_fixed_public_proxy_family_bound',n=len(h),certified_no_teacher_change=int(stable.sum()),
        maximum_possible_teacher_changes=len(h)-int(stable.sum()),all_projectors_allowed=True,parameter_budget_ignored_as_relaxation=True,
        zero_and_full_kernel_control_changes=checks,numpy_torch_interval_error=interval_error,baseline_score_control=control,
        per_teacher_class=[dict(class_id=c,n=int((winner==c).sum()),maximum_changed=int((~stable&(winner==c)).sum())) for c in range(10)],
        unknown_partner_coverage=False,real_defense_evaluation=False,new_images=0,new_model_forwards=0,new_releases=0,gradients=0,
        source_sha256=digest(__file__),inputs=cfg,wall_seconds=time.monotonic()-start,
        scope='Conservative per-image/class relaxation of all full visual-kernel B reflections at one fixed public-base/authorAGA-source TA0.5 environment. Uncertified images are not observed failures. No new direction or candidate is selected.')
    torch.save(dict(lower=lower,w=w,v=v,low=low,high=high,winner=winner,source_indices=raw['indices']),out/'bound.pt')
    report['raw_sha256']=digest(out/'bound.pt');save_json(out/'results.json',report);print(json.dumps(report),flush=True)


@torch.no_grad()
def verify_branch_proxy_bound(folder):
    out=Path(folder);report=json.loads((out/'results.json').read_text());cfg=report['inputs']
    assert digest(out/'bound.pt')==report['raw_sha256']
    for name in ['kernel','proxy','AGA_original']:assert digest(cfg[name]['path'])==cfg[name]['sha256']
    proof=torch.load(out/'bound.pt',map_location='cpu');kernel=torch.load(cfg['kernel']['path'],map_location='cpu')['N']
    raw=torch.load(cfg['proxy']['path'],map_location='cpu');p=raw['p'];h=raw['cache']['h'];r=raw['cache']['r']
    z=gelu(torch.nn.functional.layer_norm(r,(768,),p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'])
    B=torch.load(cfg['AGA_original']['path'],map_location='cpu')['model.visual.transformer.resblocks.11.mlp.c_proj.weight'].double()
    hc=h-h.mean(1,keepdim=True);w=(z@B.T)@kernel;v=hc@kernel
    rest=hc-v@kernel.T;other=rest.square().mean(1)+p['eps'];rad=w.norm(dim=1)/2;cen=(v-w/2).norm(dim=1)
    lo=(other+(cen-rad).square()/768).sqrt();hi=(other+(cen+rad).square()/768).sqrt()
    functional=(p['gamma'][:,None]*p['projection'])@p['head'].T;basis=kernel.T@functional
    linear=hc@functional;bias=p['beta']@p['projection']@p['head'].T;y=torch.as_tensor(proof['winner']);ix=torch.arange(len(h))
    directions=basis[:,y].T[:,None,:]-basis.T[None]
    dot=torch.einsum('ni,nci->nc',w,directions);loss=(dot+w.norm(dim=1)[:,None]*directions.norm(dim=-1))/2
    bg=bias[y,None]-bias[None];lower=linear[ix,y,None]-linear-loss+bg*torch.where(bg>=0,lo[:,None],hi[:,None])
    error=float((lower-torch.as_tensor(proof['lower'])).abs().max());assert error<1e-8
    mask=~torch.nn.functional.one_hot(y,10).bool();minimum=float(lower[mask].min());assert minimum>error
    assert bool((lower.masked_fill(~mask,torch.inf)>0).all()) and report['certified_no_teacher_change']==len(h)
    save_json(out/'independent_validation.json',dict(status='passed_independent_torch_full_family_bound',raw_sha256=report['raw_sha256'],
        source_sha256=digest(__file__),full_bound_NumPy_torch_error=error,minimum_margin_numerator_lower_bound=minimum,
        certified_source_inputs=len(h),new_images=0,new_releases=0,
        scope='Independent full reconstruction using torch LayerNorm/QuickGELU and orthogonal-complement norm, versus the NumPy derivation. The vector-sphere proof also covers every orthogonal B-output transform in the same kernel; no peer or full native accuracy claim.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');ap.add_argument('--cost',action='store_true');ap.add_argument('--affine',action='store_true');ap.add_argument('--branch-norm',action='store_true');ap.add_argument('--bound',action='store_true');a=ap.parse_args();initialize()
    if a.branch_norm:
        if a.bound:
            verify_branch_proxy_bound(a.verify) if a.verify else branch_proxy_bound(a.protocol)
        else:
            verify_branch_norm(a.verify) if a.verify else branch_norm_check(a.protocol)
    elif a.affine: affine_audit(a.protocol)
    else: (verify_cost if a.cost else verify)(a.verify) if a.verify else (functional_cost if a.cost else run)(a.protocol)
