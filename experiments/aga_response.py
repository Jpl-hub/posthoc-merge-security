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
from local_residual_transport import cpu, forward
from mechanisms import load_author
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, initialize
from verify_levelset_audits import convert, raw_scores


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


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');ap.add_argument('--cost',action='store_true');a=ap.parse_args();initialize()
    (verify_cost if a.cost else verify)(a.verify) if a.verify else (functional_cost if a.cost else run)(a.protocol)
