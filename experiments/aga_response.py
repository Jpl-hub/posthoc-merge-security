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


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');a=ap.parse_args();initialize()
    verify(a.verify) if a.verify else run(a.protocol)
