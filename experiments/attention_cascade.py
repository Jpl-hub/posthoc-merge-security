"""A finite B10 -> attention11 -> B11 publication chart, with exact VJP."""

import torch

from cross_module_distance import extended_parameters
from subspace_probe import make_loader
from task_boundary_audit import raw_jacobians
from task_probe_distance import gelu


def parameters(model,head):
    block=model.model.visual.transformer.resblocks[11]
    p=extended_parameters(model,head)
    p.update({name:value.detach().cpu().double() for name,value in dict(
        D=model.model.visual.transformer.resblocks[10].mlp.c_proj.weight,
        gamma1=block.ln_1.weight,beta1=block.ln_1.bias,
        W=block.attn.in_proj_weight,w=block.attn.in_proj_bias).items()})
    return p|dict(eps1=block.ln_1.eps,heads=block.attn.num_heads)


def norm(x,gamma,beta,epsilon):
    centered=x-x.mean(-1,keepdim=True)
    return centered/(centered.square().mean(-1,keepdim=True)+epsilon).sqrt()*gamma+beta


def suffix(h,p):
    n,t,d=h.shape;heads=p['heads'];width=d//heads
    x=norm(h,p['gamma1'],p['beta1'],p['eps1'])
    wq,wk,wv=p['W'].chunk(3);bq,bk,bv=p['w'].chunk(3)
    query=(x[:,0]@wq.T+bq).reshape(n,heads,width)
    key=(x@wk.T+bk).reshape(n,t,heads,width).transpose(1,2)
    value=(x@wv.T+bv).reshape(n,t,heads,width).transpose(1,2)
    probability=torch.softmax(torch.einsum('nhd,nhtd->nht',query,key)/(width**.5),-1)
    average=torch.einsum('nht,nhtd->nhd',probability,value)
    r=h[:,0]+average.flatten(1)@p['O'].T+p['ob']
    z=gelu(norm(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'])
    out=r+z@p['B'].T+p['b']
    return out,dict(H=h,x=x,q=query,k=key,v=value,prob=probability,average=average,r=r,z=z)


def input_vjp(jr,context,p):
    """One score covector back through CLS attention, all K/V tokens and LN1."""
    h=context['H'];n,t,d=h.shape;heads=p['heads'];width=d//heads;classes=jr.shape[1]
    jo=(jr@p['O']).reshape(n,classes,heads,width)
    dp=context['prob'][:,None]*(torch.einsum('nchd,nhtd->ncht',jo,context['v'])
                                -(jo*context['average'][:,None]).sum(-1)[...,None])
    dv=context['prob'][:,None,:,:,None]*jo[:,:,:,None,:]
    dk=dp[...,None]*context['q'][:,None,:,None,:]/width**.5
    dq=torch.einsum('ncht,nhtd->nchd',dp,context['k'])/width**.5
    def tokens(v):return v.transpose(2,3).reshape(n,classes,t,d)
    wq,wk,wv=p['W'].chunk(3)
    dx=tokens(dv)@wv+tokens(dk)@wk
    query_contribution=torch.zeros_like(dx);query_contribution[:,:,0]=dq.flatten(2)@wq
    dx=(dx+query_contribution)*p['gamma1']
    c=h-h.mean(-1,keepdim=True);sd=(c.square().mean(-1,keepdim=True)+p['eps1']).sqrt()
    result=(dx-dx.mean(-1,keepdim=True)-c[:,None]*(dx*c[:,None]).mean(-1,keepdim=True)/sd[:,None].square())/sd[:,None]
    direct=torch.zeros_like(result);direct[:,:,0]=jr
    return result+direct


@torch.no_grad()
def capture(model,training,indices,vh):
    """Keep H10 and projected B10 inputs; do not cache the full MLP width."""
    blocks=model.model.visual.transformer.resblocks;groups={k:[] for k in ['H','t','h','actual']}
    hooks=[blocks[10].mlp.c_proj.register_forward_pre_hook(lambda _m,args:groups['t'].append((args[0].transpose(0,1).double()@vh.cuda().T).cpu())),
           blocks[11].ln_1.register_forward_pre_hook(lambda _m,args:groups['H'].append(args[0].transpose(0,1).detach().cpu())),
           model.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:groups['h'].append(args[0].detach().cpu()))]
    try:
        model.cuda().eval()
        for images,_ in make_loader(training,indices,64):groups['actual'].append(model(images.cuda()).detach().cpu())
    finally:
        for hook in hooks:hook.remove()
        model.cpu()
    return {k:torch.cat(v).double() if k in ['h','actual'] else torch.cat(v) for k,v in groups.items()}


def select(packed,rows):
    return {**{key:packed[key] for key in ['cache','up_map']},
            **{key:packed[key][rows] for key in ['take','shift','offset','teacher'] if key in packed}}


def compact(packed):
    take=packed['take']
    return {**packed,'cache':{key:value[take] for key,value in packed['cache'].items()},'take':torch.arange(len(take),device=take.device)}


def forward(packed,p,theta,basis,q,factor=1.,want_jacobian=False):
    rank=basis['u'].shape[1];size=rank**2;nd,nb=p['D'].norm(),p['B'].norm()
    cc=theta[:size].reshape(rank,rank);kb=theta[size:].reshape(q.shape[1],p['B'].shape[1])
    cache=packed['cache'];take=packed['take'];h=cache['H'][take].double();inputs=cache['t'][take].double()
    # Task singular probes perturb the CLS B10 input and its original output.
    marker=torch.zeros((h.shape[1],1),device=h.device,dtype=h.dtype);marker[0]=1
    h=h+marker[None]*(packed['shift']@packed['up_map'])[:,None,:]
    inputs=inputs+marker[None]*packed['shift'][:,None,:]
    h=h+factor*nd*(inputs@cc.T)@basis['u'].T
    out,c=suffix(h,p);z=c['z']+packed['offset'];change=factor*nb*(z@kb.T)@q.T
    final=out+cache['rounding'][take]+packed['offset']@p['B'].T+change
    if not want_jacobian:return final,c
    b=p['B']+factor*nb*q@kb
    jh,jr,_,_=raw_jacobians({'h':final,'r':c['r']},p|{'B':b})
    jinputs=input_vjp(jr,c,p)
    first=factor*nd*torch.einsum('nctu,ntv->ncuv',jinputs@basis['u'],inputs)
    second=factor*nb*((jh@q)[:,:,:,None]*z[:,None,None,:])
    return final,torch.cat([first.flatten(2),second.flatten(2)],2)
