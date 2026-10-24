"""One fixed finite two-state nonlinear functional compensation check; no images."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/merge_security_20261009')
sys.path.insert(0,str(ROOT))
from _impl.models import initialize,load_checkpoint
from _impl.io import digest,save_json
from _impl.metrics import logical_parameter_distance

OUT=ROOT/'outputs/signed_function_20261024'


def main():
    initialize();start=time.monotonic()
    plan=json.loads((ROOT/'results/data_free_compensation/joint.json').read_text())['signed_function_compensation']
    assert digest(__file__)==plan['source_sha256']
    OUT.mkdir(exist_ok=False)
    expert,base=load_checkpoint('MNIST/finetuned.pt'),load_checkpoint('zeroshot.pt')
    es,bs=expert.state_dict(),base.state_dict();p='model.visual.transformer.resblocks.11.'
    ref=ROOT/'outputs/feature_noise/artifacts/original_AGA.pt'
    assert digest(ref)=='07195e779ea2a8203aaf2ec7bccc651d00460c3a966810bc04342af5e5088607'
    recovered=torch.load(ref,map_location='cpu',weights_only=False)
    path=ROOT/'outputs/phase2_task_subspace_comparison_20261020/basis.pt'
    assert digest(path)=='99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344'
    probe=torch.load(path,map_location='cpu',weights_only=False)['73']
    assert probe['source_key']==p+'mlp.c_proj.weight'
    A,a=[es[p+k].double().cuda() for k in ['mlp.c_fc.weight','mlp.c_fc.bias']]
    gamma,beta=[es[p+k].double().cuda() for k in ['ln_2.weight','ln_2.bias']]
    A0,a0=[bs[p+k].double().cuda() for k in ['mlp.c_fc.weight','mlp.c_fc.bias']]
    g0,beta0=[bs[p+k].double().cuda() for k in ['ln_2.weight','ln_2.bias']]
    B,b=[es[p+k].double().cuda() for k in ['mlp.c_proj.weight','mlp.c_proj.bias']]
    B0,b0=[bs[p+k].double().cuda() for k in ['mlp.c_proj.weight','mlp.c_proj.bias']]
    old=es[p+'mlp.c_fc.weight'].numpy();rows={hashlib.sha256(x.tobytes()).hexdigest():i for i,x in enumerate(old)}
    assert len(rows)==3072
    order=np.array([rows[hashlib.sha256(x.tobytes()).hexdigest()] for x in recovered[p+'mlp.c_fc.weight'].numpy()])
    assert len(set(order))==3072
    for k,axis in [('mlp.c_fc.weight',0),('mlp.c_fc.bias',0),('mlp.c_proj.weight',1)]:
        assert torch.equal(recovered[p+k],es[p+k].index_select(axis,torch.as_tensor(order)))
    assert torch.equal(recovered[p+'mlp.c_proj.bias'],es[p+'mlp.c_proj.bias'])
    inv=torch.as_tensor(np.argsort(order),device='cuda')
    Am=((recovered[p+'mlp.c_fc.weight'].double().cuda()+A0)/2)[inv]
    am=((recovered[p+'mlp.c_fc.bias'].double().cuda()+a0)/2)[inv]
    Bm=((recovered[p+'mlp.c_proj.weight'].double().cuda()+B0)/2)[:,inv]
    bm=(b+b0)/2
    v=probe['u'][:,0].cuda();V=probe['v'].cuda()
    delta=(A-A0)*gamma;delta-=delta.mean(1,keepdim=True)
    _,singular,vh=torch.linalg.svd(delta,full_matrices=False);U=vh[:32].T
    native_M=A*gamma;native_mu=A@beta+a
    middle_M=Am*((gamma+g0)/2);middle_mu=Am@((beta+beta0)/2)+am
    base_M=(A0*g0)[inv];base_mu=(A0@beta0+a0)[inv]
    ct=Bm.T@v;cz=B0[:,inv].T@v;constant=(bm-b0)@v
    def inputs(generator,kind,n):
        dim=32 if kind=='task' else 768
        z=torch.randn(n,dim,generator=generator,dtype=torch.double,device='cuda')
        if kind=='full':z-=z.mean(1,keepdim=True)
        r=32 if kind=='task' else 767
        z=z/z.norm(dim=1,keepdim=True)*(768**.5)*torch.rand(n,1,generator=generator,dtype=torch.double,device='cuda').pow(1/r)
        return z@U.T if kind=='task' else z
    def features(x,M,mu):
        z=x@M.T+mu;return z*torch.sigmoid(1.702*z)
    n=8192;batch=512
    sums=torch.zeros(3072,dtype=torch.double,device='cuda');Kn=torch.zeros(3072,3072,dtype=torch.double,device='cuda')
    Kraw=torch.zeros_like(Kn);sm=torch.zeros_like(sums);cross=torch.zeros_like(sums);mean_tau=0.;energy_tau=0.
    gen=torch.Generator(device='cuda').manual_seed(2026102427)
    for kind in ['full','task']:
        for _ in range(n//batch):
            x=inputs(gen,kind,batch);h=features(x,native_M,native_mu)
            sums+=h.sum(0)/(2*n);Kn+=h.T@h/(2*n)
            if kind=='task':
                hm=features(x,middle_M,middle_mu);hz=features(x,base_M,base_mu)
                tau=hm@ct-hz@cz+constant
                sm+=hm.sum(0)/n;Kraw+=hm.T@hm/n;cross+=hm.T@tau/n
                mean_tau+=float(tau.sum()/n);energy_tau+=float(tau.square().sum()/n)
    Cn=Kn-torch.outer(sums,sums)
    Km=Kraw-torch.outer(sm,sums)-torch.outer(sums,sm)+torch.outer(sums,sums)
    c=cross-sums*mean_tau
    ridge=1e-6*float(Cn.trace()/3072);q=.5
    H=1000*Cn+q*q*Km+ridge*torch.eye(3072,dtype=torch.double,device='cuda')
    H=(H+H.T)/2;rhs=-2*q*c
    solved=torch.linalg.solve(H,torch.cat([rhs[:,None],V],dim=1))
    z,T=solved[:,0],solved[:,1:]
    w=z-T@torch.linalg.solve(V.T@T,V.T@z)
    g=H@w-rhs;projected=g-V@(V.T@g)
    residual=float(projected.norm()/rhs.norm());task_action=float((V.T@w).norm()/w.norm())
    assert residual<1e-6 and task_action<1e-8
    db=-v*(sums@w);dB=v[:,None]*w[None]
    ratioB=float(dB.norm()/B.norm());ratiob=float(db.norm()/b.norm())
    proposed={p+'mlp.c_proj.weight':(B+dB).float().cpu(),p+'mlp.c_proj.bias':(b+db).float().cpu()}
    held={};gen=torch.Generator(device='cuda').manual_seed(2026102428)
    for kind in ['full','task']:
        err=normal=tau_energy=signed=reverse=0.;flipped=0;count=4096
        for _ in range(count//batch):
            x=inputs(gen,kind,batch);h=features(x,native_M,native_mu);s=(h-sums)@w
            output=h@B.T+b;err+=float(s.square().sum()/count);normal+=float(output.square().sum()/count)
            if kind=='task':
                hm=features(x,middle_M,middle_mu);hz=features(x,base_M,base_mu)
                tau=hm@ct-hz@cz+constant;after=tau+q*((hm-sums)@w)
                tau_energy+=float(tau.square().sum()/count);signed+=float((tau*after).sum()/count)
                reverse+=float((tau+after).square().sum()/count);flipped+=int((tau*after<0).sum())
        held[kind]=dict(native_delta_rms=err**.5,native_output_rms=normal**.5,native_relative_rms=(err/normal)**.5)
        if kind=='task':held[kind].update(task_increment_rms=tau_energy**.5,signed_task_response_ratio=signed/tau_energy,relative_reversal_error=(reverse/tau_energy)**.5,task_sign_reversal_fraction=flipped/count)
    feasible=ratioB<=4 and ratiob<=4 and max(x['native_relative_rms'] for x in held.values())<=.01 and held['task']['signed_task_response_ratio']<0
    torch.save(dict(w=w.cpu(),v=v.cpu(),native_mean=sums.cpu(),task_U=U.cpu(),source_native_covariance=Cn.cpu(),source_middle_second_moment=Km.cpu(),source_task_cross=c.cpu(),source_task_energy=energy_tau),OUT/'statistics.pt')
    release=None
    if feasible:
        torch.save(proposed,OUT/'release.pt');release=digest(OUT/'release.pt')
    result=dict(status='source_feasible_release_frozen' if feasible else 'closed_fixed_functional_subproblem_failed',plan=plan,source=Path(__file__).read_text(),source_sha256=digest(__file__),heldout=held,source_projected_normal_equation_error=residual,task_B_action_error=task_action,source_ridge=ridge,B_relative_distance=ratioB,b_relative_distance=ratiob,logical_D=logical_parameter_distance(proposed,es),release_sha256=release,statistics_sha256=digest(OUT/'statistics.pt'),raw_AGA_reference_sha256=digest(ref),all3072neurons=True,task_basis_energy_fraction=float((singular[:32]**2).sum()/(singular**2).sum()),real_images=0,classification_heads=0,new_AGA_calls=0,expert_training_updates=0,source_latent_samples=16384,heldout_latent_samples=8192,linear_solves=2,optimizer_steps=0,wall_seconds=time.monotonic()-start)
    save_json(OUT/'source.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source','plan']}),flush=True)


if __name__=='__main__':main()
