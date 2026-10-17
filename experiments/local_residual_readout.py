"""One source-only feasibility check of a localized residual/readout compensation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bias_canonicalization import inputs
from clip_benchmark import digest
from postactivation_subspace import metric
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import tail
from verify_levelset_audits import raw_scores


@torch.no_grad()
def run(protocol):
    start=time.monotonic();cfg,_,checked=inputs(protocol);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    ref=cfg['source_reference'];assert digest(ref['path'])==ref['sha256'];raw=torch.load(ref['path'],map_location='cpu',weights_only=False)
    p=raw['parameters'];env=raw['proxy_parameters'];h,z,hp,zp=[raw[k].double() for k in ['h','z','hp','zp']]
    anchor=checked('zeroshot.pt');b0=anchor.model.visual.transformer.resblocks[11].mlp.c_proj.weight.double()
    p0=anchor.model.visual.proj.double();beta0=anchor.model.visual.ln_post.bias.double()
    # Reuse response-ranked task modes, never choose a plane from partner/test responses.
    v=raw['v'];u=(p['B']-b0)@v[:2].T;u-=u.mean(0);u=torch.linalg.qr(u,mode='reduced').Q
    np.testing.assert_allclose(u.T@u,torch.eye(2),rtol=0,atol=1e-10);assert float(u.sum(0).abs().max())<1e-10
    n=cfg['fit_n'];c,_,rank,sv=np.linalg.lstsq(z[:n].numpy(),(h[:n]@u).numpy(),rcond=cfg['regression_rcond']);c=torch.from_numpy(c)
    gamma,beta,proj=p['gamma'],p['beta'],p['projection'];assert gamma.abs().min()>1e-8
    j=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64);up=u/gamma[:,None];q=u.T@(gamma[:,None]*proj);t=(beta/gamma)@u
    sine=dict(B=-u@j@c.T,projection=-up@j@q,beta=(t@j@u.T)*gamma)
    cosine=dict(B=-u@c.T,projection=-up@q,beta=-(t@u.T)*gamma)
    a=sum(float((sine[k]/p[k].norm()).square().sum()) for k in sine)
    b=sum(float((sine[k]*cosine[k]).sum()/p[k].square().sum()) for k in sine)
    cc=sum(float((cosine[k]/p[k].norm()).square().sum()) for k in sine)
    before=scores(h,p);proxy_before=scores(hp,env);y=before[:n].argmax(1)
    context=torch.cat([torch.where(y==i)[0][:2] for i in range(before.shape[1])]);assert len(context)==20
    rms=(z[:n]@v.T).square().mean(0).sqrt();shift=(.01*torch.tensor([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1)
    zprobe=(z[context,None]+shift[None]).flatten(0,1);hprobe=(h[context,None]+shift[None]@p['B'].T).flatten(0,1);probe_before=scores(hprobe,p)
    records=[];points={};inside=v.T@(v@c)
    for radius in cfg['parameter_budgets']:
        budget=radius*(1-cfg['FP32_distance_reserve']);roots=np.roots([4*cc-budget**2,8*b,4*a-2*budget**2,0.,-budget**2])
        candidates=[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<=1+1e-9]
        theta=2*np.arctan(min(candidates)) if candidates else np.pi/2
        sn,cn=np.sin(theta),1-np.cos(theta);delta={k:sn*sine[k]+cn*cosine[k] for k in sine}
        ideal=p|{k:p[k]+delta[k] for k in delta};actual=p|{k:ideal[k].float().double() for k in delta};delta={k:actual[k]-p[k] for k in delta}
        distance=sum(float((delta[k]/p[k].norm()).square().sum()) for k in delta)**.5;assert distance<=radius
        rotation=torch.eye(h.shape[1],dtype=torch.float64)+u@(sn*j-cn*torch.eye(2,dtype=torch.float64))@u.T
        ideal_feature_error=float((tail(h@rotation,ideal)[1]-tail(h,p)[1]).abs().max());assert ideal_feature_error<1e-9
        hn=h+z@delta['B'].T;src=scores(hn,actual);probe=scores(hprobe+zprobe@delta['B'].T,actual)
        # Reproduce the fixed public interpolation's actual FP32 parameter arithmetic.
        changed={k:(base.float()+.5*((actual[k].float()-base.float())+torch.zeros_like(base.float()))).double() for k,base in [('B',b0),('projection',p0),('beta',beta0)]}
        new_env=env|changed;he=hp+zp@(changed['B']-env['B']).T;proxy=scores(he,new_env)
        errors=[]
        for hh,pp,expected in [(hn,actual,src),(he,new_env,proxy)]:
            pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in pp.items()};errors.append(float(abs(raw_scores(hh.numpy(),pn)-expected.numpy()).max()))
        assert max(errors)<1e-8
        ideal_hidden=h+z@(sn*sine['B']+cn*cosine['B']).T;residual=(z@c-h@u)@(sn*j-cn*torch.eye(2,dtype=torch.float64))@u.T
        identity=float((ideal_hidden-h@rotation-residual).abs().max());assert identity<1e-9
        toy=(torch.eye(len(h[0]))+.5*(rotation-torch.eye(len(h[0]))))@(torch.eye(len(h[0]))+.5*(rotation.T-torch.eye(len(h[0]))))
        toy_error=float((toy-(torch.eye(len(h[0]))-.5*cn*(u@u.T))).abs().max());assert toy_error<1e-10
        row=dict(parameter_budget=radius,actual_distance=distance,theta=float(theta),plane_midpoint_factor=float(np.cos(theta/2)**2),
            per_tensor_distance={k:float(delta[k].norm()/p[k].norm()) for k in delta},source=metric(before,src),source_fit=metric(before[:n],src[:n]),source_check=metric(before[n:],src[n:]),
            probes=metric(probe_before,probe),proxy=metric(proxy_before,proxy),ideal_normalized_feature_error=ideal_feature_error,
            finite_hidden_identity_error=identity,conditional_two_factor_identity_error=toy_error,NumPy_score_error=max(errors),
            B_inside_relative_distance=float(((delta['B']@v.T)@v).norm()/p['B'].norm()),B_outside_relative_distance=float((delta['B']-(delta['B']@v.T)@v).norm()/p['B'].norm()))
        row['source_feasible']=max(row['source']['worst_task_drift'],row['probes']['worst_task_drift'])<=cfg['task_budget'];records.append(row);points[str(radius)]=dict(delta=delta,source=src,proxy=proxy,probes=probe)
        print(radius,'distance',distance,'angle',theta,'source',row['source']['worst_task_drift'],'probe',row['probes']['worst_task_drift'],'proxy changed',row['proxy']['changed'],flush=True)
    torch.save(dict(source_reference=ref,u=u,c=c,mode_indices=raw['mode_indices'][:2],points=points),out/'diagnostic.pt');save_json(out/'results.json',records);save_json(out/'protocol.json',cfg)
    save_json(out/'execution.json',dict(completed=True,CPU_seconds=time.monotonic()-start,GPU_compute=False,source_n=len(h),fit_n=n,probe_n=len(hprobe),regression_rank=int(rank),regression_retained_condition=float(sv[0]/sv[int(rank)-1]),predictor_inside_norm=float(inside.norm()),predictor_outside_norm=float((c-inside).norm()),source_raw_sha256=ref['sha256'],raw_sha256=digest(out/'diagnostic.pt'),source_sha256=digest(__file__),new_images=0,gradients=0,expert_updates=0,new_release_count=0,novelty_established=False,unknown_partner_guarantee=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/local_residual_readout.json');run(p.parse_args().protocol)
