"""One source-only feasibility check of a localized residual/readout compensation."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from bias_canonicalization import inputs
from clip_benchmark import digest, evaluate
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from classification_completion import readout
from feature_noise import capture
from postactivation_subspace import metric
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
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
    v=raw['v'];reflection=cfg.get('mode')=='class_readout_reflection'
    if reflection:u=readout(p)[0]['Q']
    else:u=(p['B']-b0)@v[:2].T;u-=u.mean(0);u=torch.linalg.qr(u,mode='reduced').Q
    width=u.shape[1];ident=torch.eye(width,dtype=torch.float64)
    np.testing.assert_allclose(u.T@u,ident,rtol=0,atol=1e-10);assert float(u.sum(0).abs().max())<1e-10
    n=cfg['fit_n'];c,_,rank,sv=np.linalg.lstsq(z[:n].numpy(),(h[:n]@u).numpy(),rcond=cfg['regression_rcond']);c=torch.from_numpy(c)
    gamma,beta,proj=p['gamma'],p['beta'],p['projection'];assert gamma.abs().min()>1e-8
    j=torch.zeros_like(ident) if reflection else torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64);up=u/gamma[:,None];q=u.T@(gamma[:,None]*proj);t=(beta/gamma)@u
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
        theta=np.pi if reflection else (2*np.arctan(min(candidates)) if candidates else np.pi/2)
        sn,cn=np.sin(theta),1-np.cos(theta);delta={k:sn*sine[k]+cn*cosine[k] for k in sine}
        ideal=p|{k:p[k]+delta[k] for k in delta};actual=p|{k:ideal[k].float().double() for k in delta};delta={k:actual[k]-p[k] for k in delta}
        distance=sum(float((delta[k]/p[k].norm()).square().sum()) for k in delta)**.5
        if distance>radius:
            save_json(out/'results.json',[dict(status='failed_parameter_budget',parameter_budget=radius,actual_distance=distance,mode=cfg.get('mode'),rank=width,no_clipping_or_interpolation=True)])
            save_json(out/'execution.json',dict(completed=True,closed=True,reason='parameter_budget_exceeded',source_sha256=digest(__file__),CPU_seconds=time.monotonic()-start,gradients=0,new_release_count=0));save_json(out/'protocol.json',cfg);return
        rotation=torch.eye(h.shape[1],dtype=torch.float64)+u@(sn*j-cn*ident)@u.T
        ideal_feature_error=float((tail(h@rotation,ideal)[1]-tail(h,p)[1]).abs().max());assert ideal_feature_error<1e-9
        hn=h+z@delta['B'].T;src=scores(hn,actual);probe=scores(hprobe+zprobe@delta['B'].T,actual)
        # Reproduce the fixed public interpolation's actual FP32 parameter arithmetic.
        changed={k:(base.float()+.5*((actual[k].float()-base.float())+torch.zeros_like(base.float()))).double() for k,base in [('B',b0),('projection',p0),('beta',beta0)]}
        new_env=env|changed;he=hp+zp@(changed['B']-env['B']).T;proxy=scores(he,new_env)
        errors=[]
        for hh,pp,expected in [(hn,actual,src),(he,new_env,proxy)]:
            pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in pp.items()};errors.append(float(abs(raw_scores(hh.numpy(),pn)-expected.numpy()).max()))
        assert max(errors)<1e-8
        ideal_hidden=h+z@(sn*sine['B']+cn*cosine['B']).T;residual=(z@c-h@u)@(sn*j-cn*ident)@u.T
        identity=float((ideal_hidden-h@rotation-residual).abs().max());assert identity<1e-9
        toy=(torch.eye(len(h[0]))+.5*(rotation-torch.eye(len(h[0]))))@(torch.eye(len(h[0]))+.5*(rotation.T-torch.eye(len(h[0]))))
        toy_error=float((toy-(torch.eye(len(h[0]))-.5*cn*(u@u.T))).abs().max());assert toy_error<1e-10
        row=dict(parameter_budget=radius,actual_distance=distance,theta=float(theta),plane_midpoint_factor=float(np.cos(theta/2)**2),
            per_tensor_distance={k:float(delta[k].norm()/p[k].norm()) for k in delta},source=metric(before,src),source_fit=metric(before[:n],src[:n]),source_check=metric(before[n:],src[n:]),
            probes=metric(probe_before,probe),proxy=metric(proxy_before,proxy),ideal_normalized_feature_error=ideal_feature_error,
            finite_hidden_identity_error=identity,conditional_two_factor_identity_error=toy_error,NumPy_score_error=max(errors),
            B_inside_relative_distance=float(((delta['B']@v.T)@v).norm()/p['B'].norm()),B_outside_relative_distance=float((delta['B']-(delta['B']@v.T)@v).norm()/p['B'].norm()))
        row['source_feasible']=max(row['source']['worst_task_drift'],row['probes']['worst_task_drift'])<=cfg['task_budget'];records.append(row);points[str(radius)]=dict(delta=delta,source=src,proxy=proxy,probes=probe)
        if reflection:
            def certificate(params):
                image=params['projection']@(params['head']-params['head'].mean(0)).T;m=params['gamma'][:,None]*image;m-=m.mean(0);bb=params['beta']@image
                bound=bb[:,None]-bb[None,:]-h.shape[1]**.5*(m[:,:,None]-m[:,None,:]).norm(dim=0);bound.fill_diagonal_(torch.inf);best=bound.min(1).values;k=int(best.argmax())
                return dict(best_constant_class=k,worst_class_margin_lower_bound=float(best[k]),prefix_independent_collapse_certified=bool(best[k]>0),M_norm=float(m.norm()))
            middle=p|{k:.5*(p[k]+ideal[k]) for k in delta};row['ideal_expert_midpoint']=certificate(middle)
            assert row['ideal_expert_midpoint']['M_norm']<1e-8
            row['public_proxy_readout_certificate']=certificate(new_env);row['actual_budget_comparable_to_0_5']=distance<=.5
        print(radius,'distance',distance,'angle',theta,'source',row['source']['worst_task_drift'],'probe',row['probes']['worst_task_drift'],'proxy changed',row['proxy']['changed'],flush=True)
    torch.save(dict(source_reference=ref,u=u,c=c,mode_indices=raw['mode_indices'][:2],points=points),out/'diagnostic.pt');save_json(out/'results.json',records);save_json(out/'protocol.json',cfg)
    save_json(out/'execution.json',dict(completed=True,CPU_seconds=time.monotonic()-start,GPU_compute=False,source_n=len(h),fit_n=n,probe_n=len(hprobe),regression_rank=int(rank),regression_retained_condition=float(sv[0]/sv[int(rank)-1]),predictor_inside_norm=float(inside.norm()),predictor_outside_norm=float((c-inside).norm()),source_raw_sha256=ref['sha256'],raw_sha256=digest(out/'diagnostic.pt'),source_sha256=digest(__file__),new_images=0,gradients=0,expert_updates=0,new_release_count=0,novelty_established=False,unknown_partner_guarantee=False))


@torch.no_grad()
def confirm(protocol):
    start=time.monotonic();cfg,_,checked=inputs(protocol);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for ref in cfg['references'].values():assert digest(ref['path'])==ref['sha256']
    source=torch.load(cfg['references']['source_point']['path'],map_location='cpu',weights_only=False)
    parent=torch.load(source['source_reference']['path'],map_location='cpu',weights_only=False);assert digest(source['source_reference']['path'])==source['source_reference']['sha256']
    p=parent['parameters'];point=source['points'][cfg.get('point_key','0.5')];delta=point['delta'];q=p|{k:(p[k]+v).float().double() for k,v in delta.items()}
    keys={'B':'model.visual.transformer.resblocks.11.mlp.c_proj.weight','projection':'model.visual.proj','beta':'model.visual.ln_post.bias'}
    own,head=checked('MNIST/finetuned.pt'),checked('head_MNIST.pt');released=copy.deepcopy(own);edits={}
    for name,key in keys.items():
        assert torch.equal(own.state_dict()[key].double(),p[name]);edits[key]=q[name].float();released.state_dict()[key].copy_(edits[key])
    distance=sum(float(((q[k]-p[k])/p[k].norm()).square().sum()) for k in delta)**.5;assert distance<=cfg['parameter_budget']
    train=dataset('MNIST','data',own.val_preprocess,True)
    def pair(ids):
        a=[];b=[];evaluate(own,head,make_loader(train,ids,64),'cuda',logit_sink=a.append);evaluate(released,head,make_loader(train,ids,64),'cuda',logit_sink=b.append)
        return torch.cat(a).double(),torch.cat(b).double()
    a,b=pair(parent['indices']);full=metric(a,b);error=float((b-point['source']).abs().max());assert error<=cfg['score_tolerance']
    save_json(out/'full_source_validation.json',dict(**full,score_error=error,parameter_distance=distance))
    if full['worst_task_drift']>cfg['task_budget']:save_json(out/'confirmation_validation.json',dict(status='failed_full_source',full_source=full));return
    file=out/'frozen_release.pt';torch.save(edits,file);save_json(out/'pre_confirmation_freeze.json',dict(release_sha256=digest(file),parameter_distance=distance,source_point_sha256=cfg['references']['source_point']['sha256'],confirmation_drawn=False,protocol=cfg,source_sha256=digest(__file__)))
    audit=np.random.default_rng(20261016).choice(len(train),512,replace=False).tolist();split=unused(train,set(parent['indices'])|set(audit)|{41538,27330,31374},cfg);save_json(out/'confirmation_split.json',split)
    aa,bb=pair(split['indices']);zz,hh=capture(own,make_loader(train,split['indices'],64));before=scores(hh,p);after=scores(hh+zz@delta['B'].T,q)
    control=max(float((aa-before).abs().max()),float((bb-after).abs().max()));assert control<=cfg['score_tolerance']
    pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in q.items()};numpy_score=raw_scores((hh+zz@delta['B'].T).numpy(),pn);np_error=float(abs(numpy_score-after.numpy()).max());assert np_error<1e-8
    yy=before.argmax(1);context=torch.cat([torch.where(yy==i)[0][:2] for i in range(before.shape[1])]);assert len(context)==20
    v=parent['v'];rms=(parent['z'][:512]@v.T).square().mean(0).sqrt();shift=(.01*torch.tensor([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1)
    zp=(zz[context,None]+shift[None]).flatten(0,1);hp=(hh[context,None]+shift[None]@p['B'].T).flatten(0,1);probe=metric(scores(hp,p),scores(hp+zp@delta['B'].T,q))
    natural=metric(aa,bb);passed=max(natural['worst_task_drift'],probe['worst_task_drift'])<=cfg['task_budget'];labels=torch.tensor([int(train[i][1]) for i in split['indices']])
    torch.save(dict(before=aa,after=bb,h=hh,z=zz,labels=labels,indices=split['indices']),out/'confirmation.pt')
    report=dict(status='passed' if passed else 'failed',confirmation_passed=bool(passed),parameter_distance=distance,natural=natural,probes=probe,full_source=full,n=len(labels),original_correct=int((aa.argmax(1)==labels).sum()),released_correct=int((bb.argmax(1)==labels).sum()),full_model_score_error=control,NumPy_score_error=np_error,release_sha256=digest(file),confirmation_sha256=digest(out/'confirmation.pt'),split_sha256=digest(out/'confirmation_split.json'),source_sha256=digest(__file__),new_images=len(labels),full_model_forward_instances=2*len(parent['indices'])+3*len(labels),gradients=0,expert_updates=0,source_point_repaired=False,partner_used=False,test_used=False,wall_seconds=time.monotonic()-start,protocol=cfg)
    save_json(out/'confirmation_validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/local_residual_readout.json');p.add_argument('--confirm',action='store_true');a=p.parse_args();(confirm if a.confirm else run)(a.protocol)
