"""One closed-form B-output functional-kernel construction; fixed SVD ablations."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from bias_canonicalization import inputs
from classification_completion import readout
from clip_benchmark import digest, evaluate, merge
from covariance_noise import dataset
from feature_noise import capture
from input_null_transport import class_game
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import parameters
from verify_levelset_audits import raw_scores


def metric(before, after):
    y=before.argmax(1); rows=torch.arange(len(y),device=before.device)
    gaps=before[rows,y,None]-before; mask=torch.ones_like(gaps,dtype=torch.bool); mask[rows,y]=False
    assert (gaps[mask]>0).all()
    delta=after-before; ratio=(delta[rows,y,None]-delta)/gaps.masked_fill(~mask,1.)
    ratio[rows,y]=0; changed=after.argmax(1)!=y
    return dict(n=len(y),worst_task_drift=float(ratio.abs().max()),maximum_signed_harm=float((-ratio).max()),
        changed=int(changed.sum()),per_class=[dict(class_id=c,n=int((y==c).sum()),changed=int((changed&(y==c)).sum()),
            maximum_signed_harm=float((-ratio[y==c]).max()) if (y==c).any() else None) for c in range(before.shape[1])])


@torch.no_grad()
def run(protocol):
    start=time.monotonic(); cfg,_,checked=inputs(protocol); out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=False)
    assert digest(cfg['construction_split']['path'])==cfg['construction_split']['sha256']
    split=json.loads(Path(cfg['construction_split']['path']).read_text()); ids=split['indices']; assert len(ids)==1024
    save_json(out/'protocol.json',cfg);save_json(out/'construction_split.json',split)
    anchor,expert,head=checked('zeroshot.pt'),checked('MNIST/finetuned.pt'),checked('head_MNIST.pt')
    proxy=merge(anchor,expert,anchor,.5); p=on_device(parameters(expert,head)); pe=on_device(parameters(proxy,head)); p0=on_device(parameters(anchor,head))
    train=dataset('MNIST','data',expert.val_preprocess,True); loader=make_loader(train,ids,64)
    z,h=capture(expert,loader); zp,hp=capture(proxy,loader); z,h,zp,hp=[x.cuda() for x in [z,h,zp,hp]]
    before=scores(h,p); bp=scores(hp,pe); n=cfg['fit_n']; y=before[:n].argmax(1); rows=torch.arange(n,device='cuda')
    # SVD modes are ranked by actual normalized task-margin response, not singular value alone.
    ub,sb,vb=torch.linalg.svd(p['B']-p0['B'],full_matrices=False)
    with torch.enable_grad():
        hh=h[:n].detach().requires_grad_(True); ss=scores(hh,p)
        jac=torch.stack([torch.autograd.grad(ss[:,c].sum(),hh,retain_graph=c+1<ss.shape[1])[0] for c in range(ss.shape[1])],1)
    gaps=before[:n][rows,y,None]-before[:n]; mask=torch.ones_like(gaps,dtype=torch.bool);mask[rows,y]=False
    response=((jac[rows,y,None]-jac)@ub)*(z[:n]@vb.T)[:,None,:]*sb
    response/=gaps.masked_fill(~mask,1.)[:,:,None];response[rows,y]=0
    counts=torch.bincount(y,minlength=ss.shape[1]);assert (counts>0).all()
    importance=(response.square().sum(1)/counts[y,None]).sum(0)
    chosen=importance.argsort(descending=True,stable=True)[:cfg['probe_rank']]; v=vb[chosen]
    source,_=readout(p); environment,_=readout(pe)
    qe=source['Q']; leakage=environment['M']-qe@(qe.T@environment['M'])
    u,s,_=torch.linalg.svd(leakage,full_matrices=False);u=u[:,s>s[0]*1e-10]
    assert u.shape[1]>0 and float((source['M'].T@u).abs().max())<1e-8
    yp=bp[:n].argmax(1); other=bp[:n].clone();other[rows,yp]=-torch.inf;j=other.argmax(1)
    sd=lambda x,q:((x-x.mean(1,keepdim=True)).square().mean(1)+q['eps']).sqrt()
    numerator=hp[:n]@environment['M']+sd(hp[:n],pe)[:,None]*environment['b']
    gap=numerator[rows,yp]-numerator[rows,j];assert (gap>0).all()
    left=.5*p['B'].norm()*((environment['M'][:,j]-environment['M'][:,yp]).T@u)/gap[:,None]
    assert all((yp==c).any() for c in range(bp.shape[1]))
    class_rows=torch.stack([torch.einsum('no,ni->oi',left[yp==c],zp[:n][yp==c])/int((yp==c).sum()) for c in range(bp.shape[1])])
    # Fixed source contexts and finite task-SVD probes; no new image or parameter search.
    contexts=torch.cat([torch.where(y==c)[0][:cfg['probe_contexts_per_class']] for c in range(bp.shape[1])])
    rms=(z[:n]@v.T).square().mean(0).sqrt(); shift=(cfg['probe_step']*torch.tensor([-1.,1.],device='cuda')[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1)
    zprobe=(z[contexts,None]+shift[None]).flatten(0,1)
    hprobe=(h[contexts,None]+shift[None]@p['B'].T).flatten(0,1); teacher_probe=scores(hprobe,p)
    pn={k:(x.cpu().numpy() if torch.is_tensor(x) else x) for k,x in p.items()}; en={k:(x.cpu().numpy() if torch.is_tensor(x) else x) for k,x in pe.items()}
    records=[];points={};control=0.;directions={}
    for family in ['Inside','Outside','Joint']:
        g=class_rows.clone();projected=(g@v.T)@v
        if family=='Inside':g=projected
        if family=='Outside':g-=projected
        pi,direction,game=class_game(g.flatten(1).cpu().numpy()); c=torch.tensor(direction.reshape(g.shape[1:]),device='cuda')
        unit=p['B'].norm()*u@c; directions[family]=dict(C=c.cpu(),game=game,dual_weights=pi)
        for radius in cfg['parameter_budgets']:
            actual=(p['B']+radius*(1-cfg['FP32_distance_reserve'])*unit).float().double()-p['B']
            distance=float(actual.norm()/p['B'].norm()); assert distance<=radius
            source_after=scores(h+z@actual.T,p); proxy_after=scores(hp+.5*zp@actual.T,pe)
            probe_after=scores(hprobe+zprobe@actual.T,p)
            report=dict(family=family,parameter_budget=radius,actual_distance=distance,
                task_input_parallel_distance=float((actual@v.T).norm()/p['B'].norm()),
                task_input_complement_distance=float((actual-(actual@v.T)@v).norm()/p['B'].norm()),
                source=metric(before,source_after),probes=metric(teacher_probe,probe_after),proxy=metric(bp,proxy_after),linear_class_game=game)
            # Independent NumPy readout includes LN and final normalization, for every natural source input.
            ns=raw_scores((h+z@actual.T).cpu().numpy(),pn);ne=raw_scores((hp+.5*zp@actual.T).cpu().numpy(),en)
            numpy_error=max(float(abs(ns-source_after.cpu().numpy()).max()),float(abs(ne-proxy_after.cpu().numpy()).max()))
            assert numpy_error<1e-8;report['NumPy_score_error']=numpy_error
            q=hp+.5*zp@actual.T; pieces=.5*zp@actual.T@environment['M']+(sd(q,pe)-sd(hp,pe))[:,None]*environment['b']
            exact=q@environment['M']+sd(q,pe)[:,None]*environment['b']-(hp@environment['M']+sd(hp,pe)[:,None]*environment['b'])
            report['finite_numerator_identity_error']=float((pieces-exact).abs().max());assert report['finite_numerator_identity_error']<1e-8
            report['source_feasible']=max(report['source']['worst_task_drift'],report['probes']['worst_task_drift'])<=cfg['task_budget']
            key=family+'_'+str(radius);points[key]=dict(delta=actual.cpu(),source=source_after.cpu(),proxy=proxy_after.cpu(),probes=probe_after.cpu())
            if family=='Joint' and report['source_feasible']:
                # Full original-model validation precedes freezing a point for independent confirmation.
                released=copy.deepcopy(expert);released.model.visual.transformer.resblocks[11].mlp.c_proj.weight.copy_((p['B']+actual).cpu().float())
                actual_before=[];actual_after=[]
                evaluate(expert,head,make_loader(train,ids,64),'cuda',logit_sink=actual_before.append)
                evaluate(released,head,make_loader(train,ids,64),'cuda',logit_sink=actual_after.append)
                actual_before,actual_after=torch.cat(actual_before).double(),torch.cat(actual_after).double()
                error=max(float((actual_before-before.cpu()).abs().max()),float((actual_after-source_after.cpu()).abs().max()))
                assert error<=cfg['score_tolerance'];report['full_model_source']=metric(actual_before,actual_after);report['full_model_score_error']=error
                report['source_feasible']=report['full_model_source']['worst_task_drift']<=cfg['task_budget']
                if report['source_feasible']:
                    file=out/(key+'_frozen.pt');torch.save({'model.visual.transformer.resblocks.11.mlp.c_proj.weight':released.model.visual.transformer.resblocks[11].mlp.c_proj.weight.cpu()},file)
                    report['frozen_release']=dict(path=str(file),sha256=digest(file),independent_confirmation_pending=True)
            records.append(report);save_json(out/'results.json',records);print(family,radius,'source',report['source']['worst_task_drift'],'probe',report['probes']['worst_task_drift'],'proxy changes',report['proxy']['changed'],flush=True)
    torch.save(dict(parameters={k:(x.cpu() if torch.is_tensor(x) else x) for k,x in p.items()},proxy_parameters={k:(x.cpu() if torch.is_tensor(x) else x) for k,x in pe.items()},z=z.cpu(),h=h.cpu(),zp=zp.cpu(),hp=hp.cpu(),v=v.cpu(),u=u.cpu(),mode_indices=chosen.cpu(),mode_importance=importance.cpu(),directions=directions,points=points,indices=ids),out/'diagnostic.pt')
    save_json(out/'execution.json',dict(completed=True,source_n=len(ids),fit_n=n,probe_n=len(zprobe),source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),raw_sha256=digest(out/'diagnostic.pt'),wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),new_expert_updates=0,readout_Jacobian_backward_calls=before.shape[1],publication_gradient_updates=0,new_release_count=sum('frozen_release' in r for r in records),partner_used=False,test_used=False,confirmation_used=False,novelty_established=False,scope='One analytic construction with prespecified SVD ablations and separate0.5/2.0 budgets; source proxy is not an unknown-peer guarantee.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',default='configs/postactivation_subspace.json');args=ap.parse_args();run(args.protocol)
