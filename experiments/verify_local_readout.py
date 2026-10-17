"""Independent source algebra and full-model utility witnesses; never repairs points."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from author_protocol_audit import setup
from clip_benchmark import digest
from confirm_postactivation_subspace import numpy_metric
from covariance_noise import dataset
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_levelset_audits import raw_scores


def arrays(p):
    return {k:v.numpy() if torch.is_tensor(v) else v for k,v in p.items()}


@torch.no_grad()
def source(output):
    out=Path(output);meta=json.loads((out/'execution.json').read_text());assert digest(out/'diagnostic.pt')==meta['raw_sha256']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu',weights_only=False);ref=raw['source_reference'];assert digest(ref['path'])==ref['sha256']
    parent=torch.load(ref['path'],map_location='cpu',weights_only=False);cfg=json.loads((out/'protocol.json').read_text());rows=json.loads((out/'results.json').read_text())
    p,pe=arrays(parent['parameters']),arrays(parent['proxy_parameters']);h,z,hp,zp=[parent[k].numpy() for k in ['h','z','hp','zp']];u,c=raw['u'].numpy(),raw['c'].numpy();n=cfg['fit_n']
    # Independent dual normal-equation solution, versus original SVD minimum-norm solve.
    dual=z[:n].T@np.linalg.solve(z[:n]@z[:n].T,h[:n]@u);cerror=float(np.linalg.norm(dual-c)/np.linalg.norm(c));assert cerror<1e-6
    anchor,own,head,_=checked_models('MNIST');base={'B':anchor.model.visual.transformer.resblocks[11].mlp.c_proj.weight.double().numpy(),'projection':anchor.model.visual.proj.double().numpy(),'beta':anchor.model.visual.ln_post.bias.double().numpy()}
    v=parent['v'].numpy();s0=raw_scores(h,p);yy=s0[:n].argmax(1);context=np.concatenate([np.flatnonzero(yy==i)[:2] for i in range(s0.shape[1])]);rms=np.sqrt(np.mean((z[:n]@v.T)**2,axis=0));shift=(.01*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).reshape(-1,z.shape[1]);zt=(z[context,None]+shift[None]).reshape(-1,z.shape[1]);ht=(h[context,None]+(shift@p['B'].T)[None]).reshape(-1,h.shape[1])
    maximum=0.;certificates=[]
    for row in rows:
        point=raw['points'][str(row['parameter_budget'])];delta=arrays(point['delta']);q=p|{k:p[k]+a for k,a in delta.items()}
        proxy=pe|{k:(base[k].astype('float32')+np.float32(.5)*((q[k].astype('float32')-base[k].astype('float32'))+np.zeros_like(base[k],dtype='float32'))).astype('float64') for k in delta}
        for name,hh,zz,pp,dd in [('source',h,z,q,delta['B']),('proxy',hp,zp,proxy,proxy['B']-pe['B']),('probes',ht,zt,q,delta['B'])]:
            before=raw_scores(hh,pe if name=='proxy' else p);after=raw_scores(hh+zz@dd.T,pp);error=float(abs(after-point[name].numpy()).max());maximum=max(maximum,error);assert error<1e-8
            drift,changed=numpy_metric(before,after);assert abs(drift-row[name]['worst_task_drift'])<1e-7 and changed==row[name]['changed']
            pred=before.argmax(1);bad=after.argmax(1)!=pred
            for item in row[name]['per_class']:assert int((pred==item['class_id']).sum())==item['n'] and int((bad&(pred==item['class_id'])).sum())==item['changed']
        d=sum(np.linalg.norm(a)**2/np.linalg.norm(p[k])**2 for k,a in delta.items())**.5;assert abs(d-row['actual_distance'])<1e-10
        if cfg.get('mode')=='class_readout_reflection':
            reflection=np.eye(h.shape[1])-2*u@u.T;gamma=p['gamma'];pm=((reflection.T@(gamma[:,None]*p['projection']))/gamma[:,None]+p['projection'])/2
            bm=((p['beta']/gamma)@reflection*gamma+p['beta'])/2;contrast=p['head']-p['head'].mean(0);image=pm@contrast.T;m=gamma[:,None]*image;m-=m.mean(0);b=bm@image
            lower=b[:,None]-b[None,:]-np.sqrt(h.shape[1])*np.linalg.norm(m[:,:,None]-m[:,None,:],axis=0);np.fill_diagonal(lower,np.inf);k=int(lower.min(1).argmax());margin=float(lower[k].min())
            expected=row['ideal_expert_midpoint'];assert abs(margin-expected['worst_class_margin_lower_bound'])<1e-8 and np.linalg.norm(m)<1e-8 and k==expected['best_constant_class']
            certificates.append(dict(class_id=k,margin_lower_bound=margin,valid_for_all_prefix_outputs=margin>0))
    save_json(out/'independent_validation.json',dict(status='passed_source_algebra',NumPy_complete_responses_classes_and_distances=True,dual_normal_equation_coefficient_error=cerror,maximum_score_error=maximum,ideal_midpoint_certificates=certificates,source_raw_sha256=meta['raw_sha256'],checker_sha256=digest(__file__),new_images=0,gradients=0,release_repaired=False))


@torch.no_grad()
def failure(output):
    out=Path(output);cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text());assert not meta['completed'] and meta['stopped']=='full_native_task_budget_violation'
    file=out/'artifacts/predictions.pt';assert digest(file)==meta['predictions_sha256'];raw=torch.load(file,map_location='cpu',weights_only=False)
    a,b=[raw[k+'/standalone/MNIST']['logits'].double().numpy() for k in ['original','released']];pred=a.argmax(1);i=np.arange(len(pred));g=a[i,pred,None]-a;mask=np.ones(g.shape,bool);mask[i,pred]=False;g[i,pred]=1;ds=b-a;rho=(ds[i,pred,None]-ds)/g;rho[i,pred]=0
    pos,competitor=np.unravel_index(abs(rho).argmax(),rho.shape);assert abs(rho[pos,competitor])>cfg['task_budget']
    _,own,_,released,heads,_=setup(cfg);data=dataset('MNIST','data',own.val_preprocess);lo=int(pos)//128*128;hi=min(lo+128,len(data));images=torch.stack([data[j][0] for j in range(lo,hi)]).cuda();head=heads['MNIST'].cuda();errors=[];scores=[]
    for model,saved in [(own,a),(released,b)]:
        model.cuda();actual=head(model(images)).cpu().double().numpy();model.cpu();errors.append(float(abs(actual-saved[lo:hi]).max()));scores.append(actual[int(pos)-lo]);assert errors[-1]<=cfg['score_tolerance']
    before,after=scores;teacher=int(pred[pos]);ratio=((after[teacher]-after[competitor])-(before[teacher]-before[competitor]))/(before[teacher]-before[competitor]);assert abs(ratio)>cfg['task_budget']
    record=dict(status='full_task_budget_counterexample_independently_replayed',test_index=int(pos),teacher_class=teacher,competitor=int(competitor),label=int(data[int(pos)][1]),prediction_before=int(before.argmax()),prediction_after=int(after.argmax()),relative_drift=float(ratio),NumPy_cached_drift=float(rho[pos,competitor]),maximum_context_score_error=max(errors),before=before.tolist(),after=after.tolist(),raw_sha256=meta['predictions_sha256'],release_sha256=cfg['release']['sha256'],checker_sha256=digest(__file__),source_point_repaired=False,AGA_calls=0,new_release_weights=0,full_model_forward_images=2*(hi-lo))
    save_json(out/'utility_witness_validation.json',record)


@torch.no_grad()
def ln_pre(output):
    """Can this fixed source R extend through native CLIP's initial affine LN?"""
    out=Path(output);meta=json.loads((out/'execution.json').read_text())
    assert digest(out/'diagnostic.pt')==meta['raw_sha256']
    u=torch.load(out/'diagnostic.pt',map_location='cpu',weights_only=False)['u'].numpy()
    _,own,_,_=checked_models('MNIST');gamma=own.model.visual.ln_pre.weight.double().numpy()
    r=np.eye(len(gamma))-2*u@u.T;a=r-r.mean(0);target=gamma[:,None]*r;target-=target.mean(0)
    scale=np.sum(a*target,axis=0)/np.sum(a*a,axis=0);error=a*scale[None]-target
    # Each column's least-squares residual is orthogonal to its only free column.
    normal=float(np.max(np.abs(np.sum(a*error,axis=0))));assert normal<1e-10
    saved=json.loads((out/'native_ln_pre_obstruction.json').read_text())
    f,op=float(np.linalg.norm(error)),float(np.linalg.norm(error,ord=2))
    assert abs(f-saved['minimum_diagonal_affine_residual_Frobenius'])<1e-9
    assert abs(op-saved['minimum_diagonal_affine_residual_spectral'])<1e-9 and op>1e-8
    save_json(out/'native_ln_pre_validation.json',dict(status='passed_numpy_minimum_residual',
        residual_Frobenius=f,residual_spectral=op,normal_equation_error=normal,
        original_expert_sha256=saved['original_expert_sha256'],source_raw_sha256=meta['raw_sha256'],
        checker_sha256=digest(__file__),images=0,gradients=0,new_weights=0,
        scope='This fixed orthogonal coordinate conjugacy is not exactly realizable by native ln_pre diagonal affine parameters. Not a proof against all posthoc methods.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--source');g.add_argument('--failure');g.add_argument('--ln-pre');a=ap.parse_args()
    if a.source: source(a.source)
    elif a.failure: failure(a.failure)
    else: ln_pre(a.ln_pre)
