"""Independent NumPy finite-circuit audit and explicit full-model source replay."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, softmax

from clip_benchmark import digest
from covariance_noise import dataset
from cross_module_distance import extended_parameters
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models


def array(x):
    if isinstance(x, dict): return {k: array(v) for k, v in x.items()}
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else x


def attn(x, a):
    q, k, v = np.split(x@a['W'].T+a['bias'], 3, axis=-1)
    n, length, d = x.shape
    q=q[:,0].reshape(n,12,64); k=k.reshape(n,length,12,64).transpose(0,2,1,3)
    v=v.reshape(n,length,12,64).transpose(0,2,1,3)
    prob=softmax(np.einsum('nhd,nhld->nhl',q,k)/8.,axis=-1)
    y=np.einsum('nhl,nhld->nhd',prob,v).reshape(n,768)@a['O'].T+a['ob']
    return y, prob, v


def ln(x, g, b, eps):
    centered=x-x.mean(-1,keepdims=True)
    return centered/np.sqrt((centered**2).mean(-1,keepdims=True)+eps)*g+b


def classify(cache, a, p):
    r=cache['before']+attn(cache['x'],a)[0]
    pre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a']
    h=r+(pre*expit(1.702*pre))@p['B'].T+p['b']
    feature=ln(h,p['gamma'],p['beta'],p['eps'])@p['projection']
    return feature/np.linalg.norm(feature,axis=-1,keepdims=True)@p['head'].T


def stats(before, after):
    win=before.argmax(1); idx=np.arange(len(win))
    gaps=before[idx,win,None]-before; valid=gaps>0
    ds=after-before; difference=ds[idx,win,None]-ds
    rho=np.divide(difference,gaps,out=np.zeros_like(gaps),where=valid)
    changed=after.argmax(1)!=win
    return dict(n=len(win),changed_predictions=int(changed.sum()),worst_margin_drift=float(abs(rho).max()),
                max_signed_margin_damage=float((-rho).max()),
                per_class_changed=[int(changed[win==c].sum()) for c in range(10)])


def np_distance(a,b,bias):
    return float(np.sqrt(sum(np.linalg.norm(b[k]-a[k])**2/np.linalg.norm(a[k])**2 for k in (['W','O','bias'] if bias else ['W','O']))))


@torch.no_grad()
def capture_public(model, head, train, ids):
    cache={k:[] for k in ['x','before','scores']}
    block=model.model.visual.transformer.resblocks[-1]
    hooks=[block.attn.register_forward_pre_hook(lambda m,a:cache['x'].append(a[0].transpose(0,1).cpu())),
           block.ln_1.register_forward_pre_hook(lambda m,a:cache['before'].append(a[0][0].cpu()))]
    try:
        model.cuda();head.cuda()
        for images,_ in make_loader(train,ids,64):cache['scores'].append(head(model(images.cuda())).cpu())
    finally:
        model.cpu();head.cpu()
        for h in hooks:h.remove()
    return {k:torch.cat(v) for k,v in cache.items()}


@torch.no_grad()
def run(output):
    start=time.monotonic();out=Path(output);cfg=json.loads((out/'protocol.json').read_text())
    report=json.loads((out/'results.json').read_text());assert digest(out/'artifacts/finite_audit.pt')==report['finite_audit_sha256']
    assert digest(out/'artifacts/diagnostic_edits.pt')==report['diagnostic_edits_sha256']
    raw=torch.load(out/'artifacts/finite_audit.pt',map_location='cpu');d=array(raw)
    a,b,p=d['attention_parameters'],d['modified_attention_parameters'],d['tail_parameters']
    scores={};errors={};statistics={}
    for group,cache,aa,bb,before,after in [
        ('natural',d['cache'],a,b,d['original'],d['natural']),
        ('probes',d['probes'],a,b,d['probe_before'],d['probe_after']),
        ('intervention',d['cache'],d['AGA_baseline'],d['AGA_modified'],d['intervention_before'],d['intervention_after'])]:
        u=classify(cache,aa,p);v=classify(cache,bb,p)
        err=max(float(abs(u-before).max()),float(abs(v-after).max()));assert err<1e-8,(group,err)
        errors[group]=err;statistics[group]=stats(u,v);scores[group]=[u,v]
        expected=report['AGA_source_module_intervention' if group=='intervention' else group]
        assert statistics[group]['changed_predictions']==expected['changed_predictions']
        assert abs(statistics[group]['worst_margin_drift']-expected['worst_margin_drift'])<1e-8
        print('NUMPY',group,statistics[group],flush=True)
    np.testing.assert_allclose(np_distance(a,b,True),report['parameter_distance'],atol=1e-12)
    np.testing.assert_allclose(np_distance(a,b,False),report['matrix_distance'],atol=1e-12)
    assert report['parameter_distance']<=cfg['parameter_budget']
    assert np.array_equal(a['W'][:1536],b['W'][:1536])
    assert np.array_equal(a['bias'][:1536],b['bias'][:1536])
    original_y,prob,z=attn(d['cache']['x'],a);winners=d['original'].argmax(1);risks=[];comm_error=0.
    for record in d['selection']:
        i,j=record['pair'];angle=record['angle_radians'];c,s=math.cos(angle),math.sin(angle)
        diff=prob[:,i]-prob[:,j]
        vi=np.einsum('nl,nld->nd',diff,z[:,i]);vj=np.einsum('nl,nld->nd',diff,z[:,j])
        first=-s*s*vi-c*s*vj;second=-c*s*vi+s*s*vj
        delta=first@a['O'][:,i*64:(i+1)*64].T+second@a['O'][:,j*64:(j+1)*64].T
        fraction=(delta**2).sum(1)/(original_y**2).sum(1)
        risk=max(float(fraction[winners==k].mean()) for k in range(10) if (winners==k).any())
        np.testing.assert_allclose(risk,record['worst_class_mean_attention_relative_energy'],rtol=1e-10,atol=1e-12)
        risks.append((risk,[i,j]))
        if [i,j]==d['selected']['pair']:
            comm_error=float(abs(delta-(attn(d['cache']['x'],b)[0]-original_y)).max())
            assert comm_error<2e-6,comm_error
            identity_error=float(abs((first**2+second**2).sum(1)-s*s*(vi**2+vj**2).sum(1)).max())
            assert identity_error<1e-10,identity_error
    assert min(risks,key=lambda x:(x[0],x[1]))[1]==d['selected']['pair']
    print('NUMPY structural selection, exact finite commutator and distances passed',flush=True)
    svd=d['task_SVD']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert '3090' in torch.cuda.get_device_name()
    anchor,own,head,hashes=checked_models('MNIST')
    anchor_att=anchor.model.visual.transformer.resblocks[-1].attn
    aa={'W':anchor_att.in_proj_weight.double().numpy(),'bias':anchor_att.in_proj_bias.double().numpy(),
        'O':anchor_att.out_proj.weight.double().numpy(),'ob':anchor_att.out_proj.bias.double().numpy()}
    error=float(abs((a['W'][1536:]-aa['W'][1536:])@svd['vh'].T-svd['u']*svd['s']).max());assert error<1e-10,error
    # Independently reconstruct every coefficient in official per-head AGA, including its FP32 T.
    algebra_error=0.
    for src,stored in [(a,d['AGA_baseline']),(b,d['AGA_modified'])]:
        for h in range(12):
            ix=slice(h*64,(h+1)*64)
            for offset in [0,768,1536]:
                rows=slice(offset+h*64,offset+(h+1)*64);x=src['W'][rows].T;y=aa['W'][rows].T
                t=np.linalg.solve(x.T@x+1e-8*np.eye(64),x.T@y).astype(np.float32)
                w=t.T@src['W'][rows].astype(np.float32);bias=src['bias'][rows].astype(np.float32)@t
                algebra_error=max(algebra_error,float(abs(w-stored['W'][rows]).max()),float(abs(bias-stored['bias'][rows]).max()))
            x=src['O'][:,ix];y=aa['O'][:,ix]
            t=np.linalg.solve(x.T@x+1e-8*np.eye(64),x.T@y).astype(np.float32)
            w=x.astype(np.float32)@t;algebra_error=max(algebra_error,float(abs(w-stored['O'][:,ix]).max()))
    assert algebra_error<5e-6,algebra_error
    # Audit missing proxy raw outputs by replaying the already-frozen point, not by choosing new parameters.
    proxy_stats=None;proxy_num_error=None
    if report['task_pass']:
        base=copy.deepcopy(anchor);sa,se=anchor.state_dict(),own.state_dict()
        state={k:(sa[k]+.5*(se[k]-sa[k])) if sa[k].is_floating_point() else sa[k] for k in sa}
        base.load_state_dict(state,strict=True)
        changed=copy.deepcopy(base);edits=torch.load(out/'artifacts/diagnostic_edits.pt',map_location='cpu')
        changed.load_state_dict(state|{k:state[k]+.5*(v-se[k]) for k,v in edits.items()},strict=True)
        train=dataset('MNIST','data',own.val_preprocess,True);ids=json.loads((out/'source_split.json').read_text())['indices']
        c0=capture_public(base,head,train,ids);c1=capture_public(changed,head,train,ids)
        proxy_stats=stats(array(c0['scores']),array(c1['scores']))
        assert proxy_stats['changed_predictions']==report['public_base_interpolation']['changed_predictions']
        assert abs(proxy_stats['worst_margin_drift']-report['public_base_interpolation']['worst_margin_drift'])<1e-4
        torch.save(dict(before=c0,after=c1,indices=ids,explicit_coefficient=.5,
                        source_edits_sha256=report['diagnostic_edits_sha256'],role='audit_replay_not_original_capture'),out/'artifacts/public_interpolation_replay.pt')
        ix=sorted(set([report['public_base_interpolation']['witness']['row']]+list(range(16))))
        proxy_num_error=0.
        for model,cache in [(base,c0),(changed,c1)]:
            att=model.model.visual.transformer.resblocks[-1].attn
            ap=array(dict(W=att.in_proj_weight,bias=att.in_proj_bias,O=att.out_proj.weight,ob=att.out_proj.bias))
            pp=array(extended_parameters(model,head));sample={k:array(v[ix]) for k,v in cache.items()}
            replay=classify(sample,ap,pp);err=float(abs(replay-sample['scores']).max())
            assert err<cfg['full_forward_atol'],err;proxy_num_error=max(proxy_num_error,err)
    v=dict(status='passed_independent_finite_and_source_replay',results_sha256=digest(out/'results.json'),
           verifier_sha256=digest(__file__),finite_audit_sha256=report['finite_audit_sha256'],
           NumPy_score_max_abs=errors,NumPy_statistics=statistics,independent_structural_edges=66,
           finite_commutator_max_abs=comm_error,exact_norm_identity_max_abs=identity_error,
           task_SVD_identity_max_abs=error,AGA_numpy_algebra_max_abs=algebra_error,
           public_interpolation_full_model_replay=proxy_stats,public_interpolation_NumPy_vs_full_model_max_abs=proxy_num_error,
           public_interpolation_raw_role='Independent evaluation-only replay; original in-process proxy cache was not saved.',
           public_interpolation_replay_sha256=digest(out/'artifacts/public_interpolation_replay.pt') if proxy_stats else None,
           verified_original_checkpoint_sha256=hashes,original_expert_updated=False,no_parameter_selection=True,
           no_test_or_partner_loaded=True,independent_confirmation_done=False,new_release_weights=0,
           wall_seconds=time.monotonic()-start)
    save_json(out/'independent_validation.json',v);print(json.dumps(v),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
