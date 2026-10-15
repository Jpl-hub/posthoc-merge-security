"""Independent NumPy finite reflection and fractional-knapsack dual check."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_cascade import order
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert,raw_scores,residual


def sigmoid_gelu(v):return v*expit(1.702*v)


def dual_bound(values,cost,budget):
    positive=values>0;v,w=values[positive],cost[positive]
    if np.sum(w)<=budget:multiplier=0.
    else:
        ratios=v/np.maximum(w,1e-30);indices=np.argsort(-ratios)
        stop=int(np.searchsorted(np.cumsum(w[indices]),budget,side='left'));multiplier=float(ratios[indices[stop]])
    return multiplier*budget+np.maximum(values-multiplier*cost,0.).sum()


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);meta=json.loads((out/'results.json').read_text())
    assert digest(out/'point.pt')==meta['point_sha256'] and digest(out/'predictions.pt')==meta['predictions_sha256']
    for file,pin in cfg['inputs'].items():assert digest(file)==pin
    point=torch.load(out/'point.pt',map_location='cpu');saved=torch.load(out/'predictions.pt',map_location='cpu');p=convert(torch.load(cfg['mixed_diagnostic'],map_location='cpu')['parameters'])
    boundary=torch.load(cfg['boundary_diagnostic'],map_location='cpu');old=torch.load(cfg['old_calibration'],map_location='cpu');basis=torch.load(cfg['basis_diagnostic'],map_location='cpu')['basis'];va=point['input_basis'].numpy();vb=basis['B']['vh'].numpy()
    np.testing.assert_allclose(va@va.T,np.eye(len(va)),atol=1e-10);np.testing.assert_allclose(va,basis['A']['vh'].numpy(),atol=1e-12)
    x=[];hh=[];ids=[];core=[]
    for cache,indices in [(boundary['screen'],boundary['split']['indices']),(old['scale'],old['indices']['scale'])]:
        c=convert(cache);chosen,n=order(c,p,cfg['contexts_per_class']);core.extend(range(sum(len(g) for g in x),sum(len(g) for g in x)+n))
        x.append(ln(c['r'][chosen],p['gamma2'],p['beta2'],p['eps2']));hh.append(c['h'][chosen]);ids.extend(np.asarray(indices)[chosen].tolist())
    x=np.concatenate(x);h=np.concatenate(hh);natural_n=len(x);pre=x@p['A'].T+p['a'];base=h-sigmoid_gelu(pre)@p['B'].T
    ra=np.sqrt(np.mean((x@va.T)**2,0));rb=np.sqrt(np.mean((sigmoid_gelu(pre)@vb.T)**2,0));sign=np.array([-1.,1.])
    # Producer multiplies its float32 sign tensor by the Python scalar before
    # promotion to double RMS/basis values. Reconstruct that frozen operation.
    step=float(np.float32(cfg['finite_step']))
    sa=(step*sign[:,None,None]*ra[None,:,None]*va[None]).reshape(2*len(va),-1)
    sb=(step*sign[:,None,None]*rb[None,:,None]*vb[None]).reshape(2*len(vb),-1)
    xx=np.concatenate([x,(x[core,None]+sa[None]).reshape(-1,x.shape[1]),np.repeat(x[core],len(sa),axis=0)])
    bb=np.concatenate([base,np.repeat(base[core],len(sa),axis=0),np.repeat(base[core],len(sb),axis=0)+np.tile(sb,(len(core),1))@p['B'].T])
    mask=point['mask'].numpy();assert np.array_equal(mask,np.round(mask)) and mask.min()>=0 and mask.max()<=1
    projection=(p['A']@va.T)@va;delta=-2*mask[:,None]*projection;actual_a=p['A']+delta
    distance=float(np.linalg.norm(delta)/np.linalg.norm(p['A']));assert distance<=cfg['parameter_budget']+1e-12
    values={key:[] for key in ['original','endpoint','midpoint']}
    for offset in range(0,len(xx),64):
        xx0,bb0=xx[offset:offset+64],bb[offset:offset+64]
        for key,weight in [('original',p['A']),('endpoint',actual_a),('midpoint',p['A']+.5*delta)]:
            values[key].append(raw_scores(bb0+sigmoid_gelu(xx0@weight.T+p['a'])@p['B'].T,p))
    s0,s1,sm=[np.concatenate(values[key]) for key in ['original','endpoint','midpoint']]
    np.testing.assert_allclose(s0,saved['source_scores'].numpy(),atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(s1,saved['endpoint_scores'].numpy(),atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(sm[:natural_n],saved['source_midpoint_scores'].numpy(),atol=1e-9,rtol=1e-9)
    drift=float(np.abs(residual(s0,s1)).max());changed=s1[:natural_n].argmax(1)!=s0[:natural_n].argmax(1);mid_changed=sm[:natural_n].argmax(1)!=s0[:natural_n].argmax(1)
    assert abs(drift-meta['task_worst_margin_drift'])<1e-6 and int(changed.sum())==meta['native_changed'] and int(mid_changed.sum())==meta['source_midpoint_changed']
    assert np.bincount(s0[:natural_n].argmax(1)[mid_changed],minlength=len(p['head'])).tolist()==meta['source_midpoint_changed_by_class']
    labels=official_labels(cfg['target'],True)[ids];assert int((s1[:natural_n].argmax(1)==labels).sum())==meta['native_released_correct']
    # Independent dual upper bounds validate the producer's primal fractional
    # budget bounds, including the nonlinear LayerNorm scale remainder.
    head=p['head']-p['head'].mean(0);m=p['gamma'][:,None]*(p['projection']@head.T);m-=m.mean(0);bias=p['beta']@p['projection']@head.T
    count=min(cfg['bound_n'],natural_n);sd=np.sqrt(np.mean((h[:count]-h[:count].mean(1,keepdims=True))**2,1,keepdims=True)+p['eps']);numer=h[:count]@m+sd*bias;y=numer.argmax(1)
    base_pre=x[:count]@p['A'].T+p['a'];dz=sigmoid_gelu(base_pre-x[:count]@projection.T)-sigmoid_gelu(base_pre)
    cost=4*np.sum(projection**2,1)/np.sum(p['A']**2);bn=np.linalg.norm(p['B']-p['B'].mean(0),axis=0)/np.sqrt(p['B'].shape[0]);bounds=[]
    for i in range(count):
        row=[]
        for c in range(len(head)):
            if c==y[i]:continue
            coeff=-dz[i]*(p['B'].T@(m[:,y[i]]-m[:,c]))+np.abs(dz[i])*bn*abs(bias[y[i]]-bias[c])
            row.append(dual_bound(coeff,cost,cfg['parameter_budget']**2)/(numer[i,y[i]]-numer[i,c]))
        bounds.append(max(row))
    bound_error=float(np.max(np.abs(np.asarray(bounds)-saved['upper_bounds'].numpy())));assert bound_error<1e-6
    assert abs(distance-meta['parameter_distance'])<1e-12
    result=dict(status='passed_training_candidate' if meta['finite_multiclass_gate_passed'] else 'passed_failure_verification',parameter_distance=distance,
                all_task_inputs=len(xx),task_worst_margin_drift=drift,native_changed=int(changed.sum()),source_midpoint_changed=int(mid_changed.sum()),
                fractional_primal_dual_bound_maximum_error=bound_error,point_sha256=meta['point_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
                synthetic_probe_step_exact_float32_value=step,
                new_release_weights=0,test_used=False,partner_used=False,scope='Independent direct published A reconstruction and full nonlinear natural/SVD responses; fractional upper bounds checked by LP dual certificates. No task-general, trueTA, AGA or recovery claim.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();verify(json.loads(Path(args.protocol).read_text()))
