"""One bounded LayerNorm/MLP conjugacy feasibility check, not a release."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from angular_distance import signed_residual
from clip_benchmark import digest
from mixed_compensation_audit import activation, ln
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln as numpy_ln
from verify_levelset_audits import convert, raw_scores


@torch.no_grad()
def audit(cfg):
    start=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    old=torch.load(cfg['mixed_diagnostic'],map_location='cpu');point=torch.load(cfg['point'],map_location='cpu')
    p,pp=on_device(old['parameters']),on_device(old['proxy_parameters'])
    # Solve the stated budget once. No scores, data, proxy or parameter search
    # enters this analytic root: two LN vectors and one input matrix change.
    low,high=1.,2.
    for _ in range(cfg['root_steps']):
        mid=(low+high)/2
        if 2*(mid-1)**2+(1/mid-1)**2>cfg['parameter_budget']**2:high=mid
        else:low=mid
    scale=low;d=dict(A=p['A']*(1/scale-1),gamma2=p['gamma2']*(scale-1),beta2=p['beta2']*(scale-1))
    distance=sum((v.norm()/p[k].norm()).square() for k,v in d.items()).sqrt();assert distance<=cfg['parameter_budget']
    # The public scalar follows from the published fixed budget; its inverse
    # needs no original weights or training samples.
    inverse_error=max(float(((p[k]+d[k])*(scale if k=='A' else 1/scale)-p[k]).abs().max()) for k in d)
    assert inverse_error<1e-12
    gold=official_labels(cfg['target'],True);records=[];raw={}
    for name,cache,indices,params,factor in [
        ('native',point['native'],point['indices']['fit'],p,1.),
        ('scale_train',point['scale'],point['indices']['scale'],p,1.),
        ('anchor_interpolation',point['proxy'],point['indices']['proxy'],pp,cfg['coefficient'])]:
        cache=on_device(cache);altered={**params,**{key:params[key]+factor*value for key,value in d.items()}}
        pre0=ln(cache['r'],params)@params['A'].T+params['a']
        pre1=ln(cache['r'],altered)@altered['A'].T+params['a']
        h=cache['h']+(activation(pre1)[0]-activation(pre0)[0])@params['B'].T
        before,after=scores(cache['h'],params),scores(h,params);y=before.argmax(1);pred=after.argmax(1);labels=torch.tensor(gold[indices],device='cuda')
        # Separate NumPy LayerNorm, gate and final feature normalization.
        c,par,new=convert(cache),convert(params),convert(altered)
        v0=numpy_ln(c['r'],par['gamma2'],par['beta2'],par['eps2'])@par['A'].T+par['a']
        v1=numpy_ln(c['r'],new['gamma2'],new['beta2'],new['eps2'])@new['A'].T+new['a']
        hn=c['h']+(v1*expit(1.702*v1)-v0*expit(1.702*v0))@par['B'].T
        checked=raw_scores(hn,par);error=float(np.abs(checked-after.cpu().numpy()).max());assert error<1e-9
        np.testing.assert_array_equal(checked.argmax(1),pred.cpu().numpy())
        if name!='anchor_interpolation':assert (pre1-pre0).abs().max()<1e-10 and torch.equal(y,pred)
        changed=y!=pred
        records.append(dict(environment=name,n=len(y),original_correct=int((y==labels).sum()),released_correct=int((pred==labels).sum()),
                            changed=int(changed.sum()),changed_by_class=torch.bincount(y[changed],minlength=len(params['head'])).cpu().tolist(),
                            maximum_task_margin_drift=float(signed_residual(before,after).abs().max()),
                            preactivation_change_rms=float((pre1-pre0).square().mean().sqrt()),numpy_score_maximum_error=error))
        raw[name]=dict(indices=indices,labels=labels.cpu(),original_scores=before.cpu(),changed_scores=after.cpu())
    file=out/'diagnostic.pt';torch.save(dict(cfg=cfg,scale=scale,groups=raw),file)
    report=dict(status='exact_conjugacy_diagnosis_publicly_invertible_not_a_release',scale=scale,parameter_distance=float(distance),
                ideal_same_environment_halfway_gate_gain=((1+scale)/2)*((1+1/scale)/2),inverse_parameter_maximum_error=inverse_error,
                records=records,diagnostic_sha256=digest(file),source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                original_expert_updates=0,publication_optimization_steps=0,new_release_weights=0,full_model_forward_instances=0,
                partner_used=False,test_used=False,confirmation_used=False,wall_seconds=time.monotonic()-start,
                interpretation='New exact-preservation structural alternative, checked once at the fixed analytic budget. Public inverse scalar exactly restores the source parameters; not recovery resistance or a claim of AGA success. No TA/AGA without a worthwhile mechanism.')
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    assert digest(__file__)==cfg['source_sha256'];torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;audit(cfg)
