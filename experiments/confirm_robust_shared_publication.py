"""One fresh confirmation of a frozen O/B finite-environment publication."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from angular_distance import signed_residual
from boundary_game import risk_first
from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import capture_attention,normalized_input
from probe_margin_distance import scores
from robust_shared_publication import RobustPublication
from subspace_probe import make_loader,save_json
from task_probe_distance import checked_models,gelu,initialize
from verify_cross_module import ln
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores,residual


@torch.no_grad()
def confirm(cfg):
    start=time.monotonic();root=Path(cfg['output']);report=json.loads((root/'results.json').read_text())
    valid=json.loads((root/'independent_validation.json').read_text())
    assert report['confirmation_eligible'] and valid['status']=='passed_task_calibration'
    assert digest(root/'training_point.pt')==report['training_point_sha256']==valid['point_sha256']
    point=torch.load(root/'training_point.pt',map_location='cpu');p=RobustPublication(cfg)
    anchor,expert,head,pins=checked_models(cfg['target']);released=copy.deepcopy(expert)
    prefix='model.visual.transformer.resblocks.11.';keys={'O':prefix+'attn.out_proj.weight','B':prefix+'mlp.c_proj.weight'}
    delta={}
    for name,key in keys.items():
        original=expert.state_dict()[key];assert torch.equal(original.double(),p.pars[name].cpu())
        changed=(original.double()+point['release_deltas'][name]).float()
        released.state_dict()[key].copy_(changed);delta[name]=(changed.double()-original.double()).cuda()
    distance=float(sum((delta[k].norm()/p.pars[k].norm())**2 for k in keys)**.5)
    def paired(pack):
        r=pack['r']+pack['o_input']@delta['O'].T
        z=gelu(normalized_input(r,p.pars)@p.pars['A'].T+p.pars['a'])+pack['b_offset']
        return pack['teacher'],scores(pack['rounding']+r+z@(p.pars['B']+delta['B']).T+p.pars['b'],p.pars)
    before,after=paired(p.source);source_drift=float(signed_residual(before,after).abs().max())
    assert distance<=.5 and source_drift<=.5,'FP32 export failed original source budget; no repair'
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);split_file=root/'confirmation_split.json'
    used=set(i for group in p.indices.values() for i in group)
    split=json.loads(split_file.read_text()) if split_file.exists() else unused(training,used,cfg)
    assert split['seed']==cfg['confirmation_seed'] and not set(split['indices'])&used
    save_json(split_file,split)
    cache=capture_attention(expert,training,split['indices'])
    cache,ids,n=risk_first(cache,split['indices'],cfg['contexts_per_class'],p.pars)
    assert n==p.p.cfg['probe_contexts']
    pack=p.p.prepare(cache,p.pars,True);bs,aa=paired(pack);drift=float(signed_residual(bs,aa).abs().max())
    par=convert({k:v.cpu() if torch.is_tensor(v) else v for k,v in p.pars.items()});pn=convert({k:v.cpu() for k,v in pack.items()})
    do,db=delta['O'].cpu().numpy(),delta['B'].cpu().numpy();r=pn['r']+pn['o_input']@do.T
    pre=ln(r,par['gamma2'],par['beta2'],par['eps2'])@par['A'].T+par['a'];z=pre*expit(1.702*pre)+pn['b_offset']
    independent=raw_scores(pn['rounding']+r+z@(par['B']+db).T+par['b'],par)
    np.testing.assert_allclose(independent,aa.cpu().numpy(),atol=1e-9,rtol=1e-9)
    independent_drift=float(np.abs(residual(bs.cpu().numpy(),independent)).max())
    labels=torch.tensor([int(training[i][1]) for i in ids]);prediction0=bs[:len(ids)].argmax(1).cpu();prediction1=aa[:len(ids)].argmax(1).cpu()
    replay={}
    for name,model,expected in [('original',expert,prediction0),('released',released,prediction1)]:
        values=[];model.cuda().eval();head.cuda().eval()
        for images,_ in make_loader(training,ids,64):values.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu();head.cpu();actual=torch.cat(values);assert torch.equal(actual,expected),name
        replay[name]=dict(n=len(ids),correct=int((actual==labels).sum()),all_predictions_match=True)
    passed=bool(drift<=.5 and distance<=.5 and torch.equal(prediction0,prediction1))
    torch.save(dict(cache=cache,indices=ids,labels=labels,teacher=bs.cpu(),released_scores=aa.cpu(),release_deltas={k:v.cpu() for k,v in delta.items()}),root/'confirmation.pt')
    result=dict(status='passed' if passed else 'failed_task_confirmation',confirmation_passed=passed,target=cfg['target'],
        natural_n=len(ids),all_task_inputs=len(bs),natural_changes=int((prediction0!=prediction1).sum()),
        original_correct=int((prediction0==labels).sum()),released_correct=int((prediction1==labels).sum()),
        parameter_distance=distance,source_FP32_worst_drift=source_drift,worst_margin_drift=drift,numpy_worst_margin_drift=independent_drift,
        full_model_replay=replay,new_model_forwards=3*len(ids),point_sha256=report['training_point_sha256'],
        confirmation_sha256=digest(root/'confirmation.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        peer_used=False,test_used=False,selected_after_confirmation=False)
    report['checkpoint_sha256']=pins;report['independent_confirmation_run']=True
    if passed:
        torch.save(dict(release_deltas={k:v.cpu() for k,v in delta.items()},theta=point['theta'],confirmation_indices=ids,
            training_point_sha256=report['training_point_sha256'],cfg=cfg),root/'candidate.pt')
        result['candidate_sha256']=digest(root/'candidate.pt')
        report.update(status='passed_robust_shared_publication_confirmation',confirmation_passed=True,
            candidate_sha256=result['candidate_sha256'],confirmation=result,new_release_weights=1)
        save_json(root/'independent_validation.json',dict(status='passed',source_calibration=valid,confirmation=result,
            new_model_forwards=3*len(ids),candidate_sha256=result['candidate_sha256']))
    save_json(root/'results.json',report);save_json(root/'confirmation_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    initialize();confirm(json.loads(Path(args.protocol).read_text()))
