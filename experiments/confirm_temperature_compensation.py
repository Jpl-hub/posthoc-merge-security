"""Fresh target-training confirmation and full-model replay of the sole point."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from angular_distance import signed_residual
from attention_cascade import parameters
from boundary_game import risk_first
from clip_benchmark import digest
from covariance_noise import dataset
from mechanisms import load_author
from probe_distance_candidate import on_device
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, initialize
from temperature_compensation import cpu, metric, pack_inputs, predict, thermal


def edits(original, point):
    prefix='model.visual.transformer.resblocks.11.';state=original.state_dict();cfg=point['cfg']
    w=state[prefix+'attn.in_proj_weight'].double().clone();b=state[prefix+'attn.in_proj_bias'].double().clone();d=w.shape[1]
    w[:2*d]*=1.+cfg['gain_delta'];b[:2*d]*=1.+cfg['gain_delta']
    db=state[prefix+'mlp.c_proj.weight'].double().norm()*point['output_basis']@point['k']
    return {prefix+'attn.in_proj_weight':w.float(),prefix+'attn.in_proj_bias':b.float(),prefix+'mlp.c_proj.weight':(state[prefix+'mlp.c_proj.weight'].double()+db).float()}


def unused(training, ids, cfg):
    excluded=set(ids);sources={}
    audit=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    for row in audit['records']:
        path=Path(row['path']);group=row['group']
        pin=row.get('sha256')
        if pin is None:
            pin=json.loads(path.with_name(cfg['target']+'.json').read_text())['diagnostic_sha256']
        assert digest(path)==pin
        record=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        excluded.update(record[group]);sources[str(path)]=digest(path)
    for folder in Path('outputs').iterdir():
        if not folder.is_dir():continue
        case=folder/'case.json';own=cfg['target'].lower() in folder.name.lower()
        if case.exists():own=json.loads(case.read_text()).get('target')==cfg['target']
        if not own:continue
        for name in ['training_indices.json','holdout_protocol.json','confirmation_split.json']:
            path=folder/name
            if not path.exists():continue
            v=json.loads(path.read_text())
            if name=='holdout_protocol.json' and v.get('split')!='train':continue
            groups=list(v.values()) if name=='training_indices.json' else [v.get('indices',[])]
            for group in groups:
                if isinstance(group,list):excluded.update(int(x) for x in group)
            sources[str(path)]=digest(path)
    pool=sorted(set(range(len(training)))-excluded);assert len(pool)>=cfg['confirmation_n']
    chosen=np.random.default_rng(cfg['confirmation_seed']).choice(pool,cfg['confirmation_n'],replace=False).tolist()
    assert not set(chosen)&excluded
    return dict(indices=chosen,excluded_count=len(excluded),excluded_metadata_sha256=sources,seed=cfg['confirmation_seed'],split='target_train',used_for_optimization=False)


@torch.no_grad()
def confirm(cfg):
    start=time.monotonic();out=Path(cfg['output']);report=json.loads((out/'results.json').read_text())
    valid=json.loads((out/'independent_validation.json').read_text());assert valid['status']=='passed_task_calibration'
    assert digest(out/'training_point.pt')==report['training_point_sha256']==valid['point_sha256']
    point=torch.load(out/'training_point.pt',map_location='cpu');old=torch.load(out/'diagnostic.pt',map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==report['checkpoint_sha256'];p=on_device(parameters(expert,head))
    training=dataset(cfg['target'],'data',expert.val_preprocess,True)
    split_file=out/'confirmation_split.json'
    # A setup failure must never silently draw another confirmation set.
    split=json.loads(split_file.read_text()) if split_file.exists() else unused(training,old['indices'],cfg)
    assert split['seed']==cfg['confirmation_seed'] and not set(split['indices'])&set(old['indices'])
    save_json(out/'confirmation_split.json',split)
    groups={k:[] for k in ['H','h']};block=expert.model.visual.transformer.resblocks[11]
    hooks=[block.ln_1.register_forward_pre_hook(lambda _m,args:groups['H'].append(args[0].transpose(0,1).detach().cpu())),
           expert.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:groups['h'].append(args[0].detach().cpu()))]
    try:
        expert.cuda().eval()
        for images,_ in make_loader(training,split['indices'],64):expert(images.cuda())
    finally:
        for hook in hooks:hook.remove()
        expert.cpu()
    cache={k:torch.cat(v) for k,v in groups.items()};cache['h']=cache['h'].double();rounding=[]
    for start0 in range(0,len(cache['h']),cfg['batch_size']):
        h=cache['H'][start0:start0+cfg['batch_size']].cuda().double();rounding.append((cache['h'][start0:start0+len(h)].cuda().double()-thermal(h,p,1.)[0]).cpu())
    cache['rounding']=torch.cat(rounding);cache,ids,n=risk_first(cache,split['indices'],cfg['contexts_per_class'],p)
    basis=on_device(point['task_basis']);data=pack_inputs(cache,list(range(n)),p,basis,point['B_basis'],cfg)
    k=point['k'].cuda();q=point['output_basis'].cuda();metrics=metric(data,p,q,k,cfg)
    before=data['teacher'].argmax(1).cpu();after=predict(data,p,q,k).argmax(1).cpu();labels=torch.tensor([int(training[i][1]) for i in ids])
    release=copy.deepcopy(expert);changed=edits(expert,point)
    for key,value in changed.items():release.state_dict()[key].copy_(value)
    author,pin=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    assert pin==json.loads(Path('configs/sources.json').read_text())['AGA']['sha256']
    recovered0=author['attack_attention_headwise_precise'](expert.state_dict(),anchor.state_dict(),11,num_heads=12,device='cuda')
    recovered1=author['attack_attention_headwise_precise'](release.state_dict(),anchor.state_dict(),11,num_heads=12,device='cuda')
    key='model.visual.transformer.resblocks.11.attn.in_proj_weight';width=expert.state_dict()[key].shape[1]
    recovery_error=float((recovered1[key][:2*width].double()-recovered0[key][:2*width].double()).norm()/recovered0[key][:2*width].double().norm())
    bkey=key.replace('weight','bias')
    recovery_bias_error=float((recovered1[bkey][:2*width].double()-recovered0[bkey][:2*width].double()).abs().max())
    checks={};replay_n=len(ids)
    for name,model,expected in [('original',expert,before),('released',release,after)]:
        values=[];model.cuda().eval();head.cuda().eval()
        for images,_ in make_loader(training,ids,64):values.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu();head.cpu();prediction=torch.cat(values);assert torch.equal(prediction,expected[:replay_n]),name
        checks[name]=dict(n=replay_n,correct=int((prediction==labels).sum()),all_predictions_match=True)
    passed=metrics['worst_margin_drift']<=cfg['task_budget'] and metrics['distance']<=cfg['parameter_budget'] and metrics['natural_changes']==0
    torch.save(dict(cache=cache,indices=ids,teacher=data['teacher'].cpu(),released_scores=predict(data,p,q,k).cpu(),labels=labels),out/'confirmation.pt')
    if passed:torch.save(changed,out/'release_edits.pt')
    result=dict(status='passed' if passed else 'failed_task_confirmation',confirmation_passed=passed,target=cfg['target'],**metrics,
                original_correct=int((before[:len(ids)]==labels).sum()),released_correct=int((after[:len(ids)]==labels).sum()),
                full_model_replay=checks,new_model_forward_instances=3*len(ids),training_point_sha256=report['training_point_sha256'],
                author_AGA_source_QK_relative_difference=recovery_error,author_AGA_source_QK_bias_maximum_difference=recovery_bias_error,
                confirmation_sha256=digest(out/'confirmation.pt'),release_sha256=digest(out/'release_edits.pt') if passed else None,
                source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,new_release_weights=1 if passed else 0,
                partner_used=False,test_used=False,selected_after_confirmation=False,original_expert_updates=0,
                scope='Sole pre-frozen point; fresh target-training images exclude registered historical data. Full512 original/released replay validates FP32 release implementation. No proxy or midpoint classification gate.')
    save_json(out/'confirmation_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();initialize();confirm(json.loads(Path(args.protocol).read_text()))
