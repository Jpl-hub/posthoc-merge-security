"""One frozen public-anchor replacement attack with an exact matched control."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import author_recover,checkpoint_loader,digest,evaluate,merge,paired_counts
from covariance_noise import dataset
from mechanisms import load_author
from subspace_probe import make_loader,save_json
from task_probe_distance import checked_models
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


@torch.no_grad()
def refill(published,anchor,keys):
    # The attack itself receives only the published model and public anchor.
    result=copy.deepcopy(published).cpu().eval()
    for key in keys:result.state_dict()[key].copy_(anchor.state_dict()[key])
    return result


def load_case(cfg):
    root=Path(cfg['defense_evaluation']);meta=json.loads((root/'execution.json').read_text());case=json.loads((root/'case.json').read_text())
    assert json.loads((root/'independent_validation.json').read_text())['status']=='passed'
    assert digest(root/'artifacts/release.pt')==cfg['release_sha256']==meta['release_sha256']
    assert digest(root/'artifacts/predictions.pt')==meta['predictions_sha256']
    anchor,original,head,pins=checked_models(case['target']);published=copy.deepcopy(original)
    edits=torch.load(root/'artifacts/release.pt',map_location='cpu');assert set(edits)==set(cfg['keys'])
    for key,value in edits.items():published.state_dict()[key].copy_(value)
    cleaned=refill(published,anchor,cfg['keys']);control=refill(original,anchor,cfg['keys'])
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in cleaned.state_dict().items())
    raw=torch.load(root/'artifacts/predictions.pt',map_location='cpu')
    return root,case,anchor,original,head,pins,cleaned,raw


def peer_models(case,cfg):
    load=checkpoint_loader(Path('external/secure-merge-attack'));root=Path('checkpoints/ViT-B-32')
    pins={p['path'].removeprefix('ViT-B-32/'):p['sha256'] for p in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    def checked(name):assert digest(root/name)==pins[name];return load(str(root/name)).cpu().eval()
    peer=checked(case['partner']+'/finetuned.pt');head=checked('head_'+case['partner']+'.pt')
    ref=cfg['AGA_references'][case['partner']];assert digest(ref['path'])==ref['sha256']
    attacked=copy.deepcopy(peer);attacked.load_state_dict(torch.load(ref['path'],map_location='cpu'),strict=True)
    return peer,head,attacked


@torch.no_grad()
def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);(out/'artifacts').mkdir();start=time.monotonic()
    root,case,anchor,original,head,pins,cleaned,reference=load_case(cfg)
    edits={key:cleaned.state_dict()[key].clone() for key in cfg['keys']};torch.save(edits,out/'artifacts/clean_edits.pt')
    freeze={'attack_inputs':['published model','public pretrained anchor','public method locations'],'target':case['target'],'partner':case['partner'],
        'known_method_locations':cfg['keys'],'gradient_updates':0,'construction_training_images':0,'partner_loaded_before_attack_freeze':False,'test_loaded_before_attack_freeze':False,
        'clean_edits_sha256':digest(out/'artifacts/clean_edits.pt'),'defense_release_sha256':cfg['release_sha256'],'cleaned_original_and_protected_exactly_equal':True,
        'original_finetuned_used_by_attack_function':False,'original_used_for_publication_assembly_and_matched_cost_control':True,
        'removed_task_update_relative_norm':{k:float((original.state_dict()[k]-anchor.state_dict()[k]).norm()/original.state_dict()[k].norm()) for k in cfg['keys']}}
    save_json(out/'attack_freeze.json',freeze)
    peer,peer_head,peer_a=peer_models(case,cfg);heads={case['target']:head,case['partner']:peer_head}
    datasets={t:dataset(t,'data',original.val_preprocess) for t in heads};records=[];raw={}
    def record(model,stage,task):
        pred=[];labels=[];metric=evaluate(model,heads[task],make_loader(datasets[task]),'cuda',lambda a,b:(pred.append(a),labels.append(b)))
        x,y=torch.cat(pred),torch.cat(labels);np.testing.assert_array_equal(y.numpy(),official_labels(task))
        key=stage+'/'+task;raw[key]={'prediction':x,'target':y};row={'stage':stage,'dataset':task,**metric}
        row['per_class']=[{'class_id':c,'n':int((y==c).sum()),'correct':int(((y==c)&(x==y)).sum())} for c in range(heads[task].weight.shape[0])]
        baseline_stage=stage
        for preset in ['original','released']:
            baseline=reference[preset+'/'+baseline_stage+'/'+task]['prediction'];row['paired_to_'+preset]=paired_counts(baseline.eq(y),x.eq(y));row['changed_from_'+preset]=int((baseline!=x).sum())
        records.append(row);print(json.dumps(row),flush=True)
    record(cleaned,'standalone',case['target']);fused=merge(anchor,cleaned,peer,cfg['coefficient'])
    for task in heads:record(fused,'direct',task)
    del fused
    author,pin=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'),True);assert pin==json.loads(Path('configs/sources.json').read_text())['AGA']['sha256']
    cleaned_a=author_recover(cleaned,anchor,author)
    ref=cfg['AGA_references'][case['target']];base=torch.load(ref['path'],map_location='cpu');assert digest(ref['path'])==ref['sha256']
    torch.save({k:v.cpu().clone() for k,v in cleaned_a.state_dict().items() if not torch.equal(v,base[k])},out/'artifacts/clean_AGA_edits.pt')
    fused=merge(anchor,cleaned_a,peer_a,cfg['coefficient'])
    for task in heads:record(fused,'AGA',task)
    torch.save(raw,out/'artifacts/predictions.pt');save_json(out/'results.json',records);save_json(out/'protocol.json',cfg)
    save_json(out/'execution.json',{'completed':True,'attack_freeze':freeze,'predictions_sha256':digest(out/'artifacts/predictions.pt'),'clean_AGA_edits_sha256':digest(out/'artifacts/clean_AGA_edits.pt'),
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'wall_seconds':time.monotonic()-start,'attack_parameter_selection_from_test':False,'gradient_updates':0})


@torch.no_grad()
def verify(cfg):
    out=Path(cfg['output']);root,case,anchor,original,head,pins,cleaned,reference=load_case(cfg);meta=json.loads((out/'execution.json').read_text());freeze=json.loads((out/'attack_freeze.json').read_text())
    assert digest(out/'artifacts/clean_edits.pt')==freeze['clean_edits_sha256']
    assert digest(out/'artifacts/predictions.pt')==meta['predictions_sha256'] and digest(out/'artifacts/clean_AGA_edits.pt')==meta['clean_AGA_edits_sha256']
    for k,v in torch.load(out/'artifacts/clean_edits.pt',map_location='cpu').items():assert torch.equal(v,anchor.state_dict()[k])
    peer,peer_head,peer_a=peer_models(case,cfg);heads={case['target']:head,case['partner']:peer_head}
    clean_a=copy.deepcopy(original);clean_a.load_state_dict(torch.load(cfg['AGA_references'][case['target']]['path'],map_location='cpu'))
    for k,v in torch.load(out/'artifacts/clean_AGA_edits.pt',map_location='cpu').items():clean_a.state_dict()[k].copy_(v)
    raw=torch.load(out/'artifacts/predictions.pt',map_location='cpu');checks=[]
    for row in json.loads((out/'results.json').read_text()):
        stage,task=row['stage'],row['dataset'];x,y=raw[stage+'/'+task]['prediction'],raw[stage+'/'+task]['target'];np.testing.assert_array_equal(y.numpy(),official_labels(task))
        assert int((x==y).sum())==row['correct'] and len(y)==row['n']
        for c in row['per_class']:assert c['n']==int((y==c['class_id']).sum()) and c['correct']==int(((y==c['class_id'])&(x==y)).sum())
        for preset in ['original','released']:
            baseline=reference[preset+'/'+stage+'/'+task]['prediction'];assert paired_counts(baseline.eq(y),x.eq(y))==row['paired_to_'+preset]
        model=cleaned if stage=='standalone' else explicit_merge(anchor,cleaned if stage=='direct' else clean_a,peer if stage=='direct' else peer_a,cfg['coefficient'])
        data=dataset(task,'data',original.val_preprocess);chosen=np.linspace(0,len(y)-1,16,dtype=int);images=torch.stack([data[i][0] for i in chosen]).cuda()
        model.cuda().eval();heads[task].cuda().eval();actual=heads[task](model(images)).argmax(1).cpu();model.cpu();heads[task].cpu();np.testing.assert_array_equal(actual.numpy(),x[chosen].numpy())
        checks.append({'stage':stage,'task':task,'official_labels_and_counts':True,'fresh_model_predictions':16})
    save_json(out/'independent_validation.json',{'status':'passed','source_sha256':digest(__file__),'predictions_sha256':meta['predictions_sha256'],'checks':checks,'exact_matched_control_equal':True,'new_model_prediction_replays':80,'new_defense_weights':0})
    print('PUBLIC_REFILL_VALIDATION_PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for f,pin in cfg['source_sha256'].items():assert digest(f)==pin,f
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (verify if args.verify else run)(cfg)
