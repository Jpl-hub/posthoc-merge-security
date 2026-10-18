"""Frozen knowledge-source controls, not defense construction or attack tuning."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest, evaluate
from covariance_noise import dataset
from phase3_eight_task import combine, parameter_hash
from phase3_evaluate import setup, row_metrics
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_prodistill import verify_metrics


@torch.no_grad()
def run(protocol, full, leave_out):
    cfg=json.loads(Path(protocol).read_text())
    for path,sha in cfg['source_sha256'].items():assert digest(path)==sha,path
    ref=cfg['eight_task_protocol'];assert digest(ref['path'])==ref['sha256']
    eight=json.loads(Path(ref['path']).read_text())
    parents=[]
    for r in eight['inputs'].values():
        assert digest(r['path'])==r['sha256'];parents.append(json.loads(Path(r['path']).read_text()))
    parent=parents[0]
    for k in ['tasks','AGA_references','defenses']:parent[k].update(parents[1][k])
    anchor,originals,heads,_,_=setup(parent)
    models={'Public base':anchor}|{t:originals[t] for t in cfg['tasks']}
    if leave_out:
        from utils import load_model_checkpoint
        pins={r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
        for t in eight['task_order']:
            if t in originals:continue
            path='checkpoints/ViT-B-32/'+t+'/finetuned.pt';assert digest(path)==pins['ViT-B-32/'+t+'/finetuned.pt']
            originals[t]=load_model_checkpoint(path,map_location='cpu').eval().requires_grad_(False)
        models={}
        for t in cfg['tasks']:
            order=[v for v in eight['task_order'] if v!=t]
            models[t],_=combine(anchor,originals,order,eight['coefficient'])
    start=time.monotonic();group='leave_out' if leave_out else 'single';stage='full' if full else 'pilot'
    root=Path(cfg['output'])/group;out=root/stage;out.mkdir(parents=True,exist_ok=False)
    if full:assert json.loads((root/'pilot/validation.json').read_text())['status']=='passed'
    frozen={name:parameter_hash(model.state_dict()) for name,model in models.items()}
    if full:assert frozen==json.loads((root/'pilot/pre_evaluation.json').read_text())
    save_json(out/'pre_evaluation.json',frozen);save_json(out/'protocol.json',cfg)
    baselines={};source_refs={}
    if leave_out:
        p=Path(eight['output'])/'full';validation=json.loads((p/'validation.json').read_text());assert validation['status']=='passed'
        assert digest(p/'scores.pt')==validation['raw_sha256'];old=torch.load(p/'scores.pt',map_location='cpu',weights_only=False)
        baselines={t:old['MNIST/Unprotected/None/'+t] for t in cfg['tasks']}
        source_refs={'eight_expert_scores_sha256':validation['raw_sha256']}
    else:
        for t,r in cfg['native_scores'].items():
            assert digest(r['path'])==r['sha256'];old=torch.load(r['path'],map_location='cpu',weights_only=False)
            baselines[t]=old[r['key']];source_refs[t]=r
    data={t:dataset(t,'data',originals[t].val_preprocess) for t in cfg['tasks']}
    raw={};rows=[];maximum=logp_error=0.;replays=0
    for name,model in models.items():
        for task,d in data.items():
            ids=list(range(len(d))) if full else np.linspace(0,len(d)-1,64,dtype=int).tolist()
            reference=baselines[task]['logits'][ids];imported=not leave_out and name==task
            if imported:
                scores=reference;labels=baselines[task]['target'][ids];seconds=0.
            else:
                sink=[];ys=[]
                m=evaluate(model,heads[task],make_loader(d,ids),'cuda',logit_sink=sink.append,prediction_sink=lambda p,y:ys.append(y))
                scores,labels=torch.cat(sink),torch.cat(ys);seconds=m['evaluation_seconds']
            np.testing.assert_array_equal(labels.numpy(),official_labels(task)[ids])
            row=dict(model=name,dataset=task,group=group,imported_native_scores=imported,evaluation_seconds=seconds,
                model_parameter_sha256=frozen[name],comparison='Eight experts TA0.2' if leave_out else 'Original dataset expert',
                **row_metrics(scores,labels,reference,scores.shape[1]))
            logp_error=max(logp_error,verify_metrics(scores,labels,reference,row));rows.append(row)
            raw[name+'/'+task]=dict(logits=scores,target=labels,indices=ids)
            torch.save(raw,out/'scores.pt');save_json(out/'results.json',rows)
            selected=np.linspace(0,len(ids)-1,16,dtype=int)
            positions=np.array(ids)[selected] if imported else selected
            for block in sorted(set(positions//128)):
                lo,hi=int(block)*128,min((int(block)+1)*128,len(d) if imported else len(ids));sink=[]
                context=list(range(lo,hi)) if imported else ids[lo:hi]
                evaluate(model,heads[task],make_loader(d,context),'cuda',logit_sink=sink.append)
                mask=positions//128==block;actual=torch.cat(sink)[positions[mask]-lo]
                error=float((actual-scores[selected[mask]]).abs().max());maximum=max(maximum,error)
                assert error<=cfg['score_tolerance'];replays+=int(mask.sum())
            print('KNOWLEDGE_CONTROL',group,stage,name,task,row['accuracy_percent'],flush=True)
    save_json(out/'validation.json',dict(status='passed',rows=len(rows),score_replays=replays,maximum_context_score_error=maximum,
        maximum_independent_logp_error=logp_error,raw_sha256=digest(out/'scores.pt'),source_references=source_refs,
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,new_gradients=0,new_defense_weights=0,
        scope='Historically exposed tests; frozen source controls, no tuning. Target contribution is not unique-knowledge proof.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/phase3_controls.json')
    p.add_argument('--full',action='store_true');p.add_argument('--leave-out',action='store_true')
    a=p.parse_args();run(a.protocol,a.full,a.leave_out)
