"""Author-order eight-expert TA; frozen defenses, three evaluated datasets."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest, evaluate
from covariance_noise import dataset
from phase3_evaluate import apply, row_metrics, setup
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels


def parameter_hash(state):
    h=hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode());h.update(str((value.dtype,tuple(value.shape))).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def combine(anchor,models,order,coefficient):
    from task_vectors import TaskVector
    base=anchor.state_dict();states=[models[t].state_dict() for t in order]
    total=sum(TaskVector(base,s) for s in states)
    model=copy.deepcopy(anchor);state=model.state_dict();maximum=0.
    for key,p in base.items():
        if key not in total.vector:continue
        actual=p+coefficient*total.vector[key]
        expected=states[0][key].numpy()-p.numpy()
        for s in states[1:]:expected=np.add(expected,np.subtract(s[key].numpy(),p.numpy(),dtype=p.numpy().dtype),dtype=p.numpy().dtype)
        expected=np.add(p.numpy(),np.multiply(expected,coefficient,dtype=p.numpy().dtype),dtype=p.numpy().dtype)
        np.testing.assert_array_equal(actual.numpy(),expected)
        maximum=max(maximum,float(np.abs(actual.numpy()-expected).max()));state[key]=actual
    model.load_state_dict(state,strict=True)
    return model,maximum


@torch.no_grad()
def unmask(model, published, coefficient):
    """MergeGuard section7: subtract whole published parameters, not a task vector."""
    state=model.state_dict();reference=published.state_dict()
    for key,value in state.items():
        if not value.is_floating_point():continue
        recovered=value-coefficient*reference[key]
        expected=np.subtract(value.numpy(),np.multiply(reference[key].numpy(),coefficient,dtype=value.numpy().dtype),dtype=value.numpy().dtype)
        np.testing.assert_array_equal(recovered.numpy(),expected)
        state[key]=recovered
    model.load_state_dict(state,strict=True)
    return model


@torch.no_grad()
def run(protocol,full):
    cfg=json.loads(Path(protocol).read_text());assert digest(__file__)==cfg['source_sha256']
    for ref in cfg['inputs'].values():assert digest(ref['path'])==ref['sha256']
    parent=json.loads(Path(cfg['inputs']['active']['path']).read_text());extra=json.loads(Path(cfg['inputs']['additional']['path']).read_text())
    for key in ['tasks','AGA_references','defenses']:parent[key].update(extra[key])
    anchor,originals,heads,aligned,_=setup(parent)
    from utils import load_model_checkpoint
    pins={r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    for task in cfg['task_order']:
        if task in originals:continue
        path='checkpoints/ViT-B-32/'+task+'/finetuned.pt';assert digest(path)==pins['ViT-B-32/'+task+'/finetuned.pt']
        originals[task]=load_model_checkpoint(path,map_location='cpu').eval();originals[task].requires_grad_(False)
    source=Path(cfg['author_TaskVector']['path']);assert digest(source)==cfg['author_TaskVector']['sha256']
    stage='full' if full else 'pilot';root=Path(cfg['output']);out=root/stage;out.mkdir(parents=True,exist_ok=False)
    if full:assert json.loads((root/'pilot/validation.json').read_text())['status']=='passed'
    data={t:dataset(t,'data',originals[t].val_preprocess) for t in cfg['evaluate']}
    assert all(len(d)==parent['tasks'][t]['test_n'] for t,d in data.items())
    indices={t:list(range(len(d))) if full else np.linspace(0,len(d)-1,64,dtype=int).tolist() for t,d in data.items()}
    raw={};rows=[];frozen={};score_cache={};score_sources={};maximum=0.;replays=0;start=time.monotonic()
    forwards=replay_forwards=0
    if 'score_reference' in cfg:
        refs=cfg['score_reference'][stage]
        for ref in refs.values():assert digest(ref['path'])==ref['sha256']
        saved=torch.load(refs['scores']['path'],map_location='cpu',weights_only=False)
        parameters=json.loads(Path(refs['parameters']['path']).read_text())
        audit=json.loads(Path(refs['validation']['path']).read_text())
        assert audit['status']=='passed_independent_numpy_metrics_and_frozen_artifact_hashes'
        assert parameters==audit['model_parameter_hashes']
        assert refs['scores']['sha256']==audit['raw_files_sha256'][refs['scores']['path']]
        for key,item in saved.items():
            parts=key.split('/')
            if parts[1]!='Unprotected':continue
            cache=(parameters['/'.join(parts[:3])],parts[3])
            if cache in score_cache:
                assert all(torch.equal(a,b) for a,b in zip(score_cache[cache],(item['logits'],item['target'])))
            score_cache[cache]=(item['logits'],item['target'])
            score_sources[cache]=dict(**refs['scores'],key=key)
    for case in cfg['cases']:
        target=case['target']
        for scope in cfg.get('scopes',['None','Protected expert']):
            control={}
            for name in ['Unprotected']+case['defenses']:
                aligned_scope=scope=='Protected expert'
                models=dict(originals);own=aligned[target] if aligned_scope else originals[target]
                if name!='Unprotected':
                    ref=parent['defenses'][name];own=apply(own,ref['AGA_edits' if aligned_scope else 'release'],False if aligned_scope else ref.get('full_state',False))
                models[target]=own;model,error=combine(anchor,models,cfg['task_order'],cfg['coefficient'])
                if scope=='Unmask':
                    reference=cfg['no_attack_reference']['parameters'];assert digest(reference['path'])==reference['sha256']
                    baseline=json.loads(Path(reference['path']).read_text())
                    assert parameter_hash(model.state_dict())==baseline['/'.join([target,name,'None'])]
                    model=unmask(model,own,cfg['unmask_coefficient'])
                sha=parameter_hash(model.state_dict())
                key='/'.join([target,name,scope]);frozen[key]=sha
                if full:assert sha==json.loads((root/'pilot/pre_evaluation.json').read_text())[key]
                save_json(out/'pre_evaluation.json',frozen)
                for task,d in data.items():
                    cache=(sha,task);imported=cache in score_cache
                    if imported:scores,y=score_cache[cache];seconds=0.
                    else:
                        sink=[];labels=[];metric=evaluate(model,heads[task],make_loader(d,indices[task]),'cuda',logit_sink=sink.append,prediction_sink=lambda p,y:labels.append(y))
                        scores,y=torch.cat(sink),torch.cat(labels);seconds=metric['evaluation_seconds'];score_cache[cache]=(scores,y)
                        forwards+=len(indices[task])
                    np.testing.assert_array_equal(y.numpy(),official_labels(task)[indices[task]])
                    if name=='Unprotected':control[task]=scores
                    row=dict(target=target,defense=name,attack_scope=scope,dataset=task,role='target' if task==target else 'evaluated partner',
                        participants=8,coefficient=cfg['coefficient'],evaluation_seconds=seconds,reused_identical_model_scores=imported,
                        imported_score_source=score_sources.get(cache),
                        model_parameter_sha256=sha,NumPy_weight_error=error,**row_metrics(scores,y,control[task],scores.shape[1]))
                    raw[key+'/'+task]=dict(logits=scores,target=y,indices=indices[task]);rows.append(row)
                    torch.save(raw,out/'scores.pt');save_json(out/'results.json',rows)
                    for c in row['per_class']:
                        prediction=scores.numpy().argmax(1);mask=y.numpy()==c['class_id'];assert int(((prediction==y.numpy())&mask).sum())==c['correct']
                    selected=np.linspace(0,len(indices[task])-1,16,dtype=int)
                    for block in sorted(set(selected//128)):
                        lo,hi=block*128,min((block+1)*128,len(indices[task]));loc=selected[(selected>=lo)&(selected<hi)];sink=[]
                        evaluate(model,heads[task],make_loader(d,indices[task][lo:hi]),'cuda',logit_sink=sink.append)
                        replay_forwards+=int(hi-lo)
                        diff=float((torch.cat(sink)[loc-lo]-scores[loc]).abs().max());assert diff<=cfg['score_tolerance'];maximum=max(maximum,diff);replays+=len(loc)
                    print('EIGHT_TASK_ROW',stage,target,name,scope,task,row['accuracy_percent'],flush=True)
                del model,own,models
    torch.save(raw,out/'scores.pt');save_json(out/'results.json',rows);save_json(out/'protocol.json',cfg)
    save_json(out/'validation.json',dict(status='passed',rows=len(rows),raw_sha256=digest(out/'scores.pt'),score_replays=replays,
        maximum_context_score_error=maximum,wall_seconds=time.monotonic()-start,source_sha256=digest(__file__),
        new_image_forwards=forwards+replay_forwards,production_image_forwards=forwards,replay_context_image_forwards=replay_forwards,
        reused_evaluation_records=sum(r['reused_identical_model_scores'] for r in rows),new_AGA_calls=0,new_defense_weights=0,expert_training_updates=0,
        scope=cfg['experiment']+'; only three evaluated datasets, no eight-task average or complete paper reproduction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/phase3_eight_task.json');p.add_argument('--full',action='store_true');a=p.parse_args();run(a.protocol,a.full)
