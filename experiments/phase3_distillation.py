"""Fixed-budget hard-label recovery: train an attacker-owned public-base copy.

Native adaptation of PaRaMS section6/H.2, not its full official experiment.
Frozen experts, no defense changes, no test-based selection or early stopping.
"""

import argparse
import ast
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from clip_benchmark import author_recover, digest, evaluate, merge
from covariance_noise import dataset
from mechanisms import load_author
from phase3_eight_task import parameter_hash
from phase3_evaluate import apply, row_metrics, setup
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_prodistill import verify_metrics


def train(base, head, images, labels, cfg, pilot=False):
    """Same public initialization and fixed training rule for each label group."""
    torch.manual_seed(cfg['training_seed']); torch.cuda.manual_seed_all(cfg['training_seed'])
    student=copy.deepcopy(base).cuda().train().requires_grad_(False)
    student.model.visual.requires_grad_(True); head=head.cuda().eval().requires_grad_(False)
    params=[p for p in student.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    path=Path(cfg['schedule']['path']); assert digest(path)==cfg['schedule']['sha256']
    names={'assign_learning_rate','_warmup_lr','cosine_lr'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    steps=cfg['epochs']*(len(images)//cfg['batch_size']); scheduler=env['cosine_lr'](opt,cfg['lr'],cfg['warmup_steps'],steps)
    rng=torch.Generator().manual_seed(cfg['training_seed']); history=[];step=0;start=time.monotonic()
    head_before=parameter_hash({k:v.cpu() for k,v in head.state_dict().items()});torch.cuda.reset_peak_memory_stats()
    for epoch in range(1 if pilot else cfg['epochs']):
        order=torch.randperm(len(images),generator=rng); losses=[]
        for ids in ([order[:8]] if pilot else order.split(cfg['batch_size'])):
            scheduler(step);opt.zero_grad(set_to_none=True)
            logits=head(student(images[ids].cuda())); loss=torch.nn.functional.cross_entropy(logits,labels[ids].cuda())
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0);assert torch.isfinite(norm) and norm>0
            assert all(p.grad is None for p in head.parameters())
            opt.step();losses.append(float(loss.detach()));step+=1
        history.append({'epoch':epoch+1,'mean_pseudo_label_loss':float(np.mean(losses))})
        print('KD_EPOCH',epoch+1,history[-1]['mean_pseudo_label_loss'],flush=True)
    assert head_before==parameter_hash({k:v.cpu() for k,v in head.state_dict().items()})
    torch.cuda.synchronize(); seconds=time.monotonic()-start; peak=torch.cuda.max_memory_allocated()
    student.cpu().eval().requires_grad_(False);head.cpu();del opt
    assert step==(1 if pilot else steps)
    return student,dict(completed=True,backward_steps=step,training_seconds=seconds,peak_gpu_bytes=peak,
        trainable_student_parameters=sum(p.numel() for p in params),history=history,
        expert_updates=0,defense_updates=0,head_updates=0,selection_from_test=False)


def run(protocol):
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    save_json(out/'protocol.json',cfg);assert digest(__file__)==cfg['source_sha256'][__file__ if __file__ in cfg['source_sha256'] else 'experiments/phase3_distillation.py']
    for p,sha in cfg['source_sha256'].items():assert digest(p)==sha,p
    parent=json.loads(Path(cfg['frozen_inputs']['path']).read_text());assert digest(cfg['frozen_inputs']['path'])==cfg['frozen_inputs']['sha256']
    base,originals,heads,aligned,_=setup(parent);target,partner=cfg['target'],cfg['partner']
    original_hash={t:parameter_hash(m.state_dict()) for t,m in originals.items()};base_hash=parameter_hash(base.state_dict())
    torch.use_deterministic_algorithms(True)
    train_data=dataset(target,'data',originals[target].val_preprocess,training=True)
    ids=torch.randperm(len(train_data),generator=torch.Generator().manual_seed(cfg['data_seed']))[:cfg['unlabeled_n']].tolist()
    save_json(out/'training_indices.json',ids)
    # Dataset labels are discarded here. Only protected-teacher argmax labels train the student.
    images=torch.stack([train_data[i][0] for i in ids]);image_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
    teachers={'Unprotected':originals[target],cfg['defense']:apply(originals[target],parent['defenses'][cfg['defense']]['release'])}
    teacher_hash={k:parameter_hash(m.state_dict()) for k,m in teachers.items()};labels={};teacher_scores={};query_start=time.monotonic()
    with torch.no_grad():
        for name,teacher in teachers.items():
            teacher.cuda();heads[target].cuda();scores=[]
            for block in images.split(cfg['batch_size']):scores.append(heads[target](teacher(block.cuda())).cpu())
            teacher_scores[name]=torch.cat(scores);labels[name]=teacher_scores[name].argmax(1);teacher.cpu();heads[target].cpu()
    torch.save(dict(scores=teacher_scores,labels=labels,indices=ids),out/'teacher_queries.pt')
    save_json(out/'queries.json',dict(queries_per_teacher=len(ids),ground_truth_training_labels_used=False,
        teacher_parameter_sha256=teacher_hash,query_seconds=time.monotonic()-query_start,
        image_tensor_sha256=image_hash,indices_sha256=digest(out/'training_indices.json'),
        raw_sha256=digest(out/'teacher_queries.pt'),preprocess=repr(originals[target].val_preprocess),
        teacher_prediction_disagreements=int((labels['Unprotected']!=labels[cfg['defense']]).sum())))
    _,pilot=train(base,heads[target],images,labels['Unprotected'],cfg,pilot=True)
    save_json(out/'pilot.json',pilot);torch.cuda.empty_cache()
    groups={};students={};training={}
    for name,y in labels.items():
        key=next((k for k in groups if torch.equal(y,labels[k])),None)
        if key is not None:groups[key].append(name);students[name]=students[key];continue
        assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline_utc']), 'No new training group after deadline'
        model,report=train(base,heads[target],images,y,cfg);groups[name]=[name];students[name]=model
        checkpoint=out/(name+'_student.pt');torch.save(model.state_dict(),checkpoint)
        report.update(model_sha256=digest(checkpoint),parameter_sha256=parameter_hash(model.state_dict()),
            pseudo_labels_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),source_sha256=digest(__file__))
        training[name]=report;save_json(out/'training.json',training);torch.cuda.empty_cache()
    save_json(out/'freeze.json',dict(groups=groups,training=training,all_students_frozen_before_test=True,
        base_parameter_sha256=base_hash,original_parameter_sha256=original_hash,source_sha256=digest(__file__),
        protocol_sha256=digest(protocol),teacher_queries_sha256=digest(out/'teacher_queries.pt')))
    del images;data={t:dataset(t,'data',originals[t].val_preprocess) for t in [target,partner]}
    raw={};rows=[];replays=0;maximum=0.;logp_error=0.;author,_=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'),include_wrappers=True)
    for group,names in groups.items():
        student=students[group];at=time.monotonic();recovered=author_recover(student,base,author);aga_seconds=time.monotonic()-at
        models=[('Standalone',student,[target]),('TA',merge(base,student,originals[partner],.5),[target,partner]),
            ('AGA-target-only',merge(base,recovered,originals[partner],.5),[target,partner]),
            ('AGA-both',merge(base,recovered,aligned[partner],.5),[target,partner])]
        for mode,model,tasks in models:
            for task in tasks:
                sink=[];ys=[];indices=list(range(len(data[task])))
                metric=evaluate(model,heads[task],make_loader(data[task]),'cuda',logit_sink=sink.append,prediction_sink=lambda p,y:ys.append(y))
                scores,y=torch.cat(sink),torch.cat(ys);np.testing.assert_array_equal(y.numpy(),official_labels(task))
                key=group+'/'+mode+'/'+task;baseline=scores if group=='Unprotected' else raw['Unprotected/'+mode+'/'+task]['logits']
                row=dict(teacher_group=group,teacher_names=names,mode=mode,dataset=task,
                    **(metric | row_metrics(scores,y,baseline,scores.shape[1])),AGA_seconds=aga_seconds)
                logp_error=max(logp_error,verify_metrics(scores,y,baseline,row));s=scores.double().numpy();yy=y.numpy()
                ce=logsumexp(s,axis=1)-s[np.arange(len(s)),yy]
                for c in row['per_class']:
                    mask=yy==c['class_id'];before=baseline.numpy().argmax(1);after=s.argmax(1)
                    c.update(mean_cross_entropy=float(ce[mask].mean()),lost=int((mask&(before==yy)&(after!=yy)).sum()),
                        gained=int((mask&(before!=yy)&(after==yy)).sum()))
                raw[key]=dict(logits=scores,target=y,indices=indices);rows.append(row)
                torch.save(raw,out/'scores.pt');save_json(out/'results.json',rows)
                positions=np.linspace(0,len(y)-1,16,dtype=int)
                for block in sorted(set(positions//128)):
                    lo=int(block)*128;hi=min(lo+128,len(y));sinks=[]
                    evaluate(model,heads[task],make_loader(data[task],list(range(lo,hi))),'cuda',logit_sink=sinks.append)
                    selected=positions[positions//128==block];error=float((torch.cat(sinks)[selected-lo]-scores[selected]).abs().max())
                    maximum=max(maximum,error);assert error<=cfg['score_tolerance'];replays+=len(selected)
                print('KD_EVAL',group,mode,task,row['accuracy_percent'],flush=True)
    assert base_hash==parameter_hash(base.state_dict())
    assert original_hash=={t:parameter_hash(m.state_dict()) for t,m in originals.items()}
    assert teacher_hash=={t:parameter_hash(m.state_dict()) for t,m in teachers.items()}
    save_json(out/'validation.json',dict(status='passed',rows=len(rows),score_replays=replays,maximum_context_score_error=maximum,
        maximum_independent_logp_error=logp_error,raw_sha256=digest(out/'scores.pt'),expert_updates=0,defense_updates=0,
        source_sha256=digest(__file__),scope='Fixed2048 unlabeled training images, public-base student, hard teacher labels. Stronger data/training permissions; not original author full-data KD or data-free AGA. Identical finite label vectors reuse the exact same attacker training problem, not independent seeds. No test-based selection.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/phase3_distillation.json');a=p.parse_args();run(a.protocol)
