"""Official ProDistill coefficient kernel, explicit two-expert CLIP adapter.

Unlabeled training calibration; no expert/defense updates or test selection.
"""

import argparse
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from clip_benchmark import digest, evaluate
from covariance_noise import dataset
from phase3_evaluate import apply, row_metrics, setup
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels


def author(cfg):
    env=dict(torch=torch,nn=nn,np=np,copy=copy)
    for name,ref in cfg['author_files'].items():
        assert digest(ref['path'])==ref['sha256']
        tree=ast.parse(Path(ref['path']).read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and
            (name=='merge_models' or getattr(n,'name','')=='create_preresblock_activation_extractor')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),ref['path'],'exec'),env)
    return env


def check(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);env=author(cfg);torch.manual_seed(0)
    base=nn.Linear(3,2).double();left=copy.deepcopy(base);right=copy.deepcopy(base)
    with torch.no_grad():left.weight.add_(.1);right.bias.sub_(.2)
    wrapper=env['MergedModel'](base,[left,right],'elementwise');x=torch.arange(15,dtype=torch.float64).reshape(5,3)/10
    expected=F.linear(x,base.weight+.3*(left.weight-base.weight)+.3*(right.weight-base.weight),base.bias+.3*(left.bias-base.bias)+.3*(right.bias-base.bias))
    actual=wrapper(x);np.testing.assert_allclose(actual.detach().numpy(),expected.numpy(),atol=1e-12,rtol=1e-12)
    actual.square().sum().backward();alpha=wrapper.alphas[0][0];gradient=float(alpha.grad[0,0]);values=[]
    with torch.no_grad():
        for sign in [-1,1]:alpha[0,0]=.3+sign*1e-6;values.append(float(wrapper(x).square().sum()))
    error=abs((values[1]-values[0])/2e-6-gradient);assert error<1e-7
    save_json(out/'kernel_validation.json',dict(status='passed_official_coefficient_formula_and_gradient',gradient_error=error,
        source_sha256=digest(__file__),author_files=cfg['author_files'],images=0,expert_updates=0))


@torch.no_grad()
def initial(anchor, experts):
    result=copy.deepcopy(anchor);state=result.state_dict();base=anchor.state_dict()
    for expert in experts:
        for key,_ in anchor.named_parameters():state[key].add_(.3*(expert.state_dict()[key]-base[key]))
    return result


def train(cfg,case,out,anchor,experts,own,peer,env):
    start=time.monotonic();torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();target='MNIST'
    data={t:dataset(t,'data',experts[t].val_preprocess,True) for t in ['MNIST','SVHN']}
    ids={t:torch.randperm(len(data[t]),generator=torch.Generator().manual_seed(0))[-64:].tolist() for t in data}
    save_json(out/'calibration_indices.json',ids)
    images=torch.stack([data[t][i][0] for t in ids for i in ids[t]]).cuda();del data
    source=torch.tensor([0]*64+[1]*64,device='cuda');merged=initial(anchor,[own,peer])
    def prefix(model):
        model.cuda()
        with torch.no_grad():value=env['create_preresblock_activation_extractor'](model)(images).detach().cpu()
        model.cpu();return value
    xm=prefix(merged);a,b=prefix(own),prefix(peer);xt=torch.where(source.cpu()[None,:,None]==0,a,b);del images,a,b
    log=[];steps=0
    for i in range(12):
        layers=[copy.deepcopy(m.model.visual.transformer.resblocks[i]).cuda().eval() for m in [own,peer]]
        base=copy.deepcopy(anchor.model.visual.transformer.resblocks[i]).cuda()
        wrapper=env['MergedModel'](base,layers,'elementwise');wrapper.train()
        params=[p for p in wrapper.parameters() if p.requires_grad];optimizer=torch.optim.Adam(params,lr=.01)
        x,y=xm.cuda(),xt.cuda();losses=[]
        for epoch in range(100):
            order=torch.randperm(128,device='cuda');xx,yy=x[:,order],y[:,order];s=source[order]
            optimizer.zero_grad();student=wrapper.get_merged_model()(xx).permute(1,0,2).reshape(128,-1);loss=0
            for j,layer in enumerate(layers):
                with torch.no_grad():teacher=layer(yy).permute(1,0,2).reshape(128,-1)
                loss+= (F.mse_loss(student,teacher,reduction='none').sum(1)*s.eq(j)).sum()
            assert torch.isfinite(loss);loss.backward();optimizer.step();steps+=1
            if epoch in [0,99]:losses.append(float(loss.detach()))
        maximum=0.;coefficient_hash=hashlib.sha256();state=anchor.model.visual.transformer.resblocks[i].state_dict()
        with torch.no_grad():
            fun=wrapper.get_merged_model()
            for k,name in enumerate(wrapper.names):
                value=fun
                for part in name.split('.'):value=getattr(value,part)
                p0=base.state_dict()[name].cpu().numpy();expected=np.zeros_like(p0)
                for j,layer in enumerate(layers):
                    alpha=wrapper.alphas[j][k].detach().cpu().numpy();coefficient_hash.update(alpha.tobytes())
                    expected+=alpha*(layer.state_dict()[name].cpu().numpy()-p0)
                expected+=p0;actual=value.detach().cpu();np.testing.assert_allclose(actual.numpy(),expected,atol=2e-5,rtol=2e-5)
                maximum=max(maximum,float(np.abs(actual.numpy()-expected).max()));state[name]=actual
            merged.model.visual.transformer.resblocks[i].load_state_dict(state,strict=True)
            xm=fun(x).cpu();xt=torch.where(source[None,:,None]==0,layers[0](y),layers[1](y)).cpu()
        log.append(dict(layer=i,first_last_loss=losses,trainable_coefficients=sum(p.numel() for p in params),coefficient_sha256=coefficient_hash.hexdigest(),NumPy_weight_error=maximum))
        print('PRODISTILL_LAYER',case,i,losses,flush=True)
        del wrapper,optimizer,layers,base,fun,x,y,params
    torch.save(merged.state_dict(),out/'model.pt')
    save_json(out/'training.json',dict(completed=True,wall_seconds=time.monotonic()-start,backward_steps=steps,
        gradients='Only attacker elementwise merging coefficients; experts and defense checkpoints frozen',layers=log,
        model_sha256=digest(out/'model.pt'),indices_sha256=digest(out/'calibration_indices.json'),source_sha256=digest(__file__),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),expert_updates=0,defense_updates=0,selection_from_test=False))
    return merged


@torch.no_grad()
def assess(cfg,case,out,model,head_models,experts,full):
    stage='full' if full else 'pilot';folder=out/stage;folder.mkdir(exist_ok=False);raw={};rows=[];maximum=0.;replays=0
    if full:assert json.loads((out/'pilot/validation.json').read_text())['status']=='passed'
    start=time.monotonic();name=case['defense'];control=Path(cfg['output'])/f'{case["control"]:02d}'/stage/'scores.pt'
    names=[name,'Phase1_MNIST'] if name=='Unprotected' else [name]
    for defense in names:
        current=copy.deepcopy(model)
        if defense=='Phase1_MNIST':current.model.visual.ln_post.bias.copy_(torch.load(out/'phase1_tail.pt',map_location='cpu',weights_only=False))
        for task in ['MNIST','SVHN']:
            data=dataset(task,'data',experts[task].val_preprocess);ids=list(range(len(data))) if full else np.linspace(0,len(data)-1,64,dtype=int).tolist()
            scores=[];labels=[];metric=evaluate(current,head_models[task],make_loader(data,ids),'cuda',logit_sink=scores.append,prediction_sink=lambda p,y:labels.append(y))
            score=torch.cat(scores);label=torch.cat(labels);np.testing.assert_array_equal(label.numpy(),official_labels(task)[ids]);assert torch.isfinite(score).all()
            before=score if defense=='Unprotected' else raw['Unprotected/'+task]['logits'] if name=='Unprotected' else torch.load(control,map_location='cpu',weights_only=False)['Unprotected/'+task]['logits']
            row=dict(defense=defense,merger='ProDistill (two-expert adapter)',attack_scope=case['scope'],target='MNIST',partner='SVHN',dataset=task,role='target' if task=='MNIST' else 'partner',evaluation_seconds=metric['evaluation_seconds'],**row_metrics(score,label,before,10))
            loc=np.linspace(0,len(ids)-1,16,dtype=int)
            for block in sorted(set(loc//128)):
                lo,hi=block*128,min((block+1)*128,len(ids));wanted=loc[(loc>=lo)&(loc<hi)];values=[]
                evaluate(current,head_models[task],make_loader(data,ids[lo:hi]),'cuda',logit_sink=values.append);actual=torch.cat(values)[wanted-lo]
                error=float((actual-score[wanted]).abs().max());assert error<=.0005;maximum=max(maximum,error);replays+=len(wanted)
            y=label.numpy();p=score.numpy().argmax(1);b=before.numpy().argmax(1)
            assert int((p==y).sum())==row['correct'] and int(((b==y)&(p!=y)).sum())==row['paired']['lost'] and int(((b!=y)&(p==y)).sum())==row['paired']['gained']
            for item in row['per_class']:assert int(((p==y)&(y==item['class_id'])).sum())==item['correct']
            raw[defense+'/'+task]=dict(logits=score,target=label,indices=ids);rows.append(row);print('PRODISTILL_ROW',case,stage,defense,task,row['accuracy_percent'],flush=True)
    torch.save(raw,folder/'scores.pt');save_json(folder/'results.json',rows);save_json(folder/'validation.json',dict(status='passed',maximum_context_score_error=maximum,score_replays=replays,NumPy_labels_pairs_classes=True,model_sha256=digest(out/'model.pt'),raw_sha256=digest(folder/'scores.pt'),wall_seconds=time.monotonic()-start,scope='Official coefficient kernel and explicit native two-task driver; not full eight-task paper replication.'))


def run(protocol,index,full):
    cfg=json.loads(Path(protocol).read_text());assert digest(cfg['frozen_inputs']['path'])==cfg['frozen_inputs']['sha256'];assert digest(__file__)==cfg['source_sha256'];case=cfg['cases'][index]
    parent=json.loads(Path(cfg['frozen_inputs']['path']).read_text());anchor,experts,heads,aligned,_=setup(parent);env=author(cfg)
    own=experts['MNIST'] if case['scope']=='None' else aligned['MNIST'];peer=aligned['SVHN'] if case['scope']=='All experts' else experts['SVHN']
    name=case['defense'];out=Path(cfg['output'])/f'{index:02d}'
    if name!='Unprotected':own=apply(own,parent['defenses'][name]['release' if case['scope']=='None' else 'AGA_edits'])
    assert json.loads((Path(cfg['output'])/'kernel_validation.json').read_text())['status']=='passed_official_coefficient_formula_and_gradient'
    if not full:
        out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg);model=train(cfg,case,out,anchor,experts,own,peer,env)
        if name=='Unprotected':
            phase1=apply(own,parent['defenses']['Phase1_MNIST']['release' if case['scope']=='None' else 'AGA_edits'])
            key='model.visual.ln_post.bias';assert all(torch.equal(v,phase1.state_dict()[k]) for k,v in own.state_dict().items() if k!=key)
            p=anchor.state_dict()[key];beta=(p+.3*(phase1.state_dict()[key]-p))+.3*(peer.state_dict()[key]-p);torch.save(beta,out/'phase1_tail.pt')
        save_json(out/'pre_evaluation.json',dict(model_sha256=digest(out/'model.pt'),phase1_tail_sha256=digest(out/'phase1_tail.pt') if name=='Unprotected' else None,test_images_read=0,source_sha256=digest(__file__)))
    else:
        meta=json.loads((out/'training.json').read_text());assert digest(out/'model.pt')==meta['model_sha256'];model=copy.deepcopy(anchor);model.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=False),strict=True)
    freeze=json.loads((out/'pre_evaluation.json').read_text());assert digest(out/'model.pt')==freeze['model_sha256']
    if name=='Unprotected':assert digest(out/'phase1_tail.pt')==freeze['phase1_tail_sha256']
    assess(cfg,case,out,model,heads,experts,full)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='configs/phase3_prodistill.json');p.add_argument('--case',type=int,default=0);p.add_argument('--full',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    check(json.loads(Path(a.protocol).read_text())) if a.check else run(a.protocol,a.case,a.full)
