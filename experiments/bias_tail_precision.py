"""Frozen bias recovery with an FP32 prefix and independently checked FP64 readout."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bias_canonicalization import inputs
from clip_benchmark import digest, merge
from covariance_noise import dataset
from phase3_evaluate import row_metrics
from subspace_probe import make_loader, save_json
from task_probe_distance import parameters
from task_probe_distance import checked_models
from verify_covariance_noise import official_labels
from verify_levelset_audits import raw_scores


@torch.no_grad()
def capture(model, head, data, ids):
    hidden, actual = [], []
    hook=model.model.visual.ln_post.register_forward_pre_hook(lambda _m,a:hidden.append(a[0].detach().cpu()))
    model.cuda();head.cuda()
    try:
        for x,_ in make_loader(data,ids,128):actual.append(head(model(x.cuda())).cpu())
    finally:
        hook.remove();model.cpu();head.cpu()
    return torch.cat(hidden),torch.cat(actual)


def readout(h,p,beta):
    h=h.double();v=F.layer_norm(h,(h.shape[1],),p['gamma'],beta.double(),p['eps'])@p['projection']
    return (v/v.norm(dim=1,keepdim=True))@p['head'].T


@torch.no_grad()
def run(protocol,pilot,verify):
    start=time.monotonic();cfg,frozen,checked=inputs(protocol);root=Path(cfg['output']);out=root/('pilot' if pilot else 'full')
    ref=cfg['frozen_recovery_reference'];assert digest(ref['path'])==ref['sha256'];recovery=json.loads(Path(ref['path']).read_text())
    assert recovery['constructed_before_images']
    if not verify:
        if not pilot:assert json.loads((root/'pilot/validation.json').read_text())['status']=='passed'
        out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    else:assert json.loads((out/'protocol.json').read_text())==cfg
    def saved_beta(record):
        assert digest(record['path'])==record['sha256'];return torch.load(record['path'],map_location='cpu',weights_only=False)['model.visual.ln_post.bias']
    anchor=checked('zeroshot.pt');key='model.visual.ln_post.bias';raw={};rows=[];controls=[];replays=0;score_replays=0;maxerror=0.
    if verify:
        meta=json.loads((out/'execution.json').read_text());assert digest(out/'raw.pt')==meta['raw_sha256']
        raw=torch.load(out/'raw.pt',map_location='cpu',weights_only=False);rows=json.loads((out/'results.json').read_text())
    for ci,case in enumerate(cfg['cases']):
        target,partner=case['target'],case['partner'];own,peer=[checked(t+'/finetuned.pt') for t in [target,partner]]
        ref=frozen['defenses']['Phase1_'+target]['release'];release_beta=saved_beta(ref)
        assert set(torch.load(ref['path'],map_location='cpu',weights_only=False))=={key}
        ownb=own.state_dict()[key];peerb=peer.state_dict()[key];baseb=anchor.state_dict()[key]
        clean=saved_beta(recovery['records'][target+'/Unprotected']);protected_clean=saved_beta(recovery['records'][target+'/Phase1_'+target]);peer_clean=saved_beta(recovery['records'][partner+'/Unprotected'])
        combine=lambda left,right:baseb+.5*((left-baseb)+(right-baseb))
        groups=[('Standalone',own,[target],[('Unprotected','None',ownb),('Phase1_'+target,'None',release_beta),('Unprotected','Protected expert',clean),('Phase1_'+target,'Protected expert',protected_clean)]),
            ('TA',merge(anchor,own,peer,.5),[target,partner],[('Unprotected','None',combine(ownb,peerb)),('Phase1_'+target,'None',combine(release_beta,peerb)),('Unprotected','Protected expert',combine(clean,peerb)),('Phase1_'+target,'Protected expert',combine(protected_clean,peerb)),('Unprotected','All experts',combine(clean,peer_clean)),('Phase1_'+target,'All experts',combine(protected_clean,peer_clean))])]
        for method,model,tasks,variants in groups:
            for task in tasks:
                head=checked('head_'+task+'.pt');data=dataset(task,'data',own.val_preprocess);ids=np.linspace(0,len(data)-1,cfg['pilot_n'],dtype=int).tolist() if pilot else list(range(len(data)))
                prefix=f'{ci}/{method}/{task}';p=parameters(model,head);pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in p.items()}
                if verify:
                    record=raw[prefix];h=record['h'];labels=official_labels(task)[ids];np.testing.assert_array_equal(record['labels'].numpy(),labels)
                    selected=np.linspace(0,len(ids)-1,cfg['replay_n'],dtype=int)
                    for pos in selected:
                        lo=int(pos)//128*128;hi=min(lo+128,len(ids));hh,_=capture(model,head,data,ids[lo:hi]);err=float((hh[int(pos)-lo]-h[int(pos)]).abs().max());assert err<=cfg['score_tolerance'];replays+=1;maxerror=max(maxerror,err)
                        for defense,scope,beta in variants:
                            actual=readout(hh,p,beta)[int(pos)-lo];saved=record['scores'][defense+'/'+scope][int(pos)];err=float((actual-saved).abs().max());assert err<=cfg['score_tolerance'];score_replays+=1;maxerror=max(maxerror,err)
                else:
                    h,actual=capture(model,head,data,ids);labels=torch.tensor(official_labels(task)[ids]);record=dict(h=h,actual_FP32=actual,labels=labels,indices=ids,scores={});raw[prefix]=record
                values={}
                for defense,scope,beta in variants:
                    name=defense+'/'+scope;score=readout(h,p,beta);expected=raw_scores(h.double().numpy(),pn|{'beta':beta.double().numpy()});err=float(abs(score.numpy()-expected).max());assert err<1e-8;maxerror=max(maxerror,err);values[name]=score
                    if verify:
                        np.testing.assert_allclose(score.numpy(),record['scores'][name].numpy(),rtol=0,atol=1e-8)
                        row=next(r for r in rows if r['key']==prefix+'/'+name);assert int((expected.argmax(1)==labels).sum())==row['correct']
                        baseline=record['scores']['Unprotected/'+scope].numpy().argmax(1);pred=expected.argmax(1)
                        assert int((baseline!=pred).sum())==row['changed_predictions']
                        assert int(((baseline==labels)&(pred!=labels)).sum())==row['paired']['lost'] and int(((baseline!=labels)&(pred==labels)).sum())==row['paired']['gained']
                        for c in row['per_class']:assert int((labels==c['class_id']).sum())==c['n'] and int(((pred==labels)&(labels==c['class_id'])).sum())==c['correct']
                    else:
                        control=values['Unprotected/'+scope];record['scores'][name]=score
                        rows.append(dict(key=prefix+'/'+name,target=target,partner=partner,dataset=task,role='target' if task==target else 'partner',defense=defense,merger=method,attack_scope=scope,numerical_protocol='FP32 prefix; FP64 final LN/projection/normalized head',**row_metrics(score,labels,control,frozen['tasks'][task]['classes'])))
                if method=='Standalone':
                    for defense in ['Unprotected','Phase1_'+target]:
                        delta=float((values[defense+'/None']-values[defense+'/Protected expert']).abs().max());controls.append(dict(case=ci,defense=defense,max_score_difference=delta));assert delta<=cfg['score_tolerance']
                if not verify:save_json(out/'results.json',rows);torch.save(raw,out/'raw.pt')
        del groups,own,peer
    assert len(rows)==16*len(cfg['cases'])
    if verify:
        save_json(out/'validation.json',dict(status='passed',NumPy_complete_readout=True,official_labels_classes_and_pairs=True,prefix_context_replays=replays,selected_score_replays=score_replays,maximum_error=maxerror,source_sha256=digest(__file__),raw_sha256=digest(out/'raw.pt'),controls=controls))
    else:
        save_json(out/'execution.json',dict(completed=True,rows=len(rows),wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),prefix_dtype='float32',readout_dtype='float64',same_frozen_recovery_edits=True,defense_updates=0,gradients=0,raw_sha256=digest(out/'raw.pt'),source_sha256=digest(__file__),controls=controls,scope='Separate numerical protocol; does not retroactively pass the failed nativeFP32 protocol.'))


@torch.no_grad()
def certificate():
    """Read-only sufficient bounds for arbitrary prefixes and fixed native tails."""
    start=time.monotonic();torch.set_num_threads(4);protocol=Path('configs/phase3.json')
    assert digest(protocol)=='4e40f5286e9cbaa7b39b19214742c6cddaf8946af69369ef163ff596194ef7c7'
    cfg=json.loads(protocol.read_text());keys=['model.visual.ln_post.weight','model.visual.ln_post.bias','model.visual.proj'];originals={};protected={};heads={};hashes={}
    for task in ['MNIST','DTD','SVHN']:
        anchor,own,head,pins=checked_models(task);hashes[task]=pins;assert head.normalize and torch.count_nonzero(head.bias)==0
        base={k:anchor.state_dict()[k].clone() for k in keys};originals[task]={k:own.state_dict()[k].clone() for k in keys};heads[task]=head.weight.detach().double()
        ref=cfg['defenses']['Phase1_'+task]['release'];assert digest(ref['path'])==ref['sha256'];edit=torch.load(ref['path'],map_location='cpu',weights_only=False)
        protected[task]={k:edit.get(k,originals[task][k]).clone() for k in keys}
        del anchor,own,head,edit
    rows=[];maximum=0.
    for case in cfg['cases']:
        target,peer=case['target'],case['partner']
        for name,own in [('Unprotected',originals[target]),('Phase1_'+target,protected[target])]:
            for coefficient in [.5,.3]:
                if coefficient==.5:
                    tail={k:base[k]+.5*((own[k]-base[k])+(originals[peer][k]-base[k])) for k in keys}
                else:
                    tail={k:(base[k]+.3*(own[k]-base[k]))+.3*(originals[peer][k]-base[k]) for k in keys}
                gamma,beta,p=[tail[k].double() for k in keys]
                for task in [target,peer]:
                    t=heads[task];image=p@t.T;m=gamma[:,None]*image;m-=m.mean(0);b=beta@image
                    lower=b[:,None]-b[None,:]-len(gamma)**.5*(m[:,:,None]-m[:,None,:]).norm(dim=0)
                    # Independent NumPy dot products and pairwise norm, same frozen parameters.
                    q=p.numpy()@t.numpy().T;mm=gamma.numpy()[:,None]*q;mm-=mm.mean(0);bb=beta.numpy()@q
                    expected=bb[:,None]-bb[None,:]-len(gamma)**.5*np.linalg.norm(mm[:,:,None]-mm[:,None,:],axis=0)
                    np.testing.assert_allclose(lower.numpy(),expected,atol=1e-8,rtol=1e-10);maximum=max(maximum,float(np.abs(lower.numpy()-expected).max()))
                    lower.fill_diagonal_(torch.inf);k=int(lower.min(1).values.argmax());bound=float(lower[k].min())
                    rows.append(dict(target=target,partner=peer,defense=name,coefficient=coefficient,dataset=task,
                        role='target' if task==target else 'partner',best_constant_class=k,minimum_signed_bound=bound,
                        all_prefix_constant_certified_in_real_arithmetic=bound>0,readout_linear_norm=float(m.norm()),readout_bias_norm=float(b.norm())))
    out=Path('outputs/phase3_readout_scope');out.mkdir(exist_ok=False)
    save_json(out/'results.json',rows);save_json(out/'validation.json',dict(status='passed_numpy_readout_bounds',maximum_algebra_error=maximum,
        expert_sha256=hashes,protocol_sha256=digest(protocol),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        new_images=0,gradient_steps=0,new_weights=0,actual_ProDistill_training=False,formal_FP32_roundoff_certificate=False,
        scope='Sufficient real-arithmetic bound for all finite prefixes under the fixed two-expert final readout. Not an accuracy measurement or full author ProDistill reproduction.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',default='configs/bias_tail_precision.json');ap.add_argument('--pilot',action='store_true');ap.add_argument('--verify',action='store_true');ap.add_argument('--certificate',action='store_true');a=ap.parse_args()
    certificate() if a.certificate else run(a.protocol,a.pilot,a.verify)
