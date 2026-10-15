"""Official-label recount and independently assembled full TA/AGA replays."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader, digest, paired_counts
from covariance_noise import dataset
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


@torch.no_grad()
def verify(output):
    started=time.monotonic();out=Path(output);cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text());case=json.loads((out/'case.json').read_text())
    assert meta['completed'] and meta['fusion_run'] and meta['evaluation_optimizer_steps']==0
    for file,key in [('release.pt','release_sha256'),('calibration.pt','calibration_sha256'),('predictions.pt','predictions_sha256'),('released_AGA_edits.pt','AGA_edits_sha256')]:assert digest(out/'artifacts'/file)==meta[key]
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    author=Path('external/secure-merge-attack/utils_attack_B32.py')
    author_pin=hashlib.sha256(author.read_text(encoding='utf-8').encode()).hexdigest()
    assert author_pin==json.loads(Path('configs/sources.json').read_text())['AGA']['sha256']
    freeze=json.loads((out/'release_freeze.json').read_text());assert not freeze['partner_loaded'] and not freeze['test_loaded']
    target,peer=case['target'],case['partner'];assert not any(peer in k for k in freeze['checkpoint_sha256'])
    root=Path('checkpoints/ViT-B-32');load=checkpoint_loader(Path('external/secure-merge-attack'))
    for path,pin in meta['checkpoint_sha256'].items():assert digest(root/path)==pin
    anchor=load(str(root/'zeroshot.pt')).cpu().eval();own=load(str(root/target/'finetuned.pt')).cpu().eval();other=load(str(root/peer/'finetuned.pt')).cpu().eval()
    heads={t:load(str(root/f'head_{t}.pt')).cpu().eval() for t in [target,peer]}
    source=Path(cfg['publication_point']);assert digest(source/'training_point.pt')==cfg['training_point_sha256']
    point=torch.load(source/'training_point.pt',map_location='cpu');release=torch.load(out/'artifacts/release.pt',map_location='cpu');protected=copy.deepcopy(own)
    prefix='model.visual.transformer.resblocks.11.';sw=own.state_dict();d=sw[prefix+'attn.in_proj_weight'].shape[1]
    ww=sw[prefix+'attn.in_proj_weight'].double().numpy().copy();bb=sw[prefix+'attn.in_proj_bias'].double().numpy().copy();t=point['cfg']['gain_delta']
    ww[:2*d]*=1.+t;bb[:2*d]*=1.+t
    b=sw[prefix+'mlp.c_proj.weight'].double().numpy();delta=np.linalg.norm(b)*point['output_basis'].numpy()@point['k'].numpy()
    expected={prefix+'attn.in_proj_weight':ww.astype(np.float32),prefix+'attn.in_proj_bias':bb.astype(np.float32),prefix+'mlp.c_proj.weight':(b+delta).astype(np.float32)}
    assert set(expected)==set(release);error=0.
    for key,value in release.items():
        err=float(np.abs(value.numpy()-expected[key]).max());assert err<2e-7;error=max(error,err);protected.state_dict()[key].copy_(value)
    def recovered(model,task):
        ref=cfg['AGA_references'][task];assert digest(ref['path'])==ref['sha256']
        result=copy.deepcopy(model);result.load_state_dict(torch.load(ref['path'],map_location='cpu'),strict=True);return result
    own_a=recovered(own,target);other_a=recovered(other,peer);protected_a=copy.deepcopy(own_a)
    for key,value in torch.load(out/'artifacts/released_AGA_edits.pt',map_location='cpu').items():protected_a.state_dict()[key].copy_(value)
    predictions=torch.load(out/'artifacts/predictions.pt',map_location='cpu');rows=json.loads((out/'results.json').read_text())
    ids=json.loads((out/'training_indices.json').read_text());hold=json.loads((out/'holdout_protocol.json').read_text())['indices']
    labels={task:official_labels(task) for task in [target,peer]};trainlabels=official_labels(target,True)
    data={task:dataset(task,'data',own.val_preprocess) for task in [target,peer]};training=dataset(target,'data',own.val_preprocess,True)
    checks=[]
    for row in rows:
        preset,stage,task=[row[k] for k in ['preset','stage','dataset']];key='/'.join([preset,stage,task]);raw=predictions[key]
        prediction,y=raw['prediction'].numpy(),raw['target'].numpy();group=hold if stage=='holdout' else ids['confirmation'] if stage=='confirmation' else None
        expected=trainlabels[group] if group is not None else labels[task];np.testing.assert_array_equal(y,expected)
        assert int((prediction==y).sum())==row['correct'] and len(y)==row['n'];assert abs(row['top1']-row['correct']/row['n'])<1e-12
        for cls in row['per_class']:
            k=cls['class_id'];assert cls['n']==int((y==k).sum()) and cls['correct']==int(((prediction==y)&(y==k)).sum())
        if preset=='released':
            baseline=predictions['/'.join(['original',stage,task])]['prediction'];assert row['paired']==paired_counts(baseline.eq(raw['target']),raw['prediction'].eq(raw['target']))
            assert row['changed_predictions']==int(raw['prediction'].ne(baseline).sum())
        if stage=='partner_standalone':model=other
        elif stage in ['holdout','confirmation','standalone']:model=own if preset=='original' else protected
        elif stage=='AGA_standalone':model=own_a if preset=='original' else protected_a
        else:
            first=own if preset=='original' else protected;second=other
            if stage=='AGA':first=own_a if preset=='original' else protected_a;second=other_a
            assert stage in ['direct','AGA'];model=explicit_merge(anchor,first,second,cfg['coefficient'])
        chosen=np.linspace(0,len(y)-1,min(16,len(y)),dtype=int);samples=[group[i] for i in chosen] if group is not None else chosen.tolist();dataset_=training if group is not None else data[task]
        images=torch.stack([dataset_[i][0] for i in samples]).cuda();model.cuda();heads[task].cuda()
        actual=heads[task](model(images)).argmax(1).cpu().numpy();model.cpu();heads[task].cpu();np.testing.assert_array_equal(actual,prediction[chosen])
        checks.append(dict(key=key,official_n=len(y),replayed_predictions=len(chosen),prediction_match=True))
    result=dict(status='passed',source_sha256=digest(__file__),release_sha256=meta['release_sha256'],predictions_sha256=meta['predictions_sha256'],
                official_count_rows=len(rows),per_class_and_paired_counts='passed',explicit_TA_formula=True,finite_weight_reconstruction_maximum_error=error,
                fresh_prediction_instances=sum(c['replayed_predictions'] for c in checks),checks=checks,new_candidates=0,evaluation_optimizer_steps=0,first_phase_changed=False,
                author_AGA_text_sha256=author_pin,wall_seconds=time.monotonic()-started)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;verify(args.output)
