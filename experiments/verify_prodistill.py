"""Independent cached-score and frozen-artifact audit; no GPU or optimization."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln, logsumexp

from verify_covariance_noise import official_labels


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_metrics(scores, labels, before, row):
    s,y,z=scores.numpy(),labels.numpy(),before.numpy();assert np.isfinite(s).all()
    p,b=s.argmax(1),z.argmax(1);ok,was=p==y,b==y;n=len(y)
    assert n==row['n'] and int(ok.sum())==row['correct']
    assert abs(100*ok.mean()-row['accuracy_percent'])<1e-12
    lost,gained=int((was&~ok).sum()),int((~was&ok).sum())
    assert lost==row['paired']['lost'] and gained==row['paired']['gained']
    assert int((was&ok).sum())==row['paired']['both_correct'] and int((~was&~ok).sum())==row['paired']['both_wrong']
    assert int((p!=b).sum())==row['changed_predictions']
    for r in row['per_class']:
        mask=y==r['class_id'];assert int(mask.sum())==r['n']
        assert int((ok&mask).sum())==r['correct'] and int((was&mask).sum())==r['original_correct']
    m=lost+gained
    if m:
        k=np.arange(min(lost,gained)+1,dtype=np.float64)
        logp=min(0.,float((np.log(2)+logsumexp(gammaln(m+1)-gammaln(k+1)-gammaln(m-k+1)-m*np.log(2)))/np.log(10)))
    else:logp=0.
    error=abs(logp-row['mcnemar_log10_p']);assert error<1e-7
    delta=ok.astype(float)-was;half=1.96*delta.std(ddof=1)/np.sqrt(n)
    np.testing.assert_allclose(row['paired_accuracy_delta_95ci_pp'],100*np.array([delta.mean()-half,delta.mean()+half]),atol=1e-10)
    return error


def verify(root):
    root=Path(root);out=root/'independent_validation.json';assert not out.exists()
    start=time.monotonic();cfg=json.loads((root/'00/protocol.json').read_text());source=root/'source.py'
    if not source.exists():source=Path('experiments/phase3_prodistill.py')
    assert digest(source)==cfg['source_sha256']
    sizes={'MNIST':60000,'SVHN':73257,'DTD':1880};cases=[];count=0;maximum=0.
    for i,case in enumerate(cfg['cases']):
        d=root/f'{i:02d}';read=lambda p:json.loads((d/p).read_text())
        t,freeze=read('training.json'),read('pre_evaluation.json');ids=read('calibration_indices.json')
        for task,idx in ids.items():assert idx==torch.randperm(sizes[task],generator=torch.Generator().manual_seed(0))[-64:].tolist()
        assert digest(d/'calibration_indices.json')==t['indices_sha256']
        model_sha=digest(d/'model.pt');assert model_sha==t['model_sha256']==freeze['model_sha256']
        assert t['source_sha256']==cfg['source_sha256'] and t['completed'] and t['backward_steps']==1200
        assert t['expert_updates']==t['defense_updates']==0 and len(t['layers'])==12
        assert not t['selection_from_test']
        assert all(np.isfinite(z['NumPy_weight_error']) and z['NumPy_weight_error']<=2e-5 for z in t['layers'])
        checks={}
        for stage in ['pilot','full']:
            validation=read(stage+'/validation.json');raw=torch.load(d/stage/'scores.pt',map_location='cpu',weights_only=False)
            control=torch.load(root/f'{case["control"]:02d}'/stage/'scores.pt',map_location='cpu',weights_only=False)
            assert validation['status']=='passed' and digest(d/stage/'scores.pt')==validation['raw_sha256']
            assert validation['model_sha256']==model_sha
            for row in read(stage+'/results.json'):
                item=raw[row['defense']+'/'+row['dataset']];baseline=control['Unprotected/'+row['dataset']]['logits']
                np.testing.assert_array_equal(item['target'].numpy(),official_labels(row['dataset'])[item['indices']])
                maximum=max(maximum,verify_metrics(item['logits'],item['target'],baseline,row));count+=1
            checks[stage]=validation['raw_sha256']
        if freeze['phase1_tail_sha256']:assert digest(d/'phase1_tail.pt')==freeze['phase1_tail_sha256']
        cases.append(dict(case=i,model_sha256=model_sha,raw_sha256=checks))
    result=dict(status='passed_independent_numpy_metrics_and_artifact_hashes',rows_checked=count,cases=cases,
        maximum_exact_logp_error=maximum,CPU_wall_seconds=time.monotonic()-start,new_GPU_forwards=0,new_training=0,
        source_sha256=cfg['source_sha256'],audit_source_sha256=digest(__file__),
        scope='Independent full cached-score recount, calibration and model freeze checks; not an independent implementation of whole author training.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


@torch.no_grad()
def capacity(root):
    """Algebraic coefficient-map rank, not a convergence or security guarantee."""
    from clip_benchmark import checkpoint_loader
    from phase3_evaluate import apply
    root=Path(root);out=root/'capacity_audit.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(2)
    cfg=json.loads((root/'00/protocol.json').read_text());ref=cfg['frozen_inputs'];assert digest(ref['path'])==ref['sha256']
    parent=json.loads(Path(ref['path']).read_text());load=checkpoint_loader(Path('external/secure-merge-attack'))
    pins={r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    def checked(name):
        path=Path('checkpoints/ViT-B-32')/name;assert digest(path)==pins['ViT-B-32/'+name]
        return load(str(path),map_location='cpu').eval().requires_grad_(False)
    base=checked('zeroshot.pt');originals={t:checked(t+'/finetuned.pt') for t in ['MNIST','SVHN']}
    aligned={t:apply(originals[t],parent['AGA_references'][t],full=True) for t in originals}
    records=[]
    for i,c in enumerate(cfg['cases']):
        own=originals['MNIST'] if c['scope']=='None' else aligned['MNIST']
        peer=aligned['SVHN'] if c['scope']=='All experts' else originals['SVHN']
        if c['defense']!='Unprotected':own=apply(own,parent['defenses'][c['defense']]['release' if c['scope']=='None' else 'AGA_edits'])
        layers=[]
        for j in range(12):
            left=dict(own.model.visual.transformer.resblocks[j].named_parameters());right=dict(peer.model.visual.transformer.resblocks[j].named_parameters())
            count=support=0;minimum=float('inf');maximum=0.
            for name,p0 in base.model.visual.transformer.resblocks[j].named_parameters():
                a=(left[name]-p0).double();b=(right[name]-p0).double();scale=a.square()+b.square();positive=scale[scale>0]
                count+=p0.numel();support+=positive.numel()
                if positive.numel():minimum=min(minimum,float(positive.min()));maximum=max(maximum,float(positive.max()))
            layers.append(dict(layer=j,block_parameters=count,coefficient_parameters=2*count,algebraically_reachable_coordinates=support,
                reachable_fraction=support/count,positive_diagonal_min=minimum if support else None,positive_diagonal_max=maximum))
        final=torch.load(root/f'{i:02d}'/'model.pt',map_location='cpu',weights_only=False)
        control=torch.load(root/f'{c["control"]:02d}'/'model.pt',map_location='cpu',weights_only=False)
        prefix=[k for k in final if '.resblocks.' not in k or int(k.split('.resblocks.')[1].split('.')[0])<11]
        records.append(dict(case=i,configuration=c,layers=layers,all_parameters_outside_last_block_equal_to_control=all(torch.equal(final[k],control[k]) for k in prefix)))
        del final,control
    fixture=np.ones((3,3));assert np.linalg.matrix_rank(fixture)==1 and np.linalg.matrix_rank(fixture*np.eye(3))==3
    result=dict(status='completed_read_only_coefficient_capacity_audit',cases=records,source_sha256=digest(__file__),CPU_wall_seconds=time.monotonic()-start,
        new_images=0,gradient_steps=0,new_weights=0,scope='Real-algebra coordinate reachability only; Adam trajectory, finite precision, convergence and classification are not certified. Official elementwise coefficients are unconstrained; original experts remain frozen.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--capacity',action='store_true');a=p.parse_args();(capacity if a.capacity else verify)(a.root)
