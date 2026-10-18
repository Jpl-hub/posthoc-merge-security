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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args();verify(a.root)
