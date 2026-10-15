"""Independent NumPy finite-temperature reconstruction; no candidate selection."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit, softmax

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores, residual


def hidden(h, p, gain, shift, off, rounding, db, with_gate=False):
    n, length, width=h.shape;heads=p['heads'];dim=width//heads
    x=ln(h, p['gamma1'], p['beta1'], p['eps1']);x[:, 0]+=shift
    # Reconstruct the released matrices and biases, rather than scaling logits.
    weights=p['W'].copy();bias=p['w'].copy();weights[:2*width]*=gain;bias[:2*width]*=gain
    wq,wk,wv=np.split(weights, 3);bq,bk,bv=np.split(bias, 3)
    q=(x[:, 0]@wq.T+bq).reshape(n, heads, dim)
    k=(x@wk.T+bk).reshape(n, length, heads, dim).transpose(0, 2, 1, 3)
    v=(x@wv.T+bv).reshape(n, length, heads, dim).transpose(0, 2, 1, 3)
    prob=softmax(np.einsum('nhd,nhtd->nht', q, k)/np.sqrt(dim), axis=-1)
    r=h[:, 0]+np.einsum('nht,nhtd->nhd', prob, v).reshape(n, width)@p['O'].T+p['ob']
    pre=ln(r, p['gamma2'], p['beta2'], p['eps2'])@p['A'].T+p['a']
    z=pre*expit(1.702*pre)+off
    output=rounding+r+z@(p['B']+db).T+p['b']
    return (output,z) if with_gate else output


def verify(cfg):
    start=time.monotonic();root=Path(cfg['output']);report=json.loads((root/'results.json').read_text())
    assert digest(root/'training_point.pt')==report['training_point_sha256']
    assert digest(root/'diagnostic.pt')==report['diagnostic_sha256']
    assert digest(root/'predictions.pt')==report['predictions_sha256']
    assert digest(Path(cfg['cache_reference'])/'calibration.pt')==cfg['cache_sha256']
    point=torch.load(root/'training_point.pt', map_location='cpu');diagnostic=torch.load(root/'diagnostic.pt', map_location='cpu')
    cal=torch.load(Path(cfg['cache_reference'])/'calibration.pt', map_location='cpu');data=convert(diagnostic['native']);p=convert(diagnostic['parameters'])
    # The generic numeric converter intentionally casts tensor values to
    # double. Index arrays must retain their original integer semantics.
    data['take']=diagnostic['native']['take'].numpy()
    assert np.issubdtype(data['take'].dtype,np.integer)
    ids=diagnostic['indices'];lookup={i:(g,j) for g,key in [('native','fit'),('scale','scale')] for j,i in enumerate(cal['indices'][key])}
    hh=np.stack([cal[lookup[i][0]]['H'][lookup[i][1]].numpy() for i in ids])
    rounding=np.stack([cal[lookup[i][0]]['rounding'][lookup[i][1]].numpy() for i in ids])
    q=point['output_basis'].numpy();k=point['k'].numpy();db=np.linalg.norm(p['B'])*q@k
    np.testing.assert_allclose(q.T@q, np.eye(q.shape[1]), atol=1e-10, rtol=1e-10)
    # Both Q/q and K/k are normalized as a joint affine operator, not many
    # per-head denominators that would artificially enlarge the distance.
    width=p['A'].shape[1];t=cfg['gain_delta'];parts=[]
    for start0 in [0,width]:
        w=p['W'][start0:start0+width];b=p['w'][start0:start0+width]
        parts.append((np.sum((t*w)**2)+np.sum((t*b)**2))/(np.sum(w*w)+np.sum(b*b)))
    distance=float(np.sqrt(sum(parts)+(np.linalg.norm(db)/np.linalg.norm(p['B']))**2))
    assert abs(distance-report['selected']['distance'])<1e-10
    before=[];after=[];solo=[];score_error=0.;hidden_error=0.
    for start0 in range(0, len(data['take']), cfg['batch_size']):
        rows=data['take'][start0:start0+cfg['batch_size']];s=data['shifts'][start0:start0+len(rows)];off=data['offsets'][start0:start0+len(rows)]
        h=hh[rows].astype(np.float64);rnd=rounding[rows]
        b,z=hidden(h,p,1.,s,off,rnd,np.zeros_like(db),with_gate=True);a=hidden(h,p,1.+t,s,off,rnd,db)
        bs=raw_scores(b,p);as_=raw_scores(a,p)
        score_error=max(score_error,float(np.abs(bs-data['teacher'][start0:start0+len(rows)]).max()))
        hidden_error=max(hidden_error,float(np.abs(b-data['teacher_hidden'][start0:start0+len(rows)]).max()))
        np.testing.assert_allclose(as_,raw_scores(data['base'][start0:start0+len(rows)]+np.linalg.norm(p['B'])*(data['z'][start0:start0+len(rows)]@k.T)@q.T,p),atol=1e-9,rtol=1e-9)
        before.append(bs);after.append(as_)
        if start0<data['natural_n']:solo.append(raw_scores(b+z@db.T,p)[:min(len(rows),data['natural_n']-start0)])
    before=np.concatenate(before);after=np.concatenate(after);drift=float(np.abs(residual(before,after)).max());n=data['natural_n']
    assert score_error<1e-9 and hidden_error<1e-8
    assert abs(drift-report['selected']['worst_margin_drift'])<1e-6
    changed=after[:n].argmax(1)!=before[:n].argmax(1)
    assert int(changed.sum())==report['selected']['natural_changes']
    by=np.bincount(before[:n].argmax(1)[changed],minlength=len(p['head'])).tolist()
    assert by==report['selected']['changed_by_teacher_class']
    labels=official_labels(cfg['target'],True)[ids]
    old=int((before[:n].argmax(1)==labels).sum());new=int((after[:n].argmax(1)==labels).sum())
    assert old==report['original_correct'] and new==report['released_correct']
    solo=np.concatenate(solo);solo_changed=solo.argmax(1)!=before[:n].argmax(1)
    assert int(solo_changed.sum())==report['B_only_changed']
    passed=drift<=cfg['task_budget'] and distance<=cfg['parameter_budget'];assert passed==report['task_gate_passed']
    torch.save(dict(before=torch.tensor(before),after=torch.tensor(after),indices=ids),root/'independent_predictions.pt')
    result=dict(status='passed_task_calibration' if passed else 'passed_failure_verification',target=cfg['target'],all_task_inputs=len(before),natural_n=n,
                parameter_distance=distance,worst_margin_drift=drift,natural_prediction_changes=int(changed.sum()),changed_by_class=by,
                original_correct=old,released_correct=new,B_only_changed=int(solo_changed.sum()),score_error=score_error,hidden_error=hidden_error,point_sha256=report['training_point_sha256'],
                predictions_sha256=digest(root/'independent_predictions.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
                new_release_weights=0,partner_used=False,test_used=False,confirmation_used=False,
                scope='Independent NumPy all-token attention from explicitly transformed Q/K weights+biases, QuickGELU, LayerNorm, CLIP score normalization, every frozen natural/SVD input and official labels. No source-midpoint/proxy rejection rule.')
    save_json(root/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    torch.set_num_threads(4);verify(json.loads(Path(args.protocol).read_text()))
