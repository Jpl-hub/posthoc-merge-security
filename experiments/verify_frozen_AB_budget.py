"""Retrospective source-budget audit of unchanged A/B weights, never a repair step."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from attention_cascade import parameters
from clip_benchmark import digest
from covariance_noise import dataset
from task_probe_distance import checked_models, initialize
from verify_finite_comparison import suffix
from verify_levelset_audits import convert, raw_scores


@torch.no_grad()
def run(output):
    start = time.monotonic(); out = Path(output); out.mkdir(parents=True,exist_ok=False)
    cfg = json.loads(Path('configs/frozen_AB_budget_audit.json').read_text())
    for key in ['source_cache','release']: assert digest(cfg[key]) == cfg[key+'_sha256']
    data = torch.load(cfg['source_cache'],map_location='cpu'); base = data['environments']['source']
    assert len(data['indices']) == cfg['source_n']
    initialize(); assert '3090' in torch.cuda.get_device_name()
    _, own, head, hashes = checked_models('MNIST'); released = copy.deepcopy(own)
    edits = torch.load(cfg['release'],map_location='cpu')
    assert set(edits) == {'model.visual.transformer.resblocks.11.mlp.c_fc.weight',
                          'model.visual.transformer.resblocks.11.mlp.c_proj.weight'}
    distance = sum(float(((v.double()-own.state_dict()[k].double()).norm()/own.state_dict()[k].double().norm())**2)
                   for k,v in edits.items())**.5
    assert distance <= cfg['parameter_budget']
    released.load_state_dict(own.state_dict()|edits,strict=True)
    p = convert(base['parameters']); qp = parameters(released,head); qp.pop('D'); q = convert(qp)
    H = base['cache']['H'].numpy(); h = base['cache']['h'].numpy()
    h0,_ = suffix(H,p); h1,_ = suffix(H,q)
    s0 = raw_scores(h,p); s1 = raw_scores(h+h1-h0,p)
    y = s0.argmax(1); gap = s0[np.arange(len(y)),y,None]-s0; ds = s1-s0
    rho = np.divide(ds[np.arange(len(y)),y,None]-ds,gap,out=np.zeros_like(gap),where=gap>0)
    i,c = np.unravel_index(np.abs(rho).argmax(),rho.shape); index = int(data['indices'][i]); teacher = int(y[i])
    train = dataset('MNIST','data',own.val_preprocess,True); image,label = train[index]
    image = image[None].cuda(); own.cuda(); released.cuda(); head.cuda()
    before = head(own(image)).cpu().double().numpy()[0]
    after = head(released(image)).cpu().double().numpy()[0]
    error = float(max(abs(before-s0[i]).max(),abs(after-s1[i]).max()))
    assert error < cfg['full_model_score_atol']
    actual = ((after[teacher]-before[teacher])-(after[c]-before[c]))/(before[teacher]-before[c])
    np.savez(out/'scores.npz',indices=data['indices'],before=s0,after=s1,relative_drift=rho)
    result = dict(status='frozen_AB_budget_counterexample_replayed' if abs(actual)>.5 else 'frozen_AB_finite_audit_not_global_certificate',
        preregistered_commit='c3f437b',source_n=len(y),parameter_distance=distance,budget=cfg['task_budget'],
        n_over_budget=int((np.abs(rho).max(1)>.5).sum()),source_prediction_changes=int((s1.argmax(1)!=y).sum()),
        worst=dict(training_index=index,teacher_class=teacher,competitor_class=int(c),label=int(label),
            numpy_signed_drift=float(rho[i,c]),full_model_signed_drift=float(actual),full_model_score_error=error,
            prediction_before=int(before.argmax()),prediction_after=int(after.argmax()),before=before.tolist(),after=after.tolist()),
        gradients=0,new_weights=0,full_model_forwards=2,confirmation_used=False,test_used=False,peer_loaded=False,
        release_sha256=digest(cfg['release']),checkpoint_sha256=hashes,source_cache_sha256=cfg['source_cache_sha256'],
        raw_scores_sha256=digest(out/'scores.npz'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        scope=cfg['source_role'],old_confirmation_and_TA_AGA_preserved=True,repair_or_selection=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True)
    run(ap.parse_args().output)
