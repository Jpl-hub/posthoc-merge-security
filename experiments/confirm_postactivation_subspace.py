"""Independently verify all frozen ablations; confirm the unchanged Outside0.5 point."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from bias_canonicalization import inputs
from clip_benchmark import digest, evaluate
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from feature_noise import capture
from postactivation_subspace import metric
from subspace_probe import make_loader, save_json
from verify_levelset_audits import raw_scores


def numpy_metric(before, after):
    y=before.argmax(1);i=np.arange(len(y));g=before[i,y,None]-before;mask=np.ones(g.shape,bool);mask[i,y]=False
    assert (g[mask]>0).all();g[i,y]=1
    delta=after-before;r=(delta[i,y,None]-delta)/g;r[i,y]=0
    return float(abs(r).max()),int((after.argmax(1)!=y).sum())


@torch.no_grad()
def run(protocol):
    start=time.monotonic();cfg,_,checked=inputs(protocol);source=Path(cfg['source_output']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for path,pin in cfg['references'].items():assert digest(path)==pin,path
    raw=torch.load(source/'diagnostic.pt',map_location='cpu',weights_only=False);records=json.loads((source/'results.json').read_text())
    pn={k:(v.numpy() if torch.is_tensor(v) else v) for k,v in raw['parameters'].items()};pe={k:(v.numpy() if torch.is_tensor(v) else v) for k,v in raw['proxy_parameters'].items()}
    h,z,hp,zp=[raw[k].numpy() for k in ['h','z','hp','zp']];errors=[]
    source_cfg=json.loads((source/'protocol.json').read_text());v=raw['v'].numpy();n=source_cfg['fit_n'];yy=raw_scores(h[:n],pn).argmax(1)
    contexts=np.concatenate([np.flatnonzero(yy==c)[:source_cfg['probe_contexts_per_class']] for c in range(len(pn['head']))])
    rms=np.sqrt(np.mean((z[:n]@v.T)**2,axis=0));shift=(source_cfg['probe_step']*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).reshape(-1,z.shape[1])
    zp_task=(z[contexts,None]+shift[None]).reshape(-1,z.shape[1]);hp_task=(h[contexts,None]+(shift@pn['B'].T)[None]).reshape(-1,h.shape[1])
    for r in records:
        point=raw['points'][r['family']+'_'+str(r['parameter_budget'])];delta=point['delta'].numpy()
        for label,hh,zz,p,factor in [('source',h,z,pn,1.),('proxy',hp,zp,pe,.5)]:
            before=raw_scores(hh,p);after=raw_scores(hh+factor*zz@delta.T,p)
            error=float(abs(after-point[label].numpy()).max());assert error<1e-8
            drift,changed=numpy_metric(before,after);assert abs(drift-r[label]['worst_task_drift'])<1e-7 and changed==r[label]['changed']
            y=before.argmax(1);bad=after.argmax(1)!=y
            for c in r[label]['per_class']:
                assert int((y==c['class_id']).sum())==c['n'] and int((bad&(y==c['class_id'])).sum())==c['changed']
            errors.append(error)
        a,b=raw_scores(hp_task,pn),raw_scores(hp_task+zp_task@delta.T,pn)
        drift,changed=numpy_metric(a,b);assert abs(drift-r['probes']['worst_task_drift'])<1e-7 and changed==r['probes']['changed']
        assert float(abs(b-point['probes'].numpy()).max())<1e-8
        np.testing.assert_allclose(np.linalg.norm(delta)/np.linalg.norm(pn['B']),r['actual_distance'],rtol=1e-10)
    save_json(out/'source_validation.json',dict(status='passed',all_six_ablation_points=True,NumPy_complete_LN_and_normalized_scores=True,maximum_score_error=max(errors),original_joint_points_failed=True,source_raw_sha256=digest(source/'diagnostic.pt')))
    # This is the existing source-feasible ablation, not a new direction or repaired Joint point.
    point=raw['points'][cfg['point_key']];own=checked('MNIST/finetuned.pt');head=checked('head_MNIST.pt')
    state=own.state_dict();key='model.visual.transformer.resblocks.11.mlp.c_proj.weight'
    np.testing.assert_array_equal(state[key].double().numpy(),pn['B'])
    released=copy.deepcopy(own);released.state_dict()[key].copy_((raw['parameters']['B']+point['delta']).float())
    distance=float((released.state_dict()[key].double()-state[key].double()).norm()/state[key].double().norm());assert distance<=cfg['parameter_budget']
    train=dataset('MNIST','data',own.val_preprocess,True)
    def evaluate_pair(ids):
        a=[];b=[]
        evaluate(own,head,make_loader(train,ids,64),'cuda',logit_sink=a.append)
        evaluate(released,head,make_loader(train,ids,64),'cuda',logit_sink=b.append)
        return torch.cat(a).double(),torch.cat(b).double()
    a,b=evaluate_pair(raw['indices']);full=metric(a,b)
    control=float((b-point['source']).abs().max());assert control<=cfg['score_tolerance']
    save_json(out/'full_source_validation.json',dict(**full,maximum_score_error=control,parameter_distance=distance))
    if full['worst_task_drift']>cfg['task_budget']:
        save_json(out/'confirmation_validation.json',dict(status='failed_full_source',fresh_confirmation_drawn=False,parameter_distance=distance,full_source=full));return
    release=out/'frozen_release.pt';torch.save({key:released.state_dict()[key].clone()},release)
    save_json(out/'pre_confirmation_freeze.json',dict(release_sha256=digest(release),source_raw_sha256=digest(source/'diagnostic.pt'),point_key=cfg['point_key'],parameter_distance=distance,protocol=cfg,source_sha256=digest(__file__),new_direction=False,source_point_repaired=False,confirmation_drawn=False))
    exposed=np.random.default_rng(cfg['utility_seed']).choice(len(train),cfg['utility_n'],replace=False).tolist()
    split=unused(train,set(raw['indices'])|set(exposed)|{41538,27330,31374},cfg)
    assert not set(split['indices'])&set(raw['indices']);save_json(out/'confirmation_split.json',split)
    ca,cb=evaluate_pair(split['indices']);report=metric(ca,cb)
    zz,hh=capture(own,make_loader(train,split['indices'],64));expected=raw_scores(hh.numpy()+zz.numpy()@point['delta'].numpy().T,pn)
    confirmation_error=float(abs(expected-cb.numpy()).max());assert confirmation_error<=cfg['score_tolerance']
    labels=torch.tensor([int(train[i][1]) for i in split['indices']]);passed=report['worst_task_drift']<=cfg['task_budget']
    torch.save(dict(source_before=a,source_after=b,before=ca,after=cb,expected_NumPy=expected,indices=split['indices'],labels=labels),out/'confirmation.pt')
    save_json(out/'confirmation_validation.json',dict(status='passed' if passed else 'failed',fresh_confirmation_drawn=True,source_point_repaired=False,point_key=cfg['point_key'],parameter_distance=distance,confirmation=report,full_source=full,NumPy_confirmation_score_error=confirmation_error,n=len(labels),original_correct=int((ca.argmax(1)==labels).sum()),released_correct=int((cb.argmax(1)==labels).sum()),release_sha256=digest(release),raw_sha256=digest(out/'confirmation.pt'),split_sha256=digest(out/'confirmation_split.json'),source_sha256=digest(__file__),gradients=0,expert_updates=0,partner_used=False,test_used=False,wall_seconds=time.monotonic()-start,eligible_for_true_fusion=bool(passed),scope='One unchanged Outside0.5 source ablation; original Joint construct failed. Finite new training confirmation is not an all-input proof.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',default='configs/postactivation_confirmation.json');args=ap.parse_args();run(args.protocol)
