"""One fresh confirmation of the exact frozen local-transport point; no repair."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import extended_parameters
from local_residual_transport import FIELDS, capture, cpu, pack, forward, metric
from probe_distance_candidate import on_device
from probe_margin_distance import margin_drift
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, initialize
from verify_levelset_audits import raw_scores

@torch.no_grad()
def run(cfg, source_probes=False):
    out=Path(cfg['output']);start=time.monotonic();report_file=out/('source_probe_validation.json' if source_probes else 'confirmation_validation.json')
    assert not report_file.exists()
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    fixed=cfg.get('frozen_release')
    if fixed:
        assert digest(cfg['source_reference'])==cfg['diagnostic_sha256']
        source=torch.load(cfg['source_reference'],map_location='cpu');saved=torch.load(cfg['prefix_reference'],map_location='cpu')
        assert digest(fixed['path'])==fixed['sha256'];pub=torch.load(fixed['path'],map_location='cpu')
        if cfg.get('recovery_feedback_source'):
            assert not source_probes,'Source4096/probes already independently audited;do not repeat.'
            audit=json.loads(Path(cfg['source_audit_reference']).read_text());assert audit['status']=='passed'
            assert audit['raw_point_sha256']==cfg['diagnostic_sha256']
            cal=saved;source=dict(parameters=cal['p'],ids=cal['cache']['indices']['fit']+cal['cache']['indices']['scale'])
            saved=dict(vo=cal['task_probe_O'],vb=cal['cache']['basis_B']['vh'])
        raw=dict(p=source['parameters'],q=dict(source['parameters']),vo=saved['vo'],vb=saved['vb'],indices=dict(fit=source['ids']),cfg=cfg)
        for key,value in FIELDS.items():raw['q'][key]=pub.get('model.visual.transformer.resblocks.11.'+value,raw['p'][key]).double()
        if not source_probes and not cfg.get('recovery_feedback_source'):
            assert json.loads((out/'source_probe_validation.json').read_text())['status']=='passed'
            assert json.loads((out/'source_probe_independent_validation.json').read_text())['status']=='passed'
    else:
        assert not source_probes
        assert digest(out/'diagnostic.pt')==cfg['diagnostic_sha256'];raw=torch.load(out/'diagnostic.pt',map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);p=on_device(extended_parameters(expert,head));q=dict(p);edits={}
    for k,key in FIELDS.items():
        assert torch.equal(p[k].cpu(),raw['p'][k]);q[k]=raw['q'][k].cuda()
        edits['model.visual.transformer.resblocks.11.'+key]=q[k].cpu().float()
    distance=float(sum(((q[k]-p[k]).norm()/p[k].norm()).square() for k in FIELDS).sqrt());assert distance<=cfg['parameter_budget']
    file=Path(fixed['path']) if fixed else out/'frozen_release.pt'
    if not fixed:
        assert not file.exists();torch.save(edits,file)
    pin=digest(file)
    save_json(out/('pre_probe_freeze.json' if source_probes else 'pre_confirmation_freeze.json'),dict(release_sha256=pin,source_point_sha256=cfg['diagnostic_sha256'],distance=distance,
        weights_unchanged_since_source_diagnosis=True,confirmation_drawn=False,protocol=cfg,source_sha256=digest(__file__)))
    train=dataset(cfg['target'],'data',expert.val_preprocess,True);exclude=set(cfg['extra_excluded_indices'])
    for v in raw['indices'].values():exclude.update(v)
    for path in cfg['extra_index_exclusions']:exclude.update(json.loads(Path(path).read_text())['indices'])
    if source_probes:
        split=dict(indices=source['ids'],split='existing_source',used_for_optimization=True);cache=source['original']
    else:
        split=unused(train,exclude,cfg);save_json(out/'confirmation_split.json',split);assert not set(split['indices'])&exclude
        cache=capture(expert,train,split['indices'])
    data=pack(cache,p,raw['vo'].cuda(),raw['vb'].cuda(),raw['cfg'])
    before,after=forward(data,p,p),forward(data,p,q);reported=metric(before,after,len(split['indices']))
    for value in [before]:
        gap=value.max(1).values[:,None]-value;mask=~torch.nn.functional.one_hot(value.argmax(1),value.shape[1]).bool()
        assert (gap[mask]>1e-12).all(), 'No active denominator floor.'
    pn=cpu(p);qn=cpu(q);pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in pn.items()};qn={k:v.numpy() if torch.is_tensor(v) else v for k,v in qn.items()}
    dn={k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in data.items()}
    r=dn['r']+dn['H']@(qn['O']-pn['O']).T+qn['ob']-pn['ob'];center=r-r.mean(1,keepdims=True)
    pre=(center/np.sqrt((center**2).mean(1,keepdims=True)+pn['eps2'])*pn['gamma2']+pn['beta2'])@qn['A'].T+qn['a']
    checked=raw_scores(dn['rounding']+r+(pre*expit(1.702*pre)+dn['offset'])@qn['B'].T+qn['b'],qn)
    np.testing.assert_allclose(checked,after.cpu(),atol=1e-8,rtol=1e-9);np.testing.assert_array_equal(checked.argmax(1),after.cpu().argmax(1))
    if source_probes:
        physical=torch.load(cfg['source_scores_reference'],map_location='cpu');actual,actual0=physical['after'],physical['before']
    else:
        released=copy.deepcopy(expert)
        for key,val in edits.items():released.state_dict()[key].copy_(val)
        actual=[];released.cuda();head.cuda()
        for images,_ in make_loader(train,split['indices'],64):actual.append(head(released(images.cuda())).cpu())
        released.cpu();actual=torch.cat(actual).double();actual0=head(cache['actual'].float().cuda()).cpu().double();head.cpu()
    n=len(split['indices']);control=max(float((actual-after[:n].cpu()).abs().max()),float((actual0-before[:n].cpu()).abs().max()))
    assert control<=cfg['score_tolerance'] and torch.equal(actual.argmax(1),after[:n].cpu().argmax(1))
    actual_gaps=actual0.max(1).values[:,None]-actual0;mask=~torch.nn.functional.one_hot(actual0.argmax(1),actual0.shape[1]).bool()
    if fixed:assert (actual_gaps[mask]>1e-12).all()
    true_drift=float(margin_drift(actual0,actual).max())
    labels=torch.as_tensor(train.targets)[split['indices']] if fixed else torch.tensor([int(train[i][1]) for i in split['indices']])
    passed=reported['worst_task_drift']<=cfg['task_budget'] and true_drift<=cfg['task_budget'] and reported['changed']==0 and torch.equal(actual0.argmax(1),actual.argmax(1))
    raw_file=out/('source_probes.pt' if source_probes else 'confirmation.pt')
    artifact=dict(indices=split['indices'],before=before.cpu(),after=after.cpu(),actual_original=actual0,actual_release=actual,labels=labels)
    artifact['data' if fixed else 'cache']=cpu(data) if fixed else cache
    torch.save(artifact,raw_file)
    result=dict(status='passed' if passed else 'failed',confirmation_passed=bool(passed),parameter_distance=distance,
        source=reported,full_model_natural_worst_drift=true_drift,full_model_control_max_error=control,numpy_score_max_error=float(abs(checked-after.cpu().numpy()).max()),
        original_correct=int((actual0.argmax(1)==labels).sum()),released_correct=int((actual.argmax(1)==labels).sum()),n=n,
        release_sha256=pin,confirmation_sha256=digest(raw_file),source_point_sha256=cfg['diagnostic_sha256'],
        split_sha256=None if source_probes else digest(out/'confirmation_split.json'),source_sha256=digest(__file__),protocol=cfg,checkpoint_sha256=pins,
        source_point_repaired=False,confirmation_redrawn=False,optimizer_steps=0,gradients=0,partner_used=False,test_used=False,
        stage='source_probes' if source_probes else 'fresh_confirmation',task_probe_semantics='Full FP64 O/B finite tail,including LN variance; natural controls matched to FP32 whole model.',
        per_class=[dict(class_id=c,n=int(labels.eq(c).sum()),original_correct=int((actual0.argmax(1).eq(labels)&labels.eq(c)).sum()),
                        released_correct=int((actual.argmax(1).eq(labels)&labels.eq(c)).sum())) for c in range(actual.shape[1])],
        finite_probe_rows=len(before)-n,matched_tail_rows=2*len(before),
        full_model_forwards=0 if source_probes else 2*n,new_release_weights=0 if fixed else int(passed),wall_seconds=time.monotonic()-start)
    save_json(report_file,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--source-probes',action='store_true')
    a=ap.parse_args();initialize();run(json.loads(Path(a.protocol).read_text()),a.source_probes)
