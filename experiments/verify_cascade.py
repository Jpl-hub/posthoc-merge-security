"""Independent NumPy nonlinear-attention publication mechanism verification."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit,softmax

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert,raw_scores,residual


def suffix(h,p):
    n,t,d=h.shape;heads=p['heads'];width=d//heads;x=ln(h,p['gamma1'],p['beta1'],p['eps1'])
    wq,wk,wv=np.split(p['W'],3);bq,bk,bv=np.split(p['w'],3)
    q=(x[:,0]@wq.T+bq).reshape(n,heads,width)
    k=(x@wk.T+bk).reshape(n,t,heads,width).transpose(0,2,1,3)
    v=(x@wv.T+bv).reshape(n,t,heads,width).transpose(0,2,1,3)
    attention=softmax(np.einsum('nhd,nhtd->nht',q,k)/np.sqrt(width),axis=-1)
    average=np.einsum('nht,nhtd->nhd',attention,v).reshape(n,d)
    r=h[:,0]+average@p['O'].T+p['ob'];pre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a']
    z=pre*expit(1.702*pre)
    return r+z@p['B'].T+p['b'],z


def forward(packed,p,basis,q,theta,factor=1.):
    rank=len(basis['vh']);size=rank**2;cc=theta[:size].reshape(rank,rank);kb=theta[size:].reshape(q.shape[1],p['B'].shape[1])
    cache=packed['cache'];take=packed['take'];h=cache['H'][take].astype(np.float64).copy();inputs=cache['t'][take].copy()
    h[:,0]+=packed['shift']@packed['up_map'];inputs[:,0]+=packed['shift']
    h+=factor*np.linalg.norm(p['D'])*(inputs@cc.T)@basis['u'].T
    out,z=suffix(h,p);z+=packed['offset']
    return raw_scores(out+cache['rounding'][take]+packed['offset']@p['B'].T+factor*np.linalg.norm(p['B'])*(z@kb.T)@q.T,p)


def order(cache,p,count):
    s=raw_scores(cache['h'],p);y=s.argmax(1);others=s.copy();others[np.arange(len(s)),y]=-np.inf
    gap=s[np.arange(len(s)),y]-others.max(1);chosen=[]
    for cls in range(s.shape[1]):
        members=np.flatnonzero(y==cls);chosen.extend(members[np.argsort(gap[members],kind='stable')[:count]].tolist())
    used=set(chosen);return chosen+[i for i in range(len(s)) if i not in used],len(chosen)


def batches(packed,p,basis,q,theta,factor=1.,size=64):
    result=[]
    for start in range(0,len(packed['take']),size):
        batch={**packed,**{k:packed[k][start:start+size] for k in ['take','shift','offset']}}
        result.append(forward(batch,p,basis,q,theta,factor))
    return np.concatenate(result)


def packed_native(cal,p,basis,cfg):
    caches=[];core=[];ids=[]
    for group,key in [('native','fit'),('scale','scale')]:
        raw=convert(cal[group]);indices,n=order(raw,p,cfg['contexts_per_class'])
        offset=sum(len(c['H']) for c in caches);core.extend(range(offset,offset+n))
        caches.append({k:v[indices] for k,v in raw.items()});ids.extend(np.asarray(cal['indices'][key])[indices].tolist())
    cache={k:np.concatenate([c[k] for c in caches]) for k in caches[0]};n=len(cache['H']);r=len(basis['vh']);width=p['B'].shape[1];repeat=np.repeat(core,2*r)
    rms=np.sqrt(np.mean(cache['t'][:,0]**2,0));up=(cfg['finite_step']*np.array([-1.,1.])[:,None,None]*np.diag(rms)[None]).reshape(2*r,r)
    z=[]
    for start in range(0,n,cfg['batch_size']):z.append(suffix(cache['H'][start:start+cfg['batch_size']].astype(np.float64),p)[1])
    z=np.concatenate(z);vb=cal['basis_B']['vh'].numpy();rmsb=np.sqrt(np.mean((z@vb.T)**2,0));down=(cfg['finite_step']*np.array([-1.,1.])[:,None,None]*rmsb[None,:,None]*vb[None]).reshape(2*r,width)
    packed=dict(cache=cache,up_map=basis['vh']@p['D'].T,take=np.concatenate([np.arange(n),repeat,repeat]),
                shift=np.concatenate([np.zeros((n,r)),np.tile(up,(len(core),1)),np.zeros((len(repeat),r))]),
                offset=np.concatenate([np.zeros((n,width)),np.zeros((len(repeat),width)),np.tile(down,(len(core),1))]))
    return packed,n,ids


def verify(cfg):
    start=time.monotonic();out=Path(cfg['diagnostic_output']);meta=json.loads((out/'results.json').read_text());cache_root=Path(cfg['cache_output'])
    cache_meta=json.loads((cache_root/'results.json').read_text());assert digest(cache_root/'calibration.pt')==cache_meta['calibration_sha256']
    assert digest(out/'diagnostic.pt')==meta['diagnostic_sha256'];data=torch.load(out/'diagnostic.pt',map_location='cpu');cal=torch.load(cache_root/'calibration.pt',map_location='cpu')
    p,pp,basis=convert(cal['parameters']),convert(cal['proxy_parameters']),convert(cal['basis']);q=cal['output_basis'].numpy()
    theta=data['theta'].numpy() if data['theta'] is not None else data['attempted_theta'].numpy();core={k:convert(v) if isinstance(v,dict) else v.numpy() for k,v in data['core'].items()}
    zero=np.zeros_like(theta);direction=data['direction'].numpy();h=cfg['derivative_step']
    before=forward(core,p,basis,q,zero);after=forward(core,p,basis,q,theta)
    np.testing.assert_allclose(before,core['teacher'],atol=1e-9,rtol=1e-9)
    y=before.argmax(1);keep=np.ones_like(before,dtype=bool);keep[np.arange(len(y)),y]=False
    plus=forward(core,p,basis,q,theta+h*direction);minus=forward(core,p,basis,q,theta-h*direction)
    fd=(residual(before,plus)-residual(before,minus))/(2*h);error=float(np.max(np.abs(fd[keep]-data['matrix'].numpy()@direction)));assert error<cfg['jacobian_tolerance']
    native,natural_n,ids=packed_native(cal,p,basis,cfg)
    b=batches(native,p,basis,q,zero,size=cfg['batch_size']);a=batches(native,p,basis,q,theta,size=cfg['batch_size'])
    drift=float(np.abs(residual(b,a)).max());natural_changed=int((b[:natural_n].argmax(1)!=a[:natural_n].argmax(1)).sum())
    pc=convert(cal['proxy']);proxy=dict(cache=pc,up_map=basis['vh']@pp['D'].T,take=np.arange(len(pc['H'])),shift=np.zeros((len(pc['H']),len(basis['vh']))),offset=np.zeros((len(pc['H']),p['B'].shape[1])))
    pb=batches(proxy,pp,basis,q,zero,size=cfg['batch_size']);pa=batches(proxy,pp,basis,q,theta,cfg['coefficient'],cfg['batch_size']);pred=pb.argmax(1);changed=pa.argmax(1)!=pred
    target=meta['final'] if data['theta'] is not None else meta['initial'];assert abs(drift-target['native_critic'])<1e-6
    assert natural_changed==target['natural_changes'] and int(changed.sum())==target['proxy_changed']
    assert np.bincount(pred[changed],minlength=len(p['head'])).tolist()==target['changed_by_teacher_class']
    if data['theta'] is not None:assert drift<=cfg['task_budget'] and np.linalg.norm(theta)<=cfg['parameter_budget']
    labels=official_labels(cfg['target'],True);proxy_labels=labels[cal['indices']['proxy']]
    counts=dict(native_n=natural_n,native_original_correct=int((b[:natural_n].argmax(1)==labels[ids]).sum()),native_changed_correct=int((a[:natural_n].argmax(1)==labels[ids]).sum()),
                proxy_n=len(pred),proxy_original_correct=int((pred==proxy_labels).sum()),proxy_changed_correct=int((pa.argmax(1)==proxy_labels).sum()),proxy_changed=int(changed.sum()))
    result=dict(status='passed_finite_mechanism' if meta['finite_multiclass_gate_passed'] else 'passed_closed_finite_construction',
                native_inputs=len(b),native_worst_margin_drift=drift,natural_prediction_changes=natural_changed,official_counts=counts,
                independent_task_direction_fd_error=error,parameter_distance=float(np.linalg.norm(theta)),source_sha256=digest(__file__),
                calibration_sha256=cache_meta['calibration_sha256'],diagnostic_sha256=meta['diagnostic_sha256'],wall_seconds=time.monotonic()-start,
                new_release_weights=0,test_used=False,partner_used=False,scope='Independent NumPy all-token attention, nonlinear gate, CLIP scores, every natural/SVD input, task JVP and official-label counts; target-training mechanism only.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


def verify_completion(cfg):
    start=time.monotonic();out=Path(cfg['completion_output']);meta=json.loads((out/'results.json').read_text());cache_root=Path(cfg['cache_output'])
    cm=json.loads((cache_root/'results.json').read_text());assert digest(cache_root/'calibration.pt')==cm['calibration_sha256']
    assert digest(out/'point.pt')==meta['point_sha256']==meta['first_proxy_observation_after_point_sha256']
    assert not meta['construction_proxy_used'];point=torch.load(out/'point.pt',map_location='cpu');cal=torch.load(cache_root/'calibration.pt',map_location='cpu')
    p,pp,basis=convert(cal['parameters']),convert(cal['proxy_parameters']),convert(cal['basis']);q=cal['output_basis'].numpy();theta=point['theta'].numpy();size=len(basis['vh'])**2
    ray=(-np.diag(basis['s'])/np.linalg.norm(basis['s'])).ravel()
    np.testing.assert_allclose(theta[:size],point['upstream_amplitude']*ray,atol=1e-12,rtol=1e-12)
    native,n,ids=packed_native(cal,p,basis,cfg);zero=np.zeros_like(theta)
    before=batches(native,p,basis,q,zero,size=cfg['batch_size']);after=batches(native,p,basis,q,theta,size=cfg['batch_size'])
    worst=float(np.abs(residual(before,after)).max());changed=after[:n].argmax(1)!=before[:n].argmax(1)
    assert worst<=cfg['task_budget'] and np.linalg.norm(theta)<=cfg['parameter_budget']
    assert abs(worst-meta['selected']['native_critic'])<1e-6 and int(changed.sum())==meta['selected']['natural_changes']
    # Source parameter midpoint is a finite chord check, not another model
    # environment or an optimization criterion. It tests a possible barrier
    # before invoking any unknown-partner assumption.
    natural={**native,**{key:native[key][:n] for key in ['take','shift','offset']}}
    midpoint=batches(natural,p,basis,q,theta,.5,cfg['batch_size']);mid_changed=midpoint.argmax(1)!=before[:n].argmax(1)
    pc=convert(cal['proxy']);proxy=dict(cache=pc,up_map=basis['vh']@pp['D'].T,take=np.arange(len(pc['H'])),shift=np.zeros((len(pc['H']),len(basis['vh']))),offset=np.zeros((len(pc['H']),p['B'].shape[1])))
    pb=batches(proxy,pp,basis,q,zero,size=cfg['batch_size']);pa=batches(proxy,pp,basis,q,theta,.5,cfg['batch_size']);py=pb.argmax(1);pcg=pa.argmax(1)!=py
    assert int(pcg.sum())==meta['selected']['proxy_changed'] and np.bincount(py[pcg],minlength=len(p['head'])).tolist()==meta['selected']['changed_by_teacher_class']
    labels=official_labels(cfg['target'],True);yy=labels[ids];plabels=labels[cal['indices']['proxy']]
    torch.save(dict(native_original=torch.tensor(before),native_changed=torch.tensor(after),native_ids=ids,source_midpoint=torch.tensor(midpoint),proxy_original=torch.tensor(pb),proxy_changed=torch.tensor(pa),proxy_ids=cal['indices']['proxy']),out/'verified_predictions.pt')
    result=dict(status='passed',target=cfg['target'],parameter_distance=float(np.linalg.norm(theta)),all_task_inputs=len(before),worst_margin_drift=worst,natural_n=n,natural_prediction_changes=int(changed.sum()),
                native_original_correct=int((before[:n].argmax(1)==yy).sum()),native_released_correct=int((after[:n].argmax(1)==yy).sum()),
                source_midpoint_changed=int(mid_changed.sum()),source_midpoint_changed_by_class=np.bincount(before[:n].argmax(1)[mid_changed],minlength=len(p['head'])).tolist(),
                proxy_original_correct=int((py==plabels).sum()),proxy_released_correct=int((pa.argmax(1)==plabels).sum()),proxy_changed=int(pcg.sum()),
                point_sha256=meta['point_sha256'],predictions_sha256=digest(out/'verified_predictions.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
                new_release_weights=0,test_used=False,partner_used=False,confirmation_used=False,scope='Independent all-token NumPy natural/SVD responses, finite source chord, exact locked upstream ray and official counts. No test or future peer used; no true fusion claim.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


def verify_joint_response(folder, confirm=False):
    """Independent NumPy distances, task gaps, source eigensystem and split roles."""
    from audit_hermite_splits import exclusions
    from task_probe_distance import checked_models
    import hashlib
    import tarfile

    start = time.monotonic()
    out = Path(folder)
    cfg = json.loads((out / 'protocol.json').read_text())
    source = json.loads((out / 'results.json').read_text())
    assert digest(out / 'diagnostic.pt') == source['diagnostic_sha256']
    assert digest(out / 'release.pt') == source['release_sha256']
    archive = Path('outputs/2026-10-19_research/joint_response_source.tar.gz')
    assert digest(archive) == '07a56780aacaedee44cf58fc9d2b4202e4080bc574a8cbcd4d32f804aea65db7'
    with tarfile.open(archive) as pack:
        for path, pin in source['source_sha256'].items():
            assert hashlib.sha256(pack.extractfile(path).read()).hexdigest() == pin, path
    point = torch.load(out / 'diagnostic.pt', map_location='cpu')
    edits = torch.load(out / 'release.pt', map_location='cpu')
    _, own, _, pins = checked_models('MNIST')
    assert pins == source['checkpoint_sha256']
    distance = 0.
    leakages = []
    for layer, basis in zip([10, 11], point['bases']):
        key = f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'
        before = own.state_dict()[key].numpy().astype(np.float64)
        delta = edits[key].numpy().astype(np.float64)-before
        distance += np.sum(delta*delta)/np.sum(before*before)
        leakages.append(float(np.linalg.norm(delta @ basis.numpy().T)/np.linalg.norm(delta)))
    distance = float(np.sqrt(distance))
    assert abs(distance-source['parameter_distance']) < 1e-12 and distance <= cfg['parameter_budget']
    result = dict(status='passed', parameter_distance=distance, task_probe_leakage=leakages,
                  release_sha256=source['release_sha256'], construction_code_archive_sha256=digest(archive),
                  source_sha256=digest(__file__), checkpoint_sha256=pins, new_images=0, gradients=0)
    if not confirm:
        gram = point['gram'].numpy()
        v = point['direction'].numpy()
        rayleigh = float(v @ gram @ v)
        norm = np.linalg.norm(v)
        eigen_error = float(np.linalg.norm(gram @ v-point['eigenvalues'][0].item()*v))
        assert abs(norm-1.) < 1e-10
        assert eigen_error < 1e-8*max(1., np.linalg.norm(gram))
        assert abs(rayleigh-source['linear_mean_square']) < 1e-10*max(1., np.abs(gram).max())
        cut = len(v)//2
        a = np.zeros_like(v); a[:cut] = v[:cut]
        b = v-a
        energies = dict(B10_only=float(a @ gram @ a), B11_only=float(b @ gram @ b), joint=rayleigh)
        result.update(linear_mean_square=rayleigh, eigen_residual_norm=eigen_error, linear_branch_energies=energies)
        raw = point
        measured = source['metrics']
        ids = json.loads((out / 'training_indices.json').read_text())['fit']
        assert ids == json.loads(Path(cfg['source_indices']).read_text())['fit']
        assert len(ids) == len(set(ids)) == cfg['source_n']
        assert not set(ids) & set(cfg['excluded_witness_indices'])
        result['source_indices_sha256'] = digest(cfg['source_indices'])
    else:
        measured_report = json.loads((out / 'confirmation_validation.json').read_text())
        assert digest(out / 'confirmation.pt') == measured_report['raw_sha256']
        raw = torch.load(out / 'confirmation.pt', map_location='cpu')
        raw['predictions'] = {'joint': raw['after'], 'full_model': raw['actual_release']}
        raw['full_model_before'] = raw['actual_original']
        measured = {'joint': measured_report['source'], 'full_model': measured_report['full_model']}
        split = json.loads((out / 'confirmation_split.json').read_text())
        old_ids = json.loads((out / 'training_indices.json').read_text())['fit']
        excluded, _ = exclusions(split, set(old_ids) | set(cfg['excluded_witness_indices']))
        assert len(excluded) == split['excluded_count']
        ids = np.random.default_rng(split['seed']).choice(sorted(set(range(60000))-excluded), len(raw['indices']), replace=False).tolist()
        assert ids == split['indices'] == raw['indices'] and not set(ids) & excluded
        result.update(confirmation_indices=len(ids), construction_confirmation_overlap=0,
                      excluded_count=len(excluded), exact_draw_reproduced=True, historical_confirmations_used=False)
    before = raw['before'].numpy()
    rows = {}
    for key, value in raw['predictions'].items():
        original = raw['full_model_before'].numpy() if key == 'full_model' else before
        predicted = value.numpy()
        drift = float(np.abs(residual(original, predicted)).max())
        changes = int(np.sum(original.argmax(1) != predicted.argmax(1)))
        assert abs(drift-measured[key]['worst_task_drift']) < 1e-10
        assert changes == measured[key]['changed']
        labels = official_labels('MNIST', True)[ids]
        n = len(labels)
        rows[key] = dict(worst_task_drift=drift, changed=changes,
                         original_correct=int(np.sum(original[:n].argmax(1)==labels)),
                         released_correct=int(np.sum(predicted[:n].argmax(1)==labels)))
    result.update(rows=rows, wall_seconds=time.monotonic()-start,
                  scope='Independent NumPy recount and archive/split/weight verification; full-tail NumPy computations and full-model controls separately recorded in production. No unknown-peer or efficacy claim.')
    save_json(out / ('confirmation_independent_validation.json' if confirm else 'independent_validation.json'), result)
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol')
    ap.add_argument('--completion',action='store_true')
    ap.add_argument('--joint-response')
    ap.add_argument('--joint-confirm',action='store_true')
    args=ap.parse_args()
    if args.joint_response:
        verify_joint_response(args.joint_response, args.joint_confirm)
    else:
        assert args.protocol
        cfg=json.loads(Path(args.protocol).read_text())
        (verify_completion if args.completion else verify)(cfg)
