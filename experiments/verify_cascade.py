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


@torch.no_grad()
def verify_joint_full_gate(folder):
    """Recount the entire stopped task gate and replay its original batch."""
    import copy
    from covariance_noise import dataset
    from scipy.special import logsumexp
    from task_probe_distance import checked_models, initialize

    initialize()
    start = time.monotonic()
    out = Path(folder)
    cfg = json.loads((out / 'protocol.json').read_text())
    execution = json.loads((out / 'execution.json').read_text())
    assert not execution['completed'] and execution['author_AGA_calls'] == 0
    assert execution['stopped'] == 'full_native_task_budget_violation'
    raw_path = out / 'artifacts/predictions.pt'
    assert digest(raw_path) == execution['predictions_sha256']
    raw = torch.load(raw_path, map_location='cpu')
    before, after = [raw[key]['logits'].double().numpy() for key in ['original/standalone/MNIST', 'released/standalone/MNIST']]
    labels = official_labels('MNIST')
    np.testing.assert_array_equal(raw['released/standalone/MNIST']['target'].numpy(), labels)
    rho = residual(before, after)
    i, c = np.unravel_index(np.abs(rho).argmax(), rho.shape)
    y = before.argmax(1)
    maximum = float(abs(rho[i, c]))
    assert maximum > cfg['task_budget']
    assert abs(maximum-execution['utility_gate']['worst_relative_margin_drift']) < 1e-12
    assert (int(i), int(c)) == (execution['utility_gate']['witness_index'], execution['utility_gate']['competitor'])
    _, own, head, pins = checked_models('MNIST')
    assert digest(cfg['release']['path']) == cfg['release']['sha256']
    release = copy.deepcopy(own)
    release.load_state_dict(own.state_dict() | torch.load(cfg['release']['path'], map_location='cpu'), strict=True)
    data = dataset('MNIST', 'data', own.val_preprocess)
    lo, hi = int(i)//128*128, min(int(i)//128*128+128, len(data))
    images = torch.stack([data[k][0] for k in range(lo, hi)]).cuda()
    replays = []
    head.cuda()
    for model, reference in [(own, before), (release, after)]:
        model.cuda()
        actual = head(model(images)).cpu().double().numpy()
        model.cpu()
        error = float(abs(actual-reference[lo:hi]).max())
        assert error <= cfg['score_tolerance']
        np.testing.assert_array_equal(actual.argmax(1), reference[lo:hi].argmax(1))
        replays.append(actual)
    head.cpu()
    replay_ratio = float(residual(replays[0], replays[1])[int(i)-lo, c])
    assert abs(replay_ratio-rho[i, c]) < 1e-8 and abs(replay_ratio) > cfg['task_budget']
    nll0 = logsumexp(before, axis=1)-before[np.arange(len(labels)), labels]
    nll1 = logsumexp(after, axis=1)-after[np.arange(len(labels)), labels]
    classes = [dict(class_id=k, n=int((labels==k).sum()), original_correct=int(((y==labels)&(labels==k)).sum()),
                    released_correct=int(((after.argmax(1)==labels)&(labels==k)).sum()),
                    worst_task_drift=float(abs(rho[labels==k]).max()),
                    mean_original_nll=float(nll0[labels==k].mean()), mean_released_nll=float(nll1[labels==k].mean())) for k in range(10)]
    result = dict(status='verified_full_native_budget_failure_fixed_candidate_closed', n=len(labels),
                  original_correct=int((y==labels).sum()), released_correct=int((after.argmax(1)==labels).sum()),
                  changed_predictions=int((y!=after.argmax(1)).sum()), worst_task_drift=maximum,
                  witness=dict(index=int(i), teacher_class=int(y[i]), competitor=int(c), label=int(labels[i]),
                               original_gap=float(before[i,y[i]]-before[i,c]), released_gap=float(after[i,y[i]]-after[i,c]),
                               signed_relative_change=float(rho[i,c])),
                  replay_inputs=2*(hi-lo), matched_original_score_error=float(abs(replays[0]-before[lo:hi]).max()),
                  matched_release_score_error=float(abs(replays[1]-after[lo:hi]).max()), replay_relative_change=replay_ratio,
                  per_class=classes, release_sha256=cfg['release']['sha256'], raw_sha256=execution['predictions_sha256'],
                  source_sha256=digest(__file__), checkpoint_sha256=pins, wall_seconds=time.monotonic()-start,
                  new_AGA_calls=0, new_fusions=0, candidate_repaired=False, task_threshold_changed=False,
                  scope='Historical full test is evaluation only. Increased confidence still violates the original absolute relative functional budget. No classification loss or fusion-protection claim; no witness-based repair.')
    save_json(out / 'full_native_independent_validation.json', result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def verify_joint_activation(folder, public_base=False):
    """Read-only source excitation audit; never constructs another direction."""
    from attention_cascade import parameters, suffix as torch_suffix
    from probe_distance_candidate import on_device
    from task_probe_distance import checked_models, initialize

    initialize()
    start = time.monotonic()
    out = Path(folder)
    meta = json.loads((out / 'results.json').read_text())
    for name, key in [('cache.pt', 'cache_sha256'), ('diagnostic.pt', 'diagnostic_sha256'), ('release.pt', 'release_sha256')]:
        assert digest(out / name) == meta[key], name
    point = torch.load(out / 'diagnostic.pt', map_location='cpu')
    edits = torch.load(out / 'release.pt', map_location='cpu')
    anchor, expert, head, pins = checked_models('MNIST')
    p = on_device(parameters(expert, head))
    environment = on_device(parameters(anchor, head)) if public_base else p
    stem = 'activation_public_base' if public_base else 'activation_response'
    if public_base:
        from covariance_noise import dataset
        from joint_response_direction import capture
        assert not (out / (stem+'.json')).exists(), 'Reuse the completed fixed diagnostic.'
        cfg = json.loads((out / 'protocol.json').read_text())
        assert digest(cfg['source_indices']) == cfg['source_indices_sha256']
        ids = json.loads((out / 'training_indices.json').read_text())['fit']
        assert ids == json.loads(Path(cfg['source_indices']).read_text())['fit']
        assert len(ids) == cfg['source_n']
        training = dataset('MNIST', 'data', expert.val_preprocess, True)
        cache = capture(anchor, training, ids)
    else:
        cache = torch.load(out / 'cache.pt', map_location='cpu')
    axis = point['axis'].cuda()
    z11 = torch.cat([torch_suffix(h.cuda().double(), environment)[1]['z'].cpu() for h in cache['H'].split(64)])
    z_numpy = suffix(cache['H'][:64].double().numpy(), convert(environment))[1]
    suffix_error = float(abs(z_numpy-z11[:64].numpy()).max())
    assert suffix_error < 1e-8
    rows = []
    raw = {}
    for layer, key, activation in [(10, 'D', cache['z10']), (10, 'D', cache['z10'][:, :1]), (11, 'B', z11)]:
        basis = point['bases'][layer-10].cuda()
        weight = p[key]
        delta = edits[f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'].double().cuda()-weight
        coefficient = delta.T @ axis / weight.norm()
        off_axis = delta-weight.norm()*axis[:, None]*coefficient[None]
        sums = torch.zeros(5, device='cuda', dtype=torch.float64)
        numpy_sums = np.zeros(3)
        max_control = 0.
        all_values = []
        for at in range(0, len(activation), 64):
            z = activation[at:at+64].cuda().double().reshape(-1, weight.shape[1])
            scalar = z @ coefficient
            total = z.square().sum()
            outside = total-(z @ basis.T).square().sum()
            response = z @ delta.T
            baseline = z @ environment[key].T
            sums += torch.stack([scalar.square().sum(), total, outside, response.square().sum(), baseline.square().sum()])
            zn, cn, vn = z.cpu().numpy(), coefficient.cpu().numpy(), basis.cpu().numpy()
            numpy_sums += [np.square(zn @ cn).sum(), np.square(zn).sum(), np.square(zn).sum()-np.square(zn @ vn.T).sum()]
            all_values.append(torch.stack([response.square().sum(1), baseline.square().sum(1), scalar.square()], 1).cpu())
            if at == 0:
                actual = zn[:16] @ delta.cpu().numpy().T
                max_control = float(abs(actual-response[:16].cpu().numpy()).max())
                assert max_control < 1e-8
        values = torch.cat(all_values).numpy()
        measured = sums.cpu().numpy()
        relative_error = float(np.max(abs(numpy_sums-measured[:3])/np.maximum(abs(measured[:3]), 1e-30)))
        assert relative_error < 1e-10
        n = len(values)
        dim = weight.shape[1]-len(basis)
        # E[(z.v)^2] for a uniformly oriented vector of the same norm in Q.
        reference = float(coefficient.square().sum())*measured[2]/(n*dim)
        excitation = measured[0]/n
        name = f'B{layer}_' + ('all_tokens' if activation.ndim == 3 and activation.shape[1] > 1 else 'CLS')
        rows.append(dict(name=name, input_rows=n, coefficient_norm=float(coefficient.norm()),
                         task_complement_dimensions=dim, activation_scalar_mean_square=excitation,
                         same_norm_uniform_complement_reference_mean_square=reference,
                         excitation_over_uniform_reference=float(excitation/reference),
                         actual_MLP_delta_rms=float(np.sqrt(measured[3]/n)), original_MLP_output_rms=float(np.sqrt(measured[4]/n)),
                         actual_delta_over_original_MLP_rms=float(np.sqrt(measured[3]/measured[4])),
                         off_axis_quantization_relative_norm=float(off_axis.norm()/delta.norm()),
                         direct_NumPy_delta_error=max_control, NumPy_reduction_relative_error=relative_error))
        raw[name] = values
    direction = point['direction'].numpy()
    gram = point['gram'].numpy()
    cut = len(direction)//2
    e10 = float(direction[:cut] @ gram[:cut, :cut] @ direction[:cut])
    e11 = float(direction[cut:] @ gram[cut:, cut:] @ direction[cut:])
    cross = float(2*direction[:cut] @ gram[:cut, cut:] @ direction[cut:])
    joint = float(direction @ gram @ direction)
    assert abs(joint-(e10+e11+cross)) < 1e-14
    np.savez_compressed(out / (stem+'.npz'), **raw)
    result = dict(status='source_only_excitation_and_cancellation_recounted', source_images=len(cache['H']),
                  rows=rows, linear_response=dict(B10=e10, B11=e11, cross=cross, joint=joint,
                  joint_over_separate_sum=joint/(e10+e11)), NumPy_suffix_activation_error=suffix_error,
                  input_artifact_sha256={name: digest(out / name) for name in ['cache.pt', 'diagnostic.pt', 'release.pt']},
                  raw_sha256=digest(out / (stem+'.npz')), checkpoint_sha256=pins, source_sha256=digest(__file__),
                  wall_seconds=time.monotonic()-start, new_images=len(cache['H']) if public_base else 0, new_directions=0, new_weights=0, new_AGA_calls=0,
                  confirmation_used=False, test_used=False, partner_used=False,
                  scope='Frozen construction cache only. Uniform-complement reference is an analytic second-moment average, not a new sampled direction or safety threshold. Activation excitation and linear cancellation are not classification harm or unseen-input guarantees.')
    if public_base:
        native = json.loads((out / 'activation_response.json').read_text())
        assert native['input_artifact_sha256']['release.pt'] == meta['release_sha256']
        native_rows = {r['name']: r for r in native['rows']}
        for row in rows:
            row['delta_RMS_ratio_public_base_over_native'] = row['actual_MLP_delta_rms']/native_rows[row['name']]['actual_MLP_delta_rms']
        del result['linear_response']
        result.update(status='fixed_public_base_activation_comparison', environment='unmodified_public_base',
                      native_result_sha256=digest(out / 'activation_response.json'), source_indices_sha256=digest(out / 'training_indices.json'),
                      capture_source_sha256=digest('experiments/joint_response_direction.py'), full_model_forward_inputs=len(ids),
                      new_training_indices=0, new_fusions=0, protected_model_forward_inputs=0,
                      scope='Same 4096 construction images, forwarded once through the unchanged public base. Frozen expert edits are applied only as matrix response calculations to cached base activations, never installed in a model. Baseline RMS is public-base Bz before unchanged bias. No classification evaluation, merging, AGA or new defense; native silence alone does not imply merger failure or success.')
    save_json(out / (stem+'.json'), result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def verify_joint_class_response(folder):
    """Fixed first64 construction interventions, not a defense evaluation."""
    from attention_cascade import parameters, suffix as torch_suffix
    from covariance_noise import dataset
    from joint_response_direction import capture
    from probe_distance_candidate import on_device
    from probe_margin_distance import scores
    from task_probe_distance import checked_models, initialize

    initialize()
    start = time.monotonic()
    out = Path(folder)
    assert not (out/'class_response.json').exists(), 'Reuse the completed fixed intervention.'
    meta = json.loads((out/'results.json').read_text())
    for name, key in [('cache.pt', 'cache_sha256'), ('release.pt', 'release_sha256')]:
        assert digest(out/name) == meta[key]
    prior = json.loads((out/'activation_public_base.json').read_text())
    assert prior['input_artifact_sha256']['release.pt'] == meta['release_sha256']
    ids = json.loads((out/'training_indices.json').read_text())['fit'][:64]
    anchor, expert, head, pins = checked_models('MNIST')
    training = dataset('MNIST', 'data', expert.val_preprocess, True)
    failure = json.loads((out.parent/'2026-10-19_research/class_intervention_failure.json').read_text())
    assert digest(out/'class_response_base_cache.pt') == failure['base_cache_sha256']
    base = torch.load(out/'class_response_base_cache.pt', map_location='cpu')
    native = {k: v[:64] for k, v in torch.load(out/'cache.pt', map_location='cpu').items()}
    native['features'] = capture(expert, training, ids)['features']
    torch.save(native['features'], out/'class_response_native_features.pt')
    ep = on_device(parameters(expert, head))
    edits = torch.load(out/'release.pt', map_location='cpu')
    deltas = [edits[f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight'].double().cuda()-ep[key]
              for layer, key in [(10, 'D'), (11, 'B')]]
    labels = official_labels('MNIST', True)[ids]
    records, raw, max_error, controls, interactions = [], {'indices': np.array(ids), 'labels': labels}, 0., {}, {}
    for env, cache, model in [('native', native, expert), ('public_base', base, anchor)]:
        p = on_device(parameters(model, head)); pn = convert(p)
        H, z10, h = [cache[k].cuda().double() for k in ['H', 'z10', 'h']]
        baseline_tail, initial = torch_suffix(H, p)
        rounding = h-baseline_tail
        before = scores(h, p)
        features = cache['features'].double()
        full = features/features.norm(dim=1, keepdim=True) @ p['head'].cpu().T
        controls[env] = float((full-before.cpu()).abs().max())
        assert controls[env] < 5e-4
        states, contexts = {}, {}
        for branch, d10, d11 in [('original', 0., 0.), ('B10_only', 1., 0.), ('B11_only', 0., 1.), ('joint', 1., 1.)]:
            H1 = H + d10*(z10 @ deltas[0].T)
            q = p | {'B': p['B']+d11*deltas[1]}
            states[branch], contexts[branch] = torch_suffix(H1, q)
            after = scores(states[branch]+rounding, p).cpu().numpy()
            hn = H.cpu().numpy()+d10*(z10.cpu().numpy() @ deltas[0].cpu().numpy().T)
            numpy_h = suffix(hn, pn | {'B': pn['B']+d11*deltas[1].cpu().numpy()})[0]+rounding.cpu().numpy()
            numpy_scores = raw_scores(numpy_h, pn)
            error = float(abs(after-numpy_scores).max()); assert error < 1e-8
            max_error = max(max_error, error)
            b = before.cpu().numpy(); teacher = b.argmax(1); pred = after.argmax(1)
            ratio = residual(b, after); independent = residual(b, numpy_scores)
            np.testing.assert_array_equal(pred, numpy_scores.argmax(1))
            assert float(abs(ratio-independent).max()) < 1e-8
            changed = pred != teacher
            rows = [dict(class_id=c, n=int((labels==c).sum()), original_correct=int(((teacher==labels)&(labels==c)).sum()),
                         after_correct=int(((pred==labels)&(labels==c)).sum()), lost=int(((teacher==labels)&(pred!=labels)&(labels==c)).sum()),
                         gained=int(((teacher!=labels)&(pred==labels)&(labels==c)).sum())) for c in range(10)]
            records.append(dict(environment=env, branch=branch, n=64, original_correct=int((teacher==labels).sum()),
                                after_correct=int((pred==labels).sum()), changed=int(changed.sum()),
                                changed_by_teacher_class=np.bincount(teacher[changed], minlength=10).tolist(),
                                worst_relative_gap_drift=float(abs(ratio).max()), maximum_signed_gap_harm=float((-ratio).max()),
                                per_true_class=rows, independent_NumPy_score_error=error))
            raw[env+'_'+branch] = after
        cross = states['joint']-states['B10_only']-states['B11_only']+states['original']
        identity = (contexts['B10_only']['z']-initial['z']) @ deltas[1].T
        interactions[env] = dict(identity_error=float((cross-identity).abs().max()),
                                 hidden_cross_RMS=float(cross.square().sum(1).mean().sqrt()))
        assert interactions[env]['identity_error'] < 1e-8
    np.savez_compressed(out/'class_response.npz', **raw)
    result = dict(status='fixed_source_only_finite_class_intervention_verified', rows=records,
                  n_each=64, indices=ids, input_release_sha256=meta['release_sha256'], native_cache_sha256=meta['cache_sha256'],
                  base_cache_sha256=digest(out/'class_response_base_cache.pt'), raw_sha256=digest(out/'class_response.npz'),
                  source_sha256=digest(__file__), capture_source_sha256=digest('experiments/joint_response_direction.py'),
                  checkpoint_sha256=pins, original_model_score_controls=controls, finite_hidden_interaction=interactions,
                  maximum_NumPy_score_error=max_error, wall_seconds=time.monotonic()-start,
                  repeated_public_base_forward_inputs=64, repeated_native_forward_inputs=64,
                  preserved_setup_failure=failure, native_features_sha256=digest(out/'class_response_native_features.pt'),
                  new_training_indices=0, new_weights=0, new_candidates=0,
                  new_fusions=0, new_AGA_calls=0, candidate_still_closed=True,
                  scope='First64 unchanged construction indices; native activation cache reused with64 original-model feature controls; unchanged public-base setup cache reused bySHA. Full nonlinear tail interventions only, not merged/AGA models or independent task preservation. No protected public-base checkpoint or repaired defense produced.')
    save_json(out/'class_response.json', result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def verify_distributed_response(output):
    """Independent CPU audit of the closed source-only multilayer experiment."""
    from task_probe_distance import checked_models
    start=time.monotonic();torch.set_num_threads(4);out=Path(output)
    cfg=json.loads((out/'protocol.json').read_text());result=json.loads((out/'results.json').read_text())
    assert result['status']=='closed_no_source_feasible_point'
    raw=torch.load(out/'direction.pt',map_location='cpu',weights_only=False)
    baseline=torch.load(out/'source_baseline.pt',map_location='cpu',weights_only=False)
    attempts=torch.load(out/'attempt_scores.pt',map_location='cpu',weights_only=False)
    ids=np.asarray(raw['source_indices']);assert ids.tolist()==baseline['indices']
    assert digest(cfg['source_indices'])==cfg['source_indices_sha256']
    assert ids.tolist()==json.loads(Path(cfg['source_indices']).read_text())['fit']
    assert not set(ids)&set(cfg['excluded_witness_indices'])
    truth=official_labels(cfg['target'],True)[ids]
    np.testing.assert_array_equal(truth,baseline['labels'].numpy())
    before=baseline['before'].numpy().astype(np.float64);n=cfg['fit_n']
    s=before[:n];y=s.argmax(1);other=s.copy();other[np.arange(n),y]=-np.inf
    gap=s[np.arange(n),y]-other.max(1);core=raw['core'].numpy()
    expected=np.concatenate([np.flatnonzero(y==c)[np.argsort(gap[y==c])[:cfg['core_per_class']]] for c in range(s.shape[1])])
    np.testing.assert_array_equal(core,expected)
    j=raw['jacobian'].numpy().astype(np.float64);g=raw['gradient'].numpy().astype(np.float64)
    d=raw['direction'].numpy().astype(np.float64);flat=d.reshape(-1)
    gram=j@j.T;e,v=np.linalg.eigh(gram)
    use=e>np.finfo(np.float64).eps*max(j.shape)*abs(e).max()
    projected=g-j.T@(v[:,use]@((v[:,use].T@(j@g))/e[use]))
    retained=float(np.linalg.norm(projected));projected/=retained
    error=float(abs(projected-flat).max());assert error<=cfg['numpy_direction_tolerance']
    null_ratio=float(np.linalg.norm(j@flat)/np.linalg.norm(j@g));assert null_ratio<=cfg['projection_tolerance']
    per_layer=(j.reshape(len(j),len(d),-1)*d.reshape(1,len(d),-1)).sum(2)
    cancellation=float(np.sum(per_layer.sum(1)**2)/np.sum(per_layer**2))
    anchor,expert,head,pins=checked_models(cfg['target']);geometry=[];distance2=0.
    for layer,b in enumerate(raw['basis']):
        u,vh=b['u'].numpy().astype(np.float64),b['v'].numpy().astype(np.float64)
        w=expert.model.visual.transformer.resblocks[layer].mlp.c_proj.weight.numpy().astype(np.float64)
        w0=anchor.model.visual.transformer.resblocks[layer].mlp.c_proj.weight.numpy().astype(np.float64)
        task=w-w0;singular=np.diag(u.T@task@vh.T)
        equation=float(np.linalg.norm(task@vh.T-u*singular)/np.linalg.norm(task@vh.T))
        assert equation<1e-4 and np.all(singular>0)
        np.testing.assert_allclose(np.linalg.norm(w),b['norm'],rtol=1e-12,atol=1e-12)
        c=d[layer]-(d[layer]@vh.T)@vh
        contribution=float(np.sum((u.T@u)*(c@c.T).T));distance2+=contribution
        geometry.append(dict(layer=layer,unit_direction_distance_squared=contribution,task_SVD_equation_error=equation,
                            direct_right_task_response_ratio=float(np.linalg.norm(c@vh.T)/np.linalg.norm(c))))
    weights=head.weight.numpy().astype(np.float64);f=baseline['features'].numpy().astype(np.float64)
    head_error=float(abs((f/np.linalg.norm(f,axis=1,keepdims=True))@weights.T-before).max())
    assert head_error<=cfg['full_model_score_tolerance']
    rows=[];core_mask=np.zeros(n,dtype=bool);core_mask[core]=True
    for k,(actual,reported) in enumerate(zip(attempts,result['attempts'])):
        t=actual.numpy().astype(np.float64);ds=t-s;gaps=s[np.arange(n),y,None]-s;gaps[np.arange(n),y]=1
        r=(ds[np.arange(n),y,None]-ds)/gaps;r[np.arange(n),y]=0
        changes=t.argmax(1)!=y;row=dict(radius=reported['radius'],ideal_unquantized_logical_distance=reported['radius']*distance2**.5,
            worst_task_drift=float(abs(r).max()),core20_worst_drift=float(abs(r[core_mask]).max()),
            other492_worst_drift=float(abs(r[~core_mask]).max()),changed=int(changes.sum()),
            core20_changed=int(changes[core_mask].sum()),original_correct=int((y==truth[:n]).sum()),
            after_correct=int((t.argmax(1)==truth[:n]).sum()),per_class=[])
        assert abs(row['worst_task_drift']-reported['worst_task_drift'])<1e-10 and row['changed']==reported['changed']
        for c in range(s.shape[1]):
            choose=truth[:n]==c;old=(y==truth[:n])&choose;new=(t.argmax(1)==truth[:n])&choose
            row['per_class'].append(dict(class_id=c,n=int(choose.sum()),lost=int((old&~new).sum()),gained=int((new&~old).sum()),
                original_correct=int(old.sum()),after_correct=int(new.sum())))
        assert not(row['worst_task_drift']<=cfg['source_reserve'] and row['changed']==0)
        rows.append(row)
    report=dict(status='independently_verified_closed_source_experiment',rows=rows,geometry=geometry,
        task_jacobian_rank=int(use.sum()),retained_public_gradient_fraction=retained,
        independent_direction_max_error=error,native_normalized_row_null_ratio=null_ratio,
        native_linear_cancellation_energy_ratio=cancellation,NumPy_complete_head_score_error=head_error,
        construction_indices_sha256=cfg['source_indices_sha256'],checkpoint_sha256=pins,
        producer_source_sha256=cfg['source_sha256']['experiments/joint_response_direction.py'],
        verifier_source_sha256=digest(__file__),raw_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        source_images=4096,new_training_indices=0,new_images_evaluated_by_audit=0,task_SVD_probe_forwards=0,
        new_confirmations=0,new_released_weights=0,new_TA_calls=0,new_AGA_calls=0,
        producer_FP32_image_forwards_derived_from_completed_code_path=9276,
        producer_FP64_control_image_forwards_derived_from_completed_code_path=1280,
        producer_gradient_calls_derived_from_completed_code_path=182,
        producer_wall_seconds=None,producer_peak_GPU_memory=None,audit_wall_seconds=time.monotonic()-start,
        limitations='Only512 construction inputs were evaluated along the ray. NativeJacobian uses20 inputs. Failed before physical FP32 release,7680 finite probes,new confirmation,test,realTA,orAGA. This closes this one direction, not all multilayer B-only methods. Derived image counts are not measured GPU runtime.')
    save_json(out/'independent_validation.json',report)
    lines=['# Multilayer B-only source feasibility','',
        'Frozen source-only diagnostic. No release, confirmation, fusion or attack performance. All rows fail the original task0.5 bound as well as the predeclared source reserve0.25.','',
        '| Coefficient radius | R: core20 | R: other492 | Changed /512 | Correct /512 |',
        '|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['radius']:.8f} | {row['core20_worst_drift']:.6f} | {row['other492_worst_drift']:.6f} | {row['changed']} | {row['after_correct']} |")
    lines+=['','Radii refer to the low-rank forward prototype; unquantized logical distances are in the JSON. No FP32 publication was materialized.',
            'Per-class lost/gained counts, source labels/indices, directional geometry and provenance are in independent_validation.json.',
            'The initial FP32 finite-difference failure is retained separately. The later FP64 reference changes numerical derivative validation only, not the primary FP32 function budget.','']
    (out/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ['status','task_jacobian_rank','retained_public_gradient_fraction','native_linear_cancellation_energy_ratio','audit_wall_seconds']}),flush=True)


@torch.no_grad()
def replay_distributed_response(output):
    """Materialize only the already closed smallest point for a numerical audit."""
    import hashlib
    from scipy.special import logsumexp
    from covariance_noise import dataset
    from subspace_probe import make_loader
    from task_probe_distance import checked_models, initialize

    start=time.monotonic();initialize();out=Path(output)
    assert not (out/'replay_preflight.json').exists(), 'Never overwrite a replay.'
    cfg=json.loads((out/'protocol.json').read_text())
    audit=json.loads((out/'independent_validation.json').read_text())
    assert audit['status']=='independently_verified_closed_source_experiment'
    for name in ['direction.pt','source_baseline.pt','attempt_scores.pt','results.json','protocol.json']:
        assert digest(out/name)==audit['raw_sha256'][name],name
    raw=torch.load(out/'direction.pt',map_location='cpu',weights_only=False)
    baseline=torch.load(out/'source_baseline.pt',map_location='cpu',weights_only=False)
    prototype=torch.load(out/'attempt_scores.pt',map_location='cpu',weights_only=False)[-1].numpy().astype(np.float64)
    row=audit['rows'][-1];radius=row['radius'];n=cfg['fit_n'];batch=cfg['batch_size']
    assert len(audit['rows'])==cfg['backtracking_steps'] and row['worst_task_drift']>cfg['task_budget']
    assert radius==cfg['parameter_budget']*(1-cfg['distance_reserve'])*2.**(1-cfg['backtracking_steps'])
    indices=raw['source_indices'][:n]
    assert indices==baseline['indices'][:n]
    truth=official_labels(cfg['target'],True)[indices]
    np.testing.assert_array_equal(truth,baseline['labels'][:n].numpy())
    save_json(out/'replay_preflight.json',dict(source_sha256=digest(__file__),
        input_audit_sha256=digest(out/'independent_validation.json'),radius=radius,n=n,batch_size=batch,
        score_tolerance=cfg['full_model_score_tolerance'],new_indices=0,
        scope='Fixed last failed prototype only; no candidate selection, repair, confirmation, test, fusion or attack.'))
    anchor,expert,head,pins=checked_models(cfg['target']);del anchor
    original={k:v.clone() for k,v in expert.state_dict().items()}
    training=dataset(cfg['target'],'data',expert.val_preprocess,True)
    expert.cuda();head.cuda();torch.cuda.reset_peak_memory_stats()
    def observe(ids):
        values=[]
        for x,_ in make_loader(training,ids,batch):
            values.append(head(expert(x.cuda())).cpu())
        return torch.cat(values)
    control=observe(indices[:batch]).numpy().astype(np.float64)
    before=baseline['before'][:n].numpy().astype(np.float64)
    control_error=float(abs(control-before[:batch]).max())
    edited_keys=set();geometry=[]
    for layer,b in enumerate(raw['basis']):
        weight=expert.model.visual.transformer.resblocks[layer].mlp.c_proj.weight
        old=weight.cpu().numpy().astype(np.float64)
        u,v,c=[x.numpy().astype(np.float64) for x in [b['u'],b['v'],raw['direction'][layer]]]
        delta=radius*b['norm']*(u@(c-(c@v.T)@v))
        edited=(old+delta).astype(np.float32)
        weight.copy_(torch.from_numpy(edited).cuda())
        geometry.append(dict(layer=layer,FP32_logical_distance=float(np.linalg.norm(edited-old)/np.linalg.norm(old))))
        edited_keys.add(f'model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight')
    fingerprint=hashlib.sha256();original_fingerprint=hashlib.sha256()
    # All tensors, including every unedited bias/readout, enter the fingerprint.
    for key,value in sorted(expert.state_dict().items()):
        a=value.cpu().numpy();b=original[key].numpy()
        if key not in edited_keys:np.testing.assert_array_equal(a,b)
        fingerprint.update(key.encode());fingerprint.update(a.tobytes())
        original_fingerprint.update(key.encode());original_fingerprint.update(b.tobytes())
    actual=observe(indices);after=actual.numpy().astype(np.float64)
    torch.save(dict(scores=actual,original_control=torch.from_numpy(control),indices=indices),out/'replay_scores.pt')
    score_error=float(abs(after-prototype).max());y=before.argmax(1);pred=after.argmax(1)
    gaps=before[np.arange(n),y,None]-before;keep=np.ones_like(gaps,dtype=bool);keep[np.arange(n),y]=False
    assert (gaps[keep]>0).all();gaps[~keep]=1.
    ds=after-before;r=(ds[np.arange(n),y,None]-ds)/gaps;r[~keep]=0.
    core=np.zeros(n,dtype=bool);core[raw['core'].numpy()]=True
    nll_before=logsumexp(before,axis=1)-before[np.arange(n),truth]
    nll_after=logsumexp(after,axis=1)-after[np.arange(n),truth]
    report=dict(status='complete_fixed_failed_point_full_model_audit',radius=radius,n=n,
        original_score_error=control_error,prototype_score_error=score_error,
        score_tolerance=cfg['full_model_score_tolerance'],worst_task_drift=float(abs(r).max()),
        core20_worst_drift=float(abs(r[core]).max()),other492_worst_drift=float(abs(r[~core]).max()),
        changed=int((pred!=y).sum()),correct=int((pred==truth).sum()),
        actual_FP32_logical_distance=sum(g['FP32_logical_distance']**2 for g in geometry)**.5,
        geometry=geometry,checkpoint_sha256=pins,edited_tensor_count=len(edited_keys),
        in_memory_state_sha256=fingerprint.hexdigest(),original_state_sha256=original_fingerprint.hexdigest(),
        score_sha256=digest(out/'replay_scores.pt'),source_sha256=digest(__file__),
        input_audit_sha256=digest(out/'independent_validation.json'),per_class=[],
        model_image_forwards=n+batch,new_training_indices=0,new_released_weights=0,
        new_candidates=0,new_confirmations=0,new_TA_calls=0,new_AGA_calls=0,
        audit_wall_seconds=time.monotonic()-start,peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
        scope='Independent NumPy FP32 weight materialization of the fixed smallest failed prototype, not a revived candidate. No new data or radius selection.')
    for c in range(before.shape[1]):
        mask=truth==c;old=(y==truth)&mask;new=(pred==truth)&mask
        report['per_class'].append(dict(class_id=c,n=int(mask.sum()),lost=int((old&~new).sum()),
            gained=int((new&~old).sum()),NLL_before=float(nll_before[mask].mean()),NLL_after=float(nll_after[mask].mean())))
    report['numerical_check_passed']=max(control_error,score_error)<=cfg['full_model_score_tolerance']
    save_json(out/'replay_validation.json',report)
    assert report['numerical_check_passed'], 'Keep failed replay; do not change the original tolerance.'
    print(json.dumps({k:report[k] for k in ['status','prototype_score_error','worst_task_drift','changed','actual_FP32_logical_distance','audit_wall_seconds']}),flush=True)


def audit_distributed_tangent(output):
    """Separate missing source constraints from finite curvature; never edit the ray."""
    from torch.autograd import forward_ad
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from covariance_noise import dataset
    from subspace_probe import make_loader
    from task_probe_distance import checked_models, initialize

    initialize();start=time.monotonic();out=Path(output)
    assert not (out/'tangent_preflight.json').exists(), 'Never overwrite an audit.'
    cfg=json.loads((out/'protocol.json').read_text());audit=json.loads((out/'independent_validation.json').read_text())
    assert audit['status']=='independently_verified_closed_source_experiment'
    for name in ['direction.pt','source_baseline.pt','attempt_scores.pt']:
        assert digest(out/name)==audit['raw_sha256'][name],name
    save_json(out/'tangent_preflight.json',dict(source_sha256=digest(__file__),n=cfg['fit_n'],batch_size=cfg['batch_size'],
        input_audit_sha256=digest(out/'independent_validation.json'),native_score_tolerance=cfg['full_model_score_tolerance'],
        core_normalized_row_tolerance=cfg['projection_tolerance'],checksum_relative_tolerance=1e-5,
        scope='Original FP32 expert; fixed closed direction and all8 cached radii. Math attention only for forward-mode AD, with matched primal controls. No new indices, correction, direction, candidate, confirmation, test, fusion or AGA.'))
    raw=torch.load(out/'direction.pt',map_location='cpu',weights_only=False)
    base=torch.load(out/'source_baseline.pt',map_location='cpu',weights_only=False)
    attempts=torch.load(out/'attempt_scores.pt',map_location='cpu',weights_only=False)
    anchor,expert,head,pins=checked_models(cfg['target']);del anchor
    training=dataset(cfg['target'],'data',expert.val_preprocess,True)
    expert.cuda();head.cuda();direction=raw['direction'].cuda();runtime=dict(scalar=None);handles=[]
    for layer,block in enumerate(expert.model.visual.transformer.resblocks):
        b=raw['basis'][layer];u,v=b['u'].cuda(),b['v'].cuda();norm=b['norm']
        def hook(_m,args,value,layer=layer,u=u,v=v,norm=norm):
            z=args[0];perp=z-(z@v.T)@v
            return value+runtime['scalar']*norm*((perp@direction[layer].T)@u.T)
        handles.append(block.mlp.c_proj.register_forward_hook(hook))
    indices=raw['source_indices'][:cfg['fit_n']]
    assert indices==base['indices'][:cfg['fit_n']]
    values=[];tangents=[];checksum=[];forwards=0
    previous=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            for batch_index,(x,_) in enumerate(make_loader(training,indices,cfg['batch_size'])):
                x=x.cuda()
                with torch.no_grad(),forward_ad.dual_level():
                    zero=torch.zeros((),device='cuda')
                    runtime['scalar']=forward_ad.make_dual(zero,torch.ones_like(zero))
                    primal,tangent=forward_ad.unpack_dual(head(expert(x)))
                    assert tangent is not None
                    values.append(primal.cpu());tangents.append(tangent.cpu());forwards+=len(x)
                if batch_index==0:
                    with torch.enable_grad():
                        runtime['scalar']=torch.zeros((),device='cuda',requires_grad=True)
                        scores=head(expert(x));forwards+=len(x)
                        grid=torch.arange(scores.numel(),device='cuda').reshape(scores.shape)
                        for pattern in [torch.where(grid%7<3,1.,-1.),torch.where(grid%11<5,1.,-1.)]:
                            pattern=pattern/scores.numel()**.5
                            reverse=float(torch.autograd.grad((scores*pattern).sum(),runtime['scalar'],retain_graph=True)[0])
                            forward=float((tangents[0].cuda()*pattern).sum())
                            error=abs(reverse-forward)/max(1.,abs(reverse),abs(forward))
                            checksum.append(dict(reverse=reverse,forward=forward,relative_error=error))
    finally:
        for handle in handles:handle.remove()
        torch.backends.mha.set_fastpath_enabled(previous)
    primal=torch.cat(values).numpy().astype(np.float64);tangent=torch.cat(tangents).numpy().astype(np.float64)
    before=base['before'][:len(indices)].numpy().astype(np.float64)
    native_error=float(abs(primal-before).max());y=before.argmax(1);rows=np.arange(len(y))
    gaps=before[rows,y,None]-before;keep=np.ones_like(gaps,dtype=bool);keep[rows,y]=False
    assert (gaps[keep]>0).all();gaps[~keep]=1.
    linear=(tangent[rows,y,None]-tangent)/gaps;linear[~keep]=0.
    core=raw['core'].numpy();mask=np.zeros(len(y),dtype=bool);mask[core]=True
    measured=linear[core][keep[core]]/np.asarray(raw['row_norms'])
    expected=raw['jacobian'].numpy().astype(np.float64)@raw['direction'].numpy().astype(np.float64).reshape(-1)
    core_error=float(abs(measured-expected).max());records=[]
    for actual,old in zip(attempts,audit['rows']):
        a=actual.numpy().astype(np.float64);ds=a-before
        finite=(ds[rows,y,None]-ds)/gaps;finite[~keep]=0.
        first=old['radius']*linear;remainder=finite-first
        records.append(dict(radius=old['radius'],finite_R=float(abs(finite).max()),
            first_order_R=float(abs(first).max()),nonlinear_remainder_R=float(abs(remainder).max()),
            core20_first_order_R=float(abs(first[mask]).max()),core20_remainder_R=float(abs(remainder[mask]).max()),
            other492_first_order_R=float(abs(first[~mask]).max()),other492_remainder_R=float(abs(remainder[~mask]).max())))
    torch.save(dict(tangent=torch.from_numpy(tangent),primal=torch.from_numpy(primal),indices=indices),out/'tangent_scores.pt')
    report=dict(status='closed_direction_source_tangent_audit',rows=records,native_score_error=native_error,
        core_normalized_row_error=core_error,reverse_forward_checks=checksum,raw_sha256=digest(out/'tangent_scores.pt'),
        source_sha256=digest(__file__),input_audit_sha256=digest(out/'independent_validation.json'),checkpoint_sha256=pins,
        model_image_forward_instances=forwards,forward_mode_inputs=len(indices),reverse_checksum_calls=2,
        new_indices=0,new_candidates=0,new_released_weights=0,new_TA_calls=0,new_AGA_calls=0,wall_seconds=time.monotonic()-start,
        limitations='FP32 directional linearization at the original expert; finite residuals are cached source observations. No universal bound, new defense, repaired point or attack performance.')
    report['numeric_checks_passed']=native_error<=cfg['full_model_score_tolerance'] and core_error<=cfg['projection_tolerance'] and max(c['relative_error'] for c in checksum)<=1e-5
    save_json(out/'tangent_validation.json',report)
    assert report['numeric_checks_passed'], 'Retain numerical failure, do not change tolerances.'
    print(json.dumps(dict(status=report['status'],native_score_error=native_error,core_error=core_error,last=records[-1],wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol')
    ap.add_argument('--completion',action='store_true')
    ap.add_argument('--joint-response')
    ap.add_argument('--joint-confirm',action='store_true')
    ap.add_argument('--joint-full-gate')
    ap.add_argument('--joint-activation')
    ap.add_argument('--public-base', action='store_true')
    ap.add_argument('--joint-class-response')
    ap.add_argument('--distributed-response')
    ap.add_argument('--distributed-replay')
    ap.add_argument('--distributed-tangent')
    args=ap.parse_args()
    if args.distributed_tangent:
        audit_distributed_tangent(args.distributed_tangent)
    elif args.distributed_replay:
        replay_distributed_response(args.distributed_replay)
    elif args.distributed_response:
        verify_distributed_response(args.distributed_response)
    elif args.joint_class_response:
        verify_joint_class_response(args.joint_class_response)
    elif args.joint_activation:
        verify_joint_activation(args.joint_activation, args.public_base)
    elif args.joint_full_gate:
        verify_joint_full_gate(args.joint_full_gate)
    elif args.joint_response:
        verify_joint_response(args.joint_response, args.joint_confirm)
    else:
        assert args.protocol
        cfg=json.loads(Path(args.protocol).read_text())
        (verify_completion if args.completion else verify)(cfg)
