"""Independent NumPy reconstruction of frozen finite response comparisons."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
from scipy.special import expit, softmax

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores


def suffix(H, p):
    n, length, width = H.shape; heads = p['heads']; d = width//heads
    x = ln(H, p['gamma1'], p['beta1'], p['eps1'])
    wq, wk, wv = np.split(p['W'], 3); bq, bk, bv = np.split(p['w'], 3)
    q = (x[:, 0]@wq.T+bq).reshape(n, heads, d)
    k = (x@wk.T+bk).reshape(n, length, heads, d).transpose(0, 2, 1, 3)
    v = (x@wv.T+bv).reshape(n, length, heads, d).transpose(0, 2, 1, 3)
    prob = softmax(np.einsum('nhd,nhtd->nht', q, k)/np.sqrt(d), axis=-1)
    avg = np.einsum('nht,nhtd->nhd', prob, v).reshape(n, width)
    r = H[:, 0]+avg@p['O'].T+p['ob']; x2 = ln(r, p['gamma2'], p['beta2'], p['eps2'])
    pre = x2@p['A'].T+p['a']; z = pre*expit(1.702*pre)
    return r+z@p['B'].T+p['b'], dict(z=z, r=r, x=x2)


def geometry(p):
    head = p['head']-p['head'].mean(0); m = p['gamma'][:, None]*(p['projection']@head.T)
    return m-m.mean(0), p['beta']@p['projection']@head.T


def sd(h, p): return np.sqrt(np.mean((h-h.mean(1, keepdims=True))**2, axis=1, keepdims=True)+p['eps'])


def N(h, p):
    m, b = geometry(p); return h@m+sd(h, p)*b


def states(H, h, p, q):
    old, c0 = suffix(H, p); up = dict(q); up['B'] = p['B']; hu, c1 = suffix(H, up); hj, _ = suffix(H, q)
    db = q['B']-p['B']; u = hu-old; v = c0['z']@db.T; it = (c1['z']-c0['z'])@db.T
    assert np.max(np.abs(hj-old-u-v-it)) < 1e-9
    return dict(h0=h, upper=h+u, B_only=h+v, joint=h+hj-old, U=u, V=v, I=it,
                z=c1['z'], x0=c0['x'], x1=c1['x'])


def lower_median(v): return float(np.sort(v.reshape(-1))[(v.size-1)//2])


def harm(n0, n1):
    y = n0.argmax(1); rows = np.arange(len(y)); other = n1.copy(); other[rows, y] = -np.inf; k = other.argmax(1)
    gap = n0[rows, y]-n0[rows, k]; assert np.all(gap > 0)
    values = 1-(n1[rows, y]-n1[rows, k])/gap
    np.testing.assert_array_equal(values > 1, n1.argmax(1) != y)
    return y, k, gap, values


def check_record(st, p, q, labels, expected, basis):
    n0 = N(st['h0'], p); y = n0.argmax(1); center0 = n0-n0.mean(1, keepdims=True); norm0 = np.linalg.norm(center0, axis=1, keepdims=True)
    for name in ['upper', 'B_only', 'joint']:
        n1 = N(st[name], p); pred = n1.argmax(1); changed = pred != y; r = expected['components'][name]
        assert r['changed'] == int(changed.sum()) and r['released_correct'] == int((pred == labels).sum())
        assert r['changed_by_class'] == np.bincount(y[changed], minlength=n0.shape[1]).tolist()
        yy, kk, gap, values = harm(n0, n1)
        assert abs(float(values.mean())-r['mean_margin_harm']) < 1e-7
        c1 = n1-n1.mean(1, keepdims=True); alpha = np.sum(c1*center0, axis=1)/np.sum(center0**2, axis=1)
        angle = np.linalg.norm(c1/np.linalg.norm(c1, axis=1, keepdims=True)-center0/norm0, axis=1)
        transverse = np.linalg.norm(c1-alpha[:, None]*center0, axis=1)/norm0[:, 0]
        for field, value in [('radial_scale_median', alpha), ('angular_change_median', angle), ('transverse_relative_median', transverse)]:
            assert abs(lower_median(value)-r[field]) < 1e-8, field
        np.testing.assert_array_equal(pred, raw_scores(st[name], p).argmax(1))
    db = q['B']-p['B']; projected = db@basis.T@basis; m, b = geometry(p)
    ratio = np.sum(projected**2)/np.sum(db**2)
    assert abs(ratio-expected['B_task_input_parameter_energy_ratio']) < 1e-8
    terms = dict(upstream=st['U']@m, original_gate_B=st['V']@m, finite_gate_B_cross=st['I']@m,
                 normalization=(sd(st['joint'], p)-sd(st['h0'], p))*b)
    assert np.max(np.abs(sum(terms.values())-(N(st['joint'], p)-n0))) < 1e-8
    for name, value in terms.items():
        a = -(value[np.arange(len(y)), yy]-value[np.arange(len(y)), kk])/gap
        assert abs(float(a.mean())-expected['mean_harm_terms'][name]) < 1e-7


def verify(root):
    start = time.monotonic(); root = Path(root); report = json.loads((root/'results.json').read_text()); cfg = json.loads((root/'protocol.json').read_text())
    assert digest(root/'diagnostic.pt') == report['diagnostic_sha256'] and digest(root/'source_only_results.json') == report['source_only_results_sha256']
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    raw = torch.load(root/'diagnostic.pt', map_location='cpu'); labels = official_labels(cfg['target'], True)[raw['indices']]
    np.testing.assert_array_equal(labels, raw['labels'].numpy())
    cal = torch.load(cfg['source_prefix'], map_location='cpu'); lookup = {v: i for i, v in enumerate(cal['indices']['fit'])}; select = [lookup[i] for i in raw['indices']]
    np.testing.assert_array_equal(raw['environments']['native']['H'].numpy(), cal['native']['H'][select].numpy())
    va, vb = raw['basis']['A']['vh'].numpy(), raw['basis']['B']['vh'].numpy(); perm = raw['AGA_permutation'].numpy()
    pn = convert(raw['environments']['native']['parameters']); boundary = torch.load(cfg['boundary_reference'], map_location='cpu')
    x = ln(boundary['screen']['r'].numpy(), pn['gamma2'], pn['beta2'], pn['eps2']); rms = np.sqrt(np.mean((x@va.T)**2, axis=0))
    np.testing.assert_allclose(rms, raw['rms'].numpy(), atol=1e-9, rtol=1e-9)
    own_a = torch.load(cfg['original_aga'], map_location='cpu'); np.testing.assert_array_equal(pn['A'][perm].astype(np.float32), own_a['model.visual.transformer.resblocks.11.mlp.c_fc.weight'].numpy())
    source_states, source_params, checks = {}, {}, []; max_error = 0.
    for env, e in raw['environments'].items():
        p = convert(e['parameters']); H = e['H'].numpy(); h = e['h'].numpy()
        for case in cfg['cases']:
            name = case['name']; item = raw['cases'][name][env]; q = dict(p)
            for key, value in item['edits'].items(): q[key] = p[key]+value.numpy()
            st = states(H, h, p, q); row = report['records'][env][name]
            check_record(st, p, q, labels, row, vb[:, perm] if env == 'real_AGA' else vb)
            if env == 'native':
                source_states[name], source_params[name] = st, q
                data = raw['source_probe_data'][name]; take = data['take'].numpy(); teacher = N(h, p).argmax(1)
                expected_take = np.concatenate([np.flatnonzero(teacher == c)[np.argsort(np.asarray(raw['indices'])[teacher == c], kind='stable')[:cfg['probe_images_per_class']]] for c in range(len(p['head']))])
                np.testing.assert_array_equal(take, expected_take)
                shifts = np.stack([-cfg['probe_amplitude']*rms[:, None]*va, cfg['probe_amplitude']*rms[:, None]*va], axis=1).reshape(-1, va.shape[1])
                np.testing.assert_allclose(shifts, data['shifts'].numpy(), atol=1e-9, rtol=1e-9)
                xx, zz = st['x0'][take], st['x1'][take]; u0 = xx@p['A'].T+p['a']; u1 = zz@q['A'].T+q['a']
                b0 = st['h0'][take]-(u0*expit(1.702*u0))@p['B'].T; b1 = st['joint'][take]-(u1*expit(1.702*u1))@q['B'].T
                aa, bb = [], []
                for i in range(0, len(take), cfg['probe_batch_images']):
                    sl = slice(i, i+cfg['probe_batch_images']); a = (xx[sl, None]+shifts[None]).reshape(-1, xx.shape[1]); b = (zz[sl, None]+shifts[None]).reshape(-1, zz.shape[1])
                    a = a@p['A'].T+p['a']; b = b@q['A'].T+q['a']
                    aa.append(N(np.repeat(b0[sl], len(shifts), axis=0)+(a*expit(1.702*a))@p['B'].T, p))
                    bb.append(N(np.repeat(b1[sl], len(shifts), axis=0)+(b*expit(1.702*b))@q['B'].T, p))
                aa = np.concatenate(aa).reshape(data['before'].shape); bb = np.concatenate(bb).reshape(data['after'].shape)
                np.testing.assert_allclose(aa, data['before'].numpy(), atol=1e-8, rtol=1e-8); np.testing.assert_allclose(bb, data['after'].numpy(), atol=1e-8, rtol=1e-8)
                yy = teacher[take]; stable = (aa.argmax(-1) == yy[:, None, None]).all(-1); changed = bb.argmax(-1) != yy[:, None, None]; both = changed.all(-1) & stable
                check = row['source_probes']; assert int(stable.sum()) == check['teacher_stable_pairs'] and int(both.sum()) == check['protected_both_signs_changed']
                assert [int(both[yy == c].sum()) for c in range(len(p['head']))] == check['both_sign_failures_by_class']
                comm = (bb-aa)-(N(st['joint'][take], p)-N(h[take], p))[:, None, None]
                np.testing.assert_allclose(comm, data['commutator'].numpy(), atol=1e-8, rtol=1e-8)
            else:
                es, eq = source_states[name], source_params[name]; db = eq['B']-pn['B']; ze = es['z']
                if env == 'real_AGA': db, ze = db[:, perm], ze[:, perm]
                me, be = geometry(pn); mf, bf = geometry(p); coef = cfg['coefficient']; err_b = q['B']-p['B']-coef*db
                terms = dict(source_residual=coef*(N(es['joint'], pn)-N(es['h0'], pn)),
                             upper_compensation_mismatch=st['U']@mf-coef*es['U']@me,
                             conditional_activation_shift=coef*(st['z']-ze)@db.T@me,
                             readout_shift=coef*st['z']@db.T@(mf-me), attack_merge_B_remainder=st['z']@err_b.T@mf,
                             normalization_mismatch=(sd(st['joint'], p)-sd(h, p))*bf-coef*(sd(es['joint'], pn)-sd(es['h0'], pn))*be)
                actual = N(st['joint'], p)-N(h, p); error = float(np.max(np.abs(sum(terms.values())-actual))); max_error = max(max_error, error); assert error < 1e-8
                y, k, gap, _ = harm(N(h, p), N(st['joint'], p)); rows = np.arange(len(y))
                for term, value in terms.items():
                    np.testing.assert_allclose(value, item['transport'][term].numpy(), atol=1e-8, rtol=1e-8)
                    harms = -(value[rows, y]-value[rows, k])/gap
                    assert abs(float(harms.mean())-row['mean_transport_harm'][term]) < 1e-7
                np.testing.assert_allclose(N(st['joint'], p), item['numerator'].numpy(), atol=1e-8, rtol=1e-8)
            checks.append(dict(environment=env, method=name, exact_finite_class_counts=True, normalized_score_order=True))
    result = dict(status='passed', source_sha256=digest(__file__), diagnostic_sha256=report['diagnostic_sha256'],
                  matched_methods=4, matched_environments=3, matched_train_n=len(labels),
                  all_source_probe_inputs=4*len(take)*2*len(va), finite_transport_maximum_error=max_error,
                  source_only_metadata_sha256=report['source_only_results_sha256'], checks=checks, wall_seconds=time.monotonic()-start,
                  new_candidates=0, new_release_weights=0, new_model_forwards=0, partner_used=True, test_used=False,
                  scope='Independent NumPy all-token attention, actual changed tensors, QuickGELU, LayerNorm, source signed probes, all12 matched records and finite six-term transport. Retrospective only; no new efficacy/coverage conclusion.')
    save_json(root/'independent_validation.json', result); print(json.dumps(result), flush=True)


@torch.no_grad()
def centered_readout_replay(protocol):
    """Replay a pinned point after a gamma-only comparator repair; never refit."""
    import torch.nn.functional as F
    from audit_hermite_splits import exclusions
    from task_probe_distance import checked_models, initialize
    initialize();start=time.monotonic();cfg=json.loads(Path(protocol).read_text())
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    for path,pin in cfg['frozen_artifacts'].items():assert digest(path)==pin,path
    root=Path(cfg['output']);own=json.loads(Path(cfg['original_protocol']).read_text())
    cal=torch.load(root/'calibration.pt',map_location='cpu',weights_only=False)
    point=torch.load(root/'point.pt',map_location='cpu',weights_only=False)
    assert cal['cfg']==own==json.loads((root/'protocol.json').read_text())
    parent=json.loads(Path(own['index_parent']).read_text());initial=set(parent['extra_excluded_indices'])|set(own['forbidden_indices'])
    for path in parent['extra_index_exclusions']+own['extra_index_exclusions']:
        v=json.loads(Path(path).read_text());initial.update(v['indices'] if 'indices' in v else v['fit'])
    split=json.loads((root/'draw_split.json').read_text());excluded,exposed=exclusions(split,initial)
    assert len(excluded)==split['excluded_count']
    draw=np.random.default_rng(own['seed']).choice(sorted(set(range(60000))-excluded),own['source_n']+own['reserved_confirmation_n'],replace=False).tolist()
    assert draw==split['indices'] and cal['indices']==draw[:own['source_n']]
    assert not set(cal['indices'])&exposed
    assert draw[own['source_n']:]==json.loads((root/'holdout_protocol.json').read_text())['indices']
    anchor,expert,head,pins=checked_models(own['target']);assert pins==cal['checkpoint_sha256']
    p,proxy=cal['parameters'],cal['proxy_parameters'];H=np.asarray(p['head'])
    P=np.asarray(p['projection']);beta=np.asarray(p['beta']);gamma=np.asarray(p['gamma'])
    u,v,k,mu=[point[key].numpy() for key in ['u','v','k','mean']];amplitude=point['amplitude']
    qp,qb=[point['edits'][key].double().numpy() for key in ['model.visual.proj','model.visual.ln_post.bias']]
    def z(h,par):
        h=np.asarray(h);c=h-h.mean(1,keepdims=True)
        return c/np.sqrt(np.mean(c*c,axis=1,keepdims=True)+par['eps'])*par['gamma']+par['beta']
    def scores(h,par):
        a=z(h,par)@par['projection'];return a/np.linalg.norm(a,axis=1,keepdims=True)@H.T
    def replay(h,par):
        result=[];head.cuda().eval()
        gain=torch.tensor(par['gamma'],device='cuda',dtype=torch.float32)
        bias=torch.tensor(par['beta'],device='cuda',dtype=torch.float32)
        projection=torch.tensor(par['projection'],device='cuda',dtype=torch.float32)
        for batch in h.split(64):
            a=F.layer_norm(batch.float().cuda(),(768,),gain,bias,par['eps'])@projection
            result.append(head(a).cpu().double())
        head.cpu();return torch.cat(result)
    changed=p|dict(beta=qb,projection=qp)
    # Same author's float32 subtraction/addition as the frozen source point.
    proxy_changed=dict(proxy)
    for key,field in [('model.visual.proj','projection'),('model.visual.ln_post.bias','beta')]:
        base=anchor.state_dict()[key];value=point['edits'][key]
        proxy_changed[field]=(base+own['coefficient']*((value-base)+torch.zeros_like(base))).double().numpy()
    h,hf=cal['h'],cal['h_proxy'];before,after,eb,ea=[replay(hh,par) for hh,par in [(h,p),(h,changed),(hf,proxy),(hf,proxy_changed)]]
    errors=[float(np.max(np.abs(scores(hh.numpy(),par)-value.numpy()))) for hh,par,value in [(h,p,before),(h,changed,after),(hf,proxy,eb),(hf,proxy_changed,ea)]]
    controls=[float((before-cal['original_scores']).abs().max()),float((eb-cal['proxy_original_scores']).abs().max())]
    assert max(errors+controls)<=own['score_tolerance'],(errors,controls)
    assert torch.equal(before.argmax(1),cal['original_scores'].argmax(1)) and torch.equal(eb.argmax(1),cal['proxy_original_scores'].argmax(1))
    bad_gamma=float(np.max(np.abs(scores(hf.numpy(),proxy|dict(gamma=gamma))-cal['proxy_original_scores'].numpy())))
    assert abs(bad_gamma-cfg['original_wrong_gamma_error'])<1e-10 and bad_gamma>own['score_tolerance']
    U,_,Vh=np.linalg.svd(P-anchor.model.visual.proj.double().numpy(),full_matrices=False)
    Z,Zf=z(h.numpy(),p),z(hf.numpy(),proxy);centered=Z-mu
    w=P@np.linalg.solve(P.T@P,v)
    geometry=max(float(np.max(np.abs(P.T@u))),float(np.max(np.abs(U[:,:own['task_rank']].T@u))),
        float(np.max(np.abs(Vh[:own['task_rank']]@v))),abs(float(k@u)),float(np.max(np.abs(k@P+(mu@u)*v))))
    assert geometry<1e-8,geometry
    contrast=H-H.mean(0);a=contrast[own['collapse_class']]
    n=1/gamma;n/=np.linalg.norm(n);frame=np.column_stack([P,U[:,:own['task_rank']],n])
    Q,s,_=np.linalg.svd(frame,full_matrices=True);N=Q[:,np.sum(s>s[0]*1e-10):]
    C=(centered@N).T@(centered@N)/len(Z);ridge=own['ridge_relative']*np.trace(C)/C.shape[0]
    b=N.T@((v@a)*Zf.mean(0)-((w@proxy['projection'])@a)*mu)
    g=(C+ridge*np.eye(len(b)))@(N.T@u);scale=float(g@b/(b@b))
    normal_error=float(np.linalg.norm(g-scale*b)/np.linalg.norm(g));assert scale>0 and normal_error<1e-6,normal_error
    def metric(s0,s1):
        aa,bb=s0.numpy(),s1.numpy();yy=aa.argmax(1);rows=np.arange(len(aa));mask=np.arange(len(H))[None,:]!=yy[:,None]
        gap=aa[rows,yy,None]-aa
        R=float(np.max(np.abs(((bb[rows,yy,None]-bb)-gap)[mask]/gap[mask])))
        return dict(n=len(aa),changed=int(np.sum(bb.argmax(1)!=yy)),R=R)
    native=metric(before,after);environment=metric(eb,ea);yy=before.argmax(1).numpy()
    take=[np.flatnonzero(yy==j)[0] for j in range(len(H))];basis=U[:,:own['task_rank']]
    rms=np.sqrt(np.mean((Z@basis)**2,axis=0));shift=np.concatenate([-own['finite_step']*rms[:,None]*basis.T,own['finite_step']*rms[:,None]*basis.T])
    zp=np.repeat(Z[take],len(shift),axis=0)+np.tile(shift,(len(H),1))
    cap=own['parameter_budget']*(1-1e-6)/np.hypot(1/np.linalg.norm(P),np.linalg.norm(k)/np.linalg.norm(beta))
    allz=np.concatenate([Z,zp]);features=allz@P;response=cap*((allz-mu)@u)[:,None]*v
    norm=np.linalg.norm(features,axis=1,keepdims=True);f0,d0=features/norm,response/norm
    s0=f0@H.T;ds=d0@H.T;y0=s0.argmax(1);rows=np.arange(len(s0));mask=np.arange(len(H))[None,:]!=y0[:,None]
    gaps=(s0[rows,y0,None]-s0)[mask];slopes=(ds[rows,y0,None]-ds)[mask]
    cross=np.repeat(np.sum(f0*d0,axis=1),len(H)-1);square=np.repeat(np.sum(d0*d0,axis=1),len(H)-1)
    first=1.
    for factor in [1-own['source_reserve'],1+own['source_reserve']]:
        aa=slopes**2-factor**2*gaps**2*square
        bb=2*(gaps*slopes-factor**2*gaps**2*cross);cc=(1-factor**2)*gaps**2
        for a0,b0,c0 in zip(aa,bb,cc):
            coefficients=[c0,b0,a0] if abs(a0)>1e-14*(abs(b0)+abs(c0)) else [c0,b0]
            for value in np.polynomial.polynomial.polyroots(coefficients):
                if abs(value.imag)<=1e-10*max(1.,abs(value.real)) and 0<value.real<first:first=float(value.real)
    expected=(1-own['interval_reserve'])*first*cap
    amplitude_error=abs(amplitude-expected)/max(abs(expected),1e-30)
    assert amplitude_error<1e-5,amplitude_error
    def projected(x,PP):
        t=x@PP;return torch.tensor(t/np.linalg.norm(t,axis=1,keepdims=True)@H.T)
    probes=metric(projected(zp,P),projected(zp+qb-beta,qp))
    labels=np.asarray(official_labels(own['target'],True))[cal['indices']]
    labels0,labels1=eb.argmax(1).numpy(),ea.argmax(1).numpy();drop=float(np.mean(labels0==labels)-np.mean(labels1==labels))
    distance=float(np.hypot(np.linalg.norm(qp-P)/np.linalg.norm(P),np.linalg.norm(qb-beta)/np.linalg.norm(beta)))
    feasible=max(native['R'],probes['R'])<=own['task_budget'] and native['changed']==probes['changed']==0 and distance<=own['parameter_budget']
    support=feasible and drop>=own['minimum_proxy_accuracy_drop']
    torch.save(dict(before=before,after=after,proxy_before=eb,proxy_after=ea,labels=torch.tensor(labels)),root/'readout_replay.pt')
    result=dict(status='source_supported_requires_reserved_confirmation' if support else 'closed_fixed_centered_readout_source_mechanism',
        source_feasible=feasible,mechanism_supported=support,native=native,task_probes=probes,public_AGA_proxy=environment,
        native_accuracy_before=float(np.mean(before.argmax(1).numpy()==labels)),native_accuracy_after=float(np.mean(after.argmax(1).numpy()==labels)),
        proxy_accuracy_before=float(np.mean(labels0==labels)),proxy_accuracy_after=float(np.mean(labels1==labels)),proxy_accuracy_drop=drop,
        parameter_distance=distance,amplitude=amplitude,NumPy_score_errors=errors,original_full_model_control_errors=controls,
        old_wrong_gamma_error_reproduced=bad_gamma,NumPy_finite_compensation_error=geometry,NumPy_direction_normal_equation_error=normal_error,
        independent_polynomial_amplitude_relative_error=amplitude_error,
        source_index_draw_reproduced=True,excluded_training_n=len(excluded),reserved_confirmation_images_read=0,
        source_protocol=cfg['original_protocol'],source_protocol_sha256=digest(cfg['original_protocol']),
        replay_protocol_sha256=digest(protocol),verification_source_sha256=digest(__file__),
        frozen_artifacts=cfg['frozen_artifacts'],replay_sha256=digest(root/'readout_replay.pt'),
        original_completed_full_image_forwards=4*len(labels),original_failed_run_wall_seconds=None,
        new_full_image_forwards=0,cached_prefix_readout_examples=4*len(labels),expert_training_updates=0,new_AGA_calls=0,new_point_fits=0,
        wall_seconds=time.monotonic()-start,
        scope='Frozen source point after correcting proxy gamma in the comparator. Original full-forward prefix equalities passed before the comparator failure; missing after-scores reconstructed from those fixed prefixes. No new construction,images,confirmation,test or real partner, and no full-model recovery or efficacy claim.')
    save_json(root/'results.json',result)
    save_json(root/'independent_validation.json',dict(status='passed_frozen_point_replay_and_independent_NumPy_checks',
        result_sha256=digest(root/'results.json'),replay_sha256=result['replay_sha256'],
        source_sha256=digest(__file__),source_indices_reproduced=True,geometry_error=geometry,
        normal_equation_error=normal_error,amplitude_relative_error=amplitude_error,
        NumPy_score_errors=errors,original_full_model_control_errors=controls,new_full_image_forwards=0,new_point_fits=0))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output'); ap.add_argument('--centered-readout');args = ap.parse_args()
    torch.set_num_threads(4)
    centered_readout_replay(args.centered_readout) if args.centered_readout else verify(args.output)
