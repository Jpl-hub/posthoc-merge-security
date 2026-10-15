"""Matched retrospective finite decisions and source-only compensation probes.

No release construction, parameter optimization, test images, or new attacks.
Peer-dependent explanations must not be reused to select a future defense.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores
from attention_cascade import norm, parameters, suffix
from clip_benchmark import checkpoint_loader, digest, merge
from covariance_noise import dataset
from finite_compensation_transport import geometry, sigma
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, gelu, initialize, tail
from verify_covariance_noise import official_labels


def cpu(p): return {k: v.cpu() if torch.is_tensor(v) else v for k, v in p.items()}


def pinned(path, cfg):
    assert digest(path) == cfg['inputs'][path], path
    return torch.load(path, map_location='cpu')


def pars(model, head):
    p = parameters(model, head); p.pop('D'); return p


def numerator(h, p):
    m, b = geometry(p); return h@m+sigma(h, p)*b


def classify(n0, n1, labels):
    y = n0.argmax(1); rows = torch.arange(len(y), device=y.device)
    other = n1.clone(); other[rows, y] = -torch.inf; k = other.argmax(1)
    gap = n0[rows, y]-n0[rows, k]; assert bool((gap > 0).all())
    harm = 1-(n1[rows, y]-n1[rows, k])/gap; changed = n1.argmax(1) != y
    assert torch.equal(changed, harm > 1.)
    return dict(n=len(y), original_correct=int((y == labels).sum()), released_correct=int((n1.argmax(1) == labels).sum()),
                changed=int(changed.sum()), changed_by_class=torch.bincount(y[changed], minlength=n0.shape[1]).cpu().tolist(),
                mean_margin_harm=float(harm.mean()), median_margin_harm=float(harm.median())), (y, k, gap, harm)


def parts(H, h, p, q):
    old, ctx = suffix(H, p); upper = dict(q); upper['B'] = p['B']
    hu, cu = suffix(H, upper); hj, cj = suffix(H, q)
    db = q['B']-p['B']; u = hu-old; v = ctx['z']@db.T; interaction = (cu['z']-ctx['z'])@db.T
    assert float((hj-old-u-v-interaction).abs().max()) < 1e-10
    return dict(h0=h, upper=h+u, B_only=h+v, joint=h+hj-old, U=u, V=v, I=interaction,
                z0=ctx['z'], z=cu['z'], r0=ctx['r'], r1=cu['r'], x0=norm(ctx['r'], p['gamma2'], p['beta2'], p['eps2']),
                x1=norm(cu['r'], p['gamma2'], p['beta2'], p['eps2']))


def describe(st, p, q, labels, basis):
    n0 = numerator(st['h0'], p); f0 = angular_scores(scores(st['h0'], p)); results = {}
    for key in ['upper', 'B_only', 'joint']:
        nn = numerator(st[key], p); row, _ = classify(n0, nn, labels)
        cent0 = n0-n0.mean(1, keepdim=True); cent1 = nn-nn.mean(1, keepdim=True)
        alpha = (cent0*cent1).sum(1)/cent0.square().sum(1)
        row.update(radial_scale_median=float(alpha.median()), angular_change_median=float((angular_scores(scores(st[key], p))-f0).norm(dim=1).median()),
                   transverse_relative_median=float(((cent1-alpha[:, None]*cent0).norm(dim=1)/cent0.norm(dim=1)).median()))
        results[key] = row
    da = angular_scores(scores(st['upper'], p))-f0; db = angular_scores(scores(st['B_only'], p))-f0
    cosine = (da*db).sum(1)/(da.norm(dim=1)*db.norm(dim=1)).clamp_min(1e-20)
    m, b = geometry(p); nn = numerator(st['joint'], p); _, (y, k, gap, _) = classify(n0, nn, labels)
    terms = dict(upstream=st['U']@m, original_gate_B=st['V']@m, finite_gate_B_cross=st['I']@m,
                 normalization=(sigma(st['joint'], p)-sigma(st['h0'], p))*b)
    error = float((sum(terms.values())-(nn-n0)).abs().max()); assert error < 1e-8
    row = torch.arange(len(y), device=y.device); harm = {key: -(v[row, y]-v[row, k])/gap for key, v in terms.items()}
    delta = q['B']-p['B']; task = delta@basis.T@basis; full_response = st['z']@delta.T@m
    return dict(components=results, source_branch_angular_cosine_median=float(cosine.median()), finite_identity_error=error,
                mean_harm_terms={key: float(v.mean()) for key, v in harm.items()},
                B_task_input_parameter_energy_ratio=float(task.square().sum()/delta.square().sum()),
                B_task_input_class_response_energy_ratio=float((st['z']@task.T@m).square().sum()/full_response.square().sum()),
                energy_ratio_scope='Projection onto fixed32 source task-vector right singular modes, not full functional coverage. Response ratios may exceed1 because terms can cancel.'), terms


@torch.no_grad()
def source_probes(st, p, q, indices, basis, rms, cfg):
    teacher = scores(st['h0'], p).argmax(1); chosen = []
    ids = torch.tensor(indices, device='cuda')
    for c in range(len(p['head'])):
        members = (teacher == c).nonzero().flatten(); order = ids[members].argsort(stable=True)
        assert len(members) >= cfg['probe_images_per_class']; chosen.extend(members[order[:cfg['probe_images_per_class']]].tolist())
    take = torch.tensor(chosen, device='cuda'); dirs = cfg['probe_amplitude']*rms[:, None]*basis
    shifts = torch.stack([-dirs, dirs], 1).flatten(0, 1); width = len(shifts)
    x0, x1 = st['x0'][take], st['x1'][take]; z0 = gelu(x0@p['A'].T+p['a']); z1 = gelu(x1@q['A'].T+q['a'])
    base0 = st['h0'][take]-z0@p['B'].T; base1 = st['joint'][take]-z1@q['B'].T
    before, after = [], []
    for start in range(0, len(take), cfg['probe_batch_images']):
        row = slice(start, start+cfg['probe_batch_images'])
        a = (x0[row, None]+shifts[None]).flatten(0, 1); b = (x1[row, None]+shifts[None]).flatten(0, 1)
        h0 = base0[row].repeat_interleave(width, 0)+gelu(a@p['A'].T+p['a'])@p['B'].T
        h1 = base1[row].repeat_interleave(width, 0)+gelu(b@q['A'].T+q['a'])@q['B'].T
        before.append(numerator(h0, p)); after.append(numerator(h1, p))
    before = torch.cat(before).reshape(len(take), len(basis), 2, -1); after = torch.cat(after).reshape_as(before)
    y = teacher[take]; keep = (before.argmax(-1) == y[:, None, None]).all(-1)
    changed = after.argmax(-1) != y[:, None, None]
    both = changed.all(-1) & keep; either = changed.any(-1) & keep
    base_n = numerator(st['h0'][take], p); joint_n = numerator(st['joint'][take], p)
    commutator = (after-before)-(joint_n-base_n)[:, None, None]
    return dict(images=len(take), rank=len(basis), teacher_stable_pairs=int(keep.sum()), total_pairs=keep.numel(),
                protected_both_signs_changed=int(both.sum()), protected_either_sign_changed=int(either.sum()),
                images_with_both_sign_failure=int(both.any(1).sum()),
                both_sign_failures_by_class=[int(both[y == c].sum()) for c in range(len(p['head']))],
                teacher_stable_pairs_by_class=[int(keep[y == c].sum()) for c in range(len(p['head']))],
                finite_commutator_norm_median=float(commutator.norm(dim=-1).median()),
                diagnostic_only=True, no_proxy_or_release_acceptance_gate=True), dict(take=take.cpu(), before=before.cpu(), after=after.cpu(), commutator=commutator.cpu(), shifts=shifts.cpu())


@torch.no_grad()
def capture_prefix(model, training, indices):
    hs, ends = [], []; visual = model.model.visual; block = visual.transformer.resblocks[11]
    handles = [block.ln_1.register_forward_pre_hook(lambda _m, args: hs.append(args[0].transpose(0, 1).detach().cpu())),
               visual.ln_post.register_forward_pre_hook(lambda _m, args: ends.append(args[0].detach().cpu()))]
    try:
        model.cuda().eval()
        for images, _ in make_loader(training, indices, 64): model(images.cuda())
    finally:
        for h in handles: h.remove()
        model.cpu()
    return dict(H=torch.cat(hs).double(), h=torch.cat(ends).double())


@torch.no_grad()
def audit(cfg):
    start = time.monotonic(); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    old = pinned(cfg['matched_reference'], cfg); cal = pinned(cfg['source_prefix'], cfg)
    basis = pinned(cfg['basis_reference'], cfg)['basis']; va, vb = basis['A']['vh'].cuda(), basis['B']['vh'].cuda()
    anchor, expert, head, pins = checked_models(cfg['target']); p_native = pars(expert, head)
    indices = old['indices']; assert len(indices) == cfg['n']
    lookup = {i: j for j, i in enumerate(cal['indices']['fit'])}; select = torch.tensor([lookup[i] for i in indices])
    native = dict(H=cal['native']['H'][select].double(), h=cal['native']['h'][select].double(), parameters=p_native)
    assert torch.equal(p_native['A'], cal['parameters']['A'])
    source = on_device(p_native); boundary = pinned(cfg['boundary_reference'], cfg)
    r = boundary['screen']['r'].cuda(); x = norm(r, source['gamma2'], source['beta2'], source['eps2'])
    rms = (x@va.T).square().mean(0).sqrt()
    labels = torch.tensor(official_labels(cfg['target'], True)[indices], device='cuda')
    saved = dict(indices=indices, labels=labels.cpu(), basis={site: cpu(b) for site, b in basis.items()}, rms=rms.cpu(),
                 environments={'native': native}, cases={}, cfg=cfg, source_probe_data={})
    source_reports = {}; native_states = {}; released_native = {}
    for case in cfg['cases']:
        edits = pinned(case['release'], cfg); model = copy.deepcopy(expert); model.load_state_dict(expert.state_dict() | edits)
        q = on_device(pars(model, head)); st = parts(native['H'].cuda(), native['h'].cuda(), source, q)
        row, _ = describe(st, source, q, labels, vb)
        probe, raw = source_probes(st, source, q, indices, va, rms, cfg); row['source_probes'] = probe
        source_reports[case['name']] = row; native_states[case['name']] = st; released_native[case['name']] = model
        changed = {k: (q[k]-source[k]).cpu() for k in q if torch.is_tensor(q[k]) and not torch.equal(q[k], source[k])}
        assert set(changed) <= {'O', 'ob', 'W', 'w', 'A', 'a', 'B'}
        saved['cases'][case['name']] = {'native': dict(edits=changed)}; saved['source_probe_data'][case['name']] = raw
    # This record and hash are finalized BEFORE loading any peer checkpoint.
    save_json(out/'source_only_results.json', dict(status='source_only_finite_compensation_diagnosis', records=source_reports,
        partner_loaded=False, test_used=False, no_new_candidate=True, source_sha256=digest(__file__)))
    source_pin = digest(out/'source_only_results.json')
    print(json.dumps(dict(stage='source_only_completed', records=source_reports, source_only_sha256=source_pin)), flush=True)
    training = dataset(cfg['target'], 'data', expert.val_preprocess, True)
    load = checkpoint_loader(Path('external/secure-merge-attack'))
    assert digest(cfg['peer_checkpoint']) == cfg['inputs'][cfg['peer_checkpoint']]
    peer = load(cfg['peer_checkpoint']).cpu().eval(); own_a = copy.deepcopy(expert); own_a.load_state_dict(pinned(cfg['original_aga'], cfg))
    peer_a = copy.deepcopy(peer); peer_a.load_state_dict(pinned(cfg['peer_aga'], cfg))
    def key(v): return hashlib.sha256(v.float().contiguous().numpy().tobytes()).hexdigest()
    native_rows = {key(v): i for i, v in enumerate(p_native['A'])}; perm = torch.tensor([native_rows[key(v)] for v in pars(own_a, head)['A']])
    assert len(perm.unique()) == len(perm); saved['AGA_permutation'] = perm
    environments = dict(real_TA=(expert, peer), real_AGA=(own_a, peer_a)); records = {'native': source_reports}
    replay = {}; new_forwards = 0
    for env, (own, other) in environments.items():
        before = merge(anchor, own, other, cfg['coefficient']); raw = capture_prefix(before, training, indices); new_forwards += len(indices)
        raw['parameters'] = pars(before, head); saved['environments'][env] = raw; p = on_device(raw['parameters']); H, h = raw['H'].cuda(), raw['h'].cuda()
        reports = {}; chosen = torch.linspace(0, len(indices)-1, cfg['replay_n']).long(); images = torch.stack([training[indices[i]][0] for i in chosen]).cuda()
        for case in cfg['cases']:
            name = case['name']; published = released_native[name] if env == 'real_TA' else copy.deepcopy(own_a)
            if env == 'real_AGA': published.load_state_dict(own_a.state_dict() | pinned(case['AGA_edits'], cfg))
            after = merge(anchor, published, other, cfg['coefficient'])
            assert all(torch.equal(v, after.state_dict()[k]) for k, v in before.state_dict().items() if not k.startswith('model.visual.transformer.resblocks.11.'))
            q = on_device(pars(after, head)); st = parts(H, h, p, q)
            for k in ['gamma', 'beta', 'projection', 'head', 'gamma1', 'beta1', 'gamma2', 'beta2']: assert torch.equal(p[k], q[k]), (env, name, k)
            row, raw_terms = describe(st, p, q, labels, vb[:, perm.cuda()] if env == 'real_AGA' else vb)
            source_st = native_states[name]; native_q = on_device(pars(released_native[name], head))
            dbs = native_q['B']-source['B']; zs = source_st['z']
            if env == 'real_AGA': dbs = dbs[:, perm.cuda()]; zs = zs[:, perm.cuda()]
            me, be = geometry(source); mf, bf = geometry(p); coefficient = cfg['coefficient']
            dns = numerator(source_st['joint'], source)-numerator(source_st['h0'], source)
            err_b = q['B']-p['B']-coefficient*dbs
            terms = dict(source_residual=coefficient*dns,
                         upper_compensation_mismatch=st['U']@mf-coefficient*source_st['U']@me,
                         conditional_activation_shift=coefficient*(st['z']-zs)@dbs.T@me,
                         readout_shift=coefficient*st['z']@dbs.T@(mf-me),
                         attack_merge_B_remainder=st['z']@err_b.T@mf,
                         normalization_mismatch=(sigma(st['joint'], p)-sigma(st['h0'], p))*bf-coefficient*(sigma(source_st['joint'], source)-sigma(source_st['h0'], source))*be)
            actual = numerator(st['joint'], p)-numerator(st['h0'], p); error = float((sum(terms.values())-actual).abs().max()); assert error < 1e-8
            _, (y, k, gap, _) = classify(numerator(st['h0'], p), numerator(st['joint'], p), labels); rows = torch.arange(len(y), device='cuda')
            harms = {key: -(value[rows, y]-value[rows, k])/gap for key, value in terms.items()}
            row.update(transport_identity_error=error, mean_transport_harm={key: float(value.mean()) for key, value in harms.items()},
                       transport_by_class=[dict(class_id=c, n=int((y == c).sum()), mean_harm={key: float(value[y == c].mean()) for key, value in harms.items()}) for c in range(len(p['head']))],
                       B_remainder_relative_norm=float(err_b.norm()/((q['B']-p['B']).norm()).clamp_min(1e-20)))
            after.cuda().eval(); head.cuda().eval(); measured = head(after(images)).argmax(1); after.cpu(); head.cpu(); new_forwards += len(chosen)
            expected = scores(st['joint'][chosen.cuda()], p).argmax(1); assert torch.equal(measured, expected), (env, name)
            replay[env+'/'+name] = dict(n=len(chosen), predictions_match=True)
            changed = {key: (q[key]-p[key]).cpu() for key in q if torch.is_tensor(q[key]) and not torch.equal(q[key], p[key])}
            assert set(changed) <= {'O', 'ob', 'W', 'w', 'A', 'a', 'B'}
            saved['cases'][name][env] = dict(edits=changed, numerator=numerator(st['joint'], p).cpu(), transport={key: value.cpu() for key, value in terms.items()})
            reports[name] = row; del after
        records[env] = reports; print(json.dumps(dict(stage=env, records=reports)), flush=True); del before
    assert digest(out/'source_only_results.json') == source_pin
    torch.save(saved, out/'diagnostic.pt')
    result = dict(status='matched_frozen_finite_response_diagnosis', target=cfg['target'], n=len(indices), records=records,
                  source_only_results_sha256=source_pin, diagnostic_sha256=digest(out/'diagnostic.pt'), source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']),
                  checkpoint_sha256=pins, wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  new_full_model_forward_instances=new_forwards, replay=replay, new_release_weights=0, publication_updates=0, new_attack_calls=0,
                  test_used=False, partner_used=True, role='Retrospective explanation of pre-existing points only. Source-only diagnostic saved before peer loading. No future defense construction/selection may use peer-dependent records.',
                  limitation='One fixed target-training slice and one known peer. Internal signed SVD probes are not a physical merge or unknown-peer coverage. No new protection efficacy or causal generalization claim.')
    save_json(out/'results.json', result); save_json(out/'protocol.json', cfg); print(json.dumps(dict(status=result['status'], wall_seconds=result['wall_seconds'], diagnostic_sha256=result['diagnostic_sha256'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); audit(cfg)
