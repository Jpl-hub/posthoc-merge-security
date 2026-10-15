"""Finite joint-margin stress test at a source/anchor observed representation.

No publication optimization, peer, test data, recovery or new release. The
environment is a CONDITIONAL interface set, not a claimed partner distribution.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from attention_cascade import suffix
from clip_benchmark import digest
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from verify_covariance_noise import official_labels


def gap(s, y, scale):
    other = s.clone(); other[torch.arange(len(y), device=y.device), y] = -torch.inf
    return (s.gather(1, y[:, None])[:, 0]-other.max(1).values)/scale


def norms(x): return x.flatten(1).norm(dim=1)


def run(cfg):
    start = time.monotonic(); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    cache = torch.load(cfg['cache'], map_location='cpu')
    p, pp = on_device(cache['parameters']), on_device(cache['proxy_parameters'])
    labels = official_labels(cfg['target'], True)
    # Select with SOURCE and unprotected public-anchor states only, before
    # loading either protected point. No protected-response-based selection.
    ys = scores(cache['native']['h'].cuda(), p).argmax(1).cpu()
    yp = scores(cache['proxy']['h'].cuda(), pp).argmax(1).cpu()
    lookup = {x: i for i, x in enumerate(cache['indices']['fit'])}
    chosen = []
    for cls in range(len(p['head'])):
        rows = [i for i, idx in enumerate(cache['indices']['proxy'])
                if int(yp[i]) == cls == int(ys[lookup[idx]]) == int(labels[idx])]
        assert len(rows) >= cfg['per_class']
        chosen.extend(rows[:cfg['per_class']])
    ids = [cache['indices']['proxy'][i] for i in chosen]
    native_rows = [lookup[i] for i in ids]
    h = cache['proxy']['H'][chosen].cuda().double()
    source_h = cache['native']['H'][native_rows].cuda().double()
    d = h-source_h; size = norms(d); assert size.min() > 1e-8
    unit = d/size[:, None, None]
    rounding = cache['proxy']['rounding'][chosen].cuda().double()
    baseline = scores(suffix(h, pp)[0]+rounding, pp).detach()
    cache_error = float((baseline-scores(cache['proxy']['h'][chosen].cuda(), pp)).abs().max())
    assert cache_error < 1e-8
    y = baseline.argmax(1); row = torch.arange(len(y), device='cuda')
    other = baseline.clone(); other[row, y] = -torch.inf
    scale = baseline[row, y]-other.max(1).values
    save_json(out/'selection.json', dict(indices=ids, proxy_positions=chosen,
        native_positions=native_rows, per_class=cfg['per_class'],
        criterion='first target-training examples correctly classified by source and unprotected public-anchor model; selected before protected point loads'))
    cases, raw = [], {}
    calls = dict(interface_reverse_calls=0, suffix_forward_instances=0)

    def evaluate(eta, pars):
        calls['suffix_forward_instances'] += len(h)
        return scores(suffix(h+eta, pars)[0]+rounding, pars)

    for case in cfg['cases']:
        point = torch.load(case['point'], map_location='cpu')
        if case['format'] == 'O_B':
            rank = len(point['basis_O']['vh']); theta = point['theta'].cuda().double()
            u = point['basis_O']['u'].cuda().double(); v = point['basis_O']['vh'].cuda().double()
            q = point['output_basis'].cuda().double()
            delta = dict(O=p['O'].norm()*u@theta[:rank**2].reshape(rank, rank)@v,
                         B=p['B'].norm()*q@theta[rank**2:].reshape(q.shape[1], p['B'].shape[1]))
        else:
            delta = on_device(point['release_deltas'])
        distance = sum(float((value/p[key].norm()).square().sum()) for key, value in delta.items())**.5
        assert distance <= cfg['parameter_budget']
        modified = pp | {key: pp[key]+cfg['coefficient']*value for key, value in delta.items()}
        eta = torch.zeros_like(h)
        initial = evaluate(eta, modified).detach()
        active0 = initial.argmax(1) != y
        saved_steps = []; gradient_checks = []
        for iteration in range(cfg['steps']):
            with torch.enable_grad():
                leaf = eta.detach().clone().requires_grad_(True)
                b = evaluate(leaf, pp); a = evaluate(leaf, modified)
                margin = gap(a, y, scale)
                active = active0 & (margin < cfg['restored_gap'])
                if not bool(active.any()): break
                g = torch.autograd.grad(margin.sum(), leaf, retain_graph=True)[0]
                constraint_rows = [unit.flatten(1)]
                for cls in range(len(p['head'])):
                    value = (b[:, cls]-b[row, y])/scale
                    v = torch.autograd.grad(value.sum(), leaf, retain_graph=cls+1 < len(p['head']))[0].flatten(1)
                    constraint_rows.append(v/v.norm(dim=1, keepdim=True).clamp_min(1e-30))
                calls['interface_reverse_calls'] += len(p['head'])+1
            mat = torch.stack(constraint_rows, dim=1)
            gram = mat@mat.transpose(1, 2)
            flat = g.flatten(1)
            projected = flat-(mat.transpose(1, 2)@(torch.linalg.pinv(gram, rtol=1e-10)@(mat@flat[:, :, None]))).squeeze(-1)
            # Enforce the identified task-retreat coordinate exactly, including
            # floating-point projection error. This is a state-space condition.
            projected -= (projected*unit.flatten(1)).sum(1, keepdim=True)*unit.flatten(1)
            target = (cfg['restored_gap']-margin.detach()).clamp_min(0.)
            step = projected*(target/projected.square().sum(1).clamp_min(1e-30))[:, None]
            step *= (cfg['trust_fraction']*size/step.norm(dim=1).clamp_min(1e-30)).clamp_max(1)[:, None]
            step *= active[:, None]
            step = step.reshape_as(h).detach()
            derivative = (g.detach()*step).flatten(1).sum(1)
            # Actual finite-difference check; this is a proposal predictor,
            # never a replacement for the finite acceptance test.
            eps = cfg['derivative_step']
            with torch.no_grad():
                fd = (gap(evaluate(eta+eps*step, modified), y, scale)-gap(evaluate(eta-eps*step, modified), y, scale))/(2*eps)
            fd_error = float((fd-derivative).abs().max())
            assert fd_error < cfg['derivative_tolerance'], fd_error
            gradient_checks.append(fd_error)
            alpha = torch.ones(len(h), device='cuda', dtype=torch.float64)
            accepted = torch.zeros_like(active); trials = []
            with torch.no_grad():
                before = margin.detach()
                for backtrack in range(cfg['backtracks']):
                    trial = eta+alpha[:, None, None]*step
                    b1, a1 = evaluate(trial, pp), evaluate(trial, modified)
                    drift = signed_residual(baseline, b1).abs().max(1).values
                    after = gap(a1, y, scale); pred = alpha*derivative
                    actual = after-before
                    agree = actual/pred.clamp_min(1e-30)
                    valid = (active & ~accepted & (pred > 1e-12) & (actual > 1e-12)
                             & (agree >= cfg['acceptance_ratio']) & (drift <= cfg['baseline_budget'])
                             & (norms(trial) <= size) & (b1.argmax(1) == y))
                    trials.append(dict(alpha=alpha.cpu(), baseline=b1.cpu(), released=a1.cpu(), accepted=valid.cpu()))
                    accepted |= valid
                    alpha = torch.where(accepted, alpha, alpha*.5)
                    if bool((accepted | ~active).all()): break
                # The saved scale for accepted inputs remains their first
                # accepted scale; inactive/rejected inputs are unchanged.
                update = alpha*accepted
                eta += update[:, None, None]*step
                saved_steps.append(dict(step=step.cpu(), derivative=derivative.cpu(), fd=fd.cpu(),
                                        trials=trials, applied_scale=update.cpu()))
                if not bool(accepted.any()): break
        with torch.no_grad():
            final_b, final_a = evaluate(eta, pp), evaluate(eta, modified)
            restored = active0 & (gap(final_a, y, scale) >= cfg['restored_gap'])
            drift = signed_residual(baseline, final_b).abs().max(1).values
            transverse = (eta*unit).flatten(1).sum(1).abs()/size
            assert drift.max() <= cfg['baseline_budget']+1e-10
            assert transverse.max() < 1e-10 and (norms(eta) <= size*(1+1e-10)).all()
            records = [dict(label=c, n=int((y == c).sum()),
                            initially_changed=int((active0 & (y == c)).sum()),
                            restored=int((restored & (y == c)).sum())) for c in range(len(p['head']))]
            rho = norms(eta)/size
            rec = dict(case=case['name'], n=len(h), parameter_distance=distance,
                initially_changed=int(active0.sum()), restored=int(restored.sum()),
                still_changed=int((final_a.argmax(1) != y).sum()), per_class=records,
                restored_relative_norms=rho[restored].cpu().tolist(),
                relative_norm_median=float(rho[restored].median()) if restored.any() else None,
                baseline_worst_drift=float(drift.max()), maximum_transverse_error=float(transverse.max()),
                nonlinear_steps=len(saved_steps), maximum_derivative_fd_error=max(gradient_checks, default=0.))
            cases.append(rec)
            raw[case['name']] = dict(delta={k:v.cpu() for k,v in delta.items()},
                initial=initial.cpu(), final_baseline=final_b.cpu(), final_released=final_a.cpu(),
                eta=eta.cpu(), steps=saved_steps)
            print(json.dumps(rec), flush=True)
    file = out/'diagnostic.pt'
    torch.save(dict(H=h.cpu(), source_H=source_h.cpu(), rounding=rounding.cpu(),
        baseline=baseline.cpu(), indices=ids, cases=raw, cfg=cfg), file)
    result = dict(status='finite_conditional_environment_witnesses_not_new_defense', cases=cases,
        cache_replay_error=cache_error, diagnostic_sha256=digest(file),
        source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']),
        wall_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        **calls, publication_gradient_calls=0, new_weights=0, new_full_model_forwards=0,
        partner_used=False, test_used=False, confirmation_used=False,
        limitation='A witness is an upper bound on an interface robustness radius, NOT a real partner, parameter-recovery attack, coverage estimate or impossibility theorem. Failure to find one is NOT a certificate. Local/readout parameters remain fixed, so parameter uncertainty and prefix reachability are unproved. No publication optimization or candidate was generated.')
    save_json(out/'results.json', result); save_json(out/'protocol.json', cfg)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(cfg)
