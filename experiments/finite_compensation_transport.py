"""Exact finite compensation transport of preserved points; no optimization."""

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from clip_benchmark import digest
from mixed_compensation_audit import activation, environment, ln, pack
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from verify_covariance_noise import official_labels


def geometry(p):
    head = p['head']-p['head'].mean(0)
    m = p['gamma'][:, None]*(p['projection']@head.T)
    return m-m.mean(0), p['beta']@p['projection']@head.T


def sigma(h, p):
    return ((h-h.mean(1, keepdim=True)).square().mean(1, keepdim=True)+p['eps']).sqrt()


def state(item, p, ch, da, db, factor):
    r = item['r']+factor*item['o_input']@da.T if ch['site']=='O' else item['r']
    a = p['A']+factor*da if ch['site']=='A' else p['A']
    z0 = activation(ln(item['r'], p)@p['A'].T+p['a'])[0]
    z1 = activation(ln(r, p)@a.T+p['a'])[0]
    h0 = item['rounding']+item['r']+z0@p['B'].T+p['b']
    u = r-item['r']+(z1-z0)@p['B'].T
    h1 = h0+u+factor*z1@db.T
    m, b = geometry(p)
    return dict(z=z1, u=u, h0=h0, h1=h1, m=m, b=b,
                delta_sigma=sigma(h1, p)-sigma(h0, p))


def numbers(values):
    return {k:float(torch.quantile(values, q)) for k,q in [('median', .5), ('q95', .95), ('maximum', 1.)]}


def decision(st, p, terms, labels):
    h0, h1 = st['h0'], st['h1']
    n0 = h0@st['m']+sigma(h0, p)*st['b']
    n1 = h1@st['m']+sigma(h1, p)*st['b']
    delta = n1-n0
    error = float((sum(terms.values())-delta).abs().max())
    assert error<1e-7, error
    s0, s1 = scores(h0, p), scores(h1, p)
    assert torch.equal(n0.argmax(1), s0.argmax(1)) and torch.equal(n1.argmax(1), s1.argmax(1))
    y = n0.argmax(1); rows = torch.arange(len(y), device=y.device)
    competitors = n1.clone(); competitors[rows, y] = -torch.inf
    k = competitors.argmax(1); gap = n0[rows,y]-n0[rows,k]; assert gap.min()>0
    harm = {name:-(v[rows,y]-v[rows,k])/gap for name,v in terms.items()}
    total = sum(harm.values()); changed = s1.argmax(1)!=y
    assert torch.equal(total>1, changed)
    classes = len(p['head']); counts = torch.bincount(y, minlength=classes)
    per_class = []
    for c in range(classes):
        mask = y==c
        per_class.append(dict(label=c, n=int(mask.sum()), changed=int(changed[mask].sum()),
            mean_harm={name:float(v[mask].mean()) if mask.any() else None for name,v in harm.items()},
            mean_total_harm=float(total[mask].mean()) if mask.any() else None))
    rec = dict(n=len(y), original_correct=int((y.cpu()==labels).sum()),
               released_correct=int((s1.argmax(1).cpu()==labels).sum()), changed=int(changed.sum()),
               changed_by_class=torch.bincount(y[changed], minlength=classes).cpu().tolist(),
               task_worst_margin_drift=float(signed_residual(s0,s1).abs().max()),
               finite_identity_maximum_error=error, per_class=per_class,
               mean_harm={name:float(v.mean()) for name,v in harm.items()})
    raw = dict(baseline_numerator=n0.cpu(), released_numerator=n1.cpu(),
               scores0=s0.cpu(), scores1=s1.cpu(), winner=y.cpu(), competitor=k.cpu(),
               harm={name:v.cpu() for name,v in harm.items()}, terms={name:v.cpu() for name,v in terms.items()})
    return rec, raw


@torch.no_grad()
def audit(cfg):
    start=time.monotonic(); out=Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    ref=Path(cfg['mixed_reference']); assert digest(ref/'diagnostic.pt')==cfg['mixed_sha256']
    old=torch.load(ref/'diagnostic.pt', map_location='cpu')
    p, pp=on_device(old['parameters']), on_device(old['proxy_parameters'])
    gold=official_labels(cfg['target'], True); records=[]; saved={}
    for case in cfg['cases']:
        assert digest(case['point'])==case['sha256']
        point=torch.load(case['point'], map_location='cpu'); cached=old['cases'][case['name']]
        ch=on_device(cached['chart']); native, proxy=on_device(cached['native']), on_device(cached['proxy'])
        theta=point['theta'].cuda().double(); rank=ch['rank']; q=ch['q']
        assert theta.norm()<=cfg['parameter_budget']
        da=ch['na']*ch['u']@theta[:rank**2].reshape(rank,rank)@ch['vh']
        k=ch['nb']*theta[rank**2:].reshape(q.shape[1],p['B'].shape[1]); db=q@k
        source=pack(point['native'], p)
        es=state(source,p,ch,da,db,1.); me,be=geometry(p)
        # Full finite native activation matrix, not a variance-truncated PCA.
        u,s,vh=torch.linalg.svd(es['z'], full_matrices=False)
        tol=torch.finfo(s.dtype).eps*max(es['z'].shape)*s[0]
        numerical_rank=int((s>tol).sum()); u,s,vh=u[:,:numerical_rank],s[:numerical_rank],vh[:numerical_rank]
        source_delta=es['u']@me+es['z']@k.T@(q.T@me)+es['delta_sigma']*be
        source_rec,source_raw=decision(es,p,dict(source=source_delta),torch.tensor(gold[point['indices']['fit']]))
        ge=q.T@me; decoder=torch.linalg.pinv(ge, rtol=cfg['readout_rtol'])
        assert (ge@decoder-torch.eye(len(ge),device='cuda')).abs().max()<1e-8
        env_records=[]; env_saved={}; coefficient=cfg['coefficient']
        # Factorial interventions between two observed endpoints are algebraic
        # diagnostics, not extra fitted proxies or realized model fusions.
        for bits in itertools.product([0.,1.],repeat=3):
            item,pars=environment(torch.tensor(bits,device='cuda',dtype=torch.float64),native,proxy,p,pp)
            fs=state(item,pars,ch,da,db,coefficient); gf=q.T@fs['m']; c=decoder@gf
            l=((fs['z']@vh.T)/s)@u.T
            residual=fs['z']-l@es['z']
            terms=dict(
                finite_compensation_mismatch=fs['u']@fs['m']-coefficient*l@(es['u']@me)@c,
                transported_native_residual=coefficient*l@source_delta@c,
                activation_uncovered=coefficient*residual@k.T@gf,
                normalization_commutator=fs['delta_sigma']*fs['b']-coefficient*l@(es['delta_sigma']*be)@c)
            rec,rr=decision(fs,pars,terms,torch.tensor(gold[cached['indices']]))
            rec.update(environment=list(map(int,bits)),
                       activation_relative_uncovered=float(residual.norm()/fs['z'].norm()),
                       transport_l1=numbers(l.abs().sum(1)),
                       source_residual_frobenius=float(source_delta.norm()),
                       transported_residual_frobenius=float(terms['transported_native_residual'].norm()))
            env_records.append(rec);env_saved[''.join(map(str,map(int,bits)))]=dict(**rr,transport=l.cpu())
        records.append(dict(case=case['name'], source_n=len(source['r']), source_numerical_rank=numerical_rank,
                            source_feature_width=es['z'].shape[1],source_rank_tolerance=float(tol),
                            source_condition=float(s[0]/s[-1]), source_singular_extrema=[float(s[0]),float(s[-1])],
                            publication_distance=float(theta.norm()),native=source_rec,environments=env_records))
        saved[case['name']]=dict(source=source_raw, environments=env_saved, singular_values=s.cpu())
        print(json.dumps({k:v for k,v in records[-1].items() if k not in ['native','environments']}) ,flush=True)
    file=out/'diagnostic.pt';torch.save(dict(cases=saved,cfg=cfg),file)
    report=dict(status='finite_compensation_transport_diagnosis',target=cfg['target'],records=records,
                diagnostic_sha256=digest(file),source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                publication_updates=0,new_release_weights=0,new_full_model_forward_instances=0,
                test_used=False,partner_used=False,confirmation_used=False,
                limitation='Preserved finite points and target-training caches only. Hybrids are interventions, not physical fusions; empirical transport is not unseen-data or unknown-peer coverage. No hyperparameter selection or new defense.')
    save_json(out/'results.json',report); save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    audit(cfg)
