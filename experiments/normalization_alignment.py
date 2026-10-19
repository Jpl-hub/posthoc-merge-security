"""One frozen V/O gauge plus an output-mean shift; no optimizer or response tuning."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters
from local_residual_transport import cpu, forward, metric, pack
from mechanisms import load_author
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_probe_distance import checked_models, initialize
from verify_levelset_audits import convert, raw_scores

PREFIX='model.visual.transformer.resblocks.11.attn.'
KEYS=[PREFIX+k for k in ['in_proj_weight','in_proj_bias','out_proj.weight']]


def center(x):
    return x-x.mean(0,keepdim=True)


@torch.no_grad()
def build(expert,anchor,cfg):
    state=expert.state_dict();original={k:state[k].double().clone() for k in KEYS}
    w,b,o=[original[k] for k in KEYS];o0=anchor.state_dict()[KEYS[2]].double()
    u=torch.ones(768,dtype=torch.float64)/np.sqrt(768);axes=[]
    for i in range(0,768,64):
        _,_,vh=torch.linalg.svd(o[:,i:i+64]-o0[:,i:i+64],full_matrices=False)
        p=vh[0];p=p*torch.sign(p[p.abs().argmax()]);axes.append(p)
    blocks=[p[:,None]*p[None] for p in axes];P=torch.block_diag(*blocks)
    dw,db=P@w[1536:],P@b[1536:];do=o@P
    cost_vo=float(dw.norm().square()/w.norm().square()+db.norm().square()/b.norm().square())
    cost_o=float(do.norm().square()/o.norm().square());goal=cfg['parameter_budget']*(1-cfg['distance_reserve'])
    def distance2(a):return cost_vo*(a-1)**2+cost_o*(1-1/a)**2
    lo,hi=1.,cfg['gauge_bracket_max'];assert distance2(hi)>goal**2/2
    for _ in range(cfg['bisection_steps']):
        mid=(lo+hi)/2
        if distance2(mid)<=goal**2/2:lo=mid
        else:hi=mid
    a=lo;S=torch.eye(768,dtype=torch.float64)+(a-1)*P
    qw,qb=w.clone(),b.clone();qw[1536:]+=(a-1)*dw;qb[1536:]+=(a-1)*db
    qo=o+(1/a-1)*do;M=u[:,None]*torch.cat(axes)[None]
    A=float(M.norm().square()/o.norm().square());c=float(((qo-o)*M).sum()/o.norm().square())
    beta=(-c+np.sqrt(c*c+A*(goal**2-distance2(a))))/A;qo+=beta*M
    edits=dict(zip(KEYS,[qw.float(),qb.float(),qo.float()]));qO=edits[KEYS[2]].double()
    errors=[]
    for x,y in [(qO@edits[KEYS[0]][1536:].double(),o@w[1536:]),
                (qO@edits[KEYS[1]][1536:].double()[:,None],(o@b[1536:])[:,None])]:
        errors.append(float((center(x-y)).norm()/center(y).norm()))
    distances={k:float((edits[k].double()-original[k]).norm()/original[k].norm()) for k in KEYS}
    geometry=dict(gauge_scale=a,mean_coefficient=beta,per_tensor_distance=distances,
        parameter_distance=sum(x*x for x in distances.values())**.5,centered_operator_error=max(errors),
        known_gauge_TA_half_cross_factor=.25*(a+1/a-2),head_count=12,modified_V_bias_in_budget=True)
    return edits,S,geometry


@torch.no_grad()
def geometry_check(original,edits,anchor,cfg):
    author,pin=load_author(Path(cfg['author_path']));assert pin==cfg['author_text_sha256']
    source=[];np_error=[];clean_error=[]
    for i in range(0,768,64):
        w=edits[KEYS[2]][:,i:i+64];a=anchor.state_dict()[KEYS[2]][:,i:i+64]
        def solve(x,y):return x@author['solve_linear_transformation'](x,y)
        got=solve(w,a);x,y=w.double().numpy(),a.double().numpy()
        expected=x@np.linalg.lstsq(x.T@x+1e-8*np.eye(64),x.T@y,rcond=None)[0]
        np_error.append(float(abs(expected-got.double().numpy()).max()))
        source.append(float(center(got.double()-solve(original[KEYS[2]][:,i:i+64],a).double()).norm()))
        # Centering removes the added mean family, not the ridge's coordinate dependence.
        clean=solve(center(w.double()).float(),center(a.double()).float()).double()
        base=solve(center(original[KEYS[2]][:,i:i+64].double()).float(),center(a.double()).float()).double()
        clean_error.append(float((clean-base).norm()))
    assert max(np_error)<=cfg['author_weight_tolerance']
    return dict(author_numpy_max_error=max(np_error),raw_AGA_centered_O_change=source,
        centered_AGA_O_change=clean_error,scope='Weight geometry only, not classification harm or a robustness guarantee.')


@torch.no_grad()
def measure(expert,release,head,train,ids,point,cfg,out,name):
    original=capture_attention(expert,train,ids);changed=capture_attention(release,train,ids)
    p,q=on_device(point['p']),on_device(point['q']);vo,vb=[point[k].cuda() for k in ['vo','vb']]
    data=pack(original,p,vo,vb,cfg);before,after=forward(data,p,p),forward(data,p,q)
    # Probe H and perturbations are transported through S. q.O is the effective O'S.
    values=[];head.cuda()
    for c in [original,changed]:values.append(head(c['actual'].float().cuda()).cpu().double())
    head.cpu();native,released=values
    score_error=max(float((native-before[:len(ids)].cpu()).abs().max()),float((released-after[:len(ids)].cpu()).abs().max()))
    pn,qn,dn=convert(point['p']),convert(point['q']),convert(cpu(data))
    np_error=0.
    for params,wanted in [(pn,before),(qn,after)]:
        r=dn['r']+dn['H']@(params['O']-pn['O']).T
        c=r-r.mean(1,keepdims=True);z=(c/np.sqrt((c*c).mean(1,keepdims=True)+params['eps2'])*params['gamma2']+params['beta2'])@params['A'].T+params['a']
        h=dn['rounding']+r+(z*expit(1.702*z)+dn['offset'])@params['B'].T+params['b']
        np_error=max(np_error,float(abs(raw_scores(h,params)-wanted.cpu().numpy()).max()))
    assert score_error<=cfg['score_tolerance'] and np_error<=1e-8
    labels=torch.tensor([int(train[i][1]) for i in ids]);y0,y1=native.argmax(1),released.argmax(1)
    result=dict(covariant_probes=metric(before,after,len(ids)),native=metric(native,released,len(ids)),
        canonical_native_score_error=score_error,numpy_score_error=np_error,
        per_class=[dict(class_id=k,n=int(labels.eq(k).sum()),original_correct=int((y0.eq(labels)&labels.eq(k)).sum()),
            released_correct=int((y1.eq(labels)&labels.eq(k)).sum())) for k in range(10)])
    torch.save(dict(data=cpu(data),before=before.cpu(),after=after.cpu(),native=native,released=released,
        labels=labels,indices=ids),out/(name+'.pt'))
    return result


@torch.no_grad()
def run(protocol,confirm=False):
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);start=time.monotonic()
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    assert '3090' in torch.cuda.get_device_name();torch.cuda.reset_peak_memory_stats()
    anchor,expert,head,pins=checked_models(cfg['target']);train=None
    if not confirm:
        out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
        edits,S,g=build(expert,anchor,cfg);g.update(geometry_check(expert.state_dict(),edits,anchor,cfg))
        save_json(out/'geometry.json',g)
        assert g['parameter_distance']<=cfg['parameter_budget'] and g['centered_operator_error']<=cfg['operator_tolerance']
        torch.save(edits,out/'frozen_release.pt')
        assert digest(cfg['probe_reference']['path'])==cfg['probe_reference']['sha256']
        ref=torch.load(cfg['probe_reference']['path'],map_location='cpu');p=extended_parameters(expert,head)
        q=p|dict(O=edits[KEYS[2]].double()@S)
        point=dict(p=p,q=q,vo=ref['basis_O']['vh'],vb=ref['basis_B']['vh'])
        torch.save(point,out/'point.pt')
        save_json(out/'before_images.json',dict(release_sha256=digest(out/'frozen_release.pt'),point_sha256=digest(out/'point.pt'),
            checkpoint_sha256=pins,source_sha256=digest(__file__),protocol_sha256=digest(protocol)))
        name,seed='source',cfg['construction_seed']
    else:
        report=json.loads((out/'results.json').read_text());assert report['source_feasible']
        assert json.loads((out/'independent_validation.json').read_text())['status']=='passed'
        point=torch.load(out/'point.pt',map_location='cpu');edits=torch.load(out/'frozen_release.pt',map_location='cpu')
        g=json.loads((out/'geometry.json').read_text());name,seed='confirmation',cfg['confirmation_seed']
    assert not (out/(name+'.pt')).exists(),'No response-driven rerun.'
    frozen=json.loads((out/'before_images.json').read_text())
    assert digest(out/'frozen_release.pt')==frozen['release_sha256'] and digest(out/'point.pt')==frozen['point_sha256']
    release=copy.deepcopy(expert)
    for k,v in edits.items():release.state_dict()[k].copy_(v)
    train=dataset(cfg['target'],'data',expert.val_preprocess,True);excluded=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    split=unused(train,excluded,cfg|dict(confirmation_n=cfg['check_n'],confirmation_seed=seed))
    split.update(role=name,used_for_optimization=False);save_json(out/(name+'_split.json'),split)
    save_json(out/('training_indices.json' if not confirm else 'confirmation_split.json'),dict(check=split['indices']) if not confirm else split)
    result=measure(expert,release,head,train,split['indices'],point,cfg,out,name)
    feasible=all(result[k]['worst_task_drift']<=cfg['task_budget'] and result[k]['changed']==0 for k in ['native','covariant_probes'])
    result.update(status='passed' if feasible else 'closed_task_budget_failure',source_feasible=feasible,geometry=g,
        checkpoint_sha256=pins,release_sha256=frozen['release_sha256'],raw_sha256=digest(out/(name+'.pt')),
        source_sha256=digest(__file__),protocol_sha256=digest(protocol),gradients=0,expert_updates=0,new_source_forward_instances=2*len(split['indices']),
        partner_used=False,test_used=False,wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save_json(out/('confirmation_validation.json' if confirm else 'results.json'),result);print(json.dumps(result),flush=True)


def verify(folder,confirmation=False):
    from audit_hermite_splits import exclusions
    from verify_covariance_noise import official_labels
    out=Path(folder);name='confirmation' if confirmation else 'source'
    report=json.loads((out/('confirmation_validation.json' if confirmation else 'results.json')).read_text())
    assert digest(out/(name+'.pt'))==report['raw_sha256'];raw=torch.load(out/(name+'.pt'),map_location='cpu')
    p,q=[convert(torch.load(out/'point.pt',map_location='cpu')[k]) for k in ['p','q']];d=convert(raw['data']);error=0.
    for params,label in [(p,'before'),(q,'after')]:
        r=d['r']+d['H']@(params['O']-p['O']).T;c=r-r.mean(1,keepdims=True)
        z=(c/np.sqrt((c*c).mean(1,keepdims=True)+params['eps2'])*params['gamma2']+params['beta2'])@params['A'].T+params['a']
        got=raw_scores(d['rounding']+r+(z*expit(1.702*z)+d['offset'])@params['B'].T+params['b'],params)
        error=max(error,float(abs(got-raw[label].numpy()).max()));assert error<=1e-8
    for names,key in [(('before','after'),'covariant_probes'),(('native','released'),'native')]:
        a,b=[raw[k].numpy() for k in names];y=a.argmax(1);ix=np.arange(len(y));gap=a[ix,y,None]-a;gap[ix,y]=1
        delta=b-a;rho=abs((delta[ix,y,None]-delta)/np.maximum(gap,1e-12));rho[ix,y]=0
        assert abs(float(rho.max())-report[key]['worst_task_drift'])<1e-8
    cfg=json.loads((out/'protocol.json').read_text());split=json.loads((out/(name+'_split.json')).read_text())
    initial=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:initial.update(json.loads(Path(path).read_text())['indices'])
    excluded,exposed=exclusions(split,initial);assert len(excluded)==split['excluded_count']
    draw=np.random.default_rng(split['seed']).choice(sorted(set(range(60000))-excluded),cfg['check_n'],replace=False).tolist()
    assert draw==raw['indices']==split['indices'] and not set(draw)&exposed
    labels=official_labels(cfg['target'],True)[draw];np.testing.assert_array_equal(labels,raw['labels'])
    for row in report['per_class']:
        mask=labels==row['class_id'];assert int(mask.sum())==row['n']
        for label,field in [('native','original_correct'),('released','released_correct')]:
            assert int(((raw[label].numpy().argmax(1)==labels)&mask).sum())==row[field]
    _,expert,_,_=checked_models(cfg['target']);edits=torch.load(out/'frozen_release.pt',map_location='cpu')
    distance=sum(float(np.linalg.norm(v.numpy().astype('float64')-expert.state_dict()[k].numpy())**2/
        np.linalg.norm(expert.state_dict()[k].numpy().astype('float64'))**2) for k,v in edits.items())**.5
    assert abs(distance-report['geometry']['parameter_distance'])<1e-10
    save_json(out/('confirmation_independent_validation.json' if confirmation else 'independent_validation.json'),
        dict(status='passed',source_sha256=digest(__file__),raw_sha256=report['raw_sha256'],numpy_score_error=error,
            official_labels_perclass_counts=True,exact_draw_reproduced=True,excluded_n=len(excluded),parameter_distance=distance,
            scope='Independent NumPy complete-tail and relative-margin recount; feasibility is recorded separately.',source_feasible=report['source_feasible']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');ap.add_argument('--confirm',action='store_true')
    args=ap.parse_args();initialize();verify(args.verify,args.confirm) if args.verify else run(args.protocol,args.confirm)
