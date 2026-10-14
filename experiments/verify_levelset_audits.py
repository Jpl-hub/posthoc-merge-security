"""Independent NumPy reconstruction of coupling and FFN geometry records."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_probe_candidate import array


def convert(values):
    return {k:array(v) if torch.is_tensor(v) else v for k,v in values.items()}


def feature(h,p):
    t=ln(h,p['gamma'],p['beta'],p['eps'])@p['projection']
    return t/np.linalg.norm(t,axis=1,keepdims=True)


def raw_scores(h,p):
    return feature(h,p)@p['head'].T


def residual(before,after):
    y=before.argmax(1);rows=np.arange(len(y));gap=before[rows,y,None]-before;gap[rows,y]=1.
    change=after-before;value=(change[rows,y,None]-change)/np.maximum(gap,1e-12);value[rows,y]=0.
    return value


def coupling(out,data,report):
    labels=official_labels('MNIST',True)[data['indices']];np.testing.assert_array_equal(labels,array(data['labels']))
    checks=[]
    for record in report['records']:
        name=record['environment'];g=data['groups'][name];a,b=convert(g['before']),convert(g['after'])
        p,q=convert(g['parameters_before']),convert(g['parameters_after'])
        def hidden(r):
            pre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a']
            return pre*expit(1.702*pre)
        z0,z1=hidden(a['r']),hidden(b['r']);rounding=a['h']-a['r']-z0@p['B'].T-p['b']
        db=q['B']-p['B']
        hs={'baseline':a['h'],'attention_branch_only':rounding+b['r']+z1@p['B'].T+p['b'],
            'B_only':a['h']+z0@db.T,'joint':rounding+b['r']+z1@q['B'].T+p['b']}
        ss={k:raw_scores(h,p) for k,h in hs.items()};teacher=ss['baseline'].argmax(1)
        error=max(float(np.abs(s-array(g['scores'][k])).max()) for k,s in ss.items());assert error<1e-8
        for row in record['components']:
            pred=ss[row['component']].argmax(1);changed=pred!=teacher
            assert int((pred==labels).sum())==row['correct'] and int(changed.sum())==row['changed']
            assert np.bincount(teacher[changed],minlength=len(p['head'])).tolist()==row['changed_by_teacher_class']
        f0=feature(hs['baseline'],p);da=feature(hs['attention_branch_only'],p)-f0;dbf=feature(hs['B_only'],p)-f0
        cos=np.sum(da*dbf,1)/np.maximum(np.linalg.norm(da,axis=1)*np.linalg.norm(dbf,axis=1),1e-20)
        median=float(np.sort(cos)[(len(cos)-1)//2]);assert abs(median-record['feature_compensation_cosine_median'])<1e-8
        interaction=hs['joint']-hs['attention_branch_only']-hs['B_only']+hs['baseline']
        identity=float(np.abs(interaction-(z1-z0)@db.T).max());assert identity<1e-10
        checks.append({'environment':name,'normalized_scores_maximum_error':error,'interaction_identity_error':identity,
            'official_labels_and_all_component_class_counts':'passed','feature_cosine_lower_median_error':abs(median-record['feature_compensation_cosine_median'])})
    return {'checks':checks,'scope':'Independent normalized CLIP response, exact cross-term, official-label and per-class recount. Frozen training slice; no new protection or parameter selection.'}


def ffn(out,data,report):
    p=convert(data['parameters']);basis={site:convert(g) for site,g in data['basis'].items()}
    u,v,q=basis['A']['u'],basis['A']['vh'],array(data['output_basis']);r=len(v);size=r*r
    for a in [u.T,v,q.T]:np.testing.assert_allclose(a@a.T,np.eye(len(a)),atol=1e-10)
    def predict(packed,theta,factor=1.,parameters=p):
        c=convert(packed);ea=np.linalg.norm(p['A'])*u@theta[:size].reshape(r,r)@v
        eb=np.linalg.norm(p['B'])*q@theta[size:].reshape(q.shape[1],p['B'].shape[1])
        pre=c['x']@(parameters['A']+factor*ea).T+parameters['a'];z=pre*expit(1.702*pre)+c['offset']
        return raw_scores(c['base']+z@(parameters['B']+factor*eb).T,parameters)
    core=data['core'];zero=np.zeros(len(array(data['gradient'])));theta=array(data['theta']);direction=array(data['direction'])
    before=predict(core,zero);np.testing.assert_allclose(before,array(core['teacher']),atol=1e-9,rtol=1e-9)
    random=np.random.default_rng(20261016).normal(size=len(zero));random/=np.linalg.norm(random);h=data['cfg']['derivative_step']
    y=before.argmax(1);keep=np.ones_like(before,dtype=bool);keep[np.arange(len(y)),y]=False
    fd=(residual(before,predict(core,h*random))-residual(before,predict(core,-h*random)))/(2*h)
    error=float(np.abs(fd[keep]-array(data['matrix'])@random).max());assert error<1e-6
    cache=convert(data['native']);x=ln(cache['r'],p['gamma2'],p['beta2'],p['eps2']);pre=x@p['A'].T+p['a'];z=pre*expit(1.702*pre)
    packed={'x':x,'base':cache['h']-z@p['B'].T,'offset':np.zeros_like(z)}
    b=predict(packed,zero);a=predict(packed,theta);rho=float(np.abs(residual(b,a)).max())
    assert rho<=report['final']['native_critic']+1e-8
    assert abs(np.linalg.norm(theta)-report['final']['parameter_distance'])<1e-10
    assert np.linalg.norm(theta)<=.5 and report['final']['native_critic']<=.5
    np.testing.assert_array_equal(b.argmax(1),a.argmax(1))
    return {'analytic_random_direction_fd_maximum_error':error,'natural_training_n':len(x),
        'natural_training_worst_drift':rho,'coefficient_and_weight_distance_consistency':'passed',
        'scope':'Independent native natural-data NumPy response, random-direction JVP and orthogonal publication distance. This verifier does not certify the full synthetic probe envelope or proxy transfer.'}


def game(out,data,report):
    import copy
    from cross_module_distance import extended_parameters
    from covariance_noise import dataset
    from subspace_probe import make_loader
    from task_probe_distance import checked_models
    anchor,expert,head,pins=checked_models(report['target']);assert pins==report['checkpoint_sha256']
    p=convert(extended_parameters(expert,head));p0=convert(extended_parameters(anchor,head))
    basis={s:convert(data['basis_'+s]) for s in ['A','B']};u,v=basis['A']['u'],basis['A']['vh'];q=array(data['output_basis'])
    rank=len(v);size=rank**2;theta=array(data['theta']);ea=np.linalg.norm(p['A'])*u@theta[:size].reshape(rank,rank)@v
    eb=np.linalg.norm(p['B'])*q@theta[size:].reshape(q.shape[1],p['B'].shape[1])
    np.testing.assert_allclose(ea,array(data['release_deltas']['A']),atol=1e-10);np.testing.assert_allclose(eb,array(data['release_deltas']['B']),atol=1e-10)
    for a in [u.T,v,q.T]:np.testing.assert_allclose(a@a.T,np.eye(len(a)),atol=1e-10)
    task=p['A']-p0['A'];np.testing.assert_allclose(task@v.T,u*basis['A']['s'],atol=1e-9,rtol=1e-7)
    distance=float(np.sqrt((np.linalg.norm(ea)/np.linalg.norm(p['A']))**2+(np.linalg.norm(eb)/np.linalg.norm(p['B']))**2))
    assert abs(distance-report['selected']['parameter_distance'])<1e-10 and distance<=.5
    cfg=data['protocol'];n=cfg['contexts_per_class']*p['head'].shape[0];checks=[]
    for name in ['native','scale','confirmation']:
        c=convert(data[name]);x=ln(c['r'],p['gamma2'],p['beta2'],p['eps2']);pre=x@p['A'].T+p['a'];z=pre*expit(1.702*pre)
        base=c['h']-z@p['B'].T
        sa,sb=[] , []
        shifts=[]
        for value,vv in [(x,v),(z,basis['B']['vh'])]:
            rms=np.sqrt(np.mean((value@vv.T)**2,axis=0))
            shifts.append((cfg['finite_step']*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*vv[None,:,:]).reshape(2*rank,-1))
        sa,sb=shifts;count=len(sa)
        xx=np.concatenate([x,(x[:n,None,:]+sa[None,:,:]).reshape(-1,x.shape[1]),np.repeat(x[:n],count,axis=0)])
        bb=np.concatenate([base,np.repeat(base[:n],count,axis=0),np.repeat(base[:n],count,axis=0)])
        off=np.concatenate([np.zeros_like(z),np.zeros((n*count,z.shape[1])),np.tile(sb,(n,1))])
        before_pre=xx@p['A'].T+p['a'];before_z=before_pre*expit(1.702*before_pre)+off
        after_pre=xx@(p['A']+ea).T+p['a'];after_z=after_pre*expit(1.702*after_pre)+off
        before=raw_scores(bb+before_z@p['B'].T,p);after=raw_scores(bb+after_z@(p['B']+eb).T,p)
        rho=float(np.abs(residual(before,after)).max());expected=report['confirmation']['worst_margin_drift'] if name=='confirmation' else None
        if expected is not None:assert abs(rho-expected)<1e-7
        assert rho<=.5
        np.testing.assert_array_equal(before.argmax(1),after.argmax(1))
        checks.append({'group':name,'natural_n':len(x),'natural_plus_synthetic_n':len(xx),'worst_margin_drift':rho})
    assert abs(max(c['worst_margin_drift'] for c in checks[:2])-report['selected']['native_critic'])<1e-7
    indices=data['confirmation_indices'];assert len(indices)==len(set(indices))==512
    audit=json.loads(Path(cfg['index_audit']).read_text());excluded=set(i for group in data['indices'].values() for i in group)
    for row in audit['records']:
        path=Path(row['path']);group=row['group']
        obj=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        excluded.update(obj[group])
    for path in cfg['extra_confirmation_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    assert not set(indices)&excluded
    labels=official_labels(report['target'],True)[indices]
    np.testing.assert_array_equal(labels,array(data['confirmation_labels']))
    protected=copy.deepcopy(expert);block=protected.model.visual.transformer.resblocks[11].mlp
    block.c_fc.weight.data.copy_((block.c_fc.weight.double()+torch.from_numpy(ea)).to(block.c_fc.weight.dtype))
    block.c_proj.weight.data.copy_((block.c_proj.weight.double()+torch.from_numpy(eb)).to(block.c_proj.weight.dtype))
    training=dataset(report['target'],'data',expert.val_preprocess,True);replays=0
    for model,expected in [(expert,data['confirmation_original_predictions']),(protected,data['confirmation_candidate_predictions'])]:
        model.cuda().eval();pred=[]
        with torch.no_grad():
            for images,_ in make_loader(training,indices[:128],64):
                f=model(images.cuda());f=f/f.norm(dim=1,keepdim=True);pred.append((f@head.weight.cuda().T).argmax(1).cpu())
        np.testing.assert_array_equal(torch.cat(pred).numpy(),expected[:128].numpy());model.cpu();replays+=128
    return {'checks':checks,'parameter_reconstruction':'passed','task_singular_equation':'passed','parameter_distance':distance,
        'fresh_confirmation_disjoint_against_registered_history':True,'fresh_full_model_prediction_instances':replays,
        'scope':'Independent NumPy all-natural and all-finite-probe constraints, weight distance, target history exclusion and256 complete CLIP prediction replays. No true merging efficacy claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=['coupling','ffn','game'],required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);report=json.loads((out/'results.json').read_text());path=out/('candidate.pt' if args.kind=='game' else 'diagnostic.pt')
    expected=report['candidate_sha256'] if args.kind=='game' else report['diagnostic_sha256'];assert digest(path)==expected
    data=torch.load(path,map_location='cpu');result={'coupling':coupling,'ffn':ffn,'game':game}[args.kind](out,data,report)
    save_json(out/'independent_validation.json',{'status':'passed',**result,'diagnostic_sha256':digest(path),'source_sha256':digest(__file__),
        'publication_updates':0,'new_release_weights':0,'new_model_forwards':result.get('fresh_full_model_prediction_instances',0),'test_used':False})
    print(json.dumps(result),flush=True)
