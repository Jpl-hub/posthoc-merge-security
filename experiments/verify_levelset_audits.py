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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=['coupling','ffn'],required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.output);report=json.loads((out/'results.json').read_text());path=out/'diagnostic.pt';assert digest(path)==report['diagnostic_sha256']
    data=torch.load(path,map_location='cpu');result=(coupling if args.kind=='coupling' else ffn)(out,data,report)
    save_json(out/'independent_validation.json',{'status':'passed',**result,'diagnostic_sha256':digest(path),'source_sha256':digest(__file__),
        'publication_updates':0,'new_release_weights':0,'new_model_forwards':0,'test_used':False})
    print(json.dumps(result),flush=True)
