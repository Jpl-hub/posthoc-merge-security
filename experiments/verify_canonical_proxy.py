"""NumPy finite differences of the frozen target-only canonical proxy contrast."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_levelset_audits import convert, raw_scores
from verify_probe_candidate import explicit_merge
from verify_cross_module import ln


def verify(output):
    out=Path(output);report=json.loads((out/'results.json').read_text());assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    data=torch.load(out/'diagnostic.pt',map_location='cpu');cfg=data['cfg'];point=torch.load(Path(cfg['publication_point'])/'candidate.pt',map_location='cpu')
    assert digest(Path(cfg['publication_point'])/'candidate.pt')==cfg['candidate_sha256']
    anchor,expert,head,_=checked_models('MNIST');canonical=copy.deepcopy(expert)
    assert digest(cfg['canonical_reference'])==cfg['canonical_reference_sha256'];canonical.load_state_dict(torch.load(cfg['canonical_reference'],map_location='cpu'))
    p=convert(extended_parameters(expert,head));u,v=[point['basis_A'][k].numpy() for k in ['u','vh']];q=point['output_basis'].numpy();r=len(v);size=r*r
    theta=point['theta'].numpy();d=np.random.default_rng(20261016).normal(size=len(theta));d/=np.linalg.norm(d);checks=[]
    for record in report['records']:
        name=record['environment'];proxy=explicit_merge(anchor,expert if name=='ordinary' else canonical,anchor,.5);pp=convert(extended_parameters(proxy,head))
        c=convert(point['proxy'] if name=='ordinary' else data['canonical_cache']);x=ln(c['r'],pp['gamma2'],pp['beta2'],pp['eps2'])
        pre=x@pp['A'].T+pp['a'];z=pre*expit(1.702*pre);base=c['h']-z@pp['B'].T
        def predict(t):
            ea=np.linalg.norm(p['A'])*u@t[:size].reshape(r,r)@v;eb=np.linalg.norm(p['B'])*q@t[size:].reshape(q.shape[1],p['B'].shape[1])
            pre=x@(pp['A']+.5*ea).T+pp['a'];zz=pre*expit(1.702*pre)
            return raw_scores(base+zz@(pp['B']+.5*eb).T,pp)
        before=predict(np.zeros_like(theta));after=predict(theta);saved=data['records'][name]
        np.testing.assert_allclose(before,saved['teacher'].numpy(),atol=1e-8);np.testing.assert_allclose(after,saved['after'].numpy(),atol=1e-8)
        y=before.argmax(1);rows=np.arange(len(y));counts=np.bincount(y,minlength=before.shape[1]);weight=1./counts[y]/int((counts>0).sum())
        def angular(s):
            c=s-s.mean(1,keepdims=True);return c/np.linalg.norm(c,axis=1,keepdims=True)
        def gap(s):
            other=s.copy();other[rows,y]=-np.inf;return s[rows,y]-other.max(1)
        unit=np.sort(gap(angular(before)))[(len(y)-1)//2]
        loss=lambda t:-np.sum(weight*np.maximum(gap(angular(predict(t))),0))/unit
        h=1e-6;fd=(loss(h*d)-loss(-h*d))/(2*h);analytic=saved['gradient'].numpy()@d
        assert abs(fd-analytic)<1e-6
        changed=after.argmax(1)!=y;assert int(changed.sum())==record['frozen_point_changed']
        checks.append({'environment':name,'random_gradient_fd_error':abs(fd-analytic),'normalized_scores_and_changed_class_counts':'passed'})
    a,b=[data['records'][name]['projected_gradient'].numpy() for name in ['ordinary','author_canonical']]
    cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)));assert abs(cosine-report['task_tangent_gradient_cosine'])<1e-10
    save_json(out/'independent_validation.json',{'status':'passed','checks':checks,'gradient_cosine':cosine,'source_sha256':digest(__file__),
        'diagnostic_sha256':digest(out/'diagnostic.pt'),'new_model_forwards':0,'publication_updates':0,'test_used':False,'partner_used':False,
        'scope':'Independent NumPy normalized scores, finite-difference gradient and stored projected-gradient cosine; no unknown-peer transfer guarantee.'})
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
