"""Independent NumPy finite transport, minimum-norm and classification checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import qr
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores, residual


def geometry(p):
    w=p['head']-p['head'].mean(0);m=p['gamma'][:,None]*(p['projection']@w.T)
    return m-m.mean(0),p['beta']@p['projection']@w.T


def sd(h,p):
    return np.sqrt(np.mean((h-h.mean(1,keepdims=True))**2,1,keepdims=True)+p['eps'])


def forward(item,p,ch,da,db,factor):
    r=item['r']+factor*item['o_input']@da.T if ch['site']=='O' else item['r']
    a=p['A']+factor*da if ch['site']=='A' else p['A']
    def hidden(r,a):
        v=ln(r,p['gamma2'],p['beta2'],p['eps2'])@a.T+p['a']
        return v*expit(1.702*v)
    z0,z=hidden(item['r'],p['A']),hidden(r,a)
    h0=item['rounding']+item['r']+z0@p['B'].T+p['b']
    u=r-item['r']+(z-z0)@p['B'].T;h1=h0+u+factor*z@db.T
    return z,u,h0,h1,sd(h1,p)-sd(h0,p)


def verify(output):
    out=Path(output);start=time.monotonic();report=json.loads((out/'results.json').read_text())
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    data=torch.load(out/'diagnostic.pt',map_location='cpu');cfg=data['cfg'];ref=Path(cfg['mixed_reference'])/'diagnostic.pt'
    assert digest(ref)==cfg['mixed_sha256'];old=torch.load(ref,map_location='cpu')
    p,pp=convert(old['parameters']),convert(old['proxy_parameters']);gold=official_labels(cfg['target'],True)
    checks=[]
    for case,record in zip(cfg['cases'],report['records']):
        assert digest(case['point'])==case['sha256'];point=torch.load(case['point'],map_location='cpu')
        previous=old['cases'][case['name']];ch=convert(previous['chart']);native,proxy=convert(previous['native']),convert(previous['proxy'])
        theta=point['theta'].numpy();r=ch['rank'];q=ch['q'];da=ch['na']*ch['u']@theta[:r*r].reshape(r,r)@ch['vh']
        k=ch['nb']*theta[r*r:].reshape(q.shape[1],p['B'].shape[1]);db=q@k
        source=convert(point['native']);v=ln(source['r'],p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a']
        source={**source,'rounding':source['h']-source['r']-(v*expit(1.702*v))@p['B'].T-p['b']}
        ze,ue,h0,h1,dse=forward(source,p,ch,da,db,1.)
        me,be=geometry(p);source_delta=(h1-h0)@me+dse*be
        saved=data['cases'][case['name']];np.testing.assert_allclose(source_delta,saved['source']['terms']['source'].numpy(),atol=1e-8,rtol=1e-7)
        before,after=raw_scores(h0,p),raw_scores(h1,p);labels=gold[point['indices']['fit']]
        assert int((before.argmax(1)!=after.argmax(1)).sum())==record['native']['changed']
        assert int((after.argmax(1)==labels).sum())==record['native']['released_correct']
        assert abs(np.abs(residual(before,after)).max()-record['native']['task_worst_margin_drift'])<1e-7
        assert np.linalg.norm(theta)<=cfg['parameter_budget'] and record['native']['task_worst_margin_drift']<=cfg['task_budget']
        n0=h0@me+sd(h0,p)*be
        gain_change=np.sum(source_delta*n0,1,keepdims=True)/np.sum(n0*n0,1,keepdims=True)
        radial=gain_change*n0;angular=source_delta-radial
        source_scale=dict(positive_numerator_gain_quantiles=np.quantile(1+gain_change,[0,.5,1.]).tolist(),
                          angular_residual_fraction=float(np.linalg.norm(angular)/np.linalg.norm(source_delta)),
                          interpretation='Positive source numerator gain preserves ordering, but its transported finite effect need not be collinear. Orthogonal decomposition is response geometry, not a new transform.')
        ge=q.T@me;decoder=np.linalg.pinv(ge,rcond=cfg['readout_rtol'])
        np.testing.assert_allclose(q@ge,me,atol=1e-9,rtol=1e-9)
        # An independent QR proves the saved full-rank transport rows lie in
        # col(Z_e); together with L Z_e=Z_f this checks the minimum-norm solve.
        assert record['source_numerical_rank']==ze.shape[1], 'Rank-deficient chart needs a separate verifier; do not silently certify it'
        qc,rc=qr(ze,mode='economic',check_finite=True)
        diagonal=float(np.abs(np.diag(rc)).min());assert diagonal>record['source_rank_tolerance']
        env_checks=[]
        for rec in record['environments']:
            bits=rec['environment'];name=''.join(map(str,bits));entry=saved['environments'][name]
            c={key:value+bits[0]*(proxy[key]-value) for key,value in native.items()}
            local={'A','a','B','b','gamma2','beta2'};readout={'gamma','beta','projection'}
            par={key:value+bits[1 if key in local else 2]*(pp[key]-value) if key in local|readout else value for key,value in p.items()}
            z,u,hb,ha,ds=forward(c,par,ch,da,db,cfg['coefficient']);mf,bf=geometry(par);gf=q.T@mf;cross=decoder@gf
            l=entry['transport'].numpy();uncovered=z-l@ze
            minimum_norm_error=float(np.linalg.norm(l-(l@qc)@qc.T)/max(np.linalg.norm(l),1e-20));assert minimum_norm_error<1e-6
            l1=np.sum(np.abs(l),1);np.testing.assert_allclose(np.quantile(l1,[.5,.95,1.]),list(rec['transport_l1'].values()),atol=1e-7,rtol=1e-7)
            rate=np.linalg.norm(uncovered)/np.linalg.norm(z);assert abs(rate-rec['activation_relative_uncovered'])<1e-7
            terms=dict(finite_compensation_mismatch=u@mf-cfg['coefficient']*l@(ue@me)@cross,
                       transported_native_residual=cfg['coefficient']*l@source_delta@cross,
                       activation_uncovered=cfg['coefficient']*uncovered@k.T@gf,
                       normalization_commutator=ds*bf-cfg['coefficient']*l@(dse*be)@cross)
            for key,value in terms.items():np.testing.assert_allclose(value,entry['terms'][key].numpy(),atol=2e-7,rtol=2e-6)
            nb=hb@mf+sd(hb,par)*bf;na=ha@mf+sd(ha,par)*bf
            identity_error=float(np.abs(sum(terms.values())-(na-nb)).max());assert identity_error<2e-7
            sb,sa=raw_scores(hb,par),raw_scores(ha,par)
            for val,key in [(sb,'scores0'),(sa,'scores1')]:np.testing.assert_allclose(val,entry[key].numpy(),atol=1e-10,rtol=1e-9)
            y=sb.argmax(1);changed=sa.argmax(1)!=y;rows=np.arange(len(y));comp=na.copy();comp[rows,y]=-np.inf;winner=comp.argmax(1)
            gap=nb[rows,y]-nb[rows,winner];harms={key:-(value[rows,y]-value[rows,winner])/gap for key,value in terms.items()}
            np.testing.assert_array_equal(sum(harms.values())>1,changed)
            assert int(changed.sum())==rec['changed'] and np.bincount(y[changed],minlength=len(p['head'])).tolist()==rec['changed_by_class']
            labels=gold[previous['indices']]
            assert int((y==labels).sum())==rec['original_correct'] and int((sa.argmax(1)==labels).sum())==rec['released_correct']
            for key,value in harms.items():np.testing.assert_allclose(value,entry['harm'][key].numpy(),atol=1e-6,rtol=1e-6)
            radial_term=cfg['coefficient']*l@radial@cross
            angular_term=cfg['coefficient']*l@angular@cross
            np.testing.assert_allclose(radial_term+angular_term,terms['transported_native_residual'],atol=1e-8,rtol=1e-8)
            residual_harm={key:float(np.mean(-(value[rows,y]-value[rows,winner])/gap))
                           for key,value in [('native_radial_gain',radial_term),('native_angular_residual',angular_term)]}
            for row in rec['per_class']:
                mask=y==row['label'];assert int(mask.sum())==row['n'] and int(changed[mask].sum())==row['changed']
                for key,value in harms.items():
                    if mask.any():assert abs(float(value[mask].mean())-row['mean_harm'][key])<1e-6
            env_checks.append(dict(environment=bits,minimum_norm_row_space_relative_error=minimum_norm_error,finite_identity_maximum_error=identity_error,changed=int(changed.sum()),transported_residual_harm_split=residual_harm,classification_and_per_class_counts='passed'))
        checks.append(dict(case=case['name'],source_qr_minimum_abs_diagonal=diagonal,source_response_scale=source_scale,environments=env_checks))
    result=dict(status='passed',checks=checks,diagnostic_sha256=report['diagnostic_sha256'],source_sha256=digest(__file__),
                wall_seconds=time.monotonic()-start,publication_updates=0,new_model_forward_instances=0,
                scope='Independent NumPy full finite native and hybrid responses, QR minimum-norm check, four-term identities, exact margin-crossing and official-label class recount. No unseen-data, unknown-peer or new release claim.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args();verify(args.output)
