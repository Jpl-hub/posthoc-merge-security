"""Independent NumPy mixed derivatives, task projection and finite predictions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores, residual


def activated(x):
    s=expit(1.702*x)
    return x*s, s+1.702*x*s*(1-s)


def score_jacobian(h,p):
    c=h-h.mean(1,keepdims=True);sd=np.sqrt(np.mean(c*c,1,keepdims=True)+p['eps'])
    t=(c/sd*p['gamma']+p['beta'])@p['projection'];tn=np.linalg.norm(t,axis=1,keepdims=True);f=t/tn;s=f@p['head'].T
    j=(p['head'][None,:,:]-s[:,:,None]*f[:,None,:])/tn[:,:,None]
    j=(j@p['projection'].T)*p['gamma']
    j=(j-j.mean(-1,keepdims=True)-c[:,None,:]*np.mean(j*c[:,None,:],-1,keepdims=True)/sd[:,:,None]**2)/sd[:,:,None]
    return s,j


def coefficients(item,p,ch,jh):
    x=ln(item['r'],p['gamma2'],p['beta2'],p['eps2']);z,slope=activated(x@p['A'].T+p['a'])
    if ch['site']=='O':
        r=item['r'];c=r-r.mean(1,keepdims=True);sd=np.sqrt(np.mean(c*c,1,keepdims=True)+p['eps2'])
        back=((jh@p['B'])*slope[:,None,:])@p['A']*p['gamma2']
        back=(back-back.mean(-1,keepdims=True)-c[:,None,:]*np.mean(back*c[:,None,:],-1,keepdims=True)/sd[:,:,None]**2)/sd[:,:,None]
        left=(jh+back)@ch['u'];right=item['o_input']@ch['vh'].T
    else:left=((jh@p['B'])*slope[:,None,:])@ch['u'];right=x@ch['vh'].T
    first=ch['na']*(left[:,:,:,None]*right[:,None,None,:])
    second=ch['nb']*((jh@ch['q'])[:,:,:,None]*z[:,None,None,:])
    return np.concatenate([first.reshape(len(x),jh.shape[1],-1),second.reshape(len(x),jh.shape[1],-1)],2)


def verify(output):
    start=time.monotonic();out=Path(output);report=json.loads((out/'results.json').read_text());path=out/'diagnostic.pt'
    assert digest(path)==report['diagnostic_sha256'];raw=torch.load(path,map_location='cpu');p=convert(raw['parameters']);pp=convert(raw['proxy_parameters'])
    labels=official_labels(report['target'],True);checks=[]
    for rec in report['records']:
        data=raw['cases'][rec['case']];ch=convert(data['chart']);native=convert(data['native']);proxy=convert(data['proxy'])
        y=data['winner'].numpy();comp=data['competitor'].numpy();gap=data['gap'].numpy();rows=np.arange(len(y));classes=len(p['head'])
        weights=np.eye(classes)[y].T/np.bincount(y,minlength=classes)[:,None]
        zero=np.zeros(data['frozen_theta'].numel());frozen=data['frozen_theta'].numpy()
        def env(t):
            c={k:v+t[0]*(proxy[k]-v) for k,v in native.items()}
            local={'A','a','B','b','gamma2','beta2'};readout={'gamma','beta','projection'}
            pars={k:v+t[1 if k in local else 2]*(pp[k]-v) if k in local|readout else v for k,v in p.items()}
            return c,pars
        def forward(t,theta,factor=1.):
            c,pars=env(t);size=ch['rank']**2
            da=ch['na']*ch['u']@theta[:size].reshape(ch['rank'],ch['rank'])@ch['vh']
            db=ch['nb']*ch['q']@theta[size:].reshape(ch['q'].shape[1],pars['B'].shape[1])
            r=c['r']+factor*c['o_input']@da.T if ch['site']=='O' else c['r']
            a=pars['A']+factor*da if ch['site']=='A' else pars['A']
            z,_=activated(ln(r,pars['gamma2'],pars['beta2'],pars['eps2'])@a.T+pars['a'])
            h=c['rounding']+r+z@(pars['B']+factor*db).T+pars['b']
            s=raw_scores(h,pars);s-=s.mean(1,keepdims=True)
            return s/np.linalg.norm(s,axis=1,keepdims=True)
        def values(t,theta):
            s=forward(t,theta)
            return -weights@((s[rows,y]-s[rows,comp])/gap)
        def gradient(t):
            c,pars=env(t);z,_=activated(ln(c['r'],pars['gamma2'],pars['beta2'],pars['eps2'])@pars['A'].T+pars['a'])
            h=c['rounding']+c['r']+z@pars['B'].T+pars['b'];s,jh=score_jacobian(h,pars)
            s-=s.mean(1,keepdims=True);norm=np.linalg.norm(s,axis=1,keepdims=True);q=s/norm
            jh=(jh-jh.mean(1,keepdims=True)-q[:,:,None]*np.sum(q[:,:,None]*jh,1,keepdims=True))/norm[:,:,None]
            jm=(jh[rows,y]-jh[rows,comp])/gap[:,None]
            return -weights@coefficients(c,pars,ch,jm[:,None,:])[:,0,:]
        g0=gradient(np.zeros(3));g1=gradient(np.ones(3))
        np.testing.assert_allclose(g0,data['g0'].numpy(),atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(g1,data['g1'].numpy(),atol=1e-8,rtol=1e-8)
        step=raw['cfg']['derivative_step'];components=[]
        for i in range(3):
            axis=np.eye(3)[i]*step;components.append((gradient(axis)-gradient(-axis))/(2*step))
        mixed=sum(components);expected=data['mixed'].numpy()
        error=np.linalg.norm(mixed-expected)/np.linalg.norm(expected);assert error<1e-6
        for actual,stored in zip(components,data['components']):np.testing.assert_allclose(actual,stored.numpy(),atol=1e-6,rtol=1e-5)
        # Independent four-corner differences test the mixed quantity through
        # actual normalized CLIP outputs, without differentiating the Jacobian.
        ht=hc=1e-4;four_errors=[]
        for direction in [data['direction'].numpy(),frozen/np.linalg.norm(frozen)]:
            measured=(values(np.ones(3)*ht,hc*direction)-values(np.ones(3)*ht,-hc*direction)
                     -values(-np.ones(3)*ht,hc*direction)+values(-np.ones(3)*ht,-hc*direction))/(4*ht*hc)
            target=expected@direction;err=np.linalg.norm(measured-target)/max(np.linalg.norm(target),1e-20)
            assert err<1e-3;four_errors.append(float(err))
        core=convert(data['core']);teacher,jh=score_jacobian(core['h'],p);jac=coefficients(core,p,ch,jh)
        yy=teacher.argmax(1);rr=np.arange(len(yy));keep=np.ones_like(teacher,dtype=bool);keep[rr,yy]=False
        gg=teacher[rr,yy,None]-teacher;gg[rr,yy]=1.
        matrix=((jac[rr,yy,None,:]-jac)/gg[:,:,None])[keep]
        projected=data['projected'].numpy();dual=data['projection_dual'].numpy()
        np.testing.assert_allclose(matrix.T@dual,(expected-projected).T,atol=1e-7,rtol=1e-7)
        null=float(np.abs(matrix@projected.T).max());assert null<1e-6
        alpha=data['alpha'].numpy();direction=data['direction'].numpy();assert alpha.min()>=0 and abs(alpha.sum()-1)<1e-12
        primal=float(np.min(projected@direction));upper=float(np.linalg.norm(alpha@projected));assert primal<=upper+1e-10
        assert abs(primal-rec['linear_maxmin_primal'])<1e-8 and abs(upper-rec['linear_maxmin_dual_upper'])<1e-8
        np.testing.assert_array_equal(labels[data['indices']],data['labels'].numpy())
        for row,t in zip(rec['frozen_counts'],[np.zeros(3),np.ones(3)]):
            b=forward(t,zero);a=forward(t,frozen,.5);pred=b.argmax(1);changed=a.argmax(1)!=pred
            assert int(changed.sum())==row['changed']
            assert int((pred==labels[data['indices']]).sum())==row['original_correct']
            assert np.bincount(y[changed],minlength=classes).tolist()==row['changed_by_native_class']
        exact=(values(np.ones(3),.5*frozen)-values(np.ones(3),zero))-(values(np.zeros(3),.5*frozen)-values(np.zeros(3),zero))
        np.testing.assert_allclose(exact,rec['frozen_class_exact_harm'],atol=1e-9,rtol=1e-9)
        linear=.5*(g1-g0)@frozen;np.testing.assert_allclose(linear,rec['frozen_class_linear_harm'],atol=1e-8,rtol=1e-8)
        finite=None
        if data['finite'] and data['finite']['theta'] is not None:
            theta=data['finite']['theta'].numpy();size=ch['rank']**2
            do=ch['na']*ch['u']@theta[:size].reshape(ch['rank'],ch['rank'])@ch['vh']
            db=ch['nb']*ch['q']@theta[size:].reshape(ch['q'].shape[1],p['B'].shape[1])
            old=next(c for c in raw['cfg']['cases'] if c['name']==rec['case']);assert digest(old['point'])==old['sha256']
            point=torch.load(old['point'],map_location='cpu');cfg=point['protocol'];n=cfg['probe_contexts'];results=[]
            for name in ['native','scale']:
                c=convert(point[name]);r=c['r'];o=c['o_input'];z,_=activated(ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'])
                rounding=c['h']-r-z@p['B'].T-p['b'];vb=point['basis_B']['vh'].numpy();vo=ch['vh'];shifts=[]
                for val,v in [(o,vo),(z,vb)]:
                    rms=np.sqrt(np.mean((val@v.T)**2,0));shifts.append((cfg['finite_step']*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None,:,:]).reshape(2*ch['rank'],-1))
                so,sb=shifts;count=len(so)
                rr=np.concatenate([r,(r[:n,None,:]+(so@p['O'].T)[None,:,:]).reshape(-1,r.shape[1]),np.repeat(r[:n],count,0)])
                oo=np.concatenate([o,(o[:n,None,:]+so[None,:,:]).reshape(-1,o.shape[1]),np.repeat(o[:n],count,0)])
                rounding=np.concatenate([rounding,np.repeat(rounding[:n],count,0),np.repeat(rounding[:n],count,0)])
                offset=np.concatenate([np.zeros_like(z),np.zeros((n*count,z.shape[1])),np.tile(sb,(n,1))])
                def pred(delta_o,delta_b):
                    r2=rr+oo@delta_o.T;zz=activated(ln(r2,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'])[0]+offset
                    return raw_scores(rounding+r2+zz@(p['B']+delta_b).T+p['b'],p)
                b=pred(np.zeros_like(do),np.zeros_like(db));a=pred(do,db)
                rho=float(np.abs(residual(b,a)).max());assert rho<=.5
                results.append({'group':name,'natural_n':len(r),'with_probes_n':len(rr),'worst_margin_drift':rho,'natural_changed':int((a[:len(r)].argmax(1)!=b[:len(r)].argmax(1)).sum())})
            expected_finite=rec['finite']['selected'];assert abs(max(v['worst_margin_drift'] for v in results)-expected_finite['task_worst_margin_drift'])<1e-7
            assert abs(np.linalg.norm(theta)-expected_finite['distance'])<1e-12
            before_proxy=forward(np.ones(3),zero);after_proxy=forward(np.ones(3),theta,.5)
            assert int((before_proxy.argmax(1)!=after_proxy.argmax(1)).sum())==expected_finite['proxy_changed']
            finite={'all_natural_and_all_frozen_singular_probes':results,'proxy_changed':expected_finite['proxy_changed'],'distance':float(np.linalg.norm(theta))}
        checks.append({'case':rec['case'],'mixed_jacobian_full_numpy_relative_error':float(error),'four_corner_output_relative_errors':four_errors,
                       'projection_row_space_identity':'passed','projection_null_maximum':null,'linear_maxmin_primal':primal,'linear_maxmin_dual_upper':upper,
                       'frozen_official_labels_and_class_counts':'passed','finite':finite})
    result={'status':'passed','checks':checks,'diagnostic_sha256':digest(path),'source_sha256':digest(__file__),'wall_seconds':time.monotonic()-start,
            'new_release_weights':0,'new_full_model_forward_instances':0,'publication_updates':0,'partner_used':False,'test_used':False,'confirmation_used':False,
            'scope':'NumPy normalized CLIP tail, all-coordinate analytic first Jacobians, mixed central differences, output four-corner differences, task projection identities, convex bounds and all finite natural/singular-probe constraints. Cached source/anchor models only; no true fusion or unknown-partner coverage claim.'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
