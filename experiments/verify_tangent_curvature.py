"""Independent NumPy margin Hessian and finite task-envelope verification."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores, residual


def verify(output,finite_output=None):
    root=Path(output);report=json.loads((root/'results.json').read_text());path=root/'diagnostic.pt';assert digest(path)==report['diagnostic_sha256']
    data=torch.load(path,map_location='cpu');cfg=data['cfg'];out=root
    if finite_output:
        out=Path(finite_output);report=json.loads((out/'results.json').read_text());path=out/'diagnostic.pt';assert digest(path)==report['diagnostic_sha256']
        point=torch.load(path,map_location='cpu');data['theta'],data['direction']=point['theta'],point['direction']
    p=convert(data['parameters']);u,v=[data['basis_A'][k].numpy() for k in ['u','vh']]
    q=data['output_basis'].numpy();r=len(v);size=r*r;direction=data['direction'].numpy();theta=data['theta'].numpy()
    task_rows=data['task_row_basis'].numpy();np.testing.assert_allclose(task_rows@direction,0,atol=1e-9)
    assert np.linalg.norm(theta)<=cfg['parameter_budget']
    def predict(packed,t,factor=1.):
        c=convert(packed);ea=np.linalg.norm(p['A'])*u@t[:size].reshape(r,r)@v;eb=np.linalg.norm(p['B'])*q@t[size:].reshape(q.shape[1],p['B'].shape[1])
        pre=c['x']@(p['A']+factor*ea).T+p['a'];z=pre*expit(1.702*pre)+c['offset']
        return raw_scores(c['base']+z@(p['B']+factor*eb).T,p)
    core=data['core'];before=core['teacher'].numpy();y=before.argmax(1);rows=np.arange(len(y));keep=np.ones_like(before,dtype=bool);keep[rows,y]=False
    gap=before[rows,y,None]-before;gap[~keep]=1.
    count=np.bincount(y,minlength=before.shape[1]);weights=1./count[y]/before.shape[1]/(before.shape[1]-1)
    def objective(t):
        s=predict(core,t);return np.sum(weights[:,None]*np.where(keep,(s[rows,y,None]-s)/gap,0))
    zero=np.zeros_like(direction);h=cfg['curvature_fd_step'];fd=(objective(h*direction)-2*objective(zero)+objective(-h*direction))/h**2
    if not finite_output:
        assert abs(fd-report['direction_curvature'])/max(1,abs(report['direction_curvature']))<cfg['hessian_relative_tolerance']
        k=data['krylov_basis'].numpy();tri=data['tridiagonal'].numpy();np.testing.assert_allclose(k.T@k,np.eye(k.shape[1]),atol=1e-9)
        eig,vec=np.linalg.eigh(tri);ritz=k@vec[:,-1];alignment=abs(ritz@direction);assert 1-alignment<1e-9
        savedhd=data['projected_hessian_direction'].numpy();resnorm=float(np.linalg.norm(savedhd-float(direction@savedhd)*direction))
        assert abs(resnorm-report['Ritz_residual_norm'])<1e-8
    old=Path(cfg['old_calibration_reference'])/'calibration.pt';assert digest(old)==cfg['old_calibration_sha256']
    cached=torch.load(old,map_location='cpu');vb=cached['basis_B']['vh'].numpy()
    checks=[]
    for name,raw in [('native',data['native']),('scale',cached['scale'])]:
        c=convert(raw);x=ln(c['r'],p['gamma2'],p['beta2'],p['eps2']);pre=x@p['A'].T+p['a'];z=pre*expit(1.702*pre);base=c['h']-z@p['B'].T
        if name=='scale':
            s=raw_scores(c['h'],p);yy=s.argmax(1);others=s.copy();others[np.arange(len(s)),yy]=-np.inf;g=s[np.arange(len(s)),yy]-others.max(1)
            selected=[]
            for cls in range(s.shape[1]):
                members=np.flatnonzero(yy==cls);selected.extend(members[np.argsort(g[members],kind='stable')[:cfg['contexts_per_class']]].tolist())
            order=selected+[i for i in range(len(s)) if i not in set(selected)];x,z,base=x[order],z[order],base[order]
        n=cfg['contexts_per_class']*before.shape[1];shifts=[]
        for val,vv in [(x,v),(z,vb)]:
            rms=np.sqrt(np.mean((val@vv.T)**2,axis=0));shifts.append((cfg['finite_step']*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*vv[None,:,:]).reshape(2*r,-1))
        sa,sb=shifts;num=len(sa);packed={'x':np.concatenate([x,(x[:n,None,:]+sa).reshape(-1,x.shape[1]),np.repeat(x[:n],num,axis=0)]),
            'base':np.concatenate([base,np.repeat(base[:n],num,axis=0),np.repeat(base[:n],num,axis=0)]),
            'offset':np.concatenate([np.zeros_like(z),np.zeros((n*num,z.shape[1])),np.tile(sb,(n,1))])}
        packed={key:torch.from_numpy(val) for key,val in packed.items()}
        teacher=predict(packed,zero);endpoint=predict(packed,theta);mid=predict(packed,theta,.5)
        rho=float(np.abs(residual(teacher,endpoint)).max());assert rho<=cfg['margin_fraction']
        changed=mid.argmax(1)!=teacher.argmax(1)
        checks.append({'group':name,'natural_n':len(x),'natural_plus_probes_n':len(teacher),'endpoint_worst_margin_drift':rho,
            'midpoint_changed_natural':int(changed[:len(x)].sum()),'midpoint_changed_natural_plus_probes':int(changed.sum())})
    worst=max(c['endpoint_worst_margin_drift'] for c in checks);assert abs(worst-report['final']['task_worst_margin_drift'])<1e-7
    key='natural_midpoint_changed' if finite_output else 'source_midpoint_changed'
    checked='midpoint_changed_natural' if finite_output else 'midpoint_changed_natural_plus_probes'
    assert checks[0][checked]==report['final'][key]
    result={'status':'passed','checks':checks,'curvature_directional_fd':fd,'source_sha256':digest(__file__),'diagnostic_sha256':digest(path),
        'original_report_source_n_scope':'source_n=6656 includes4096 natural inputs and2560 finite probes; both counts independently separated here.',
        'new_model_forwards':0,'new_release_weights':0,'partner_used':False,'test_used':False,
        'scope':'One finite task-tangent Ritz direction and corrected point; no global extremum, infeasibility or unknown-peer guarantee.'}
    if not finite_output:result.update(analytic_curvature=report['direction_curvature'],curvature_relative_error=abs(fd-report['direction_curvature'])/abs(report['direction_curvature']),Krylov_direction_alignment=alignment,Ritz_residual_norm=resnorm)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--finite-output');args=parser.parse_args();torch.set_num_threads(4);verify(args.output,args.finite_output)
