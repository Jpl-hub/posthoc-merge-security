"""Independent NumPy class Hessian forms and restricted minimax dual bound."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_levelset_audits import convert, raw_scores


def verify(output):
    out=Path(output);report=json.loads((out/'results.json').read_text());path=out/'diagnostic.pt';assert digest(path)==report['diagnostic_sha256']
    data=torch.load(path,map_location='cpu');p=convert(data['parameters']);core=convert(data['core']);basis=data['krylov_basis'].numpy()
    u,v=[data['basis_A'][k].numpy() for k in ['u','vh']];q=data['output_basis'].numpy();r=len(v);size=r*r;hs=data['hessians'].numpy()
    teacher=core['teacher'];y=teacher.argmax(1);winner_rows=np.arange(len(y));keep=np.ones_like(teacher,dtype=bool);keep[winner_rows,y]=False
    gap=teacher[winner_rows,y,None]-teacher;gap[~keep]=1.
    def values(z):
        t=basis@z;ea=np.linalg.norm(p['A'])*u@t[:size].reshape(r,r)@v;eb=np.linalg.norm(p['B'])*q@t[size:].reshape(q.shape[1],p['B'].shape[1])
        pre=core['x']@(p['A']+ea).T+p['a'];hidden=pre*expit(1.702*pre)+core['offset']
        s=raw_scores(core['base']+hidden@(p['B']+eb).T,p);ratios=(s[winner_rows,y,None]-s)/gap
        return np.array([ratios[(y==c)[:,None]&keep].mean() for c in range(len(p['head']))])
    rng=np.random.default_rng(20261017);a=rng.normal(size=basis.shape[1]);a/=np.linalg.norm(a);b=rng.normal(size=len(a));b/=np.linalg.norm(b)
    h=data['cfg']['hessian_fd_step'];zero=np.zeros_like(a);fd=(values(h*a)-2*values(zero)+values(-h*a))/h**2
    expected=np.einsum('i,cij,j->c',a,hs,a);relative=np.abs(fd-expected)/np.maximum(1,np.abs(expected));assert relative.max()<1e-3
    mixed=(values(h*(a+b))-values(h*(a-b))-values(h*(-a+b))+values(-h*(a+b)))/(4*h*h)
    expmixed=np.einsum('i,cij,j->c',a,hs,b);mixed_relative=np.abs(mixed-expmixed)/np.maximum(1,np.abs(expmixed));assert mixed_relative.max()<1e-3
    weights=np.array(report['dual']['simplex_weights']);assert weights.min()>=0 and abs(weights.sum()-1)<1e-12
    spectral=max(0,float(np.linalg.eigvalsh(np.einsum('c,cij->ij',weights,hs))[-1]))
    assert abs(spectral-report['dual']['spectral_upper_bound'])<1e-8
    d=data['direction_in_fixed_basis'].numpy();np.testing.assert_allclose(np.linalg.norm(d),1,atol=1e-10)
    individual=np.einsum('i,cij,j->c',d,hs,d);np.testing.assert_allclose(individual,report['dual']['unit_direction_per_class'],atol=1e-8,rtol=1e-8)
    result={'status':'passed','class_quadratic_relative_fd_error_maximum':float(relative.max()),'class_bilinear_relative_fd_error_maximum':float(mixed_relative.max()),
        'simplex_weights_verified':True,'restricted_spectral_upper_bound':spectral,'worst_class_direction_curvature':float(individual.min()),
        'source_sha256':digest(__file__),'diagnostic_sha256':digest(path),'new_model_forwards':0,'new_release_weights':0,'partner_used':False,'test_used':False,
        'scope':'Conditional16-dimensional geometric span only; valid spectral upper bound does not depend on optimizer convergence. Not a global max-min solution or a bound on actual classification accuracy.'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
