"""Finite geometric-space max-min class curvature and its convex relaxation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from clip_benchmark import digest
from subspace_probe import save_json


def solve(hessians,steps):
    """Any simplex weight gives a valid restricted-space spectral upper bound."""
    count=len(hessians)
    def spectral(weights):
        matrix=np.einsum('c,cij->ij',weights,hessians);values,vectors=np.linalg.eigh(matrix);q=vectors[:,-1]
        return float(values[-1]),np.einsum('i,cij,j->c',q,hessians,q)
    initial=np.ones(count)/count
    result=minimize(lambda w:spectral(w),initial,jac=True,method='SLSQP',bounds=[(0.,1.)]*count,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':steps,'ftol':1e-10,'disp':False})
    weights=np.maximum(result.x,0);weights/=weights.sum();matrix=np.einsum('c,cij->ij',weights,hessians)
    values,vectors=np.linalg.eigh(matrix);direction=vectors[:,-1];response=np.einsum('i,cij,j->c',direction,hessians,direction)
    return {'simplex_weights':weights.tolist(),'spectral_upper_bound':float(max(0,values[-1])),
        'unit_direction_worst_class_curvature':float(response.min()),'unit_direction_per_class':response.tolist(),
        'solver_success':bool(result.success),'solver_message':str(result.message),'solver_iterations':int(result.nit),
        'scope':'The bound is valid for these fixed16 directions for any returned simplex weights, even if the nonsmooth dual solver does not reach optimum. It is not an upper bound outside this restricted space.'},direction


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    path=Path(cfg['curvature_reference'])/'diagnostic.pt';assert digest(path)==cfg['curvature_diagnostic_sha256']
    data=torch.load(path,map_location='cpu');parent=json.loads(path.with_name('independent_validation.json').read_text());assert parent['status']=='passed'
    to=lambda d:{k:v.cuda().double() if torch.is_tensor(v) else v for k,v in d.items()}
    p=to(data['parameters']);core=to(data['core']);u,v=[data['basis_A'][k].cuda().double() for k in ['u','vh']]
    q=data['output_basis'].cuda().double();basis=data['krylov_basis'].cuda().double();r=len(v);size=r*r;dim=basis.shape[1]
    teacher=core['teacher'];winner=teacher.argmax(1);hess=[];records=[]
    def predict(z,subset):
        c=basis@z;ea=p['A'].norm()*u@c[:size].reshape(r,r)@v;eb=p['B'].norm()*q@c[size:].reshape(q.shape[1],p['B'].shape[1])
        pre=core['x'][subset]@(p['A']+ea).T+p['a'];hidden=pre*torch.sigmoid(1.702*pre)+core['offset'][subset]
        h=core['base'][subset]+hidden@(p['B']+eb).T
        # Elementary LayerNorm avoids the fused-layer double-backward discrepancy.
        centered=h-h.mean(1,keepdim=True);normal=centered/(centered.square().mean(1,keepdim=True)+p['eps']).sqrt()
        f=(normal*p['gamma']+p['beta'])@p['projection'];f=f/f.norm(dim=1,keepdim=True)
        return f@p['head'].T
    zero=torch.zeros(dim,device='cuda',dtype=torch.float64)
    objective_functions=[]
    for cls in range(teacher.shape[1]):
        subset=winner==cls;t=teacher[subset];rows=torch.arange(len(t),device='cuda');y=t.argmax(1)
        keep=torch.ones_like(t,dtype=torch.bool);keep[rows,y]=False;gap=(t[rows,y,None]-t).masked_fill(~keep,1.)
        def f(z,subset=subset,y=y,rows=rows,gap=gap,keep=keep):
            s=predict(z,subset);return ((s[rows,y,None]-s)/gap)[keep].mean()
        matrix=torch.autograd.functional.hessian(f,zero,create_graph=False);asym=float((matrix-matrix.T).abs().max());assert asym<1e-7
        matrix=(matrix+matrix.T)/2;hess.append(matrix.cpu().numpy());objective_functions.append(f)
        eig=torch.linalg.eigvalsh(matrix);records.append({'class':cls,'contexts':int(subset.sum()),'minimum_curvature':float(eig[0]),'maximum_curvature':float(eig[-1]),'hessian_symmetry_error':asym})
    hs=np.stack(hess);solution,z=solve(hs,cfg['dual_solver_steps']);z_t=torch.from_numpy(z).cuda();checks=[];h=cfg['hessian_fd_step']
    for cls,f in enumerate(objective_functions):
        fd=float((f(h*z_t)-2*f(zero)+f(-h*z_t))/h**2);analytic=float(z@hs[cls]@z)
        error=abs(fd-analytic)/max(1.,abs(analytic));assert error<cfg['hessian_relative_tolerance'];checks.append({'class':cls,'fd':fd,'analytic':analytic,'relative_error':error})
    file=out/'diagnostic.pt';torch.save({'hessians':torch.from_numpy(hs),'direction_in_fixed_basis':torch.from_numpy(z),'krylov_basis':data['krylov_basis'],
        'core':data['core'],'parameters':data['parameters'],'basis_A':data['basis_A'],'output_basis':data['output_basis'],'cfg':cfg},file)
    report={'status':'fixed_geometric_space_class_curvature_game','space_dimension':dim,'classes':records,'dual':solution,
        'quadratic_half_chord_mean_relative_margin_drop_upper_bound':cfg['parameter_budget']**2*solution['spectral_upper_bound']/8,
        'finite_difference_checks':checks,'coefficient_Hessian_rows':len(hess)*dim,'scalar_gradient_builds':len(hess),
        'publication_updates':0,'new_release_weights':0,'new_model_forwards':0,'proxy_models':0,'partner_used':False,'test_used':False,'confirmation_used':False,
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'diagnostic_sha256':digest(file),
        'limitation':'Finite task-context average across competing margins per class. A quadratic local bound does not certify every input, higher-order finite steps or full-space impossibility.'}
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;run(cfg)
