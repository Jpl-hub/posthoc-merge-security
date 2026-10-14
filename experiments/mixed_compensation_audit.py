"""Target/anchor mixed response of two frozen compensation charts; no release."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from angular_distance import signed_residual
from clip_benchmark import digest, merge
from cross_module_distance import CrossProblem, extended_parameters
from ffn_levelset import FFNProblem
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models
from verify_covariance_noise import official_labels


def ln(x, p):
    c = x - x.mean(-1, keepdim=True)
    return c / (c.square().mean(-1, keepdim=True) + p['eps2']).sqrt() * p['gamma2'] + p['beta2']


def activation(x):
    s = torch.sigmoid(1.702 * x)
    return x * s, s + 1.702 * x * s * (1-s)


def angle(h, p):
    # Exact ordering statistic, centered and normalized to remove common scale.
    head = p['head'] - p['head'].mean(0)
    m = p['gamma'][:, None] * (p['projection'] @ head.T)
    m = m - m.mean(0, keepdim=True)
    b = p['beta'] @ p['projection'] @ head.T
    c = h-h.mean(1, keepdim=True)
    sd = (c.square().mean(1, keepdim=True)+p['eps']).sqrt()
    a = h@m+sd*b
    norm = a.norm(dim=1, keepdim=True)
    q = a/norm
    ja = m.T[None, :, :] + b[None, :, None]*c[:, None, :]/(h.shape[1]*sd[:, :, None])
    jq = (ja-q[:, :, None]*(q[:, :, None]*ja).sum(1, keepdim=True))/norm[:, :, None]
    return q, jq


def coefficient_jacobian(item, p, chart, jh):
    x = ln(item['r'], p)
    z, slope = activation(x@p['A'].T+p['a'])
    if chart['site'] == 'O':
        r = item['r']; c = r-r.mean(1, keepdim=True)
        sd = (c.square().mean(1, keepdim=True)+p['eps2']).sqrt()
        back = ((jh@p['B'])*slope[:, None, :])@p['A']*p['gamma2']
        back = (back-back.mean(-1, keepdim=True)-c[:, None, :]*(back*c[:, None, :]).mean(-1, keepdim=True)/sd[:, :, None].square())/sd[:, :, None]
        left = (jh+back)@chart['u']; right = item['o_input']@chart['vh'].T
    else:
        left = ((jh@p['B'])*slope[:, None, :])@chart['u']; right = x@chart['vh'].T
    first = chart['na']*(left[:, :, :, None]*right[:, None, None, :])
    second = chart['nb']*((jh@chart['q'])[:, :, :, None]*z[:, None, None, :])
    return torch.cat([first.flatten(2), second.flatten(2)], 2)


def pack(raw, p):
    c = on_device(raw)
    z, _ = activation(ln(c['r'], p)@p['A'].T+p['a'])
    return {'r':c['r'], 'o_input':c['o_input'], 'rounding':c['h']-c['r']-z@p['B'].T-p['b']}


def environment(t, native, proxy, p, pp):
    c = {k:v+t[0]*(proxy[k]-v) for k,v in native.items()}
    local = {'A','a','B','b','gamma2','beta2'}
    readout = {'gamma','beta','projection'}
    par = {k:v+t[1 if k in local else 2]*(pp[k]-v) if k in local|readout else v for k,v in p.items()}
    return c, par


def response(item, p, chart, theta, factor):
    n = chart['rank']**2
    da = chart['na']*chart['u']@theta[:n].reshape(chart['rank'],chart['rank'])@chart['vh']
    db = chart['nb']*chart['q']@theta[n:].reshape(chart['q'].shape[1],p['B'].shape[1])
    r = item['r']+factor*item['o_input']@da.T if chart['site']=='O' else item['r']
    a = p['A']+factor*da if chart['site']=='A' else p['A']
    z, _ = activation(ln(r,p)@a.T+p['a'])
    h = item['rounding']+r+z@(p['B']+factor*db).T+p['b']
    return angle(h,p)[0]


def cosine(a,b):
    return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))


def problem_from_point(point,p,chart):
    cls = CrossProblem if chart['site']=='O' else FFNProblem
    obj = cls.__new__(cls); obj.p=p; obj.cfg=point['protocol']
    obj.r=chart['rank']; obj.a_size=obj.r**2; obj.b_rows=chart['q'].shape[1]; obj.b_width=p['B'].shape[1]
    obj.na=chart['na']; obj.nb=chart['nb']
    obj.basis={'A':{'u':chart['u'],'vh':chart['vh']}, 'B':on_device(point['basis_B'])}
    obj.output_basis=chart['q']; obj.q=chart['q']; obj.ua=chart['u']; obj.va=chart['vh']
    obj.native=[obj.prepare(point[k],p,True) for k in ['native','scale']]
    return obj


def audit(cfg):
    out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    anchor,expert,head,pins=checked_models(cfg['target']); p=on_device(extended_parameters(expert,head))
    proxy_model=merge(anchor,expert,anchor,cfg['coefficient']); pp=on_device(extended_parameters(proxy_model,head)); del proxy_model
    gold=official_labels(cfg['target'],True)
    raw={'parameters':{k:v.cpu() if torch.is_tensor(v) else v for k,v in p.items()},
         'proxy_parameters':{k:v.cpu() if torch.is_tensor(v) else v for k,v in pp.items()},'cases':{},'cfg':cfg}
    records=[]; jvp_calls=0
    for case in cfg['cases']:
        assert digest(case['point'])==case['sha256']
        point=torch.load(case['point'],map_location='cpu'); site='O' if case['name']=='O_B' else 'A'
        basis=on_device(point['basis_'+site]); q=point['output_basis'].cuda().double()
        chart={'site':site,'rank':len(basis['vh']),'u':basis['u'],'vh':basis['vh'],'q':q,'na':p[site].norm(),'nb':p['B'].norm()}
        ids=point['indices']['proxy'][:cfg['diagnostic_n']]; positions={i:k for k,i in enumerate(point['indices']['fit'])}
        offsets=[positions[i] for i in ids]
        nc={k:v[offsets] for k,v in point['native'].items()}; pc={k:v[:len(ids)] for k,v in point['proxy'].items()}
        native,proxy=pack(nc,p),pack(pc,pp); zero3=torch.zeros(3,device='cuda',dtype=torch.float64)
        ztheta=torch.zeros_like(point['theta'],device='cuda'); theta=point['theta'].cuda().double()
        before=response(native,p,chart,ztheta,1.); y=before.argmax(1); rows=torch.arange(len(y),device='cuda')
        other=before.clone(); other[rows,y]=-torch.inf; competitor=other.argmax(1)
        gap=before[rows,y]-before[rows,competitor]; counts=torch.bincount(y,minlength=before.shape[1]); assert bool((counts>0).all())
        weights=torch.nn.functional.one_hot(y,len(counts)).T.double()/counts[:,None]
        def gradient(t):
            c,par=environment(t,native,proxy,p,pp)
            h=c['rounding']+c['r']+activation(ln(c['r'],par)@par['A'].T+par['a'])[0]@par['B'].T+par['b']
            _,jh=angle(h,par); jm=(jh[rows,y]-jh[rows,competitor])/gap[:,None]
            return -weights@coefficient_jacobian(c,par,chart,jm[:,None,:])[:,0,:]
        g0=gradient(zero3); g1=gradient(torch.ones_like(zero3))
        component=[]
        for k in range(3):
            v=torch.zeros_like(zero3);v[k]=1
            component.append(torch.autograd.functional.jvp(gradient,zero3,v)[1].detach()); jvp_calls+=1
        mixed=sum(component)
        eps=cfg['derivative_step']; fd=(gradient(zero3+eps)-gradient(zero3-eps))/(2*eps)
        fd_error=float((fd-mixed).norm()/mixed.norm()); assert fd_error<cfg['derivative_relative_tolerance']
        integrals=[]
        for order in cfg['quadrature_orders']:
            nodes,ws=np.polynomial.legendre.leggauss(order); integral=torch.zeros_like(g0)
            for node,w in zip((nodes+1)/2,ws/2):
                _,value=torch.autograd.functional.jvp(gradient,torch.full_like(zero3,float(node)),torch.ones_like(zero3)); jvp_calls+=1
                integral+=float(w)*value.detach()
            integrals.append(integral)
        n=point['protocol']['contexts_per_class']*len(counts)
        core={k:torch.cat([point[g][k][:n] for g in ['native','scale']]).cuda().double() for k in nc}
        jh,_,_,teacher=raw_jacobians(core,p); jac=coefficient_jacobian(core,p,chart,jh)
        yy=teacher.argmax(1); rr=torch.arange(len(yy),device='cuda'); keep=torch.ones_like(teacher,dtype=torch.bool);keep[rr,yy]=False
        gg=(teacher[rr,yy,None]-teacher).masked_fill(~keep,1.)
        matrix=((jac[rr,yy,None,:]-jac)/gg[:,:,None])[keep]
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False); rank=int((s>torch.finfo(s.dtype).eps*max(matrix.shape)*s[0]).sum())
        u,s,vh=u[:,:rank],s[:rank],vh[:rank]
        projected=mixed-(mixed@vh.T)@vh
        projection_dual=(u/s[None,:])@(vh@mixed.T)
        gram=(projected@projected.T).cpu().numpy(); scale=max(float(np.linalg.norm(gram,2)),1e-30)
        solved=minimize(lambda a: .5*float(a@gram@a)/scale,np.ones(len(counts))/len(counts),
                        jac=lambda a:gram@a/scale,bounds=[(0.,1.)]*len(counts),
                        constraints={'type':'eq','fun':lambda a:a.sum()-1.,'jac':lambda a:np.ones_like(a)},
                        method='SLSQP',options={'maxiter':cfg['dual_steps'],'ftol':1e-12})
        alpha=np.maximum(solved.x,0);alpha/=alpha.sum(); combo=torch.tensor(alpha,device='cuda')@projected
        direction=combo/combo.norm().clamp_min(1e-30)
        pair=response(proxy,pp,chart,theta,.5)-response(native,p,chart,theta,.5)
        base=response(proxy,pp,chart,ztheta,1.)-before
        exact=-(pair-base)[rows,y]+(pair-base)[rows,competitor]
        exact/=gap
        linear_per_class=.5*(g1-g0)@theta
        exact_per_class=weights@exact
        frozen=[]
        for name,c,par in [('native',native,p),('anchor_interpolation',proxy,pp)]:
            b=response(c,par,chart,ztheta,1.); a=response(c,par,chart,theta,.5); pred=b.argmax(1); changed=a.argmax(1)!=pred
            frozen.append({'environment':name,'n':len(ids),'original_correct':int((pred.cpu()==torch.tensor(gold[ids])).sum()),
                           'changed':int(changed.sum()),'changed_by_native_class':torch.bincount(y[changed],minlength=len(counts)).cpu().tolist()})
        finite=None; finite_raw=None
        if case['name']==cfg['finite_direction_case']:
            problem=problem_from_point(point,p,chart)
            trials=[]; selected=None
            for i in range(cfg['backtracking_steps']):
                step=cfg['parameter_budget']*.5**i; t=step*direction
                with torch.no_grad():
                    rho=float(problem.residuals(t).abs().max())
                    if rho<=cfg['margin_fraction']:selected=t;break
                trials.append({'step':step,'task_worst_margin_drift':rho})
            if selected is not None:
                with torch.no_grad():
                    proxy_scores=response(proxy,pp,chart,selected,.5); proxy_base=response(proxy,pp,chart,ztheta,1.)
                    changed=proxy_scores.argmax(1)!=proxy_base.argmax(1)
                    nt=problem.predict(problem.native[0],p,selected); b=problem.native[0]['teacher']
                    finite={'distance':float(selected.norm()),'task_worst_margin_drift':rho,'natural_n':len(point['native']['h']),
                            'natural_prediction_changes':int((nt[:len(point['native']['h'])].argmax(1)!=b[:len(point['native']['h'])].argmax(1)).sum()),
                            'proxy_n':len(ids),'proxy_changed':int(changed.sum()),'proxy_changed_by_native_class':torch.bincount(y[changed],minlength=len(counts)).cpu().tolist()}
            finite_raw={'theta':selected.cpu() if selected is not None else None,'trials':trials}
            finite={'trials':trials,'selected':finite,'role':'One new mixed-derivative direction from the origin; task-only finite feasibility, no publication optimization.'}
        rec={'case':case['name'],'n':len(ids),'teacher_class_counts':counts.cpu().tolist(),'task_row_rank':rank,
             'mixed_fd_relative_error':fd_error,'component_norms':dict(zip(['upstream','local_operator','readout'],[float(v.norm()) for v in component])),
             'quadrature_orders':cfg['quadrature_orders'],'integrated_mixed_relative_errors':[float((v-(g1-g0)).norm()/(g1-g0).norm()) for v in integrals],
             'source_mixed_to_finite_secant_cosine':cosine(mixed,g1-g0),'projected_mixed_to_proxy_gradient_cosine':cosine(projected.mean(0),g1.mean(0)-vh.T@(vh@g1.mean(0))),
             'projected_mixed_retained_fraction':float(projected.norm()/mixed.norm()),'projected_mixed_task_null_max':float((matrix@projected.T).abs().max()),
             'linear_maxmin_primal':float((projected@direction).min()),'linear_maxmin_dual_upper':float(combo.norm()),'dual_solver_success':bool(solved.success),'dual_solver_message':str(solved.message),
             'frozen_mixed_linear_vs_exact_class_error':float((linear_per_class-exact_per_class).norm()/exact_per_class.norm().clamp_min(1e-30)),
             'frozen_class_exact_harm':exact_per_class.cpu().tolist(),'frozen_class_linear_harm':linear_per_class.cpu().tolist(),'frozen_counts':frozen,'finite':finite}
        records.append(rec);print(json.dumps(rec),flush=True)
        raw['cases'][case['name']]={'chart':{k:v.cpu() if torch.is_tensor(v) else v for k,v in chart.items()},'native':{k:v.cpu() for k,v in native.items()},
            'proxy':{k:v.cpu() for k,v in proxy.items()},'core':{k:v.cpu() for k,v in core.items()},'indices':ids,'labels':torch.tensor(gold[ids]),'winner':y.cpu(),'competitor':competitor.cpu(),'gap':gap.cpu(),
            'g0':g0.detach().cpu(),'g1':g1.detach().cpu(),'components':[v.cpu() for v in component],'mixed':mixed.cpu(),'projected':projected.cpu(),'projection_dual':projection_dual.cpu(),
            'quadratures':[v.cpu() for v in integrals],'alpha':torch.tensor(alpha),'direction':direction.cpu(),'frozen_theta':theta.cpu(),'finite':finite_raw,'cfg_point':point['protocol']}
    file=out/'diagnostic.pt';torch.save(raw,file)
    report={'status':'target_anchor_mixed_compensation_diagnosis','target':cfg['target'],'records':records,'parameter_budget':cfg['parameter_budget'],'task_budget':cfg['margin_fraction'],
        'diagnostic_sha256':digest(file),'checkpoint_sha256':pins,'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),
        'read_only_mixed_JVP_calls':jvp_calls,'new_full_model_forward_instances':0,'publication_optimization_steps':0,'new_release_weights':0,
        'partner_used':False,'test_used':False,'confirmation_used':False,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'limitation':'One observable target/anchor secant, not coverage of unknown peers. Intermediate tail environments are an integration path, not independently selected model proxies. A symmetric uncertainty set cannot force uniformly signed first-order harm.'}
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for name,pin in cfg['source_sha256'].items():assert digest(name)==pin,name
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    audit(cfg)
