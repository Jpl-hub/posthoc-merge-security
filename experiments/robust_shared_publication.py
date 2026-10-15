"""Bounded O/B publication against two source-derived realizable opponents.

The opponents are the zero environment and the independently solved common
prefix environment. This is a finite empirical game, NOT a continuous-family
or unknown-partner robustness guarantee. No peer/test/confirmation selection.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from attention_cascade import suffix
from clip_benchmark import digest
from cross_module_distance import CrossProblem, normalized_input
from joint_compensation_trust import positive_root
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import gelu


def frozen_problem(point,pars,cfg):
    p=CrossProblem.__new__(CrossProblem);p.target=cfg['target'];p.cfg=point['protocol'].copy()
    p.r=cfg['probe_rank'];p.a_size=p.r**2;p.b_rows=point['output_basis'].shape[1];p.b_width=pars['B'].shape[1]
    p.p=pars;p.basis={'A':on_device(point['basis_O']),'B':on_device(point['basis_B'])}
    p.output_basis=point['output_basis'].cuda().double();p.na=pars['O'].norm();p.nb=pars['B'].norm()
    p.native=[p.prepare(point[group],pars,True) for group in ['native','scale']]
    return p


class RobustPublication:
    def __init__(self,cfg):
        self.cfg=cfg
        for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
        ev=Path(cfg['environment_root']);meta=json.loads((ev/'results.json').read_text());validation=json.loads((ev/'independent_validation.json').read_text())
        assert meta['finite_multiclass_signal'] and validation['status']=='passed' and validation['finite_multiclass_signal']
        assert validation['diagnostic_sha256']==cfg['inputs'][str(ev/'diagnostic.pt')]
        cal=torch.load(cfg['cache'],map_location='cpu');self.pars=on_device(cal['parameters']);self.pp=on_device(cal['proxy_parameters'])
        point=torch.load(cfg['initial_point'],map_location='cpu');self.p=frozen_problem(point,self.pars,cfg)
        self.initial=point['theta'].cuda().double();self.source={key:torch.cat([x[key] for x in self.p.native]) for key in self.p.native[0]}
        self.teacher=self.source['teacher'];self.y=self.teacher.argmax(1)
        self.natural_rows=torch.cat([torch.arange(len(point['native']['r']),device='cuda'),
            len(self.p.native[0]['r'])+torch.arange(len(point['scale']['r']),device='cuda')])
        nc=self.p.cfg['probe_contexts'];self.core=torch.cat([torch.arange(nc,device='cuda'),len(self.p.native[0]['r'])+torch.arange(nc,device='cuda')])
        env=torch.load(ev/'diagnostic.pt',map_location='cpu')['theta'].cuda().reshape(cfg['probe_rank'],cfg['probe_rank'])
        basis=on_device(cal['basis']);raw=on_device(cal['proxy']);self.environments=[]
        # The pair is fixed before publication optimization. No environment
        # count search, future partner or test-dependent opponent choice.
        for name,s in [('zero',torch.zeros_like(env)),('learned_shared',env)]:
            h=raw['H'].double()+self.pars['D'].norm()*(raw['t'].double()@s.T)@basis['u'].T
            out,context=suffix(h,self.pp);o=context['average'].flatten(1)
            item=dict(r=context['r'],o_input=o,rounding=raw['rounding'],
                b_offset=torch.zeros(len(h),self.pp['B'].shape[1],device='cuda',dtype=torch.float64),
                task_input=o@self.p.basis['A']['vh'].T)
            item['teacher']=scores(out+raw['rounding'],self.pp)
            y=item['teacher'].argmax(1);other=item['teacher'].clone();other[torch.arange(len(y),device='cuda'),y]=-torch.inf
            gap=item['teacher'][torch.arange(len(y),device='cuda'),y]-other.max(1).values
            self.environments.append(dict(name=name,data=item,y=y,gap=gap))
        self.indices=point['indices'];self.dimension=len(self.initial)

    def objective(self,theta,return_scores=False):
        values=[];predictions=[]
        for env in self.environments:
            s=self.p.predict(env['data'],self.pp,theta,.5);y=env['y'];row=torch.arange(len(y),device='cuda')
            other=s.clone();other[row,y]=-torch.inf
            margin=(s[row,y]-other.max(1).values)/env['gap']
            damage=torch.sigmoid(-margin/self.cfg['margin_temperature'])
            values.extend(damage[y==c].mean() for c in range(s.shape[1]));predictions.append(s)
        a=torch.stack(values);tau=self.cfg['opponent_temperature']
        game=-tau*(torch.logsumexp(-a/tau,0)-math.log(len(a)))
        obj=game+self.cfg['distance_weight']*theta.square().sum()
        return (obj,a,predictions) if return_scores else obj

    @torch.no_grad()
    def evaluate(self,theta):
        source=self.p.predict(self.source,self.pars,theta);err=signed_residual(self.teacher,source)
        obj,a,envs=self.objective(theta,True);records=[]
        for env,ss in zip(self.environments,envs):
            y=env['y'];changed=ss.argmax(1)!=y
            records.append(dict(name=env['name'],n=len(y),changed=int(changed.sum()),classes_changed=sum(bool((changed&(y==c)).any()) for c in range(ss.shape[1])),
                per_class=[dict(label=c,n=int((y==c).sum()),changed=int((changed&(y==c)).sum())) for c in range(ss.shape[1])]))
        metrics=dict(parameter_distance=float(theta.norm()),worst_task_drift=float(err.abs().max()),
            natural_changes=int((source[self.natural_rows].argmax(1)!=self.y[self.natural_rows]).sum()),
            objective=float(obj),environment_class_damage=a.cpu().tolist(),environments=records,
            worst_environment_changed=min(x['changed'] for x in records),worst_environment_classes=min(x['classes_changed'] for x in records))
        return dict(source=source,residual=err,environment_scores=envs,metrics=metrics)

    @torch.no_grad()
    def factors(self,theta,state):
        risk=state['residual'].abs().max(1).values;rows=set(self.core.cpu().tolist())
        for cls in range(len(self.pars['head'])):
            members=(self.y==cls).nonzero().flatten();rows.add(int(members[risk[members].argmax()]))
        selected=torch.tensor(sorted(rows),device='cuda');item={k:v[selected] for k,v in self.source.items()}
        co=theta[:self.p.a_size].reshape(self.p.r,self.p.r);k=theta[self.p.a_size:].reshape(self.p.b_rows,self.p.b_width)
        r=item['r']+self.p.na*(item['task_input']@co.T)@self.p.basis['A']['u'].T
        z=gelu(normalized_input(r,self.pars)@self.pars['A'].T+self.pars['a'])+item['b_offset']
        b=self.pars['B']+self.p.nb*self.p.output_basis@k
        h=item['rounding']+r+z@b.T+self.pars['b'];jh,jr,_,now=raw_jacobians({'r':r,'h':h},self.pars|{'B':b})
        left=self.p.na*(jr@self.p.basis['A']['u']);right=self.p.nb*(jh@self.p.output_basis)
        teacher=item['teacher'];y=teacher.argmax(1);ar=torch.arange(len(y),device='cuda');keep=torch.ones_like(teacher,dtype=torch.bool);keep[ar,y]=False
        gap=(teacher[ar,y,None]-teacher).masked_fill(~keep,1.)
        aa=((left[ar,y,None]-left)/gap[:,:,None])[keep];bb=((right[ar,y,None]-right)/gap[:,:,None])[keep]
        ss=item['task_input'][:,None].expand(-1,len(self.pars['head']),-1)[keep];zz=z[:,None].expand(-1,len(self.pars['head']),-1)[keep]
        gram=(aa@aa.T)*(ss@ss.T)+(bb@bb.T)*(zz@zz.T)
        eigen,v=torch.linalg.eigh(gram);tol=torch.finfo(eigen.dtype).eps*max(len(gram),len(theta))*eigen.abs().max();ok=eigen>tol
        inverse=torch.where(ok,1/eigen.clamp_min(tol),0.)
        def solve(value):return v@((v.T@value)*inverse)
        def jv(value):
            c=value[:self.p.a_size].reshape(self.p.r,self.p.r);k=value[self.p.a_size:].reshape(self.p.b_rows,self.p.b_width)
            return (aa*(ss@c.T)).sum(1)+(bb*(zz@k.T)).sum(1)
        def jt(value):return torch.cat([(aa.T@(value[:,None]*ss)).flatten(),(bb.T@(value[:,None]*zz)).flatten()])
        return dict(selected=selected,residual=signed_residual(teacher,now)[keep],solve=solve,jv=jv,jt=jt,rank=int(ok.sum()))


@torch.no_grad()
def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    p=RobustPublication(cfg);theta=p.initial.clone();state=p.evaluate(theta);initial=state['metrics']
    assert initial['parameter_distance']<=.5 and initial['worst_task_drift']<=.5 and initial['natural_changes']==0
    radius=cfg['trust_initial'];history=[];snaps=[];accepted=0;reason='fixed_trial_cap';gradient_calls=0
    for trial in range(cfg['maximum_trials']):
        f=p.factors(theta,state)
        with torch.enable_grad():
            leaf=theta.detach().clone().requires_grad_(True);value=p.objective(leaf);g=torch.autograd.grad(value,leaf)[0]
        gradient_calls+=1;tangent=g-f['jt'](f['solve'](f['jv'](g)))
        if tangent.norm()<1e-15:reason='no_source_feasible_tangent';break
        if theta.norm()>=cfg['parameter_budget']*.98:
            radial=theta-f['jt'](f['solve'](f['jv'](theta)))
            dot=float(tangent@radial)
            if dot>0 and radial.norm()>1e-15:tangent-=radial*dot/(radial@radial)
        step=radius*tangent/tangent.norm();alpha=min(1.,positive_root(float(step@step),float(theta@step),float(theta@theta)-(.5*(1-1e-12))**2));step*=alpha
        predicted=float(g@step)
        if predicted<=1e-14:reason='no_positive_feasible_gain';break
        eps=cfg['derivative_step'];fd=float((p.objective(theta+eps*step)-p.objective(theta-eps*step))/(2*eps));assert abs(fd-predicted)<cfg['derivative_tolerance']
        proposal=p.evaluate(theta+step);actual=proposal['metrics']['objective']-state['metrics']['objective'];ratio=actual/predicted
        linear=float((f['residual']+f['jv'](step)).abs().max());m=proposal['metrics']
        ok=bool(m['parameter_distance']<=.5 and m['worst_task_drift']<=.5 and m['natural_changes']==0 and linear<=.5 and actual>1e-12 and ratio>=cfg['acceptance_ratio'])
        record=dict(trial=trial+1,accepted=ok,proposal=m,before=state['metrics'],predicted_gain=predicted,actual_gain=actual,agreement_ratio=ratio,active_linear_worst=linear,
            trust_radius=radius,active_rank=f['rank'],active_rows=f['selected'].cpu().tolist(),fd_error=abs(fd-predicted))
        history.append(record);snaps.append(dict(theta=theta.cpu(),step=step.cpu(),gradient=g.cpu(),source=proposal['source'].cpu(),
            environment_scores=[x.cpu() for x in proposal['environment_scores']],active_r=f['residual'].cpu(),active_Jstep=f['jv'](step).cpu()))
        print(json.dumps(record),flush=True)
        if ok:
            theta+=step;state=proposal;accepted+=1
            if ratio>=cfg['expansion_ratio']:radius=min(cfg['trust_maximum'],2*radius)
        else:radius*=.5
        if radius<cfg['trust_minimum']:reason='finite_source_feasibility_radius_exhausted';break
    m=state['metrics'];signal=(m['worst_environment_changed']>=cfg['minimum_changed'] and m['worst_environment_classes']>=cfg['minimum_changed_classes'] and m['objective']>initial['objective'])
    delta={'O':p.p.na*p.p.basis['A']['u']@theta[:p.p.a_size].reshape(p.p.r,p.p.r)@p.p.basis['A']['vh'],
           'B':p.p.nb*p.p.output_basis@theta[p.p.a_size:].reshape(p.p.b_rows,p.p.b_width)}
    torch.save(dict(theta=theta.cpu(),release_deltas={k:v.cpu() for k,v in delta.items()},cfg=cfg),out/'training_point.pt')
    torch.save(dict(theta=theta.cpu(),initial_theta=p.initial.cpu(),source_teacher=p.teacher.cpu(),
        environments=[dict(name=e['name'],data={k:v.cpu() for k,v in e['data'].items()}) for e in p.environments],
        snapshots=snaps,final_source=state['source'].cpu(),final_environment_scores=[x.cpu() for x in state['environment_scores']],cfg=cfg),out/'diagnostic.pt')
    result=dict(status='robust_shared_publication_source_candidate' if signal else 'closed_robust_shared_publication_no_finite_signal',target=cfg['target'],
        method='finite_joint_O_B_publication_against_solved_common_prefix_opponents',initial=initial,final=m,
        source_calibration_passed=True,confirmation_eligible=bool(signal),confirmation_passed=False,selected_step=accepted,accepted_updates=accepted,
        trials=len(history),stop_reason=reason,explicit_reverse_gradient_calls=gradient_calls,analytic_task_jacobian_calls=len(history),
        initial_point=cfg['initial_point'],initial_point_sha256=cfg['inputs'][cfg['initial_point']],initialization='preserved source-only O/B compensation point; no peer/test selection in this new optimization',
        opponent_count=2,opponents_frozen_before_optimization=True,continuous_environment_minimax_solved=False,
        training_point_sha256=digest(out/'training_point.pt'),diagnostic_sha256=digest(out/'diagnostic.pt'),
        source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        new_release_weights=0,original_expert_updates=0,peer_used=False,test_used=False,confirmation_used=False,
        limitation='Exact entropy-regularized min over TWO source-derived realizable environments and classes, not the continuous prefix family or unknown peers. Original total0.5 budgets unchanged. Source-specific gate does not become a universal proxy requirement; true TA/AGA remains essential.')
    save_json(out/'results.json',result);save_json(out/'history.json',history);save_json(out/'training_indices.json',p.indices);save_json(out/'protocol.json',cfg)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(cfg)
