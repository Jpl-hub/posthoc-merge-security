"""Finite joint publication with exact boundary Hermite elimination and reserve."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from attention_cascade import suffix
from boundary_game import risk_first
from boundary_hermite import HermiteChart,load_chart,path_pack
from clip_benchmark import digest
from cross_module_distance import CrossProblem,final_hidden
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import initialize


def mlp_jet(r,dr,p):
    c=r-r.mean(1,keepdim=True);dc=dr-dr.mean(1,keepdim=True);sd=(c.square().mean(1,keepdim=True)+p['eps2']).sqrt()
    x=c/sd*p['gamma2']+p['beta2'];dx=(dc-c*(c*dc).mean(1,keepdim=True)/sd.square())/sd*p['gamma2']
    a=x@p['A'].T+p['a'];da=dx@p['A'].T;sig=torch.sigmoid(1.702*a)
    return a*sig,(sig+1.702*a*sig*(1-sig))*da


class AnalyticHermite(HermiteChart):
    def factors(self,c,derivatives=True,create_graph=False):
        a=path_pack(self.endpoints,self.roots,self.p);d={k:v[:,1]-v[:,0] for k,v in self.endpoints.items()}
        do=self.delta_O(c);r=a['r']+a['o_input']@do.T;dr=d['r']+d['o_input']@do.T
        z,dz=mlp_jet(r,dr,self.p);z0,dz0=mlp_jet(a['r'],d['r'],self.p)
        h0=a['rounding']+r+z@self.p['B'].T+self.p['b'];dh0=d['rounding']+dr+dz@self.p['B'].T
        th=a['teacher_h'];dth=d['rounding']+d['r']+dz0@self.p['B'].T
        f=torch.cat([z,dz]);target=torch.cat([th-h0,dth-dh0])/self.nb
        scale=f.norm(dim=1,keepdim=True);f=f/scale;target=target/scale;fa,ta=f,target
        take=self.cfg['canonical_path_rows'];take=take+[len(self.roots)+i for i in take];f,target=f[take],target[take]
        gram=f@f.T;part=target.T@torch.linalg.solve(gram,f)
        return part,f,gram,target,fa,ta


class Publication:
    def __init__(self,cfg):
        self.cfg=cfg
        for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
        valid=json.loads((Path(cfg['mechanism_reference'])/'independent_validation.json').read_text());assert valid['status']=='passed_mechanism'
        saved,base=load_chart(cfg);self.saved=saved;self.p=base.p
        self.chart=AnalyticHermite(base.p,base.basis,base.endpoints,base.roots,base.space,cfg);self.q=base.space['Q'];self.size=cfg['probe_rank']**2
        self.dimension=self.size+self.q.shape[1]*self.p['B'].shape[1]
        self.zero=torch.zeros(self.dimension,device='cuda',dtype=torch.float64)
        self.ids=saved['indices'];self.core_n=cfg['contexts_per_class']*len(self.p['head'])
        raw,ids,n=risk_first(saved['cache'],self.ids,cfg['contexts_per_class'],self.p);assert n==self.core_n
        cp=CrossProblem.__new__(CrossProblem);cp.cfg=dict(probe_contexts=n,finite_step=cfg['finite_step'],full_B_completion=True)
        cp.r=cfg['probe_rank'];cp.a_size=self.size;cp.b_rows=self.p['B'].shape[0];cp.b_width=self.p['B'].shape[1]
        cp.basis={'A':base.basis,'B':on_device(saved['bases']['B'])};cp.na=self.chart.na;cp.nb=self.chart.nb;cp.output_basis=None
        natural=cp.prepare(raw,self.p,True);self.natural_n=len(ids);self.ordered_ids=ids
        tube=[]
        for exponent in cfg['offset_exponents']:
            for sign in [-1.,1.]:
                t=base.roots+sign*(2.**-exponent)*torch.minimum(base.roots,1-base.roots)
                a=path_pack(base.endpoints,t,self.p);tube.append(dict(r=a['r'],o_input=a['o_input'],rounding=a['rounding'],
                    b_offset=torch.zeros(len(t),self.p['B'].shape[1],device='cuda',dtype=torch.float64)))
        self.data={k:torch.cat([natural[k]]+[a[k] for a in tube]) for k in ['r','o_input','rounding','b_offset']}
        self.teacher=self.forward(self.data,*self.decode(self.zero)[:2]);self.y=self.teacher.argmax(1)
        self.active=set(range(n));self.rejected=set()
        # The two environments are recomputed from source-only frozen prefix
        # parameters, never loaded from failed publication responses.
        cal=torch.load(cfg['environment_cache'],map_location='cpu');pp=on_device(cal['proxy_parameters']);self.pp=pp
        for key in ['O','B']:assert torch.equal(cal['parameters'][key],self.p[key].cpu())
        env=torch.load(Path(cfg['environment_reference'])/'diagnostic.pt',map_location='cpu')['theta'].cuda().reshape(cfg['probe_rank'],cfg['probe_rank'])
        basis=on_device(cal['basis']);raw=on_device(cal['proxy']);self.environments=[]
        for name,s in [('zero',torch.zeros_like(env)),('learned_shared',env)]:
            h=raw['H'].double()+cal['parameters']['D'].norm().cuda()*(raw['t'].double()@s.T)@basis['u'].T
            out,context=suffix(h,pp);o=context['average'].flatten(1)
            a=dict(r=context['r'],o_input=o,rounding=raw['rounding'],b_offset=torch.zeros(len(h),pp['B'].shape[1],device='cuda',dtype=torch.float64))
            teacher=scores(out+raw['rounding'],pp);y=teacher.argmax(1);ar=torch.arange(len(y),device='cuda');other=teacher.clone();other[ar,y]=-torch.inf
            self.environments.append(dict(name=name,data=a,teacher=teacher,y=y,gap=teacher[ar,y]-other.max(1).values))

    def decode(self,theta):
        c=theta[:self.size].reshape(self.cfg['probe_rank'],self.cfg['probe_rank']);l=theta[self.size:].reshape(self.q.shape[1],self.p['B'].shape[1])
        k,info=self.chart.complete(c,self.q@l,True,True)
        return c,k,info

    def forward(self,data,c,k,p=None,factor=1.):
        p=self.p if p is None else p;r=data['r']+factor*data['o_input']@self.chart.delta_O(c).T
        z=mlp_jet(r,torch.zeros_like(r),p)[0]+data['b_offset']
        return scores(data['rounding']+r+z@p['B'].T+p['b']+factor*self.chart.nb*z@k.T,p)

    def objective(self,theta,detail=False):
        c,k,info=self.decode(theta);values=[];predictions=[]
        for e in self.environments:
            ss=self.forward(e['data'],c,k,self.pp,.5);ar=torch.arange(len(e['y']),device='cuda');other=ss.clone();other[ar,e['y']]=-torch.inf
            margin=(ss[ar,e['y']]-other.max(1).values)/e['gap'];damage=torch.sigmoid(-margin/self.cfg['margin_temperature'])
            values.extend(damage[e['y']==i].mean() for i in range(len(self.p['head'])));predictions.append(ss)
        a=torch.stack(values);tau=self.cfg['opponent_temperature'];distance=c.square().sum()+k.square().sum()
        value=-tau*(torch.logsumexp(-a/tau,0)-math.log(len(a)))+self.cfg['distance_weight']*distance
        return (value,a,predictions,c,k,info) if detail else value

    @torch.no_grad()
    def evaluate(self,theta):
        obj,a,env,c,k,info=self.objective(theta,True);ss=self.forward(self.data,c,k);err=signed_residual(self.teacher,ss);records=[]
        for e,s in zip(self.environments,env):
            change=s.argmax(1)!=e['y'];records.append(dict(name=e['name'],n=len(change),changed=int(change.sum()),
                per_class=[dict(label=i,n=int((e['y']==i).sum()),changed=int((change&(e['y']==i)).sum())) for i in range(len(self.p['head']))]))
        metric=dict(objective=float(obj),parameter_distance=float((c.square().sum()+k.square().sum()).sqrt()),worst_task_drift=float(err.abs().max()),
            natural_changes=int((ss[:self.natural_n].argmax(1)!=self.y[:self.natural_n]).sum()),
            hermite_equality_error=float((info['F_all']@k.T-info['target_all']).abs().max()),constraint_condition=float(torch.linalg.cond(info['F'])),environments=records)
        return dict(metrics=metric,scores=ss,residual=err,environment_scores=env,C=c,K=k)

    def task_jacobian(self,theta,state):
        risk=state['residual'].abs().max(1).values;selected=set(range(self.core_n))|self.rejected
        for cls in range(len(self.p['head'])):
            members=(self.y==cls).nonzero().flatten();selected.add(int(members[risk[members].argmax()]))
        assert len(selected)<=self.cfg['maximum_active_inputs'];selected=torch.tensor(sorted(selected),device='cuda')
        data={k:v[selected] for k,v in self.data.items()};teacher=self.teacher[selected];leaf=theta.detach().clone().requires_grad_(True)
        c,k,_=self.decode(leaf);value=signed_residual(teacher,self.forward(data,c,k));mask=torch.ones_like(teacher,dtype=torch.bool);mask[torch.arange(len(selected),device='cuda'),teacher.argmax(1)]=False;value=value[mask]
        rows=[];batch=self.cfg['jacobian_batch']
        for start in range(0,len(value),batch):
            take=torch.arange(start,min(start+batch,len(value)),device='cuda');seeds=torch.nn.functional.one_hot(take,len(value)).double()
            rows.append(torch.autograd.grad(value,leaf,grad_outputs=seeds,is_grads_batched=True,retain_graph=True)[0].detach())
        jac=torch.cat(rows);return jac,value.detach(),selected,len(rows),len(value)


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats();p=Publication(cfg)
    theta=p.zero.clone();state=p.evaluate(theta);initial=state['metrics'];radius=cfg['trust_initial'];history=[];snapshots=[];accepted=0;reason='fixed_trial_cap';objective_calls=task_reverse=task_duals=0
    for trial in range(cfg['maximum_trials']):
        with torch.enable_grad():
            leaf=theta.detach().clone().requires_grad_(True);value=p.objective(leaf);gradient=torch.autograd.grad(value,leaf)[0].detach();objective_calls+=1
            jac,residual,rows,batches,duals=p.task_jacobian(theta,state);task_reverse+=batches;task_duals+=duals
        gram=jac@jac.T;d,v=torch.linalg.eigh(gram);tol=torch.finfo(d.dtype).eps*max(jac.shape)*d.abs().max();good=d>tol
        def solve(z):return v@((v.T@z)*torch.where(good,1/d.clamp_min(tol),0.))
        tangent=gradient-jac.T@solve(jac@gradient)
        if tangent.norm()<1e-14:reason='no_source_feasible_tangent';break
        step=radius*tangent/tangent.norm();predicted=float(gradient@step)
        if predicted<=1e-14:reason='no_positive_gain';break
        with torch.no_grad():
            proposal=p.evaluate(theta+step);eps=cfg['derivative_step'];fd=float((p.objective(theta+eps*step)-p.objective(theta-eps*step))/(2*eps));fd_error=abs(fd-predicted)
            assert fd_error<cfg['derivative_tolerance']
            actual=proposal['metrics']['objective']-state['metrics']['objective'];ratio=actual/predicted;linear=float((residual+jac@step).abs().max());m=proposal['metrics']
            ok=bool(m['parameter_distance']<=cfg['parameter_budget'] and m['worst_task_drift']<=cfg['construction_task_reserve'] and m['natural_changes']==0
                and m['hermite_equality_error']<cfg['linear_equality_tolerance'] and m['constraint_condition']<cfg['maximum_condition']
                and linear<=cfg['construction_task_reserve'] and actual>1e-12 and ratio>=cfg['acceptance_ratio'])
            row=dict(trial=trial+1,accepted=ok,proposal=m,before=state['metrics'],predicted_gain=predicted,actual_gain=actual,agreement_ratio=ratio,
                trust_radius=radius,active_linear_worst=linear,active_rows=rows.cpu().tolist(),active_rank=int(good.sum()),objective_fd_error=fd_error)
            history.append(row);snapshots.append(dict(theta=theta.cpu(),step=step.cpu(),gradient=gradient.cpu(),C=proposal['C'].cpu(),K=proposal['K'].cpu(),
                source_scores=proposal['scores'].cpu(),environment_scores=[s.cpu() for s in proposal['environment_scores']],active_residual=residual.cpu(),active_Jstep=(jac@step).cpu()))
            print(json.dumps(row),flush=True)
            if ok:theta=theta+step;state=proposal;accepted+=1;radius=min(cfg['trust_maximum'],2*radius) if ratio>=cfg['expansion_ratio'] else radius
            else:radius*=.5;p.rejected.add(int(proposal['residual'].abs().max(1).values.argmax()))
            if radius<cfg['trust_minimum']:reason='fixed_minimum_radius';break
            if time.monotonic()-start>=cfg['maximum_wall_seconds']:reason='frozen_wall_cost_cap';break
    with torch.no_grad():
        final=state['metrics'];eligible=bool(accepted>0 and final['parameter_distance']>1e-6 and final['objective']>initial['objective'])
        deltas=dict(O=p.chart.delta_O(state['C']).cpu(),B=(p.chart.nb*state['K']).cpu())
        torch.save(dict(theta=theta.cpu(),release_deltas=deltas,cfg=cfg),out/'training_point.pt')
        torch.save(dict(snapshots=snapshots,theta=theta.cpu(),teacher=p.teacher.cpu(),data={k:v.cpu() for k,v in p.data.items()},
            environments=[dict(name=e['name'],data={k:v.cpu() for k,v in e['data'].items()},teacher=e['teacher'].cpu()) for e in p.environments],
            final_scores=state['scores'].cpu(),final_C=state['C'].cpu(),final_K=state['K'].cpu(),final_environment_scores=[s.cpu() for s in state['environment_scores']],cfg=cfg),out/'diagnostic.pt')
    result=dict(status='source_Hermite_publication_point' if eligible else 'closed_no_source_Hermite_point',method='source_boundary_hidden_Hermite_elimination_and_finite_shared_environment_game',
        target=cfg['target'],initial=initial,final=final,trials=len(history),accepted_updates=accepted,selected_step=accepted,stop_reason=reason,
        source_natural_n=p.natural_n,source_task_inputs=len(p.teacher),construction_task_reserve=cfg['construction_task_reserve'],parameter_budget=.5,task_budget=.5,
        confirmation_eligible=eligible,confirmation_passed=False,new_release_weights=0,true_TA_run=False,author_AGA_run=False,peer_used=False,test_used=False,old_failed_point_used=False,
        explicit_reverse_gradient_calls=objective_calls+task_reverse,publication_objective_reverse_calls=objective_calls,
        task_batched_reverse_calls=task_reverse,task_dual_directions=task_duals,autograd_task_jacobian_calls=objective_calls,original_expert_updates=0,source_sha256=digest(__file__),
        training_point_sha256=digest(out/'training_point.pt'),diagnostic_sha256=digest(out/'diagnostic.pt'),wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitation='Source-only Hermite boundary paths and two realizable environments; no transverse input or unknown-peer coverage. Reserved finite source budget is not a generalization guarantee. Sole final point requires new independent confirmation before trueTA/AGA.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);save_json(out/'history.json',history);save_json(out/'training_indices.json',{'fit':p.ids});print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();cfg=json.loads(Path(a.protocol).read_text());initialize()
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    run(cfg)
