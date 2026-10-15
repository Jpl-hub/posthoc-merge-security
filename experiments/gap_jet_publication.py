"""Decision-normal boundary retraction; source-only finite publication game.

Unlike full-hidden interpolation, free B changes are applied BEFORE retracting
only one class gap and its path derivative. All old frozen modules are read-only.
"""

import argparse
import json
from pathlib import Path

import torch

from boundary_hermite import HermiteChart,path_pack
from clip_benchmark import digest
from hermite_publication import Publication,mlp_jet
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import initialize


class GapJetChart(HermiteChart):
    def __init__(self,*args,pairs,**kwargs):
        super().__init__(*args,**kwargs)
        self.pairs=pairs
        p=self.p
        linear=p['gamma'][:,None]*p['projection'];self.linear=linear-linear.mean(0)
        self.bias=p['beta']@p['projection']
        w=p['head'][pairs[:,0]]-p['head'][pairs[:,1]]
        self.normal=w@self.linear.T;self.d=w@self.bias
        self.b=self.normal.norm(dim=1);self.v=self.normal/self.b[:,None]
        self.lower=self.b-self.d.abs()/(p['B'].shape[0]**.5)
        assert self.lower.min()>self.cfg['minimum_monotone_slope']

    def gap_slope(self,h,dh):
        centered=h-h.mean(1,keepdim=True)
        sd=(centered.square().mean(1)+self.p['eps']).sqrt()
        t=h@self.linear+sd[:,None]*self.bias
        dt=dh@self.linear+((centered*dh).mean(1)/sd)[:,None]*self.bias
        w=self.p['head'][self.pairs[:,0]]-self.p['head'][self.pairs[:,1]]
        norm=t.norm(dim=1);num=(t*w).sum(1)
        return num/norm,(dt*w).sum(1)/norm-num*(t*dt).sum(1)/norm**3

    def complete(self,c,free=None,derivatives=True,create_graph=False):
        assert derivatives
        p=self.p;a=path_pack(self.endpoints,self.roots,p)
        d={k:v[:,1]-v[:,0] for k,v in self.endpoints.items()}
        do=self.delta_O(c);r=a['r']+a['o_input']@do.T;dr=d['r']+d['o_input']@do.T
        z,dz=mlp_jet(r,dr,p);z0,dz0=mlp_jet(a['r'],d['r'],p)
        h0=a['rounding']+r+z@p['B'].T+p['b'];dh0=d['rounding']+dr+dz@p['B'].T
        dth=d['rounding']+d['r']+dz0@p['B'].T
        free=torch.zeros_like(p['B']) if free is None else free
        hf=h0+self.nb*z@free.T;dhf=dh0+self.nb*dz@free.T
        hc=hf-hf.mean(1,keepdim=True);aa=(hf*self.normal).sum(1)
        q0=hc.square().mean(1)+p['eps'];q1=(hc*self.v).mean(1);q2=self.v.square().mean(1)
        # Unique root of aa+b*s+d*sqrt(q0+2*q1*s+q2*s*s)=0.
        # b>|d|/sqrt(width) proves global monotonicity, not a contraction.
        den=self.b.square()-self.d.square()*q2
        rad=self.b.square()*q0-2*aa*self.b*q1+aa.square()*q2+self.d.square()*(q1.square()-q2*q0)
        assert (den>0).all() and (rad>0).all()
        shift=(-aa*self.b+self.d.square()*q1-self.d*rad.sqrt())/den
        ht=hf+shift[:,None]*self.v
        centered=ht-ht.mean(1,keepdim=True);sd=(centered.square().mean(1)+p['eps']).sqrt()
        length=(ht@self.linear+sd[:,None]*self.bias).norm(dim=1)
        normal=self.normal+self.d[:,None]*centered/(ht.shape[1]*sd[:,None])
        slope_v=(normal*self.v).sum(1)/length
        teacher_gap,teacher_slope=self.gap_slope(a['teacher_h'],dth)
        multiplier=(teacher_slope-(normal*dhf).sum(1)/length)/slope_v
        dht=dhf+multiplier[:,None]*self.v
        f=torch.cat([z,dz]);desired=torch.cat([ht-h0,dht-dh0])/self.nb
        scale=f.norm(dim=1,keepdim=True);fa=f/scale;ta=desired/scale
        take=self.cfg['canonical_path_rows'];take=take+[len(self.roots)+i for i in take]
        f=fa[take];target=ta[take];gram=f@f.T
        correction=(target-f@free.T).T@torch.linalg.solve(gram,f)
        k=free+correction
        h=h0+self.nb*z@k.T;dh=dh0+self.nb*dz@k.T
        gap,slope=self.gap_slope(h,dh)
        info=dict(F=f,gram=gram,target=target,F_all=fa,target_all=ta,part=correction,
            gap_error=(gap-teacher_gap).abs().max(),slope_error=(slope-teacher_slope).abs().max(),
            hidden_change=(h-a['teacher_h']).norm()/a['teacher_h'].norm(),
            monotone_lower=self.lower.min(),directional_slope_min=slope_v.min(),
            correction_norm=correction.norm(),free_norm=free.norm())
        return k,info


class GapPublication(Publication):
    def __init__(self,cfg):
        super().__init__(cfg)
        base=self.chart
        self.chart=GapJetChart(base.p,base.basis,base.endpoints,base.roots,base.space,cfg,pairs=self.saved['pairs'].cuda())
        self.teacher=self.forward(self.data,*self.decode(self.zero)[:2]);self.y=self.teacher.argmax(1)

    @torch.no_grad()
    def evaluate(self,theta):
        state=super().evaluate(theta)
        _,_,info=self.decode(theta)
        for key in ['gap_error','slope_error','hidden_change','monotone_lower','directional_slope_min','correction_norm','free_norm']:
            state['metrics'][key]=float(info[key])
        return state


def run(cfg):
    # The frozen trust-region controller is copied, rather than changing the old
    # controller or silently substituting its class through module globals.
    import math,time
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats();p=GapPublication(cfg)
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
                and m['gap_error']<cfg['score_tolerance'] and m['slope_error']<cfg['slope_tolerance']
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
    result=dict(status='source_gap_jet_point' if eligible else 'closed_no_source_gap_jet_point',method='decision_normal_boundary_jet_retraction_and_finite_shared_environment_game',
        target=cfg['target'],initial=initial,final=final,trials=len(history),accepted_updates=accepted,selected_step=accepted,stop_reason=reason,
        source_natural_n=p.natural_n,source_task_inputs=len(p.teacher),construction_task_reserve=cfg['construction_task_reserve'],parameter_budget=.5,task_budget=.5,
        confirmation_eligible=eligible,confirmation_passed=False,new_release_weights=0,true_TA_run=False,author_AGA_run=False,peer_used=False,test_used=False,old_failed_point_used=False,
        explicit_reverse_gradient_calls=objective_calls+task_reverse,publication_objective_reverse_calls=objective_calls,
        task_batched_reverse_calls=task_reverse,task_dual_directions=task_duals,autograd_task_jacobian_calls=objective_calls,original_expert_updates=0,source_sha256=digest(__file__),
        training_point_sha256=digest(out/'training_point.pt'),diagnostic_sha256=digest(out/'diagnostic.pt'),wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitation='Only20 source activation-chord pair jets and finite task constraints. No transverse/unknown-peer coverage. Two fixed source-realizable environments, not all possible shared environments. No proxy count necessary gate; no tests/peers used. Not zero-gradient.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);save_json(out/'history.json',history);save_json(out/'training_indices.json',{'fit':p.ids});print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();cfg=json.loads(Path(a.protocol).read_text());initialize()
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    run(cfg)
