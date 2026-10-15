"""A common, finite, task-SVD prefix environment adversary; no publication.

One parameter S is shared across all images/tokens and both source/reference
task checks. Exact finite nonlinear acceptance is mandatory at every step.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from attention_cascade import input_vjp, select, suffix
from cascade_game import Cascade
from clip_benchmark import digest
from joint_compensation_trust import positive_root
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from verify_covariance_noise import official_labels


def prefix_forward(packed, pars, s, basis, nd, jacobian=False):
    cache = packed['cache']; take = packed['take']
    h = cache['H'][take].double(); t = cache['t'][take].double()
    marker = torch.zeros(h.shape[1], 1, device='cuda', dtype=torch.float64); marker[0] = 1.
    h = h+marker[None]*(packed['shift']@packed['up_map'])[:, None]
    t = t+marker[None]*packed['shift'][:, None]
    h = h+nd*(t@s.T)@basis['u'].T
    out, context = suffix(h, pars)
    final = out+cache['rounding'][take]+packed['offset']@pars['B'].T
    result = scores(final, pars)
    if not jacobian: return result
    _, jr, _, check = raw_jacobians({'h':final, 'r':context['r']}, pars)
    assert float((result-check).abs().max()) < 1e-8
    back = input_vjp(jr, context, pars)
    j = nd*torch.einsum('nctu,ntv->ncuv', back@basis['u'], t)
    return result, j.flatten(2)


class SharedGame:
    def __init__(self, cfg):
        self.cfg = cfg
        for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
        self.p = Cascade(cfg); self.rank = cfg['probe_rank']; self.dim = self.rank**2
        self.zero = torch.zeros(self.dim, device='cuda', dtype=torch.float64)
        point = torch.load(cfg['protected_point'], map_location='cpu')
        size = len(point['basis_O']['vh'])**2; theta = point['theta'].cuda().double()
        u, v, q = [x.cuda().double() for x in [point['basis_O']['u'], point['basis_O']['vh'], point['output_basis']]]
        self.delta = dict(O=self.p.p['O'].norm()*u@theta[:size].reshape(len(v),len(v))@v,
                          B=self.p.p['B'].norm()*q@theta[size:].reshape(len(q.T),self.p.p['B'].shape[1]))
        self.modified = self.p.pp | {k:self.p.pp[k]+cfg['coefficient']*val for k,val in self.delta.items()}
        self.nt = self.p.native['teacher']; self.bt = self.p.proxy['teacher']
        self.y = self.bt.argmax(1); self.rows = torch.arange(len(self.y),device='cuda')
        other = self.bt.clone(); other[self.rows,self.y] = -torch.inf
        self.gap = self.bt[self.rows,self.y]-other.max(1).values
        self.classes = self.bt.shape[1]
        self.native_core = list(self.p.core)
        # Public-reference core chosen before observing protected responses.
        self.proxy_core = []
        for cls in range(self.classes):
            members = (self.y == cls).nonzero().flatten()
            self.proxy_core.extend(members[self.gap[members].argsort(stable=True)[:cfg['proxy_contexts_per_class']]].cpu().tolist())
        self.labels = torch.tensor(official_labels(cfg['target'],True)[self.p.indices['proxy']], device='cuda')
        initial = self.predict(self.p.proxy,self.modified,self.zero)
        self.initial_protected = initial; self.initial_changed = initial.argmax(1) != self.y

    def predict(self, packed, pars, theta, jacobian=False):
        output, matrices = [], []
        for lo in range(0,len(packed['take']),self.cfg['batch_size']):
            batch = select(packed,slice(lo,lo+self.cfg['batch_size']))
            result = prefix_forward(batch,pars,theta.reshape(self.rank,self.rank),self.p.basis,self.p.nd,jacobian)
            if jacobian: output.append(result[0]); matrices.append(result[1])
            else: output.append(result)
        ss = torch.cat(output)
        return (ss,torch.cat(matrices)) if jacobian else ss

    def objective(self, protected, gradient=False):
        if gradient: protected = protected.detach().requires_grad_(True)
        other = protected.clone(); other[self.rows,self.y] = -torch.inf
        margin = (protected[self.rows,self.y]-other.max(1).values)/self.gap
        # The class adversary emphasizes the highest finite restoration loss.
        losses = self.cfg['margin_temperature']*torch.nn.functional.softplus(-margin/self.cfg['margin_temperature'])
        per_class = torch.stack([losses[self.y == c].mean() for c in range(self.classes)])
        tau = self.cfg['class_temperature']
        value = -tau*(torch.logsumexp(per_class/tau,0)-math.log(self.classes))
        if gradient:
            grad = torch.autograd.grad(value,protected)[0]
            return value.detach(), per_class.detach(), grad
        return value,per_class

    @torch.no_grad()
    def evaluate(self, theta):
        n = self.predict(self.p.native,self.p.p,theta)
        b = self.predict(self.p.proxy,self.p.pp,theta)
        a = self.predict(self.p.proxy,self.modified,theta)
        nr, br = signed_residual(self.nt,n), signed_residual(self.bt,b)
        value, losses = self.objective(a)
        now = a.argmax(1); restored = self.initial_changed & (now == self.y)
        rec = dict(parameter_distance=float(theta.norm()), source_worst_task=float(nr.abs().max()),
                   reference_worst_task=float(br.abs().max()),
                   source_natural_changes=int((n[:self.p.natural_n].argmax(1)!=self.nt[:self.p.natural_n].argmax(1)).sum()),
                   reference_changes=int((b.argmax(1)!=self.y).sum()),
                   protected_changed=int((now!=self.y).sum()), prediction_restored=int(restored.sum()),
                   restored_classes=sum(bool((restored & (self.y==c)).any()) for c in range(self.classes)),
                   protected_original_correct=int((self.initial_protected.argmax(1)==self.labels).sum()),
                   protected_current_correct=int((now==self.labels).sum()), objective=float(value),
                   per_class_restoration_loss=losses.cpu().tolist(),
                   per_class=[dict(label=c,n=int((self.y==c).sum()),initial_changed=int((self.initial_changed&(self.y==c)).sum()),
                       now_changed=int(((now!=self.y)&(self.y==c)).sum()),restored=int((restored&(self.y==c)).sum())) for c in range(self.classes)])
        return dict(source=n,baseline=b,protected=a,source_r=nr,baseline_r=br,metrics=rec)

    @torch.no_grad()
    def constraints(self, theta, state):
        all_j, all_r, locations = [],[],[]
        for name,packed,pars,teacher,core,res in [
            ('source',self.p.native,self.p.p,self.nt,self.native_core,state['source_r']),
            ('reference',self.p.proxy,self.p.pp,self.bt,self.proxy_core,state['baseline_r'])]:
            yy=teacher.argmax(1);risk=res.abs().max(1).values
            rows=set(core)
            for cls in range(self.classes):
                members=(yy==cls).nonzero().flatten();rows.add(int(members[risk[members].argmax()]))
            rows=sorted(rows);locations.append(dict(group=name,rows=rows));selected=torch.tensor(rows,device='cuda')
            value,j=self.predict(select(packed,selected),pars,theta,True)
            before=teacher[selected]; y=before.argmax(1); ar=torch.arange(len(y),device='cuda')
            keep=torch.ones_like(before,dtype=torch.bool);keep[ar,y]=False
            gap=(before[ar,y,None]-before).masked_fill(~keep,1.)
            all_j.append(((j[ar,y,None]-j)/gap[:,:,None])[keep])
            all_r.append(signed_residual(before,value)[keep])
        j,r=torch.cat(all_j),torch.cat(all_r)
        u,s,vh=torch.linalg.svd(j,full_matrices=False)
        keep=s>torch.finfo(s.dtype).eps*max(j.shape)*s[0]
        return j,r,u[:,keep],s[keep],vh[keep],locations


@torch.no_grad()
def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.cuda.reset_peak_memory_stats();p=SharedGame(cfg)
    theta=p.zero.clone();state=p.evaluate(theta);initial=state['metrics'];history=[];snapshots=[];accepted=0
    radius=cfg['trust_initial'];reason='fixed_trial_cap';gradients=0;fd_max=0.;jacobians=0;autograd_error=None;direct_checks=0
    resumed=0
    if cfg.get('resume_diagnostic'):
        previous=torch.load(cfg['resume_diagnostic'],map_location='cpu')
        parent=Path(cfg['resume_diagnostic']).parent
        prior=json.loads((parent/'results.json').read_text())
        validation=json.loads((parent/'independent_validation.json').read_text());assert validation['status']=='passed'
        assert prior['diagnostic_sha256']==cfg['inputs'][cfg['resume_diagnostic']]
        for key in ['inputs','protected_point','probe_rank','contexts_per_class','finite_step','coefficient',
                    'parameter_budget','task_budget','margin_temperature','class_temperature','maximum_trials']:
            if key!='inputs':assert previous['cfg'][key]==cfg[key],key
        theta=previous['theta'].cuda();state=p.evaluate(theta);initial=prior['initial']
        assert state['metrics']==prior['final']
        history=json.loads((parent/'history.json').read_text());snapshots=previous['snapshots']
        accepted=prior['accepted_environment_updates'];gradients=prior['explicit_environment_gradient_calls']
        jacobians=prior['analytic_jacobian_calls'];fd_max=prior['maximum_objective_fd_error']
        autograd_error=prior['full_shared_gradient_autograd_relative_error'];direct_checks=prior['additional_direct_environment_autograd_checks']
        last=history[-1];radius=last['trust_radius']
        if last['accepted'] and last['agreement_ratio']>=cfg['expansion_ratio']:radius=min(cfg['trust_maximum'],radius*2.)
        elif not last['accepted']:radius*=.5
        resumed=len(history)
    for trial in range(resumed,cfg['maximum_trials']):
        j,r,u,s,vh,locations=p.constraints(theta,state);jacobians+=1
        protected,jp=p.predict(p.p.proxy,p.modified,theta,True);jacobians+=1
        with torch.enable_grad():value,losses,score_grad=p.objective(protected,True)
        g=torch.einsum('nc,ncd->d',score_grad,jp);gradients+=1
        if trial == resumed:
            with torch.enable_grad():
                leaf=theta.detach().clone().requires_grad_(True)
                check=p.objective(p.predict(p.p.proxy,p.modified,leaf))[0]
                direct=torch.autograd.grad(check,leaf)[0]
            check_error=float((g-direct).norm()/direct.norm().clamp_min(1e-30))
            autograd_error=max(autograd_error or 0.,check_error);direct_checks+=1
            assert check_error<cfg['gradient_tolerance'],check_error
        tangent=g-vh.T@(vh@g)
        # The task is an INEQUALITY, not r=0. At a feasible point its minimum
        # normal correction is zero. Forcing a feasible residual back to zero
        # can oppose the true objective and stall despite an available tangent.
        target=r-r.clamp(-cfg['task_budget'],cfg['task_budget']) if cfg.get('inequality_feasibility_normal',False) else r
        normal=-vh.T@((u.T@target)/s)
        legacy_normal=-vh.T@((u.T@r)/s)
        legacy_normal*=min(1.,cfg['normal_fraction']*radius/max(float(legacy_normal.norm()),1e-30))
        retained=float(tangent.norm()/g.norm().clamp_min(1e-30))
        if tangent.norm()<1e-15:reason='no_projected_shared_direction';break
        normal*=min(1.,cfg['normal_fraction']*radius/max(float(normal.norm()),1e-30))
        tangent/=tangent.norm();step_size=positive_root(1.,float(normal@tangent),float(normal@normal)-radius**2)
        step=normal+step_size*tangent
        alpha=min(1.,positive_root(float(step@step),float(theta@step),float(theta@theta)-(.5*(1-1e-12))**2))
        step*=alpha;pred=float(g@step)
        if pred<=1e-14:reason='no_predicted_finite_gain';break
        # Full reference finite directional check of the actual objective.
        eps=cfg['derivative_step']
        plus=p.objective(p.predict(p.p.proxy,p.modified,theta+eps*step))[0]
        minus=p.objective(p.predict(p.p.proxy,p.modified,theta-eps*step))[0]
        fd=float((plus-minus)/(2*eps));error=abs(fd-pred);fd_max=max(fd_max,error)
        assert error<cfg['derivative_tolerance'],error
        proposal=p.evaluate(theta+step);m=proposal['metrics'];actual=m['objective']-state['metrics']['objective'];ratio=actual/pred
        linear=float((r+j@step).abs().max())
        ok=(m['parameter_distance']<=cfg['parameter_budget'] and m['source_worst_task']<=cfg['task_budget']
            and m['reference_worst_task']<=cfg['task_budget'] and m['source_natural_changes']==0 and m['reference_changes']==0
            and linear<=cfg['task_budget'] and actual>1e-12 and ratio>=cfg['acceptance_ratio'])
        record=dict(trial=trial+1,accepted=bool(ok),before=state['metrics'],proposal=m,predicted_gain=pred,
                    actual_gain=actual,agreement_ratio=ratio,active_linear_worst=linear,trust_radius=radius,
                    shared_tangent_retained_fraction=retained,active_rank=len(s),active_task_rows=len(j),fd_error=error,locations=locations,
                    normal_target_maximum=float(target.abs().max()),legacy_zero_residual_normal_gradient_dot=float(g@legacy_normal))
        history.append(record);snapshots.append(dict(theta=theta.cpu(),step=step.cpu(),gradient=g.cpu(),
            tangent_retained_fraction=retained,source=proposal['source'].cpu(),baseline=proposal['baseline'].cpu(),
            protected=proposal['protected'].cpu(),active_residual=r.cpu(),active_Jstep=(j@step).cpu(),
            score_gradient=score_grad.cpu(),objective_fd=fd))
        print(json.dumps(record),flush=True)
        if ok:
            theta+=step;state=proposal;accepted+=1
            if ratio>=cfg['expansion_ratio']:radius=min(cfg['trust_maximum'],radius*2.)
        else:radius*=.5
        if radius<cfg['trust_minimum']:reason='finite_trust_radius_exhausted';break
    final=state['metrics']
    signal=(final['prediction_restored']>=cfg['minimum_restored'] and final['restored_classes']>=cfg['minimum_restored_classes'])
    artifact=out/'diagnostic.pt'
    torch.save(dict(theta=theta.cpu(),initial_protected=p.initial_protected.cpu(),source_teacher=p.nt.cpu(),
        reference_teacher=p.bt.cpu(),snapshots=snapshots,final_source=state['source'].cpu(),
        final_baseline=state['baseline'].cpu(),final_protected=state['protected'].cpu(),indices=p.p.indices,cfg=cfg),artifact)
    report=dict(status='finite_shared_environment_signal' if signal else 'closed_fixed_shared_environment_no_class_signal',
        target=cfg['target'],initial=initial,final=final,finite_multiclass_signal=bool(signal),
        stop_reason=reason,trials=len(history),accepted_environment_updates=accepted,
        source_natural_n=p.p.natural_n,source_all_task_inputs=len(p.nt),reference_n=len(p.bt),
        explicit_environment_gradient_calls=gradients,analytic_jacobian_calls=jacobians,
        additional_direct_environment_autograd_checks=direct_checks,
        full_shared_gradient_autograd_relative_error=autograd_error,
        resumed_existing_proposals=resumed,total_proposal_cap=cfg['maximum_trials'],
        inequality_feasibility_normal=cfg.get('inequality_feasibility_normal',False),
        publication_parameter_updates=0,original_expert_updates=0,new_release_weights=0,
        parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],
        diagnostic_sha256=digest(artifact),source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
        maximum_objective_fd_error=fd_max,wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        peer_used=False,test_used=False,confirmation_used=False,new_full_model_forwards=0,
        limitation='One finite common rank32 task-SVD B10 environment path. Applies to all images and both source/reference checks, but is not a future peer distribution. Source task constraints and public-reference constraints checked on ALL inputs. Local projections only propose; every accepted point is finite feasible. A failed finite path is not a proof that all realizable directions fail. No new publication or TA/AGA efficacy.')
    save_json(out/'results.json',report);save_json(out/'history.json',history);save_json(out/'protocol.json',cfg)
    save_json(out/'training_indices.json',p.p.indices);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(cfg)
