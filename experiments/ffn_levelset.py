"""Target-only MLP input/output level-set feasibility diagnosis; no new release."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import Problem, angular_scores, signed_residual
from boundary_game import risk_first
from classification_completion import readout
from clip_benchmark import digest, merge
from cross_module_distance import extended_parameters, normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models, gelu


class FFNProblem(Problem):
    """Only publication coordinates change; expert and classifier are frozen."""
    def __init__(self, target, cfg):
        self.target, self.cfg = target, cfg
        self.radius, self.r = cfg['parameter_budget'], cfg['probe_rank']
        self.a_size = self.r**2
        self.anchor, self.expert, self.head, self.hashes = checked_models(target)
        self.p = on_device(extended_parameters(self.expert, self.head))
        proxy = merge(self.anchor, self.expert, self.anchor, cfg['proxy_beta'])
        self.pp = on_device(extended_parameters(proxy, self.head)); del proxy
        root = Path(cfg['boundary_reference'])
        assert digest(root/'diagnostic.pt') == cfg['diagnostic_sha256']
        cached = torch.load(root/'diagnostic.pt', map_location='cpu')
        fit, ids, n = risk_first(cached['screen'], cached['split']['indices'], cfg['contexts_per_class'], self.p)
        scale_root = Path(cfg['old_calibration_reference'])
        assert digest(scale_root/'calibration.pt') == cfg['old_calibration_sha256']
        scale_cal = torch.load(scale_root/'calibration.pt', map_location='cpu')
        scale, sid, ns = risk_first(scale_cal['scale'], scale_cal['indices']['scale'], cfg['contexts_per_class'], self.p)
        assert n == ns
        self.indices = {'fit':ids, 'scale':sid, 'proxy':cached['split']['indices'][:cfg['proxy_n']]}
        self.cache = {'native':fit, 'scale':scale, 'proxy':cached['proxy']}
        anchor_p = on_device(extended_parameters(self.anchor, self.head))
        u, s, vh = torch.linalg.svd(self.p['A']-anchor_p['A'], full_matrices=False)
        small = on_device({k:v[:cfg['selection_n']] for k,v in fit.items()})
        jh, _, z, teacher = raw_jacobians(small, self.p)
        x = normalized_input(small['r'], self.p)
        pre = x@self.p['A'].T+self.p['a']; sig = torch.sigmoid(1.702*pre)
        slope = sig+1.702*pre*sig*(1-sig)
        du = ((jh@self.p['B'])*slope[:,None,:])@u
        response = du*(x@vh.T)[:,None,:]
        # Rank singular modes by functional response, never singular value alone.
        ac = angular_scores(teacher); norm = (teacher-teacher.mean(1,keepdim=True)).norm(dim=1,keepdim=True)
        response = (response-response.mean(1,keepdim=True)-ac[:,:,None]*(ac[:,:,None]*response).sum(1,keepdim=True))/norm[:,:,None]
        y = teacher.argmax(1); count = torch.bincount(y,minlength=teacher.shape[1]); weight = 1./count[y].double()/int((count>0).sum())
        importance = (response.square().sum(1)*weight[:,None]).sum(0)
        chosen = importance.argsort(descending=True,stable=True)[:self.r]
        self.ua, self.va = u[:,chosen], vh[chosen]
        self.basis = {'A':{'u':self.ua,'vh':self.va,'selected':chosen,'s':s[chosen]}, 'B':on_device(scale_cal['basis_B'])}
        self.space, self.geometry = readout(self.p,cfg['readout_relative_tolerance'])
        self.q = self.space['Q']; self.b_rows, self.b_width = self.q.shape[1], self.p['B'].shape[1]
        self.na, self.nb = self.p['A'].norm(), self.p['B'].norm()
        self.native = [self.prepare(self.cache[g],self.p,True) for g in ['native','scale']]
        self.proxy = self.prepare(self.cache['proxy'],self.pp,False)
        t = self.proxy['teacher']; self.winner=t.argmax(1); self.rows=torch.arange(len(t),device='cuda')
        count=torch.bincount(self.winner,minlength=t.shape[1]); self.weights=1./count[self.winner].double()/int((count>0).sum())
        self.unit=self.gap(angular_scores(t)).median(); assert self.unit>0
        self.selection={'selected_A_modes':chosen.cpu().tolist(),'diagonal_response_fraction':float(importance[chosen].sum()/importance.sum()),
            'rule':'Class-balanced unit-Frobenius centered normalized score response on target-training contexts; fixed rank32.',
            'task_singular_values':s.cpu().tolist()}

    @torch.no_grad()
    def prepare(self, cache, p, probes):
        c=on_device(cache); x=normalized_input(c['r'],p); z=gelu(x@p['A'].T+p['a'])
        base=c['h']-z@p['B'].T
        xx,bb,oo=[x],[base],[torch.zeros_like(z)]
        if probes:
            n=self.cfg['contexts_per_class']*p['head'].shape[0]
            shifts=[]
            for val,v in [(x,self.va),(z,self.basis['B']['vh'])]:
                rms=(val@v.T).square().mean(0).sqrt()
                shifts.append((self.cfg['finite_step']*torch.tensor([-1.,1.],device='cuda')[:,None,None]*rms[None,:,None]*v[None,:,:]).flatten(0,1))
            sa,sb=shifts; count=len(sa)
            xx += [(x[:n,None,:]+sa[None,:,:]).flatten(0,1),x[:n].repeat_interleave(count,dim=0)]
            bb += [base[:n].repeat_interleave(count,dim=0)]*2
            oo += [torch.zeros((n*count,z.shape[1]),device='cuda',dtype=torch.float64),sb.repeat(n,1)]
        packed={'x':torch.cat(xx),'base':torch.cat(bb),'offset':torch.cat(oo)}
        packed['task_input']=packed['x']@self.va.T
        zero=torch.zeros(self.a_size+self.b_rows*self.b_width,device='cuda',dtype=torch.float64)
        packed['teacher']=self.predict(packed,p,zero)
        return packed

    def predict(self, packed, p, theta, factor=1.):
        ca=theta[:self.a_size].reshape(self.r,self.r); kb=theta[self.a_size:].reshape(self.b_rows,self.b_width)
        pre=packed['x']@p['A'].T+p['a']+factor*self.na*(packed['task_input']@ca.T)@self.ua.T
        z=gelu(pre)+packed['offset']
        h=packed['base']+z@p['B'].T+factor*self.nb*(z@kb.T)@self.q.T
        return scores(h,p)

    def stress(self,s):
        return -(self.weights*self.gap(angular_scores(s)).relu()).sum()/self.unit

    def local_matrix(self, packed, theta):
        ca=theta[:self.a_size].reshape(self.r,self.r); kb=theta[self.a_size:].reshape(self.b_rows,self.b_width)
        b=self.p['B']+self.nb*self.q@kb
        pre=packed['x']@self.p['A'].T+self.p['a']+self.na*(packed['task_input']@ca.T)@self.ua.T
        sig=torch.sigmoid(1.702*pre); slope=sig+1.702*pre*sig*(1-sig); z=gelu(pre)+packed['offset']
        h=packed['base']+z@b.T
        jh,_,_,after=raw_jacobians({'h':h,'r':torch.zeros_like(h)},self.p|{'B':b})
        first=self.na*((((jh@b)*slope[:,None,:])@self.ua)[:,:,:,None]*packed['task_input'][:,None,None,:])
        second=self.nb*((jh@self.q)[:,:,:,None]*z[:,None,None,:])
        jac=torch.cat([first.flatten(2),second.flatten(2)],2)
        teacher=packed['teacher']; y=teacher.argmax(1); rows=torch.arange(len(y),device='cuda')
        keep=torch.ones_like(teacher,dtype=torch.bool);keep[rows,y]=False
        gap=(teacher[rows,y,None]-teacher).masked_fill(~keep,1.)
        assert float(gap[keep].min())>1e-8
        return ((jac[rows,y,None,:]-jac)/gap[:,:,None])[keep],signed_residual(teacher,after)[keep]

    def critic(self, theta):
        residual=self.residuals(theta); offset=int(residual.abs().argmax())
        witness=None
        for g in self.native:
            if offset<g['teacher'].numel():
                row=offset//g['teacher'].shape[1]; witness={k:v[row:row+1] for k,v in g.items()};break
            offset-=g['teacher'].numel()
        n=self.cfg['contexts_per_class']*self.p['head'].shape[0]
        core={k:torch.cat([g[k][:n] for g in self.native]) for k in self.native[0]}
        packed={k:torch.cat([core[k],witness[k]]) for k in core}
        j,r=self.local_matrix(packed,theta); u,s,vh=torch.linalg.svd(j,full_matrices=False)
        rank=int((s>torch.finfo(torch.float64).eps*max(j.shape)*s[0]).sum())
        return u[:,:rank],s[:rank],vh[:rank],r


def diagnose(target,cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    p=FFNProblem(target,cfg);zero=torch.zeros(p.a_size+p.b_rows*p.b_width,device='cuda',dtype=torch.float64)
    leaf=zero.clone().requires_grad_(True); proxy_obj=lambda t:p.stress(p.predict(p.proxy,p.pp,t,cfg['proxy_beta']))
    gradient=torch.autograd.grad(proxy_obj(leaf),leaf)[0]
    u,s,vh,r=p.critic(zero);direction=gradient-vh.T@(vh@gradient);retained=float(direction.norm()/gradient.norm());direction/=direction.norm()
    steps=[{'step':cfg['parameter_budget']*.5**k,**p.metrics(cfg['parameter_budget']*.5**k*direction)} for k in range(cfg['backtracking_steps'])]
    feasible=next((a for a in steps if a['native_critic']<=cfg['margin_fraction']),None)
    history=[];theta=None;jacobian_calls=1
    if feasible:
        theta=min(2*feasible['step'],cfg['parameter_budget'])*direction
        for _ in range(cfg['correction_steps']):
            before=p.metrics(theta)
            if before['native_critic']<=cfg['margin_fraction']:break
            u,s,vh,r=p.critic(theta);jacobian_calls+=1;q=-vh.T@((u.T@r)/s)
            accepted=None
            for k in range(cfg['backtracking_steps']):
                trial=theta+(.5**k)*q
                if float(trial.norm())>cfg['parameter_budget']:continue
                metrics=p.metrics(trial)
                if metrics['native_critic']<before['native_critic']-1e-12:accepted=(trial,metrics);break
            history.append({'before':before,'correction_norm':float(q.norm()),'accepted':accepted is not None,'after':accepted[1] if accepted else None})
            if accepted is None:break
            theta=accepted[0]
    final=p.metrics(theta) if theta is not None else None
    # Structural contribution diagnosis only, not a recovery or candidate gate.
    anchor_p=on_device(extended_parameters(p.anchor,p.head));natural=p.native[0]
    x=natural['x'][:len(p.cache['native']['h'])];base=natural['base'][:len(x)];teacher=natural['teacher'][:len(x)]
    replace_a=scores(base+gelu(x@anchor_p['A'].T+p.p['a'])@p.p['B'].T,p.p)
    n=p.cfg['contexts_per_class']*p.p['head'].shape[0]; core={k:v[:n] for k,v in natural.items()}
    matrix,_=p.local_matrix(core,zero)
    fdstep=cfg['derivative_step'];fd=(signed_residual(core['teacher'],p.predict(core,p.p,fdstep*direction))-signed_residual(core['teacher'],p.predict(core,p.p,-fdstep*direction)))/(2*fdstep)
    keep=torch.ones_like(core['teacher'],dtype=torch.bool);keep[torch.arange(n,device='cuda'),core['teacher'].argmax(1)]=False
    jerror=float((fd[keep]-matrix@direction).abs().max());assert jerror<1e-5
    report={'status':'target_only_FFN_input_output_geometry_diagnosis','target':target,'mode_selection':p.selection,
        'local_task_rank':len(s),'coefficient_dimension':len(zero),'proxy_gradient_retained_fraction':retained,
        'native_null_response':float((matrix@direction).abs().max()),'analytic_JVP_error':jerror,
        'finite_steps':steps,'first_feasible':feasible,'correction_history':history,'final':final,
        'input_task_update_removal':{'n':len(x),'changed_teacher_predictions':int((replace_a.argmax(1)!=teacher.argmax(1)).sum()),
            'worst_margin_drift':float(signed_residual(teacher,replace_a).abs().max()),'role':'training-only structural contribution; not a defense success or robustness criterion'},
        'test_used':False,'partner_used':False,'publication_updates':0,'diagnostic_correction_updates':len(history),
        'reverse_gradient_calls':1,'analytic_jacobian_calls':jacobian_calls,'new_full_model_forwards':0,'new_release_weights':0,
        'checkpoint_sha256':p.hashes,'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path'])}
    raw={'theta':theta.cpu() if theta is not None else None,'direction':direction.cpu(),'gradient':gradient.cpu(),
        'matrix':matrix.cpu(),'core':{k:v.cpu() for k,v in core.items()},'native':p.cache['native'],
        'parameters':{k:v.cpu() if torch.is_tensor(v) else v for k,v in p.p.items()},
        'basis':{site:{k:v.cpu() if torch.is_tensor(v) else v for k,v in values.items()} for site,values in p.basis.items()},
        'output_basis':p.q.cpu(),'indices':p.indices,'cfg':cfg}
    torch.save(raw,out/'diagnostic.pt'); report['diagnostic_sha256']=digest(out/'diagnostic.pt')
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',p.indices)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    diagnose(args.target,cfg)
