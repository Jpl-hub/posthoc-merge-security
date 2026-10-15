"""Target-only attention-mediated finite compensation and class-adversarial game."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores,signed_residual
from attention_cascade import capture,compact,forward,input_vjp,parameters,select,suffix
from boundary_game import risk_first
from classification_completion import readout
from clip_benchmark import digest,merge
from covariance_noise import dataset
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models


def cpu(p):return {k:v.cpu() if torch.is_tensor(v) else v for k,v in p.items()}


def prepare(target,cfg):
    root=Path(cfg['cache_output']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    anchor,expert,head,pins=checked_models(target);p=on_device(parameters(expert,head));pa=on_device(parameters(anchor,head))
    source=Path(cfg['boundary_reference'])/'diagnostic.pt';assert digest(source)==cfg['boundary_sha256']
    boundary=torch.load(source,map_location='cpu');ids=boundary['split']['indices']
    old=Path(cfg['old_calibration_reference'])/'calibration.pt';assert digest(old)==cfg['old_calibration_sha256']
    cal=torch.load(old,map_location='cpu');scale_ids=cal['indices']['scale']
    training=dataset(target,'data',expert.val_preprocess,True)
    u,s,vh=torch.linalg.svd(p['D']-pa['D'],full_matrices=False)
    small=capture(expert,training,ids[:cfg['selection_n']],vh)
    c=on_device(small);h,ctx=suffix(c['H'].double(),p)
    jh,jr,_,teacher=raw_jacobians({'h':c['h'].double(),'r':ctx['r']},p)
    j=input_vjp(jr,ctx,p)
    response=torch.einsum('nctu,ntu->ncu',j@u,c['t'])
    ac=angular_scores(teacher);scale=(teacher-teacher.mean(1,keepdim=True)).norm(dim=1,keepdim=True)
    response=(response-response.mean(1,keepdim=True)-ac[:,:,None]*(ac[:,:,None]*response).sum(1,keepdim=True))/scale[:,:,None]
    winner=teacher.argmax(1);counts=torch.bincount(winner,minlength=teacher.shape[1]);weights=1./counts[winner]/int((counts>0).sum())
    importance=(response.square().sum(1)*weights[:,None]).sum(0)
    chosen=importance.argsort(descending=True,stable=True)[:cfg['probe_rank']]
    basis=dict(u=u[:,chosen].cpu(),vh=vh[chosen].cpu(),s=s[chosen].cpu(),selected=chosen.cpu())
    space,_=readout(p);saved=dict(parameters=cpu(p),proxy_parameters=None,basis=basis,output_basis=space['Q'].cpu(),basis_B=cal['basis_B'],indices={})
    del small,c,j,jh,jr,response,ctx
    proxy=merge(anchor,expert,anchor,cfg['coefficient']);saved['proxy_parameters']=parameters(proxy,head)
    for group,model,which in [('native',expert,ids),('scale',expert,scale_ids),('proxy',proxy,ids[:cfg['proxy_n']])]:
        cache=capture(model,training,which,basis['vh']);pars=p if group!='proxy' else on_device(saved['proxy_parameters'])
        errors=[];rounding=[]
        for offset in range(0,len(which),cfg['batch_size']):
            h,_=suffix(cache['H'][offset:offset+cfg['batch_size']].cuda().double(),pars)
            actual=cache['actual'][offset:offset+cfg['batch_size']].cuda().double();actual=actual/actual.norm(dim=1,keepdim=True)
            errors.append(float((scores(h,pars)-actual@pars['head'].T).abs().max()))
            rounding.append((cache['h'][offset:offset+cfg['batch_size']].cuda().double()-h).cpu())
        assert max(errors)<cfg['full_cache_tolerance'];cache['rounding']=torch.cat(rounding)
        saved[group]=cache;saved['indices'][{'native':'fit','scale':'scale','proxy':'proxy'}[group]]=which
        print(json.dumps({'captured':group,'n':len(which),'tail_score_maximum_error':max(errors)}),flush=True)
    file=root/'calibration.pt';torch.save(saved,file)
    report=dict(target=target,calibration_sha256=digest(file),checkpoint_sha256=pins,selected_modes=chosen.cpu().tolist(),
                functional_response_fraction=float(importance[chosen].sum()/importance.sum()),
                task_singular_residual=float(((p['D']-pa['D'])@basis['vh'].cuda().T-basis['u'].cuda()*basis['s'].cuda()).norm()),
                new_full_model_forward_instances=len(ids)+len(scale_ids)+cfg['proxy_n']+cfg['selection_n'],
                test_used=False,partner_used=False,confirmation_used=False,expert_updated=False,
                wall_seconds=time.monotonic()-start,source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']))
    save_json(root/'results.json',report);save_json(root/'training_indices.json',saved['indices']);save_json(root/'protocol.json',cfg)
    print(json.dumps(report),flush=True)


class Cascade:
    def __init__(self,cfg):
        self.cfg=cfg;root=Path(cfg['cache_output']);meta=json.loads((root/'results.json').read_text());assert digest(root/'calibration.pt')==meta['calibration_sha256']
        cal=torch.load(root/'calibration.pt',map_location='cpu');self.cal=cal;self.p=on_device(cal['parameters']);self.pp=on_device(cal['proxy_parameters']);self.basis=on_device(cal['basis']);self.q=cal['output_basis'].cuda()
        self.rank=cfg['probe_rank'];self.size=self.rank**2;self.dimension=self.size+self.q.shape[1]*self.p['B'].shape[1]
        self.zero=torch.zeros(self.dimension,device='cuda',dtype=torch.float64);self.nd=self.p['D'].norm();self.nb=self.p['B'].norm()
        self.core=[];reordered=[];indices={}
        for group,key in [('native','fit'),('scale','scale')]:
            cache,ids,n=risk_first(cal[group],cal['indices'][key],cfg['contexts_per_class'],self.p)
            self.core.extend(range(sum(len(g['h']) for g in reordered),sum(len(g['h']) for g in reordered)+n));reordered.append(cache);indices[key]=ids
        self.indices=indices|{'proxy':cal['indices']['proxy']}
        combined={k:torch.cat([g[k] for g in reordered]).cuda() for k in reordered[0]}
        self.natural_n=len(combined['H']);self.n_fit=len(reordered[0]['H'])
        self.native=self.packed(combined,self.p,True);self.proxy=self.packed(on_device(cal['proxy']),self.pp,False)
        s=angular_scores(self.proxy['teacher']);self.winner=s.argmax(1);self.rows=torch.arange(len(s),device='cuda')
        others=s.clone();others[self.rows,self.winner]=-torch.inf;self.gaps=s[self.rows,self.winner]-others.max(1).values
        assert self.gaps.min()>0

    def packed(self,cache,p,probes):
        n=len(cache['H']);device='cuda';width=p['B'].shape[1]
        take=[torch.arange(n,device=device)];shift=[torch.zeros((n,self.rank),device=device,dtype=torch.float64)];offset=[torch.zeros((n,width),device=device,dtype=torch.float64)]
        up_map=self.basis['vh']@p['D'].T
        if probes:
            contexts=torch.tensor(self.core,device=device);count=2*self.rank
            rms=cache['t'][:,0].square().mean(0).sqrt();up=(self.cfg['finite_step']*torch.tensor([-1.,1.],device=device)[:,None,None]*torch.diag(rms)[None]).flatten(0,1)
            z=[]
            for start in range(0,n,self.cfg['batch_size']):z.append(suffix(cache['H'][start:start+self.cfg['batch_size']].double(),p)[1]['z'])
            z=torch.cat(z);vb=self.cal['basis_B']['vh'].cuda();rmsb=(z@vb.T).square().mean(0).sqrt()
            down=(self.cfg['finite_step']*torch.tensor([-1.,1.],device=device)[:,None,None]*rmsb[None,:,None]*vb[None]).flatten(0,1)
            repeat=contexts.repeat_interleave(count);take.extend([repeat,repeat]);shift.extend([up.repeat(len(contexts),1),torch.zeros((len(repeat),self.rank),device=device,dtype=torch.float64)])
            offset.extend([torch.zeros((len(repeat),width),device=device,dtype=torch.float64),down.repeat(len(contexts),1)])
        packed=dict(cache=cache,up_map=up_map,take=torch.cat(take),shift=torch.cat(shift),offset=torch.cat(offset))
        packed['teacher']=self.predict(packed,p,self.zero).detach();return packed

    def predict(self,packed,p,theta,factor=1.):
        values=[]
        for start in range(0,len(packed['take']),self.cfg['batch_size']):
            batch=select(packed,slice(start,start+self.cfg['batch_size']));h,_=forward(batch,p,theta,self.basis,self.q,factor)
            values.append(scores(h,p))
        return torch.cat(values)

    def residuals(self,theta):return signed_residual(self.native['teacher'],self.predict(self.native,self.p,theta))

    def class_harm(self,theta):
        s=angular_scores(self.predict(self.proxy,self.pp,theta,self.cfg['coefficient']));others=s.clone();others[self.rows,self.winner]=-torch.inf
        harm=1.-(s[self.rows,self.winner]-others.max(1).values)/self.gaps
        return torch.stack([harm[self.winner==c].mean() for c in range(s.shape[1])])

    def objective(self,theta):
        values=self.class_harm(theta);tau=self.cfg['class_temperature']
        game=-tau*(torch.logsumexp(-values/tau,0)-torch.log(torch.tensor(float(len(values)),device='cuda')))
        return theta.square().sum()/self.cfg['parameter_budget']**2+game

    @torch.no_grad()
    def metrics(self,theta):
        native=self.predict(self.native,self.p,theta);proxy=self.predict(self.proxy,self.pp,theta,self.cfg['coefficient']);changed=proxy.argmax(1)!=self.winner
        return dict(parameter_distance=float(theta.norm()),native_critic=float(signed_residual(self.native['teacher'],native).abs().max()),
                    natural_changes=int((native[:self.natural_n].argmax(1)!=self.native['teacher'][:self.natural_n].argmax(1)).sum()),
                    proxy_changed=int(changed.sum()),proxy_n=len(changed),changed_by_teacher_class=torch.bincount(self.winner[changed],minlength=proxy.shape[1]).cpu().tolist(),
                    class_harm=self.class_harm(theta).cpu().tolist(),objective=float(self.objective(theta)))

    @torch.no_grad()
    def critic(self,theta):
        residual=self.residuals(theta);worst=int(residual.abs().max(1).values.argmax());rows=torch.tensor(self.core+[worst],device='cuda')
        packed=select(self.native,rows);h,j=forward(packed,self.p,theta,self.basis,self.q,want_jacobian=True);after=scores(h,self.p)
        teacher=packed['teacher'];y=teacher.argmax(1);ri=torch.arange(len(y),device='cuda');keep=torch.ones_like(teacher,dtype=torch.bool);keep[ri,y]=False
        gaps=(teacher[ri,y,None]-teacher).masked_fill(~keep,1.)
        matrix=((j[ri,y,None]-j)/gaps[:,:,None])[keep];r=signed_residual(teacher,after)[keep]
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False);rank=int((s>torch.finfo(s.dtype).eps*max(matrix.shape)*s[0]).sum())
        return u[:,:rank],s[:rank],vh[:rank],r,packed,matrix


def correct(p,theta,steps,counters):
    history=[]
    for _ in range(steps):
        before=p.metrics(theta)
        if before['native_critic']<=p.cfg['task_budget']:return theta,history
        if counters['jacobians']>=p.cfg['maximum_jacobian_calls']:return None,history
        u,s,vh,r,_,_=p.critic(theta);counters['jacobians']+=1;q=-vh.T@((u.T@r)/s);accepted=None
        for k in range(p.cfg['backtracking_steps']):
            proposed=theta+(.5**k)*q
            if proposed.norm()>p.cfg['parameter_budget']:continue
            metrics=p.metrics(proposed)
            if metrics['native_critic']<before['native_critic']-1e-12:accepted=proposed;break
        history.append(dict(before=before,correction_norm=float(q.norm()),accepted=accepted is not None))
        if accepted is None:return None,history
        theta=accepted;counters['corrections']+=1
    return (theta if p.metrics(theta)['native_critic']<=p.cfg['task_budget'] else None),history


def diagnose(cfg):
    out=Path(cfg['diagnostic_output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=Cascade(cfg)
    theta=p.zero.clone();s=p.basis['s'];theta[:p.size]=(-cfg['seed_distance']*torch.diag(s)/s.norm()).flatten()
    initial=p.metrics(theta);counters=dict(jacobians=0,corrections=0);selected,history=correct(p,theta,cfg['seed_corrections'],counters)
    final=p.metrics(selected) if selected is not None else None;probe=theta if selected is None else selected
    *_,packed,matrix=p.critic(probe);counters['jacobians']+=1
    direction=torch.cos(torch.arange(p.dimension,device='cuda',dtype=torch.float64));direction/=direction.norm();eps=cfg['derivative_step']
    plus=p.predict(packed,p.p,probe+eps*direction);minus=p.predict(packed,p.p,probe-eps*direction);teacher=packed['teacher'];y=teacher.argmax(1)
    keep=torch.ones_like(teacher,dtype=torch.bool);keep[torch.arange(len(y),device='cuda'),y]=False
    fd=(signed_residual(teacher,plus)-signed_residual(teacher,minus))/(2*eps);error=float((fd[keep]-matrix@direction).abs().max());assert error<cfg['jacobian_tolerance']
    signal=final is not None and final['proxy_changed']>=cfg['minimum_finite_changes'] and sum(v>0 for v in final['changed_by_teacher_class'])>=cfg['minimum_changed_classes']
    packed=compact(packed)
    artifact=out/'diagnostic.pt';torch.save(dict(theta=selected.cpu() if selected is not None else None,attempted_theta=theta.cpu(),core={k:cpu(v) if isinstance(v,dict) else v.cpu() for k,v in packed.items()},matrix=matrix.cpu(),direction=direction.cpu(),cfg=cfg),artifact)
    report=dict(status='finite_attention_cascade_mechanism' if signal else 'closed_no_finite_multiclass_mechanism',target=cfg['target'],initial=initial,final=final,history=history,
                finite_multiclass_gate_passed=signal,analytic_direction_fd_error=error,**counters,diagnostic_sha256=digest(artifact),source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                expert_updated=False,new_release_weights=0,test_used=False,partner_used=False,confirmation_used=False,wall_seconds=time.monotonic()-start)
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['prepare','diagnose'],required=True);ap.add_argument('--protocol',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='prepare':prepare(cfg['target'],cfg)
    else:diagnose(cfg)
