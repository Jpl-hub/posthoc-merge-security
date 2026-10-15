"""Frozen finite gate displacement with implicit source-normal compensation."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from angular_distance import signed_residual,angular_scores
from attention_cascade import suffix
from boundary_game import risk_first
from classification_completion import readout,ideal_output
from clip_benchmark import digest,merge
from covariance_noise import dataset
from cross_module_distance import CrossProblem,extended_parameters,normalized_input
from null_bias_leakage import analyze_scores,slope
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader,save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import capture,checked_models,gelu,initialize
from verify_covariance_noise import official_labels


def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    return v


def prefix_key(key):
    stem='model.visual.transformer.resblocks.'
    return ((key.startswith(stem) and int(key[len(stem):].split('.')[0])<11) or
            key.startswith(('model.visual.conv1.','model.visual.class_embedding','model.visual.positional_embedding','model.visual.ln_pre.')))


def pack(pre,h,p,offset=None):
    offset=torch.zeros_like(pre) if offset is None else offset
    z=gelu(pre)+offset
    return dict(pre=pre,h=h,offset=offset,z=z,teacher=scores(h,p))


def forward(data,p,d,K,Q,V,factor=1.):
    Z=gelu(data['pre']+factor*d)+data['offset']
    up=(Z-data['z'])@p['B'].T;down=factor*(Z@V.T@K)@Q.T
    h=data['h']+up+down
    return scores(h,p),dict(h=h,up=up,down=down,Z=Z)


def class_values(data,actual,name):
    aa=angular_scores(actual);base=angular_scores(data['teacher']);yy=data['teacher'].argmax(1);rr=torch.arange(len(yy),device='cuda')
    alt=aa.clone();alt[rr,yy]=-torch.inf;runner=alt.argmax(1);old=base.clone();old[rr,yy]=-torch.inf
    damage=1-(aa[rr,yy]-aa[rr,runner])/(base[rr,yy]-old.max(1).values);values=[];details=[]
    for c in range(actual.shape[1]):
        ix=(yy==c).nonzero().flatten();order=damage[ix].argsort(stable=True);position=int(.25*(len(ix)-1));i=ix[order[position]]
        values.append(damage[i]);details.append(dict(environment=name,label=c,n=len(ix),sample=int(i),competitor=int(runner[i])))
    return values,details


class GateProblem:
    def __init__(self,cfg,source,p,U,V,envs,space):
        self.cfg=cfg;self.source=source;self.p=p;self.U=U;self.V=V;self.envs=envs;self.Q=space['Q'];self.space=space
        teacher=source['teacher'];ang=angular_scores(teacher);yy=teacher.argmax(1);tmp=ang.clone();tmp[torch.arange(len(yy),device='cuda'),yy]=-torch.inf
        g=ang[torch.arange(len(yy),device='cuda'),yy]-tmp.max(1).values;assert (g>0).all()
        count=torch.bincount(yy,minlength=len(p['head']));w=1/g.square()/count[yy];self.w=w/w.mean()
        self.na=p['a'].norm();self.nb=p['B'].norm()
    def complete(self,t):
        d=self.na*(self.U@t);s=self.source;Z=gelu(s['pre']+d)+s['offset'];hg=s['h']+(Z-s['z'])@self.p['B'].T
        _,target,_=ideal_output(hg,s['h'],self.space,self.p['eps']);X=Z@self.V.T;gram=X.T@(self.w[:,None]*X)
        ridge=self.cfg['ridge_relative']*gram.trace()/len(gram);K=torch.linalg.solve(gram+ridge*torch.eye(len(gram),device='cuda'),X.T@(self.w[:,None]*target))
        D=(t.square().sum()+K.square().sum()/self.nb.square()).sqrt()
        return d,K,D,dict(ridge=ridge,gram=gram,target=target,X=X)
    def critics(self,d,K):
        values=[];details=[]
        for name,data,p in self.envs:
            actual,_=forward(data,p,d,K,self.Q,self.V,self.cfg['coefficient']);vv,dd=class_values(data,actual,name);values.extend(vv);details.extend(dd)
        return torch.stack(values),details
    @torch.no_grad()
    def metrics(self,t,quantize=False):
        d,K,D,fit=self.complete(t)
        if quantize:
            d=(self.p['a']+d).float().double()-self.p['a'];db=(self.p['B']+self.Q@K.T@self.V).float().double()-self.p['B']
            # FP32 rounds outside the low-rank chart: materialized matrices decide.
            def material(data,p,factor):
                Z=gelu(data['pre']+factor*d)+data['offset'];h=data['h']+(Z-data['z'])@p['B'].T+factor*Z@db.T
                return scores(h,p),h
            ss,sh=material(self.source,self.p,1.);D=(d.norm().square()/self.na.square()+db.norm().square()/self.nb.square()).sqrt()
        else:ss,parts=forward(self.source,self.p,d,K,self.Q,self.V);sh=parts['h'];db=self.Q@K.T@self.V
        sr,sp=analyze_scores(self.source['teacher'],ss);environment=[];arrays=[];qvalues=[];details=[]
        for name,data,p in self.envs:
            if quantize:es,eh=material(data,p,self.cfg['coefficient'])
            else:es,ep=forward(data,p,d,K,self.Q,self.V,self.cfg['coefficient']);eh=ep['h']
            er,ap=analyze_scores(data['teacher'],es);environment.append(dict(name=name,**er));arrays.append(dict(name=name,before=data['teacher'],after=es,h=eh,**ap))
            vv,dd=class_values(data,es,name);qvalues.extend(vv);details.extend(dd)
        q=torch.stack(qvalues)
        report=dict(distance=float(D),source=sr,environment=environment,minimum_class_quantile=float(q.min()),class_quantiles=q.cpu().tolist(),critics=details,
            source_feasible=bool(D<=self.cfg['parameter_budget'] and sr['worst_task_drift']<=self.cfg['source_reserve'] and sr['changed']==0),
            compensation_gram_condition=float(torch.linalg.cond(fit['gram']+fit['ridge']*torch.eye(len(K),device='cuda'))),compensation_relative_residual=float(((fit['X']@K-fit['target'])*self.w.sqrt()[:,None]).norm()/((fit['target']*self.w.sqrt()[:,None]).norm().clamp_min(1e-20))))
        return report,dict(t=t,d=d,K=K,delta_B=db,source_before=self.source['teacher'],source_after=ss,source_h=sh,environment=arrays)


def dual_step(q,G,step):
    q=q.cpu().numpy();G=G.cpu().numpy();H=G@G.T;scale=max(np.linalg.norm(G,axis=1).max()*step,np.ptp(q),1e-12)
    def fun(pi):return (pi@q+step*np.sqrt(max(pi@H@pi,0)))/scale
    def jac(pi):return (q+step*(H@pi)/max(np.sqrt(max(pi@H@pi,0)),1e-20))/scale
    pi=np.zeros(len(q));pi[q.argmin()]=1
    sol=minimize(fun,pi,jac=jac,bounds=[(0,1)]*len(q),constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(len(x)))],method='SLSQP',options=dict(ftol=1e-12,maxiter=200))
    if not sol.success:return None,dict(solver_failed=True,message=str(sol.message),iterations=int(sol.nit))
    v=sol.x@G;norm=np.linalg.norm(v);direction=v/max(norm,1e-30);primal=float(np.min(q+step*(G@direction)));dual=float(sol.x@q+step*norm)
    if dual-primal>=1e-5*max(1,abs(dual)):return None,dict(solver_failed=True,message='affine model duality gap',primal=primal,dual=dual)
    return torch.tensor(direction,device='cuda'),dict(pi=sol.x.tolist(),primal=primal,dual=dual,gap=dual-primal)


def run(cfg):
    start=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);torch.cuda.reset_peak_memory_stats()
    for path,pin in cfg['inputs'].items():assert digest(path)==pin,path
    cal=torch.load(cfg['calibration_reference'],map_location='cpu');cached=torch.load(cfg['environment_cache'],map_location='cpu');anchor,expert,head,pins=checked_models(cfg['target'])
    p=on_device(extended_parameters(expert,head));p0=on_device(extended_parameters(anchor,head));raw,ids,n=risk_first(cal['cache'],cal['indices'],cfg['contexts_per_class'],p);raw=on_device(raw)
    with torch.no_grad():
        x=normalized_input(raw['r'],p);pre=x@p['A'].T+p['a'];u,s,v=torch.linalg.svd(p['A']-p0['A'],full_matrices=False);nn=cfg['selection_n'];J,_,_,_=raw_jacobians({k:a[:nn] for k,a in raw.items()},p);kp=(J@p['B'])*slope(pre[:nn])[:,None]
        sc=scores(raw['h'][:nn],p);yy=sc.argmax(1);rr=torch.arange(nn,device='cuda');gap=sc[rr,yy,None]-sc;gap[rr,yy]=1
        response=(kp@u)*(x[:nn]@v.T)[:,None]*s[None,None];rel=(response[rr,yy,None]-response)/gap[:,:,None];rel[rr,yy]=0;count=torch.bincount(yy,minlength=len(p['head']));importance=(rel.square().sum(1)/count[yy,None]).sum(0)
        chosen=importance.argsort(descending=True,stable=True)[:cfg['probe_rank']];U=u[:,chosen];Va=v[chosen];V=on_device(cal['bases']['B'])['vh'];space,_=readout(p)
        cp=CrossProblem.__new__(CrossProblem);cp.cfg=dict(probe_contexts=n,finite_step=cfg['finite_step'],full_B_completion=True);cp.r=cfg['probe_rank'];cp.basis={'A':on_device(cal['bases']['O']),'B':on_device(cal['bases']['B'])};cp.na=p['O'].norm();cp.nb=p['B'].norm();cp.a_size=cp.r**2;cp.b_rows=p['B'].shape[0];cp.b_width=p['B'].shape[1];cp.output_basis=None
        data=cp.prepare(cpu(raw),p,True);pr=normalized_input(data['r'],p)@p['A'].T+p['a'];Z=gelu(pr)+data['b_offset'];hh=data['rounding']+data['r']+Z@p['B'].T+p['b'];source=pack(pr,hh,p,data['b_offset'])
        rms=(x@Va.T).square().mean(0).sqrt();F=p['A']@Va.T*rms[None];frames=[];extra_pre=[];extra_h=[]
        for i in range(n):
            jh,_,_,_=raw_jacobians(dict(r=raw['r'][i:i+1],h=raw['h'][i:i+1]),p);ju=((jh[0]@p['B'])*slope(pre[i])[None])@F;y=int(sc[i].argmax());normal=ju[y,None]-ju;normal=normal[torch.arange(len(p['head']),device='cuda')!=y]
            _,sv,frame=torch.linalg.svd(normal,full_matrices=True);rank=int((sv>sv[0]*torch.finfo(sv.dtype).eps*max(normal.shape)).sum());shift=cfg['neighborhood_radius']*torch.cat([frame,-frame])@F.T
            extra_pre.append(pre[i]+shift);extra_h.append(raw['h'][i]+(gelu(pre[i]+shift)-gelu(pre[i]))@p['B'].T);frames.append(dict(frame=frame.cpu(),rank=rank))
        extra=pack(torch.cat(extra_pre),torch.cat(extra_h),p);source={k:torch.cat([source[k],extra[k]]) for k in source}
    envs=[];ecaches=[];training=dataset(cfg['target'],'data',expert.val_preprocess,True);env_ids=cached['indices']['proxy'];pe=on_device(cached['proxy_parameters'])
    for coeff in cfg['prefix_coefficients']:
        name=f'prefix_{coeff:.3f}'
        print(json.dumps(dict(stage='shared_prefix_environment_capture',coefficient=coeff,n=len(env_ids))),flush=True)
        with torch.no_grad():
            if coeff==.5:
                he,context=suffix(on_device(cached['proxy'])['H'].double(),pe);h=he+on_device(cached['proxy'])['rounding'];xx=normalized_input(context['r'],pe);cache=dict(x=xx.cpu(),h=h.cpu(),actual=cached['proxy']['actual'])
            else:
                model=merge(anchor,expert,anchor,.5);state=model.state_dict();sa=anchor.state_dict();se=expert.state_dict()
                changed=[]
                for key in state:
                    if prefix_key(key):state[key].copy_(sa[key]+coeff*(se[key]-sa[key]));changed.append(key)
                assert changed and not any('.11.' in key for key in changed)
                cache=capture(model,training,env_ids);del model
            pp=on_device(cache);original=scores(pp['h'],pe);actual=(pp['actual']/pp['actual'].norm(dim=1,keepdim=True))@pe['head'].T;error=float((original-actual).abs().max());assert error<5e-4
            ee=pack(pp['x']@pe['A'].T+pe['a'],pp['h'],pe);envs.append((name,ee,pe));ecaches.append(dict(name=name,prefix_coefficient=coeff,cache=cache,baseline_full_model_score_error=error))
    torch.save(cpu(dict(environments=ecaches,source_indices=ids,environment_indices=env_ids,frames=frames,U=U,V=V,input_V=Va,rms=rms,chosen=chosen,importance=importance,cfg=cfg)),out/'geometry.pt')
    problem=GateProblem(cfg,source,p,U,V,envs,space)
    with torch.no_grad():
        center=-U.T@pre[:len(ids)].mean(0);assert center.norm()>1e-12;t=cfg['initial_gate_norm']*center/center.norm();initial,point=problem.metrics(t)
    history=[];trace=[];jacobians=[];stops=[];gradient_calls=0;accepted=0
    print(json.dumps(dict(stage='finite_initial_compensation',**initial)),flush=True)
    if not initial['source_feasible']:stops.append('fixed_finite_gate_seed_not_source_feasible')
    else:
        for iteration in range(cfg['maximum_updates']):
            leaf=t.clone().requires_grad_(True);d,K,D,fit=problem.complete(leaf);q,critics=problem.critics(d,K)
            gradients=[]
            for i in range(len(q)):gradients.append(torch.autograd.grad(q[i],leaf,retain_graph=True)[0]);gradient_calls+=1
            gd=torch.autograd.grad(D.square(),leaf)[0];gradient_calls+=1;G=torch.stack(gradients);direction,dual=dual_step(q.detach(),(G+cfg['distance_weight']*gd[None]).detach(),cfg['step'])
            jacobians.append(cpu(dict(t=t,q=q,G=G,distance_gradient=gd,direction=direction,critics=critics,dual=dual)));trials=[];success=False
            if direction is None:stops.append('frozen_affine_class_environment_solver_not_certified');break
            for k in range(cfg['backtracking_steps']):
                step=cfg['step']*.5**k;proposal=t+step*direction;metrics,trial_point=problem.metrics(proposal);prediction=float((q.detach()+step*(G.detach()@direction)).min()-q.detach().min());actual=metrics['minimum_class_quantile']-float(q.detach().min())
                ok=bool(metrics['source_feasible'] and prediction>1e-8 and actual>=max(cfg['minimum_improvement'],cfg['acceptance_fraction']*prediction))
                trials.append(dict(step=step,prediction=prediction,actual=actual,accepted=ok,metrics=metrics));trace.append(cpu(dict(t=proposal,d=trial_point['d'],K=trial_point['K'],source_after=trial_point['source_after'],environment_after=[a['after'] for a in trial_point['environment']],metrics=metrics)))
                if ok:t=proposal;point=trial_point;accepted+=1;success=True;break
            history.append(dict(iteration=iteration+1,dual=dual,trials=trials,accepted=success));print(json.dumps(history[-1]),flush=True);save_json(out/'history.json',history)
            if not success:stops.append('no_source_feasible_actual_multiclass_improvement_in_frozen_trust_steps');break
    final,point=problem.metrics(t,quantize=True);labels=official_labels(cfg['target'],True);environment=[]
    for report,arrays in zip(final['environment'],point['environment']):
        lab=torch.tensor(labels[env_ids],device='cuda');a=arrays['before'].argmax(1);b=arrays['after'].argmax(1);report.update(original_correct=int((a==lab).sum()),protected_correct=int((b==lab).sum()),lost=int(((a==lab)&(b!=lab)).sum()),gained=int(((a!=lab)&(b==lab)).sum()));environment.append(report)
    support=bool(final['source_feasible'] and all(e['lost']-e['gained']>=cfg['minimum_net_losses'] and e['classes_changed']>=cfg['minimum_classes'] for e in environment))
    checks={}
    if final['source_feasible']:
        released=copy.deepcopy(expert);stem='model.visual.transformer.resblocks.11.mlp.';released.state_dict()[stem+'c_fc.bias'].copy_((p['a']+point['d']).float().cpu());released.state_dict()[stem+'c_proj.weight'].copy_((p['B']+point['delta_B']).float().cpu());take=ids[:cfg['full_replay_n']]
        for name,model,expected in [('original',expert,point['source_before'][:len(take)].argmax(1)),('prototype',released,point['source_after'][:len(take)].argmax(1))]:
            model.cuda().eval();head.cuda().eval();pred=[]
            for images,_ in make_loader(training,take,64):pred.append(head(model(images.cuda())).argmax(1).cpu())
            model.cpu();head.cpu();assert torch.equal(torch.cat(pred),expected.cpu()),name;checks[name]=dict(n=len(take),all_predictions_match=True)
    torch.save(cpu(dict(selected=point,trace=trace,jacobians=jacobians,initial=initial,history=history,cfg=cfg)),out/'diagnostic.pt')
    result=dict(target=cfg['target'],method='finite_gate_coordinate_with_implicit_class_normal_compensation',status='mechanism_supported_requires_fresh_confirmation' if support else 'closed_finite_gate_compensation',mechanism_supported=support,confirmation_eligible=support,confirmation_passed=False,new_release_weights=0,new_best=False,
        initial=initial,selected=final,history=history,stop_reasons=stops,accepted_updates=accepted,selected_step=accepted,explicit_reverse_gradient_calls=gradient_calls,analytic_compensator_solves=1+len(trace)+len(jacobians)+1,
        original_expert_updated=False,old_candidate_loaded=False,peer_loaded=False,test_loaded=False,confirmation_loaded=False,source_natural_n=len(ids),source_task_inputs=len(source['pre']),environment_n=len(env_ids),environment_full_model_forwards=2*len(env_ids),full_replay=checks,
        diagnostic_sha256=digest(out/'diagnostic.pt'),geometry_sha256=digest(out/'geometry.pt'),checkpoint_sha256=pins,source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitation='One fixed gate/implicit-compensator chart, finite source-reserved trust acceptance; three nodes of one actual shared prefix task-vector family. These nodes are not an unknown-peer distribution or continuous certificate. Gradients propose only; complete finite source/class responses decide. Source.25 reserve within original dual.5 budgets. No closed transport point, peer or exposed confirmation responses used. No anti-recovery claim.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);save_json(out/'training_indices.json',dict(fit=ids,environment=env_ids));print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();cfg=json.loads(Path(args.protocol).read_text());initialize()
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    run(cfg)
