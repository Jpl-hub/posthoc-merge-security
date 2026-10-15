"""One finite task-subspace neuron-reflection game with a budget certificate."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds,LinearConstraint,milp

from angular_distance import signed_residual
from boundary_game import risk_first
from clip_benchmark import digest
from finite_compensation_transport import geometry,sigma
from mixed_compensation_audit import ln
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import gelu
from verify_covariance_noise import official_labels


def cpu(p):return {k:v.cpu() if torch.is_tensor(v) else v for k,v in p.items()}


@torch.no_grad()
def inputs(cfg,p,basis):
    boundary=torch.load(cfg['boundary_diagnostic'],map_location='cpu');old=torch.load(cfg['old_calibration'],map_location='cpu')
    caches=[];ids=[];core=[]
    for cache,which in [(boundary['screen'],boundary['split']['indices']),(old['scale'],old['indices']['scale'])]:
        c,order,n=risk_first(cache,which,cfg['contexts_per_class'],p);offset=sum(len(g['h']) for g in caches)
        core.extend(range(offset,offset+n));caches.append(c);ids.extend(order)
    raw={k:torch.cat([c[k] for c in caches]).cuda() for k in ['r','h']};x=ln(raw['r'],p);z=gelu(x@p['A'].T+p['a']);base=raw['h']-z@p['B'].T
    n=len(x);core=torch.tensor(core,device='cuda');va=basis['A']['vh'].cuda();vb=basis['B']['vh'].cuda();r=len(va)
    ra=(x@va.T).square().mean(0).sqrt();rb=(z@vb.T).square().mean(0).sqrt();sign=torch.tensor([-1.,1.],device='cuda')
    sa=(cfg['finite_step']*sign[:,None,None]*ra[None,:,None]*va[None]).flatten(0,1)
    sb=(cfg['finite_step']*sign[:,None,None]*rb[None,:,None]*vb[None]).flatten(0,1)
    xx=torch.cat([x,(x[core,None]+sa[None]).flatten(0,1),x[core].repeat_interleave(2*r,0)])
    bb=torch.cat([base,base[core].repeat_interleave(2*r,0),base[core].repeat_interleave(2*r,0)+sb.repeat(len(core),1)@p['B'].T])
    return dict(x=xx,base=bb,core=core,natural_n=n,indices=ids)


def fractional_upper(values,cost,budget):
    keep=values>0;v,w=values[keep],cost[keep];order=np.argsort(-v/np.maximum(w,1e-30));v,w=v[order],w[order]
    previous=np.cumsum(w)-w;fraction=np.clip((budget-previous)/np.maximum(w,1e-30),0.,1.)
    return float(v@fraction)


@torch.no_grad()
def construct(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    for file,pin in cfg['inputs'].items():assert digest(file)==pin,file
    original=torch.load(cfg['mixed_diagnostic'],map_location='cpu');p=on_device(original['parameters'])
    basis=torch.load(cfg['basis_diagnostic'],map_location='cpu')['basis'];packed=inputs(cfg,p,basis)
    va=basis['A']['vh'].cuda();projection=(p['A']@va.T)@va
    pre=packed['x']@p['A'].T+p['a'];signal=packed['x']@projection.T
    z=gelu(pre);endpoint=gelu(pre-2*signal)-z;middle=gelu(pre-signal)-z
    h=packed['base']+z@p['B'].T;teacher=scores(h,p);classes=teacher.shape[1];core=packed['core']
    cost=4*projection.square().sum(1)/p['A'].square().sum();w=cost.cpu().numpy();budget=cfg['parameter_budget']**2
    m,b=geometry(p);bn=(p['B']-p['B'].mean(0)).norm(dim=0)/(p['B'].shape[0]**.5)
    # A fractional knapsack gives a true per-input upper bound for every
    # binary mask at the source parameter midpoint, even ignoring task cost.
    count=min(cfg['bound_n'],packed['natural_n']);n0=h[:count]@m+sigma(h[:count],p)*b;y=n0.argmax(1);possible=[];bound_rows=[]
    for i in range(count):
        values=[]
        for c in range(classes):
            if c==int(y[i]):continue
            gap=float(n0[i,y[i]]-n0[i,c]);mm=m[:,y[i]]-m[:,c];bb=b[y[i]]-b[c]
            contribution=-middle[i]*(p['B'].T@mm)+middle[i].abs()*bn*bb.abs()
            values.append(fractional_upper(contribution.cpu().numpy(),w,budget)/gap)
        bound_rows.append(max(values));possible.append(max(values)>=1.)
    bound_class=torch.bincount(y[torch.tensor(possible,device='cuda')],minlength=classes).cpu().tolist()
    mask=torch.zeros(len(cost),device='cuda',dtype=torch.float64);solver=None;reason='fractional_midpoint_bound_excludes_multiclass_signal'
    if sum(possible)>=cfg['minimum_finite_changes'] and sum(n>0 for n in bound_class)>=cfg['minimum_changed_classes']:
        # Finite per-neuron endpoint changes are exact. Only the final CLIP
        # readout in these generator constraints is linearized; all final
        # natural/probe and midpoint decisions are recomputed nonlinearly.
        jh,_,_,_=raw_jacobians({'h':h[core],'r':torch.zeros_like(h[core])},p)
        s=teacher[core];yy=s.argmax(1);rows=torch.arange(len(s),device='cuda');keep=torch.ones_like(s,dtype=torch.bool);keep[rows,yy]=False
        gaps=(s[rows,yy,None]-s).masked_fill(~keep,1.)
        jac=(jh[rows,yy,None]-jh)/gaps[:,:,None]
        native=((jac@p['B'])*endpoint[core,None,:])[keep].cpu().numpy()
        nfit=cfg['objective_n'];numer=h[:nfit]@m+sigma(h[:nfit],p)*b;pred=numer.argmax(1);other=numer.clone();other[torch.arange(nfit,device='cuda'),pred]=-torch.inf;runner=other.argmax(1)
        gap=numer[torch.arange(nfit,device='cuda'),pred]-numer[torch.arange(nfit,device='cuda'),runner]
        response=-middle[:nfit]*((m[:,pred]-m[:,runner]).T@p['B'])/gap[:,None]
        harm=torch.stack([response[pred==c].mean(0) for c in range(classes)]).cpu().numpy()
        a=np.column_stack([native,np.zeros(len(native))]);class_a=np.column_stack([-harm,np.ones(classes)])
        matrix=np.vstack([a,class_a,np.r_[w,0.][None]])
        low=np.r_[np.full(len(native),-cfg['generator_margin_fraction']),np.full(classes,-np.inf),-np.inf]
        high=np.r_[np.full(len(native),cfg['generator_margin_fraction']),np.zeros(classes),budget]
        objective=np.r_[-w/budget,-1.]
        solved=milp(objective,integrality=np.r_[np.ones(len(w)),0.],bounds=Bounds(np.zeros(len(w)+1),np.r_[np.ones(len(w)),np.inf]),
                    constraints=LinearConstraint(matrix,low,high),options=dict(time_limit=cfg['solver_seconds'],node_limit=cfg['solver_node_limit'],mip_rel_gap=cfg['solver_gap']))
        solver=dict(status=int(solved.status),message=solved.message,incumbent=solved.x is not None)
        if solved.x is not None:
            assert np.max(np.abs(solved.x[:-1]-np.round(solved.x[:-1])))<1e-4
            mask=torch.tensor(np.round(solved.x[:-1]),device='cuda');assert float(cost@mask)<=budget+1e-10
            solver.update(objective=float(solved.fun),dual_bound=float(solved.mip_dual_bound),relative_gap=float(solved.mip_gap),nodes=int(solved.mip_node_count),worst_class_linear_harm=float(solved.x[-1]))
            reason='one_frozen_finite_reflection_mask'
    # Freeze the sole discrete parameter transform before finite validation.
    file=out/'point.pt';torch.save(dict(mask=mask.cpu(),input_basis=va.cpu(),cfg=cfg),file);pin=digest(file)
    after=scores(h+(endpoint*mask)@p['B'].T,p);mid=scores(h[:packed['natural_n']]+(middle[:packed['natural_n']]*mask)@p['B'].T,p)
    drift=float(signed_residual(teacher,after).abs().max());n=packed['natural_n'];native_changed=after[:n].argmax(1)!=teacher[:n].argmax(1);mid_changed=mid.argmax(1)!=teacher[:n].argmax(1)
    by=torch.bincount(teacher[:n].argmax(1)[mid_changed],minlength=classes).cpu().tolist();distance=float((cost@mask).sqrt())
    passed=drift<=cfg['task_budget'] and int(mid_changed.sum())>=cfg['minimum_finite_changes'] and sum(v>0 for v in by)>=cfg['minimum_changed_classes']
    labels=torch.tensor(official_labels(cfg['target'],True)[packed['indices']],device='cuda')
    torch.save(dict(source_scores=teacher.cpu(),endpoint_scores=after.cpu(),source_midpoint_scores=mid.cpu(),indices=packed['indices'],upper_bounds=torch.tensor(bound_rows),cfg=cfg),out/'predictions.pt')
    report=dict(target=cfg['target'],status='finite_reflection_mechanism' if passed else 'closed_reflection_no_valid_multiclass_barrier',reason=reason,
                fractional_bound_possible=sum(possible),fractional_bound_n=count,fractional_bound_possible_by_class=bound_class,selected_neurons=int(mask.sum()),parameter_distance=distance,
                task_worst_margin_drift=drift,all_task_inputs=len(h),natural_n=n,native_changed=int(native_changed.sum()),source_midpoint_changed=int(mid_changed.sum()),source_midpoint_changed_by_class=by,
                native_original_correct=int((teacher[:n].argmax(1)==labels).sum()),native_released_correct=int((after[:n].argmax(1)==labels).sum()),solver=solver,finite_multiclass_gate_passed=passed,
                point_sha256=pin,predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),
                wall_seconds=time.monotonic()-start,new_release_weights=0,expert_updated=False,test_used=False,partner_used=False,confirmation_used=False,
                limitation='One task-SVD input reflection subset. Finite hidden responses exact, MILP readout constraints are only a generator. No unknown-peer coverage, originalAGA result, or recovery guarantee.')
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;construct(cfg)
