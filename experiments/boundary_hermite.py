"""Source-only decision-boundary value/derivative preserving publication chart."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from boundary_game import risk_first
from classification_completion import feature_space,raw_score_target
from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import capture_attention,extended_parameters,final_hidden,normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models,gelu,initialize


def path_pack(endpoints,t,p):
    out={k:endpoints[k][:,0]+t[:,None]*(endpoints[k][:,1]-endpoints[k][:,0]) for k in ['r','o_input','rounding']}
    out['teacher_h']=out['rounding']+final_hidden(out['r'],p)
    return out


class HermiteChart:
    def __init__(self,params,basis,endpoints,roots,space,cfg):
        self.p,self.basis,self.endpoints,self.roots,self.space,self.cfg=params,basis,endpoints,roots,space,cfg
        self.rank=cfg['probe_rank'];self.na=params['O'].norm();self.nb=params['B'].norm()
        self.q=torch.eye(params['B'].shape[0],device=params['B'].device,dtype=params['B'].dtype) if cfg.get('completion_target')=='hidden' else space['Q']

    def delta_O(self,c):return self.na*self.basis['u']@c@self.basis['vh']

    def path_targets(self,c,t):
        a=path_pack(self.endpoints,t,self.p);r=a['r']+a['o_input']@self.delta_O(c).T
        z=gelu(normalized_input(r,self.p)@self.p['A'].T+self.p['a'])
        h0=a['rounding']+r+z@self.p['B'].T+self.p['b']
        if self.cfg.get('completion_target')=='hidden':target=a['teacher_h']-h0
        else:_,target,_=raw_score_target(h0,a['teacher_h'],self.space,self.p['eps'],self.cfg['fixed_point_iterations'])
        return z,target/self.nb

    def factors(self,c,derivatives=True,create_graph=False):
        def fn(t):return self.path_targets(c,t)
        if derivatives:
            (z,y),(dz,dy)=torch.autograd.functional.jvp(fn,self.roots,torch.ones_like(self.roots),create_graph=create_graph)
            f=torch.cat([z,dz]);target=torch.cat([y,dy])
        else:f,target=fn(self.roots)
        scale=f.norm(dim=1,keepdim=True);f=f/scale;target=target/scale
        f_all,target_all=f,target
        if self.cfg.get('canonical_path_rows') is not None:
            rows=self.cfg['canonical_path_rows'];n=len(self.roots)
            rows=rows+[n+i for i in rows] if derivatives else rows
            f,target=f[rows],target[rows]
        gram=f@f.T
        part=target.T@torch.linalg.solve(gram,f)
        return part,f,gram,target,f_all,target_all

    def complete(self,c,free=None,derivatives=True,create_graph=False):
        part,f,gram,target,f_all,target_all=self.factors(c,derivatives,create_graph)
        k=part if free is None else part+free-(free@f.T)@torch.linalg.solve(gram,f)
        return k,dict(F=f,gram=gram,target=target,part=part,F_all=f_all,target_all=target_all)

    def predict_h(self,a,c,k,p=None,factor=1.):
        p=self.p if p is None else p
        r=a['r']+factor*a['o_input']@self.delta_O(c).T
        z=gelu(normalized_input(r,p)@p['A'].T+p['a'])+a.get('b_offset',0.)
        return a['rounding']+r+z@(p['B']+factor*self.nb*self.q@k).T+p['b']

    def predict(self,a,c,k,p=None,factor=1.):return scores(self.predict_h(a,c,k,p,factor),self.p if p is None else p)


@torch.no_grad()
def prepare_data(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    anchor,expert,head,pins=checked_models(cfg['target']);pcpu=extended_parameters(expert,head);p=on_device(pcpu)
    training=dataset(cfg['target'],'data',expert.val_preprocess,True)
    split=unused(training,set(),dict(cfg,confirmation_n=cfg['construction_n'],confirmation_seed=cfg['construction_seed']))
    split.update(used_for_optimization=True,role='predeclared_new_construction_only_not_confirmation')
    save_json(out/'construction_split.json',split);save_json(out/'training_indices.json',{'fit':split['indices']})
    cache=capture_attention(expert,training,split['indices']);raw=on_device(cache);raw['rounding']=raw['h']-final_hidden(raw['r'],p)
    # Actual task-update components are ranked by their finite-source class
    # Jacobian action, not by the singular value alone.
    base=extended_parameters(anchor,head);n=cfg['selection_n'];small={k:v[:n] for k,v in raw.items()}
    jh,jr,_,ss=raw_jacobians(small,p);y=ss.argmax(1);counts=torch.bincount(y,minlength=len(p['head']));w=1./counts[y].double()/int((counts>0).sum())
    hidden=gelu(normalized_input(small['r'],p)@p['A'].T+p['a']);bases={};geometry={}
    for name,inputs,jac in [('O',small['o_input'],jr),('B',hidden,jh)]:
        u,s,vh=torch.linalg.svd((pcpu[name]-base[name]).cuda(),full_matrices=False)
        response=(jac@u)*(inputs@vh.T)[:,None,:]*s
        importance=(response.square().sum(1)*w[:,None]).sum(0);chosen=importance.argsort(descending=True,stable=True)[:cfg['probe_rank']]
        bases[name]=dict(u=u[:,chosen],s=s[chosen],vh=vh[chosen])
        geometry[name]=dict(shape=list(p[name].shape),selected=chosen.cpu().tolist(),functional_fraction=float(importance[chosen].sum()/importance.sum()),
            spectral_top32_functional_fraction=float(importance[:cfg['probe_rank']].sum()/importance.sum()))
    before=scores(raw['h'],p);winner=before.argmax(1);others=before.clone();row=torch.arange(len(before),device='cuda');others[row,winner]=-torch.inf
    runner=others.argmax(1);gap=before[row,winner]-others.max(1).values
    coordinates=raw['o_input']@bases['O']['vh'].T;unit=coordinates.square().mean(0).sqrt().clamp_min(1e-12);coordinates=coordinates/unit
    aidx=[];bidx=[];pairs=[]
    for cls in range(len(p['head'])):
        members=(winner==cls).nonzero().flatten();chosen=members[gap[members].argsort(stable=True)[:cfg['paths_per_class']]]
        assert len(chosen)==cfg['paths_per_class']
        for i in chosen:
            other=int(runner[i]);possible=(winner==other).nonzero().flatten();dist=(coordinates[possible]-coordinates[i]).square().sum(1)
            j=possible[dist.argmin()];aidx.append(int(i));bidx.append(int(j));pairs.append((cls,other))
    endpoints={k:torch.stack([raw[k][aidx],raw[k][bidx]],1) for k in ['r','o_input','rounding']}
    pairs=torch.tensor(pairs,device='cuda');ar=torch.arange(len(pairs),device='cuda')
    def pairgap(t):
        a=path_pack(endpoints,t,p);ss=scores(a['teacher_h'],p);return ss[ar,pairs[:,0]]-ss[ar,pairs[:,1]]
    lo=torch.zeros(len(pairs),device='cuda',dtype=torch.float64);hi=torch.ones_like(lo)
    assert (pairgap(lo)>0).all() and (pairgap(hi)<0).all()
    for _ in range(cfg['root_iterations']):
        mid=(lo+hi)/2;positive=pairgap(mid)>0;lo=torch.where(positive,mid,lo);hi=torch.where(positive,hi,mid)
    roots=(lo+hi)/2;root_error=float(pairgap(roots).abs().max());assert root_error<cfg['root_tolerance']
    root_scores=scores(path_pack(endpoints,roots,p)['teacher_h'],p);pairtop=torch.stack([root_scores[ar,pairs[:,0]],root_scores[ar,pairs[:,1]]],1)
    genuine=root_scores.max(1).values-pairtop.min(1).values<cfg['root_tolerance']
    space,space_info=feature_space(p);saved=dict(parameters=pcpu,bases={k:{a:b.cpu() for a,b in v.items()} for k,v in bases.items()},
        endpoints={k:v.cpu() for k,v in endpoints.items()},roots=roots.cpu(),pairs=pairs.cpu(),space={k:v.cpu() for k,v in space.items()},
        cache=cache,indices=split['indices'],path_source_positions=aidx,path_end_positions=bidx,checkpoint_sha256=pins,cfg=cfg)
    torch.save(saved,out/'calibration.pt');save_json(out/'geometry.json',dict(bases=geometry,space=space_info,root_gap_maximum=root_error,
        endpoint_indices=[[split['indices'][a],split['indices'][b]] for a,b in zip(aidx,bidx)],pairs=pairs.cpu().tolist(),genuine_top_class_boundaries=int(genuine.sum()),
        wall_seconds=time.monotonic()-start,original_expert_updates=0,new_full_model_forwards=len(split['indices']),calibration_sha256=digest(out/'calibration.pt')))
    save_json(out/'protocol.json',cfg);print(json.dumps({'calibration_sha256':digest(out/'calibration.pt'),'root_error':root_error,'paths':len(pairs)}),flush=True)


def load_chart(cfg):
    path=Path(cfg.get('calibration_reference',str(Path(cfg['output'])/'calibration.pt')))
    if cfg.get('calibration_sha256'):assert digest(path)==cfg['calibration_sha256']
    saved=torch.load(path,map_location='cpu');p=on_device(saved['parameters'])
    chart=HermiteChart(p,on_device(saved['bases']['O']),on_device(saved['endpoints']),saved['roots'].cuda(),on_device(saved['space']),cfg)
    return saved,chart


def diagnose(cfg):
    start=time.monotonic();out=Path(cfg['output'])
    if cfg.get('calibration_reference'):out.mkdir(parents=True,exist_ok=False)
    saved,chart=load_chart(cfg);p=chart.p;calibration=Path(cfg.get('calibration_reference',str(out/'calibration.pt')))
    c=-cfg['mechanism_O_distance']*torch.diag(chart.basis['s'])/chart.basis['s'].norm();records=[];raw=on_device(saved['cache']);raw['rounding']=raw['h']-final_hidden(raw['r'],p)
    before=scores(raw['h'],p);pairs=saved['pairs'].cuda();ar=torch.arange(len(pairs),device='cuda');snap={}
    for name,kind in [('uncompensated',None),('value_only',False),('value_and_derivative',True)]:
        if kind is None:k=torch.zeros(len(chart.q.T),p['B'].shape[1],device='cuda',dtype=torch.float64);finfo=None
        else:k,finfo=chart.complete(c,derivatives=kind)
        def residual_at(t):
            a=path_pack(chart.endpoints,t,p);diff=chart.predict(a,c,k)-scores(a['teacher_h'],p)
            return diff-diff.mean(1,keepdim=True)
        root_value,root_slope=torch.autograd.functional.jvp(residual_at,chart.roots,torch.ones_like(chart.roots))
        finite=[]
        for exponent in cfg['offset_exponents']:
            alpha=2.**-exponent
            for sign in [-1.,1.]:
                # Stay inside the real-context chord, with a fixed local scale.
                t=chart.roots+sign*alpha*torch.minimum(chart.roots,1-chart.roots)
                a=path_pack(chart.endpoints,t,p);teacher=scores(a['teacher_h'],p);ss=chart.predict(a,c,k)
                g=teacher[ar,pairs[:,0]]-teacher[ar,pairs[:,1]];dg=(ss-teacher)[ar,pairs[:,0]]-(ss-teacher)[ar,pairs[:,1]]
                finite.append(dict(exponent=exponent,sign=sign,max_pair_relative_drift=float((dg.abs()/g.abs()).max()),
                    pair_gap=g.detach().cpu(),pair_gap_change=dg.detach().cpu(),maximum_raw_score_error=float((ss-teacher).abs().max())))
        natural=chart.predict(raw,c,k);drift=signed_residual(before,natural);dist=float((c.square().sum()+k.square().sum()).sqrt())
        condition=None if finfo is None else float(torch.linalg.cond(finfo['F']))
        eq=None if finfo is None else float((finfo['F_all']@k.T-finfo['target_all']).abs().max())
        row=dict(method=name,parameter_distance=dist,root_score_error=float(root_value.abs().max()),root_score_derivative_error=float(root_slope.abs().max()),
            natural_worst_task=float(drift.abs().max()),natural_changed=int((natural.argmax(1)!=before.argmax(1)).sum()),
            constraint_matrix_condition=condition,linear_equality_maximum_error=eq,finite=[{a:b for a,b in v.items() if not torch.is_tensor(b)} for v in finite])
        records.append(row);snap[name]=dict(C=c.cpu(),K=k.detach().cpu(),root_value=root_value.cpu(),root_slope=root_slope.cpu(),finite=finite,natural_scores=natural.detach().cpu())
        print(json.dumps(row),flush=True)
    verified=records[-1];near=max(x['max_pair_relative_drift'] for x in verified['finite'][-2:]);far=max(x['max_pair_relative_drift'] for x in verified['finite'][:2])
    supported=bool(verified['root_score_error']<cfg['score_tolerance'] and verified['root_score_derivative_error']<cfg['derivative_tolerance']
        and verified['constraint_matrix_condition']<cfg['maximum_condition'] and verified['linear_equality_maximum_error']<cfg['linear_equality_tolerance'])
    supported=supported and near<=far/cfg['minimum_relative_decay']
    # Pointwise finite decay is reported, never advertised as an interval
    # curvature certificate or unseen-input/unknown-peer coverage.
    torch.save(dict(records=snap,teacher=before.cpu(),calibration_sha256=digest(calibration)),out/'mechanism.pt')
    result=dict(status='boundary_Hermite_chart_numerically_supported' if supported else 'closed_boundary_Hermite_chart_failed',
        mechanism_supported=supported,near_far_relative_error_ratio=near/max(far,1e-20),records=records,new_release_weights=0,original_expert_updates=0,
        independent_confirmation_run=False,true_TA_run=False,author_AGA_run=False,peer_used=False,test_used=False,
        exposed_confirmation_values_used=False,calibration_sha256=digest(calibration),mechanism_sha256=digest(out/'mechanism.pt'),
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        limitation='Hermite matching on20 target-training activation chords, not a uniform bound transverse to chords or future-peer coverage. One fixed finite seed is mechanism-only and not a releasable candidate. No amplitude/path/rank sweep.')
    save_json(out/'mechanism.json',result);save_json(out/'protocol.json',cfg)
    save_json(out/'training_indices.json',{'fit':saved['indices']});print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());initialize()
    for name,pin in cfg['source_sha256'].items():assert digest(name)==pin,name
    prepare_data(cfg) if args.prepare else diagnose(cfg)
