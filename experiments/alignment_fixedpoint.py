"""One source-only AGA fixed-point head relocation and closed-form B compensation."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters, final_hidden, normalized_input
from local_residual_transport import cpu, forward, metric, pack
from mechanisms import load_author
from probe_distance_candidate import on_device
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, gelu, initialize
from verify_levelset_audits import convert, raw_scores


def relocate(original, anchor, cfg):
    pairs=[]
    for start in range(0,768,64):
        w,a=original[:,start:start+64],anchor[:,start:start+64]
        qe,qa=[torch.linalg.qr(x,mode='reduced').Q for x in [w,a]]
        residual=qa-qe@(qe.T@qa);residual-=qe@(qe.T@residual)
        normal,s,vh=torch.linalg.svd(residual,full_matrices=False)
        assert float(s.min())>1e-10 and float(s.max())<1-1e-10
        q=(qa@vh.T-normal*s)/(1-s.square()).sqrt()
        assert float((q.T@normal).abs().max())<1e-9
        assert float((q.T@q-torch.eye(64)).abs().max())<1e-9
        pairs.append((q,normal,a))
    def at(angle):
        parts=[]
        for q,n,a in pairs:
            basis=q*np.cos(angle)-n*np.sin(angle)
            u,s,vh=torch.linalg.svd(basis.T@a,full_matrices=False)
            assert float(s.min())>2*np.sqrt(cfg['ridge'])
            z=(s+(s.square()-4*cfg['ridge']).sqrt())/2
            parts.append((basis@(u*z)@vh).float().double())
        return torch.cat(parts,1)
    goal=cfg['parameter_budget']/np.sqrt(2)*(1-cfg['distance_reserve'])
    distance=lambda w:float((w-original).norm()/original.norm())
    low,high=0.,np.pi/4
    assert distance(at(low))<goal<distance(at(high)),'Fixed weight-only bracket fails; no alternate path.'
    for _ in range(cfg['bisection_steps']):
        mid=(low+high)/2
        if distance(at(mid))<=goal:low=mid
        else:high=mid
    return at(low),dict(angle=float(low),O_relative_distance=distance(at(low)),O_budget=goal)


@torch.no_grad()
def run(protocol):
    start=time.monotonic();cfg=json.loads(Path(protocol).read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for p,s in cfg['source_sha256'].items():assert digest(p)==s,p
    assert '3090' in torch.cuda.get_device_name();save_json(out/'protocol.json',cfg)
    anchor,expert,head,pins=checked_models(cfg['target'])
    p=extended_parameters(expert,head);a=extended_parameters(anchor,head)
    O,geometry=relocate(p['O'],a['O'],cfg)
    author,_=load_author(Path(cfg['author_path']),True);errors=[];np_errors=[];equations=[]
    for i in range(0,768,64):
        x=O[:,i:i+64].float();t=author['solve_linear_transformation'](x,a['O'][:,i:i+64].float())
        errors.append(float((x@t-x).abs().max()))
        xn,an=x.double().numpy(),a['O'][:,i:i+64].numpy()
        gram=xn.T@xn+cfg['ridge']*np.eye(64)
        independent=xn@np.linalg.lstsq(gram,xn.T@an,rcond=None)[0]
        np_errors.append(float(abs(independent-(x@t).double().numpy()).max()))
        equations.append(float(abs(xn.T@(an-xn)-cfg['ridge']*np.eye(64)).max()))
    geometry.update(author_O_fixedpoint_maximum_weight_error=max(errors),NumPy_author_output_error=max(np_errors),
        quantized_ridge_fixedpoint_equation_error=max(equations),source_sha256=digest(__file__),checkpoint_sha256=pins)
    save_json(out/'geometry.json',geometry);assert max(errors)<=cfg['author_weight_tolerance']
    assert max(np_errors)<=cfg['author_weight_tolerance']
    torch.save(O,out/'O_before_images.pt')
    train=dataset(cfg['target'],'data',expert.val_preprocess,True)
    excluded=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    split=unused(train,excluded,cfg|dict(confirmation_n=cfg['fit_n']+cfg['check_n'],confirmation_seed=cfg['construction_seed']))
    split.update(used_for_optimization=True,role='Source construction only; not independent confirmation.')
    ids=dict(fit=split['indices'][:cfg['fit_n']],check=split['indices'][cfg['fit_n']:])
    save_json(out/'construction_split.json',split);save_json(out/'training_indices.json',ids)
    cache={k:capture_attention(expert,train,v) for k,v in ids.items()}
    pg=on_device(p);f=on_device(cache['fit']);r=f['r']+f['o_input']@(O-p['O']).cuda().T
    z=gelu(normalized_input(r,pg)@pg['A'].T+pg['a'])
    target=final_hidden(f['r'],pg)-r-z@pg['B'].T-pg['b']
    coefficient,_,rank,singular=np.linalg.lstsq(z.cpu().numpy(),target.cpu().numpy(),rcond=cfg['regression_rcond'])
    q=p|dict(O=O,B=(p['B']+torch.from_numpy(coefficient).T).float().double())
    distances={k:float((q[k]-p[k]).norm()/p[k].norm()) for k in ['O','B']}
    distance=sum(v*v for v in distances.values())**.5
    residual=float(np.linalg.norm(z.cpu().numpy()@coefficient-target.cpu().numpy())/np.linalg.norm(target.cpu().numpy()))
    report=dict(status='source_parameter_budget_failed',parameter_distance=distance,per_matrix_distance=distances,
        fit_rank=int(rank),fit_relative_hidden_residual=residual,geometry=geometry,source_feasible=False,
        gradients=0,expert_updates=0,source_sha256=digest(__file__),protocol_sha256=digest(protocol),
        partner_used=False,test_used=False,confirmation_used=False,new_release_count=0,groups={})
    point=torch.load(cfg['probe_reference']['path'],map_location='cpu');assert digest(cfg['probe_reference']['path'])==cfg['probe_reference']['sha256']
    vo,vb=point['basis_O']['vh'],point['basis_B']['vh'];assert len(vo)==len(vb)==32
    torch.save(dict(p=p,q=q,vo=vo,vb=vb,cfg=cfg,indices=ids),out/'diagnostic.pt')
    report['diagnostic_sha256']=digest(out/'diagnostic.pt')
    if distance<=cfg['parameter_budget']:
        pg,qg=on_device(p),on_device(q);qn=convert(q);np_error=0.;full_error=0.
        released=copy.deepcopy(expert)
        for k,suffix in [('O','attn.out_proj.weight'),('B','mlp.c_proj.weight')]:
            released.state_dict()['model.visual.transformer.resblocks.11.'+suffix].copy_(q[k].float())
        for name,c in cache.items():
            data=pack(c,pg,vo.cuda(),vb.cuda(),cfg);before,after=forward(data,pg,pg),forward(data,pg,qg)
            rr=data['r']+data['H']@(qg['O']-pg['O']).T;cn=rr.cpu().numpy();center=cn-cn.mean(1,keepdims=True)
            pre=(center/np.sqrt((center**2).mean(1,keepdims=True)+p['eps2'])*qn['gamma2']+qn['beta2'])@qn['A'].T+qn['a']
            h=data['rounding'].cpu().numpy()+cn+(pre*expit(1.702*pre)+data['offset'].cpu().numpy())@qn['B'].T+qn['b']
            error=float(abs(raw_scores(h,qn)-after.cpu().numpy()).max());assert error<1e-8;np_error=max(np_error,error)
            actual=[];released.cuda();head.cuda()
            for images,_ in make_loader(train,ids[name],64):actual.append(head(released(images.cuda())).cpu())
            released.cpu();native=head(c['actual'].float().cuda()).cpu().double();head.cpu();actual=torch.cat(actual).double()
            n=len(ids[name]);error=max(float((native-before[:n].cpu()).abs().max()),float((actual-after[:n].cpu()).abs().max()))
            full_error=max(full_error,error);assert error<=cfg['score_tolerance']
            report['groups'][name]=dict(cached=metric(before,after,n),full_model=metric(native,actual,n))
        report.update(status='source_task_budget_failed',NumPy_score_error=np_error,full_model_score_error=full_error)
        report['source_feasible']=all(max(v['cached']['worst_task_drift'],v['full_model']['worst_task_drift'])<=cfg['task_budget'] and v['cached']['changed']==v['full_model']['changed']==0 for v in report['groups'].values())
        if report['source_feasible']:report['status']='source_passed_independent_confirmation_required'
    report.update(wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),closed=not report['source_feasible'])
    save_json(out/'results.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True)
    args=parser.parse_args();initialize();run(args.protocol)
