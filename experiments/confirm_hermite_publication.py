"""Sole fresh target-training confirmation of the frozen Hermite publication."""

import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from angular_distance import signed_residual
from boundary_game import risk_first
from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import CrossProblem,capture_attention,normalized_input
from hermite_publication import Publication
from probe_margin_distance import scores
from subspace_probe import make_loader,save_json
from task_probe_distance import checked_models,gelu,initialize
from verify_downstream_first import convert
from verify_hidden_hermite import mlp_jet
from verify_levelset_audits import raw_scores,residual


@torch.no_grad()
def confirm(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());valid=json.loads((out/'independent_validation.json').read_text())
    assert rep['confirmation_eligible'] and valid['status']=='passed_source_task_constraints' and not rep['confirmation_passed']
    assert digest(out/'training_point.pt')==rep['training_point_sha256']==valid['training_point_sha256']
    point=torch.load(out/'training_point.pt',map_location='cpu');p=Publication(cfg);anchor,expert,head,pins=checked_models(cfg['target']);released=copy.deepcopy(expert)
    prefix='model.visual.transformer.resblocks.11.';keys={'O':prefix+'attn.out_proj.weight','B':prefix+'mlp.c_proj.weight'};delta={}
    for name,key in keys.items():
        original=expert.state_dict()[key];assert torch.equal(original.double(),p.p[name].cpu())
        changed=(original.double()+point['release_deltas'][name]).float();released.state_dict()[key].copy_(changed);delta[name]=(changed.double()-original.double()).cuda()
    distance=float(sum((delta[k].norm()/p.p[k].norm())**2 for k in delta)**.5)
    def forward(data):
        r=data['r']+data['o_input']@delta['O'].T;z=gelu(normalized_input(r,p.p)@p.p['A'].T+p.p['a'])+data['b_offset']
        return scores(data['rounding']+r+z@(p.p['B']+delta['B']).T+p.p['b'],p.p)
    source_drift=float(signed_residual(p.teacher,forward(p.data)).abs().max());assert distance<=.5 and source_drift<=cfg['construction_task_reserve'],'FP32 source reserve failed; stop without repair'
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);split_file=out/'confirmation_split.json'
    split=json.loads(split_file.read_text()) if split_file.exists() else unused(training,set(p.ids),cfg)
    assert split['seed']==cfg['confirmation_seed'] and not set(split['indices'])&set(p.ids);save_json(split_file,split)
    cache=capture_attention(expert,training,split['indices']);cache,ids,n=risk_first(cache,split['indices'],cfg['contexts_per_class'],p.p)
    cp=CrossProblem.__new__(CrossProblem);cp.cfg=dict(probe_contexts=n,finite_step=cfg['finite_step'],full_B_completion=True);cp.r=cfg['probe_rank'];cp.a_size=cp.r**2
    cp.b_rows=p.p['B'].shape[0];cp.b_width=p.p['B'].shape[1];cp.na=p.chart.na;cp.nb=p.chart.nb;cp.output_basis=None;cp.basis={'A':p.chart.basis,'B':{k:v.cuda() for k,v in p.saved['bases']['B'].items()}}
    data=cp.prepare(cache,p.p,True);before=data['teacher'];after=forward(data);drift=float(signed_residual(before,after).abs().max())
    pn=convert(p.p);dn=convert(data);do,db=delta['O'].cpu().numpy(),delta['B'].cpu().numpy();r=dn['r']+dn['o_input']@do.T;z,_=mlp_jet(r,np.zeros_like(r),pn);z+=dn['b_offset']
    independent=raw_scores(dn['rounding']+r+z@(pn['B']+db).T+pn['b'],pn);np.testing.assert_allclose(independent,after.cpu().numpy(),atol=1e-8,rtol=1e-9)
    independent_drift=float(np.abs(residual(before.cpu().numpy(),independent)).max());pred0=before[:len(ids)].argmax(1).cpu();pred1=after[:len(ids)].argmax(1).cpu();labels=torch.tensor([int(training[i][1]) for i in ids]);checks={}
    for name,model,expected in [('original',expert,pred0),('released',released,pred1)]:
        values=[];model.cuda().eval();head.cuda().eval()
        for images,_ in make_loader(training,ids,64):values.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu();head.cpu();actual=torch.cat(values);assert torch.equal(actual,expected),name
        checks[name]=dict(n=len(ids),correct=int((actual==labels).sum()),all_predictions_match=True)
    passed=bool(drift<=.5 and distance<=.5 and torch.equal(pred0,pred1));torch.save(dict(cache=cache,indices=ids,labels=labels,teacher=before.cpu(),released_scores=after.cpu(),release_deltas={k:v.cpu() for k,v in delta.items()}),out/'confirmation.pt')
    result=dict(status='passed' if passed else 'failed_task_confirmation',confirmation_passed=passed,natural_n=len(ids),task_inputs=len(before),parameter_distance=distance,
        source_FP32_worst_drift=source_drift,worst_margin_drift=drift,numpy_worst_margin_drift=independent_drift,natural_changes=int((pred0!=pred1).sum()),
        original_correct=int((pred0==labels).sum()),released_correct=int((pred1==labels).sum()),full_model_replay=checks,new_model_forwards=3*len(ids),
        confirmation_sha256=digest(out/'confirmation.pt'),training_point_sha256=rep['training_point_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        peer_used=False,test_used=False,confirmation_redrawn=False,point_repaired=False)
    save_json(out/'source_independent_validation.json',valid);rep.update(checkpoint_sha256=pins,confirmation=result,confirmation_passed=passed,new_release_weights=int(passed),
        status='passed_Hermite_publication_confirmation' if passed else 'closed_Hermite_publication_confirmation_failure')
    if passed:
        torch.save(dict(release_deltas={k:v.cpu() for k,v in delta.items()},confirmation_indices=ids,training_point_sha256=rep['training_point_sha256'],cfg=cfg),out/'candidate.pt')
        rep['candidate_sha256']=digest(out/'candidate.pt');result['candidate_sha256']=rep['candidate_sha256']
    save_json(out/'independent_validation.json',dict(status='passed' if passed else 'confirmed_constraint_failure',source_calibration=valid,confirmation=result,new_model_forwards=3*len(ids),new_release_weights=int(passed)))
    save_json(out/'confirmation_validation.json',result);save_json(out/'results.json',rep);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();initialize();confirm(json.loads(Path(a.protocol).read_text()))
