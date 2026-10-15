"""Evaluate the sole confirmed temperature release under unchanged TA/author AGA."""

import argparse
import copy
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from confirm_temperature_compensation import edits
from covariance_noise import run
from subspace_probe import make_loader, save_json


@torch.no_grad()
def construct(original,anchor,head,training,indices,cfg,target):
    root=Path(cfg['publication_point']);meta=json.loads((root/'results.json').read_text())
    valid=json.loads((root/'independent_validation.json').read_text());confirmed=json.loads((root/'confirmation_validation.json').read_text())
    assert valid['status']=='passed_task_calibration' and confirmed['status']=='passed' and confirmed['confirmation_passed']
    assert digest(root/'training_point.pt')==cfg['training_point_sha256']==meta['training_point_sha256']==confirmed['training_point_sha256']
    assert digest(root/'release_edits.pt')==cfg['release_edits_sha256']==confirmed['release_sha256']
    point=torch.load(root/'training_point.pt',map_location='cpu');expected=edits(original,point);changes=torch.load(root/'release_edits.pt',map_location='cpu')
    assert set(expected)==set(changes)
    released=copy.deepcopy(original)
    for key,value in changes.items():
        assert torch.equal(value,expected[key]),key
        released.state_dict()[key].copy_(value)
    prefix='model.visual.transformer.resblocks.11.';w=original.state_dict()[prefix+'attn.in_proj_weight'].double();b=original.state_dict()[prefix+'attn.in_proj_bias'].double();d=w.shape[1]
    dw=changes[prefix+'attn.in_proj_weight'].double()-w;db=changes[prefix+'attn.in_proj_bias'].double()-b
    parts=[float((dw[i:i+d].square().sum()+db[i:i+d].square().sum())/(w[i:i+d].square().sum()+b[i:i+d].square().sum())) for i in [0,d]]
    original_b=original.state_dict()[prefix+'mlp.c_proj.weight'].double()
    parts.append(float(((changes[prefix+'mlp.c_proj.weight'].double()-original_b).norm()/original_b.norm()).square()))
    distance=sum(parts)**.5
    assert abs(distance-meta['selected']['distance'])<1e-6,'FP32 export distance differs from the bounded coefficients'
    actual={k for k,v in released.state_dict().items() if not torch.equal(v,original.state_dict()[k])};assert actual==set(changes)
    geometry=dict(method='fixed_same_sign_QK_temperature_finite_margin_distillation',source_parameter_distance=meta['selected']['distance'],export_parameter_distance=distance,
                  source_task_drift=meta['selected']['worst_margin_drift'],task_confirmation=confirmed,gain=meta['gain'],
                  point_sha256=meta['training_point_sha256'],construction_GPU_seconds=meta['wall_seconds'],original_checkpoint_updated=False,
                  proxy_used_for_construction=False,partner_used_for_construction=False,test_used_for_construction=False,
                  symbolic_parameter_budget=cfg['parameter_budget'],FP32_export_distance_error=distance-meta['selected']['distance'])
    return released,changes,{'reference':{'path':str(root/'training_point.pt'),'sha256':meta['training_point_sha256']}},geometry


def holdout(original,training,indices,cfg,target):
    split=json.loads((Path(cfg['publication_point'])/'confirmation_split.json').read_text())
    return make_loader(training,split['indices']),dict(**split,target=target,role='Replay sole pre-frozen target-training confirmation; no selection')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--target',required=True);ap.add_argument('--partner',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());run(args,constructor=construct,holdout_loader=holdout)
    out=Path(args.output);meta=json.loads((out/'execution.json').read_text());meta.update(evaluation_optimizer_steps=0,construction_optimizer_steps=0,
        construction_spectral_distillation_solves=1,construction_nonlinear_correction_updates=0,original_expert_parameters_updated=False,
        training_point_sha256=cfg['training_point_sha256'],primary_entrypoint_sha256=digest(__file__),test_parameter_selection=False)
    save_json(out/'execution.json',meta)
