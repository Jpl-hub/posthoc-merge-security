"""Evaluate one frozen cross-module publication point with unchanged TA and AGA."""

import argparse
import copy
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from covariance_noise import run
from subspace_probe import make_loader, save_json


@torch.no_grad()
def construct(original,anchor,head,training,indices,cfg,target):
    root=Path(cfg["publication_point"]);meta=json.loads((root/"results.json").read_text())
    validation=json.loads((root/"independent_validation.json").read_text())
    assert meta["target"]==target and meta["confirmation_passed"] and validation["status"]=="passed"
    assert digest(root/"candidate.pt")==cfg["candidate_sha256"]==meta["candidate_sha256"]
    assert validation["new_model_forwards"]>=256
    cal=torch.load(root/"candidate.pt",map_location="cpu")
    for model in [original,anchor,head]:
        for parameter in model.parameters():parameter.requires_grad_(False)
    released=copy.deepcopy(original);prefix="model.visual.transformer.resblocks.11."
    keys=cfg.get("publication_keys",{"O":prefix+"attn.out_proj.weight","B":prefix+"mlp.c_proj.weight"})
    assert set(keys)==set(cal["release_deltas"])
    edits={}
    for site,key in keys.items():
        weight=original.state_dict()[key];delta=cal["release_deltas"][site]
        assert delta.shape==weight.shape
        edits[key]=(weight.double()+delta).to(weight.dtype)
        released.state_dict()[key].copy_(edits[key])
    distance=sum(float(((edits[key].double()-original.state_dict()[key].double()).norm()/original.state_dict()[key].double().norm())**2) for key in keys.values())**.5
    assert distance<=cfg["parameter_budget"]*(1+1e-6)
    actual={key for key,value in released.state_dict().items() if not torch.equal(value,original.state_dict()[key])}
    assert actual==set(keys.values())
    reference={"path":str(root/"candidate.pt"),"sha256":meta["candidate_sha256"]}
    updates=meta.get("optimizer_updates",meta.get("accepted_updates"))
    assert updates is not None
    geometry={"method":meta["method"],"publication_point":reference,"parameter_distance":distance,
              "selected_step":meta["selected_step"],"construction_optimizer_updates":updates,
              "explicit_reverse_gradient_calls":meta["explicit_reverse_gradient_calls"],"coefficient_jvp_calls":meta.get("coefficient_jvp_calls",0),
              "analytic_task_jacobian_calls":meta.get("analytic_task_jacobian_calls",0),
              "nonlinear_correction_updates":meta.get("nonlinear_correction_updates",0),
              "construction_GPU_seconds":meta["wall_seconds"],"task_confirmation":meta["confirmation"],
              "original_checkpoint_parameters_updated":False,"future_partner_used_for_construction":False,"test_used_for_construction":False}
    return released,edits,{"reference":reference},geometry


def holdout(original,training,indices,cfg,target):
    cal=torch.load(Path(cfg["publication_point"])/"candidate.pt",map_location="cpu")
    chosen=cal["confirmation_indices"]
    return make_loader(training,chosen),{"indices":chosen,"split":"train","target":target,
        "role":"Replay of frozen point confirmation; never used to select or restart optimization","seed":cfg["confirmation_seed"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",required=True)
    parser.add_argument("--partner",required=True);parser.add_argument("--protocol",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text())
    run(args,constructor=construct,holdout_loader=holdout)
    out=Path(args.output);meta=json.loads((out/"execution.json").read_text())
    point=json.loads((Path(cfg["publication_point"])/"results.json").read_text())
    updates=point.get("optimizer_updates",point.get("accepted_updates"))
    meta.update(evaluation_optimizer_steps=0,construction_optimizer_steps=updates,
                optimizer_steps=updates,construction_nonlinear_correction_updates=point.get("nonlinear_correction_updates",0),
                construction_analytic_task_jacobian_calls=point.get("analytic_task_jacobian_calls",0),original_expert_parameters_updated=False,
                publication_parameter_optimization=True,test_parameter_selection=False,
                primary_entrypoint_sha256=digest(__file__),construction_point_sha256=point["candidate_sha256"])
    save_json(out/"execution.json",meta)
