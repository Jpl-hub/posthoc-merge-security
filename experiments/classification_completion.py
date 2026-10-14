"""Exact normalized-score level sets for task-only conditional distillation."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores,signed_residual
from clip_benchmark import digest
from cross_module_distance import extended_parameters,final_hidden
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import checked_models


def readout(p,rtol=1e-10):
    contrast=p["head"]-p["head"].mean(0)
    image=p["projection"]@contrast.T
    m=p["gamma"][:,None]*image;m=m-m.mean(0)
    bias=p["beta"]@image
    u,s,_=torch.linalg.svd(m,full_matrices=False)
    rank=int((s>s[0]*rtol).sum());assert rank>0
    q=u[:,:rank];g=q.T@m
    condition=float((s[0]/s[rank-1])**2);assert condition<1e12
    v=torch.linalg.solve(g@g.T,g@bias);w=v@q.T;wc=w-w.mean()
    row_error=float((v@g-bias).norm()/bias.norm().clamp_min(1e-20))
    a=1.-wc.square().mean();assert row_error<1e-8 and a>1e-8
    return {"M":m,"b":bias,"Q":q,"v":v,"w":w,"a":a}, {"rank":rank,"classes":len(contrast),"G_squared_condition":condition,"bias_row_space_relative_error":row_error,"quadratic_a":float(a)}


def sigma(h,epsilon):
    return ((h-h.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)+epsilon).sqrt()


def ideal_output(h0,teacher,space,epsilon):
    q,v,w,a=[space[k] for k in ["Q","v","w","a"]]
    se=sigma(teacher,epsilon)
    projected=(teacher-h0)@q
    star=h0+projected@q.T+se*w
    centered=star-star.mean(-1,keepdim=True);wc=w-w.mean()
    b0=(centered*wc).mean(-1,keepdim=True);c0=centered.square().mean(-1,keepdim=True)+epsilon
    rho=c0/((b0.square()+a*c0).sqrt()+b0)
    target=projected+(se-rho)*v
    return h0+target@q.T,target,rho


@torch.no_grad()
def analyze(target,cfg):
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    root=Path(cfg["cache_reference"]);assert digest(root/"calibration.pt")==cfg["calibration_sha256"]
    cache=torch.load(root/"calibration.pt",map_location="cpu")
    _,expert,head,hashes=checked_models(target);p=on_device(extended_parameters(expert,head))
    space,geometry=readout(p,cfg["readout_relative_tolerance"])
    basis=on_device(cache["basis_O"]);co=-cfg["parameter_budget"]*torch.diag(basis["s"])/basis["s"].norm()
    eo=p["O"].norm()*basis["u"]@co@basis["vh"]
    records=[]
    for group in ["native","scale"]:
        data=on_device(cache[group]);h=data["h"]
        numer=h@space["M"]+sigma(h,p["eps"])*space["b"]
        direct=angular_scores(scores(h,p));analytic=numer/numer.norm(dim=1,keepdim=True)
        identity=float((direct-analytic).abs().max());assert identity<1e-10
        rounding=h-final_hidden(data["r"],p)
        h0=rounding+final_hidden(data["r"]+data["o_input"]@eo.T,p)
        ideal,target_q,rho=ideal_output(h0,h,space,p["eps"])
        changed_numer=ideal@space["M"]+sigma(ideal,p["eps"])*space["b"]
        numerator_error=float((changed_numer-numer).norm()/numer.norm())
        direction_error=float((angular_scores(scores(ideal,p))-direct).abs().max())
        variance_error=float((sigma(ideal,p["eps"])-rho).abs().max())
        assert numerator_error<1e-10 and direction_error<1e-10 and variance_error<1e-10
        records.append({"group":group,"n":len(h),"direction_identity_maximum_error":identity,
                        "ideal_numerator_relative_error":numerator_error,"ideal_direction_maximum_error":direction_error,
                        "ideal_sigma_error":variance_error,"ideal_raw_margin_drift":float(signed_residual(scores(h,p),scores(ideal,p)).abs().max()),
                        "ideal_hidden_relative_change":float((ideal-h).norm()/h.norm()),"required_Q_correction_rms":float(target_q.square().sum(1).mean().sqrt())})
    artifact=out/"readout.pt";torch.save(space,artifact)
    report={"target":target,"status":"exact_level_set_identity_verified_only","geometry":geometry,"records":records,
            "checkpoint_sha256":hashes,"readout_sha256":digest(artifact),"calibration_sha256":cfg["calibration_sha256"],
            "source_sha256":digest(__file__),"protocol_sha256":digest(cfg["protocol_path"]),
            "ideal_O_relative_distance":cfg["parameter_budget"],"uniform_B_completion_constructed":False,
            "new_release_weights":0,"optimizer_updates":0,"new_model_forwards":0,"test_used":False,"partner_used":False,
            "wall_seconds":time.monotonic()-start}
    save_json(out/"results.json",report);save_json(out/"protocol.json",cfg);print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",required=True);parser.add_argument("--protocol",required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg["protocol_path"]=args.protocol
    for file,sha in cfg["source_sha256"].items():assert digest(file)==sha,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    analyze(args.target,cfg)
