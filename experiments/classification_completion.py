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


def feature_space(p,rtol=1e-10):
    """Hidden directions whose projected feature change stays in the class subspace."""
    contrast=p["head"]-p["head"].mean(0)
    uc,sc,_=torch.linalg.svd(contrast.T,full_matrices=False)
    rank=int((sc>sc[0]*rtol).sum());assert rank>0
    uc=uc[:,:rank]
    linear=p["gamma"][:,None]*p["projection"];linear=linear-linear.mean(0)
    ul,sl,vl=torch.linalg.svd(linear,full_matrices=False)
    assert int((sl>sl[0]*rtol).sum())==linear.shape[1]
    s=(ul/sl[None,:])@(vl@uc)
    q,r=torch.linalg.qr(s,mode="reduced")
    error=float((s.T@linear-uc.T).abs().max());assert error<1e-8
    bias=p["beta"]@p["projection"]
    return {"L":linear,"c":bias,"U":uc,"S":s,"Q":q,"R":r}, {"rank":rank,"classes":len(contrast),"feature_map_condition":float(sl[0]/sl[-1]),"pure_feature_class_change_error":error}


def raw_score_target(h0,teacher,space,epsilon,iterations=16):
    linear,bias,u,s,r=[space[k] for k in ["L","c","U","S","R"]]
    te=teacher@linear+sigma(teacher,epsilon)*bias
    a=(te/te.norm(dim=1,keepdim=True))@u
    complement=(1.-a.square().sum(1,keepdim=True)).clamp_min(0).sqrt();assert complement.min()>1e-8
    g0=(h0@linear)@u;aux=h0@linear-g0@u.T
    cg=bias@u;ca=bias-cg@u.T
    centered_s=s-s.mean(0)
    bound=((a@centered_s.T).norm(dim=1,keepdim=True)*ca.norm()/complement+(cg@centered_s.T).norm())/(h0.shape[1]**.5)
    assert bound.max()<1.-1e-8,"Raw-score variance map is not certified contractive"
    rho=sigma(h0,epsilon)
    for _ in range(iterations):
        length=(aux+rho*ca).norm(dim=1,keepdim=True)/complement
        dg=length*a-g0-rho*cg
        final=h0+dg@s.T
        rho=sigma(final,epsilon)
    # Re-evaluate the last output at the final scalar, so fixed-point error is explicit.
    length=(aux+rho*ca).norm(dim=1,keepdim=True)/complement
    dg=length*a-g0-rho*cg;final=h0+dg@s.T
    return final,dg@r.T,{"contraction_bound_maximum":float(bound.max().detach()),
                        "fixed_point_sigma_error":float((sigma(final,epsilon)-rho).abs().max().detach())}


def amplitude_target(h0,teacher,space,p,margin_fraction,steps=40):
    """Solve a positive class amplitude inside the original raw-margin tolerance."""
    q,v,w,a=[space[k] for k in ["Q","v","w","a"]]
    linear=p["gamma"][:,None]*p["projection"];linear=linear-linear.mean(0)
    bias=p["beta"]@p["projection"]
    se=sigma(teacher,p["eps"]);te=teacher@linear+se*bias;teacher_norm=te.norm(dim=1,keepdim=True)
    aux=h0-(h0@q)@q.T;task=(teacher@q)@q.T+se*w
    auxc=aux-aux.mean(1,keepdim=True);tc=task-task.mean(1,keepdim=True);wc=w-w.mean()
    base=auxc.square().mean(1,keepdim=True)+p["eps"]
    task_norm=tc.square().mean(1,keepdim=True);cross=(tc*wc).mean(1,keepdim=True)
    asym_rho=((cross.square()+a*task_norm).sqrt()-cross)/a
    asym_t=task@linear+asym_rho*(bias-w@linear)
    asym_alpha=teacher_norm/asym_t.norm(dim=1,keepdim=True)
    target=1.-margin_fraction/2.
    assert asym_alpha.min()>target+1e-8,"Specified raw-margin level cannot be certified in this class plane"
    def family(c):
        disc=(c.square()*cross.square()+a*(base+c.square()*task_norm)).sqrt()
        rho=(base+c.square()*task_norm)/(disc+c*cross)
        h=aux+c*task-rho*w;t=h@linear+rho*bias
        alpha=c*teacher_norm/t.norm(dim=1,keepdim=True)
        dr=(c*(cross.square()+a*task_norm)/disc-cross)/a
        dt=task@linear+dr*(bias-w@linear)
        derivative=teacher_norm/t.norm(dim=1,keepdim=True)*(1.-c*(t*dt).sum(1,keepdim=True)/t.square().sum(1,keepdim=True))
        return h,alpha,derivative,rho
    with torch.no_grad():
        low=torch.zeros_like(teacher_norm);high=torch.ones_like(low)
        for _ in range(16):
            insufficient=family(high)[1]<target
            if not insufficient.any():break
            high=torch.where(insufficient,2*high,high)
        assert (family(high)[1]>=target).all(),"Positive amplitude bracket failed"
        for _ in range(steps):
            mid=(low+high)/2;below=family(mid)[1]<target
            low=torch.where(below,mid,low);high=torch.where(below,high,mid)
        numerical=(low+high)/2
    _,alpha,derivative,_=family(numerical)
    assert derivative.abs().min()>1e-8,"Ill-conditioned amplitude root"
    # Attach the implicit first derivative; the numerical root is not a learned per-image parameter.
    c=numerical-(alpha-target)/derivative.detach()
    h,actual,_,rho=family(c)
    correction=(c*teacher-h0)@q+(c*se-rho)*v
    assert (h-(h0+correction@q.T)).abs().max()<1e-8
    return h,correction,{"target_raw_gap_scale":target,"amplitude_minimum":float(c.min().detach()),
                         "amplitude_maximum":float(c.max().detach()),"asymptotic_gap_scale_minimum":float(asym_alpha.min().detach()),
                         "amplitude_root_error":float((actual-target).abs().max().detach()),"root_derivative_minimum":float(derivative.abs().min().detach())}


@torch.no_grad()
def analyze(target,cfg):
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    root=Path(cfg["cache_reference"]);assert digest(root/"calibration.pt")==cfg["calibration_sha256"]
    cache=torch.load(root/"calibration.pt",map_location="cpu")
    _,expert,head,hashes=checked_models(target);p=on_device(extended_parameters(expert,head))
    raw_mode=cfg.get("preserve_centered_raw_scores",False);band_mode=cfg.get("preserve_raw_margin_band",False)
    space,geometry=(feature_space if raw_mode else readout)(p,cfg["readout_relative_tolerance"])
    basis=on_device(cache["basis_O"]);co=-cfg["parameter_budget"]*torch.diag(basis["s"])/basis["s"].norm()
    eo=p["O"].norm()*basis["u"]@co@basis["vh"]
    records=[]
    for group in ["native","scale"]:
        data=on_device(cache[group]);h=data["h"]
        direct=angular_scores(scores(h,p))
        if raw_mode:
            t=h@space["L"]+sigma(h,p["eps"])*space["c"]
            raw_scores=t/t.norm(dim=1,keepdim=True)@p["head"].T
            analytic=angular_scores(raw_scores);numer=None
        else:
            numer=h@space["M"]+sigma(h,p["eps"])*space["b"]
            analytic=numer/numer.norm(dim=1,keepdim=True)
        identity=float((direct-analytic).abs().max());assert identity<1e-10
        rounding=h-final_hidden(data["r"],p)
        h0=rounding+final_hidden(data["r"]+data["o_input"]@eo.T,p)
        if band_mode:
            ideal,target_q,root_info=amplitude_target(h0,h,space,p,cfg["margin_fraction"],cfg["amplitude_bisection_steps"])
            before=scores(h,p);after=scores(ideal,p);factor=root_info["target_raw_gap_scale"]
            numerator_error=float(((after-after.mean(1,keepdim=True))-factor*(before-before.mean(1,keepdim=True))).abs().max())
            variance_error=root_info["amplitude_root_error"]
        elif raw_mode:
            ideal,target_q,root_info=raw_score_target(h0,h,space,p["eps"],cfg["fixed_point_iterations"])
            before=scores(h,p);after=scores(ideal,p)
            numerator_error=float(((after-after.mean(1,keepdim=True))-(before-before.mean(1,keepdim=True))).abs().max())
            variance_error=root_info["fixed_point_sigma_error"]
        else:
            ideal,target_q,rho=ideal_output(h0,h,space,p["eps"]);root_info={}
            changed_numer=ideal@space["M"]+sigma(ideal,p["eps"])*space["b"]
            numerator_error=float((changed_numer-numer).norm()/numer.norm())
            variance_error=float((sigma(ideal,p["eps"])-rho).abs().max())
        direction_error=float((angular_scores(scores(ideal,p))-direct).abs().max())
        assert numerator_error<1e-10 and direction_error<1e-10 and variance_error<1e-10
        records.append({"group":group,"n":len(h),"direction_identity_maximum_error":identity,
                        "ideal_numerator_relative_error":numerator_error,"numerator_error_measure":"centered_score_maximum_absolute_error" if band_mode or raw_mode else "numerator_relative_Frobenius_error","ideal_direction_maximum_error":direction_error,
                        "ideal_sigma_error":variance_error,"ideal_raw_margin_drift":float(signed_residual(scores(h,p),scores(ideal,p)).abs().max()),
                        "ideal_hidden_relative_change":float((ideal-h).norm()/h.norm()),"required_Q_correction_rms":float(target_q.square().sum(1).mean().sqrt()),**root_info})
    artifact=out/"readout.pt";torch.save(space,artifact)
    report={"target":target,"status":"controlled_raw_margin_level_set_verified_only" if band_mode else "exact_raw_score_level_set_verified_only" if raw_mode else "exact_level_set_identity_verified_only","geometry":geometry,"records":records,
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
