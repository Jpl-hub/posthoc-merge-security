"""Independent NumPy reconstruction of Attention-to-MLP publication diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_probe_candidate import array, features


def ln(value,gamma,beta,epsilon):
    centered=value-value.mean(-1,keepdims=True)
    return centered/np.sqrt(np.mean(centered**2,-1,keepdims=True)+epsilon)*gamma+beta


def ln_jvp(value,direction,gamma,epsilon):
    centered=value-value.mean(-1,keepdims=True);dc=direction-direction.mean(-1,keepdims=True)
    sd=np.sqrt(np.mean(centered**2,-1,keepdims=True)+epsilon)
    return (dc/sd-centered*np.mean(centered*dc,-1,keepdims=True)/sd**3)*gamma


def verify(output):
    output=Path(output);report=json.loads((output/"direction.json").read_text());meta=json.loads((output/"mechanism.json").read_text())
    assert digest(output/"direction.pt")==report["direction_sha256"]
    assert digest(output/"calibration.pt")==meta["calibration_sha256"]
    saved=torch.load(output/"direction.pt",map_location="cpu");cal=torch.load(output/"calibration.pt",map_location="cpu");cfg=saved["cfg"]
    anchor,expert,head,hashes=checked_models(report["target"]);assert hashes==report["checkpoint_sha256"]
    raw=extended_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in raw.items()}
    rank=cfg["probe_rank"];size=rank**2
    uo,vo=[array(cal["basis_O"][k]) for k in ["u","vh"]];vb=array(cal["basis_B"]["vh"])
    original=extended_parameters(anchor,head);task=p["O"]-array(original["O"])
    np.testing.assert_allclose(task@vo.T,uo*array(cal["basis_O"]["s"]),atol=1e-9,rtol=1e-7)
    for basis in [uo.T,vo,vb]:np.testing.assert_allclose(basis@basis.T,np.eye(rank),atol=1e-9)
    normo,normb=np.linalg.norm(p["O"]),np.linalg.norm(p["B"])
    theta,direction=[array(saved[k]) for k in ["theta","direction"]]
    def delta(coeff):return normo*uo@coeff[:size].reshape(rank,rank)@vo,normb*coeff[size:].reshape(p["B"].shape[0],rank)@vb
    eo,eb=delta(theta);do,db=delta(direction)
    i=next((j for j,row in enumerate(report["finite_steps"]) if row["feasible"]),0)
    finite=report["finite_steps"][i];trial=array(saved["finite_thetas"][i]);to,tb=delta(trial)
    def hidden(r):
        pre=ln(r,p["gamma2"],p["beta2"],p["eps2"])@p["A"].T+p["a"]
        return pre*expit(1.702*pre)
    realr=np.concatenate([array(cal[g]["r"]) for g in ["native","scale"]]);realz=np.concatenate([array(cal[g]["o_input"]) for g in ["native","scale"]])
    teacher=np.concatenate([array(cal[g]["h"]) for g in ["native","scale"]])
    rounding=teacher-realr-hidden(realr)@p["B"].T-p["b"]
    q0=hidden(realr)@vb.T;ridge=cfg["ridge_relative_floor"]*np.linalg.eigvalsh(q0.T@q0)[-1]
    assert abs(ridge-saved["ridge"])<1e-9
    conditional=[]
    for coeff in [theta,trial]:
        o,b=delta(coeff);r=realr+realz@o.T;z=hidden(r);q=z@vb.T
        residual=teacher-rounding-r-z@p["B"].T-p["b"]
        kb=np.linalg.solve(q.T@q+ridge*np.eye(rank),q.T@residual).T
        error=float(np.max(np.abs(kb/normb-coeff[size:].reshape(p["B"].shape[0],rank))));assert error<1e-7;conditional.append(error)
    residuals,jvps,finite_max,cache_checks=[],[],[],[]
    for g in ["native","scale"]:
        z,r,h=[array(cal[g][k]) for k in ["o_input","r","h"]];n=cfg["probe_contexts"];count=2*rank
        c=h-r-hidden(r)@p["B"].T-p["b"]
        aerr=float(np.max(np.abs(z@p["O"].T+p["ob"]-array(cal[g]["attn_output"]))))
        actual=array(cal[g]["actual"]);actual=actual/np.linalg.norm(actual,axis=1,keepdims=True)@p["head"].T
        serr=float(np.max(np.abs(features(h,expert.model.visual)@p["head"].T-actual)))
        assert aerr<5e-4 and serr<5e-4;cache_checks.append({"group":g,"n":len(r),"attention_error":aerr,"zero_score_error":serr})
        so=(cfg["finite_step"]*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((z@vo.T)**2,0))[None,:,None]*vo[None,:,:]).reshape(count,-1)
        sb=(cfg["finite_step"]*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((hidden(r)@vb.T)**2,0))[None,:,None]*vb[None,:,:]).reshape(count,-1)
        rr=np.concatenate([r,(r[:n,None,:]+(so@p["O"].T)[None,:,:]).reshape(-1,r.shape[1]),np.repeat(r[:n],count,axis=0)])
        zz=np.concatenate([z,(z[:n,None,:]+so[None,:,:]).reshape(-1,z.shape[1]),np.repeat(z[:n],count,axis=0)])
        cc=np.concatenate([c,np.repeat(c[:n],count,axis=0),np.repeat(c[:n],count,axis=0)])
        off=np.concatenate([np.zeros((len(r)+n*count,p["B"].shape[1])),np.tile(sb,(n,1))])
        def full(o,b):
            r1=rr+zz@o.T;return cc+r1+(hidden(r1)+off)@(p["B"]+b).T+p["b"]
        h0,h1=full(np.zeros_like(eo),np.zeros_like(eb)),full(eo,eb)
        r1=rr+zz@eo.T;dr=zz@do.T
        normalized=ln(r1,p["gamma2"],p["beta2"],p["eps2"]);dx=ln_jvp(r1,dr,p["gamma2"],p["eps2"])
        pre=normalized@p["A"].T+p["a"];prob=expit(1.702*pre)
        dz=(prob+1.702*pre*prob*(1-prob))*(dx@p["A"].T)
        dh=dr+dz@(p["B"]+eb).T+(hidden(r1)+off)@db.T
        t=(ln(h1,p["gamma"],p["beta"],p["eps"]))@p["projection"];f=t/np.linalg.norm(t,axis=1,keepdims=True)
        dt=ln_jvp(h1,dh,p["gamma"],p["eps"])@p["projection"]
        df=(dt-f*np.sum(f*dt,axis=1,keepdims=True))/np.linalg.norm(t,axis=1,keepdims=True)
        ds=df@p["head"].T
        before=features(h0,expert.model.visual)@p["head"].T;after=f@p["head"].T
        winner=before.argmax(1);rows=np.arange(len(before));gap=before[rows,winner,None]-before;gap[rows,winner]=1.
        delta_score=after-before;res=(delta_score[rows,winner,None]-delta_score)/np.maximum(gap,1e-12);res[rows,winner]=0.
        jj=(ds[rows,winner,None]-ds)/np.maximum(gap,1e-12);jj[rows,winner]=0.
        score=features(full(to,tb),expert.model.visual)@p["head"].T-before
        finite_res=(score[rows,winner,None]-score)/np.maximum(gap,1e-12);finite_res[rows,winner]=0.
        residuals.append(res.flatten());jvps.append(jj.flatten());finite_max.append(float(np.abs(finite_res).max()))
    res,jj=np.concatenate(residuals),np.concatenate(jvps)
    np.testing.assert_allclose(res,array(saved["native_residual"]),atol=1e-7,rtol=1e-7)
    np.testing.assert_allclose(jj,array(saved["native_directional_derivative"]),atol=1e-7,rtol=1e-7)
    assert abs(max(finite_max)-finite["native_critic"])<1e-7
    result={"status":"passed","source_sha256":digest(__file__),"direction_sha256":report["direction_sha256"],
            "calibration_sha256":meta["calibration_sha256"],"conditional_coefficient_error":max(conditional),
            "native_jvp_maximum_error":float(np.abs(jj-array(saved["native_directional_derivative"])).max()),
            "native_residual_maximum_error":float(np.abs(res-array(saved["native_residual"])).max()),
            "finite_step":finite["step"],"finite_step_feasible":finite["feasible"],"finite_native_critic":max(finite_max),
            "cache_checks":cache_checks,"new_model_forwards":0,"new_release_weights":0,"test_used":False,"partner_used":False}
    save_json(output/"independent_validation.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
