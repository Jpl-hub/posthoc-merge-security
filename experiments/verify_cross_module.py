"""Independent NumPy reconstruction of Attention-to-MLP publication diagnostics."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from cross_module_distance import extended_parameters
from covariance_noise import dataset
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
    cache_path=output/"calibration.pt" if (output/"calibration.pt").exists() else Path(meta["cache_reference"])/"calibration.pt"
    assert digest(cache_path)==meta["calibration_sha256"]
    saved=torch.load(output/"direction.pt",map_location="cpu");cal=torch.load(cache_path,map_location="cpu");cfg=saved["cfg"]
    anchor,expert,head,hashes=checked_models(report["target"]);assert hashes==report["checkpoint_sha256"]
    raw=extended_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in raw.items()}
    rank=cfg["probe_rank"];size=rank**2
    uo,vo=[array(cal["basis_O"][k]) for k in ["u","vh"]];vb=array(cal["basis_B"]["vh"])
    original=extended_parameters(anchor,head);task=p["O"]-array(original["O"])
    np.testing.assert_allclose(task@vo.T,uo*array(cal["basis_O"]["s"]),atol=1e-9,rtol=1e-7)
    for basis in [uo.T,vo,vb]:np.testing.assert_allclose(basis@basis.T,np.eye(rank),atol=1e-9)
    normo,normb=np.linalg.norm(p["O"]),np.linalg.norm(p["B"])
    class_space=None
    if cfg.get("class_output_completion"):
        class_space=torch.load(output/'readout.pt',map_location='cpu')
        oq=array(class_space['Q']);np.testing.assert_allclose(oq.T@oq,np.eye(oq.shape[1]),atol=1e-10)
    theta,direction=[array(saved[k]) for k in ["theta","direction"]]
    def delta(coeff):
        eo=normo*uo@coeff[:size].reshape(rank,rank)@vo
        if class_space is not None:eb=normb*oq@coeff[size:].reshape(oq.shape[1],p['B'].shape[1])
        else:eb=normb*coeff[size:].reshape(p["B"].shape) if cfg.get("full_B_completion") else normb*coeff[size:].reshape(p["B"].shape[0],rank)@vb
        return eo,eb
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
    conditional=[];comparisons=[]
    for coeff in [theta,trial]:
        o,b=delta(coeff);r=realr+realz@o.T;z=hidden(r);q=z@vb.T
        residual=teacher-rounding-r-z@p["B"].T-p["b"]
        kb=np.linalg.solve(q.T@q+ridge*np.eye(rank),q.T@residual).T
        if class_space is not None:
            from verify_classification_completion import target_numpy
            linear=p['gamma'][:,None]*p['projection'];linear=linear-linear.mean(0);bias=p['beta']@p['projection']
            w=array(class_space['w']);a=1.-np.mean((w-w.mean())**2)
            base=teacher-residual
            ideal,_,_=target_numpy(base,teacher,oq,w,a,linear,bias,p['eps'],1.-cfg['margin_fraction']/2.)
            projected=(ideal-base)@oq
            full=np.linalg.solve(z@z.T+ridge*np.eye(len(z)),projected).T@z
            error=float(np.max(np.abs(full/normb-coeff[size:].reshape(oq.shape[1],p['B'].shape[1]))))
        elif cfg.get("full_B_completion"):
            full=np.linalg.solve(z@z.T+ridge*np.eye(len(z)),residual).T@z
            error=float(np.max(np.abs(full/normb-coeff[size:].reshape(p["B"].shape))))
            restricted=kb@vb
            of=float(np.linalg.norm(z@full.T-residual)**2+ridge*np.linalg.norm(full)**2)
            ore=float(np.linalg.norm(z@restricted.T-residual)**2+ridge*np.linalg.norm(restricted)**2)
            assert of<=ore*(1+1e-7)
            comparisons.append({"O_relative_distance":float(np.linalg.norm(o)/normo),"full_B_relative_distance":float(np.linalg.norm(full)/normb),
                                "restricted_B_relative_distance":float(np.linalg.norm(restricted)/normb),"full_regularized_objective":of,"restricted_regularized_objective":ore})
        else:error=float(np.max(np.abs(kb/normb-coeff[size:].reshape(p["B"].shape[0],rank))))
        assert error<1e-7;conditional.append(error)
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
    result["same_O_full_vs_restricted_B"]=comparisons
    save_json(output/"independent_validation.json",result);print(json.dumps(result),flush=True)


def verify_game(output):
    output=Path(output);report=json.loads((output/"results.json").read_text());path=output/"candidate.pt"
    assert digest(path)==report["candidate_sha256"]
    cal=torch.load(path,map_location="cpu");cfg=cal["protocol"]
    _,expert,head,hashes=checked_models(report["target"]);assert hashes==report["checkpoint_sha256"]
    pp=extended_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in pp.items()}
    rnk=cfg["probe_rank"];sz=rnk**2;theta=array(cal["theta"])
    uo,vo=[array(cal["basis_O"][k]) for k in ["u","vh"]];vb=array(cal["basis_B"]["vh"])
    eo=np.linalg.norm(p["O"])*uo@theta[:sz].reshape(rnk,rnk)@vo
    if cfg.get('class_output_completion'):
        oq=array(cal['output_basis']);np.testing.assert_allclose(oq.T@oq,np.eye(oq.shape[1]),atol=1e-10)
        eb=np.linalg.norm(p['B'])*oq@theta[sz:].reshape(oq.shape[1],p['B'].shape[1])
    else:eb=np.linalg.norm(p["B"])*theta[sz:].reshape(p["B"].shape[0],rnk)@vb
    for site,delta in [("O",eo),("B",eb)]:np.testing.assert_allclose(delta,array(cal["release_deltas"][site]),atol=1e-8,rtol=1e-7)
    distance=float(np.sqrt(np.linalg.norm(eo)**2/np.linalg.norm(p["O"])**2+np.linalg.norm(eb)**2/np.linalg.norm(p["B"])**2))
    assert distance<=cfg["parameter_budget"]*(1+1e-12) and abs(distance-report["selected"]["parameter_distance"])<1e-8
    def hidden(r):
        pre=ln(r,p["gamma2"],p["beta2"],p["eps2"])@p["A"].T+p["a"]
        return pre*expit(1.702*pre)
    r=np.concatenate([array(cal[g]["r"]) for g in ["native","scale"]]);z=np.concatenate([array(cal[g]["o_input"]) for g in ["native","scale"]])
    h=np.concatenate([array(cal[g]["h"]) for g in ["native","scale"]]);c=h-r-hidden(r)@p["B"].T-p["b"]
    q0=hidden(r)@vb.T;ridge=cfg["ridge_relative_floor"]*np.linalg.eigvalsh(q0.T@q0)[-1]
    r1=r+z@eo.T;hh=hidden(r1);q=hh@vb.T
    if cfg.get('class_output_completion'):
        from verify_classification_completion import target_numpy
        space=torch.load(output/'readout.pt',map_location='cpu')
        np.testing.assert_allclose(oq,array(space['Q']),atol=1e-12)
        linear=p['gamma'][:,None]*p['projection'];linear=linear-linear.mean(0);bias=p['beta']@p['projection']
        w=array(space['w']);a=1.-np.mean((w-w.mean())**2);base=c+r1+hh@p['B'].T+p['b']
        ideal,_,_=target_numpy(base,h,oq,w,a,linear,bias,p['eps'],1.-cfg['margin_fraction']/2.)
        kb=np.linalg.solve(hh@hh.T+ridge*np.eye(len(hh)),(ideal-base)@oq).T@hh
        coefficient_error=float(np.max(np.abs(kb/np.linalg.norm(p['B'])-theta[sz:].reshape(oq.shape[1],p['B'].shape[1]))))
    else:
        kb=np.linalg.solve(q.T@q+ridge*np.eye(rnk),q.T@(h-c-r1-hh@p["B"].T-p["b"])).T
        coefficient_error=float(np.max(np.abs(kb/np.linalg.norm(p["B"])-theta[sz:].reshape(p["B"].shape[0],rnk))))
    assert coefficient_error<1e-7
    groups={};confirmed=None
    for g in ["native","scale","confirmation"]:
        r,z,h=[array(cal[g][k]) for k in ["r","o_input","h"]];n=cfg["probe_contexts"];count=2*rnk
        c=h-r-hidden(r)@p["B"].T-p["b"]
        so=(cfg["finite_step"]*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((z@vo.T)**2,0))[None,:,None]*vo[None,:,:]).reshape(count,-1)
        sb=(cfg["finite_step"]*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((hidden(r)@vb.T)**2,0))[None,:,None]*vb[None,:,:]).reshape(count,-1)
        rr=np.concatenate([r,(r[:n,None,:]+(so@p["O"].T)[None,:,:]).reshape(-1,r.shape[1]),np.repeat(r[:n],count,axis=0)])
        zz=np.concatenate([z,(z[:n,None,:]+so[None,:,:]).reshape(-1,z.shape[1]),np.repeat(z[:n],count,axis=0)])
        cc=np.concatenate([c,np.repeat(c[:n],count,axis=0),np.repeat(c[:n],count,axis=0)])
        off=np.concatenate([np.zeros((len(r)+n*count,p["B"].shape[1])),np.tile(sb,(n,1))])
        def full(o,b):
            r1=rr+zz@o.T;return cc+r1+(hidden(r1)+off)@(p["B"]+b).T+p["b"]
        before=features(full(np.zeros_like(eo),np.zeros_like(eb)),expert.model.visual)@p["head"].T
        after=features(full(eo,eb),expert.model.visual)@p["head"].T
        winner=before.argmax(1);rows=np.arange(len(before));gap=before[rows,winner,None]-before;gap[rows,winner]=1.
        ds=after-before;drift=np.abs((ds[rows,winner,None]-ds)/np.maximum(gap,1e-12));drift[rows,winner]=0.
        maximum=float(drift.max());changed=int((before[:len(r)].argmax(1)!=after[:len(r)].argmax(1)).sum())
        groups[g]={"maximum_margin_drift":maximum,"real_n":len(r),"probe_instances":2*n*count,"changed_real_predictions":changed}
        if g!="confirmation":assert maximum<=cfg["margin_fraction"]*(1+1e-6)
        else:confirmed=(before[:len(r)].argmax(1),after[:len(r)].argmax(1))
    limit=cfg["margin_fraction"];rho=groups["confirmation"]["maximum_margin_drift"]
    assert abs(rho-report["confirmation"]["worst_margin_drift"])<1e-6
    assert (rho<=limit)==report["confirmation_passed"]
    labels=array(cal["confirmation_labels"]);before,after=confirmed
    np.testing.assert_array_equal(before,array(cal["confirmation_original_predictions"]))
    np.testing.assert_array_equal(after,array(cal["confirmation_candidate_predictions"]))
    counts={"n":len(labels),"original_correct":int((before==labels).sum()),"candidate_correct":int((after==labels).sum()),"changed_predictions":int((before!=after).sum())}
    for key,value in counts.items():assert report["confirmation"][key]==value
    indices=cal["confirmation_indices"];old=set(i for group in cal["indices"].values() for i in group)
    parent=torch.load(Path(cfg["margin_reference"])/(report["target"]+"_diagnostic.pt"),map_location="cpu")
    old.update(parent["indices"]["confirmation"])
    assert not set(indices).intersection(old) and len(set(indices))==len(indices)
    fresh=[]
    if rho<=limit:
        protected=copy.deepcopy(expert);block=protected.model.visual.transformer.resblocks[11]
        with torch.no_grad():
            block.attn.out_proj.weight.add_(torch.from_numpy(eo).float())
            block.mlp.c_proj.weight.add_(torch.from_numpy(eb).float())
        training=dataset(report["target"],"data",expert.val_preprocess,True)
        for start in range(0,min(128,len(indices)),32):
            images=torch.stack([training[i][0] for i in indices[start:start+32]]).cuda()
            for name,model,pred in [("original",expert,before),("candidate",protected,after)]:
                with torch.no_grad():
                    model.cuda();head.cuda();actual=head(model(images)).argmax(1).cpu().numpy()
                    model.cpu();head.cpu()
                assert np.array_equal(actual,pred[start:start+len(images)])
                fresh.append({"model":name,"offset":start,"n":len(images),"prediction_match":True})
    result={"status":"passed" if rho<=limit else "confirmed_constraint_failure","source_sha256":digest(__file__),
            "candidate_sha256":digest(path),"parameter_distance":distance,"O_relative_distance":float(np.linalg.norm(eo)/np.linalg.norm(p["O"])),
            "B_relative_distance":float(np.linalg.norm(eb)/np.linalg.norm(p["B"])),"conditional_coefficient_error":coefficient_error,
            "task_constraints":groups,"confirmation_counts":counts,"new_release_weights":0,
            "new_model_forwards":sum(row["n"] for row in fresh),"fresh_replays":fresh,"test_used":False,"partner_used":False}
    save_json(output/"independent_validation.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if (Path(args.output)/"results.json").exists():verify_game(args.output)
    else:verify(args.output)
