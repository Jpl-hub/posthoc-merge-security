"""Independent NumPy construction and official-label/fresh-forward verification."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import DTD

from clip_benchmark import checkpoint_loader,digest,merge
from covariance_noise import dataset
from subspace_probe import save_json
from verify_covariance_noise import official_labels


def array(value):
    return value.detach().cpu().numpy().astype(np.float64,copy=False)


def params(model,head):
    v=model.model.visual
    return dict(gamma=array(v.ln_post.weight),beta=array(v.ln_post.bias),eps=v.ln_post.eps,
                projection=array(v.proj),head=array(head.weight))


def numerator_space(p):
    contrast=p["head"]-p["head"].mean(0)
    a=p["gamma"][:,None]*(p["projection"]@contrast.T)
    a-=a.mean(0)
    return a,p["beta"]@p["projection"]@contrast.T


def numpy_features(h,p):
    x=h-h.mean(1,keepdims=True)
    x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True)+p["eps"])
    return (x*p["gamma"]+p["beta"])@p["projection"]


def numpy_boundary(h,d,a,k,labels,eps):
    h0=h-h.mean(1,keepdims=True);d0=d-d.mean(1,keepdims=True)
    q0=np.mean(h0*h0,axis=1)+eps;q1=np.mean(h0*d0,axis=1);q2=np.mean(d0*d0,axis=1)
    linear=h@a;before=linear+np.sqrt(q0)[:,None]*k
    correct=before.argmax(1)==labels;row=np.arange(len(h))
    gap=linear[row,labels,None]-linear;c=k[labels,None]-k
    with np.errstate(divide="ignore",invalid="ignore"):
        threshold=-gap/c
        disc=q1[:,None]**2+q2[:,None]*(threshold**2-q0[:,None])
        root=np.sqrt(disc.clip(0))
        candidates=np.stack([(-q1[:,None]-root)/q2[:,None],(-q1[:,None]+root)/q2[:,None]])
    valid=(correct[:,None]&(np.abs(c)>1e-12)&(threshold>np.sqrt(eps))&(q2[:,None]>1e-20)&(disc>=0))[None]
    candidates[~(valid&(candidates>1e-12)&np.isfinite(candidates))]=np.inf
    return float(candidates.min())


@torch.no_grad()
def verify(out,probe_n):
    out=Path(out);cfg=json.loads((out/"protocol.json").read_text());meta=json.loads((out/"execution.json").read_text());case=json.loads((out/"case.json").read_text())
    target,partner_name=case["target"],case["partner"]
    assert meta["completed"] and meta["optimizer_steps"]==0 and not meta["parameter_search"]
    for path,pin in cfg["source_sha256"].items():assert digest(path)==pin,path
    assert digest("experiments/covariance_noise.py")==meta["runner_sha256"]
    for filename,key in [("release.pt","release_sha256"),("calibration.pt","calibration_sha256"),("predictions.pt","predictions_sha256")]:
        assert digest(out/"artifacts"/filename)==meta[key]
    load=checkpoint_loader(Path("external/secure-merge-attack"));root=Path("checkpoints/ViT-B-32")
    for path,pin in meta["checkpoint_sha256"].items():assert digest(root/path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    anchor=load(str(root/"zeroshot.pt")).cpu().eval();original=load(str(root/target/"finetuned.pt")).cpu().eval();head=load(str(root/f"head_{target}.pt")).cpu().eval()
    release=torch.load(out/"artifacts/release.pt",map_location="cpu");cal=torch.load(out/"artifacts/calibration.pt",map_location="cpu")
    if "cache_path" in cal:
        assert digest(cal["cache_path"])==cal["cache_sha256"]
        cached=torch.load(cal["cache_path"],map_location="cpu")
    else:
        assert cfg["noise_family"]=="projection_bias" and cal["calibration_source"]=="fresh_target_training"
        cached=cal
    parent=cached
    if "reference" in cached:
        assert digest(cached["reference"]["path"])==cached["reference"]["sha256"]
        parent=torch.load(cached["reference"]["path"],map_location="cpu")
    indices=json.loads((out/"training_indices.json").read_text())
    labels=official_labels(target,True)
    proxy=merge(anchor,original,anchor,cfg["proxy_beta"])
    p,pp=params(original,head),params(proxy,head)
    family=cfg.get("noise_family","readout_output")
    algebra={}
    if family=="projection_bias":
        generator=torch.Generator().manual_seed(cfg["noise_seed"])
        requested=int(torch.randint(head.weight.shape[0],(1,),generator=generator))
        assert requested==cal["requested_class"]
        b=p["projection"]
        projector=np.eye(len(b))-b@np.linalg.pinv(b,rcond=cfg["numerical_relative_tolerance"])
        contrast=p["head"]-p["head"].mean(0)
        leakage=projector@pp["projection"]@contrast.T
        v=np.linalg.pinv(leakage.T,rcond=cfg["numerical_relative_tolerance"])@array(cal["desired"])
        error=float(np.linalg.norm(v-array(cal["direction"]))/np.linalg.norm(v))
        assert error<1e-7
        q=v@pp["projection"]@contrast.T;winner=int(q.argmax());rivals=np.arange(len(q))!=winner
        assert winner==cal["collapse_class"]
        t=numpy_features(array(cached["proxy_residual"]),pp);scores=t@p["head"].T
        bound=float(np.max((scores[:,rivals]-scores[:,winner,None])/(cfg["proxy_beta"]*(q[winner]-q[rivals]))))
        amplitude=cfg["bias_safety"]*bound
        np.testing.assert_allclose(amplitude,cal["amplitude"],rtol=1e-7,atol=1e-7)
        key="model.visual.ln_post.bias"
        expected=(p["beta"]+cal["amplitude"]*array(cal["direction"])).astype(np.float32)
        np.testing.assert_array_equal(release[key].numpy(),expected)
        assert set(release)=={key}
        error_beta=(array(release[key])-p["beta"])@b
        algebra=dict(direction_relative_error=error,ideal_embedding_null_error=float(np.linalg.norm(v@b)),
                     fp32_embedding_constant_error=float(np.linalg.norm(error_beta)),amplitude=amplitude,
                     proxy_bias_winner=winner)
    else:
        generator=torch.Generator().manual_seed(cfg["noise_seed"])
        expected_random=torch.randn(head.weight.shape[0],3072,generator=generator,dtype=torch.float64)
        expected_random-=expected_random.mean(0)
        assert torch.equal(expected_random,cal["random"])
        a,k=numerator_space(p);ap,_=numerator_space(pp)
        matrix=np.column_stack([a,np.ones(len(a))/np.sqrt(len(a))])
        u,s,_=np.linalg.svd(matrix,full_matrices=False);u=u[:,s>s[0]*cfg["numerical_relative_tolerance"]]
        leakage=ap-u@(u.T@ap)
        r=np.linalg.pinv(leakage.T,rcond=cfg["numerical_relative_tolerance"])@array(cal["random"])
        r/=np.linalg.norm(r)
        error=float(np.linalg.norm(r-array(cal["operator"])))
        assert error<1e-7
        bounds={}
        for group in ["fit","scale"]:
            z,h=[array(parent[group+suffix]) for suffix in ["_hidden","_residual"]]
            d=(z-array(cal["mean"]))@r.T
            bounds[group]=numpy_boundary(h,d,a,k,labels[indices[group]],p["eps"])
        amplitude=cfg["boundary_safety"]*min(bounds.values())
        np.testing.assert_allclose(amplitude,cal["amplitude"],rtol=1e-7,atol=1e-7)
        prefix="model.visual.transformer.resblocks.11.mlp.c_proj."
        for part,delta in [("weight",cal["delta"]),("bias",cal["bias_delta"])]:
            expected=(array(original.state_dict()[prefix+part])+array(delta)).astype(np.float32)
            np.testing.assert_array_equal(release[prefix+part].numpy(),expected)
        assert set(release)=={prefix+"weight",prefix+"bias"}
        algebra=dict(operator_error=error,native_readout_error=float(np.linalg.norm(a.T@r)),
                     native_boundary=bounds,amplitude=amplitude)
    del proxy
    released=copy.deepcopy(original)
    for key,value in release.items():released.state_dict()[key].copy_(value)
    raw=torch.load(out/"artifacts/predictions.pt",map_location="cpu");rows=json.loads((out/"results.json").read_text());holdout=json.loads((out/"holdout_protocol.json").read_text())
    train=dataset(target,"data",original.val_preprocess,True)
    tests={target:dataset(target,"data",original.val_preprocess)}
    heads={target:head}
    models={"original":original,"released":released}
    if meta["fusion_run"]:
        partner=load(str(root/partner_name/"finetuned.pt")).cpu().eval()
        heads[partner_name]=load(str(root/f"head_{partner_name}.pt")).cpu().eval()
        tests[partner_name]=dataset(partner_name,"data",original.val_preprocess)
        def recovered(model,task):
            ref=cfg["AGA_references"][task];assert digest(ref["path"])==ref["sha256"]
            result=copy.deepcopy(model);result.load_state_dict(torch.load(ref["path"],map_location="cpu"));return result
        ao=recovered(original,target);other=recovered(partner,partner_name)
        ar=copy.deepcopy(ao)
        assert digest(out/"artifacts/released_AGA_edits.pt")==meta["AGA_edits_sha256"]
        aga_edits=torch.load(out/"artifacts/released_AGA_edits.pt",map_location="cpu")
        assert set(aga_edits)==set(release)
        for key,value in aga_edits.items():ar.state_dict()[key].copy_(value)
        models.update(partner=partner,original_AGA=ao,released_AGA=ar)
        for name,model in [("original",original),("released",released)]:models[name+"_direct"]=merge(anchor,model,partner,cfg["coefficient"])
        for name,model in [("original",ao),("released",ar)]:models[name+"_AGA_fused"]=merge(anchor,model,other,cfg["coefficient"])
    counts=[];fresh=0
    for row in rows:
        preset,stage,task=row["preset"],row["stage"],row["dataset"]
        key="/".join([preset,stage,task]);stored=raw[key];prediction=stored["prediction"].numpy();truth=stored["target"].numpy()
        if stage=="holdout":
            if target=="DTD":
                data=DTD("data",split="val",partition=1,transform=original.val_preprocess)
                root_data=Path("data/dtd/dtd");classes=sorted(p.name for p in (root_data/"images").iterdir() if p.is_dir())
                names=(root_data/"labels/val1.txt").read_text().splitlines();official=np.array([classes.index(n.split("/")[0]) for n in names]);selected=list(range(len(data)))
            else:data=train;selected=holdout["indices"];official=labels[selected]
        elif stage=="confirmation":data=train;selected=indices["confirmation"];official=labels[selected]
        else:data=tests[task];selected=list(range(len(data)));official=official_labels(task)
        np.testing.assert_array_equal(truth,official)
        assert len(truth)==row["n"] and int((prediction==truth).sum())==row["correct"]
        for entry in row["per_class"]:
            choose=truth==entry["class_id"]
            assert entry["n"]==int(choose.sum()) and entry["correct"]==int(((prediction==truth)&choose).sum())
        if preset=="released":
            baseline=raw["/".join(["original",stage,task])]["prediction"].numpy()
            bc,rc=baseline==truth,prediction==truth
            assert row["paired"]["lost"]==int((bc&~rc).sum()) and row["paired"]["gained"]==int((~bc&rc).sum())
        if stage in ["holdout","confirmation","standalone"]:model=models[preset]
        elif stage=="partner_standalone":model=models["partner"]
        elif stage=="direct":model=models[preset+"_direct"]
        elif stage=="AGA_standalone":model=models[preset+"_AGA"]
        else:model=models[preset+"_AGA_fused"]
        images=torch.stack([data[i][0] for i in selected[:probe_n]])
        model.cuda().eval();classifier=heads[task].cuda().eval()
        features=model(images.cuda());actual_prediction=classifier(features).argmax(1).cpu().numpy()
        np.testing.assert_array_equal(actual_prediction,prediction[:len(actual_prediction)])
        model.cpu();classifier.cpu();fresh+=len(actual_prediction)
        counts.append(dict(key=key,correct=row["correct"],n=row["n"],unique_predictions=int(np.unique(prediction).size)))
    feature_check=None
    if family=="projection_bias":
        selected=indices["confirmation"][:probe_n];images=torch.stack([train[i][0] for i in selected]).cuda()
        original.cuda();released.cuda();f0=original(images).double().cpu();f1=released(images).double().cpu();original.cpu();released.cpu()
        shift=(array(release["model.visual.ln_post.bias"])-p["beta"])@p["projection"]
        measured=array(f1-f0)
        feature_check=dict(n=len(selected),actual_feature_delta_rms=float(np.sqrt(np.mean(np.sum(measured**2,axis=1)))),
                           max_error_from_constant_shift=float(np.max(np.abs(measured-shift))))
    result=dict(status="passed",algebra=algebra,official_counts=counts,fresh_predictions=fresh,
                native_feature_check=feature_check,source_sha256=digest(__file__),raw_predictions_sha256=meta["predictions_sha256"],
                new_protection_weights=0,new_attacks=0)
    save_json(out/"independent_validation.json",result)
    print(json.dumps(dict(output=str(out),status="passed",algebra=algebra,fresh_predictions=fresh,feature_check=feature_check)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--probe-n",type=int,default=32)
    args=parser.parse_args();verify(args.output,args.probe_n)
