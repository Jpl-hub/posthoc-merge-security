"""Independent parameter/constraint reconstruction and official prediction replay."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torchvision.datasets import DTD

from clip_benchmark import checkpoint_loader,digest
from covariance_noise import dataset
from subspace_probe import save_json
from verify_covariance_noise import official_labels


PREFIX="model.visual.transformer.resblocks.11.mlp."


def array(x):return x.detach().cpu().numpy().astype(np.float64,copy=False)


def features(h,v):
    centered=h-h.mean(1,keepdims=True)
    normalized=centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+v.ln_post.eps)
    t=(normalized*array(v.ln_post.weight)+array(v.ln_post.bias))@array(v.proj)
    return t/np.linalg.norm(t,axis=1,keepdims=True)


def explicit_merge(anchor,own,peer,coefficient):
    result=copy.deepcopy(anchor)
    for k,dest in result.state_dict().items():
        if dest.is_floating_point():
            a=anchor.state_dict()[k];dest.copy_(a+coefficient*((own.state_dict()[k]-a)+(peer.state_dict()[k]-a)))
    return result


@torch.no_grad()
def verify(output,probe_n):
    out=Path(output);cfg=json.loads((out/"protocol.json").read_text());meta=json.loads((out/"execution.json").read_text());case=json.loads((out/"case.json").read_text())
    assert meta["completed"] and meta["optimizer_steps"]==0
    assert cfg["coordinate_form"]=="matrix"
    for path,pin in cfg["source_sha256"].items():assert digest(path)==pin,path
    for name,key in [("release.pt","release_sha256"),("calibration.pt","calibration_sha256"),("predictions.pt","predictions_sha256")]:
        assert digest(out/"artifacts"/name)==meta[key]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    load=checkpoint_loader(Path("external/secure-merge-attack"));root=Path("checkpoints/ViT-B-32")
    for path,pin in meta["checkpoint_sha256"].items():assert digest(root/path)==pin,path
    target,peer_name=case["target"],case["partner"]
    anchor=load(str(root/"zeroshot.pt")).cpu().eval();original=load(str(root/target/"finetuned.pt")).cpu().eval();head=load(str(root/f"head_{target}.pt")).cpu().eval()
    cal=torch.load(out/"artifacts/calibration.pt",map_location="cpu");release=torch.load(out/"artifacts/release.pt",map_location="cpu")
    d,c,h,gp,hf=[array(cal[k]) for k in ["direction","coefficients","gram_native","gram_proxy","gram_native_features"]]
    assert digest(Path(cal["probe_reference"])/"artifacts/probes.pt")==cal["probe_sha256"]
    r=cfg["probe_rank"];assert len(c)==2*r*r and np.linalg.norm(c)<=cfg["parameter_budget"]+1e-10
    np.testing.assert_allclose(c,cal["amplitude"]*d,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(d),1.,atol=1e-10)
    np.testing.assert_allclose(d@h@d,cal["curvature"],rtol=1e-7,atol=1e-10)
    assert d@(gp-hf)@d>0
    eigenvalues,q=np.linalg.eigh((gp-hf+gp.T-hf.T)/2)
    visible=eigenvalues>1e-10*max(np.linalg.norm(gp),np.linalg.norm(hf))
    q=q[:,visible]
    assert np.linalg.norm(d-q@(q.T@d))<1e-5
    optimum=float(np.linalg.eigvalsh(q.T@h@q)[0])
    np.testing.assert_allclose(optimum,cal["curvature"],rtol=1e-4,atol=1e-7)
    blocks=original.model.visual.transformer.resblocks[11].mlp
    weights={"A":array(blocks.c_fc.weight),"B":array(blocks.c_proj.weight)}
    delta={}
    for site,key,offset in [("A","c_fc.weight",0),("B","c_proj.weight",r*r)]:
        b=cal["basis"][site];u,v=array(b["u"]),array(b["vh"])
        np.testing.assert_allclose(u.T@u,np.eye(r),atol=1e-8)
        np.testing.assert_allclose(v@v.T,np.eye(r),atol=1e-8)
        delta[site]=np.linalg.norm(weights[site])*u@c[offset:offset+r*r].reshape(r,r)@v
        expected=(weights[site]+delta[site]).astype(np.float32)
        np.testing.assert_allclose(expected,release[PREFIX+key].numpy(),rtol=2e-7,atol=2e-8)
    assert set(release)=={PREFIX+"c_fc.weight",PREFIX+"c_proj.weight"}
    x,raw_h=array(cal["native"]["x"]),array(cal["native"]["h"])
    pre=x@weights["A"].T+array(blocks.c_fc.bias);z=pre*expit(1.702*pre)
    newpre=x@(weights["A"]+delta["A"]).T+array(blocks.c_fc.bias)
    znew=newpre*expit(1.702*newpre)
    dh=znew@(weights["B"]+delta["B"]).T-z@weights["B"].T
    f0=features(raw_h,original.model.visual);f1=features(raw_h+dh,original.model.visual)
    pa=(newpre-pre)@array(cal["basis"]["A"]["u"])/cal["scales"]["A"]
    pb=dh@array(cal["basis"]["B"]["u"])/cal["scales"]["B"]
    parts=[float(np.mean(np.sum(v*v,axis=1))) for v in [f1-f0,pa,pb]]
    assert sum(parts)<=cfg["response_budget"]**2*(1+1e-6)
    algebra=dict(coefficient_distance=float(np.linalg.norm(c)),absolute_ideal_distance=float(np.sqrt(sum(np.sum(e*e) for e in delta.values()))),independent_min_curvature=optimum,visible_dimension=int(visible.sum()),native_nonlinear_parts=parts,native_nonlinear_total=sum(parts),strict_visibility_quadratic=float(d@(gp-hf)@d),changed_teacher_fit=int(np.sum(np.argmax(f0@array(head.weight).T,axis=1)!=np.argmax(f1@array(head.weight).T,axis=1))))
    released=copy.deepcopy(original)
    for k,v in release.items():released.state_dict()[k].copy_(v)
    raw=torch.load(out/"artifacts/predictions.pt",map_location="cpu");rows=json.loads((out/"results.json").read_text())
    train=dataset(target,"data",original.val_preprocess,True);tests={target:dataset(target,"data",original.val_preprocess)}
    heads={target:head};models={"original":original,"released":released};labels=official_labels(target,True)
    if meta["fusion_run"]:
        peer=load(str(root/peer_name/"finetuned.pt")).cpu().eval();heads[peer_name]=load(str(root/f"head_{peer_name}.pt")).cpu().eval();tests[peer_name]=dataset(peer_name,"data",original.val_preprocess)
        def recovered(model,task):
            ref=cfg["AGA_references"][task];assert digest(ref["path"])==ref["sha256"]
            result=copy.deepcopy(model);result.load_state_dict(torch.load(ref["path"],map_location="cpu"));return result
        ao=recovered(original,target);other=recovered(peer,peer_name);ar=copy.deepcopy(ao)
        assert digest(out/"artifacts/released_AGA_edits.pt")==meta["AGA_edits_sha256"]
        for k,v in torch.load(out/"artifacts/released_AGA_edits.pt",map_location="cpu").items():ar.state_dict()[k].copy_(v)
        models.update(partner=peer,original_AGA=ao,released_AGA=ar)
        for preset,own in [("original",original),("released",released)]:models[preset+"_direct"]=explicit_merge(anchor,own,peer,cfg["coefficient"])
        for preset,own in [("original",ao),("released",ar)]:models[preset+"_AGA_fused"]=explicit_merge(anchor,own,other,cfg["coefficient"])
    indices=json.loads((out/"training_indices.json").read_text());hold=json.loads((out/"holdout_protocol.json").read_text());checks=[];fresh=0
    for row in rows:
        preset,stage,task=[row[k] for k in ["preset","stage","dataset"]];key="/".join([preset,stage,task]);p=raw[key]["prediction"].numpy();y=raw[key]["target"].numpy()
        if stage=="holdout" and target=="DTD":
            data=DTD("data",split="val",partition=1,transform=original.val_preprocess);selected=list(range(len(data)))
            rootdata=Path("data/dtd/dtd");classes=sorted(q.name for q in (rootdata/"images").iterdir() if q.is_dir());truth=np.array([classes.index(n.split("/")[0]) for n in (rootdata/"labels/val1.txt").read_text().splitlines()])
        elif stage in ["holdout","confirmation"]:
            data=train;selected=hold["indices"] if stage=="holdout" else indices["confirmation"];truth=labels[selected]
        else:data=tests[task];selected=list(range(len(data)));truth=official_labels(task)
        np.testing.assert_array_equal(y,truth);assert len(y)==row["n"] and int((p==y).sum())==row["correct"]
        for item in row["per_class"]:
            choose=y==item["class_id"];assert item["n"]==int(choose.sum()) and item["correct"]==int(((p==y)&choose).sum())
        if preset=="released":
            baseline=raw["/".join(["original",stage,task])]["prediction"].numpy()
            assert row["changed_predictions"]==int((p!=baseline).sum())
            assert row["paired"]["lost"]==int(((baseline==y)&(p!=y)).sum()) and row["paired"]["gained"]==int(((baseline!=y)&(p==y)).sum())
        if stage in ["holdout","confirmation","standalone"]:model=models[preset]
        elif stage=="partner_standalone":model=models["partner"]
        elif stage=="AGA_standalone":model=models[preset+"_AGA"]
        elif stage=="direct":model=models[preset+"_direct"]
        else:model=models[preset+"_AGA_fused"]
        images=torch.stack([data[i][0] for i in selected[:probe_n]]).cuda();model.cuda();classifier=heads[task].cuda()
        prediction=classifier(model(images)).argmax(1).cpu().numpy();np.testing.assert_array_equal(prediction,p[:len(prediction)])
        model.cpu();classifier.cpu();fresh+=len(prediction);checks.append(dict(key=key,correct=row["correct"],n=row["n"]))
    result=dict(status="passed",algebra=algebra,official_counts=checks,fresh_predictions=fresh,explicit_merge_formula=True,verifier_sha256=digest(__file__),raw_predictions_sha256=meta["predictions_sha256"],new_attacks=0,new_release_weights=0)
    save_json(out/"independent_validation.json",result)
    print(json.dumps(dict(output=str(out),status="passed",algebra=algebra,fresh_predictions=fresh)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--probe-n",type=int,default=32)
    args=parser.parse_args();verify(args.output,args.probe_n)
