"""Task-matrix probes and a bounded local distance problem, without task retraining."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from clip_benchmark import checkpoint_loader, digest
from covariance_noise import dataset
from subspace_probe import make_loader, save_json, split_training


PREFIX="model.visual.transformer.resblocks.11.mlp."


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def checked_models(target):
    load=checkpoint_loader(Path("external/secure-merge-attack"))
    root=Path("checkpoints/ViT-B-32")
    pins={p["path"].removeprefix("ViT-B-32/"):p["sha256"]
          for p in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    hashes={}
    def checked(name):
        hashes[name]=digest(root/name)
        assert hashes[name]==pins[name],name
        model=load(str(root/name)).cpu().eval()
        for p in model.parameters():p.requires_grad_(False)
        return model
    return checked("zeroshot.pt"),checked(target+"/finetuned.pt"),checked("head_"+target+".pt"),hashes


def parameters(model,head):
    v=model.model.visual;block=v.transformer.resblocks[11]
    assert type(block.mlp.gelu).__name__=="QuickGELU"
    assert head.normalize and torch.count_nonzero(head.bias)==0
    return {name:value.detach().cpu().double() for name,value in dict(
        A=block.mlp.c_fc.weight,a=block.mlp.c_fc.bias,
        B=block.mlp.c_proj.weight,b=block.mlp.c_proj.bias,
        gamma=v.ln_post.weight,beta=v.ln_post.bias,projection=v.proj,head=head.weight).items()}|dict(eps=v.ln_post.eps)


@torch.no_grad()
def capture(model,training,indices):
    x,h,t=[],[],[]
    block=model.model.visual.transformer.resblocks[11]
    hooks=[block.mlp.c_fc.register_forward_pre_hook(lambda _m,args:x.append(args[0][0].detach().cpu())),
           model.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:h.append(args[0].detach().cpu()))]
    try:
        model.cuda().eval()
        for images,_ in make_loader(training,indices,64):t.append(model(images.cuda()).detach().cpu())
    finally:
        for hook in hooks:hook.remove()
        model.cpu()
    return dict(x=torch.cat(x).double(),h=torch.cat(h).double(),actual=torch.cat(t).double())


def gelu(x):return x*torch.sigmoid(1.702*x)


def tail(h,p):
    normalized=F.layer_norm(h,(h.shape[-1],),p["gamma"],p["beta"],p["eps"])
    t=normalized@p["projection"]
    return t,t/t.norm(dim=-1,keepdim=True)


def mlp_delta(cache,p,ea,eb):
    x=cache["x"];pre=x@p["A"].T+p["a"];z=gelu(pre)
    altered=gelu(x@(p["A"]+ea).T+p["a"])@(p["B"]+eb).T
    return altered-z@p["B"].T


def classify(cache,p,ea,eb):
    _,f=tail(cache["h"]+mlp_delta(cache,p,ea,eb),p)
    return f@p["head"].T


def score_gradient(cache,p):
    with torch.enable_grad():
        h=cache["h"].clone().requires_grad_(True)
        _,f=tail(h,p);contrast=p["head"]-p["head"].mean(0)
        scores=f@contrast.T
        grad=[torch.autograd.grad(scores[:,k].sum(),h,retain_graph=k+1<len(contrast))[0]
              for k in range(len(contrast))]
    return torch.stack(grad,1).detach()


@torch.no_grad()
def importance(cache,p,svd):
    x=cache["x"];pre=x@p["A"].T+p["a"];prob=torch.sigmoid(1.702*pre)
    derivative=prob+1.702*pre*prob*(1-prob);z=pre*prob
    gh=score_gradient(cache,p)
    result={}
    for site in ["A","B"]:
        u,s,vh=[svd[site][k] for k in ["u","s","vh"]]
        inputs=x if site=="A" else z
        activation=inputs@vh.T
        upstream=(gh@p["B"])*derivative[:,None,:] if site=="A" else gh
        response=(upstream@u)*activation[:,None,:]*s[None,None,:]
        result[site]=dict(activation_energy=(activation.square().mean(0)*s.square()),
                          class_sensitivity=response.square().sum(1).mean(0),response=response)
    return result


@torch.no_grad()
def diagnose(target,protocol,output):
    cfg=json.loads(Path(protocol).read_text());initialize()
    for path,pin in cfg["source_sha256"].items():assert digest(path)==pin,path
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/"artifacts").mkdir()
    save_json(out/"protocol.json",cfg)
    anchor,expert,head,hashes=checked_models(target)
    p=parameters(expert,head);p0=parameters(anchor,head)
    training=dataset(target,"data",expert.val_preprocess,True)
    n=min(cfg["maximum_calibration_group"],len(training)//3)
    indices=split_training(training,{k:n for k in ["fit","scale","confirmation"]},cfg["seed"])
    save_json(out/"training_indices.json",indices)
    groups={k:capture(expert,training,indices[k][:cfg["diagnostic_n"]]) for k in ["fit","confirmation"]}
    svd={}
    for site in ["A","B"]:
        matrix=p[site]-p0[site]
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
        svd[site]=dict(u=u,s=s,vh=vh)
    imp={g:importance(cache,p,svd) for g,cache in groups.items()}
    rows=[];summaries={};checks=[]
    for site in ["A","B"]:
        s=svd[site]["s"];fit=imp["fit"][site];confirm=imp["confirmation"][site]
        chosen=fit["class_sensitivity"].argsort(descending=True)[:cfg["probe_rank"]]
        svd[site]["selected"]=chosen
        spectral=s[:cfg["probe_rank"]].square().sum()/s.square().sum()
        overlap=len(set(chosen.tolist())&set(confirm["class_sensitivity"].argsort(descending=True)[:cfg["probe_rank"]].tolist()))
        summaries[site]=dict(shape=list(p[site].shape),task_norm=float(s.norm()),original_norm=float(p[site].norm()),
            top32_parameter_energy=float(spectral),selected=chosen.tolist(),
            selected_sensitivity_fraction=float(fit["class_sensitivity"][chosen].sum()/fit["class_sensitivity"].sum()),
            top32_sigma_sensitivity_fraction=float(fit["class_sensitivity"][:cfg["probe_rank"]].sum()/fit["class_sensitivity"].sum()),
            selected_confirmation_fraction=float(confirm["class_sensitivity"][chosen].sum()/confirm["class_sensitivity"].sum()),
            top32_overlap_fit_confirmation=overlap,
            spearman_sigma_class=float(spearmanr(s.numpy(),fit["class_sensitivity"].numpy()).statistic),
            spearman_fit_confirmation=float(spearmanr(fit["class_sensitivity"].numpy(),confirm["class_sensitivity"].numpy()).statistic))
        for i in range(len(s)):
            rows.append(dict(site=site,index=i,sigma=float(s[i]),fit_activation=float(fit["activation_energy"][i]),
                fit_class_sensitivity=float(fit["class_sensitivity"][i]),
                confirmation_class_sensitivity=float(confirm["class_sensitivity"][i]),selected=bool((chosen==i).any())))
        for i in sorted({0,int(chosen[0]),len(s)-1}):
            mode=s[i]*svd[site]["u"][:,i,None]@svd[site]["vh"][i,None,:]
            ea,eb=[cfg["finite_step"]*mode if k==site else torch.zeros_like(p[k]) for k in ["A","B"]]
            before=classify(groups["fit"],p,torch.zeros_like(p["A"]),torch.zeros_like(p["B"]))
            actual=classify(groups["fit"],p,ea,eb)-before;actual-=actual.mean(1,keepdim=True)
            predicted=cfg["finite_step"]*imp["fit"][site]["response"][:,:,i]
            checks.append(dict(site=site,index=i,finite_step=cfg["finite_step"],
                derivative_relative_error=float((actual-predicted).norm()/predicted.norm().clamp_min(1e-20))))
    replay=[]
    for group,cache in groups.items():
        t,f=tail(cache["h"],p);real=cache["actual"]/cache["actual"].norm(dim=-1,keepdim=True)
        replay.append(dict(group=group,n=len(t),feature_relative_error=float((t-cache["actual"]).norm()/cache["actual"].norm()),
                           changed_predictions=int((f@p["head"].T).argmax(1).ne((real@p["head"].T).argmax(1)).sum())))
    # Two actual model forwards verify the analytic/cached intervention on four training images.
    full=[]
    images=torch.stack([training[i][0] for i in indices["fit"][:4]]).cuda()
    for site,key in [("A","c_fc.weight"),("B","c_proj.weight")]:
        i=int(svd[site]["selected"][0]);s=svd[site]["s"]
        change=cfg["finite_step"]*s[i]*svd[site]["u"][:,i,None]@svd[site]["vh"][i,None,:]
        tensor=expert.state_dict()[PREFIX+key];original=tensor.clone()
        try:
            tensor.copy_((tensor.double()+change).float());expert.cuda();actual=expert(images).double().cpu();expert.cpu()
        finally:
            expert.cpu();expert.state_dict()[PREFIX+key].copy_(original)
        ea,eb=[change if k==site else torch.zeros_like(p[k]) for k in ["A","B"]]
        cache={k:v[:4] for k,v in groups["fit"].items()}
        expected,_=tail(cache["h"]+mlp_delta(cache,p,ea,eb),p)
        full.append(dict(site=site,n=4,relative_error=float((actual-expected).norm()/actual.norm())))
        assert torch.equal(expert.state_dict()[PREFIX+key],original)
    assert all(r["changed_predictions"]==0 and r["feature_relative_error"]<1e-5 for r in replay)
    assert all(r["relative_error"]<1e-4 for r in full)
    torch.save(dict(svd=svd,groups=groups,importance=imp),out/"artifacts/probes.pt")
    save_json(out/"directions.json",rows)
    result=dict(target=target,status="passed",summaries=summaries,derivative_checks=checks,native_reconstruction=replay,
                actual_full_model_checks=full,calibration_n=cfg["diagnostic_n"],probe_rank=cfg["probe_rank"],
                original_parameters_updated=False,gradient_optimizer_steps=0,labels_used_for_selection=False,
                partner_loaded=False,test_loaded=False,source_sha256=digest(__file__),protocol_sha256=digest(protocol),
                probes_sha256=digest(out/"artifacts/probes.pt"),checkpoint_sha256=hashes)
    save_json(out/"results.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",required=True)
    parser.add_argument("--protocol",default="configs/task_probe_distance.json")
    parser.add_argument("--output",required=True)
    args=parser.parse_args();diagnose(args.target,args.protocol,args.output)
