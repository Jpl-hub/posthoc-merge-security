"""Class-readout output-complement noise with a finite native margin bound."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torchvision.datasets import DTD

from clip_benchmark import digest, merge
from covariance_noise import run
from decision_audit import classify
from subspace_probe import make_loader


def parameters(model, head):
    v = model.model.visual
    return dict(gamma=v.ln_post.weight.double(), beta=v.ln_post.bias.double(),
                eps=v.ln_post.eps, projection=v.proj.double(), head=head.weight.double())


def readout(params):
    contrast = params["head"]-params["head"].mean(0)
    a = params["gamma"][:, None]*(params["projection"] @ contrast.T)
    a -= a.mean(0)
    return a, params["beta"] @ params["projection"] @ contrast.T


def first_native_boundary(h, direction, a, offset, labels, eps):
    """First positive alpha where unchanged linear numerator loses a class tie."""
    center = h-h.mean(1,keepdim=True)
    d = direction-direction.mean(1,keepdim=True)
    q0, q1, q2 = center.square().mean(1)+eps, (center*d).mean(1), d.square().mean(1)
    linear = h @ a
    numerator = linear+q0.sqrt()[:,None]*offset
    correct = numerator.argmax(1).eq(labels)
    row = torch.arange(len(h))
    gap = linear[row,labels,None]-linear
    c = offset[labels,None]-offset
    threshold = -gap/c
    valid = correct[:,None] & c.abs().gt(1e-12) & threshold.gt(eps**.5) & q2[:,None].gt(1e-20)
    disc = q1[:,None].square()+q2[:,None]*(threshold.square()-q0[:,None])
    valid &= disc.ge(0)
    root = disc.clamp_min(0).sqrt()
    candidates = torch.stack([(-q1[:,None]-root)/q2[:,None],(-q1[:,None]+root)/q2[:,None]])
    valid = valid[None] & candidates.gt(1e-12) & torch.isfinite(candidates)
    candidates[~valid] = torch.inf
    bound = float(candidates.min())
    assert bound > 0 and bound < float("inf"), "No finite native amplitude bound; stop rather than invent a cap"
    return bound, int(correct.sum())


@torch.no_grad()
def construct(original, anchor, head, training, indices, cfg, target):
    ref = Path(cfg["native_cache"][target])
    meta = json.loads((ref/"execution.json").read_text())
    assert json.loads((ref/"independent_validation.json").read_text())["status"] == "passed"
    assert indices == json.loads((ref/"training_indices.json").read_text())
    assert digest(ref/"artifacts/calibration.pt") == meta["calibration_sha256"]
    cache = torch.load(ref/"artifacts/calibration.pt",map_location="cpu")
    parent = cache
    if "reference" in cache:
        assert digest(cache["reference"]["path"]) == cache["reference"]["sha256"]
        parent = torch.load(cache["reference"]["path"],map_location="cpu")
    z, zs, h, hs = [parent[k] for k in ["fit_hidden","scale_hidden","fit_residual","scale_residual"]]
    mean = z.mean(0)
    native_params = parameters(original,head)
    a, offset = readout(native_params)
    matrix = torch.cat([a,torch.ones(len(a),1,dtype=torch.float64)/len(a)**.5],dim=1)
    u,s,_ = torch.linalg.svd(matrix,full_matrices=False)
    basis = u[:,s>s[0]*cfg["numerical_relative_tolerance"]]
    proxy = merge(anchor,original,anchor,cfg["proxy_beta"])
    proxy_params = parameters(proxy,head)
    ap,_ = readout(proxy_params)
    leakage = ap-basis@(basis.T@ap)
    left, singular, right = torch.linalg.svd(leakage,full_matrices=False)
    keep = singular>singular[0]*cfg["numerical_relative_tolerance"]
    assert int(keep.sum()) == head.weight.shape[0]-1
    generator = torch.Generator().manual_seed(cfg["noise_seed"])
    random = torch.randn(head.weight.shape[0],z.shape[1],generator=generator,dtype=torch.float64)
    random -= random.mean(0)
    operator = left[:,keep] @ ((right[keep]@random)/singular[keep,None])
    norm = float(operator.norm())
    operator /= norm
    assert float((a.T@operator).norm()/a.norm()) < 1e-10
    expected = right[keep].T@(right[keep]@random)/norm
    assert float((ap.T@operator-expected).norm()/expected.norm()) < 1e-8
    labels_all = getattr(training,"targets",getattr(training,"_labels",None))
    bounds = {}
    for group,values,residual in [("fit",z,h),("scale",zs,hs)]:
        labels = torch.tensor([int(labels_all[i]) for i in indices[group]])
        response = (values-mean)@operator.T
        bound,n_correct = first_native_boundary(residual,response,a,offset,labels,native_params["eps"])
        bounds[group] = dict(first_boundary=bound,original_correct=n_correct,n=len(labels))
    amplitude = cfg["boundary_safety"]*min(v["first_boundary"] for v in bounds.values())
    layer = original.model.visual.transformer.resblocks[11].mlp.c_proj
    delta = amplitude*operator
    bias_delta = -delta@mean
    released = copy.deepcopy(original).cpu().eval()
    edited = released.model.visual.transformer.resblocks[11].mlp.c_proj
    edited.weight.copy_((layer.weight.double()+delta).float())
    edited.bias.copy_((layer.bias.double()+bias_delta).float())
    changed = {k for k in released.state_dict() if not torch.equal(released.state_dict()[k],original.state_dict()[k])}
    prefix = "model.visual.transformer.resblocks.11.mlp.c_proj."
    assert changed == {prefix+"weight",prefix+"bias"}
    actual = edited.weight.double()-layer.weight.double()
    diagnostics = []
    for name,model,params,latent,residual,ids,factor in [
        ("native_fit",original,native_params,z,h,indices["fit"],1),
        ("native_scale",original,native_params,zs,hs,indices["scale"],1),
        ("proxy_fit",proxy,proxy_params,cache["proxy_hidden"],cache["proxy_residual"],indices["fit"],cfg["proxy_beta"]),
    ]:
        response = factor*amplitude*(latent-mean)@operator.T
        before = classify(residual,params)[0].argmax(1)
        after = classify(residual+response,params)[0].argmax(1)
        labels = torch.tensor([int(labels_all[i]) for i in ids])
        original_correct = before.eq(labels)
        if name.startswith("native"):
            assert torch.all(after[original_correct].eq(labels[original_correct]))
        per_class = [dict(class_id=k,n=int(labels.eq(k).sum()),
                         original_correct=int((before.eq(labels)&labels.eq(k)).sum()),
                         protected_correct=int((after.eq(labels)&labels.eq(k)).sum())) for k in range(len(offset))]
        diagnostics.append(dict(context=name,n=len(labels),original_correct=int(original_correct.sum()),
                                protected_correct=int(after.eq(labels).sum()),per_class=per_class))
    edits = {k:released.state_dict()[k].clone() for k in changed}
    calibration = dict(cache_path=str(ref/"artifacts/calibration.pt"),cache_sha256=meta["calibration_sha256"],
                       mean=mean,basis=basis,readout=a,offset=offset,proxy_readout=ap,
                       operator=operator,random=random,amplitude=amplitude,operator_before_normalization=norm,
                       singular=singular,delta=delta,bias_delta=bias_delta)
    geometry = dict(output_task_rank=basis.shape[1],noise_rank=int(keep.sum()),input_PCA_used=False,
                    readout_leakage_fraction=float(leakage.norm()/ap.norm()),
                    native_readout_error=float((a.T@actual).norm()/(a.norm()*actual.norm())),
                    amplitude=amplitude,native_boundary=bounds,
                    relative_weight_change=float(actual.norm()/layer.weight.double().norm()),
                    diagnostics=diagnostics,partner_used=False,parameter_search=False,optimizer_steps=0,
                    old_output_complement_was_publicly_clearable=True,
                    limitation="Native linear class numerator preserved; LN variance offset constrained on training only. Existing public-clearability finding remains, no new cleaner tested.")
    return released,edits,calibration,geometry


def holdout_loader(original,training,indices,cfg,target):
    if target=="DTD":
        data=DTD("data",split="val",partition=1,transform=original.val_preprocess)
        assert len(data)==1880
        return make_loader(data),dict(dataset="DTD",split="val1",n=len(data),
            labels_sha256=digest("data/dtd/dtd/labels/val1.txt"),used_in_construction=False)
    used=set(sum(indices.values(),[]))
    pool=torch.tensor([i for i in range(len(training)) if i not in used])
    generator=torch.Generator().manual_seed(cfg["holdout_seed"])
    chosen=pool[torch.randperm(len(pool),generator=generator)[:cfg["holdout_n"]]].tolist()
    return make_loader(training,chosen),dict(dataset=target,split="train",n=len(chosen),
        indices=chosen,excluded_design_indices=True,seed=cfg["holdout_seed"],used_in_construction=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",required=True)
    parser.add_argument("--partner",required=True)
    parser.add_argument("--protocol",default="configs/decision_noise.json")
    parser.add_argument("--output",required=True)
    run(parser.parse_args(),constructor=construct,holdout_loader=holdout_loader)
