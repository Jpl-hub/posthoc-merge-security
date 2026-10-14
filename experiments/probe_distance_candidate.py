"""One local maximum-distance displacement with task-probe/teacher constraints."""

import argparse
import copy
import json
from pathlib import Path

import torch

from clip_benchmark import digest, merge
from covariance_noise import run
from decision_noise import holdout_loader
from subspace_probe import save_json
from task_probe_distance import PREFIX, capture, gelu, mlp_delta, parameters, tail


def on_device(values):
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in values.items()}


def jvp_features(cache,p,dh):
    h=cache["h"];center=h-h.mean(1,keepdim=True)
    sigma=(center.square().mean(1,keepdim=True)+p["eps"]).sqrt()
    dc=dh-dh.mean(-1,keepdim=True)
    dl=dc/sigma[:,None,:]-center[:,None,:]*(center[:,None,:]*dc).mean(-1,keepdim=True)/sigma[:,None,:].pow(3)
    dt=(dl*p["gamma"])@p["projection"]
    t,f=tail(h,p)
    return (dt-f[:,None,:]*(dt*f[:,None,:]).sum(-1,keepdim=True))/t.norm(dim=-1)[:,None,None]


@torch.no_grad()
def jacobians(cache,p,basis,weight_norms,coordinate_form="diagonal",scales=None):
    x=cache["x"];pre=x@p["A"].T+p["a"]
    prob=torch.sigmoid(1.702*pre);z=pre*prob
    derivative=prob+1.702*pre*prob*(1-prob)
    a,b=basis["A"],basis["B"];r=a["u"].shape[1]
    a_input=x@a["vh"].T
    if coordinate_form=="matrix":
        output_a=(derivative[:,None,:]*a["u"].T[None,:,:])@p["B"].T
        dha=(weight_norms["A"]*output_a[:,:,None,:]*a_input[:,None,:,None]).flatten(1,2)
        dhb=(weight_norms["B"]*b["u"].T[None,:,None,:]*(z@b["vh"].T)[:,None,:,None]).flatten(1,2)
    else:
        da=weight_norms["A"]*a_input[:,:,None]*a["u"].T[None,:,:]
        dha=(da*derivative[:,None,:])@p["B"].T
        dhb=weight_norms["B"]*(z@b["vh"].T)[:,:,None]*b["u"].T[None,:,:]
    dh=torch.cat([dha,dhb],1)
    jf=jvp_features(cache,p,dh)
    pa=pre@a["u"];pb=(z@p["B"].T+p["b"])@b["u"]
    scale_a=pa.square().sum(-1).mean().sqrt().clamp_min(1e-12) if scales is None else scales["A"]
    scale_b=pb.square().sum(-1).mean().sqrt().clamp_min(1e-12) if scales is None else scales["B"]
    variables=r*r if coordinate_form=="matrix" else r
    ja=torch.zeros(len(x),2*variables,r,dtype=x.dtype,device=x.device)
    if coordinate_form=="matrix":
        ja[:,:variables,:]=(weight_norms["A"]*torch.eye(r,device=x.device)[None,:,None,:]*a_input[:,None,:,None]).flatten(1,2)/scale_a
    else:ja[:,:r,:]=torch.diag_embed(weight_norms["A"]*a_input)/scale_a
    jb=(dh@b["u"])/scale_b
    return jf,ja,jb,dict(A=scale_a,B=scale_b)


def displacement(coefficients,basis,norms):
    r=basis["A"]["u"].shape[1]
    if len(coefficients)==2*r*r:
        return {site:norms[site]*basis[site]["u"]@coefficients[offset:offset+r*r].reshape(r,r)@basis[site]["vh"]
                for site,offset in [("A",0),("B",r*r)]}
    return {site:norms[site]*(basis[site]["u"]*coefficients[offset:offset+r])@basis[site]["vh"]
            for site,offset in [("A",0),("B",r)]}


@torch.no_grad()
def nonlinear(cache,p,basis,norms,scales,coefficients):
    e=displacement(coefficients,basis,norms)
    x=cache["x"];pre=x@p["A"].T+p["a"]
    pre_new=x@(p["A"]+e["A"]).T+p["a"]
    dh=mlp_delta(cache,p,e["A"],e["B"])
    _,f0=tail(cache["h"],p);_,f1=tail(cache["h"]+dh,p)
    residuals=[f1-f0,(pre_new-pre)@basis["A"]["u"]/scales["A"],dh@basis["B"]["u"]/scales["B"]]
    losses=[float(v.square().sum(-1).mean()) for v in residuals]
    return sum(losses),losses,int((f0@p["head"].T).argmax(1).ne((f1@p["head"].T).argmax(1)).sum())


@torch.no_grad()
def construct(original,anchor,head,training,indices,cfg,target):
    ref=Path(cfg["probe_references"][target])
    diagnostic=json.loads((ref/"results.json").read_text())
    assert diagnostic["status"]=="passed" and not diagnostic["test_loaded"]
    assert digest(ref/"artifacts/probes.pt")==diagnostic["probes_sha256"]
    assert indices==json.loads((ref/"training_indices.json").read_text())
    stored=torch.load(ref/"artifacts/probes.pt",map_location="cpu")
    p_cpu=parameters(original,head);anchor_params=parameters(anchor,head);p=on_device(p_cpu)
    basis={site:{key:stored["svd"][site][key][:,stored["svd"][site]["selected"]].cuda()
                    if key=="u" else stored["svd"][site][key][stored["svd"][site]["selected"]].cuda()
                    for key in ["u","vh"]} for site in ["A","B"]}
    norms={site:p[site].norm() for site in ["A","B"]}
    cache_cpu=capture(original,training,indices["fit"]);cache=on_device(cache_cpu)
    proxy=merge(anchor,original,anchor,cfg["proxy_beta"])
    proxy_cache_cpu=capture(proxy,training,indices["fit"]);proxy_cache=on_device(proxy_cache_cpu)
    pp=on_device(parameters(proxy,head));del proxy
    def gram(j):return torch.einsum("nkd,nld->kl",j,j)/len(j)
    form=cfg.get("coordinate_form","diagonal")
    if form=="matrix":
        pre=cache["x"]@p["A"].T+p["a"]
        scales={"A":(pre@basis["A"]["u"]).square().sum(-1).mean().sqrt(),
                "B":((gelu(pre)@p["B"].T+p["b"])@basis["B"]["u"]).square().sum(-1).mean().sqrt()}
        hf=h=gp=None
        for start in range(0,len(cache["x"]),cfg["jacobian_chunk"]):
            end=start+cfg["jacobian_chunk"]
            local={k:v[start:end] for k,v in cache.items()}
            other={k:v[start:end] for k,v in proxy_cache.items()}
            jf,ja,jb,_=jacobians(local,p,basis,norms,form,scales)
            jproxy=jacobians(other,pp,basis,norms,form)[0]*cfg["proxy_beta"]
            weight=len(local["x"])/len(cache["x"])
            block_feature=weight*gram(jf);block_h=block_feature+weight*(gram(ja)+gram(jb));block_gp=weight*gram(jproxy)
            hf=block_feature if hf is None else hf+block_feature
            h=block_h if h is None else h+block_h;gp=block_gp if gp is None else gp+block_gp
            del jf,ja,jb,jproxy
        visibility=gp-hf
        values,q=torch.linalg.eigh((visibility+visibility.T)/2)
        keep=values>1e-10*max(float(gp.norm()),float(hf.norm()))
    else:
        jf,ja,jb,scales=jacobians(cache,p,basis,norms)
        jproxy=jacobians(proxy_cache,pp,basis,norms)[0]*cfg["proxy_beta"]
        hf=gram(jf);h=hf+gram(ja)+gram(jb);gp=gram(jproxy)
        values,q=torch.linalg.eigh((gp+gp.T)/2)
        keep=values>=cfg["proxy_eigenvalue_fraction"]*values[-1]
    h=(h+h.T)/2;gp=(gp+gp.T)/2
    assert keep.any(),"No proxy/native feature separation in this coordinate space; stop"
    q=q[:,keep];ev,v=torch.linalg.eigh(q.T@h@q)
    direction=q@v[:,0];curvature=float(direction@h@direction)
    assert curvature>0 and torch.isfinite(direction).all()
    upper=min(cfg["parameter_budget"],cfg["response_budget"]/curvature**.5)
    trials=[]
    def check(alpha):
        total,parts,changed=nonlinear(cache,p,basis,norms,scales,alpha*direction)
        trials.append(dict(alpha=alpha,total=total,parts=parts,teacher_changed=changed))
        return total<=cfg["response_budget"]**2
    if check(upper):amplitude=upper
    else:
        low,high=0.,upper
        for _ in range(cfg["scalar_solver_limit"]):
            middle=(low+high)/2
            if check(middle):low=middle
            else:high=middle
        amplitude=low
    assert amplitude>1e-10,"Only zero feasible; stop"
    coefficients=amplitude*direction;ed=displacement(coefficients,basis,norms)
    loss,parts,changed=nonlinear(cache,p,basis,norms,scales,coefficients)
    proxy_e={site:cfg["proxy_beta"]*ed[site] for site in ["A","B"]}
    _,fp0=tail(proxy_cache["h"],pp)
    _,fp1=tail(proxy_cache["h"]+mlp_delta(proxy_cache,pp,proxy_e["A"],proxy_e["B"]),pp)
    proxy_rms=float((fp1-fp0).square().sum(-1).mean().sqrt())
    native_rms=parts[0]**.5
    proxy_teacher_changed=int((fp0@pp["head"].T).argmax(1).ne((fp1@pp["head"].T).argmax(1)).sum())
    # Regression onto opposite neuron scalings checks an obvious gauge explanation.
    numerator=(ed["A"]*p["A"]).sum(1)-(ed["B"]*p["B"]).sum(0)
    denominator=p["A"].square().sum(1)+p["B"].square().sum(0)
    explained=float((numerator.square()/denominator.clamp_min(1e-20)).sum()/sum(e.square().sum() for e in ed.values()))
    geometry=dict(method="bounded_task_singular_probe_distance",coordinate_form=form,variable_dimension=len(coefficients),
        coefficient_distance=float(coefficients.norm()),parameter_budget=cfg["parameter_budget"],
        native_response_budget=cfg["response_budget"],native_feature_rms=native_rms,
        native_probe_rms=dict(A=parts[1]**.5,B=parts[2]**.5),native_fit_teacher_changed=changed,
        proxy_feature_rms=proxy_rms,proxy_teacher_changed=proxy_teacher_changed,
        proxy_visible_dimension=int(keep.sum()),native_curvature=curvature,
        neuron_scaling_explained_fraction=explained,scalar_solver_evaluations=len(trials),
        layer_relative_distance={s:float(ed[s].norm()/norms[s]) for s in ed},
        absolute_parameter_distance=float(sum(e.square().sum() for e in ed.values()).sqrt()),
        task_vector_relative_distance=float(sum(e.square().sum() for e in ed.values()).sqrt()/
            sum((p_cpu[s]-anchor_params[s]).square().sum() for s in ed).sqrt()),
        optimizer_steps=0,gradient_parameter_updates=0,labels_used=False,partner_used=False,
        local_linearized_solver=True,global_optimum_claim=False,
        phase1_bias_used=False,parameter_search=False)
    cal=dict(basis={s:{k:v.cpu() for k,v in b.items()} for s,b in basis.items()},native=cache_cpu,proxy=proxy_cache_cpu,
        gram_native=h.cpu(),gram_native_features=hf.cpu(),gram_proxy=gp.cpu(),direction=direction.cpu(),coefficients=coefficients.cpu(),
        amplitude=amplitude,curvature=curvature,scales={k:float(v) for k,v in scales.items()},trials=trials,
        probe_reference=str(ref),probe_sha256=diagnostic["probes_sha256"])
    save_json(Path(cfg["diagnostic_output"])/f"{target}_construction.json",geometry)
    torch.save(cal,Path(cfg["diagnostic_output"])/f"{target}_diagnostic.pt")
    assert explained<cfg["maximum_scaling_explanation"],"Distance dominated by neuron scaling; stop"
    assert proxy_rms>max(native_rms,1e-6),"No measured proxy/native functional separation; stop"
    released=copy.deepcopy(original).cpu().eval();edits={}
    for site,key in [("A","c_fc.weight"),("B","c_proj.weight")]:
        full=PREFIX+key
        released.state_dict()[full].copy_((p_cpu[site]+ed[site].cpu()).float())
        edits[full]=released.state_dict()[full].clone()
    changed_keys={k for k in original.state_dict() if not torch.equal(original.state_dict()[k],released.state_dict()[k])}
    assert changed_keys<=set(edits) and changed_keys
    geometry["edited_keys"]=sorted(changed_keys)
    total=sum(v.double().square().sum() for v in original.state_dict().values() if v.is_floating_point())
    geometry["full_model_relative_distance"]=geometry["absolute_parameter_distance"]/float(total.sqrt())
    return released,edits,cal,geometry


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",required=True);parser.add_argument("--partner",required=True)
    parser.add_argument("--protocol",default="configs/probe_distance_candidate.json")
    parser.add_argument("--output",required=True)
    run(parser.parse_args(),constructor=construct,holdout_loader=holdout_loader)
