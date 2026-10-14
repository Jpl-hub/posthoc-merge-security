"""Task-SVD Attention output changes with downstream MLP output distillation."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from angular_distance import Problem, angular_scores
from clip_benchmark import digest, merge
from conditional_task_distance import DistilledProblem, diagnose, optimize
from covariance_noise import dataset
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, gelu, parameters


def extended_parameters(model, head):
    block=model.model.visual.transformer.resblocks[11]
    return parameters(model,head)|{k:v.detach().cpu().double() for k,v in dict(
        O=block.attn.out_proj.weight,ob=block.attn.out_proj.bias,
        gamma2=block.ln_2.weight,beta2=block.ln_2.bias).items()}|dict(eps2=block.ln_2.eps)


def normalized_input(r,p):
    return F.layer_norm(r,(r.shape[-1],),p["gamma2"],p["beta2"],p["eps2"])


def final_hidden(r,p):
    return r+gelu(normalized_input(r,p)@p["A"].T+p["a"])@p["B"].T+p["b"]


@torch.no_grad()
def capture_attention(model,training,indices):
    """Observe the exact MHA pre-output input; never invert an attention matrix."""
    block=model.model.visual.transformer.resblocks[11]
    assert not block.attn.batch_first
    groups={k:[] for k in ["o_input","attn_output","r","h","actual"]}; active={}
    old_linear=F.linear
    def record_linear(value,weight,bias=None):
        if weight.data_ptr()==active["o_pointer"]:
            active["calls"]+=1
            groups["o_input"].append(value.reshape(-1,active["batch"],value.shape[-1])[0].detach().cpu())
        return old_linear(value,weight,bias)
    hooks=[block.attn.register_forward_hook(lambda _m,_args,out:groups["attn_output"].append(out[0][0].detach().cpu())),
           block.ln_2.register_forward_pre_hook(lambda _m,args:groups["r"].append(args[0][0].detach().cpu())),
           model.model.visual.ln_post.register_forward_pre_hook(lambda _m,args:groups["h"].append(args[0].detach().cpu()))]
    try:
        model.cuda().eval();active["o_pointer"]=block.attn.out_proj.weight.data_ptr();F.linear=record_linear
        for images,_ in make_loader(training,indices,64):
            active.update(batch=len(images),calls=0)
            groups["actual"].append(model(images.cuda()).detach().cpu())
            assert active["calls"]==1,"MHA bypassed the exact pre-output observation"
    finally:
        F.linear=old_linear
        for hook in hooks:hook.remove()
        model.cpu()
    cache={k:torch.cat(v).double() for k,v in groups.items()}
    assert all(len(v)==len(indices) for v in cache.values())
    return cache


@torch.no_grad()
def output_jacobian(cache,p):
    """Analytic normalized class-score Jacobian wrt the Attention residual."""
    h=cache["h"];c=h-h.mean(1,keepdim=True);sd=(c.square().mean(1,keepdim=True)+p["eps"]).sqrt()
    t=(c/sd*p["gamma"]+p["beta"])@p["projection"];tn=t.norm(dim=1,keepdim=True);f=t/tn
    s=f@p["head"].T
    j=(p["head"][None,:,:]-s[:,:,None]*f[:,None,:])/tn[:,None,:]
    j=(j@p["projection"].T)*p["gamma"]
    j=(j-j.mean(-1,keepdim=True)-c[:,None,:]*(j*c[:,None,:]).mean(-1,keepdim=True)/sd[:,None,:].square())/sd[:,None,:]
    u=angular_scores(s);cs=(s-s.mean(1,keepdim=True)).norm(dim=1,keepdim=True)
    j=(j-j.mean(1,keepdim=True)-u[:,:,None]*(u[:,:,None]*j).sum(1,keepdim=True))/cs[:,None,:]
    r=cache["r"];rc=r-r.mean(1,keepdim=True);rs=(rc.square().mean(1,keepdim=True)+p["eps2"]).sqrt()
    pre=normalized_input(r,p)@p["A"].T+p["a"];sig=torch.sigmoid(1.702*pre)
    slope=sig+1.702*pre*sig*(1-sig)
    back=((j@p["B"])*slope[:,None,:])@p["A"]*p["gamma2"]
    back=(back-back.mean(-1,keepdim=True)-rc[:,None,:]*(back*rc[:,None,:]).mean(-1,keepdim=True)/rs[:,None,:].square())/rs[:,None,:]
    return j+back,s


class CrossProblem(Problem):
    def __init__(self,target,cfg):
        self.target,self.cfg=target,cfg;self.radius=cfg["parameter_budget"];self.form="matrix";self.r=cfg["probe_rank"];self.a_size=self.r**2
        parent=Path(cfg["margin_reference"])/(target+"_diagnostic.pt")
        self.parent_sha=digest(parent);meta=json.loads(parent.with_name(target+".json").read_text())
        assert self.parent_sha==meta["diagnostic_sha256"]
        previous=torch.load(parent,map_location="cpu")
        self.anchor,self.expert,self.head,self.hashes=checked_models(target)
        self.p_cpu=extended_parameters(self.expert,self.head);self.p=on_device(self.p_cpu)
        self.b_width=self.p["B"].shape[1] if cfg.get("full_B_completion") else self.r
        self.b_rows=self.p["B"].shape[0]
        self.output_basis=None
        self.prior_confirmation_indices=previous["indices"]["confirmation"]
        if cfg.get("cache_reference"):
            root=Path(cfg["cache_reference"]);meta=json.loads((root/"mechanism.json").read_text())
            assert digest(root/"calibration.pt")==cfg["calibration_sha256"]==meta["calibration_sha256"]
            assert meta["checkpoint_sha256"]==self.hashes and meta["parent_sha256"]==self.parent_sha
            cache=torch.load(root/"calibration.pt",map_location="cpu")
            proxy=merge(self.anchor,self.expert,self.anchor,cfg["proxy_beta"])
            self.pp=on_device(extended_parameters(proxy,self.head));del proxy
            self.complete(cache,cache["basis_O"],cache["basis_B"],cache["indices"])
            save_json(Path(cfg["output"])/"mechanism.json",{"target":target,"cache_reference":str(root),"calibration_sha256":meta["calibration_sha256"],"checkpoint_sha256":self.hashes,"new_cache_model_forwards":0})
            return
        training=dataset(target,"data",self.expert.val_preprocess,True)
        indices={k:previous["indices"][k] for k in ["fit","scale"]}
        cache={g:capture_attention(self.expert,training,indices[k]) for g,k in [("native","fit"),("scale","scale")]}
        proxy=merge(self.anchor,self.expert,self.anchor,cfg["proxy_beta"])
        self.pp=on_device(extended_parameters(proxy,self.head));cache["proxy"]=capture_attention(proxy,training,indices["fit"]);del proxy
        cache_checks=[]
        for g in ["native","scale","proxy"]:
            item=on_device(cache[g]);p=self.pp if g=="proxy" else self.p
            attention_error=float((item["o_input"]@p["O"].T+p["ob"]-item["attn_output"]).abs().max())
            actual=(item["actual"]/item["actual"].norm(dim=1,keepdim=True))@p["head"].T
            score_error=float((scores(item["h"],p)-actual).abs().max())
            assert attention_error<5e-4 and score_error<5e-4,(g,attention_error,score_error)
            cache_checks.append({"group":g,"n":len(item["h"]),"attention_output_maximum_error":attention_error,"zero_transform_score_maximum_error":score_error})
        p0=extended_parameters(self.anchor,self.head);task=self.p_cpu["O"]-p0["O"]
        u,s,vh=torch.linalg.svd(task,full_matrices=False)
        mode={"u":u.cuda(),"s":s.cuda(),"vh":vh.cuda()};imp={};checks=[]
        for g in ["native","scale"]:
            small=on_device({k:v[:cfg["selection_n"]] for k,v in cache[g].items()})
            jac,teacher=output_jacobian(small,self.p)
            response=(jac@mode["u"])*(small["o_input"]@mode["vh"].T)[:,None,:]
            winners=teacher.argmax(1);counts=torch.bincount(winners,minlength=teacher.shape[1])
            weights=1./counts[winners].double()/int((counts>0).sum())
            imp[g]=(response.square().sum(1)*weights[:,None]).sum(0)
            for i in [0,int(imp[g].argmax()),len(s)-1]:
                dr=(small["o_input"]@mode["vh"][i])[:,None]*mode["u"][:,i]
                h=cfg["selection_derivative_step"]
                rounding=small["h"]-final_hidden(small["r"],self.p)
                plus=angular_scores(scores(rounding+final_hidden(small["r"]+h*dr,self.p),self.p))
                minus=angular_scores(scores(rounding+final_hidden(small["r"]-h*dr,self.p),self.p))
                error=float(((plus-minus)/(2*h)-response[:,:,i]).norm()/response[:,:,i].norm().clamp_min(1e-20))
                assert error<1e-3,(g,i,error)
                checks.append({"group":g,"mode":i,"relative_error":error})
        chosen=imp["native"].argsort(descending=True)[:self.r]
        cpu_chosen=chosen.cpu()
        o_basis={"u":u[:,cpu_chosen],"vh":vh[cpu_chosen],"s":s[cpu_chosen],"selected":cpu_chosen}
        # B uses the previously frozen target-only task-singular basis, not a peer-selected basis.
        self.complete(cache,o_basis,previous["basis"]["B"],indices)
        summary={"target":target,"site":"block11 attn.out_proj and mlp.c_proj","selection_rule":"class-balanced unit-Frobenius normalized-score sensitivity, target training only",
                 "selected_O_modes":chosen.cpu().tolist(),"fit_selected_response_fraction":float(imp["native"][chosen].sum()/imp["native"].sum()),
                 "scale_selected_response_fraction":float(imp["scale"][chosen].sum()/imp["scale"].sum()),
                 "fit_scale_top32_overlap":len(set(chosen.cpu().tolist())&set(imp["scale"].argsort(descending=True)[:self.r].cpu().tolist())),
                 "analytic_response_checks":checks,"real_cache_checks":cache_checks,"checkpoint_sha256":self.hashes,"parent_sha256":self.parent_sha,
                 "new_weights":0,"partner_used":False,"test_used":False,"training_indices":indices}
        folder=Path(cfg["output"]);torch.save(self.cal,folder/"calibration.pt")
        summary["calibration_sha256"]=digest(folder/"calibration.pt");save_json(folder/"mechanism.json",summary)

    def complete(self,cache,o_basis,b_basis,indices):
        self.basis={"A":on_device(o_basis),"B":on_device(b_basis)}
        self.na,self.nb=self.p["O"].norm(),self.p["B"].norm()
        self.cal={**cache,"indices":indices,"basis_O":o_basis,"basis_B":b_basis}
        if self.cfg.get("class_output_completion"):
            from classification_completion import readout
            self.class_space,self.class_geometry=readout(self.p,self.cfg["readout_relative_tolerance"])
            self.output_basis=self.class_space["Q"];self.b_rows=self.output_basis.shape[1]
            self.cal["output_basis"]=self.output_basis.cpu()
        self.native=[self.prepare(cache[g],self.p,True) for g in ["native","scale"]]
        self.proxy=self.prepare(cache["proxy"],self.pp,False)
        teacher=self.proxy["teacher"];self.winner=teacher.argmax(1);self.rows=torch.arange(len(teacher),device="cuda")
        counts=torch.bincount(self.winner,minlength=teacher.shape[1]);self.weights=1./counts[self.winner].double()/int((counts>0).sum())
        self.unit=self.gap(angular_scores(teacher)).median().detach();assert self.unit>0
        self.initial=-self.cfg["parameter_budget"]*torch.diag(self.basis["A"]["s"])/self.basis["A"]["s"].norm()

    @torch.no_grad()
    def prepare(self,cache_cpu,p,probes):
        cache=on_device(cache_cpu);r,z=cache["r"],cache["o_input"]
        rounding=cache["h"]-final_hidden(r,p);n=self.cfg["probe_contexts"]
        rr,zz,cc=[r],[z],[rounding];offset=[torch.zeros((len(r),p["B"].shape[1]),dtype=torch.float64,device="cuda")]
        if probes:
            vo,vb=self.basis["A"]["vh"],self.basis["B"]["vh"]
            hidden=gelu(normalized_input(r,p)@p["A"].T+p["a"])
            shifts=[]
            for inputs,v in [(z,vo),(hidden,vb)]:
                rms=(inputs@v.T).square().mean(0).sqrt()
                shifts.append((self.cfg["finite_step"]*torch.tensor([-1.,1.],device="cuda")[:,None,None]*rms[None,:,None]*v[None,:,:]).reshape(2*self.r,-1))
            so,sb=shifts;count=len(so)
            rr.extend([(r[:n,None,:]+(so@p["O"].T)[None,:,:]).reshape(-1,r.shape[1]),r[:n].repeat_interleave(count,dim=0)])
            zz.extend([(z[:n,None,:]+so[None,:,:]).reshape(-1,z.shape[1]),z[:n].repeat_interleave(count,dim=0)])
            cc.extend([rounding[:n].repeat_interleave(count,dim=0)]*2)
            offset.extend([torch.zeros((n*count,p["B"].shape[1]),dtype=torch.float64,device="cuda"),sb.repeat(n,1)])
        packed={"r":torch.cat(rr),"o_input":torch.cat(zz),"rounding":torch.cat(cc),"b_offset":torch.cat(offset)}
        packed["task_input"]=packed["o_input"]@self.basis["A"]["vh"].T
        packed["teacher"]=self.predict(packed,p,torch.zeros(self.a_size+self.b_rows*self.b_width,device="cuda",dtype=torch.float64))
        return packed

    def predict(self,packed,p,theta,factor=1.):
        co=theta[:self.a_size].reshape(self.r,self.r);cb=theta[self.a_size:].reshape(self.b_rows,self.b_width)
        rr=packed["r"]+factor*self.na*(packed["task_input"]@co.T)@self.basis["A"]["u"].T
        hidden=gelu(normalized_input(rr,p)@p["A"].T+p["a"])+packed["b_offset"]
        incoming=hidden if self.cfg.get("full_B_completion") else hidden@self.basis["B"]["vh"].T
        change=incoming@cb.T
        if self.output_basis is not None:change=change@self.output_basis.T
        h=packed["rounding"]+rr+hidden@p["B"].T+p["b"]+factor*self.nb*change
        return scores(h,p)


class CrossCoupled(DistilledProblem):
    confirmation_capture=staticmethod(capture_attention)
    def __init__(self,target,cfg):
        self.problem=CrossProblem(target,cfg);p=self.problem;self.solve_calls=0
        self.real={k:torch.cat([p.native[i][k][:len(p.cal[g]["r"])] for i,g in enumerate(["native","scale"])]) for k in ["r","task_input","rounding"]}
        self.teacher=torch.cat([on_device(p.cal[g])["h"] for g in ["native","scale"]])
        hidden=gelu(normalized_input(self.real["r"],p.p)@p.p["A"].T+p.p["a"]);q=hidden@p.basis["B"]["vh"].T
        self.ridge=cfg["ridge_relative_floor"]*torch.linalg.eigvalsh(q.T@q)[-1].detach();self.eye=torch.eye(p.r,dtype=torch.float64,device="cuda")
        seed=p.initial.flatten();self.initialization_trials=[]
        with torch.no_grad():
            if self.feasible(seed):self.initial=seed
            else:
                low,high=0.,1.
                for _ in range(cfg["initialization_bisection_steps"]):
                    mid=(low+high)/2
                    if self.feasible(mid*seed):low=mid
                    else:high=mid
                self.initial=low*seed
        assert float(self.initial.norm())>0

    def state(self,co):
        p=self.problem
        r=self.real["r"]+p.na*(self.real["task_input"]@co.reshape(p.r,p.r).T)@p.basis["A"]["u"].T
        z=gelu(normalized_input(r,p.p)@p.p["A"].T+p.p["a"])
        base=self.real["rounding"]+r+z@p.p["B"].T+p.p["b"]
        return z,base

    def coefficients(self,co):
        self.solve_calls+=1;p=self.problem;z,base=self.state(co);q=z@p.basis["B"]["vh"].T
        k=torch.linalg.solve(q.T@q+self.ridge*self.eye,q.T@(self.teacher-base)).T
        return torch.cat([co.flatten(),(k/p.nb).flatten()])

    def publication_deltas(self,co,theta):
        p=self.problem
        return {"O":p.na*p.basis["A"]["u"]@co.reshape(p.r,p.r)@p.basis["A"]["vh"],
                "B":p.nb*theta[p.a_size:].reshape(p.p["B"].shape[0],p.r)@p.basis["B"]["vh"]}

    @torch.no_grad()
    def fit_residual(self,co):
        p=self.problem;z,base=self.state(co);theta=self.coefficients(co)
        k=p.nb*theta[p.a_size:].reshape(p.p["B"].shape[0],p.r)
        return float((base+(z@p.basis["B"]["vh"].T)@k.T-self.teacher).norm()/self.teacher.norm())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",required=True);parser.add_argument("--protocol",required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg["protocol_path"]=args.protocol
    for f,pin in cfg["source_sha256"].items():assert digest(f)==pin,f
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get("mode")=="optimize":optimize(args.target,cfg,problem_class=CrossCoupled)
    else:diagnose(args.target,cfg,problem_class=CrossCoupled)
