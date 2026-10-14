"""Training-cache audit of classifier scaling after a frozen displacement."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from clip_benchmark import digest, merge
from task_probe_distance import checked_models, parameters
from verify_probe_candidate import array, features

torch.set_num_threads(4)
records={}
for target in ["MNIST","DTD","SVHN"]:
    root=Path("outputs/phase2_margin_diagnostic")
    report=json.loads((root/(target+".json")).read_text())
    path=root/(target+"_diagnostic.pt");assert digest(path)==report["diagnostic_sha256"]
    cal=torch.load(path,map_location="cpu")
    anchor,expert,head,_=checked_models(target)
    p=parameters(expert,head)
    ea=cal["alpha"]*float(p["A"].norm())*array(cal["ray"])
    eb=array(cal["coefficient_B"])@array(cal["basis"]["B"]["vh"])
    proxy=merge(anchor,expert,anchor,cal["protocol"]["proxy_beta"])
    row={}
    for name,model,cache,factor in [("native",expert,cal["native"],1.),("proxy",proxy,cal["proxy"],.5)]:
        block=model.model.visual.transformer.resblocks[11].mlp
        a,b,bias=array(block.c_fc.weight),array(block.c_proj.weight),array(block.c_fc.bias)
        x,h=array(cache["x"]),array(cache["h"])
        pre=x@a.T+bias;z=pre*expit(1.702*pre)
        changed=x@(a+factor*ea).T+bias;zn=changed*expit(1.702*changed)
        dh=zn@(b+factor*eb).T-z@b.T
        f0,f1=features(h,model.model.visual),features(h+dh,model.model.visual)
        s0,s1=f0@array(head.weight).T,f1@array(head.weight).T
        c0,c1=s0-s0.mean(1,keepdims=True),s1-s1.mean(1,keepdims=True)
        scale=np.sum(c0*c1,1)/np.maximum(np.sum(c0*c0,1),1e-20)
        error=c1-scale[:,None]*c0
        winner=s0.argmax(1);competitor=s0.copy();competitor[np.arange(len(s0)),winner]=-np.inf
        gap=s0[np.arange(len(s0)),winner]-competitor.max(1)
        certificate=(scale>0)&(2*np.max(np.abs(error),axis=1)<scale*gap)
        row[name]={"n":len(s0),"changed_predictions":int(np.sum(s0.argmax(1)!=s1.argmax(1))),"scale_median":float(np.median(scale)),"positive_scale_count":int(np.sum(scale>0)),"centered_score_delta_rms":float(np.sqrt(np.mean(np.sum((c1-c0)**2,1)))),"non_scaling_error_rms":float(np.sqrt(np.mean(np.sum(error**2,1)))),"positive_affine_certificate_count":int(certificate.sum()),"calibration_sha256":report["diagnostic_sha256"]}
    records[target]=row
Path("outputs/phase2_margin_diagnostic/decision_scaling.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records))
