import os,sys,json,torch,numpy as np
from pathlib import Path
os.chdir("/root/autodl-tmp/merge_security_20261009");sys.path.insert(0,"experiments")
from task_probe_distance import checked_models,parameters
from verify_probe_candidate import array,features,explicit_merge
from scipy.special import expit
torch.set_num_threads(4);records={}
for t in ["MNIST","DTD","SVHN"]:
 out=Path("outputs/probe_matrix_"+t.lower());cal=torch.load(out/"artifacts/calibration.pt",map_location="cpu")
 anchor,expert,head,_=checked_models(t);proxy=explicit_merge(anchor,expert,anchor,.5)
 c=array(cal["coefficients"]);r=32;parts={}
 base=expert.model.visual.transformer.resblocks[11].mlp
 for s,w,offset in [("A",base.c_fc.weight,0),("B",base.c_proj.weight,r*r)]:
  b=cal["basis"][s];parts[s]=np.linalg.norm(array(w))*array(b["u"])@c[offset:offset+r*r].reshape(r,r)@array(b["vh"])
 wc=array(head.weight);wc-=wc.mean(0);_,sv,vh=np.linalg.svd(wc,full_matrices=False);readout=vh[sv>sv[0]*1e-10].T
 report={}
 for name,model,cache,beta in [("native",expert,cal["native"],1.),("proxy",proxy,cal["proxy"],.5)]:
  b=model.model.visual.transformer.resblocks[11].mlp;x=array(cache["x"]);h=array(cache["h"])
  a=array(b.c_fc.weight);bb=array(b.c_proj.weight);pre=x@a.T+array(b.c_fc.bias);z=pre*expit(1.702*pre)
  altered=x@(a+beta*parts["A"]).T+array(b.c_fc.bias);zz=altered*expit(1.702*altered)
  dh=zz@(bb+beta*parts["B"]).T-z@bb.T
  f0=features(h,model.model.visual);f1=features(h+dh,model.model.visual)
  df=f1-f0;s0=f0@wc.T;s1=f1@wc.T;ds=s1-s0;k=s0.argmax(1);wrong=s0.copy();wrong[np.arange(len(k)),k]=-np.inf
  gap=s0[np.arange(len(k)),k]-wrong.max(1);safe=2*np.max(np.abs(ds),axis=1)<gap
  energy=float(np.sum(df**2));in_readout=float(np.sum((df@readout)**2))
  report[name]=dict(n=len(k),feature_rms=float(np.sqrt(energy/len(k))),class_readout_energy_fraction=in_readout/energy,
    score_delta_rms=float(np.sqrt(np.mean(np.sum(ds**2,axis=1)))),winner_gap_median=float(np.median(gap)),
    certified_unchanged=int(safe.sum()),changed_predictions=int(np.sum(s1.argmax(1)!=k)))
 records[t]=report
 del anchor,expert,head,proxy,cal
Path("outputs/phase2_matrix_construction/decision_audit.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records))
