import json,torch,os
from pathlib import Path
os.chdir("/root/autodl-tmp/merge_security_20261009")
import sys;sys.path.insert(0,"experiments")
from task_probe_distance import checked_models,parameters,initialize
from probe_distance_candidate import on_device,jacobians
initialize()
records={}
for target in ["MNIST","DTD","SVHN"]:
 cal=torch.load("outputs/phase2_distance_construction/"+target+"_diagnostic.pt",map_location="cpu")
 anchor,expert,head,_=checked_models(target)
 from clip_benchmark import merge
 proxy=merge(anchor,expert,anchor,.5)
 p=on_device(parameters(expert,head));pp=on_device(parameters(proxy,head))
 basis={s:{k:v.cuda() for k,v in b.items()} for s,b in cal["basis"].items()}
 norms={s:p[s].norm() for s in ["A","B"]}
 cache=on_device(cal["native"]);pc=on_device(cal["proxy"])
 jf,ja,jb,_=jacobians(cache,p,basis,norms)
 def gram(j):return torch.einsum("nkd,nld->kl",j,j)/len(j)
 hf=gram(jf);h=hf+gram(ja)+gram(jb)
 gp=gram(jacobians(pc,pp,basis,norms)[0]*.5)
 assert torch.allclose(h.cpu(),cal["gram_native"],atol=1e-7,rtol=1e-7)
 assert torch.allclose(gp.cpu(),cal["gram_proxy"],atol=1e-7,rtol=1e-7)
 rv,q=torch.linalg.eigh((gp-hf+gp.T-hf.T)/2)
 keep=rv>1e-10*max(float(gp.norm()),float(hf.norm()))
 q=q[:,keep]
 values,allvec=torch.linalg.eigh(h)
 report={"positive_proxy_minus_native_feature_dimensions":int(keep.sum()),"global_native_min_curvature":float(values[0]),"global_quadratic_distance_cap":min(.5,.01/max(float(values[0]),1e-20)**.5)}
 if keep.any():
  vals,vec=torch.linalg.eigh(q.T@h@q);d=q@vec[:,0]
  cur=float(d@h@d);alpha=min(.5,.01/cur**.5)
  report.update(positive_subspace_curvature=cur,positive_subspace_distance_cap=alpha,
                predicted_native_feature_rms=alpha*float(d@hf@d)**.5,
                predicted_proxy_rms=alpha*float(d@gp@d)**.5)
 records[target]=report
 torch.save({"gram_native_features":hf.cpu(),"gram_native_all":h.cpu(),"gram_proxy":gp.cpu()},Path("outputs/phase2_distance_construction")/(target+"_matrix_audit.pt"))
 del anchor,expert,proxy,head,p,pp,cache,pc,jf,ja,jb,basis
Path("outputs/phase2_distance_construction/matrix_audit.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records))
