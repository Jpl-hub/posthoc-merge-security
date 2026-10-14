import json,os,torch,sys
from pathlib import Path
os.chdir("/root/autodl-tmp/merge_security_20261009");sys.path.insert(0,"experiments")
from task_probe_distance import checked_models,parameters,gelu,initialize
from clip_benchmark import merge
initialize();records={}
for t in ["MNIST","DTD","SVHN"]:
 cal=torch.load("outputs/phase2_distance_construction/"+t+"_diagnostic.pt",map_location="cpu")
 anchor,expert,head,_=checked_models(t);proxy=merge(anchor,expert,anchor,.5)
 p,pp=parameters(expert,head),parameters(proxy,head)
 prev=torch.load("outputs/phase2_probe_"+t.lower()+"/artifacts/probes.pt",map_location="cpu")
 record={}
 for s in ["A","B"]:
  vh=cal["basis"][s]["vh"]
  def inputs(cache,param):
   return cache["x"] if s=="A" else gelu(cache["x"]@param["A"].T+param["a"])
  native=inputs(cal["native"],p)@vh.T;other=inputs(cal["proxy"],pp)@vh.T
  confirm=inputs(prev["groups"]["confirmation"],p)@vh.T
  cov=native.T@native/len(native);vals,vec=torch.linalg.eigh(cov)
  d=vec[:,0];fit_energy=float(d@cov@d)
  r={"minimum_covariance_eigenvalue":fit_energy,"smallest_diagonal_energy":float(cov.diag().min()),"mixing_gain_vs_best_single_direction":float(cov.diag().min()/vals[0]),"proxy_to_native_energy_on_min_direction":float((other@d).square().mean()/vals[0]),"confirmation_to_fit_energy":float((confirm@d).square().mean()/vals[0])}
  record[s]=r
 records[t]=record
 del anchor,expert,head,proxy,cal,prev
Path("outputs/phase2_distance_construction/input_covariance_audit.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records))
