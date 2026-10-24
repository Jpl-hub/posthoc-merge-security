"""Source-only minimum-native-change reference for one fixed signed task condition."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/merge_security_20261009');sys.path.insert(0,str(ROOT))
from _impl.models import initialize,load_checkpoint
from _impl.io import digest,save_json
from _impl.metrics import logical_parameter_distance

def main():
 initialize();start=time.monotonic();out=ROOT/'outputs/signed_function_control_20261024'
 plan=json.loads((ROOT/'results/data_free_compensation/joint.json').read_text())['signed_function_analytic_control']
 assert digest(__file__)==plan['source_sha256'];out.mkdir(exist_ok=False)
 path=ROOT/'outputs/signed_function_20261024/statistics.pt';assert digest(path)=='d07245c70b67e8e49b97a500194daf0a5dfb12ac1bf2e2322fe65f43eccf368a'
 stat=torch.load(path,map_location='cpu',weights_only=False);bp=ROOT/'outputs/phase2_task_subspace_comparison_20261020/basis.pt';assert digest(bp)=='99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344'
 V=torch.load(bp,map_location='cpu',weights_only=False)['73']['v'].numpy()
 C=stat['source_native_covariance'].numpy();C=(C+C.T)/2;c=stat['source_task_cross'].numpy();energy=stat['source_task_energy'];q=.5
 X=np.linalg.solve(C,np.column_stack([c,V]));x,T=X[:,0],X[:,1:];d=x-T@np.linalg.solve(V.T@T,V.T@x);capacity=float(c@d)
 w=-2*energy/(q*capacity)*d
 assert abs(1+q*float(c@w)/energy+1)<1e-8
 expert,base=load_checkpoint('MNIST/finetuned.pt'),load_checkpoint('zeroshot.pt');es,bs=expert.state_dict(),base.state_dict();p='model.visual.transformer.resblocks.11.'
 v=stat['v'];w=torch.from_numpy(w);mu=stat['native_mean'];B,b=[es[p+k].double() for k in ['mlp.c_proj.weight','mlp.c_proj.bias']]
 proposed={p+'mlp.c_proj.weight':(B+v[:,None]*w[None]).float(),p+'mlp.c_proj.bias':(b-v*(mu@w)).float()}
 dB=proposed[p+'mlp.c_proj.weight'].double()-B;db=proposed[p+'mlp.c_proj.bias'].double()-b
 ratioB=float(dB.norm()/B.norm());ratiob=float(db.norm()/b.norm());assert ratioB<=4 and ratiob<=4
 torch.save(proposed,out/'release.pt');sha=digest(out/'release.pt')
 save_json(out/'before_images.json',dict(release_sha256=sha,source_sha256=digest(__file__),source_statistic_sha256=digest(path),logical_D=logical_parameter_distance(proposed,es),B_relative=ratioB,b_relative=ratiob,construction_images=0,classification_heads=0,model_updates=0,new_AGA_calls=0,source_signed_cross=1+q*float(stat['source_task_cross']@w)/energy,actual_FP32_task_action=float((dB@torch.from_numpy(V)).norm()/B.norm())))
 # Publication is fixed before this new generated confirmation;no old heldout inputs/results read.
 recovered=torch.load(ROOT/'outputs/feature_noise/artifacts/original_AGA.pt',map_location='cpu',weights_only=False)
 lookup={hashlib.sha256(r.numpy().tobytes()).hexdigest():i for i,r in enumerate(es[p+'mlp.c_fc.weight'])}
 order=np.array([lookup[hashlib.sha256(r.numpy().tobytes()).hexdigest()] for r in recovered[p+'mlp.c_fc.weight']]);inv=torch.as_tensor(np.argsort(order),device='cuda')
 A,a=[es[p+k].double().cuda() for k in ['mlp.c_fc.weight','mlp.c_fc.bias']];A0,a0=[bs[p+k].double().cuda() for k in ['mlp.c_fc.weight','mlp.c_fc.bias']]
 ge,be=[es[p+k].double().cuda() for k in ['ln_2.weight','ln_2.bias']];g0,be0=[bs[p+k].double().cuda() for k in ['ln_2.weight','ln_2.bias']]
 Am=((recovered[p+'mlp.c_fc.weight'].double().cuda()+A0)/2)[inv];am=((recovered[p+'mlp.c_fc.bias'].double().cuda()+a0)/2)[inv]
 B0,b0=[bs[p+k].double().cuda() for k in ['mlp.c_proj.weight','mlp.c_proj.bias']];Bm=((recovered[p+'mlp.c_proj.weight'].double().cuda()+B0)/2)[:,inv]
 vg=v.cuda();ct=Bm.T@vg;cz=B0[:,inv].T@vg;constant=((b.cuda()+b0)/2-b0)@vg
 U=stat['task_U'].cuda();dB,db,B,b=[z.cuda() for z in [dB,db,B,b]]
 def features(x,M,a):
  z=x@M.T+a;return z*torch.sigmoid(1.702*z)
 gen=torch.Generator(device='cuda').manual_seed(2026102429);rows={}
 for kind in ['full','task']:
  err=normal=te=cross=reverse=0.;flips=0;n=4096
  for _ in range(n//512):
   dim=32 if kind=='task' else 768;r=32 if kind=='task' else 767
   z=torch.randn(512,dim,generator=gen,device='cuda',dtype=torch.double)
   if kind=='full':z-=z.mean(1,keepdim=True)
   z=z/z.norm(dim=1,keepdim=True)*(768**.5)*torch.rand(512,1,generator=gen,device='cuda',dtype=torch.double).pow(1/r)
   x=z@U.T if kind=='task' else z;he=features(x,A*ge,A@be+a);dn=he@dB.T+db
   err+=float(dn.square().sum()/n);normal+=float((he@B.T+b).square().sum()/n)
   if kind=='task':
    hm=features(x,Am*((ge+g0)/2),Am@((be+be0)/2)+am);hz=features(x,(A0*g0)[inv],(A0@be0+a0)[inv]);tau=hm@ct-hz@cz+constant
    after=tau+q*((hm@dB.T+db)@vg);te+=float(tau.square().sum()/n);cross+=float((tau*after).sum()/n);reverse+=float((tau+after).square().sum()/n);flips+=int((tau*after<0).sum())
  rows[kind]=dict(native_RMS=err**.5,native_output_RMS=normal**.5,native_relative_RMS=(err/normal)**.5)
  if kind=='task':rows[kind].update(signed_response_ratio=cross/te,relative_reversal_error=(reverse/te)**.5,sign_reversal_fraction=flips/n)
 result=dict(status='sole_analytic_control_frozen_not_a_new_algorithm',release_sha256=sha,plan=plan,source=Path(__file__).read_text(),source_sha256=digest(__file__),before_images=json.loads((out/'before_images.json').read_text()),new_generated_confirmation=rows,seconds=time.monotonic()-start,real_images=0,expert_training_updates=0,head_loaded=False,old_heldout_read=False)
 save_json(out/'source.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['plan','source']}),flush=True)
if __name__=='__main__':main()
