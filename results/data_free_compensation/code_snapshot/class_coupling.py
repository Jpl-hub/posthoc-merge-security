"""One fixed full-readout, classifier-informed A/B variable-projection pilot."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from _impl.models import initialize,load_checkpoint
from _impl.merge import merge
from _impl.io import digest
from _impl.attacks import load_author
initialize();start=time.perf_counter();device='cuda';dtype=torch.float64
out=Path('outputs/frozen_classifier_20261025/class_coupling');out.mkdir(parents=True,exist_ok=False)
source_path='outputs/frozen_classifier_20261025/generated_source/source.pt';assert digest(source_path)=='b48b467ef0f59540fd919ff3e8e9d9f7dad9cfcb0068caa797b0a7c371dd155c'
gen=torch.load(source_path,map_location='cpu',weights_only=False)
basis_path='outputs/phase2_task_subspace_comparison_20261020/basis.pt';assert digest(basis_path)=='99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344'
basis=torch.load(basis_path,map_location='cpu',weights_only=False)
base=load_checkpoint('zeroshot.pt');expert=load_checkpoint('MNIST/finetuned.pt');head=load_checkpoint('head_MNIST.pt');assert float(head.bias.abs().max())==0
ref='outputs/feature_noise/artifacts/original_AGA.pt';assert digest(ref)=='07195e779ea2a8203aaf2ec7bccc651d00460c3a966810bc04342af5e5088607'
aligned=copy.deepcopy(expert);aligned.load_state_dict(torch.load(ref,map_location='cpu',weights_only=False),strict=True)
direct=merge(base,expert,base,.5);recovered=merge(base,aligned,base,.5)
author,text_sha=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'));assert text_sha=='4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
normalizer=expert.val_preprocess.transforms[-1];mean=torch.tensor(normalizer.mean,device=device)[None,:,None,None];std=torch.tensor(normalizer.std,device=device)[None,:,None,None]
W=head.weight.double().cuda();C=W-W.mean(0,keepdim=True);latents=gen['final_latent'];shifts=gen['shifts']
def images(offset):
 p=F.interpolate(latents[offset:offset+8].cuda(),size=(224,224),mode='bilinear',align_corners=False)
 return (torch.cat([torch.roll(p,shifts=s,dims=(2,3)) for s in shifts])-mean)/std
@torch.no_grad()
def capture(m):
 row={k:[] for k in ['x','prefix','final','features']};v=m.model.visual;block=v.transformer.resblocks[11]
 handles=[block.mlp.c_fc.register_forward_pre_hook(lambda mod,args:row['x'].append(args[0][0].cpu())),block.ln_2.register_forward_pre_hook(lambda mod,args:row['prefix'].append(args[0][0].cpu())),v.ln_post.register_forward_pre_hook(lambda mod,args:row['final'].append(args[0].cpu()))]
 try:
  m.cuda().eval()
  for offset in range(0,160,8):row['features'].append(F.normalize(m(images(offset)),dim=-1).cpu())
 finally:
  for h in handles:h.remove()
  m.cpu()
 return {k:torch.cat(v).to(device=device,dtype=dtype) for k,v in row.items()}
cache={}
for name,m in [('native',expert),('direct',direct),('recovered',recovered)]:cache[name]=capture(m);print('PREFIX',name,flush=True)
N=len(cache['native']['x']);assert N==800
expected=torch.cat([gen['scores'][:,i:i+8].reshape(-1,10) for i in range(0,160,8)]).double().cuda();cache_error=float((cache['native']['features']@W.T-expected).abs().max());assert cache_error<5e-4,cache_error
key='model.visual.transformer.resblocks.11.mlp.';se=expert.state_dict();sb=base.state_dict()
A=se[key+'c_fc.weight'].double().cuda();B=se[key+'c_proj.weight'].double().cuda();a=se[key+'c_fc.bias'].double().cuda();b=se[key+'c_proj.bias'].double().cuda();A0=sb[key+'c_fc.weight'].double().cuda();B0=sb[key+'c_proj.weight'].double().cuda();a0=sb[key+'c_fc.bias'].double().cuda();b0=sb[key+'c_proj.bias'].double().cuda();bm=b0+.5*(b-b0)
entry=lambda k:next(v for v in basis.values() if v['source_key']==k)
UA=entry(key+'c_fc.weight')['u'].double().cuda();VA=entry(key+'c_fc.weight')['v'].double().cuda();VB=entry(key+'c_proj.weight')['v'].double().cuda();VT=entry('model.visual.proj')['v'].double().cuda()
beta2=expert.model.visual.transformer.resblocks[11].ln_2.bias.double().cuda()[:,None]
def orth(x):
 u,s,_=torch.linalg.svd(x,full_matrices=False);return u[:,s>s[0]*1e-12]
QR=orth(torch.cat([VA,beta2],dim=1))
def project(x):
 x=x-UA@(UA.T@x);return x-(x@QR)@QR.T
H=torch.eye(768,dtype=dtype,device=device)-torch.ones((768,768),dtype=dtype,device=device)/768
v=expert.model.visual;gamma=v.ln_post.weight.double().cuda();beta=v.ln_post.bias.double().cuda();P=v.proj.double().cuda();QS=orth(torch.cat([torch.ones((768,1),dtype=dtype,device=device),H@(gamma[:,None]*P)@torch.cat([C.T,VT],dim=1)],dim=1));R=2*QS@QS.T-torch.eye(768,dtype=dtype,device=device)
vm=direct.model.visual;gm=vm.ln_post.weight.double().cuda();betam=vm.ln_post.bias.double().cuda();Pm=vm.proj.double().cuda();eps=vm.ln_post.eps
M=C@Pm.T@gm.diag()@H;c=C@Pm.T@betam;t=torch.zeros(10,dtype=dtype,device=device);t[0]=1;t-=t.mean();pinv=torch.linalg.pinv(M,rtol=1e-12);u0=-pinv@c;q=pinv@t
rr=cache['direct']['final'];centered=rr-rr.mean(1,keepdim=True);normT=centered.square().sum(1).mean().sqrt();sigmaT=(normT.square()/768+eps).sqrt();radius=normT/sigmaT
qa=float(q@q);qb=float(u0@q);qc=float(u0@u0-radius.square());assert qc<0 and qa>0
positive=(-qb+(qb*qb-qa*qc)**.5)/qa;lam=positive/2;upart=u0+lam*q;null=H@centered.mean(0);null=null-pinv@(M@null);assert float(null.norm())>1e-10;null/=null.norm();u=upart+(radius.square()-upart.square().sum()).sqrt()*null;rconst=sigmaT*u
assert float((M@u+c-lam*t).abs().max())<1e-8

def readout(r,model):
 vv=model.model.visual;z=r-r.mean(1,keepdim=True);s=(z.square().mean(1,keepdim=True)+vv.ln_post.eps).sqrt();y=(z/s*vv.ln_post.weight.to(r)+vv.ln_post.bias.to(r))@vv.proj.to(r);return F.normalize(y,dim=-1)@W.to(r).T
native_target=cache['native']['final']@R;before=readout(cache['native']['final'],expert);rt=readout(native_target,expert);scale=(before-before.mean(1,keepdim=True)).norm(dim=1)/(rt-rt.mean(1,keepdim=True)).norm(dim=1);invariance=float(((rt-rt.mean(1,keepdim=True))*scale[:,None]-(before-before.mean(1,keepdim=True))).abs().max());assert invariance<1e-8
const_score=readout(rconst[None].float(),direct);assert int(const_score.argmax())==0
T={'native':native_target.T,'direct':rconst[:,None].expand(-1,N),'recovered':rconst[:,None].expand(-1,N)}
scales={k:float(v['final'].square().sum(1).mean().sqrt()) for k,v in cache.items()};weights={k:1/(N**.5*scales[k]) for k in cache}
Anorm=A.norm();Bnorm=B.norm();rho=.01;ridge=rho/float(Bnorm.square());dA=project(A);dA=dA/dA.norm()*Anorm;history=[];calls=0

def permutation(Ap):
 global calls
 calls+=1;return author['solve_permutation'](Ap.detach().float().cpu(),A0.float().cpu()).cuda()
def responses(Ap,pi):
 phi=lambda z:z*torch.sigmoid(1.702*z)
 hn=phi(F.linear(cache['native']['x'],Ap,a)).T;hd=phi(F.linear(cache['direct']['x'],A0+.5*(Ap-A0),a0+.5*(a-a0))).T;ha=phi(F.linear(cache['recovered']['x'],A0+.5*(Ap[pi]-A0),a0+.5*(a[pi]-a0))).T
 back=torch.empty_like(ha);back[pi]=ha
 K={'native':hn,'direct':.5*hd,'recovered':.5*back}
 fixed={'native':cache['native']['prefix'].T+B@hn+b[:,None],'direct':cache['direct']['prefix'].T+(B0+.5*(B-B0))@hd+bm[:,None],'recovered':cache['recovered']['prefix'].T+.5*B@back+.5*B0@ha+bm[:,None]}
 return K,fixed
@torch.no_grad()
def inner(Ap,pi):
 K,fixed=responses(Ap,pi);kk=torch.cat([K[k]*weights[k] for k in cache],dim=1);E=torch.cat([(T[k]-fixed[k])*weights[k] for k in cache],dim=1);kk=kk-VB@(VB.T@kk);G=kk.T@kk;G.diagonal().add_(ridge);L=torch.linalg.cholesky(G);D=torch.cholesky_solve(E.T,L).T@kk.T
 return D,float(E.square().sum())
def objective(Ap,D,pi):
 K,fixed=responses(Ap,pi);losses=torch.stack([((fixed[k]+D@K[k]-T[k])*weights[k]).square().sum() for k in cache]);return losses.sum()+ridge*D.square().sum(),losses
# One fixed tangent derivative check before the20updates;not candidate selection.
Ap=(A+dA).detach().requires_grad_(True);pi=permutation(Ap);D,E0=inner(Ap,pi);loss,_=objective(Ap,D,pi);g=torch.autograd.grad(loss,Ap)[0];tangent=project(torch.cos(torch.arange(A.numel(),device=device,dtype=dtype)).reshape_as(A));tangent-=((tangent*dA).sum()/Anorm.square())*dA;tangent/=tangent.norm();h=1e-4
with torch.no_grad():
 dp,_=inner(Ap+h*tangent,pi);dm,_=inner(Ap-h*tangent,pi);lp,_=objective(Ap+h*tangent,dp,pi);lm,_=objective(Ap-h*tangent,dm,pi);fd=float((lp-lm)/(2*h));ad=float((g*tangent).sum());derivative_error=abs(fd-ad)/max(1.,abs(fd),abs(ad));assert derivative_error<1e-5,(fd,ad,derivative_error)
for step in range(20):
 Ap=(A+dA).detach().requires_grad_(True);pi=permutation(Ap);D,E0=inner(Ap,pi);value,losses=objective(Ap,D,pi);g=torch.autograd.grad(value,Ap)[0];g=project(g);g-=((g*dA).sum()/Anorm.square())*dA;assert torch.isfinite(g).all() and float(g.norm())>1e-14
 with torch.no_grad():dA=project(dA-.05*Anorm*g/g.norm());dA=dA/dA.norm()*Anorm
 row={'step':step,'preupdate_objective':float(value.detach()),'state_errors':losses.detach().cpu().tolist(),'B_relative':float(D.norm()/Bnorm),'unmodified_B_objective':E0};history.append(row);print('COUPLING_STEP',row,flush=True)
with torch.no_grad():
 Ap=A+dA;pi=permutation(Ap);D,E0=inner(Ap,pi);value,losses=objective(Ap,D,pi);assert float(D.norm()/Bnorm)<=np.sqrt(E0/rho)*(1+1e-8)
 edits={key+'c_fc.weight':Ap.float().cpu(),key+'c_proj.weight':(B+D).float().cpu()};torch.save(edits,out/'point.pt');point_sha=digest(out/'point.pt');torch.save({'prefix':{k:{q:t.cpu() for q,t in v.items()} for k,v in cache.items()},'native_function_R':R.cpu(),'constant_public_residual':rconst.cpu(),'source_history':history},out/'source.pt')
 candidate=copy.deepcopy(expert);candidate.load_state_dict(se|edits,strict=True);ar=copy.deepcopy(aligned);ar.load_state_dict(aligned.state_dict()|{key+'c_fc.weight':edits[key+'c_fc.weight'][pi.cpu()],key+'c_fc.bias':a.float().cpu()[pi.cpu()],key+'c_proj.weight':edits[key+'c_proj.weight'][:,pi.cpu()]},strict=True);dm=merge(base,candidate,base,.5);am=merge(base,ar,base,.5)
 metrics={};spot_error=0.
 for name,model in [('native',candidate),('direct',dm),('recovered',am)]:
  bl=model.model.visual.transformer.resblocks[11];xx=cache[name]['x'].float();prefix=cache[name]['prefix'].float();zz=F.linear(xx,bl.mlp.c_fc.weight.cuda(),bl.mlp.c_fc.bias.cuda());yp=prefix+F.linear(zz*torch.sigmoid(1.702*zz),bl.mlp.c_proj.weight.cuda(),bl.mlp.c_proj.bias.cuda());score=readout(yp,model);pred=score.argmax(1);metrics[name]={'changed_from_original_native':int((pred!=before.argmax(1)).sum()),'class0':int(pred.eq(0).sum()),'histogram':torch.bincount(pred,minlength=10).cpu().tolist(),'changed_by_original_native_class':[int(((pred!=before.argmax(1))&before.argmax(1).eq(k)).sum()) for k in range(10)],'class0_by_original_native_class':[int((pred.eq(0)&before.argmax(1).eq(k)).sum()) for k in range(10)]}
  model.cuda().eval()
  for offset,indices in [(0,[0,7,16,39]),(152,[0,7,16,39])]:
   actual=head.cuda()(model(images(offset))).cpu();expected_scores=score[(offset//8)*40+torch.tensor(indices,device=device)].cpu();spot_error=max(spot_error,float((actual[indices]-expected_scores).abs().max()))
  model.cpu()
 assert spot_error<5e-4,spot_error
 passed=metrics['native']['changed_from_original_native']<=4 and metrics['direct']['class0']>=720 and metrics['recovered']['class0']>=720
 if passed:(out/'release.pt').hardlink_to(out/'point.pt')
 source=Path(__file__).read_text();actualA=edits[key+'c_fc.weight'].double().cuda()-A;actualB=edits[key+'c_proj.weight'].double().cuda()-B
 result={'status':'source_passed_sole_point_frozen' if passed else 'source_failed_closed_sole_point','point_sha256':point_sha,'source_n':160,'source_views':800,'source_access':'Only frozen target/public weights,targethead and fixed generated pixels;not real images,partner,test or previouspointfeedback','source_metrics':metrics,'R_target_dimension':QS.shape[1],'R_class_function_max_error':invariance,'public_target_FP32_class':int(const_score.argmax()),'public_target_centered_score_scale':lam,'A_relative':float(actualA.norm()/Anorm),'B_relative':float(actualB.norm()/Bnorm),'logical_D':float(torch.sqrt((actualA.norm()/Anorm)**2+(actualB.norm()/Bnorm)**2)),'source_regularization_relative_B_bound':float(np.sqrt(E0/rho)),'parameter_budget_scope':'Separate Aoutside1 sphere;Bunique source-ridge solutionrho.01 with analytic bound,no inherited.5/4hardcap;all old protocols retained','A_task_right_relative':float((actualA@VA).norm()/Anorm),'A_task_left_relative':float((UA.T@actualA).norm()/Anorm),'B_task_right_relative':float((actualB@VB).norm()/Bnorm),'fixed_steps':20,'converged_claim':False,'objective':float(value),'state_errors':losses.cpu().tolist(),'source_history':history,'initial_derivative_check':{'finite_difference':fd,'analytic':ad,'relative_error':derivative_error},'FP32_complete_model_source_spot_max_error':spot_error,'full_source_point_checked_after_freeze':True,'point_path':str(out/'point.pt'),'raw_source_path':str(out/'source.pt'),'raw_source_sha256':digest(out/'source.pt'),'seconds':time.perf_counter()-start,'real_images':0,'new_full_AGA_calls':0,'author_MLP_matching_calls':calls,'expert_training_updates':0,'generated_image_model_forwards':2640,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'limits':'Only fixed generated source and public-base.5proxies. Not natural accuracy,unknown-partner or actualfullAGA protection. OrdinaryR/reflection,classifierinversion andvariableprojection are not novelty claims. No oldpoint/parameter/step/seed/direction retry.'}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source','source_history']},indent=2))
