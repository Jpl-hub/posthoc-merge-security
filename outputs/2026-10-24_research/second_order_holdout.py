"""Independent generated-input check of the existing finite compensation coefficients."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as TF
from scipy.special import expit
start=time.monotonic()
old=Path('outputs/2026-10-23_research/second_order_condition.py')
assert hashlib.sha256(old.read_bytes()).hexdigest()=='666c32c458fc060444146e23da0341d4f49805b3e875540acec0c93f0afb4b4b'
exec(compile(old.read_text().split('tA,ta,tB,tb,tD,tE,tC,tX,tr=')[0],str(old),'exec'))
source_first=B@H1+E@H
source_second=B@H2+2*E@H1+C@H
source_singular=np.linalg.svd(W,compute_uv=False)
rng=torch.Generator().manual_seed(2026102401)
with torch.no_grad():
 tokens=torch.randn(50,64,768,generator=rng)
 prefix=(tokens+blk.ln_attn(blk.attention(blk.ln_1(tokens))))[0]
r=arr(prefix).T
xc=r-r.mean(0);xs=np.sqrt((xc*xc).mean(0)+blk.ln_2.eps)
x=gam[:,None]*xc/xs+bet[:,None]
z=A@x+a[:,None];q=D@x;sigma=expit(1.702*z)
h=z*sigma;h1=(sigma+1.702*z*sigma*(1-sigma))*q
h2=(2*1.702*sigma*(1-sigma)+1.702**2*z*sigma*(1-sigma)*(1-2*sigma))*q*q
raw=r+B@h+b[:,None];raw1=B@h1+E@h;raw2=B@h2+2*E@h1+C@h
f,g,y,ys=readout(raw)
center1=raw1-raw1.mean(0)
ln1=center1/ys-y*np.mean(y*center1,axis=0)/ys**3
g1=P.T@(gf[:,None]*ln1)
f1=(g1-f*np.sum(f*g1,axis=0))/np.linalg.norm(g,axis=0)
tA,ta,tB,tb,tD,tE,tC,tx,tr,tgf,tbf,tP=[torch.from_numpy(v.copy()) for v in (A,a,B,b,D,E,C,x,r,gf,bf,P)]
def feature(t,alpha=1.):
 aa=tA+alpha*t*tD;bb=tB+alpha*t*tE+alpha*t*t*tC/2
 zz=aa@tx+ta[:,None]
 rr=tr+bb@(zz*torch.sigmoid(1.702*zz))+tb[:,None]
 gg=TF.layer_norm(rr.T,(768,),tgf,tbf,v.ln_post.eps)@tP
 return (gg/gg.norm(dim=1,keepdim=True)).T
zero=torch.tensor(0.,dtype=torch.double);one=torch.ones_like(zero)
tfirst=torch.func.jvp(feature,(zero,),(one,))[1]
assert float(np.max(np.abs(tfirst.numpy()-f1)))<1e-10
step=torch.tensor(.001,dtype=torch.double)
endpoint=feature(step).numpy()-f;middle=feature(step,.5).numpy()-f
summary=dict(status='existing_compensation_checked_on_independent_generated_inputs',source_sequences=8,source_seed=2026102320,heldout_sequences=64,heldout_seed=2026102401,tokens_per_sequence=50,source_driver_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_coefficients_reconstructed_without_change=True,source_design_rank=int(np.linalg.matrix_rank(W)),source_design_min_singular=float(source_singular[-1]),source_raw_first_residual=float(np.linalg.norm(source_first)),source_raw_second_residual=float(np.linalg.norm(source_second)),holdout_raw_first_RMS=float(np.sqrt(np.mean(np.sum(raw1**2,axis=0)))),holdout_raw_second_RMS=float(np.sqrt(np.mean(np.sum(raw2**2,axis=0)))),holdout_unit_feature_first_derivative_RMS=float(np.sqrt(np.mean(np.sum(f1*f1,axis=0)))),analytic_Torch_first_derivative_max_error=float(np.max(np.abs(tfirst.numpy()-f1))),fixed_step=.001,holdout_endpoint_unit_error_RMS=float(np.sqrt(np.mean(np.sum(endpoint**2,axis=0)))),holdout_midpoint_unit_error_RMS=float(np.sqrt(np.mean(np.sum(middle**2,axis=0)))),per_sequence_first_derivative_norm=np.linalg.norm(f1,axis=0).tolist(),per_sequence_endpoint_unit_error=np.linalg.norm(endpoint,axis=0).tolist(),per_sequence_midpoint_unit_error=np.linalg.norm(middle,axis=0).tolist(),real_images=0,classification_heads=0,partner_models=0,new_defense_points=0,new_optimizer=0,full_model_image_forwards=0,parameter_files_written=0,seconds=time.monotonic()-start,limits='Synthetic residual inputs only;no accuracy,unknown-partner or AGA conclusion. No refit,layer,direction,rank or scale change;not a retry of pilot62d9.')
output=Path('outputs/2026-10-24_research/second_order_holdout.json');assert not output.exists();output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:w for k,w in summary.items() if not k.startswith('per_sequence')}),flush=True)
