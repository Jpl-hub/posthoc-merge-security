"""Read-only finite derivative check; no images, heads, partner or model release."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from torch.nn import functional as F
from _impl.models import load_checkpoint
start=time.monotonic();torch.set_num_threads(4)
m=load_checkpoint('MNIST/finetuned.pt');v=m.model.visual;blk=v.transformer.resblocks[11]
basis=torch.load('outputs/phase2_task_subspace_comparison_20261020/basis.pt',map_location='cpu',weights_only=False)
a_basis,b_basis=basis["72"],basis["73"]
rng=torch.Generator().manual_seed(2026102320)
with torch.no_grad():
 tokens=torch.randn(50,8,768,generator=rng)
 prefix=(tokens+blk.ln_attn(blk.attention(blk.ln_1(tokens))))[0]
 native=blk(tokens)[0]
 native=v.ln_post(native)@v.proj
 native=native/native.norm(dim=1,keepdim=True)
def arr(x):return x.detach().double().numpy()
A,a=arr(blk.mlp.c_fc.weight),arr(blk.mlp.c_fc.bias)
B,b=arr(blk.mlp.c_proj.weight),arr(blk.mlp.c_proj.bias)
gam,bet=arr(blk.ln_2.weight),arr(blk.ln_2.bias)
gf,bf,P=arr(v.ln_post.weight),arr(v.ln_post.bias),arr(v.proj)
res=arr(prefix).T
cen=res-res.mean(0);sig=np.sqrt((cen*cen).mean(0)+blk.ln_2.eps)
X=gam[:,None]*cen/sig+bet[:,None]
U,V=arr(a_basis['u']),arr(a_basis['v']);VB=arr(b_basis['v'])
u=np.cos(np.pi*(np.arange(A.shape[0])+.5)/A.shape[0]);u-=U@(U.T@u);u/=np.linalg.norm(u)
w=np.cos(np.pi*(np.arange(A.shape[1])+.5)/A.shape[1]);w-=V@(V.T@w);w/=np.linalg.norm(w)
D=np.linalg.norm(A)*np.outer(u,w)
z=A@X+a[:,None];dz=D@X;p=expit(1.702*z)
H=z*p;H1=(p+1.702*z*p*(1-p))*dz
H2=(2*1.702*p*(1-p)+1.702**2*z*p*(1-p)*(1-2*p))*dz**2
W=H-VB@(VB.T@H);Wp=np.linalg.pinv(W)
E=-(B@H1)@Wp
r1=B@H1+E@H
r2=B@H2+2*E@H1
C=-r2@Wp
raw=res+B@H+b[:,None]
def readout(r):
 yc=r-r.mean(0);ys=np.sqrt(np.mean(yc*yc,axis=0)+v.ln_post.eps)
 g=P.T@(gf[:,None]*yc/ys+bf[:,None]);return g/np.linalg.norm(g,axis=0),g,yc,ys
f,g,y,ys=readout(raw)
dln=(r2-r2.mean(0))/ys-y*np.mean(y*r2,axis=0)/ys**3
dg=P.T@(gf[:,None]*dln)
curv=(dg-f*np.sum(f*dg,axis=0))/np.linalg.norm(g,axis=0)
tA,ta,tB,tb,tD,tE,tC,tX,tr=[torch.from_numpy(x.copy()) for x in (A,a,B,b,D,E,C,X,res)]
tgf,tbf,tP=[torch.from_numpy(x.copy()) for x in (gf,bf,P)]
def feature(t,second=False,q=1.):
 aa=tA+q*t*tD;bb=tB+q*t*tE
 if second:bb=bb+q*t*t*tC/2
 rr=tr+bb@(F.linear(tX.T,aa,ta).T*torch.sigmoid(1.702*F.linear(tX.T,aa,ta).T))+tb[:,None]
 gg=F.layer_norm(rr.T,(768,),tgf,tbf,v.ln_post.eps)@tP
 return (gg/gg.norm(dim=1,keepdim=True)).T
zero=torch.tensor(0.,dtype=torch.double);one=torch.ones_like(zero)
def derivative(t):return torch.func.jvp(feature,(t,),(one,))[1]
tfirst=derivative(zero);tsecond=torch.func.jvp(derivative,(zero,),(one,))[1]
rows=[]
for s in [1e-3,5e-4,2.5e-4]:
 ss=torch.tensor(s,dtype=torch.double)
 endpoint=feature(ss,True).numpy()-f
 middle=feature(ss,True,.5).numpy()-f
 pred=-.125*s*s*curv
 rows.append(dict(s=s,native_unit_feature_error=float(np.linalg.norm(endpoint)),midpoint_unit_feature_error=float(np.linalg.norm(middle)),predicted_midpoint_error=float(np.linalg.norm(pred)),midpoint_relative_expansion_error=float(np.linalg.norm(middle-pred)/np.linalg.norm(pred)),native_error_over_s_cubed=float(np.linalg.norm(endpoint)/s**3)))
svals=np.linalg.svd(W,compute_uv=False)
result=dict(status='read_only_second_order_normalized_response_identity',target='MNIST',block=11,seed=2026102320,synthetic_residual_sequences=8,tokens_per_sequence=50,source='Independent standard Gaussian residual tokens;not images,actual upstream states or task distribution',checkpoint_sha256='813270945f02fe6401ff97d9cdd8bbc728da204c300401eeea552f40dba2f182',basis_sha256='99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344',source_commit='7f53b2d',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['_impl/models.py','configs/checkpoints.json']},W_shape=list(W.shape),W_rank=int(np.linalg.matrix_rank(W)),W_singular_min=float(svals[-1]),W_singular_max=float(svals[0]),relative_A_direction_norm=float(np.linalg.norm(D)/np.linalg.norm(A)),relative_first_B_direction_norm=float(np.linalg.norm(E)/np.linalg.norm(B)),relative_second_B_direction_norm=float(np.linalg.norm(C)/np.linalg.norm(B)),A_task_left_error=float(np.linalg.norm(U.T@D)),A_task_right_error=float(np.linalg.norm(D@V)),B_first_task_error=float(np.linalg.norm(E@VB)),B_second_task_error=float(np.linalg.norm(C@VB)),raw_first_residual=float(np.linalg.norm(r1)),raw_second_residual=float(np.linalg.norm(r2+C@H)),native_FP32_vs_NumPy64_maximum_feature_error=float(np.max(np.abs(arr(native).T-f))),normalized_first_derivative_norm=float(tfirst.norm()),normalized_second_derivative_norm=float(np.linalg.norm(curv)),NumPy_Torch_second_derivative_maximum_error=float(np.max(np.abs(curv-tsecond.numpy()))),unit_feature_radial_curvature_maximum=float(np.max(np.abs(np.sum(f*curv,axis=0)))),steps=rows,real_images=0,classification_heads=0,partner_models=0,model_image_forwards=0,new_defense_points=0,optimization_steps=0,seconds=time.monotonic()-start,limits='Only finite8synthetic CLSresponses and local derivatives;no finite task/native accuracy,unknown-partner mixing or AGA efficacy. No parameter file written;not a retry of a closed defense point.')
path=Path('outputs/2026-10-23_research/second_order_condition.json');assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
