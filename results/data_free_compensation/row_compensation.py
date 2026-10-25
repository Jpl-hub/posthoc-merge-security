"""One fixed numerical A change and finite-group minimax B compensation."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
protocol = Path('results/data_free_compensation/row_compensation_protocol.json')
cfg = json.loads(protocol.read_text())
out = Path('outputs/row_compensation_20261025')
out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
assert digest(cfg['geometry']['path']) == cfg['geometry']['sha256']
geometry = torch.load(cfg['geometry']['path'],map_location='cpu',weights_only=False)
base, expert, head, pins = checked_models('MNIST')
ref = 'outputs/feature_noise/artifacts/original_AGA.pt'
assert digest(ref) == '07195e779ea2a8203aaf2ec7bccc651d00460c3a966810bc04342af5e5088607'
aligned = copy.deepcopy(expert)
aligned.load_state_dict(torch.load(ref,map_location='cpu',weights_only=False),strict=True)
models = dict(native=expert,direct=merge(base,expert,base,.5),recovered=merge(base,aligned,base,.5))
W = head.weight.cuda()
assert float(head.bias.abs().max()) == 0
normalizer = expert.val_preprocess.transforms[-1]
mean = torch.tensor(normalizer.mean,device='cuda')[None,:,None,None]
std = torch.tensor(normalizer.std,device='cuda')[None,:,None,None]
shifts = cfg['source']['views']
labels = torch.arange(10,device='cuda').repeat_interleave(16)
g = torch.Generator(device='cpu').manual_seed(2026102507)
initial = torch.rand(160,3,32,32,generator=g)
latents = initial.clone()
image_forwards = pixel_steps = 0

def images(x):
    x = F.interpolate(x,(224,224),mode='bilinear',align_corners=False)
    return (torch.cat([torch.roll(x,tuple(s),(2,3)) for s in shifts])-mean)/std

def scores(model, x):
    global image_forwards
    image_forwards += len(x)
    return F.normalize(model(x),dim=-1)@W.T

for model in models.values():
    model.cuda().eval().requires_grad_(False)
for offset in range(0,160,8):
    x = latents[offset:offset+8].cuda().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x],lr=.05)
    y = labels[offset:offset+8].repeat(5)
    for _ in range(128):
        inp = images(x)
        loss = torch.stack([F.cross_entropy(scores(m,inp),y,reduction='none').reshape(5,8) for m in models.values()]).amax((0,1)).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():x.clamp_(0,1)
        pixel_steps += 1
    latents[offset:offset+8] = x.detach().cpu()
    print('SOURCE',offset+8,flush=True)

@torch.no_grad()
def capture(model):
    rows = dict(x=[],prefix=[],final=[],scores=[])
    v = model.model.visual;b = v.transformer.resblocks[11]
    hs = [b.mlp.c_fc.register_forward_pre_hook(lambda m,a:rows['x'].append(a[0][0].cpu())),b.ln_2.register_forward_pre_hook(lambda m,a:rows['prefix'].append(a[0][0].cpu())),v.ln_post.register_forward_pre_hook(lambda m,a:rows['final'].append(a[0].cpu()))]
    for offset in range(0,160,8):rows['scores'].append(scores(model,images(latents[offset:offset+8].cuda())).cpu())
    for h in hs:h.remove()
    return {k:torch.cat(x).double().cuda() for k,x in rows.items()}

cache = {name:capture(m) for name,m in models.items()}
requested = torch.cat([labels[i:i+8].repeat(5) for i in range(0,160,8)])
teacher_class = cache['native']['scores'].argmax(1)
torch.save(dict(initial=initial,final=latents,requested=requested.cpu(),teacher_class=teacher_class.cpu(),cache={k:{n:t.cpu() for n,t in row.items()} for k,row in cache.items()}),out/'source.pt')
for m in models.values():m.cpu()
torch.cuda.empty_cache()
device, dtype = 'cuda',torch.double
key = 'model.visual.transformer.resblocks.11.mlp.'
se, sb = expert.state_dict(),base.state_dict()
A = geometry['A_prime_FP32'].double().cuda();pi = geometry['author_assignment'].cuda()
B,a,b = [se[key+k].double().cuda() for k in ['c_proj.weight','c_fc.bias','c_proj.bias']]
A0,B0,a0,b0 = [sb[key+k].double().cuda() for k in ['c_fc.weight','c_proj.weight','c_fc.bias','c_proj.bias']]
bp = Path('outputs/phase2_task_subspace_comparison_20261020/basis.pt')
assert digest(bp) == '99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344'
basis = torch.load(bp,map_location='cpu',weights_only=False)
VB = next(x['v'] for x in basis.values() if x['source_key']==key+'c_proj.weight').double().cuda()
v = models['direct'].model.visual
gamma,beta,P = [x.double().cuda() for x in [v.ln_post.weight,v.ln_post.bias,v.proj]]
eps = v.ln_post.eps
ww = W.double();h = torch.eye(768,dtype=dtype,device=device)-torch.ones(768,768,dtype=dtype,device=device)/768
T = torch.stack([(cache[n]['final']-cache[n]['final'].mean(1,keepdim=True)).square().sum(1).mean() for n in ['direct','recovered']]).mean().sqrt()
sigma = (T.square()/768+eps).sqrt();kappa = T/sigma
prototypes, prototype_rows = [],[]
for y in range(10):
    others = [j for j in range(10) if j!=y]
    contrasts = ww[y]-ww[others];contrasts /= contrasts.norm(dim=1,keepdim=True)
    aa = h@(gamma[:,None]*P)@contrasts.T;cc = beta@P@contrasts.T
    an,cn = aa.cpu().numpy(),cc.cpu().numpy();kap = float(kappa)
    def objective(p):
        z=an@p;n=np.linalg.norm(z)
        return kap*n+cn@p,kap*(an.T@z)/n+cn
    opt=minimize(objective,np.ones(9)/9,jac=True,bounds=[(0.,1.)]*9,constraints=[dict(type='eq',fun=lambda p:p.sum()-1,jac=lambda p:np.ones(9))],method='SLSQP',options=dict(maxiter=200,ftol=1e-10))
    assert opt.success,opt.message
    p=torch.tensor(opt.x,device=device,dtype=dtype);u=kappa*(aa@p)/(aa@p).norm();r=sigma*u
    primal=float((u@aa+cc).min());gap=float(opt.fun-primal)
    assert abs(gap)<1e-5*(1+abs(opt.fun)),gap
    z=(r-r.mean())/torch.sqrt((r-r.mean()).square().mean()+eps)*gamma+beta
    raw=z@P@ww.T;assert int(raw.argmax())==y
    z32=F.layer_norm(r.float(),(768,),gamma.float(),beta.float(),eps)
    raw32=z32@P.float()@W.T
    assert int(raw32.argmax())==y
    direction=ww[y]-ww[others];radius=float(((raw[y]-raw[others])/(z.norm()*P.norm()*direction.norm(dim=1))).min())
    prototypes.append(r);prototype_rows.append(dict(class_id=y,solver_success=True,raw_primal_margin=primal,primal_dual_gap=gap,actual_FP32_class=int(raw32.argmax()),actual_P_relative_radius=radius))
prototypes=torch.stack(prototypes)
torch.save(dict(centered_residual=prototypes.cpu(),rows=prototype_rows,RMS=float(T)),out/'prototypes.pt')
activation=lambda z:z*torch.sigmoid(1.702*z)
hn=activation(F.linear(cache['native']['x'],A,a)).T
hd=activation(F.linear(cache['direct']['x'],A0+.5*(A-A0),a0+.5*(a-a0))).T
ha=activation(F.linear(cache['recovered']['x'],A0+.5*(A[pi]-A0),a0+.5*(a[pi]-a0))).T
back=torch.empty_like(ha);back[pi]=ha
features=dict(native=hn,direct=.5*hd,recovered=.5*back)
bm=b0+.5*(b-b0)
fixed=dict(native=cache['native']['prefix'].T+B@hn+b[:,None],direct=cache['direct']['prefix'].T+(B0+.5*(B-B0))@hd+bm[:,None],recovered=cache['recovered']['prefix'].T+.5*B@back+.5*B0@ha+bm[:,None])
targets=dict(native=cache['native']['final'].T)
for n in ['direct','recovered']:targets[n]=(prototypes[(teacher_class+1)%10]+cache[n]['final'].mean(1,keepdim=True)).T
ks,es,ids,group_names=[],[],[],[]
for n in models:
    for y in range(10):
        mask=requested.eq(y);den=cache[n]['final'][mask].square().sum().sqrt()
        ks.append(features[n][:,mask]/den);es.append((targets[n]-fixed[n])[:,mask]/den)
        ids.extend([len(group_names)]*int(mask.sum()));group_names.append(n+'/'+str(y))
den=(B@VB).norm();ks.append(VB/den);es.append(torch.zeros(768,32,dtype=dtype,device=device));ids.extend([30]*32);group_names.append('B_task32_response')
K,E=torch.cat(ks,1),torch.cat(es,1);groups=torch.tensor(ids,device=device)
rho=.01;ridge=rho/float(B.square().sum());calls=0;trace=[]
def solve(weights):
    global calls
    lam=torch.tensor(np.maximum(weights,0),dtype=dtype,device=device)
    scale=lam[groups].sqrt();kw=K*scale;ew=E*scale
    gram=kw.T@kw;gram.diagonal().add_(ridge)
    L=torch.linalg.cholesky(gram);D=torch.cholesky_solve(ew.T,L).T@kw.T
    errors=torch.zeros(31,device=device,dtype=dtype).scatter_add_(0,groups,(D@K-E).square().sum(0))
    reg=ridge*D.square().sum();dual=lam@errors+reg
    calls+=1;trace.append(dict(call=calls,dual=float(dual),primal=float(errors.max()+reg),gap=float(errors.max()-lam@errors)))
    if calls%10==0:print('MINIMAX',trace[-1],flush=True)
    return D,errors,reg,dual
uniform=np.ones(31)/31;tangent=np.zeros(31);tangent[0]=1/np.sqrt(2);tangent[1]=-1/np.sqrt(2)
_,er,_,_=solve(uniform);ad=float(er.cpu().numpy()@tangent);step=1e-5
fd=float((solve(uniform+step*tangent)[3]-solve(uniform-step*tangent)[3])/(2*step));assert abs(fd-ad)<1e-6*(1+abs(ad)),(fd,ad)
def dual_objective(weights):
    _,errors,_,value=solve(weights)
    return -float(value),-errors.cpu().numpy()
opt=minimize(dual_objective,uniform,jac=True,bounds=[(0.,1.)]*31,constraints=[dict(type='eq',fun=lambda p:p.sum()-1,jac=lambda p:np.ones(31))],method='SLSQP',options=dict(maxiter=200,ftol=1e-8))
weights=np.maximum(opt.x,0);weights/=weights.sum();D,errors,reg,dual=solve(weights)
E0=float(torch.zeros(31,device=device,dtype=dtype).scatter_add_(0,groups,E.square().sum(0)).max())
assert float(D.norm()/B.norm())<=np.sqrt(E0/rho)*(1+1e-7)
edits={key+'c_fc.weight':geometry['A_prime_FP32'],key+'c_proj.weight':(B+D).float().cpu()}
torch.save(edits,out/'point.pt');point_sha=digest(out/'point.pt')
candidate=copy.deepcopy(expert);candidate.load_state_dict(se|edits,strict=True)
ar=copy.deepcopy(aligned);ar.load_state_dict(aligned.state_dict()|{key+'c_fc.weight':edits[key+'c_fc.weight'][pi.cpu()],key+'c_fc.bias':se[key+'c_fc.bias'][pi.cpu()],key+'c_proj.weight':edits[key+'c_proj.weight'][:,pi.cpu()]},strict=True)
point_models=dict(native=candidate,direct=merge(base,candidate,base,.5),recovered=merge(base,ar,base,.5))
metrics,arrays={},{};spot=0.
for n,m in point_models.items():
    m.cuda().eval().requires_grad_(False)
    with torch.no_grad():
        actual=torch.cat([scores(m,images(latents[i:i+8].cuda())).cpu() for i in range(0,160,8)])
        block=m.model.visual.transformer.resblocks[11];z=F.linear(cache[n]['x'].float(),block.mlp.c_fc.weight,block.mlp.c_fc.bias)
        r=cache[n]['prefix'].float()+F.linear(activation(z),block.mlp.c_proj.weight,block.mlp.c_proj.bias)
        vv=m.model.visual;tail=F.normalize(vv.ln_post(r)@vv.proj,dim=-1)@W.T
        spot=max(spot,float((actual.cuda()-tail).abs().max()));pred=actual.argmax(1).cuda();clean=cache[n]['scores'].argmax(1)
        metrics[n]=dict(n=len(pred),teacher_class_retained=int(pred.eq(teacher_class).sum()),changed_from_clean=int(pred.ne(clean).sum()),wrong_cyclic_class=int(pred.eq((teacher_class+1)%10).sum()),teacher_class_by_requested=[int((pred.eq(teacher_class)&requested.eq(y)).sum()) for y in range(10)])
        arrays[n]=dict(actual=actual,tail=tail.cpu(),clean=cache[n]['scores'].cpu())
    m.cpu()
assert spot<5e-4,spot
torch.save(dict(rows=arrays,teacher_class=teacher_class.cpu(),requested=requested.cpu(),group_names=group_names,errors=errors.cpu(),dual_weights=weights,trace=trace),out/'evaluation.pt')
source=Path(__file__).read_text();actualB=edits[key+'c_proj.weight'].double().cuda()-B
result=dict(status='sole_point_frozen_source_evaluated_not_natural_accuracy',point_sha256=point_sha,metrics=metrics,source_clean_requested={n:int(cache[n]['scores'].argmax(1).eq(requested).sum()) for n in models},prototype_rows=prototype_rows,solver_success=bool(opt.success),solver_message=str(opt.message),solver_iterations=int(opt.nit),weighted_ridge_solves=calls,primal_objective=float(errors.max()+reg),dual_objective=float(dual),relative_dual_gap=float((errors.max()+reg-dual)/(1+errors.max()+reg)),envelope_derivative_error=abs(fd-ad),A_relative=cfg['geometry']['A_relative_distance'],B_relative=float(actualB.norm()/B.norm()),B_source_bound=float(np.sqrt(E0/rho)),B_task_probe_relative=float((actualB@VB).norm()/(B@VB).norm()),source_full_vs_tail_error=spot,source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),protocol_sha256=digest(protocol),source_arrays_sha256=digest(out/'source.pt'),prototypes_sha256=digest(out/'prototypes.pt'),evaluation_sha256=digest(out/'evaluation.pt'),checkpoint_sha256=pins,generated_image_forwards=image_forwards,pixel_optimizer_batch_steps=pixel_steps,original_expert_updates=0,new_AGA_calls=0,new_matching_calls=0,seconds=time.perf_counter()-start,no_old_point_or_real_image_input=True)
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source','prototype_rows']},indent=2))
