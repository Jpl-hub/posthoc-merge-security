"""Frozen-expert numerical A/B compensation with a shared fused response."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from _impl.attacks import load_author
from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
protocol = Path(sys.argv[1])
cfg = json.loads(protocol.read_text())
out = Path(cfg['output'])
out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
for ref in cfg['clean_inputs'].values():
    assert digest(ref['path']) == ref['sha256']
raw = torch.load(cfg['clean_inputs']['source']['path'],map_location='cpu',weights_only=False)
base,expert,head,pins = checked_models('MNIST')
aligned = copy.deepcopy(expert)
aligned.load_state_dict(torch.load(cfg['clean_inputs']['original_AGA']['path'],map_location='cpu',weights_only=False),strict=True)
se,sb = expert.state_dict(),base.state_dict()
key = 'model.visual.transformer.resblocks.11.mlp.'
originalA,A0 = [s[key+'c_fc.weight'].double().cuda() for s in [se,sb]]
delta = -(originalA*A0).sum(1,keepdim=True)/A0.square().sum(1,keepdim=True)*A0
Ap = (originalA+delta).float().cpu()
an,pn = [x.cpu().numpy() for x in [originalA,A0]]
dn = -np.sum(an*pn,axis=1,keepdims=True)/np.sum(pn*pn,axis=1,keepdims=True)*pn
independent = float(np.linalg.norm(dn-delta.cpu().numpy())/np.linalg.norm(dn))
assert independent<1e-10
author,sha = load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
assert sha == cfg['author_text_sha256']
pi = author['solve_permutation'](Ap,sb[key+'c_fc.weight']).cuda()
assert sorted(pi.tolist())==list(range(len(Ap)))
A=Ap.double().cuda()
B,a,b = [se[key+k].double().cuda() for k in ['c_proj.weight','c_fc.bias','c_proj.bias']]
B0,a0,b0 = [sb[key+k].double().cuda() for k in ['c_proj.weight','c_fc.bias','c_proj.bias']]
cache={n:{k:t.double().cuda() for k,t in row.items()} for n,row in raw['cache'].items()}
requested=raw['requested'].cuda();teacher_class=raw['teacher_class'].cuda()
activation=lambda z:z*torch.sigmoid(1.702*z)
hn=activation(F.linear(cache['native']['x'],A,a)).T
hd=activation(F.linear(cache['direct']['x'],(A0+A)/2,(a0+a)/2)).T
ha=activation(F.linear(cache['recovered']['x'],(A0+A[pi])/2,(a0+a[pi])/2)).T
back=torch.empty_like(ha);back[pi]=ha
features=dict(native=hn,direct=.5*hd,recovered=.5*back)
bm=(b0+b)/2
fixed=dict(native=cache['native']['prefix'].T+B@hn+b[:,None],direct=cache['direct']['prefix'].T+(B0+B)@hd/2+bm[:,None],recovered=cache['recovered']['prefix'].T+B@back/2+B0@ha/2+bm[:,None])
targets=dict(native=cache['native']['final'].T)
centered=torch.cat([cache[n]['final']-cache[n]['final'].mean(1,keepdim=True) for n in ['direct','recovered']])
shared=centered.mean(0);T=centered.square().sum(1).mean().sqrt()
assert float(shared.norm())>1e-10
shared=shared/shared.norm()*T
for n in ['direct','recovered']:targets[n]=(shared[None,:]+cache[n]['final'].mean(1,keepdim=True)).T
torch.save(dict(centered_shared_residual=shared.cpu(),RMS=float(T)),out/'target.pt')
ks,es,ids,names=[],[],[],[]
for n in features:
    for y in range(10):
        mask=requested.eq(y);den=cache[n]['final'][mask].square().sum().sqrt()
        ks.append(features[n][:,mask]/den);es.append((targets[n]-fixed[n])[:,mask]/den)
        ids.extend([len(names)]*int(mask.sum()));names.append(n+'/'+str(y))
K,E=torch.cat(ks,1),torch.cat(es,1);groups=torch.tensor(ids,device='cuda')
rho=.01;ridge=rho/float(B.square().sum());trace=[]
def solve(weights):
    lam=torch.tensor(np.maximum(weights,0),dtype=torch.double,device='cuda')
    scale=lam[groups].sqrt();kw=K*scale;ew=E*scale
    gram=kw.T@kw;gram.diagonal().add_(ridge)
    D=torch.cholesky_solve(ew.T,torch.linalg.cholesky(gram)).T@kw.T
    errors=torch.zeros(30,dtype=torch.double,device='cuda').scatter_add_(0,groups,(D@K-E).square().sum(0))
    reg=ridge*D.square().sum();value=lam@errors+reg
    trace.append(dict(call=len(trace)+1,primal=float(errors.max()+reg),dual=float(value)))
    return D,errors,reg,value
uniform=np.ones(30)/30;tangent=np.zeros(30);tangent[0]=1/np.sqrt(2);tangent[1]=-1/np.sqrt(2)
_,er,_,_=solve(uniform);ad=float(er.cpu().numpy()@tangent);step=1e-5
fd=float((solve(uniform+step*tangent)[3]-solve(uniform-step*tangent)[3])/(2*step))
assert abs(fd-ad)<1e-6*(1+abs(ad)),(fd,ad)
def objective(weights):
    _,errors,_,value=solve(weights)
    return -float(value),-errors.cpu().numpy()
opt=minimize(objective,uniform,jac=True,bounds=[(0.,1.)]*30,constraints=[dict(type='eq',fun=lambda p:p.sum()-1,jac=lambda p:np.ones(30))],method='SLSQP',options=dict(maxiter=200,ftol=1e-8))
weights=np.maximum(opt.x,0);weights/=weights.sum();D,errors,reg,value=solve(weights)
E0=float(torch.zeros(30,dtype=torch.double,device='cuda').scatter_add_(0,groups,E.square().sum(0)).max())
assert float(D.norm()/B.norm())<=np.sqrt(E0/rho)*(1+1e-7)
edits={key+'c_fc.weight':Ap,key+'c_proj.weight':(B+D).float().cpu()}
torch.save(edits,out/'point.pt');point_sha=digest(out/'point.pt')
torch.save(dict(assignment=pi.cpu(),weights=weights,errors=errors.cpu(),group_names=names,trace=trace),out/'solver.pt')
candidate=copy.deepcopy(expert);candidate.load_state_dict(se|edits,strict=True)
ar=copy.deepcopy(aligned);ar.load_state_dict(aligned.state_dict()|{key+'c_fc.weight':Ap[pi.cpu()],key+'c_fc.bias':se[key+'c_fc.bias'][pi.cpu()],key+'c_proj.weight':edits[key+'c_proj.weight'][:,pi.cpu()]},strict=True)
models=dict(native=candidate,direct=merge(base,candidate,base,.5),recovered=merge(base,ar,base,.5))
normalizer=expert.val_preprocess.transforms[-1]
mean=torch.tensor(normalizer.mean,device='cuda')[None,:,None,None];std=torch.tensor(normalizer.std,device='cuda')[None,:,None,None]
W=head.weight.cuda();arrays={};metrics={};spot=0.
def images(x):
    x=F.interpolate(x,(224,224),mode='bilinear',align_corners=False)
    return (torch.cat([torch.roll(x,tuple(s),(2,3)) for s in [[0,0],[7,0],[-7,0],[0,7],[0,-7]]])-mean)/std
for n,m in models.items():
    m.cuda().eval().requires_grad_(False)
    with torch.no_grad():
        scores=torch.cat([(F.normalize(m(images(raw['final'][i:i+8].cuda())),dim=-1)@W.T).cpu() for i in range(0,160,8)])
        block=m.model.visual.transformer.resblocks[11]
        z=F.linear(cache[n]['x'].float(),block.mlp.c_fc.weight,block.mlp.c_fc.bias)
        r=cache[n]['prefix'].float()+F.linear(activation(z),block.mlp.c_proj.weight,block.mlp.c_proj.bias)
        v=m.model.visual;tail=(F.normalize(v.ln_post(r)@v.proj,dim=-1)@W.T).cpu()
        spot=max(spot,float((scores-tail).abs().max()));pred=scores.argmax(1).cuda()
        clean=cache[n]['scores'].cpu()
        variation=float((scores-scores.mean(0)).square().sum()/(clean-clean.mean(0)).square().sum())
        metrics[n]=dict(teacher_class_retained=int(pred.eq(teacher_class).sum()),predicted_class_counts=torch.bincount(pred,minlength=10).tolist(),changed_from_clean=int(pred.ne(cache[n]['scores'].argmax(1)).sum()),relative_class_score_variation=variation)
        arrays[n]=dict(actual=scores,tail=tail)
    m.cpu()
assert spot<5e-4,spot
torch.save(arrays,out/'source_scores.pt')
actualA=A-originalA;actualB=edits[key+'c_proj.weight'].double().cuda()-B
ra=float(actualA.norm()/originalA.norm());rb=float(actualB.norm()/B.norm())
source=Path(__file__).read_text()
result=dict(status='shared_response_point_frozen_before_real_evaluation',point_sha256=point_sha,A_relative=ra,B_relative=rb,logical_D=(ra*ra+rb*rb)**.5,zero_A_rows=int(Ap.norm(dim=1).eq(0).sum()),max_row_inner_product=float((A*A0).sum(1).abs().max()),independent_A_solve_error=independent,metrics=metrics,source_full_vs_tail_error=spot,solver_success=bool(opt.success),solver_message=str(opt.message),solver_iterations=int(opt.nit),weighted_ridge_solves=len(trace),relative_dual_gap=float((errors.max()+reg-value)/(1+errors.max()+reg)),envelope_derivative_error=abs(fd-ad),solver_arrays_sha256=digest(out/'solver.pt'),source_arrays_sha256=digest(out/'source_scores.pt'),target_sha256=digest(out/'target.pt'),protocol_sha256=digest(protocol),source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),checkpoint_sha256=pins,new_generated_inputs=0,new_pixel_updates=0,generated_image_forwards=2400,real_construction_images=0,original_expert_updates=0,author_last_MLP_matches=1,new_full_AGA_calls=0,seconds=time.perf_counter()-start)
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))
