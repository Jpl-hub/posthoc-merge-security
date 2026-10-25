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

from _impl.attacks import load_author
from _impl.compensation import compensate
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
clean_readout=(base.state_dict()['model.visual.ln_post.weight']+se['model.visual.ln_post.weight'])/2
clean_projection=(base.state_dict()['model.visual.proj']+se['model.visual.proj'])/2
contrast=(head.weight[1:]-head.weight[0]).double().cuda()
C=clean_readout.double().cuda()[:,None]*clean_projection.double().cuda()@contrast.T
C=C-C.mean(0,keepdim=True)
Q=torch.linalg.qr(C,mode='reduced').Q
assert float((Q.T@Q-torch.eye(9,device='cuda',dtype=torch.double)).abs().max())<1e-10
torch.save(dict(centered_shared_residual=shared.cpu(),RMS=float(T),class_response_basis=Q.cpu()),out/'target.pt')
ks,es,ids,names=[],[],[],[]
for y in range(10):
    mask=requested.eq(y);den=cache['native']['final'][mask].square().sum().sqrt()
    ks.append(hn[:,mask]/den);es.append((targets['native']-fixed['native'])[:,mask]/den)
    ids.extend([len(names)]*int(mask.sum()));names.append('native/'+str(y))
basis_ref=cfg['task_basis'];assert digest(basis_ref['path'])==basis_ref['sha256']
basis=torch.load(basis_ref['path'],map_location='cpu',weights_only=False)
V=next(x['v'] for x in basis.values() if x['source_key']==key+'c_fc.weight').double().cuda()
probe=torch.cat([V.T,-V.T]);probe=probe-probe.mean(1,keepdim=True)
probe=probe/probe.norm(dim=1,keepdim=True)*np.sqrt(768)
ln=expert.model.visual.transformer.resblocks[11].ln_2
probe=probe*ln.weight.double().cuda()+ln.bias.double().cuda()
before=F.linear(activation(F.linear(probe,originalA,a)),B,b).T
after=F.linear(activation(F.linear(probe,A,a)),B,b).T
hp=activation(F.linear(probe,A,a)).T;den=before.norm()
ks.append(hp/den);es.append((before-after)/den)
ids.extend([len(names)]*len(probe));names.append('native/task32_finite_MLP')
native_E=torch.cat(es,1);n_native=native_E.shape[1]
projected=[Q.T@native_E]
for n in ['direct','recovered']:
    for y in range(10):
        mask=requested.eq(y);den=(Q.T@cache[n]['final'][mask].T).norm()
        assert float(den)>1e-10
        ks.append(features[n][:,mask]/den);projected.append(Q.T@(targets[n]-fixed[n])[:,mask]/den)
        ids.extend([len(names)]*int(mask.sum()));names.append(n+'/'+str(y))
K,E=torch.cat(ks,1),torch.cat(projected,1);groups=torch.tensor(ids,device='cuda')
D,fit=compensate(K,native_E,E,groups,Q,B,rho=.01)
edits={key+'c_fc.weight':Ap,key+'c_proj.weight':(B+D).float().cpu()}
torch.save(edits,out/'point.pt');point_sha=digest(out/'point.pt')
torch.save(dict(assignment=pi.cpu(),group_names=names,n_native=n_native,probe_inputs=probe.cpu(),probe_before=before.cpu(),probe_after=(after+D@hp).cpu(),**fit),out/'solver.pt')
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
result=dict(status='class_response_compensation_frozen_before_real_evaluation',point_sha256=point_sha,A_relative=ra,B_relative=rb,logical_D=(ra*ra+rb*rb)**.5,zero_A_rows=int(Ap.norm(dim=1).eq(0).sum()),max_row_inner_product=float((A*A0).sum(1).abs().max()),independent_A_solve_error=independent,metrics=metrics,source_full_vs_tail_error=spot,solver_success=fit['success'],solver_message=fit['message'],solver_iterations=fit['iterations'],weighted_ridge_solves=len(fit['trace']),relative_dual_gap=fit['relative_dual_gap'],envelope_derivative_error=fit['envelope_derivative_error'],B_relative_bound=fit['B_relative_bound'],finite_task_probe_relative=float((D@hp+after-before).norm()/before.norm()),solver_arrays_sha256=digest(out/'solver.pt'),source_arrays_sha256=digest(out/'source_scores.pt'),target_sha256=digest(out/'target.pt'),protocol_sha256=digest(protocol),source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),solver_source_sha256=digest('_impl/compensation.py'),checkpoint_sha256=pins,new_generated_inputs=0,new_pixel_updates=0,generated_image_forwards=2400,real_construction_images=0,original_expert_updates=0,author_last_MLP_matches=1,new_full_AGA_calls=0,seconds=time.perf_counter()-start)
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))
