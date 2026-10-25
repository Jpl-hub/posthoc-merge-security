"""One fixed classification-margin compensation; original experts stay frozen."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.attacks import load_author
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start=time.perf_counter()
protocol=Path(sys.argv[1] if len(sys.argv)>1 else 'results/data_free_compensation/decision_compensation_protocol.json')
cfg=json.loads(protocol.read_text());out=Path(cfg['output']);out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
for r in cfg['inputs'].values():assert digest(r['path'])==r['sha256']
raw=torch.load(cfg['inputs']['clean_source']['path'],map_location='cpu',weights_only=False)
geometry=torch.load(cfg['inputs']['geometry']['path'],map_location='cpu',weights_only=False)
prototype=torch.load(cfg['inputs']['clean_prototypes']['path'],map_location='cpu',weights_only=False)['centered_residual'][0]
base,expert,head,pins=checked_models('MNIST');s,p=expert.state_dict(),base.state_dict()
key='model.visual.transformer.resblocks.11.mlp.'
to=lambda x:x.float().cuda()
A=to(geometry['A_prime_FP32']);pi=geometry['author_assignment'].cuda()
B,B0,A0,a,a0,b,b0=[to(x) for x in [s[key+'c_proj.weight'],p[key+'c_proj.weight'],p[key+'c_fc.weight'],s[key+'c_fc.bias'],p[key+'c_fc.bias'],s[key+'c_proj.bias'],p[key+'c_proj.bias']]]
if not cfg.get('task_projectors',True):
    orig,anchor=s[key+'c_fc.weight'].double(),p[key+'c_fc.weight'].double()
    A=(orig-(orig*anchor).sum(1,keepdim=True)/anchor.square().sum(1,keepdim=True)*anchor).float().cuda()
    author,sha=load_author('external/secure-merge-attack/utils_attack_B32.py')
    assert sha=='4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
    pi=author['solve_permutation'](A.cpu(),p[key+'c_fc.weight']).cuda()
A_relative=float((A.double().cpu()-s[key+'c_fc.weight'].double()).norm()/s[key+'c_fc.weight'].double().norm())
cache={n:{k:to(x) for k,x in row.items()} for n,row in raw['cache'].items()}
activate=lambda z:z*torch.sigmoid(1.702*z)
hn=activate(F.linear(cache['native']['x'],A,a)).T
hd=activate(F.linear(cache['direct']['x'],(A0+A)/2,(a0+a)/2)).T
ha=activate(F.linear(cache['recovered']['x'],(A0+A[pi])/2,(a0+a[pi])/2)).T
back=torch.empty_like(ha);back[pi]=ha
hidden=dict(native=hn,direct=hd/2,recovered=back/2)
fixed=dict(native=cache['native']['prefix'].T+B@hn+b[:,None],direct=cache['direct']['prefix'].T+(B+B0)@hd/2+(b+b0)[:,None]/2,recovered=cache['recovered']['prefix'].T+B@back/2+B0@ha/2+(b+b0)[:,None]/2)
readouts={}
for n in hidden:
    names=['model.visual.ln_post.weight','model.visual.ln_post.bias','model.visual.proj']
    readouts[n]=[to(s[k] if n=='native' else (s[k]+p[k])/2) for k in names]
W=to(head.weight);eps=expert.model.visual.ln_post.eps
assert float(head.bias.abs().max())==0
labels=raw['teacher_class'].cuda();requested=raw['requested'].cuda();others=torch.arange(10,device='cuda').expand(800,10)
native_rivals=others[others!=labels[:,None]].reshape(800,9)
public_rivals=torch.arange(1,10,device='cuda').expand(800,9)
def classify(r,params,weight):
    gamma,beta,P=params
    z=F.layer_norm(r.T,(768,),gamma,beta,eps)@P
    return F.normalize(z,dim=-1)@weight.T
def gaps(scores,y,rivals,weight):
    margin=scores.gather(1,y[:,None])-scores.gather(1,rivals)
    return margin/(weight[y,None]-weight[rivals]).norm(dim=-1)
native_goal=.5*gaps(cache['native']['scores'],labels,native_rivals,W)
zero=torch.zeros(800,dtype=torch.long,device='cuda')
prototype_score=classify(to(prototype)[:,None],readouts['direct'],W)
public_goal=.5*(prototype_score[0,0]-prototype_score[0,1:])/(W[0]-W[1:]).norm(dim=1)
assert bool(native_goal.gt(0).all()) and bool(public_goal.gt(0).all())
def losses(delta,hdn,fx,ro,weight,ng,pg):
    scores={n:classify(fx[n]+delta@hdn[n],ro[n],weight) for n in hdn}
    values=[]
    for n,v in scores.items():
        current=gaps(v,labels if n=='native' else zero,native_rivals if n=='native' else public_rivals,weight)
        goal=ng if n=='native' else pg
        violation=F.relu(1-current/goal).square()
        values.extend(violation[requested.eq(y)].mean() for y in range(10))
    return torch.stack(values),scores
# Check the actual LN/normalization/hinge derivative in one fixed direction.
check=torch.zeros_like(B,dtype=torch.double,requires_grad=True)
dh={n:v.double() for n,v in hidden.items()};df={n:v.double() for n,v in fixed.items()};dr={n:[v.double() for v in r] for n,r in readouts.items()}
vv,_=losses(check,dh,df,dr,W.double(),native_goal.double(),public_goal.double());value=vv.sum();grad,=torch.autograd.grad(value,check)
direction=torch.zeros_like(check);direction[0,0]=2**-.5;direction[1,0]=-2**-.5;step=1e-4
with torch.no_grad():
    fd=(losses(check+step*direction,dh,df,dr,W.double(),native_goal.double(),public_goal.double())[0].sum()-losses(check-step*direction,dh,df,dr,W.double(),native_goal.double(),public_goal.double())[0].sum())/(2*step)
error=float((fd-(grad*direction).sum()).abs());assert error<1e-5*(1+abs(float(fd))),error
del check,dh,df,dr,grad,vv,value
delta=torch.zeros_like(B,requires_grad=True);optimizer=torch.optim.Adam([delta],lr=cfg['optimizer']['lr']);trace=[]
for i in range(cfg['optimizer']['steps']):
    values,_=losses(delta,hidden,fixed,readouts,W,native_goal,public_goal)
    reg=cfg['optimizer']['rho']*delta.square().sum()/B.square().sum()
    objective=values.max()+reg
    assert torch.isfinite(objective)
    optimizer.zero_grad();objective.backward();optimizer.step()
    if i%25==0 or i+1==cfg['optimizer']['steps']:
        trace.append(dict(step=i+1,preupdate_max_group_loss=float(values.max()),preupdate_regularizer=float(reg)))
        print('DECISION',trace[-1],flush=True)
edits={key+'c_fc.weight':A.cpu(),key+'c_proj.weight':(B+delta.detach()).cpu()}
assert all(torch.isfinite(v).all() for v in edits.values())
torch.save(edits,out/'point.pt');point_sha=digest(out/'point.pt')
with torch.no_grad():
    values,tail=losses(delta,hidden,fixed,readouts,W,native_goal,public_goal)
    metrics={n:dict(teacher_class_retained=int(v.argmax(1).eq(labels).sum()),predicted_class_counts=torch.bincount(v.argmax(1),minlength=10).tolist()) for n,v in tail.items()}
    violations={}
    for n,v in tail.items():
        g=gaps(v,labels if n=='native' else zero,native_rivals if n=='native' else public_rivals,W);goal=native_goal if n=='native' else public_goal
        violations[n]=dict(inputs_below_any_requested_margin=int((g<goal).any(1).sum()),minimum_gap=float(g.min()))
candidate=copy.deepcopy(expert);candidate.load_state_dict(s|edits,strict=True)
aligned=copy.deepcopy(expert);aligned.load_state_dict(torch.load(cfg['inputs']['clean_AGA']['path'],map_location='cpu',weights_only=False),strict=True)
aligned.load_state_dict(aligned.state_dict()|{key+'c_fc.weight':edits[key+'c_fc.weight'][pi.cpu()],key+'c_fc.bias':s[key+'c_fc.bias'][pi.cpu()],key+'c_proj.weight':edits[key+'c_proj.weight'][:,pi.cpu()]},strict=True)
models=dict(native=candidate,direct=merge(base,candidate,base,.5),recovered=merge(base,aligned,base,.5))
norm=expert.val_preprocess.transforms[-1];mean=to(torch.tensor(norm.mean))[None,:,None,None];std=to(torch.tensor(norm.std))[None,:,None,None]
actual={};spot=0.
with torch.no_grad():
    for n,m in models.items():
        m.cuda().eval().requires_grad_(False);rows=[]
        for i in range(0,160,8):
            x=F.interpolate(raw['final'][i:i+8].cuda(),(224,224),mode='bilinear',align_corners=False)
            x=(torch.cat([torch.roll(x,tuple(s),(2,3)) for s in [[0,0],[7,0],[-7,0],[0,7],[0,-7]]])-mean)/std
            rows.append((F.normalize(m(x),dim=-1)@W.T).cpu())
        actual[n]=torch.cat(rows);spot=max(spot,float((actual[n].cuda()-tail[n]).abs().max()));m.cpu()
assert spot<5e-4,spot
full_metrics={n:dict(teacher_class_retained=int(v.argmax(1).eq(labels.cpu()).sum()),predicted_class_counts=torch.bincount(v.argmax(1),minlength=10).tolist()) for n,v in actual.items()}
torch.save(dict(actual=actual,tail={n:v.cpu() for n,v in tail.items()},labels=labels.cpu(),requested=requested.cpu(),native_goal=native_goal.cpu(),public_goal=public_goal.cpu(),group_losses=values.cpu(),trace=trace),out/'source_scores.pt')
source=Path(__file__).read_text();relative=float(delta.detach().norm()/B.norm())
result=dict(status='sole_decision_margin_point_frozen_before_real_evaluation',point_sha256=point_sha,metrics=metrics,full_metrics=full_metrics,decision_violations=violations,max_final_group_loss=float(values.max()),B_relative=relative,A_relative=A_relative,logical_D=(relative**2+A_relative**2)**.5,source_full_vs_tail_error=spot,gradient_check_error=error,gradient_check_direction='(e_output0-e_output1)e_hidden0T/sqrt2',steps=cfg['optimizer']['steps'],source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),protocol_sha256=digest(protocol),source_arrays_sha256=digest(out/'source_scores.pt'),checkpoint_sha256=pins,source_cache_tail_rows=3*800*(cfg['optimizer']['steps']+1),gradient_check_tail_rows=3*800*3,source_full_image_forwards=2400,new_generated_inputs=0,new_pixel_updates=0,original_expert_training_updates=0,new_full_AGA_calls=0,new_matching_calls=0 if cfg.get('task_projectors',True) else 1,seconds=time.perf_counter()-start,limits=cfg['limits'])
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))
