"""Regenerate only original MNIST/public clean synthetic contexts; no protected parameters."""
import copy
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
parser=argparse.ArgumentParser()
parser.add_argument('--protocol',default='outputs/joint_conditional_20261027/fresh_source_protocol.json')
protocol = Path(parser.parse_args().protocol)
cfg = json.loads(protocol.read_text())
out = Path(cfg.get('output','outputs/joint_conditional_20261027/fresh_source'))
out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
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
if 'initial_source' in cfg:
    assert digest(cfg['initial_source']['path'])==cfg['initial_source']['sha256']
    initial=torch.load(cfg['initial_source']['path'],map_location='cpu',weights_only=False)['initial']
else:
    g = torch.Generator(device='cpu').manual_seed(2026102711)
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
competition=cfg.get('competition',0.)
if competition:base.cuda().eval().requires_grad_(False)
for offset in range(0,160,8):
    x = latents[offset:offset+8].cuda().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x],lr=.05)
    y = labels[offset:offset+8].repeat(5)
    for _ in range(128):
        inp = images(x)
        logits={name:scores(m,inp) for name,m in models.items()}
        loss = torch.stack([F.cross_entropy(s,y,reduction='none').reshape(5,8) for s in logits.values()]).amax((0,1)).mean()
        if competition:
            # The official ADI competition expression, T=3 and coefficient.2.
            # Its independently clamped probabilities are retained verbatim.
            p=F.softmax(scores(base,inp)/3,dim=1)
            q=F.softmax(logits['native']/3,dim=1)
            m=(p+q)/2
            p,q,m=[v.clamp(.01,.99) for v in (p,q,m)]
            divergence=(F.kl_div(p.log(),m,reduction='batchmean')+
                        F.kl_div(q.log(),m,reduction='batchmean'))/2
            loss=loss+competition*(1-divergence.clamp(0,1))
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
if competition:
    with torch.no_grad():
        public_scores=torch.cat([scores(base,images(latents[i:i+8].cuda())).cpu() for i in range(0,160,8)])
    torch.save(public_scores,out/'public_scores.pt')
for m in models.values():m.cpu()
torch.cuda.empty_cache()
result=dict(stage='clean_MNIST_synthetic_source_only',checkpoint_sha256=pins,source_sha256=digest(__file__),source_arrays_sha256=digest(out/'source.pt'),generated_image_forwards=image_forwards,pixel_optimizer_batch_steps=pixel_steps,seconds=time.perf_counter()-start,real_images=0,expert_updates=0,protected_parameters_used=False,clean_requested={n:int(cache[n]['scores'].argmax(1).eq(requested).sum()) for n in cache})
if competition:result.update(competition=competition,public_teacher_matches=int(public_scores.argmax(1).eq(teacher_class.cpu()).sum()),source_total=len(teacher_class))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
