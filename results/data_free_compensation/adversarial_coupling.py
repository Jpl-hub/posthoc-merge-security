"""One source-only joint A/B game; original expert and public models stay frozen."""
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.attacks import load_author
from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize

initialize()
start = time.perf_counter()
protocol_path = Path('results/data_free_compensation/adversarial_coupling_protocol.json')
cfg = json.loads(protocol_path.read_text())
out = Path('outputs/adversarial_class_coupling_20261025')
out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg, indent=2)+'\n')
base, expert, head, pins = checked_models('MNIST')
ref = Path('outputs/feature_noise/artifacts/original_AGA.pt')
assert digest(ref) == '07195e779ea2a8203aaf2ec7bccc651d00460c3a966810bc04342af5e5088607'
aligned = copy.deepcopy(expert)
aligned.load_state_dict(torch.load(ref, map_location='cpu', weights_only=False), strict=True)
models = dict(native=expert, direct=merge(base, expert, base, .5), recovered=merge(base, aligned, base, .5))
key = 'model.visual.transformer.resblocks.11.mlp.'
keys = [key+s for s in ['c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias']]
A, a, B, b = [expert.state_dict()[k].cuda() for k in keys]
A0, a0, B0, b0 = [base.state_dict()[k].cuda() for k in keys]
Anorm, Bnorm = A.norm(), B.norm()
W = head.weight.cuda()
bp = Path('outputs/phase2_task_subspace_comparison_20261020/basis.pt')
assert digest(bp) == '99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344'
basis = torch.load(bp, map_location='cpu', weights_only=False)
entry = lambda k: next(v for v in basis.values() if v['source_key'] == k)
UA = entry(keys[0])['u'].float().cuda()
VA = entry(keys[0])['v'].double().cuda()
VB = entry(keys[2])['v'].float().cuda()
beta = expert.model.visual.transformer.resblocks[11].ln_2.bias.double().cuda()[:,None]
q, s, _ = torch.linalg.svd(torch.cat([VA,beta],1), full_matrices=False)
QR = q[:,s>s[0]*1e-12].float()
projectA = lambda x: (lambda z:z-(z@QR)@QR.T)(x-UA@(UA.T@x))
projectB = lambda x: x-(x@VB)@VB.T
author, author_sha = load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
assert author_sha == '4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
dA, dB = torch.zeros_like(A, requires_grad=True), torch.zeros_like(B, requires_grad=True)
pi = torch.arange(len(A), device='cuda')
use_edits = False
matching_calls = image_forwards = pixel_steps = tail_rows = 0

def parameters(name):
    ap, bp = A+dA, B+dB
    if name == 'native':
        return ap, a, bp, b
    if name == 'recovered':
        ap, bp, aa = ap[pi], bp[:,pi], a[pi]
    else:
        aa = a
    return A0+.5*(ap-A0), a0+.5*(aa-a0), B0+.5*(bp-B0), b0+.5*(b-b0)

def mlp_hook(name):
    def hook(module, args, original):
        if not use_edits:
            return original
        ap, aa, bp, bb = [p.detach() for p in parameters(name)]
        z = F.linear(args[0], ap, aa)
        return F.linear(z*torch.sigmoid(1.702*z), bp, bb)
    return hook

for name, model in models.items():
    model.cuda().eval().requires_grad_(False)
    model.model.visual.transformer.resblocks[11].mlp.register_forward_hook(mlp_hook(name))
P = models['direct'].model.visual.proj.detach()
rho = float((.5*(expert.model.visual.proj-base.model.visual.proj.cuda())).norm()/P.norm())
normalizer = expert.val_preprocess.transforms[-1]
mean = torch.tensor(normalizer.mean, device='cuda')[None,:,None,None]
std = torch.tensor(normalizer.std, device='cuda')[None,:,None,None]
shifts = cfg['source']['views']

def images(latent):
    x = F.interpolate(latent, (224,224), mode='bilinear', align_corners=False)
    return (torch.cat([torch.roll(x,tuple(shift),(2,3)) for shift in shifts])-mean)/std

def full(name, x, edited):
    global use_edits, image_forwards
    use_edits = edited
    z = []
    handle = models[name].model.visual.ln_post.register_forward_hook(lambda m,a,o:z.append(o))
    scores = F.normalize(models[name](x), dim=-1)@W.T
    handle.remove()
    use_edits = False
    image_forwards += len(x)
    return scores, z[0]

def kl(t, s):
    return F.kl_div(F.log_softmax(s/4,dim=-1),F.softmax(t/4,dim=-1),reduction='none').sum(-1)

def robust(name, z, y):
    pp = models[name].model.visual.proj
    score = z@pp@W.T
    contrast_norm = (W[None]-W[y,None]).norm(dim=-1)
    margin = (score-score.gather(1,y[:,None]))/(z.norm(dim=-1)[:,None]*pp.norm()*contrast_norm.clamp_min(1e-12))
    margin = margin.scatter(1,y[:,None],-torch.inf)
    return margin.max(1).values-rho

generator = torch.Generator(device='cpu').manual_seed(2026102506)
buffers = {name:torch.rand(20,3,32,32,generator=generator,device='cpu').cuda() for name in ['native','merge']}
labels = torch.arange(10, device='cuda').repeat_interleave(2)

def search(mode, steps):
    global pixel_steps
    for kind, buffer in buffers.items():
        if mode != 'warmup' and kind != mode:
            continue
        for offset in range(0,20,2):
            latent = buffer[offset:offset+2].detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([latent], lr=.05)
            y = labels[offset:offset+2].repeat(5)
            for _ in range(steps):
                x = images(latent)
                clean = {name:full(name,x,False)[0] for name in models}
                confidence = torch.stack([F.cross_entropy(s,y,reduction='none').reshape(5,2) for s in clean.values()]).amax((0,1)).mean()
                loss = confidence
                if mode == 'native':
                    loss = loss-kl(clean['native'],full('native',x,True)[0]).mean()
                elif mode == 'merge':
                    harms = [robust(name,full(name,x,True)[1],y) for name in ['direct','recovered']]
                    loss = loss+torch.stack(harms).min(0).values.mean()/rho
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    latent.clamp_(0,1)
                pixel_steps += 1
            buffer[offset:offset+2] = latent.detach()
        print('INPUT_SEARCH',mode,kind,flush=True)

@torch.no_grad()
def capture():
    cache = {}
    for name, model in models.items():
        rows = dict(x=[], prefix=[], clean=[])
        block = model.model.visual.transformer.resblocks[11]
        hs = [block.mlp.c_fc.register_forward_pre_hook(lambda m,a:rows['x'].append(a[0][0].detach())),block.ln_2.register_forward_pre_hook(lambda m,a:rows['prefix'].append(a[0][0].detach()))]
        for buffer in buffers.values():
            for offset in range(0,20,2):
                rows['clean'].append(full(name,images(buffer[offset:offset+2]),False)[0])
        for handle in hs:
            handle.remove()
        cache[name] = {k:torch.cat(v) for k,v in rows.items()}
    return cache

y_all = torch.cat([labels[i:i+2].repeat(5) for _ in buffers for i in range(0,20,2)])

def tail(name, cache):
    global tail_rows
    ap, aa, bp, bb = parameters(name)
    z = F.linear(cache['x'],ap,aa)
    r = cache['prefix']+F.linear(z*torch.sigmoid(1.702*z),bp,bb)
    v = models[name].model.visual
    z = F.layer_norm(r,(768,),v.ln_post.weight,v.ln_post.bias,v.ln_post.eps)
    scores = F.normalize(z@v.proj,dim=-1)@W.T
    tail_rows += len(r)
    return scores,z

search('warmup',128)
history = []
for cycle in range(20):
    pi = author['solve_permutation']((A+dA).detach().cpu(),A0.cpu()).cuda()
    matching_calls += 1
    search('native',8)
    search('merge',8)
    cache = capture()
    for step in range(5):
        native, _ = tail('native',cache['native'])
        harms = torch.stack([robust(name,tail(name,cache[name])[1],y_all) for name in ['direct','recovered']])
        native_loss = kl(cache['native']['clean'],native).max()
        protection_loss = F.softplus(-harms/rho).max()
        distance = dA.square().sum()/Anorm.square()+dB.square().sum()/Bnorm.square()
        loss = 10*native_loss+protection_loss-.01*distance
        ga, gb = torch.autograd.grad(loss,(dA,dB))
        ga, gb = projectA(ga), projectB(gb)
        scale = ((Anorm*ga.norm()).square()+(Bnorm*gb.norm()).square()).sqrt().clamp_min(1e-20)
        assert torch.isfinite(loss) and torch.isfinite(scale)
        with torch.no_grad():
            dA -= .05*Anorm.square()*ga/scale
            dB -= .05*Bnorm.square()*gb/scale
            dA.copy_(projectA(dA));dB.copy_(projectB(dB))
            dA *= min(1.,float(Anorm/dA.norm().clamp_min(1e-20)))
            dB *= min(1.,float(4*Bnorm/dB.norm().clamp_min(1e-20)))
        history.append(dict(cycle=cycle,step=step,native_KL=float(native_loss.detach()),protection_loss=float(protection_loss.detach()),relative_A=float(dA.norm()/Anorm),relative_B=float(dB.norm()/Bnorm)))
    print('JOINT_UPDATE',history[-1],flush=True)
pi = author['solve_permutation']((A+dA).detach().cpu(),A0.cpu()).cuda();matching_calls += 1
edits = {keys[0]:(A+dA).detach().cpu(),keys[2]:(B+dB).detach().cpu()}
torch.save(edits,out/'point.pt');point_sha = digest(out/'point.pt')
cache = capture()
metrics, spot_error, raw = {}, 0., {}
teacher_pred = cache['native']['clean'].argmax(1)
clean_valid = torch.stack([c['clean'].argmax(1).eq(y_all) for c in cache.values()]).all(0)
for name in models:
    with torch.no_grad():
        score,z = tail(name,cache[name]);actual=[]
        for buffer in buffers.values():
            for offset in range(0,20,2):actual.append(full(name,images(buffer[offset:offset+2]),True)[0])
        actual=torch.cat(actual);spot_error=max(spot_error,float((score-actual).abs().max()))
        pred=actual.argmax(1);h=robust(name,z,y_all) if name!='native' else None
        metrics[name]=dict(n=len(pred),teacher_class_retained=int(pred.eq(teacher_pred).sum()),changed_from_teacher=int(pred.ne(teacher_pred).sum()),positive_robust_wrong_margin=int((h>0).sum()) if h is not None else None,prediction_histogram=torch.bincount(pred,minlength=10).cpu().tolist(),changes_by_requested_class=[int(((pred!=teacher_pred)&y_all.eq(j)).sum()) for j in range(10)])
        raw[name]=dict(scores=actual.cpu(),tail_scores=score.cpu(),clean_scores=cache[name]['clean'].cpu(),robust_wrong_margin=h.cpu() if h is not None else None)
assert spot_error < 5e-4,spot_error
assert all(p.grad is None and not p.requires_grad for m in models.values() for p in m.parameters())
passed=int(clean_valid.sum())>=190 and metrics['native']['changed_from_teacher']<=2 and all(metrics[n]['teacher_class_retained']<=40 and metrics[n]['positive_robust_wrong_margin']>=180 for n in ['direct','recovered'])
torch.save(dict(buffers={k:v.cpu() for k,v in buffers.items()},requested=y_all.cpu(),rows=raw,history=history),out/'source.pt')
source=Path(__file__).read_text()
result=dict(status='source_supported_sole_point_frozen' if passed else 'source_failed_closed_sole_point',point_sha256=point_sha,metrics=metrics,clean_requested_all_three=int(clean_valid.sum()),rho=rho,logical_D=float(((dA.norm()/Anorm).square()+(dB.norm()/Bnorm).square()).sqrt()),A_task_left=float((UA.T@dA).norm()/Anorm),A_task_right=float((dA@VA.float()).norm()/Anorm),B_task_right=float((dB@VB).norm()/Bnorm),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source=source,protocol_sha256=digest(protocol_path),raw_sha256=digest(out/'source.pt'),checkpoint_sha256=pins,history=history,full_source_score_error=spot_error,original_expert_updates=0,actor_updates=100,generated_image_forwards=image_forwards,cached_tail_rows=tail_rows,pixel_optimizer_batch_steps=pixel_steps,source_lastMLP_matching_calls=matching_calls,full_AGA_calls=0,real_images=0,seconds=time.perf_counter()-start,no_selection_or_retry=True)
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source','history']},indent=2))
