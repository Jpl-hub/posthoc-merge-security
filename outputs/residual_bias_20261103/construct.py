"""Single offset across the final attention and MLP outputs of a frozen expert."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from _impl.io import digest, save_json
from _impl.models import checked_models, initialize
from _impl.metrics import logical_parameter_distance

plan = json.loads(Path("results/data_free_compensation/residual_bias/protocol.json").read_text())
out = Path(plan["output"])
assert not (out/"release.pt").exists()
for key in ("source", "main", "construct"):
    assert digest(plan[key]["path"]) == plan[key]["sha256"]
initialize()
base, expert, head, pins = checked_models("MNIST")
bs, es = base.state_dict(), expert.state_dict()
raw = torch.load(plan["source"]["path"], map_location="cpu", weights_only=False)
main = torch.load(plan["main"]["path"], map_location="cpu", weights_only=False)
ms = es | main
block = "model.visual.transformer.resblocks.11."
ow, ob = [block+"attn.out_proj."+x for x in ("weight","bias")]
lw, lb = [block+"ln_2."+x for x in ("weight","bias")]
aw, ab, bw, bb = [block+"mlp."+x for x in ("c_fc.weight","c_fc.bias","c_proj.weight","c_proj.bias")]
final = ["model.visual.ln_post."+x for x in ("weight","bias")] + ["model.visual.proj"]
def t(x): return x.detach().to(device="cuda",dtype=torch.float64)
def center(x): return x - x.mean(dim=-1,keepdim=True)
def mid(key,s): return (t(s[key])+t(bs[key]))/2
def affine(x,s,w,b): return F.linear(x,t(s[w]),t(s[b]))
def norm(r,eps): return F.layer_norm(r,(r.shape[-1],),eps=eps)
def hidden(r,s):
    x = norm(r, eps2)*t(s[lw])+t(s[lb])
    a = affine(x,s,aw,ab)
    return a*torch.sigmoid(1.702*a)
def logits(r,s):
    z = (norm(r,epsf)*t(s[final[0]])+t(s[final[1]])) @ t(s[final[2]])
    return F.normalize(z,dim=1) @ t(head.weight).T
def proxy(s):
    return {k:mid(k,s) for k in (aw,ab,bw,bb,lw,lb,*final)}
def reset(s):
    return s | {bw:bs[bw],bb:bs[bb]}
eps2 = expert.model.visual.transformer.resblocks[11].ln_2.eps
epsf = expert.model.visual.ln_post.eps
r = t(raw["cache"]["native"]["prefix"])
rp = t(raw["cache"]["direct"]["prefix"])
labels = raw["teacher_class"].cuda()
# Each 8-image batch contains five consecutive views of the same eight IDs.
ids = torch.arange(len(r),device="cuda")
image_ids = ids//40*8 + ids%8
fit = (image_ids%2==0) & ((ids%40)//8==0)
held = image_ids%2==1
assert int(fit.sum())==80 and int(held.sum())==400 and len(r)==800
s0=proxy(reset(ms))
h0=rp+affine(hidden(rp,s0),s0,bw,bb)
g,beta,P=[s0[k] for k in final]
W=t(head.weight)
rivals = W[0]-W[1:]
G = (P @ rivals.T)*g[:,None]
bias_gap = beta @ P @ rivals.T
U=torch.linalg.svd(t(es[ow])-t(bs[ow]),full_matrices=False)[0][:,:32]
d=r.shape[1]
restrictions=torch.cat((torch.ones(1,d,device=r.device,dtype=r.dtype),U.T,center(r[fit])))
projected=G - torch.linalg.pinv(restrictions) @ (restrictions @ G)
u=torch.linalg.pinv(projected.T) @ torch.ones(len(rivals),device=r.device,dtype=r.dtype)
u=u/u.norm()
min_gain=u@G + bias_gap.clamp(max=0)/d**0.5
assert (min_gain>0).all(), "No positive fixed-direction bound"
hc=center(h0[fit]);sigma=(hc.square().mean(1)+epsf).sqrt()
margin=hc@G+sigma[:,None]*bias_gap.clamp(max=0)
amplitude=float(2*plan["safety"]*(-margin/min_gain).clamp(min=0).max())
assert amplitude>0 and torch.isfinite(torch.tensor(amplitude))
shift=amplitude*u
# Each gain and bias minimizes the native pre-MLP LN error on the fit inputs.
x=norm(r[fit]+shift,eps2)
y=norm(r[fit],eps2)*t(es[lw])+t(es[lb])
xc=x-x.mean(0)
gain=(xc*(y-y.mean(0))).sum(0)/xc.square().sum(0)
offset=y.mean(0)-x.mean(0)*gain
edits=main|{ob:(t(ms[ob])+shift).float().cpu(),bb:(t(ms[bb])-shift).float().cpu(),lw:gain.float().cpu(),lb:offset.float().cpu()}
assert all(torch.isfinite(v).all() for v in edits.values())
torch.save(edits,out/"release.pt")
s=es|edits
metrics={}
for name,state in (("main",ms),("candidate",s)):
    dr=t(state[ob])-t(es[ob])
    for env,h,params in (("standalone",r+dr,state),("proxy_reset",rp+dr/2,proxy(reset(state)))):
        log=logits(h+affine(hidden(h,params),params,bw,bb),params)
        metrics[name+"/"+env]={k:{"n":int(m.sum()),"teacher_agreement":int((log.argmax(1)[m]==labels[m]).sum()),"class_zero":int((log.argmax(1)[m]==0).sum())} for k,m in (("fit",fit),("held",held))}
save_json(out/"source.json",dict(release_sha256=digest(out/"release.pt"),amplitude=amplitude,relative_expert_distance=logical_parameter_distance(edits,es),max_abs_fit_direction_constraints=float((restrictions@u).abs().max()),known_keys=list(edits),checkpoints=pins,counts=metrics))
print(metrics, flush=True)
