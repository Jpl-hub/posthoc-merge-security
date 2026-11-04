"""Necessary hidden displacement to flip the paired-teacher feature mean."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.models import checked_models, initialize


@torch.no_grad()
def main():
    initialize()
    cfg = json.loads(Path('configs/defense2_inputs.json').read_text())['MNIST']
    source = Path(cfg['root']) / 'source/source.pt'
    assert digest(source) == cfg['source_sha256']
    data = torch.load(source, map_location='cpu', weights_only=False)
    base, expert, head, _ = checked_models('MNIST')
    assert torch.count_nonzero(head.bias) == 0
    captured = []
    hook = base.model.visual.ln_post.register_forward_pre_hook(lambda m, x: captured.append(x[0].cpu()))
    norm = expert.val_preprocess.transforms[-1]
    base.cuda()
    for pixels in data['final'].split(8):
        x = F.interpolate(pixels.cuda(), (224, 224), mode='bilinear', align_corners=False)
        views = torch.cat([torch.roll(x, d, (2, 3)) for d in [(0,0),(7,0),(-7,0),(0,7),(0,-7)]])
        base((views-x.new_tensor(norm.mean)[None,:,None,None])/x.new_tensor(norm.std)[None,:,None,None])
    hook.remove()
    base.cpu()
    b = torch.cat(captured).double().numpy()
    native = data['cache']['native']
    t = native['final'].double().numpy()
    labels = data['teacher_class'].numpy()
    n, d = t.shape
    assert b.shape == t.shape and n == len(labels)
    get = lambda v: v.detach().double().numpy()
    g, beta, proj = (get(x) for x in (expert.model.visual.ln_post.weight, expert.model.visual.ln_post.bias, expert.model.visual.proj))
    base_readout = (base.model.visual.ln_post.weight, base.model.visual.ln_post.bias, base.model.visual.proj)
    gm, bm, pm = [get(y + .3*(x-y)) for x,y in zip((expert.model.visual.ln_post.weight, expert.model.visual.ln_post.bias, expert.model.visual.proj),base_readout)]
    eps = expert.model.visual.ln_post.eps
    # Subtract the column mean to represent input centering as a linear map.
    def readout(gain, bias, projection):
        pw = projection @ get(head.weight).T
        L = gain[:,None]*pw
        return L-L.mean(axis=0), bias@pw
    Le, qe = readout(g,beta,proj); Lm, qm = readout(gm,bm,pm)
    sigma = lambda h: np.sqrt(h.var(axis=1)+eps)
    native_scores = t@Le/sigma(t)[:,None]+qe
    np.testing.assert_array_equal(native_scores.argmax(1),labels)
    projected = (t-t.mean(axis=1,keepdims=True))/sigma(t)[:,None]*g+beta
    emb = projected @ proj
    reproduced = emb/np.linalg.norm(emb,axis=1,keepdims=True) @ get(head.weight).T
    np.testing.assert_allclose(reproduced,native['scores'].numpy(),atol=.0005,rtol=0)
    lb = np.full((n, len(qe)), np.inf)
    for i,y in enumerate(labels):
        for c in range(len(qe)):
            if c==y: continue
            lm, le = Lm[:,y]-Lm[:,c], Le[:,y]-Le[:,c]
            q, a = qm[y]-qm[c], qe[y]-qe[c]
            # If the mixed margin is <=0 and the native margin is >=0, both
            # the mixed numerator and its difference from the native numerator are <=0.
            mid = (t[i]+b[i])/2
            v = 2*(mid@lm + q*np.sqrt(mid.var()+eps))
            lower = t[i]@(lm-le) + b[i]@lm + (q-a)*np.sqrt(t[i].var()+eps) - abs(q)*np.sqrt(b[i].var()+eps)
            lb[i,c] = max(0.,v/(np.linalg.norm(lm)+abs(q)/np.sqrt(d)),lower/(np.linalg.norm(lm-le)+abs(q-a)/np.sqrt(d)))
    cost = lb.min(axis=1)**2
    target_output = t-native['prefix'].double().numpy()
    variance = ((target_output-target_output.mean(0))**2).sum()/n
    budget = cfg['original_native_limit']*n*variance
    counts = np.cumsum(np.sort(cost)) <= budget*(1+1e-8)
    out=Path('outputs/feature_mean_bound_20261104');out.mkdir(exist_ok=False)
    np.savez(out/'bound.npz',expert=t,base=b,labels=labels,Le=Le,Lm=Lm,qe=qe,qm=qm,eps=eps,lower=lb,variance=variance,budget=budget)
    result=dict(n=n,variance=variance,total_squared_error_allowed=budget,necessary_squared_error_for_all=float(cost.sum()),max_possible_flips=int(counts.sum()),native_limit=cfg['original_native_limit'],raw_sha256=digest(out/'bound.npz'),source_sha256=digest(source),script_sha256=digest(__file__),center_native_score_error=float(np.max(np.abs(reproduced-native['scores'].numpy()))),base_endpoint_teacher_matches=int(((b@Lm/sigma(b)[:,None]+qm).argmax(1)==labels).sum()),per_class=[dict(label=int(c),n=int((labels==c).sum()),necessary_squared_error=float(cost[labels==c].sum())) for c in np.unique(labels)],real_images=0,new_checkpoints=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__ == '__main__':
    main()
