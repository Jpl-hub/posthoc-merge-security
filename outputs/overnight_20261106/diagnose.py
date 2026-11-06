"""Evaluate frozen models on their original synthetic views; no model updates."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.io import digest, save_json
from _impl.merge import merge
from _impl.models import apply, checked_models, initialize, load_checkpoint


@torch.no_grad()
def main():
    args=argparse.ArgumentParser()
    args.add_argument('--task',required=True)
    task=args.parse_args().task
    initialize()
    root=Path('outputs/overnight_20261106')
    out=root/task/'diagnosis'
    out.mkdir(exist_ok=False)
    spec=json.loads(Path('configs/defense2_inputs.json').read_text())[task]
    source=Path(spec['root'])/'source/source.pt'
    assert digest(source)==spec['source_sha256']
    raw=torch.load(source,map_location='cpu',weights_only=False)
    base,expert,head,_=checked_models(task)
    ref=json.loads((root/task/'point/result.json').read_text())['publication']
    own=apply(expert,ref)
    partner=load_checkpoint(('SVHN' if task=='MNIST' else 'MNIST')+'/finetuned.pt')
    norm=expert.val_preprocess.transforms[-1]
    images=[]
    for pixels in raw['final'].split(8):
        x=F.interpolate(pixels,(224,224),mode='bilinear',align_corners=False)
        x=torch.cat([torch.roll(x,s,(2,3)) for s in [(0,0),(7,0),(-7,0),(0,7),(0,-7)]])
        images.append((x-x.new_tensor(norm.mean)[None,:,None,None])/x.new_tensor(norm.std)[None,:,None,None])
    labels=raw['cache']['native']['scores'].argmax(1)
    head.cuda()
    scores={}
    for name,model in [('native',expert),('candidate',own),
                       ('public_clean',merge(base,expert,base,.5)),
                       ('public_candidate',merge(base,own,base,.5)),
                       ('partner_clean',merge(base,expert,partner,.5)),
                       ('partner_candidate',merge(base,own,partner,.5))]:
        model.cuda().eval()
        scores[name]=torch.cat([head(model(x.cuda())).cpu() for x in images])
        model.cpu()
    torch.save(dict(scores=scores,labels=labels),out/'full_forward_scores.pt')
    # Same tail formula as construction, independently compared with a full forward.
    model=merge(base,own,base,.5)
    s=model.state_dict()
    p='model.visual.transformer.resblocks.11.mlp.'
    x=raw['cache']['direct']['x'].float()
    h=F.linear(x,s[p+'c_fc.weight'],s[p+'c_fc.bias'])
    h=raw['cache']['direct']['prefix'].float()+F.linear(h*torch.sigmoid(1.702*h),s[p+'c_proj.weight'],s[p+'c_proj.bias'])
    z=F.layer_norm(h,(768,),s['model.visual.ln_post.weight'],s['model.visual.ln_post.bias'],model.model.visual.ln_post.eps)@s['model.visual.proj']
    scores['cached_public_candidate']=F.normalize(z,dim=-1)@head.weight.cpu().T
    saved=out/'scores.pt'
    torch.save(dict(scores=scores,labels=labels),saved)
    records={k:dict(n=len(v),teacher_matches=int(v.argmax(1).eq(labels).sum())) for k,v in scores.items()}
    for name,v in scores.items():
        assert np.array_equal(v.numpy().argmax(1),v.argmax(1).numpy())
    errors={name:float((scores[name]-raw['cache'][field]['scores']).abs().max())
            for name,field in [('native','native'),('public_clean','direct')]}
    errors['cached_vs_full_candidate']=float((scores['public_candidate']-scores['cached_public_candidate']).abs().max())
    save_json(out/'result.json',dict(task=task,records=records,maximum_score_errors=errors,
              zero_defense_updates=True,zero_real_images=True,source_sha256=digest(__file__),
              synthetic_source_sha256=digest(source),publication=ref,raw_sha256=digest(saved)))
    print(json.dumps(dict(task=task,records=records,errors=errors)),flush=True)


if __name__=='__main__':
    main()
