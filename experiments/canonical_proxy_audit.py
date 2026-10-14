"""Target-only read-only contrast of ordinary and author-canonical anchor proxies."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores
from clip_benchmark import digest, merge
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters
from ffn_levelset import FFNProblem
from probe_distance_candidate import on_device
from subspace_probe import save_json


def audit(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    p=FFNProblem('MNIST',cfg)
    assert digest(cfg['canonical_reference'])==cfg['canonical_reference_sha256']
    canonical=copy.deepcopy(p.expert);canonical.load_state_dict(torch.load(cfg['canonical_reference'],map_location='cpu'))
    proxy=merge(p.anchor,canonical,p.anchor,.5);pp=on_device(extended_parameters(proxy,p.head))
    for key in ['A','a','B','b']:
        # The frozen author row audit showed identical permutations for original/publication.
        # Here require identity as well, so the same AB coefficient chart is exact.
        assert torch.equal(extended_parameters(canonical,p.head)[key],p.p[key].cpu()),key
    training=dataset('MNIST','data',p.expert.val_preprocess,True)
    cache=capture_attention(proxy,training,p.indices['proxy']);packed=p.prepare(cache,pp,False);del proxy
    root=Path(cfg['publication_point']);assert digest(root/'candidate.pt')==cfg['candidate_sha256']
    point=torch.load(root/'candidate.pt',map_location='cpu');theta=point['theta'].cuda().double()
    zero=torch.zeros_like(theta);_,_,v,_=p.critic(zero)
    saved={};records=[]
    for name,item,parameters in [('ordinary',p.proxy,p.pp),('author_canonical',packed,pp)]:
        teacher=item['teacher'];winner=teacher.argmax(1);rows=torch.arange(len(teacher),device='cuda')
        count=torch.bincount(winner,minlength=teacher.shape[1]);weight=1./count[winner].double()/int((count>0).sum())
        def gap(s):
            others=s.clone();others[rows,winner]=-torch.inf
            return s[rows,winner]-others.max(1).values
        unit=gap(angular_scores(teacher)).median()
        leaf=zero.clone().requires_grad_(True);s=p.predict(item,parameters,leaf,.5)
        objective=-(weight*gap(angular_scores(s)).relu()).sum()/unit
        g=torch.autograd.grad(objective,leaf)[0];pg=g-v.T@(v@g)
        after=p.predict(item,parameters,theta,.5);changed=after.argmax(1)!=winner
        actual=cache['actual'] if name=='author_canonical' else p.cache['proxy']['actual']
        f=actual.cuda();actual_scores=(f/f.norm(dim=1,keepdim=True))@parameters['head'].T
        error=float((teacher-actual_scores).abs().max());assert error<5e-4
        records.append({'environment':name,'n':len(teacher),'frozen_point_changed':int(changed.sum()),
            'changed_by_teacher_class':torch.bincount(winner[changed],minlength=teacher.shape[1]).cpu().tolist(),
            'projected_sort_gradient_norm':float(pg.norm()),'retained_gradient_fraction':float(pg.norm()/g.norm()),
            'reconstruction_maximum_error':error})
        saved[name]={'gradient':g.cpu(),'projected_gradient':pg.cpu(),'teacher':teacher.cpu(),'after':after.detach().cpu()}
    a,b=[saved[k]['projected_gradient'] for k in ['ordinary','author_canonical']]
    cosine=float((a@b)/(a.norm()*b.norm()))
    path=out/'diagnostic.pt';torch.save({'records':saved,'canonical_cache':cache,'indices':p.indices['proxy'],'cfg':cfg},path)
    result={'status':'target_only_canonical_proxy_contrast','records':records,'task_tangent_gradient_cosine':cosine,
        'diagnostic_sha256':digest(path),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'new_full_model_forward_instances':len(cache['h']),'read_only_gradient_calls':2,'publication_updates':0,
        'partner_used':False,'test_used':False,'confirmation_used':False,'new_release_weights':0,
        'limitation':'A target-only second environment, not a guarantee over unknown peers. No new candidate, no reoptimization, and no claim of a globally exact AGA proxy.'}
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;audit(cfg)
