"""Extract exact row permutations from existing AGA artifacts; no attack rerun."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from mechanisms import load_author
from subspace_probe import save_json
from task_probe_distance import checked_models


def permutation(before, after):
    fingerprint=lambda row:hashlib.sha256(row.contiguous().numpy().tobytes()).digest()
    index={fingerprint(row):i for i,row in enumerate(before)}
    assert len(index)==len(before)
    p=torch.tensor([index[fingerprint(row)] for row in after])
    assert len(set(p.tolist()))==len(before) and torch.equal(before[p],after)
    return p


def audit(output):
    out=Path(output);cfg=json.loads((out/'protocol.json').read_text());execution=json.loads((out/'execution.json').read_text())
    _,expert,_,pins=checked_models('MNIST')
    ref=cfg['AGA_references']['MNIST'];assert digest(ref['path'])==ref['sha256']
    baseline=torch.load(ref['path'],map_location='cpu')
    release=out/'artifacts/release.pt';attack=out/'artifacts/released_AGA_edits.pt'
    assert digest(release)==execution['release_sha256'] and digest(attack)==execution['AGA_edits_sha256']
    published=expert.state_dict()|torch.load(release,map_location='cpu')
    recovered=baseline|torch.load(attack,map_location='cpu')
    prefix='model.visual.transformer.resblocks.11.mlp.'
    p0=permutation(expert.state_dict()[prefix+'c_fc.weight'],baseline[prefix+'c_fc.weight'])
    p1=permutation(published[prefix+'c_fc.weight'],recovered[prefix+'c_fc.weight'])
    for source,dest,p in [(expert.state_dict(),baseline,p0),(published,recovered,p1)]:
        assert torch.equal(source[prefix+'c_fc.bias'][p],dest[prefix+'c_fc.bias'])
        assert torch.equal(source[prefix+'c_proj.weight'][:,p],dest[prefix+'c_proj.weight'])
    result={'status':'passed_exact_AGA_row_permutation_extraction','neurons':len(p0),'changed_permutation_positions':int((p0!=p1).sum()),
        'original_to_AGA_permutation':p0.tolist(),'published_to_AGA_permutation':p1.tolist(),
        'bias_and_B_column_consistency':'passed','checkpoint_sha256':pins,'release_sha256':digest(release),'AGA_edits_sha256':digest(attack),
        'source_sha256':digest(__file__),'new_model_forwards':0,'new_attack_calls':0,'new_release_weights':0,'test_used':False,'partner_used':False,
        'scope':'Exact algebra of the existing author MLP permutation. A changed assignment is a coordinate mismatch explanation, not a quantified causal attribution of the entire TA/AGA accuracy gap.'}
    save_json(out/'coordinate_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


@torch.no_grad()
def head_geometry(output):
    """Separate removable head coordinates from changed column spaces; no images."""
    start=time.monotonic();out=Path(output);assert not out.exists()
    cfg=json.loads(Path('configs/frozen_coupling_audit.json').read_text())
    anchor,expert,_,pins=checked_models('MNIST')
    def checked(path,pin):
        assert digest(path)==pin,path
        return torch.load(path,map_location='cpu')
    released=expert.state_dict()|checked(cfg['release'],cfg['release_sha256'])
    recovered=checked(cfg['original_aga'],cfg['original_aga_sha256'])
    restored=recovered|checked(cfg['published_aga_edits'],cfg['published_aga_edits_sha256'])
    source=Path('external/secure-merge-attack/utils_attack_B32.py')
    spec=json.loads(Path('configs/phase3.json').read_text())
    assert digest(source)==spec['AGA']['raw_sha256']
    author,_=load_author(source,True)
    key='model.visual.transformer.resblocks.11.attn.out_proj.weight'
    original,changed,base=[s[key] for s in [expert.state_dict(),released,anchor.state_dict()]]
    records=[];maximum=0.
    for head in range(12):
        sl=slice(head*64,(head+1)*64)
        w,d,a=[x.double() for x in [original[:,sl],changed[:,sl]-original[:,sl],base[:,sl]]]
        q=torch.linalg.qr(w,mode='reduced').Q
        tangent=q@(q.T@d);normal=d-tangent
        u=np.linalg.svd(w.numpy(),full_matrices=False)[0]
        np.testing.assert_allclose(tangent.numpy(),u@(u.T@d.numpy()),atol=1e-11,rtol=1e-10)
        def projection(x):
            z=torch.linalg.qr(x,mode='reduced').Q
            return z@(z.T@a)
        coordinate=torch.linalg.lstsq(w,w+tangent).solution
        minimum=float(torch.linalg.svdvals(coordinate).min());assert minimum>1e-8
        erased=float((projection(w+tangent)-projection(w)).abs().max());assert erased<1e-10
        ideal=projection(w+d)-projection(w)
        for state,wanted in [(original,recovered),(changed,restored)]:
            x=state[:,sl];actual=x@author['solve_linear_transformation'](x,base[:,sl])
            err=float((actual-wanted[key][:,sl]).abs().max());maximum=max(maximum,err)
            assert err<2e-5,err
        observed=restored[key][:,sl].double()-recovered[key][:,sl].double()
        records.append(dict(head=head,delta_squared=float(d.square().sum()),
            normal_squared=float(normal.square().sum()),tangent_squared=float(tangent.square().sum()),
            ideal_recovered_delta_squared=float(ideal.square().sum()),
            author_recovered_delta_squared=float(observed.square().sum()),
            removable_coordinate_minimum_singular=minimum,tangent_projection_maximum_error=erased,
            ideal_to_author_delta_maximum_error=float((ideal-observed).abs().max())))
    sums={k:sum(r[k] for r in records) for k in ['delta_squared','normal_squared','tangent_squared','ideal_recovered_delta_squared','author_recovered_delta_squared']}
    result=dict(status='source_only_head_geometry_checked_not_defense',records=records,totals=sums,
        normal_energy_fraction=sums['normal_squared']/sums['delta_squared'],
        restored_delta_norm_ratio=(sums['author_recovered_delta_squared']/sums['delta_squared'])**.5,
        original_author_CPU_to_frozen_GPU_maximum_weight_error=maximum,
        scope='Only frozen MNIST O/B, public anchor and existing target-only parameter recovery. No peer, image, test, confirmation, gradient, new release or accuracy claim.',
        hypothesis='If head-coordinate erasure dominates the old O/B compensation loss, most original O change should lie in unchanged head column spaces and disappear on recovery. Energy and finite restored operator must be reported, not treated as classification damage.',
        source_sha256=digest(__file__),author_sha256=digest(source),checkpoint_sha256=pins,
        frozen_inputs={cfg[k]:cfg[k+'_sha256'] for k in ['release','original_aga','published_aga_edits']},
        old_protocol_sha256=digest('configs/frozen_coupling_audit.json'),wall_seconds=time.monotonic()-start,
        images_loaded=0,new_release_weights=0,gradients=0,partner_used=False)
    save_json(out,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--head-geometry',action='store_true');args=parser.parse_args();torch.set_num_threads(4)
    (head_geometry if args.head_geometry else audit)(args.output)
