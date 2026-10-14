"""Extract exact row permutations from existing AGA artifacts; no attack rerun."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from clip_benchmark import digest
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);audit(args.output)
