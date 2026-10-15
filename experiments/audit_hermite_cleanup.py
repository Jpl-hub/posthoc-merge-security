"""Post-freeze source-only parameter replacement identity, no new attack search."""

import argparse,copy,json,time
from pathlib import Path
import torch
from clip_benchmark import digest
from subspace_probe import save_json
from task_probe_distance import checked_models


@torch.no_grad()
def run(cfg):
    start=time.monotonic();out=Path(cfg['output']);meta=json.loads((out/'results.json').read_text());valid=json.loads((out/'independent_validation.json').read_text())
    assert meta['confirmation_passed'] and valid['status']=='passed'
    assert digest(out/'candidate.pt')==meta['candidate_sha256'];point=torch.load(out/'candidate.pt',map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);release=copy.deepcopy(expert);prefix='model.visual.transformer.resblocks.11.'
    keys=dict(O=prefix+'attn.out_proj.weight',B=prefix+'mlp.c_proj.weight')
    for name,key in keys.items():release.state_dict()[key].copy_((expert.state_dict()[key].double()+point['release_deltas'][name]).float())
    changed=[k for k,v in release.state_dict().items() if not torch.equal(v,expert.state_dict()[k])];assert set(changed)==set(keys.values())
    for key in keys.values():release.state_dict()[key].copy_(anchor.state_dict()[key])
    equal=all(torch.equal(v,anchor.state_dict()[k] if k in keys.values() else expert.state_dict()[k]) for k,v in release.state_dict().items());assert equal
    # The existing independently verified refill control uses this exact source
    # checkpoint and these same public locations. No partner or test is read.
    prior=Path('outputs/boundary_public_refill');oldcfg=json.loads((prior/'protocol.json').read_text());oldfreeze=json.loads((prior/'attack_freeze.json').read_text());oldvalid=json.loads((prior/'independent_validation.json').read_text())
    assert set(oldcfg['keys'])==set(keys.values()) and oldvalid['status']=='passed'
    oldpins=json.loads((Path(oldcfg['defense_evaluation'])/'release_freeze.json').read_text())['checkpoint_sha256']
    for name,pin in pins.items():assert oldpins[name]==pin,name
    edits=torch.load(prior/'artifacts/clean_edits.pt',map_location='cpu');assert digest(prior/'artifacts/clean_edits.pt')==oldfreeze['clean_edits_sha256']
    for key in keys.values():assert torch.equal(edits[key],anchor.state_dict()[key])
    result=dict(status='matched_public_replacement_erases_release_exactly',candidate_sha256=meta['candidate_sha256'],keys=keys,
        all_other_expert_parameters_unchanged=True,cleaned_model_equals_cleaned_unprotected_expert=True,
        same_control_as_prior_verified_public_refill=True,prior_control='outputs/boundary_public_refill',
        prior_clean_edits_sha256=oldfreeze['clean_edits_sha256'],checkpoint_sha256=pins,
        gradient_updates=0,new_model_forwards=0,peer_loaded=False,test_loaded=False,new_recovery_accuracy_evaluation=False,
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        limitation='Known-location public base replacement removes both edits exactly. Prior cleaned-control utility counts are historical matched evidence, not newly measured recovery accuracy for this candidate. No anti-cleaning claim; no adaptive attack or defense retuning.')
    save_json(out/'public_cleanup_identity.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();torch.set_num_threads(4);run(json.loads(Path(a.protocol).read_text()))
