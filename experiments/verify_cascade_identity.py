"""Reuse a full independent validation only after exact finite-map identity."""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from subspace_probe import save_json


def verify(cfg):
    out=Path(cfg['completion_output']);reference=Path(cfg['resume_point']).parent
    current=json.loads((out/'results.json').read_text());old=json.loads((reference/'results.json').read_text());validation=json.loads((reference/'independent_validation.json').read_text())
    assert validation['status']=='passed' and validation['point_sha256']==old['point_sha256']
    assert digest(out/'point.pt')==current['point_sha256'] and digest(reference/'point.pt')==old['point_sha256']
    assert digest(reference/'verified_predictions.pt')==validation['predictions_sha256']
    a=torch.load(reference/'point.pt',map_location='cpu');b=torch.load(out/'point.pt',map_location='cpu')
    assert torch.equal(a['theta'],b['theta']) and a['upstream_amplitude']==b['upstream_amplitude']
    for key in ['cache_output','batch_size','probe_rank','contexts_per_class','finite_step','coefficient']:
        assert a['cfg'][key]==b['cfg'][key],key
    for file in ['experiments/attention_cascade.py','experiments/cascade_game.py']:
        assert digest(file)==a['cfg']['source_sha256'][file]==b['cfg']['source_sha256'][file]
    assert digest(Path(cfg['cache_output'])/'calibration.pt')=='aaf492c814b1b8baca7b8146824e0502ab1419171ed8b26c09a15c6dab4f91dd'
    assert current['selected']==old['selected']
    witnesses=json.loads((out/'adversary_witnesses.json').read_text());assert witnesses['current_always_included']
    result={**validation,'status':'passed_exact_identity_to_independently_verified_point','point_sha256':current['point_sha256'],
            'reference_point_sha256':old['point_sha256'],'reference_validation_sha256':digest(reference/'independent_validation.json'),
            'raw_prediction_reference':str(reference/'verified_predictions.pt'),'exact_coefficient_and_forward_identity':True,
            'new_model_forward_instances':0,'new_independent_response_runs':0,'source_sha256':digest(__file__),
            'scope':'Exact bitwise coordinates, pinned cache, identical forward source and probe inputs. Reuses previous complete NumPy9728-input validation and raw predictions; not a repeated experiment or a new defense.'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();verify(json.loads(Path(args.protocol).read_text()))
