"""Recount exclusion metadata and reproduce both draws without image responses."""

import argparse,json
from pathlib import Path
import numpy as np
import torch
from clip_benchmark import digest
from subspace_probe import save_json


def exclusions(split,initial):
    excluded=set(initial);confirmed=set();audit=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    for row in audit['records']:
        path=Path(row['path']);assert digest(path)==split['excluded_metadata_sha256'][str(path)]
        record=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        values=set(record[row['group']]);excluded.update(values)
        if 'confirm' in row['group']:confirmed.update(values)
    for name,pin in split['excluded_metadata_sha256'].items():
        path=Path(name);assert digest(path)==pin,name
        if path.suffix!='.json':continue
        values=json.loads(path.read_text())
        if path.name=='training_indices.json':
            for key,ids in values.items():
                if isinstance(ids,list):excluded.update(ids)
                if 'confirm' in key:confirmed.update(ids)
        elif path.name in ['holdout_protocol.json','confirmation_split.json']:
            excluded.update(values.get('indices',[]));confirmed.update(values.get('indices',[]))
    return excluded,confirmed


def run(cfg):
    parent=Path(cfg['calibration_reference']).parent;out=Path(cfg['output']);source=json.loads((parent/'construction_split.json').read_text());fresh=json.loads((out/'confirmation_split.json').read_text())
    sources=set(source['indices']);old,exposed=exclusions(source,set());assert len(old)==source['excluded_count']
    selected=np.random.default_rng(cfg['construction_seed']).choice(sorted(set(range(60000))-old),cfg['construction_n'],replace=False).tolist();assert selected==source['indices']
    old2,exposed2=exclusions(fresh,sources);assert len(old2)==fresh['excluded_count']
    selected2=np.random.default_rng(cfg['confirmation_seed']).choice(sorted(set(range(60000))-old2),cfg['confirmation_n'],replace=False).tolist();assert selected2==fresh['indices']
    assert not sources&exposed and not set(fresh['indices'])&(old2|sources)
    result=dict(status='passed_all_registered_split_exclusions_and_exact_draws',construction_n=len(sources),historical_excluded_n=len(old),historical_confirmation_and_holdout_indices=len(exposed),
        fresh_n=len(fresh['indices']),fresh_excluded_n=len(old2),construction_confirmation_overlap=0,historical_exposed_used_for_construction=0,
        original_construction_draw_reproduced=True,original_confirmation_draw_reproduced=True,image_responses_loaded=False,new_weights=0,
        construction_split_sha256=digest(parent/'construction_split.json'),confirmation_split_sha256=digest(out/'confirmation_split.json'),source_sha256=digest(__file__),
        scope='Index metadata only, including historical tensor indices; no old confirmation predictions, new image forwards, redraw, parameter update or model selection.')
    save_json(out/'split_independence_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);a=p.parse_args();torch.set_num_threads(4);run(json.loads(Path(a.protocol).read_text()))
