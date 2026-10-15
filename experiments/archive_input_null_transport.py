"""Post-freeze split, signed linear bound, retention and artifact audit; no search."""
import argparse,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from clip_benchmark import digest
from subspace_probe import save_json
from verify_downstream_first import convert


def archive(cfg):
    out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());ver=json.loads((out/'independent_validation.json').read_text())
    assert ver['status']=='passed_closed_mechanism' and not rep['mechanism_supported']
    assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256'];d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'))
    excluded=set();metadata={};generic=set()
    audit=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    for row in audit['records']:
        if 'confirm' not in row['group']:continue
        path=Path(row['path']);r=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        excluded.update(r[row['group']]);metadata[str(path)]=digest(path)
    for folder in Path('outputs').iterdir():
        if not folder.is_dir():continue
        case=folder/'case.json';own=cfg['target'].lower() in folder.name.lower()
        if case.exists():own=json.loads(case.read_text()).get('target')==cfg['target']
        if not own:continue
        for name in ['confirmation_split.json','holdout_protocol.json']:
            path=folder/name
            if not path.exists():continue
            r=json.loads(path.read_text())
            if name=='holdout_protocol.json' and r.get('split')!='train':continue
            excluded.update(r.get('indices',[]));metadata[str(path)]=digest(path)
        path=folder/'training_indices.json'
        if path.exists():
            r=json.loads(path.read_text())
            if isinstance(r.get('confirmation'),list):generic.update(r['confirmation'])
    source=set(d['source_indices']);reference=set(d['environment_indices']);assert not source&excluded and not reference&excluded
    split=dict(status='passed',source_n=len(source),reference_n=len(reference),registered_independent_confirmation_n=len(excluded),independent_source_overlap=0,independent_reference_overlap=0,
        generic_historical_confirmation_source_overlap=len(source&generic),generic_historical_confirmation_reference_overlap=len(reference&generic),
        confirmation_values_loaded=False,metadata_sha256=metadata,source_is_construction_not_independent=True)
    save_json(out/'split_audit.json',split)
    env=convert(torch.load(cfg['environment_cache'],map_location='cpu'));p=env['proxy_parameters'];g=d['environment'];h=g['h'];delta=g['linear'];contrast=p['head']-p['head'].mean(0)
    M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0);b=p['beta']@p['projection']@contrast.T
    sd=np.sqrt(np.mean((h-h.mean(1,keepdims=True))**2,1)+p['eps']);N=h@M+sd[:,None]*b;y=g['before'].argmax(1);rows=np.arange(len(y));mask=np.ones_like(N,dtype=bool);mask[rows,y]=False
    den=N[rows,y,None]-N;lin=delta@M;damage=(lin-lin[rows,y,None])/np.where(mask,den,1)
    actual_alpha=rep['source_ray'][-1]['alpha'];exact_scale=cfg['parameter_budget']/actual_alpha
    bound=dict(selected_point_linear_maximum_damage=float(np.max(np.where(mask,damage,-np.inf))),
        hypothetical_full_budget_scaled_linear_maximum_damage=float(np.max(np.where(mask,damage*exact_scale,-np.inf))),
        finite_joint_maximum_damage=json.loads((out/'signed_margin_attribution.json').read_text())['environment']['maximum_damage'],
        source_reserve=cfg['source_reserve'],original_task_budget=cfg['task_budget'],
        note='Linear-only scale extrapolation for attribution; not a finite model evaluation or new candidate. Quantized point lies close to the exact ray. The .125 source step met original.5 but failed predeclared reserve.25; it was not reference-evaluated. No impossibility claim over the full original feasible set.')
    save_json(out/'linear_scope_audit.json',bound)
    previous=Path('outputs/phase2_null_bias_leakage_mnist');manifest=json.loads((previous/'source_freeze.json').read_text());count=0
    for name,pin in manifest['artifacts'].items():assert digest(previous/name)==pin,name;count+=1
    retention=json.loads((previous/'retention_integrity.json').read_text())
    for name,pin in retention['old_best_and_phase1'].items():assert digest(name)==pin,name
    save_json(out/'retention_integrity.json',dict(status='passed',prior_null_bias_artifacts_verified=count,old_best_and_phase1=retention['old_best_and_phase1'],no_historical_files_deleted_or_changed=True))
    rep.update(independently_verified=True,status='closed_independently_verified_input_null_transport',verification='independent_validation.json',true_TA_run=False,author_AGA_run=False,independent_confirmation_run=False,phase1_unchanged=True)
    save_json(out/'results.json',rep)
    files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['source_freeze.json','source_history.bundle']}
    save_json(out/'source_freeze.json',dict(construction_commit='b14d863',independent_verifier_commit='4854077',artifacts=files,
        source_sha256={name:digest(name) for name in ['experiments/input_null_transport.py','experiments/verify_input_null_transport.py',__file__,'configs/input_null_transport.json']},
        construction_archive='outputs/phase2_input_null_source_b14d863.tar.gz',construction_archive_sha256=digest('outputs/phase2_input_null_source_b14d863.tar.gz'),heavy_files_server_only=True))
    print(json.dumps(dict(status=rep['status'],split={k:v for k,v in split.items() if k!='metadata_sha256'},linear_scope=bound,prior_artifacts_preserved=count)))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();archive(json.loads(Path(args.protocol).read_text()))
