"""Frozen controls, split exclusions, recovery scope and immutable artifact audit."""
import argparse,json,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from attention_cascade import suffix
from clip_benchmark import digest
from cross_module_distance import normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import gelu,initialize
from verify_cascade import suffix as numpy_suffix
from verify_cross_module import ln
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores


@torch.no_grad()
def archive(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());valid=json.loads((out/'independent_validation.json').read_text())
    assert valid['status']=='passed_closed_mechanism' and not rep['mechanism_supported']
    assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256'];g=torch.load(out/'geometry.pt',map_location='cpu');excluded=set();metadata={};generic=set()
    audit=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    for row in audit['records']:
        if 'confirm' not in row['group']:continue
        path=Path(row['path']);r=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices'];excluded.update(r[row['group']]);metadata[str(path)]=digest(path)
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
    source=set(g['source_indices']);reference=set(g['environment_indices']);assert not source&excluded and not reference&excluded
    save_json(out/'split_audit.json',dict(status='passed',source_n=len(source),reference_n=len(reference),registered_independent_confirmation_n=len(excluded),independent_source_overlap=0,independent_reference_overlap=0,generic_historical_confirmation_source_overlap=len(source&generic),generic_historical_confirmation_reference_overlap=len(reference&generic),confirmation_values_loaded=False,metadata_sha256=metadata,source_is_construction_not_independent=True))
    # An existing weight-only artifact: no exposed confirmations or future peer.
    baseline=Path('outputs/boundary_mnist/artifacts/release.pt');pin='953007daee0d6ba6f4c40de8aa738cfb5052db65270436aa4f0fe5e04aefaa46';assert digest(baseline)==pin
    old=torch.load(baseline,map_location='cpu');cal=torch.load(cfg['calibration_reference'],map_location='cpu');env=torch.load(cfg['environment_cache'],map_location='cpu');p=on_device(env['proxy_parameters']);H=env['proxy']['H'].double().cuda();h,ctx=suffix(H,p);rounding=env['proxy']['rounding'].cuda()
    stem='model.visual.transformer.resblocks.11.';dO=old[stem+'attn.out_proj.weight'].double()-cal['parameters']['O'];dB=old[stem+'mlp.c_proj.weight'].double()-cal['parameters']['B']
    r=ctx['r']+.5*ctx['average'].flatten(1)@dO.cuda().T;hp=rounding+r+gelu(normalized_input(r,p)@p['A'].T+p['a'])@(p['B']+.5*dB.cuda()).T+p['b'];before=scores(h+rounding,p);after=scores(hp,p)
    pn=convert(env['proxy_parameters']);hn,zn=numpy_suffix(env['proxy']['H'].double().numpy(),pn);rn=hn-zn@pn['B'].T-pn['b'];avg=np.linalg.solve(pn['O'],(rn-env['proxy']['H'][:,0].double().numpy()-pn['ob']).T).T;rn+=.5*avg@dO.numpy().T
    zp=ln(rn,pn['gamma2'],pn['beta2'],pn['eps2'])@pn['A'].T+pn['a'];znew=zp/(1+np.exp(-1.702*zp));hnew=env['proxy']['rounding'].numpy()+rn+znew@(pn['B']+.5*dB.numpy()).T+pn['b'];independent=raw_scores(hnew,pn);error=float(np.max(np.abs(independent-after.cpu().numpy())));assert error<1e-8
    torch.save(dict(before=before.cpu(),after=after.cpu(),baseline_sha256=pin,environment_indices=g['environment_indices']),out/'frozen_OB_control.pt')
    save_json(out/'frozen_OB_control.json',dict(status='passed',n=len(before),changed=int((before.argmax(1)!=after.argmax(1)).sum()),maximum_numpy_score_error=error,baseline_sha256=pin,raw_predictions_sha256=digest(out/'frozen_OB_control.pt'),only_weight_artifact_loaded=True,role='Post-freeze same-source-environment positive control. Never used for constructing, selecting or repairing new gate parameters; no real partner/test/confirmation responses.'))
    save_json(out/'recovery_scope.json',dict(status='algebraic_erasure_risk',edited_keys=[stem+'mlp.c_fc.bias',stem+'mlp.c_proj.weight'],public_base_overwrite_removes_all_release_increments=True,original_and_protected_cleaned_parameters_identical=True,cleanup_utility_cost_measured=False,cleaned_TA_or_AGA_measured=False,robustness_proved=False,scope='Only these two final-MLP tensors changed. Replacing both with public base yields identical cleaned original/protected models. This does not say cleanup preserves utility; no new defense or attack performance is claimed.'))
    previous=Path('outputs/phase2_input_null_transport_mnist');manifest=json.loads((previous/'source_freeze.json').read_text());count=0
    for name,pin in manifest['artifacts'].items():assert digest(previous/name)==pin,name;count+=1
    retained=json.loads((previous/'retention_integrity.json').read_text())
    for name,pin in retained['old_best_and_phase1'].items():assert digest(name)==pin,name
    save_json(out/'retention_integrity.json',dict(status='passed',previous_transport_artifacts_verified=count,old_best_and_phase1=retained['old_best_and_phase1'],no_historical_artifacts_modified=True))
    rep.update(status='closed_independently_verified_finite_gate_compensation',independently_verified=True,verification='independent_validation.json',true_TA_run=False,author_AGA_run=False,independent_confirmation_run=False,phase1_unchanged=True)
    save_json(out/'results.json',rep)
    save_json(out/'audit_cost.json',dict(wall_seconds=time.monotonic()-start,new_full_model_forwards=0,new_publication_updates=0,source_sha256=digest(__file__),scope='Lightweight frozen-control and archival audit; primary construction and NumPy verifier walls reported separately. Failed first verifier log retained; its wall was not timed.'))
    files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['source_freeze.json','source_history.bundle']}
    save_json(out/'source_freeze.json',dict(construction_commit='09701a5',first_verifier_commit='386781c',corrected_verifier_commit='e549851',artifacts=files,source_sha256={f:digest(f) for f in ['experiments/finite_gate_compensation.py','experiments/verify_finite_gate_compensation.py',__file__,'configs/finite_gate_compensation.json']},source_archive='outputs/phase2_finite_gate_source_09701a5.tar.gz',source_archive_sha256=digest('outputs/phase2_finite_gate_source_09701a5.tar.gz'),first_verifier_failure='verification_failure_v1.log; CPU vs CUDA nonunique SVD column signs; only coordinate replay changed, no primary rerun or tolerance relaxation',heavy_files_server_only=True))
    print(json.dumps(dict(status=rep['status'],retained_artifacts=count,frozen_OB_control_changed=int((before.argmax(1)!=after.argmax(1)).sum()),audit_seconds=time.monotonic()-start)))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();initialize();archive(json.loads(Path(args.protocol).read_text()))
