"""Post-freeze margin attribution, index audit and immutable artifact manifest."""
import os,json,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from clip_benchmark import digest
from subspace_probe import save_json
from verify_downstream_first import convert


def attribution(group,p):
    h=group['h'];dh=group['dh'];linear=group['linear'];even=dh-linear
    contrast=p['head']-p['head'].mean(0);M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0);b=p['beta']@p['projection']@contrast.T
    # Include the actual LayerNorm epsilon, not its small-variance approximation.
    sigma=lambda v:np.sqrt(np.mean((v-v.mean(1,keepdims=True))**2,1)+p['eps'])
    N=h@M+sigma(h)[:,None]*b;components=dict(linear=np.broadcast_to(linear@M,N.shape),even=even@M,variance=(sigma(h+dh)-sigma(h))[:,None]*b)
    total=sum(components.values());error=float(np.abs(total-((h+dh)@M+sigma(h+dh)[:,None]*b-N)).max());assert error<1e-9
    y=group['before'].argmax(1);rows=np.arange(len(y));mask=np.ones_like(N,dtype=bool);mask[rows,y]=False;g=N[rows,y,None]-N
    harm={k:-(v[rows,y,None]-v)/np.where(mask,g,1.) for k,v in components.items()};joint=sum(harm.values());j=np.argmax(np.where(mask,joint,-np.inf),1)
    crossing=(joint[rows,j]>1);np.testing.assert_array_equal(crossing,group['after'].argmax(1)!=y)
    picked={k:v[rows,j] for k,v in harm.items()}
    return dict(n=len(y),actual_crossings=int(crossing.sum()),maximum_joint_numerator_damage=float(joint[rows,j].max()),median_joint_numerator_damage=float(np.median(joint[rows,j])),
        component_mean_at_worst_joint_competitor={k:float(v.mean()) for k,v in picked.items()},
        linear_and_even_opposite_sign_fraction=float(np.mean(picked['linear']*picked['even']<0)),exact_identity_error=error,
        per_class=[dict(label=c,n=int((y==c).sum()),crossings=int((crossing&(y==c)).sum()),maximum_damage=float(joint[rows,j][y==c].max())) for c in range(len(p['head']))],
        selection_used=False,role='Post-freeze exact attribution only; worst competitor is never used to design parameters.')


if __name__=='__main__':
    cfg=json.loads(Path('configs/null_bias_leakage.json').read_text());out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());valid=json.loads((out/'independent_validation.json').read_text());assert valid['status']=='passed_closed_mechanism'
    assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256'];data=convert(torch.load(out/'diagnostic.pt',map_location='cpu'))
    p=convert(torch.load(cfg['calibration_reference'],map_location='cpu'))['parameters'];env=convert(torch.load(cfg['environment_cache'],map_location='cpu'))
    attrs=dict(source=attribution(data['source'],p),environment=attribution(data['environment'],env['proxy_parameters']),source_sha256=digest(__file__),new_parameter_updates=0,new_forwards=0)
    save_json(out/'signed_margin_attribution.json',attrs)
    # Audit ONLY registered split indices. No confirmation activations/scores.
    excluded=set();metadata={};generic=set();generic_metadata={}
    audit=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    for row in audit['records']:
        path=Path(row['path'])
        if 'confirm' not in row['group']:continue
        record=json.loads(path.read_text()) if path.suffix=='.json' else torch.load(path,map_location='cpu')['indices']
        excluded.update(record[row['group']]);metadata[str(path)]=digest(path)
    for folder in Path('outputs').iterdir():
        if not folder.is_dir():continue
        case=folder/'case.json';own=cfg['target'].lower() in folder.name.lower()
        if case.exists():own=json.loads(case.read_text()).get('target')==cfg['target']
        if not own:continue
        for name in ['confirmation_split.json','holdout_protocol.json']:
            path=folder/name
            if not path.exists():continue
            record=json.loads(path.read_text())
            if name=='holdout_protocol.json' and record.get('split')!='train':continue
            excluded.update(record.get('indices',[]));metadata[str(path)]=digest(path)
        path=folder/'training_indices.json'
        if path.exists():
            record=json.loads(path.read_text())
            if isinstance(record.get('confirmation'),list):generic.update(record['confirmation']);generic_metadata[str(path)]=digest(path)
    source=set(data['source_indices']);reference=set(data['environment_indices']);assert not source&excluded and not reference&excluded
    split=dict(status='passed_independent_confirmation_exclusions',original_direction_training_data_used=False,
        source_n=len(source),reference_n=len(reference),independent_confirmations_registered=len(excluded),independent_overlap_source=0,independent_overlap_reference=0,
        generic_historical_calibration_evaluation_overlap_source=len(source&generic),generic_historical_calibration_evaluation_overlap_reference=len(reference&generic),
        direction_depends_only_on_frozen_expert_and_public_base=True,confirmation_responses_loaded=False,metadata_sha256=metadata,generic_metadata_sha256=generic_metadata,
        limitation='Generic benchmark stages called confirmation may replay historical calibration pictures and are not new blind confirmation; these overlaps are reported separately. No independent confirmation responses are used for direction, selection, repair or source geometry.')
    save_json(out/'split_audit.json',split)
    normd=float(np.linalg.norm(data['delta_FP32']));metrics=dict(raw_parameter_displacement=normd,original_bias_norm=float(np.linalg.norm(p['a'])),
        per_edited_tensor_distance=rep['parameter_distance'],edited_tensor='block11.mlp.c_fc.bias',modified_weight_matrices=0,
        projected_task_leakage_operator_norm=rep['task_SVD_projected_leakage_singular_values'][0],
        class_normal_linear_leakage_upper_for_exact_kernel_at_public_half=normd*rep['task_SVD_projected_leakage_singular_values'][0]/8,
        scope='Per-edited-tensor normalized distances across different edit sites are not equal whole-model Euclidean displacements and are not protection-quality measures.')
    save_json(out/'distance_audit.json',metrics)
    # Quantify why the sufficient continuous-ball certificate is inconclusive.
    va=data['basis_A'];F=(p['A']@va.T)*data['rms_A'][None]
    L=p['gamma'][:,None]*p['projection'];L-=L.mean(0);cc=p['beta']@p['projection']
    ql=np.sqrt(np.linalg.eigvalsh(L.T@L)[-1])+np.linalg.norm(cc)/np.sqrt(768)
    bn=np.sqrt(np.linalg.eigvalsh(p['B']@p['B'].T)[-1]);fn=np.sqrt(np.linalg.eigvalsh(F.T@F)[-1]);m1=1.5*bn*fn
    h=data['source']['h'][:rep['neighborhood_contexts']];sd=np.sqrt(np.mean((h-h.mean(1,keepdims=True))**2,1)+p['eps']);tn=np.linalg.norm(h@L+sd[:,None]*cc,axis=1)
    bound_audit=dict(status='uniform_spectral_bound_inconclusive',radius=cfg['neighborhood_radius'],normal_rank_by_context=[v['normal_rank'] for v in data['neighborhood']],
        tangent_dimension_by_context=[v['tangent_dimension'] for v in data['neighborhood']],
        positive_denominator_proofs=sum(v['denominator_nonzero_proved'] for v in data['bounds']),certified_relative_task_balls=rep['certified_neighborhoods'],
        displacement_bound=m1*cfg['neighborhood_radius'],minimum_actual_hidden_sigma=float(sd.min()),minimum_actual_unnormalized_feature_norm=float(tn.min()),
        worst_feature_change_bound_ratio=float((ql*m1*cfg['neighborhood_radius']/tn).max()),
        sufficient_nondegenerate_radius_upper=float(np.minimum(sd*np.sqrt(768)/m1,tn/(ql*m1)).min()),
        finite_sampled_task_violations=sum(v['worst_task_drift']>cfg['task_budget'] for v in data['neighborhood']),radius_changed=False,
        scope='All32-dimensional ball bound uses valid global norm products. Its failure is not a witnessed violation; finite frame probes are not a continuous proof. No radius shrink, reconstruction or new direction.')
    save_json(out/'neighborhood_bound_audit.json',bound_audit)
    # Frozen previous round manifests and first-stage/best-point fingerprints.
    count=0
    for folder in ['phase2_gap_jet_mechanism_mnist','phase2_gap_jet_publication_mnist','phase2_gap_jet_eval_mnist_svhn']:
        base=Path('outputs')/folder;m=json.loads((base/'source_freeze.json').read_text())
        for name,pin in m['artifacts'].items():assert digest(base/name)==pin,(folder,name);count+=1
    pins={'outputs/phase2_boundary_game_mnist/candidate.pt':'e0ee5e5eeb170d550401724dfe0a8c6ff39273afdde79f46a207a4370b2398a4',
        'outputs/phase1_frozen_20261014/source.zip':'602dd96dea9786e68bbfe74c3ae2e5eaf1a053382fb6f9637ea21a05c772b5ae',
        'outputs/projection_bias_noise_mnist/artifacts/release.pt':'6c5c29354c85bea33dccb3ec60c3f89fcd05664c6281cd693a0129c6dde40e71',
        'outputs/projection_bias_noise_dtd/artifacts/release.pt':'0dfad6a9d8748a96f1403a77ce12728a829126b1a3cdd69ec2e7ada1248c1f5d',
        'outputs/projection_bias_noise_svhn/artifacts/release.pt':'454a5afcf23e9859c1c41fb39dbb0e4a46981802a6e2d4f93a081088044f4c29'}
    for name,pin in pins.items():assert digest(name)==pin,name
    save_json(out/'retention_integrity.json',dict(status='passed',prior_gap_jet_pins=count,old_best_and_phase1=pins,all_old_data_unchanged=True))
    rep.update(independently_verified=True,verification='independent_validation.json',status='closed_independently_verified_output_null_bias',true_TA_run=False,author_AGA_run=False,independent_confirmation_run=False,phase1_unchanged=True)
    save_json(out/'results.json',rep)
    files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ['source_freeze.json','source_history.bundle']}
    save_json(out/'source_freeze.json',dict(construction_commit='1a1707f',first_verifier_commit='589c043',verified_implementation_commit='48bd665',artifacts=files,
        source_sha256={name:digest(name) for name in ['experiments/null_bias_leakage.py','experiments/verify_null_bias_leakage.py',__file__,'configs/null_bias_leakage.json']},
        failed_verifier_log_preserved='verification_failure_v1.log',failure_reason='Verifier expected a context dict from NumPy suffix which actually returns the activation matrix; only verifier fixed, no rerun/reselection of primary direction.',heavy_files_server_only=True))
    print(json.dumps(dict(status=rep['status'],source_max=rep['source']['worst_task_drift'],environment_changes=0,distance=metrics,split_audit={k:v for k,v in split.items() if k not in ['metadata_sha256','generic_metadata_sha256']},attribution=attrs)))
