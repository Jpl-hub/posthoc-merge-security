"""Curate verified fixed gap-jet results and hash retained artifacts; no selection."""
import json,shutil,hashlib
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    out=Path('outputs/phase2_gap_jet_publication_mnist');ev=Path('outputs/phase2_gap_jet_eval_mnist_svhn')
    report=read(out/'results.json');validation=read(ev/'independent_validation.json');rows=read(ev/'results.json');execution=read(ev/'execution.json')
    assert report['confirmation_passed'] and validation['status']=='passed' and validation['official_count_rows']==17
    assert sha(out/'candidate.pt')==report['candidate_sha256']=='7f748688e95843a27f0b86e67b3f3d50d4b191c1948cde8b4c2ed63e473b3bd9'
    assert sha(ev/'artifacts/predictions.pt')==execution['predictions_sha256']
    def row(preset,stage,task):return next(r for r in rows if (r['preset'],r['stage'],r['dataset'])==(preset,stage,task))
    target=row('released','AGA','MNIST');assert target['correct']==9931 and target['paired']['exact_mcnemar_p']==1.
    summary=dict(status='closed_verified_no_target_defense',new_best=False,protection_effect_verified=False,
        candidate_sha256=report['candidate_sha256'],training_point_sha256=report['training_point_sha256'],confirmation=report['confirmation'],
        old_best=dict(native=99.71,TA=78.77,AGA=77.85,unchanged=True),
        metrics=[{k:r[k] for k in ['preset','stage','dataset','correct','n','top1','changed_predictions','paired','per_class'] if k in r} for r in rows],
        independent_real_verification=validation,matched_source_chart_audit=read(out/'matched_chart_audit.json'),
        known_public_replacement=read(out/'public_cleanup_identity.json'),split_audit=read(out/'split_independence_audit.json'),
        forward_evaluation_seconds=sum(r.get('evaluation_seconds',0.) for r in rows),AGA_recovery_compute_seconds='not_separately_measured',
        cross_task_extended=False,parameters_changed_after_confirmation=False,parameters_changed_after_test=False,
        mathematical_fact='For every selected pair, ||a||>|b|/sqrt(width) certifies unique scalar numerator root; independent Brent and SVD verify finite shared-matrix realization and normalized gap/path-derivative constraints.',
        negative_fact='Reducing full-hidden constraints to pair jets retains finite hidden freedom and passes one fresh split, but this fixed chart/controller does not cause target fusion failure. No general impossibility or future-peer coverage proved.',
        next_hypothesis='Study multi-class decision-normal response over a source-defined task boundary neighborhood, with a finite residual-to-margin bound and shared-parameter realizability. First establish whether non-classification tangent freedom changes the joint decision-normal response; do not extend this point, re-use exposed confirmation values, or add arbitrary environments.')
    write(out/'summary.json',summary)
    shutil.copyfile(out/'results.json',out/'source_results_before_real_eval.json')
    report.update(status='closed_verified_no_target_defense',true_TA_run=True,author_AGA_run=True,real_evaluation=str(ev),real_evaluation_verified=True,
        protection_effect_verified=False,new_best=False,point_repaired_after_confirmation=False,parameter_search_after_test=False,
        true_metrics={task+'_'+label:100*row('released',stage,task)['top1'] for task in ['MNIST','SVHN'] for label,stage in [('TA','direct'),('AGA','AGA')]})
    report['true_metrics']['MNIST_standalone']=100*row('released','standalone','MNIST')['top1'];write(out/'results.json',report)
    # Complete prior Hermite freeze pins; old weights/source/evidence are read-only.
    checked=0
    for folder in ['phase2_boundary_hermite_mnist','phase2_hidden_hermite_mnist','phase2_hidden_hermite_canonical_mnist','phase2_hermite_publication_mnist','phase2_hermite_eval_mnist_svhn']:
        base=Path('outputs')/folder;freeze=read(base/'source_freeze.json')
        for name,pin in freeze['artifacts'].items():assert sha(base/name)==pin,(folder,name);checked+=1
        for name,pin in freeze['source_archives'].items():assert sha(name)==pin
    pins={
        'outputs/phase2_boundary_game_mnist/candidate.pt':'e0ee5e5eeb170d550401724dfe0a8c6ff39273afdde79f46a207a4370b2398a4',
        'outputs/phase2_ffn_game_mnist/candidate.pt':'1c5ffd5e9e16b44a104dcc29825dbd2cac1a8a2cd534b1d203f5b18815732f3b',
        'outputs/phase2_robust_shared_publication_mnist/training_point.pt':'dfb2c7e234e818221652e118f368d2b4e0a230bd6e6f21ccff5bf66bf2b81593',
        'outputs/phase1_frozen_20261014/source.zip':'602dd96dea9786e68bbfe74c3ae2e5eaf1a053382fb6f9637ea21a05c772b5ae',
        'outputs/projection_bias_noise_mnist/artifacts/release.pt':'6c5c29354c85bea33dccb3ec60c3f89fcd05664c6281cd693a0129c6dde40e71',
        'outputs/projection_bias_noise_dtd/artifacts/release.pt':'0dfad6a9d8748a96f1403a77ce12728a829126b1a3cdd69ec2e7ada1248c1f5d',
        'outputs/projection_bias_noise_svhn/artifacts/release.pt':'454a5afcf23e9859c1c41fb39dbb0e4a46981802a6e2d4f93a081088044f4c29'}
    for name,pin in pins.items():assert sha(name)==pin,name
    write(out/'retention_integrity.json',dict(status='passed',prior_Hermite_artifact_pins=checked,prior_best_failed_and_phase1_pins=pins,phase1_unchanged=True,no_history_deleted=True))
    for base in [Path('outputs/phase2_gap_jet_mechanism_mnist'),out,ev]:
        artifacts={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name not in ['source_freeze.json','source_history.bundle']}
        write(base/'source_freeze.json',dict(construction_commit='482fdab',source_selected_point_commit='e124d13',confirmation_and_eval_weight_pin_commit='56d3986',
            source_archive='outputs/phase2_gap_jet_source_e124d13.tar.gz',source_archive_sha256=sha('outputs/phase2_gap_jet_source_e124d13.tar.gz'),
            artifacts=artifacts,report_source_sha256=sha(__file__),heavy_artifacts_server_only=True,all_prior_artifacts_preserved=True))
    print(json.dumps(dict(status=summary['status'],MNIST_native=report['true_metrics']['MNIST_standalone'],MNIST_TA=report['true_metrics']['MNIST_TA'],MNIST_AGA=report['true_metrics']['MNIST_AGA'],target_AGA_paired=target['paired'],retained_prior_Hermite_pins=checked,new_best=False)))
