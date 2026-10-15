"""Archive the fixed Hermite experiment and its independent failure evidence.

Reporting only: no new parameter selection, image forwards or optimization.
Run from the project root after all frozen validations finish.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import json,hashlib,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'experiments')
from verify_downstream_first import convert
from verify_hidden_hermite import path_jet
from verify_levelset_audits import raw_scores

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
root=Path.cwd();out=root/'outputs/phase2_hermite_publication_mnist';ev=root/'outputs/phase2_hermite_eval_mnist_svhn'
report=read(out/'results.json');valid=read(out/'independent_validation.json');assert valid['status']=='passed' and report['confirmation_passed']
evvalid=read(ev/'independent_validation.json');assert evvalid['status']=='passed';execution=read(ev/'execution.json');rows=read(ev/'results.json')
assert sha(out/'candidate.pt')==report['candidate_sha256']=='65ee6895018ae99c28fd1de9fdbc01d85672ca9972f8893d419e2a412b7a78f7'
assert sha(ev/'artifacts/predictions.pt')==execution['predictions_sha256']
cfg=read(root/'configs/hermite_publication.json');cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));point=convert(torch.load(out/'training_point.pt',map_location='cpu'));d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'))
p=cal['parameters'];bs=cal['bases']['O'];size=cfg['probe_rank']**2;c=point['theta'][:size].reshape(cfg['probe_rank'],cfg['probe_rank']);do=np.linalg.norm(p['O'])*bs['u']@c@bs['vh'];j=path_jet(cal['endpoints'],cal['roots'],p,do)
f=np.r_[j['z'],j['dz']];target=np.r_[j['teacher']-j['h0'],j['dteacher']-j['dh0']]/np.linalg.norm(p['B']);scale=np.linalg.norm(f,axis=1,keepdims=True);f=f/scale;target=target/scale
part=np.linalg.lstsq(f,target,rcond=None)[0].T;k=d['final_K'];null=k-part
split=dict(upstream_distance=float(np.linalg.norm(c)),boundary_particular_B_distance=float(np.linalg.norm(part)),free_B_distance=float(np.linalg.norm(null)),
 part_null_inner_product=float(np.sum(part*null)),total_distance=float(np.sqrt(np.sum(c*c)+np.sum(k*k))),analysis_used_for_selection=False,new_parameter_updates=0)
write(out/'distance_decomposition.json',split)
# Independently verify the sufficient-bound failure; do not infer impossibility.
sp=cal['space'];teacher=path_jet(cal['endpoints'],cal['roots'],p,np.zeros_like(do))['teacher'];center=teacher-teacher.mean(1,keepdims=True);sigma=np.sqrt(np.mean(center*center,1,keepdims=True)+p['eps'])
L,bias,u,s=sp['L'],sp['c'],sp['U'],sp['S'];t=teacher@L+sigma*bias;a=(t/np.linalg.norm(t,axis=1,keepdims=True))@u;comp=np.sqrt(1-np.sum(a*a,1,keepdims=True));cg=bias@u;ca=bias-cg@u.T;cs=s-s.mean(0)
bounds=(np.linalg.norm(a@cs.T,axis=1,keepdims=True)*np.linalg.norm(ca)/comp+np.linalg.norm(cg@cs.T))/np.sqrt(teacher.shape[1])
oldroot=root/'outputs/phase2_boundary_hermite_mnist';failure=read(oldroot/'failure.json');np.testing.assert_allclose(bounds.ravel(),failure['bounds'],atol=1e-10,rtol=1e-10)
write(oldroot/'failure_independent_validation.json',dict(status='confirmed_sufficient_contraction_condition_failed',bound_min=float(bounds.min()),bound_max=float(bounds.max()),failed_roots=int((bounds>=1-1e-8).sum()),
 numerical_error=float(np.abs(bounds.ravel()-np.array(failure['bounds'])).max()),no_impossibility_claim=True,no_threshold_relaxation=True,new_release_weights=0))
metrics=[{k:r[k] for k in ['preset','stage','dataset','correct','n','top1','changed_predictions','paired'] if k in r} for r in rows]
target_ag=next(r for r in rows if r['preset']=='released' and r['stage']=='AGA' and r['dataset']=='MNIST')
assert target_ag['correct']==9929
summary=dict(status='closed_true_fusion_ineffective_after_independent_preservation',source_point=report['training_point_sha256'],candidate_sha256=report['candidate_sha256'],
 confirmation=report['confirmation'],metrics=metrics,old_best=dict(native=99.71,TA=78.77,AGA=77.85,unchanged=True),
 target_AGA_changed=target_ag['changed_predictions'],target_AGA_paired=target_ag['paired'],new_best=False,defense_effect_verified=False,
 independent_real_verification=evvalid,public_recovery_risk=read(out/'public_cleanup_identity.json'),distance_decomposition=split,
 execution_wall_seconds=execution.get('wall_seconds'),forward_evaluation_wall_seconds=sum(r.get('evaluation_seconds',0.) for r in rows),cross_task_extended=False,
 mechanism_fact='Value+derivative preservation gives finite quadratic suppression along fixed legal source chords. The from-zero point passes ONE truly new training confirmation, not a universal certificate. It does not cause significant true target fusion loss.',
 limitations=['Full hidden Hermite matching is stronger than class-gap matching; its cost is measured, not proved to be the cause of weak protection.',
 'Source objective improvement and .206 parameter distance still produce only4/6 empirical environment changes; no tested response entails future-peer classification failure.',
 'Known-location public base replacement erases both edits exactly. Historical matched-control counts are not a new recovery-accuracy evaluation.',
 'Original task checkpoints/phase1 frozen, .5/.5 budgets unchanged; no use of exposed confirmation values, peer or test for construction; no after-test retuning.'])
write(out/'summary.json',summary)
if not (out/'source_results_before_real_eval.json').exists():shutil.copyfile(out/'results.json',out/'source_results_before_real_eval.json')
report.update(status='closed_verified_true_fusion_ineffective',true_TA_run=True,author_AGA_run=True,real_evaluation=str(ev.relative_to(root)),real_evaluation_verified=True,
 protection_effect_verified=False,new_best=False,point_repaired_after_confirmation=False,parameter_search_after_test=False,
 true_metrics=dict(MNIST_standalone=99.70,MNIST_TA=99.36,MNIST_AGA=99.29,SVHN_standalone=100*25371/26032,SVHN_TA=100*23910/26032,SVHN_AGA=100*23595/26032))
write(out/'results.json',report)
archives={
 'outputs/phase2_boundary_hermite_source_450cb80.tar.gz':'a4e825510d257317d97820bc979876350ea5ced4bc479c0720ee858ae72101ea',
 'outputs/phase2_hidden_hermite_source_8b6734d.tar.gz':'589242a3c7a233a33226a55f6cf759b0bac960621107c9dc4653ed6c5149f5c2',
 'outputs/phase2_hermite_joint_source_d969223.tar.gz':'769b890838d8aaee9e9da9bd443bf45b5b26695b8a6828c3ed5830cf39c43f4a',
 'outputs/phase2_hermite_confirm_source_e8833c9.tar.gz':'cfdbeac6a5677bd74dda23d95c80a69dd28295f8077a769e9a0943442daa53cd'}
for name,pin in archives.items():assert sha(root/name)==pin,name
folders={'phase2_boundary_hermite_mnist':'450cb80','phase2_hidden_hermite_mnist':'8b6734d','phase2_hidden_hermite_canonical_mnist':'74f6d2d','phase2_hermite_publication_mnist':'d969223','phase2_hermite_eval_mnist_svhn':'5f9cbfb'}
for name,commit in folders.items():
 base=root/'outputs'/name;art={}
 for file in sorted(base.rglob('*')):
  if file.is_file() and file.name not in ['source_freeze.json','source_history.bundle']:art[str(file.relative_to(base))]=sha(file)
 write(base/'source_freeze.json',dict(construction_commit=commit,confirmation_and_eval_plan_commit='e8833c9',frozen_export_commit='5f9cbfb',
  validation_commit='403e400',source_archives=archives,artifacts=art,heavy_artifacts_server_only=True,
  split_validation_commit='1e9bebe',report_source_sha256=sha(Path(__file__)),
  validation_sources={name:sha(root/name) for name in ['experiments/verify_hidden_hermite.py','experiments/verify_hermite_publication.py','experiments/confirm_hermite_publication.py','experiments/audit_hermite_cleanup.py','experiments/audit_hermite_splits.py']},
  replay='Use450cb80 for first score-target calibration/failure;8b6734d for redundant hidden prototype;74f6d2d canonicalizes exact reversed paths and verifies all original equations. d969223 contains the joint source path;be861a5/e8833c9 contain confirmation and prior eval plan.5f9cbfb fills only the confirmed weight SHA. Full history preserved in final Git bundle. No reconstruction should mix historical config source pins with other revisions.'))
print(json.dumps(dict(status=summary['status'],distance_decomposition=split,full_verification_rows=evvalid['official_count_rows'],new_best=False)))
