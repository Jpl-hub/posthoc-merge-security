# Structural unit control: task retention passed, target protection absent

**Status:** production and independent verification completed. All 17 rows passed the official-label, per-class and paired-count checks. The verifier reproduced 272 selected scores in their original 128-image batch contexts, with the frozen score tolerance of 0.0005 and identical selected predictions. It checked the unchanged raw-prediction and AGA-edit hashes; this was a verification run, not a second production experiment. The completion receipt is in [evaluation/independent_validation.json](evaluation/independent_validation.json).

This is a known-component mechanism control,not an original Defense2 claim. No previous failed point was repaired.

| Check | Max full relative-gap drift | Outcome |
|---|---:|---|
| Original4096 cache,ideal old-unit deletion |0.004744306|Necessary check passed|
| New512 actual source |0.000259783|Passed|
| Source1280 task probes |0.001497613|Passed|
| Fresh512 actual confirmation |0.000241174|Passed|
| Confirmation1280 task probes |0.001722225|Passed|
| Complete10000 MNIST |0.009210947|Observed full-task gate passed|

The fixed logical A/a/B parameter distance is0.4999994994. Unit1230 and eta0.3217181725 were fixed from original target construction data only. Original expert updates:0. No unit/axis/sign/layer/strength/data/threshold retry. Whole-task accuracy stays9969/10000 with0 prediction changes; this historical test is not a first-blind evaluation.

| Merge / recovery scope | MNIST: unprotected → control | SVHN: unprotected → control |
|---|---:|---:|
|TA0.5|99.37% → 99.35%|92.50% → 92.64%|
|OfficialAGA target-only → TA0.5|99.30% → 99.30%|92.78% → 92.90%|
|OfficialAGA both → TA0.5|99.32% → 99.33%|91.77% → 91.90%|

MNIST lost/gained counts are3/1,2/2,2/3; paired p values0.625,1,1. There is no material target protection. Partner accuracy improves in this fixed comparison; it must not be reported as target damage. Full class counts and unrounded data are in evaluation/results.json and main_table.csv.

The original authorAGA recovered this single frozen release once,using unchanged headwise solves,precision,regularization and MLP matching. Original/partner recoveries and native controls were reused by their pinned hashes. Attack time3.45 s; total production148.11 s,128096 full-model image-forward instances,peak reported GPU858857984 bytes. Aligned matrix distance remains0.4552321,which is not a safety guarantee or the original publication budget. Source/confirmation took17.64/17.81 s; each has1024 whole-model natural image forwards,2560 block-probe forwards(1280 per model),and512 original block-control forwards.

The zero-activation relation is structural only for the newly replaced unit. The original unit contribution is removed,and FP32 rounding is separately measured. Task-SVD input projections of the A/B changes have distances0.0253343/0.0619604; their outside energy does not imply fusion harm. The construction uses known dummy-neuron and LN-affine ideas; data-free canonical detection/cleanup was disclosed before images. Its adaptive recovery accuracy was not run,as the main fixed originalAGA/TA control already fails.

Reproduction uses existing entrypoints at their pinned revisions:

```bash
python experiments/finite_compensation_bound.py --structural-unit configs/structural_unit_check.json
python experiments/verify_finite_compensation_bound.py --structural-unit outputs/phase2_ln_zero_unit_source_mnist_20261020
python experiments/ln_affine_release.py --protocol configs/structural_unit_release.json
python experiments/ln_affine_release.py --verify outputs/phase2_structural_unit_mnist_20261020
python experiments/ln_affine_release.py --protocol configs/structural_unit_release.json --confirm
python experiments/ln_affine_release.py --verify outputs/phase2_structural_unit_mnist_20261020 --confirm
python experiments/evaluate_local_residual_transport.py --protocol configs/structural_unit_eval.json
python experiments/evaluate_local_residual_transport.py --verify outputs/phase2_structural_unit_eval_20261020
```

The first verification attempt lost SSH connectivity and left an empty log without a receipt. After reconnection, a process and file check confirmed that no verifier remained active. That empty log was preserved as structural_unit_eval_verify.connection_lost.log; only the unfinished verification was restarted. It completed successfully, and its receipt was copied back and hash-checked. Local source/protocol freezes:7635e1c(cache),9515732(actual source),4e7d881(complete evaluation),all before their corresponding execution. Connector outages prevented remote preregistration; later upload must not be described as prior remote registration. Models,raw activations,predictions and source/attack weights stay on3090; no4090 use,no PPT. This fixed control is closed against weight repair.
