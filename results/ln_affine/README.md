# Defense2: fixed affine Q control — closed

**No effective new defense.** This point also cannot be listed as a strict logical-projection distance0.5 candidate.

The preregistered native packed-QKV tensor metric was0.499999500; independently separating the modified Q projection and bias gives **0.696969066**. Q weight/bias relative distances are0.569853/0.401290. The earlier V/O interaction similarly gives0.736941103 under this comparison. Original release files, packed metrics and evaluation facts remain unchanged; no point was rescaled or retried. See [budget recount](budget_conventions.json).

The same independent recount was applied to the frozen baselines: PaRaMS Attention ablation0.4060698203 remains within the parameter0.5 budget; full official PaRaMS74.4862962043 remains over-budget. Their historical packed-tensor distances0.2648484944/47.2573024155 and all predictions are retained. This does not turn finite task confirmation into a universal task guarantee.

Only last-block Q weight and its bias changed, using the known LayerNorm affine input gauge. The original expert remains frozen. The gauge itself is prior work; it is not claimed as an original defense. Source weights were frozen before new images, and evaluation before any peer/test use.

Source512 R=1.91741686e-06; fresh512 R=8.98226721e-07; all checked predictions unchanged. Each also passed640 finite task-SVD input probes. Complete MNIST10000:99.69%, zero changed, R=4.14791464e-05. These are finite checks, not universal functional preservation in FP32.

| Setting | MNIST control | MNIST point | SVHN control | SVHN point | MNIST changed |
|---|---:|---:|---:|---:|---:|
| TA0.5 | 99.37000% | 99.37000% | 92.49770% | 92.49770% | 0 |
| Author AGA target only + TA | 99.30000% | 99.30000% | 92.78196% | 92.75123% | 0 |
| Author AGA both + TA | 99.32000% | 99.32000% | 91.77167% | 91.76398% | 0 |

Target predictions are identical to matched unprotected in all three merging settings. AGA-standalone changes3 predictions, loses2 correct/gains1; this does not establish fusion protection. Partner changes/losses and all class NLL/lost/gained counts are retained in [the detailed trace](evaluation/mechanism_and_class_loss.json).

The causal trace uses identical actual AGA prefixes on512 original source images: query relative change34.768%, median attention TV11.248%, yet target-only fusion signed margin harm95th percentile1.829%. A changed function/operator/attention map is insufficient for harmful classification. NumPy attention error5.55e-16; full native attention control2.94e-7.

Wall times: source17.98s, fresh confirmation16.41s, full evaluation150.01s including author AGA6.59s, causal trace9.00s. Independent272 scores were replayed in their original128-image contexts; replay/recount overhead is additional and not assumed zero. No expert updates or reverse gradients.

Reproduction: source `experiments/ln_affine_release.py`, configs `ln_affine_release.json` and `ln_affine_eval.json`; existing native evaluator reused. Source Git22b91a7 / remote8e76176 precedes images; eval Git1cf04e8 / remote02532cd precedes tests. Exact executed code, splits, weights and raw scores stay3090. Historical tests are reused, not first blind tests.

The analytic data-free gauge canonicalization limitation was declared before execution. No rank, head, scale, layer or seed retry. O/B and A/B remain effects-only references because of their separate task-budget counterexamples.
