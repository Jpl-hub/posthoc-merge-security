# Normalization / alignment interaction: closed

One fixed source-only point; original expert frozen, no gradients or tuning after responses. This is a test of known components, not a new validated defense.

Joint parameter distance: **0.4999995002** (budget 0.5, including V bias). Source512 / fresh512 task drift: 8.6940079e-07 / 1.195428e-06; both zero changed predictions. The full MNIST test set has zero changed predictions and worst relative task drift **0.00012052549** (budget 0.5). This is measured FP32 evidence, not a universal floating-point guarantee. Real-valued functional preservation follows the stated normalization identity.

| Setting | MNIST unprotected | MNIST fixed point | SVHN unprotected | SVHN fixed point |
|---|---:|---:|---:|---:|
| TA(0.5) | 99.37% | 99.34% | 92.50% | 92.60% |
| Author AGA target only + TA | 99.30% | 99.26% | 92.78% | 92.47% |
| Author AGA both + TA | 99.32% | 99.31% | 91.77% | 91.35% |

**Conclusion:** no meaningful protection of MNIST. Target-only lost6/gained2 vs matched unprotected (McNemar p=0.2891); both lost5/gained4 (p=1). Significant partner losses of 0.3150 / 0.4226 percentage points do not establish target protection. No rank/axis/scale/layer/seed retry.

The V and O perturbations put 81.08% / 82.39% of squared parameter energy outside their respective32 task input probes. Raw AGA retains a matrix difference, yet the target median signed margin harm is only2.55%, with95th percentile15.75% for target-only. Parameter geometry and SVD singular values do not measure classification importance. Neither number is an unknown-partner guarantee.

Centered-before-AGA removes the added mean family (per-head residual O differences about1e-6 from finite precision/ridge). This limitation was declared before images; no adaptive classification accuracy is claimed because the point already fails the original main evaluation. The known V/O symmetry, SVD and mean invariance are not originality claims.

Production wall times: source 20.43s, fresh confirmation 19.32s, real evaluation 141.57s (AGA call 7.29s included). These exclude independent recount/replay overhead; 272 scores were independently replayed within their original128-image batch contexts, plus source/confirmation NumPy and exact index-draw audits. Matched original predictions / original and partner AGA weights were reused by SHA.

Source was committed locally at711d470 before images; real evaluation configuration at0d9910d before test/peer evaluation. GitHub read/write tools remain unavailable in this session despite the installed/enabled plugin: no new upload claimed. Heavy weights, raw scores and exact source indices remain3090. Protocols, source/model/raw fingerprints and lightweight statistics are retained here. Historical tests are reused, not fresh blind tests.

Canonical protocols are [source](../../configs/normalization_alignment.json) and [evaluation](../../configs/normalization_alignment_eval.json); copies with identical configuration content were omitted. Full server output directories retain their execution-time protocols.
