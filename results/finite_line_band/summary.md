# Finite margin-band feasibility study

The sole point is **closed: fresh task-retention confirmation failed**. This is not a new effective Defense 2. No historical point was repaired.

| Fixed check | Inputs / probes | Max relative margin drift | Prediction changes | Outcome |
|---|---:|---:|---:|---|
| Construction, actual FP32 | 4096 | 0.247918 | 0 | Pass |
| Finite task-B probes | 640 | 0.217628 | — | Pass |
| Finite task-A probes | 640 | 0.218655 | — | Pass |
| Fresh training confirmation | 512 | 3.616268 | 0 | **Fail** |
| Complete test / true peer TA / full AGA | 0 | — | — | Blocked by confirmation |

The original parameter and task budgets are both 0.5. The actual logical B distance is 0.49999949985; the right-task32 response fraction is 6.50e−9. All other published parameters are unchanged. Construction native accuracy stays 4090/4096; confirmation stays 511/512. Accuracy equality does not imply the fixed all-class relative-gap constraint.

Only one fresh input violates the task budget, across three class pairs. The largest violation increases an original 0.797528 gap to 3.681604 (signed relative change +3.616268). The original teacher already misclassifies that input; the fixed protocol still includes it. The confirmation median/99th percentile/max R are 0.093128 / 0.241127 / 3.616268. Neither dropping an incorrect input nor replacing the maximum with an average is permitted.

The full finite source constraints work on their constrained inputs. They do not establish response control on unseen activations. This is evidence of a generalization failure for this fixed 3040-dimensional shared coefficient family; it does not isolate dimension, curvature or coverage as the sole cause. No radius,rank,axis,data,seed or task-budget retry follows this point. All 512 confirmation indices remain excluded from future construction.

The preregistered source-only public-base/original-AGA proxy falls from 510/512 to 438/512; this is a source diagnostic, not real partner/complete AGA protection or exclusive knowledge recovery. Its result did not select or repair weights.

Numerics: original sparse auto-QDLDL returned MaxTime (889.14 s, one iteration), the identical FAER replay was actively terminated after about890 s, and the explicitly amended dense equivalent problem returned optimal (74.23 s,20 iterations). All failures and both computational amendments are retained. The science data/objective/budgets did not change; the originally registered one-solve computational stop rule did change explicitly. The normalized linear objective is bracketed by 0.001674235671 and 0.001674236035 with a residual-aware bounded-ball dual check. This is not a classification guarantee.

Source validation took15.40 s with4352 FP32 image-forward instances including original/point128 replay; fresh confirmation14.90 s with1024 instances. Original source construction had4096 native plus512 public-environment forwards before solver status. No whole-task or new peer/AGA evaluation followed the failed gate. Heavy models,raw scores,serialized constraints and solver vectors stay on3090. Git contains code,protocols,and lightweight results.

Reproduction uses the existing two entrypoints only (run each step once in its frozen source revision):

```bash
python experiments/finite_compensation_bound.py --rank-one-band configs/finite_line_band.json
# Preserve the initial solver status; computational amendments use its exact serialized data.
python experiments/finite_compensation_bound.py --dense-line-system outputs/phase2_finite_line_band_mnist_20261020
python experiments/verify_finite_compensation_bound.py --line-point outputs/phase2_finite_line_band_mnist_20261020 --protocol results/finite_line_band/source_protocol.json
python experiments/verify_finite_compensation_bound.py --audit-line outputs/phase2_finite_line_band_mnist_20261020 --stage source
python experiments/verify_finite_compensation_bound.py --line-holdout outputs/phase2_finite_line_band_mnist_20261020 --protocol results/finite_line_band/confirmation_protocol.json
python experiments/verify_finite_compensation_bound.py --audit-line outputs/phase2_finite_line_band_mnist_20261020 --stage confirmation
# STOP: passed=false. Do not run full-task/peer/AGA,redraw,or repair.
python experiments/verify_finite_compensation_bound.py --line-closure outputs/phase2_finite_line_band_mnist_20261020
```

Source gate was frozen in local2e8bc18 before execution; its remote upload bc37e93e followed a connector interruption. Confirmation/full-task source was frozen in local97606ff and verified remote8ef3c25a before drawing/forwarding confirmation. Source hashes are stage-specific in the corresponding result/protocol; current verifier adds read-only closure reproduction. These stages must not all be described as executing the current file hash. Standard SVD,quartics,rank-one edits,and SOCP are not claimed novel.
