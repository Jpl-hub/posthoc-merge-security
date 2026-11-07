# Direction controls

The frozen NBP release is compared at its original update norm with a simple projected class contrast and five fixed random-nullspace draws. Seed20261106 is reused from the completed component controls; seeds20261107–20261110 are new. No draw is selected or used to alter NBP.

The simple direction is `N @ N.T @ ((P0 + Pe)/2) @ (w_requested - mean(W))`, normalized to the frozen NBP update norm. The requested class is drawn with the original construction seed20261014 before evaluation. It uses no multi-class least-squares solve.

MNIST/SVHN/DTD use the same fixed companions, datasets, TA0.5 and target-only author AGA as the existing component controls. All utility failures and all seed results will be reported. Matching unprotected logits are reused by SHA and parameter fingerprint. Each new direction has full native target and both merged datasets, with original128-context verification. These are ablations, not new main-method versions.

Completed:75 new complete evaluation rows,15 imported unprotected rows,1440 original128-context selected-score replays. Maximum replay difference1.3352e-5 (limit5e-4). Independent NumPy construction checks passed for all15 new directions (maximum relative error2.653e-8). The original inherited replay-count estimate was320; the actual count is480 per task. No registered numerical setting was changed.

| Task | NBP AGA+TA | Class projection AGA+TA | Random null, mean ± sample SD | Random range |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 9.80 | 9.80 | 50.27 ± 19.28 | 19.45–68.59 |
| SVHN | 6.70 | 6.70 | 50.42 ± 25.80 | 14.58–84.88 |
| DTD | 2.13 | 2.13 | 3.45 ± 1.31 | 2.13–5.16 |

The projected class contrast matches NBP on these three tasks. This comparison does not establish that the multi-class pseudoinverse is necessary. The five fixed random draws are all retained; they are not a search or a basis for selecting a release.
