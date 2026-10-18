# ProDistill: fixed two-expert evaluation

Cells are protected task / partner accuracy (%). Attacker uses64 unlabeled train images per task and task membership; only merger coefficients receive gradients. This is a native two-expert adapter of the official kernel, not the original eight-task table.

## DTD / MNIST

| Defense | No alignment | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|
| Unprotected | 73.14 / 99.51 | 73.09 / 99.52 | 73.40 / 99.47 |
| Defense 1 | 2.13 / 98.06 | 2.13 / 98.07 | 2.13 / 97.71 |

## SVHN / MNIST

| Defense | No alignment | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|
| Unprotected | 96.73 / 99.45 | 96.45 / 99.45 | 96.56 / 99.44 |
| Defense 1 | 6.70 / 9.80 | 6.70 / 9.80 | 6.70 / 9.80 |

† O/B and A/B have independently confirmed task0.5 violations: effect references only. Defense1 uses a different parameter constraint. Frozen standalone utility, SVD probes and budgets remain in ../utility.json; no defense is retrained or selected from these results.

Every scope has a matched unprotected control. Equal accuracy does not imply equal predictions: raw per-class counts and lost/gained pairs are retained. Bonferroni correction covers12 protected comparisons in this batch. Historically exposed tests are not new blind evaluation.

Defense1 shares all observed prefixes/blocks with the unprotected control. Its identical coefficient training is reused only after exact input-weight equality; its own frozen final LN bias is retained. This attack does not optimize final LN/P. Failure against that bias does not establish resistance to attacks that cover it; the separately reported bias canonicalization already restores it.

6 coefficient fits,7200 backwards,0 expert/defense updates. Training wall time 287.21s; full evaluation sub-cost 301.56s. 384 original-context score replays passed with maximum error 0. Assessment wall times including their replays are retained per case; the sub-costs are not whole-project GPU time.
