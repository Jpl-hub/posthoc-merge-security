# ProDistill: fixed two-expert evaluation

MNIST protected / SVHN partner (%). Attacker uses64 unlabeled train images per task and task membership; only merger coefficients receive gradients. This is a native two-expert adapter of the official kernel, not the original eight-task table.

| Defense | No alignment | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|
| Unprotected | 99.45 / 96.73 | 99.42 / 96.83 | 99.44 / 96.56 |
| Defense 1 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 |
| O/B† | 99.45 / 96.10 | 99.42 / 96.16 | 99.42 / 95.79 |
| A/B† | 99.50 / 96.43 | 99.42 / 96.46 | 99.43 / 96.15 |

† O/B and A/B have independently confirmed task0.5 violations: effect references only. Defense1 uses a different parameter constraint. Frozen standalone utility, SVD probes and budgets remain in ../utility.json; no defense is retrained or selected from these results.

Every scope has a matched unprotected control. Equal accuracy does not imply equal predictions: raw per-class counts and lost/gained pairs are retained. Bonferroni correction covers18 protected comparisons in this batch. Historically exposed tests are not new blind evaluation.

Defense1 shares all observed prefixes/blocks with the unprotected control. Its identical coefficient training is reused only after exact input-weight equality; its own frozen final LN bias is retained. This attack does not optimize final LN/P. Failure against that bias does not establish resistance to attacks that cover it; the separately reported bias canonicalization already restores it.

Nine coefficient fits,10800 backwards,0 expert/defense updates. Training wall time 442.96s; full evaluation sub-cost 421.90s. 384 original-context score replays passed with maximum error 0. Assessment wall times including their replays are retained per case; the sub-costs are not whole-project GPU time.
