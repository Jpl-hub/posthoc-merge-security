# Phase 3 evaluation results

Accuracy (%). Frozen defenses; historically exposed test sets. No defense was selected or updated from these results.

*Phase 1 and full PaRaMS are different-budget references. † O/B has a known task-preservation violation. A/B has finite confirmation only.*

## MNIST → SVHN: TA

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.37 / 92.50 | 99.30 / 92.78 | 99.32 / 91.77 |
| Phase 1 (LN bias)* | 99.69 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 |
| O/B† | 99.71 | 78.77 / 62.94 | 79.53 / 65.82 | 77.85 / 61.42 |
| A/B | 99.65 | 91.07 / 77.46 | 85.42 / 75.50 | 79.20 / 69.84 |
| PaRaMS (full)* | 99.69 | 8.92 / 9.15 | 99.30 / 92.79 | 99.32 / 91.77 |
| PaRaMS (attention) | 99.69 | 99.37 / 92.51 | 99.30 / 92.78 | 99.32 / 91.77 |

## MNIST → SVHN: TIES

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.34 / 94.37 | 99.32 / 94.81 | 99.36 / 94.18 |
| Phase 1 (LN bias)* | 99.69 | 9.82 / 9.69 | 9.82 / 9.69 | 9.82 / 9.69 |
| O/B† | 99.71 | 64.31 / 54.27 | 57.13 / 50.75 | 54.06 / 44.53 |
| A/B | 99.65 | 69.44 / 66.83 | 65.76 / 66.21 | 62.69 / 61.80 |
| PaRaMS (full)* | 99.69 | 8.92 / 9.05 | 99.32 / 94.81 | 99.36 / 94.18 |
| PaRaMS (attention) | 99.69 | 99.35 / 93.92 | 99.32 / 94.81 | 99.36 / 94.18 |

## Evidence and limits

- Independently verified cases: 1/4; result rows: 78.
- TA coefficient 0.5; TIES top 20%, mass signs, disjoint mean, coefficient 1.0. Each has a separate unprotected control.
- Original AGA leaves final LN bias unchanged. Phase 1 accuracy collapse does not establish coverage against arbitrary recovery.
- Per-class counts, paired losses/gains, approximate paired 95% intervals and exact McNemar log-p are in results.json/CSV. Bonferroni correction covers all non-native protected comparisons in this generated batch.
- Figures show fixed deterministic configurations, not seed means. Significance is not a defense-success definition.
- No DARE, full TSV-Merge, data-assisted adaptation or distillation result is claimed in this batch.
