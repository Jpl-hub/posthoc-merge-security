# Phase 3 evaluation results

Accuracy (%). Frozen defenses; historically exposed test sets. No defense was selected or updated from these results.

*Phase 1 and full PaRaMS are different-budget references. † Task-preservation bound exceeded. Earlier finite confirmations do not prove global feasibility.*

The read-only 512-image audit also finds an A/B violation. Its historical confirmation and merge results remain valid observations; it cannot be presented as globally satisfying the 0.5 task bound.

## DTD → MNIST: TA

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 79.41 | 69.31 / 99.45 | 68.40 / 99.45 | 68.78 / 99.42 |
| Phase 1 LN bias* | 79.41 | 2.13 / 21.09 | 2.13 / 21.09 | 2.13 / 21.05 |

## DTD → MNIST: TIES

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 79.41 | 74.26 / 99.60 | 73.51 / 99.60 | 73.51 / 99.61 |
| Phase 1 LN bias* | 79.41 | 2.13 / 9.58 | 2.13 / 9.58 | 4.26 / 9.58 |

## MNIST → DTD: TA

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.45 / 69.31 | 99.43 / 69.41 | 99.42 / 68.78 |
| Phase 1 (LN bias)* | 99.69 | 9.80 / 2.13 | 9.80 / 2.13 | 9.80 / 2.13 |
| O/B† | 99.71 | 42.99 / 63.24 | 47.18 / 63.88 | 46.01 / 62.61 |
| A/B† | 99.65 | 49.40 / 59.41 | 36.77 / 59.68 | 35.54 / 58.19 |
| PaRaMS (full)* | 99.69 | 8.89 / 2.07 | 99.43 / 69.41 | 99.42 / 68.78 |
| PaRaMS (attention) | 99.69 | 99.46 / 69.26 | 99.43 / 69.41 | 99.42 / 68.78 |

## MNIST → DTD: TIES

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.60 / 74.26 | 99.60 / 74.41 | 99.61 / 73.51 |
| Phase 1 (LN bias)* | 99.69 | 9.82 / 2.13 | 9.82 / 2.13 | 9.82 / 2.13 |
| O/B† | 99.71 | 43.97 / 52.66 | 39.88 / 55.48 | 39.68 / 53.67 |
| A/B† | 99.65 | 50.60 / 49.41 | 36.69 / 50.21 | 35.78 / 47.39 |
| PaRaMS (full)* | 99.69 | 8.92 / 2.13 | 99.60 / 74.36 | 99.61 / 73.46 |
| PaRaMS (attention) | 99.69 | 99.60 / 71.17 | 99.60 / 74.41 | 99.61 / 73.51 |

## MNIST → SVHN: TA

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.37 / 92.50 | 99.30 / 92.78 | 99.32 / 91.77 |
| Phase 1 (LN bias)* | 99.69 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 |
| O/B† | 99.71 | 78.77 / 62.94 | 79.53 / 65.82 | 77.85 / 61.42 |
| A/B† | 99.65 | 91.07 / 77.46 | 85.42 / 75.50 | 79.20 / 69.84 |
| PaRaMS (full)* | 99.69 | 8.92 / 9.15 | 99.30 / 92.79 | 99.32 / 91.77 |
| PaRaMS (attention) | 99.69 | 99.37 / 92.51 | 99.30 / 92.78 | 99.32 / 91.77 |

## MNIST → SVHN: TIES

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 99.69 | 99.34 / 94.37 | 99.32 / 94.81 | 99.36 / 94.18 |
| Phase 1 (LN bias)* | 99.69 | 9.82 / 9.69 | 9.82 / 9.69 | 9.82 / 9.69 |
| O/B† | 99.71 | 64.31 / 54.27 | 57.13 / 50.75 | 54.06 / 44.53 |
| A/B† | 99.65 | 69.44 / 66.83 | 65.76 / 66.21 | 62.69 / 61.80 |
| PaRaMS (full)* | 99.69 | 8.92 / 9.05 | 99.32 / 94.81 | 99.36 / 94.18 |
| PaRaMS (attention) | 99.69 | 99.35 / 93.92 | 99.32 / 94.81 | 99.36 / 94.18 |

## SVHN → MNIST: TA

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 97.46 | 92.50 / 99.37 | 91.56 / 99.39 | 91.77 / 99.32 |
| Phase 1 LN bias* | 97.46 | 6.70 / 9.80 | 6.70 / 9.80 | 6.70 / 9.80 |

## SVHN → MNIST: TIES

Each entry is **protected-task / partner-task** accuracy.

| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |
|---|---:|---:|---:|---:|
| Unprotected | 97.46 | 94.37 / 99.34 | 93.75 / 99.37 | 94.18 / 99.36 |
| Phase 1 LN bias* | 97.46 | 17.22 / 20.04 | 17.14 / 20.04 | 16.10 / 19.42 |

## MNIST → SVHN: NUFILT

Official kernels with a native weight adapter; fixed author defaults. Both task orders are reported. This is not the full FusionBench application/table.

| Task order | Defense | No attack | AGA (protected expert) | AGA (all experts) |
|---|---|---:|---:|---:|
| MNIST → SVHN | Unprotected | 99.43 / 94.45 | 99.39 / 94.68 | 99.39 / 93.75 |
| MNIST → SVHN | Phase 1 (LN bias)* | 99.43 / 94.45 | 99.39 / 94.68 | 99.39 / 93.75 |
| MNIST → SVHN | A/B† | 99.33 / 94.46 | 99.14 / 94.61 | 99.08 / 93.72 |
| SVHN → MNIST | Unprotected | 99.19 / 95.11 | 99.07 / 95.35 | 99.14 / 94.51 |
| SVHN → MNIST | Phase 1 (LN bias)* | 99.19 / 95.11 | 99.07 / 95.35 | 99.14 / 94.51 |
| SVHN → MNIST | A/B† | 99.13 / 95.21 | 98.86 / 95.38 | 98.94 / 94.53 |

Phase 1 scores reuse the exact same merged models after checking all selected inputs are identical. Unselected biases, LN, fc2 and visual projection remain at the public anchor. No defense was changed. Merger gates were optimized on parameter projection losses only: 12 distinct two-task merges × 50 global backward steps, with zero expert/defense updates. Full evaluation uses frozen pilot merge weights.

## Evidence and limits

- Independently verified cases: 4/4; result rows: 244.
- TA coefficient 0.5; TIES top 20%, mass signs, disjoint mean, coefficient 1.0. Each has a separate unprotected control.
- Original AGA leaves final LN bias unchanged. Phase 1 accuracy collapse does not establish coverage against arbitrary recovery.
- Per-class counts, paired losses/gains, approximate paired 95% intervals and exact McNemar log-p are in results.json/CSV. Bonferroni correction covers all non-native protected comparisons in this generated batch.
- Figures show fixed deterministic configurations, not seed means. Significance is not a defense-success definition.
- No DARE, full TSV-Merge, data-assisted adaptation or distillation result is claimed in this batch.

## Finite utility and parameter audit

Fixed 512 training images per task; read-only evaluation, historical overlap possible. This is not a global or fresh-confirmation certificate.

| Defense | Task | Changed-tensor distance | Worst margin drift | Inputs above 0.5 |
|---|---|---:|---:|---:|
| Phase1_MNIST | MNIST | 620.441 | 0.0010786 | 0/512 |
| O/B | MNIST | 0.29904 | 5.36907 | 1/512 |
| A/B | MNIST | 0.364139 | 39.9202 | 1/512 |
| PaRaMS | MNIST | 47.2573 | 4.68955e-05 | 0/512 |
| PaRaMS_Attention | MNIST | 0.264848 | 1.86113e-06 | 0/512 |
| Phase1_DTD | DTD | 902.068 | 0.0183107 | 0/512 |
| Phase1_SVHN | SVHN | 472.809 | 4.30566e-05 | 0/512 |
