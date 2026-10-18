# Paper-defined Unmask recovery: eight experts, three evaluated tasks

Fixed MergeGuard section7 operator: merged parameters minus0.3 times the whole published expert. TA0.2, one known protected expert; no AGA composition. Native CLIP paper-formula adaptation, not official execution or complete Table5 replication. Accuracy (%); no eight-task average.

## Protected task: MNIST

Cells: MNIST / DTD / SVHN (target / evaluated partners).

| Defense | No attack | Unmask (0.3) |
|---|---:|---:|
| Unprotected | 94.00 / 52.18 / 74.42 | 7.52 / 1.97 / 9.55 |
| Defense 1* | 9.80 / 5.85 / 6.70 | 10.28 / 2.07 / 7.76 |
| O/B† | 51.73 / 50.64 / 40.57 | 11.35 / 1.97 / 17.59 |
| A/B† | 61.94 / 50.21 / 53.09 | 10.89 / 2.13 / 12.85 |
| PaRaMS (full)* | 9.82 / 2.13 / 9.65 | 9.58 / 1.70 / 14.45 |
| PaRaMS (attention) | 94.00 / 52.13 / 74.45 | 7.51 / 1.97 / 9.55 |

## Protected task: DTD

Cells: DTD / MNIST / SVHN (target / evaluated partners).

| Defense | No attack | Unmask (0.3) |
|---|---:|---:|
| Unprotected | 52.18 / 94.00 / 74.42 | 2.55 / 10.00 / 9.56 |
| Defense 1* | 2.13 / 72.22 / 52.48 | 2.13 / 10.28 / 7.76 |

## Protected task: SVHN

Cells: SVHN / MNIST / DTD (target / evaluated partners).

| Defense | No attack | Unmask (0.3) |
|---|---:|---:|
| Unprotected | 74.42 / 94.00 / 52.18 | 9.15 / 7.91 / 2.13 |
| Defense 1* | 6.70 / 9.80 / 27.07 | 19.59 / 11.35 / 2.23 |

† O/B and A/B violate the task-preservation budget; effect references only. * Defense 1 has different constraints; full PaRaMS is over-budget. The attention ablation is not the complete PaRaMS method. Historically exposed tests; no tuning.

Per-class and paired statistics compare each row with its own unprotected control at the same attack scope. Bonferroni correction covers all 21 protected comparisons. Recovery is attacked minus unattacked protected accuracy; remaining gap is matched unprotected minus protected accuracy. Neither distance nor high accuracy proves unique-knowledge extraction. Independent NumPy metric recount is separate from the producer's bitwise NumPy reconstruction of all merged parameter tensors; this is not a second implementation of the complete author attack.
