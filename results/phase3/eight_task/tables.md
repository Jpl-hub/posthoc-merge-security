# Eight participating experts; three evaluated tasks

Original author task order, TA0.2, and AGA applied to the protected expert. Accuracy (%); no eight-task average. The other five partner datasets were not evaluated. This setting changes both coefficient and participant count relative to the two-expert TA0.5 table.

## Protected task: MNIST

Cells: MNIST / DTD / SVHN (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 94.00 / 52.18 / 74.42 | 93.23 / 52.34 / 73.90 |
| Defense 1* | 9.80 / 5.85 / 6.70 | 9.80 / 5.90 / 6.70 |
| O/B† | 51.73 / 50.64 / 40.57 | 54.22 / 50.80 / 44.63 |
| A/B† | 61.94 / 50.21 / 53.09 | 59.16 / 50.59 / 51.97 |
| PaRaMS (full)* | 9.82 / 2.13 / 9.65 | 93.23 / 52.34 / 73.90 |
| PaRaMS (attention) | 94.00 / 52.13 / 74.45 | 93.23 / 52.34 / 73.90 |

## Protected task: DTD

Cells: DTD / MNIST / SVHN (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 52.18 / 94.00 / 74.42 | 51.38 / 94.06 / 74.46 |
| Defense 1* | 2.13 / 72.22 / 52.48 | 2.13 / 72.30 / 52.47 |

## Protected task: SVHN

Cells: SVHN / MNIST / DTD (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 74.42 / 94.00 / 52.18 | 72.53 / 93.78 / 52.18 |
| Defense 1* | 6.70 / 9.80 / 27.07 | 6.70 / 9.80 / 27.07 |

† O/B and A/B violate the task-preservation budget; effect references only. * Defense 1 has different constraints; full PaRaMS is over-budget. The attention ablation is not the complete PaRaMS method. Historically exposed tests; no tuning.

Per-class and paired statistics compare each row with its own unprotected control at the same attack scope. Bonferroni correction covers all 42 protected comparisons. Recovery is attacked minus unattacked protected accuracy; remaining gap is matched unprotected minus protected accuracy. Neither distance nor high accuracy proves unique-knowledge extraction. Independent NumPy metric recount is separate from the producer's bitwise NumPy reconstruction of all merged parameter tensors; this is not a second implementation of the complete author attack.
