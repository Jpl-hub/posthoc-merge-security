# Eight participating experts; three evaluated tasks

Original author task order, TA0.2, and AGA applied to the protected expert. Accuracy (%); no eight-task average. The other five partner datasets were not evaluated. This setting changes both coefficient and participant count relative to the two-expert TA0.5 table.

## Protected task: MNIST

Cells: MNIST / DTD / SVHN (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 94.00 / 52.18 / 74.42 | 93.23 / 52.34 / 73.90 |
| Defense 2 | 10.26 / 4.31 / 16.18 | 11.35 / 2.82 / 15.06 |

## Protected task: DTD

Cells: DTD / MNIST / SVHN (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 52.18 / 94.00 / 74.42 | 51.38 / 94.06 / 74.46 |
| Defense 2 | 3.24 / 8.87 / 7.11 | 4.89 / 8.85 / 8.63 |

## Protected task: SVHN

Cells: SVHN / MNIST / DTD (target / evaluated partners).

| Defense | No attack | AGA (protected expert) |
|---|---:|---:|
| Unprotected | 74.42 / 94.00 / 52.18 | 72.53 / 93.78 / 52.18 |
| Defense 2 | 13.28 / 17.24 / 5.90 | 11.64 / 8.93 / 4.89 |

Same frozen task-SVD-directed releases as the two-expert TA0.5 experiment; no reconstruction or selection from these scores. All three standalone gates already passed. Known-family global-coordinate recovery still applies. Historically exposed tests; no tuning. Eight experts participate but only MNIST,DTD,SVHN are evaluated; no eight-task average. No all-expert AGA run in this setting.

Per-class and paired statistics compare each row with its own unprotected control at the same attack scope. Bonferroni correction covers all 18 protected comparisons. Recovery is attacked minus unattacked protected accuracy; remaining gap is matched unprotected minus protected accuracy. Neither distance nor high accuracy proves unique-knowledge extraction. Independent NumPy metric recount is separate from the producer's bitwise NumPy reconstruction of all merged parameter tensors; this is not a second implementation of the complete author attack.
