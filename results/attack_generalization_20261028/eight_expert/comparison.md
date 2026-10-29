# Eight-expert fusion

Eight experts, TA coefficient 0.2. Only three full datasets are evaluated; no eight-task average. AGA processes only the protected expert. Same frozen releases; no reconstruction or tuning.

| Protected task | Defense | AGA | MNIST | SVHN | DTD |
| --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | None | 94.00 | 74.42 | 52.18 |
| MNIST | Defense 2 | None | 9.81 | 7.05 | 41.22 |
| MNIST | Unprotected | Target | 93.23 | 73.90 | 52.34 |
| MNIST | Defense 2 | Target | 9.80 | 6.70 | 33.35 |
| SVHN | Unprotected | None | 94.00 | 74.42 | 52.18 |
| SVHN | Defense 2 | None | 9.80 | 6.70 | 32.66 |
| SVHN | Unprotected | Target | 93.78 | 72.53 | 52.18 |
| SVHN | Defense 2 | Target | 9.80 | 6.70 | 33.46 |
| DTD | Unprotected | None | 94.00 | 74.42 | 52.18 |
| DTD | Defense 2 | None | 90.72 | 65.04 | 2.39 |
| DTD | Unprotected | Target | 94.06 | 74.46 | 51.38 |
| DTD | Defense 2 | Target | 76.66 | 54.28 | 2.13 |
