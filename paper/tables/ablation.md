Equal-norm direction controls (%). Within each task, NBP and the two random controls have the same update norm before FP32 storage. Random nullspace projects a fixed random vector into the expert projection kernel. Random direction uses that vector before projection. Companions and TA coefficient 0.5 match the main table.

| Task | Update | Standalone | TA | AGA+TA |
| --- | --- | ---: | ---: | ---: |
| MNIST | None | 99.69 | 99.37 | 99.30 |
| MNIST | NBP | 99.69 | 9.80 | 9.80 |
| MNIST | Random nullspace | 99.69 | 65.21 | 63.74 |
| MNIST | Random direction | 9.74 | 9.74 | 9.74 |
| SVHN | None | 97.46 | 92.50 | 91.56 |
| SVHN | NBP | 97.46 | 6.70 | 6.70 |
| SVHN | Random nullspace | 97.46 | 60.13 | 59.22 |
| SVHN | Random direction | 6.38 | 6.38 | 6.38 |
| DTD | None | 79.41 | 69.31 | 68.40 |
| DTD | NBP | 79.41 | 2.13 | 2.13 |
| DTD | Random nullspace | 79.41 | 2.13 | 2.13 |
| DTD | Random direction | 2.13 | 2.13 | 2.13 |
