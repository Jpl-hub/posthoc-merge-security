# Defense 1 on additional expert pairs

All entries use complete test sets. Standalone reports target accuracy. Each merged cell gives target / partner accuracy. TA uses coefficient 0.5. AGA processes the target before merging, including for the unprotected control.

| Target / partner | Defense | Standalone | TA | AGA + TA |
| --- | --- | --- | --- | --- |
| RESISC45/MNIST | Unprotected | 96.11 | 91.08 / 99.43 | 90.38 / 99.41 |
| RESISC45/MNIST | PaRaMS | 96.11 | 2.57 / 8.92 | 90.37 / 99.41 |
| RESISC45/MNIST | Defense 1 | 96.11 | 2.44 / 27.60 | 2.44 / 27.56 |
| Cars/MNIST | Unprotected | 77.66 | 73.17 / 99.41 | 72.53 / 99.41 |
| Cars/MNIST | PaRaMS | 77.66 | 0.72 / 8.92 | 72.53 / 99.41 |
| Cars/MNIST | Defense 1 | 77.64 | 0.55 / 10.10 | 0.55 / 10.10 |
| GTSRB/MNIST | Unprotected | 98.73 | 93.25 / 99.12 | 92.22 / 99.17 |
| GTSRB/MNIST | PaRaMS | 98.73 | 2.79 / 8.92 | 92.22 / 99.17 |
| GTSRB/MNIST | Defense 1 | 98.73 | 1.19 / 35.83 | 1.19 / 35.80 |
| SUN397/MNIST | Unprotected | 75.28 | 71.92 / 99.46 | 71.58 / 99.47 |
| SUN397/MNIST | PaRaMS | 75.28 | 0.30 / 8.92 | 71.58 / 99.47 |
| SUN397/MNIST | Defense 1 | 75.27 | 0.25 / 10.28 | 0.25 / 10.28 |
