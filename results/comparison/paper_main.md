# Standalone and two-expert merging

Standalone accuracy refers to the target expert. Each merged cell gives target / partner accuracy. TA uses coefficient 0.5. AGA processes the target before merging, with the same procedure for the unprotected control. Defense 2 uses the frozen A/B release. All entries use complete test sets.

| Target / partner | Defense | Standalone | TA | AGA + TA |
| --- | --- | --- | --- | --- |
| MNIST/SVHN | Unprotected | 99.69 | 99.37 / 92.50 | 99.30 / 92.78 |
| MNIST/SVHN | PaRaMS | 99.69 | 8.92 / 9.15 | 99.30 / 92.79 |
| MNIST/SVHN | Defense 1 | 99.69 | 9.80 / 6.70 | 9.80 / 6.70 |
| MNIST/SVHN | Defense 2 | 99.57 | 25.15 / 13.26 | 2.99 / 2.02 |
| SVHN/MNIST | Unprotected | 97.46 | 92.50 / 99.37 | 91.56 / 99.39 |
| SVHN/MNIST | PaRaMS | 97.46 | 9.09 / 8.95 | 91.56 / 99.39 |
| SVHN/MNIST | Defense 1 | 97.46 | 6.70 / 9.80 | 6.70 / 9.80 |
| SVHN/MNIST | Defense 2 | 96.66 | 2.26 / 0.22 | 1.86 / 0.24 |
| DTD/MNIST | Unprotected | 79.41 | 69.31 / 99.45 | 68.40 / 99.45 |
| DTD/MNIST | PaRaMS | 79.41 | 2.07 / 8.92 | 68.35 / 99.45 |
| DTD/MNIST | Defense 1 | 79.41 | 2.13 / 9.80 | 2.13 / 9.80 |
| DTD/MNIST | Defense 2 | 76.33 | 2.13 / 99.22 | 2.07 / 99.02 |
