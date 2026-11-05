# Defense 2 in eight-expert merging

This table uses one frozen Defense 2 release. Eight experts, TA coefficient 0.2; AGA processes only the protected expert. Seven complete test datasets, including all evaluated partner tasks. EuroSAT is excluded; no eight-task mean. Lower protected-task accuracy favors protection; losses on other tasks are reported without averaging.

| Target | Defense | Processing | MNIST | SVHN | DTD | GTSRB | RESISC45 | Cars | SUN397 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | TA | 94.00 | 74.42 | 52.18 | 65.08 | 72.00 | 62.06 | 63.76 |
| MNIST | Defense 2 | TA | 24.71 | 13.16 | 15.21 | 17.14 | 50.10 | 34.87 | 19.73 |
| MNIST | Unprotected | AGA | 93.23 | 73.90 | 52.34 | 65.16 | 72.19 | 62.02 | 63.78 |
| MNIST | Defense 2 | AGA | 23.35 | 12.88 | 12.18 | 14.91 | 42.22 | 31.95 | 10.02 |
| SVHN | Unprotected | TA | 94.00 | 74.42 | 52.18 | 65.08 | 72.00 | 62.06 | 63.76 |
| SVHN | Defense 2 | TA | 13.56 | 15.25 | 7.61 | 8.45 | 16.27 | 8.99 | 1.06 |
| SVHN | Unprotected | AGA | 93.78 | 72.53 | 52.18 | 64.87 | 72.19 | 62.04 | 63.80 |
| SVHN | Defense 2 | AGA | 13.95 | 11.03 | 6.86 | 7.01 | 14.10 | 7.37 | 0.91 |
| DTD | Unprotected | TA | 94.00 | 74.42 | 52.18 | 65.08 | 72.00 | 62.06 | 63.76 |
| DTD | Defense 2 | TA | 94.45 | 74.81 | 4.84 | 21.08 | 65.30 | 52.88 | 50.26 |
| DTD | Unprotected | AGA | 94.06 | 74.46 | 51.38 | 65.04 | 72.05 | 62.03 | 63.76 |
| DTD | Defense 2 | AGA | 94.36 | 74.85 | 3.24 | 14.96 | 58.14 | 48.48 | 43.19 |
