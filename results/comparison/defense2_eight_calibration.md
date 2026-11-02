# Output calibration in eight-expert merging

Same frozen updated Defense 2 releases; eight experts, TA coefficient 0.2. AGA processes the protected expert. Each merged model is calibrated independently using 64 fixed random images and its public classifier, without real images, labels or changed-module knowledge. Full archived scores are reused. Seven complete datasets; EuroSAT is excluded, and no eight-task mean is reported.

| Target | Defense | Processing | MNIST | SVHN | DTD | GTSRB | RESISC45 | Cars | SUN397 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | TA | 94.04 | 74.61 | 38.67 | 60.45 | 69.10 | 51.39 | 53.75 |
| MNIST | Defense 2 | TA | 24.03 | 26.74 | 44.52 | 31.80 | 57.97 | 50.70 | 41.78 |
| MNIST | Unprotected | AGA | 93.19 | 74.12 | 38.67 | 60.54 | 69.17 | 51.46 | 53.79 |
| MNIST | Defense 2 | AGA | 22.99 | 25.76 | 45.96 | 31.97 | 58.02 | 51.19 | 44.89 |
| SVHN | Unprotected | TA | 94.04 | 74.61 | 38.67 | 60.45 | 69.10 | 51.39 | 53.75 |
| SVHN | Defense 2 | TA | 28.70 | 23.74 | 39.47 | 33.30 | 48.19 | 33.60 | 36.97 |
| SVHN | Unprotected | AGA | 93.82 | 72.82 | 38.67 | 60.13 | 69.17 | 51.56 | 53.76 |
| SVHN | Defense 2 | AGA | 25.50 | 17.16 | 43.46 | 39.71 | 58.43 | 40.92 | 43.74 |
| DTD | Unprotected | TA | 94.04 | 74.61 | 38.67 | 60.45 | 69.10 | 51.39 | 53.75 |
| DTD | Defense 2 | TA | 94.19 | 74.41 | 26.17 | 54.80 | 66.24 | 53.35 | 51.87 |
| DTD | Unprotected | AGA | 94.07 | 74.67 | 38.35 | 60.40 | 69.19 | 51.52 | 53.79 |
| DTD | Defense 2 | AGA | 94.23 | 74.54 | 28.46 | 56.54 | 66.62 | 52.32 | 52.93 |
