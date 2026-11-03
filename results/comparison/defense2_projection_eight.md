# Task projection in eight-expert merging

Eight experts participate in TA with coefficient 0.2. AGA processes the protected expert. Output calibration uses 64 random images. The removed condition drops the two projections used to construct A and keeps the task-specific synthetic inputs. Seven complete test sets are available. The original EuroSAT split is unverified.

| Task | Projection | Standalone | TA | +calib. | AGA | +calib. |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | 99.69 | 94.00 | 94.04 | 93.23 | 93.19 |
| MNIST | Retained | 99.57 | 24.71 | 24.03 | 23.35 | 22.99 |
| MNIST | Removed | 99.68 | 16.75 | 17.53 | 18.56 | 22.31 |
| SVHN | Unprotected | 97.46 | 74.42 | 74.61 | 72.53 | 72.82 |
| SVHN | Retained | 96.66 | 15.25 | 23.74 | 11.03 | 17.16 |
| SVHN | Removed | 97.41 | 27.24 | 21.94 | 30.91 | 29.48 |
| DTD | Unprotected | 79.41 | 52.18 | 38.67 | 51.38 | 38.35 |
| DTD | Retained | 76.33 | 4.84 | 26.17 | 3.24 | 28.46 |
| DTD | Removed | 77.87 | 2.07 | 7.87 | 2.34 | 2.39 |
