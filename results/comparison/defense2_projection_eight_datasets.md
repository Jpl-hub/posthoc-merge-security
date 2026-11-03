# Seven test sets with the task projection removed

Eight experts participate in TA with coefficient 0.2. AGA processes the protected expert. Output calibration uses 64 random images. The removed condition drops the two projections used to construct A and keeps the task-specific synthetic inputs. Seven complete test sets are available. The original EuroSAT split is unverified.

| Protected task | Processing | MNIST | SVHN | DTD | GTSRB | RESISC45 | Cars | SUN397 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | TA | 16.75 | 12.31 | 2.23 | 14.82 | 10.16 | 4.78 | 1.13 |
| MNIST | TA + calibration | 17.53 | 12.59 | 43.67 | 30.29 | 60.70 | 45.28 | 44.58 |
| MNIST | AGA | 18.56 | 14.70 | 2.61 | 8.97 | 14.65 | 7.41 | 1.24 |
| MNIST | AGA + calibration | 22.31 | 21.55 | 44.15 | 33.09 | 61.54 | 45.52 | 44.57 |
| SVHN | TA | 20.94 | 27.24 | 4.84 | 3.54 | 2.13 | 1.89 | 0.62 |
| SVHN | TA + calibration | 18.73 | 21.94 | 42.07 | 36.51 | 67.81 | 44.45 | 50.49 |
| SVHN | AGA | 20.89 | 30.91 | 5.59 | 3.56 | 2.19 | 2.38 | 0.81 |
| SVHN | AGA + calibration | 38.58 | 29.48 | 43.30 | 47.47 | 67.90 | 48.46 | 50.73 |
| DTD | TA | 81.19 | 58.77 | 2.07 | 0.69 | 2.97 | 3.52 | 0.87 |
| DTD | TA + calibration | 93.74 | 72.65 | 7.87 | 39.94 | 64.94 | 42.59 | 46.73 |
| DTD | AGA | 80.62 | 59.25 | 2.34 | 0.74 | 4.51 | 3.46 | 0.74 |
| DTD | AGA + calibration | 94.15 | 74.57 | 2.39 | 40.27 | 60.57 | 41.94 | 38.98 |
