# Protected-task accuracy before and after output calibration

Same frozen updated Defense 2 releases; eight experts, TA coefficient 0.2. AGA processes the protected expert. Each merged model is calibrated independently using 64 fixed random images and its public classifier, without real images, labels or changed-module knowledge. Full archived scores are reused. Seven complete datasets; EuroSAT is excluded, and no eight-task mean is reported.

| Task | Processing | Unprotected | +calib. | Defense 2 | +calib. |
| --- | --- | --- | --- | --- | --- |
| MNIST | TA | 94.00 | 94.04 | 24.71 | 24.03 |
| MNIST | AGA | 93.23 | 93.19 | 23.35 | 22.99 |
| SVHN | TA | 74.42 | 74.61 | 15.25 | 23.74 |
| SVHN | AGA | 72.53 | 72.82 | 11.03 | 17.16 |
| DTD | TA | 52.18 | 38.67 | 4.84 | 26.17 |
| DTD | AGA | 51.38 | 38.35 | 3.24 | 28.46 |
