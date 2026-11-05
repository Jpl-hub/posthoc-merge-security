# Defense 2 recovery table

This table uses one frozen Defense 2 release throughout. TA coefficient 0.5; AGA processes the protected expert. Calibration uses 64 random images. ProDistill uses 64 unlabeled training images per task and coefficient initialization 0.3, with no AGA preprocessing. MNIST is paired with SVHN; SVHN and DTD are paired with MNIST. Known-module replacement is reported separately.

| Task | Standalone | TA | AGA | AGA+calib. | MLP match | ProDistill |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | 99.57 | 25.15 | 2.99 | 10.76 | 9.39 | 99.43 |
| SVHN | 96.66 | 2.26 | 1.86 | 6.99 | 7.96 | 96.18 |
| DTD | 76.33 | 2.13 | 2.07 | 5.00 | 2.29 | 66.70 |
