# Partner accuracy for Defense 2

This table uses one frozen Defense 2 release throughout. TA coefficient 0.5; AGA processes the protected expert. Calibration uses 64 random images. ProDistill uses 64 unlabeled training images per task and coefficient initialization 0.3, with no AGA preprocessing. MNIST is paired with SVHN; SVHN and DTD are paired with MNIST. Known-module replacement is reported separately.

| Target | TA | AGA | AGA+calib. | MLP match | ProDistill |
| --- | --- | --- | --- | --- | --- |
| MNIST | 13.26 | 2.02 | 8.75 | 4.33 | 96.22 |
| SVHN | 0.22 | 0.24 | 11.53 | 10.29 | 99.27 |
| DTD | 99.22 | 99.02 | 99.47 | 99.13 | 99.51 |
