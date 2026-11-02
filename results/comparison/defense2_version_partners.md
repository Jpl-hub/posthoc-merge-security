# Partner accuracy for the same Defense 2 versions

The earlier and updated Defense 2 releases use distinct frozen construction rules; each row uses one release throughout. TA coefficient 0.5; AGA processes the protected expert. Calibration uses 64 random images. ProDistill uses 64 unlabeled training images per task and coefficient initialization 0.3, with no AGA preprocessing. MNIST is paired with SVHN; SVHN and DTD are paired with MNIST. Known-module replacement is reported separately; neither version resists it.

| Target | Version | TA | AGA | AGA+calib. | MLP match | ProDistill |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | Earlier | 6.70 | 6.70 | 54.05 | 6.70 | 96.49 |
| MNIST | Updated | 13.26 | 2.02 | 8.75 | 4.33 | 96.22 |
| SVHN | Earlier | 9.80 | 9.80 | 69.99 | 9.80 | 99.28 |
| SVHN | Updated | 0.22 | 0.24 | 11.53 | 10.29 | 99.27 |
| DTD | Earlier | 98.44 | 75.64 | 99.42 | 96.78 | 99.51 |
| DTD | Updated | 99.22 | 99.02 | 99.47 | 99.13 | 99.51 |
