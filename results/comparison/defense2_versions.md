# Defense 2 versions on the protected task

The earlier and updated Defense 2 releases use distinct frozen construction rules; each row uses one release throughout. TA coefficient 0.5; AGA processes the protected expert. Calibration uses 64 random images. ProDistill uses 64 unlabeled training images per task and coefficient initialization 0.3, with no AGA preprocessing. MNIST is paired with SVHN; SVHN and DTD are paired with MNIST. Known-module replacement is reported separately; neither version resists it.

| Task | Version | Standalone | TA | AGA | AGA+calib. | MLP match | ProDistill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Earlier | 99.61 | 9.80 | 9.80 | 56.34 | 9.80 | 99.45 |
| MNIST | Updated | 99.57 | 25.15 | 2.99 | 10.76 | 9.39 | 99.43 |
| SVHN | Earlier | 96.98 | 6.70 | 6.70 | 53.84 | 6.70 | 95.78 |
| SVHN | Updated | 96.66 | 2.26 | 1.86 | 6.99 | 7.96 | 96.18 |
| DTD | Earlier | 76.12 | 2.13 | 2.13 | 52.61 | 2.13 | 63.83 |
| DTD | Updated | 76.33 | 2.13 | 2.07 | 5.00 | 2.29 | 66.70 |
