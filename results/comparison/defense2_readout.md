# Effect of adding readout compensation to Defense 2

TA uses coefficient 0.5. AGA-T processes the target and AGA-B both experts. B reset replaces the target's final MLP output weight and bias with public base values. In AGA-T + B, AGA precedes B reset, then TA. Each cell in that column gives raw accuracy followed by accuracy after negating every class score. ProDistill starts at coefficient 0.3 and uses 64 unlabeled training images per task and 100 epochs per block. Standalone accuracy is reported for the protected expert.

| Target | Dataset | Defense | Standalone | TA | AGA-T | AGA-B | B reset | AGA-T + B | ProDistill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | MNIST | A/B | 99.57 | 25.15 | 2.99 | 1.33 | 99.36 | 99.28 / 0.00 | 99.43 |
| MNIST | MNIST | A/B + readout | 99.57 | 0.07 | 0.08 | 0.09 | 0.05 | 0.07 / 61.78 | 98.63 |
| MNIST | SVHN | A/B | -- | 13.26 | 2.02 | 1.50 | 92.49 | 92.64 / 0.07 | 96.22 |
| MNIST | SVHN | A/B + readout | -- | 0.34 | 0.48 | 0.61 | 0.88 | 1.05 / 56.73 | 92.52 |
| SVHN | SVHN | A/B | 96.66 | 2.26 | 1.86 | 1.87 | 92.32 | 91.38 / 0.07 | 96.18 |
| SVHN | SVHN | A/B + readout | 96.66 | 7.76 | 7.76 | 7.76 | 0.38 | 0.45 / 62.45 | 83.45 |
| SVHN | MNIST | A/B | -- | 0.22 | 0.24 | 0.25 | 99.37 | 99.38 / 0.00 | 99.27 |
| SVHN | MNIST | A/B + readout | -- | 10.28 | 10.28 | 10.28 | 0.00 | 0.00 / 65.61 | 91.69 |
| DTD | DTD | A/B | 76.33 | 2.13 | 2.07 | 2.07 | 66.22 | 64.95 / 0.00 | 66.70 |
| DTD | DTD | A/B + readout | 76.33 | 1.81 | 1.70 | 1.76 | 0.53 | 0.59 / 11.86 | 30.85 |
| DTD | MNIST | A/B | -- | 99.22 | 99.02 | 98.81 | 99.44 | 99.47 / 0.00 | 99.51 |
| DTD | MNIST | A/B + readout | -- | 0.05 | 0.08 | 0.10 | 0.01 | 0.01 / 87.59 | 99.37 |
