# Effect of adding readout compensation to Defense 2

TA uses coefficient 0.5. AGA-T processes the target and AGA-B both experts. B reset replaces the target's final MLP output weight and bias before TA. ProDistill uses 64 unlabeled training images per task and 100 epochs per block. Dashes indicate unavailable results or a standalone partner measurement outside the protocol.

| Target | Dataset | Defense | Standalone | TA | AGA-T | AGA-B | B reset | AGA-T + reset | ProDistill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | MNIST | A/B | 99.57 | 25.15 | 2.99 | 1.33 | 99.36 | 99.28 | 99.43 |
| MNIST | MNIST | A/B + readout | 99.57 | 0.07 | 0.08 | 0.09 | 0.05 | 0.07 | 98.63 |
| MNIST | SVHN | A/B | -- | 13.26 | 2.02 | 1.50 | 92.49 | 92.64 | 96.22 |
| MNIST | SVHN | A/B + readout | -- | 0.34 | 0.48 | 0.61 | 0.88 | 1.05 | 92.52 |
| DTD | DTD | A/B | 76.33 | 2.13 | 2.07 | 2.07 | 66.22 | 64.95 | 66.70 |
| DTD | DTD | A/B + readout | 76.33 | 1.81 | 1.70 | 1.76 | 0.53 | 0.59 | -- |
| DTD | MNIST | A/B | -- | 99.22 | 99.02 | 98.81 | 99.44 | 99.47 | 99.51 |
| DTD | MNIST | A/B + readout | -- | 0.05 | 0.08 | 0.10 | 0.01 | 0.01 | -- |
