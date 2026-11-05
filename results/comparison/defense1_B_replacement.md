# Defense 1 under public MLP output replacement

For both defenses in the table, replace the target's final MLP output weight and bias with the public base values. Replacement follows AGA and precedes TA at 0.5. Standalone is measured after replacement. AGA-T processes the target and AGA-B both experts. Defense 1 modifies the final LayerNorm bias, which this operation leaves intact.

| Target | Dataset | Defense | Standalone | TA | AGA-T | AGA-B |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | MNIST | Unprotected | 99.68 | 99.35 | 99.29 | 99.30 |
| MNIST | MNIST | Defense 1 | 99.68 | 9.80 | 9.80 | 9.80 |
| MNIST | SVHN | Unprotected | -- | 92.45 | 92.79 | 91.72 |
| MNIST | SVHN | Defense 1 | -- | 6.70 | 6.70 | 6.70 |
| SVHN | SVHN | Unprotected | 97.43 | 92.40 | 91.47 | 91.74 |
| SVHN | SVHN | Defense 1 | 97.43 | 6.70 | 6.70 | 6.70 |
| SVHN | MNIST | Unprotected | -- | 99.38 | 99.39 | 99.33 |
| SVHN | MNIST | Defense 1 | -- | 9.80 | 9.80 | 9.80 |
| DTD | DTD | Unprotected | 79.15 | 68.19 | 67.18 | 67.71 |
| DTD | DTD | Defense 1 | 79.15 | 2.13 | 2.13 | 2.13 |
| DTD | MNIST | Unprotected | -- | 99.47 | 99.45 | 99.42 |
| DTD | MNIST | Defense 1 | -- | 9.80 | 9.80 | 9.80 |
