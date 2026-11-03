# Components of the current Defense 2

Same current Defense 2 releases. A and B include their respective biases. A-only and B-only retain the selected changes without refitting. Gaussian perturbations match each of the four tensor-change norms, with one fixed seed. TA coefficient 0.5; AGA-T processes the protected expert and AGA-B both experts. Low merged accuracy is useful only alongside preserved standalone accuracy. No output calibration or distillation is added to these component comparisons.

| Task | Variant | Standalone | TA | AGA-T | AGA-B |
| --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | 99.69 | 99.37 | 99.30 | 99.32 |
| MNIST | A only | 99.68 | 99.37 | 99.33 | 99.35 |
| MNIST | B only | 0.06 | 16.49 | 15.73 | 14.98 |
| MNIST | Gaussian | 99.67 | 99.34 | 99.31 | 99.33 |
| MNIST | A+B | 99.57 | 25.15 | 2.99 | 1.33 |
| SVHN | Unprotected | 97.46 | 92.50 | 91.56 | 91.77 |
| SVHN | A only | 97.41 | 92.44 | 91.46 | 91.74 |
| SVHN | B only | 0.55 | 2.40 | 2.52 | 2.44 |
| SVHN | Gaussian | 94.20 | 92.76 | 91.81 | 92.13 |
| SVHN | A+B | 96.66 | 2.26 | 1.86 | 1.87 |
| DTD | Unprotected | 79.41 | 69.31 | 68.40 | 68.78 |
| DTD | A only | 78.62 | 68.03 | 65.00 | 65.37 |
| DTD | B only | 1.86 | 2.39 | 2.45 | 2.45 |
| DTD | Gaussian | 66.33 | 69.63 | 67.87 | 67.87 |
| DTD | A+B | 76.33 | 2.13 | 2.07 | 2.07 |
