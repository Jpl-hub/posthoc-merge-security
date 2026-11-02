# Recovery of the fixed task-projection controls

The same fixed projection controls as the preceding ablation. MLP matching uses public weights without images. B reset replaces the last MLP output matrix and bias with the public base, requiring knowledge of the modified location; its unprotected control receives the same replacement. Both use TA coefficient 0.5. ProDistill uses 64 additional unlabeled training images per task, 100 epochs per block and coefficient initialization 0.3; no AGA preprocessing. Data, indices and training steps match the retained-projection comparison. These recovery conditions are separate, not one common attacker. Standalone columns precede recovery. MNIST and SVHN synthetic source-condition failures remain reported; no new defense updates.

| Task | Projection | Standalone | MLP match | B reset | ProDistill |
| --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | 99.69 | 99.37 | 99.35 | 99.45 |
| MNIST | Retained | 99.57 | 9.39 | 99.36 | 99.43 |
| MNIST | Removed | 99.68 | 11.14 | 99.34 | 99.07 |
| SVHN | Unprotected | 97.46 | 92.50 | 92.40 | 96.73 |
| SVHN | Retained | 96.66 | 7.96 | 92.32 | 96.18 |
| SVHN | Removed | 97.41 | 3.24 | 92.26 | 95.82 |
| DTD | Unprotected | 79.41 | 69.31 | 68.19 | 73.14 |
| DTD | Retained | 76.33 | 2.29 | 66.22 | 66.70 |
| DTD | Removed | 77.87 | 2.50 | 65.96 | 20.53 |
