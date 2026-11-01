# ProDistill recovery comparison

Two-expert adaptation of the fixed author coefficient kernel. The attacker uses 64 unlabeled training images per task, task identity, and 100 epochs per block. Initial coefficient 0.3; no AGA preprocessing. Final LayerNorm and projection are not optimized. Defense 1 reuses identical unprotected blocks. Lower target accuracy favors protection; partner accuracy is reported separately. Standalone accuracy is reported in the main attack table.

| Target | Defense | Target accuracy | Partner | Partner accuracy |
| --- | --- | --- | --- | --- |
| MNIST | Unprotected | 99.45 | SVHN | 96.73 |
| MNIST | PaRaMS | 89.27 | SVHN | 64.77 |
| MNIST | Defense 1 | 9.80 | SVHN | 6.70 |
| MNIST | Defense 2 | 99.45 | SVHN | 96.49 |
| SVHN | Unprotected | 96.73 | MNIST | 99.45 |
| SVHN | PaRaMS | 66.22 | MNIST | 89.55 |
| SVHN | Defense 1 | 6.70 | MNIST | 9.80 |
| SVHN | Defense 2 | 95.78 | MNIST | 99.28 |
| DTD | Unprotected | 73.14 | MNIST | 99.51 |
| DTD | PaRaMS | 7.29 | MNIST | 88.39 |
| DTD | Defense 1 | 2.13 | MNIST | 30.51 |
| DTD | Defense 2 | 63.83 | MNIST | 99.51 |
