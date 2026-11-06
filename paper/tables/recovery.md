Protected-task accuracy after the specified recovery procedures (\%). B reset replaces the final MLP output weight and bias after AGA, then applies TA. ProDistill uses its fixed layer-wise fitting protocol; final LayerNorm and projection are not optimized. The control receives the same processing.

| Procedure | Defense | MNIST | SVHN | DTD |
| --- | --- | --- | --- | --- |
| B reset | None | 99.29 | 91.47 | 67.18 |
| B reset | NBP | 9.80 | 6.70 | 2.13 |
| ProDistill | None | 99.45 | 96.73 | 73.14 |
| ProDistill | PaRaMS | 89.27 | 66.22 | 7.29 |
| ProDistill | NBP | 9.80 | 6.70 | 2.13 |
