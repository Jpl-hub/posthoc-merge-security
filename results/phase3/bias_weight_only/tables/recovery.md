Target accuracy after recovery (%). B reset replaces the final MLP output weight and bias after AGA, then applies TA. ProDistill fits block coefficients; final LayerNorm and projection are fixed. Bias projection modifies the final normalization bias before TA. Each unprotected control receives the same operation. NBP and control have identical predictions after bias projection.

| Procedure | Defense | MNIST | SVHN | DTD |
| --- | --- | ---: | ---: | ---: |
| B reset | None | 99.29 | 91.47 | 67.18 |
| B reset | NBP | 9.80 | 6.70 | 2.13 |
| ProDistill | None | 99.45 | 96.73 | 73.14 |
| ProDistill | PaRaMS | 89.27 | 66.22 | 7.29 |
| ProDistill | NBP | 9.80 | 6.70 | 2.13 |
| Bias projection | None | 99.37 | 92.50 | 69.26 |
| Bias projection | NBP | 99.37 | 92.50 | 69.26 |
