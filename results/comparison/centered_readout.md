# Centered affine readout transfer

The frozen rule is the same for MNIST, SVHN and DTD. Construction uses weights, the public base and the fixed classifier. No images are used in construction. TA uses coefficient 0.5. ProDistill uses 64 unlabeled images and 100 epochs per block.

| Target | Standalone | TA | AGA target | AGA both | B reset after AGA | ProDistill | ProDistill + fixed sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | 99.57 | 9.74 | 9.74 | 9.74 | 53.73 | 10.32 | 10.09 |
| SVHN | 96.66 | 15.94 | 15.94 | 15.94 | 0.66 | 15.94 | 56.73 |
| DTD | 76.33 | 4.15 | 2.18 | 2.18 | 17.93 | 2.13 | 13.46 |

The same frozen publication preserves standalone predictions. The sign column applies one fixed score negation after the ProDistill result. It is reported because the earlier readout point exposed a reversible score-order failure.
