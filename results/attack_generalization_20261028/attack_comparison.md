# Attack comparison

All accuracies are percentages. Only completed, verified evaluations appear; missing entries are --.

| Target | Partner | Defense | Processing | AGA scope | Standalone | Target accuracy | Partner accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | SVHN | Unprotected | TA / AGA | None | 99.69 | 99.37 | 92.50 |
| MNIST | SVHN | Unprotected | TA / AGA | Protected expert | 99.69 | 99.30 | 92.78 |
| MNIST | SVHN | Unprotected | TA / AGA | All experts | 99.69 | 99.32 | 91.77 |
| MNIST | SVHN | NativeConstraintClass_MNIST | TA / AGA | None | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | NativeConstraintClass_MNIST | TA / AGA | Protected expert | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | NativeConstraintClass_MNIST | TA / AGA | All experts | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Without_both_SVD_components | TA / AGA | None | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Without_both_SVD_components | TA / AGA | Protected expert | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Without_both_SVD_components | TA / AGA | All experts | 99.69 | 9.80 | 6.70 |
| SVHN | MNIST | Unprotected | TA / AGA | None | 97.46 | 92.50 | 99.37 |
| SVHN | MNIST | Unprotected | TA / AGA | Protected expert | 97.46 | 91.56 | 99.39 |
| SVHN | MNIST | Unprotected | TA / AGA | All experts | 97.46 | 91.77 | 99.32 |
| SVHN | MNIST | NativeConstraintClass_SVHN | TA / AGA | None | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | NativeConstraintClass_SVHN | TA / AGA | Protected expert | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | NativeConstraintClass_SVHN | TA / AGA | All experts | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Without_both_SVD_components | TA / AGA | None | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Without_both_SVD_components | TA / AGA | Protected expert | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Without_both_SVD_components | TA / AGA | All experts | 97.45 | 6.70 | 9.80 |
| DTD | MNIST | Unprotected | TA / AGA | None | 79.41 | 69.31 | 99.45 |
| DTD | MNIST | Unprotected | TA / AGA | Protected expert | 79.41 | 68.40 | 99.45 |
| DTD | MNIST | Unprotected | TA / AGA | All experts | 79.41 | 68.78 | 99.42 |
| DTD | MNIST | NativeConstraintClass_DTD | TA / AGA | None | 76.12 | 2.13 | 98.44 |
| DTD | MNIST | NativeConstraintClass_DTD | TA / AGA | Protected expert | 76.12 | 2.13 | 75.64 |
| DTD | MNIST | NativeConstraintClass_DTD | TA / AGA | All experts | 76.12 | 2.13 | 68.61 |
| DTD | MNIST | Without_both_SVD_components | TA / AGA | None | 78.09 | 2.13 | 37.09 |
| DTD | MNIST | Without_both_SVD_components | TA / AGA | Protected expert | 78.09 | 2.13 | 17.44 |
| DTD | MNIST | Without_both_SVD_components | TA / AGA | All experts | 78.09 | 2.13 | 16.70 |
| MNIST | SVHN | Unprotected | Component deletion | None | 99.69 | 99.37 | 92.50 |
| MNIST | SVHN | Unprotected | Component deletion | Protected expert | 99.69 | 99.30 | 92.78 |
| MNIST | SVHN | Unprotected | Component deletion | All experts | 99.69 | 99.32 | 91.77 |
| MNIST | SVHN | A_only | Component deletion | None | 99.70 | 99.37 | 92.53 |
| MNIST | SVHN | A_only | Component deletion | Protected expert | 99.70 | 99.31 | 92.58 |
| MNIST | SVHN | A_only | Component deletion | All experts | 99.70 | 99.34 | 91.54 |
| MNIST | SVHN | B_only | Component deletion | None | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | B_only | Component deletion | Protected expert | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | B_only | Component deletion | All experts | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | Unprotected | 64-noise calibration | None | -- | 99.34 | 92.32 |
| MNIST | SVHN | Unprotected | 64-noise calibration | Protected expert | -- | 99.29 | 92.58 |
| MNIST | SVHN | Unprotected | 64-noise calibration | All experts | -- | 99.29 | 91.59 |
| MNIST | SVHN | NativeConstraintClass_MNIST | 64-noise calibration | None | -- | 89.21 | 85.53 |
| MNIST | SVHN | NativeConstraintClass_MNIST | 64-noise calibration | Protected expert | -- | 56.34 | 54.05 |
| MNIST | SVHN | NativeConstraintClass_MNIST | 64-noise calibration | All experts | -- | 56.28 | 52.43 |
| SVHN | MNIST | Unprotected | 64-noise calibration | None | -- | 92.32 | 99.34 |
| SVHN | MNIST | Unprotected | 64-noise calibration | Protected expert | -- | 91.34 | 99.34 |
| SVHN | MNIST | Unprotected | 64-noise calibration | All experts | -- | 91.59 | 99.29 |
| SVHN | MNIST | NativeConstraintClass_SVHN | 64-noise calibration | None | -- | 84.56 | 87.66 |
| SVHN | MNIST | NativeConstraintClass_SVHN | 64-noise calibration | Protected expert | -- | 53.84 | 69.99 |
| SVHN | MNIST | NativeConstraintClass_SVHN | 64-noise calibration | All experts | -- | 51.46 | 69.80 |
| MNIST | SVHN | Without_both_SVD_components | 64-noise calibration | None | -- | 49.28 | 46.07 |
| MNIST | SVHN | Without_both_SVD_components | 64-noise calibration | Protected expert | -- | 49.98 | 46.19 |
| MNIST | SVHN | Without_both_SVD_components | 64-noise calibration | All experts | -- | 49.95 | 45.19 |
| SVHN | MNIST | Without_both_SVD_components | 64-noise calibration | None | -- | 50.46 | 60.21 |
| SVHN | MNIST | Without_both_SVD_components | 64-noise calibration | Protected expert | -- | 51.31 | 62.71 |
| SVHN | MNIST | Without_both_SVD_components | 64-noise calibration | All experts | -- | 50.22 | 59.56 |
| DTD | MNIST | Unprotected | 64-noise calibration | None | -- | 55.69 | 99.46 |
| DTD | MNIST | Unprotected | 64-noise calibration | Protected expert | -- | 55.00 | 99.45 |
| DTD | MNIST | Unprotected | 64-noise calibration | All experts | -- | 55.00 | 99.39 |
| DTD | MNIST | Without_both_SVD_components | 64-noise calibration | None | -- | 57.18 | 99.42 |
| DTD | MNIST | Without_both_SVD_components | 64-noise calibration | Protected expert | -- | 52.77 | 99.42 |
| DTD | MNIST | Without_both_SVD_components | 64-noise calibration | All experts | -- | 52.77 | 99.38 |

Standalone in public-replacement rows is measured after replacement. Calibration rows reuse fused scores and do not imply a new standalone evaluation. No_target uses the public base in place of the entire target expert at the unchanged coefficient. A-only/B-only are frozen deletions, not separately optimized methods. Current coverage is three datasets, not eight.
