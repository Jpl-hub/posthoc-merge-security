# Attack comparison

All accuracies are percentages. Only completed, verified evaluations appear; missing entries are --.

| Target | Partner | Defense | Processing | AGA scope | Standalone | Target accuracy | Partner accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | SVHN | Unprotected | TA / AGA | None | 99.69 | 99.37 | 92.50 |
| MNIST | SVHN | Unprotected | TA / AGA | Protected expert | 99.69 | 99.30 | 92.78 |
| MNIST | SVHN | Unprotected | TA / AGA | All experts | 99.69 | 99.32 | 91.77 |
| MNIST | SVHN | Defense 2 | TA / AGA | None | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 | TA / AGA | Protected expert | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 | TA / AGA | All experts | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | TA / AGA | None | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | TA / AGA | Protected expert | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | TA / AGA | All experts | 99.69 | 9.80 | 6.70 |
| SVHN | MNIST | Unprotected | TA / AGA | None | 97.46 | 92.50 | 99.37 |
| SVHN | MNIST | Unprotected | TA / AGA | Protected expert | 97.46 | 91.56 | 99.39 |
| SVHN | MNIST | Unprotected | TA / AGA | All experts | 97.46 | 91.77 | 99.32 |
| SVHN | MNIST | Defense 2 | TA / AGA | None | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 | TA / AGA | Protected expert | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 | TA / AGA | All experts | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | TA / AGA | None | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | TA / AGA | Protected expert | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | TA / AGA | All experts | 97.45 | 6.70 | 9.80 |
| DTD | MNIST | Unprotected | TA / AGA | None | 79.41 | 69.31 | 99.45 |
| DTD | MNIST | Unprotected | TA / AGA | Protected expert | 79.41 | 68.40 | 99.45 |
| DTD | MNIST | Unprotected | TA / AGA | All experts | 79.41 | 68.78 | 99.42 |
| DTD | MNIST | Defense 2 | TA / AGA | None | 76.12 | 2.13 | 98.44 |
| DTD | MNIST | Defense 2 | TA / AGA | Protected expert | 76.12 | 2.13 | 75.64 |
| DTD | MNIST | Defense 2 | TA / AGA | All experts | 76.12 | 2.13 | 68.61 |
| DTD | MNIST | Defense 2 w/o SVD | TA / AGA | None | 78.09 | 2.13 | 37.09 |
| DTD | MNIST | Defense 2 w/o SVD | TA / AGA | Protected expert | 78.09 | 2.13 | 17.44 |
| DTD | MNIST | Defense 2 w/o SVD | TA / AGA | All experts | 78.09 | 2.13 | 16.70 |
| MNIST | SVHN | A only | TA / AGA | None | 99.70 | 99.37 | 92.53 |
| MNIST | SVHN | A only | TA / AGA | Protected expert | 99.70 | 99.31 | 92.58 |
| MNIST | SVHN | A only | TA / AGA | All experts | 99.70 | 99.34 | 91.54 |
| MNIST | SVHN | B+b only | TA / AGA | None | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | B+b only | TA / AGA | Protected expert | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | B+b only | TA / AGA | All experts | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | Unprotected | 64-noise calibration | None | 99.69 | 99.34 | 92.32 |
| MNIST | SVHN | Unprotected | 64-noise calibration | Protected expert | 99.69 | 99.29 | 92.58 |
| MNIST | SVHN | Unprotected | 64-noise calibration | All experts | 99.69 | 99.29 | 91.59 |
| MNIST | SVHN | Defense 2 | 64-noise calibration | None | 99.61 | 89.21 | 85.53 |
| MNIST | SVHN | Defense 2 | 64-noise calibration | Protected expert | 99.61 | 56.34 | 54.05 |
| MNIST | SVHN | Defense 2 | 64-noise calibration | All experts | 99.61 | 56.28 | 52.43 |
| SVHN | MNIST | Unprotected | 64-noise calibration | None | 97.46 | 92.32 | 99.34 |
| SVHN | MNIST | Unprotected | 64-noise calibration | Protected expert | 97.46 | 91.34 | 99.34 |
| SVHN | MNIST | Unprotected | 64-noise calibration | All experts | 97.46 | 91.59 | 99.29 |
| SVHN | MNIST | Defense 2 | 64-noise calibration | None | 96.98 | 84.56 | 87.66 |
| SVHN | MNIST | Defense 2 | 64-noise calibration | Protected expert | 96.98 | 53.84 | 69.99 |
| SVHN | MNIST | Defense 2 | 64-noise calibration | All experts | 96.98 | 51.46 | 69.80 |
| MNIST | SVHN | Defense 2 w/o SVD | 64-noise calibration | None | 99.69 | 49.28 | 46.07 |
| MNIST | SVHN | Defense 2 w/o SVD | 64-noise calibration | Protected expert | 99.69 | 49.98 | 46.19 |
| MNIST | SVHN | Defense 2 w/o SVD | 64-noise calibration | All experts | 99.69 | 49.95 | 45.19 |
| SVHN | MNIST | Defense 2 w/o SVD | 64-noise calibration | None | 97.45 | 50.46 | 60.21 |
| SVHN | MNIST | Defense 2 w/o SVD | 64-noise calibration | Protected expert | 97.45 | 51.31 | 62.71 |
| SVHN | MNIST | Defense 2 w/o SVD | 64-noise calibration | All experts | 97.45 | 50.22 | 59.56 |
| DTD | MNIST | Unprotected | 64-noise calibration | None | 79.41 | 55.69 | 99.46 |
| DTD | MNIST | Unprotected | 64-noise calibration | Protected expert | 79.41 | 55.00 | 99.45 |
| DTD | MNIST | Unprotected | 64-noise calibration | All experts | 79.41 | 55.00 | 99.39 |
| DTD | MNIST | Defense 2 w/o SVD | 64-noise calibration | None | 78.09 | 57.18 | 99.42 |
| DTD | MNIST | Defense 2 w/o SVD | 64-noise calibration | Protected expert | 78.09 | 52.77 | 99.42 |
| DTD | MNIST | Defense 2 w/o SVD | 64-noise calibration | All experts | 78.09 | 52.77 | 99.38 |
| MNIST | SVHN | Defense 1 | TA / AGA (reference) | None | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 1 | TA / AGA (reference) | Protected expert | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 1 | TA / AGA (reference) | All experts | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Task-directed permutation | TA / AGA (reference) | None | 99.69 | 10.98 | 14.32 |
| MNIST | SVHN | Task-directed permutation | TA / AGA (reference) | Protected expert | 99.69 | 9.80 | 9.81 |
| MNIST | SVHN | Task-directed permutation | TA / AGA (reference) | All experts | 99.69 | 9.80 | 9.83 |
| DTD | MNIST | Task-directed permutation | TA / AGA (reference) | None | 79.41 | 2.39 | 10.29 |
| DTD | MNIST | Task-directed permutation | TA / AGA (reference) | Protected expert | 79.41 | 2.39 | 10.09 |
| DTD | MNIST | Task-directed permutation | TA / AGA (reference) | All experts | 79.41 | 2.39 | 10.09 |
| SVHN | MNIST | Task-directed permutation | TA / AGA (reference) | None | 97.46 | 7.30 | 9.45 |
| SVHN | MNIST | Task-directed permutation | TA / AGA (reference) | Protected expert | 97.46 | 6.45 | 12.19 |
| SVHN | MNIST | Task-directed permutation | TA / AGA (reference) | All experts | 97.46 | 6.45 | 12.22 |
| MNIST | SVHN | Unprotected | Public replacement A | None | 99.68 | 99.37 | 92.51 |
| MNIST | SVHN | Defense 2 | Public replacement A | None | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | Public replacement A | None | 9.80 | 9.80 | 6.70 |
| MNIST | SVHN | No target | Public replacement A | None | 48.25 | 84.06 | 95.76 |
| MNIST | SVHN | Unprotected | Public replacement B+b | None | 99.68 | 99.35 | 92.45 |
| MNIST | SVHN | Defense 2 | Public replacement B+b | None | 99.68 | 99.35 | 92.49 |
| MNIST | SVHN | Defense 2 w/o SVD | Public replacement B+b | None | 99.68 | 99.33 | 92.37 |
| MNIST | SVHN | Unprotected | Public replacement MLP | None | 99.67 | 99.35 | 92.44 |
| MNIST | SVHN | Defense 2 | Public replacement MLP | None | 99.67 | 99.35 | 92.44 |
| MNIST | SVHN | Defense 2 w/o SVD | Public replacement MLP | None | 99.67 | 99.35 | 92.44 |
| SVHN | MNIST | Unprotected | Public replacement A | None | 97.42 | 92.45 | 99.37 |
| SVHN | MNIST | Defense 2 | Public replacement A | None | 6.70 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | Public replacement A | None | 6.70 | 6.70 | 9.80 |
| SVHN | MNIST | No target | Public replacement A | None | 31.61 | 52.92 | 99.46 |
| SVHN | MNIST | Unprotected | Public replacement B+b | None | 97.43 | 92.40 | 99.38 |
| SVHN | MNIST | Defense 2 | Public replacement B+b | None | 97.42 | 92.39 | 99.37 |
| SVHN | MNIST | Defense 2 w/o SVD | Public replacement B+b | None | 97.46 | 92.26 | 99.35 |
| SVHN | MNIST | Unprotected | Public replacement MLP | None | 97.38 | 92.35 | 99.37 |
| SVHN | MNIST | Defense 2 | Public replacement MLP | None | 97.38 | 92.35 | 99.37 |
| SVHN | MNIST | Defense 2 w/o SVD | Public replacement MLP | None | 97.38 | 92.35 | 99.37 |
| DTD | MNIST | Unprotected | Public replacement A | None | 79.41 | 68.88 | 99.46 |
| DTD | MNIST | Defense 2 | Public replacement A | None | 3.56 | 2.13 | 97.21 |
| DTD | MNIST | Defense 2 w/o SVD | Public replacement A | None | 2.13 | 2.13 | 33.72 |
| DTD | MNIST | No target | Public replacement A | None | 44.41 | 39.52 | 99.46 |
| DTD | MNIST | Unprotected | Public replacement B+b | None | 79.15 | 68.19 | 99.47 |
| DTD | MNIST | Defense 2 | Public replacement B+b | None | 78.94 | 66.81 | 99.45 |
| DTD | MNIST | Defense 2 w/o SVD | Public replacement B+b | None | 78.03 | 65.32 | 99.47 |
| DTD | MNIST | Unprotected | Public replacement MLP | None | 78.99 | 67.45 | 99.47 |
| DTD | MNIST | Defense 2 | Public replacement MLP | None | 78.99 | 67.45 | 99.47 |
| DTD | MNIST | Defense 2 w/o SVD | Public replacement MLP | None | 78.99 | 67.45 | 99.47 |
| MNIST | SVHN | Random noise | TA / AGA | None | 99.65 | 99.34 | 92.59 |
| MNIST | SVHN | Random noise | TA / AGA | Protected expert | 99.65 | 99.30 | 92.87 |
| MNIST | SVHN | Random noise | TA / AGA | All experts | 99.65 | 99.30 | 91.86 |
| SVHN | MNIST | Random noise | TA / AGA | None | 97.30 | 92.66 | 99.35 |
| SVHN | MNIST | Random noise | TA / AGA | Protected expert | 97.30 | 91.71 | 99.37 |
| SVHN | MNIST | Random noise | TA / AGA | All experts | 97.30 | 91.92 | 99.31 |
| SVHN | MNIST | A only | TA / AGA | None | 97.46 | 92.48 | 99.35 |
| SVHN | MNIST | A only | TA / AGA | Protected expert | 97.46 | 91.18 | 99.37 |
| SVHN | MNIST | A only | TA / AGA | All experts | 97.46 | 91.42 | 99.33 |
| SVHN | MNIST | B+b only | TA / AGA | None | 6.81 | 6.70 | 9.80 |
| SVHN | MNIST | B+b only | TA / AGA | Protected expert | 6.81 | 6.70 | 9.80 |
| SVHN | MNIST | B+b only | TA / AGA | All experts | 6.81 | 6.70 | 9.80 |
| DTD | MNIST | Random noise | TA / AGA | None | 62.39 | 65.53 | 99.51 |
| DTD | MNIST | Random noise | TA / AGA | Protected expert | 62.39 | 64.41 | 99.50 |
| DTD | MNIST | Random noise | TA / AGA | All experts | 62.39 | 64.20 | 99.41 |
| DTD | MNIST | A only | TA / AGA | None | 78.94 | 67.61 | 99.43 |
| DTD | MNIST | A only | TA / AGA | Protected expert | 78.94 | 65.05 | 99.43 |
| DTD | MNIST | A only | TA / AGA | All experts | 78.94 | 65.11 | 99.36 |
| DTD | MNIST | B+b only | TA / AGA | None | 5.00 | 2.13 | 97.25 |
| DTD | MNIST | B+b only | TA / AGA | Protected expert | 5.00 | 2.13 | 97.16 |
| DTD | MNIST | B+b only | TA / AGA | All experts | 5.00 | 2.13 | 95.64 |
| MNIST | SVHN | Unprotected | TIES | None | 99.69 | 99.34 | 94.37 |
| MNIST | SVHN | Unprotected | TIES | Protected expert | 99.69 | 99.32 | 94.81 |
| MNIST | SVHN | Unprotected | TIES | All experts | 99.69 | 99.36 | 94.18 |
| MNIST | SVHN | Unprotected | DARE+TA | None | 99.69 | 99.36 | 92.52 |
| MNIST | SVHN | Unprotected | DARE+TA | Protected expert | 99.69 | 99.30 | 92.78 |
| MNIST | SVHN | Unprotected | DARE+TA | All experts | 99.69 | 99.31 | 91.77 |
| MNIST | SVHN | Defense 2 | TIES | None | 99.61 | 29.30 | 28.00 |
| MNIST | SVHN | Defense 2 | TIES | Protected expert | 99.61 | 29.51 | 23.73 |
| MNIST | SVHN | Defense 2 | TIES | All experts | 99.61 | 28.18 | 20.52 |
| MNIST | SVHN | Defense 2 | DARE+TA | None | 99.61 | 11.52 | 7.44 |
| MNIST | SVHN | Defense 2 | DARE+TA | Protected expert | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 | DARE+TA | All experts | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | TIES | None | 99.69 | 18.18 | 14.64 |
| MNIST | SVHN | Defense 2 w/o SVD | TIES | Protected expert | 99.69 | 35.58 | 25.44 |
| MNIST | SVHN | Defense 2 w/o SVD | TIES | All experts | 99.69 | 35.26 | 27.41 |
| MNIST | SVHN | Defense 2 w/o SVD | DARE+TA | None | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | DARE+TA | Protected expert | 99.69 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | DARE+TA | All experts | 99.69 | 9.80 | 6.70 |
| SVHN | MNIST | Unprotected | TIES | None | 97.46 | 94.37 | 99.34 |
| SVHN | MNIST | Unprotected | TIES | Protected expert | 97.46 | 93.75 | 99.37 |
| SVHN | MNIST | Unprotected | TIES | All experts | 97.46 | 94.18 | 99.36 |
| SVHN | MNIST | Unprotected | DARE+TA | None | 97.46 | 92.51 | 99.37 |
| SVHN | MNIST | Unprotected | DARE+TA | Protected expert | 97.46 | 91.59 | 99.37 |
| SVHN | MNIST | Unprotected | DARE+TA | All experts | 97.46 | 91.77 | 99.32 |
| SVHN | MNIST | Defense 2 | TIES | None | 96.98 | 28.89 | 36.25 |
| SVHN | MNIST | Defense 2 | TIES | Protected expert | 96.98 | 27.43 | 32.27 |
| SVHN | MNIST | Defense 2 | TIES | All experts | 96.98 | 27.64 | 32.06 |
| SVHN | MNIST | Defense 2 | DARE+TA | None | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 | DARE+TA | Protected expert | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 | DARE+TA | All experts | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | TIES | None | 97.45 | 8.16 | 10.33 |
| SVHN | MNIST | Defense 2 w/o SVD | TIES | Protected expert | 97.45 | 14.06 | 11.48 |
| SVHN | MNIST | Defense 2 w/o SVD | TIES | All experts | 97.45 | 14.83 | 12.15 |
| SVHN | MNIST | Defense 2 w/o SVD | DARE+TA | None | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | DARE+TA | Protected expert | 97.45 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | DARE+TA | All experts | 97.45 | 6.70 | 9.80 |
| DTD | MNIST | Unprotected | TIES | None | 79.41 | 74.26 | 99.60 |
| DTD | MNIST | Unprotected | TIES | Protected expert | 79.41 | 73.51 | 99.60 |
| DTD | MNIST | Unprotected | TIES | All experts | 79.41 | 73.51 | 99.61 |
| DTD | MNIST | Unprotected | DARE+TA | None | 79.41 | 69.47 | 99.47 |
| DTD | MNIST | Unprotected | DARE+TA | Protected expert | 79.41 | 68.40 | 99.46 |
| DTD | MNIST | Unprotected | DARE+TA | All experts | 79.41 | 68.94 | 99.43 |
| DTD | MNIST | Defense 2 | TIES | None | 76.12 | 3.30 | 90.68 |
| DTD | MNIST | Defense 2 | TIES | Protected expert | 76.12 | 2.50 | 88.85 |
| DTD | MNIST | Defense 2 | TIES | All experts | 76.12 | 2.50 | 88.27 |
| DTD | MNIST | Defense 2 | DARE+TA | None | 76.12 | 2.29 | 98.68 |
| DTD | MNIST | Defense 2 | DARE+TA | Protected expert | 76.12 | 2.13 | 84.31 |
| DTD | MNIST | Defense 2 | DARE+TA | All experts | 76.12 | 2.13 | 81.05 |
| DTD | MNIST | Defense 2 w/o SVD | TIES | None | 78.09 | 3.40 | 49.55 |
| DTD | MNIST | Defense 2 w/o SVD | TIES | Protected expert | 78.09 | 2.34 | 53.27 |
| DTD | MNIST | Defense 2 w/o SVD | TIES | All experts | 78.09 | 2.39 | 49.49 |
| DTD | MNIST | Defense 2 w/o SVD | DARE+TA | None | 78.09 | 2.13 | 46.73 |
| DTD | MNIST | Defense 2 w/o SVD | DARE+TA | Protected expert | 78.09 | 2.13 | 16.41 |
| DTD | MNIST | Defense 2 w/o SVD | DARE+TA | All experts | 78.09 | 2.13 | 15.73 |
| MNIST | SVHN | Unprotected | MLP matching | None | 99.69 | 99.37 | 92.50 |
| MNIST | SVHN | Defense 2 | MLP matching | None | 99.61 | 9.80 | 6.70 |
| MNIST | SVHN | Defense 2 w/o SVD | MLP matching | None | 99.69 | 9.80 | 6.70 |
| SVHN | MNIST | Unprotected | MLP matching | None | 97.46 | 92.50 | 99.37 |
| SVHN | MNIST | Defense 2 | MLP matching | None | 96.98 | 6.70 | 9.80 |
| SVHN | MNIST | Defense 2 w/o SVD | MLP matching | None | 97.45 | 6.70 | 9.80 |
| DTD | MNIST | Unprotected | MLP matching | None | 79.41 | 69.31 | 99.45 |
| DTD | MNIST | Defense 2 | MLP matching | None | 76.12 | 2.13 | 96.78 |
| DTD | MNIST | Defense 2 w/o SVD | MLP matching | None | 78.09 | 2.13 | 20.66 |
| DTD | MNIST | Defense 2 | 64-noise calibration | None | 76.12 | 59.31 | 99.45 |
| DTD | MNIST | Defense 2 | 64-noise calibration | Protected expert | 76.12 | 52.61 | 99.42 |
| DTD | MNIST | Defense 2 | 64-noise calibration | All experts | 76.12 | 52.34 | 99.38 |

Standalone in public-replacement rows is measured after replacement. Calibration rows reuse fused scores and do not imply a new standalone evaluation. No_target uses the public base in place of the entire target expert at the unchanged coefficient. A-only/B-only are frozen deletions, not separately optimized methods. Current coverage is three datasets, not eight.
