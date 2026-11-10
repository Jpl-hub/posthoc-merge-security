Target accuracy (%): higher standalone (S), lower after merging. A--B: our B/32 tests; Trap²/CE use separately trained experts. TA coefficients are 0.5 (two experts) and 0.2 (eight). Avg. weights tasks equally. C: published L/14 results (MergeGuard), Table 3, with the original protocol.

**A. Seven-task evaluation of released experts (ViT-B/32)**

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | None | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| Standalone | PaRaMS | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| Standalone | NBP | 75.27 | 77.64 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| 2-expert TA | None | 71.92 | 73.17 | 91.08 | 92.50 | 93.25 | 99.37 | 69.31 | 84.37 |
| 2-expert TA | PaRaMS | 0.30 | 0.72 | 2.57 | 9.09 | 2.79 | 8.92 | 2.07 | 3.78 |
| 2-expert TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 | 3.29 |
| 2-expert AGA + TA | None | 71.58 | 72.53 | 90.38 | 91.56 | 92.22 | 99.30 | 68.40 | 83.71 |
| 2-expert AGA + TA | PaRaMS | 71.58 | 72.53 | 90.37 | 91.56 | 92.22 | 99.30 | 68.35 | 83.70 |
| 2-expert AGA + TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 | 3.29 |
| 8-expert TA | None | 63.76 | 62.06 | 72.00 | 74.42 | 65.08 | 94.00 | 52.18 | 69.07 |
| 8-expert TA | PaRaMS | 0.28 | 0.55 | 2.70 | 9.71 | 1.77 | 9.82 | 2.13 | 3.85 |
| 8-expert TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 | 3.29 |
| 8-expert AGA + TA | None | 63.56 | 61.61 | 71.46 | 72.53 | 63.29 | 93.23 | 51.38 | 68.15 |
| 8-expert AGA + TA | PaRaMS | 63.56 | 61.61 | 71.46 | 72.53 | 63.29 | 93.23 | 51.38 | 68.15 |
| 8-expert AGA + TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 | 3.29 |

**B. Three-task comparison with Trap² (ViT-B/32, two experts)**

| Defense | MNIST / S | MNIST / TA | MNIST / AGA+TA | SVHN / S | SVHN / TA | SVHN / AGA+TA | DTD / S | DTD / TA | DTD / AGA+TA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 99.69 | 99.37 | 99.30 | 97.46 | 92.50 | 91.56 | 79.41 | 69.31 | 68.40 |
| PaRaMS | 99.69 | 8.92 | 99.30 | 97.46 | 9.09 | 91.56 | 79.41 | 2.07 | 68.35 |
| NBP | 99.69 | 9.80 | 9.80 | 97.46 | 6.70 | 6.70 | 79.41 | 2.13 | 2.13 |
| CE (Trap²) | 99.35 | 99.55 | 99.50 | 96.42 | 96.33 | 95.80 | 71.60 | 72.02 | 70.90 |
| Trap² | 99.33 | 9.74 | 9.74 | 96.60 | 6.70 | 6.70 | 65.37 | 2.13 | 2.13 |

**C. Published comparison from MergeGuard (ViT-L/14)**

Source: [MergeGuard, Table 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.pdf). Original protocol; seven common tasks shown.

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | PaRaMS | 82.32 | 92.39 | 97.37 | 98.11 | 99.24 | 99.69 | 84.15 |
| Standalone | MergeGuard | 81.52 | 88.29 | 97.25 | 96.82 | 98.25 | 99.27 | 82.16 |
| TA | PaRaMS | 56.77 | 38.61 | 67.86 | 71.95 | 52.25 | 96.02 | 48.72 |
| TA | MergeGuard | 54.27 | 38.24 | 56.50 | 47.28 | 12.91 | 11.35 | 46.65 |
