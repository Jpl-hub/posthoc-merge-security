Defense comparison on ViT-B/32 (accuracy, %). Higher is better standalone; lower after merging. Each training method has its own CE control. $^\dagger$: paper-based MergeGuard implementation.

**A. Seven-task evaluation of released experts**

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

**B. Three-task comparison with training-based defenses (two experts)**

| Defense | MNIST / S | MNIST / TA | MNIST / AGA+TA | SVHN / S | SVHN / TA | SVHN / AGA+TA | DTD / S | DTD / TA | DTD / AGA+TA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 99.69 | 99.37 | 99.30 | 97.46 | 92.50 | 91.56 | 79.41 | 69.31 | 68.40 |
| PaRaMS | 99.69 | 8.92 | 99.30 | 97.46 | 9.09 | 91.56 | 79.41 | 2.07 | 68.35 |
| NBP | 99.69 | 9.80 | 9.80 | 97.46 | 6.70 | 6.70 | 79.41 | 2.13 | 2.13 |
| CE (Trap$^2$) | 99.35 | 99.55 | 99.50 | 96.42 | 96.33 | 95.80 | 71.60 | 72.02 | 70.90 |
| Trap$^2$ | 99.33 | 9.74 | 9.74 | 96.60 | 6.70 | 6.70 | 65.37 | 2.13 | 2.13 |
| CE (MergeGuard) | 99.63 | 99.45 | 99.41 | 97.25 | 93.70 | 92.84 | 79.68 | 69.95 | 69.04 |
| MergeGuard$^\dagger$ | 99.60 | 99.22 | 99.24 | 96.80 | 90.63 | 88.54 | 74.20 | 65.80 | 65.90 |
