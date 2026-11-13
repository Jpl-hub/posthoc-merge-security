MergeLock comparison (%)

Paper implementation; all completed tasks and outcomes are retained.

| Setting | Method | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | None | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| Standalone | MergeLock | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| 2-expert TA | None | 71.92 | 73.17 | 91.08 | 92.50 | 93.25 | 99.37 | 69.31 |
| 2-expert TA | MergeLock | 0.22 | 0.49 | 3.10 | 9.51 | 3.03 | 8.70 | 2.66 |
| 2-expert AGA + TA | None | 71.58 | 72.53 | 90.38 | 91.56 | 92.22 | 99.30 | 68.40 |
| 2-expert AGA + TA | MergeLock | 71.58 | 72.48 | 90.38 | 91.56 | 92.22 | 99.30 | 68.40 |
| 8-expert TA | None | 63.76 | 62.06 | 72.00 | 74.42 | 65.08 | 94.00 | 52.18 |
| 8-expert TA | MergeLock | 0.22 | 0.49 | 3.03 | 9.95 | 2.74 | 9.24 | 2.61 |
| 8-expert AGA + TA | None | 63.56 | 61.61 | 71.46 | 72.53 | 63.29 | 93.23 | 51.38 |
| 8-expert AGA + TA | MergeLock | 63.57 | 61.62 | 71.46 | 72.52 | 63.29 | 93.23 | 51.38 |
