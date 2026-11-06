Standalone and protected-task accuracy (%) for seven tasks. Standalone accuracy should remain high; lower merged accuracy indicates stronger protection. TA uses coefficient 0.5 for two experts and 0.2 for eight experts. AGA processes the protected expert before TA, including in each matching control. Two-expert companions are SVHN for MNIST and MNIST otherwise. Eight-expert merging protects one expert at a time; EuroSAT participates without a reported test score or eight-task average.

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | None | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| Standalone | PaRaMS | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| Standalone | NBP | 75.27 | 77.64 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| 2-expert TA | None | 71.92 | 73.17 | 91.08 | 92.50 | 93.25 | 99.37 | 69.31 |
| 2-expert TA | PaRaMS | 0.30 | 0.72 | 2.57 | 9.09 | 2.79 | 8.92 | 2.07 |
| 2-expert TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
| 2-expert AGA | None | 71.58 | 72.53 | 90.38 | 91.56 | 92.22 | 99.30 | 68.40 |
| 2-expert AGA | PaRaMS | 71.58 | 72.53 | 90.37 | 91.56 | 92.22 | 99.30 | 68.35 |
| 2-expert AGA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
| 8-expert TA | None | 63.76 | 62.06 | 72.00 | 74.42 | 65.08 | 94.00 | 52.18 |
| 8-expert TA | PaRaMS | 0.28 | 0.55 | 2.70 | 9.71 | 1.77 | 9.82 | 2.13 |
| 8-expert TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
| 8-expert AGA | None | 63.56 | 61.61 | 71.46 | 72.53 | 63.29 | 93.23 | 51.38 |
| 8-expert AGA | PaRaMS | 63.56 | 61.61 | 71.46 | 72.53 | 63.29 | 93.23 | 51.38 |
| 8-expert AGA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
