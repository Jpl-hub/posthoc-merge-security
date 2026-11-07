Standalone and protected-task accuracy (%) with matched checkpoints and processing. Bold scores mark the lowest merged accuracy per setting. TA coefficients are 0.5 for two experts and 0.2 for eight experts. Avg. is the unweighted mean of the seven displayed task scores. Companions are SVHN for MNIST and MNIST otherwise. EuroSAT participates in eight-expert merging.

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
