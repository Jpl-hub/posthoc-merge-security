Seven protected tasks in two-expert merging. Standalone accuracy should remain high; lower merged protected-task accuracy indicates stronger protection. TA uses coefficient 0.5. AGA processes the protected expert, including in each matching unprotected control. MNIST pairs with SVHN; other targets pair with MNIST. All values are complete-test accuracy (\%).

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standalone | None | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| Standalone | PaRaMS | 75.28 | 77.66 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| Standalone | NBP | 75.27 | 77.64 | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 |
| TA | None | 71.92 | 73.17 | 91.08 | 92.50 | 93.25 | 99.37 | 69.31 |
| TA | PaRaMS | 0.30 | 0.72 | 2.57 | 9.09 | 2.79 | 8.92 | 2.07 |
| TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
| AGA + TA | None | 71.58 | 72.53 | 90.38 | 91.56 | 92.22 | 99.30 | 68.40 |
| AGA + TA | PaRaMS | 71.58 | 72.53 | 90.37 | 91.56 | 92.22 | 99.30 | 68.35 |
| AGA + TA | NBP | 0.25 | 0.55 | 2.44 | 6.70 | 1.19 | 9.80 | 2.13 |
