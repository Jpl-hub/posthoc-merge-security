Complete companion-task accuracy (%) for TIES and DARE+TA. Column names identify the protected expert. Its companion is SVHN for MNIST and MNIST otherwise. Same frozen models and settings as the protected-task table.

| Procedure | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TIES | None | 99.60 | 99.65 | 99.64 | 99.34 | 99.49 | 94.37 | 99.60 |
| TIES | PaRaMS | 8.92 | 8.90 | 8.98 | 8.95 | 8.92 | 9.05 | 8.93 |
| TIES | NBP | 10.28 | 9.82 | 10.28 | 19.35 | 8.92 | 9.69 | 9.58 |
| AGA + TIES | None | 99.60 | 99.64 | 99.64 | 99.37 | 99.50 | 94.81 | 99.60 |
| AGA + TIES | PaRaMS | 99.60 | 99.64 | 99.64 | 99.37 | 99.50 | 94.81 | 99.60 |
| AGA + TIES | NBP | 10.28 | 9.82 | 10.28 | 18.12 | 8.92 | 9.69 | 9.58 |
| DARE+TA | None | 99.45 | 99.42 | 99.44 | 99.37 | 99.12 | 92.52 | 99.47 |
| DARE+TA | PaRaMS | 9.57 | 9.58 | 9.58 | 9.70 | 9.53 | 7.15 | 9.57 |
| DARE+TA | NBP | 10.28 | 9.82 | 8.92 | 9.80 | 9.82 | 23.00 | 9.82 |
| AGA + DARE+TA | None | 99.45 | 99.41 | 99.41 | 99.37 | 99.15 | 92.78 | 99.46 |
| AGA + DARE+TA | PaRaMS | 99.45 | 99.41 | 99.41 | 99.37 | 99.15 | 92.78 | 99.46 |
| AGA + DARE+TA | NBP | 10.28 | 9.82 | 8.92 | 9.80 | 9.82 | 22.89 | 9.82 |
