Complete-test accuracy after labeled coefficient selection (%). Each method separately maximizes the mean target/companion accuracy on the same 256 training examples per task, using TA coefficients 0.1--1.0 in steps of 0.1. Coefficients are frozen before test evaluation. AGA processes the protected expert first; both task updates use the selected coefficient.

| Task | Defense | Coeff. | Target | Companion |
| --- | --- | ---: | ---: | ---: |
| MNIST | None | 0.6 | 99.36 | 93.24 |
| MNIST | PaRaMS | 0.6 | 99.36 | 93.24 |
| MNIST | NBP | 0.7 | 11.35 | 19.59 |
| SVHN | None | 0.6 | 91.98 | 99.42 |
| SVHN | PaRaMS | 0.6 | 91.98 | 99.42 |
| SVHN | NBP | 0.7 | 68.21 | 87.44 |
| DTD | None | 0.7 | 71.12 | 99.69 |
| DTD | PaRaMS | 0.7 | 71.12 | 99.69 |
| DTD | NBP | 0.2 | 2.13 | 29.49 |
