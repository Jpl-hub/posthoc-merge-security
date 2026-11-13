Full-test AGA+TA accuracy (%) with coefficient 0.5 and FP32 readout (R32). T and C denote protected and companion tasks. MNIST uses SVHN as companion; the other tasks use MNIST.

| Task | Precision | None (T/C) | NBP (T/C) |
| --- | --- | ---: | ---: |
| Cars | FP32 | 72.53 / 99.41 | 0.55 / 10.10 |
| Cars | FP16 + R32 | 72.52 / 99.41 | 0.55 / 10.10 |
| Cars | BF16 + R32 | 72.52 / 99.40 | 0.55 / 10.10 |
| SVHN | FP32 | 91.56 / 99.39 | 6.70 / 9.80 |
| SVHN | FP16 + R32 | 91.58 / 99.39 | 6.70 / 9.80 |
| SVHN | BF16 + R32 | 91.59 / 99.38 | 6.70 / 9.80 |
| MNIST | FP32 | 99.30 / 92.78 | 9.80 / 6.70 |
| MNIST | FP16 + R32 | 99.30 / 92.78 | 9.80 / 6.70 |
| MNIST | BF16 + R32 | 99.30 / 92.77 | 9.80 / 6.70 |
