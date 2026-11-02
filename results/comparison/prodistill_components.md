# Final MLP components after ProDistill

Complete test accuracy. Other parameters remain fixed at the trained ProDistill model. A and B include their respective biases. Initial values use the same 0.3 initialization; trained values come from one completed joint coefficient fit. These are fixed component substitutions, not separately trained A-only or B-only attacks. Both-trained scores are reused from the same frozen model. No new optimization.

| Target | Dataset | Initial A/B | Trained A | Trained B | Both trained |
| --- | --- | --- | --- | --- | --- |
| MNIST | MNIST | 28.06 | 98.79 | 46.07 | 99.43 |
| MNIST | SVHN | 22.18 | 93.20 | 47.27 | 96.22 |
| SVHN | SVHN | 8.14 | 86.67 | 20.19 | 96.18 |
| SVHN | MNIST | 10.18 | 86.55 | 21.27 | 99.27 |
| DTD | DTD | 8.67 | 41.17 | 31.38 | 66.70 |
| DTD | MNIST | 99.47 | 99.49 | 99.50 | 99.51 |
