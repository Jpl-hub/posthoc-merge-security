Target accuracy after recovery (%). Each method and its control receive the same operation; access is listed per group.

| Defense | MNIST | SVHN | DTD |
| --- | ---: | ---: | ---: |
| **MLP reset: no images; final readout fixed** | | | |
| None | 99.29 | 91.47 | 67.18 |
| NBP | 9.80 | 6.70 | 2.13 |
| **ProDistill: 64 images/task; final readout fixed** | | | |
| None | 99.45 | 96.73 | 73.14 |
| PaRaMS | 89.27 | 66.22 | 7.29 |
| NBP | 9.80 | 6.70 | 2.13 |
| **Bias projection: no images; final bias edited** | | | |
| None | 99.37 | 92.50 | 69.26 |
| NBP | 99.37 | 92.50 | 69.26 |
