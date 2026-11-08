Target accuracy after recovery (%). Group headings state image access and final-parameter edits. MLP replacement follows AGA; bias projection precedes TA. Controls receive the same operation. Bias projection gives identical NBP/control predictions.

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
