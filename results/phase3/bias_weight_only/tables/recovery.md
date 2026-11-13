Target accuracy after recovery (%; lower is better protection). Bold/underline: best/second distinct scores per group; all-equal groups are unmarked. Recovery uses public MLP reset, ProDistill, or bias projection.

| Defense | MNIST | SVHN | DTD |
| --- | ---: | ---: | ---: |
| **MLP reset: no images; final readout fixed** | | | |
| None | <u>99.29</u> | <u>91.47</u> | <u>67.18</u> |
| NBP | **9.80** | **6.70** | **2.13** |
| **ProDistill: 64 images/task; final readout fixed** | | | |
| None | 99.45 | 96.73 | 73.14 |
| PaRaMS | <u>89.27</u> | <u>66.22</u> | <u>7.29</u> |
| NBP | **9.80** | **6.70** | **2.13** |
| **Bias projection + TA: no images; final bias edited** | | | |
| None | 99.37 | 92.50 | 69.26 |
| NBP | 99.37 | 92.50 | 69.26 |
