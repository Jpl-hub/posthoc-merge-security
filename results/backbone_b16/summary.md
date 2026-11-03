# Defense 1 on ViT-B/16

ViT-B/16, fixed TA coefficient 0.5, three protected tasks. AGA-T processes the protected expert; AGA-B processes both experts. Cal. uses 64 fixed random images and no real training data. Defense construction uses no images. The same author AGA functions are adapted from the B/32 entry. All standalone predictions are preserved, but output calibration restores near matched-control accuracy. No Defense 2 construction in this batch.

| Task | Defense | Standalone | TA | AGA-T | AGA-B | TA+cal. | AGA-T+cal. | AGA-B+cal. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | 99.76 | 99.40 | 99.35 | 99.40 | 99.38 | 99.33 | 99.38 |
| MNIST | Defense 1 | 99.76 | 28.71 | 28.70 | 28.71 | 99.38 | 99.33 | 99.37 |
| SVHN | Unprotected | 97.86 | 95.44 | 95.03 | 95.26 | 95.19 | 94.81 | 95.08 |
| SVHN | Defense 1 | 97.86 | 6.70 | 6.70 | 6.70 | 95.18 | 94.73 | 94.98 |
| DTD | Unprotected | 82.07 | 73.56 | 72.39 | 72.61 | 63.14 | 61.38 | 61.49 |
| DTD | Defense 1 | 82.07 | 2.13 | 2.13 | 2.13 | 63.19 | 62.18 | 62.23 |
