Eight-expert target accuracy (%; lower is better protection). MergeGuard is our paper-based B/32 implementation; CE uses the same continuation schedule. None uses the original expert. TA coefficient: 0.2.

| Setting | Defense | MNIST | SVHN | DTD | GTSRB |
| --- | --- | ---: | ---: | ---: | ---: |
| TA | None | 94.00 | 74.42 | 52.18 | 65.08 |
| TA | CE | 95.14 | 76.07 | 52.55 | 73.26 |
| TA | MergeGuard | 93.20 | 72.38 | 50.21 | 60.83 |
| AGA + TA | None | 93.23 | 72.53 | 51.38 | 63.29 |
| AGA + TA | CE | 94.67 | 73.97 | 51.97 | 71.03 |
| AGA + TA | MergeGuard | 92.31 | 69.33 | 50.59 | 62.01 |
