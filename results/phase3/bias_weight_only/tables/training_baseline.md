Training-based comparisons with a separate matched CE control for each method. Scores are full-test target accuracy (%); TA uses 0.5 and AGA aligns the target. Trap$^2$ uses author training code; MergeGuard is our paper-based B/32 reproduction. Training details and differences from the main table are stated in the text.

| Task | Training | Alone | TA | AGA+TA |
| --- | --- | ---: | ---: | ---: |
| **Author Trap$^2$ training** | | | | |
| MNIST | CE | 99.35 | 99.55 | 99.50 |
| MNIST | Trap$^2$ | 99.33 | 9.74 | 9.74 |
| SVHN | CE | 96.42 | 96.33 | 95.80 |
| SVHN | Trap$^2$ | 96.60 | 6.70 | 6.70 |
| DTD | CE | 71.60 | 72.02 | 70.90 |
| DTD | Trap$^2$ | 65.37 | 2.13 | 2.13 |
| **Paper-based MergeGuard reproduction** | | | | |
| MNIST | CE | 99.63 | 99.45 | 99.41 |
| MNIST | MergeGuard | 99.60 | 99.22 | 99.24 |
| SVHN | CE | 97.25 | 93.70 | 92.84 |
| SVHN | MergeGuard | 96.80 | 90.63 | 88.54 |
| DTD | CE | 79.68 | 69.95 | 69.04 |
| DTD | MergeGuard | 74.20 | 65.80 | 65.90 |
