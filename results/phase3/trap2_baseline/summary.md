Training-based protection on three tasks. CE and Trap$^2$ use the same native CLIP training setup and a frozen visual projection. Scores are full-test target accuracy (%); TA uses 0.5. AGA aligns the target expert. These newly trained experts differ from those in the main table.

| Task | Training | Alone | TA | AGA+TA |
| --- | --- | ---: | ---: | ---: |
| MNIST | CE | 99.35 | 99.55 | 99.50 |
| MNIST | Trap$^2$ | 99.33 | 9.74 | 9.74 |
| SVHN | CE | 96.42 | 96.33 | 95.80 |
| SVHN | Trap$^2$ | 96.60 | 6.70 | 6.70 |
| DTD | CE | 71.60 | 72.02 | 70.90 |
| DTD | Trap$^2$ | 65.37 | 2.13 | 2.13 |
