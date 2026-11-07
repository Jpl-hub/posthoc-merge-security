Direction controls on MNIST (%). The three perturbations have the same update norm. The random nullspace direction is projected into the original visual-projection kernel; the unconstrained random direction uses the same sampled vector before projection. All controls use the same MNIST/SVHN pair and TA coefficient 0.5.

| Update | Standalone | TA | AGA + TA |
| --- | ---: | ---: | ---: |
| None | 99.69 | 99.37 | 99.30 |
| NBP | 99.69 | 9.80 | 9.80 |
| Random nullspace | 99.69 | 65.21 | 63.74 |
| Random direction | 9.74 | 9.74 | 9.74 |
