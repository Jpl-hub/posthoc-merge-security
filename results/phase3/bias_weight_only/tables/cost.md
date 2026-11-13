CPU construction time in seconds: median [min, max] over five fresh processes, each followed by one warm call. Both methods use the same four-CPU affinity. Loading and verification are excluded; cold calls include author import/JIT.

| Call | Task | NBP | PaRaMS |
| --- | --- | ---: | ---: |
| Cold | MNIST | 0.58 [0.55, 0.58] | 160.49 [158.63, 167.47] |
| Cold | SVHN | 0.62 [0.61, 0.66] | 158.72 [157.69, 161.97] |
| Cold | DTD | 0.62 [0.59, 0.66] | 159.85 [158.73, 163.04] |
| Warm | MNIST | 0.51 [0.48, 0.54] | 159.83 [156.07, 164.15] |
| Warm | SVHN | 0.52 [0.51, 0.54] | 159.20 [156.96, 160.25] |
| Warm | DTD | 0.51 [0.49, 0.55] | 158.51 [157.53, 160.94] |
