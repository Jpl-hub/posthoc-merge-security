CLIP ViT-B/16 protected-task accuracy (%). The same NBP construction is applied to three experts. Two-expert TA uses coefficient 0.5; AGA processes the protected expert before merging.

| Setting | Defense | MNIST | SVHN | DTD |
| --- | --- | ---: | ---: | ---: |
| Standalone | None | 99.76 | 97.86 | 82.07 |
| Standalone | NBP | 99.76 | 97.86 | 82.07 |
| TA | None | 99.40 | 95.44 | 73.56 |
| TA | NBP | 28.71 | 6.70 | 2.13 |
| AGA + TA | None | 99.35 | 95.03 | 72.39 |
| AGA + TA | NBP | 28.70 | 6.70 | 2.13 |
