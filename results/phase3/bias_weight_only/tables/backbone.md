Transfer to ViT-B/16: standalone and target accuracy (%). TA uses 0.5; AGA aligns the protected expert.

| Setting | Defense | MNIST | SVHN | DTD |
| --- | --- | ---: | ---: | ---: |
| Standalone | None | 99.76 | 97.86 | 82.07 |
| Standalone | NBP | 99.76 | 97.86 | 82.07 |
| TA | None | <u>99.40</u> | <u>95.44</u> | <u>73.56</u> |
| TA | NBP | **28.71** | **6.70** | **2.13** |
| AGA + TA | None | <u>99.35</u> | <u>95.03</u> | <u>72.39</u> |
| AGA + TA | NBP | **28.70** | **6.70** | **2.13** |
