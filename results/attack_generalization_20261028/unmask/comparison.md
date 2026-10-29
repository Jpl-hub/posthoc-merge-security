# Unmask

Eight experts, TA coefficient 0.2. 5 complete datasets evaluated; no eight-task average. Same frozen releases; no reconstruction or tuning. Subtract 0.2 times the whole published visual weights from the fused model, including in the unprotected control. No AGA. The protected and matched unprotected models make identical predictions on all five datasets; low accuracy here does not establish an additional defense success.

| Protected task | Defense | Attack | MNIST | SVHN | DTD | GTSRB | RESISC45 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Unprotected | Unmask | 11.35 | 17.97 | 6.06 | 5.13 | 5.46 |
| MNIST | Defense 2 | Unmask | 11.35 | 17.97 | 6.06 | 5.13 | 5.46 |
| SVHN | Unprotected | Unmask | 11.35 | 12.58 | 6.81 | 4.66 | 5.75 |
| SVHN | Defense 2 | Unmask | 11.35 | 12.58 | 6.81 | 4.66 | 5.75 |
| DTD | Unprotected | Unmask | 11.35 | 15.35 | 6.12 | 5.60 | 5.06 |
| DTD | Defense 2 | Unmask | 11.35 | 15.35 | 6.12 | 5.60 | 5.06 |
