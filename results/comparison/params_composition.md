# Fixed PaRaMS compositions on MNIST

MNIST is protected; SVHN is the partner. TA coefficient 0.5. AGA-T processes the target; AGA-B processes both experts. ProDistill uses 64 additional unlabeled images per task and 100 epochs per block, with initial coefficient 0.3. Official default PaRaMS is applied to each frozen defense; construction uses no images.

| Defense | Processing | MNIST | SVHN |
| --- | --- | --- | --- |
| Unprotected | Standalone | 99.69 | -- |
| Unprotected | TA | 99.37 | 92.50 |
| Unprotected | AGA-T + TA | 99.30 | 92.78 |
| Unprotected | AGA-B + TA | 99.32 | 91.77 |
| Unprotected | ProDistill | 99.45 | 96.73 |
| Defense 1 + PaRaMS | Standalone | 99.69 | -- |
| Defense 1 + PaRaMS | TA | 9.80 | 6.70 |
| Defense 1 + PaRaMS | AGA-T + TA | 9.80 | 6.70 |
| Defense 1 + PaRaMS | AGA-B + TA | 9.80 | 6.70 |
| Defense 1 + PaRaMS | ProDistill | 9.80 | 6.70 |
| Defense 2 + PaRaMS | Standalone | 99.57 | -- |
| Defense 2 + PaRaMS | TA | 9.74 | 6.38 |
| Defense 2 + PaRaMS | AGA-T + TA | 2.99 | 2.03 |
| Defense 2 + PaRaMS | AGA-B + TA | 1.33 | 1.50 |
| Defense 2 + PaRaMS | ProDistill | 89.74 | 59.37 |
