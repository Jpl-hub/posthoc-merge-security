# Known-module recovery of the same compositions

Extra knowledge: the modified location. Defense 1 uses final-bias projection; Defense 2 replaces the last MLP output weight and bias with public base values. Each control undergoes the same operation. AGA followed by module recovery is a composed procedure, separate from AGA alone. TA coefficient 0.5; no real attack data.

| Defense | Processing | Standalone | MNIST | SVHN |
| --- | --- | --- | --- | --- |
| D1 control | Known module | 99.69 | 99.37 | 92.50 |
| Defense 1 + PaRaMS | Known module | 99.69 | 8.92 | 9.16 |
| D2 control | Known module | 99.68 | 99.35 | 92.45 |
| Defense 2 + PaRaMS | Known module | 99.65 | 8.92 | 9.15 |
| D1 control | AGA + known module | 99.68 | 99.30 | 92.79 |
| Defense 1 + PaRaMS | AGA + known module | 99.68 | 99.30 | 92.79 |
| D2 control | AGA + known module | 99.64 | 99.29 | 92.79 |
| Defense 2 + PaRaMS | AGA + known module | 99.65 | 99.28 | 92.64 |
