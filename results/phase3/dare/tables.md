# Fixed DARE+TA and original AGA

MNIST protected / SVHN partner. Drop0.2, TA0.5; all seeds42/43/44 retained. Cells are mean ± sample SD (%) across three masks, not a confidence interval.

| Defense | No attack: target / partner | AGA (protected expert): target / partner | AGA (all experts): target / partner |
|---|---:|---:|---:|
| Unprotected | 99.37 ± 0.01 / 92.51 ± 0.01 | 99.30 ± 0.00 / 92.81 ± 0.04 | 99.32 ± 0.01 / 91.78 ± 0.02 |
| Defense 1 | 25.32 ± 4.88 / 25.92 ± 9.56 | 24.20 ± 5.60 / 25.50 ± 10.30 | 23.87 ± 5.90 / 24.90 ± 10.78 |
| O/B† | 78.76 ± 0.44 / 63.19 ± 0.52 | 79.52 ± 0.57 / 66.01 ± 0.71 | 77.79 ± 0.72 / 61.61 ± 0.67 |
| A/B† | 91.83 ± 0.61 / 77.73 ± 0.56 | 86.11 ± 0.95 / 75.87 ± 0.56 | 80.22 ± 1.58 / 70.30 ± 0.71 |
| PaRaMS (full)* | 9.45 ± 0.22 / 7.59 ± 0.44 | 99.30 ± 0.00 / 92.81 ± 0.04 | 99.32 ± 0.01 / 91.78 ± 0.02 |
| PaRaMS (attention) | 99.36 ± 0.00 / 92.49 ± 0.03 | 99.30 ± 0.00 / 92.81 ± 0.04 | 99.32 ± 0.01 / 91.78 ± 0.02 |

† O/B and A/B violate the task0.5 budget; effect references only. * Full official PaRaMS is over-budget. Defense1 has different constraints. Frozen native accuracies and per-class/paired counts remain in each seed record.

Each seed/scope has its own unprotected control. Images repeat across seeds: per-seed paired tests are not pooled. DARE is a merger component, not an additional recovery attack. Official mask function is unchanged; native CLIP adaptation and FP32 reconstruction arithmetic are explicit. This does not reproduce the original LM/GLUE hyperparameter search.

Full evaluation wall time: 2242.62s; independent selected-score replays: 2016, maximum error 0. Replay wall time is additional and was not separately timed.
