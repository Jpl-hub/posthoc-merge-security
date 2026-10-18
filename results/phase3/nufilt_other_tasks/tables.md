# Missing NUFILT protected-task cells

Accuracy (%), protected task / partner. Every Phase1 result has the exact same predictions and scores as its matched unprotected merger; selected-input identity was checked before reuse. Official kernels, native adapter; no calibration images. These are merger evaluations, not new recovery attacks.

| Protected task / partner | AGA scope | Order | Unprotected = Defense1 |
|---|---|---|---:|
| DTD / MNIST | None | DTD → MNIST | 72.18 / 99.45 |
| DTD / MNIST | Protected expert | DTD → MNIST | 70.59 / 99.45 |
| DTD / MNIST | All experts | DTD → MNIST | 70.90 / 99.42 |
| DTD / MNIST | None | MNIST → DTD | 72.71 / 99.51 |
| DTD / MNIST | Protected expert | MNIST → DTD | 70.96 / 99.51 |
| DTD / MNIST | All experts | MNIST → DTD | 71.44 / 99.48 |
| SVHN / MNIST | Protected expert | SVHN → MNIST | 94.40 / 99.25 |
| SVHN / MNIST | Protected expert | MNIST → SVHN | 93.47 / 99.44 |

SVHN/MNIST without AGA and with all-expert AGA reuse earlier identical-model results after target/partner relabeling; they were not retrained here. DTD full scores survived interruption: only the unfinished original-context replay was completed; its original total wall time is unavailable. Frozen weights, thresholds and raw scores were unchanged.
