# Frozen O/B under NUFILT and original AGA

O/B is an effect reference with known task0.5 violations. Every accuracy cell is MNIST protected / SVHN partner (%). Two orders are reported separately, not as random seeds.

| Merge order | Recovery scope | Matched unprotected | O/B | Remaining gap (pp) |
|---|---|---:|---:|---:|
| MNIST → SVHN | None | 99.43 / 94.45 | 87.57 / 85.42 | 11.86 / 9.04 |
| MNIST → SVHN | Protected expert | 99.39 / 94.68 | 99.24 / 93.80 | 0.15 / 0.87 |
| MNIST → SVHN | All experts | 99.39 / 93.75 | 99.23 / 92.73 | 0.16 / 1.02 |
| SVHN → MNIST | None | 99.19 / 95.11 | 87.02 / 89.64 | 12.17 / 5.47 |
| SVHN → MNIST | Protected expert | 99.07 / 95.35 | 98.97 / 95.07 | 0.10 / 0.28 |
| SVHN → MNIST | All experts | 99.14 / 94.51 | 98.92 / 93.70 | 0.22 / 0.80 |

Only12 O/B dataset results and six merged models are new. The24 matched unprotected/Phase1 records reuse hash-verified earlier scores; unchanged selected weights establish the Phase1 reuse. This is an official NUFILT kernel with native CLIP adaptation, not the complete FusionBench paper experiment.

All per-class and paired counts passed NumPy recount; 576 selected-score replays in original contexts, maximum error 0. Full raw scores and merged weights remain on3090.

The original aggregate wall timers are invalid because a replay index overwrote the timer. They are retained unchanged. Valid sub-costs: merger construction 259.85s, full evaluation 203.81s,300 merger-gate backwards,0 expert/defense updates. The filesystem campaign interval was 1236.15s including pilot, full evaluation, replays and gaps; it is not isolated GPU timing.
