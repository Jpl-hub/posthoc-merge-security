# Finite A/B compensation: closed test

One MNIST point, same target and SVHN partner. Construction used no real images or partner weights. The point is not adopted: native utility remains, but known B replacement and fixed ProDistill recover. AGA protection also weakens relative to the current main method.

| Setting | MNIST accuracy (%) | SVHN partner accuracy (%) |
| --- | ---: | ---: |
| Standalone | 99.58 | — |
| TA | 26.50 | 17.68 |
| AGA, target | 17.71 | 11.17 |
| AGA, both | 16.96 | 10.10 |
| AGA, target + public B replacement | 99.29 | 92.69 |
| ProDistill | 99.17 | 95.89 |

All32 complete rows, per-class totals and576 original-batch score replays passed. The missing evaluation-directory error was fixed before any evaluation; its original log remains. Original finite-difference failure and the independent forward/reverse verification remain separate records. No scientific input, parameter range or objective changed during the numerical diagnosis.

Raw weights, scores and executed sources remain on21516 under `outputs/compensation_20261106`. Complete compact data: [compensation_result.json](compensation_result.json).
