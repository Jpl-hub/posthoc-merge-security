# PaRaMS result sources

All seven tasks use the full author-default PaRaMS transformation: MLP permutation and diagonal attention scaling, seed 0. Two-expert TA uses coefficient 0.5; eight-expert TA uses 0.2. AGA aligns the protected expert only. The companion is SVHN for MNIST and MNIST otherwise. These are evaluations on the same released experts as NBP, not a reproduction of every setting in the PaRaMS paper.

## Full-test counts

Each cell is correct / total. These 35 cells were independently recomputed from the source records and checked against the main comparison on 2026-11-13.

| Protected task | Standalone | 2-expert TA | 2-expert AGA+TA | 8-expert TA | 8-expert AGA+TA |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUN397 | 14944 / 19850 | 60 / 19850 | 14208 / 19850 | 56 / 19850 | 12617 / 19850 |
| Cars | 6245 / 8041 | 58 / 8041 | 5832 / 8041 | 44 / 8041 | 4954 / 8041 |
| RESISC45 | 6055 / 6300 | 162 / 6300 | 5693 / 6300 | 170 / 6300 | 4502 / 6300 |
| SVHN | 25371 / 26032 | 2366 / 26032 | 23836 / 26032 | 2529 / 26032 | 18880 / 26032 |
| GTSRB | 12470 / 12630 | 353 / 12630 | 11647 / 12630 | 224 / 12630 | 7994 / 12630 |
| MNIST | 9969 / 10000 | 892 / 10000 | 9930 / 10000 | 982 / 10000 | 9323 / 10000 |
| DTD | 1493 / 1880 | 39 / 1880 | 1285 / 1880 | 40 / 1880 | 966 / 1880 |

## Evaluation records

| Task | Two-expert counts |
| --- | --- |
| SUN397 | [source](../phase3/bias_weight_only/eight_expert/remaining_pairs_result.json) |
| Cars | [source](../phase3/bias_weight_only/eight_expert/remaining_pairs_result.json) |
| RESISC45 | [source](../phase3/bias_weight_only/eight_expert/remaining_pairs_result.json) |
| SVHN | [source](other_tasks/evaluation/full/01_SVHN_MNIST/results.json) |
| GTSRB | [source](../phase3/bias_weight_only/eight_expert/remaining_pairs_result.json) |
| MNIST | [source](../phase3/results.json) |
| DTD | [source](other_tasks/evaluation/full/00_DTD_MNIST/results.json) |

The four-task file lives under `eight_expert/` for historical reasons, but explicitly records `eight_expert_scope: false`; its records are two-expert tests. Each record includes the original result, raw-score, execution-script and verification paths with SHA-256 hashes.

Eight-expert counts come from [the three-task summary](eight_expert/summary.json) and [the four-task extension](../phase3/bias_weight_only/eight_expert/extension/results.json). The extension includes [execution hashes](../phase3/bias_weight_only/eight_expert/extension/execution.json), [independent statistics](../phase3/bias_weight_only/eight_expert/extension/independent_statistics.json) and [score replay verification](../phase3/bias_weight_only/eight_expert/extension/validation.json).

## Checkpoints and implementations

The protocols pin the complete SHA-256 hashes and paths for protected releases and AGA-transformed weights: [MNIST/SVHN/DTD](eight_expert/evaluation.json), [SUN397/Cars/RESISC45/GTSRB](../phase3/bias_weight_only/eight_expert/extension_evaluation.json). Short hashes below identify the same entries.

| Task | PaRaMS release SHA-256 prefix | AGA weights SHA-256 prefix |
| --- | --- | --- |
| SUN397 | `1131f4188b2ba821` | `3ea87c212bb92e32` |
| Cars | `37fb19d0e1498d71` | `1fbfef40a5cd7e4d` |
| RESISC45 | `7011a52b98d4b59f` | `e88047d968a140d2` |
| SVHN | `e8ef5b1784c8c77c` | `a5907b9cba77d06c` |
| GTSRB | `2a6c6f94d3d8d734` | `8601c472ff3a50ef` |
| MNIST | `5a7bca84cc9a4052` | `a441587682b6ef57` |
| DTD | `27da7f2fcb2b0102` | `fdba7aa377a6bb52` |

PaRaMS author commit: `4159a9a8bf253088694154d944f4583db8eca60e`. Source: `external/PaRaMS/src/PaRaMS/PaRaMS_algorithm.py`, SHA-256 `40c114a121907d5633d29a49984efa9be2ac03e29e649e801fed2dccad89fb60`.

AGA author commit: `fb61429f5e6f3517102d506697a150ac26abbdea`; author module and normalized source hash are recorded under `AGA` in both protocols. The test denominators are SUN397 19,850; Cars 8,041; RESISC45 6,300; SVHN 26,032; GTSRB 12,630; MNIST 10,000; DTD test1 1,880. No EuroSAT accuracy enters the average.

Rebuild tables with `python -m experiment.report --output results/phase3/bias_weight_only/tables`. The exporter checks frozen input hashes. Heavy model files and raw score archives remain on the experiment servers; the source records specify their paths and hashes.
