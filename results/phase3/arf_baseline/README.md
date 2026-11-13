# ARF: matched training comparison

The configuration documents the paper equation/prose discrepancy and our training and sampling choices. These results evaluate this adaptation; they do not establish reproduction of the authors’ ARF results. The companion is the original SVHN expert.

| Method | Standalone ↑ | TA ↓ | AGA + TA (target aligned) ↓ | AGA + TA (both aligned) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Matched CE | 99.62 | 99.27 | 99.22 | 99.25 |
| ARF | 99.55 | 9.64 | 99.25 | 99.24 |

Both runs start from the public base and use the same five-epoch schedule and training split. The ARF run adds the declared distance penalty and the existing invertible attention transformation. All 14 target/companion rows are retained in `summary.json`; 224 score replays match exactly.

Reproduce from the repository root:

```bash
python -m experiment.train_baseline --config configs/arf_baseline.json --task MNIST --defense CE
python -m experiment.train_baseline --config configs/arf_baseline.json --task MNIST --defense ARF
python -m experiment.train_baseline --config configs/arf_baseline.json --task MNIST --evaluate
```

DTD was added on 13 November using the same constructor, loss and optimizer settings, with 76 epochs and the official partition-1 train/validation split. The epoch count follows the Task Arithmetic training schedule, not unpublished ARF settings. Its matched CE run starts from the same public base; the earlier MergeGuard CE model is not reused because it starts from a fine-tuned expert. The fixed companion is MNIST. Run the same three commands with `--task DTD`; the existing MNIST results are not rerun or replaced.

The fixed DTD run is now complete. On 1,880 test images, the matched CE control gives 73.46% standalone and 57.77% after target-only AGA+TA (70.90% for ARF). ARF gives 75.32% standalone, 2.34% under direct TA, 70.90% with target-only AGA+TA, and 71.01% when both experts are aligned. The fixed MNIST companion rows are retained in the raw evaluation record. All 48 attention-matrix margins are satisfied (distance/anchor-norm ratio 0.05385--0.05744; required minimum 0.05); the complete replay has 224 score checks with zero error. These are paper-prose adaptations, not official ARF results, and are not combined with the seven-task released-expert ranking.

The original MNIST configuration hash and execution records in `summary.json` remain unchanged. The current configuration also includes DTD; each new run records the configuration it actually used.
