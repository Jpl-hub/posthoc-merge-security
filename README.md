# Nullspace Bias Perturbation (NBP)

NBP is a post-hoc modification of frozen CLIP vision experts. It changes only the final LayerNorm bias, using the expert, the public pretrained model and a fixed classifier. Construction uses no real or synthetic images and no optimizer. The original projection cancels the update in exact arithmetic; parameter merging can expose it as a constant feature offset.

The results concern specified merging and recovery procedures. Public projection of the final bias removes the added component; the method does not provide general white-box protection. Finite-precision utility is measured separately.

## Reproduce the result tables and figures

From the repository root, with Python 3.12 and the packages in [requirements.txt](requirements.txt):

```bash
python -m experiment.report --output outputs/nbp_tables --figure
python -m pytest -q
```

This command needs no checkpoints, images or GPU. It reads frozen result counts, checks their input hashes, and exports Markdown, TeX, CSV and vector PDF figures. Coefficient curves show all recorded calibration points (256 training images per task); selected full-test results are reported separately. [Recorded table exports](results/phase3/bias_weight_only/tables/) are included for comparison. Manuscript sources and PDFs are maintained separately.

## Run model experiments

| Operation | Entry |
| --- | --- |
| Construct NBP | [experiment/construct.py](experiment/construct.py), `--method bias` |
| Construct the paper-based MergeLock comparator | [experiment/construct.py](experiment/construct.py), `--method mergelock --protocol configs/mergelock_baseline.json` |
| Standalone evaluation | [experiment/native.py](experiment/native.py) |
| Two-expert merging | [experiment/main_table.py](experiment/main_table.py) |
| Eight-expert merging | [experiment/eight_task.py](experiment/eight_task.py) |
| AGA and public-parameter recovery | [experiment/recover.py](experiment/recover.py) |
| Layer-wise coefficient fitting | [experiment/distill.py](experiment/distill.py) |
| Direction controls and weight analysis | [experiment/bias_ablation.py](experiment/bias_ablation.py), [experiment/analyze_bias.py](experiment/analyze_bias.py) |
| Trap², MergeGuard and ARF training comparisons | [experiment/train_baseline.py](experiment/train_baseline.py) |

Use the fixed protocols linked in the [result index](results/comparison/README.md). Model and dataset setup is described in [docs/engineering.md](docs/engineering.md). Checkpoint sizes and hashes are in [configs/checkpoints.json](configs/checkpoints.json); author-code versions are in [configs/sources.json](configs/sources.json). The additional PaRaMS dependencies are in [requirements-params.txt](requirements-params.txt).

The class-choice stability check uses five fixed seeds on seven tasks. Construct all controls before evaluating them:

```bash
python -m experiment.bias_ablation --protocol configs/nbp_stability.json
python -m experiment.bias_ablation --protocol configs/nbp_stability.json --evaluate
```

These controls use the same NBP rule and do not replace the main releases. Report all five draws, including any repeated class indices; they are not independent expert-training runs. The protocol reuses archived unprotected scores after checking file and model hashes. For a fresh run without those archives, set `legacy` to `{}` to evaluate the identical unprotected models.

Precision and construction cost use the existing evaluation and construction entries:

```bash
python -m experiment.native --precision-protocol configs/nbp_precision.json
python -m experiment.native --precision-protocol configs/nbp_precision_merge.json
python -m experiment.construct --benchmark --method bias --target MNIST --protocol configs/nbp_cost.json --output outputs/cost/MNIST_bias
```

Precision tests keep parameter storage in FP32 and vary actual AMP operations; the separate FP32-readout control covers final LayerNorm, projection and classifier. The merging test uses fixed author AGA references and reports both target and companion accuracy. Cost tests use five fresh processes, each with a cold and warm construction, under the same four-CPU affinity. Run both `bias` and `params` for each task listed in the cost protocol, without concurrent computation.

The MergeLock comparator implements Section 3.4 of [Model Unmerging](https://arxiv.org/abs/2509.01548). Its author repository contained only a README when checked on November 13, 2026. The configuration records our choices for the unspecified random-matrix distribution, scaling range and seed. It is a paper implementation, not an official-code reproduction; standalone accuracy must be reported alongside merging results.

`_impl/` contains shared implementations, `experiment/` contains command-line entries, `configs/` pins inputs, and `results/` stores scientific records. Models, images and raw scores stay outside Git. Historical explorations are available in Git history; they are not current method variants. Old record keys such as `Defense1` are preserved to keep their original hashes and are displayed as NBP by the exporter.
