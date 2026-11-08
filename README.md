# Nullspace Bias Perturbation (NBP)

NBP is a post-hoc modification of frozen CLIP vision experts. It changes only the final LayerNorm bias, using the expert, the public pretrained model and a fixed classifier. Construction uses no real or synthetic images and no optimizer. The original projection cancels the update in exact arithmetic; parameter merging can expose it as a constant feature offset.

The results concern specified merging and recovery procedures. Public projection of the final bias removes the added component; the method does not provide general white-box protection. Finite-precision utility is measured separately.

## Reproduce the result tables and figures

From the repository root, with Python 3.12 and the packages in [requirements.txt](requirements.txt):

```bash
python -m experiment.report --output outputs/nbp_tables --figure
python -m pytest -q
```

This command needs no checkpoints, images or GPU. It reads frozen result counts, checks their input hashes, and exports Markdown, TeX, CSV and vector PDF figures. [Recorded table exports](results/phase3/bias_weight_only/tables/) are included for comparison. Manuscript sources and PDFs are maintained separately.

## Run model experiments

| Operation | Entry |
| --- | --- |
| Construct NBP | [experiment/construct.py](experiment/construct.py), `--method bias` |
| Standalone evaluation | [experiment/native.py](experiment/native.py) |
| Two-expert merging | [experiment/main_table.py](experiment/main_table.py) |
| Eight-expert merging | [experiment/eight_task.py](experiment/eight_task.py) |
| AGA and public-parameter recovery | [experiment/recover.py](experiment/recover.py) |
| Layer-wise coefficient fitting | [experiment/distill.py](experiment/distill.py) |
| Direction controls and weight analysis | [experiment/bias_ablation.py](experiment/bias_ablation.py), [experiment/analyze_bias.py](experiment/analyze_bias.py) |

Use the fixed protocols linked in the [result index](results/comparison/README.md). Model and dataset setup is described in [docs/engineering.md](docs/engineering.md). Checkpoint sizes and hashes are in [configs/checkpoints.json](configs/checkpoints.json); author-code versions are in [configs/sources.json](configs/sources.json). The additional PaRaMS dependencies are in [requirements-params.txt](requirements-params.txt).

`_impl/` contains shared implementations, `experiment/` contains command-line entries, `configs/` pins inputs, and `results/` stores scientific records. Models, images and raw scores stay outside Git. Historical explorations are available in Git history; they are not current method variants. Old record keys such as `Defense1` are preserved to keep their original hashes and are displayed as NBP by the exporter.
