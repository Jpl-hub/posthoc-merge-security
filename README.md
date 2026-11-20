# Nullspace Bias Perturbation (NBP)

NBP protects a released CLIP vision expert against model merging by changing its final LayerNorm bias. The expert’s visual projection cancels the update in standalone use. Merging changes the projection and can shift the image embeddings. Construction uses the frozen expert, its pretrained base and its task classifier, without images or training.

Across 42 ordered ViT-B/32 expert pairs, mean target accuracy after AGA alignment and Task Arithmetic drops from 84.81% to 3.30%. Standalone accuracy changes by less than 0.03 percentage points on each of the seven tasks. See the [per-task results](results/phase3/bias_weight_only/tables/partners.md) and the [final-bias recovery control](results/phase3/bias_weight_only/tables/cleanup.md).

## Reproduce the tables

Run from the repository root with Python 3.12. The table exporter uses the standard library and the included evaluation records.

```bash
python -m experiment.report --output outputs/nbp_tables
```

Add `--figure` to generate the experiment plots with NumPy and Matplotlib. The [result index](results/comparison/README.md) links each experiment to its protocol and measurements.

## Protect an expert

Follow the [setup instructions](docs/engineering.md#inputs) for dependencies, checkpoints and classification heads. For example, to protect MNIST:

```bash
python -m experiment.construct --method bias --target MNIST --output outputs/nbp/MNIST
```

The [method implementation](_impl/defenses.py) computes the bias update. The [experiment guide](docs/engineering.md) covers standalone evaluation, model merging and recovery.
