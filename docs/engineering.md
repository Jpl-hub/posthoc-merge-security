# Setup and reproduction

Run all commands from the repository root. The tested GPU environment uses Python3.12, PyTorch2.3.0+cu121 and NumPy1.26.4. Install `requirements.txt` and, for constructing the PaRaMS reference, `requirements-params.txt`.

## Inputs

Place the pinned author implementation at `external/secure-merge-attack`, the CLIP checkpoints and classification heads at `checkpoints/ViT-B-32` or `checkpoints/ViT-B-16`, and datasets at `data/`. Exact author revisions, checkpoint identifiers, sizes and SHA256 values are in `configs/sources.json` and the two checkpoint manifests. `POSTHOC_BACKBONE=ViT-B-16` selects the second backbone; B/32 is the default.

The checkpoints come from the Task Arithmetic release. Their image encoders use OpenAI-pretrained CLIP through OpenCLIP. The fixed heads contain normalized, template-averaged class-name text embeddings scaled by the pretrained logit scale; they are not fitted linear probes. The same head is used for all methods on a task. NBP does not train either the encoder or the head.

`_impl/data.py` defines the evaluated splits and independently reads their labels. Inference uses each checkpoint's stored validation preprocessing. DTD uses partition1. EuroSAT participates in eight-expert merging, but its original expert train/test membership could not be recovered; the reported task average therefore covers seven validated test sets.

## Commands

```bash
python -m experiment.construct --help
python -m experiment.main_table --help
python -m experiment.native --help
python -m experiment.eight_task --help
python -m experiment.recover --help
python -m experiment.distill --help
python -m experiment.report --output outputs/nbp_tables --figure
python -m pytest -q
```

The construction entry's `--method bias` calls `_impl.defenses.construct_bias`. It reads frozen weights and a fixed classifier, computes a projection kernel, selects the class direction and obtains the update norm from a LayerNorm bound. It uses zero construction images and zero optimizer steps.

Each model batch specifies its own source and weight hashes, split, task order, coefficient and recovery scope. Use the recorded execution version to reproduce a historical batch; do not replace its fingerprints with those of newer code. The evaluator's `--verify` recounts complete labels and per-class/paired outcomes, then replays selected scores in their original128-image batch context. A pilot is an implementation check, not a replacement for a complete test set.

`experiment.report` only reads frozen records. Generated tables and figures go to the requested output directory, separate from input records. The default output is ignored by Git.

## Historical records

Stopped numerical-compensation experiments and their report code are available at commit `f1d7e496` and earlier. The active repository exposes the NBP method, matched baselines and their evidence. Unique model and raw-score files are not deleted by this source cleanup. Removed internal metadata on the prior compute host was preserved in its system-disk archive before updating that host.

```bash
git show <execution-commit>:<source-path>
git archive <execution-commit> | tar -x -C <empty-directory>
```
