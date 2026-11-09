# Setup and reproduction

Run all commands from the repository root. The tested GPU environment uses Python 3.12, PyTorch 2.3.0+cu121 and NumPy 1.26.4. Install `requirements.txt` and, for constructing the PaRaMS reference, `requirements-params.txt`.

## Inputs

Place the pinned author implementation at `external/secure-merge-attack`, the CLIP checkpoints and classification heads at `checkpoints/ViT-B-32` or `checkpoints/ViT-B-16`, and datasets at `data/`. Exact author revisions, checkpoint identifiers, sizes and SHA256 values are in `configs/sources.json` and the two checkpoint manifests. `POSTHOC_BACKBONE=ViT-B-16` selects the second backbone; B/32 is the default.

The checkpoints come from the Task Arithmetic release. Their image encoders use OpenAI-pretrained CLIP through OpenCLIP. The fixed heads contain normalized, template-averaged class-name text embeddings scaled by the pretrained logit scale; they are not fitted linear probes. The same head is used for all methods on a task. NBP does not train either the encoder or the head. Experiments load the released heads directly: they do not rebuild templates, tokenize class names or regenerate text embeddings. Use the exact head files in the checkpoint manifest when reproducing these results; a newly built head is a different input even if it follows the same prompt recipe.

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

Each model batch specifies its own source and weight hashes, split, task order, coefficient and recovery scope. Use the recorded execution version to reproduce a historical batch; do not replace its fingerprints with those of newer code. The evaluator's `--verify` recounts complete labels and per-class/paired outcomes, then replays selected scores in their original 128-image batch context. A pilot is an implementation check, not a replacement for a complete test set.

`experiment.report` only reads frozen records. Generated tables and figures go to the requested output directory, separate from input records. The default output is ignored by Git.

## Historical records

Stopped numerical-compensation experiments and their report code are available at commit `f1d7e496` and earlier. The active repository exposes the NBP method, matched baselines and their evidence. Unique model and raw-score files are not deleted by this source cleanup. Removed internal metadata on the prior compute host was preserved in its system-disk archive before updating that host.

```bash
git show <execution-commit>:<source-path>
git archive <execution-commit> | tar -x -C <empty-directory>
```

## Regenerate publications and clean AGA references

The construction command defaults to NBP and supports all seven evaluated targets. It does not run the stopped permutation method:

```bash
python -m experiment.construct --target MNIST --output outputs/nbp/MNIST
```

Repeat this command for the other tasks when reproducing the complete table. Historical protocols retain their original artifact paths and hashes. For newly regenerated files, make a new evaluation configuration with their actual hashes; do not edit the frozen result records.

After preparing all public checkpoints, this uses the same author AGA kernel and shared loader to generate the clean references, and connects the newly constructed bias files to the current evaluation configuration:

```python
import json
from pathlib import Path
import torch
from cfg import AUTHOR
from _impl.attacks import load_author, author_recover
from _impl.io import digest
from _impl.models import initialize, load_checkpoint

initialize()
cfg = json.loads(Path('configs/evaluate.json').read_text())
author, _ = load_author(AUTHOR / cfg['AGA']['file'], True)
base = load_checkpoint('zeroshot.pt')
folder = Path('outputs/clean_aga')
folder.mkdir(parents=True, exist_ok=False)
for task in cfg['tasks']:
    expert = load_checkpoint(f'{task}/finetuned.pt')
    aligned = author_recover(expert, base, author)
    path = folder / f'{task}.pt'
    torch.save(aligned.state_dict(), path)
    cfg['AGA_references'][task] = {'path': str(path), 'sha256': digest(path)}
    path = Path(f'outputs/nbp/{task}/release.pt')
    ref = {'path': str(path), 'sha256': digest(path)}
    cfg['defenses']['NBP_'+task].update(release=ref, AGA_edits=ref)
Path('outputs/evaluate.json').write_text(json.dumps(cfg, indent=2)+'\n')
```

Then run `python -m experiment.main_table --protocol outputs/evaluate.json --case 0 --pilot` and its `--verify` check before the complete case. Cases 0--6 follow the ordered task list in the configuration. The source fingerprint in each historical result identifies the executed version; original raw-score hashes are not expected to match a fresh run on different hardware.

## Additional training-based comparison

Trap² uses the pinned official training functions listed in `configs/trap2_baseline.json`. Place that checkout at `external/Trap2`. This comparison trains new experts and is separate from NBP and PaRaMS on the previously released checkpoints.

```bash
python -m experiment.train_baseline --task DTD --defense Trap2
python -m experiment.train_baseline --task DTD --defense CE
python -m experiment.train_baseline --task DTD --evaluate
```

Repeat for MNIST and SVHN. The matched CE control uses the same training loop with zero protection penalty. Both use the native public base and fixed classifier; the visual projection stays frozen as in the author's HF implementation. DTD uses its official validation split. MNIST/SVHN validation comes from 20% of training data, so the full official test sets remain untouched during training. This differs from the author's test-derived validation split and is recorded in the protocol. The author's training-log column named `Test` receives the validation loader, not test images.

The evaluator uses the best validation checkpoints, original fixed companions, TA at 0.5, and AGA applied to the target or both experts. It saves complete logits and uses the shared independent recount/replay. No new NBP checkpoint or NBP result is inferred from these runs.

MergeGuard is implemented from its CVPR paper, not an author release. The declared settings and implementation choices are in `configs/mergeguard_baseline.json`. Its CE control starts from the same fine-tuned expert; training and importance use training/validation images only. The stored `best.pt` is the final fixed-epoch model for this protocol.

```bash
python -m experiment.train_baseline --config configs/mergeguard_baseline.json --task MNIST --defense MergeGuard
python -m experiment.train_baseline --config configs/mergeguard_baseline.json --task MNIST --defense CE
python -m experiment.train_baseline --config configs/mergeguard_baseline.json --task MNIST --evaluate
```

Use the same commands for SVHN and DTD. Do not treat these B/32 adaptations as the authors' published L/14 numbers. Completed counts are in `results/phase3/trap2_baseline/summary.json` and `results/phase3/mergeguard_baseline/summary.json`; both are included in the standard table exporter.
