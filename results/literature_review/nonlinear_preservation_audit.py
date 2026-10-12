"""Inspect the frozen MNIST architecture on CPU; never modify its weights."""
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/nonlinear_preservation_audit'
protocol = json.loads((OUT / 'protocol.json').read_text())
assert not (OUT / 'results.json').exists(), 'Do not overwrite an audit'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(4)
torch.set_grad_enabled(False)
sys.path.insert(0, str(ROOT / 'experiments'))
from clip_benchmark import checkpoint_loader

checkpoint = ROOT / 'checkpoints/ViT-B-32/MNIST/finetuned.pt'
assert sha(checkpoint) == protocol['checkpoint_sha256']
loader = checkpoint_loader(ROOT / 'external/secure-merge-attack')
model = loader(checkpoint, map_location='cpu').eval()
block = model.model.visual.transformer.resblocks[0]
source_file = Path(inspect.getfile(type(block)))
assert sha(source_file) == protocol['open_clip_model_sha256']
assert type(block).__name__ == 'ResidualAttentionBlock'
assert type(block.mlp.gelu).__name__ == 'QuickGELU'
assert isinstance(block.ln_attn, torch.nn.Identity)
assert isinstance(block.mlp.ln, torch.nn.Identity)
assert tuple(block.mlp.c_fc.weight.shape) == (3072, 768)
assert tuple(block.mlp.c_proj.weight.shape) == (768, 3072)
assert block.ln_1 is not block.ln_2
result = dict(protocol=protocol, checkpoint_verified=True,
    package_version=importlib.metadata.version('open_clip_torch'),
    loaded_block_class=type(block).__module__ + '.' + type(block).__name__,
    loaded_activation_class=type(block.mlp.gelu).__module__ + '.' + type(block.mlp.gelu).__name__,
    source_sha256=sha(source_file),
    block_forward_source=inspect.getsource(type(block).forward),
    activation_source=inspect.getsource(type(block.mlp.gelu)),
    LN1_LN2_are_distinct_modules=True,
    LN1_LN2_weights_exactly_equal=bool(torch.equal(block.ln_1.weight, block.ln_2.weight)),
    LN1_LN2_biases_exactly_equal=bool(torch.equal(block.ln_1.bias, block.ln_2.bias)),
    MLP_hidden_dimension=3072, hidden_normalization='Identity',
    attention_output_normalization='Identity',
    script_sha256=sha(__file__), protocol_sha256=sha(OUT / 'protocol.json'),
    new_model_forwards=0, new_protected_weights=0, new_classification_results=0,
    completed=True)
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
