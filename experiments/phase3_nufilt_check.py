"""CPU-only official NUFILT kernel and checkpoint-compatibility checks, not an attack run."""

import ast
import hashlib
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn

from clip_benchmark import checkpoint_loader, digest


def load_official():
    root = Path('external/NUFILT')
    pins = {
        'fusion_bench/method/nufilt/utils.py': 'feaef2cb710d345521d6a51de49e434e144be09d6d23111c92708ed802a67983',
        'fusion_bench/models/filter_lora.py': '1d0897cb65c6bbc10520dafb8294be1d281034d812fa7be6374bf9babd7828ad',
        'fusion_bench/method/nufilt/nufilt.py': '68381a159aa37b538f77225aa68f497272acb9f8d59f16600e4a494d7176af75',
        'config/method/nufilt/nufilt.yaml': 'fbd950dde20625169b797d492e67244bde370c3920aae7a8c66a5a357961bfb6',
    }
    for p, sha in pins.items(): assert digest(root/p) == sha, p
    env = dict(torch=torch, Tensor=Tensor, nn=nn, Tuple=Tuple)
    for path, names in [('fusion_bench/method/nufilt/utils.py', {'_svd', 'svd'}),
                        ('fusion_bench/models/filter_lora.py', {'ResidualProjGate', 'FilterLoRA'})]:
        tree = ast.parse((root/path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root/path), 'exec'), env)
    return env, pins


def run():
    torch.set_num_threads(1)
    env, pins = load_official()
    torch.manual_seed(42)
    base = nn.Linear(16, 12)
    task = torch.randn(12, 16)*.02; previous = torch.randn(12, 16)*.02
    module = env['FilterLoRA'](base, task, True, previous, 3, 4, 2, 'cpu')
    with torch.no_grad(): module.gate.B.normal_(std=.01)
    gate = module.gate.get_weight()
    total = previous+task@gate
    explicit = ((task@module.task_V-total@module.task_V)**2).sum()
    explicit += ((previous@module.pre_V-total@module.pre_V)**2).sum()
    objective_error = float((module.solve_lora()-explicit).abs().detach())
    assert objective_error < 1e-6
    x = torch.randn(7,16)
    with torch.no_grad():
        source_forward = module(x)
        merged_forward = nn.functional.linear(x, module.merge_to_base(), base.bias)
        observed_difference = float((source_forward-merged_forward).abs().max())
    # Kernel correctness evidence does not certify architecture/checkpoint conversion.
    loader = checkpoint_loader(Path('external/secure-merge-attack'))
    checkpoint = Path('checkpoints/ViT-B-32/zeroshot.pt')
    assert digest(checkpoint) == '55a47b9d1b36051fc58a0da2a7c1d597c9194901d460ae8775c245bfeefa391f'
    original = loader(str(checkpoint)).cpu().eval()
    actual = [name for name, m in original.named_modules()
              if isinstance(m, nn.Linear) and not list(m.children()) and any(k in name for k in ['attn','fc1'])]
    expected_hf_count = 12*5  # q/k/v/out projection and fc1 per block, official configuration.
    protocol = json.loads(Path('configs/phase3.json').read_text())
    # Map the intended HF selection, without executing an incomplete native adapter.
    selected = {f'model.visual.transformer.resblocks.{i}.{suffix}' for i in range(12)
                for suffix in ['attn.in_proj_weight', 'attn.out_proj.weight', 'mlp.c_fc.weight']}
    coverage = []
    for task in protocol['tasks']:
        spec = protocol['defenses']['Phase1_'+task]
        ignored = []
        for stage, ref in [('release', spec['release']), ('AGA_edits', spec['AGA_edits'])]:
            assert digest(ref['path']) == ref['sha256']
            edits = torch.load(ref['path'], map_location='cpu', weights_only=False)
            touched = sorted(set(edits) & selected)
            ignored.append(dict(stage=stage, artifact_sha256=ref['sha256'], edited_keys=sorted(edits),
                                selected_weight_keys_in_edits=touched))
        coverage.append(dict(target=task, records=ignored,
            selected_weights_identical_to_matching_unprotected_input=all(not r['selected_weight_keys_in_edits'] for r in ignored)))
    result = dict(status='source_kernel_checked_direct_checkpoint_adapter_not_validated',
        repository='https://github.com/zihuanqiu/NUFILT', commit='4cd298fb7ea22bd4bcd343748fb4b8e3f9815324',
        source_sha256=pins, cpu_only=True, real_model_attacks=0, protected_weights_changed=False,
        official_projection_loss_vs_explicit_matrix_error=objective_error,
        official_gate_forward_vs_fused_linear_max_error=observed_difference,
        forward_equivalence_note='Nonzero BA need not be symmetric: forward x(P+BA) differs from fused-linear x(P+BA)^T. The official surrogate targets the fused matrix; this check does not invalidate final fused weights.',
        native_OpenAI_module_matches=actual, native_OpenAI_module_count=len(actual),
        expected_HF_selected_linear_count=expected_hf_count,
        direct_module_name_filter_faithful=len(actual)==expected_hf_count,
        phase1_official_selected_parameter_dependency=coverage,
        conditional_invariance='The literal official algorithm reads only selected task weights. If these inputs, anchor, seed and task order match, its final merged weights match; skipped final LN bias never enters the merge. This does not establish absolute task accuracy without a validated implementation on our checkpoints.',
        required_before_full_run=['Validate packed QKV splitting and c_fc/fc1 mapping.',
            'Preserve official skip rules for biases, LayerNorm, fc2 and visual projection; do not invent averaging.',
            'Check original first-task behavior, both fixed task orders and final fused matrix computation.',
            'Validate native/HF checkpoint and classifier conversion before interpreting accuracy.'],
        official_defaults=dict(lora_rank=64, null_rank=128, gradient_probe_rank=8, adam_lr=.001, max_steps=50),
        source_check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = Path('outputs/phase3/nufilt_source_check.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
