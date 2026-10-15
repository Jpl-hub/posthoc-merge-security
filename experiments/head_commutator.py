"""One preregistered source-only finite inter-head feasibility test; no optimizer."""

import argparse
import contextlib
import copy
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clip_benchmark import digest, merge
from covariance_noise import dataset
from cross_module_distance import extended_parameters, final_hidden
from mechanisms import load_author
from params_reference_eval import fresh_split
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models

PREFIX = 'model.visual.transformer.resblocks.11.attn.'


@torch.no_grad()
def capture(model, head, data, indices):
    block = model.model.visual.transformer.resblocks[-1]
    values = {k: [] for k in ['x', 'before', 'attention', 'r', 'h', 'scores', 'labels']}
    hooks = [block.attn.register_forward_pre_hook(lambda m, a: values['x'].append(a[0].transpose(0, 1).detach().cpu())),
             block.ln_1.register_forward_pre_hook(lambda m, a: values['before'].append(a[0][0].detach().cpu())),
             block.attn.register_forward_hook(lambda m, a, y: values['attention'].append(y[0][0].detach().cpu())),
             block.ln_2.register_forward_pre_hook(lambda m, a: values['r'].append(a[0][0].detach().cpu())),
             model.model.visual.ln_post.register_forward_pre_hook(lambda m, a: values['h'].append(a[0].detach().cpu()))]
    model.cuda().eval(); head.cuda().eval()
    try:
        for x, y in make_loader(data, indices, 64):
            feature = model(x.cuda()); values['scores'].append(head(feature).cpu()); values['labels'].append(y)
    finally:
        model.cpu(); head.cpu()
        for hook in hooks: hook.remove()
    return {k: torch.cat(v) for k, v in values.items()}


def attention_parameters(model):
    a = model.model.visual.transformer.resblocks[-1].attn
    return {k: v.detach().cpu().double() for k, v in dict(W=a.in_proj_weight, bias=a.in_proj_bias,
                O=a.out_proj.weight, ob=a.out_proj.bias).items()}


def attention(x, a):
    q, k, v = F.linear(x, a['W'], a['bias']).chunk(3, -1)
    n, length, d = x.shape; heads = 12; width = d//heads
    q = q[:, 0].reshape(n, heads, width)
    k = k.reshape(n, length, heads, width).transpose(1, 2)
    v = v.reshape(n, length, heads, width).transpose(1, 2)
    probability = (torch.einsum('nhd,nhld->nhl', q, k)/math.sqrt(width)).softmax(-1)
    z = torch.einsum('nhl,nhld->nhd', probability, v).reshape(n, d)
    return F.linear(z, a['O'], a['ob']), probability, v


def rotated(a, pair, angle):
    i, j = pair; width = 64; c, s = math.cos(angle), math.sin(angle)
    r = torch.eye(768, dtype=a['W'].dtype, device=a['W'].device)
    u = torch.arange(width, device=r.device)+width*i; v = torch.arange(width, device=r.device)+width*j
    r[u, u] = c; r[v, v] = c; r[u, v] = -s; r[v, u] = s
    w = a['W'].clone(); b = a['bias'].clone()
    w[1536:] = r@w[1536:]; b[1536:] = r@b[1536:]
    return a | dict(W=w, bias=b, O=a['O']@r.T)


def coefficient(a, pair):
    ix = torch.cat([torch.arange(i*64, (i+1)*64, device=a['W'].device) for i in pair])
    return float(a['W'][1536+ix].square().sum()/a['W'].square().sum()
                 +a['bias'][1536+ix].square().sum()/a['bias'].square().sum()
                 +a['O'][:, ix].square().sum()/a['O'].square().sum())


def distance(a, b, include_bias=True):
    keys = ['W', 'O', 'bias'] if include_bias else ['W', 'O']
    return sum(float(((b[k]-a[k]).norm()/a[k].norm())**2) for k in keys)**.5


def tail(cache, a, p):
    y = attention(cache['x'], a)[0]
    r = cache['before']+y
    return scores(final_hidden(r, p), p)


def margins(before, after):
    winner = before.argmax(1); yy = winner[:, None]
    gap = before.gather(1, yy)-before; valid = gap>0
    ds = after-before
    drift = (ds.gather(1, yy)-ds)/gap.masked_fill(~valid, 1.)
    magnitude = drift.abs().masked_fill(~valid, 0.)
    damage = (-drift).masked_fill(~valid, 0.)
    row, rival = np.unravel_index(int(magnitude.argmax()), tuple(magnitude.shape))
    changed = before.argmax(1).ne(after.argmax(1))
    return dict(n=len(before), changed_predictions=int(changed.sum()), worst_margin_drift=float(magnitude.max()),
                max_signed_margin_damage=float(damage.max()),
                witness=dict(row=int(row), teacher=int(winner[row]), competitor=int(rival),
                             teacher_margin=float(gap[row, rival]), after_margin=float(after[row, winner[row]]-after[row, rival])),
                per_class=[dict(class_id=c, n=int(winner.eq(c).sum()), changed=int((winner.eq(c)&changed).sum()),
                                worst_margin_drift=float(magnitude[winner.eq(c)].max()) if winner.eq(c).any() else None,
                                max_signed_margin_damage=float(damage[winner.eq(c)].max()) if winner.eq(c).any() else None)
                           for c in range(before.shape[1])])


@torch.no_grad()
def run(protocol):
    cfg = json.loads(Path(protocol).read_text()); out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False); (out/'artifacts').mkdir()
    save_json(out/'protocol.json', cfg); save_json(out/'case.json', dict(target='MNIST', source_only=True))
    for path, sha in cfg['source_sha256'].items(): assert digest(path)==sha, path
    assert '3090' in torch.cuda.get_device_name()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    start = time.monotonic(); anchor, own, head, hashes = checked_models('MNIST')
    train = dataset('MNIST', 'data', own.val_preprocess, True)
    ids = fresh_split(dict(confirmation_seed=cfg['seed'], confirmation_n=cfg['source_n']), out)
    split = json.loads((out/'confirmation_split.json').read_text())
    split.update(role='construction_only_not_independent_confirmation', used_for_optimization=False,
                 used_for_structural_selection=True)
    save_json(out/'source_split.json', split); (out/'confirmation_split.json').unlink()
    save_json(out/'training_indices.json', dict(fit=ids))
    cache_cpu = capture(own, head, train, ids)
    cache = {k: v.cuda().double() for k, v in cache_cpu.items()}
    a = on_device(attention_parameters(own)); p = on_device(extended_parameters(own, head))
    a0 = on_device(attention_parameters(anchor))
    original = tail(cache, a, p); baseline_error=float((original-cache['scores']).abs().max())
    assert baseline_error<cfg['full_forward_atol'], baseline_error
    y0, probability, value = attention(cache['x'], a)
    winners = original.argmax(1); records=[]
    for i in range(12):
        for j in range(i+1, 12):
            k=coefficient(a, [i, j]); angle=2*math.asin(min(cfg['construction_radius']/(2*math.sqrt(k)), 1/math.sqrt(2)))
            mixed=rotated(a, [i,j], angle)
            residual=attention(cache['x'], mixed)[0]-y0
            fraction=residual.square().sum(1)/y0.square().sum(1).clamp_min(1e-30)
            risk=max(float(fraction[winners.eq(c)].mean()) for c in range(10) if winners.eq(c).any())
            records.append(dict(pair=[i,j], angle_radians=angle, K=k, worst_class_mean_attention_relative_energy=risk))
    selected=min(records, key=lambda r:(r['worst_class_mean_attention_relative_energy'],r['pair']))
    save_json(out/'selection.json', dict(rule=cfg['selection_rule'], selected=selected, structural_edges=records,
                                       no_classification_or_fusion_result_used_in_selection=True))
    mixed=rotated(a, selected['pair'], selected['angle_radians'])
    # Materialize once in original FP32. All following checks use these exact values.
    mixed={k:v.float().double() for k,v in mixed.items()}
    natural=tail(cache,mixed,p); natural_report=margins(original,natural)
    svd_u,svd_s,svd_vh=torch.linalg.svd(a['W'][1536:]-a0['W'][1536:],full_matrices=False)
    excitation=(cache['x'].reshape(-1,768)@svd_vh.T).square().mean(0)*svd_s.square()
    selected_modes=excitation.argsort(descending=True,stable=True)[:cfg['probe_rank']]
    contexts=torch.stack([winners.eq(c).nonzero()[0,0] for c in range(10)])
    vectors=svd_vh[selected_modes]; rms=(cache['x'][:,0]@vectors.T).square().mean(0).sqrt()
    shifts=(cfg['probe_fraction']*torch.tensor([-1.,1.],device='cuda')[:,None,None]*rms[None,:,None]*vectors[None,:,:]).reshape(-1,768)
    count=len(shifts); probe_x=cache['x'][contexts].repeat_interleave(count,0).clone();probe_x[:,0]+=shifts.repeat(10,1)
    probe_cache=dict(x=probe_x,before=cache['before'][contexts].repeat_interleave(count,0))
    probe_before=tail(probe_cache,a,p);probe_after=tail(probe_cache,mixed,p); probe_report=margins(probe_before,probe_after)
    edits={PREFIX+'in_proj_weight':mixed['W'].cpu().float(),PREFIX+'in_proj_bias':mixed['bias'].cpu().float(),PREFIX+'out_proj.weight':mixed['O'].cpu().float()}
    torch.save(edits,out/'artifacts/diagnostic_edits.pt')
    save_json(out/'transform_freeze.json',dict(selected=selected,parameter_distance=distance(a,mixed),
              matrix_distance=distance(a,mixed,False),edits_sha256=digest(out/'artifacts/diagnostic_edits.pt'),
              source_indices_sha256=digest(out/'source_split.json'),source_sha256=cfg['source_sha256'],
              promotion=False, frozen_before_source_function_acceptance=True, original_expert_updated=False))
    material=copy.deepcopy(own); material.load_state_dict(material.state_dict()|edits,strict=True)
    actual_cpu=capture(material,head,train,ids)
    reconstruction_error=float((actual_cpu['scores'].double()-natural.cpu()).abs().max())
    assert reconstruction_error<cfg['full_forward_atol'],reconstruction_error
    task_pass=max(natural_report['worst_margin_drift'],probe_report['worst_margin_drift'])<=cfg['task_budget']
    # Fixed author layer function is sufficient here because all other layers and MLP matrices are unchanged.
    author,author_sha=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    assert author_sha==cfg['AGA_text_sha256']
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        aligned0=author['attack_attention_headwise_precise'](own.state_dict(),anchor.state_dict(),11,num_heads=12,device='cuda')
        aligned1=author['attack_attention_headwise_precise'](material.state_dict(),anchor.state_dict(),11,num_heads=12,device='cuda')
    (out/'AGA_module.log').write_text(log.getvalue())
    ad0={k:aligned0[PREFIX+name].double().cuda() for k,name in [('W','in_proj_weight'),('bias','in_proj_bias'),('O','out_proj.weight'),('ob','out_proj.bias')]}
    ad1={k:aligned1[PREFIX+name].double().cuda() for k,name in [('W','in_proj_weight'),('bias','in_proj_bias'),('O','out_proj.weight'),('ob','out_proj.bias')]}
    aligned_distance=sum(float(((ad1[k]-ad0[k]).norm()/a[k].norm())**2) for k in ['W','O'])**.5
    # A source-input module intervention, not full-model AGA accuracy or real merging.
    intervention0=tail(cache,ad0,p);intervention1=tail(cache,ad1,p)
    intervention=margins(intervention0,intervention1)
    proxy_report=None;proxy_cache=None;proxy_after=None
    if task_pass:
        public=merge(anchor,own,anchor,.5);public_cache_cpu=capture(public,head,train,ids)
        public_parameters=on_device(extended_parameters(public,head));public_a=on_device(attention_parameters(public))
        public_changed={k:public_a[k]+.5*(mixed[k]-a[k]) for k in public_a}
        proxy_cache={k:v.cuda().double() for k,v in public_cache_cpu.items()}
        proxy_before=tail(proxy_cache,public_a,public_parameters);proxy_after=tail(proxy_cache,public_changed,public_parameters)
        proxy_report=margins(proxy_before,proxy_after)
    source_classes=0 if proxy_report is None else sum(x['changed']>0 for x in proxy_report['per_class'])
    signals=task_pass and source_classes>=cfg['minimum_source_changed_classes']
    artifact=dict(cache=cache_cpu,attention_parameters={k:v.cpu() for k,v in a.items()},
                  modified_attention_parameters={k:v.cpu() for k,v in mixed.items()},tail_parameters={k:v.cpu() if torch.is_tensor(v) else v for k,v in p.items()},
                  selected=selected,selection=records,original=original.cpu(),natural=natural.cpu(),actual=actual_cpu['scores'],
                  probes={k:v.cpu() for k,v in probe_cache.items()},probe_before=probe_before.cpu(),probe_after=probe_after.cpu(),
                  task_SVD=dict(u=svd_u.cpu(),s=svd_s.cpu(),vh=svd_vh.cpu(),selected=selected_modes.cpu(),excitation=excitation.cpu(),shifts=shifts.cpu(),contexts=contexts.cpu()),
                  AGA_baseline={k:v.cpu() for k,v in ad0.items()},AGA_modified={k:v.cpu() for k,v in ad1.items()},
                  intervention_before=intervention0.cpu(),intervention_after=intervention1.cpu(),proxy_report=proxy_report)
    torch.save(artifact,out/'artifacts/finite_audit.pt')
    report=dict(status='source_signal_pending_independent_review' if signals else ('closed_source_task_constraint_failed' if not task_pass else 'closed_no_finite_multiclass_signal_in_fixed_test'),
                source_n=len(ids),selected=selected,parameter_distance=distance(a,mixed),matrix_distance=distance(a,mixed,False),
                parameter_budget=cfg['parameter_budget'],task_budget=cfg['task_budget'],natural=natural_report,probes=probe_report,
                source_original_correct=int(cache['scores'].argmax(1).eq(cache['labels']).sum()),source_modified_correct=int(natural.argmax(1).eq(cache['labels']).sum()),
                task_pass=task_pass,AGA_aligned_matrix_distance=aligned_distance,AGA_source_module_intervention=intervention,
                intervention_is_full_model_AGA=False,public_base_interpolation=proxy_report,source_signal=signals,
                baseline_reconstruction_max_abs=baseline_error,modified_reconstruction_max_abs=reconstruction_error,
                new_release_weights=0,diagnostic_edits_sha256=digest(out/'artifacts/diagnostic_edits.pt'),
                finite_audit_sha256=digest(out/'artifacts/finite_audit.pt'),original_checkpoint_sha256=hashes,
                source_sha256=cfg['source_sha256'],official_AGA_sha256=author_sha,gradients=0,optimizer_updates=0,
                future_partner_loaded=False,test_data_loaded=False,confirmation_data_used=False,
                wall_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),GPU=torch.cuda.get_device_name())
    save_json(out/'results.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True)
    run(parser.parse_args().protocol)
