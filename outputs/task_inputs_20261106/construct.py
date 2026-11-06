"""One fixed A/B compensation point against coefficient recovery and B reset.

Uses the existing derivative of the author's 100-step coefficient fit.
The target expert and public base remain frozen.
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from _impl.attacks import load_author
from _impl.io import check_files, digest, new_output, read_json, save_json
from _impl.models import apply, checked_models, initialize
from results.data_free_compensation.synthetic_recovery.differentiate import recover_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    cfg = read_json(parser.parse_args().protocol)
    check_files(cfg['source_sha256'])
    initialize()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    out = new_output(cfg['output'], cfg)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    base, original, head, pins = checked_models('MNIST')
    parent = apply(original, cfg['parent'])
    check_files({cfg['synthetic_source']['path']: cfg['synthetic_source']['sha256'],
                 cfg['recovered_model']['path']: cfg['recovered_model']['sha256']})
    raw = torch.load(cfg['synthetic_source']['path'], map_location='cpu', weights_only=False)
    if 'retention_source' in cfg:
        ref=cfg['retention_source']; assert digest(ref['path'])==ref['sha256']
        old=torch.load(ref['path'],map_location='cpu',weights_only=False)
        for name in ('native','direct','recovered'):
            raw['cache'][name]={k:torch.cat((old['cache'][name][k],v)) for k,v in raw['cache'][name].items()}
        del old
    ref = cfg['synthetic_partner_source']
    check_files({ref['path']: ref['sha256']})
    other = torch.load(ref['path'], map_location='cpu', weights_only=False)
    pixels = torch.cat((raw['final'][cfg['synthetic_source']['indices']],
                        other[ref['key']][ref['indices']])).cuda().double()
    del other
    norm = original.val_preprocess.transforms[-1]
    images = F.interpolate(pixels.float(), (224, 224), mode='bilinear', align_corners=False)
    images = (images - images.new_tensor(norm.mean)[None, :, None, None]) / images.new_tensor(norm.std)[None, :, None, None]
    recovered = copy.deepcopy(original)
    recovered.load_state_dict(torch.load(cfg['recovered_model']['path'], map_location='cpu', weights_only=False))
    official_scores = []
    head.cuda()

    def inputs(model):
        rows = []
        hook = model.model.visual.transformer.resblocks[11].register_forward_pre_hook(
            lambda m, args: rows.append(args[0].detach().cpu()))
        with torch.no_grad():
            model.cuda()
            for x in images.split(16):
                features = model(x)
                if model is recovered:
                    official_scores.append(head(features).detach().cpu())
            model.cpu()
        hook.remove()
        return torch.cat(rows, 1).cuda().double()

    xm, xt, xb = inputs(recovered), inputs(parent)[:, :64], inputs(base)[:, 64:]
    with torch.no_grad():
        original.cuda(); head.cuda()
        labels = torch.cat([head(original(x)).argmax(1) for x in images[:64].split(16)])
        original.cpu(); head.cpu()
    block = copy.deepcopy(base.model.visual.transformer.resblocks[11]).cuda().double().eval()
    names = tuple(dict(block.named_parameters()))
    anchor = tuple(p.detach() for p in block.parameters())
    initial = tuple(p.detach().cuda().double() for p in parent.model.visual.transformer.resblocks[11].parameters())
    clean = tuple(p.detach().cuda().double() for p in original.model.visual.transformer.resblocks[11].parameters())
    iA, ia, iB, ib = [names.index(k) for k in ('mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias')]
    beta = initial[names.index('ln_2.bias')]
    U, _, Vh = torch.linalg.svd(clean[iA] - anchor[iA], full_matrices=False)
    U, V = U[:, :cfg['rank']], Vh[:cfg['rank']].T
    native_ln = copy.deepcopy(parent.model.visual.ln_post).cuda().double()
    native_proj = parent.model.visual.proj.detach().cuda().double()
    attack_ln = copy.deepcopy(recovered.model.visual.ln_post).cuda().double()
    attack_proj = recovered.model.visual.proj.detach().cuda().double()
    public_ln = copy.deepcopy(native_ln)
    public_ln.load_state_dict({k: (v + base.model.visual.ln_post.state_dict()[k].cuda().double()) / 2
                              for k, v in native_ln.state_dict().items()})
    public_proj = (native_proj + base.model.visual.proj.detach().cuda().double()) / 2
    native = {k: raw['cache']['native'][k].cuda().double() for k in ('x', 'prefix', 'scores')}
    contexts = {k: {n: raw['cache'][k][n].cuda().double() for n in ('x', 'prefix')}
                for k in ('direct', 'recovered')}
    head = head.cuda().double()
    attack, _ = load_author(Path(cfg['AGA_source']['path']))
    # Recompute the current assignment in every call, including line searches.
    # Differentiate its selected branch; the discrete assignment has no gradient.
    del raw, recovered, original, parent, base, images, pixels, Vh
    act = lambda x: x * torch.sigmoid(1.702 * x)
    call = lambda p, x: functional_call(block, dict(zip(names, p)), (x,))
    public_target = call(anchor, xb).detach()

    def inner_loss(alpha, delta, target):
        w = tuple(b + a * d for b, a, d in zip(anchor, alpha, delta))
        student = call(w, xm)
        return (student[:, :64] - target).square().sum() + (student[:, 64:] - public_target).square().sum()

    def recovery_scores(p):
        return head(attack_ln(call(p, xm[:, :64])[0]) @ attack_proj)

    def margin(scores, teacher_class):
        true = scores.gather(1, teacher_class[:, None]).flatten()
        other = scores.scatter(1, teacher_class[:, None], -torch.inf).amax(1)
        return (other - true).clamp_max(0).mean()

    def measure(p):
        return (margin(recovery_scores(p), labels),)

    def class_scores(p):
        current_order=attack['solve_permutation'](p[iA].detach().float().cpu(),anchor[iA].float().cpu()).cuda()
        hidden = native['prefix'] + F.linear(act(F.linear(native['x'], p[iA], p[ia])), p[iB], p[ib])
        result = {'native': head(native_ln(hidden) @ native_proj)}
        for name, row in contexts.items():
            a, aa, b = p[iA], p[ia], p[iB]
            if name == 'recovered':
                a, aa, b = a[current_order], aa[current_order], b[:, current_order]
            z = act(F.linear(row['x'], (a + anchor[iA]) / 2, (aa + anchor[ia]) / 2))
            hidden = row['prefix'] + F.linear(z, (b + anchor[iB]) / 2, (p[ib] + anchor[ib]) / 2)
            result[name] = head(public_ln(hidden) @ public_proj)
            if name == 'recovered':
                hidden = row['prefix'] + F.linear(z, anchor[iB], anchor[ib])
                result['B_reset'] = head(public_ln(hidden) @ public_proj)
        return result

    def recovered_forward(p):
        p = tuple(v.detach() for v in p)
        delta = tuple(v - b for v, b in zip(p, anchor))
        target = call(p, xt).detach()
        alpha = tuple(torch.full_like(v, .3, requires_grad=True) for v in p)
        optimizer = torch.optim.Adam(alpha, lr=.01, foreach=False)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            inner_loss(alpha, delta, target).backward()
            optimizer.step()
        return tuple((b + a * d).detach() for b, a, d in zip(anchor, alpha, delta))

    with torch.no_grad():
        fixed = class_scores(initial)
        fixed['native'] = native['scores']  # Original expert, not current release.
        fixed = {k: v.detach() for k, v in fixed.items()}
    protected_recovery = recovered_forward(initial)
    clean_recovery = recovered_forward(clean)
    initial_check = dict(
        target_images=64, public_images=64, real_images=0, defense_updates=0,
        official_fp32_target_matches=int(torch.cat(official_scores)[:64].argmax(1).eq(labels.cpu()).sum()),
        simulated_fp64_target_matches=int(recovery_scores(protected_recovery).argmax(1).eq(labels).sum()),
        unprotected_fp64_target_matches=int(recovery_scores(clean_recovery).argmax(1).eq(labels).sum()),
        scope='Fixed initial source comparison; no selection or parameter changes from these results.')
    save_json(out / 'initial_recovery_check.json', initial_check)
    print('INITIAL_RECOVERY', initial_check, flush=True)
    del protected_recovery, clean_recovery, official_scores
    releases = {}
    for name, strength in cfg['conditions'].items():
        folder = out / name
        folder.mkdir()
        variables = [torch.nn.Parameter(initial[i].clone()) for i in (iA, iB, ib)]

        def effective():
            a, b, bias = variables
            d = a - clean[iA]
            a = clean[iA] + d * (cfg['A_limit'] * clean[iA].norm() / d.norm().clamp_min(1e-30)).clamp(max=1)
            d = b - clean[iB]
            b = clean[iB] + d * (cfg['B_limit'] * clean[iB].norm() / d.norm().clamp_min(1e-30)).clamp(max=1)
            p = list(initial)
            p[iA], p[ia], p[iB], p[ib] = a, initial[ia] - (a - initial[iA]) @ beta, b, bias
            return tuple(p)

        def kl(score, target):
            t = cfg['temperature']
            return t*t*F.kl_div(F.log_softmax(score/t, 1), F.softmax(target/t, 1), reduction='batchmean')

        optimizer = torch.optim.LBFGS(variables, **cfg['solver'])
        records = []
        start = time.monotonic()

        def closure():
            optimizer.zero_grad(set_to_none=True)
            p = effective()
            scores = class_scores(p)
            native_kl = kl(scores['native'], fixed['native'])
            public_kl = torch.stack([kl(scores[k], fixed[k]) for k in contexts]).mean()
            delta_A = p[iA] - clean[iA]
            task_change = ((U.T @ delta_A).square().sum() / (U.T @ (clean[iA]-anchor[iA])).square().sum()
                           + (delta_A @ V).square().sum() / ((clean[iA]-anchor[iA]) @ V).square().sum())
            retain = native_kl + public_kl + task_change
            retain.backward(retain_graph=bool(strength))
            assert strength == 1.0
            recovered = recovered_forward(p)
            recovered_margin = float(measure(recovered)[0].detach())
            replacement_margin = margin(scores['B_reset'], fixed['native'].argmax(1))
            active = 'distillation' if recovered_margin <= float(replacement_margin.detach()) else 'B_reset'
            if active == 'distillation':
                value, gradient, _ = recover_gradient(inner_loss, measure, anchor,
                    tuple(v.detach() for v in p), lambda w: call(w, xt))
                assert abs(value-recovered_margin) < 1e-5 * (1+abs(value))
                changing = (iA, ia, iB, ib)
                torch.autograd.backward(tuple(p[i] for i in changing), tuple(-strength * gradient[i] for i in changing))
            else:
                (-strength * replacement_margin).backward()
            objective = float(retain.detach()) - strength * min(recovered_margin, float(replacement_margin.detach()))
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in variables)
            # The shared recovery derivative kernel has independent forward/reverse validation.
            # Actual finite objectives are recomputed by the library line search.
            records.append(dict(evaluation=len(records)+1, objective=objective, native_KL=float(native_kl.detach()),
                                public_KL=float(public_kl.detach()), task_change=float(task_change.detach()),
                                recovered_margin=recovered_margin,
                                B_reset_margin=float(replacement_margin.detach()), active=active,
                                seconds=time.monotonic()-start))
            save_json(folder/'progress.json', records)
            print(name, records[-1], flush=True)
            return torch.tensor(objective, device='cuda', dtype=torch.float64)

        closure()  # Record the original point before the library update.
        optimizer.step(closure)
        # Freeze only the final library point, with its actual FP32 rounding.
        rounded = tuple(v.detach().float().double() if i in (iA, ia, iB, ib) else v.detach()
                        for i, v in enumerate(effective()))
        point = {f'model.visual.transformer.resblocks.11.{names[i]}': rounded[i].float().cpu()
                 for i in (iA, ia, iB, ib)}
        torch.save(point, folder/'point.pt')
        scores = {k: v.detach().cpu() for k, v in class_scores(rounded).items()}
        rec = recovered_forward(rounded)
        scores['distilled'] = recovery_scores(rec).detach().cpu()
        torch.save(dict(scores=scores, reference={k:v.cpu() for k,v in fixed.items()},
                        recovery_labels=labels.cpu()), folder/'source_scores.pt')
        result = dict(release={'path':str(folder/'point.pt'), 'sha256':digest(folder/'point.pt')},
                      records=records, iterations=int(optimizer.state[variables[0]]['n_iter']),
                      seconds=time.monotonic()-start, real_construction_images=0, original_expert_updates=0,
                      native_n=len(fixed['native']),recovery_n=len(labels),
                      relative_edits={names[i]:float((rounded[i]-clean[i]).norm()/clean[i].norm()) for i in (iA,ia,iB,ib)},
                      native_matches=int((scores['native'].argmax(1)==fixed['native'].argmax(1).cpu()).sum()),
                      B_reset_matches=int((scores['B_reset'].argmax(1)==fixed['native'].argmax(1).cpu()).sum()),
                      recovery_matches=int((scores['distilled'].argmax(1)==labels.cpu()).sum()),
                      A_relative=float((rounded[iA]-clean[iA]).norm()/clean[iA].norm()),
                      B_relative=float((rounded[iB]-clean[iB]).norm()/clean[iB].norm()),
                      checkpoint_sha256=pins, source_scores_sha256=digest(folder/'source_scores.pt'))
        save_json(folder/'result.json', result)
        releases[name] = result['release']
        save_json(out/'releases.json', releases)
        print('FROZEN', name, {k:v for k,v in result.items() if k not in ('records','checkpoint_sha256')}, flush=True)
    save_json(out/'completed.json', {'releases':releases})


if __name__ == '__main__':
    main()
