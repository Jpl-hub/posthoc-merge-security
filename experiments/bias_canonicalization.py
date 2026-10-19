"""Architecture-aware LN-bias recovery; supplementary adaptation, not author AGA."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader, digest, evaluate, merge
from covariance_noise import dataset
from phase3_evaluate import apply, row_metrics
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def inputs(protocol):
    cfg=json.loads(Path(protocol).read_text())
    assert cfg['GPU'].removeprefix('RTX') in torch.cuda.get_device_name()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for path,pin in cfg['source_sha256'].items(): assert digest(path)==pin,path
    assert digest(cfg['frozen_protocol'])==cfg['frozen_protocol_sha256']
    frozen=json.loads(Path(cfg['frozen_protocol']).read_text())
    pins={r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    load=checkpoint_loader(Path('external/secure-merge-attack'))
    def checked(name):
        path=Path('checkpoints/ViT-B-32')/name; assert digest(path)==pins['ViT-B-32/'+name]
        model=load(str(path)).cpu().eval(); model.requires_grad_(False); return model
    return cfg,frozen,checked


@torch.no_grad()
def canonicalize(model, independent=False):
    """Minimum-norm bias with the same published visual projection response."""
    p = model.model.visual.proj.detach().cpu().double()
    beta = model.model.visual.ln_post.bias.detach().cpu().double()
    q, r = torch.linalg.qr(p, mode='reduced')
    assert torch.linalg.matrix_rank(r) == p.shape[1]
    ideal = torch.from_numpy(np.linalg.lstsq(p.numpy().T, (beta@p).numpy(), rcond=None)[0]) if independent else q@(q.T@beta)
    error = float(((ideal-beta)@p).abs().max())
    assert error < 1e-8, 'Ideal function-preservation identity failed.'
    result = copy.deepcopy(model)
    result.model.visual.ln_post.bias.copy_(ideal.float())
    actual = result.model.visual.ln_post.bias.double()
    return result, dict(ideal_projection_error=error,
        FP32_projection_error=float(((actual-beta)@p).abs().max()),
        bias_norm_before=float(beta.norm()), bias_norm_after=float(actual.norm()))


def affine_readout(gamma,beta,projection):
    """Same normalized tail function, native LN, with no secret or reference."""
    linear=gamma[:,None]*projection;linear=linear-linear.mean(0)
    offset=beta@projection;scale=(linear.square().sum()+offset.square().sum()).sqrt()
    assert float(scale)>0
    return torch.ones_like(gamma),torch.ones_like(beta),(linear+offset[None]/len(gamma))/scale,scale


@torch.no_grad()
def affine_family(folder):
    """Read-only check of the closed point's tail equivalence class, not an attack evaluation."""
    root=Path(folder);output=root/'affine_family_limit.json';assert not output.exists();start=time.monotonic();torch.set_num_threads(4)
    freeze=json.loads((root/'before_images.json').read_text());report=json.loads((root/'results.json').read_text())
    for file,pin in [('point.pt',freeze['point_sha256']),('frozen_release.pt',freeze['release_sha256']),('source.pt',report['raw_sha256'])]:assert digest(root/file)==pin
    point=torch.load(root/'point.pt',map_location='cpu');edits=torch.load(root/'frozen_release.pt',map_location='cpu');raw=torch.load(root/'source.pt',map_location='cpu')
    assert point['site']=='READOUT';h=raw['inputs'].double();g=point['gamma'];head=raw['head_weight'];d=len(g)
    rows={};packed=[];invariants=[];C=np.eye(d)-np.ones((d,d))/d
    for label,b,p in [('original',point['beta'],point['projection']),('published',edits['model.visual.ln_post.bias'].double(),edits['model.visual.proj'].double())]:
        gc,bc,pc,scale=affine_readout(g,b,p);packed.append(pc)
        gn,bn,pn=g.numpy(),b.numpy(),p.numpy();L=C@(gn[:,None]*pn);offset=bn@pn;K=np.vstack([L,offset]);s=np.linalg.norm(K)
        expected=(L+offset[None]/d)/s;np.testing.assert_allclose(expected,pc.numpy(),rtol=0,atol=1e-12)
        invariants.append(K/s);singular=np.linalg.svd(K,compute_uv=False);assert singular[1]>singular[0]*1e-8
        def score(hh,gain,bias,projection):
            z=torch.nn.functional.layer_norm(hh,(d,),gain,bias,point['eps']);f=z@projection
            return f/f.norm(dim=1,keepdim=True)@head.to(f).T
        before=score(h,g,b,p);same=score(h,gc,bc,pc);identity=float((same-before).abs().max());assert identity<1e-8
        hn=h.numpy();hc=hn-hn.mean(1,keepdims=True);u=hc/np.sqrt((hc*hc).mean(1,keepdims=True)+point['eps'])
        fb=(u*gn+bn)@pn;fc=(u+1)@expected
        nb=fb/np.linalg.norm(fb,axis=1,keepdims=True)@head.numpy().T;nc=fc/np.linalg.norm(fc,axis=1,keepdims=True)@head.numpy().T
        numpy_error=max(float(abs(nb-nc).max()),float(abs(nc-same.numpy()).max()));assert numpy_error<1e-8
        native=score(h.float(),gc.float(),bc.float(),pc.float()).double()
        cached=raw['before'] if label=='original' else raw['after'];error=float((native-cached).abs().max())
        source_cfg=json.loads((root/'protocol.json').read_text());assert error<=source_cfg['score_tolerance']
        delta=native-cached;y=cached.argmax(1);ix=torch.arange(len(y));gap=cached[ix,y,None]-cached;gap[ix,y]=1
        drift=(delta[ix,y,None]-delta)/gap.clamp_min(1e-12);drift[ix,y]=0
        rows[label]=dict(affine_map_second_singular_value=float(singular[1]),positive_scale=float(scale),
            NumPy_packing_error=float(abs(expected-pc.numpy()).max()),FP64_cached_score_identity_error=identity,
            NumPy_complete_readout_identity_error=numpy_error,
            FP32_cached_score_error=error,FP32_cached_relative_margin_drift=float(drift.abs().max()),
            FP32_cached_prediction_changes=int((native.argmax(1)!=cached.argmax(1)).sum()))
    difference=float((packed[0]-packed[1]).abs().max());unequal=int((packed[0].float()!=packed[1].float()).sum())
    save_json(output,dict(status='read_only_native_affine_equivalence_limit_checked',source_n=len(h),rows=rows,
        normalized_affine_invariant_max_abs_difference=float(abs(invariants[0]-invariants[1]).max()),
        canonical_projection_max_abs_difference=difference,FP32_canonical_projection_unequal_entries=unequal,
        canonical_models_not_claimed_bitwise_equal=unequal>0,readout_FP32_threshold=source_cfg['score_tolerance'],
        source_cache_sha256=report['raw_sha256'],release_sha256=freeze['release_sha256'],source_sha256=digest(__file__),
        new_images=0,full_model_forwards=0,cached_FP32_canonical_readout_instances=2*len(h),new_AGA_calls=0,
        gradients=0,new_defense_releases=0,partner_or_test_or_confirmation_used=False,wall_seconds=time.monotonic()-start,
        original_paper_or_attack_novelty_claim=False,real_merging_or_recovery_accuracy_evaluated=False,
        scope='Fixed-prefix/all-hidden-input full normalized visual-feature equivalence only, rank>=2/epsilon>0. Not classification-only preservation, changed-prefix OB/AB, empirical task0.5 necessity or a universal no-defense theorem. Matching all merger inputs would require a new frozen protocol and matched controls.'))


def class_faces(folder):
    """All fixed unit-gap class faces, using weights only; not image witnesses."""
    from itertools import combinations
    root=Path(folder);output=root/'class_face_limit.json';assert not output.exists();start=time.monotonic();torch.set_num_threads(4)
    freeze=json.loads((root/'before_images.json').read_text());report=json.loads((root/'results.json').read_text())
    assert digest(root/'point.pt')==freeze['point_sha256'] and digest(root/'source.pt')==report['raw_sha256']
    point=torch.load(root/'point.pt',map_location='cpu');head=torch.load(root/'source.pt',map_location='cpu')['head_weight'].numpy()
    gamma,beta,p=[point[k].numpy() for k in ['gamma','beta','projection']];d=len(gamma);k=len(head);t=head-head.mean(0)
    L=gamma[:,None]*p;L-=L.mean(0);M=L@t.T;c=(beta@p)@t.T
    U,s,V=np.linalg.svd(M,full_matrices=False);tol=np.finfo(float).eps*max(M.shape)*s[0];keep=s>tol;assert int(keep.sum())==k-1
    pairs=list(combinations(range(k),2));q=np.full((len(pairs),k),-2/k)
    for n,(i,j) in enumerate(pairs):q[n,i]+=1;q[n,j]+=1
    u=(((q-c)@V[keep].T)/s[keep])@U[:,keep].T
    # Independent QR on K-1 class differences, not the SVD pseudoinverse.
    A=M[:,:-1]-M[:,-1,None];Q,R=np.linalg.qr(A,mode='reduced');rhs=(q-c)[:,:-1]-(q-c)[:,-1,None]
    qr=np.linalg.solve(R.T,rhs.T).T@Q.T;error=float(abs(qr-u).max());assert error<1e-8
    residual=float(abs(u@M+c-q).max());assert residual<1e-8
    norms=np.linalg.norm(u,axis=1);covered=norms<np.sqrt(d)*(1-1e-8);rows=[];reconstruction=0.
    for n,(i,j) in enumerate(pairs):
        gap=None
        if covered[n]:
            h=np.sqrt(point['eps']/(1-norms[n]**2/d))*u[n];hc=h-h.mean();actual=hc/np.sqrt(np.mean(hc*hc)+point['eps'])
            reconstruction=max(reconstruction,float(abs(actual-u[n]).max()));scores=actual@M+c
            others=[a for a in range(k) if a not in [i,j]];gap=float(min(scores[i],scores[j])-scores[others].max());assert gap>1-1e-8
        rows.append(dict(class_pair=[i,j],normalized_hidden_norm=float(norms[n]),inside_LN_open_ball=bool(covered[n]),strict_gap_to_other_classes=gap))
    assert reconstruction<1e-8
    save_json(output,dict(status='weight_only_class_face_conditions_checked',classes=k,task_affine_rank=int(keep.sum()),class_pairs=len(pairs),
        covered_class_pairs=int(covered.sum()),complete_adjacency_condition_passed=bool(covered.all()),rows=rows,
        maximum_hidden_norm_fraction=float(norms.max()/np.sqrt(d)),QR_SVD_witness_error=error,affine_score_residual=residual,
        LN_inverse_reconstruction_error=reconstruction,source_sha256=digest(__file__),point_sha256=freeze['point_sha256'],
        head_source_cache_sha256=report['raw_sha256'],new_images=0,full_model_forwards=0,new_candidates=0,gradients=0,wall_seconds=time.monotonic()-start,
        scope='Synthetic normalized hidden-state adjacency witnesses from original weights only, not MNIST images or task counterexamples. Conditional fixed-prefix/all-hidden-input classifier rigidity, not finite-task0.5 or changed-prefix OB/AB impossibility. No alternate gap scale, rank or witness optimization.'))


@torch.no_grad()
def run(protocol, full=False, verify=False):
    start = time.monotonic()
    cfg, frozen, checked = inputs(protocol)
    root = Path(cfg['output']); tasks = sorted({t for case in cfg['cases'] for t in case.values()})
    if not full and not verify:
        root.mkdir(parents=True, exist_ok=False); (root/'artifacts').mkdir()
        save_json(root/'protocol.json', cfg); records = {}
        if cfg.get('frozen_recovery_reference'):
            ref=cfg['frozen_recovery_reference']; assert digest(ref['path'])==ref['sha256']
            previous=json.loads(Path(ref['path']).read_text()); assert previous['constructed_before_images']
            records=previous['records']
        for task in ([] if records else tasks):
            own = checked(task+'/finetuned.pt')
            for name in ['Unprotected', 'Phase1_'+task]:
                model = own if name == 'Unprotected' else apply(own, frozen['defenses'][name]['release'])
                clean, geometry = canonicalize(model)
                ref = root/'artifacts'/(task+'_'+name+'.pt')
                torch.save({'model.visual.ln_post.bias': clean.model.visual.ln_post.bias.cpu()}, ref)
                records[task+'/'+name] = dict(path=str(ref), sha256=digest(ref), **geometry)
        save_json(root/'freeze.json', dict(records=records, constructed_before_images=True,
            protocol_sha256=digest(root/'protocol.json'), attack_data=0, defense_updates=0))
    else:
        assert cfg == json.loads((root/'protocol.json').read_text())
        if full: assert json.loads((root/'pilot/validation.json').read_text())['status'] == 'passed'
    freeze = json.loads((root/'freeze.json').read_text())
    assert freeze['constructed_before_images'] and digest(root/'protocol.json') == freeze['protocol_sha256']
    out = root/('full' if full else 'pilot')
    if not verify: out.mkdir(exist_ok=False)
    raw = torch.load(out/'predictions.pt', map_location='cpu', weights_only=False) if verify else {}
    rows = json.loads((out/'results.json').read_text()) if verify else []
    if verify: assert digest(out/'predictions.pt') == json.loads((out/'execution.json').read_text())['predictions_sha256']
    controls, replays, maximum_error = [], 0, 0.
    anchor = checked('zeroshot.pt')
    for ci, case in enumerate(cfg['cases']):
        target, partner = case['target'], case['partner']
        own, peer = [checked(t+'/finetuned.pt') for t in (target, partner)]
        heads = {t: checked('head_'+t+'.pt') for t in (target, partner)}
        data = {t: dataset(t, 'data', own.val_preprocess) for t in heads}
        for t in data: assert len(data[t]) == frozen['tasks'][t]['test_n']
        clean_peer = canonicalize(peer, True)[0] if verify else apply(peer, freeze['records'][partner+'/Unprotected'])
        for name in ['Unprotected', 'Phase1_'+target]:
            release = own if name == 'Unprotected' else apply(own, frozen['defenses'][name]['release'])
            clean = canonicalize(release, True)[0] if verify else apply(release, freeze['records'][target+'/'+name])
            fusion = explicit_merge if verify else merge
            groups = [('Standalone', 'Protected expert', clean, [target]),
                ('TA', 'Protected expert', fusion(anchor, clean, peer, .5), list(heads)),
                ('TA', 'All experts', fusion(anchor, clean, clean_peer, .5), list(heads))]
            baseline = {} if name == 'Unprotected' else baseline
            for method, scope, model, evaluated in groups:
                for task in evaluated:
                    ids = list(range(len(data[task]))) if full else np.linspace(0,len(data[task])-1,cfg['pilot_n'],dtype=int).tolist()
                    key = '/'.join((str(ci),name,method,scope,task))
                    if verify:
                        record = raw[key]; logits = record['logits']; labels = official_labels(task)[ids]
                        np.testing.assert_array_equal(labels,record['target'].numpy())
                        row = next(r for r in rows if r['key']==key); prediction = logits.numpy().argmax(1)
                        assert int((prediction==labels).sum()) == row['correct']
                        control = raw['/'.join((str(ci),'Unprotected',method,scope,task))]['logits'].numpy().argmax(1)
                        assert int((prediction!=control).sum()) == row['changed_predictions']
                        assert int(((control==labels)&(prediction!=labels)).sum()) == row['paired']['lost']
                        assert int(((control!=labels)&(prediction==labels)).sum()) == row['paired']['gained']
                        for c in row['per_class']:
                            mask = labels==c['class_id']; assert int(mask.sum())==c['n'] and int(((prediction==labels)&mask).sum())==c['correct']
                        for pos in np.linspace(0,len(ids)-1,cfg['replay_n'],dtype=int):
                            lo=int(pos)//128*128; hi=min(lo+128,len(ids))
                            x=torch.stack([data[task][i][0] for i in ids[lo:hi]]).cuda()
                            model.cuda(); heads[task].cuda(); actual=heads[task](model(x)).cpu()[int(pos)-lo]; model.cpu(); heads[task].cpu()
                            err=float((actual-logits[int(pos)]).abs().max()); maximum_error=max(maximum_error,err)
                            assert err<=cfg['score_tolerance'] and int(actual.argmax())==int(prediction[int(pos)]); replays+=1
                    else:
                        scores, labels = [], []
                        metric=evaluate(model,heads[task],make_loader(data[task],ids),'cuda',logit_sink=scores.append,prediction_sink=lambda p,y:labels.append(y))
                        scores, labels=torch.cat(scores),torch.cat(labels)
                        np.testing.assert_array_equal(labels.numpy(),official_labels(task)[ids]); assert torch.isfinite(scores).all()
                        if name=='Unprotected': baseline[method,scope,task]=scores
                        rows.append(dict(key=key,target=target,partner=partner,defense=name,merger=method,attack_scope=scope,
                            recovery='Published-projection LN-bias canonicalization',dataset=task,role='target' if task==target else 'partner',
                            **row_metrics(scores,labels,baseline[method,scope,task],frozen['tasks'][task]['classes']),evaluation_seconds=metric['evaluation_seconds']))
                        raw[key]=dict(logits=scores,target=labels,indices=ids)
                        save_json(out/'results.json',rows); torch.save(raw,out/'predictions.pt')
                        if not full and method=='Standalone':
                            before=[]; evaluate(release,heads[task],make_loader(data[task],ids),'cuda',logit_sink=before.append)
                            err=float((scores-torch.cat(before)).abs().max()); controls.append(dict(case=ci,defense=name,max_score_error=err))
                            save_json(out/'function_controls.json',controls); assert err<=cfg['score_tolerance']
            del groups,release,clean
    assert len(rows)==len(cfg['cases'])*10
    if verify:
        save_json(out/'validation.json',dict(status='passed',NumPy_labels_classes_pairs=True,NumPy_lstsq_bias_reconstruction=True,
            explicit_TA=True,selected_score_replays=replays,maximum_score_error=maximum_error,predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__)))
    else:
        save_json(out/'execution.json',dict(completed=True,rows=len(rows),wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),
            defense_updates=0,gradients=0,test_parameter_selection=False,full=full,predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol',default='configs/bias_canonicalization.json')
    ap.add_argument('--full',action='store_true'); ap.add_argument('--verify',action='store_true')
    mode=ap.add_mutually_exclusive_group();mode.add_argument('--affine-family');mode.add_argument('--class-faces');args=ap.parse_args()
    if args.affine_family:affine_family(args.affine_family)
    elif args.class_faces:class_faces(args.class_faces)
    else:run(args.protocol,args.full,args.verify)
