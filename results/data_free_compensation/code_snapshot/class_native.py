"""Frozen-point native evaluation with historical-score reuse and new train draw."""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path('/root/autodl-tmp/merge_security_20261009')
sys.path.insert(0, str(ROOT))
from _impl.data import dataset, make_loader, official_labels
from _impl.evaluate import evaluate
from _impl.io import digest, save_json
from _impl.metrics import row_metrics, task_change
from _impl.models import initialize, load_checkpoint
OUT = ROOT / 'outputs/frozen_classifier_20261025/class_coupling/native'


def numpy_check(before, after, labels, row):
    a, b = before.numpy(), after.numpy()
    p, q = a.argmax(1), b.argmax(1)
    good, new = p == labels, q == labels
    lost, gained = int((good & ~new).sum()), int((~good & new).sum())
    assert int(new.sum()) == row['correct']
    assert int((p != q).sum()) == row['changed_predictions']
    assert (lost, gained) == (row['paired']['lost'], row['paired']['gained'])
    differences = new.astype(float) - good.astype(float)
    half = 1.96*differences.std(ddof=1)/len(labels)**.5
    np.testing.assert_allclose([100*(differences.mean()-half),100*(differences.mean()+half)], row['paired_accuracy_delta_95ci_pp'], atol=1e-12)
    n = lost+gained
    logp = min(0., math.log(2)+float(np.logaddexp.reduce([math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2) for k in range(min(lost,gained)+1)]))) if n else 0.
    assert abs(logp/math.log(10)-row['mcnemar_log10_p']) < 1e-8
    for c in row['per_class']:
        m = labels == c['class_id']
        assert (int(m.sum()),int(new[m].sum()),int(good[m].sum())) == (c['n'],c['correct'],c['original_correct'])
    return dict(official_labels=True, per_class=True, paired_CI=True, scalar_lgamma_exact_McNemar=True)


def main():
    initialize()
    plan = json.loads((ROOT/'results/data_free_compensation/joint.json').read_text())['classifier_coupling_plan']
    assert digest(__file__) == plan['native_evaluation_source_sha256']
    pub = ROOT/'outputs/frozen_classifier_20261025/class_coupling/point.pt'
    assert digest(pub) == plan['point_sha256']
    OUT.mkdir(exist_ok=False)
    # Conservative metadata exclusion;not a claim that every historical raw tensor was indexed.
    excluded, manifest, seen = set(), [], set()
    def collect(obj):
        if isinstance(obj, list):
            if obj and all(isinstance(x,int) and not isinstance(x,bool) for x in obj):
                return {x for x in obj if 0 <= x < 60000}
            found = set()
            for x in obj: found.update(collect(x))
            return found
        if isinstance(obj,dict):
            found = set()
            for x in obj.values(): found.update(collect(x))
            return found
        return set()
    for top in [ROOT/'outputs',ROOT/'results']:
        for folder, dirs, names in os.walk(top, followlinks=True):
            st = os.stat(folder); key = (st.st_dev, st.st_ino)
            if key in seen: dirs[:] = []; continue
            seen.add(key)
            for name in names:
                if not name.endswith('.json'): continue
                path = Path(folder)/name
                if path.stat().st_size > 16000000: continue
                try: ids = collect(json.loads(path.read_text()))
                except (ValueError,UnicodeError): continue
                if ids:
                    excluded.update(ids)
                    manifest.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),integer_values_excluded=len(ids)))
    available = np.array(sorted(set(range(60000))-excluded))
    assert len(available) >= 256, ('Insufficient metadata-disjoint training indices',len(available))
    indices = sorted(np.random.default_rng(2026102504).choice(available,256,replace=False).tolist())
    save_json(OUT/'training_indices.json',dict(indices=indices,seed=2026102504,excluded_count=len(excluded),eligible_count=len(available),historical_json_records=manifest,scope='New draw excluding discovered historical JSON integer-index lists;all historical raw PT files not exhaustively indexed,not a global-first-blind claim. Current construction uses0real images.'))
    save_json(OUT/'protocol.json',dict(plan=plan,training_indices_sha256=digest(OUT/'training_indices.json'),full_accuracy_loss_limit_pp=.5,training_is_descriptive=True,source_sha256=digest(__file__)))
    expert, head = load_checkpoint('MNIST/finetuned.pt'), load_checkpoint('head_MNIST.pt')
    model = copy.deepcopy(expert)
    model.load_state_dict(expert.state_dict() | torch.load(pub,map_location='cpu',weights_only=False),strict=True)
    baseline = ROOT/'outputs/alignment_compensation_20261024/native/full.pt'
    assert digest(baseline) == '2166c96219ba1e774a661a954b8444fe0bcddf45fbd676fe00560bd5e024f0be'
    old = torch.load(baseline,map_location='cpu',weights_only=False)
    rows = {}; start = time.monotonic(); forwards = 0; replays = 0; maxerr = 0.
    for stage,training,ids in [('training',True,indices),('full',False,list(range(10000)))]:
        data = dataset('MNIST','data',expert.val_preprocess,training)
        labels = official_labels('MNIST',training)[ids]
        if training:
            logits=[];evaluate(expert,head,make_loader(data,ids),'cuda',logit_sink=logits.append);before=torch.cat(logits).double();forwards+=len(ids)
        else:
            np.testing.assert_array_equal(old['indices'],ids);np.testing.assert_array_equal(old['labels'],labels);before=old['before']
        logits=[];runtime=evaluate(model,head,make_loader(data,ids),'cuda',logit_sink=logits.append);after=torch.cat(logits).double();forwards+=len(ids)
        row=row_metrics(after,torch.as_tensor(labels),before,10);row['strict_task']=task_change(before.numpy(),after.numpy());row['runtime']=runtime
        row['numpy_validation']=numpy_check(before,after,labels,row)
        torch.save(dict(indices=ids,labels=labels,before=before,after=after),OUT/(stage+'.pt'))
        row['raw_sha256']=digest(OUT/(stage+'.pt'))
        wanted=np.linspace(0,len(ids)-1,16,dtype=int);blocks=sorted(set(wanted//128));context=[i for k in blocks for i in range(k*128,min((k+1)*128,len(ids)))];chosen=[context.index(int(i)) for i in wanted]
        for name,current,expected in [('protected',model,after)]+([('original',expert,before)] if training else []):
            logits=[];evaluate(current,head,make_loader(data,[ids[i] for i in context]),'cuda',logit_sink=logits.append);actual=torch.cat(logits).double()[chosen]
            err=float((actual-expected[wanted]).abs().max());assert err<=5e-4;np.testing.assert_array_equal(actual.argmax(1),expected[wanted].argmax(1));maxerr=max(maxerr,err);forwards+=len(context);replays+=len(wanted)
        rows[stage]=row;save_json(OUT/'partial.json',rows);print(stage,row['correct'],row['n'],row['strict_task'],flush=True)
    result=dict(rows=rows,passed_full_accuracy=100*(int((old['before'].argmax(1).numpy()==old['labels']).sum())-rows['full']['correct'])/10000<=.5,publication_sha256=digest(pub),source_sha256=digest(__file__),source=Path(__file__).read_text(),new_real_image_forwards=forwards,context_score_replays=replays,maximum_context_score_error=maxerr,seconds=time.monotonic()-start,baseline_full_reused_sha256=digest(baseline),training_indices_sha256=digest(OUT/'training_indices.json'))
    save_json(OUT/'results.json',result);(OUT/'partial.json').unlink();print('NATIVE_DONE',result['passed_full_accuracy'],forwards,replays,maxerr,flush=True)


if __name__=='__main__':main()
