"""Paired counts, exact McNemar and full relative class-gap checks."""

import numpy as np
import torch
from scipy.special import gammaln, logsumexp
from scipy.stats import binomtest


def paired_counts(reference, correct):
    lost = int((reference & ~correct).sum())
    gained = int((~reference & correct).sum())
    return dict(
        both_correct=int((reference & correct).sum()),
        lost=lost,
        gained=gained,
        both_wrong=int((~reference & ~correct).sum()),
        loss_pp=100 * (lost - gained) / len(correct),
        exact_mcnemar_p=float(binomtest(lost, lost + gained).pvalue) if lost + gained else 1.0,
    )


def paired_logp(lost, gained):
    n = lost + gained
    if n == 0:
        return 0.0
    k = np.arange(min(lost, gained) + 1)
    return float(
        min(
            0.0,
            np.log(2)
            + logsumexp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
            - n * np.log(2),
        )
        / np.log(10)
    )


def row_metrics(scores, labels, before, classes):
    prediction = scores.argmax(1)
    correct = prediction.eq(labels)
    pair = paired_counts(before.argmax(1).eq(labels), correct)
    # Paired normal interval describes image-sampling uncertainty, not seed variation.
    differences = correct.double() - before.argmax(1).eq(labels).double()
    half = 1.96 * float(differences.std(unbiased=True)) / len(labels) ** 0.5
    return dict(
        n=len(labels),
        correct=int(correct.sum()),
        accuracy_percent=100 * float(correct.double().mean()),
        changed_predictions=int(prediction.ne(before.argmax(1)).sum()),
        paired=pair,
        mcnemar_log10_p=paired_logp(pair["lost"], pair["gained"]),
        paired_accuracy_delta_95ci_pp=[
            100 * (float(differences.mean()) - half),
            100 * (float(differences.mean()) + half),
        ],
        per_class=[
            dict(
                class_id=c,
                n=int(labels.eq(c).sum()),
                correct=int((correct & labels.eq(c)).sum()),
                original_correct=int((before.argmax(1).eq(labels) & labels.eq(c)).sum()),
            )
            for c in range(classes)
        ],
    )


def logical_parameter_distance(edits, original):
    total = 0.0
    for k, v in edits.items():
        a, b = v.double(), original[k].double()
        pieces = (
            zip(a.chunk(3, 0), b.chunk(3, 0))
            if k.endswith(("attn.in_proj_weight", "attn.in_proj_bias"))
            else [(a, b)]
        )
        for changed, before in pieces:
            if float(before.norm()) == 0:
                assert torch.equal(changed, before)
                continue
            total += float((changed - before).norm().square() / before.norm().square())
    return total**0.5


def task_change(before, after):
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matching score matrices required")
    winner = a.argmax(1)
    rows = np.arange(len(a))
    keep = np.arange(a.shape[1])[None, :] != winner[:, None]
    gap = a[rows, winner, None] - a
    if not np.all(gap[keep] > 0):
        raise ValueError("Teacher class tie: relative gap undefined")
    delta = b - a
    return dict(
        n=len(a),
        changed=int(np.sum(b.argmax(1) != winner)),
        R=float(np.max(np.abs((delta[rows, winner, None] - delta)[keep] / gap[keep]))),
    )
