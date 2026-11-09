# Final LayerNorm and projection: scope of a local change

This is a weight-only feasibility check, not a new defense or an accuracy experiment. The original seven NBP releases remain unchanged.

With the prefix fixed, write the embedding as `z diag(gamma) P + beta^T P`, where `z` is centered. Its affine map is completely determined by

```text
A = (I - 11^T/d) diag(gamma) P
c = beta^T P.
```

If A has full column rank, the same map is realized by gain=1, projection=A, and bias solving `A^T bias=c` with minimum norm. These are standard affine folding and projection operations. Any changes limited to these final three tensors that preserve the complete embedding for every centered normalized state have the same A,c and the same folded representation in exact arithmetic. This conclusion does not cover approximate task preservation or a changed prefix.

All seven original experts have rank512 for A. The independent CPU NumPy QR solve agrees with the GPU SVD solve within the declared1e-9 relative tolerance. Sixteen deterministic hidden-vector checks use the real LayerNorm equation and epsilon; they are not image evaluation or an empirical accuracy guarantee.

| Task | Rank | NumPy relative error | Relative embedding error | Remaining bias difference / original NBP update |
| --- | ---: | ---: | ---: | ---: |
| SUN397 | 512 | 2.358e-13 | 6.276e-15 | 2.249e-08 |
| Cars | 512 | 1.068e-13 | 5.274e-15 | 2.084e-08 |
| RESISC45 | 512 | 2.701e-13 | 6.421e-15 | 2.290e-08 |
| SVHN | 512 | 4.030e-13 | 5.224e-15 | 2.316e-08 |
| GTSRB | 512 | 2.487e-13 | 6.470e-15 | 2.211e-08 |
| MNIST | 512 | 6.048e-13 | 4.990e-15 | 2.222e-08 |
| DTD | 512 | 9.061e-14 | 6.256e-15 | 2.425e-08 |

The remaining differences for the stored FP32 NBP tensors reflect their nonzero projection residual. No full-model predictions or recovery accuracy are measured here. No images, partner weights, classifier, optimizer or new release are used. The source is the existing `experiment/analyze_bias.py --tail-equivalence` entry; its exact SHA and frozen input protocols are in protocol.json.

## Noise retained by the original bias projection

A bias increment in col(P) survives projection onto col(P), but changes the standalone embedding by a nonzero constant v. For the public-base TA0.5 proxy its offset is `v (I + P^dagger P0)/4`. We check the maximum ratio of the proxy offset norm to the native offset norm over all v. This is a source-only operator norm; it says nothing by itself about class errors, standalone utility, or unknown partners. No candidate direction, magnitude or released model is selected.

| Task | Maximum proxy/native embedding-offset ratio | Independent NumPy relative error |
| --- | ---: | ---: |
| SUN397 | 5.6626 | 1.912e-13 |
| Cars | 7.8713 | 1.419e-13 |
| RESISC45 | 4.3826 | 1.972e-13 |
| SVHN | 4.7493 | 2.320e-13 |
| GTSRB | 13.1301 | 7.663e-14 |
| MNIST | 8.9985 | 1.073e-13 |
| DTD | 4.4261 | 1.529e-13 |

These finite gains justify examining the utility trade-off of nonzero native offsets; they do not establish a successful improvement. The scalar ratio must not be reported as classification protection. The existing `--projection-gain` analysis uses only each target and the public base.

## Class-relevant bias changes

The embedding ratios above include directions that change no class decision. For a fixed zero-bias classifier, let `Wc = W - mean(W)` and `M = P Wc^T`. Replacing any final bias by its orthogonal projection onto col(M) preserves all centered class-score numerators. Feature normalization multiplies these numerators by a positive scalar, so the class decision is unchanged whenever the normalized embedding is defined. Unlike projection onto col(P), this operation also uses the published classifier. It need not preserve probabilities, the embedding, or tasks using other classifiers.

For bias changes that survive this class-aware projection, write `M = U S V^T` using its nonzero singular values and `delta_bias = U S^-1 y`. Native centered class-score offsets have norm `||y||`. In the public-base TA0.5 proxy, the corresponding offset is

```text
y^T S^-1 U^T (P + P0) Wc^T / 4.
```

Its spectral norm is the largest possible proxy/native ratio within this surviving subspace. Both sides are unnormalized score numerators, not normalized logits or accuracies. This calculation uses no partner, images, chosen amplitude or new release.

| Task | Rank of M | Maximum proxy/native class-offset ratio |
| --- | ---: | ---: |
| SUN397 | 396 | 1.0111 |
| Cars | 195 | 0.7151 |
| RESISC45 | 44 | 0.5123 |
| SVHN | 9 | 0.5558 |
| GTSRB | 42 | 0.6775 |
| MNIST | 9 | 0.5404 |
| DTD | 46 | 0.5078 |

Independent NumPy pseudoinverse calculations agree within 2.84e-14 relative error. These values do not provide evidence for large class-offset amplification by changing only the surviving bias component. They do not rule out a task-margin-dependent utility trade-off or changes to the prefix. No image accuracy or new defense success is claimed. `class_gain.json` contains the measurements; reproduce with `python -m experiment.analyze_bias --class-gain --output outputs/class_gain`.
