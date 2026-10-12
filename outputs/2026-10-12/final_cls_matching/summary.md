# Final CLS matching: negative result

Source freeze: `b292a0928add2481dbc3b6c15577a441f150557a`. Target-only MNIST,1024 training calibration images and256 disjoint confirmation images. No labels, future partners, gradient optimization or response reconstruction. The setup repair reproduced both saved FP32 releases exactly. Eight-expert TA0.2; full MNIST/SVHN tests only.

| Setting | Native MNIST | Fusion MNIST | Fusion SVHN |
|---|---:|---:|---:|
| Original, direct | 99.6900% | 94.0000% | 74.4238% |
| Decoy, direct | 99.6900% | 94.0000% | 74.2163% |
| Original, author AGA | 99.6800% | 93.2300% | 73.9014% |
| Decoy, author AGA | 99.6800% | 93.0100% | 72.9833% |
| Coherent control, author AGA | 99.6800% | 93.2400% | 73.8245% |
| Shared public c_fc reset | 99.6800% | 93.9300% | 74.3278% |

The defense is ineffective: direct fusion remains94.00% MNIST. Original AGA changes MNIST93.23%→93.01% (−0.22pp) and SVHN73.9014%→72.9833% (−0.9181pp). Raw paired p-values are0.003838 and4.94e−39, without multiplicity correction; detectable differences are not a large practical collapse. Public c_fc reset reaches93.93% MNIST, only0.07pp below original direct fusion. No further recovery attacks are run; they are not claimed to pass.

## What the measurements establish

- At the same final layer and on identical1024 images, all-token PCA needs764 dimensions to preserve99.99% input energy; CLS-only PCA needs505. The readout uses only CLS after the final tokenwise MLP. Two distinct patch-output interventions leave matched-batch features bitwise unchanged. This causal simplification does not apply to earlier layers.
- After excluding the exact LayerNorm null direction, the CLS complement has262 dimensions. Every one of3072 neuron rows is assigned the intended wrong anchor by the unchanged author solver. Coherent control changes no row identities. Input-row matching evidence can therefore be changed while maintaining actual native MNIST99.69%; targeting an estimator is already known, including DDO, and no novelty is claimed.
- Confirmation preactivation change is3.82% of native response,6.49% on real direct-fusion MNIST inputs and8.60% on direct-fusion SVHN inputs. Normalized feature changes are0.0207/0.0616/0.0793. Fusion inputs enter the local perturbation equation `delta_z=beta*delta_W*x+beta*delta_b` as predicted. The perturbation does act on merged computation, but in this module and construction it does not create enough classification errors. Wrong matching or larger response energy alone cannot establish protection.
- The principal-solver geometry is validated, not universal security. Output columns are unchanged and another matching rule could use them. We do not infer that all final-layer edits or all complement defenses are ineffective.

## Verification and next step

Fifteen fresh full-test runs plus six validated baseline aliases. Independent validation checked all21 counts, official labels, paired changes and32-image prefixes, release reconstruction, FP32 span leakage, analytic amplitude bounds and actual author assignments. Separate c_fc prehook capture plus NumPy/SciPy recomputed the CLS covariance and projector: relative covariance error3.72e−16, projector error1.02e−10. Code and input/source/checkpoint fingerprints are retained. All raw weights/certificates/predictions stay on the server.

Stop this candidate without changing strength, rank or permutation. In already profiled task-critical modules, the next diagnostic should measure signed classification-margin effects along target-only fusion proxies, and confirm them on disjoint target training images. Geometry or feature RMS must be treated as proxies until actual native/direct/original-AGA measurements support them. The old48-direction beta-path family remains closed; this note does not authorize a sweep or claim a new successful construction. Necessary adaptive attacks follow a useful core candidate.

Exact paper Table6 reproduction and the original EuroSAT split remain unresolved. This experiment evaluates two tasks of an eight-expert merger and reports no eight-task mean.
