# Complement matching: negative result

Source freeze: `6ef8a7807719c6da2c27e7e613aefa80d00d46ca`. Target-only MNIST construction, 256 training calibration and 256 disjoint training confirmation images; no labels, future partners, gradients or response fits. Eight-expert TA0.2; two official test sets measured. Not an exact paper Table6 reproduction.

| Setting | Native MNIST | Fusion MNIST | Fusion SVHN |
|---|---:|---:|---:|
| Original, direct | 99.69% | 94.00% | 74.42% |
| Decoy, direct | 99.69% | 94.00% | 74.42% |
| Original, author AGA | 99.68% | 93.23% | 73.90% |
| Decoy, author AGA | 99.68% | 93.24% | 73.89% |
| Coherent control, author AGA | 99.68% | 93.24% | 73.91% |
| Shared public c_fc reset | 99.66% | 92.07% | 74.13% |

Exact numerators and denominators, matched changes and tests are in results.json. The tiny changes do not show effective protection. Coherent control is measured under all attacks; it is not assumed to have identical perturbation energy.

The 99.99%-energy all-token PCA spans have ranks758/761/762. After excluding the LayerNorm exact null direction, remaining dimensions are9/6/5. Only one complete pair at layer6 is feasible under this fixed projected-anchor row construction. Author AGA actually swaps those two rows; layers7/8 stay unchanged. This isolates insufficient matching coverage, not failed implementation or a universal impossibility of complement protection. No amplitude or rank rescue.

Independent validation passed21 full prediction counts and21 prefix replays, official labels, actual release reconstruction, independent row-assignment and amplitude bounds, public-reset identity and raw/source fingerprints. The candidate is closed.

A follow-up is justified by a different causal subspace definition: in CLIP's final residual block, MLP patch-token outputs are not read by the image head; only the CLS token enters ln_post and projection. Thus preserving all final-block token variance imposes unnecessary constraints. A separate fixed final-CLS candidate will test this fact, with larger calibration and independent confirmation; no conclusion yet about its generalization or utility.
