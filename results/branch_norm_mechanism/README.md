# A finite B-branch mechanism check, not a released defense

The proposed skip/branch variance mechanism was not supported. No model was published, no fresh confirmation was consumed, and no native TA/AGA result is claimed for this check. The original experts and all earlier releases remain unchanged.

The calculation uses the last-block residual `h=s+Bz`, with fixed upstream computation, biases, LayerNorm and visual projection. Let `K` be an orthonormal basis of the kernel of `[1, diag(gamma) P]`. Its dimension is255. Consider only orthogonal changes to the B-output coordinates inside this kernel. This is a restricted family of known orthogonal transformations, not a claimed invention.

## What was actually checked

The fixed prototype reflects the whole kernel: `B'=(I-2KK^T)B`. Its quantized relative B distance is0.3947733144 (ideal family maximum0.3947733144649071). This is already the farthest point in this family:

$$\frac{\|B'-B\|_F}{\|B\|_F}\leq\frac{2\|K^TB\|_F}{\|B\|_F}=0.3947733144.$$

The maximum follows from `||(U-I)K^T B|| <= 2||K^T B||` for every orthogonal `U`, and is achieved by `U=-I`. Thus the0.5 limit was not binding here. Trying more reflection ranks or angles would not fix an unfinished distance maximization.

For `N` spanning any reflected subspace, the exact real-arithmetic chord identity is

$$\sigma(h_\lambda)^2-\sigma(h)^2=
\frac{-4\lambda\langle N^Ts,N^TBz\rangle-4\lambda(1-\lambda)\|N^TBz\|^2}{768}.$$

On the512 original construction contexts, the full-kernel endpoint has task drift0.03760796 and the half chord0.01206938, both with zero changed predictions. However, sigma **increases** by6.33–15.26% at the endpoint and1.55–4.26% at the half chord. The intended contraction did not occur. These are cached last-block counterfactuals, not a complete native model or independent task confirmation.

The fixed covariance certificates were inconclusive: the relevant two matrices have eigenvalue ranges[-7.8730,0.7035] and[-10.0080,2.2603]. They do not establish a whole-family impossibility. Separately, an exact scalar class-numerator calculation shows that arbitrary variance contraction alone cannot change any of these512 source decisions. The checked B chord retains a strictly positive margin lower bound3.80844351, including its quantized linear readout leakage.

## A whole-family bound in one public source environment

To go beyond the expert's self chord, a second preregistered, read-only calculation uses the existing public-base/author-AGA-expert TA0.5 cache. It uses construction training contexts, not a future partner, test images or independent confirmation. It neither selects a transformation nor evaluates a new defense.

For this fixed merged last block, put `w=K^T B_AGA z_f` and `v=K^T h_f`. Any kernel reflection changes its hidden state by `-Kq`, where `q=Pi w`. Every such q lies on the sphere with center `w/2` and radius `||w||/2`. The same sphere also contains `q=(w-Uw)/2` for **every orthogonal U** in the kernel, so the bound covers the whole orthogonal family, not just the fixed full reflection.

The author MLP permutation is determined by the unchanged A rows; left multiplication of B therefore commutes with that fixed column permutation. This is why the family can be followed through AGA into this particular merged tail. No other attention, upstream or readout parameters are edited.

Decompose the centered hidden state into its kernel and orthogonal-complement coordinates. The exact extrema of its possible norm give an interval `[sigma_min,sigma_max]`. For a teacher/competitor pair, let `t=K^T diag(gamma_f)P_f(H_y-H_k)` and let `a,b` be its original linear and bias numerator terms. Then

$$N'_{yk}\geq a-\frac{w^Tt+\|w\|\|t\|}{2}
+\begin{cases}\sigma_{\min}b,&b\geq0,\\\sigma_{\max}b,&b<0.\end{cases}$$

Allowing a different worst transformation for every image and class, and ignoring the parameter budget, only enlarges the family. The lower bound is nevertheless positive for every teacher/competitor pair on all512 contexts: minimum0.4954739471. Independent Torch reconstruction of the complete LayerNorm/QuickGELU/readout agrees with the NumPy calculation to1.18e-13.

**Scope:** all kernel-orthogonal B-output changes are certified unable to alter those512 decisions in that one fixed float64 source environment. This is not an unknown-partner guarantee, a universal FP32 result, a task0.5 proof, or a rejection criterion imposed on other defenses. It does not cover nonorthogonal changes, edits outside the visual kernel, changes to other modules, or O/B's nonzero class-numerator compensation.

The result explains a limitation of this particular family. The next unresolved requirement is a finite, native joint compensation that preserves the original task while allowing a signed class-numerator mismatch after recovery and fusion. No such new construction is claimed here.

## Reproduction and retained evidence

- [Fixed counterfactual](results.json) and [independent norm, scalar and chord verification](independent_validation.json).
- [Fixed public-environment family bound](public_proxy_bound/results.json) and [independent complete reconstruction](public_proxy_bound/independent_validation.json).
- Existing entry point: `experiments/aga_response.py --branch-norm`; configs `branch_norm_mechanism.json` and `branch_norm_bound.json`. Add `--bound` for the public-environment calculation, or `--verify <existing output directory>` for its independent audit.
- Producer source versions are pinned in each config. Use the matching historical Git source or the archived `executed_source.py` on3090, not a later HEAD with a different source hash. Original directories must not be overwritten.
- Source prototype preregistration: local `d5bc46e`, GitHub `1807bffc`; family-bound preregistration: local `abf0dfd`. The latter was local before execution, rather than an externally timestamped GitHub preregistration.
- Heavy caches stay on3090 under `outputs/phase2_branch_norm_mechanism_20261019` and `outputs/phase2_branch_proxy_bound_20261019`. Raw SHA256 values are recorded in the linked JSON. The first verifier's missing-import failure log is retained; only that import was repaired.
- Prototype5.27s and512 repeated cached last-block contexts; bound2.39s and no new model forwards. Audit time is additional. Zero new images, gradients, expert updates or released models.
