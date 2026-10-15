# Output-null bias transport: one mechanism test

## Frozen hypothesis

Decision-normal Gap-Jet is closed. This test changes neither its point nor its controller. It tests a different algebraic mechanism: a **last-MLP input bias translation in the exact kernel of the original MLP output matrix**. A and B are unchanged. This is not last-LayerNorm bias protection, a neuron reflection/permutation, scaling or Gaussian noise.

The expert checkpoint is frozen. Let z=Ax+a and let phi(z)=z sigmoid(1.702z). Write

$$\phi(z)=z/2+e(z),\qquad e(-z)=e(z).$$

For one common bias displacement d with B_e d=0,

$$\delta h_e=[e(z+d)-e(z)]B_e^T.$$

The linear component is identically zero **for every input**, but the even nonlinear component generally is not. This is not a function-preserving symmetry and does not establish task utility. Under a shared environment B_f and publication coefficient lambda,

$$\delta h_f=\underbrace{\lambda B_fd/2}_{\ell_f}
+\underbrace{[e(z_f+\lambda d)-e(z_f)]B_f^T}_{v_f}.$$

This is an exact finite identity. In TA(.5), B_f=(B_e+B_p)/2 before publication, hence ell_f=B_p d/8 when B_e d=0. With a public-base interpolation B_p=B_0, B_0d=-DeltaB d. It is the small task update's projected action, not the full anchor norm, that controls leakage. This gives a conditional source-only falsification target. Unknown B_p d, upstream states, readout and the nonlinear term remain uncontrolled.

## Unique analytic construction

Use only the existing legal4096 construction cache, original expert and public base. No old protection or exposed confirmation responses are read. Set P_N=I-B_e^+B_e; Q is the complete centered class-numerator column basis. Compute

$$G=Q^T(B_e-B_0)P_N.$$

Choose the leading right singular vector v of G, with its largest-magnitude coordinate positive (deterministic sign). Define

$$d=(0.5)(1-10^{-6})\|a_e\|v.$$

This maximizes the projected task-B leakage norm within the chosen output kernel and distance ball; it does **not** maximize actual classification damage. The actual FP32 bias is materialized and distance rechecked≤.5. No amplitude, seed, layer or rank sweep, nonlinear optimization or continuation of a closed point. One finite direction only. The small export reserve is numerical, not a larger budget.

## Signed classification and class adversary

For centered actual scores s'=alpha s+r, compute all competitors' signed crossing excess

$$E_{ij}=r_j-r_y-\alpha(s_y-s_j).$$

Except ties, max_j E_ij>0 is exactly an argmax change. All per-class counts are recorded; residual RMS or large hidden changes are not used as classification evidence. Raw score components are not added through nonlinear normalization. The hidden linear/even split is checked separately.

The linear leakage ell is constant across inputs. For each source-reference teacher class, average the negative **linear numerator** change divided by each image's original teacher/runner-up numerator gap. This gives ten common-parameter rows L_c. The restricted linear game is

$$\max_{\|d\|\le R, B_ed=0}\min_{\pi\in\Delta_C}\sum_c\pi_cL_cd.$$

For any feasible class weights pi,

$$\text{value}\le R\|P_NL^T\pi\|_2.$$

A linear program minimizes the infinity norm of the dual vector; the resulting two-norm gives a **valid upper bound**, not a claim to have solved the full nonlinear game. Directed class differences can cancel under a class opponent, even when individual classes have large leakage. The certificate excludes the nonlinear even and LN-variance terms and does not rule out all multi-class protection. Source finite utility and actual joint crossings decide this mechanism.

The environmental family behind the leakage identity is the shared matrix segment B_eta=B_e-eta DeltaB, acting on all pictures. No independent per-image environment is introduced. Actual finite checking uses **one complete public-base .5 interpolation model** from the pinned source cache, not a hybrid of states or arbitrary fake partners. Neither this segment nor this interpolation certifies unknown-partner coverage. No outer unknown-environment minimax is claimed.

## Multi-class task neighborhood

Retain the established source natural/O-B task probes. Separately define x(u)=LN2(r)+V_A^T diag(RMS)u, ||u||≤.01, with V_A the32 leading right singular directions of the actual matrix A_e-A_0. SVD specifies coordinates, not sufficient task function. This is an activation-space neighborhood, not a claim that every point is image-realizable.

For two lowest source teacher-gap contexts per class, form the nine competitor-gap derivatives in these coordinates. Their SVD gives a complete orthonormal frame of functional normals and tangents (no assumed rank). Evaluate both signs of **all32** frame directions with full finite MLP, LayerNorm and CLIP normalization. These1280 probes are not a continuous certificate.

An additional uniform bound applies to the entire32-dimensional ball. For QuickGELU, |phi'|≤1.5 and |phi''|≤1.5 follow from |q|sigmoid(q)(1-sigmoid(q))≤1/e and slope parameter1.702. Let F=A V_A^T diag(RMS), M1=1.5||B|| ||F|| and M2=1.5||B|| ||F||². Then ||Dh||≤M1, ||D²h||≤M2.

For the final normalized feature t(h)=hL+sigma(h)c, set K=||L||+||c||/sqrt(width). On the ball, sigma_min=sigma(h0)-M1 rho/sqrt(width), t_min=||t(h0)||-K M1 rho. Only if both are positive, a pair weight w has valid bounds

$$G_h\le\|w\|K/t_{min},\qquad
H_h\le\|w\|[6K^2/t_{min}^2+\|c\|/(d\,\sigma_{min}t_{min})],$$
$$H_u\le H_hM_1^2+G_hM_2.$$

For pair-gap error e=g_release-g_teacher, Taylor's theorem therefore gives

$$|e(u)|\le |e(0)|+\rho\|De(0)\|+\rho^2(H_{u,e}+H_{u,r})/2.$$

The corresponding teacher-gap lower bound uses its exact center derivative and H_u. A positive lower bound plus error upper≤.5 times that lower bound certifies the original relative task budget on that ball. A loose/failed bound is **not** a witnessed task violation; finite violations are recorded separately. No sampled Hessian is advertised as a uniform proof, no radius is shrunk to make certificates pass.

## Decision and preservation

The one mechanism-specific continuation rule is full finite source/probe and neighborhood drift≤.5, no natural prediction changes, and at least16/512 actual reference changes covering five classes. This is a preregistered cost gate for this particular leakage hypothesis, not a universal necessary condition or a protection claim. If absent, no publication optimization, fresh confirmation, TA/AGA or parameter adjustment follows. If supported, freeze the point first, obtain genuinely new train confirmation, then run true TA/AGA exactly once.

All scores, derivatives, directions, class dual and prototype are saved. Independent NumPy must reconstruct the kernel projection without the Torch SVD, all finite scores and crossing counts, frame derivatives and continuous-bound calculation. Original/prototype FP32 predictions on16 source pictures are checked by complete models. Only3090; phase1 and old releases unchanged.

## Prior work and novelty boundary

GELU even-linear identities and parameter symmetries are known; see [Identifiable Equivariant Networks](https://arxiv.org/html/2601.21645v2). [PaRaMS](https://arxiv.org/html/2503.07661v1) already studies distance with functionality preservation; task-matrix SVD comes from [TSV](https://arxiv.org/html/2412.00081v2), and [ESM](https://arxiv.org/html/2602.20208v1) studies activation-aware task subspaces. [GRAIL](https://proceedings.mlr.press/v328/tang26a.html) also shows post-hoc linear reconstruction compensation is established tooling. This diagnostic does not claim any of these tools as innovation. [AGA/ARF](https://arxiv.org/html/2606.30360v1) remain attack versus tuning-stage defense; neither ARF nor Table6 is reproduced here.

No novelty, robust protection or paper-ready result is established before actual evidence. The specific proposed interaction is output-kernel bias transport and its finite class-normal leakage; its usefulness is falsifiable by the fixed checks above.

## Results

**The one fixed mechanism is independently falsified. No release, fresh confirmation, true TA or authorAGA was run. Old O/B remains99.71% independent,78.77% TA,77.85% AGA.**

|Check|Verified result|
|---|---:|
|Edited tensor|Last MLP input bias only; A/B unchanged|
|Normalized edited-tensor distance|.4999994999≤.5|
|Raw Euclidean displacement / original bias norm|12.30661 /24.61324|
|Source4096 natural plus2560 task probes|0/6656 changed; worst relative drift.1703500|
|New multi-class normal/tangent probes|0/1280 changed; worst drift.1704066|
|Complete public .5 interpolation|**0/512 changed**, all10 classes unchanged|
|Source-reference all-pair maximum numerator damage|**.0745751**, crossing requires>1|
|Constant-linear class-game dual upper|2.57e−16, independently recounted|
|Continuous32D task balls certified|0/20; sufficient bound inconclusive|
|Exact kernel error / FP32 materialization error|7.83e−14 /2.19e−8 per unit bias|

The distance metric is the unchanged sum of squared **per-edited-tensor** relative displacements, here one bias tensor. Comparing its.5 directly with old O/B's.299 does not compare whole-model Euclidean distances or defense strengths. This prototype has no functionally empty linear effect in the source, but QuickGELU's even response is real; independent finite checks, not the norm, determine its relevance.

Full source/actual-environment normalized scores, every class, signed crossing excess, projected task matrix, FP32 bias, class dual weights, all1280 neighborhood responses and sufficient bounds are retained. Independent NumPy used a row-Gram kernel projector and9x9 eigenproblem rather than the original output-matrix SVD. Maximum finite score error5.68e−14, score-Jacobian error1.69e−14, directional finite-difference error8.23e−10. Original Torch frame **orientation** was replayed as input data, while its orthogonality,9-dimensional functional normal space,23-dimensional tangent space and finite responses were independently checked; no claim of an independent orientation algorithm.

Original/prototype32 full-model predictions match. Construction16.26s, independent verification15.02s, peak allocation2.169GB on3090; no gradient or parameter optimization updates, no optimization restart. A verifier interface error (NumPy suffix returns activations rather than a context dictionary) was fixed only in the verifier; original failure log/commit589c043 are retained. The primary direction and its scores were not rerun or altered. The unmeasured failed verification/reporting overhead is not included in those timed subtotals.

Index-only audit excludes12,621 registered independent confirmation/holdout indices: zero source or reference overlap. Generic historical calibration-evaluation overlaps are also zero. The direction itself depends only on original/public weights; source training responses generate diagnostics/probes, never a replacement direction. New confirmation samples were not drawn. Phase1, frozen O/B and all previous Gap-Jet artifacts are hash-verified unchanged.

### Why norm-maximal leakage fails here

Let N be the exact class-ranking numerator. For each original teacher y and competitor j,

$$\delta N_y-\delta N_j=(\ell_y-\ell_j)+(v_y-v_j)+\delta\sigma(b_y-b_j).$$

The relative signed sum must exceed1 in the adverse direction to cross. At the frozen joint-worst competitor in each of512 reference inputs, mean contributions were linear−.01067, even+.01923, variance+.0000933. Linear and even signs oppose in88.67% of inputs. Maximum **joint** damage is only.07458; no component norm implies crossing. The decomposition identity error is5.10e−14. These are post-freeze source attributions and never used to choose a sign, coefficient or new point.

At the actual source, the corresponding all-pair maximum damage is.09926 and original predictions remain intact. Output-kernel membership makes the source **linear** part vanish, but does not provide the directional nonlinear amplification needed for this protection mechanism.

### Exact obstruction for a constant linear leakage

The numeric class dual has a symbolic explanation. For class c, define positive a_cj as the class-average inverse teacher/runner-up numerator gap when runner-up is j. Let a_c=sum_j a_cj and T_cj=a_cj/a_c. Every row of T sums to1. The average linear-damage row is a common right linear map applied to

$$a_cigl(M_c-\sum_jT_{cj}M_jigr).$$

A finite row-stochastic T has a stationary distribution q≥0. Taking pi_c proportional to q_c/a_c gives

$$\sum_c\pi_c a_c(M_c-\sum_jT_{cj}M_j)=0.$$

Therefore the max-over-d/min-over-class value for this **constant linear component** is exactly0 (d=0 attains0). A norm-maximal d cannot make every class's average runner-up gap strictly worse. This does not forbid damage to a subset of classes, and does not apply to input-dependent even or variance terms. The actual no-crossing result for the sole nonlinear point is separate empirical evidence, not a theorem that every bias translation or post-hoc defense fails.

### Why finite neighborhood success is not a certificate

All20 contexts have9 independent class normals and23 task-coordinate tangents, and all1280 signed-frame points satisfy the finite task criterion. The uniform bound proves nonzero variance and feature denominator on every radius.01 ball; nevertheless its worst-case Hessian remainder cannot certify the.5 relative-gap budget on any of them. At these centers the hidden displacement bound is1.06048 and the maximal bound on feature-change/norm is.42424. The bound is sufficient and conservative; 0/20 certificates is **not**20 task violations. The radius was not reduced and no samples were added to manufacture a pass.

## Next hypothesis, not a launched method

A common constant shift has the class-cycle obstruction above. An input-conditioned linear escape would instead have E x with B_e E=0, for example E=P_N C V_A in task-SVD coordinates. Its shared-environment linear term is lambda B_f E x/2; class-conditioned x statistics prevent the previous stationary-distribution cancellation from applying automatically. The nonlinear source term remains [e(z+E x)-e(z)]B_e^T and may still make preservation impossible.

Before proposing optimization, the required evidence is a **finite shared-matrix** class-adversary separation with adverse sign and adequate margin, together with an actual multi-class task-neighborhood error bound. A projected gradient or positive singular value would not suffice. This is a new unverified hypothesis, not permission to rescale the closed bias point, mechanically swap sites, assume unknown-partner coverage, or invoke the old O/B/A-B optimizers. No input-conditioned matrix, fresh confirmation or real fusion result is claimed here.
