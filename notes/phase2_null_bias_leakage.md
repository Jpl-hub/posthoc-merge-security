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

Pending the one frozen run and independent verification.
