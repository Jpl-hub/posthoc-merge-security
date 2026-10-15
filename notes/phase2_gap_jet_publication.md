# Decision-normal boundary retraction

## Hypothesis and frozen scope

The full-hidden Hermite release preserved fresh task function but did not protect true merging (MNIST TA99.36%,AGA99.29%). Its boundary equations constrain 768 hidden coordinates per value/derivative. The new hypothesis is that retaining only the decision-normal value and derivative leaves additional finite freedom for compensation mismatch. This is a hypothesis about a new chart, not evidence that its remaining freedom transfers to unknown partners.

One algorithm, MNIST first; source expert unchanged; only last-block O/B publication edits. Same allowed construction4096 and20 activation chords, never previously exposed confirmation values. One fixed mechanism check, then a separate **from-zero** publication, maximum16 proposals/600seconds. Distance/task budgets remain .5/.5, source reserve remains .25. Fresh confirmation512 seed20261026, dynamic exclusion of all registered prior confirmation/holdout indices. No restart, redraw, repair or test/partner selection.

## Nonlinear equation and solvability

Let final hidden vector be h, width d=768, sigma(h)=sqrt(||h-mean(h)||²/d+eps). Let L=diag(gamma)P with column means removed, c=beta P. For class pair i,j, w=head_i-head_j:

$$g_{ij}(h)=\frac{a^Th+b\,\sigma(h)}{\|h^TL+\sigma(h)c\|_2},\qquad a=Lw,\quad b=cw.$$

The positive denominator includes final CLIP feature normalization. Set v=a/||a|| and solve the **numerator**, not a hidden-feature fixed point:

$$F(s)=a^T(h_f+sv)+b\,\sigma(h_f+sv)=0.$$

Since |d sigma(h_f+sv)/ds|≤1/sqrt(d),

$$F'(s)\geq\|a\|-|b|/\sqrt d=:m.$$

If m>0 (checked for every pair), F is strictly increasing and surjective: a unique root exists globally on this line. Epsilon prevents zero variance. This certifies the scalar root, **not** the whole classifier, denominator nondegeneracy, finite task budget, or unknown-input/partner behavior.

For q0=var(h_f)+eps, q1=mean(center(h_f)v), q2=mean(v²), A=a^T h_f and B=||a||, set T=B²-b²q2>0. The root with the correct unsquared sign is

$$s=\frac{-AB+b^2q_1-b\sqrt{B^2q_0-2ABq_1+A^2q_2+b^2(q_1^2-q_2q_0)}}{T}.$$

The implementation checks its discriminant, actual finite gap and numerical conditioning; an independent verifier uses a bracket |F(0)|/m+1 and Brent's algorithm instead of this formula.

At h*=h_f+sv, write n=a+b center(h*)/(d sigma(h*)). Then the normalized gap derivative along a chord is n^T h'/||h*L+sigma(h*)c||. Correct only along v:

$$h'^*=h'_f+v\frac{g'_{teacher}-n^T h'_f/\|t(h^*)\|}{n^Tv/\|t(h^*)\|}.$$

No assertion that this is the minimum Euclidean correction in every direction; it is a uniquely solvable decision-normal line correction. 'Minimal' refers to two scalar functional constraints per boundary, not a proved minimal globally sufficient task description.

## Shared parameter realization

O update: deltaO=||O|| U C V^T using the previously source-response-selected task-SVD directions. Free deltaB/||B||=Q_class L_free is applied **before** boundary correction. Let z,z' be complete finite QuickGELU/LN2 activations after O; F stacks their row-normalized values and derivatives. A correction K_c solves

$$F K_c^T=Y-F K_f^T,\qquad K=K_f+K_c.$$

Canonical36 rows remove only two reversed-path duplicates; all40 original equations and all20 pair jets are checked. F must have full row rank/condition≤1e6. Torch uses the row Gram solve; independent NumPy uses SVD least-squares. This produces one common B matrix, not per-input inference parameters. It preserves the requested jets, while allowing tangential hidden changes induced by K_f. It does not enforce all other class gaps at the root. Finite all-class natural/probe/chord constraints remain mandatory.

## Finite game and acceptance

Distance D²=||C||²+||K||_F² equals the actual normalized O/B matrix distance. Optimize the unchanged source-only finite class/environment soft-min objective plus .01D². The adversary ranges over the20 entries (two fixed source-realizable shared-prefix environments × ten classes), with entropy regularization .01. These two environments are a finite subgame, **not** all shared parameter perturbations or a partner distribution.

The frozen trust-region controller computes actual classification gradients, projects against active all-class task-margin Jacobians, and accepts only positive finite gain with predicted/actual agreement≥.1, all6816 source task inputs≤.25 and D≤.5. It also checks exact boundary jets. Never infer success from parameter distance, proxy count or hidden change. The one terminal source-selected point requires independent numerical replay and one fresh train confirmation (≤.5) before TA/authorAGA.

Original checkpoints remain frozen; gradients optimize publication coefficients only. Explicit AD calls and runtime are recorded; this is not zero-gradient or zero-training-cost protection.

## Relation to prior work

- [PaRaMS](https://arxiv.org/html/2503.07661v1),§4: MLP permutations and attention scaling preserve functionality while separating parameters. Distance plus function preservation is already prior art; no novelty claim for that objective or known symmetries.
- [TSV](https://arxiv.org/html/2412.00081v2),§3–4: decomposes per-layer task matrices and studies singular interference/whitening. Our use of task SVD is a structural tool; it is not a functional guarantee or a new SVD method.
- [ESM](https://arxiv.org/html/2602.20208v1),§3: activation-shift-aware decomposition and polarized scaling improve merging. Response-aware subspaces already have precedent; neither those results nor ours certify unknown-partner coverage.
- [AGA/ARF](https://arxiv.org/html/2606.30360v1),§3,5: AGA is recovery; ARF is repulsive **fine-tuning followed by invertible attention protection**. This work does not reproduce ARF/Table6. Our publication coefficient optimization differs from task fine-tuning but still has optimization cost.

Possible contribution under investigation: decision-normal nonlinear boundary retraction retaining finite compensation freedom. Root solvers, derivative distillation, SVD and trust regions are known tools. A publishable protection contribution has not been established by this construction.

## Results

The fixed non-releasable mechanism seed passed the scalar/jet checks: monotonic derivative lower bound3.4442, canonical feature-row condition90.55, independent Brent/SVD coefficient difference1.86e−16, gap error7.39e−13, slope error2.27e−13, objective directional derivative difference2.92e−8. Hidden response changes by55.32% while its selected pair jets remain preserved: this establishes finite representation freedom, not task preservation.

That seed has worst finite task drift1.0143 and two natural changed predictions, so it is **not releasable and not rescaled**. The separate candidate optimization starts from zero. No new protection success claimed.


The sole from-zero construction accepted9 of16 proposals (56.52s, peak2.299GB,1189 explicit reverse-AD calls). Terminal D=.1635149; all6816 source task inputs have worst relative gap drift.2315751, no natural prediction changes. Canonical row condition87.59, boundary hidden change24.90%. Source-realizable environment changes3/512 in each, involving classes2,4,5: weak empirical interference, not defense success. All16 acceptance decisions were independently replayed with Brent roots/SVD least-squares (score error1.05e−12, coefficient error1.34e−15, objective direction error2.44e−12, task J-step error7.98e−8).

Fresh confirmation512 seed20261026:511/512 correct before and after, zero changed predictions; all3072 natural/probe inputs worst drift.4259195≤.5. FP32 D=.1635149, source drift.2315751. NumPy and1536 original/release full-model forwards match. All registered historical exposed confirmations excluded. No redraw or repair. Frozen candidate SHA256 `7f748688e95843a27f0b86e67b3f3d50d4b191c1948cde8b4c2ed63e473b3bd9`.

Construction/protocol source commit482fdab, sole point and confirmation plan e124d13, confirmed weight/evaluation pin56d3986. Server-only e124d13 archive SHA256 `6a1691fa7272dca819e8095ebb52e7ddcfd4259d1b3e8da3e30906cfb27c89d9`. Actual test evaluation follows once; historical MNIST/SVHN test sets are not a new blind test.


## Verified real evaluation and closure

All17 official count groups, class statistics, paired counts and272 sampled full-model TA/AGA predictions were independently verified. No evaluation optimization or candidate selection occurred.

|Metric|Unprotected|Gap-jet release|Frozen best O/B|
|---|---:|---:|---:|
|MNIST independent|9969/10000 (99.69%)|9970/10000 (99.70%)|99.71%|
|MNIST TA(.5)|9937/10000 (99.37%)|9934/10000 (99.34%)|78.77%|
|MNIST authorAGA|9932/10000 (99.32%)|9931/10000 (99.31%)|77.85%|
|SVHN TA(.5)|24079/26032 (92.4977%)|23932/26032 (91.9330%)|See frozen O/B evidence|
|SVHN authorAGA|23890/26032 (91.7717%)|23656/26032 (90.8728%)|See frozen O/B evidence|

MNIST TA:25changed,13lost/10gained,paired exactp=.67764. AGA:17changed,8lost/7gained,p=1.0. This is **no target defense**, not a near-breakthrough. SVHN loss is auxiliary partner damage. SVHN independent25371/26032=97.4608% remains unchanged. No extension to protecting DTD/SVHN or additional partner selection.

|MNIST class|Test n|Unprotected AGA correct|Release AGA correct|
|---|---:|---:|---:|
|0|980|975|974|
|1|1135|1133|1134|
|2|1032|1024|1024|
|3|1010|1009|1008|
|4|982|977|976|
|5|892|885|884|
|6|958|942|946|
|7|1028|1017|1014|
|8|974|971|971|
|9|1009|999|1000|

Known-location replacement of O/B with the public base removes both edits exactly; independently checked against the historical matched cleaned control. No new adaptive attack, new recovery-accuracy evaluation or anti-cleaning claim. All original checkpoints, phase1 releases/archive, best O/B and historical failed points remain hash-verified unchanged. Weights, source archive, raw predictions and all trial snapshots remain on3090; only code/light evidence synchronized.

The fixed route is closed. Its actual coefficient distance is.1635, not a proof that all directions up to.5 are ineffective. The finite path ended at its predeclared16 proposals; no larger budget, extra steps, changed seed or confirmation repair was attempted. Forward evaluation203.21s excludes AGA parameter-recovery/load time, which was not separately measured. Source construction56.52s, independent source replay71.98s, fresh confirmation24.98s; no zero-gradient claim.

## What the negative result establishes

1. Full-hidden equality is **not required** for a numerically stable source boundary chart: the unique scalar root has a verified positive derivative bound, and the same shared B matrix preserves every requested normalized pair value/derivative. This resolves a conditional solvability issue; it does not solve protection.
2. At the same final C/L coordinates, the full-hidden chart has D=.18454, source worst.25711 and2/512 changes per environment. The gap chart has D=.16351, source worst.23158 and3/512 changes. Reducing hidden constraints increases measured non-scaling freedom, but does not create broad margin crossings. This post-freeze source ablation is not a second release and is never selected for true evaluation.
3. The precise decision relation is centered scores s'=alpha*s+r, where alpha=<s,s'>/||s||². For every teacher class y and competitor j:

   $$g'_{yj}=lpha g_{yj}+r_y-r_j.$$

   With alpha>0, crossing requires r_j-r_y>alpha*g_yj. Source median alpha=.9730 and residualRMS=.2669 give no crossings; fixed source environments have median alpha=.8535/.8345 and residualRMS=.8667/.7742 but only3 crossings each. RMS and hidden distance do not certify a sufficiently adverse, samplewise class-normal residual. The decomposition was independently checked to3.50e−15, with margin-crossing counts exactly matching argmax counts.
4. Boundary chord jets are not a sufficient task description. Exact counterexample: teacher gap g(t,u)=t, released gap g'(t,u)=t+u. Along u=0, every value and tangential derivative matches, yet relative gap error |u|/|t| is unbounded near t=0. Thus a proof on fixed chords cannot be promoted to transverse or unknown-fusion coverage. This counterexample disproves that inference, not the existence of all possible post-hoc defenses.

## Next falsifiable question

Can task preservation be expressed over a source-defined **multi-class boundary neighborhood**, including independent decision-normal derivatives, while leaving a shared-parameter direction whose finite joint response has the adverse sign and amplitude required by the identity above? Before a new publication optimization, specify a reachable neighborhood, bound the normal remainder relative to teacher gaps, and check actual shared-matrix feasibility. A continuous representation ball or arbitrary environment list does not establish unknown-partner coverage. SVD provides directions; it does not prove sufficiency.

No such neighborhood certificate, candidate or cross-task defense is established here. The next study must use source-only data, exclude the newly exposed seed20261026 confirmation responses, start with a distinct testable mechanism, and preserve both.5 budgets. Do not recycle this point with extra steps or tune against the observed TA/AGA.
