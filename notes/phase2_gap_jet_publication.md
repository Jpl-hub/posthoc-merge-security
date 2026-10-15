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

Pending frozen mechanism check and independent replay. No new protection success claimed.
