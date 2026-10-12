# Complement matching probe

One target-only, fixed candidate on MNIST MLP layers 6–8. No additional gradient training or response fitting. This is an original-AGA mechanism test, not a claim of new mathematics or broad security.

## Construction

From 256 target training images, estimate centered c_fc input covariance and mean μ. Retain the smallest PCA span with at least 99.99% input energy. Append the exact LayerNorm affine-null direction to the preserved span. Let P be its orthogonal projector and Q=I−P. Public pretrained c_fc rows are normalized using the author's epsilon convention, yielding a_j. Fix π(i)=(i+1536) mod 3072, a union of disjoint swaps, without looking at partners or tests.

For original row w_i set z_i=a_{π(i)}Q. Add δw_i=λ_i z_i and δb_i=−δw_i μ. On μ+range(P), the linear map is unchanged. Outside this affine span its local error is δw_i(x−μ); subsequent nonlinear computations can amplify it. Low PCA tail energy alone does not guarantee accuracy.

For every competing anchor k≠π(i), original AGA prefers the intended wrong anchor when

`<w_i,a_π−a_k> + λ_i <z_i,a_π−a_k> ≥ 0.001 ||w_i||`.

The released row's cosine denominator is positive and common to all competitors, so it cancels in this inequality. Solve the resulting scalar lower and upper bounds once. Add 1% relative slack and keep only feasible complete swap pairs; unselected rows receive zero edits. If all rows in a bijection have unique intended maxima, it is also the unique optimal sum assignment. With partially selected rows the assignment still requires actual verification; we do not assume the global result.

The coherent control uses the same λ_i and a_iQ in place of a_πQ. It tests whether any observed AGA degradation comes from adversarial identity geometry rather than simply adding complement energy. Control energy and utility are measured rather than assumed equal. Bias compensation preserves a mean, not all input responses.

## Frozen measurements

- Standalone MNIST, direct eight-expert TA (coefficient 0.2), and unchanged author AGA on the target before TA. Evaluate full MNIST and SVHN tests for fusion. Keep unprotected and coherent controls under the same attacks.
- Independently replay actual author assignments, held-out training feature changes, exact nonlinear MLP-only permutation equivalence, release parameters, source/checkpoint fingerprints, full prediction counts, official labels and 32-image prefixes.
- After core tests, replace the edited public c_fc matrices and biases. This yields exactly the same state for all presets. It is a limitation test, not a precondition that blocks constructing the candidate.
- Save raw activations, certificates, release edits, assignments and predictions only on the server. Commit code, protocol and compact real results. No strength/rank/sign search or selection from tests.

## Literature and scope

AGA §3.2 and its pinned source use normalized input-row cosine assignment and jointly permute c_fc, bias and c_proj. The method is copied through the repository's existing AST loader, unchanged: https://github.com/krumpguo/secure-merge-attack/tree/fb61429f5e6f3517102d506697a150ac26abbdea . Paper: https://arxiv.org/html/2606.30360v1 . This candidate changes the matching evidence outside a measured target input span rather than relying on an exact hidden permutation alone. No proof of novelty follows from that difference.

NUFILT motivates studying task-dependent representations; its task-vector SVD subspace is not the same object as the activation PCA span used here. We reuse the existing covariance/PCA implementation and do not claim to run NUFILT's full trained method or inherit its theorem. Existing complement experiments show several distinct failures: no signed merging harm despite a response proxy; restoration by original AGA; and inexpensive public replacement. They do not establish that every complement direction has no merging effect. This probe isolates the AGA matching failure mode before further recovery tests.
