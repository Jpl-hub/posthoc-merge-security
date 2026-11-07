# Direction controls

The frozen NBP release is compared at its original update norm with a simple projected class contrast and five fixed random-nullspace draws. Seed20261106 is reused from the completed component controls; seeds20261107–20261110 are new. No draw is selected or used to alter NBP.

The simple direction is `N @ N.T @ ((P0 + Pe)/2) @ (w_requested - mean(W))`, normalized to the frozen NBP update norm. The requested class is drawn with the original construction seed20261014 before evaluation. It uses no multi-class least-squares solve.

MNIST/SVHN/DTD use the same fixed companions, datasets, TA0.5 and target-only author AGA as the existing component controls. All utility failures and all seed results will be reported. Matching unprotected logits are reused by SHA and parameter fingerprint. Each new direction has full native target and both merged datasets, with original128-context verification. These are ablations, not new main-method versions.
