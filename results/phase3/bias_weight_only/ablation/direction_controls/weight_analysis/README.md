# Class choice and storage rounding

Run `python -m experiment.analyze_bias --directions --output outputs/class_geometry` after preparing the frozen checkpoints and seven original NBP publications. This analysis uses weights only: no images, new publications or parameter selection.

`results.json` contains every class index for two formulas: the NBP minimum-norm least-squares direction and direct class projection into the same nullspace. All748 class indices have positive proxy class gaps for both formulas. This establishes feasibility of the proxy magnitude rule, not test accuracy for these1,496 directions. The original seven releases are unchanged. The independent NumPy least-squares comparison is included per task.

The21 precision records round both original/released biases and the expert projection, then compute the residual in FP64. They are not low-precision forward evaluations. Cars has nonfinite released FP16 bias values. FP32 is the evaluated deployment precision.

`protocol.json` records the executed source SHA and the source protocols. Checkpoint, head and original-publication SHA values are included per task.

## Minimum norm for the same proxy condition

Run `python -m experiment.analyze_bias --minimum-norm --output outputs/minimum_norm`. This keeps each original release's largest proxy-offset class, coefficient0.5 and safety factor2. It changes only the mathematical question: how small can a kernel perturbation be while satisfying the same class-margin inequalities?

With nullspace basis N, class-difference readouts q_j and unprotected lower-bound deficits c_j, the program minimizes ||a||²/2 subject to0.5 q_jᵀNa ≥2 max(c_j,0). Nonnegative dual variables give a lower bound on the objective; feasibility and the primal–dual gap are reported alongside the solver status. This is a standard convex program, not a new defense claim.

| Task | Original update norm | Minimum norm estimate | Ratio |
| --- | ---: | ---: | ---: |
| SUN397 | 284725.70 | 12905.68 | 0.0453 |
| Cars | 933555.30 | 11250.81 | 0.0121 |
| RESISC45 | 35088.11 | 8759.77 | 0.2497 |
| SVHN | 11220.33 | 7772.77 | 0.6927 |
| GTSRB | 43171.78 | 14579.04 | 0.3377 |
| MNIST | 12752.00 | 6329.63 | 0.4964 |
| DTD | 26156.66 | 10829.09 | 0.4140 |

All seven problems pass the recorded feasibility and optimality tolerances. Maximum absolute scaled-constraint violation is4.00e-15, relative primal–dual gap magnitude4.28e-15, and relative native-kernel residual4.90e-16. These are weight-only results: no optimized checkpoint was saved, no images were evaluated, and no main release was replaced.

The existing multi-class least-squares fit can require much larger parameters than its own margin condition needs. This supports simplifying the construction; it does not establish lower-precision inference quality, behavior with unknown partners, or resistance to public bias projection. Every solution remains in the same removable kernel.
