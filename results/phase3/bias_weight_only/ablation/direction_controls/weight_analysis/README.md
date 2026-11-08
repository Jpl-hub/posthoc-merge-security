# Class choice and storage rounding

Run `python -m experiment.analyze_bias --directions --output outputs/class_geometry` after preparing the frozen checkpoints and seven original NBP publications. This analysis uses weights only: no images, new publications or parameter selection.

`results.json` contains every class index for two formulas: the NBP minimum-norm least-squares direction and direct class projection into the same nullspace. All748 class indices have positive proxy class gaps for both formulas. This establishes feasibility of the proxy magnitude rule, not test accuracy for these1,496 directions. The original seven releases are unchanged. The independent NumPy least-squares comparison is included per task.

The21 precision records round both original/released biases and the expert projection, then compute the residual in FP64. They are not low-precision forward evaluations. Cars has nonfinite released FP16 bias values. FP32 is the evaluated deployment precision.

`protocol.json` records the executed source SHA and the source protocols. Checkpoint, head and original-publication SHA values are included per task.
