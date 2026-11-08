Implemented recovery settings. All procedures can read the published weights. AGA and MLP replacement use the public base; coefficient fitting and selection use the public base and expert checkpoints. Bias projection needs only the published bias and projection. The changed parameters describe recovery itself; subsequent fusion also combines final normalization and projection parameters.

| Procedure | Images/task | Labels | Changed parameters | Budget |
| --- | --- | --- | --- | --- |
| AGA | 0 | No | Attention and MLP blocks | One author recovery pass |
| MLP replacement | 0 | No | Last MLP output matrix and bias | One replacement after AGA |
| ProDistill | 64 | No | Element-wise block coefficients | 100 steps/block |
| Coefficient selection | 256 | Yes | Shared TA coefficient | 10 values, 0.1--1.0 |
| Bias projection | 0 | No | Final LayerNorm bias | One least-squares solve |
