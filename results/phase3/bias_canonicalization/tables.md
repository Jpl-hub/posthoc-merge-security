# Targeted LN-bias recovery

Each fused cell reports protected-task / partner-task accuracy (%), TA(0.5).

| Protected task | Partner | Recovery scope | Defense 1: no recovery | Defense 1: bias recovery | Unprotected: same recovery |
|---|---|---|---:|---:|---:|
| MNIST | SVHN | Protected expert | 9.80 / 6.70 | 99.37 / 92.50 | 99.37 / 92.50 |
| MNIST | SVHN | All experts | 9.80 / 6.70 | 99.37 / 92.50 | 99.37 / 92.50 |
| MNIST | DTD | Protected expert | 9.80 / 2.13 | 99.45 / 69.26 | 99.45 / 69.26 |
| MNIST | DTD | All experts | 9.80 / 2.13 | 99.45 / 69.26 | 99.45 / 69.26 |

The recovery uses only the published visual projection and final LN bias. It projects the bias onto the projection column space; no clean expert, public base, images or partner enters recovery construction. The base and partner are used only for subsequent TA evaluation.

This is an architecture-aware supplementary adaptive diagnostic, not original AGA or a claimed author attack reproduction. Defense weights are unchanged. Both recovery scopes and their matching unprotected controls are retained; tests were previously observed.

Scope: the two MNIST cases passed the unchanged 0.0005 function-control threshold on RTX3090. The four-case pilot stopped at DTD (0.00082493); SVHN as protected task was not reached. That failure remains recorded. Earlier RTX4090 pilot verification is separate. No threshold or recovery coefficient was changed.

Native full-test comparison is separate from replay verification: all 10000 MNIST predictions remain unchanged, but the maximum protected-model score difference before/after cleanup is 0.00069427, above the pilot/replay tolerance0.0005. Worst relative margin drift is0.00112328. The ideal real-arithmetic identity does not imply exact FP32 score agreement. See native_function_audit.json; the original tolerance is unchanged.

Full evaluation: 286.89s. Independent full-model selected-score replays: 320, maximum error 0. Replay cost is additional; concurrent workloads mean these wall times are not an isolated speed benchmark.

Every recovered Defense 1 prediction equals its matching recovered unprotected control, including standalone MNIST. Against the frozen no-recovery predictions, MNIST gains 8962 correct predictions and loses 5 with SVHN; it gains 8966 and loses 1 with DTD. Both recovery scopes give the same counts. Partner net gains are 22335/26032 (SVHN) and 1262/1880 (DTD).

Per-class counts and comparisons to matching recovered unprotected controls are in results.json. Recovery gains against the frozen no-recovery models, exact paired p values and both raw-score fingerprints are in recovery_pairs.json. Raw scores and six frozen recovery edits stay on the server. No all-attack or cross-protected-task robustness claim.
