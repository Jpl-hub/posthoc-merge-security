# Labelled calibration for the TA coefficient

This is an additional data-assisted recovery check, not a change to NBP. The attacker has256 labelled official training examples per task. Examples may have been used to train the original expert; all test sets remain separate. NBP construction uses no images. The same grid, calibration indices and selection rules apply to unprotected, PaRaMS and NBP models.

AGA first processes only the protected expert. Both task updates then use a shared TA coefficient from0.1 to1.0 in steps of0.1. The primary rule maximizes the unweighted mean of target and companion calibration accuracy; the secondary rule maximizes target accuracy. Ties favor the coefficient closest to0.5, then the smaller coefficient. `selected.json` and `selected_test_protocol.json` were frozen before test evaluation. Test scores do not enter selection.

The calibration stage completed30 cases:270 dataset rows and4,320 original-context score replays. Eight selected test cases completed48 full dataset rows and768 replays, all with zero replay error. The18 selected method/objective choices were independently recovered from `calibration_counts.csv`; see `selection_validation.json`. Full calibration per-class records and logits remain in the original server output. Each CSV count comes from its recorded result SHA in `calibration_validation.json`.

With the primary rule, NBP target accuracy is11.35% on MNIST,68.21% on SVHN and2.13% on DTD. SVHN therefore partially recovers relative to the fixed0.5 result,6.70%. Matching unprotected models selected by the same primary rule give99.36%,91.97%,71.12%. This result does not support protection at every fusion coefficient.

`readout_analysis.json` uses the already frozen releases and both expert projections after publication. It checks all grid readouts without new image forwards. It is explanatory analysis, not construction or coefficient selection. The second expert has never been used to construct the NBP release.

Reproduction uses the existing main-table evaluator with `calibration_protocol.json`, followed by its fixed selection rule and `selected_test_protocol.json`. Exact training indices are in `calibration_indices.json`. The three PaRaMS AGA edit files were regenerated once using the pinned author kernel because their old serialized files were unavailable; their execution records are included in `aligned/`. No defense parameters were changed. The initial import failure occurred before image evaluation, and its log is retained.
