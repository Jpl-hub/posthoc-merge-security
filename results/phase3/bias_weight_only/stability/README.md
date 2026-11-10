# NBP class-choice stability

Seven tasks, seeds 0–4, original constructor, no image feedback. Every control was saved before evaluation. The original seed-20261014 releases remain unchanged. RESISC45 seeds 1 and 4 both select class 40 and are both reported.

Each task contains 30 complete-test rows: five matching unprotected controls plus five rows per NBP draw (standalone, TA target/companion, AGA+TA target/companion). The 210 rows include 175 new protected evaluations. The original raw scores and small bias releases remain on the experiment server at the paths recorded in execution.json and freeze.json.

All seven independent label recounts and 3,360 original-context score replays passed. The maximum replay logit error is 1.33514404296875e-5, below the unchanged 5e-4 tolerance. This is not an accuracy confidence interval over independently trained experts.

Reproduce the evaluations with `configs/nbp_stability.json`; export tables with `python -m experiment.report`. The generated `../tables/class_stability.md`, `.tex`, and `.json` report the aggregate and every draw. MNIST AGA+TA accuracy ranges from 9.58% to 30.20%; all other tasks stay below 10%. Maximum standalone loss is under 0.03 percentage points. These results do not address public bias projection.
