# Finite projection/bias compensation: fixed-point result

One max-min conic solve in the task-SVD complement, frozen expert and source training data. This is a data-assisted source experiment; it is separate from the existing three-task Defense2 releases.

| Evaluation | Images | Before (%) | After (%) | Predictions changed | Relative class-gap change |
|---|---:|---:|---:|---:|---:|
| Construction train | 1024 | 99.80 | 99.80 | 0 | 0.247617 |
| New train confirmation | 512 | 99.80 | 99.80 | 0 | 0.433406 |
| Complete MNIST test | 10000 | 99.69 | 99.64 | 12 | 72.989474 |

On the 1024 construction images, merging the author-AGA target with the public base gave 99.71% before the edit and 9.67% after it. This is a construction environment, not a held-out partner or an attack-test result. The source minimum class-0 gap is 4.369236; all 1024 public-control predictions are class 0.

The genuinely unread 512-image training confirmation passed. Complete test accuracy lost 0.05 percentage points: 8 formerly correct images became wrong, 3 formerly wrong images became correct, and 12 total predictions changed. However, 62 images exceeded the original relative-gap limit 0.5. The largest value was 72.989474: an original class-9 versus class-5 gap of 0.0237379 changed by 1.7326174. That image still predicted class 9. Independent NumPy reproduced every prediction and all four standalone counts; maximum score error was 1.52e-5, below the unchanged 5e-4 numerical tolerance.

**Decision:** preserve this exact point as a failed complete-task check under the registered protocol. Do not change its amplitude, class, source, budget or direction using these test results. Real-partner TA and original AGA were not started after the failure. This does not invalidate the previous frozen three-task method or its original-AGA accuracies of 9.80% / 2.39% / 6.45%.

**Mechanism:** the finite compensation preserves the source mean and bounds all source class gaps, not every possible target activation. Source and fresh training checks did not cover all near-boundary test responses. The failure is real, rather than a NumPy comparator issue or merely a large parameter distance. The accuracy loss and the stricter gap failure are reported separately.

**Scale and cost:** logical joint distance 2.828424296, within the separately preregistered 2sqrt(2) range and outside the historical 0.5 comparison. Task limit 0.5 was retained. One standard Clarabel solve, 25 iterations, 46.91 seconds. Source optimization used an already frozen cache (0 new full image forwards); native confirmation/test added 21024 image forwards in 20.05 seconds. No expert training or new AGA calls. Solver-run total wall time is unknown because the final JSON boolean write failed; point, scores and failure log were retained, and a read-only audit rebuilt the report without solving again.

Reproduction source stages and hashes are indexed in `../task_directed_pairing/frozen_source_history.json`; raw prefixes, edits, conic problem and scores remain on the 3090. GitHub connection is unavailable, so current commits are local and are not described as published.
