# Fixed joint-response direction: task gate

One frozen B10/B11 direction, parameter distance 0.4999995. Original task budget 0.5; no change after confirmation or tests.

| Evaluation set | Inputs | Worst relative gap drift | Prediction changes | Task gate |
|---|---:|---:|---:|---|
| Construction + finite probes | 6656 | 7.93762e-05 | 0 | Pass |
| Fresh train + finite probes | 3072 | 0.0313674 | 0 | Pass |
| Complete MNIST test | 10000 | 1.06848 | 0 | Fail |

Complete-test accuracy is 99.69% before and after protection; all predictions match. 2 images exceed the functional budget.
The independently replayed witness (test index 8527, class 4 versus 9) increases the gap from 0.04598331 to 0.09511566. Increased confidence still violates the original absolute relative-gap constraint.

**TA and original AGA were not run:** the preregistered native-task gate stopped this fixed candidate. This is neither a feasible-defense result nor evidence about its attack resistance. No radius, rank, layer, output axis or weights were changed to repair it.

## Coupling and limits

On the construction inputs, B10 alone has worst drift 0.00568254, B11 alone 0.00569164, and their fixed combination 7.93762e-05. Each has zero construction prediction changes; this does not establish harmful isolated branches.
The minimum empirical joint response and finite task-SVD probes support source compensation, not unseen-input preservation. Standard SVD, Rayleigh optimization and Gauss-Newton geometry are not novelty claims. Historically exposed tests are not a first blind evaluation.

Construction: 152.82s; confirmation: 53.50s; stopped full-task evaluation: 21.46s. Independent audits and their costs remain separate. No optimizer, expert updates or new attack fits.
