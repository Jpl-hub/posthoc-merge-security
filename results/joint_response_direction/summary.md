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


## Parameter distance versus source activation

| Branch / inputs | Relative parameter distance | MLP output change (RMS %) | Activation energy / uniform-complement mean |
|---|---:|---:|---:|
| B10 all tokens | 0.039248 | 0.303353 | 0.00709173 |
| B10 CLS | 0.039248 | 0.084877 | 0.00119874 |
| B11 CLS | 0.498457 | 0.043667 | 8.0648e-06 |

All rows use the 4,096 original construction images. B10 includes separate all-token and CLS statistics; B11 uses CLS. MLP output here means the linear Bz contribution before the unchanged bias. The uniform-complement column is an analytic second-moment reference, with no sampled or optimized replacement direction.
The joint linear response energy is 0.000666904 times the sum of isolated-branch energies: coupling exists, while the dominant B11 change is weakly excited on these inputs.
This diagnoses the frozen source solution only. It does not prove the cause of every unseen-input violation or establish attack efficacy. No rewhitening, new weights, threshold changes or additional TA/AGA evaluations follow from this table.
