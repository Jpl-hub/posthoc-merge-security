# 最新实际融合结果

**新版没有改善防护效果。** MNIST单独使用99.69%，直接融合99.23%，AGA只处理保护模型后融合99.06%，处理两个模型后融合98.93%。不保护时对应融合结果为99.37%、99.30%、99.32%。

这份固定模型仍未通过此前的全部保持检查，原失败结论保留。补测仅把真实融合数字交代清楚，没有改变参数或预算。独立17行／272分数重放已通过。[简明对照表与双方结果](diagnostic/summary.md)。以下保留原主协议的完整技术记录；其中“未运行TA／AGA”指后续补测之前的停止状态。

---

# Joint O/B with variable original-AGA recovery feedback

**Main candidate closed: full-task R budget failed. The original gated evaluation stopped before TA/AGA; a later fixed-model fusion diagnostic was completed at the user’s request.**

This round implemented a joint method experiment,not another fixed-prefix/minimum-B control. C and K start at zero;the objective combines task-outside distance with the signed class response THROUGH variable original-AGA O-column recovery. Native full finite responses and original task-singular probes constrain the same path. Only the frozen target,public base and existing lawful4096 training inputs enter construction. No old protected weights,confirmation,test or peer select parameters. The existing finite tangent/retraction solver is reused;ridge differentiation,SVD and constrained optimization are known tools,not novelty claims.

| Gate | Observation | Decision |
|---|---|---|
| Quantized logical parameter distance | 0.1103276984 / 0.5 | passed |
| 4096 construction images | R0.249842845;0 prediction changes | passed |
| 5120 original task-direction finite probes | R0.249842661 | passed |
| Genuinely new512 training confirmation | FP32 R0.337044315;512/512 preserved | passed |
| Confirmation incl2560 task probes | R0.337047653 | passed |
| Complete10000 MNIST | R25.994735939;18 images/37 pairs over0.5 | **failed** |
| Complete native accuracy | 9969→9969 correct;5 predictions changed,lost2/gained2 | accuracy alone insufficient |
| True TA/originalAGA | 0 new calls | stopped before attacks |

The largest observed ratio is an INCREASE in an already wrong teacher margin: image4507 has true label1,teacher8,competitor1,gap0.20562458→5.55078125. It is neither a machine-zero denominator nor proof that every violation damages a formerly correct classification. All original mistakes remain in the declared absolute-relative-function budget. The strict0.5 gate is unchanged;we do not discard this witness or repair the point from test responses. Two native rows/32 selected scores were replayed in their original128-image contexts (3872 full image instances),maximum error0.Official labels,per-class counts and paired losses agree.

## What the source mechanism actually did

O/B distances are0.038929775/0.103231166. Inside the ORIGINAL task-right32 input frames they are5.17e-9/0.061856097;combined outside distance is0.091356578. The source path accepted21 joint updates and stopped at its fixed finite retraction rule,below the distance cap. This is a local numerical stopping result under the0.25 construction reserve,not a global parameter-distance limit. Actual evaluation R remains0.5.

| Native source counterfactual | Worst full R | Prediction changes |
|---|---:|---:|
| O only | 4.424693 | 1 |
| B task-inside | 9.387734 | 2 |
| B task-outside | 0.250808 | 0 |
| B full | 9.050295 | 2 |
| O+B inside | 1.740956 | 1 |
| O+B outside | 5.925758 | 1 |
| Joint | 0.249843 | 0 |

These changes are relative to the teacher prediction;the isolated O/B changes above correct some original mistakes rather than reducing source accuracy. Joint compensation deliberately restores the original function,including its mistakes. Removing B's inside component is not justified by the source result. Conversely,parameter task-outside directions are not free functional directions.

In the fixed512-input **public source proxy**,baseline510 correct becomes364;147 predictions change (146 formerly correct lost). Classes0/1/3 account for135 changes. B-full alone already gives147 changes,while recovered O alone gives0. This does not establish that feedback through O caused the proxy effect,nor unknown-peer transfer or real AGA robustness. The components are read-only FP64 decompositions of the frozen physical change,not separately trained/published defenses. All16 native/public component rows and per-class losses are retained;Torch/NumPy agree within3.91e-14. Source scores remain on3090.

The experiment isolates an unresolved issue: finite source compensation and a recovery-aware source objective can coexist,but neither guarantees complete task-function retention. It does NOT prove that posthoc protection or joint minmax is impossible. We do not continue the same point with more steps,another frame/rank/seed or a relaxed budget. A next candidate needs a concrete cross-input retention mechanism;no additional candidate is currently frozen.

## Reproducibility and numerical scope

[Construction protocol](../../configs/recovery_feedback.json),[new confirmation](../../configs/recovery_feedback_validation.json),[gated real evaluation](../../configs/recovery_feedback_eval.json). Source construction/final audit code is pinned by the protocols and Git history;the later source component verifier records its own SHA separately.

Independent NumPy source/probe scores agree within3.40e-9,quantized weights reconstruct exactly,official column fits agree within5.96e-8,and the variable-recovery gradient check agrees within4.56e-10 relative error. Full source/proxy FP32 controls remain below1.21e-5 (threshold5e-4). Fresh split,official labels and one-time seed2026102013 were independently reconstructed. Historical test data are not called first-use blind evaluation.

Two PRE-optimization derivative-check failures remain. The fixed central difference crossed one runner-up max branch. Its correct active-branch derivative matches autograd;verification was repaired without changing the true max/ReLU objective,gradient,epsilon or tolerance. Source calibration from the instrumented failure is reused through a symlink. Initial failure forward cost9216 image instances is separate from the corrected run4608;initial wall times were not measured separately. Corrected source220.45s,new confirmation18.39s,full native stage20.51s,source decomposition10.45s;these are stage timings,not a fabricated total campaign wall time.

Sole frozen release SHA256: `af0491135d9d0c77c2c1d86ccd2b9a34849664f050207f0ac6eb626542ae561c`.
Point SHA256: `fbed41ec8665c6463776f5e450d7e6ab194f5bd1887c4682ac1a406cb0265c38`.
Original-only calibration SHA256: `24a025541f20305192e30f5c0ddba3b01608ca90b216db2a546ed4f5d205124f`.

[Raw artifact manifest](artifact_manifest.json):491218280 unique bytes remain on3090;original experts,first-stage release and prior failure evidence are retained. One exact542728-byte duplicate was replaced by a verified symlink;28 regenerable bytecode files757430bytes removed. No new optimizer,class,experiment module,PPT or scheduler. GitHub plugin has no callable connection;new local commits are pending synchronization,not reported as uploaded.
