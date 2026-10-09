# 功能重建与融合输入

**结论：固定公开特征时，改变拟合输入可以缩小融合残差。残差没有证明任务信息不可恢复；本轮没有新防御，也没有在真实保护模型上验证新攻击。**

## 协议

CLIP ViT-B/32，官方 pretrained、SVHN 与 MNIST 专家，Task Arithmetic 系数 0.5。仅重建 SVHN 的 block 3–6 MLP，其他模块保持发布控制模型。复用作者 TaskVector、ImageClassifier/get_logits 与项目原有 ridge 求解器，不增加优化器或专家微调。

所有拟合使用同一冻结的 256 张无标签 SVHN 训练图，每层 12,800 个 token；第一层固定，第二层与偏置闭式拟合。正则比率固定 1e-4，每层独立拟合，不按分类结果选择超参数。检查使用各任务 32 张校准图，与拟合及评价互斥。

评价仍是已经查看过的 SVHN 4,000 张、MNIST 2,000 张训练子集；不是新的 held-out validation，也不是官方 test。首次四组使用相同图像预算；第二轮另有明确的调用成本，见下文。

| 第一层特征 | 拟合输入 | 独立 SVHN | 融合 SVHN | 融合 MNIST |
|---|---|---|---|---|
| 原模型，无重建 | — | 98.625% | 89.925% | 99.450% |
| 公开 pretrained | 原专家 | 98.625% | 89.275% | 99.450% |
| 公开 pretrained | 原融合模型 | 98.625% | 89.500% | 99.450% |
| 发布控制模型 | 原专家 | 98.625% | 89.925% | 99.450% |
| 发布控制模型 | 原融合模型 | 98.625% | 89.925% | 99.450% |
| 公开 pretrained，第二轮 | 第一轮重建模型融合后 | 98.625% | 89.500% | 99.450% |

“发布控制模型”指本轮未保护的原 SVHN 专家。其第一层特征和原融合模型输入均是诊断对照的可用量；若真实保护改变了这些表示，不能默认攻击者仍能取得相同量。

## 数学上区分两件事

令 H 为固定第一层与 QuickGELU 生成的特征，附加常数列；Y 是发布 MLP 在所选输入上的输出，A 包含第二层权重和偏置，A0 来自 pretrained。沿用目标：

$$
\min_A\;\|HA-Y\|_F^2+\lambda\|A-A_0\|_F^2,
\qquad \lambda=10^{-4}\|H\|_F^2/N.
$$

本轮 N=12,800、特征数 3,073，使用等价 primal 系统。所有新求解的一阶最优性残差小于 1e-8。使用发布控制模型的第一层时，Y=H A_S，因此

$$
\widehat A-A_S=-\lambda(H^TH+\lambda I)^{-1}(A_S-A_0).
$$

这解释了相同特征的正对照为何容易恢复拟合目标，但不保证任意未见输入、参数或分类决策完全相同。公开第一层不能直接使用这个恒等式，因为它与原专家的隐藏特征不同。

另一方面，专家功能近似并不保证参数融合后的功能近似。公开特征重建令专家第一层更新为零；与 MNIST 融合后，其第一层为 W1,0 + 0.5 ΔW1,M。原融合模型还保留 0.5 ΔW1,S。第二层对独立功能的补偿未必在这个改变后的特征上继续成立。

本轮在相同图像上冻结原专家和原融合模型输入，分别测量重建专家算子与参数融合算子的误差。这样比较局部算子时不会同时改变上游输入。它检验误差来源，不能单独证明某个 block 导致了分类错误。

## 局部检查

32 张互斥 SVHN 校准图上，block 3–6 原融合输入相对原专家输入的范数变化为 34.20%–45.34%。这不是准确率或能量占比。

| 固定公开特征的拟合输入 | 专家算子相对输出误差 | 参数融合算子相对输出误差 |
|---|---|---|
| 原专家输入 | 3.021%–3.594% | 1.554%–1.821% |
| 原融合输入 | 2.250%–3.107% | 1.167%–1.573% |
| 第一轮重建模型融合后的输入 | 2.250%–3.107% | 1.167%–1.573% |

以上均在冻结的原融合输入上检查，分母为对应原算子的完整输出范数。两种融合输入拟合的结果接近，但不能由四舍五入后的范围或相同准确率推出模型相同。使用发布特征时，局部误差更小；完整检查包含两任务、两输入上下文和任务增量归一化，见 [64 条诊断](../../2026-10-10/reconstruction_context/local_errors.json)及[第二轮 32 条诊断](../../2026-10-10/adaptive_context/local_errors.json)。

## 第二轮：使用攻击者自己的融合输入

先用原来的公开特征重建得到 Ŝ1；攻击者可按原 Task Arithmetic 构建 merge(W0, Ŝ1, MNIST)。再在同一 256 张图上捕获它的输入，用发布 MLP 输出作为目标，独立重建四层得到 Ŝ2。第二轮拟合不访问原融合模型输入；原融合输入只用于事后诊断。

每层累计使用 25,600 个拟合目标 token 行，第一轮为 12,800；这是两倍的拟合目标调用预算，并非相同查询预算。该计数只覆盖拟合目标 MLP 的输出行，排除输入捕获、事后诊断和完整评价。第一轮拟合权重复用服务器缓存，新四次拟合共 9.70 s，包含输入捕获、求解、最优性检查和模型复制；这是一次调用计时，不是速度基准。

融合 SVHN 为 3,580/4,000；相对原模型损失 19 张、增加 2 张，净差 17 张（0.425 pp）。相对第一轮的 3,571 张正确，有 10 张变对、1 张变错，净增 9 张。独立 SVHN 两轮正确状态相同，但相对原专家都有 1 张变错、1 张变对。不能将相同正确总数称为完全恢复。配对计数只作已经查看过的训练子集上的描述。[服务器原始预测核验](../../2026-10-10/adaptive_context/validation.json)。

![第二轮评价及预算](../../2026-10-10/adaptive_context/figures/context_accuracy.png)

## 与前作的关系

[PLeaS v2 §4.2](https://arxiv.org/html/2407.02447v2#S4.SS2)已经以逐层最小二乘匹配置换后组合特征。我们的固定公开 MLP 特征、单个发布模块目标及融合输入重拟合不是完整 PLeaS 复现；功能拟合本身不作为创新。作者公开优化器使用 Adam，本轮沿用已实现的闭式 ridge，没有加入其训练流程。[阅读范围与固定源码](reading_update.md#pleas拟合融合功能已有前作)。

当前更明确的评估要求是：未来保护必须面对攻击者在其自行构建的融合输入上重建发布功能。仅让参数不对应或独立重建略有融合损失，证据不足。尚未得到能通过该条件的新构造。

## 记录与复核

每次准确率测量指一个配置在一个任务及运行方式上预测固定图片，对照标签计算正确比例。一个配置测独立 SVHN、融合 SVHN、融合 MNIST 三项；记录条数不是新方法数、独立重复次数或新增图片数。

四组控制源码提交 `0cae27c`，执行 15 次准确率测量：9 项新增配置结果、6 项旧对照重测。新执行 12 次拟合，复用 4 次拟合；总实验 112.74 s。[协议](../../2026-10-10/reconstruction_context/protocol.json)、[元数据](../../2026-10-10/reconstruction_context/metadata.json)、[核验](../../2026-10-10/reconstruction_context/validation.json)。

第二轮源码提交 `e157850`，执行 9 次准确率测量：3 项新增配置结果、6 项旧对照重测。新执行 4 次拟合，复用 4 次拟合；总实验 68.86 s。[协议](../../2026-10-10/adaptive_context/protocol.json)、[元数据](../../2026-10-10/adaptive_context/metadata.json)、[核验](../../2026-10-10/adaptive_context/validation.json)。两组对照合计新增 4 个重建配置、12 项配置结果，另有 12 次旧对照重测。每组原模型和旧 256 张重建的 6 项逐图预测与前次记录各自完全一致；重复运行不算独立种子。

服务器保存权重、原始 logits/预测、冻结输入与完整日志，指纹记录在 metadata.json。这里只保留轻量 JSON 和图表。

```bash
python experiments/operator_diagnostics.py --reconstruction-context --config configs/reconstruction_context.json --output outputs/2026-10-10/reconstruction_context
python experiments/operator_diagnostics.py --reconstruction-context --config configs/adaptive_context.json --output outputs/2026-10-10/adaptive_context
python experiments/validate_results.py --reconstruction-context --output outputs/2026-10-10/reconstruction_context
python experiments/validate_results.py --reconstruction-context --output outputs/2026-10-10/adaptive_context
python experiments/plot_results.py --reconstruction-context --output outputs/2026-10-10/adaptive_context
```

首次运行保留输出目录，不覆盖既有记录；再次运行需另选新目录。复核原始四组代码使用记录的提交；当前入口兼容两种固定配置。
