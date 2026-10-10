# 四块完整替换：局部编辑的恢复下限

只恢复冻结的旧 MNIST 保护，不重新生成保护。索引 3–6 的四个完整 block 统一换为公开 anchor，包括 Attention、MLP 和两处 LayerNorm；48 项参数覆盖旧候选的全部 16 项编辑。没有校准查询、拟合或梯度步骤。SVHN 伙伴保持原样，系数固定 0.5。

| 官方完整测试 | 未保护，同样替换 | 旧保护，同样替换 |
|---|---:|---:|
| 独立 MNIST | 9,671 / 10,000 = 96.71% | 9,671 / 10,000 = 96.71% |
| 融合 MNIST | 9,634 / 10,000 = 96.34% | 9,634 / 10,000 = 96.34% |
| 融合 SVHN | 24,717 / 26,032 = 94.9485% | 24,717 / 26,032 = 94.9485% |

两组恢复后的**全部模型参数与逐图预测严格相同**。作业 64.29 秒，六行记录均核验。旧保护曾依赖已知伙伴；它只作为历史攻击控制，不是本轮未知伙伴保护候选。[逐图结果](results.json)、[编辑覆盖及参数证明](support_audit.json)、[核验](validation.json)、[冻结源码](source_validation.json)。

## 条件明确的结论

令 $B$ 是这四个完整 block，$P$ 为任何只编辑 $B$ 内参数、保持架构和外部参数不变的后处理，$R_B$ 把这些参数换为公开值。必有

\[
R_B(P(\theta))=R_B(\theta).
\]

因此，对当前 MNIST 专家、固定 SVHN 伙伴、融合系数和测试集，**任何同四块内的后处理都有 96.34% 的公开恢复下限**。增加块内 Attention、MLP、LayerNorm 的联合依赖，不能改变该恒等式。

相对固定原直接融合 99.37%，这条攻击将可造成的下降限制在 3.03 pp 以内。当前旧候选的 FC 替换仍能达到 99.25%，攻击者可以选择更好的结果；不能用较差的完整替换掩盖这个漏洞。

这不是所有局部后处理无效的结论：修改块外参数、改计算结构、换伙伴或系数，均须重新分析。这里也没有测试新的跨模块保护或证明全局最小编辑集合。

```bash
python experiments/subspace_interference.py --evaluate-saved outputs/2026-10-10/subspace_compensation_mnist --recovery-groups pretrained_block --output outputs/block_reset
python experiments/validate_results.py --subspace-test --output outputs/block_reset
```
