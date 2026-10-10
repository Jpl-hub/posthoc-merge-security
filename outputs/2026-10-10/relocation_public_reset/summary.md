# 实际编辑支持的公开替换下限

**Cars 的 V＋Attention 输出投影＋MLP 替换后，八专家融合为 60.107%。这不是受保护模型的实测恢复结果。**它约束上一条已停止的响应迁移路线及任何同支持构造。

支持限定 block 9、10、11：替换 V 的 weight／bias 切片、Attention 输出投影和完整 MLP。共 24 个参数字段，其中 6 个只替换 V 切片；Q、K、两类 LayerNorm 及区外全部参数保持原专家值。

| 未保护专家的处理 | 独立 Cars | 八专家融合 Cars |
|---|---:|---:|
| 原始 | 1911/2048 =93.311% | 1346/2048 =65.723% |
| 替换全部实际编辑支持 | 1746/2048 =85.254% | 1231/2048 =60.107% |

原模型两行逐图预测与[上一轮定位](../task_profile/summary.md)完全一致。相对原融合，130 张失去正确分类、15 张获得正确分类，净差 115 张，即 5.615 pp。独立替换损失为 8.057 pp。

若后处理 \(P_S\) 仅修改以上坐标，且架构、head 与区外参数保持原样，公开替换满足

\[
R_S(P_S(\theta))=R_S(\theta).
\]

因此在本固定八专家 TA 系数 0.2、同一 2048 图上，攻击者至少能达到 60.107%，相对原始直接融合的防御下降最多为 5.615 pp。完整 block 替换的 59.521% 仅给出了更宽松的 6.201 pp 上界。

这个上界高于预定 5 pp 门槛，所以**尚不能据此排除所有同支持构造**。判定文件的 `proposed_support_can_meet_gap=true` 仅表示替换下限未排除该门槛，不表示已有构造能达到。超过门槛的空间只有 0.615 pp；更强恢复可能提高下限。它不保证总体、官方测试、其他伙伴或未保护最佳攻击对照的性能差。

数据复用此前已看过的 Cars 训练确认集，无新选择、拟合、梯度或 AGA。只有四行评价，作业 26.71 秒；不是独立重复或新验证集。原始预测、实际替换后的参数状态与日志仅保存在服务器。

[计数与配对表](results.json)、[精确支持切片](support.json)、[判定](decision.json)、[逐图／参数／标签核验](validation.json)、[冻结源码核验](source_validation.json)。源代码 `8da2a16`。

```bash
python experiments/public_reset.py --config configs/relocation_public_reset.json --output outputs/reset_floor
python experiments/validate_results.py --public-reset --output outputs/reset_floor
```

这条诊断收紧了[已停止构造](../task_relocation_cars/summary.md)的可行空间，未产生有效候选。后续应先证明构造的计算可表示性和效用，再验证剩余的恢复差值；不能靠调大同支持噪声突破上述下限。
