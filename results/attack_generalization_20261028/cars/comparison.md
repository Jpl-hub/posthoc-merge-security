# Cars完整评价

八专家参与，TA系数0.2；保护对象仍是MNIST、SVHN或DTD，Cars仅作为其他参与任务评价。完整测试集8041张，类别和文件顺序与AGA作者代码一致。准确率单位为%。

| 保护任务 | 模型 | 处理 | Cars |
|---|---|---|---:|
| MNIST | Unprotected | TA | 62.06 |
| MNIST | Defense 1 | TA | 2.25 |
| MNIST | Defense 2 | TA | 55.03 |
| MNIST | Unprotected | AGA target | 62.02 |
| MNIST | Defense 1 | AGA target | 2.24 |
| MNIST | Defense 2 | AGA target | 42.36 |
| SVHN | Unprotected | TA | 62.06 |
| SVHN | Defense 1 | TA | 3.84 |
| SVHN | Defense 2 | TA | 47.16 |
| SVHN | Unprotected | AGA target | 62.04 |
| SVHN | Defense 1 | AGA target | 3.84 |
| SVHN | Defense 2 | AGA target | 36.89 |
| DTD | Unprotected | TA | 62.06 |
| DTD | Defense 1 | TA | 2.89 |
| DTD | Defense 2 | TA | 53.72 |
| DTD | Unprotected | AGA target | 62.03 |
| DTD | Defense 1 | AGA target | 2.90 |
| DTD | Defense 2 | AGA target | 38.59 |

同一组冻结权重在Cars上的补测，不是新的防御方法或构造实验。无保护TA约62.06%；防御一使Cars降至2.25%～3.84%，数值防御二为47.16%～55.03%。AGA处理被保护专家后，Cars也有变化，因此需要同时报告所有伙伴任务。低伙伴准确率表示连带损害，不能记成目标保护的额外成功。

18条完整与18条pilot记录已逐类、配对和精确McNemar复核；共576个原128批上下文分数重放，最大误差0。完整评价耗时1200.15秒，新图片前向128656次，重放36450次；没有重新构造防御或求解AGA。
