# Defense 2 攻击与对照清单

范围：冻结的末层 MLP A/B 数值补偿，含 SVD 为主方法，去 SVD 为消融。不重新构造防御。

| 项目 | MNIST | SVHN | DTD | 本轮动作 |
|---|---|---|---|---|
| 完整独立、TA0.5、AGA仅目标/双方 | 已核验 | 已核验 | 已核验 | 复用 |
| 去SVD完整独立、TA、AGA | 已核验 | 已核验 | 已核验 | 复用 |
| 主方法64噪声输出校正 | 已核验 | 已核验 | 缺失 | 补DTD |
| 去SVD64噪声输出校正 | 已核验 | 已核验 | 已核验 | 复用 |
| A-only、B+b-only严格删项 | 已核验 | 缺失 | 缺失 | 复用MNIST；后两项待补 |
| 最后MLP公开替换A/B+b/完整MLP | 缺失 | 缺失 | 缺失 | 第一批；匹配无保护及无目标贡献 |
| 非AGA公开匹配 | 缺失 | 缺失 | 缺失 | 先验证可恢复置换，再评价当前发布 |
| 等张量改动范数随机噪声 | 缺失 | 缺失 | 缺失 | 固定种子和范数后评价 |
| TIES/DARE/NUFILT/ProDistill | 旧方法有记录 | 旧方法有记录 | 旧方法有记录 | 不冒充当前发布已覆盖；优先TIES/DARE |
| 八专家、八数据集评价 | 未完成 | 未完成 | 未完成 | 旧表只测三数据集；当前主发布尚未覆盖 |

## 实现及来源

- AGA：现有 `_impl.attacks.author_recover`，作者固定源码指纹在 protocol.json；本轮不重复已有结果。
- 输出校正：[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)，固定作者代码 e04d8643be91c2cce63f33e07760ff75d5aa3ad0；64噪声是图像域适配，不是官方CLIP复现。
- 公开模块替换：本项目现有 `public_mlp_reset` 的明确局部对照；新函数只接收发布模型与基座，不接收干净专家。属于位置已知的白盒替换，不称知识反演。
- [Git Re-Basin](https://arxiv.org/abs/2209.04836)、[作者weight_matching](https://github.com/samuela/git-re-basin/blob/main/src/weight_matching.py)：已阅读联合入边/出边权重成本；原代码没有CLIP ViT的排列规格。MLP隐藏单元可独立换序，packed QKV不能随意跨头重排，残差通道须跨层一致，LayerNorm及投影须同步。当前状态为源码审查，未运行；不能把AGA改名。
- 已有 `public_patch_recovery` 只匹配公开patch通道，适用于已知全网共同通道置换，不能冒称通用Git Re-Basin。

## 资源与八任务缺项

当前12997为RTX3090；Python3.12.3、torch2.3.0+cu121，FP32/禁TF32。只有MNIST、SVHN、DTD完整测试数据与对应三专家/头。其余五专家、分类头和官方测试划分要从历史归档核验或补齐；不能先写八任务平均。服务器旧目录无.git，本批另建源码工作副本，资产只链接，不覆盖旧目录。

GitHub插件及本机HTTPS本轮连接失败；7f06a12e是最近成功读回的远端提交，不声称它仍是当前远端HEAD。已暂停原无数据蒸馏诊断，转入本次攻击和对照。
