# PaRaMS：固定输出校正对照

所有数字均为准确率（%）。MNIST的伙伴为SVHN，SVHN和DTD的伙伴为MNIST。复用官方默认设置的三份冻结发布；TA系数0.5。每个融合模型使用相同64张随机图估计输出先验，再按固定概率除法校正。攻击不需要真实图片、干净专家或修改位置。此为语言模型校正方法的图像域适配，不是原论文的官方CLIP实现。

| Protected task | Processing | Model | Target before | Target corrected | Partner before | Partner corrected |
|---|---|---|---:|---:|---:|---:|
| MNIST | TA | Unprotected | 99.37 | 99.34 | 92.50 | 92.32 |
| MNIST | TA | PaRaMS | 8.92 | 9.86 | 9.15 | 11.89 |
| MNIST | AGA target + TA | Unprotected | 99.30 | 99.29 | 92.78 | 92.58 |
| MNIST | AGA target + TA | PaRaMS | 99.30 | 99.29 | 92.79 | 92.58 |
| MNIST | AGA both + TA | Unprotected | 99.32 | 99.29 | 91.77 | 91.59 |
| MNIST | AGA both + TA | PaRaMS | 99.32 | 99.29 | 91.77 | 91.59 |
| SVHN | TA | Unprotected | 92.50 | 92.32 | 99.37 | 99.34 |
| SVHN | TA | PaRaMS | 9.09 | 8.45 | 8.95 | 8.35 |
| SVHN | AGA target + TA | Unprotected | 91.56 | 91.34 | 99.39 | 99.34 |
| SVHN | AGA target + TA | PaRaMS | 91.56 | 91.34 | 99.39 | 99.34 |
| SVHN | AGA both + TA | Unprotected | 91.77 | 91.59 | 99.32 | 99.29 |
| SVHN | AGA both + TA | PaRaMS | 91.77 | 91.59 | 99.32 | 99.29 |
| DTD | TA | Unprotected | 69.31 | 55.69 | 99.45 | 99.46 |
| DTD | TA | PaRaMS | 2.07 | 2.87 | 8.92 | 8.57 |
| DTD | AGA target + TA | Unprotected | 68.40 | 55.00 | 99.45 | 99.45 |
| DTD | AGA target + TA | PaRaMS | 68.35 | 55.00 | 99.45 | 99.45 |
| DTD | AGA both + TA | Unprotected | 68.78 | 55.00 | 99.42 | 99.39 |
| DTD | AGA both + TA | PaRaMS | 68.78 | 55.00 | 99.42 | 99.39 |

单独输出校正未恢复PaRaMS的直接融合能力；AGA则能恢复，校正后仍接近同处理的无保护对照。不能据此给不同防御排一个与攻击条件无关的总名次。DTD无保护模型经过校正也由69.31%降到55.69%，必须保留这个对照。

## 同一输出校正下的目标任务

仅直接TA后校正；防御一、二复用已核结果。

| Model | MNIST | SVHN | DTD |
|---|---:|---:|---:|
| Unprotected | 99.34 | 92.32 | 55.69 |
| PaRaMS | 9.86 | 8.45 | 2.87 |
| Defense 1 | 99.35 | 92.24 | 57.71 |
| Defense 2 | 89.21 | 84.56 | 59.31 |

校正后的低准确率表示该固定攻击未恢复融合能力；并不代表可抵抗其他攻击。本批36条完整分数记录、18个配对比较经独立NumPy复核，完整索引、逐类计数、配对区间及精确McNemar一致。新增576次随机图前向、0次真实图前向、0次训练、0次AGA求解，耗时30.57秒。原分数和核验脚本在outputs/params_calibration_20261030；配置、指纹及逐类记录见calibration_protocol.json和calibration.json。
