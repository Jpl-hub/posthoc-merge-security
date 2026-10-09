# 源码审计

审计日期：2026-10-09；作者源码保持原样，实验使用新入口。

| 来源 | 固定 commit |
|---|---|
| [secure-merge-attack](https://github.com/krumpguo/secure-merge-attack) | `fb61429f5e6f3517102d506697a150ac26abbdea` |
| [task_vectors](https://github.com/mlfoundations/task_vectors) | `826a64c67082fab0f40628233287948f0f8d7fa3` |

## Attention AGA

`utils_attack_B32.py::solve_linear_transformation` 实现带 $10^{-8}$ 岭正则的正规方程，内部 float64，返回 float32。`attack_attention_headwise_precise` 默认 12 heads，读取 `model.visual.transformer.resblocks.{i}.attn`，将 packed QKV 按第 0 维分成三个矩阵，然后逐 head 独立求 Q/K/V/O 的四个恢复矩阵。

Q/K/V 使用 $W_p^TT\approx W_{anchor}^T$，恢复为 $T^TW_p$；bias 作为行向量乘 $T$。O 对 $W_p[:,head]T\approx W_{anchor}[:,head]$ 求解；输出 bias 原样复制。估计过程不联合强制 Q/K 逆转置、V/O 逆变换关系，也不搜索 head correspondence。

`universal_attention_attack_params` 默认处理 12 层，调用上述函数时没有把 head 数暴露为参数；异常会被捕获并继续，甚至最后仍打印完成。因此新入口要显式检查每层键、返回值数量、输出有限性，不能只看日志“Completed”。

P0/P1/P3 用 AST 从固定原文件提取三个函数并直接执行，函数体未改；这样排除与这些函数无关的 JAX 依赖，同时保存原文件 SHA256。toy Attention 也使用 12 heads，模型宽度缩小为 96。

## MLP AGA

`solve_permutation` 将两组第一层权重转到 NumPy，对行做 L2 归一化，成本为负 cosine，用 SciPy `linear_sum_assignment`。它将 `restore_indices[col_ind]=row_ind`，返回使 protected 行对齐 anchor 的索引。

`attack_vit_mlp_only` 同步恢复 `c_fc.weight` 的行、`c_fc.bias` 和 `c_proj.weight` 的列；`c_proj.bias` 不动。正确匹配时与原微调 MLP 参数完全相同。论文附录的未归一化欧氏 margin 证明不能直接当作该 cosine 实现的充分条件，见 `theory_notes.md`。

## 真实 checkpoint 与环境依赖

`utils.py::load_model_checkpoint` 使用 `torch.load(weights_only=False, map_location='cpu')` 加载 pickled 整模型，随后 patch `ln_attn`。`setup_checkpoint_compat` 注册历史模块路径，但带有作者绝对路径，并引用旧 `open_clip.model.VisualTransformer`。这与新 OpenCLIP 版本可能不兼容，不能假设任意 state_dict 都能直接载入。

`modeling.py::ImageEncoder` 用 OpenCLIP `create_model_and_transforms(..., pretrained='openai')`，默认删除语言 Transformer，视觉模块由 `encode_image` 调用。`ImageEncoder.load` 存在递归调用自身的问题；`load_from_state_dict` 还引用未定义变量且按 classmethod 使用 `self`。新入口不能沿用这两个不完整方法。

## Task Arithmetic

`TaskVector.__init__` 对 pretrained 和 fine-tuned state_dict 逐键相减，跳过 int64/uint8；`__add__` 相加；`apply_to` 执行 $\theta_{pre}+\lambda\sum_i(\theta_i-\theta_{pre})$。

`main_task_arithmetic_attack_B32.py` 的 `scaling_coef_=0.2`，处理八个任务，每次只保护并攻击一个，其余七个用未保护 checkpoint。它覆盖命令行的数据路径、保存路径、batch size、model/device；不能把原样执行当作可配置复现。我们保留原文件，另建实验入口。

原 Task Arithmetic README 的双任务示例用 0.8，不是作者本仓库八任务攻击脚本的 0.2。新实验应显式记录 coefficient、任务数量与 checkpoint 来源，不能混用比较。

本次入口直接复用攻击仓库的 `TaskVector.__init__` 与 `__add__`。两任务自适应攻击需要内存中的变换后 anchor，因此仅保留应用 task vector 的薄适配层；八任务入口直接用原 `apply_to`。分类推理复用 `ImageClassifier` 和 `get_logits`，新增循环只负责完整计数、计时和显存记录。重新执行全部对照后才更新结果。

## 数据与分类 head

`eval.py` 用图像编码器、分类 head 构建 `ImageClassifier`，在 test loader 上计算 top-1。`heads.py` 查找 `head_{dataset}.pt`；不存在时使用预训练文字编码器、数据集 classnames 和 templates 创建 zero-shot head。

**目录导入不一致：** `eval.py`/`heads.py` 导入 `dataset.*`，实际仓库目录名为 `datasets/`。同目录内 registry 也使用 `dataset` 导入。目录未提供可安装 package 声明，需在新入口中处理或自行读取数据。

MNIST 数据封装调用 torchvision 自动下载 train/test；Cars 等数据仍需对应外部资源。配置项主要由 `args.py` 定义，主脚本又覆盖关键配置。`--openclip-cachedir` 在当前 `ImageEncoder` 创建函数中没有传入。

## 公开资源缺失

固定 commit 包含攻击、Task Arithmetic、评估和数据包装代码，README 只有项目名称及一句介绍。未发现完整保护实现、ARF 训练脚本、保护 checkpoint、分类 head、模型权重或环境锁文件。函数注释提及 `protect_attention_with_general_invertible`，但仓库无该函数。

Task Arithmetic 上游 README 提供 [Google Drive checkpoint 入口](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)。若不可访问，可使用同源 CLIP 的已公开下游模型做独立实验，但必须标明不是论文指定的原始 checkpoint，也不是论文表格复现。

## 其他防御源码

PaRaMS 官方仓库：[ISCT-W/PaRaMS](https://github.com/ISCT-W/PaRaMS)，固定 commit `4159a9a8bf253088694154d944f4583db8eca60e`。原 `src/PaRaMS/PaRaMS_algorithm.py` 的 SHA256 为 `40c114a121907d5633d29a49984efa9be2ac03e29e649e801fed2dccad89fb60`，用原模块的 `params` 接口，不重写核心算法。

- MLP 原代码用 JAX 构造成本、SciPy Hungarian 求与 anchor 最不相似的置换，而非随机置换。ViT 每层置换组互不依赖，一次完整 assignment pass 足以解决各组固定线性成本；本次显式设 `max_iter=1`，不同于上游默认 200。
- 对角保护配置采用上游 `[0.05, 20]` uniform。实现让 Q 和 V 共享同一组因子，K 与 O 用其倒数；不能把它描述成独立 Q/V 缩放。
- 通过原始模块加载保留 JAX 依赖，使用 CPU JAX；不修改作者文件。

MergeLock 官方仓库：[hetailang/Merge-Lock](https://github.com/hetailang/Merge-Lock)，固定 commit `f01aea9d7dddf626b6b6536d2a73894feb833fb2` 仅含 README，暂无可复用保护实现。未把我们的正交基变换直接冠名为 MergeLock。源码指纹见 `configs/sources.json`。
