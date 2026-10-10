# Post-hoc Model Merging Security

研究问题：不再微调已训练模型，通过后处理抑制未经授权的模型融合，同时保持 standalone utility，并评估自适应恢复攻击。

[明日汇报](outputs/2026-10-11/report.md)与[讲稿](outputs/2026-10-11/talk.md)：目前没有达标的新防御。旧候选被强恢复绕过，新联合构造因效用损失和公开组合恢复淘汰；全部真实结果与失败记录保留。

## 实验原则

- 不执行额外专家梯度训练；闭式参数拟合明确记录校准数据与调用预算。
- toy/matrix 机制验证与真实数据集准确率分开记录。
- 使用作者真实 AGA 函数；源代码保持原样并固定 commit。
- 严格等价保护检查函数等价性；近似后处理明确校准来源与独立效用损失，再评价攻击及融合效果。
- 数据、checkpoint、完整日志和依赖缓存仅保存在服务器。
- 不上传凭据、登录状态、模型权重或原始数据。

## 依赖与源码

当前 GPU 环境：Python 3.12.3、PyTorch 2.3.0+cu121、RTX 4090 D 24 GB；依赖装在独立环境，原 3090 环境与成果保留。先确认现有 torch CUDA 可用；请勿为安装其他依赖盲目覆盖 CUDA PyTorch。

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 open-clip-torch==2.0.2 gdown==5.2.0 huggingface_hub==0.24.7
git clone https://github.com/krumpguo/secure-merge-attack.git external/secure-merge-attack
git -C external/secure-merge-attack checkout fb61429f5e6f3517102d506697a150ac26abbdea
git clone https://github.com/mlfoundations/task_vectors.git external/task_vectors
git -C external/task_vectors checkout 826a64c67082fab0f40628233287948f0f8d7fa3
```

Checkpoint 来源是 [Task Arithmetic 官方 README](https://github.com/mlfoundations/task_vectors#checkpoints) 的 Google Drive。默认只下载双任务所需文件；`--all-tasks` 扩展到八任务。`configs/checkpoints.json` 固定已取得文件的 SHA256；未取得文件不填造指纹。下载文件不进入 Git。

## 运行

```bash
python experiments/record_environment.py
python experiments/download_checkpoints.py
python experiments/mechanisms.py --seeds 20261009 20261010 20261011
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5 --attack-all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/low_activity_posthoc.py
python experiments/validate_results.py
python experiments/plot_results.py
python experiments/build_report.py
```

`clip_benchmark.py` 读取 `checkpoints/ViT-B-32/`，使用 MNIST/SVHN 官方 test splits；数据下载到 `data/`。保护只在 MNIST 模型上施加；第二模型是未保护的 SVHN。0.2 与 0.5 在评价前固定，不在 test set 上调参。这是两任务独立复现，不能直接与论文八任务表格相比。

默认输出 `outputs/2026-10-09/`。脚本支持 `--output`；benchmark 也支持 `--data`、`--checkpoints`、`--author-repo`、`--batch-size` 等配置。所有重型操作在服务器执行，本地只保存轻量代码与交付文件。

PaRaMS 基线直接调用 [官方实现](https://github.com/ISCT-W/PaRaMS)，源码指纹在 `configs/sources.json`。需另装 CPU JAX；独立 ViT MLP 置换组使用一轮精确 assignment。

```bash
git clone https://github.com/ISCT-W/PaRaMS.git external/PaRaMS
git -C external/PaRaMS checkout 4159a9a8bf253088694154d944f4583db8eca60e
python -m pip install -r requirements-params.txt
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --protection params --attack-all --output outputs/2026-10-09/params
```

## 研究记录

配对对齐基线复用 [RotationSymmetry](https://github.com/zhengzaiyi/RotationSymmetry) 的原始 SVD 与 FFN matching。先运行默认正交保护实验，保留其服务器上的 protected state，再执行：

```bash
git clone https://github.com/zhengzaiyi/RotationSymmetry.git external/RotationSymmetry
git -C external/RotationSymmetry checkout b89d081429d6e6351f3449d60df5562dfa146664
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/coupled_alignment.py
```

源码路径与指纹见 `configs/sources.json`。本次覆盖正交变换及既定 head 对应关系；不包含一般 GL 白化或跨 head 匹配。

- [源码审计](outputs/2026-10-09/code_audit.md)
- [理论与适用条件](outputs/2026-10-09/theory_notes.md)
- [环境](outputs/2026-10-09/environment.md)
- [执行记录](outputs/2026-10-09/experiment_log.md)
- [中文组会报告](outputs/2026-10-09/group_meeting.md)
- [实验详细记录与完整对照](outputs/2026-10-09/experiment_details.md)
- [组会前结果表](outputs/2026-10-09/results.csv)
- [八任务资源与缺项](outputs/2026-10-09/eight_task/status.md)
- [文献核对](outputs/2026-10-09/literature_triage.md)
- [研究方向审查](outputs/2026-10-09/research_direction_review.md)
- [秩编辑与完整 FFN 重表示：数学依据、攻击与全部失败结果](outputs/2026-10-09/candidate_direction.md)
- [组会后调整：真实任务补空间扰动](outputs/2026-10-09/post_meeting/review.md)
- [首次补空间实验：完整测试](outputs/2026-10-09/post_meeting/subspace_initial.md)
- [成分分解、MLP 扰动与无数据恢复](outputs/2026-10-09/post_meeting/followup.md)
- [有界补空间编辑：双角色开发筛选](outputs/2026-10-09/post_meeting/bounded_notes.md)
- [中层联合依赖与功能重建对照](outputs/2026-10-09/post_meeting/operator_notes.md)
- [NUFILT：文献核对与任务子空间检查](outputs/2026-10-09/post_meeting/nufilt_notes.md)

本轮新增入口（服务器运行；重新执行应指定新的输出目录以保留旧结果）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/rank_probe.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/polynomial_probe.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/polynomial_probe.py --config configs/polynomial_multilayer.json --output outputs/2026-10-09/polynomial_multilayer
python experiments/prepare_eight_task_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --pretrained-checks DTD GTSRB --output outputs/2026-10-09/eight_task/pretrained_1828
```

秩编辑使用无数据对照与明确标注的合成校准；FFN 新映射和恢复攻击有闭式参数拟合，不能称完全不拟合参数。单层改写保住独立效用但可恢复，四层改写损失超预算，均不计作有效保护。分类 head 与数据入口检查不计作八任务融合复现。

本仓库不将论文报告数字记作我们的结果。head permutation 被自适应 matching 恢复时应作为负面结果保留。当前未宣称得到可抵抗自适应攻击的强防御。

组会后新增独立的真实校准实验，保留此前所有结果。先准备官方 SVHN train（图片仅留服务器），再运行；开发选择使用训练集内互斥子集，不使用官方 test 选配置。

```bash
python -c 'from torchvision.datasets import SVHN; SVHN("data", split="train", download=True)'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 JAX_PLATFORMS=cpu python experiments/subspace_probe.py
python experiments/validate_results.py --subspace --output outputs/2026-10-09/post_meeting/subspace_probe
python experiments/plot_results.py --subspace --output outputs/2026-10-09/post_meeting/subspace_probe
```

配置固定单层、输入主子空间保留阈值及随机扰动强度，不叠加旋转、置换或压缩。最终测试仅评价开发集选出的一个配置；攻击者使用独立训练图片重新估计 PCA，包含原 AGA、投影消除和 pretrained 补全。训练集开发子集并非原专家训练时的 held-out validation，属于本次局限。

后续对照复用上一轮服务器上的发布模型与冻结子空间。因果分解和 MLP 配置来自开发筛选；独立无数据攻击扫描所有线性矩阵，不能仅测试原 AGA 就认定有效保护。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/subspace_diagnostics.py
python experiments/validate_results.py --subspace-diagnostics --output outputs/2026-10-09/post_meeting/subspace_diagnostics
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/subspace_diagnostics.py --norm-cleanup --config configs/anchor_cleanup.json --output outputs/2026-10-09/post_meeting/anchor_cleanup
python experiments/validate_results.py --norm-cleanup --output outputs/2026-10-09/post_meeting/anchor_cleanup
python experiments/plot_results.py --subspace-diagnostics --output outputs/2026-10-09/post_meeting/subspace_diagnostics
```

八个 Task Arithmetic 官方专家与全部分类头已取得并核验。八专家按作者代码系数 0.2 融合，在七个完整测试集上完成评估；SUN397 归档与第 1 划分图片已核验，EuroSAT 原随机索引仍缺。现有结果与论文表格尚未对齐，不能报告完整八任务均值。[计数、来源及差异](outputs/2026-10-09/eight_task/status.md)。

多层投影删除与范数保持反射使用同一入口，先检验开发集效用与无数据角度清理；只有通过门槛才运行官方测试。两角色本轮均未通过，未进行最终测试。输入诊断不计作准确率实验。JSON 保留原始字节，便于跨平台核对记录的指纹。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/bounded_complement.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/bounded_complement.py --config configs/bounded_svhn.json --output outputs/2026-10-09/post_meeting/bounded_svhn
python experiments/validate_results.py --bounded-complement --output outputs/2026-10-09/post_meeting/bounded_complement
python experiments/validate_results.py --bounded-complement --output outputs/2026-10-09/post_meeting/bounded_svhn
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/bounded_complement.py --distribution-shift --output outputs/2026-10-09/post_meeting/distribution_shift
python experiments/validate_results.py --distribution-shift --output outputs/2026-10-09/post_meeting/distribution_shift
python experiments/plot_results.py --bounded-complement --output outputs/2026-10-09/post_meeting/bounded_complement
python experiments/plot_results.py --bounded-complement --output outputs/2026-10-09/post_meeting/bounded_svhn
python experiments/plot_results.py --distribution-shift --output outputs/2026-10-09/post_meeting/distribution_shift
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/bounded_complement.py --scope-audit --output outputs/2026-10-09/post_meeting/scope_audit
python experiments/validate_results.py --scope-audit --output outputs/2026-10-09/post_meeting/scope_audit
```

中层定位使用同一作者评价与融合入口，并复用现有闭式重建；没有提出新保护。固定新增互斥训练子集复核联合替换和功能恢复，保留逐图配对计数。原始预测只在服务器保留；服务器验证会检查其指纹与计数。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/operator_diagnostics.py
python experiments/validate_results.py --operator-diagnostics --output outputs/2026-10-09/post_meeting/operator_diagnostics
python experiments/plot_results.py --operator-diagnostics --output outputs/2026-10-09/post_meeting/operator_diagnostics
```

固定预算重建进一步比较第一层特征来源与拟合输入。第二轮可使用重建模型自己融合后的输入，但增加一轮拟合目标调用；这仍是未保护模型上的恢复控制。[协议、数学与本轮结果](outputs/2026-10-09/post_meeting/reconstruction_notes.md)。

```bash
python experiments/operator_diagnostics.py --reconstruction-context --config configs/reconstruction_context.json --output outputs/2026-10-10/reconstruction_context
python experiments/operator_diagnostics.py --reconstruction-context --config configs/adaptive_context.json --output outputs/2026-10-10/adaptive_context
python experiments/validate_results.py --reconstruction-context --output outputs/2026-10-10/adaptive_context
python experiments/plot_results.py --reconstruction-context --output outputs/2026-10-10/adaptive_context
```

## 当前子空间实验

固定 MNIST 保护、SVHN 伙伴，单个扰动强度；无梯度微调，闭式响应补偿使用无标签训练校准图。官方完整测试加载已保存的候选参数，额外功能重建单列。[机制与全部结果](outputs/2026-10-09/post_meeting/subspace_interference_notes.md)。在服务器运行，输出目录须为新目录：

```bash
python experiments/subspace_interference.py --config configs/subspace_compensation_mnist.json --output outputs/mnist_dev
python experiments/validate_results.py --subspace-interference --output outputs/mnist_dev
python experiments/subspace_interference.py --evaluate-saved outputs/mnist_dev --output outputs/mnist_test
python experiments/validate_results.py --subspace-test --output outputs/mnist_test
python experiments/plot_results.py --subspace-interference --output outputs/mnist_test
```

完整编辑集合替换、联合模块依赖与分类输出恢复另列为机制检验。[数学边界与真实结果](outputs/2026-10-09/post_meeting/recovery_boundary_notes.md)。分类输出恢复只拟合公共特征，当前是已知伙伴、相同分类 head 的额外攻击，不能称原 AGA 或新防御：

联合依赖对照区分 LayerNorm 与 Attention＋MLP；任务更新删除检验复用已冻结保护参数，不改变扰动或重新补偿。以下均为已查看训练子集上的机制筛选，失败后按协议停止：

```bash
python experiments/operator_diagnostics.py --config configs/operator_support_mnist_norms.json --output outputs/norm_support
python experiments/validate_results.py --operator-diagnostics --output outputs/norm_support
python experiments/plot_results.py --operator-diagnostics --output outputs/norm_support
python experiments/context_erasure.py --redundancy-probe --config configs/redundancy_erasure_mnist.json --output outputs/redundancy_erasure
python experiments/validate_results.py --redundancy-erasure --output outputs/redundancy_erasure
python experiments/plot_results.py --redundancy-erasure --output outputs/redundancy_erasure
```

分类输出恢复：

```bash
python experiments/operator_diagnostics.py --readout-recovery --config configs/readout_recovery_mnist.json --output outputs/readout_dev
python experiments/validate_results.py --readout-recovery --output outputs/readout_dev
python experiments/operator_diagnostics.py --evaluate-readout outputs/readout_dev --output outputs/readout_test
python experiments/validate_results.py --readout-recovery --output outputs/readout_test
```

固定 readout 的查询分类稳定性与精确单次变化诊断使用同一入口；其界限只适用于固定线性 readout，oracle 确认集结果不计作防御：

```bash
python experiments/operator_diagnostics.py --readout-stability outputs/readout_dev --output outputs/readout_stability
python experiments/validate_results.py --readout-stability --output outputs/readout_stability
python experiments/operator_diagnostics.py --readout-sensitivity outputs/readout_stability --output outputs/readout_sensitivity
```

## 工程检查

```bash
python -m ruff check experiments
python -m ruff format --check experiments
python -m compileall -q experiments
```

研究参考：[On the Vulnerability of Parameter-Level Defenses to Model Merging](https://arxiv.org/abs/2606.30360v1)，[Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors)。
