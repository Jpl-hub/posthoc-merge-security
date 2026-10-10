# Post-hoc Model Merging Security

对已微调专家施加发布前后处理，保持原任务能力，检验对未经授权融合及自适应恢复的保护。目标独立准确率损失 ≤0.5 个百分点，同时完整报告性能与防御权衡；不进行额外梯度微调，有闭式拟合的实验明确记录校准数据。

**目前没有达标的新防御。**最新 MNIST 诊断检查 12 个任务主方向：联合精确方向删除无独立准确率损失，融合只下降 0.6836 pp；完整支持替换下降 7.6172 pp，不能替代细粒度删除代价。方向由防御方提供，属于 oracle 条件诊断，尚未构造保护或运行其 AGA。此前 Cars 七个稳定分离方向删除几乎无代价；跨模块候选的直接融合仅下降 0.0373 pp，原版 AGA 没有收益。旧四层候选被 FC 替换和功能重建绕过。全部真实结果保留。

## 当前结果

- [MNIST／DTD 局部任务删除](outputs/2026-10-11/local_erasure/summary.md)：两个固定点、七条恢复路径，无性能门槛；MNIST 原版 AGA 后下降 6.78 pp，重缩放恢复目标至 98.90% 并损失伙伴能力，未获得有效新防御。
- [MNIST 主任务方向删除代价](outputs/2026-10-11/critical_subspace_mnist/summary.md)：细粒度联合删除仍便宜，固定方向族已停止；[近期代码论文精读](outputs/2026-10-11/reading_focus.md)核对 AGA、NUFILT 与 SLOA 的机制边界。
- [丢弃专家的完整测试基线](outputs/2026-10-11/expert_omission/summary.md)：固定系数，分别将目标任务向量置零，并检查伙伴任务；用于明确当前准确率指标的防御空间，不是原版 AGA 或恢复原权重的结果。
- [沿输入子空间删除更新](outputs/2026-10-11/subspace_recovery_cars/summary.md)：七个精确方向投影的任务代价为零；公开协方差规则只覆盖部分方向，保护后恢复未验证。
- [Cars 细粒度预检](outputs/2026-10-11/fine_subspace_cars/summary.md)：独立确认响应分离，实际编辑支持替换的条件恢复下限为 61.67%，未进入保护构造。
- [Cars 跨模块功能敏感性](outputs/2026-10-10/functional_contrast_cars/summary.md)：两个冻结幅度、三个完整测试集、原版 AGA 及公开替换同条件对照；24 方向族的校准一阶 RMS 对比上限约 1.0933 倍，未获得有效防御。
- [MNIST／Cars／DTD 关键区域定位](outputs/2026-10-10/task_profile/summary.md)：Cars 9–11 层联合替换的独立损失 11.47 pp、八专家融合下降 6.20 pp；这是未保护模型的替换基线。
- [Cars 局部响应迁移](outputs/2026-10-10/task_relocation_cars/summary.md)：只使用目标与公开预训练模型，首层逆解未收敛；无独立准确率、AGA 或防御成功结果。
- [实际编辑支持的替换下限](outputs/2026-10-10/relocation_public_reset/summary.md)：V／Attention 输出／MLP 替换达到 60.11%；同支持保护相对当前原始融合最多下降 5.62 pp，未获得有效候选。
- [10 月 11 日汇报](outputs/2026-10-11/report.md)与[讲稿](outputs/2026-10-11/talk.md)：10 月 10 日晚更新，围绕恢复漏洞和机制结论。
- [目标专家任务定位](outputs/2026-10-10/task_support_mnist/summary.md)：仅 MNIST 选择，独立贡献不能替代融合恢复代价；未找到合格局部支持。
- [四个完整 block 替换](outputs/2026-10-10/subspace_compensation_mnist_block_reset/summary.md)：同四块内、架构不变的编辑具有 96.34% 的条件恢复下限。
- [LN 规范化＋FC 替换](outputs/2026-10-10/subspace_compensation_mnist_canonical_fc/summary.md)：公开消去精确缩放绑定，旧候选仍恢复至 99.25%。
- [七任务 PaRaMS／作者 AGA](outputs/2026-10-10/eight_task/params_seven_protection/summary.md)：八专家参与融合，七个完整测试集；本配置未抵抗 AGA。EuroSAT 原划分及 AGA Table 6 仍未对齐。
- [只恢复受保护专家的 AGA](outputs/2026-10-10/subspace_compensation_mnist_selected_aga/summary.md)：保持伙伴原样的完整测试对照。
- [联合构造、推导与淘汰依据](outputs/2026-10-10/joint_transfer_notes.md)：冻结配置、全部已测攻击与无重拟合诊断。
- [固定响应可表示性诊断](outputs/2026-10-10/response_transfer_probe/summary.md)：数值 LS 没有改善确认误差，公开符号可精确还原 Attention 输出。
- [类别响应重编译](outputs/2026-10-10/class_response_recompile/summary.md)：一次闭式拟合，独立确认损失 6.75 pp，按冻结门槛停止。
- [实验前必要条件](outputs/2026-10-10/search_constraints.md)：完整支持替换、分类头零空间及响应可表示性。

## 环境与资源

GPU 环境为 Python 3.12.3、PyTorch 2.3.0+cu121、RTX 4090 D 24 GB。依赖使用独立环境；先确认现有 CUDA PyTorch 可用。

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 open-clip-torch==2.0.2 gdown==5.2.0 huggingface_hub==0.24.7
python -m pip install -r requirements-params.txt
git clone https://github.com/krumpguo/secure-merge-attack.git external/secure-merge-attack
git -C external/secure-merge-attack checkout fb61429f5e6f3517102d506697a150ac26abbdea
git clone https://github.com/mlfoundations/task_vectors.git external/task_vectors
git -C external/task_vectors checkout 826a64c67082fab0f40628233287948f0f8d7fa3
git clone https://github.com/ISCT-W/PaRaMS.git external/PaRaMS
git -C external/PaRaMS checkout 4159a9a8bf253088694154d944f4583db8eca60e
```

源码 pin 见 [configs/sources.json](configs/sources.json)。作者 AGA 函数体保持原样；同时保存原始字节和统一换行后的指纹。

权重来自 [Task Arithmetic 官方仓库](https://github.com/mlfoundations/task_vectors#checkpoints)，17 个已取得文件的指纹见 [configs/checkpoints.json](configs/checkpoints.json)。默认目录为 `checkpoints/ViT-B-32/`、`data/`、`external/`；资源准备和缺项见[八任务状态](outputs/2026-10-09/eight_task/status.md)。

数据、checkpoint、原始预测、完整日志及缓存只保存在服务器；Git 和本地只保留代码、协议、轻量计数、核验记录及图表。凭据与登录状态不进入仓库。

## 运行当前实验

以下命令在服务器执行。每次使用新的输出目录，避免覆盖已保存结果；环境变量固定运行平台、随机哈希及计算线程。保护、融合、恢复及评价均使用已有作者代码，额外攻击单列。

局部任务删除使用两个运行前固定发布点；所有攻击均运行，无准确率淘汰门槛。重缩放和输出拟合单列于原版 AGA，三套完整测试不称八任务均值。

```bash
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/local_erasure.py --output outputs/local_deletion_check
python experiments/validate_results.py --local-erasure --output outputs/local_deletion_check
python experiments/plot_results.py --local-erasure --output outputs/local_deletion_check
```

最新方向与替换代价的预检入口如下；两轮都没有构造保护模型。复核投影诊断需要服务器父实验的原始激活。丢弃专家检查使用完整官方测试，预处理图片只缓存在服务器内存。

```bash
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/critical_subspace.py --output outputs/critical_check
python experiments/validate_results.py --critical-subspace --output outputs/critical_check
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/fine_subspace.py --config configs/fine_subspace_cars.json --output outputs/fine_check
python experiments/validate_results.py --fine-subspace --output outputs/fine_check
python experiments/plot_results.py --fine-subspace --output outputs/fine_check
python experiments/subspace_recovery.py --config configs/subspace_recovery_cars.json --output outputs/projection_check
python experiments/validate_results.py --subspace-recovery --output outputs/projection_check
python experiments/plot_results.py --subspace-recovery --output outputs/projection_check
python experiments/expert_omission.py --config configs/expert_omission.json --output outputs/omission_check
python experiments/validate_results.py --expert-omission --output outputs/omission_check
```

此前的跨模块敏感性构造已得到负结果，不继续调幅度。如下入口用于复核冻结协议。

```bash
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/functional_contrast.py --config configs/functional_contrast_cars.json --output outputs/contrast_check
python experiments/validate_results.py --functional-contrast --output outputs/contrast_check
python experiments/plot_results.py --functional-contrast --output outputs/contrast_check
```

七任务对照：每次保护一个专家，八专家按固定 TA 系数 0.2 融合。PaRaMS 使用一轮独立置换组 assignment 和预先固定的 Attention 缩放。

```bash
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --protection params --protection-checks MNIST DTD SVHN GTSRB RESISC45 Cars SUN397 --output outputs/params_check
python experiments/validate_results.py --eight-protection --output outputs/params_check
python experiments/plot_results.py --eight-protection --output outputs/params_check
python experiments/build_report.py --eight-protection --output outputs/params_check
```

任务定位不生成保护，使用 MNIST 标签选择替换支持；伙伴在选择冻结后仅用于评价。训练确认结果不与官方测试混表。

```bash
python experiments/operator_diagnostics.py --task-support --config configs/task_support_mnist.json --output outputs/task_support
python experiments/validate_results.py --task-support --output outputs/task_support
python experiments/plot_results.py --task-support --output outputs/task_support
```

三任务定位与一次 Cars 响应迁移；迁移构造已经在逆解门槛停止。每次重跑构造使用新目录；最后一条核验命令需要服务器上的原始诊断张量。

```bash
python experiments/operator_diagnostics.py --task-profile --config configs/task_profile.json --output outputs/task_profile
python experiments/validate_results.py --task-profile --output outputs/task_profile
python experiments/task_relocation.py --config configs/task_relocation_cars.json --output outputs/cars_relocation
python experiments/task_relocation.py --root-diagnostic --output outputs/cars_relocation
python experiments/validate_results.py --task-relocation --output outputs/cars_relocation
python experiments/public_reset.py --config configs/relocation_public_reset.json --output outputs/reset_floor
python experiments/validate_results.py --public-reset --output outputs/reset_floor
```

旧四层候选的保存与完整测试入口；后续恢复复用冻结保护，不重新生成；AGA 只运行受保护专家。双任务系数为 0.5，与七任务结果分开报告。

```bash
python experiments/subspace_interference.py --config configs/subspace_compensation_mnist.json --output outputs/mnist_dev
python experiments/validate_results.py --subspace-interference --output outputs/mnist_dev
python experiments/subspace_interference.py --evaluate-saved outputs/mnist_dev --output outputs/mnist_test
python experiments/validate_results.py --subspace-test --output outputs/mnist_test
python experiments/subspace_interference.py --evaluate-saved outputs/mnist_dev --recovery-groups aga_selected --output outputs/mnist_selected_aga
python experiments/validate_results.py --subspace-test --output outputs/mnist_selected_aga
python experiments/subspace_interference.py --evaluate-saved outputs/mnist_dev --recovery-groups pretrained_block --output outputs/mnist_block_reset
python experiments/validate_results.py --subspace-test --output outputs/mnist_block_reset
python experiments/subspace_interference.py --evaluate-saved outputs/mnist_dev --recovery-groups canonical_fc --output outputs/mnist_canonical_fc
python experiments/validate_results.py --subspace-test --output outputs/mnist_canonical_fc
```

`validation.json` 在服务器重新统计原始逐图预测、配对计数，并检查源码、参数及数据子集指纹。本地缺原始预测时不能声称完成同一核验。论文报告数字不记作本项目结果，训练开发子集不与官方 test 拼表。

## 工程检查与研究记录

```bash
python -m ruff check experiments
python -m ruff format --check experiments
python -m compileall -q experiments
```

- [文献核对](outputs/2026-10-09/post_meeting/reading_update.md)：NUFILT、AGA、PaRaMS、MergeBarrier、MergeLock、GRAIL、ReACT 等。
- [组会后要求与调整](outputs/2026-10-09/post_meeting/review.md)。
- [旧候选机制及全部对照](outputs/2026-10-09/post_meeting/subspace_interference_notes.md)。
- [恢复边界与额外功能攻击](outputs/2026-10-09/post_meeting/recovery_boundary_notes.md)。
- [早期实验详细记录](outputs/2026-10-09/experiment_details.md)与[代码审计](outputs/2026-10-09/code_audit.md)。

研究参考：[On the Vulnerability of Parameter-Level Defenses to Model Merging](https://arxiv.org/abs/2606.30360v1)。当前未宣称得到抗自适应恢复的新防御或完成论文八任务原表复现。
