# Post-hoc Model Merging Security

研究问题：不再微调已训练模型，通过后处理抑制未经授权的模型融合，同时保持 standalone utility，并评估自适应恢复攻击。

## 实验原则

- 不执行模型训练或额外微调。
- toy/matrix 机制验证与真实数据集准确率分开记录。
- 使用作者真实 AGA 函数；源代码保持原样并固定 commit。
- 保护后先检查函数等价性，再评价攻击及融合效果。
- 数据、checkpoint、完整日志和依赖缓存仅保存在服务器。
- 不上传凭据、登录状态、模型权重或原始数据。

## 依赖与源码

当前 GPU 环境：Python 3.12.3、PyTorch 2.3.0+cu121、RTX 3090 24 GB。先确认现有 torch CUDA 可用；请勿为安装其他依赖盲目覆盖 CUDA PyTorch。

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 open-clip-torch==2.0.2 gdown==5.2.0 huggingface_hub==0.24.7
git clone https://github.com/krumpguo/secure-merge-attack.git external/secure-merge-attack
git -C external/secure-merge-attack checkout fb61429f5e6f3517102d506697a150ac26abbdea
git clone https://github.com/mlfoundations/task_vectors.git external/task_vectors
git -C external/task_vectors checkout 826a64c67082fab0f40628233287948f0f8d7fa3
```

Checkpoint 来源是 [Task Arithmetic 官方 README](https://github.com/mlfoundations/task_vectors#checkpoints) 的 Google Drive。仅下载 B/32 的 pretrained、MNIST、SVHN 和对应 classification heads。`checkpoint_manifest.json` 保存文件来源、大小及 SHA256；下载文件不进入 Git。

## 运行

```bash
python experiments/record_environment.py
python experiments/download_checkpoints.py
python experiments/mechanisms.py --seeds 20261009 20261010 20261011
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
python experiments/validate_results.py
python experiments/plot_results.py
```

`clip_benchmark.py` 读取 `checkpoints/ViT-B-32/`，使用 MNIST/SVHN 官方 test splits；数据下载到 `data/`。保护只在 MNIST 模型上施加；第二模型是未保护的 SVHN。0.2 与 0.5 在评价前固定，不在 test set 上调参。这是两任务独立复现，不能直接与论文八任务表格相比。

默认输出 `outputs/2026-10-09/`。脚本支持 `--output`；benchmark 也支持 `--data`、`--checkpoints`、`--author-repo`、`--batch-size` 等配置。所有重型操作在服务器执行，本地只保存轻量代码与交付文件。

## 研究记录

- [源码审计](outputs/2026-10-09/code_audit.md)
- [理论与适用条件](outputs/2026-10-09/theory_notes.md)
- [环境](outputs/2026-10-09/environment.md)
- [执行记录](outputs/2026-10-09/experiment_log.md)
- [中文组会报告](outputs/2026-10-09/group_meeting.md)
- [结果表](outputs/2026-10-09/results.csv)

本仓库不将论文报告数字记作我们的结果。head permutation 被自适应 matching 恢复时应作为负面结果保留。当前未宣称得到可抵抗自适应攻击的强防御。

## 工程检查

```bash
python -m ruff check experiments scripts
python -m ruff format --check experiments scripts
python -m compileall -q experiments
```

研究参考：[On the Vulnerability of Parameter-Level Defenses to Model Merging](https://arxiv.org/abs/2606.30360v1)，[Editing Models with Task Arithmetic](https://github.com/mlfoundations/task_vectors)。
