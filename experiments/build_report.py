"""Generate concise Chinese reports and a unified CSV from executed results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read(root: Path, name: str, default=None):
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def mean(rows, field):
    return statistics.mean(row[field] for row in rows)


def run(root: Path):
    mechanism = read(root, "mechanism_results.json", [])
    real = read(root, "clip_results.json", [])
    ln = read(root, "layernorm/layernorm_results.json", [])
    mm = read(root, "mechanism_metadata.json", {})
    cm = read(root, "clip_metadata.json", {})
    env = read(root, "environment.json", {})
    lm = read(root, "layernorm/layernorm_metadata.json", {})
    all_rows = mechanism + real + ln
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted(set().union(*(row.keys() for row in all_rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    mlp = [r for r in mechanism if r["experiment"] == "mlp_permutation" and r["magnitude"] == 0.01]
    attn = [r for r in mechanism if r["experiment"] == "attention_qk_vo" and r["magnitude"] == 0.01]
    heads = [r for r in mechanism if r["experiment"] == "heads_plus_orthogonal"]
    p1 = [r for r in mechanism if r["experiment"] == "column_space_invariance"]
    synthetic_table = table(
        ["机制；合成 task change = 1%", "保护输出误差", "AGA 参数误差", "AGA 输出误差"],
        [
            ["MLP 置换", f"{mean(mlp, 'protected_output_error'):.2e}", "0", "0"],
            [
                "Attention QK/VO",
                f"{mean(attn, 'protected_output_error'):.2e}",
                f"{mean(attn, 'recovered_weight_error'):.2e}",
                f"{mean(attn, 'recovered_output_error'):.2e}",
            ],
        ],
    )
    head_table = table(
        ["跨 head 置换 + 正交基变换", "相对输出误差；3 seeds 平均"],
        [
            ["保护 standalone", f"{mean(heads, 'protected_output_error'):.2e}"],
            ["原逐 head AGA", f"{mean(heads, 'recovered_output_error'):.4f}"],
            ["子空间 matching + AGA", f"{mean(heads, 'adaptive_output_error'):.4f}"],
        ],
    )

    def accuracy(group, dataset, coef=None):
        hit = [
            r
            for r in real
            if r["group"] == group and r["dataset"] == dataset and r["coefficient"] == coef
        ]
        return f"{100 * hit[0]['top1']:.2f}%" if hit else "未完成"

    real_table = table(
        ["组别", "系数", "MNIST", "SVHN"],
        [
            [
                "A：未保护 MNIST 模型",
                "—",
                accuracy("A_unprotected", "MNIST"),
                accuracy("A_unprotected", "SVHN"),
            ],
            [
                "B：后处理保护",
                "—",
                accuracy("B_protected", "MNIST"),
                accuracy("B_protected", "SVHN"),
            ],
            ["C：B 经 AGA 恢复", "—", accuracy("C_AGA", "MNIST"), accuracy("C_AGA", "SVHN")],
            *[
                [label, str(coef), accuracy(group, "MNIST", coef), accuracy(group, "SVHN", coef)]
                for coef in [0.2, 0.5]
                for group, label in [
                    ("D_unprotected_merge", "D：未保护融合"),
                    ("E_protected_merge", "E：保护后直接融合"),
                    ("F_AGA_merge", "F：恢复后融合"),
                ]
            ],
        ],
    )
    ln_text = "候选实验尚未完成，不计入防御效果结论。"
    if ln and lm:
        groups = [
            ("LN_null_only", "空方向扰动 standalone"),
            ("LN_null_plus_basis", "空方向 + 基变换 standalone"),
            ("LN_null_AGA", "原 AGA standalone"),
            ("LN_null_canonical_AGA", "投影消除 + AGA standalone"),
            ("LN_null_AGA_merge", "原 AGA 后融合"),
            ("LN_null_canonical_AGA_merge", "投影消除 + AGA 后融合"),
            ("canonical_unprotected_merge", "canonical 未保护融合对照"),
        ]
        strengths = sorted({r["strength"] for r in ln})
        values = []
        for group, label in groups:
            values.append(
                [
                    label,
                    *[
                        f"{100 * next(r['top1'] for r in ln if r['group'] == group and r['strength'] == s):.2f}%"
                        for s in strengths
                    ],
                ]
            )
        ln_text = table(["MNIST 全测试集", *[f"扰动强度 {s:g}" for s in strengths]], values)
        ln_text += "\n\n所有参数修改均为解析后处理，无训练、无校准数据。保护误差须通过脚本阈值 $10^{-4}$。canonicalization 使用公开 LayerNorm 参数，不需要秘密或原始训练数据。比较原 AGA 与自适应攻击时，同时保留 canonical 未保护融合对照。"
        ln_text += "\n\n![LayerNorm 后处理探索](figures/layernorm_exploration.png)"
    meeting = rf"""# 2026-10-09 组会：无微调的模型融合保护

## 研究问题

对已经训练好的模型增加后处理保护：不再微调，尽量不使用原始训练数据，保持 standalone utility，阻止未经授权融合，并抵抗恢复攻击。

今天没有执行模型训练。完成 {len(mechanism)} 条机制验证、{len(real)} 条真实模型评估{("、" + str(len(ln)) + " 条 LayerNorm 候选评估") if lm else ""}。尚未得到可抵抗自适应攻击的强防御。

## 论文与实现

AGA 用 pretrained 权重作 anchor：Attention 对 Q/K/V/O 分别逐 head 最小二乘对齐；MLP 用 cosine + Hungarian 恢复隐藏神经元顺序。ARF 在微调时扩大 Attention 与 anchor 的距离，因此不满足本课题的“仅后处理”。

作者公开代码未包含完整保护、ARF 训练和 checkpoint。今天使用 Task Arithmetic 官方 B/32 checkpoint，自行实现函数等价保护，直接调用固定版本的作者 AGA 函数。属于独立复现，不是论文八任务表格的原样复现。

## 已完成：机制验证

3 个固定种子；GELU MLP 宽度 48、隐藏维 96；Attention 宽度 96、12 heads。参数变化是合成扰动，不是下游训练。

{synthetic_table}

MLP 的 15 个测试配置均 100% 恢复真实置换；恢复后的参数误差为零。Attention 在不同扰动幅度下仍保持保护函数等价，但 AGA 独立估计投影会留下恢复误差。

![机制验证](figures/mechanisms.png)

## 已完成：真实 CLIP 对照

MNIST test：10,000；SVHN test：26,032。保护对象是 MNIST 微调模型，第二专家是 SVHN。两个融合系数预先固定，未使用测试集选择系数。测试仅评价，没有校准。

{real_table}

保护前后 16 张真实图像上的特征相对误差为 {cm.get("protected_output_error", float("nan")):.2e}。原模型与保护模型在两个完整测试集上的正确样本数相同。AGA 后全 checkpoint 参数相对误差为 {cm.get("recovered_weight_error", float("nan")):.2e}；该指标含未参与视觉推理的遗留参数，不能替代 feature error 或 accuracy。AGA 特征误差为 {cm.get("recovered_output_error", float("nan")):.4f}。

![真实模型融合](figures/clip_merging.png)

## 已完成：后处理边界

对于 $W^p=W_{{ft}}P$、可逆 $P$，理想 LS 的拟合值为

$$
W^a=\Pi_{{\operatorname{{col}}(W_{{ft}})}}W_{{pre}},\qquad
W^a-W_{{ft}}=-\Pi_{{ft}}\tau.
$$

这一结论已在论文附录证明。我们的直接 LS 数值验证中，不同正交、对角及稠密保护矩阵与统一投影结果的最大相对差异为 {max(r["ideal_projection_error"] for r in p1):.2e}。作者带正则的正规方程与 float32 转换在高条件数时偏离理想结果；这些数值误差不构成安全防御。

跨 head 置换保持函数，但会破坏原代码默认对应关系：

{head_table}

自适应子空间 matching 在全部 3 个种子中 100% 恢复 head correspondence，故不把跨 head 置换当成强防御。

![跨 head 置换](figures/head_permutation.png)

## 后处理候选：LayerNorm 空方向

若 $y=\gamma\odot z+\beta$ 且 $\mathbf1^Tz=0$，取归一化 $r\propto1/\gamma$，则 $r^T(y-\beta)=0$。对紧随 LayerNorm 的 Linear 作

$$
W'=W+ur^T,\qquad b'=b-u(r^T\beta),
$$

可改变参数列空间而保持该层在合法输入上的输出。它超出了“仅右乘可逆矩阵”的限制，且无需训练数据。候选施加在 QKV 与 MLP 第一层，再与基变换保护组合。

但 $r$ 由公开 LayerNorm 参数确定。攻击者可解析消去该方向：

$$
W_c=W-(Wr)r^T,\qquad b_c=b+(Wr)(r^T\beta).
$$

因此必须评价“canonicalization + AGA”，不能只观察原 AGA 是否失效。

{ln_text}

## 结论与下一步

1. 小任务变化下，置换和 head 内可逆变换可被 AGA 绕过；扩大变换幅度不改变理想投影结果。
2. 精确函数等价后处理确有可用自由度，但已知对称性往往也给攻击者提供 canonicalization。
3. 下一步优先分析“不属于易消去对称性”的小效用损失扰动：用少量公开、未标注校准输入的低活跃方向构造解析扰动，并预先固定 standalone 损失预算。该方向尚未验证，是否允许校准数据需与导师确认。
4. 先保留 MNIST/SVHN 两任务对照，加入 head matching、LayerNorm 投影消除及直接 LS 攻击；候选通过后再扩展任务、种子和融合方法。

## 审读发现

MLP 附录的原始欧氏 margin 条件不能直接保证代码中的 cosine matching；固定反例满足条件却得到错误置换。ARF 印刷式 (19) 的 `min` 与正文 margin 行为不一致，已核对 PDF。由于 ARF 训练代码未公开，不能据此推断其实际实现或实验无效。详情见 `theory_notes.md`。

参考：[论文 v1](https://arxiv.org/abs/2606.30360v1)、[作者代码](https://github.com/krumpguo/secure-merge-attack)、[Task Arithmetic](https://github.com/mlfoundations/task_vectors)。数字均取自本次程序输出；论文数字未填入结果表。
"""
    (root / "group_meeting.md").write_text(meeting, encoding="utf-8")
    environment = f"""# 实验环境

日期：2026-10-09；运行位置：租用服务器；本地仅保存轻量文件。

| 项目 | 记录 |
|---|---|
| GPU | {env.get("gpu")}，24 GB |
| Compute capability | 8.6 |
| Python | 3.12.3 |
| PyTorch | {env.get("torch")} |
| PyTorch CUDA runtime | {env.get("torch_cuda")} |
| 驱动 | 595.71.05；nvidia-smi 支持上限 CUDA 13.2 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| OpenCLIP | 2.0.2 |
| torchvision | 0.18.0+cu121 |
| 主机内存 | 总计约 375.8 GiB；检查时可用约 322.9 GiB；是主机视图 |
| 数据盘 | 50 GiB；checkpoint/数据仅保存在此盘 |
| 系统盘 | 30 GiB；依赖安装在现有 Miniconda |
| TF32 | 关闭 |
| CPU 线程 | PyTorch / OpenBLAS：4 |

现有 CUDA PyTorch 已通过 GPU 算子检查，无重装。新增依赖命令：

```bash
/root/miniconda3/bin/python -m pip install numpy==1.26.4 scipy==1.13.1 gdown==5.2.0 huggingface_hub==0.24.7 open-clip-torch==2.0.2
```

最初未锁版本的 pip 解析拟升级 NumPy，已在安装前停止并改为上面的兼容版本。完整安装日志、环境 JSON 与 pip freeze 留在服务器。仓库提交研究所需版本信息，排除机器连接和认证资料。

计时为实际 wall time，包含各脚本自身记录的加载/下载等阶段。显存峰值为 PyTorch 当前进程 `max_memory_allocated`，不是整机占用；P0 总峰值 {mm.get("peak_gpu_bytes", 0) / 2**20:.2f} MiB。基准过程有一次短暂并行候选启动，随后停止并排队；因此不将本次耗时作为严格性能比较。
"""
    (root / "environment.md").write_text(environment, encoding="utf-8")
    log = f"""# 执行记录

| 阶段 | 状态 | 实际记录 |
|---|---|---|
| 环境 | 完成 | 3090；现有 torch 2.3.0+cu121；GPU 算子可用 |
| 论文 | 完成 | 阅读本地 28 页 PDF；视觉核对第 7、13、19 页相关公式 |
| 源码 | 完成 | 两个官方仓库固定 commit；原文件未修改 |
| P0/P1/P3 | 完成 | {len(mechanism)} 行；3 seeds；{mm.get("elapsed_seconds", 0):.3f} s |
| P2 | {"完成" if cm else "运行中"} | {len(real)} 行；full MNIST/SVHN；{cm.get("total_seconds", 0):.3f} s |
| LayerNorm 候选 | {"完成" if lm else "运行中"} | {len(ln)} 行；{lm.get("total_seconds", 0):.3f} s |
| 结果验证 | 已通过基础检查 | 函数等价、投影恒等式、样本数及 accuracy 计数 |

服务器工作目录：`/root/autodl-tmp/merge_security_20261009/`。完整日志位于其 `outputs/2026-10-09/`；本地未下载 checkpoint、数据或依赖缓存。

实际运行入口：

```bash
python experiments/record_environment.py
python experiments/mechanisms.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
python experiments/validate_results.py
python experiments/plot_results.py
python experiments/build_report.py
```

解释器实际为 `/root/miniconda3/bin/python`。网络代理只用于下载，无模型传入本地磁盘。

已处理问题：非交互 shell 未初始化 Python PATH；本地 GitHub 直连失败后使用现有代理；服务器 Drive/HF 直连受限后使用 SSH 代理转发；MNIST 原源返回 404 后由 torchvision 切换 S3；旧 checkpoint 缺少 `src.modeling` 模块路径，用新入口兼容注册解决。初次机制入口因 SciPy 尚未安装退出，安装完成后成功重跑。

保护耗时：{cm.get("protection_seconds", 0):.3f} s；作者恢复耗时：{cm.get("recovery_seconds", 0):.3f} s。参数、feature error 与 accuracy 计量对象不同，报告分别列出。
"""
    (root / "experiment_log.md").write_text(log, encoding="utf-8")
    speech = f"""# 组会讲稿（约 4 分钟）

老师，今天我把问题明确为：对已经训练好的模型，完全不再微调，通过后处理保护它的任务能力，避免未经授权的模型融合，同时保持模型单独使用的性能。

我先把论文和公开代码对应起来。AGA 对 Attention 的每个 head 分别做最小二乘，利用 pretrained 模型作为 anchor；MLP 则用余弦相似度和 Hungarian 找回神经元顺序。论文的 ARF 需要在微调时修改目标，所以不能直接用于我们这个限制下的问题。

今天已经完成机制实验和真实 CLIP 实验。机制实验用了三个固定种子，确认置换、QK 和 VO 配对变换都保持函数。MLP 的置换被完全找回。Attention 的恢复不是精确逆变换，而是把 anchor 投影到微调权重的列空间。这个结论论文附录已经给出，我做了不同可逆矩阵、条件数和数值求解方法的验证，没有把它当成新贡献。

真实实验使用 Task Arithmetic 官方 ViT-B/32 的 MNIST、SVHN checkpoint，测试了全部一万张 MNIST 和两万六千多张 SVHN 图像。MNIST 模型原始和保护后都是 {accuracy("A_unprotected", "MNIST")}，AGA 后是 {accuracy("C_AGA", "MNIST")}。在两任务融合系数 0.5 下，未保护融合为 {accuracy("D_unprotected_merge", "MNIST", 0.5)}，保护后直接融合为 {accuracy("E_protected_merge", "MNIST", 0.5)}，AGA 恢复后融合为 {accuracy("F_AGA_merge", "MNIST", 0.5)}。因此这类可逆保护能破坏直接融合，但对 AGA 没有稳健保护。

在老师提出的后处理方向上，我做了两个进一步检查。跨 head 置换能让原代码失效，但加上一个子空间 matching 就能全部找回，所以它只是实现缺口。

另一个候选利用 LayerNorm 的输入约束：它的输出落在一个已知仿射超平面上。我向后续 Linear 添加在这个超平面上恒为零的参数分量，并补偿 bias。这样无需训练、也无需训练数据，理论上保持函数，却可以改变 AGA 所用的参数列空间。这是今天实际实现的新切入口，不只是重复原有旋转。

不过这个空方向由公开 LayerNorm 参数决定，攻击者也能投影去掉它。所以我同时加入了 canonicalization 加 AGA 的攻击对照。这个实验无论结果是否支持候选，都能帮助我们区分“函数等价自由度”和“真正难恢复的保护”。具体结果在报告候选表中，没有完成的部分不会当作结论。

目前我不认为已经得到强防御。下一步建议先把这类易消去对称性的边界分析清楚，再研究允许极小 standalone 损失的解析扰动。如果允许少量公开未标注数据，可以利用低活跃方向设计扰动，但仍要避免回到完整微调，并把自适应攻击作为必要对照。我们先在两个任务上把证据做扎实，再扩展规模。
"""
    (root / "speech.md").write_text(speech, encoding="utf-8")
    print("REPORTS_WRITTEN", root, "rows", len(all_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    run(Path(parser.parse_args().output))
