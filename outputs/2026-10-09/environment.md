# 实验环境

日期：2026-10-09；运行位置：租用服务器；本地仅保存轻量文件。

| 项目 | 记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24 GB |
| Compute capability | 8.6 |
| Python | 3.12.3 |
| PyTorch | 2.3.0+cu121 |
| PyTorch CUDA runtime | 12.1 |
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

计时为实际 wall time，包含各脚本自身记录的加载/下载等阶段。显存峰值为 PyTorch 当前进程 `max_memory_allocated`，不是整机占用；P0 总峰值 11.31 MiB。基准过程有一次短暂并行候选启动，随后停止并排队；因此不将本次耗时作为严格性能比较。
