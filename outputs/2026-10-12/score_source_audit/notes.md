# SCORE：代码核查，不是新防御

论文：[Bridging Domains through Subspace-Aware Model Merging](https://arxiv.org/html/2603.05768v1)，2026-03-06，正文第 5 节明确链接 [研究代码](https://github.com/VirtualSpaceman/score_cvpr26)。固定 commit 与三个源码指纹见 `source.json`。

已读 `score.py`、`task_vectors.py` 和合并 dispatcher。主方法 `ours_v1` 对各任务矩阵做 SVD，拼接主要左右方向并以极分解得到公共正交基，在该基中保留对角项、剪掉部分偏离均值的非对角项，再重建合并矩阵。合并本身没有梯度优化；专家来自此前训练。这是域泛化合并方法，不能当作本项目后处理安全防御或完整复现。

对本项目的启发是参数干扰可能被算子重表示和滤波缓解，不能只看直接 Task Arithmetic。当前 MNIST 候选已经被更便宜的 PCA 清理恢复，因此不为这个失败候选新增 SCORE 完整测试，也不把公共 SVD/极分解当创新。

本轮同时重读 [NUFILT v2 §3.4/A.1](https://arxiv.org/html/2509.21413v2)：其子空间对齐结论依赖低秩分解、误差界和谱间隙，并不保证校准 PCA 补空间对任务分类为零影响，更不是抵抗清理的安全定理。本项目后续需给出功能保留与清理代价的独立依据。

固定源码主函数已在 CPU 上通过一次 2 个任务、128×64 满秩合成矩阵的独立 NumPy 复算：相对误差 4.84e−14，bias 误差 1.11e−16。源码矩阵项求和，而非矩阵项取平均；不能默认直接套用本项目的 TA 系数。见 `primitive.json`。这只核验一个 primitive，不是完整 SCORE、CLIP 或安全实验。
