# EuroSAT 划分与 AGA 权重身份

本轮没有取得 AGA 原始 EuroSAT 索引或其专家权重指纹，Table 6 精确复现仍未完成。没有创建随机划分、下载全数据或把公开镜像称为论文原测试集。

- 固定 AGA commit `fb61429f` 的仓库树只有代码，没有 checkpoint、训练入口或划分清单；本次读取公开 issue 也没有得到这些文件。入口仍使用本地 `finetuned.pt`，八专家 TA 系数为 0.2。
- [Task Arithmetic 原讨论](https://github.com/mlfoundations/task_vectors/issues/1)在此前已核对：作者明确 21600/2700/2700 随机划分，但没有给出原索引或种子。本次再次打开没有新增证据，不把它计作新进展。
- [AdaMerging 固定 README](https://github.com/EnnengYang/AdaMerging/blob/cea1165ad900113b8a52ae5ff190626576635f9b/README.md)指向 Task Arithmetic 官方权重，并提供已处理数据的百度网盘链接。本次页面获取失败；未读取归档，不能推断其 EuroSAT 测试集身份。
- [tanganke/eurosat](https://huggingface.co/datasets/tanganke/eurosat/tree/43750fc422c7fbd4289e2df3f7473617a5937acc)固定版本的公开元数据为训练 21600、测试 2700。查看器仅给 image/label 两列，两个预览没有原文件名；相同样本数不足以证明划分相同。没有读取完整 Parquet。服务器直接请求在 20 秒连接超时后停止，未修改环境。

来源指纹、公开仓库树和元数据见 `source.json`、`splits.json`、`preview.json`、`parquet.json`。预览图片的临时签名 URL 已去除；没有把图片存到本地。

另对固定 AGA 与 Task Arithmetic 源码做了直接比较：`templates.py` 在统一换行后逐字节相同；两处 head 构建函数、完整 `ClassificationHead` 类及分类器前向的 AST 相同。八任务模板逐一渲染也一致，见 `head_source_audit.json`。因此当前公开代码没有给出另一套 head 公式或提示词来解释表格差异。**这不证明已保存 head 的张量、类别顺序、依赖版本或专家权重相同。**没有重建、替换 head，也没有新增 GPU 评价。

下一步应取得可核对的样本索引或归档出处，再做逐文件身份检查。若只能取得另一个公开划分，可另立具名基准，但不能替代 AGA 原表。继续从作者入口的 head、预处理与参数来源检查差异，避免重复搜索同一划分讨论或按测试数值调系数。
