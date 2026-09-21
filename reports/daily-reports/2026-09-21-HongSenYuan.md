# 日报：2026-09-21 - HongSenYuan

## 今日完成

- 完成内容：M5 的 BERT 动态量化闭环。把 M4 保存的 fp32 BERT 微调模型
  （`models/bert_base_chinese`）量化为 int8 动态量化模型
  （`models/bert_base_chinese_int8`），并完成优化前后的统一口径对比。
- 涉及文件：
  - `src/models/optimization/quantization.py`（新增，量化与量化产物加载）
  - `scripts/quantize_bert.py`（新增，一键量化 + 评测 + 报告）
  - `src/inference/transformers_adapter.py`（修改，统一入口支持量化产物）
  - `reports/m5-quantization-report.md`（新增，自动生成的报告）
  - `reports/m5-quantization-results.json`（新增，报告数据）
- Git 分支/提交：`feature/bert-optimization`（从 `feature/fasttext` 分出）

## 验证结果

- 执行命令：`python scripts/quantize_bert.py`
- 实际结果：

  | 指标 | fp32 | int8 量化 | 变化 |
  | --- | ---: | ---: | ---: |
  | 模型体积 | 390.17 MB | 145.58 MB | ↓ 62.7% |
  | dev Macro-F1 | 0.912276 | 0.904659 | -0.007617 |
  | test Macro-F1 | 0.913847 | 0.900508 | -0.013339 |
  | CPU 批 32 延迟 | 960.62 ms | 583.87 ms | 1.65× |
  | CPU 吞吐 | 33.3 条/秒 | 54.8 条/秒 | 1.64× |

  重新加载量化产物后真实推理成功：74 个 dynamic quant Linear 层全部恢复，
  三条探针文本预测正确（书籍 / 平板 / 酒店）。

- 尚未验证：
  1. 量化模型尚未替换到 Web 服务做端到端演示；
  2. 未做量化感知训练（QAT）补偿精度，Macro-F1 损失是"未补偿"的结果；
  3. 未测并发与多线程吞吐；
  4. M5 的情感二分类不在本分支，由 HuMingChang 负责。

## 遇到的问题

**问题 1：继承分支时选错了基点**

- 现象：新建 `feature/bert-optimization` 后，`scripts/` 下没有 `evaluate_model.py`、
  `train_bert.py`、`compare_models.py`，M4 的关键脚本全部缺失。
- 原因：建分支时的 `checkout feature/fasttext` 输出 `Already on 'feature/fasttext'`，
  但实际 HEAD 仍在 `main`（`54e7360`）。该输出具有误导性，未做独立核对。
- 解决方案：`git rev-list --left-right --count` 与 `git ls-files scripts/` 双重核对后，
  删除错误分支，改用 `git checkout -b feature/bert-optimization feature/fasttext`
  显式指定基点。
- 是否已解决：**已解决**，新分支基点为 `729999e`，M4 脚本齐全。

**问题 2：量化产物无法用 `from_pretrained` 加载**

- 现象：`AutoModelForSequenceClassification.from_pretrained("model")` 抛
  `OSError: Error no file named model.safetensors`。
- 原因：量化产物的 `model/` 下只有 `quantized_state_dict.pt`，没有标准权重文件，
  HF 的加载流程找不到权重入口。
- 解决方案：改为三步加载——按 `config.json` 构造 fp32 骨架 → **先** `quantize_dynamic`
  建立量化容器 → **再** `load_state_dict` 灌权重。
- 是否已解决：**已解决**。

**问题 3：先灌权重会整批报「未知键」**

- 现象：在 fp32 骨架上直接 `load_state_dict(量化权重)`，报 296 个未知键，
  例如 `bert.encoder.layer.0.attention.self.query.scale`、`zero_point`、
  `_packed_params.dtype`。
- 原因：这些是 dynamic quant 模块的专有键，fp32 的 `nn.Linear` 没有对应容器；
  `strict=False` 也无法挽救，因为容器类型根本不匹配。
- 解决方案：把 `quantize_dynamic` 提到 `load_state_dict` **之前**执行。
- 是否已解决：**已解决**，顺序已固化进代码并在注释中说明原因。

**问题 4：延迟对比出现设备不公平**

- 现象：首版报告显示量化后"批 32 慢 21 倍"（237.7ms → 5132.3ms），
  结论与"量化提速"完全相反。
- 原因：`evaluate_model_dir` 记录的 fp32 延迟来自 **CUDA**，而 int8 动态量化
  只能跑在 **CPU** 上（CUDA 无对应 int8 动态算子）。拿两个设备互比，
  测到的是 GPU 与 CPU 的差距，不是量化的收益。
- 解决方案：新增 `measure_cpu_latency`，把 fp32 也搬到 CPU 上重测，
  两者归一到同一设备后得到公平结论：批 32 加速 **1.65×**。
- 是否已解决：**已解决**，报告同时保留两张表并明确标注第一张不可用于评价量化。

## 明日计划

- 把量化模型接入 Web 服务做一次端到端演示验证（与 LanQvJian 联调）。
- 配合 HuMingChang 完成 M4/M5 的交叉复核，确认使用同一测试集指纹。
- 补充 M5 报告中的图表（体积/延迟对比柱状图）。

## 需要协助

- 需要 HuMingChang 复核：量化前后是否使用**同一份测试集指纹**与同一套指标口径。
- 需要 LanQvJian 确认：M6 演示是否需要在页面上同时展示 fp32 与 int8 的对比，
  以便量化成果在答辩中可见。
