# M5 BERT 动态量化报告

本报告由 `scripts/quantize_bert.py` 渲染，数字来自真实运行。

数据来源说明（如实记录，便于复核）：

- dev/test 指标、模型体积、量化层数、重新加载验证：来自 `python scripts/quantize_bert.py` 的完整运行，原始数据存于 `reports/m5-quantization-results.json`。
- 量化模型的 `models/bert_base_chinese_int8/metrics.json` 由该 JSON 回填生成（字段经 `src.models.artifact.write_metrics_json` 写入，保证与契约结构一致）；数值均为上述某次真实评测的实测结果，未做任何调整。
- 「两者都在 CPU 上」那张延迟表：由一次独立的 CPU 基准测量得到后并入结果 JSON，因为主评测流程记录的 fp32 延迟来自 CUDA，与 int8 的 CPU 延迟不可比。
- 可用 `python scripts/quantize_bert.py --render-only` 从已有 JSON 重新渲染本报告，不会重跑实验。

## 1. 优化方式与理由

采用 **动态量化（dynamic quantization）**：用 `torch.quantization.quantize_dynamic` 把全部 `nn.Linear` 的权重压成 int8，激活值在推理时按 batch 动态定标。

选择该方向的理由（三选一的取舍）：

| 方向 | 可行性 | 结论 |
| --- | --- | --- |
| 动态量化 | 无需重新训练，一行调用 | **采用** |
| 知识蒸馏 | 需从头训学生模型；M4 单次微调已耗 22.3 分钟 | 时间不足，放弃 |
| 模型剪枝 | 需再训练恢复精度，否则 Macro-F1 明显下降 | 时间不足，放弃 |

量化参数：方法 `torch.quantization.quantize_dynamic`，dtype `qint8`，共量化 **74 个 Linear 层**，耗时 3.3s。

## 2. 优化前后对比

| 指标 | 优化前（fp32） | 优化后（int8 动态量化） | 变化 |
| --- | ---: | ---: | ---: |
| 模型体积 | 390.17 MB | 145.58 MB | **↓ 62.7%** |
| dev Macro-F1 | 0.912276 | 0.904659 | -0.007617 |
| test Macro-F1 | 0.913847 | 0.900508 | -0.013339 |
| 量化 Linear 层数 | 0 | 74 | — |

### 推理成本（各自默认设备，**不可直接比较**）

| 指标 | 优化前 | 优化后 | 说明 |
| --- | ---: | ---: | --- |
| 单条延迟 (ms) | 20.325 | 121.624 | 中位数 |
| 批 32 延迟 (ms) | 237.708 | 5132.288 | 中位数 |
| 吞吐 (条/秒) | 134.6 | 6.2 | 由批 32 推出 |
| 测量设备 | cuda | cpu | **设备不同** |

上表**不能**用来评价量化效果：fp32 跑在 CUDA 上，int8 动态量化只能跑在 CPU 上（CUDA 没有对应的 int8 动态算子）。这张表反映的是 GPU 与 CPU 的差距，不是量化的收益。量化效果见下一节。

### 推理成本（**两者都在 CPU 上，可比**）

动态量化的部署场景是 CPU，因此把 fp32 也搬到 CPU 上再测一次，这样两个数字才归一到同一设备。

| 指标 | 优化前 fp32 (CPU) | 优化后 int8 (CPU) | 加速比 |
| --- | ---: | ---: | ---: |
| 单条延迟 (ms) | 70.88 | 64.92 | **1.09×** |
| 批 32 延迟 (ms) | 960.62 | 583.87 | **1.65×** |
| 吞吐 (条/秒) | 33.3 | 54.8 | **1.65×** |

## 3. 重新加载与真实推理验证

AGENTS.md §5.6 要求「不得把模型成功保存视为优化成功」。本节记录**重新加载量化产物**后的真实推理结果。

- 重新加载成功：`True`
- 重新加载后 dynamic quant Linear 层数：`74`（0 表示产物损坏）

| 探针文本 | 预测类别索引 | 置信度 |
| --- | ---: | ---: |
| 这本书内容很好，印刷清晰，非常值得推荐 | 0 | 0.999244 |
| 手机电池续航很差，用了一天就没电了 | 1 | 0.998585 |
| 酒店的床单有异味，前台态度也不好 | 9 | 0.999716 |

## 4. 加载方式的注意事项

量化产物**不能**用 `AutoModelForSequenceClassification.from_pretrained` 直接加载：该目录的 `model/` 下只有 `quantized_state_dict.pt`，没有 `model.safetensors`，`from_pretrained` 会抛 `OSError`。正确顺序是三步：

1. 按 `config.json` 构造 fp32 骨架；
2. **先**调用 `quantize_dynamic` 把子模块换成 dynamic quant 容器；
3. **再** `load_state_dict` 灌入量化权重。

第 2、3 步不能调换：量化 state_dict 含 `scale` / `zero_point` / `_packed_params.dtype` 等专用键，fp32 的 `nn.Linear` 没有对应容器，先灌权重会整批报「未知键」。

实现见 `src/models/optimization/quantization.py` 的 `load_quantized_model`；`src/inference/transformers_adapter.py` 会按 `metadata.json` 的 `quantization` 字段自动选择该路径，因此量化模型同样能走统一推理入口。

## 5. 结论与局限

**结论**：动态量化在**体积**和**CPU 吞吐**上收益明确，但**精度有可测量的损失**。

| 维度 | 结果 | 评价 |
| --- | --- | --- |
| 模型体积 | 390.17 MB → 145.58 MB（↓ 62.7%） | 收益显著 |
| test Macro-F1 | 0.913847 → 0.900508（-0.013339） | 有损失，需权衡 |
| CPU 批 32 吞吐 | 33.3 → 54.8 条/秒 | 收益明显 |

### 本实验的局限（如实记录，未验证项不冒充结论）

1. **精度损失未做补偿**。动态量化直接作用于 M4 权重，没有经过量化感知训练（QAT）或蒸馏补偿，因此 Macro-F1 必然下降。要缩小差距需要重新训练，不在本次时间预算内。
2. **峰值性能未测**。延迟是在本机 CPU 上、单进程、无并发条件下测得；未测多线程/多进程吞吐，也未与 ONNX Runtime 等专用推理引擎对比。
3. **量化 API 有弃用风险**。运行日志显示 `torch.quantize_per_tensor` 等接口已被 PyTorch 标记为 deprecated（见 pytorch/pytorch#184982），未来版本可能需要改用 `torchao` 的量化方案。
4. **未做端到端 Web 联调**。量化模型已能通过 `src.inference.load_predictor` 统一入口加载（本报告第 3 节已验证），但尚未替换到 Web 服务上实际演示。
5. **未测情感任务**。M5 的情感二分类由 HuMingChang 负责，本报告只覆盖类别分类。

### 部署建议

- **CPU-only 环境**：推荐使用 int8 量化模型，换取 ~63% 的体积缩减与更高的吞吐。
- **有 GPU 时**：优先用 fp32 原模型，GPU 上的 fp32 仍远快于 CPU 上的 int8（见第 2 节第一张表），量化在 CUDA 上没有收益。
- **对精度敏感**：用 fp32，或先做量化感知训练再量化。
