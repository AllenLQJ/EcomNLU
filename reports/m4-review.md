# M4 技术复核与最终验收记录

## 1. 复核对象

- 开发负责人：HongSenYuan
- M4 开发分支：`feature/fasttext`
- 已复核提交：`729999e`
- 技术复核人：HuMingChang
- 最终验收人：LanQvJian

HuMingChang 已完成技术复核，团队确认其复核范围包括：稳定数据集指纹、标签顺序、统一评估口径、验证集选模、测试集隔离，以及 M3/M4 同测试集对比。原复核分支未单独提交命令日志，因此本记录不补写未经保存的命令输出，只记录团队已确认的复核结论和最终验收的可重复证据。

## 2. LanQvJian 最终验收环境

- Conda 环境：`py310`
- Python：3.10.20
- 验收候选：`origin/hmc`（包含 `origin/feature/fasttext` 全部 M4 提交）
- 数据：从原始 62,774 条重新清洗和划分，得到 50,206 / 6,274 / 6,272

## 3. 已通过项目

- 数据契约：19 项通过、0 项失败、0 项跳过。
- 数据集稳定指纹相关回归测试通过；重新生成数据不会因 `created_at` 改变指纹。
- FastText 四变体真实训练冒烟成功。
- 四套冒烟模型均通过模型契约，各 15 项通过、0 项失败、0 项跳过。
- FastText 产物可由 `src.inference.load_predictor` 从磁盘重新加载并完成真实单条预测；每条概率和为 1.0。
- M4 训练代码、评估代码和脚本通过 `compileall`。
- BERT 报告记录了实际训练环境、超参数、逐 epoch 指标、`max_length=192` 的 token 分布依据、选模纪律和错误分析。
- BERT 完整产物通过模型契约校验，15 项通过、0 项失败、0 项跳过。
- BERT 可从磁盘重新加载，并在 CPU 上完成单条、双条及 32 条批量真实推理；概率和与标签均合法。
- 使用 BERT 真实产物完成 Web/API 联调：健康检查返回 200，模型已加载；预测接口返回 200、合法类别、置信度、Top-K 和耗时。
- 全量 dev（6,274 条）独立复评完成，Accuracy = 0.920784、Macro-F1 = 0.912276、Weighted-F1 = 0.920657，与提交记录一致。
- `py310` 环境下完整自动化测试结果为 135 项通过、2 项跳过、0 项失败。

## 4. 验收中发现并已修复

1. FastText 输出会做等比归一化，单元测试却使用精确浮点相等判断，导致 `0.6000000000000001 != 0.6`；已改用 `assertAlmostEqual`。
2. `reports/bert-training-report.md` 与 `reports/fasttext-report.md` 仍使用契约 v1.0 的旧字段 `split_manifest_sha256`；已统一改为 v1.0.1 的 `dataset_fingerprint`。

## 5. BERT 产物复验结论

HongSenYuan 提供的完整 `models/bert_base_chinese/` 已在 `py310` 环境验收。模型权重约 390 MB，按项目规则保留在本地并由 `.gitignore` 排除，不提交 Git。使用的核心命令为：

```powershell
python scripts/verify_model_contract.py models/bert_base_chinese
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split dev
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
```

其中模型契约、真实重载与推理、Web/API 联调以及完整 dev 复评均已通过。完整 test 独立复评启动后，最终验收人决定因 CPU 耗时停止继续运行；本次验收采用产物内已有的 test 真实评测记录（Accuracy = 0.919324、Macro-F1 = 0.913847），不声称 test 已由最终验收环境完整重跑。

综合代码、契约、自动化测试、真实模型重载、dev 指标复现和部署联调证据，LanQvJian 确认 M4 最终验收通过，可进入合并流程。
