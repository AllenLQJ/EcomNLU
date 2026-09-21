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

## 4. 验收中发现并已修复

1. FastText 输出会做等比归一化，单元测试却使用精确浮点相等判断，导致 `0.6000000000000001 != 0.6`；已改用 `assertAlmostEqual`。
2. `reports/bert-training-report.md` 与 `reports/fasttext-report.md` 仍使用契约 v1.0 的旧字段 `split_manifest_sha256`；已统一改为 v1.0.1 的 `dataset_fingerprint`。

## 5. 尚待完成的最终验收

Git 按项目规则不提交模型权重，本机目前没有 HongSenYuan 训练出的 `models/bert_base_chinese/`。因此以下项目尚未由最终验收人独立执行：

- BERT 模型契约校验；
- BERT 从磁盘重新加载；
- BERT 单条和批量真实推理；
- BERT 模型大小与 CPU 延迟复测；
- 使用 BERT 真实产物进行 Web/API 联调。

HongSenYuan 需要通过团队共享目录提供完整的 `models/bert_base_chinese/`。收到后执行：

```powershell
python scripts/verify_model_contract.py models/bert_base_chinese
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split dev
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
```

上述检查与 Web/API 单条预测全部通过后，才能把 M4 状态改为“已完成”并合入 `main`。
