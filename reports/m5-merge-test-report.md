# M5 合并测试报告

- 报告日期：2026-09-22
- 执行人：HuMingChang
- 被合并分支：`feature/bert-optimization`（HongSenYuan，tip `ad9197d`）
- 合并目标分支：`feature/sentiment-baseline`（本分支，合并后 tip `2ec24eb1`）
- 目的：验证 M5 两条并行支线（情感二分类基线 / BERT 优化）能否组合、是否存在冲突与报错
- 说明：本报告所有数值均来自本机真实运行，未运行的项目在第五节明确标注「未验证」

## 1. 结论摘要

| 检查项 | 结果 |
| --- | --- |
| 与 `feature/bert-optimization` 的冲突数 | **0** |
| 与最新 `main` 的冲突数 | **0** |
| 全量自动化测试 | **Ran 178 tests → OK**（退出码 0） |
| 模型契约校验 | 5 个模型各 **15/15 通过** |
| 数据契约校验 | **19/19 通过** |
| 端到端预测（情感 / 类别） | 均正常，指标与既有报告一致 |
| 编译与模块导入 | `compileall` 全通过，关键模块导入无异常 |
| 结论 | **两条 M5 支线可直接组合，未发现冲突，未发现报错** |

## 2. 合并对象与提交历史

合并后 `feature/sentiment-baseline` 的历史（双父合并提交共 2 个）：

```text
*   2ec24eb  merge: sync latest main (M5 contract v1.2 finalization) into feature/sentiment-baseline
|\            ├─ parent 1: 83bcfbf（本分支）
| |           └─ parent 2: 6bc3552（main，组长 M5 契约定稿）
| * 6bc3552  fix(m5): finalize sentiment contract and acceptance
* |   83bcfbf  merge: integrate M5 BERT optimization into feature/sentiment-baseline
|\ \            ├─ parent 1: 4714563（本分支原 tip）
| | |           └─ parent 2: ad9197d（feature/bert-optimization tip）
| * | ad9197d  fix(compare): show the measurement device and regenerate the comparison table
| * | ddab0ef  chore(optimization): align quantized model name with the team convention
| * | 6d6e6a3  Merge remote-tracking branch 'origin/main' into feature/bert-optimization
| * | 6691e63  docs(report): add M5 quantization report and daily report
| * | 9e6f257  feat(optimization): add BERT dynamic quantization with reload verification
* | | 4714563  merge: sync main (M4 acceptance + M2 review) into feature/sentiment-baseline
```

| 提交 | 类型 | 说明 |
| --- | --- | --- |
| `83bcfbf` | 合并提交 | 并入 BERT 优化（`ad9197d`） |
| `2ec24eb` | 合并提交 | 并入最新 `main`（`6bc3552`，M5 契约定稿） |

## 3. 冲突检查

采用**非破坏性预演**与**真实合并**双重验证：预演的 tree 哈希与真实合并产出的 tree 哈希逐位一致，说明预演结论可信。

| 合并 | merge-base | 预演 tree | 实际 tree | 冲突文件 | 是否一致 |
| --- | --- | --- | --- | --- | --- |
| `bert-optimization` → 本分支 | `9454abe` | `4d0745c2` | `4d0745c2` | 无 | 一致 |
| 最新 `main` → 本分支 | `4714563` | `3039a282` | `3039a282` | 无 | 一致 |

真实合并的回显均为 `Automatic merge went well`，`git diff --name-only --diff-filter=U` 输出为空。

**为什么没有冲突**：两条支线的改动文件集合完全不相交。

- 情感侧改动：`configs/base.yaml`、`docs/contracts.md`、`scripts/{predict,train_sentiment,verify_model_contract,...}.py`、`src/models/sentiment/*`、`src/inference/{contract,factory,sklearn_adapter}.py`、`tests/test_sentiment.py` 等。
- 优化侧改动：`src/models/optimization/quantization.py`、`scripts/quantize_bert.py`、`src/inference/transformers_adapter.py`、`scripts/compare_models.py`、`reports/m5-quantization-*`、`reports/model-comparison.*`。

唯一"同目录"的文件是 `src/inference/`，但情感侧改的是 `sklearn_adapter.py` / `factory.py` / `contract.py`，优化侧改的是 `transformers_adapter.py`，没有落到同一文件。

需要特别记录的一点：`main` 的 `6bc3552` 修改了本分支**自己的** M5 文件（`configs/base.yaml`、`scripts/predict.py`、`src/models/sentiment/artifacts.py` 等）与 3 个测试文件，因此本次同步是必要的——若不做这一步，后续测试覆盖的是过期状态。

## 4. 合并内容

### 4.1 并入的 BERT 优化（9 个文件，+1951 / −17）

| 文件 | 变化 | 内容 |
| --- | --- | --- |
| `src/models/optimization/quantization.py` | 新增 305 行 | int8 动态量化实现与 `load_quantized_model` |
| `scripts/quantize_bert.py` | 新增 522 行 | 量化实验与报告渲染入口 |
| `src/inference/transformers_adapter.py` | +26 / −4 | 按 `metadata.json` 的 `quantization` 字段切换量化加载路径，并强制量化产物在 CPU 推理 |
| `scripts/compare_models.py` | +19 / −2 | 对比表新增「设备」列，跨设备行给出不可比警告 |
| `reports/m5-quantization-report.md` | 新增 | M5 量化报告 |
| `reports/m5-quantization-results.json` | 新增 862 行 | 量化实验原始结果 |
| `reports/model-comparison.md` / `.json` | 修改 | 对比表刷新 |
| `reports/daily-reports/2026-09-21-HongSenYuan.md` | 新增 | 个人日报 |

### 4.2 并入的最新 main（22 个文件）

核心是组长 `6bc3552` 的 **M5 契约定稿**：

- `docs/contracts.md` 正式升到 **v1.2**：新增 `sentiment_tfidf_logreg` / `sentiment_tfidf_svm` / `sentiment_fasttext_word` / `sentiment_bert_base_chinese` 四个模型名；`label_mapping.json` 按 `metadata.task` 分两套规则；`confusion_matrix` 支持 2×2；新增可选字段 `per_product_category`；明确「原始 `label` 0 = 负向、1 = 正向」；统一 `load_predictor` 且不新增情感专用协议。
- `configs/base.yaml`、`src/evaluation/device.py`、`src/inference/*`、`src/models/sentiment/*` 相应微调。
- `scripts/{predict,predict_sentiment_svm,train_sentiment,verify_data_contract,verify_model_contract}.py`。
- `tests/{test_contract_validators,test_data_pipeline,test_sentiment}.py`（新增用例，测试总数 177 → 178）。

## 5. 测试与校验结果

全部在**合并后的工作区**执行，不是沿用既有结论。

### 5.1 全量自动化测试

```text
python -m unittest discover -s tests
→ Ran 178 tests in 20.414s
→ OK
```

| 阶段 | 测试数 | 结果 |
| --- | --- | --- |
| 合并 `bert-optimization` 后 | 177 | OK |
| 再同步最新 `main` 后 | 178 | OK |

### 5.2 契约校验

```text
python scripts/verify_model_contract.py
→ models/sentiment_tfidf_logreg  通过 15 项，失败 0 项
→ models/sentiment_tfidf_svm     通过 15 项，失败 0 项
→ models/tfidf_logreg            通过 15 项，失败 0 项
→ models/tfidf_rf                通过 15 项，失败 0 项
→ models/tfidf_svm               通过 15 项，失败 0 项
→ 结论：模型契约校验通过。

python scripts/verify_data_contract.py
→ 通过 19 项，失败 0 项，跳过 0 项
→ 结论：数据契约校验通过。
```

其中情感模型的两项关键断言在合并后依然成立：`label_mapping` 与 `task.sentiment_categories` 一致（情感空间，不是 10 类），且统一入口 `load_predictor` 可加载并推理。

### 5.3 端到端预测

| 场景 | 命令 | 结果 |
| --- | --- | --- |
| 情感单条（负向） | `predict_sentiment_svm.py --text "质量太差了，用两天就坏了"` | 负向 0.9997（Top-2：负向 0.9997 / 正向 0.0003） |
| 情感单条（正向） | `predict_sentiment_svm.py --text "非常好用，推荐购买"` | 正向 0.9917（Top-2：正向 0.9917 / 负向 0.0083） |
| 情感 dev | `predict_sentiment_svm.py --eval dev` | Macro-F1 **0.9335** / Acc 0.9335 |
| 情感 test | `predict_sentiment_svm.py --eval test` | Macro-F1 **0.9362** / Acc 0.9362 |
| 类别 test | `predict_svm.py --eval test` | Macro-F1 **0.8883** / Acc 0.8956 |

情感 test Macro-F1 0.9362 与 `reports/m5-sentiment-baseline.md` 一致；类别 test Macro-F1 0.8883 与 M3 选定的 `tfidf_svm` 一致。**合并未造成任何指标回归。**

### 5.4 编译与导入

```text
compileall(src / scripts / tests) → True
IMPORT OK   src.inference.transformers_adapter
IMPORT OK   src.models.optimization.quantization
IMPORT OK   src.inference.factory
IMPORT OK   src.web.service
IMPORT OK   src.models.sentiment.report
```

环境：conda env `AI`，torch 2.6.0+cu124、transformers 5.17.0、fasttext 可用。

## 6. 未验证项（如实标注）

以下内容本次**没有**验证，不作为结论：

1. **BERT 与量化产物的真实推理未在本机复现**。本机 `models/` 下只有 5 个 sklearn 模型（`tfidf_svm` / `tfidf_logreg` / `tfidf_rf` / `sentiment_tfidf_svm` / `sentiment_tfidf_logreg`），没有 `bert_base_chinese` 与 `bert_base_chinese_quantized`（模型权重被 `.gitignore` 排除）。因此：
   - `src/inference/transformers_adapter.py` 新增的量化加载分支**只做了导入与语法层面的检查**；
   - `reports/m5-quantization-report.md` 中的量化指标（体积 390.17 → 145.58 MB、test Macro-F1 0.913847 → 0.900508 等）**引用自 HongSenYuan 的报告，本机未复现**。
2. **未做量化模型的统一入口装载验证**。需要 `models/bert_base_chinese_quantized/` 才能执行，属 M6 部署阶段。
3. **未做 Web/API 联调**。`src/web/service.py` 的 `SENTIMENT_ENABLED` 仍为 `False`，前端尚未接入情感预测。
4. **未做跨设备性能复测**。量化报告已明确 fp32 与 int8 的延迟分属 cuda / cpu，本报告不重复引用其性能数字。
5. **未与 `main` 的最新一次文档提交合并**。远端 `main` 已前进到 `4c2ddc3 docs(presentation): assign demo assets and offline fallback`（仅 3 个文档文件，+229/−2），本分支尚未包含；预演显示无冲突。该提交不涉及代码与测试，不影响本报告结论。

## 7. 风险与建议

1. **演示环境缺 BERT 权重**。离线方案 A 级要求 `models/bert_base_chinese` 可加载，但本机不存在该目录。请 LanQvJian 在彩排前确认演示机上权重齐备，否则需降级到 B 级（`models/tfidf_svm`）。
2. **提交纪律**。按 `reports/final-presentation/task-assignment.md` 第八节，最终材料走 `docs/presentation-*` 个人分支，**不得直接向受保护的 `main` 推送**。本次全部工作在 `feature/sentiment-baseline` 分支上完成，未对 `main` 做任何提交。
3. **建议的后续步骤**：本分支已完整包含 `main`（`HEAD..main` 为空），所以组长可直接对本分支发起合并请求；或按 M5 验收要求先完成跨成员复核。

## 8. 复现命令

```powershell
# 1) 冲突预演（非破坏性，不改变工作区）
git merge-tree --write-tree --name-only 4714563 ad9197d
git merge-tree --write-tree --name-only 83bcfbf 6bc3552

# 2) 全量测试
python -m unittest discover -s tests

# 3) 契约校验
python scripts/verify_model_contract.py
python scripts/verify_data_contract.py

# 4) 端到端预测
python scripts/predict_sentiment_svm.py --text "质量太差了，用两天就坏了"
python scripts/predict_sentiment_svm.py --eval test
python scripts/predict_svm.py --eval test
```

> 本机提交说明：该仓库所在环境的 git 引用事务写入不可靠（`git commit` 可能产生孤立根提交），因此合并统一采用
> `git merge --no-commit` 解冲突 + `git write-tree` + `git commit-tree -p <父1> -p <父2>` 显式建对象的方式，
> 再直接写引用文件，最后用 `git log --oneline -3` 复核父提交。本报告中的 tree 哈希可用于独立核对。
