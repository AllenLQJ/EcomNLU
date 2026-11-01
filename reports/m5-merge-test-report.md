# M5 合并测试报告

- 报告日期：2026-09-22
- 执行人：HuMingChang（合并复核）、LanQvJian（独立产物验收，见第 9 节）
- 被合并分支：`feature/bert-optimization`（HongSenYuan，tip `ad9197d`）
- 合并目标分支：`feature/sentiment-baseline`（tip `e351c76`）
- 验收分支：`fix/m5-quantization-final`（基于 `feature/sentiment-baseline`，已并入最新 `main`，tip `b52d9c9`）
- 目的：验证 M5 两条并行支线（情感二分类基线 / BERT 优化）能否组合、是否存在冲突与报错
- 说明：本报告所有数值均来自本机真实运行，未运行的项目在第六节明确标注「未验证」

## 1. 结论摘要

| 检查项 | 结果 |
| --- | --- |
| 与 `feature/bert-optimization` 的冲突数 | **0** |
| 与最新 `main` 的冲突数 | **0**（`fix/m5-quantization-final` 落后 `main` 为 0） |
| 全量自动化测试 | **unittest 223 项 OK（跳过 5）/ pytest 220 passed, 3 skipped** |
| 模型契约校验 | 6 个模型各 **15/15 通过**（含量化产物） |
| 数据契约校验 | **19/19 通过** |
| 量化产物重载与真实预测 | **已独立验证**，见第 9 节 |
| Web/API 联调（量化类别模型 + 情感模型） | **10/10 通过** |
| 编译与空白检查 | `compileall` 通过，`git diff --check` 无空白错误 |
| 结论 | **两条 M5 支线可直接组合；量化产物已完成独立验收** |

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

## 6. 由「未验证」转为「已验证」的项目

本节原先是「未验证项」清单。经 `fix/m5-quantization-final` 分支上的独立验收（2026-09-22，LanQvJian，
Conda `py310`），原先 5 项中已有 4 项转为已验证，如实更新如下：

| 原未验证项 | 现状 |
| --- | --- |
| 1. BERT 与量化产物的真实推理未复现 | **已验证**，见第 9 节 |
| 2. 未做量化模型的统一入口装载验证 | **已验证**：`verify_model_contract.py` 15/15，含「统一入口可加载并推理」 |
| 3. 未做 Web/API 联调 | **已验证**：量化类别模型 + 情感模型，10/10 通过 |
| 4. 未做跨设备性能复测 | **未做**（本机无 CUDA）。本次测量全部在 CPU 上进行，fp32 与 int8 同设备可比，见第 9 节 |
| 5. 未与 `main` 最新提交合并 | **已验证**：`fix/m5-quantization-final` 落后 `origin/main`（`ce13fd4`）为 **0** |

仍未验证的事项：

- **跨设备（CUDA）性能对比**：本机无可用 NVIDIA 设备，只完成 CPU 内的 fp32 / int8 对比。
- **量化指标与 `reports/m5-quantization-report.md` 原值存在差异**，原因未最终确认（见第 9 节第 4 条）。

## 6.1 环境要求的一处修正（必须记录）

量化保存原先在 `py310` 中必然失败：

```text
torch.save(state_dict) → transformers/__init__.py 的模块级 __getattr__ 懒加载
  → 导入 transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3
  → import cv2 → RuntimeError: module compiled against ABI version 0x1000009
                 but this version of numpy is 0x2000000
```

**根因**：环境中的 `opencv-python 4.5.5.64` 按 NumPy 1.x ABI 编译，与 NumPy 2.2.6 不兼容。
**注意卸载不是解法**——该 `import cv2` 无保护，会变成 `ModuleNotFoundError`，同样失败。

**处理**：升级到 `opencv-python 4.14.0.94`（4.10 起才提供 NumPy 2 兼容 wheel），
并在 `requirements.txt` 中显式加入 `opencv-python>=4.10,<5.0` 及原因注释，使其可复现。

**本次验收记录的环境**：Python 3.10.20、numpy 2.2.6、torch 2.12.0+cpu、transformers 5.16.1、
opencv-python 4.14.0、scikit-learn 1.7.2、设备 CPU。

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

## 9. 量化产物的独立验收记录（LanQvJian，2026-09-22）

在 `fix/m5-quantization-final` 分支上，用 Conda `py310` 重建并独立验收量化产物。
所有数字来自实际运行，未沿用既有报告结论。

### 9.1 产物

```text
models/bert_base_chinese_quantized/
├── model/quantized_state_dict.pt    149,068.8 KB（145.6 MB）
├── model/config.json
├── tokenizer/{tokenizer.json,tokenizer_config.json}
├── label_mapping.json
├── metadata.json
└── metrics.json
```

### 9.2 模型契约

```text
python scripts/verify_model_contract.py models/bert_base_chinese_quantized
→ 通过 15 项，失败 0 项，跳过 0 项
```

其中「统一入口可加载并推理」为真实执行，非跳过。

### 9.3 真实重载与预测

```text
python scripts/predict.py --model models/bert_base_chinese_quantized \
  --text "这本书内容很好，印刷清晰，非常值得推荐" --top-k 3
→ 书籍 0.9992（Top-3：书籍 0.9992 / 平板 0.0001 / 洗发水 0.0001）

python scripts/predict.py --model models/bert_base_chinese_quantized \
  --text "酒店的床单有异味，前台态度也不好" --top-k 3
→ 酒店 0.9997（Top-3：酒店 0.9997 / 衣服 0.0001 / 蒙牛 0.0000）
```

### 9.4 量化前后对比（本次实测）

| 指标 | fp32 源模型 | int8 量化模型 |
| --- | ---: | ---: |
| 体积 | 409,126,369 字节（390.2 MB） | 152,647,957 字节（145.6 MB） |
| dev Macro-F1 | 0.912276 | 0.904306 |
| test Macro-F1 | 0.913847 | 0.902649 |
| CPU 单条延迟 | 59.44 ms | 33.11 ms |
| CPU 批 32 延迟 | 721.22 ms | 463.02 ms |
| 量化 Linear 层 | — | 74 |

结论：体积降至 37.3%，CPU 单条延迟降至 55.7%，dev Macro-F1 下降 0.0080、test 下降 0.0112。

**与原报告的差异必须记录**：fp32 指标与 `reports/m5-quantization-report.md`
**完全一致**（0.912276 / 0.913847），说明数据划分与评估口径相同；但**量化后指标不同**
（原报告 dev 0.904659、test 0.900508）。最可能的原因是 **torch 版本差异**——原报告环境为
torch 2.6.0+cu124，本次为 2.12.0+cpu，动态量化的行为会随版本变化。该差异**尚未最终确认**，
引用量化指标时必须连同运行环境一起标注，不得把两个环境的数值混写成同一次实验。

### 9.5 Web/API 联调

以量化模型为主类别模型、`sentiment_tfidf_svm` 为情感模型启动服务，**10/10 通过**：

| 检查 | 结果 |
| --- | --- |
| `GET /api/health` | 200，`model_name=bert_base_chinese_quantized`，`sentiment_enabled=true` |
| 「这本书内容很好…」 | 书籍 0.9992 / 情感 正向 / 67.3 ms |
| 「酒店的床单有异味…」 | 酒店 0.9997 / 情感 负向 / 41.6 ms |
| 「这款手机很好用…」top_k=5 | 返回 5 条，概率降序 |
| 空输入 / 超长 / 非法 top_k | 分别返回 `EMPTY_REVIEW` / `REVIEW_TOO_LONG` / `INVALID_TOP_K` |

### 9.6 本次修复的两个产物缺陷

| 缺陷 | 修复 |
| --- | --- |
| `metadata.model_name` 为 `bert_base_chinese`，与目录名 `bert_base_chinese_quantized` 不一致（契约 §2.3 要求一致） | `save_quantized_model` 改为取目标目录名 |
| `quantization.source_model_dir` 写入本机绝对路径（AGENTS.md §7.1 禁止） | 新增 `_as_project_relative()`，转为相对项目根的 POSIX 路径；项目外路径写成「项目外」 |

同时新增 `tests/test_quantization.py`（9 项）——量化模块此前无任何测试覆盖。
