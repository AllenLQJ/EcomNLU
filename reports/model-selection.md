# 主部署模型决策（M4）

- 负责人：HongSenYuan
- 日期：2026-09-21
- 决策依据数据：`reports/model-comparison.md`（全部读取自各模型目录真实评测产出的 `metrics.json`）
- 选模指标：**验证集 Macro-F1**（AGENTS.md §6；类别不均衡下 Accuracy 只作辅助）

## 1. 决策

| 角色 | 模型目录 | 用途 |
| --- | --- | --- |
| **主部署模型** | `models/bert_base_chinese` | M6 Web/API 的默认模型，负责交互式单条评论预测 |
| 轻量备选模型 | `models/fasttext_word_tuned` | 演示机资源受限、需要高吞吐批处理或需要 CPU 低延迟时的替代模型 |

两个模型都满足 `docs/contracts.md` 第二章的产物契约（`scripts/verify_model_contract.py` 校验通过），
都由 `src.inference.load_predictor` 统一加载，**切换部署模型只需改配置里的模型目录，不需要改代码**。

## 2. 依据

dev 为选模划分，test 只在定稿后评测一次用于最终确认：

| 模型 | dev Macro-F1 | test Macro-F1 | test Acc | 模型大小 | 单条延迟 (CPU) | 吞吐 (CPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **bert_base_chinese** | **0.9123** | **0.9138** | 0.9193 | 390.2 MB | 343.90 ms | 3.5 条/秒 |
| fasttext_word_tuned | 0.8607 | 0.8655 | 0.8830 | 46.8 MB | 1.41 ms | 618.6 条/秒 |
| fasttext_char_tuned | 0.8546 | 0.8539 | 0.8747 | 3.6 MB | 0.06 ms | 21,720.2 条/秒 |
| fasttext_word | 0.8431 | 0.8478 | 0.8795 | 24.4 MB | 1.08 ms | 652.0 条/秒 |
| fasttext_char | 0.8290 | 0.8378 | 0.8702 | 1.9 MB | 0.23 ms | 5,896.4 条/秒 |

1. **精度差距是决定性的**：BERT 的 dev Macro-F1 比最强的 FastText 变体高 **5.16 个百分点**
   （0.9123 vs 0.8607），test 上高 **4.83 个百分点**（0.9138 vs 0.8655），
   且 dev 与 test 的排序一致，不存在「dev 选出来的模型在 test 上失效」的情况。
2. **成本差距同样真实**：BERT 体积是 `fasttext_word_tuned` 的 8.3 倍，
   CPU 单条延迟 244 倍、吞吐只有 1/175。这个差距没有被粉饰，而是作为「轻量备选」保留的理由。
3. **场景匹配**：M6 的演示是**单条评论交互式预测**，BERT 在 CPU 上单条约 0.34 s，
   用户感知完全可接受；反过来，若业务要求批量处理上万条评论或部署在不适合加载 390 MB 模型的环境，
   就应当换成 `fasttext_word_tuned`：牺牲约 4.8 个 Macro-F1 点，换来两个数量级的速度与 8 倍小的体积。

## 3. 生效前提与限制

1. **推理设备实测为 CPU**：两个模型的延迟与吞吐都测自项目 venv（`torch 2.14.0+cpu`），
   `metrics.json` 的 `cost.measured_on` 均为 `cpu`，因此上表的延迟可以直接横向比较。
   **GPU 上的推理延迟未实测**（属 M5 范围），不能假设它会更快。
2. **必须本地加载**：预训练权重、微调权重都在 `models/` 下（已被 `.gitignore` 排除），
   演示环境需随仓库分发/拷贝这些目录，不依赖联网下载。
3. **`max_length=192` 随模型走**：它由本数据集 token 分布实测确定（`reports/bert-token-lengths.json`），
   推理必须复用该值，改小会让长评论被截断、指标下降。
4. **已知短板**：主部署模型在「热水器」上的 test recall 只有 0.6316（F1 0.7347），
   在「平板 ↔ 洗发水」「平板 → 衣服」上存在集中混淆，详见 `reports/error-analysis.md`。
   这不是换模型能解决的问题，需要数据侧复核标注或调整类别定义。

## 4. 尚未完成、必须复核的事项

1. **传统基线尚未参评**：M3 的 TF-IDF 线性模型与随机森林还没有训练（对比表中标注为「未提供」）。
   AGENTS.md §12.4 要求所有模型在同一份无泄漏测试集上比较，本题结论目前是
   **「在已完成模型中」的最优**；基线补齐后需要重新跑一次 `scripts/compare_models.py` 确认结论，
   而不是假定它不会变。
2. **优化模型未做**（量化 / 蒸馏 / 剪枝，M5 范围）：没有「更小且精度接近 BERT」的候选，
   所以当前的成本-精度取舍只有 BERT 与 FastText 两个端点。
3. **未做类别加权实验**：`scripts/train_bert.py --class-weights` 已实现但未运行，
   小类（热水器）的召回是否因此改善尚未验证。

## 5. 复现命令

```powershell
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split dev
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
python scripts/compare_models.py --error-analysis-model models/bert_base_chinese --split test
python scripts/verify_model_contract.py models/bert_base_chinese
```
