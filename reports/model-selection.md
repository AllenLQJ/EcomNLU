# 主部署模型决策（M4）

- 负责人：HongSenYuan
- 日期：2026-09-21（已合并 main 的 M3 基线与契约 v1.0.1 后更新）
- 选模指标：**验证集 Macro-F1**（AGENTS.md §6；类别不均衡下 Accuracy 只作辅助）
- 数据来源（数值都来自真实运行，来源分别标注）：
  - M4（FastText、BERT）：本分支的 `reports/model-comparison.md`，读取各模型 `metrics.json`；
  - M3（TF-IDF 三个基线）：队友的 `reports/m3-baseline-comparison.md`（本机 `models/tfidf_*`
    目录未拷贝，因此没有并入同一张自动生成的表）。
- 两个阶段的产物使用**同一份数据**：数据集指纹 `c9b76e66e05c96dd…`（契约 §2.3，由
  `data/processed` 的 train/dev/test 三个 CSV 内容哈希算出），因此是同一测试集上的比较。

## 1. 决策

| 角色 | 模型目录 | 依据 |
| --- | --- | --- |
| **主部署模型** | `models/bert_base_chinese` | 验证集 Macro-F1 最高（0.9123），测试集 0.9138 同样最高；CPU 单条 183.8 ms，交互式单条预测可接受 |
| 低成本备选 | `models/tfidf_svm` | M3 选定基线：测试集 0.8883，106 MB、单条 3.57 ms、2,228 条/秒；精度只低 2.55 点，速度高约 51 倍 |
| 轻量备选 | `models/fasttext_word_tuned` | 46.8 MB、单条 0.21 ms；精度比主模型低 4.83 点，作为最小体积方案 |

所有候选都满足 `docs/contracts.md` 第二章的产物契约，并由 `src.inference.load_predictor`
统一加载（本分支的 5 个 M4 模型已用契约校验器逐个验证通过），**切换部署模型只需改模型目录**。

## 2. 依据：跨阶段对比（数值来源见「来源」列）

| 模型 | dev Macro-F1 | test Macro-F1 | test Acc | 模型大小 | 单条延迟 (CPU) | 吞吐 (CPU) | 来源 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **bert_base_chinese** | **0.9123** | **0.9138** | 0.9193 | 390.2 MB | 183.80 ms | 3.9 条/秒 | M4 `model-comparison.md` |
| tfidf_svm | 0.8923 | 0.8883 | 0.8956 | 106.0 MB | 3.57 ms | 2,228 条/秒 | M3 `m3-baseline-comparison.md` |
| tfidf_logreg | 0.8721 | 0.8777 | 0.8887 | 45.0 MB | 0.86 ms | 4,204 条/秒 | M3 `m3-baseline-comparison.md` |
| fasttext_word_tuned | 0.8607 | 0.8655 | 0.8830 | 46.8 MB | 0.21 ms | 1,917 条/秒 | M4 `model-comparison.md` |
| fasttext_char_tuned | 0.8546 | 0.8539 | 0.8747 | 3.6 MB | 0.04 ms | 17,518 条/秒 | M4 `model-comparison.md` |
| fasttext_word | 0.8431 | 0.8478 | 0.8795 | 24.4 MB | 0.21 ms | 1,833 条/秒 | M4 `model-comparison.md` |
| fasttext_char | 0.8290 | 0.8378 | 0.8702 | 1.9 MB | 0.03 ms | 25,179 条/秒 | M4 `model-comparison.md` |
| tfidf_rf | 0.8165 | 0.8281 | 0.8635 | 900.5 MB | 77.56 ms | 294 条/秒 | M3 `m3-baseline-comparison.md` |

1. **精度格局清楚**：主模型 BERT 的 dev Macro-F1 比第二名 `tfidf_svm` 高 **2.00 个百分点**，
   比最强的轻量方案 `fasttext_word_tuned` 高 **5.16 个百分点**；dev 与 test 的排序完全一致，
   没有出现「验证集选出来、测试集反转」的情况。
2. **随机森林垫底**（0.8165 / 0.8281，900 MB、77.6 ms），与 AGENTS.md §5.2 的提醒一致：
   不能假定随机森林是文本分类的最优传统模型，本项目实测 SVM 明显更好。
3. **成本取舍有中间档**：`tfidf_svm` 用 106 MB 和 3.57 ms 换到 0.8883，比 BERT 只低 2.55 点、
   快约 51 倍。因此「按场景选模型」比「选一个万能模型」更合理：
   - 交互式单条演示（M6）：BERT，0.18 s/条在体验上无感；
   - 批量打分 / 低配机器：`tfidf_svm`（或更极端的 `fasttext_char`，1.9 MB、1.7 万条/秒）；
   - 精度优先但体积受限：`fasttext_word_tuned`。
4. **短板一致，说明是数据问题而不是模型问题**：三个模型族在「热水器」上的召回都是最低
   （BERT 0.6316、tfidf_svm 0.5439、FastText 族同类问题），且都集中混淆「洗发水 ↔ 平板」「平板 → 衣服」。
   主部署模型在这个短板类别上依然是最好的，因此这是换模型无法解决的、属于标注与类别定义的问题
   （详见 `reports/error-analysis.md`）。

## 3. 生效前提与限制

1. **推理延迟为 CPU 实测**：M4 各模型用项目 venv（`torch 2.14.0+cpu`）测量，M3 各模型同样是 CPU；
   M4 与 M3 现在共用同一个测量函数 `src.evaluation.metrics.measure_inference_cost`
   （1 次预热 + 20 次单条取中位数、批量 32 条重复 5 次取中位数），因此两列可以直接比较。
   **GPU 上的推理延迟未测量**（属 M5 范围），不能假设一定更快。
2. **必须本地加载**：权重都在 `models/` 下且被 `.gitignore` 排除，演示环境要提前拷贝这些目录，不依赖联网下载。
3. **`max_length=192` 随 BERT 模型走**：它由本数据集 token 分布实测确定
   （`reports/bert-token-lengths.json`，覆盖率 96.29%），推理必须复用该值。
4. **小类风险**：若业务上特别关心「热水器」，当前所有模型都不理想，需要数据侧复核或调整类别定义。

## 4. 尚未完成、必须复核的事项

1. **同一张自动生成的表还未包含 M3**：本机没有 `models/tfidf_{logreg,svm,rf}`（`.gitignore` 排除，
   需本地训练）。运行下面的命令后，`reports/model-comparison.md` 会自动把 M3 三个模型并入同一张表，
   届时本节的跨阶段汇总可以退化为单一来源：

   ```powershell
   python scripts/train_baseline.py        # 约 6-7 分钟（本机 RF 训练较慢，可能更久）
   python scripts/compare_models.py --error-analysis-model models/bert_base_chinese --split test
   ```

   > 本次合并时尝试在本机重跑 M3 训练，RF（200 棵树 × 40 万维）阶段超过工具时限、被中断两次，
   > 因此第 2 节的 M3 数值直接引用队友已发布的报告（已标注来源，未做任何改写或推算）。
2. **优化模型未做**（量化 / 蒸馏 / 剪枝，M5 范围）：没有「更小且精度接近 BERT」的候选，
   成本-精度取舍目前只有 BERT / SVM / FastText 三个档位。
3. **未做类别加权实验**：`scripts/train_bert.py --class-weights` 已实现但未运行，
   小类（热水器）召回是否因此改善尚未验证。

## 5. 复现命令

```powershell
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split dev
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
python scripts/compare_models.py --error-analysis-model models/bert_base_chinese --split test
python scripts/verify_model_contract.py models/bert_base_chinese
```
