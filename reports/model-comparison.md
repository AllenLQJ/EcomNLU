# 模型对比

> 所有数值读取自各模型目录中真实评测产出的 `metrics.json`；未完成的模型标注为「未提供」，
> 不填估算值。选模依据是**验证集 Macro-F1**（AGENTS.md §6），测试集仅用于最终确认。

| 模型 | dev Macro-F1 | dev Acc | test Macro-F1 | test Acc | 模型大小 (MB) | 单条延迟 (ms) | 批量32 (ms) | 吞吐 (条/秒) | 设备 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sentiment_tfidf_svm | 0.9335 | 0.9335 | 0.9362 | 0.9362 | 23.6 | 1.37 | 11.74 | 2726.0 | cpu |
| sentiment_tfidf_logreg | 0.9308 | 0.9308 | 0.9369 | 0.9369 | 17.5 | 0.39 | 9.47 | 3379.9 | cpu |
| bert_base_chinese | 0.9123 | 0.9208 | 0.9138 | 0.9193 | 390.2 | 140.81 | 6875.75 | 4.7 | cpu |
| bert_base_chinese_quantized | 0.9043 | 0.9127 | 0.9026 | 0.9134 | 145.6 | 57.96 | 2042.36 | 15.7 | cpu |
| tfidf_svm | 0.8922 | 0.8954 | 0.8883 | 0.8956 | 106.0 | 2.88 | 18.18 | 1760.5 | cpu |
| tfidf_logreg | 0.8755 | 0.8878 | 0.8779 | 0.8895 | 44.9 | 12.47 | 26.00 | 1230.8 | cpu |
| fasttext_word_tuned | 0.8607 | 0.8797 | 0.8655 | 0.8830 | 46.8 | 0.21 | 16.69 | 1917.1 | cpu |
| fasttext_char_tuned | 0.8546 | 0.8757 | 0.8539 | 0.8747 | 3.6 | 0.04 | 1.83 | 17517.9 | cpu |
| fasttext_word | 0.8431 | 0.8757 | 0.8478 | 0.8795 | 24.4 | 0.21 | 17.46 | 1832.6 | cpu |
| fasttext_char | 0.8172 | 0.8572 | 0.8378 | 0.8702 | 1.9 | 0.10 | 5.93 | 5393.0 | cpu |
| tfidf_rf | 0.8109 | 0.8497 | 0.8240 | 0.8594 | 898.4 | 76.78 | 109.71 | 291.7 | cpu |

> 上表均为 **CPU** 口径。BERT 族在早期版本中曾用 CUDA 实测延迟，与 int8 量化的 CPU 结果
> 跨设备不可比（见 `reports/m5-quantization-report.md` 第 2 节），已统一改为 CPU 复测。
> 引用任一行时必须连同「设备」列一起引用，不得跨设备比较。

## 数值来源与一处复现性说明

除 FastText 外，上表所有数值均读取自**本机** `models/<模型名>/metrics.json`，
对应产物可用 `scripts/verify_model_contract.py` 重新加载验收。

**FastText 一族的四个数值来自 HongSenYuan 机器上的运行**（记录在其提交的
`reports/fasttext-summary.json` 与 `reports/fasttext-report.md`），因为该族模型产物
（`models/fasttext_*`）从未拷贝到撰写本报告的环境。summary 中的 `model_dir`、
`train_file` 与 `dev_file` 现为相对项目根的路径。此为已知的产物未同步问题，不是数据问题：
两处的 `dataset_fingerprint` 一致。

**复现性说明（如实记录，原因未证实）**：在本机按报告记录的同一条命令
（`python scripts/train_fasttext.py --autotune-duration 120 --evaluate-test`，本机
`fasttext-wheel 0.9.2` 与记录版本一致、训练数据指纹一致）重训后，数值与上表不符：

| 变体 | 上表 dev | 本机重训 dev | 差异 |
| --- | ---: | ---: | ---: |
| `fasttext_word_tuned` | 0.8607 | 0.8616 | +0.0009（复现良好） |
| `fasttext_char_tuned` | 0.8546 | 0.7354 | −0.1192 |
| `fasttext_word` | 0.8431 | 0.7184 | −0.1247 |
| `fasttext_char` | 0.8172 | 0.7132 | −0.1040 |

**假设（未证实）**：fastText 训练为异步 SGD，默认 `thread` 取 CPU 核数（本机 8 核），
默认参数只训练 5 个 epoch，对线程交错的敏感度高于 `word_tuned`（56 epoch，复现良好）。
本报告未对该假设做受控实验，故**上表仍采用原结果**，本机重训产物不作为对外报告数值。
若需在答辩中说明，应如实表述为「存在跨机器复现差异，原因待查」，不得声称已复现。

## 模型族覆盖情况

| 模型族 | 状态 |
| --- | --- |
| 传统基线 (TF-IDF) | tfidf_logreg、tfidf_svm |
| 随机森林 | tfidf_rf |
| FastText | fasttext_char、fasttext_char_tuned、fasttext_word、fasttext_word_tuned（数值来自 HongSenYuan 机器，见上节） |
| BERT | bert_base_chinese |
| 优化模型 | bert_base_chinese_quantized |

情感任务（M5 扩展任务）另有 `sentiment_tfidf_svm`、`sentiment_tfidf_logreg` 两个模型，
与主任务使用同一测试集划分口径，但标签集为 `负向` / `正向`，不与上表 10 类结果横向比较。
