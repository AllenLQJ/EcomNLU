# 模型对比

> 所有数值读取自各模型目录中真实评测产出的 `metrics.json`；未完成的模型标注为「未提供」，
> 不填估算值。选模依据是**验证集 Macro-F1**（AGENTS.md §6），测试集仅用于最终确认。

| 模型 | dev Macro-F1 | dev Acc | test Macro-F1 | test Acc | 模型大小 (MB) | 单条延迟 (ms) | 批量32 (ms) | 吞吐 (条/秒) | 设备 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bert_base_chinese | 0.9123 | 0.9208 | 0.9138 | 0.9193 | 390.2 | 20.76 | 239.98 | 133.3 | cuda |
| bert_base_chinese_quantized | 0.9047 | 0.9139 | 0.9005 | 0.9109 | 145.6 | 121.62 | 5132.29 | 6.2 | cpu |
| fasttext_word_tuned | 0.8607 | 0.8797 | 0.8655 | 0.8830 | 46.8 | 0.21 | 16.69 | 1917.1 | cpu |
| fasttext_char_tuned | 0.8546 | 0.8757 | 0.8539 | 0.8747 | 3.6 | 0.04 | 1.83 | 17517.9 | cpu |
| fasttext_word | 0.8431 | 0.8757 | 0.8478 | 0.8795 | 24.4 | 0.21 | 17.46 | 1832.6 | cpu |
| fasttext_char | 0.8172 | 0.8572 | 0.8378 | 0.8702 | 1.9 | 0.10 | 5.93 | 5393.0 | cpu |

> **注意**：上表的延迟与吞吐分别在不同设备上实测（本机出现的设备：cpu、cuda），**跨设备行不可直接比较**。
> 比较优化前后性能时，必须先把两者归一到同一设备（见 `reports/m5-quantization-report.md` 的 CPU 对比表）。

## 模型族覆盖情况

| 模型族 | 状态 |
| --- | --- |
| 传统基线 (TF-IDF) | 未提供（本机 `models/` 下没有该族模型目录：未训练或未拷贝） |
| 随机森林 | 未提供（本机 `models/` 下没有该族模型目录：未训练或未拷贝） |
| FastText | fasttext_char、fasttext_char_tuned、fasttext_word、fasttext_word_tuned |
| BERT | bert_base_chinese |
| 优化模型 | bert_base_chinese_quantized |
