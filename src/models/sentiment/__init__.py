"""情感二分类基线（M5 扩展任务）的训练、产物写出与报告渲染。

模块划分与 ``src/models/baseline/`` 保持同构，便于对照阅读：

- :mod:`src.models.sentiment.estimators`：候选模型、训练与验证集选模；
- :mod:`src.models.sentiment.artifacts`：按契约写出模型产物；
- :mod:`src.models.sentiment.report`：渲染对比报告。

与类别任务共享的组件（不重复实现）：

- 特征：``src.features.text_features`` 的 TF-IDF 向量器（含清洗与分词）；
- 评估：``src.evaluation.metrics.compute_metrics_block`` 等；
- 推理：``src.inference.load_predictor``（任务无关，情感模型与类别模型共用）。
"""
