"""用 **sentiment_tfidf_svm**（TF-IDF + 校准后的线性 SVM，情感二分类）做预测与打分。

本脚本与 ``scripts/predict_svm.py`` 同构，是 ``scripts/predict.py`` 的薄封装：
只把默认模型固定为 ``models/sentiment_tfidf_svm``、默认任务固定为情感分类，
其余参数与功能完全一致——**不复制任何推理或评估逻辑**，避免各个预测脚本
各写一套而逐渐分叉（见 ``scripts/predict.py`` 的模块文档字符串）。

与类别模型脚本的关键差别（都由共享入口内部处理，这里无需重复实现）：
    - 标签空间是 ``["负向", "正向"]``，取自 ``configs/base.yaml`` 的
      ``task.sentiment_categories``，而不是 10 个商品类别；
    - 划分 CSV 中的真实标签在 ``label`` 列，取值是原始 **0/1**
      （0=差评/负向、1=好评/正向，数据契约 §1.2），必须翻译成中文类名
      才能与预测结果比对；
    - 不输出「词面捷径」提示，该提示只对商品类别任务成立。

Usage:
    # 不带参数：默认对验证集 dev.csv 做评估，先列逐条预测预览再给真实分数
    python scripts/predict_sentiment_svm.py
    python scripts/predict_sentiment_svm.py --preview 0     # 只看分数，不列预览

    # 单条评论：给出情感、置信度与两类概率
    python scripts/predict_sentiment_svm.py --text "质量太差了，用两天就坏了"
    python scripts/predict_sentiment_svm.py --text "非常好用，推荐购买"

    # 批量：从 CSV 读取一列评论（CSV 自带 label 列时会显示预测对错与准确率）
    python scripts/predict_sentiment_svm.py --file data/processed/dev.csv --limit 20
    python scripts/predict_sentiment_svm.py --file reviews.csv --text-column review

    # 对某个划分跑评估分数（Macro-F1、Accuracy、正负两类指标）
    python scripts/predict_sentiment_svm.py --eval test
    python scripts/predict_sentiment_svm.py --eval dev --output reports/score-sentiment-dev.json

    # 临时改用另一个情感模型（覆盖默认值）
    python scripts/predict_sentiment_svm.py --model models/sentiment_tfidf_logreg --text "还行吧"

输入文件：
    - **不带参数时读 ``data/processed/dev.csv``**：它自带 ``label`` 列，
      因此一条命令即可同时自检「预测是否正常」与「真实分数是多少」；
    - ``--text`` 直接给一条评论；``--file`` 给含评论列的 CSV；
    - ``--eval`` 读取 ``data/processed/{dev,test}.csv``（需先完成 M2 划分）。

输出文件：
    无。结果打印到标准输出；``--eval`` 加 ``--output`` 可另存指标 JSON。

模型简介：
    本模型是 M5 依据**验证集 Macro-F1** 选出的情感二分类模型
    （验证集 0.9335 / 测试集 0.9362，见 ``reports/m5-sentiment-baseline.md``）。
    两点提醒：
    它与候选 ``sentiment_tfidf_logreg`` 的性能差距在噪声内（测试集 Macro-F1
    只差 0.0005，约 2 条样本），之所以交付本模型只是因为按契约用验证集选模；
    体积约 23.6 MB、单条延迟约 1.49 ms，均高于逻辑回归，若更看重部署成本应选 logreg。
    已知失败模式（否定作用域、反讽、评价对象不明确）见
    ``reports/m5-sentiment-error-analysis.md``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import TASK_SENTIMENT, main  # noqa: E402


MODEL_DIR = "models/sentiment_tfidf_svm"
TASK = TASK_SENTIMENT


if __name__ == "__main__":
    raise SystemExit(main(default_model=MODEL_DIR, default_task=TASK))
