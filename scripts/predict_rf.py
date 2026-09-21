"""用 **tfidf_rf**（TF-IDF + 随机森林）做预测与打分。

本脚本是 ``scripts/predict.py`` 的薄封装，只把默认模型固定为 ``models/tfidf_rf``，
其余参数与功能完全一致——**不复制任何推理或评估逻辑**，避免四个脚本各写一套而逐渐分叉。

Usage:
    # 单条评论：给出类别、置信度与 Top-K 概率
    python scripts/predict_rf.py --text "这本书内容不错，作者水平很高"

    # 批量：从 CSV 读取一列评论
    python scripts/predict_rf.py --file reviews.csv --text-column review

    # 对某个划分跑评估分数（Macro-F1、Accuracy、逐类别指标）
    python scripts/predict_rf.py --eval test
    python scripts/predict_rf.py --eval dev --output reports/score-rf-dev.json

    # 临时改用别的模型（覆盖默认值）
    python scripts/predict_rf.py --model models/tfidf_svm --text "酒店服务很差"

输入文件：
    - ``--text`` 直接给一条评论；``--file`` 给含评论列的 CSV；
    - ``--eval`` 读取 ``data/processed/{dev,test}.csv``（需先完成 M2 划分）。

输出文件：
    无。结果打印到标准输出；``--eval`` 加 ``--output`` 可另存指标 JSON。

模型简介与使用提醒：
    随机森林在本数据集的稀疏高维 TF-IDF 特征上**三项指标均低于两个线性模型**，
    且体积约 900 MB、单条延迟约 72 ms，**不适合部署**。
    它在本项目中的作用是「与「投满分」项目对应的对照模型」，
    用来证明「随机森林并不必然是文本分类的最佳传统模型」。
    加载时会占用较多内存与时间，请勿在演示现场临时切换。
    完整对比见 ``reports/m3-baseline-comparison.md``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import main  # noqa: E402


MODEL_DIR = "models/tfidf_rf"


if __name__ == "__main__":
    raise SystemExit(main(default_model=MODEL_DIR))
