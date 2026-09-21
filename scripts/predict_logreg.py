"""用 **tfidf_logreg**（TF-IDF + 逻辑回归）做预测与打分。

本脚本是 ``scripts/predict.py`` 的薄封装，只把默认模型固定为 ``models/tfidf_logreg``，
其余参数与功能完全一致——**不复制任何推理或评估逻辑**，避免四个脚本各写一套而逐渐分叉。

Usage:
    # 不带参数：默认对验证集 dev.csv 做评估，先列逐条预测预览再给真实分数
    python scripts/predict_logreg.py
    python scripts/predict_logreg.py --preview 0      # 只看分数，不列预览

    # 单条评论：给出类别、置信度与 Top-K 概率
    python scripts/predict_logreg.py --text "这本书内容不错，作者水平很高"

    # 批量：从 CSV 读取一列评论（CSV 自带标签列时会显示预测对错与准确率）
    python scripts/predict_logreg.py --file data/processed/dev.csv --limit 20
    python scripts/predict_logreg.py --file reviews.csv --text-column review

    # 对某个划分跑评估分数（Macro-F1、Accuracy、逐类别指标）
    python scripts/predict_logreg.py --eval test
    python scripts/predict_logreg.py --eval dev --output reports/score-logreg-dev.json

    # 临时改用别的模型（覆盖默认值）
    python scripts/predict_logreg.py --model models/tfidf_svm --text "酒店服务很差"

输入文件：
    - **不带参数时读 ``data/processed/dev.csv``**：它自带真实标签，
      因此一条命令即可同时自检「预测是否正常」与「真实分数是多少」；
    - ``--text`` 直接给一条评论；``--file`` 给含评论列的 CSV；
    - ``--eval`` 读取 ``data/processed/{dev,test}.csv``（需先完成 M2 划分）。

输出文件：
    无。结果打印到标准输出；``--eval`` 加 ``--output`` 可另存指标 JSON。

模型简介：
    逻辑回归是三个基线中**体积最小（约 45 MB）、单条延迟最低（约 0.6 ms）**的模型，
    适合对响应速度要求高的场景；Macro-F1 略低于线性 SVM。
    完整对比与取舍依据见 ``reports/m3-baseline-comparison.md``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import main  # noqa: E402


MODEL_DIR = "models/tfidf_logreg"


if __name__ == "__main__":
    raise SystemExit(main(default_model=MODEL_DIR))
