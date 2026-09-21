"""按接口契约写出情感模型的产物。

模块职责：
    把「已训练的模型 + 特征向量器 + 标签映射 + 元数据 + 指标」按 ``docs/contracts.md``
    §2.1 规定的目录结构写到 ``models/<model_name>/``，使 ``src/inference/`` 的统一
    入口与 ``scripts/verify_model_contract.py`` 能够加载并校验。

主要输入与输出：
    输入：已训练模型、已拟合向量器、情感标签顺序、元数据与指标字典。
    输出：``models/<model_name>/`` 下的 ``model.joblib``、``vectorizer.joblib``、
        ``label_mapping.json``、``metadata.json``、``metrics.json``。

关键设计理由与约束：

1. **情感模型的 ``label_mapping.json`` 就是情感映射本身。**
   ``categories`` 为 ``["负向", "正向"]``，取值来源是 ``configs/base.yaml`` 的
   ``task.sentiment_categories``。这是契约 v1.2 §2.1 的规定：
   §2.1 现行那句「必须与 ``data/processed/label_mapping.json`` 逐字节相同」
   只能适用于主任务——情感模型的标签集与 10 个商品类**必然不同**，
   套用该句会让任何情感模型在校验第一步就失败（草案「更正二」已说明这一点）。
   因此本模块不复制商品映射，而是按任务写出对应的那一份。

2. **``metadata.json`` 的 ``task`` 必须是 ``sentiment_classification``。**
   校验器据此选择比对基准（主任务比对数据契约、情感任务比对配置），
   M5 的模型对比也据此区分两个任务。

3. **``dataset_fingerprint`` 与类别模型同算法。**
   直接复用 :func:`src.models.baseline.artifacts.dataset_fingerprint`：
   指纹只由 train/dev/test 三个 CSV 的内容决定，因此情感模型与 M3/M4 的类别模型
   指纹**完全相同**，这正是「使用同一份无泄漏测试集」的硬证据（契约 §2.3）。

4. **写出前先自检，不留半成品。**
   模型缺 ``predict_proba``、``classes_`` 与情感标签不一致、向量器未拟合、
   元数据缺字段，都在这里直接报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.evaluation.metrics import COST_KEYS, METRICS_BLOCK_KEYS, compute_metrics_block
from src.features.text_features import save_vectorizer
from src.models.baseline.artifacts import (
    METADATA_REQUIRED_KEYS,
    dataset_fingerprint,
    sha256_of,
    write_json,
)


__all__ = [
    "SENTIMENT_TASK",
    "SENTIMENT_METADATA_REQUIRED_KEYS",
    "build_sentiment_label_mapping",
    "build_sentiment_metadata",
    "build_sentiment_metrics",
    "build_per_product_category_metrics",
    "write_sentiment_model_body",
    "write_sentiment_metrics_file",
    "write_sentiment_artifact",
    "dataset_fingerprint",
    "sha256_of",
]


# metadata.json 中标识情感任务的取值，与 scripts/verify_model_contract.py 的
# ALLOWED_TASKS 一致。
SENTIMENT_TASK = "sentiment_classification"

# 与类别模型完全相同的必需字段集合（契约 §2.3）。显式重新导出一次，
# 使调用方不必同时 import 两个 artifacts 模块。
SENTIMENT_METADATA_REQUIRED_KEYS: tuple[str, ...] = METADATA_REQUIRED_KEYS


def build_sentiment_label_mapping(labels: Sequence[str]) -> dict[str, Any]:
    """构造情感模型的 ``label_mapping.json`` 内容。

    结构严格与契约 §1.3 的三字段同形（``categories`` / ``label2id`` / ``id2label``），
    不添加额外键——这样既有的标签映射校验逻辑与推理适配器可以原样复用。

    Args:
        labels: 情感标签顺序，取自 ``configs/base.yaml`` 的 ``task.sentiment_categories``。

    Returns:
        可直接序列化的情感标签映射字典。

    Raises:
        ValueError: 标签数不是 2 时抛出。
    """
    pair = [str(label) for label in labels]
    if len(pair) != 2:
        raise ValueError(
            f"情感标签必须恰好 2 个（0 负向、1 正向），实际为 {pair}"
        )
    return {
        "categories": pair,
        "label2id": {name: index for index, name in enumerate(pair)},
        "id2label": {str(index): name for index, name in enumerate(pair)},
    }


def build_sentiment_metadata(
    *,
    model_name: str,
    created_at: str,
    random_seed: int,
    processed_dir: Path,
    train_rows: int,
    hyperparameters: Mapping[str, Any],
    device: str = "cpu",
    notes: str = "",
) -> dict[str, Any]:
    """构造情感模型的 ``metadata.json`` 内容。

    Args:
        model_name: 情感模型名，如 ``sentiment_tfidf_logreg``。
        created_at: 带时区的 ISO 8601 创建时间。
        random_seed: 训练使用的随机种子。
        processed_dir: ``data/processed`` 目录，用于实时计算数据集指纹（契约 §2.3）。
        train_rows: 实际参与训练的行数（不含验证集与测试集）。
        hyperparameters: 完整到可复现训练的超参数。
        device: 实际训练设备。scikit-learn 没有 GPU 实现，固定为 ``cpu``。
        notes: 备注。

    Returns:
        可直接序列化的元数据字典。

    Raises:
        FileNotFoundError: 三个划分文件不齐时抛出——此时无法证明数据版本，不应产出模型。
    """
    return {
        "model_name": str(model_name),
        "task": SENTIMENT_TASK,
        "framework": "scikit-learn",
        "framework_version": _sklearn_version(),
        "created_at": created_at,
        "random_seed": int(random_seed),
        "dataset_fingerprint": dataset_fingerprint(processed_dir),
        "train_rows": int(train_rows),
        # 契约 §2.3：max_length 仅 BERT 类模型填实际值，其余必须为 null
        "max_length": None,
        "device": str(device),
        "hyperparameters": dict(hyperparameters),
        "notes": str(notes),
    }


def _sklearn_version() -> str:
    """返回当前 scikit-learn 版本。"""
    import sklearn

    return sklearn.__version__


def build_per_product_category_metrics(
    frame: pd.DataFrame,
    *,
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str],
    category_column: str,
) -> dict[str, dict[str, Any]]:
    """统计每个商品类别内部的情感表现。

    M5 任务要求报告「各商品类别内的情感表现」。整体指标会把 10 个类别混在一起，
    掩盖掉「模型在酒店评论上判断情感很准、在书籍评论上明显更差」这类差异，
    因此必须按类别拆开看。本字段是 M5 报告要求的附加指标，
    **不属于契约 §2.4 的必填结构**，因此放在顶层独立字段中，不改动 ``dev`` / ``test`` 块。

    Args:
        frame: 评估集合（通常是测试集）的 DataFrame，需含类别列。
        true_labels: 与 ``frame`` 同序的真实情感类名。
        predicted_labels: 与 ``frame`` 同序的预测情感类名。
        labels: 情感标签顺序。
        category_column: 商品类别列名。

    Returns:
        ``{商品类别: 指标块}``，指标块结构与 ``metrics.json`` 的单集合块一致，
        另加 ``support``、``positive_rate``（该类别真实正例占比）与
        ``predicted_positive_rate``。类别顺序按出现顺序，保证多次运行可 diff。

    Raises:
        ValueError: 三个序列长度不一致时抛出。
        KeyError: 缺少商品类别列时抛出。
    """
    if not (len(frame) == len(true_labels) == len(predicted_labels)):
        raise ValueError(
            "frame、真实标签与预测标签长度必须一致："
            f"{len(frame)} / {len(true_labels)} / {len(predicted_labels)}"
        )
    if category_column not in frame.columns:
        raise KeyError(f"缺少商品类别列：{category_column}")

    label_list = [str(label) for label in labels]
    positive_name = label_list[-1]

    results: dict[str, dict[str, Any]] = {}
    category_values = frame[category_column].astype(str).tolist()
    for category in dict.fromkeys(category_values):
        positions = [
            index for index, value in enumerate(category_values) if value == category
        ]
        if not positions:
            continue
        group_true = [true_labels[index] for index in positions]
        group_pred = [predicted_labels[index] for index in positions]
        block = compute_metrics_block(group_true, group_pred, categories=label_list)
        block["support"] = len(positions)
        block["positive_rate"] = round(
            sum(1 for value in group_true if value == positive_name) / len(positions), 6
        )
        block["predicted_positive_rate"] = round(
            sum(1 for value in group_pred if value == positive_name) / len(positions), 6
        )
        results[str(category)] = block
    return results


def build_sentiment_metrics(
    *,
    model_name: str,
    labels: Sequence[str],
    dev_block: Mapping[str, Any],
    test_block: Mapping[str, Any] | None,
    cost: Mapping[str, Any],
    per_product_category: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造情感模型的 ``metrics.json`` 内容并校验结构。

    Args:
        model_name: 情感模型名。
        labels: 情感标签顺序。
        dev_block: 验证集指标块，不得为 ``None``（契约要求验证集指标必须先产出）。
        test_block: 测试集指标块；``None`` 表示尚未评估，序列化为 ``null``。
        cost: 推理成本块。
        per_product_category: 各商品类别内的情感表现，M5 报告要求；
            未评估时传 ``None``。

    Returns:
        可直接序列化的指标字典。

    Raises:
        ValueError: 缺字段、标签顺序不符或混淆矩阵尺寸不符时抛出。
    """
    if dev_block is None:
        raise ValueError("验证集指标块不得为空：契约要求 dev 必须先产出")

    label_list = [str(label) for label in labels]
    for name, block in (("dev", dev_block), ("test", test_block)):
        if block is None:
            continue
        missing = [key for key in METRICS_BLOCK_KEYS if key not in block]
        if missing:
            raise ValueError(f"{name} 指标块缺少字段：{missing}")
        if set(block["per_class"]) != set(label_list):
            raise ValueError(
                f"{name}.per_class 的键必须是 {label_list}，实际 {sorted(block['per_class'])}"
            )
        matrix = block["confusion_matrix"]
        if len(matrix) != len(label_list) or any(
            len(row) != len(label_list) for row in matrix
        ):
            raise ValueError(
                f"{name}.confusion_matrix 必须是 {len(label_list)}×{len(label_list)}"
            )

    missing_cost = [key for key in COST_KEYS if key not in cost]
    if missing_cost:
        raise ValueError(f"cost 缺少字段：{missing_cost}")

    return {
        "model_name": str(model_name),
        "task": SENTIMENT_TASK,
        "categories": label_list,
        "dev": dict(dev_block),
        "test": dict(test_block) if test_block is not None else None,
        "cost": dict(cost),
        "per_product_category": (
            None
            if per_product_category is None
            else {str(key): dict(value) for key, value in per_product_category.items()}
        ),
    }


def write_sentiment_model_body(
    *,
    models_dir: Path,
    model_name: str,
    estimator: Any,
    vectorizer: Any,
    labels: Sequence[str],
    metadata: Mapping[str, Any],
) -> Path:
    """写出情感模型的模型本体、标签映射与元数据（不含 ``metrics.json``）。

    单独提供这一步，是因为推理成本必须**通过统一入口加载后实测**
    （契约 §2.4 要求实测而非估计），而加载又要求产物已就位。

    Args:
        models_dir: ``models/`` 目录。
        model_name: 模型名，用于子目录名。
        estimator: 已训练的情感模型。
        vectorizer: 已拟合的向量器。
        labels: 情感标签顺序。
        metadata: :func:`build_sentiment_metadata` 的结果。

    Returns:
        写出的模型目录路径。

    Raises:
        ValueError: 模型缺少 ``predict_proba``、``classes_`` 与情感标签不一致、
            或元数据缺字段时抛出。
    """
    import joblib

    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"{type(estimator).__name__} 没有 predict_proba，无法满足契约 §2.5 的概率输出要求"
        )
    missing = [key for key in SENTIMENT_METADATA_REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata 缺少必需字段：{missing}")

    label_list = [str(label) for label in labels]
    model_dir = Path(models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    classes = [str(value) for value in getattr(estimator, "classes_", [])]
    if set(classes) != set(label_list):
        raise ValueError(
            f"模型的 classes_ 必须是 {label_list}，实际为 {classes}。"
            "训练标签请用 src.models.sentiment.estimators.translate_labels 转换。"
        )

    joblib.dump(estimator, model_dir / "model.joblib")
    save_vectorizer(vectorizer, model_dir / "vectorizer.joblib")
    write_json(build_sentiment_label_mapping(label_list), model_dir / "label_mapping.json")
    write_json(dict(metadata), model_dir / "metadata.json")
    return model_dir


def write_sentiment_metrics_file(model_dir: Path, metrics: Mapping[str, Any]) -> Path:
    """写出情感模型的 ``metrics.json``。

    Args:
        model_dir: 模型目录。
        metrics: :func:`build_sentiment_metrics` 的结果。

    Returns:
        写出的 ``metrics.json`` 路径。
    """
    path = Path(model_dir) / "metrics.json"
    write_json(dict(metrics), path)
    return path


def write_sentiment_artifact(
    *,
    models_dir: Path,
    model_name: str,
    estimator: Any,
    vectorizer: Any,
    labels: Sequence[str],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """一次性写出一个情感模型目录的全部契约产物。

    Args:
        models_dir: ``models/`` 目录。
        model_name: 模型名。
        estimator: 已训练模型。
        vectorizer: 已拟合向量器。
        labels: 情感标签顺序。
        metadata: 元数据字典。
        metrics: 指标字典。

    Returns:
        写出的模型目录路径。
    """
    model_dir = write_sentiment_model_body(
        models_dir=models_dir,
        model_name=model_name,
        estimator=estimator,
        vectorizer=vectorizer,
        labels=labels,
        metadata=metadata,
    )
    write_sentiment_metrics_file(model_dir, metrics)
    return model_dir


def load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件，供报告脚本回读产物。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
