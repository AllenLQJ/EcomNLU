"""统一的模型评估与成本测量。

对应 ``docs/contracts.md`` §2.4：所有模型都必须产出结构一致的 ``metrics.json``，
其中的数值必须来自真实运行，不得估算或编造（AGENTS.md §6、§7.4）。

本模块只依赖 numpy 之外的标准库与 scikit-learn，训练脚本与优化脚本共用，
避免每个模型各写一份指标计算而产生口径差异。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

# 契约 §2.4 规定的指标名，顺序与 configs/base.yaml 的 evaluation.report_metrics 对齐。
REPORT_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)

COST_KEYS = (
    "model_size_bytes",
    "single_sample_latency_ms",
    "batch_32_latency_ms",
    "throughput_per_sec",
    "measured_on",
    "measured_at",
)

BEIJING = timezone(timedelta(hours=8))


def now_iso() -> str:
    """返回北京时间 ISO 8601 时间戳（秒精度）。

    统一用带时区的北京时间，便于与团队其它报告的时间字段对照。
    """
    return datetime.now(BEIJING).replace(microsecond=0).isoformat()


def directory_size_bytes(path: Path) -> int:
    """递归统计目录或文件的字节数。"""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    categories: Sequence[str],
) -> dict[str, Any]:
    """计算契约要求的全部指标。

    ``categories`` 的顺序即 ``label_mapping.json`` 的顺序，直接决定 ``per_class``
    与 ``confusion_matrix`` 的行列顺序（契约 §2.4：行是真实类别、列是预测类别）。
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true 与 y_pred 长度不一致：{len(y_true)} vs {len(y_pred)}")
    if not y_true:
        raise ValueError("评估样本为空，无法计算指标")

    labels = list(range(len(categories)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        category: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, category in enumerate(categories)
    }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def measure_cost(
    predict_batch: Callable[[list[str]], list[Any]],
    sample_texts: Sequence[str],
    model_size_bytes: int,
    device: str,
    warmup: int = 3,
    repeats: int = 5,
) -> dict[str, Any]:
    """实测模型的体积、单条延迟、批量延迟与吞吐量（契约 §2.4 的 ``cost``）。

    ``predict_batch`` 接收文本列表并返回预测结果，由调用方绑定到具体模型。
    先做 ``warmup`` 次预热再计时，避免把首次调用的初始化开销记入延迟
    （这正是本项目早期一次 BERT 计时探针得到不可信结果的原因）。
    """
    if not sample_texts:
        raise ValueError("sample_texts 不能为空")

    for _ in range(max(0, warmup)):
        predict_batch([sample_texts[0]])

    single_started = time.perf_counter()
    for index in range(repeats):
        predict_batch([sample_texts[index % len(sample_texts)]])
    single_ms = (time.perf_counter() - single_started) / repeats * 1000

    batch = [sample_texts[index % len(sample_texts)] for index in range(32)]
    predict_batch(batch)
    batch_started = time.perf_counter()
    for _ in range(repeats):
        predict_batch(batch)
    batch_seconds = (time.perf_counter() - batch_started) / repeats

    return {
        "model_size_bytes": int(model_size_bytes),
        "single_sample_latency_ms": round(single_ms, 3),
        "batch_32_latency_ms": round(batch_seconds * 1000, 3),
        "throughput_per_sec": round(len(batch) / batch_seconds, 2),
        "measured_on": device,
        "measured_at": now_iso(),
    }


def write_metrics_json(
    model_dir: Path,
    model_name: str,
    categories: Sequence[str],
    dev: dict[str, Any],
    cost: dict[str, Any],
    test: dict[str, Any] | None = None,
) -> Path:
    """写出契约 §2.4 规定的 ``metrics.json``。

    ``test`` 默认必须为 ``None``：选模阶段只允许使用验证集，测试集只在最终确认时
    评估一次（契约 §2.4、AGENTS.md §6）。
    """
    payload = {
        "model_name": model_name,
        "categories": list(categories),
        "dev": dev,
        "test": test,
        "cost": cost,
    }
    missing = [key for key in COST_KEYS if key not in cost]
    if missing:
        raise ValueError(f"cost 缺少字段：{missing}")
    for name, block in (("dev", dev), ("test", test)):
        if block is None:
            continue
        absent = [key for key in REPORT_METRICS if key not in block]
        if absent:
            raise ValueError(f"{name} 指标缺少字段：{absent}")

    path = Path(model_dir) / "metrics.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_metrics(model_dir: Path) -> dict[str, Any]:
    """读取模型目录下的 ``metrics.json``。"""
    path = Path(model_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))
