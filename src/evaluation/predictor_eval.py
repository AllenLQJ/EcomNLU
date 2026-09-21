"""任意契约模型目录的评测入口。

``docs/contracts.md`` §2.4 要求所有模型给出结构一致的指标，并且**通过统一推理入口
评测**，这样"离线评测口径"与"线上服务口径"必然相同。本模块因此不感知具体框架：
它只调用 ``src.inference.load_predictor``，用哪种模型由 ``metadata.json`` 决定。

测试集纪律：``split="test"`` 只允许在模型定稿后调用，且不应参与任何选模决策
（AGENTS.md §6、契约 §2.4）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.metrics import (
    compute_metrics_block,
    directory_size_bytes,
    measure_inference_cost,
)
from src.models.artifact import PROCESSED_DIR, read_split, write_metrics_json

SUMMARY_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)

# 指标精度：统一四舍五入到 6 位小数，保证重新评测时与已产出的 metrics.json 逐位一致。
METRIC_DIGITS = 6


def summarize(block: dict[str, Any]) -> dict[str, Any]:
    """只保留汇总指标，便于写入报告而不必携带混淆矩阵。"""
    return {key: block[key] for key in SUMMARY_KEYS}


def round_metrics_block(block: dict[str, Any], digits: int = METRIC_DIGITS) -> dict[str, Any]:
    """把指标块中的浮点数统一舍入到指定位数。

    指标本身只由 ``src.evaluation.metrics.compute_metrics_block`` 计算一次
    （AGENTS.md §12.4 要求所有模型同一口径）；本函数只做精度归一，让 M4 期间产出的
    6 位小数指标在重新评测后仍然逐位可比。
    """
    rounded: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        elif key == "per_class" and isinstance(value, dict):
            rounded[key] = {
                name: {
                    field: round(number, digits) if isinstance(number, float) else number
                    for field, number in values.items()
                }
                for name, values in value.items()
            }
        else:
            rounded[key] = value
    return rounded


def model_size_bytes(model_dir: Path) -> int:
    """统计模型本体的体积，排除 label_mapping/metadata/metrics 等说明文件。

    体积对比（AGENTS.md §5.6）关心的是"要加载的模型有多大"，因此按模型本体类型
    分别统计：FastText 取 ``model.bin``，Transformers 取 ``model/`` 目录。
    """
    model_dir = Path(model_dir)
    for name in ("model.bin", "model.joblib"):
        candidate = model_dir / name
        if candidate.is_file():
            return directory_size_bytes(candidate)
    model_subdir = model_dir / "model"
    if model_subdir.is_dir():
        return directory_size_bytes(model_subdir)
    return directory_size_bytes(model_dir)


def evaluate_model_dir(
    model_dir: Path,
    categories: Sequence[str],
    split: str,
    processed_dir: Path | None = None,
    cost_sample_size: int = 64,
) -> dict[str, Any]:
    """通过统一入口在指定划分上评测模型，并把结果合并进 ``metrics.json``。

    Args:
        model_dir: 契约模型目录（含模型本体、label_mapping.json、metadata.json）。
        categories: 类别顺序，决定 per_class 与混淆矩阵的行列顺序。
        split: ``dev`` 或 ``test``。
        processed_dir: 契约数据目录，默认 ``data/processed``。
        cost_sample_size: 测量延迟所用的样本数。

    Returns:
        该划分的指标块（含 per_class 与 confusion_matrix）。
    """
    from src.inference.factory import load_predictor

    rows = read_split(split, categories, processed_dir)
    texts = [text for text, _ in rows]
    category_list = [str(name) for name in categories]
    y_true = [category_list[label] for _, label in rows]

    predictor = load_predictor(model_dir)
    started = time.perf_counter()
    probabilities = predictor.predict_proba(texts)
    elapsed = time.perf_counter() - started
    # 契约 §2.5：预测类别即最高概率类别。用同一份概率推出预测，避免两套逻辑不一致。
    labels = [max(item, key=item.get) for item in probabilities]

    block = round_metrics_block(
        compute_metrics_block(y_true, labels, categories=category_list)
    )
    block["elapsed_seconds"] = round(elapsed, 3)

    cost = measure_inference_cost(
        predictor.predict_proba,
        probe_texts=texts[:cost_sample_size],
        model_size_bytes=model_size_bytes(model_dir),
        device=str(getattr(predictor, "device", "cpu")),
    )

    metrics_path = Path(model_dir) / "metrics.json"
    existing: dict[str, Any] = {}
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
    write_metrics_json(
        model_dir=Path(model_dir),
        model_name=existing.get("model_name", Path(model_dir).name),
        categories=categories,
        dev=block if split == "dev" else existing.get("dev"),
        cost=cost,
        test=block if split == "test" else existing.get("test"),
    )
    return block
