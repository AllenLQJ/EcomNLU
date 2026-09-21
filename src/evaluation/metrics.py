"""统一的模型评估：指标、混淆矩阵、推理成本与对比报告。

模块职责：
    为所有模型提供**同一套**评估口径，产出契约 §2.4 规定的 ``metrics.json`` 结构，
    并渲染 AGENTS.md §6 要求的模型对比报告。三个模型只有在完全相同的指标实现与
    同一份测试集上评估，比较才有意义（AGENTS.md §12.4）。

主要输入与输出：
    输入：真实标签、预测标签、类别的固定顺序、模型目录与预测回调。
    输出：符合契约结构的指标字典；Markdown 对比报告。

关键设计理由与约束：

1. **类别顺序一律显式传入。** 混淆矩阵的行列顺序、``per_class`` 的键、
   报告里的表格顺序全部取自 ``label_mapping.json`` 的 ``categories``。
   依赖 sklearn 的内部排序会让不同模型的矩阵行列语义不一致，而混淆矩阵
   一旦顺序错位，结论就会完全错误且不易察觉。

2. **推理成本通过统一预测接口实测。** ``docs/contracts.md`` §2.4 要求
   ``cost`` 实测而非估计；这里传入的是 ``predict_proba`` 回调，因此测到的是
   包含向量化在内的端到端推理耗时，而不是只测模型前向。

3. **测试集纪律由调用流程保证。** 本模块不区分「选模」与「确认」，
   但 :func:`render_comparison_report` 会把选模依据（验证集）与最终确认
   （测试集）分开呈现，避免读者误以为测试集参与了选模。
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


BEIJING_TIMEZONE = timezone(timedelta(hours=8))

# 契约 §2.4 规定的指标块字段，缺一不可。
METRICS_BLOCK_KEYS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "per_class",
    "confusion_matrix",
)

# 契约 §2.4 规定的成本字段。
COST_KEYS: tuple[str, ...] = (
    "model_size_bytes",
    "single_sample_latency_ms",
    "batch_32_latency_ms",
    "throughput_per_sec",
    "measured_on",
    "measured_at",
)

# 延时测量的重复次数与批大小。取中位数以降低偶发抖动的影响。
LATENCY_REPEATS = 20
BATCH_SIZE = 32
BATCH_REPEATS = 5


def beijing_timestamp() -> str:
    """返回带时区的 ISO 8601 北京时间字符串。

    返回：
        形如 ``2026-09-21T14:30:00+08:00`` 的时间字符串。
    """
    return datetime.now(BEIJING_TIMEZONE).isoformat(timespec="seconds")


def compute_metrics_block(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    categories: Sequence[str],
) -> dict[str, Any]:
    """计算契约要求的单个集合的指标块。

    Args:
        y_true: 真实类别名序列。
        y_pred: 预测类别名序列。
        categories: 类别顺序，决定 ``per_class`` 的键与混淆矩阵的行列顺序。

    Returns:
        含 ``accuracy``、``macro_precision``、``macro_recall``、``macro_f1``、
        ``weighted_f1``、``per_class`` 与 ``confusion_matrix`` 的字典。

    Raises:
        ValueError: 两个序列长度不一致，或出现 ``categories`` 之外的类别时抛出。
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"真实标签与预测标签长度不一致：{len(y_true)} vs {len(y_pred)}")

    category_list = [str(c) for c in categories]
    known = set(category_list)
    unknown_true = sorted(set(y_true) - known)
    unknown_pred = sorted(set(y_pred) - known)
    if unknown_true or unknown_pred:
        raise ValueError(
            f"出现未在 categories 中的类别：真实 {unknown_true}，预测 {unknown_pred}"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=category_list, zero_division=0
    )
    per_class = {
        category: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, category in enumerate(category_list)
    }

    matrix = confusion_matrix(y_true, y_pred, labels=category_list)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            np.mean([per_class[c]["precision"] for c in category_list])
        ),
        "macro_recall": float(np.mean([per_class[c]["recall"] for c in category_list])),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=category_list, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=category_list, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": [[int(value) for value in row] for row in matrix.tolist()],
    }


def directory_size_bytes(path: Path) -> int:
    """统计一个目录（或单个文件）的总字节数。

    Args:
        path: 模型目录或文件路径。

    Returns:
        该路径下所有文件的大小之和；路径不存在时返回 0。
    """
    if path.is_file():
        return int(path.stat().st_size)
    if not path.is_dir():
        return 0
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def measure_inference_cost(
    predict_proba: Callable[[list[str]], list[dict[str, float]]],
    *,
    probe_texts: Sequence[str],
    model_size_bytes: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """实测推理成本，产出契约 §2.4 的 ``cost`` 块。

    Args:
        predict_proba: 统一预测接口的 ``predict_proba``，测的是包含特征化的端到端耗时。
        probe_texts: 用于测量批量的文本；不足 :data:`BATCH_SIZE` 条时会重复填充。
        model_size_bytes: 模型目录的总字节数。
        device: 实际测量设备，本项目为 ``"cpu"``。

    Returns:
        含 ``model_size_bytes``、``single_sample_latency_ms``、
        ``batch_32_latency_ms``、``throughput_per_sec``、``measured_on``、
        ``measured_at`` 的字典。
    """
    texts = [str(text) for text in probe_texts] or ["手机很好用"]
    single_text = [texts[0]]

    # 先跑一次预热，避免把首次调用的惰性初始化（jieba 词典、sklearn 的校验路径）
    # 计入耗时，那样测到的是冷启动而不是稳态性能。
    predict_proba(single_text)

    single_samples: list[float] = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        predict_proba(single_text)
        single_samples.append((time.perf_counter() - start) * 1000.0)

    batch = (texts * (BATCH_SIZE // len(texts) + 1))[:BATCH_SIZE]
    batch_samples: list[float] = []
    for _ in range(BATCH_REPEATS):
        start = time.perf_counter()
        predict_proba(batch)
        batch_samples.append((time.perf_counter() - start) * 1000.0)

    single_median = statistics.median(single_samples)
    batch_median = statistics.median(batch_samples)

    return {
        "model_size_bytes": int(model_size_bytes),
        "single_sample_latency_ms": round(single_median, 3),
        "batch_32_latency_ms": round(batch_median, 3),
        "throughput_per_sec": round(len(batch) / (batch_median / 1000.0), 1),
        "measured_on": device,
        "measured_at": beijing_timestamp(),
    }


def category_name_shortcut_shares(
    frame: "pd.DataFrame",
    *,
    text_column: str,
    category_column: str,
    threshold: float = 0.95,
) -> dict[str, float]:
    """统计每个类别中「评论文本直接包含类别名」比例超过阈值的类别。

    用途：识别指标被**词面捷径**抬高的类别。若某类别的评论几乎都会直接写出类别名
    （典型是品牌类，例如「蒙牛」），分类器只需匹配这一个词就能全对，
    该类的 Precision / Recall 便无法代表模型对同类商品文本的泛化能力，
    报告中必须如实说明，否则会把「记住了品牌名」误读成「理解了商品类别」。

    参数：
        frame：含文本列与类别列的 DataFrame（通常传测试集）。
        text_column：评论文本列名。
        category_column：类别列名。
        threshold：判定比例，默认 0.95。

    返回：
        ``{类别: 比例}``，仅含比例达到阈值的类别；没有命中的类别不出现。
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 pandas DataFrame")
    for column in (text_column, category_column):
        if column not in frame.columns:
            raise KeyError(f"缺少必需列：{column}")

    shares: dict[str, float] = {}
    for category, group in frame.groupby(category_column):
        texts = group[text_column].astype("string").fillna("")
        if texts.empty:
            continue
        # 用 Python 的字符串包含判断，而不是 pandas 的 .str.contains：
        # 类别名里可能含正则元字符，且含中文时 RE2 的字符类语义与预期不同。
        hits = sum(1 for text in texts if str(category) in str(text))
        share = hits / len(texts)
        if share >= threshold:
            shares[str(category)] = share
    return shares


def render_comparison_report(
    *,
    model_results: Mapping[str, Mapping[str, Any]],
    selected_model: str,
    categories: Sequence[str],
    selection_metric: str,
    dataset_summary: Mapping[str, Any],
    vectorizer_config: Mapping[str, Any],
    ablation_results: Sequence[Mapping[str, Any]] = (),
    notes: Sequence[str] = (),
) -> str:
    """渲染模型对比报告。

    Args:
        model_results: ``{模型名: {"dev": 指标块, "test": 指标块, "cost": 成本块, "meta": ...}}``。
        selected_model: 依据验证集指标选出的模型名。
        categories: 类别顺序。
        selection_metric: 选模依据的指标名，本项目为 ``macro_f1``。
        dataset_summary: 数据规模摘要。
        vectorizer_config: 特征配置摘要。
        ablation_results: 特征方案对照结果，每项含 ``name``、``macro_f1``、``note``。
        notes: 追加到报告的备注条目。

    Returns:
        完整的 Markdown 报告文本。
    """
    lines: list[str] = [
        "# 传统模型基线对比报告（M3）",
        "",
        "本报告由 `scripts/train_baseline.py` 生成，所有指标均来自真实运行结果。",
        "",
        "## 1. 评估口径",
        "",
        f"- 数据规模：训练集 {dataset_summary['train_rows']:,} 条、"
        f"验证集 {dataset_summary['dev_rows']:,} 条、测试集 {dataset_summary['test_rows']:,} 条",
        f"- 类别顺序：{'、'.join(categories)}（取自 `label_mapping.json`，固定不变）",
        f"- **选模依据：验证集 {selection_metric}**，不使用测试集做任何选择或调参",
        f"- 训练集划分随机种子：{dataset_summary['random_seed']}",
        f"- 数据集指纹：`{dataset_summary['dataset_fingerprint'][:16]}…`"
        "（所有模型共用同一份；按契约 §2.3 由 train/dev/test 三个 CSV 的内容哈希算出，"
        "不受 `created_at` 影响）",
        "",
        "## 2. 特征配置",
        "",
        "| 项目 | 取值 |",
        "| --- | --- |",
    ]
    for key, value in vectorizer_config.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## 3. 模型对比（AGENTS.md §6 要求）",
            "",
            f"按验证集 {selection_metric} 排序，**最优为 `{selected_model}`**：",
            "",
            f"| 模型 | 验证集 Macro-F1 | 验证集 Accuracy | 测试集 Macro-F1 | 测试集 Accuracy | "
            f"模型大小 | 单条延迟 | 批量32延迟 | 吞吐量 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    ordered = sorted(
        model_results.items(),
        key=lambda item: item[1]["dev"]["macro_f1"],
        reverse=True,
    )
    for name, result in ordered:
        dev = result["dev"]
        test = result.get("test")
        cost = result["cost"]
        marker = " ← 选定" if name == selected_model else ""
        test_f1 = f"{test['macro_f1']:.4f}" if test else "未评估"
        test_acc = f"{test['accuracy']:.4f}" if test else "未评估"
        lines.append(
            f"| `{name}`{marker} | {dev['macro_f1']:.4f} | {dev['accuracy']:.4f} | "
            f"{test_f1} | {test_acc} | {cost['model_size_bytes'] / 1024 / 1024:.2f} MB | "
            f"{cost['single_sample_latency_ms']:.2f} ms | {cost['batch_32_latency_ms']:.2f} ms | "
            f"{cost['throughput_per_sec']:.0f} 条/秒 |"
        )

    lines.extend(
        [
            "",
            "说明：测试集指标在**选模完成之后**才计算，仅用于确认最终表现与横向对比；",
            "`metrics.json` 中每个模型都记录了测试集结果，其 `metadata.notes` 亦注明",
            "选模依据为验证集，测试集未参与任何模型选择或超参数调整。",
            "",
            "## 4. 各类别详细指标（选定模型）",
            "",
            f"模型 `{selected_model}` 在测试集上的逐类别表现：",
            "",
            "| 类别 | Precision | Recall | F1 | 支持数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    selected = model_results[selected_model]
    block = selected.get("test") or selected["dev"]
    set_name = "测试集" if selected.get("test") else "验证集"
    for category in categories:
        values = block["per_class"][category]
        lines.append(
            f"| {category} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']:,} |"
        )
    lines.append("")
    lines.append(f"以上为{set_name}结果。混淆矩阵见对应的 `metrics.json`。")

    if ablation_results:
        lines.extend(
            [
                "",
                "## 5. 特征方案对照（验证集实测）",
                "",
                "为避免凭直觉选择分词方案，先用最快的模型在验证集上实测比较：",
                "",
                "| 方案 | 验证集 Macro-F1 | 说明 |",
                "| --- | ---: | --- |",
            ]
        )
        for row in ablation_results:
            lines.append(f"| {row['name']} | {row['macro_f1']:.4f} | {row['note']} |")

    if notes:
        lines.extend(["", "## 6. 结论与已知局限", ""])
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def top_confusions(
    matrix: Sequence[Sequence[int]], categories: Sequence[str], *, limit: int = 10
) -> list[tuple[str, str, int]]:
    """从混淆矩阵中取出最容易被混淆的类别对（不含对角线）。

    Args:
        matrix: 行真实、列预测的混淆矩阵。
        categories: 类别顺序，需与矩阵行列一致。
        limit: 返回的最大条目数。

    Returns:
        ``[(真实类别, 预测类别, 次数), ...]``，按次数降序。
    """
    pairs: list[tuple[str, str, int]] = []
    for row_index, true_category in enumerate(categories):
        for column_index, predicted_category in enumerate(categories):
            if row_index == column_index:
                continue
            count = int(matrix[row_index][column_index])
            if count:
                pairs.append((str(true_category), str(predicted_category), count))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:limit]
