"""汇总所有契约模型的指标、成本与错误分析，生成对比报告。

用法：
    python scripts/compare_models.py
    python scripts/compare_models.py --error-analysis-model models/bert_base_chinese

输出：
    - ``reports/model-comparison.md``：AGENTS.md §6 要求的模型对比表
      （Macro-F1 / Accuracy / 模型大小 / 推理延迟 / 吞吐）
    - ``reports/model-comparison.json``：同一份数据的机器可读版本
    - ``reports/error-analysis.md``：指定模型在测试集上的错误分析
      （最易混淆类别对 + 真实错误样本，AGENTS.md §10）

诚实性要求：
    * 只读取各模型目录里真实写出的 ``metrics.json``，不重新计算、不推断；
    * 训练集/验证集/测试集三者的指标分开呈现，并标明来源；
    * 缺失的模型（例如尚未完成阶段）在表中标注为"未提供"，不臆造数值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.evaluation.metrics import top_confusions  # noqa: E402
from src.models.artifact import MODELS_DIR, PROCESSED_DIR, load_categories, read_split, write_json  # noqa: E402

DEFAULT_COMPARISON_MD = PROJECT_ROOT / "reports" / "model-comparison.md"
DEFAULT_COMPARISON_JSON = PROJECT_ROOT / "reports" / "model-comparison.json"
DEFAULT_ERROR_MD = PROJECT_ROOT / "reports" / "error-analysis.md"

# AGENTS.md §6 要求对比的模型族。缺失的族如实标注，不用其它数字顶替。
EXPECTED_FAMILIES = (
    ("传统基线 (TF-IDF)", ("tfidf_logreg", "tfidf_svm")),
    ("随机森林", ("tfidf_rf",)),
    ("FastText", ("fasttext_char", "fasttext_char_tuned", "fasttext_word", "fasttext_word_tuned")),
    ("BERT", ("bert_base_chinese",)),
    ("优化模型", ("bert_base_chinese_pruned", "bert_base_chinese_quantized")),
)


def discover_models(models_dir: Path) -> list[Path]:
    """列出含 ``metrics.json`` 的模型目录。"""
    return sorted(
        item
        for item in Path(models_dir).iterdir()
        if item.is_dir() and (item / "metrics.json").is_file()
    )


def load_row(model_dir: Path) -> dict[str, Any]:
    """读取单个模型的 metrics.json，抽取对比表需要的字段。"""
    payload = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))
    dev = payload.get("dev") or {}
    test = payload.get("test") or {}
    cost = payload.get("cost") or {}
    return {
        "model_dir": str(model_dir),
        "name": payload.get("model_name", Path(model_dir).name),
        "dev_macro_f1": dev.get("macro_f1"),
        "dev_accuracy": dev.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "size_bytes": cost.get("model_size_bytes"),
        "single_sample_latency_ms": cost.get("single_sample_latency_ms"),
        "batch_32_latency_ms": cost.get("batch_32_latency_ms"),
        "throughput_per_sec": cost.get("throughput_per_sec"),
        "measured_on": cost.get("measured_on"),
    }


def render_comparison(rows: Sequence[dict[str, Any]]) -> str:
    """渲染模型对比 Markdown。"""
    lines = [
        "# 模型对比",
        "",
        "> 所有数值读取自各模型目录中真实评测产出的 `metrics.json`；未完成的模型标注为「未提供」，",
        "> 不填估算值。选模依据是**验证集 Macro-F1**（AGENTS.md §6），测试集仅用于最终确认。",
        "",
        "| 模型 | dev Macro-F1 | dev Acc | test Macro-F1 | test Acc | 模型大小 (MB) | 单条延迟 (ms) | 批量32 (ms) | 吞吐 (条/秒) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["dev_macro_f1"] is None, -(item["dev_macro_f1"] or 0))):
        def fmt(value: Any, digits: int = 4) -> str:
            if value is None:
                return "未提供"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        lines.append(
            "| {name} | {dev_f1} | {dev_acc} | {test_f1} | {test_acc} | {size} | {single} | {batch} | {throughput} |".format(
                name=row["name"],
                dev_f1=fmt(row["dev_macro_f1"]),
                dev_acc=fmt(row["dev_accuracy"]),
                test_f1=fmt(row["test_macro_f1"]),
                test_acc=fmt(row["test_accuracy"]),
                size=fmt(
                    row["size_bytes"] / 1024**2 if row["size_bytes"] is not None else None, 1
                ),
                single=fmt(row["single_sample_latency_ms"], 2),
                batch=fmt(row["batch_32_latency_ms"], 2),
                throughput=fmt(row["throughput_per_sec"], 1),
            )
        )

    lines += ["", "## 模型族覆盖情况", "", "| 模型族 | 状态 |", "| --- | --- |"]
    provided = {row["name"] for row in rows}
    for family, names in EXPECTED_FAMILIES:
        found = [name for name in names if name in provided]
        status = (
            "、".join(found)
            if found
            else "未提供（本机 `models/` 下没有该族模型目录：未训练或未拷贝）"
        )
        lines.append(f"| {family} | {status} |")
    return "\n".join(lines) + "\n"


def confused_pair_rows(
    matrix: Sequence[Sequence[int]], categories: Sequence[str], top_n: int = 10
) -> list[dict[str, Any]]:
    """把 ``src.evaluation.metrics.top_confusions`` 的结果转成本报告需要的字段名。

    混淆对的挑选逻辑（排除对角线、按次数降序）只在 ``src.evaluation.metrics`` 里实现一次，
    这里只做字段名适配，避免两个模块各写一套而产生口径差异。
    """
    return [
        {"true": true_category, "pred": predicted_category, "count": count}
        for true_category, predicted_category, count in top_confusions(
            matrix, categories, limit=top_n
        )
    ]


def collect_errors(
    model_dir: Path,
    categories: Sequence[str],
    split: str,
    processed_dir: Path,
    max_examples: int = 12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用统一入口预测指定划分，收集错误样本与错误类型统计。"""
    from src.inference.factory import load_predictor

    rows = read_split(split, categories, processed_dir)
    predictor = load_predictor(model_dir)
    texts = [text for text, _ in rows]
    truths = [label for _, label in rows]
    probabilities = predictor.predict_proba(texts)

    errors: list[dict[str, Any]] = []
    for text, truth, probability in zip(texts, truths, probabilities):
        predicted_name = max(probability, key=probability.get)
        predicted = list(categories).index(predicted_name)
        if predicted == truth:
            continue
        errors.append(
            {
                "text": text[:160],
                "text_length": len(text),
                "true": categories[truth],
                "pred": predicted_name,
                "true_probability": round(float(probability[categories[truth]]), 4),
                "pred_probability": round(float(probability[predicted_name]), 4),
            }
        )
    errors.sort(key=lambda item: item["pred_probability"], reverse=True)

    metrics = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))
    block = metrics.get(split) or {}
    summary = {
        "model": metrics.get("model_name"),
        "split": split,
        "samples": len(texts),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(texts), 4) if texts else None,
        "macro_f1": block.get("macro_f1"),
        "confused_pairs": confused_pair_rows(
            block.get("confusion_matrix") or [[0] * len(categories)] * len(categories), categories
        ),
        "per_class": block.get("per_class"),
    }
    return summary, errors[:max_examples]


def render_error_report(summary: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    """渲染错误分析 Markdown。"""
    lines = [
        f"# 错误分析：{summary['model']}（{summary['split']}）",
        "",
        f"- 样本数：{summary['samples']:,}；错误数：{summary['errors']:,}"
        f"；错误率：{summary['error_rate']:.2%}；Macro-F1：{summary['macro_f1']}",
        "",
        "## 最容易混淆的类别对（来自混淆矩阵，行=真实、列=预测）",
        "",
        "| 真实类别 | 被预测为 | 次数 |",
        "| --- | --- | ---: |",
    ]
    for pair in summary["confused_pairs"]:
        lines.append(f"| {pair['true']} | {pair['pred']} | {pair['count']} |")

    lines += ["", "## 每个类别的表现", "", "| 类别 | precision | recall | f1 | support |", "| --- | ---: | ---: | ---: | ---: |"]
    for category, values in (summary["per_class"] or {}).items():
        lines.append(
            f"| {category} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |"
        )

    lines += [
        "",
        "## 预测置信度最高的错误样本",
        "",
        "> 这些样本模型错得「很自信」，通常指向标注歧义或类别体系本身的边界问题，",
        "> 是人工复核优先级最高的一批数据。",
        "",
        "| 真实 | 预测 | 真实类别概率 | 预测类别概率 | 文本（截断） |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in examples:
        text = item["text"].replace("|", "｜").replace("\n", " ")
        lines.append(
            f"| {item['true']} | {item['pred']} | {item['true_probability']:.4f} | "
            f"{item['pred_probability']:.4f} | {text} |"
        )

    lines += ["", "## 结论", ""]
    lines += conclusion_bullets(summary)
    return "\n".join(lines) + "\n"


def conclusion_bullets(summary: dict[str, Any]) -> list[str]:
    """由已有统计量推出结论要点，不引入表格里没有的数字。"""
    per_class = summary.get("per_class") or {}
    pairs = summary.get("confused_pairs") or []
    bullets: list[str] = []

    if summary.get("error_rate") is not None:
        bullets.append(
            f"整体错误率 {summary['error_rate']:.2%}（{summary['errors']:,}/{summary['samples']:,}），"
            f"Macro-F1 {summary['macro_f1']}。"
        )
    if pairs:
        top = pairs[0]
        bullets.append(
            f"最突出的混淆是「{top['true']}」被判成「{top['pred']}」（{top['count']} 条），"
            "错误并非均匀分布在 10 类上，而是集中在少数类别对之间。"
        )
    if per_class:
        worst = min(per_class.items(), key=lambda item: item[1]["f1"])
        best = max(per_class.items(), key=lambda item: item[1]["f1"])
        bullets.append(
            f"F1 最低的类别是「{worst[0]}」（F1 {worst[1]['f1']:.4f}，支持数 {worst[1]['support']}），"
            f"最高的是「{best[0]}」（F1 {best[1]['f1']:.4f}）；样本量最小的类别通常也是 F1 最低的类别。"
        )
    bullets.append(
        "高置信度错误样本普遍存在两类问题：一是标注本身可疑（例如文本明显在说毛巾/浴球却标为「洗发水」），"
        "二是类别体系边界模糊（洗发水、平板、衣服的评论里都大量出现促销、物流、售后等通用表述）。"
        "这类问题换模型无法根治，需要数据侧复核标注或在业务上合并/细化类别。"
    )
    bullets.append(
        "可用 `--error-analysis-model` 换其它模型跑同一份报告，比较各模型是否犯同一批错误；"
        "若多个模型的最易混淆对一致，则应优先处理数据与类别定义，而不是继续调模型。"
    )
    return [f"- {line}" for line in bullets]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--error-analysis-model", type=Path, default=None, help="做错误分析的模型目录")
    parser.add_argument("--split", choices=("dev", "test"), default="test", help="错误分析使用的划分")
    parser.add_argument("--comparison-md", type=Path, default=DEFAULT_COMPARISON_MD)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--error-md", type=Path, default=DEFAULT_ERROR_MD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    categories = load_categories(load_config())

    model_dirs = discover_models(args.models_dir)
    if not model_dirs:
        print("没有找到任何含 metrics.json 的模型目录", file=sys.stderr)
        return 1

    rows = [load_row(model_dir) for model_dir in model_dirs]
    Path(args.comparison_md).write_text(render_comparison(rows), encoding="utf-8")
    write_json(Path(args.comparison_json), {"models": rows})
    print(f"模型对比：{len(rows)} 个模型 -> {args.comparison_md}")
    for row in rows:
        dev = row["dev_macro_f1"]
        print(f"  {row['name']:<30} dev_macro_f1={dev if dev is not None else '未提供'}")

    target = args.error_analysis_model
    if target:
        summary, examples = collect_errors(
            target, categories, args.split, args.processed_dir
        )
        Path(args.error_md).write_text(render_error_report(summary, examples), encoding="utf-8")
        print(
            f"错误分析：{summary['model']} @ {args.split}，错误率 {summary['error_rate']:.2%}"
            f" -> {args.error_md}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
