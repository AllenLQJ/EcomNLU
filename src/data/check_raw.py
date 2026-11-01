"""原始数据质量检查（只读）。

对 ``data/raw/online_shopping_10_cats.csv`` 做一次可复现的质量检查，输出证据报告，
用于证实或推翻 ``data/README.md`` 中记录的已知问题，并为 M2 的清洗规则和
README 中的 ``max_length`` 讨论提供数据依据。

本脚本**只读**原始数据：不写入 ``data/raw``，不做任何清洗、去重或修改。

用法::

    python src/data/check_raw.py
    python src/data/check_raw.py --output-dir reports/data-quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "data-quality"

# 与 scripts/download_data.py 及 data/README.md 保持一致
EXPECTED_SHA256 = "f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb"
EXPECTED_COLUMNS = ["cat", "label", "review"]
EXPECTED_ROWS = 62_774
VALID_LABELS = {"0", "1"}

# data/README.md 中记录的已知问题，用于自动比对
DOCUMENTED_MISSING_REVIEW = 1
DOCUMENTED_DUPLICATE_REVIEWS = 49

# 控制字符（保留 \t \n \r），含 C0 与 C1 区段
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")
HTML_ENTITY_PATTERN = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,6}|#x[0-9a-fA-F]{2,6});")
HTML_TAG_PATTERN = re.compile(r"<[^<>\n]{0,40}>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CJK_PATTERN = re.compile(r"[一-鿿]")

ANOMALY_LABELS = {
    "blank": "空白文本",
    "control_chars": "异常控制字符",
    "html_entities": "HTML 实体",
    "html_tags": "HTML 标签",
    "urls": "URL",
    "leading_or_trailing_space": "首尾空白",
    "no_cjk_characters": "不含中文字符",
}

LENGTH_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)
SHORT_TEXT_CANDIDATES = (1, 2, 3, 5, 10)
EXAMPLE_LIMIT = 5
EXAMPLE_MAX_CHARS = 80

LOGGER = logging.getLogger("check_raw")


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """记录单个检查阶段的开始、结束与耗时。"""
    start = time.perf_counter()
    LOGGER.info("[%s] 开始", name)
    try:
        yield
    finally:
        LOGGER.info("[%s] 完成，耗时 %.2fs", name, time.perf_counter() - start)


def sha256_of(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{public_data_path(config_path)}")
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def resolve_from_root(relative_path: str) -> Path:
    """把配置中的相对路径解析为项目根目录下的绝对路径。"""
    return (PROJECT_ROOT / relative_path).resolve()


def public_data_path(path: Path) -> str:
    """报告里只写相对项目根的路径；项目外路径不保留绝对位置。"""
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "项目外"


def load_raw_frame(raw_path: Path) -> pd.DataFrame:
    """读取原始 CSV。

    全部按字符串读取并关闭默认 NA 推断，使空值和异常字面量（如 ``nan``）保持原样，
    便于如实统计，而不是被 pandas 悄悄转换。
    """
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"原始数据不存在：{public_data_path(raw_path)}\n请先运行：python scripts/download_data.py"
        )
    return pd.read_csv(
        raw_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )


def truncate(text: str, limit: int = EXAMPLE_MAX_CHARS) -> str:
    """截断长文本用于报告展示。"""
    return text if len(text) <= limit else text[:limit] + "…"


def check_file(raw_path: Path) -> dict[str, Any]:
    """文件级校验：大小与 SHA-256。"""
    actual_hash = sha256_of(raw_path)
    return {
        "path": public_data_path(raw_path),
        "size_bytes": raw_path.stat().st_size,
        "sha256": actual_hash,
        "sha256_matches_expected": actual_hash == EXPECTED_SHA256,
    }


def check_schema(frame: pd.DataFrame) -> dict[str, Any]:
    """字段名、顺序与行数校验。"""
    columns = list(frame.columns)
    return {
        "columns": columns,
        "columns_match_expected": columns == EXPECTED_COLUMNS,
        "row_count": int(len(frame)),
        "row_count_matches_expected": int(len(frame)) == EXPECTED_ROWS,
    }


def check_missing(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """缺失值统计：区分纯空白与疑似占位符字面量。"""
    placeholders = {"nan", "null", "none", "n/a", "na", "\\n"}
    per_column: dict[str, Any] = {}
    for column in columns:
        series = frame[column]
        stripped = series.str.strip()
        blank_mask = stripped.eq("")
        placeholder_mask = stripped.str.lower().isin(placeholders) & ~blank_mask
        per_column[column] = {
            "blank": int(blank_mask.sum()),
            "placeholder_literal": int(placeholder_mask.sum()),
            "example_rows": [
                {"row_index": int(index), "value": truncate(str(series.at[index]), 40)}
                for index in series.index[blank_mask | placeholder_mask][:EXAMPLE_LIMIT]
            ],
        }
    return per_column


def check_labels(
    frame: pd.DataFrame, category_column: str, sentiment_column: str, expected_categories: list[str]
) -> dict[str, Any]:
    """标签取值校验与类别 × 情感交叉分布。"""
    category_series = frame[category_column].str.strip()
    category_counts = category_series.value_counts().to_dict()

    unexpected_categories = sorted(set(category_counts) - set(expected_categories))
    missing_categories = sorted(set(expected_categories) - set(category_counts))

    sentiment_series = frame[sentiment_column].str.strip()
    sentiment_counts = sentiment_series.value_counts().to_dict()
    invalid_sentiment_mask = ~sentiment_series.isin(VALID_LABELS)

    crosstab: dict[str, dict[str, int]] = {}
    valid_mask = ~invalid_sentiment_mask
    if valid_mask.any():
        valid_frame = frame.loc[valid_mask].copy()
        valid_frame[category_column] = valid_frame[category_column].str.strip()
        valid_frame[sentiment_column] = valid_frame[sentiment_column].str.strip()
        table = pd.crosstab(valid_frame[category_column], valid_frame[sentiment_column])
        for category, row in table.iterrows():
            crosstab[str(category)] = {str(key): int(value) for key, value in row.items()}

    # 情感正负完全相等的类别，通常意味着上游数据做过人工配平而非自然分布
    perfectly_balanced = sorted(
        category
        for category, counts in crosstab.items()
        if counts.get("0") is not None and counts.get("0") == counts.get("1")
    )

    return {
        "category_counts": {str(key): int(value) for key, value in category_counts.items()},
        "category_count": len(category_counts),
        "unexpected_categories": unexpected_categories,
        "missing_categories": missing_categories,
        "sentiment_counts": {str(key): int(value) for key, value in sentiment_counts.items()},
        "invalid_sentiment_count": int(invalid_sentiment_mask.sum()),
        "invalid_sentiment_examples": sentiment_series[invalid_sentiment_mask]
        .head(EXAMPLE_LIMIT)
        .tolist(),
        "category_by_sentiment": crosstab,
        "perfectly_balanced_categories": perfectly_balanced,
    }


def duplicate_stats(
    keys: pd.Series, categories: pd.Series, sentiments: pd.Series
) -> dict[str, Any]:
    """统计给定文本键下的重复、冲突与去重后样本量。

    ``keys`` 决定口径：传原文即为原文口径，传归一化文本即为归一化口径。
    空白键会被排除，避免把缺失评论误算成重复。
    """
    working = pd.DataFrame({"key": keys, "cat": categories, "label": sentiments})
    working = working.loc[working["key"].ne("")]

    sizes = working["key"].value_counts()
    duplicated_sizes = sizes[sizes > 1]

    grouped = working.groupby("key")
    conflicts = (grouped["cat"].nunique() > 1) | (grouped["label"].nunique() > 1)
    conflicts = conflicts.reindex(duplicated_sizes.index, fill_value=False)

    examples: list[dict[str, Any]] = []
    for key in duplicated_sizes.sort_values(ascending=False).index[:EXAMPLE_LIMIT]:
        rows = working.loc[working["key"].eq(key)]
        examples.append(
            {
                "review": truncate(str(key)),
                "occurrences": int(len(rows)),
                "categories": sorted(rows["cat"].unique().tolist()),
                "labels": sorted(rows["label"].unique().tolist()),
            }
        )

    return {
        "extra_rows": int(duplicated_sizes.sum() - len(duplicated_sizes)),
        "duplicated_groups": int(len(duplicated_sizes)),
        "largest_group_size": int(sizes.max()) if len(sizes) else 0,
        "conflicting_groups": int(conflicts.sum()),
        "conflicting_rows": int(duplicated_sizes[conflicts].sum()),
        "rows_after_dedup": int(working["key"].nunique()),
        "examples": examples,
    }


def check_duplicates(
    frame: pd.DataFrame, category_column: str, sentiment_column: str, text_column: str
) -> dict[str, Any]:
    """重复检查：完全重复行，以及原文/归一化两种口径下的文本重复与冲突。

    "冲突重复"指同一评论文本对应多个类别或多个情感标签。这类样本不能直接用于
    训练，也会在划分数据集时造成跨集合泄漏，是 M2 必须最先处理的问题。

    归一化口径（去首尾空白并合并连续空白）会合并"仅内部空白不同"的文本，因此其
    重复行数可能**高于**原文口径。两者差异记录在 ``normalization_merge_groups``，
    报告中必须分别标注口径，不能混用。
    """
    raw_keys = frame[text_column]
    normalized_keys = frame[text_column].str.strip().str.replace(r"\s+", " ", regex=True)
    categories = frame[category_column].str.strip()
    sentiments = frame[sentiment_column].str.strip()

    return {
        "exact_duplicate_rows": int(
            frame.duplicated(subset=[category_column, sentiment_column, text_column]).sum()
        ),
        "by_raw_text": duplicate_stats(raw_keys, categories, sentiments),
        "by_normalized_text": duplicate_stats(normalized_keys, categories, sentiments),
        "normalization_merge_groups": int(
            raw_keys[raw_keys.ne("")].nunique()
            - normalized_keys[normalized_keys.ne("")].nunique()
        ),
    }


def check_text_anomalies(frame: pd.DataFrame, text_column: str) -> dict[str, Any]:
    """异常文本统计：空白、控制字符、HTML 实体与标签、URL、无中文。"""
    series = frame[text_column]
    stripped = series.str.strip()

    masks = {
        "blank": stripped.eq(""),
        "control_chars": series.str.contains(CONTROL_CHAR_PATTERN, regex=True, na=False),
        "html_entities": series.str.contains(HTML_ENTITY_PATTERN, regex=True, na=False),
        "html_tags": series.str.contains(HTML_TAG_PATTERN, regex=True, na=False),
        "urls": series.str.contains(URL_PATTERN, regex=True, na=False),
        "leading_or_trailing_space": series.ne(stripped),
    }

    result: dict[str, Any] = {
        name: {
            "count": int(mask.sum()),
            "examples": [truncate(str(value)) for value in series[mask].head(EXAMPLE_LIMIT)],
        }
        for name, mask in masks.items()
    }

    no_cjk_mask = ~series.str.contains(CJK_PATTERN, regex=True, na=False)
    result["no_cjk_characters"] = {
        "count": int(no_cjk_mask.sum()),
        "examples": [truncate(str(value)) for value in series[no_cjk_mask].head(EXAMPLE_LIMIT)],
    }
    return result


def check_lengths(frame: pd.DataFrame, text_column: str) -> dict[str, Any]:
    """字符长度分布，为传统模型截断与 BERT max_length 讨论提供依据。

    注意：这里统计的是**字符**长度。BERT 的 ``max_length`` 需要 tokenizer 实测的
    token 长度分布，属于 M4 的独立步骤，不能直接用字符长度替代。
    """
    lengths = frame[text_column].str.len()

    quantiles = {
        f"p{int(q * 1000) / 10:g}": float(lengths.quantile(q)) for q in LENGTH_QUANTILES
    }
    short_thresholds = {
        f"le_{threshold}": int(lengths.le(threshold).sum()) for threshold in SHORT_TEXT_CANDIDATES
    }

    longest = lengths.nlargest(EXAMPLE_LIMIT)
    shortest = lengths.nsmallest(EXAMPLE_LIMIT)
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(round(lengths.mean(), 2)),
        "median": float(lengths.median()),
        "std": float(round(lengths.std(), 2)),
        "quantiles": quantiles,
        "short_text_counts": short_thresholds,
        "longest_examples": [
            {"length": int(lengths.at[index]), "review": truncate(str(frame.at[index, text_column]))}
            for index in longest.index
        ],
        "shortest_examples": [
            {"length": int(lengths.at[index]), "review": truncate(str(frame.at[index, text_column]))}
            for index in shortest.index
        ],
    }


def check_category_balance(category_counts: dict[str, int]) -> dict[str, Any]:
    """类别不均衡程度量化。"""
    if not category_counts:
        return {"imbalance_ratio": None, "majority_share": None, "minority_share": None}
    total = sum(category_counts.values())
    largest = max(category_counts.values())
    smallest = min(category_counts.values())

    # 多个类别样本数完全相同，通常意味着上游对每类做了固定数量截断，而非自然分布
    count_frequencies: dict[int, int] = {}
    for count in category_counts.values():
        count_frequencies[count] = count_frequencies.get(count, 0) + 1
    tied_counts = sorted(
        (count for count, times in count_frequencies.items() if times > 1), reverse=True
    )

    return {
        "imbalance_ratio": round(largest / smallest, 2) if smallest else None,
        "majority_share": round(largest / total, 4),
        "minority_share": round(smallest / total, 4),
        "tied_sample_counts": [int(count) for count in tied_counts],
    }


def build_report(raw_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """执行全部检查并组装报告。"""
    data_config = config.get("data", {})
    category_column = data_config.get("category_column", "cat")
    sentiment_column = data_config.get("sentiment_column", "label")
    text_column = data_config.get("text_column", "review")
    expected_categories = config.get("task", {}).get("categories", [])

    with timed_stage("读取原始数据"):
        frame = load_raw_frame(raw_path)

    with timed_stage("文件与字段校验"):
        file_info = check_file(raw_path)
        schema = check_schema(frame)

    with timed_stage("缺失值检查"):
        missing = check_missing(frame, list(frame.columns))

    with timed_stage("标签取值检查"):
        labels = check_labels(frame, category_column, sentiment_column, expected_categories)
        balance = check_category_balance(labels["category_counts"])

    with timed_stage("重复检查"):
        duplicates = check_duplicates(frame, category_column, sentiment_column, text_column)

    with timed_stage("异常文本检查"):
        anomalies = check_text_anomalies(frame, text_column)

    with timed_stage("文本长度分析"):
        lengths = check_lengths(frame, text_column)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "file": file_info,
        "schema": schema,
        "missing": missing,
        "labels": labels,
        "category_balance": balance,
        "duplicates": duplicates,
        "text_anomalies": anomalies,
        "lengths": lengths,
    }


def compare_with_documented_claims(report: dict[str, Any]) -> list[str]:
    """把实测结果与 data/README.md 记录的说法逐条比对。"""
    lines: list[str] = []

    missing_review = report["missing"].get("review", {}).get("blank", 0)
    lines.append(
        f"缺失评论：文档记录 {DOCUMENTED_MISSING_REVIEW} 条，实测 {missing_review} 条 → "
        + ("一致" if missing_review == DOCUMENTED_MISSING_REVIEW else "**不一致，需更新 data/README.md**")
    )

    duplicates = report["duplicates"]
    raw_duplicates = duplicates["by_raw_text"]
    normalized_duplicates = duplicates["by_normalized_text"]
    lines.append(
        f"重复评论文本：文档记录 {DOCUMENTED_DUPLICATE_REVIEWS} 行，实测原文口径 "
        f"{raw_duplicates['extra_rows']} 行（{raw_duplicates['duplicated_groups']} 组）→ "
        + ("一致" if raw_duplicates["extra_rows"] == DOCUMENTED_DUPLICATE_REVIEWS else "**不一致**")
        + f"；归一化口径 {normalized_duplicates['extra_rows']} 行"
        f"（{normalized_duplicates['duplicated_groups']} 组，差异来自 "
        f"{duplicates['normalization_merge_groups']} 组仅内部空白不同的文本）"
    )
    lines.append(
        f"冲突重复：{normalized_duplicates['conflicting_groups']} 组重复文本对应多个类别或情感"
        f"（涉及 {normalized_duplicates['conflicting_rows']} 行）→ "
        + (
            "**不能直接 drop_duplicates 保留首条，必须先定冲突处理规则**"
            if normalized_duplicates["conflicting_groups"]
            else "无冲突"
        )
    )

    balance = report["category_balance"]
    lines.append(
        f"类别不均衡：最大/最小类样本比 {balance['imbalance_ratio']}，"
        f"最大类占比 {balance['majority_share']:.2%} → "
        + ("与文档描述一致" if (balance["imbalance_ratio"] or 0) > 1.5 else "**未体现明显不均衡**")
    )

    short_count = report["lengths"]["short_text_counts"]["le_2"]
    lines.append(
        f"超短评论：长度 ≤ 2 字符的有 {short_count} 条，最短 {report['lengths']['min']} 字符；"
        f"超长评论最长 {report['lengths']['max']} 字符 → 与文档描述一致"
    )
    return lines


def render_console_summary(report: dict[str, Any]) -> str:
    """生成面向终端的中文结论摘要。"""
    file_info = report["file"]
    duplicates = report["duplicates"]
    raw_duplicates = duplicates["by_raw_text"]
    normalized_duplicates = duplicates["by_normalized_text"]
    labels = report["labels"]
    lengths = report["lengths"]
    anomalies = report["text_anomalies"]
    balance = report["category_balance"]

    lines = [
        "",
        "=" * 68,
        "原始数据质量检查结论",
        "=" * 68,
        f"文件      : {file_info['path']} ({file_info['size_bytes']:,} 字节)",
        f"SHA-256   : {'匹配' if file_info['sha256_matches_expected'] else '不匹配'}",
        f"字段      : {report['schema']['columns']}",
        f"行数      : {report['schema']['row_count']:,}"
        f" ({'符合预期' if report['schema']['row_count_matches_expected'] else '与预期不符'})",
        "",
        f"类别数    : {labels['category_count']}（预期 10）",
        f"  未预期类别: {labels['unexpected_categories'] or '无'}",
        f"  缺失类别  : {labels['missing_categories'] or '无'}",
        f"情感取值  : {labels['sentiment_counts']}",
        f"  非法情感标签: {labels['invalid_sentiment_count']} 条",
        f"不均衡比  : 最大/最小 = {balance['imbalance_ratio']}"
        f"（最大类占比 {balance['majority_share']:.2%}）",
        "  并列样本数: "
        + ("、".join(f"{count:,}" for count in balance["tied_sample_counts"]) or "无")
        + "（多个类别样本数完全相同，提示上游做过固定数量截断）",
        "  情感配平类别: "
        + ("、".join(labels["perfectly_balanced_categories"]) or "无")
        + "（正负样本完全相等，提示上游做过人工配平）",
        "",
        f"完全重复行        : {duplicates['exact_duplicate_rows']:,}",
        f"重复(原文口径)    : {raw_duplicates['extra_rows']:,} 行 / "
        f"{raw_duplicates['duplicated_groups']:,} 组（最大一组 {raw_duplicates['largest_group_size']} 条）",
        f"重复(归一化口径)  : {normalized_duplicates['extra_rows']:,} 行 / "
        f"{normalized_duplicates['duplicated_groups']:,} 组"
        f"（差异来自 {duplicates['normalization_merge_groups']} 组仅内部空白不同的文本）",
        f"冲突重复组        : {normalized_duplicates['conflicting_groups']:,}"
        f"（同一文本对应多个类别或情感，涉及 {normalized_duplicates['conflicting_rows']:,} 行）",
        f"按归一化去重后    : {normalized_duplicates['rows_after_dedup']:,} 条",
        "",
        f"空白文本      : {anomalies['blank']['count']:,}",
        f"控制字符      : {anomalies['control_chars']['count']:,}",
        f"HTML 实体     : {anomalies['html_entities']['count']:,}",
        f"HTML 标签     : {anomalies['html_tags']['count']:,}",
        f"URL           : {anomalies['urls']['count']:,}",
        f"不含中文字符  : {anomalies['no_cjk_characters']['count']:,}",
        f"首尾空白      : {anomalies['leading_or_trailing_space']['count']:,}",
        "",
        f"字符长度  : min={lengths['min']} max={lengths['max']} "
        f"mean={lengths['mean']} median={lengths['median']}",
        "分位数    : "
        + " ".join(f"{name}={value:g}" for name, value in lengths["quantiles"].items()),
        "超短计数  : "
        + " ".join(
            f"≤{name.split('_')[1]}字符={value:,}"
            for name, value in lengths["short_text_counts"].items()
        ),
        "",
        "-" * 68,
        "与 data/README.md 已记录说法的比对",
        "-" * 68,
    ]
    lines.extend(f"- {line}" for line in compare_with_documented_claims(report))
    lines.append("=" * 68)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """生成 Markdown 版质量报告，作为 M2 的可复现交付物。"""
    labels = report["labels"]
    lengths = report["lengths"]
    duplicates = report["duplicates"]
    raw_duplicates = duplicates["by_raw_text"]
    normalized_duplicates = duplicates["by_normalized_text"]
    anomalies = report["text_anomalies"]
    balance = report["category_balance"]
    file_info = report["file"]

    parts = [
        "# 原始数据质量报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 数据文件：`{file_info['path']}`（{file_info['size_bytes']:,} 字节）",
        f"- SHA-256：`{file_info['sha256']}`"
        f"（{'与预期一致' if file_info['sha256_matches_expected'] else '**与预期不一致**'}）",
        f"- 运行环境：Python {report['environment']['python']}，pandas {report['environment']['pandas']}",
        "",
        "> 本报告由 `src/data/check_raw.py` 自动生成，只读检查，未修改原始数据。",
        "",
        "## 1. 字段与行数",
        "",
        f"- 字段：`{report['schema']['columns']}`",
        f"- 行数：{report['schema']['row_count']:,}",
        "",
        "## 2. 缺失值",
        "",
        "| 字段 | 空白 | 疑似占位符 |",
        "| --- | ---: | ---: |",
    ]
    for column, info in report["missing"].items():
        parts.append(f"| `{column}` | {info['blank']} | {info['placeholder_literal']} |")

    parts.extend(
        [
            "",
            "## 3. 类别分布",
            "",
            "| 类别 | 样本数 | 占比 |",
            "| --- | ---: | ---: |",
        ]
    )
    total = sum(labels["category_counts"].values())
    for category, count in sorted(
        labels["category_counts"].items(), key=lambda item: item[1], reverse=True
    ):
        parts.append(f"| {category} | {count:,} | {count / total:.2%} |")
    parts.extend(
        [
            "",
            f"- 类别数：{labels['category_count']}",
            f"- 不均衡比（最大/最小）：{balance['imbalance_ratio']}",
            f"- 未预期类别：{labels['unexpected_categories'] or '无'}",
            f"- 缺失类别：{labels['missing_categories'] or '无'}",
        ]
    )
    if balance["tied_sample_counts"]:
        tied = "、".join(f"{count:,}" for count in balance["tied_sample_counts"])
        parts.append(
            f"- **观察到并列样本数**：{tied} 条（多个类别样本数完全相同）。"
            "这通常说明上游对部分类别做了固定数量截断，即当前的不均衡程度包含人为截断因素，"
            "不完全是自然分布。撰写报告时需如实说明，并据此选择 class weight 等处理方式。"
        )

    parts.extend(
        [
            "",
            "## 4. 类别 × 情感分布",
            "",
            "| 类别 | 负向(0) | 正向(1) |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, counts in labels["category_by_sentiment"].items():
        parts.append(f"| {category} | {counts.get('0', 0):,} | {counts.get('1', 0):,} |")
    parts.append(
        f"\n情感取值统计：{labels['sentiment_counts']}；非法标签 {labels['invalid_sentiment_count']} 条。"
    )
    balanced = labels["perfectly_balanced_categories"]
    if balanced:
        parts.append(
            f"\n**注意到 {len(balanced)} 个类别（{'、'.join(balanced)}）的正负样本数完全相等。**"
            "这不符合真实评论的自然分布，说明上游对情感标签做过人工配平。"
            "扩展任务（情感二分类）因此比真实场景简单，报告中应如实说明这一限制，"
            "不能把高分直接解读为情感建模能力强。"
        )

    parts.extend(
        [
            "",
            "## 5. 重复情况",
            "",
            "重复度按两种文本口径分别统计，两者不可混用：",
            "",
            "| 口径 | 重复行 | 重复组 | 最大一组 | 去重后剩余 |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| 原文 | {raw_duplicates['extra_rows']:,} | {raw_duplicates['duplicated_groups']:,} | "
            f"{raw_duplicates['largest_group_size']} | {raw_duplicates['rows_after_dedup']:,} |",
            f"| 归一化（去首尾空白并合并连续空白） | {normalized_duplicates['extra_rows']:,} | "
            f"{normalized_duplicates['duplicated_groups']:,} | {normalized_duplicates['largest_group_size']} | "
            f"{normalized_duplicates['rows_after_dedup']:,} |",
            "",
            f"- 完全重复行（三个字段全部相同）：{duplicates['exact_duplicate_rows']:,}",
            f"- 两种口径的差异来自 {duplicates['normalization_merge_groups']} 组仅内部空白不同的文本",
            f"- **冲突重复组**：{normalized_duplicates['conflicting_groups']:,}"
            f"（涉及 {normalized_duplicates['conflicting_rows']:,} 行）",
            "  即同一评论文本对应多个类别或多个情感标签，既无法直接用于训练，也会造成跨集合泄漏。",
            "",
            "### 重复最多的文本样本",
            "",
            "| 出现次数 | 类别 | 情感 | 评论文本 |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for example in normalized_duplicates["examples"]:
        categories = " / ".join(example["categories"])
        sentiment = " / ".join(example["labels"])
        parts.append(
            f"| {example['occurrences']} | {categories} | {sentiment} | {example['review']} |"
        )

    parts.extend(
        [
            "",
            "## 6. 异常文本",
            "",
            "| 检查项 | 数量 |",
            "| --- | ---: |",
        ]
    )
    for name, info in anomalies.items():
        parts.append(f"| {ANOMALY_LABELS.get(name, name)} | {info['count']:,} |")

    example_titles = {
        "blank": "空白文本样本",
        "control_chars": "异常控制字符样本（用 repr 显示，便于观察不可见字符）",
        "no_cjk_characters": "不含中文字符样本",
    }
    for name, title in example_titles.items():
        info = anomalies[name]
        if not info["count"]:
            continue
        parts.extend(["", f"### {title}", ""])
        for example in info["examples"]:
            shown = repr(example) if name == "control_chars" else example
            parts.append(f"- `{shown}`")

    parts.extend(
        [
            "",
            "## 7. 文本长度分布（字符）",
            "",
            f"- 最短 {lengths['min']}，最长 {lengths['max']}，"
            f"均值 {lengths['mean']}，中位数 {lengths['median']}",
            "",
            "| 分位数 | 长度 |",
            "| --- | ---: |",
        ]
    )
    for name, value in lengths["quantiles"].items():
        parts.append(f"| {name} | {value:g} |")

    parts.extend(["", "| 长度阈值 | 样本数 |", "| --- | ---: |"])
    for name, value in lengths["short_text_counts"].items():
        parts.append(f"| ≤ {name.split('_')[1]} 字符 | {value:,} |")

    parts.extend(["", "### 最长的评论文本", "", "| 长度 | 文本 |", "| ---: | --- |"])
    for example in lengths["longest_examples"]:
        parts.append(f"| {example['length']} | {example['review']} |")

    parts.extend(
        [
            "",
            "> 注意：以上为**字符**长度。BERT 的 `max_length` 必须由 tokenizer 实测的 "
            "token 长度分布确定，属于 M4 的独立步骤，不能直接用字符长度替代。",
            "",
            "## 8. 与 `data/README.md` 已记录说法的比对",
            "",
        ]
    )
    parts.extend(f"- {line}" for line in compare_with_documented_claims(report))
    parts.append("")
    return "\n".join(parts)


def write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """写出 JSON 与 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "raw_quality_report.json"
    markdown_path = output_dir / "raw_quality_report.md"

    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="原始数据质量检查（只读）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    config = load_config(args.config)
    raw_path = resolve_from_root(config["data"]["raw_path"])

    report = build_report(raw_path, config)
    json_path, markdown_path = write_outputs(report, args.output_dir)

    print(render_console_summary(report))
    print(f"\nJSON 报告    : {public_data_path(json_path)}")
    print(f"Markdown 报告: {public_data_path(markdown_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
