"""统计本数据集的 token 长度分布，为 ``max_length`` 提供依据。

AGENTS.md §5.4 与契约 §2.3 都要求 ``max_length`` 由**本数据集**的 token 长度分布
决定，不得照搬其它项目的 32。本模块负责给出这个依据，并把结论落成报告，
使 ``metadata.json`` 里的 ``max_length`` 有可追溯的来源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.models.artifact import PROCESSED_DIR, read_split

# 候选截断长度：既覆盖常见取值，也让报告能呈现"覆盖率—代价"的取舍。
CANDIDATE_LENGTHS = (32, 48, 64, 96, 128, 160, 192, 256, 384, 512)


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """逐条统计 token 数（含 ``[CLS]``、``[SEP]``，不做截断）。"""
    return [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in texts
    ]


def distribution_summary(lengths: Sequence[int]) -> dict[str, Any]:
    """给出分位数、极值与各候选截断长度的覆盖率。"""
    import numpy as np

    values = np.asarray(lengths, dtype=np.int64)
    quantiles = {
        f"p{int(q * 100)}": float(np.quantile(values, q))
        for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    return {
        "samples": int(values.size),
        "mean": round(float(values.mean()), 2),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "quantiles": quantiles,
        "coverage": {
            str(cap): round(float((values <= cap).mean() * 100), 2)
            for cap in CANDIDATE_LENGTHS
        },
        "over_512": int((values > 512).sum()),
    }


def recommend_max_length(coverage: dict[str, float], min_coverage: float = 95.0) -> int:
    """在候选长度中选出**覆盖率首次达到阈值的那个**，即满足覆盖的最小截断长度。"""
    for cap in CANDIDATE_LENGTHS:
        if coverage[str(cap)] >= min_coverage:
            return cap
    return CANDIDATE_LENGTHS[-1]


def compute(
    pretrained_dir: Path,
    split: str = "train",
    processed_dir: Path | None = None,
    min_coverage: float = 95.0,
) -> dict[str, Any]:
    """用给定分词器统计某个划分的 token 长度分布并给出 ``max_length`` 建议。"""
    from transformers import AutoTokenizer

    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    tokenizer = AutoTokenizer.from_pretrained(str(pretrained_dir))
    mapping_rows = read_split(split, None, directory)
    texts = [text for text, _ in mapping_rows]
    lengths = token_lengths(tokenizer, texts)
    summary = distribution_summary(lengths)
    summary["split"] = split
    summary["tokenizer"] = str(pretrained_dir)
    summary["recommend_max_length"] = recommend_max_length(summary["coverage"], min_coverage)
    summary["min_coverage"] = min_coverage
    return summary
