"""对清洗后的评论数据集进行可复现的分层划分。

模块职责：
    把清洗后的评论数据按 8:1:1 划分为训练集、验证集与测试集，
    并校验划分结果是否存在跨集合的文本泄漏。

主要输入与输出：
    输入：清洗后的 pandas DataFrame，以及文本列名、类别列名、目标比例与随机种子。
    输出：SplitResult（各集合的数据表与划分报告所需的事实数据），
    并提供跨集合文本泄漏检查结果与类别分布表，供 `scripts/split_data.py` 生成 Markdown 报告。

关键设计理由与约束：
    两个约束必须同时满足：

    1. 训练 / 验证 / 测试划分必须按类别分层，因为数据集严重不均衡
       （最小类别 575 条，最大类别 10,000 条）。
    2. 同一条评论文本绝不能出现在一个以上的集合中，否则报告的测试分数会因泄漏而虚高。

    直接使用 ``train_test_split(..., stratify=cat)`` 只能满足第一个约束：
    原始数据中同一文本会出现在多个类别下。因此本模块把每一个不同的评论文本视为一个分组，
    以评论文本为分组单位整体分配到各集合，并按配置的比例贪心逼近目标，
    从而在绝不拆分任何分组的前提下保持类别比例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")

# 配额修复的目标容差（行）。接口契约的校验标准是 ±1.0 行，
# 这里收紧到 ±0.5 以留出余量，避免因边界舍入被判不合格。
QUOTA_TOLERANCE = 0.5
MAX_REPAIR_MOVES = 200


@dataclass
class SplitResult:
    """各集合的数据表，以及生成划分报告所需的事实数据。

    Attributes:
        splits: 集合名称到该集合数据表的映射。
        ratios: 集合名称到目标占比的映射。
        seed: 控制分组处理顺序的随机种子。
        group_count: 文本分组总数。
        multi_category_groups: 同时出现在多个类别下的文本分组数。
        largest_group_size: 最大文本分组的行数。
    """

    splits: dict[str, pd.DataFrame]
    ratios: dict[str, float]
    seed: int
    group_count: int
    multi_category_groups: int
    largest_group_size: int
    # 配额修复阶段搬运的分组数，0 表示贪心分配的结果已经满足容差要求。
    repair_moves: int = 0


def _repair_quota_overshoot(
    *,
    group_to_split: dict[Any, str],
    group_counts: dict[Any, dict[Any, float]],
    assigned: dict[str, dict[Any, float]],
    targets: dict[str, dict[Any, float]],
    categories: Sequence[Any],
    split_names: Sequence[str],
    tolerance: float = QUOTA_TOLERANCE,
    max_moves: int = MAX_REPAIR_MOVES,
) -> int:
    """把超出类别配额容差的整组样本搬到欠配的集合。

    贪心分配以「整组投放」为单位，当某类别包含跨类别的重复文本组时会略微超出配额；
    接口契约要求各类别的分层比例落在 ±1.0 行以内，因此需要一轮修复。

    修复仍然只搬运**整组**，所以「同一评论文本不跨划分边界」这一硬约束不会被破坏。
    搬运时会同时检查两件事：源集合在该组涉及的每个类别上都必须处于超配状态
    （否则搬走会造成新的亏缺），且目标集合接收后不得超出容差。

    参数：
        group_to_split：分组文本到目标集合的映射，将被就地更新。
        group_counts：分组文本到「各类别行数」的映射。
        assigned：各集合各类别的当前行数，将被就地更新。
        targets：各集合各类别的目标行数。
        categories：全部类别。
        split_names：全部集合名称，其顺序用于确定性的取最小操作。
        tolerance：允许的最大偏差（行）。
        max_moves：搬运次数上限，用于兜底防止死循环。

    返回：
        实际搬运的分组数。
    """
    moves = 0
    while moves < max_moves:
        # 找出当前偏差最大且确实超配的（集合, 类别）
        worst: tuple[float, str, Any] | None = None
        for category in categories:
            for name in split_names:
                excess = assigned[name][category] - targets[name][category]
                if excess > tolerance and (worst is None or excess > worst[0]):
                    worst = (excess, name, category)
        if worst is None:
            # 所有集合的所有类别都已落入容差
            break

        _, over_split, category = worst
        under_split = min(
            split_names,
            key=lambda name: assigned[name][category] - targets[name][category],
        )
        if under_split == over_split:
            break

        candidate_text: Any = None
        candidate_counts: dict[Any, float] = {}
        for text, name in group_to_split.items():
            if name != over_split:
                continue
            counts = group_counts[text]
            if any(
                assigned[over_split][c] - targets[over_split][c] <= 0 for c in counts
            ):
                continue
            if any(
                assigned[under_split][c] + value - targets[under_split][c] > tolerance
                for c, value in counts.items()
            ):
                continue
            candidate_text = text
            candidate_counts = counts
            break

        if candidate_text is None:
            # 在容差约束下已经找不到可搬运的整组样本，停止修复
            break

        group_to_split[candidate_text] = under_split
        for c, value in candidate_counts.items():
            assigned[over_split][c] -= value
            assigned[under_split][c] += value
        moves += 1
    return moves


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    """校验划分比例是否合法。

    Args:
        ratios: 集合名称到目标占比的映射。

    Raises:
        ValueError: 当比例映射为空，或各项比例之和不等于 1.0 时抛出。
    """
    if not ratios:
        raise ValueError("At least one split ratio is required")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total!r}")


def split_by_group(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    category_column: str,
    ratios: Mapping[str, float],
    seed: int = 42,
    split_names: Sequence[str] = DEFAULT_SPLIT_NAMES,
) -> SplitResult:
    """划分数据表，保证同一评论文本不跨越任何集合边界。

    按类别配额贪心分配文本分组：每个文本分组整体进入某一个集合，
    因此同一文本不会同时出现在训练集与测试集，避免指标泄漏。

    Args:
        dataframe: 待划分的清洗后数据表。
        text_column: 存放评论文本的列名，即分组键。
        category_column: 用于分层的列名。
        ratios: 集合名称到目标占比的映射，所有集合的占比之和为 1。
        seed: 随机种子，控制同样大小分组的处理顺序。
        split_names: 集合的顺序，用于确定性的平局决胜。

    Returns:
        一个 :class:`SplitResult`，每个集合对应一个数据表。

    Raises:
        ValueError: 当比例非法、输入数据表为空，或某个集合划分后为空时抛出。
        KeyError: 当集合名称未出现在比例映射中，或数据表缺少必需列时抛出。
        RuntimeError: 当存在未能分配到任何集合的行时抛出。
    """
    _validate_ratios(ratios)
    missing = [name for name in split_names if name not in ratios]
    if missing:
        raise KeyError(f"Split names missing from ratios: {missing}")
    for column in (text_column, category_column):
        if column not in dataframe.columns:
            raise KeyError(f"Missing required column: {column}")
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe")

    categories = list(pd.unique(dataframe[category_column]))
    totals = dataframe[category_column].value_counts()
    targets = {
        name: {category: ratios[name] * float(totals.get(category, 0)) for category in categories}
        for name in split_names
    }

    composition = (
        dataframe.groupby(text_column, sort=False)[category_column]
        .value_counts()
        .unstack(fill_value=0)
    )
    composition = composition.reindex(columns=categories, fill_value=0)

    group_sizes = composition.sum(axis=1)
    multi_category_groups = int((composition.gt(0).sum(axis=1) > 1).sum())

    # 先用随机种子打乱，再按分组大小做稳定的降序排序，使处理顺序可复现，
    # 同时让大分组在剩余配额还宽裕的时候就被分配下去。
    shuffled_index = np.random.default_rng(seed).permutation(len(composition))
    composition = composition.iloc[shuffled_index]
    composition = composition.loc[
        group_sizes.sort_values(ascending=False, kind="stable").index
    ]

    assigned: dict[str, dict[Any, float]] = {
        name: {category: 0.0 for category in categories} for name in split_names
    }
    group_to_split: dict[Any, str] = {}
    group_counts: dict[Any, dict[Any, float]] = {}

    for text, row in composition.iterrows():
        counts = {category: float(row[category]) for category in categories if row[category]}
        group_counts[text] = counts
        best_split = split_names[0]
        best_score = float("inf")
        for name in split_names:
            score = 0.0
            for category, count in counts.items():
                target = targets[name][category]
                if target <= 0:
                    # 某类别在该集合的配额为 0 时，其代价实际上相当于无穷大，
                    # 但不能除以零，因此用极小值代替。
                    target = 1e-9
                score += (assigned[name][category] + count) / target
            if score < best_score - 1e-12:
                best_score = score
                best_split = name
        group_to_split[text] = best_split
        for category, count in counts.items():
            assigned[best_split][category] += count

    # 贪心分配以「整组投放」为单位，当某个类别含跨类别的重复文本组时会略微超出配额。
    # 接口契约要求各类别的分层比例落在 ±1.0 行以内，因此这里做一轮配额修复：
    # 仍然只搬运**整组**，所以「同一评论文本不跨集合」的约束不会被破坏。
    repair_moves = _repair_quota_overshoot(
        group_to_split=group_to_split,
        group_counts=group_counts,
        assigned=assigned,
        targets=targets,
        categories=categories,
        split_names=split_names,
    )

    split_labels = dataframe[text_column].map(group_to_split)
    if split_labels.isna().any():
        raise RuntimeError("Some rows were not assigned to a split")

    splits: dict[str, pd.DataFrame] = {}
    for name in split_names:
        subset = dataframe.loc[(split_labels == name).to_numpy()].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"Split '{name}' ended up empty; check the ratios")
        splits[name] = subset

    return SplitResult(
        splits=splits,
        ratios={name: float(ratios[name]) for name in split_names},
        seed=seed,
        group_count=int(len(composition)),
        multi_category_groups=multi_category_groups,
        largest_group_size=int(group_sizes.max()),
        repair_moves=repair_moves,
    )


def check_text_leakage(
    splits: Mapping[str, pd.DataFrame], *, text_column: str
) -> dict[str, Any]:
    """校验没有任何评论文本出现在一个以上的集合中。

    Args:
        splits: 集合名称到该集合数据表的映射。
        text_column: 存放评论文本的列名。

    Returns:
        各集合两两之间的重叠数量，以及出现在多个集合中的文本数量。
        任何非零值都表示该划分必须被拒绝。
    """
    text_sets = {
        name: set(frame[text_column].astype("string").fillna(""))
        for name, frame in splits.items()
    }
    names = list(splits)
    overlaps: dict[str, int] = {}
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            overlaps[f"{first}__{second}"] = len(text_sets[first] & text_sets[second])

    seen: dict[str, set[str]] = {}
    for name, texts in text_sets.items():
        for text in texts:
            seen.setdefault(text, set()).add(name)
    leaking = sum(1 for owners in seen.values() if len(owners) > 1)

    return {
        "pairwise_overlaps": overlaps,
        "total_overlapping_texts": int(leaking),
        "unique_texts_per_split": {name: len(texts) for name, texts in text_sets.items()},
        "leakage_free": leaking == 0 and all(value == 0 for value in overlaps.values()),
    }


def distribution_table(
    splits: Mapping[str, pd.DataFrame], *, category_column: str
) -> list[dict[str, Any]]:
    """为每个集合统计各类别的数量与占比。

    Args:
        splits: 集合名称到该集合数据表的映射。
        category_column: 类别列名。

    Returns:
        逐条记录「集合 / 类别 / 数量 / 占该集合比例 / 占该类别比例」的列表。
    """
    rows: list[dict[str, Any]] = []
    for name, frame in splits.items():
        counts = frame[category_column].value_counts()
        total = int(frame.shape[0])
        for category, count in counts.items():
            rows.append(
                {
                    "split": name,
                    "category": str(category),
                    "count": int(count),
                    "share_of_split": count / total if total else 0.0,
                    "share_of_category": count / float(counts.sum()),
                }
            )
    return rows


def render_split_report(
    result: SplitResult,
    *,
    leakage: Mapping[str, Any],
    source_path: str,
    output_paths: Mapping[str, str],
    text_column: str,
    category_column: str,
    sentiment_column: str,
) -> str:
    """把划分结果与泄漏检查结果渲染成 Markdown 报告。

    报告正文属于程序输出，其内容与措辞保持原样，便于与历史报告对比。

    Args:
        result: :func:`split_by_group` 返回的划分结果。
        leakage: :func:`check_text_leakage` 返回的泄漏检查结果。
        source_path: 输入文件路径，写入报告。
        output_paths: 集合名称到输出文件路径的映射，写入报告。
        text_column: 评论文本列名。
        category_column: 类别列名。
        sentiment_column: 情感标签列名。

    Returns:
        完整的 Markdown 报告文本。
    """
    splits = result.splits
    total_rows = sum(int(frame.shape[0]) for frame in splits.values())

    lines: list[str] = [
        "# 数据集划分报告",
        "",
        "## 1. 划分策略",
        "",
        f"- 输入文件：`{source_path}`",
        f"- 随机种子：{result.seed}",
        f"- 目标比例："
        + "、".join(f"{name} {ratio:.0%}" for name, ratio in result.ratios.items()),
        f"- 分层依据：`{category_column}`（类别）",
        f"- 分组依据：`{text_column}`（评论文本，同一文本整体进入同一集合）",
        f"- 文本分组总数：{result.group_count:,}",
        f"- 跨多个类别的文本组数：{result.multi_category_groups:,}",
        f"- 最大文本组行数：{result.largest_group_size:,}",
        f"- 配额修复搬运的分组数：{result.repair_moves:,}",
        "",
        "说明：原始数据中同一评论文本可能出现在多个类别下。若直接按行随机划分，",
        "同一文本会同时落入训练集和测试集，造成指标泄漏。因此本流程以评论文本为单位",
        "整体分配，并在分配时按类别配额贪心选择目标比例偏差最小的集合，",
        "从而同时满足「按类别分层」与「无跨集合文本泄漏」。",
        "",
        "由于以整组为单位投放，当某类别包含跨类别的重复文本组时会略微超出配额。",
        "因此贪心分配之后还有一轮**配额修复**：把超配集合中的整组样本搬到欠配集合，",
        "使各类别的分层比例落入接口契约的 ±1.0 行容差内。",
        "修复同样只搬运整组，不会破坏「同一评论文本不跨集合」这一硬约束。",
        f"本次共搬运 {result.repair_moves:,} 个分组。",
        "",
        "## 2. 各集合规模",
        "",
        "| 集合 | 样本数 | 实际占比 | 目标占比 | 偏差 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, frame in splits.items():
        actual = frame.shape[0] / total_rows
        target = result.ratios[name]
        lines.append(
            f"| {name} | {frame.shape[0]:,} | {actual:.4%} | {target:.4%} | "
            f"{actual - target:+.4%} |"
        )
    lines.append(f"| **合计** | **{total_rows:,}** | 100% | 100% | - |")

    lines.extend(
        [
            "",
            "## 3. 类别分布（分层校验）",
            "",
            "| 类别 | 全集 | " + " | ".join(f"{name} 数量" for name in splits) + " | "
            + " | ".join(f"{name} 占该类别" for name in splits) + " |",
            "| --- | ---: | " + " | ".join("---:" for _ in splits) + " | "
            + " | ".join("---:" for _ in splits) + " |",
        ]
    )
    category_totals = pd.concat(splits.values())[category_column].value_counts()
    for category in category_totals.index:
        cells = []
        shares = []
        for name, frame in splits.items():
            count = int((frame[category_column] == category).sum())
            cells.append(f"{count:,}")
            shares.append(f"{count / category_totals[category]:.2%}")
        lines.append(
            f"| {category} | {int(category_totals[category]):,} | "
            + " | ".join(cells)
            + " | "
            + " | ".join(shares)
            + " |"
        )

    lines.extend(["", "## 4. 情感分布", ""])
    for name, frame in splits.items():
        counts = frame[sentiment_column].value_counts().sort_index()
        detail = "、".join(f"label={label}: {int(count):,}" for label, count in counts.items())
        positive = counts.get(1, 0) / max(int(counts.sum()), 1)
        lines.append(f"- {name}：{detail}（正向占比 {positive:.2%}）")

    lines.extend(
        [
            "",
            "## 5. 跨集合文本泄漏检查",
            "",
            "| 检查项 | 结果 |",
            "| --- | --- |",
        ]
    )
    for pair, count in leakage["pairwise_overlaps"].items():
        left, right = pair.split("__")
        lines.append(f"| {left} 与 {right} 的重复文本数 | {count} |")
    for name, count in leakage["unique_texts_per_split"].items():
        lines.append(f"| {name} 唯一文本数 | {count:,} |")
    lines.append(f"| 出现在多个集合的文本数 | {leakage['total_overlapping_texts']} |")
    lines.append(
        f"| **结论** | {'通过：无明显跨集合文本泄漏' if leakage['leakage_free'] else '**失败：存在泄漏**'} |"
    )

    lines.extend(["", "## 6. 输出文件", ""])
    for name, path in output_paths.items():
        lines.append(f"- {name}：`{path}`（{splits[name].shape[0]:,} 行）")
    lines.append("")
    return "\n".join(lines)
