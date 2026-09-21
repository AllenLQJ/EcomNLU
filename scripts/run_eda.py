"""生成探索性数据分析（EDA）图表、EDA 报告以及 max_length 结论。

本脚本是 M2 数据闭环的收尾环节：它不训练任何模型，只负责把已经落盘的
数据产物转换成可展示、可答辩的统计图表和结论文字。

Usage:
    python scripts/run_eda.py

输入文件：
    - 原始 CSV：由 ``configs/*.yaml`` 中的 ``data.raw_path`` 指向，只读，
      用于报告原始样本量；
    - 清洗统计 JSON：``reports/data-cleaning-stats.json``，由
      ``clean_data.py`` 产出，提供清洗各步骤的前后行数，用于绘制清洗漏斗图；
    - 三个划分文件：``data/processed/{train,dev,test}.csv``，由
      ``split_data.py`` 产出（8:1:1 分层划分）。

输出文件：
    - 6 张图：``reports/figures/`` 下的类别分布图、类别-情感堆叠图、
      评论长度直方图、长度箱线图、长度累计分布图（CDF）、清洗漏斗图；
    - EDA 报告：``reports/eda-report.md``（路径可由 ``--report`` 覆盖）；
    - 长度覆盖率表：``reports/eda-length-coverage.json``，即
      ``max_length_coverage_on_train`` 字段，供评估报告直接引用，
      保证「图表结论」与「训练时实际使用的 max_length」来自同一次统计。

关键设计理由：
    1. 图表必须描述「模型真正拿来训练的数据」，因此类别、情感、长度三类
       统计统一基于清洗后的数据与 train 划分，而不是原始 CSV，避免出现
       报告与训练数据不一致的假象；
    2. ``REJECTED_BASELINE_LENGTH = 32`` 是原「投满分」中文新闻分类项目为
       短标题设定的 max_length。这里刻意把它作为一个**被否定的对照项**保留，
       目的是用本数据集的真实覆盖率证明：新闻标题的 ``max_len=32`` 不能
       照搬过来，否则会截断一半以上的电商评论；
    3. 覆盖率表交给 ``reports/eda-report.md``（经
       ``build_max_length_recommendation`` 转成结论文字）以及
       ``reports/eda-length-coverage.json``（机器可读）共同使用，使 BERT
       阶段的 max_length 决策有可追溯的量化依据，而不是凭感觉选取。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.config import ensure_directories, load_config, project_path  # noqa: E402
from src.data.eda import (  # noqa: E402
    MAX_LENGTH_CANDIDATES,
    collect_word_frequencies,
    plot_category_distribution,
    plot_category_sentiment,
    plot_cleaning_funnel,
    plot_length_boxplot,
    plot_length_cdf,
    plot_length_histogram,
    plot_top_words_by_category,
    plot_wordcloud,
    render_eda_report,
    setup_chinese_font,
)

# 原「投满分」新闻分类项目针对短标题使用了 max_length=32。
# 这里保留该值仅作为被否定的对照，用来证明它不能沿用到电商评论上。
REJECTED_BASELINE_LENGTH = 32


def parse_args() -> argparse.Namespace:
    """解析命令行参数：配置文件路径、图表输出目录与 EDA 报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--figures-dir", default="reports/figures", help="Figure directory")
    parser.add_argument("--report", default="reports/eda-report.md", help="Report path")
    return parser.parse_args()


def build_max_length_recommendation(table: list[dict], train_rows: int) -> str:
    """把覆盖率表转成一段有数据支撑、可追溯的 max_length 结论文字。

    参数：
        table：``plot_length_cdf`` 返回的覆盖率表，每行包含
            ``max_length``、``coverage``、``truncated_rows`` 与 ``truncated``。
        train_rows：训练集样本量，用于在结论中标注统计口径。

    返回：
        Markdown 片段，包含「不能照搬 max_len=32」的否决理由、推荐值
        ``max_length=128``、高覆盖备选 ``max_length=256``，以及满足 95%
        覆盖率下限的最小候选值。
    """
    by_length = {row["max_length"]: row for row in table}
    rejected = by_length[REJECTED_BASELINE_LENGTH]
    covered_128 = by_length[128]
    covered_256 = by_length[256]

    # 在候选中挑出覆盖率仍不低于 95% 的最小长度，作为「足够用」的下限。
    sufficient = [row for row in table if row["coverage"] >= 0.95]
    recommended = sufficient[0] if sufficient else table[-1]

    return (
        "**不能照搬原投满分项目的 `max_len=32`。** 本数据集的评论文本远长于新闻标题："
        f"`max_length=32` 只能完整保留 {rejected['coverage']:.2%} 的评论，"
        f"会截断 {rejected['truncated_rows']:,} 条（{rejected['truncated']:.2%}）样本，"
        "超过一半的评论文本信息会被丢弃，其中包含大量判断类别和情感的关键描述。\n\n"
        f"- **推荐默认值 `max_length=128`**：完整保留 {covered_128['coverage']:.2%} 的训练样本，"
        f"只截断 {covered_128['truncated_rows']:,} 条；在覆盖率与显存/耗时之间最平衡。\n"
        f"- **高覆盖备选 `max_length=256`**：完整保留 {covered_256['coverage']:.2%}，"
        f"截断量降到 {covered_256['truncated_rows']:,} 条，代价是注意力计算量显著上升。\n"
        f"- **95% 覆盖率下限**：满足该标准的最小候选值是 `max_length={recommended['max_length']}`"
        f"（覆盖率 {recommended['coverage']:.2%}）。\n"
        "- 训练时应记录实际使用的 max_length，并在评估报告中与本节结论对应；"
        "若显存不足而必须下调，需说明下调后新增的截断比例，不能静默修改。\n"
        f"- 训练集规模为 {train_rows:,} 条，以上覆盖率均在训练集上统计，"
        "验证集与测试集必须使用同一 max_length，以保证评估公平。"
    )


def main() -> int:
    """执行完整 EDA 流程并写出图表、报告与长度覆盖率表。

    流程：先校验原始 CSV、清洗结果、清洗统计 JSON 与三个划分文件是否齐备
    （缺失时抛出 ``FileNotFoundError``，且错误信息中直接给出需要先运行的
    脚本名），随后依次绘制 6 张图、把长度覆盖率表转成 max_length 结论、
    渲染 ``reports/eda-report.md``，最后把覆盖率表写入
    ``reports/eda-length-coverage.json`` 供报告与后续 BERT 阶段引用。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    sentiment_column = data_config["sentiment_column"]

    raw_path = project_path(data_config["raw_path"])
    clean_path = project_path("data/interim/reviews_clean.csv")
    cleaning_stats_path = project_path("reports/data-cleaning-stats.json")
    for required in (raw_path, clean_path, cleaning_stats_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing input: {required}. Run check_data_quality.py, clean_data.py "
                "and split_data.py first."
            )

    font_name = setup_chinese_font()
    print(f"[eda] chinese font: {font_name or 'NOT FOUND'}")

    raw = pd.read_csv(raw_path, encoding="utf-8")
    clean = pd.read_csv(clean_path, encoding="utf-8")
    cleaning_stats = json.loads(cleaning_stats_path.read_text(encoding="utf-8"))

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "dev", "test"):
        split_path = project_path(f"{data_config['processed_dir']}/{name}.csv")
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing split: {split_path}. Run 'python scripts/split_data.py' first."
            )
        splits[name] = pd.read_csv(split_path, encoding="utf-8")

    figures_dir = project_path(args.figures_dir)
    ensure_directories(figures_dir)

    figure_paths = {
        "category": f"{args.figures_dir}/category_distribution.png",
        "sentiment": f"{args.figures_dir}/category_sentiment_stacked.png",
        "length_hist": f"{args.figures_dir}/review_length_histogram.png",
        "length_box": f"{args.figures_dir}/review_length_boxplot.png",
        "length_cdf": f"{args.figures_dir}/review_length_cdf.png",
        "cleaning": f"{args.figures_dir}/cleaning_funnel.png",
        "wordcloud": f"{args.figures_dir}/wordcloud_train.png",
        "top_words": f"{args.figures_dir}/top_words_by_category.png",
    }

    conclusions: dict[str, str] = {}
    conclusions["category"] = plot_category_distribution(
        clean, category_column=category_column, output_path=project_path(figure_paths["category"])
    )
    print("[eda] figure: category_distribution.png")

    conclusions["sentiment"] = plot_category_sentiment(
        clean,
        category_column=category_column,
        sentiment_column=sentiment_column,
        output_path=project_path(figure_paths["sentiment"]),
    )
    print("[eda] figure: category_sentiment_stacked.png")

    conclusions["length_hist"] = plot_length_histogram(
        clean[text_column], output_path=project_path(figure_paths["length_hist"])
    )
    print("[eda] figure: review_length_histogram.png")

    conclusions["length_box"] = plot_length_boxplot(
        clean,
        text_column=text_column,
        category_column=category_column,
        output_path=project_path(figure_paths["length_box"]),
    )
    print("[eda] figure: review_length_boxplot.png")

    conclusions["length_cdf"], length_table = plot_length_cdf(
        splits["train"][text_column],
        output_path=project_path(figure_paths["length_cdf"]),
        candidates=MAX_LENGTH_CANDIDATES,
    )
    print("[eda] figure: review_length_cdf.png")

    conclusions["cleaning"] = plot_cleaning_funnel(
        cleaning_stats["steps"], output_path=project_path(figure_paths["cleaning"])
    )
    print("[eda] figure: cleaning_funnel.png")

    # 分词与词频统计。只统计训练集：与后续 TF-IDF 的拟合口径保持一致，
    # 避免用测试集信息影响特征工程相关的判断。
    # jieba 对 5 万条评论分词约需 40 秒，因此只遍历一遍，同时得到整体与各类别词频。
    print(f"[eda] tokenizing train split ({splits['train'].shape[0]:,} rows), this takes a while...")
    word_frequencies, category_frequencies, kept_token_total = collect_word_frequencies(
        splits["train"],
        text_column=text_column,
        category_column=category_column,
    )
    print(
        f"[eda] tokenized: kept {kept_token_total:,} tokens, "
        f"{len(word_frequencies):,} distinct words in wordcloud, "
        f"{len(category_frequencies)} categories"
    )

    conclusions["wordcloud"] = plot_wordcloud(
        word_frequencies, output_path=project_path(figure_paths["wordcloud"])
    )
    print("[eda] figure: wordcloud_train.png")

    conclusions["top_words"] = plot_top_words_by_category(
        category_frequencies, output_path=project_path(figure_paths["top_words"])
    )
    print("[eda] figure: top_words_by_category.png")

    recommendation = build_max_length_recommendation(length_table, int(splits["train"].shape[0]))

    report = render_eda_report(
        conclusions=conclusions,
        figure_paths=figure_paths,
        length_table=length_table,
        font_name=font_name,
        word_frequencies=word_frequencies,
        dataset_summary={
            "raw_rows": int(raw.shape[0]),
            "clean_rows": int(clean.shape[0]),
            "train_rows": int(splits["train"].shape[0]),
            "dev_rows": int(splits["dev"].shape[0]),
            "test_rows": int(splits["test"].shape[0]),
        },
        max_length_recommendation=recommendation,
    )
    report_path = project_path(args.report)
    ensure_directories(report_path.parent)
    report_path.write_text(report, encoding="utf-8")

    stats_path = project_path("reports/eda-length-coverage.json")
    stats_path.write_text(
        json.dumps(
            {
                "max_length_coverage_on_train": length_table,
                "chinese_font": font_name,
                "word_frequency_on_train": {
                    "kept_token_total": kept_token_total,
                    "distinct_words": len(word_frequencies),
                    "top_50": [
                        {"word": word, "count": count} for word, count in word_frequencies[:50]
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[eda] top 20 words (train):")
    for rank, (word, count) in enumerate(word_frequencies[:20], start=1):
        print(f"[eda]   {rank:>2}. {word}  {count:,}")
    print("[eda] max_length coverage (train):")
    for row in length_table:
        print(
            f"[eda]   max_length={row['max_length']:>4} coverage={row['coverage']:.4%} "
            f"truncated={row['truncated_rows']:,}"
        )
    print(f"[eda] report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
