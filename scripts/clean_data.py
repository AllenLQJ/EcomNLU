"""清洗原始评论数据集，并写出中间层 CSV 文件与清洗报告。

本脚本读取只读的原始数据，按固定顺序执行清洗步骤：去除空白文本、清理异常控制字符
与 HTML 实体、剔除无意义超短文本、删除重复评论等；随后把清洗结果写入中间层 CSV，
并生成 Markdown 清洗报告和 JSON 步骤统计，便于追溯每一步删除了多少行、修改了多少行。

Usage:
    python scripts/clean_data.py

输入文件：
    - 原始数据集：由配置项 data.raw_path 指定，只读，不会被修改。
    - 配置文件：默认 configs/base.yaml，可用 --config 指定其他 YAML 配置。

输出文件：
    - 中间层清洗数据：默认 data/interim/reviews_clean.csv，可用 --output 覆盖。
    - Markdown 清洗报告：默认 reports/data-cleaning-report.md，可用 --report 覆盖。
    - JSON 步骤统计：默认 reports/data-cleaning-stats.json，可用 --stats-json 覆盖。

前置依赖：
    - 无强制前置脚本，但建议先运行 check_data_quality.py 了解原始数据的质量问题。
    - 本脚本的输出是 split_data.py 的必需输入，划分数据集前必须先运行本脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 先把项目根目录加入 sys.path，再导入 src 包，因此下面的导入位置必须保持在此处
from src.data.cleaning import clean_reviews, render_cleaning_report  # noqa: E402  # 延迟导入
from src.data.config import ensure_directories, load_config, project_path  # noqa: E402  # 延迟导入
from src.data.raw_io import load_raw_dataframe, write_contract_csv  # noqa: E402  # 延迟导入


# 中间层清洗数据的默认输出路径
DEFAULT_OUTPUT = "data/interim/reviews_clean.csv"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果，包含 config、output、report 与 stats_json 字段。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Interim CSV path")
    parser.add_argument(
        "--report",
        default="reports/data-cleaning-report.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--stats-json",
        default="reports/data-cleaning-stats.json",
        help="Machine-readable step statistics destination",
    )
    return parser.parse_args()


def main() -> int:
    """执行数据清洗流水线，并写出中间层 CSV、Markdown 报告与 JSON 统计文件。

    Returns:
        int: 进程退出码，成功时返回 0。
    """
    args = parse_args()
    # 加载 YAML 配置并取出数据列名与路径等配置项
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    sentiment_column = data_config["sentiment_column"]

    # 解析原始数据、输出数据与报告的实际路径
    raw_path = project_path(data_config["raw_path"])
    output_path = project_path(args.output)
    report_path = project_path(args.report)

    print(f"[clean] reading raw dataset: {raw_path}")
    raw = load_raw_dataframe(raw_path)
    # 记录清洗前的行数，用于报告清洗前后样本量变化
    rows_before = int(raw.shape[0])

    # 执行清洗步骤，result 中保留了每一步的行数变化统计
    result = clean_reviews(
        raw,
        text_column=text_column,
        category_column=category_column,
        sentiment_column=sentiment_column,
    )
    cleaned = result.dataframe

    # 按数据契约写出中间层清洗数据：UTF-8 不带 BOM、不写出行索引、
    # 列顺序固定为 sample_id, review, cat, label
    ensure_directories(output_path.parent, report_path.parent)
    write_contract_csv(cleaned, output_path)

    # 渲染并写出 Markdown 清洗报告
    report = render_cleaning_report(
        result,
        source_path=data_config["raw_path"],
        output_path=args.output,
    )
    report_path.write_text(report, encoding="utf-8")

    # 汇总每一步清洗的行数变化，写出机器可读的 JSON 统计文件
    stats_path = project_path(args.stats_json)
    ensure_directories(stats_path.parent)
    stats_path.write_text(
        json.dumps(
            {
                "rows_before": result.rows_before,
                "rows_after": result.rows_after,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "rows_before": step.rows_before,
                        "rows_after": step.rows_after,
                        "rows_removed": step.rows_removed,
                        "rows_modified": step.rows_modified,
                    }
                    for step in result.steps
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[clean] stats written: {stats_path}")

    # 逐条打印清洗步骤的统计信息
    print(f"[clean] steps applied: {len(result.steps)}")
    for step in result.steps:
        print(
            f"[clean]   {step.name}: rows {step.rows_before:,} -> {step.rows_after:,} "
            f"(removed {step.rows_removed:,}, modified {step.rows_modified:,})"
        )
    print(f"[clean] rows: {rows_before:,} -> {cleaned.shape[0]:,}")
    print(f"[clean] output written: {output_path}")
    print(f"[clean] report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
