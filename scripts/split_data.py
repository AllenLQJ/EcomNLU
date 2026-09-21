"""把清洗后的数据集划分为可复现、无泄漏的训练集、验证集与测试集。

本脚本按评论文本分组进行分层划分（同一条评论文本只会进入一个集合），使用配置中的
固定随机种子保证结果可复现；划分完成后执行跨集合文本重叠检查，确认不存在同一文本
同时出现在训练集与测试集等泄漏情况。划分结果写入处理后数据目录，同时生成 Markdown
划分报告与 JSON 统计文件。检测到文本泄漏时脚本返回非 0 退出码，表示该划分必须废弃。

Usage:
    python scripts/split_data.py

输入文件：
    - 清洗后的中间层 CSV：默认 data/interim/reviews_clean.csv，可用 --input 覆盖。
    - 配置文件：默认 configs/base.yaml，可用 --config 指定其他 YAML 配置。

输出文件：
    - 划分后的数据集：写入配置项 data.processed_dir 下的 train.csv、dev.csv、test.csv。
    - Markdown 划分报告：默认 reports/data-split-report.md，可用 --report 覆盖。
    - JSON 划分统计：默认 reports/data-split-stats.json，可用 --stats-json 覆盖。

前置依赖：
    - 必须先运行 scripts/clean_data.py 生成清洗后的中间层 CSV。
    - 本脚本拒绝回退到原始文件读取，缺少清洗结果时直接报错退出，
      以保证处理后的数据集始终来自文档化的数据流水线。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 先把项目根目录加入 sys.path，再导入第三方包与 src 包，因此下面的导入位置必须保持在此处
import pandas as pd  # noqa: E402  # 延迟导入

from src.data.config import ensure_directories, load_config, project_path  # noqa: E402  # 延迟导入
from src.data.manifest import write_contract_artifacts  # noqa: E402  # 延迟导入
from src.data.raw_io import write_contract_csv  # noqa: E402  # 延迟导入
from src.data.splitting import (  # noqa: E402  # 延迟导入
    check_text_leakage,
    render_split_report,
    split_by_group,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果，包含 config、input、report 与 stats_json 字段。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument(
        "--input", default="data/interim/reviews_clean.csv", help="Cleaned CSV path"
    )
    parser.add_argument(
        "--report", default="reports/data-split-report.md", help="Markdown report path"
    )
    parser.add_argument(
        "--stats-json", default="reports/data-split-stats.json", help="Stats path"
    )
    return parser.parse_args()


def main() -> int:
    """执行数据集划分、文本泄漏检查，并写出划分文件、报告与统计。

    Returns:
        int: 进程退出码，划分成功且无泄漏时返回 0，检测到文本泄漏时返回 1。
    """
    args = parse_args()
    # 加载 YAML 配置，取出列名、划分比例与随机种子
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    sentiment_column = data_config["sentiment_column"]
    seed = int(config["project"]["random_seed"])

    # 只接受清洗后的数据，缺少该文件时明确提示先运行 clean_data.py
    input_path = project_path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}. Run 'python scripts/clean_data.py' first."
        )

    # 划分比例来自配置，保证可复现
    ratios = {
        "train": float(data_config["train_ratio"]),
        "dev": float(data_config["dev_ratio"]),
        "test": float(data_config["test_ratio"]),
    }

    print(f"[split] reading cleaned dataset: {input_path}")
    dataframe = pd.read_csv(input_path, encoding="utf-8")
    print(f"[split] rows={dataframe.shape[0]:,} seed={seed} ratios={ratios}")

    # 按评论文本分组划分，避免同一文本跨集合
    result = split_by_group(
        dataframe,
        text_column=text_column,
        category_column=category_column,
        ratios=ratios,
        seed=seed,
    )

    # 检查各集合之间是否存在重复评论文本
    leakage = check_text_leakage(result.splits, text_column=text_column)

    # 把 train/dev/test 写入处理后数据目录。
    # 走 write_contract_csv 而不是 DataFrame.to_csv，由它强制保证契约要求的
    # 列顺序 sample_id, review, cat, label 与 UTF-8 无 BOM、不写行索引。
    processed_dir = project_path(data_config["processed_dir"])
    ensure_directories(processed_dir)
    output_paths: dict[str, str] = {}
    split_filenames: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, frame in result.splits.items():
        relative = f"{data_config['processed_dir']}/{name}.csv"
        write_contract_csv(frame, project_path(relative))
        output_paths[name] = relative
        split_filenames[name] = f"{name}.csv"
        counts[name] = int(frame.shape[0])
        print(f"[split]   {name}: {frame.shape[0]:,} rows -> {relative}")

    # 生成契约要求的 label_mapping.json 与 split_manifest.json。
    # 清洗前后样本量取自清洗阶段写出的统计文件，保证清单记录的是本次真实运行的结果。
    cleaning_stats_path = project_path("reports/data-cleaning-stats.json")
    if not cleaning_stats_path.is_file():
        raise FileNotFoundError(
            f"未找到清洗统计文件：{cleaning_stats_path}，请先运行 'python scripts/clean_data.py'。"
        )
    cleaning_stats = json.loads(cleaning_stats_path.read_text(encoding="utf-8"))
    label_mapping_path, manifest_path = write_contract_artifacts(
        processed_dir=processed_dir,
        categories=config["task"]["categories"],
        split_files=split_filenames,
        seed=seed,
        raw_path=data_config["raw_path"],
        ratios=ratios,
        rows_before=int(cleaning_stats["rows_before"]),
        rows_after=int(cleaning_stats["rows_after"]),
        counts=counts,
        leakage=leakage,
    )
    print(f"[split] contract artifact written: {label_mapping_path}")
    print(f"[split] contract artifact written: {manifest_path}")

    # 渲染 Markdown 划分报告并写出报告与 JSON 统计文件
    report = render_split_report(
        result,
        leakage=leakage,
        source_path=args.input,
        output_paths=output_paths,
        text_column=text_column,
        category_column=category_column,
        sentiment_column=sentiment_column,
    )
    report_path = project_path(args.report)
    stats_path = project_path(args.stats_json)
    ensure_directories(report_path.parent, stats_path.parent)
    report_path.write_text(report, encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "ratios": result.ratios,
                "seed": result.seed,
                "group_count": result.group_count,
                "multi_category_groups": result.multi_category_groups,
                "largest_group_size": result.largest_group_size,
                "repair_moves": result.repair_moves,
                "sizes": {name: int(frame.shape[0]) for name, frame in result.splits.items()},
                "leakage": leakage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 打印泄漏检查结果；一旦发现泄漏就返回 1，提示该划分不可用于训练
    print(f"[split] leakage check: {leakage['pairwise_overlaps']}")
    print(f"[split] overlapping texts across splits: {leakage['total_overlapping_texts']}")
    print(f"[split] leakage_free={leakage['leakage_free']}")
    print(f"[split] report written: {report_path}")
    if not leakage["leakage_free"]:
        print("[split] ERROR: leakage detected, split must be rejected")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
