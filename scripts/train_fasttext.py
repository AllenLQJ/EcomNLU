"""训练 FastText 模型（字符级与词级、默认参数与自动调参）。

用法：
    python scripts/train_fasttext.py
    python scripts/train_fasttext.py --autotune-duration 60 --evaluate-test

输入：
    - ``data/processed/train.csv``、``dev.csv``（必须先通过
      ``python scripts/verify_data_contract.py``）。

输出：
    - ``models/fasttext_char``、``models/fasttext_char_tuned``、
      ``models/fasttext_word``、``models/fasttext_word_tuned``
      每个目录含 ``model.bin``、``label_mapping.json``、``metadata.json``、``metrics.json``。
    - ``reports/fasttext-summary.json``：全部变体的验证集指标与语料生成统计。

纪律：
    默认只评测验证集；``--evaluate-test`` 才会评测测试集，且每个模型只评一次
    （契约 §2.4、AGENTS.md §6）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.artifact import MODELS_DIR, PROCESSED_DIR, write_json  # noqa: E402
from src.models.fasttext.train import run  # noqa: E402

DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "fasttext-summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="契约数据目录")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="模型产物根目录")
    parser.add_argument(
        "--autotune-duration",
        type=int,
        default=120,
        help="每个自动调参变体的搜索秒数（默认 120）",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="定稿后评估测试集一次，写入 metrics.json 的 test 块",
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="汇总 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        autotune_duration=args.autotune_duration,
        evaluate_test=args.evaluate_test,
    )
    write_json(args.summary_json, summary)

    print("\n=== 验证集 Macro-F1 对比 ===")
    for name, item in summary["variants"].items():
        print(f"  {name:<24} {item['dev']['macro_f1']:.4f}")
    if args.evaluate_test:
        print("\n=== 测试集（各评一次） ===")
        for name, item in summary["variants"].items():
            if "test" in item:
                print(f"  {name:<24} {item['test']['macro_f1']:.4f}")
    print(f"\n汇总已写入：{args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
