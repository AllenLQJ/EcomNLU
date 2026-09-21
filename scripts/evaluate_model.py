"""对任意契约模型目录做一次真实评测，并把结果写回该模型的 ``metrics.json``。

用法：
    python scripts/evaluate_model.py --model-dir models/fasttext_word --split dev
    python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
    python scripts/evaluate_model.py --all --split dev

说明：
    * 评测走 ``src.inference`` 的统一入口，因此"评测口径 = 线上推理口径"；
    * 指标口径与字段见 ``docs/contracts.md`` §2.4；
    * ``--split test`` 只应用于最终确认，不得用于选模（AGENTS.md §6）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.evaluation.predictor_eval import evaluate_model_dir, summarize  # noqa: E402
from src.models.artifact import MODELS_DIR, PROCESSED_DIR, load_categories  # noqa: E402


def discover_model_dirs(models_dir: Path) -> list[Path]:
    """列出所有含 metadata.json 的模型目录，按名称排序。"""
    return sorted(
        item
        for item in Path(models_dir).iterdir()
        if item.is_dir() and (item / "metadata.json").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", type=Path, action="append", default=[], help="模型目录，可重复")
    parser.add_argument("--all", action="store_true", help="评测 models/ 下全部模型")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="评测划分")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="模型根目录")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="契约数据目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    categories = load_categories(load_config())

    targets = list(args.model_dir)
    if args.all:
        targets.extend(discover_model_dirs(args.models_dir))
    if not targets:
        print("未指定模型：请用 --model-dir 或 --all", file=sys.stderr)
        return 2

    print(f"评测划分：{args.split}；模型数：{len(targets)}")
    for model_dir in targets:
        block = evaluate_model_dir(model_dir, categories, args.split, args.processed_dir)
        summary = summarize(block)
        print(
            f"  {Path(model_dir).name:<28} "
            f"macro_f1={summary['macro_f1']:.4f} acc={summary['accuracy']:.4f} "
            f"weighted_f1={summary['weighted_f1']:.4f} "
            f"({block['elapsed_seconds']:.1f}s)"
        )
    print("指标已写入各模型目录的 metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
