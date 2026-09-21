"""训练 BERT 中文分类模型。

用法（GPU 环境示例）：
    python scripts/train_bert.py --batch-size 32 --epochs 3
    python scripts/train_bert.py --class-weights --model-name bert_base_chinese_cw

输入：
    - ``models/pretrained/bert-base-chinese/``：先运行
      ``python scripts/fetch_pretrained.py bert-base-chinese``；
    - ``data/processed/{train,dev}.csv``：先通过 ``python scripts/verify_data_contract.py``。

输出：
    - ``models/<model-name>/``：``model/``、``tokenizer/``、``label_mapping.json``、
      ``metadata.json``（``metrics.json`` 由 ``scripts/evaluate_model.py`` 在真实评测后写入）；
    - ``reports/bert-training-summary.json``：超参、环境、逐 epoch 指标与耗时。

注意：
    ``max_length`` 的默认值 192 来自 ``src/models/bert/token_stats.py`` 对本数据集
    train 划分的实测分布（覆盖率首次达到 95%），不是照搬其它项目。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.artifact import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from src.models.bert.train import BertTrainConfig, run  # noqa: E402

DEFAULT_PRETRAINED = MODELS_DIR / "pretrained" / "bert-base-chinese"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "bert-training-summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-dir", type=Path, default=DEFAULT_PRETRAINED, help="本地预训练模型目录")
    parser.add_argument("--model-name", default="bert_base_chinese", help="产物目录名")
    parser.add_argument("--max-length", type=int, default=192, help="截断长度（默认 192，由 token 分布实测确定）")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="预热步数占比")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖 dropout 概率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--no-bf16", action="store_true", help="关闭 cuda 上的 bf16 混合精度")
    parser.add_argument("--class-weights", action="store_true", help="按类别频次倒数给损失加权")
    parser.add_argument("--log-every-steps", type=int, default=100, help="日志间隔步数")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="契约数据目录")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="模型产物根目录")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="训练摘要输出路径")
    parser.add_argument("--notes", default="", help="写入 metadata.notes 的说明")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BertTrainConfig(
        pretrained_dir=args.pretrained_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        use_bf16=not args.no_bf16,
        class_weights=args.class_weights,
        log_every_steps=args.log_every_steps,
        notes=args.notes
        or f"BERT 中文微调；max_length={args.max_length} 由本数据集 token 分布实测确定",
    )
    summary = run(
        config,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        history_path=args.summary_json,
    )
    print(
        f"\n最佳验证集 Macro-F1：{summary['best_dev_macro_f1']:.4f}"
        f"（第 {summary['best_epoch']} 轮），总耗时 {summary['total_seconds']:.0f}s"
    )
    print(f"模型目录：{summary['model_dir']}")
    print(f"训练摘要：{args.summary_json}")
    print(
        "下一步：用统一入口评测并写入 metrics.json\n"
        f"  python scripts/evaluate_model.py --model-dir {summary['model_dir']} --split dev\n"
        f"  python scripts/evaluate_model.py --model-dir {summary['model_dir']} --split test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
