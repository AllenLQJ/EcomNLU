"""加载已训练的模型并对评论做预测，或对某个划分跑出评估分数。

Usage:
    # 单条评论：给出类别、置信度与 Top-K 概率
    python scripts/predict.py --text "手机很好用，电池耐用"

    # 指定模型与 Top-K
    python scripts/predict.py --model models/tfidf_logreg --text "这本书不错" --top-k 5

    # 批量：从 CSV 读取一列评论
    python scripts/predict.py --file reviews.csv --text-column review

    # 对某个划分跑完整评估分数（Macro-F1、Accuracy、逐类别指标）
    python scripts/predict.py --eval test
    python scripts/predict.py --eval dev --model models/tfidf_logreg

输入文件：
    - ``--text``：直接给一条评论文本，不需要额外文件；
    - ``--file``：一个 CSV，用 ``--text-column`` 指定评论列（默认 ``review``）；
    - ``--eval``：``data/processed/{dev,test}.csv``，需要先完成 M2 的数据划分。

输出文件：
    无。结果打印到标准输出；评估模式下可选把指标 JSON 写到 ``--output``。

关键设计理由与约束：
    本脚本**不实现任何推理逻辑**，只负责参数解析与结果展示。真正的加载与预测全部走
    ``src.inference.load_predictor``，也就是 M6 的 FastAPI 服务将要调用的同一个入口。
    这样命令行看到的分数与将来 Web 页面看到的分数必然一致，
    联调时可以先确认「模型这一侧是对的」，再排查 API 那一侧。

    预测时传入的是**原始评论文本**：清洗与分词由模型目录内的 ``vectorizer.joblib``
    在内部完成（见 ``docs/contracts.md`` §2.2 与 ``src/features/text_features.py``），
    因此这里不需要也不应该预先清洗文本。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.config import load_config, project_path  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    category_name_shortcut_shares,
    compute_metrics_block,
)
from src.inference import load_predictor  # noqa: E402


DEFAULT_MODEL_NAME = "tfidf_svm"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数：
        argv：参数列表；``None`` 表示从 ``sys.argv`` 读取。
            显式传入是为了让 ``scripts/predict_<model>.py`` 这类薄封装脚本能够复用本函数。

    返回：
        含模型路径、输入方式与展示选项的解析结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    parser.add_argument(
        "--model",
        default=None,
        help=f"模型目录。省略时优先用 models/{DEFAULT_MODEL_NAME}，否则取第一个可用模型",
    )
    parser.add_argument("--text", default=None, help="单条评论文本")
    parser.add_argument("--file", default=None, help="含评论文本的 CSV 路径")
    parser.add_argument(
        "--text-column", default="review", help="--file 模式下评论所在的列名"
    )
    parser.add_argument(
        "--eval",
        choices=["dev", "test"],
        default=None,
        help="对某个划分跑评估分数（需要 data/processed/ 下的划分文件）",
    )
    parser.add_argument("--top-k", type=int, default=3, help="展示前 K 个类别，默认 3")
    parser.add_argument("--limit", type=int, default=None, help="只预测前 N 条，便于抽查")
    parser.add_argument("--output", default=None, help="--eval 模式下把指标 JSON 写到该路径")
    return parser.parse_args(argv)


def resolve_model_dir(config: dict, explicit: str | None) -> Path:
    """决定使用哪个模型目录。

    参数：
        config：项目配置。
        explicit：用户显式传入的模型目录；相对路径按**项目根目录**解析，
            这样从任意工作目录运行结果都一致。

    返回：
        模型目录路径。

    异常：
        FileNotFoundError：指定的目录不存在，或没有任何可用的模型目录时抛出。
    """
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = project_path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"模型目录不存在：{path}")
        return path

    models_dir = project_path("models")
    preferred = models_dir / DEFAULT_MODEL_NAME
    if preferred.is_dir():
        return preferred

    candidates = sorted(
        path
        for path in models_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    ) if models_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"{models_dir} 下没有已训练的模型。请先运行 'python scripts/train_baseline.py'。"
        )
    return candidates[0]


def print_single_predictions(predictor, texts: list[str], *, top_k: int) -> float:
    """执行预测并打印逐条结果。

    计时在函数内部完成：预测才是耗时部分，若在调用处取时间会测到接近 0 的值。

    参数：
        predictor：统一入口返回的预测器。
        texts：原始评论文本列表。
        top_k：展示前 K 个类别。

    返回：
        整批推理耗时（毫秒）。
    """
    started = time.perf_counter()
    probabilities = predictor.predict_proba(texts)
    labels = predictor.predict(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    per_item = elapsed_ms / max(len(texts), 1)

    for index, (text, label, item) in enumerate(zip(texts, labels, probabilities), start=1):
        ranked = sorted(item.items(), key=lambda pair: -pair[1])[:top_k]
        print(f"[{index}] 评论：{text}")
        print(f"    预测类别：{label}（置信度 {item.get(label, 0.0):.4f}）")
        print(
            "    Top-"
            + str(len(ranked))
            + "："
            + "  ".join(f"{name} {value:.4f}" for name, value in ranked)
        )
        if index < len(texts):
            print()
    print(f"\n共 {len(texts)} 条，总耗时 {elapsed_ms:.1f} ms，平均 {per_item:.2f} ms/条")
    return elapsed_ms


def run_evaluation(args: argparse.Namespace, predictor, model_dir: Path) -> int:
    """在指定划分上计算评估分数。

    参数：
        args：命令行参数。
        predictor：统一入口返回的预测器。
        model_dir：模型目录，用于打印来源。

    返回：
        进程退出码。
    """
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    categories = [str(c) for c in config["task"]["categories"]]

    split_path = project_path(f"{data_config['processed_dir']}/{args.eval}.csv")
    if not split_path.is_file():
        raise FileNotFoundError(
            f"缺少划分文件 {split_path}。请先运行 'python scripts/split_data.py'。"
        )
    frame = pd.read_csv(split_path, encoding="utf-8")
    if args.limit:
        frame = frame.head(args.limit)

    texts = frame[text_column].astype(str).tolist()
    truth = frame[category_column].astype(str).tolist()

    print(f"模型：{model_dir}")
    print(f"数据：{args.eval}（{len(texts):,} 条）\n")
    started = time.perf_counter()
    predictions = predictor.predict(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    block = compute_metrics_block(truth, predictions, categories=categories)
    print(f"Macro-F1      {block['macro_f1']:.4f}")
    print(f"Accuracy      {block['accuracy']:.4f}")
    print(f"Macro-Precision {block['macro_precision']:.4f}")
    print(f"Macro-Recall    {block['macro_recall']:.4f}")
    print(f"Weighted-F1   {block['weighted_f1']:.4f}")
    print(f"推理耗时      {elapsed_ms:.1f} ms（{elapsed_ms / max(len(texts), 1):.2f} ms/条）")

    print("\n逐类别：")
    print(f"  {'类别':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'支持数':>8}")
    for category in categories:
        values = block["per_class"][category]
        print(
            f"  {category:<6} {values['precision']:>10.4f} {values['recall']:>10.4f} "
            f"{values['f1']:>10.4f} {values['support']:>8,}"
        )

    # 词面捷径提示：这类类别的指标被「评论直接写出类别名」抬高，不能当作泛化能力
    shortcuts = category_name_shortcut_shares(
        frame, text_column=text_column, category_column=category_column
    )
    if shortcuts:
        detail = "、".join(
            f"{name}（{share:.1%} 的评论直接含类别名）" for name, share in shortcuts.items()
        )
        print(f"\n注意：{detail}，该类别的指标被词面捷径抬高，不代表泛化能力。")

    if args.output:
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model_dir": str(model_dir),
                    "split": args.eval,
                    "rows": len(texts),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "metrics": block,
                    "category_name_shortcuts": shortcuts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n指标已写入：{output_path}")
    return 0


def main(argv: list[str] | None = None, *, default_model: str | None = None) -> int:
    """脚本入口。

    参数：
        argv：参数列表；``None`` 表示从 ``sys.argv`` 读取。
        default_model：未通过 ``--model`` 指定时使用的模型目录。
            ``scripts/predict_logreg.py`` 等薄封装脚本通过它固定各自的模型，
            从而复用本函数而不必复制任何逻辑。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args(argv)
    config = load_config(args.config)
    model_dir = resolve_model_dir(config, args.model or default_model)
    predictor = load_predictor(model_dir)

    if args.eval:
        return run_evaluation(args, predictor, model_dir)

    if args.text:
        texts = [args.text]
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{file_path}")
        frame = pd.read_csv(file_path, encoding="utf-8")
        if args.text_column not in frame.columns:
            raise KeyError(
                f"{file_path} 中没有列 {args.text_column!r}；现有列：{list(frame.columns)}"
            )
        texts = frame[args.text_column].astype(str).tolist()
        if args.limit:
            texts = texts[: args.limit]
    else:
        raise SystemExit("请用 --text 给一条评论、用 --file 给 CSV，或用 --eval 跑评估分数。")

    if args.top_k < 1 or args.top_k > 10:
        raise SystemExit(f"--top-k 必须在 1-10 之间，实际为 {args.top_k}")

    print(f"模型：{model_dir}\n")
    print_single_predictions(predictor, texts, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
