"""加载已训练的模型并对评论做预测，或对某个划分跑出评估分数。

Usage:
    # 不带参数：默认对验证集 dev.csv 做评估，先列出逐条预测预览再给出真实分数
    python scripts/predict.py
    python scripts/predict.py --preview 0        # 只看分数，不列逐条预览

    # 单条评论：给出类别、置信度与 Top-K 概率
    python scripts/predict.py --text "手机很好用，电池耐用"

    # 指定模型与 Top-K
    python scripts/predict.py --model models/tfidf_logreg --text "这本书不错" --top-k 5

    # 批量：从 CSV 读取一列评论（CSV 自带标签列时会一并显示预测对错与准确率）
    python scripts/predict.py --file data/processed/dev.csv --limit 20
    python scripts/predict.py --file reviews.csv --text-column review

    # 对某个划分跑完整评估分数（Macro-F1、Accuracy、逐类别指标）
    python scripts/predict.py --eval test                 # 默认 dev
    python scripts/predict.py --eval test --output scores.json

    # 情感二分类模型（M5）：脚本会读模型自身的 metadata.task，自动切换标签空间
    # 与真实标签列，无需额外参数
    python scripts/predict.py --model models/sentiment_tfidf_svm --text "质量太差，用两天就坏了"
    python scripts/predict.py --model models/sentiment_tfidf_svm --eval test

输入文件：
    - 不带参数：读 ``data/processed/dev.csv``（验证集，自带真实标签）。
      之所以默认用它，是因为它既有数据又有标签，一条命令即可同时自检预测与分数；
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

    类别任务与情感任务共用本脚本，**按模型目录内 ``metadata.json`` 的 ``task`` 字段分流**：
    类别任务读 ``task.categories`` 与 ``data.category_column``（``cat`` 列本身已是中文类名）；
    情感任务读 ``task.sentiment_categories`` 与 ``data.sentiment_column``（``label`` 列是原始
    0/1，必须先经 ``translate_labels`` 翻成「负向/正向」再比对，否则准确率会恒为 0）。
    分流依据是产物自述而非模型名，因此将来新增模型不必回来改本脚本。
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

#: M3 交付了三个类别模型的薄封装脚本，M5 追加一个情感模型的薄封装脚本。
#: 本常量只作为「未指定 --model 时的兜底」保留给 predict.py 自身使用。

#: 两个任务的取值。薄封装脚本通过 ``main(default_task=...)`` 指定，
#: 不指定时由 :func:`resolve_task` 从模型目录的 ``metadata.json`` 读出。
TASK_CATEGORY = "category_classification"
TASK_SENTIMENT = "sentiment_classification"
TASK_CHOICES = (TASK_CATEGORY, TASK_SENTIMENT)

#: 打印时对标签的称呼，只影响文案，不参与任何计算。
LABEL_NOUN_BY_TASK = {TASK_CATEGORY: "类别", TASK_SENTIMENT: "情感"}


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
    parser.add_argument(
        "--task",
        choices=list(TASK_CHOICES),
        default=None,
        help=(
            "按哪个任务解释模型输出。省略时读模型目录内 metadata.json 的 task 字段；"
            "两者都没有时按类别任务处理"
        ),
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
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="评估模式下列出前 N 条逐条预测（0 表示不列出），默认 10",
    )
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


def resolve_task(config: dict, model_dir: Path, explicit: str | None) -> str:
    """决定按哪个任务解释模型的输出。

    参数：
        config：项目配置（本函数暂未使用，保留以与其它解析函数签名一致）。
        model_dir：模型目录，其 ``metadata.json`` 的 ``task`` 字段是主要依据。
        explicit：用户或薄封装脚本显式指定的任务；为真值时不看产物。

    返回：
        :data:`TASK_CATEGORY` 或 :data:`TASK_SENTIMENT`。

    说明：
        优先显式值，其次读产物自述，最后回退到类别任务——回退是为了兼容
        M3 早期生成、``metadata.json`` 里还没有 ``task`` 字段的模型产物。
        之所以用产物自述而不是模型名判断，是因为模型名一旦增删就要回来改本脚本，
        而 ``task`` 字段是契约 §2.3 已经要求落盘的。
    """
    if explicit:
        return str(explicit)

    metadata_path = model_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        task = metadata.get("task")
        if task:
            return str(task)
    return TASK_CATEGORY


def build_label_space(config: dict, task: str) -> tuple[list[str], str]:
    """按任务给出标签顺序与真实标签列名。

    参数：
        config：项目配置。
        task：任务标识。

    返回：
        ``(labels, truth_column)``。

    异常：
        ValueError：情感任务的标签不是恰好 2 个时抛出。标签顺序是契约
            v1.2 草案 §2.2 规定的 ``label_mapping.json`` 取值来源，
            长度不对说明配置被改坏了，必须显式报错。

    说明：
        两个任务的标签与列名都**只从 config 读**，不在脚本里硬编码，
        以满足契约 v1.2 草案 §2.6「标签来源唯一」的要求。
    """
    data_config = config["data"]
    if task == TASK_SENTIMENT:
        labels = [str(value) for value in config["task"]["sentiment_categories"]]
        if len(labels) != 2:
            raise ValueError(f"task.sentiment_categories 必须恰好 2 个，实际为 {labels}")
        return labels, data_config["sentiment_column"]
    return (
        [str(value) for value in config["task"]["categories"]],
        data_config["category_column"],
    )


def read_truth_labels(
    frame: pd.DataFrame, *, truth_column: str, task: str, labels: list[str]
) -> list[str]:
    """取出真实标签列并按任务归一成类名，供与预测结果直接比对。

    参数：
        frame：待评估的数据（已读入的划分 CSV 或 ``--file`` 指定的 CSV）。
        truth_column：真实标签所在列。
        task：任务标识。
        labels：标签顺序，情感任务下用于 0/1 到中文类名的翻译。

    返回：
        与 ``frame`` 行数一致的真实类名列表。

    异常：
        KeyError：数据缺少真实标签列时抛出。

    说明：
        类别任务的 ``cat`` 列本身已是中文类名，转字符串即可；情感任务的
        ``label`` 列按数据契约 §1.2 是 0/1，**必须翻译**——若直接拿 ``"0"``
        去比预测出的「负向」，准确率会恒为 0 且不易察觉，属于会静默出错的坑。
        翻译函数从 ``src.models.sentiment`` 延迟导入，避免类别任务也被牵连加载
        情感模块（那里会拉起 sklearn 的 SVC 等实现）。
    """
    if truth_column not in frame.columns:
        raise KeyError(
            f"数据缺少真实标签列 {truth_column!r}；现有列：{list(frame.columns)}"
        )

    raw = frame[truth_column].tolist()
    if task == TASK_SENTIMENT:
        from src.models.sentiment.estimators import translate_labels  # noqa: PLC0415

        return translate_labels(raw, labels)
    return [str(value) for value in raw]


def print_single_predictions(
    predictor,
    texts: list[str],
    *,
    top_k: int,
    truth: list[str] | None = None,
    label_noun: str = "类别",
) -> float:
    """执行预测并打印逐条结果。

    计时在函数内部完成：预测才是耗时部分，若在调用处取时间会测到接近 0 的值。

    参数：
        predictor：统一入口返回的预测器。
        texts：原始评论文本列表。
        top_k：展示前 K 个标签。
        truth：可选的真实标签；提供时会在每条前标出 ✓/✗ 并在末尾给出准确率。
            只有在输入数据自带标签时才可用（例如直接喂 ``data/processed/dev.csv``）。
        label_noun：标签的称呼，类别任务为「类别」、情感任务为「情感」，只影响文案。

    返回：
        整批推理耗时（毫秒）。
    """
    started = time.perf_counter()
    probabilities = predictor.predict_proba(texts)
    labels = predictor.predict(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    per_item = elapsed_ms / max(len(texts), 1)

    correct = 0
    for index, (text, label, item) in enumerate(zip(texts, labels, probabilities), start=1):
        ranked = sorted(item.items(), key=lambda pair: -pair[1])[:top_k]
        prefix = ""
        if truth is not None:
            is_correct = label == truth[index - 1]
            correct += int(is_correct)
            prefix = "✓ " if is_correct else "✗ "
        print(f"{prefix}[{index}] 评论：{text}")
        print(f"    预测{label_noun}：{label}（置信度 {item.get(label, 0.0):.4f}）")
        if truth is not None:
            print(f"    真实{label_noun}：{truth[index - 1]}")
        print(
            "    Top-"
            + str(len(ranked))
            + "："
            + "  ".join(f"{name} {value:.4f}" for name, value in ranked)
        )
        if index < len(texts):
            print()
    print(f"\n共 {len(texts)} 条，总耗时 {elapsed_ms:.1f} ms，平均 {per_item:.2f} ms/条")
    if truth is not None:
        print(f"其中预测正确 {correct}/{len(texts)}，准确率 {correct / max(len(texts), 1):.4f}")
    return elapsed_ms


def run_evaluation(
    args: argparse.Namespace,
    predictor,
    model_dir: Path,
    *,
    task: str,
    labels: list[str],
    truth_column: str,
) -> int:
    """在指定划分上计算评估分数。

    参数：
        args：命令行参数。
        predictor：统一入口返回的预测器。
        model_dir：模型目录，用于打印来源。
        task：任务标识。类别任务才会附加「词面捷径」提示——该提示依赖
            「评论里直接写出类别名」这一现象，对情感任务没有对应含义。
        labels：该任务的标签空间，用于固定逐类指标的输出顺序。
        truth_column：划分 CSV 中真实标签所在的列。

    返回：
        进程退出码。
    """
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    label_noun = LABEL_NOUN_BY_TASK.get(task, "标签")

    split_path = project_path(f"{data_config['processed_dir']}/{args.eval}.csv")
    if not split_path.is_file():
        raise FileNotFoundError(
            f"缺少划分文件 {split_path}。请先运行 'python scripts/split_data.py'。"
        )
    frame = pd.read_csv(split_path, encoding="utf-8")
    if args.limit:
        frame = frame.head(args.limit)

    texts = frame[text_column].astype(str).tolist()
    truth = read_truth_labels(
        frame, truth_column=truth_column, task=task, labels=labels
    )

    print(f"模型：{model_dir}")
    print(f"任务：{task}")
    print(f"数据：{args.eval}（{len(texts):,} 条）\n")
    started = time.perf_counter()
    predictions = predictor.predict(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # 逐条预览：放在指标之前，便于先确认「预测本身是否正常」再读分数。
    # 只预览前面若干条，避免 6000 余行输出把终端刷爆。
    if args.preview > 0:
        shown = min(args.preview, len(texts))
        print(f"逐条预览（前 {shown} 条，✓ 表示预测与真实{label_noun}一致）：")
        for index in range(shown):
            flag = "✓" if predictions[index] == truth[index] else "✗"
            text = texts[index]
            if len(text) > 34:
                text = text[:34] + "…"
            print(
                f"  {flag} 预测 {predictions[index]:<4} / 真实 {truth[index]:<4}  {text}"
            )
        print()

    block = compute_metrics_block(truth, predictions, categories=labels)
    print(f"Macro-F1      {block['macro_f1']:.4f}")
    print(f"Accuracy      {block['accuracy']:.4f}")
    print(f"Macro-Precision {block['macro_precision']:.4f}")
    print(f"Macro-Recall    {block['macro_recall']:.4f}")
    print(f"Weighted-F1   {block['weighted_f1']:.4f}")
    print(f"推理耗时      {elapsed_ms:.1f} ms（{elapsed_ms / max(len(texts), 1):.2f} ms/条）")

    print(f"\n逐{label_noun}：")
    print(f"  {label_noun:<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'支持数':>8}")
    for category in labels:
        values = block["per_class"][category]
        print(
            f"  {category:<6} {values['precision']:>10.4f} {values['recall']:>10.4f} "
            f"{values['f1']:>10.4f} {values['support']:>8,}"
        )

    # 词面捷径提示：这类类别的指标被「评论直接写出类别名」抬高，不能当作泛化能力。
    # 该现象只对商品类别任务成立（评论里会写出「手机」「酒店」），情感任务没有对应
    # 含义，因此只在类别任务下计算，情感任务写空字典以保证输出 JSON 结构稳定。
    shortcuts: dict[str, float] = {}
    if task == TASK_CATEGORY:
        shortcuts = category_name_shortcut_shares(
            frame, text_column=text_column, category_column=truth_column
        )
        if shortcuts:
            detail = "、".join(
                f"{name}（{share:.1%} 的评论直接含类别名）"
                for name, share in shortcuts.items()
            )
            print(f"\n注意：{detail}，该类别的指标被词面捷径抬高，不代表泛化能力。")

    if args.output:
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model_dir": str(model_dir),
                    "task": task,
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


def main(
    argv: list[str] | None = None,
    *,
    default_model: str | None = None,
    default_task: str | None = None,
) -> int:
    """脚本入口。

    参数：
        argv：参数列表；``None`` 表示从 ``sys.argv`` 读取。
        default_model：未通过 ``--model`` 指定时使用的模型目录。
            ``scripts/predict_logreg.py`` 等薄封装脚本通过它固定各自的模型，
            从而复用本函数而不必复制任何逻辑。
        default_task：未通过 ``--task`` 指定时按哪个任务解释输出。
            情感模型的薄封装脚本传 ``"sentiment_classification"``；留空时
            由 :func:`resolve_task` 从模型的 ``metadata.json`` 自行判断。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args(argv)
    config = load_config(args.config)
    model_dir = resolve_model_dir(config, args.model or default_model)
    task = resolve_task(config, model_dir, args.task or default_task)
    labels, truth_column = build_label_space(config, task)
    predictor = load_predictor(model_dir)

    if args.eval:
        return run_evaluation(
            args,
            predictor,
            model_dir,
            task=task,
            labels=labels,
            truth_column=truth_column,
        )

    if args.text:
        texts = [args.text]
        truth: list[str] | None = None
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{file_path}")
        frame = pd.read_csv(file_path, encoding="utf-8")
        if args.text_column not in frame.columns:
            raise KeyError(
                f"{file_path} 中没有列 {args.text_column!r}；现有列：{list(frame.columns)}"
            )
        if args.limit:
            frame = frame.head(args.limit)
        texts = frame[args.text_column].astype(str).tolist()
        # CSV 若自带真实标签列（例如直接喂 data/processed/dev.csv），就一并校验对错。
        # 列名随任务变化：类别任务看 cat，情感任务看 label（原始 0/1，需要翻译）。
        truth = (
            read_truth_labels(
                frame, truth_column=truth_column, task=task, labels=labels
            )
            if truth_column in frame.columns
            else None
        )
    else:
        # 未指定任何输入时，默认直接使用验证集 dev.csv：它自带真实标签，
        # 因此可以一次给出逐条预测预览与真实评估分数，便于快速自检。
        print(
            "未指定输入，默认使用 data/processed/dev.csv。\n"
            "如需其他输入：--text 单条评论、--file 指定 CSV、--eval dev|test 切换划分。\n"
        )
        args.eval = "dev"
        return run_evaluation(
            args,
            predictor,
            model_dir,
            task=task,
            labels=labels,
            truth_column=truth_column,
        )

    if args.top_k < 1 or args.top_k > 10:
        raise SystemExit(f"--top-k 必须在 1-10 之间，实际为 {args.top_k}")

    print(f"模型：{model_dir}")
    print(f"任务：{task}\n")
    print_single_predictions(
        predictor,
        texts,
        top_k=args.top_k,
        truth=truth,
        label_noun=LABEL_NOUN_BY_TASK.get(task, "标签"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
