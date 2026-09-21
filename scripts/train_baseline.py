"""训练 M3 的 TF-IDF 传统模型基线，并产出符合接口契约的模型产物。

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --quick           # 小样本冒烟，用于快速验证流程
    python scripts/train_baseline.py --skip-ablation   # 跳过特征方案对照

输入文件：
    - ``data/processed/{train,dev,test}.csv``（数据契约产物，需先运行 split_data.py）
    - ``data/processed/label_mapping.json``
    - ``data/processed/split_manifest.json``（记入模型元数据，作为数据版本指纹）
    - ``configs/base.yaml``

输出文件：
    - ``models/tfidf_logreg/``、``models/tfidf_svm/``、``models/tfidf_rf/``
      各含 ``model.joblib``、``vectorizer.joblib``、``label_mapping.json``、
      ``metadata.json``、``metrics.json``（契约 §2.1）
    - ``reports/m3-baseline-comparison.md``：模型对比报告
    - ``reports/m3-baseline-results.json``：机器可读的完整结果

关键设计理由与约束：
    - **向量器只在训练集上拟合**，验证集与测试集只做变换，避免特征泄漏；
    - **选模只看验证集 Macro-F1**，测试集在选模完成之后才评估，仅用于确认与对比；
    - **推理成本通过统一入口 ``src.inference.load_predictor`` 实测**，
      测的是「加载产物 -> 清洗 -> 分词 -> 向量化 -> 预测」的端到端耗时，
      既满足契约 §2.4 的实测要求，也顺带验证了 M6 的加载路径确实可用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.data.config import ensure_directories, load_config, project_path  # noqa: E402
from src.evaluation.device import (  # noqa: E402
    describe_environment,
    render_environment_note,
    resolve_device,
)
from src.evaluation.metrics import (  # noqa: E402
    beijing_timestamp,
    category_name_shortcut_shares,
    compute_metrics_block,
    directory_size_bytes,
    measure_inference_cost,
    render_comparison_report,
    top_confusions,
)
from src.features.text_features import (  # noqa: E402
    DEFAULT_VECTORIZER_CONFIG,
    build_tfidf_vectorizer,
    load_split,
    vectorize,
    vectorizer_signature,
)
from src.models.baseline.artifacts import (  # noqa: E402
    build_metadata,
    build_metrics,
    sha256_of,
    write_metrics_file,
    write_model_body,
)
from src.models.baseline.estimators import (  # noqa: E402
    MODEL_NAMES,
    select_best_per_model,
    select_overall,
    train_all_candidates,
)

# 特征方案对照表。目的是用实测数据决定分词方案，而不是凭直觉选择。
# 每项为 (展示名, 向量器配置, 说明)。
# 说明：bigram 与字符级方案必须设 max_features 上限。中文的 2-3gram 在 min_df=2 下
# 词表可达百万级，不仅训练与推理成本失控，产物体积也无法接受。上限值写入报告，
# 使读者知道这是受约束的对照而不是无上限的理想配置。
ABLATION_CONFIGS: tuple[tuple[str, dict, str], ...] = (
    (
        "词级 unigram（默认）",
        {"analyzer_mode": "word", "ngram_range": (1, 1)},
        "jieba 分词 + 单字词，最小词长 2，过滤停用词；词表不设上限",
    ),
    (
        "词级 unigram + bigram",
        {"analyzer_mode": "word", "ngram_range": (1, 2), "max_features": 400_000},
        "加入相邻词组合，可捕捉「服务 态度」这类搭配；词表上限 40 万",
    ),
    (
        "字符级 2-3 gram",
        {"analyzer_mode": "char", "ngram_range": (2, 3), "max_features": 400_000},
        "不依赖分词，对未登录词与口语化表达更稳健；词表上限 40 万",
    ),
)

# 冒烟模式下的采样规模与精简网格，用于快速验证流程而不是产出最终结果。
QUICK_TRAIN_ROWS = 3000
QUICK_DEV_ROWS = 800
QUICK_GRIDS: dict[str, tuple[dict, ...]] = {
    "tfidf_logreg": ({"C": 1.0},),
    "tfidf_svm": ({"C": 1.0},),
    "tfidf_rf": ({"n_estimators": 20, "max_features": "sqrt"},),
}
QUICK_VECTORIZER_OVERRIDES: dict = {"max_features": 20000}


def log(message: str) -> None:
    """打印带刷新的一行进度，便于观察长耗时步骤。

    参数：
        message：要打印的内容。
    """
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回：
        含 ``config``、``quick``、``skip_ablation``、``models`` 的解析结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML 配置路径")
    parser.add_argument(
        "--quick", action="store_true", help="小样本冒烟模式：抽样训练、精简网格"
    )
    parser.add_argument(
        "--skip-ablation", action="store_true", help="跳过特征方案对照，直接用默认配置"
    )
    parser.add_argument(
        "--report",
        default="reports/m3-baseline-comparison.md",
        help="对比报告输出路径",
    )
    parser.add_argument(
        "--results-json",
        default="reports/m3-baseline-results.json",
        help="机器可读结果输出路径",
    )
    return parser.parse_args()


def subsample(frame: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    """按类别分层抽样，用于冒烟模式。

    逐类别等比例抽取，保证抽样后每个类别都有样本——否则冒烟运行时可能出现
    缺失类别，让 ``compute_metrics_block`` 的类别校验失败，掩盖真正的流程问题。

    参数：
        frame：待抽样数据表，需含 ``cat`` 列。
        limit：目标总行数上限。
        seed：随机种子。

    返回：
        抽样后的数据表；行数已不超过上限时原样返回。
    """
    if len(frame) <= limit:
        return frame
    ratio = limit / len(frame)
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("cat", sort=True):
        take = max(1, min(len(group), int(round(len(group) * ratio))))
        parts.append(group.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_ablation(
    *,
    train_texts,
    train_labels,
    dev_texts,
    dev_labels,
    categories,
    seed: int,
    quick: bool,
    on_progress,
) -> tuple[list[dict], dict]:
    """用最快的模型在验证集上对照若干特征方案。

    以 LinearSVC 作为探针模型：它没有概率输出、也不做交叉验证，单次拟合最快，
    适合只用来比较特征方案。这里的模型不会被保存为契约产物。

    参数：
        train_texts：训练集文本。
        train_labels：训练集标签。
        dev_texts：验证集文本。
        dev_labels：验证集标签。
        categories：类别顺序。
        seed：随机种子。
        quick：是否冒烟模式（缩小词表以加速）。
        on_progress：进度回调。

    返回：
        ``(对照结果列表, 最优向量器配置)``。若全部方案都失败，返回默认配置。
    """
    results: list[dict] = []
    best_config: dict | None = None
    best_score = -1.0

    for name, overrides, note in ABLATION_CONFIGS:
        config = {**DEFAULT_VECTORIZER_CONFIG, **overrides}
        if quick:
            config.update(QUICK_VECTORIZER_OVERRIDES)
        on_progress(f"[ablation] 拟合特征方案：{name} {overrides}")
        try:
            vectorizer = build_tfidf_vectorizer(**config)
            vectorizer, train_matrix, matrices = vectorize(
                vectorizer, train_texts, {"dev": dev_texts}
            )
            probe = LinearSVC(dual=True, max_iter=2000, random_state=seed)
            probe.fit(train_matrix, list(train_labels))
            predictions = [str(v) for v in probe.predict(matrices["dev"])]
            metrics = compute_metrics_block(
                list(dev_labels), predictions, categories=categories
            )
            score = float(metrics["macro_f1"])
            results.append(
                {
                    "name": name,
                    "macro_f1": score,
                    "accuracy": float(metrics["accuracy"]),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "note": note,
                }
            )
            on_progress(
                f"[ablation]   -> Macro-F1={score:.4f} 词表={len(vectorizer.vocabulary_):,}"
            )
            if score > best_score:
                best_score = score
                best_config = config
        except Exception as exc:  # 单一方案失败不应中断整个训练
            on_progress(f"[ablation]   -> 方案失败：{type(exc).__name__}: {exc}")
            results.append(
                {"name": name, "macro_f1": 0.0, "accuracy": 0.0, "note": f"失败：{exc}"}
            )

    if best_config is None:
        on_progress("[ablation] 所有方案均失败，回退到默认特征配置")
        best_config = dict(DEFAULT_VECTORIZER_CONFIG)
    return results, best_config


def main() -> int:
    """执行训练、选模、产物写出与报告渲染。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    categories = [str(c) for c in config["task"]["categories"]]
    seed = int(config["project"]["random_seed"])

    processed_dir = project_path(data_config["processed_dir"])
    models_dir = project_path("models")
    label_mapping_path = processed_dir / "label_mapping.json"
    manifest_path = processed_dir / "split_manifest.json"

    for required in (label_mapping_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"缺少契约产物 {required}，请先运行 'python scripts/split_data.py'。"
            )

    # --- 1. 载入划分 ------------------------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "dev", "test"):
        frame = load_split(
            processed_dir / f"{name}.csv",
            text_column=text_column,
            category_column=category_column,
        )
        if args.quick:
            limit = QUICK_TRAIN_ROWS if name == "train" else QUICK_DEV_ROWS
            frame = subsample(frame, limit, seed=seed)
        frames[name] = frame
        log(f"[data] {name}: {len(frame):,} 行")

    train_texts = frames["train"][text_column].tolist()
    dev_texts = frames["dev"][text_column].tolist()
    test_texts = frames["test"][text_column].tolist()
    train_labels = frames["train"][category_column].astype(str).tolist()
    dev_labels = frames["dev"][category_column].astype(str).tolist()
    test_labels = frames["test"][category_column].astype(str).tolist()

    # --- 2. 特征方案对照 --------------------------------------------------
    ablation_results: list[dict] = []
    if args.skip_ablation:
        vectorizer_config = dict(DEFAULT_VECTORIZER_CONFIG)
        if args.quick:
            vectorizer_config.update(QUICK_VECTORIZER_OVERRIDES)
        log("[ablation] 已跳过，采用默认特征配置")
    else:
        ablation_results, vectorizer_config = run_ablation(
            train_texts=train_texts,
            train_labels=train_labels,
            dev_texts=dev_texts,
            dev_labels=dev_labels,
            categories=categories,
            seed=seed,
            quick=args.quick,
            on_progress=log,
        )

    # --- 3. 拟合向量器（只在训练集上）-------------------------------------
    log(f"[features] 配置：{vectorizer_config}")
    vectorizer = build_tfidf_vectorizer(**vectorizer_config)
    fit_started = time.perf_counter()
    vectorizer, train_matrix, matrices = vectorize(
        vectorizer, train_texts, {"dev": dev_texts, "test": test_texts}
    )
    signature = vectorizer_signature(vectorizer)
    log(
        f"[features] 词表 {signature['vocabulary_size']:,} 维，"
        f"训练矩阵 {train_matrix.shape}，耗时 {time.perf_counter() - fit_started:.1f}s"
    )

    # --- 4. 训练候选并选模（只看验证集）-----------------------------------
    grids = QUICK_GRIDS if args.quick else None
    candidates = train_all_candidates(
        train_matrix=train_matrix,
        train_labels=train_labels,
        dev_matrix=matrices["dev"],
        dev_labels=dev_labels,
        categories=categories,
        random_seed=seed,
        grids=grids,
        model_names=MODEL_NAMES,
        on_progress=log,
    )
    best_per_model = select_best_per_model(candidates)
    selected = select_overall(best_per_model)
    log(
        f"[select] 依据验证集 Macro-F1 选定：{selected.model_name} "
        f"(Macro-F1={selected.dev_macro_f1:.4f})"
    )

    # --- 5. 写出产物、经统一入口实测成本、评估测试集 ----------------------
    from src.inference import load_predictor  # 延迟导入，避免训练路径强依赖推理层

    model_results: dict[str, dict] = {}
    ensure_directories(models_dir)
    # 设备按模型族判定：scikit-learn 没有 GPU 实现，即使本机有 CUDA 也必须记 cpu
    device = resolve_device("scikit-learn")
    log(f"[device] scikit-learn 实际使用设备：{device}")
    log(f"[device] 环境：{describe_environment()}")
    for name, candidate in best_per_model.items():
        model_dir = write_model_body(
            models_dir=models_dir,
            model_name=name,
            estimator=candidate.estimator,
            vectorizer=vectorizer,
            label_mapping_source=label_mapping_path,
            metadata=build_metadata(
                model_name=name,
                created_at=beijing_timestamp(),
                random_seed=seed,
                split_manifest_path=manifest_path,
                train_rows=len(frames["train"]),
                hyperparameters={
                    **candidate.hyperparameters,
                    "vectorizer": signature,
                    "quick_mode": args.quick,
                },
                device=device,
                notes=(
                    "选模依据为验证集 Macro-F1，测试集仅在选模完成后用于确认与横向对比，"
                    "未参与任何模型选择或超参数调整。"
                    + ("".join(candidate.extra_notes))
                ),
            ),
        )

        # 通过统一入口加载，既测端到端成本，也验证 M6 的加载路径确实可用
        predictor = load_predictor(model_dir)
        cost = measure_inference_cost(
            predictor.predict_proba,
            probe_texts=test_texts[:32] or ["手机很好用"],
            model_size_bytes=directory_size_bytes(model_dir),
        )
        test_predictions = predictor.predict(test_texts)
        test_block = compute_metrics_block(
            test_labels, test_predictions, categories=categories
        )

        write_metrics_file(
            model_dir,
            build_metrics(
                model_name=name,
                categories=categories,
                dev_block=candidate.dev_metrics,
                test_block=test_block,
                cost=cost,
            ),
        )
        model_results[name] = {
            "dev": candidate.dev_metrics,
            "test": test_block,
            "cost": cost,
            "hyperparameters": candidate.hyperparameters,
            "description": candidate.description,
            "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
        }
        log(
            f"[artifact] {name} 已写出：验证集 Macro-F1={candidate.dev_metrics['macro_f1']:.4f}，"
            f"测试集 Macro-F1={test_block['macro_f1']:.4f}，"
            f"大小={cost['model_size_bytes'] / 1024 / 1024:.2f} MB，"
            f"单条={cost['single_sample_latency_ms']:.1f} ms"
        )

    # --- 6. 报告 ----------------------------------------------------------
    notes = [
        render_environment_note(),
        "随机森林在本数据集的稀疏高维 TF-IDF 特征上训练与推理成本最高，"
        "因此**不能假定它优于线性模型**；实际排名以本报告第 3 节的验证集 Macro-F1 为准。",
        "类别不均衡（最大类 10,000 条、最小类 574 条），排序依据固定为 Macro-F1，"
        "Accuracy 仅为辅助指标。",
    ]
    if args.quick:
        notes.insert(
            0,
            "**本报告来自 `--quick` 冒烟模式**（抽样训练、精简网格、限制词表），"
            "仅用于验证流程可跑通，**不得作为最终实验结果引用**。",
        )

    worst_confusions = top_confusions(
        model_results[selected.model_name]["test"]["confusion_matrix"],
        categories,
        limit=5,
    )
    if worst_confusions:
        # 只陈述实测到的混淆对，不臆测语义原因；语义解释需要证据，否则就会写出
        # 与数据不符的结论（本项目曾出现过「举例平板与手机混淆」而实测并无此对的错误）。
        notes.append(
            "选定模型在测试集上最容易被混淆的类别对（按次数降序）："
            + "；".join(
                f"{true}→{pred}（{count} 次）" for true, pred, count in worst_confusions
            )
            + "。这些是实测结果；报告中不对其语义原因下断言，"
            "若需要解释应另做错误分析（例如人工检查误分类样本是否缺少类别信号）。"
        )

    # 词面捷径披露：某些类别的评论几乎都直接写出类别名，该类指标会被抬高
    shortcut_shares = category_name_shortcut_shares(
        frames["test"],
        text_column=text_column,
        category_column=category_column,
    )
    if shortcut_shares:
        detail = "、".join(
            f"「{category}」（{share:.1%} 的测试评论直接包含类别名）"
            for category, share in sorted(shortcut_shares.items(), key=lambda kv: -kv[1])
        )
        notes.append(
            f"**指标被词面捷径抬高的类别**：{detail}。"
            "这类类别（典型是品牌类）的评论几乎必然出现类别词，分类器只要匹配该词即可全对，"
            "因此其 Precision / Recall **不能代表模型对同类商品文本的泛化能力**；"
            "阅读第 4 节的逐类别指标时必须结合这一点，"
            "并把整体 Macro-F1 理解为「包含可被词面捷径解决的类别」的综合值。"
        )

    macro_f1_spread = max(
        entry["test"]["macro_f1"] for entry in model_results.values()
    ) - min(entry["test"]["macro_f1"] for entry in model_results.values())
    notes.append(
        f"三个模型的测试集 Macro-F1 极差为 {macro_f1_spread:.4f}，"
        "说明在本特征方案下模型族之间的差距有限，"
        "进一步提升更可能来自特征（例如 BERT 的上下文表示）而非换用更复杂的传统模型。"
    )

    report = render_comparison_report(
        model_results=model_results,
        selected_model=selected.model_name,
        categories=categories,
        selection_metric="macro_f1",
        dataset_summary={
            "train_rows": len(frames["train"]),
            "dev_rows": len(frames["dev"]),
            "test_rows": len(frames["test"]),
            "random_seed": seed,
            # 测试集指纹直接取自磁盘上的清单文件，作为「所有模型共用同一份数据」的证据
            "split_manifest_sha256": sha256_of(manifest_path),
        },
        vectorizer_config=signature,
        ablation_results=ablation_results,
        notes=notes,
    )

    report_path = project_path(args.report)
    results_path = project_path(args.results_json)
    ensure_directories(report_path.parent, results_path.parent)
    report_path.write_text(report, encoding="utf-8")
    results_path.write_text(
        json.dumps(
            {
                "quick_mode": args.quick,
                "selected_model": selected.model_name,
                "selection_metric": "macro_f1",
                "vectorizer": signature,
                "vectorizer_config": vectorizer_config,
                "ablation": ablation_results,
                "tuning": [
                    {
                        "model_name": c.model_name,
                        "hyperparameters": c.hyperparameters,
                        "dev_macro_f1": c.dev_metrics["macro_f1"],
                        "dev_accuracy": c.dev_metrics["accuracy"],
                    }
                    for c in candidates
                ],
                "models": model_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"[done] 报告：{report_path}")
    log(f"[done] 结果：{results_path}")
    log(f"[done] 总耗时 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
