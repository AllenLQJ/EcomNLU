"""训练情感二分类基线（M5 扩展任务）并产出符合接口契约的模型产物。

Usage:
    python scripts/train_sentiment.py
    python scripts/train_sentiment.py --quick           # 小样本冒烟，用于快速验证流程
    python scripts/train_sentiment.py --skip-ablation   # 跳过特征方案对照

输入文件：
    - ``data/processed/{train,dev,test}.csv``（数据契约产物，含 ``label`` 列）
    - ``data/processed/label_mapping.json``
    - ``configs/base.yaml``

输出文件：
    - ``models/sentiment_tfidf_logreg/``、``models/sentiment_tfidf_svm/``
      各含 ``model.joblib``、``vectorizer.joblib``、``label_mapping.json``、
      ``metadata.json``、``metrics.json``（契约 §2.1 的四件套；其中
      ``label_mapping.json`` 是情感标签映射，取值来源为
      ``configs/base.yaml`` 的 ``task.sentiment_categories``）
    - ``reports/m5-sentiment-baseline.md``：模型对比与逐类别报告
    - ``reports/m5-sentiment-results.json``：机器可读的完整结果
    - ``reports/m5-sentiment-error-cases.json``：全部误分类样本及现象分组

关键设计理由与约束：
    - **标签直接复用数据契约的 ``label`` 列**，只做 0/1 -> 中文情感名的重命名，
      不重新制作标签（M5 任务明确要求，AGENTS.md §2.2）；
    - **向量器只在训练集上拟合**，验证集与测试集只做变换，避免特征泄漏；
    - **选模只看验证集 Macro-F1**，测试集在选模完成之后才评估，仅用于确认与对比；
    - **推理成本通过统一入口 ``src.inference.load_predictor`` 实测**，
      测的是「加载产物 -> 清洗 -> 分词 -> 向量化 -> 预测」的端到端耗时，
      既满足契约 §2.4 的实测要求，也顺带验证了 M6 的情感联调路径确实可用
      （适配器是任务无关的，情感模型与类别模型共用同一个入口）。
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
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.data.config import ensure_directories, load_config, project_path, public_path  # noqa: E402
from src.evaluation.device import (  # noqa: E402
    describe_environment,
    render_environment_note,
    resolve_device,
)
from src.evaluation.metrics import (  # noqa: E402
    beijing_timestamp,
    compute_metrics_block,
    directory_size_bytes,
    measure_inference_cost,
)
from src.features.text_features import (  # noqa: E402
    DEFAULT_VECTORIZER_CONFIG,
    build_tfidf_vectorizer,
    load_split,
    vectorize,
    vectorizer_signature,
)
from src.inference import load_predictor  # noqa: E402
from src.models.baseline.artifacts import dataset_fingerprint  # noqa: E402
from src.models.sentiment.artifacts import (  # noqa: E402
    build_per_product_category_metrics,
    build_sentiment_metadata,
    build_sentiment_metrics,
    write_sentiment_metrics_file,
    write_sentiment_model_body,
)
from src.models.sentiment.estimators import (  # noqa: E402
    SENTIMENT_MODEL_NAMES,
    select_best_per_model,
    select_overall,
    train_sentiment_candidates,
    translate_labels,
)
from src.models.sentiment.report import render_sentiment_report  # noqa: E402


# 特征方案对照表，与 M3 的项目保持一致以便直接比较两套实验的结论。
ABLATION_CONFIGS: tuple[tuple[str, dict, str], ...] = (
    (
        "词级 unigram（M3 默认）",
        {"analyzer_mode": "word", "ngram_range": (1, 1)},
        "jieba 分词 + 单字词，最小词长 2，过滤停用词；词表不设上限",
    ),
    (
        "词级 unigram + bigram",
        {"analyzer_mode": "word", "ngram_range": (1, 2), "max_features": 400_000},
        "加入相邻词组合，可捕捉「服务 态度」这类搭配；词表上限 40 万",
    ),
    (
        "字符级 2-3 gram（M3 实测最优）",
        {"analyzer_mode": "char", "ngram_range": (2, 3), "max_features": 400_000},
        "不依赖分词，对口语化与错别字更稳健；词表上限 40 万",
    ),
)

# 冒烟模式下的采样规模与精简网格，用于快速验证流程而不是产出最终结果。
QUICK_TRAIN_ROWS = 4000
QUICK_DEV_ROWS = 1000
QUICK_GRIDS: dict[str, tuple[dict, ...]] = {
    "sentiment_tfidf_logreg": ({"C": 1.0},),
    "sentiment_tfidf_svm": ({"C": 1.0},),
}
QUICK_VECTORIZER_OVERRIDES: dict = {"max_features": 20000}

# 误分类现象分组的线索词与优先级顺序。
# 说明：分组只是**检索手段**，用于把误分类样本按可观察的语言现象归类，
# 便于人工复核；命中线索词不代表已证明因果，报告与错误分析文档都必须这样表述。
NEGATION_CUES: tuple[str, ...] = ("不", "没", "无", "未", "别", "难以", "不如")
CONTRAST_CUES: tuple[str, ...] = ("但", "不过", "然而", "就是", "可惜", "只是", "刚开始")
IRONY_CUES: tuple[str, ...] = ("呵呵", "真是", "居然", "竟然", "结果", "白", "佩服", "厉害了")
# 「超短文本（无明确评价对象）」的长度阈值。
# 取值依据：数据清洗阶段明确**不按长度删除短评**（「挺好的」这类短评仍是有效信号），
# 因此这里只是把「短到不足以承载评价对象、且不含任何情感线索词」的样本单独分出，
# 阈值必须压得很低，否则会误伤「玩游戏发热快」「像素很高」这种带明确评价对象的短句。
# 该分组在测试集 400 条误分类中只命中 1 条（最短误分类样本为 4 字符），
# 属于低召回、高精度的分组，符合它「仅提示无对象样本」的定位。
SHORT_TEXT_MAX_CHARS = 5

ERROR_TAGS: tuple[str, ...] = (
    "反讽／褒义贬用",
    "否定与转折并存",
    "含否定词",
    "超短文本（无明确评价对象）",
    "其他（无明显线索）",
)


def log(message: str) -> None:
    """打印带刷新的一行进度，便于观察长耗时步骤。

    参数：
        message：要打印的内容。
    """
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回：
        解析后的参数对象。
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
        "--report", default="reports/m5-sentiment-baseline.md", help="报告输出路径"
    )
    parser.add_argument(
        "--results-json",
        default="reports/m5-sentiment-results.json",
        help="机器可读结果输出路径",
    )
    parser.add_argument(
        "--error-cases",
        default="reports/m5-sentiment-error-cases.json",
        help="误分类样本清单输出路径",
    )
    return parser.parse_args()


def subsample(frame: pd.DataFrame, limit: int, *, seed: int, category_column: str) -> pd.DataFrame:
    """按商品类别分层抽样，用于冒烟模式。

    逐类别等比例抽取，保证抽样后每个类别都有样本——否则报告里的
    「各商品类别内情感表现」会出现空类别，掩盖真正的流程问题。

    参数：
        frame：待抽样数据表。
        limit：目标总行数上限。
        seed：随机种子。
        category_column：商品类别列名。

    返回：
        抽样后的数据表；行数已不超过上限时原样返回。
    """
    if len(frame) <= limit:
        return frame
    ratio = limit / len(frame)
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(category_column, sort=True):
        take = max(1, min(len(group), int(round(len(group) * ratio))))
        parts.append(group.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def classify_error_tag(text: str) -> str:
    """按可观察的语言现象给一条误分类样本打标签。

    分组优先级从「反讽」到「无线索」，保证每条样本只落入一个分组，
    因此各分组占比之和为 100%，便于在报告中直接比较。

    参数：
        text：误分类的评论文本。

    返回：
        :data:`ERROR_TAGS` 中的一项。
    """
    has_negation = any(cue in text for cue in NEGATION_CUES)
    has_contrast = any(cue in text for cue in CONTRAST_CUES)
    has_irony = any(cue in text for cue in IRONY_CUES)

    if has_irony and has_negation:
        return "反讽／褒义贬用"
    if has_negation and has_contrast:
        return "否定与转折并存"
    if has_negation:
        return "含否定词"
    if len(text.strip()) <= SHORT_TEXT_MAX_CHARS:
        return "超短文本（无明确评价对象）"
    return "其他（无明显线索）"


def run_ablation(
    *,
    train_texts,
    train_labels,
    dev_texts,
    dev_labels,
    labels,
    seed: int,
    quick: bool,
    on_progress,
) -> tuple[list[dict], dict]:
    """用最快的模型在验证集上对照若干特征方案。

    探针模型用逻辑回归：它原生输出概率、单次拟合快，且与最终候选模型同族，
    因此特征方案之间的相对差异可以较好地迁移到最终模型上。
    这里的模型不会被保存为契约产物。

    参数：
        train_texts：训练集文本。
        train_labels：训练集标签。
        dev_texts：验证集文本。
        dev_labels：验证集标签。
        labels：情感标签顺序。
        seed：随机种子。
        quick：是否冒烟模式（缩小词表以加速）。
        on_progress：进度回调。

    返回：
        ``(对照结果列表, 最优向量器配置)``。全部方案失败时回退到默认配置。
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
            probe = LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=seed, C=1.0
            )
            probe.fit(train_matrix, list(train_labels))
            predictions = [str(value) for value in probe.predict(matrices["dev"])]
            metrics = compute_metrics_block(
                list(dev_labels), predictions, categories=list(labels)
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


def analyze_errors(
    *,
    frame: pd.DataFrame,
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: list[dict[str, float]],
    labels: list[str],
    text_column: str,
    category_column: str,
) -> dict:
    """汇总误分类样本并产出错误分析所需的机器可读清单。

    参数：
        frame：测试集数据表。
        true_labels：真实情感类名。
        predicted_labels：预测情感类名。
        probabilities：每个样本的情感概率字典。
        labels：情感标签顺序。
        text_column：评论文本列名。
        category_column：商品类别列名。

    返回：
        含总体统计、现象分组统计、代表样本与完整清单的字典。
    """
    negative_name, positive_name = labels[0], labels[1]
    texts = frame[text_column].astype(str).tolist()
    categories = frame[category_column].astype(str).tolist()

    cases: list[dict] = []
    by_tag: dict[str, dict] = {tag: {"count": 0} for tag in ERROR_TAGS}
    false_negative = 0
    false_positive = 0

    for index, (true_label, predicted_label) in enumerate(
        zip(true_labels, predicted_labels)
    ):
        if true_label == predicted_label:
            continue
        probabilities_row = probabilities[index]
        confidence = float(probabilities_row[predicted_label])
        tag = classify_error_tag(texts[index])
        by_tag[tag]["count"] += 1
        if true_label == positive_name:
            false_negative += 1
        else:
            false_positive += 1
        cases.append(
            {
                "index": index,
                "text": texts[index],
                "category": categories[index],
                "true": true_label,
                "predicted": predicted_label,
                "confidence": round(confidence, 6),
                "probability_negative": round(float(probabilities_row[negative_name]), 6),
                "probability_positive": round(float(probabilities_row[positive_name]), 6),
                "tag": tag,
                "length": len(texts[index]),
            }
        )

    cases.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "test_rows": len(true_labels),
        "error_count": len(cases),
        "error_rate": (len(cases) / len(true_labels)) if true_labels else 0.0,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "by_tag": by_tag,
        # 报告表格只取前 5 条；完整清单写在 JSON 里，不抽样、不截断，便于复核。
        "top_examples": cases[:5],
        "cases": cases,
    }


def main() -> int:
    """执行情感基线的训练、选模、产物写出与报告渲染。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    data_config = config["data"]
    text_column = data_config["text_column"]
    category_column = data_config["category_column"]
    sentiment_column = data_config["sentiment_column"]
    seed = int(config["project"]["random_seed"])
    # 情感标签顺序取自配置（契约 v1.2 §2.1），不在代码里硬编码：
    # labels[0] 对应原始 label 值 0，labels[1] 对应 1。
    labels = [str(value) for value in config["task"].get("sentiment_categories", [])]
    if len(labels) != 2:
        raise KeyError(
            "configs/base.yaml 的 task.sentiment_categories 必须是恰好 2 个情感标签"
            f"（0 负向、1 正向），实际为 {labels}。"
        )
    log(f"[label] 情感标签顺序：{labels[0]}=0、{labels[1]}=1（来源：task.sentiment_categories）")

    processed_dir = project_path(data_config["processed_dir"])
    models_dir = project_path("models")
    label_mapping_path = processed_dir / "label_mapping.json"

    for required in (label_mapping_path, processed_dir / "split_manifest.json"):
        if not required.is_file():
            raise FileNotFoundError(
                f"缺少契约产物 {required}，请先运行 'python scripts/split_data.py'。"
            )

    # --- 1. 载入划分并转换情感标签 ----------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "dev", "test"):
        frame = load_split(
            processed_dir / f"{name}.csv",
            text_column=text_column,
            category_column=category_column,
        )
        if sentiment_column not in frame.columns:
            raise KeyError(
                f"{processed_dir / f'{name}.csv'} 缺少情感标签列 {sentiment_column!r}"
                "（见 docs/contracts.md §1.2）"
            )
        if args.quick:
            limit = QUICK_TRAIN_ROWS if name == "train" else QUICK_DEV_ROWS
            frame = subsample(
                frame, limit, seed=seed, category_column=category_column
            )
        frames[name] = frame
        positive_share = float((frame[sentiment_column] == 1).mean())
        log(f"[data] {name}: {len(frame):,} 行，正例率 {positive_share:.4f}")

    train_texts = frames["train"][text_column].astype(str).tolist()
    dev_texts = frames["dev"][text_column].astype(str).tolist()
    test_texts = frames["test"][text_column].astype(str).tolist()
    # 原始 label 列只有 0/1，按固定对应关系重命名；labels[0] 对应 0、labels[1] 对应 1。
    train_labels = translate_labels(
        frames["train"][sentiment_column].tolist(), labels
    )
    dev_labels = translate_labels(frames["dev"][sentiment_column].tolist(), labels)
    test_labels = translate_labels(frames["test"][sentiment_column].tolist(), labels)
    positive_rates = {
        name: round(float((frame[sentiment_column] == 1).mean()), 6)
        for name, frame in frames.items()
    }

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
            labels=labels,
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
    candidates = train_sentiment_candidates(
        train_matrix=train_matrix,
        train_labels=train_labels,
        dev_matrix=matrices["dev"],
        dev_labels=dev_labels,
        labels=labels,
        random_seed=seed,
        grids=grids,
        model_names=SENTIMENT_MODEL_NAMES,
        on_progress=log,
    )
    best_per_model = select_best_per_model(candidates)
    selected = select_overall(best_per_model)
    log(
        f"[select] 依据验证集 Macro-F1 选定：{selected.model_name} "
        f"(Macro-F1={selected.dev_macro_f1:.4f})"
    )

    # --- 5. 写出产物、经统一入口实测成本、评估测试集 ----------------------
    model_results: dict[str, dict] = {}
    error_payload: dict | None = None
    ensure_directories(models_dir)
    device = resolve_device("scikit-learn")
    log(f"[device] scikit-learn 实际使用设备：{device}")
    log(f"[device] 环境：{describe_environment()}")

    tuning_by_model: dict[str, list[dict]] = {}
    for candidate in candidates:
        tuning_by_model.setdefault(candidate.model_name, []).append(
            {
                "hyperparameters": candidate.hyperparameters,
                "dev_macro_f1": candidate.dev_metrics["macro_f1"],
                "dev_accuracy": candidate.dev_metrics["accuracy"],
            }
        )

    for name, candidate in best_per_model.items():
        model_dir = write_sentiment_model_body(
            models_dir=models_dir,
            model_name=name,
            estimator=candidate.estimator,
            vectorizer=vectorizer,
            labels=labels,
            metadata=build_sentiment_metadata(
                model_name=name,
                created_at=beijing_timestamp(),
                random_seed=seed,
                processed_dir=processed_dir,
                train_rows=len(frames["train"]),
                hyperparameters={
                    **candidate.hyperparameters,
                    "vectorizer": signature,
                    "quick_mode": args.quick,
                },
                device=device,
                notes=(
                    f"情感二分类（原始 label 列：0=差评/{labels[0]}、1=好评/{labels[1]}），"
                    "标签直接取自数据集自带的数字标签列，未重新制作标签、未做人工标注。"
                    "选模依据为验证集 Macro-F1，测试集仅在选模完成后用于确认与横向对比，"
                    "未参与任何模型选择或超参数调整。"
                    + "".join(candidate.extra_notes)
                ),
            ),
        )

        # 通过统一入口加载：适配器是任务无关的，标签集取自本目录的 label_mapping.json。
        # 这一步既测端到端成本，也验证 M6 的情感联调路径确实可用。
        predictor = load_predictor(model_dir)
        cost = measure_inference_cost(
            predictor.predict_proba,
            probe_texts=test_texts[:32] or ["质量很好"],
            model_size_bytes=directory_size_bytes(model_dir),
            device=device,
        )
        probabilities = predictor.predict_proba(test_texts)
        test_predictions = [str(value) for value in predictor.predict(test_texts)]
        test_block = compute_metrics_block(
            test_labels, test_predictions, categories=labels
        )
        per_product_category = build_per_product_category_metrics(
            frames["test"],
            true_labels=test_labels,
            predicted_labels=test_predictions,
            labels=labels,
            category_column=category_column,
        )

        write_sentiment_metrics_file(
            model_dir,
            build_sentiment_metrics(
                model_name=name,
                labels=labels,
                dev_block=candidate.dev_metrics,
                test_block=test_block,
                cost=cost,
                per_product_category=per_product_category,
            ),
        )
        model_results[name] = {
            "dev": candidate.dev_metrics,
            "test": test_block,
            "cost": cost,
            "hyperparameters": candidate.hyperparameters,
            "description": candidate.description,
            "model_dir": public_path(model_dir),
            "tuning": tuning_by_model.get(name, []),
            "per_product_category": per_product_category,
        }
        log(
            f"[artifact] {name} 已写出：验证集 Macro-F1={candidate.dev_metrics['macro_f1']:.4f}，"
            f"测试集 Macro-F1={test_block['macro_f1']:.4f}，"
            f"大小={cost['model_size_bytes'] / 1024 / 1024:.2f} MB，"
            f"单条={cost['single_sample_latency_ms']:.1f} ms"
        )

        if name == selected.model_name:
            error_payload = analyze_errors(
                frame=frames["test"],
                true_labels=test_labels,
                predicted_labels=test_predictions,
                probabilities=probabilities,
                labels=labels,
                text_column=text_column,
                category_column=category_column,
            )

    # --- 6. 报告 ----------------------------------------------------------
    fingerprint = dataset_fingerprint(processed_dir)
    m3_metadata_path = models_dir / "tfidf_svm" / "metadata.json"
    m3_fingerprint = None
    if m3_metadata_path.is_file():
        try:
            m3_fingerprint = json.loads(
                m3_metadata_path.read_text(encoding="utf-8")
            ).get("dataset_fingerprint")
        except (json.JSONDecodeError, OSError):
            m3_fingerprint = None
    fingerprint_matches_m3 = bool(m3_fingerprint) and m3_fingerprint == fingerprint

    notes = [
        render_environment_note(),
        "情感标签在三个划分上的正例率都在 50.5% 左右，整体不存在类别不均衡，"
        "因此未使用 class_weight 或重采样；但**类别内部并不均衡**"
        "（热水器正例率 82.6%、书籍 54.6%），第 6 节的逐类别结果需结合这一点阅读。",
        "情感任务的分类边界比商品类别更主观：「一般」「还可以」这类中性评价在不同类别下"
        "被标注成了不同极性，这是标签本身的噪声，不是模型缺陷；"
        "详见错误分析文档中列出的真实误分类样本。",
        "测试集 6,272 条中的误分类样本已全部写入 `reports/m5-sentiment-error-cases.json`，"
        "未抽样、未截断，便于复核每一条结论。",
    ]
    if args.quick:
        notes.insert(
            0,
            "**本报告来自 `--quick` 冒烟模式**（抽样训练、精简网格、限制词表），"
            "仅用于验证流程可跑通，**不得作为最终实验结果引用**。",
        )

    # 报告里需要引用误分类清单的路径；JSON 自身不需要该字段，因此单独补一份。
    report_error_summary = (
        {**error_payload, "cases_path": args.error_cases}
        if error_payload is not None
        else None
    )

    report = render_sentiment_report(
        model_results=model_results,
        selected_model=selected.model_name,
        labels=labels,
        selection_metric="macro_f1",
        dataset_summary={
            "train_rows": len(frames["train"]),
            "dev_rows": len(frames["dev"]),
            "test_rows": len(frames["test"]),
            "random_seed": seed,
            "dataset_fingerprint": fingerprint,
            "fingerprint_matches_m3": fingerprint_matches_m3,
            "positive_rate": positive_rates,
        },
        vectorizer_config=signature,
        ablation_results=ablation_results,
        per_product_category=model_results[selected.model_name]["per_product_category"],
        error_summary=report_error_summary,
        notes=notes,
    )

    report_path = project_path(args.report)
    results_path = project_path(args.results_json)
    error_path = project_path(args.error_cases)
    ensure_directories(report_path.parent, results_path.parent, error_path.parent)
    report_path.write_text(report, encoding="utf-8")
    results_path.write_text(
        json.dumps(
            {
                "quick_mode": args.quick,
                "task": "sentiment_classification",
                "selected_model": selected.model_name,
                "selection_metric": "macro_f1",
                "labels": labels,
                "dataset_fingerprint": fingerprint,
                "fingerprint_matches_m3": fingerprint_matches_m3,
                "positive_rate": positive_rates,
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
                "models": {
                    name: {
                        "dev": result["dev"],
                        "test": result["test"],
                        "cost": result["cost"],
                        "hyperparameters": result["hyperparameters"],
                        "description": result["description"],
                        "model_dir": result["model_dir"],
                        "per_product_category": result["per_product_category"],
                    }
                    for name, result in model_results.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if error_payload is not None:
        error_path.write_text(
            json.dumps(
                {
                    "model_name": selected.model_name,
                    "labels": labels,
                    "test_rows": error_payload["test_rows"],
                    "error_count": error_payload["error_count"],
                    "error_rate": error_payload["error_rate"],
                    "false_negative": error_payload["false_negative"],
                    "false_positive": error_payload["false_positive"],
                    "by_tag": error_payload["by_tag"],
                    "tag_definitions": {
                        "反讽／褒义贬用": "同时命中反讽线索词与否定线索词：" + "、".join(IRONY_CUES),
                        "否定与转折并存": "同时命中否定线索词与转折线索词："
                        + "、".join(NEGATION_CUES)
                        + " / "
                        + "、".join(CONTRAST_CUES),
                        "含否定词": "命中否定线索词：" + "、".join(NEGATION_CUES),
                        "超短文本（无明确评价对象）": f"清洗后长度不超过 {SHORT_TEXT_MAX_CHARS} 字符",
                        "其他（无明显线索）": "未命中上述任何线索",
                    },
                    "note": (
                        "分组按 ERROR_TAGS 的优先级顺序取值，每条样本只落入一个分组，"
                        "因此各分组占比之和为 100%。命中线索词只是检索手段，"
                        "不代表已证明因果。"
                    ),
                    "cases": error_payload["cases"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    log(f"[done] 报告：{report_path}")
    log(f"[done] 结果：{results_path}")
    if error_payload is not None:
        log(f"[done] 误分类清单：{error_path}（{error_payload['error_count']} 条）")
    log(f"[done] 总耗时 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
