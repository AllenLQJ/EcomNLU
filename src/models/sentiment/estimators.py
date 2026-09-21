"""情感二分类的候选模型、训练与验证集选模。

模块职责：
    定义 M5 情感任务的两个候选模型（逻辑回归、校准后的线性 SVM），在训练集上训练、
    在验证集上按 Macro-F1 选模，并记录每个超参数组合的验证集表现作为调参过程证据。

主要输入与输出：
    输入：特征矩阵、训练标签、验证集特征与标签、情感标签顺序。
    输出：已训练的模型对象、每个候选配置的验证集指标、以及被选中的最优配置。

关键设计理由与约束：

1. **标签顺序由配置注入，不在本模块硬编码。**
   情感类名与顺序取自 ``configs/base.yaml`` 的 ``task.sentiment_categories``，
   由调用方（``scripts/train_sentiment.py``）传入。这样标签顺序只有一处定义，
   校验器与 Web 层读的是同一份配置，不会出现「训练脚本按 0/1 顺序、
   某处按字母顺序」这类静默错位（契约 v1.2 草案 §2.6）。

2. **训练标签复用数据契约的 ``label`` 列，不重新制作标签。**
   M2 已经产出带 ``label`` 列的 ``train/dev/test.csv``，且与 M3 使用同一份划分
   （跨集合重复文本为 0）。情感任务只做 0/1 -> 中文名的重命名，不新增任何标注。

3. **选模只看验证集。** :func:`select_overall` 直接复用 M3 的实现，
   其入参里根本没有测试集，从签名上排除用测试集选模的可能（AGENTS.md §6）。

4. **线性 SVM 必须校准后才能满足契约。** ``LinearSVC`` 只有 ``decision_function``，
   没有 ``predict_proba``；``src/inference/sklearn_adapter.py`` 会直接拒绝这类模型。
   因此 ``sentiment_tfidf_svm`` 用 ``CalibratedClassifierCV`` 包装后再保存。

5. **不需要类别加权。** 情感标签在三个划分上的正例率都在 50.5% 左右
   （train 50.56%、dev 50.54%、test 50.53%），不存在类别不均衡，
   因此不设置 ``class_weight``，也不做重采样。**但内部类别的正例率并不均衡**
   （热水器 82.6%、书籍 54.6%），这一点在报告的各商品类别分析中单独披露。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation.metrics import compute_metrics_block
from src.models.baseline.estimators import (
    TrainedCandidate,
    select_best_per_model,
    select_overall,
)


__all__ = [
    "SENTIMENT_MODEL_NAMES",
    "SENTIMENT_MODEL_DESCRIPTIONS",
    "SENTIMENT_TUNING_GRIDS",
    "build_sentiment_estimator",
    "train_sentiment_candidate",
    "train_sentiment_candidates",
    "select_best_per_model",
    "select_overall",
    "translate_labels",
]


# 情感模型的目录名。契约 §2.1 的 model_name 表在 v1.1 中尚无情感任务条目，
# v1.2 草案 §2.1 已按 <情感任务前缀>_<同名模型> 的规则补齐命名，
# 本模块与之一致：sentiment_tfidf_logreg / sentiment_tfidf_svm。
SENTIMENT_MODEL_NAMES: tuple[str, ...] = (
    "sentiment_tfidf_logreg",
    "sentiment_tfidf_svm",
)

SENTIMENT_MODEL_DESCRIPTIONS: dict[str, str] = {
    "sentiment_tfidf_logreg": "TF-IDF + 逻辑回归（二分类，原生输出概率）",
    "sentiment_tfidf_svm": (
        "TF-IDF + 线性 SVM（经 CalibratedClassifierCV 校准，以满足契约概率要求）"
    ),
}

# 超参数网格，刻意保持很小。
# 理由与 M3 相同：可用时间有限，把时间花在评估严谨性与错误分析上的收益
# 高于扩大网格。每个组合的验证集结果都会写入报告，作为「确实调过参」的证据。
SENTIMENT_TUNING_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "sentiment_tfidf_logreg": (
        {"C": 1.0},
        {"C": 10.0},
    ),
    "sentiment_tfidf_svm": (
        {"C": 0.5},
        {"C": 1.0},
    ),
}


def translate_labels(values: Sequence[Any], labels: Sequence[str]) -> list[str]:
    """把原始 ``label`` 列的 0/1 转成中文情感类名。

    参数：
        values：``label`` 列的值序列，允许是 int 或可转成 int 的字符串。
        labels：情感标签顺序，长度为 2，取自 ``configs/base.yaml`` 的
            ``task.sentiment_categories``。``labels[0]`` 对应原始值 0，
            ``labels[1]`` 对应原始值 1。

    返回：
        与输入等长的情感类名列表。

    异常：
        ValueError：``labels`` 长度不是 2、或出现 0/1 之外的值时抛出。
            数据契约 §1.2 规定 ``label`` 只能是 0 或 1，出现别的值说明数据不合格，
            必须显式报错而不是静默丢样本——静默丢弃会让评估集的规模与报告不符，
            且不易察觉。
    """
    pair = list(labels)
    if len(pair) != 2:
        raise ValueError(f"情感标签必须恰好 2 个，实际为 {pair}")
    negative_name, positive_name = pair

    translated: list[str] = []
    for raw in values:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"label 取值无法解析为整数：{raw!r}") from exc
        if value == 0:
            translated.append(negative_name)
        elif value == 1:
            translated.append(positive_name)
        else:
            raise ValueError(
                f"label 只能是 0（{negative_name}）或 1（{positive_name}），"
                f"实际出现 {value!r}（见 docs/contracts.md §1.2）"
            )
    return translated


def build_sentiment_estimator(
    model_name: str, hyperparameters: Mapping[str, Any], *, random_seed: int
) -> Any:
    """按名称与超参数构造一个**未训练**的情感估计器。

    参数：
        model_name：``sentiment_tfidf_logreg`` / ``sentiment_tfidf_svm`` 之一。
        hyperparameters：该模型的超参数。
        random_seed：随机种子，写入所有带随机性的模型。

    返回：
        sklearn 估计器实例。

    异常：
        ValueError：模型名不在允许列表内时抛出。
    """
    params = dict(hyperparameters)
    if model_name == "sentiment_tfidf_logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_seed,
            **params,
        )
    if model_name == "sentiment_tfidf_svm":
        # LinearSVC 没有 predict_proba，必须校准；契约 §2.5 要求概率输出。
        base = LinearSVC(dual=True, max_iter=5000, random_state=random_seed, **params)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    raise ValueError(
        f"未知情感模型名 {model_name!r}，允许：{SENTIMENT_MODEL_NAMES}"
    )


def train_sentiment_candidate(
    model_name: str,
    hyperparameters: Mapping[str, Any],
    *,
    train_matrix: Any,
    train_labels: Sequence[str],
    dev_matrix: Any,
    dev_labels: Sequence[str],
    labels: Sequence[str],
    random_seed: int,
) -> TrainedCandidate:
    """训练单个情感候选配置并在验证集上评估。

    参数：
        model_name：模型名。
        hyperparameters：超参数。
        train_matrix：训练集特征矩阵。
        train_labels：训练集标签（中文情感类名）。
        dev_matrix：验证集特征矩阵。
        dev_labels：验证集标签。
        labels：情感标签顺序，决定混淆矩阵行列语义。
        random_seed：随机种子。

    返回：
        :class:`~src.models.baseline.estimators.TrainedCandidate`，
        含已训练模型与验证集指标。复用 M3 的数据容器，语义完全一致。
    """
    estimator = build_sentiment_estimator(
        model_name, hyperparameters, random_seed=random_seed
    )
    estimator.fit(train_matrix, list(train_labels))
    predictions = [str(value) for value in estimator.predict(dev_matrix)]
    metrics = compute_metrics_block(
        list(dev_labels), predictions, categories=list(labels)
    )
    notes: list[str] = []
    if model_name == "sentiment_tfidf_svm":
        notes.append(
            "为满足契约 §2.5 的概率输出要求，LinearSVC 外包装了 "
            "CalibratedClassifierCV（sigmoid，cv=3），训练成本约为单次拟合的 3 倍。"
        )
    return TrainedCandidate(
        model_name=model_name,
        hyperparameters=dict(hyperparameters),
        estimator=estimator,
        dev_metrics=metrics,
        description=SENTIMENT_MODEL_DESCRIPTIONS.get(model_name, ""),
        extra_notes=notes,
    )


def train_sentiment_candidates(
    *,
    train_matrix: Any,
    train_labels: Sequence[str],
    dev_matrix: Any,
    dev_labels: Sequence[str],
    labels: Sequence[str],
    random_seed: int,
    grids: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    model_names: Sequence[str] = SENTIMENT_MODEL_NAMES,
    on_progress: Callable[[str], None] | None = None,
) -> list[TrainedCandidate]:
    """训练所有情感候选配置并返回结果列表。

    参数：
        train_matrix：训练集特征矩阵。
        train_labels：训练集标签。
        dev_matrix：验证集特征矩阵。
        dev_labels：验证集标签。
        labels：情感标签顺序。
        random_seed：随机种子。
        grids：超参数网格，默认取 :data:`SENTIMENT_TUNING_GRIDS`。
        model_names：要训练的模型名序列。
        on_progress：进度回调，接收一行文本。

    返回：
        :class:`TrainedCandidate` 列表，顺序与 ``model_names`` 及网格顺序一致。

    异常：
        ValueError：某个模型的网格为空时抛出。
    """
    active_grids = dict(grids or SENTIMENT_TUNING_GRIDS)
    results: list[TrainedCandidate] = []
    for name in model_names:
        grid = list(active_grids.get(name, ()))
        if not grid:
            raise ValueError(f"情感模型 {name} 的超参数网格为空，无法训练")
        for hyperparameters in grid:
            if on_progress:
                on_progress(f"训练 {name} {hyperparameters} ...")
            candidate = train_sentiment_candidate(
                name,
                hyperparameters,
                train_matrix=train_matrix,
                train_labels=train_labels,
                dev_matrix=dev_matrix,
                dev_labels=dev_labels,
                labels=labels,
                random_seed=random_seed,
            )
            if on_progress:
                on_progress(
                    f"  -> {name} {hyperparameters} 验证集 Macro-F1 = "
                    f"{candidate.dev_macro_f1:.4f}"
                )
            results.append(candidate)
    return results
