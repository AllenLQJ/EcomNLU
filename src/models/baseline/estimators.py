"""TF-IDF 传统机器学习基线的候选模型、训练与选模。

模块职责：
    定义 M3 的三个候选模型（逻辑回归、线性 SVM、随机森林），在训练集上训练、
    在验证集上按 Macro-F1 选模，并记录每个超参数组合的验证集表现作为调参过程证据。

主要输入与输出：
    输入：特征矩阵、训练标签、验证集特征与标签。
    输出：已训练的模型对象、每个候选配置的验证集指标、以及被选中的最优配置。

关键设计理由与约束：

1. **训练标签直接使用中文类别名，不使用整数 id。**
   这样 ``estimator.classes_`` 就是中文类别名，``src/inference/sklearn_adapter.py``
   的 ``_to_category`` 可以直接解析，无需额外的映射表；同时混淆矩阵与分类报告的
   类别标识与契约的 ``categories`` 天然一致。

2. **选模只看验证集。** :func:`select_best` 的入参里根本没有测试集，
   从签名上排除用测试集选模的可能（AGENTS.md §6）。

3. **不只比较准确率。** 由于类别不均衡（最大类 10,000 条、最小类 574 条），
   排序依据固定为 Macro-F1；Accuracy 只作为辅助列一并报告。

4. **线性 SVM 必须校准后才能满足契约。** 契约 §2.5 要求模型输出概率，
   而 ``LinearSVC`` 只有 ``decision_function``，没有 ``predict_proba``；
   ``src/inference/sklearn_adapter.py`` 会直接拒绝这类模型。因此
   ``tfidf_svm`` 用 ``CalibratedClassifierCV`` 包装后再保存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.sparse import spmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation.metrics import compute_metrics_block


# 契约 §2.1 允许的 model_name 中属于本模块的三个。
MODEL_NAMES: tuple[str, ...] = ("tfidf_logreg", "tfidf_svm", "tfidf_rf")

MODEL_DESCRIPTIONS: dict[str, str] = {
    "tfidf_logreg": "TF-IDF + 逻辑回归（多分类，原生输出概率）",
    "tfidf_svm": "TF-IDF + 线性 SVM（经 CalibratedClassifierCV 校准确，以满足契约概率要求）",
    "tfidf_rf": "TF-IDF + 随机森林（与「投满分」项目对应的对照模型）",
}

# 超参数网格。
# 说明：网格刻意保持很小。M3 的可用时间有限，且随机森林在 5 万条 × 10 万维稀疏特征上
# 单次训练成本很高，把时间花在扩大网格上的收益远低于把它花在评估严谨性与错误分析上。
# 每个组合的验证集结果都会写入报告，作为「确实调过参」的证据。
TUNING_GRIDS: dict[str, tuple[dict[str, Any], ...]] = {
    "tfidf_logreg": (
        {"C": 1.0},
        {"C": 10.0},
    ),
    "tfidf_svm": (
        {"C": 0.5},
        {"C": 1.0},
    ),
    "tfidf_rf": (
        {"n_estimators": 200, "max_features": "sqrt"},
    ),
}


@dataclass
class TrainedCandidate:
    """一个已训练的超参数组合及其验证集表现。"""

    model_name: str
    hyperparameters: dict[str, Any]
    estimator: Any
    dev_metrics: dict[str, Any]
    description: str = ""
    extra_notes: list[str] = field(default_factory=list)

    @property
    def dev_macro_f1(self) -> float:
        """验证集 Macro-F1，选模依据。"""
        return float(self.dev_metrics["macro_f1"])


def build_estimator(
    model_name: str, hyperparameters: Mapping[str, Any], *, random_seed: int
) -> Any:
    """按名称与超参数构造一个**未训练**的估计器。

    参数：
        model_name：``tfidf_logreg`` / ``tfidf_svm`` / ``tfidf_rf`` 之一。
        hyperparameters：该模型的超参数。
        random_seed：随机种子，写入所有带随机性的模型。

    返回：
        sklearn 估计器实例。

    异常：
        ValueError：模型名不在允许列表内时抛出。
    """
    params = dict(hyperparameters)
    if model_name == "tfidf_logreg":
        return LogisticRegression(
            # lbfgs 在多分类下是原生多分类求解，比逐对（OvR）更适合 10 类任务。
            # 该求解器不使用 n_jobs（它是单线程的稀疏求解器），因此不传该参数以免误导。
            solver="lbfgs",
            max_iter=1000,
            random_state=random_seed,
            **params,
        )
    if model_name == "tfidf_svm":
        # LinearSVC 没有 predict_proba，必须校准；契约 §2.5 要求概率输出。
        base = LinearSVC(
            dual=True, max_iter=5000, random_state=random_seed, **params
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    if model_name == "tfidf_rf":
        return RandomForestClassifier(
            n_jobs=-1,
            random_state=random_seed,
            **params,
        )
    raise ValueError(f"未知模型名 {model_name!r}，允许：{MODEL_NAMES}")


def train_candidate(
    model_name: str,
    hyperparameters: Mapping[str, Any],
    *,
    train_matrix: spmatrix,
    train_labels: Sequence[str],
    dev_matrix: spmatrix,
    dev_labels: Sequence[str],
    categories: Sequence[str],
    random_seed: int,
) -> TrainedCandidate:
    """训练单个候选配置并在验证集上评估。

    参数：
        model_name：模型名。
        hyperparameters：超参数。
        train_matrix：训练集特征矩阵。
        train_labels：训练集标签（中文类别名）。
        dev_matrix：验证集特征矩阵。
        dev_labels：验证集标签。
        categories：类别顺序，决定混淆矩阵行列语义。
        random_seed：随机种子。

    返回：
        :class:`TrainedCandidate`，含已训练模型与验证集指标。
    """
    estimator = build_estimator(model_name, hyperparameters, random_seed=random_seed)
    estimator.fit(train_matrix, list(train_labels))
    predictions = estimator.predict(dev_matrix)
    metrics = compute_metrics_block(
        list(dev_labels), [str(value) for value in predictions], categories=categories
    )
    notes: list[str] = []
    if model_name == "tfidf_svm":
        notes.append(
            "为满足契约 §2.5 的概率输出要求，LinearSVC 外包装了 "
            "CalibratedClassifierCV（sigmoid，cv=3），训练成本约为单次拟合的 3 倍。"
        )
    return TrainedCandidate(
        model_name=model_name,
        hyperparameters=dict(hyperparameters),
        estimator=estimator,
        dev_metrics=metrics,
        description=MODEL_DESCRIPTIONS.get(model_name, ""),
        extra_notes=notes,
    )


def train_all_candidates(
    *,
    train_matrix: spmatrix,
    train_labels: Sequence[str],
    dev_matrix: spmatrix,
    dev_labels: Sequence[str],
    categories: Sequence[str],
    random_seed: int,
    grids: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    model_names: Sequence[str] = MODEL_NAMES,
    on_progress: Callable[[str], None] | None = None,
) -> list[TrainedCandidate]:
    """训练所有候选配置并返回结果列表。

    参数：
        train_matrix：训练集特征矩阵。
        train_labels：训练集标签。
        dev_matrix：验证集特征矩阵。
        dev_labels：验证集标签。
        categories：类别顺序。
        random_seed：随机种子。
        grids：超参数网格，默认取 :data:`TUNING_GRIDS`。
        model_names：要训练的模型名序列。
        on_progress：进度回调，接收一行文本。

    返回：
        :class:`TrainedCandidate` 列表，顺序与 ``model_names`` 及网格顺序一致。

    异常：
        ValueError：某个模型的网格为空时抛出。
    """
    active_grids = dict(grids or TUNING_GRIDS)
    results: list[TrainedCandidate] = []
    for name in model_names:
        grid = list(active_grids.get(name, ()))
        if not grid:
            raise ValueError(f"模型 {name} 的超参数网格为空，无法训练")
        for hyperparameters in grid:
            if on_progress:
                on_progress(f"训练 {name} {hyperparameters} ...")
            candidate = train_candidate(
                name,
                hyperparameters,
                train_matrix=train_matrix,
                train_labels=train_labels,
                dev_matrix=dev_matrix,
                dev_labels=dev_labels,
                categories=categories,
                random_seed=random_seed,
            )
            if on_progress:
                on_progress(
                    f"  -> {name} {hyperparameters} 验证集 Macro-F1 = "
                    f"{candidate.dev_macro_f1:.4f}"
                )
            results.append(candidate)
    return results


def select_best_per_model(
    candidates: Sequence[TrainedCandidate],
) -> dict[str, TrainedCandidate]:
    """为每个模型名挑出验证集 Macro-F1 最高的配置。

    参数：
        candidates：全部候选结果。

    返回：
        ``{模型名: 最优候选}``。同一模型名下的多个超参数组合在此收敛为一个。
    """
    best: dict[str, TrainedCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.model_name)
        if current is None or candidate.dev_macro_f1 > current.dev_macro_f1:
            best[candidate.model_name] = candidate
    return best


def select_overall(best_per_model: Mapping[str, TrainedCandidate]) -> TrainedCandidate:
    """在所有模型中选出验证集 Macro-F1 最高的一个。

    注意本函数的入参里没有任何测试集数据，因此不可能用测试集选模。

    参数：
        best_per_model：每个模型名的最优候选。

    返回：
        全局最优候选。

    异常：
        ValueError：传入的映射为空时抛出。
    """
    if not best_per_model:
        raise ValueError("没有候选模型可供选择")
    return max(best_per_model.values(), key=lambda candidate: candidate.dev_macro_f1)


def predict_labels(estimator: Any, matrix: spmatrix) -> list[str]:
    """用估计器预测类别名，统一转成字符串列表。

    参数：
        estimator：已训练模型。
        matrix：特征矩阵。

    返回：
        预测的中文类别名列表。
    """
    return [str(value) for value in estimator.predict(matrix)]


def classes_of(estimator: Any) -> list[str]:
    """取估计器的 ``classes_``，用于核对标签顺序未被意外改变。

    参数：
        estimator：已训练模型。

    返回：
        类别名列表。

    异常：
        ValueError：模型缺少 ``classes_`` 或类别数少于 2 时抛出——
            前者说明模型类型不符，后者说明训练数据缺少类别，都属于必须显式报错的情况。
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise ValueError(f"{type(estimator).__name__} 缺少 classes_，无法核对类别顺序")
    values = [str(value) for value in np.asarray(classes).tolist()]
    if len(values) < 2:
        raise ValueError(f"模型的类别数只有 {len(values)}，训练数据可能缺少类别")
    return values
