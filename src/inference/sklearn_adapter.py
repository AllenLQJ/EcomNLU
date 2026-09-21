"""scikit-learn 模型适配器（TF-IDF + 线性模型 / 随机森林等）。

**预处理要求**：本适配器把原始文本直接交给 ``vectorizer.joblib``。因此训练阶段
必须把分词和全部文本清洗步骤**封装进向量器内部**（例如用
``Pipeline``/``TfidfVectorizer(tokenizer=...)``），否则推理阶段的预处理会与训练
阶段不一致，准确率会无报错地下降（AGENTS.md §5.2）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.contract import (
    BaseCategoryPredictor,
    PredictorError,
    load_label_mapping,
    validate_texts,
)


class SklearnCategoryPredictor(BaseCategoryPredictor):
    """加载 ``model.joblib`` 与 ``vectorizer.joblib`` 完成预测。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        mapping = load_label_mapping(self.model_dir)
        self.categories: list[str] = list(mapping["categories"])
        self.label2id: dict[str, int] = dict(mapping["label2id"])
        self.id2category = {int(index): name for name, index in self.label2id.items()}

        self.model: Any = self._load("model.joblib")
        self.vectorizer: Any = self._load("vectorizer.joblib")
        self._ensure_probability_support()
        self._columns = self._resolve_columns()

    def _load(self, filename: str) -> Any:
        """加载 joblib 产物，给出面向使用者的错误。"""
        path = self.model_dir / filename
        if not path.is_file():
            raise PredictorError(f"缺少 {filename}：{path}（见 docs/contracts.md §2.2）")
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底
            raise PredictorError("未安装 joblib，请先执行 pip install -r requirements.txt") from exc
        try:
            return joblib.load(path)
        except Exception as exc:
            raise PredictorError(f"{filename} 加载失败：{type(exc).__name__}: {exc}") from exc

    def _ensure_probability_support(self) -> None:
        """确认模型能输出概率。

        ``LinearSVC`` 这类模型只有 ``decision_function``，没有 ``predict_proba``。
        契约 §2.5 要求返回概率，这不是部署阶段能补救的，必须在训练阶段决定，
        所以这里直接给出可执行的修复方式而不是静默降级。
        """
        if hasattr(self.model, "predict_proba"):
            return
        model_name = type(self.model).__name__
        raise PredictorError(
            f"{model_name} 不支持 predict_proba，无法满足契约 §2.5 的概率输出要求。"
            "修复方式（需在训练阶段完成）：改用 LogisticRegression，"
            "或用 CalibratedClassifierCV 包装后重新保存 model.joblib。"
        )

    def _to_category(self, raw: Any) -> str | None:
        """把模型内部类别标识转成中文类别名。"""
        name = str(raw).strip()
        if name in self.label2id:
            return name
        try:
            return self.id2category.get(int(name))
        except (TypeError, ValueError):
            return None

    def _resolve_columns(self) -> list[str]:
        """把 ``predict_proba`` 的列位置映射到中文类别名。

        不能假定列顺序就是 0..9：sklearn 按 ``classes_`` 的升序排列概率列，
        训练时若某个类别没有样本，列数还会少于 10。
        """
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise PredictorError(
                "模型的 classes_ 缺失，无法确定概率列对应的类别顺序，拒绝预测以避免标签错位"
            )
        columns: list[str] = []
        for raw in classes:
            category = self._to_category(raw)
            if category is None:
                raise PredictorError(f"模型的类别 {raw!r} 不在 label_mapping.json 中")
            columns.append(category)
        return columns

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """返回每个样本覆盖全部类别的概率。"""
        items = validate_texts(texts)
        if not items:
            return []
        matrix = self.vectorizer.transform(items)
        raw_probabilities = self.model.predict_proba(matrix)

        results: list[dict[str, float]] = []
        for row in raw_probabilities:
            probabilities = {category: 0.0 for category in self.categories}
            for position, category in enumerate(self._columns):
                probabilities[category] = float(row[position])
            results.append(probabilities)
        return results
