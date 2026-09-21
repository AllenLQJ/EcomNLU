"""FastText 模型适配器（字符级或词级）。

FastText 的原生输出与契约 §2.5 有四处差异，都在这里抹平：

1. 标签带 ``__label__`` 前缀，需要剥离；
2. 只返回 top-k，需要补齐成完整的类别集合；
3. 输入不能含换行（FastText 把换行当作文档分隔符），需要先压平；
4. **训练时的切分方式必须在推理时一致**。FastText 只按空格切词：若模型是在
   空格分隔的字符序列（字符级）或 jieba 词序列（词级）上训练的，那么直接喂原始
   句子会让整句变成一个 token，预测结果基本等于乱猜。切分方式从 ``metadata.json``
   的 ``preprocess.tokenize`` 读取，取值 ``none`` / ``chars`` / ``jieba_words``；
   字段缺省时按 ``none`` 处理，与旧模型兼容。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

from src.inference.contract import (
    BaseCategoryPredictor,
    PredictorError,
    load_label_mapping,
    load_metadata,
    validate_texts,
)


LABEL_PREFIX = "__label__"
WHITESPACE_PATTERN = re.compile(r"\s+")

# 切分方式，需与训练脚本写入 metadata.json 的取值一致。
TOKENIZE_MODES = ("none", "chars", "jieba_words")

# fastText 的概率来自 float32 softmax，实测会略微超过 1（如 1.0000100135803223）。
# 这是数值精度问题而非模型问题，因此允许极小的越界后再夹紧，而不是直接判为非法概率。
PROBABILITY_TOLERANCE = 1e-4


def _chars_tokenizer(text: str) -> str:
    """字符级切分：与 ``src/models/fasttext/data_io.py`` 的 tokenize_chars 一致。"""
    return " ".join(list(text))


def _jieba_words_tokenizer(text: str) -> str:
    """词级切分：与训练脚本的 tokenize_words 保持一致。"""
    try:
        import jieba
    except ImportError as exc:
        raise PredictorError(
            "该模型的 metadata 声明了 jieba 词级切分，但未安装 jieba，"
            "请执行 pip install -r requirements.txt"
        ) from exc
    return " ".join(jieba.lcut(text))


def resolve_tokenizer(mode: str) -> Callable[[str], str] | None:
    """把 ``preprocess.tokenize`` 的取值解析为切分函数，``none`` 返回 None。"""
    if mode == "none":
        return None
    if mode == "chars":
        return _chars_tokenizer
    if mode == "jieba_words":
        return _jieba_words_tokenizer
    raise PredictorError(
        f"metadata.preprocess.tokenize 取值非法：{mode!r}，允许 {TOKENIZE_MODES}"
    )



def apply_numpy2_compat() -> bool:
    """修补 fasttext-wheel 0.9.2 与 NumPy 2 的不兼容，返回是否应用了补丁。

    ``fasttext`` 的 ``predict`` 内部使用 ``np.array(probs, copy=False)``。NumPy 2.0
    起 ``copy=False`` 的含义变为"禁止复制，需要复制就报错"，而 ``probs`` 是元组，
    必然需要复制，因此 ``predict`` 一定抛 ``ValueError``——训练能跑、预测必崩。

    上游 ``fasttext-wheel`` 最新版本就是 0.9.2，官方 ``fasttext`` 最新 0.9.3 发布于
    NumPy 2 之前，两者都带同一个缺陷，无法靠升级解决。

    这里把 ``fasttext.FastText`` 模块内的 ``np`` 换成垫片：只把
    ``array(x, copy=False)`` 转发为 ``asarray(x)``，其余属性原样透传。
    因此 fasttext 自身的标签/概率处理逻辑完全不变。

    **仅当 NumPy 主版本 >= 2 时生效。** 若团队把 NumPy 降级到 1.x，本补丁自动失效，
    不会留下无效代码。
    """
    import numpy

    if int(numpy.__version__.split(".")[0]) < 2:
        return False

    from fasttext import FastText as fasttext_module

    if getattr(fasttext_module, "_numpy2_compat_applied", False):
        return True

    class _NumpyCompat:
        """只修正 ``array(x, copy=False)``，其余转发给真实 numpy。"""

        def __init__(self, real: Any) -> None:
            self._real = real

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

        def array(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("copy") is False:
                kwargs = {key: value for key, value in kwargs.items() if key != "copy"}
                return self._real.asarray(obj, *args, **kwargs)
            return self._real.array(obj, *args, **kwargs)

    fasttext_module.np = _NumpyCompat(numpy)
    fasttext_module._numpy2_compat_applied = True
    return True


class FastTextCategoryPredictor(BaseCategoryPredictor):
    """加载 ``model.bin`` 完成预测。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        mapping = load_label_mapping(self.model_dir)
        self.categories: list[str] = list(mapping["categories"])

        model_path = self.model_dir / "model.bin"
        if not model_path.is_file():
            raise PredictorError(
                f"缺少 model.bin：{model_path}（见 docs/contracts.md §2.2）"
            )
        try:
            import fasttext
        except ImportError as exc:
            raise PredictorError(
                "未安装 fasttext，请先执行 pip install -r requirements.txt"
            ) from exc
        self.numpy2_compat_applied = apply_numpy2_compat()
        try:
            self.model: Any = fasttext.load_model(str(model_path))
        except Exception as exc:
            raise PredictorError(f"model.bin 加载失败：{type(exc).__name__}: {exc}") from exc

        # 推理时的切分必须与训练时一致，否则预测结果与随机猜测无异。
        self.tokenize_mode = self._read_tokenize_mode()
        self._tokenizer = resolve_tokenizer(self.tokenize_mode)

    def _read_tokenize_mode(self) -> str:
        """读取训练时使用的切分方式；``metadata.json`` 缺失或未声明时按 ``none``。"""
        try:
            metadata = load_metadata(self.model_dir)
        except PredictorError:
            return "none"
        preprocess = metadata.get("preprocess")
        if not isinstance(preprocess, dict):
            return "none"
        mode = preprocess.get("tokenize", "none")
        if not isinstance(mode, str):
            raise PredictorError(
                f"metadata.preprocess.tokenize 必须是字符串，实际为 {mode!r}"
            )
        return mode

    def _flatten(self, text: str) -> str:
        """把文本压成单行，避免 FastText 把换行当成文档分隔符。"""
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    def _prepare(self, text: str) -> str:
        """压平为单行后，按训练时的切分方式切分。"""
        flattened = self._flatten(text)
        if self._tokenizer is None:
            return flattened
        return self._tokenizer(flattened)

    @staticmethod
    def _strip_prefix(label: str) -> str:
        """剥离 FastText 的 ``__label__`` 前缀。"""
        return label[len(LABEL_PREFIX) :] if label.startswith(LABEL_PREFIX) else label

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """返回每个样本覆盖全部类别的概率。

        FastText 的 ``predict`` 给出的是对全部标签做 softmax 后的概率，因此请求
        ``k = 类别总数`` 时各项之和已经为 1，不需要再做归一化。
        """
        items = validate_texts(texts)
        results: list[dict[str, float]] = []
        for text in items:
            labels, scores = self.model.predict(
                self._prepare(text), k=len(self.categories)
            )
            if len(labels) != len(scores):
                raise PredictorError(
                    f"FastText 返回的标签数 {len(labels)} 与概率数 {len(scores)} 不一致"
                )
            probabilities = {category: 0.0 for category in self.categories}
            seen: set[str] = set()
            for label, score in zip(labels, scores):
                if not isinstance(label, str):
                    raise PredictorError(
                        f"FastText 返回的标签必须是字符串，实际为 {type(label).__name__}"
                    )
                category = self._strip_prefix(label)
                if category not in probabilities:
                    raise PredictorError(
                        f"FastText 返回未知标签 {label!r}，不在 label_mapping.json 中"
                    )
                if category in seen:
                    raise PredictorError(f"FastText 重复返回标签 {label!r}")
                try:
                    probability = float(score)
                except (TypeError, ValueError) as exc:
                    raise PredictorError(
                        f"FastText 标签 {label!r} 的概率不是数值：{score!r}"
                    ) from exc
                if (
                    not math.isfinite(probability)
                    or probability < -PROBABILITY_TOLERANCE
                    or probability > 1.0 + PROBABILITY_TOLERANCE
                ):
                    raise PredictorError(
                        f"FastText 标签 {label!r} 的概率必须是 [0, 1] 内的有限数，"
                        f"实际为 {score!r}"
                    )
                # 夹紧到 [0, 1]：覆盖 float32 softmax 的微小越界（见本模块开头说明）。
                probability = min(1.0, max(0.0, probability))
                seen.add(category)
                probabilities[category] = probability

            probability_sum = sum(probabilities.values())
            if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-3):
                raise PredictorError(
                    f"FastText 概率和必须约为 1，实际为 {probability_sum:.6f}；"
                    "请检查模型标签与 label_mapping.json 是否一致"
                )
            results.append(probabilities)
        return results
