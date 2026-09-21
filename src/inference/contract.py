"""推理层的统一接口定义。

对应 ``docs/contracts.md`` §2.5。本模块**不导入任何重型依赖**（torch、
transformers、fasttext），以保证 ``import src.inference`` 足够快；各框架的依赖由
对应适配器自行按需导入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class PredictorError(RuntimeError):
    """推理层可预期的错误。

    消息面向使用者，说明哪里不符合契约以及怎么修，不包含 Python 堆栈。
    """


@runtime_checkable
class CategoryPredictor(Protocol):
    """所有模型对外的统一预测接口（契约 §2.5）。"""

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """返回每个样本的 ``{中文类别名: 概率}``，必须覆盖全部类别。"""

    def predict(self, texts: list[str]) -> list[str]:
        """返回每个样本的预测中文类别名。"""


class BaseCategoryPredictor:
    """适配器基类。

    ``predict`` 一律由 ``predict_proba`` 取最高概率项得出，子类只需要实现
    ``predict_proba``。这样可保证预测结果与概率自洽——
    ``scripts/verify_model_contract.py`` 会校验 ``predict`` 必须等于最高概率类别，
    集中实现可避免各适配器各写一套而产生不一致。
    """

    categories: list[str]

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """子类必须实现。"""
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[str]:
        """按最高概率取类别。"""
        return [max(item, key=item.get) for item in self.predict_proba(texts)]


def read_json(model_dir: Path, filename: str) -> dict[str, Any]:
    """读取模型目录下的 JSON 文件，失败时抛出面向使用者的错误。"""
    path = Path(model_dir) / filename
    if not path.is_file():
        raise PredictorError(f"缺少 {filename}：{path}（见 docs/contracts.md §2.1）")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictorError(f"{filename} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise PredictorError(f"{filename} 的顶层必须是 JSON 对象")
    return data


def load_label_mapping(model_dir: Path) -> dict[str, Any]:
    """读取并校验 ``label_mapping.json``。"""
    mapping = read_json(model_dir, "label_mapping.json")
    categories = mapping.get("categories")
    if not isinstance(categories, list) or not categories:
        raise PredictorError("label_mapping.json 缺少非空的 categories 列表")
    if not all(isinstance(name, str) for name in categories):
        raise PredictorError("label_mapping.json 的 categories 必须全部是字符串")
    label2id = mapping.get("label2id")
    expected_label2id = {name: index for index, name in enumerate(categories)}
    if label2id != expected_label2id:
        raise PredictorError(
            "label_mapping.json 的 label2id 必须与 categories 顺序一致；"
            f"期望 {expected_label2id}，实际 {label2id!r}"
        )

    id2label = mapping.get("id2label")
    expected_id2label = {str(index): name for index, name in enumerate(categories)}
    if id2label != expected_id2label:
        raise PredictorError(
            "label_mapping.json 的 id2label 必须与 categories 顺序一致；"
            f"期望 {expected_id2label}，实际 {id2label!r}"
        )
    return mapping


def load_metadata(model_dir: Path) -> dict[str, Any]:
    """读取 ``metadata.json``。"""
    return read_json(model_dir, "metadata.json")


def validate_texts(texts: list[str]) -> list[str]:
    """校验输入是字符串列表，返回普通 list。"""
    if isinstance(texts, str):
        raise PredictorError("texts 必须是字符串列表，不能直接传单个字符串")
    if not isinstance(texts, (list, tuple)):
        raise PredictorError(f"texts 必须是列表，实际为 {type(texts).__name__}")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise PredictorError(f"texts[{index}] 必须是字符串，实际为 {type(text).__name__}")
    return list(texts)
