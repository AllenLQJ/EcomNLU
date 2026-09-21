"""Transformers（BERT 等）模型适配器。

从模型目录下的 ``model/`` 与 ``tokenizer/`` 子目录加载，二者都由
``save_pretrained`` 产出（契约 §2.2）。``max_length`` 取自 ``metadata.json``——
契约 §2.3 要求该值由本数据集的 token 长度分布决定，不得照搬其它项目。

本适配器同时承担 M5 量化产物的加载：``metadata.json`` 里出现 ``quantization``
字段时改走 :mod:`src.models.optimization.quantization` 的量化加载路径，并**强制在
CPU 上推理**——int8 动态量化只对 CPU 有加速，放到 GPU 上反而更慢。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.contract import (
    BaseCategoryPredictor,
    PredictorError,
    load_label_mapping,
    load_metadata,
    validate_texts,
)


DEFAULT_BATCH_SIZE = 32


class TransformersCategoryPredictor(BaseCategoryPredictor):
    """加载 ``model/`` 与 ``tokenizer/`` 完成预测。"""

    def __init__(self, model_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_dir = Path(model_dir)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise PredictorError(
                f"batch_size 必须是正整数，实际为 {batch_size!r}"
            )
        self.batch_size = batch_size

        mapping = load_label_mapping(self.model_dir)
        self.categories: list[str] = list(mapping["categories"])
        self.label2id: dict[str, int] = dict(mapping["label2id"])

        metadata = load_metadata(self.model_dir)
        self.max_length = metadata.get("max_length")
        if not isinstance(self.max_length, int) or isinstance(self.max_length, bool):
            raise PredictorError(
                f"metadata.json 的 max_length 必须是整数，实际为 {self.max_length!r}"
                "（契约 §2.3：transformers 模型必须填写实际的 max_length）"
            )
        if self.max_length <= 0:
            raise PredictorError(f"metadata.json 的 max_length 必须为正，实际为 {self.max_length}")

        model_path = self.model_dir / "model"
        tokenizer_path = self.model_dir / "tokenizer"
        for path in (model_path, tokenizer_path):
            if not path.is_dir():
                raise PredictorError(
                    f"缺少目录 {path.name}/：{path}（见 docs/contracts.md §2.2）"
                )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise PredictorError(
                "未安装 torch / transformers，请先执行 pip install -r requirements.txt"
            ) from exc

        # metadata 里的 quantization 字段决定加载路径：量化产物的 ``model/`` 下只有
        # ``quantized_state_dict.pt``，没有 model.safetensors，必须走专用加载器。
        self.is_quantized: bool = isinstance(metadata.get("quantization"), dict)

        try:
            if self.is_quantized:
                from src.models.optimization.quantization import load_quantized_model

                self.model, self.tokenizer = load_quantized_model(self.model_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path)
                )
        except Exception as exc:
            raise PredictorError(f"模型或分词器加载失败：{type(exc).__name__}: {exc}") from exc

        self._torch = torch
        # 量化模型固定在 CPU：int8 动态量化的收益来自 CPU 上的 oneDNN/fbgemm 算子，
        # 搬到 CUDA 上既无对应算子、又会因权重反复反量化而变慢。
        if self.is_quantized:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self._check_label_count()
        self._check_declared_labels()

    def _check_label_count(self) -> None:
        """确认输出维度与类别数一致。"""
        num_labels = getattr(self.model.config, "num_labels", None)
        if num_labels != len(self.categories):
            raise PredictorError(
                f"模型的 num_labels 为 {num_labels}，label_mapping.json 有 "
                f"{len(self.categories)} 个类别，二者必须一致"
            )

    def _check_declared_labels(self) -> None:
        """若模型自身声明了可识别的类别名，校验其顺序与 label_mapping 一致。

        顺序错位不会报错，只会让所有预测静默对应到错误的类别，因此这里主动拦截。
        模型若把类别存成 ``"0"``~``"9"`` 这类无意义标识，则跳过本项检查。
        """
        id2label = getattr(self.model.config, "id2label", None)
        if not isinstance(id2label, dict):
            return
        declared = [id2label.get(index) for index in range(len(self.categories))]
        if not all(isinstance(name, str) and name in self.label2id for name in declared):
            return
        if declared != self.categories:
            raise PredictorError(
                f"模型声明的类别顺序 {declared} 与 label_mapping.json 的 "
                f"{self.categories} 不一致，预测结果会整体错位"
            )

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """分批推理并返回每个样本覆盖全部类别的概率。"""
        items = validate_texts(texts)
        results: list[dict[str, float]] = []
        torch = self._torch

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probabilities:
                results.append(
                    {
                        category: float(row[index])
                        for index, category in enumerate(self.categories)
                    }
                )
        return results
