"""BERT 分类模型的构建。

与参考项目（``_04_bert/_03_bert_classifier_model.py``）的差别及原因：

* 参考项目用 ``BertModel`` + 自写 ``nn.Linear`` 头，并以裸 ``state_dict``（``.pt``）保存。
  本项目契约 §2.2 要求模型保存为 ``save_pretrained`` 的 ``model/`` 与 ``tokenizer/``
  目录，因此改用 ``AutoModelForSequenceClassification``，它自带分类头且能直接
  ``save_pretrained``。
* 参考项目的 ``max_len=32`` 直接沿用自新闻标题任务，本项目必须由 token 长度分布决定
  （见 ``src/models/bert/token_stats.py``，实测推荐 192）。

``id2label`` 按 ``label_mapping.json`` 的顺序写入模型配置：统一入口会校验模型声明的
类别顺序与 label_mapping 一致，写错会导致所有预测静默对应到错误类别。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def build_model(
    pretrained_dir: Path,
    categories: Sequence[str],
    dropout: float | None = None,
) -> Any:
    """从本地预训练目录构建序列分类模型。

    Args:
        pretrained_dir: 本地预训练模型目录（由 ``scripts/fetch_pretrained.py`` 下载）。
        categories: ``label_mapping.json`` 的类别顺序，写入 ``id2label``。
        dropout: 可选，覆盖 ``hidden_dropout_prob`` 与 ``attention_probs_dropout_prob``。

    Returns:
        ``AutoModelForSequenceClassification`` 实例（未移动到任何设备）。
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification

    pretrained_dir = Path(pretrained_dir)
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(
            f"预训练模型目录不存在：{pretrained_dir}；"
            "请先运行 python scripts/fetch_pretrained.py bert-base-chinese"
        )

    config = AutoConfig.from_pretrained(str(pretrained_dir))
    config.num_labels = len(categories)
    config.id2label = {index: name for index, name in enumerate(categories)}
    config.label2id = {name: index for index, name in enumerate(categories)}
    if dropout is not None:
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        str(pretrained_dir), config=config
    )
    return model


def parameter_count(model: Any) -> int:
    """统计可训练参数量，写入报告用于体积与规模对比。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
