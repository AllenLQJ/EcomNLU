"""BERT 微调的数据管道：Dataset、动态 padding 的 collate_fn 与 DataLoader。

对应 AGENTS.md §5.4 的要求：独立 ``Dataset``、``DataLoader`` 与 ``collate_fn``，
并且支持动态 padding。这里采用"先一次性编码、再按批动态补齐"的做法：

* 编码阶段只做截断不做补齐，因此不会有"整批按 max_length 补齐"的额外算力浪费；
* ``collate_fn`` 把同一批补齐到**该批最长样本**的长度，短句批次的算力显著低于长句批次。

标签只使用 ``cat`` 列，``sample_id`` 是追溯用的行号，按契约 §1.2 必须被忽略。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

ENCODE_CHUNK_SIZE = 4096


class ReviewDataset(Dataset):
    """评论分类数据集，返回 ``(input_ids, label)``。"""

    def __init__(
        self,
        rows: Sequence[tuple[str, int]],
        tokenizer: Any,
        max_length: int,
        desc: str = "",
    ) -> None:
        if max_length <= 0:
            raise ValueError(f"max_length 必须为正，实际为 {max_length}")
        self.max_length = max_length
        self.labels: list[int] = [int(label) for _, label in rows]
        self.input_ids: list[list[int]] = []

        texts = [text for text, _ in rows]
        for start in range(0, len(texts), ENCODE_CHUNK_SIZE):
            chunk = texts[start : start + ENCODE_CHUNK_SIZE]
            encoded = tokenizer(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.input_ids.extend(encoded["input_ids"])
        if len(self.input_ids) != len(self.labels):
            raise RuntimeError(
                f"编码结果与标签数量不一致：{len(self.input_ids)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.input_ids[index], self.labels[index]


def make_collate_fn(pad_token_id: int) -> Callable[[list[tuple[list[int], int]]], dict[str, torch.Tensor]]:
    """返回按批动态补齐的 ``collate_fn``。"""

    def collate_fn(batch: list[tuple[list[int], int]]) -> dict[str, torch.Tensor]:
        sequences, labels = zip(*batch)
        longest = max(len(item) for item in sequences)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for item in sequences:
            padding = longest - len(item)
            input_ids.append(list(item) + [pad_token_id] * padding)
            attention_mask.append([1] * len(item) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(list(labels), dtype=torch.long),
        }

    return collate_fn


def build_dataloader(
    rows: Sequence[tuple[str, int]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """构建 DataLoader；动态 padding 由 ``collate_fn`` 完成。"""
    dataset = ReviewDataset(rows, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def batch_length_stats(loader: DataLoader, max_batches: int = 20) -> dict[str, float]:
    """统计若干批次的实际补齐长度，用于说明动态 padding 的效果。"""
    lengths: list[int] = []
    for index, batch in enumerate(loader):
        lengths.append(int(batch["input_ids"].shape[1]))
        if index + 1 >= max_batches:
            break
    if not lengths:
        return {"batches": 0}
    return {
        "batches": len(lengths),
        "mean_padded_length": round(sum(lengths) / len(lengths), 1),
        "max_padded_length": max(lengths),
    }
