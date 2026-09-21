"""FastText 训练文件的生成与校验。

FastText 要求每行形如 ``__label__类别 分词后的文本``。AGENTS.md §5.3 明确要求
生成时必须校验制表符、换行符、空文本和非法标签，"不能默认每一行都能安全拆成两个字段"，
因此本模块逐行校验并把处理结果统计出来，而不是静默丢弃或静默写出坏行。

字符级与词级两种切分共用同一套写出逻辑，只在 ``tokenize`` 上分叉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

LABEL_PREFIX = "__label__"

# FastText 以换行分隔样本、以空格分隔词，因此这两类字符必须先行拦截。
FORBIDDEN_CHARS = ("\n", "\r", "\t")


@dataclass
class BuildStats:
    """生成训练文件的统计，用于报告与日志。"""

    written: int = 0
    skipped_empty: int = 0
    skipped_invalid_label: int = 0
    sanitized: int = 0
    examples_skipped: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.written + self.skipped_empty + self.skipped_invalid_label

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "written": self.written,
            "skipped_empty": self.skipped_empty,
            "skipped_invalid_label": self.skipped_invalid_label,
            "sanitized": self.sanitized,
        }


def tokenize_chars(text: str) -> str:
    """字符级切分：每个汉字/符号作为一个 token。"""
    return " ".join(list(text))


def tokenize_words(text: str) -> str:
    """词级切分：jieba 精确模式分词。"""
    import jieba

    return " ".join(jieba.lcut(text))


def sanitize(text: str) -> tuple[str, bool]:
    """把制表符等分隔符替换为空格，返回 ``(处理后的文本, 是否发生过替换)``。

    换行在数据契约里已经由清洗阶段消除，但这里仍然兜底替换：一旦漏进训练文件，
    FastText 会把一条样本切成两条，属于静默的数据破坏。
    """
    cleaned = text
    for char in FORBIDDEN_CHARS:
        if char in cleaned:
            cleaned = cleaned.replace(char, " ")
    cleaned = " ".join(cleaned.split())
    return cleaned, cleaned != text


def build_fasttext_file(
    rows: Iterable[tuple[str, int]],
    categories: Sequence[str],
    output_path: Path,
    tokenize: Callable[[str], str],
) -> BuildStats:
    """把 ``(review, 类别 id)`` 写成 FastText 训练文件。

    Args:
        rows: 来自 ``data/processed`` 的样本，类别以整数 id 给出。
        categories: ``label_mapping.json`` 的顺序，id 即下标。
        output_path: 输出文件路径，父目录会自动创建。
        tokenize: 切分函数，通常传 ``tokenize_chars`` 或 ``tokenize_words``。

    Returns:
        生成统计，包含写出条数与各类跳过条数。

    Raises:
        ValueError: 类别 id 越界时抛出，避免把错误标签写进训练文件。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = BuildStats()

    with output_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for index, (text, label_id) in enumerate(rows):
            if not 0 <= int(label_id) < len(categories):
                raise ValueError(
                    f"第 {index} 条样本的类别 id={label_id} 越界，"
                    f"合法范围 0~{len(categories) - 1}"
                )
            cleaned, changed = sanitize(text)
            if changed:
                stats.sanitized += 1

            tokens = tokenize(cleaned).strip()
            if not tokens:
                # 只有空白或切分后无 token，FastText 会得到没有特征的样本。
                stats.skipped_empty += 1
                if len(stats.examples_skipped) < 5:
                    stats.examples_skipped.append(text[:30])
                continue

            category = categories[int(label_id)]
            if LABEL_PREFIX in tokens or "\x00" in tokens:
                stats.skipped_invalid_label += 1
                if len(stats.examples_skipped) < 5:
                    stats.examples_skipped.append(text[:30])
                continue

            file_obj.write(f"{LABEL_PREFIX}{category} {tokens}\n")
            stats.written += 1

    if stats.written == 0:
        raise ValueError(f"{output_path} 没有任何有效样本，请检查输入数据")
    return stats


def read_fasttext_file(path: Path, limit: int = 3) -> list[str]:
    """读取前若干行，用于日志与人工核对文件格式。"""
    lines: list[str] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            lines.append(line.rstrip("\n"))
            if len(lines) >= limit:
                break
    return lines
