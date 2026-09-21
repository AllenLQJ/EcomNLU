"""原始数据集与产物的读写工具。

模块职责：
    集中处理原始 CSV 的读取、文件哈希计算与契约要求的 CSV 写出，使数据流水线各处
    使用一致的文件访问方式与编码约定。

主要输入与输出：
    输入：文件路径。
    输出：已读取的 :class:`pandas.DataFrame`；或写入磁盘的 UTF-8 CSV。

关键设计理由与约束：
    原始数据是只读资产：本模块只提供读取与哈希能力，**不提供任何写入原始目录的函数**，
    从接口上杜绝误覆盖原始数据的可能（AGENTS.md 要求原始数据必须保留、不得覆盖）。
    契约要求产物 CSV 编码为 UTF-8 且不带 BOM、不写出 pandas 行索引，
    因此写出统一走 :func:`write_contract_csv`。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd


# 数据契约规定的列顺序，任何写出的划分文件都必须严格遵守。
CONTRACT_COLUMNS: tuple[str, ...] = ("sample_id", "review", "cat", "label")


def sha256_of_file(path: Path) -> str:
    """计算文件的 SHA-256 摘要。

    分块读取，避免把上百 MB 的文件一次性载入内存。

    参数：
        path：文件路径。

    返回：
        64 位小写十六进制哈希字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    """读取原始评论 CSV。

    参数：
        csv_path：原始 CSV 路径。

    返回：
        原始数据表，不含额外处理，保持与磁盘内容一致。

    异常：
        FileNotFoundError：文件不存在时抛出，并提示先运行下载脚本。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"未找到原始数据集：{csv_path}。请先运行 'python scripts/download_data.py'。"
        )
    return pd.read_csv(csv_path, encoding="utf-8")


def write_contract_csv(
    dataframe: pd.DataFrame, path: Path, columns: Sequence[str] = CONTRACT_COLUMNS
) -> None:
    """按数据契约要求写出 CSV。

    要求：UTF-8 不带 BOM（``encoding="utf-8"`` 而非 ``utf-8-sig``）、不写出行索引、
    列顺序严格为 ``sample_id, review, cat, label``。

    参数：
        dataframe：待写出的数据表。
        path：输出路径；其父目录不存在时自动创建。
        columns：输出列顺序，默认取契约规定的顺序。

    异常：
        KeyError：数据表缺少契约要求的列时抛出，避免写出结构不合规的文件。
    """
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"写出契约 CSV 时缺少必需列：{missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.loc[:, list(columns)].to_csv(path, index=False, encoding="utf-8")
