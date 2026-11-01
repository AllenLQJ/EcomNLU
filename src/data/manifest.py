"""接口契约要求的数据产物清单生成。

模块职责：
    生成并写出 ``data/processed/label_mapping.json`` 与 ``data/processed/split_manifest.json``，
    这两个文件是接口契约（``docs/contracts.md`` 第一节）规定的强制产物：
    前者冻结类别到整数 id 的映射，后者记录数据来源、清洗与划分的全部可核查事实。

主要输入与输出：
    输入：配置中的类别顺序与随机种子、原始数据哈希、清洗步骤统计、划分结果与泄漏检查结果。
    输出：两个 UTF-8 JSON 文件，写在 ``data/processed/`` 下。

关键设计理由与约束：
    - ``label_mapping.json`` 一旦生成即冻结：混淆矩阵与分类报告的类别顺序全部依赖它，
      因此本模块只按 ``configs/base.yaml`` 的 ``task.categories`` 顺序生成，
      不允许调用方传入自定义顺序。
    - ``split_manifest.json`` 中的 ``sample_id`` 可追溯性、文件哈希、行数与泄漏计数
      全部取自本次真实运行的结果，**不得手写或估计**，否则该文件就失去了证明力。
    - ``created_at`` 必须带时区，否则契约校验会判定为不合法。
    - 字符串值不得写成 ``<...>`` 形式：契约禁止保留未替换的占位符。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.config import public_path
from src.data.raw_io import CONTRACT_COLUMNS, sha256_of_file


# 契约要求 cleaning.rules 必须包含且不得留空的四个键，
# 其余键是本项目额外补充的规则说明，用于让清洗过程完整可查。
CLEANING_RULES: dict[str, str] = {
    # --- 契约要求的四项 ---
    "empty_review": "drop",
    "duplicate_review": "drop_within_category_only",
    "control_chars": "strip_chars_keep_row",
    "short_text_threshold": "none",
    # --- 本项目额外的规则说明 ---
    "unicode_format_chars": "strip_chars_keep_row",
    "html_entities": "unescape",
    "html_tags": "keep",
    "whitespace": "normalize_including_unicode_spaces",
    "meaningless_text": "drop_if_no_cjk_and_no_latin_letter",
    "urls": "keep",
}

# 泄漏检查在 manifest 中使用的键名，由契约固定。
LEAKAGE_KEYS: tuple[str, ...] = (
    "train_dev_overlap",
    "train_test_overlap",
    "dev_test_overlap",
)

BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def beijing_timestamp() -> str:
    """返回带时区的 ISO 8601 北京时间字符串。

    返回：
        形如 ``2026-09-21T11:30:00+08:00`` 的时间字符串。
        契约要求 ``created_at`` 必须带时区，否则校验不通过。
    """
    return datetime.now(BEIJING_TIMEZONE).isoformat(timespec="seconds")


def build_label_mapping(categories: Sequence[str]) -> dict[str, Any]:
    """按契约要求构造标签映射。

    id 严格等于类别在 ``categories`` 中的下标，
    满足 ``label2id[类别] == categories.index(类别)``。

    参数：
        categories：类别名称序列，必须与 ``configs/base.yaml`` 中 ``task.categories``
            的顺序完全一致。

    返回：
        含 ``categories`` / ``label2id`` / ``id2label`` 三个字段的字典。
    """
    category_list = [str(category) for category in categories]
    if len(set(category_list)) != len(category_list):
        raise ValueError(f"类别名称存在重复：{category_list}")
    return {
        "categories": category_list,
        "label2id": {name: index for index, name in enumerate(category_list)},
        "id2label": {str(index): name for index, name in enumerate(category_list)},
    }


def leakage_for_manifest(leakage: Mapping[str, Any]) -> dict[str, Any]:
    """把泄漏检查结果转换成契约规定的字段结构。

    参数：
        leakage：:func:`src.data.splitting.check_text_leakage` 的返回值，
            其 ``pairwise_overlaps`` 以 ``train__dev`` 形式作键。

    返回：
        含 ``based_on`` 与三个 ``*_overlap`` 计数的字典。
        ``based_on`` 固定为 ``normalized_text``：本项目划分时以清洗后的文本为分组依据，
        清洗已包含空白归一化，因此泄漏检查的实际口径就是归一化文本。
    """
    overlaps = leakage.get("pairwise_overlaps", {})
    manifest: dict[str, Any] = {"based_on": "normalized_text"}
    for key in LEAKAGE_KEYS:
        left, right = key.replace("_overlap", "").split("_", 1)
        manifest[key] = int(overlaps.get(f"{left}__{right}", 0))
    return manifest


def build_split_manifest(
    *,
    seed: int,
    raw_path: str,
    ratios: Mapping[str, float],
    rows_before: int,
    rows_after: int,
    counts: Mapping[str, int],
    file_hashes: Mapping[str, str],
    leakage: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """构造划分清单。

    参数：
        seed：随机种子，必须与配置中的 ``project.random_seed`` 一致。
        raw_path：原始数据路径，必须与配置中的 ``data.raw_path`` 一致。
        ratios：各集合的目标比例。
        rows_before：清洗前样本量。
        rows_after：清洗后样本量，必须等于三个划分的行数之和。
        counts：各集合行数。
        file_hashes：``{文件名: sha256}``，用于证明所有模型使用的是同一份数据。
        leakage：契约结构的泄漏检查结果。
        created_at：时间戳；为 None 时取当前北京时间。

    返回：
        可直接序列化为 JSON 的字典。
    """
    return {
        "created_at": created_at or beijing_timestamp(),
        "random_seed": int(seed),
        "source": {
            "path": raw_path,
            # 哈希由调用方从真实文件计算后传入，不在此处重新读取，
            # 避免清单与实际使用的文件不一致。
        },
        "cleaning": {
            "rules": dict(CLEANING_RULES),
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
        },
        "split": {
            "strategy": "stratified_by_cat",
            "ratios": {name: float(value) for name, value in ratios.items()},
            "dedup_key": "normalized_text",
        },
        "counts": {name: int(value) for name, value in counts.items()},
        "files": {
            name: {"sha256": digest, "rows": int(counts[name.replace(".csv", "")])}
            for name, digest in file_hashes.items()
        },
        "leakage_check": dict(leakage),
    }


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """以 UTF-8 写出 JSON。

    参数：
        payload：待序列化的内容。
        path：输出路径；父目录不存在时自动创建。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_contract_artifacts(
    *,
    processed_dir: Path,
    categories: Sequence[str],
    split_files: Mapping[str, str],
    seed: int,
    raw_path: str,
    ratios: Mapping[str, float],
    rows_before: int,
    rows_after: int,
    counts: Mapping[str, int],
    leakage: Mapping[str, Any],
) -> tuple[Path, Path]:
    """按契约写出 ``label_mapping.json`` 与 ``split_manifest.json``。

    文件哈希在本函数内从磁盘上的真实文件计算，确保清单描述的就是实际写出的数据。

    参数：
        processed_dir：``data/processed`` 目录。
        categories：类别顺序，取自配置。
        split_files：``{集合名: 文件名}``。
        seed：随机种子。
        raw_path：原始数据路径。
        ratios：各集合目标比例。
        rows_before：清洗前样本量。
        rows_after：清洗后样本量。
        counts：各集合行数。
        leakage：:func:`src.data.splitting.check_text_leakage` 的返回值。

    返回：
        ``(label_mapping.json 路径, split_manifest.json 路径)``。

    异常：
        FileNotFoundError：划分文件不存在时抛出，避免写出哈希缺失的清单。
    """
    label_mapping_path = processed_dir / "label_mapping.json"
    manifest_path = processed_dir / "split_manifest.json"

    # 先确认所有划分文件都已写出，再计算哈希，避免记录到过期或不存在的文件
    file_hashes: dict[str, str] = {}
    for split_name, filename in split_files.items():
        file_path = processed_dir / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"划分文件不存在，无法生成清单：{public_path(file_path)}")
        file_hashes[filename] = sha256_of_file(file_path)

    write_json(build_label_mapping(categories), label_mapping_path)

    raw_file = Path(raw_path)
    manifest = build_split_manifest(
        seed=seed,
        raw_path=public_path(raw_path),
        ratios=ratios,
        rows_before=rows_before,
        rows_after=rows_after,
        counts=counts,
        file_hashes=file_hashes,
        leakage=leakage_for_manifest(leakage),
    )
    # source.sha256 必须来自真实的原始文件；原始数据缺失时明确报错而不是留空
    if not raw_file.is_file():
        raise FileNotFoundError(f"原始数据不存在，无法记录其哈希：{public_path(raw_file)}")
    manifest["source"]["sha256"] = sha256_of_file(raw_file)

    write_json(manifest, manifest_path)
    return label_mapping_path, manifest_path


__all__ = [
    "CONTRACT_COLUMNS",
    "CLEANING_RULES",
    "beijing_timestamp",
    "build_label_mapping",
    "build_split_manifest",
    "leakage_for_manifest",
    "write_contract_artifacts",
    "write_json",
]
