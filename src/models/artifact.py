"""模型产物契约的共用实现（``docs/contracts.md`` 第二章）。

FastText、BERT 与优化脚本都要产出同一套四件套，并且都必须满足以下硬性要求：

* ``label_mapping.json`` 必须与 ``data/processed/label_mapping.json`` **逐字节相同**，
  只能复制不能重新生成（契约 §2.1）；
* ``metadata.json`` 必须包含 §2.3 的全部字段，其中 ``split_manifest_sha256``
  是"所有模型使用同一份无泄漏测试集"的唯一硬证据；
* 训练数据只能来自 ``data/processed/`` 中通过数据契约校验的划分。

本模块把这些要求集中实现一次，避免各训练脚本各写一套而出现细微不一致。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.data.config import PROJECT_ROOT
from src.evaluation.metrics import now_iso

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_SPLITS = ("train", "dev", "test")


def sha256_of(path: Path) -> str:
    """计算文件的 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_categories(config: dict[str, Any]) -> list[str]:
    """从配置中读出固定类别顺序（契约 §1.3：顺序即 id 分配依据）。"""
    categories = config["task"]["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("configs/base.yaml 的 task.categories 必须是非空列表")
    return [str(name) for name in categories]


def load_label_mapping(processed_dir: Path | None = None) -> dict[str, Any]:
    """读取数据契约产出的 ``label_mapping.json``。"""
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    path = directory / "label_mapping.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}；请先运行 scripts/split_data.py 生成契约数据（M2）"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def split_manifest_sha256(processed_dir: Path | None = None) -> str:
    """计算 ``split_manifest.json`` 的 SHA-256，作为训练数据的指纹。"""
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    manifest = directory / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"缺少 {manifest}；请先运行 scripts/split_data.py 生成契约数据（M2）"
        )
    return sha256_of(manifest)


def read_split(
    split: str,
    categories: list[str] | None = None,
    processed_dir: Path | None = None,
) -> list[tuple[str, int]]:
    """读取某个划分，返回 ``(review, 类别 id)`` 列表。

    只用标准库 ``csv``：项目 venv 与 GPU 训练环境都能直接运行本模块，
    不需要额外安装 pandas。

    注意：``sample_id`` 是追溯用的行号，**不是特征**，契约 §1.2 要求训练代码忽略它。
    """
    if split not in REQUIRED_SPLITS:
        raise ValueError(f"split 必须是 {REQUIRED_SPLITS} 之一，实际为 {split!r}")
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    path = directory / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}；请先运行 scripts/split_data.py")

    mapping = load_label_mapping(directory)
    category_list = list(categories) if categories else list(mapping["categories"])
    label2id = {name: index for index, name in enumerate(category_list)}

    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"review", "cat", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path.name} 的列必须包含 {sorted(required)}，实际为 {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            review = (row.get("review") or "").strip()
            category = (row.get("cat") or "").strip()
            if not review:
                raise ValueError(f"{path.name}:{line_number} 的 review 为空，数据不合格")
            if category not in label2id:
                raise ValueError(
                    f"{path.name}:{line_number} 的 cat={category!r} 不在类别表中"
                )
            rows.append((review, label2id[category]))
    if not rows:
        raise ValueError(f"{path.name} 没有任何样本")
    return rows


def copy_label_mapping(model_dir: Path, processed_dir: Path | None = None) -> Path:
    """把数据契约的 ``label_mapping.json`` 逐字节复制进模型目录。

    契约 §2.1 明确要求"直接复制，不得重新生成"，因此这里用 ``shutil.copyfile``
    复制原始字节，并校验复制结果与源文件完全一致。
    """
    directory = Path(processed_dir) if processed_dir else PROCESSED_DIR
    source = directory / "label_mapping.json"
    if not source.is_file():
        raise FileNotFoundError(f"缺少 {source}；无法复制标签映射")
    target = Path(model_dir) / "label_mapping.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if target.read_bytes() != source.read_bytes():
        raise RuntimeError(f"复制后的 {target} 与源文件不一致")
    return target


def write_metadata(
    model_dir: Path,
    model_name: str,
    framework: str,
    framework_version: str,
    train_rows: int,
    hyperparameters: dict[str, Any],
    device: str,
    max_length: int | None = None,
    task: str = "category_classification",
    random_seed: int = 42,
    notes: str = "",
    extra: dict[str, Any] | None = None,
    processed_dir: Path | None = None,
) -> Path:
    """写出契约 §2.3 规定的 ``metadata.json``。

    ``max_length`` 对 transformers 类模型必须是正整数、对其余模型必须是 ``None``
    （``scripts/verify_model_contract.py`` 会按模型类型校验）。
    """
    if framework == "transformers" and not isinstance(max_length, int):
        raise ValueError("transformers 模型必须填写整数 max_length（契约 §2.3）")
    if framework != "transformers" and max_length is not None:
        raise ValueError("非 transformers 模型的 max_length 必须为 null（契约 §2.3）")

    payload: dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "framework": framework,
        "framework_version": framework_version,
        "created_at": now_iso(),
        "random_seed": random_seed,
        "split_manifest_sha256": split_manifest_sha256(processed_dir),
        "train_rows": int(train_rows),
        "hyperparameters": hyperparameters,
        "max_length": max_length,
        "device": device,
        "notes": notes,
    }
    if extra:
        payload.update(extra)

    path = Path(model_dir) / "metadata.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def write_json(path: Path, payload: Any) -> Path:
    """以 UTF-8 写出 JSON，保留中文。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
