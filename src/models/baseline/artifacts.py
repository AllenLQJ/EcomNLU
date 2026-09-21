"""按接口契约写出模型产物。

模块职责：
    把「已训练的模型 + 特征向量器 + 标签映射 + 元数据 + 指标」按 ``docs/contracts.md``
    §2.1 规定的目录结构写到 ``models/<model_name>/``，使 ``src/inference/`` 的统一
    入口与 ``scripts/verify_model_contract.py`` 能够加载并校验。

主要输入与输出：
    输入：已训练模型、已拟合向量器、数据契约的标签映射、元数据与指标字典。
    输出：``models/<model_name>/`` 下的 ``model.joblib``、``vectorizer.joblib``、
        ``label_mapping.json``、``metadata.json``、``metrics.json``。

关键设计理由与约束：

1. **``label_mapping.json`` 必须与数据契约那份逐字节相同。**
   契约 §2.1 明确要求「直接复制，不得重新生成」，因此这里走
   :func:`src.features.text_features.copy_label_mapping` 的字节复制。
   重新序列化会因键序或缩进差异产生不同的文件，而混淆矩阵的类别顺序依赖它。

2. **``dataset_fingerprint`` 必须由三个划分 CSV 的内容算出。**
   契约 §2.3（v1.0.1）称它是「所有模型使用同一份无泄漏测试集的唯一硬证据」。
   算法固定为 `train` → `dev` → `test` 顺序取三个 CSV 的 SHA-256，
   用半角冒号连接后再取一次 SHA-256，见 :func:`dataset_fingerprint`。
   **刻意不使用 ``split_manifest.json`` 的整体哈希**：该文件含 ``created_at``，
   每次重新生成数据都会变化，会让同一份数据产生不同指纹，
   模型被误判为「基于旧数据训练」。这是 v1.0 的缺陷，v1.0.1 已勘误，不得回退。

3. **``max_length`` 对非 transformers 模型必须为 ``null``。**
   契约 §2.3 规定该字段仅 BERT 类模型填实际值；校验器会检查这一点。

4. **写出前先自检，不留半成品。**
   如果模型缺少 ``predict_proba``、类别与 ``label_mapping`` 不一致，或向量器未拟合，
   这里直接报错。这些问题一旦写进产物，要到 M6 部署时才会暴露，代价更高。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import sklearn
from pathlib import Path

from src.evaluation.metrics import COST_KEYS, METRICS_BLOCK_KEYS
from src.features.text_features import copy_label_mapping, save_vectorizer


# metadata.json 的必需字段，与 scripts/verify_model_contract.py 保持一致。
# 注意：v1.0.1 勘误后，原先的 ``split_manifest_sha256`` 已被 ``dataset_fingerprint`` 取代
# （见 docs/contracts.md §2.3 与
# reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md）。
METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "model_name",
    "task",
    "framework",
    "framework_version",
    "created_at",
    "random_seed",
    "dataset_fingerprint",
    "train_rows",
    "hyperparameters",
    "max_length",
    "device",
)

# 计算数据集指纹时固定的划分顺序，契约 §2.3 规定该顺序不得调换。
FINGERPRINT_SPLIT_ORDER: tuple[str, ...] = ("train", "dev", "test")


def sha256_of(path: Path) -> str:
    """计算文件的 SHA-256。

    参数：
        path：文件路径。

    返回：
        64 位小写十六进制哈希。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(processed_dir: Path) -> str:
    """按契约 §2.3 计算数据集指纹。

    算法固定：先分别取 ``train.csv``、``dev.csv``、``test.csv`` 的 SHA-256（小写十六进制），
    按 ``train:dev:test`` 用半角冒号连接，再对该字符串取 SHA-256。

    该指纹**刻意只由三个 CSV 的内容决定**，不使用 ``split_manifest.json`` 的整体哈希。
    原因：清单里含 ``created_at``，每次重新生成数据都会变化；即使三个 CSV 字节完全一致，
    清单哈希仍会漂移，模型会被误判为「基于旧数据训练」。这是 v1.0 的缺陷，
    v1.0.1 已勘误，**不得回退**。

    参数：
        processed_dir：``data/processed`` 目录。

    返回：
        64 位小写十六进制指纹。

    异常：
        FileNotFoundError：任一划分文件缺失时抛出，避免产出含义不明的指纹。
    """
    digests: list[str] = []
    for split in FINGERPRINT_SPLIT_ORDER:
        path = Path(processed_dir) / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"计算数据集指纹需要 {path}，请先运行 'python scripts/split_data.py'。"
            )
        digests.append(sha256_of(path))
    return hashlib.sha256(":".join(digests).encode("utf-8")).hexdigest()


def build_metadata(
    *,
    model_name: str,
    created_at: str,
    random_seed: int,
    processed_dir: Path,
    train_rows: int,
    hyperparameters: Mapping[str, Any],
    device: str = "cpu",
    notes: str = "",
) -> dict[str, Any]:
    """构造 ``metadata.json`` 的内容。

    参数：
        model_name：模型名，须为契约允许的取值。
        created_at：带时区的 ISO 8601 创建时间。
        random_seed：训练使用的随机种子。
        processed_dir：``data/processed`` 目录，用于实时计算数据集指纹（契约 §2.3）。
            注意这里传入的是**目录**而不是 ``split_manifest.json``：v1.0.1 勘误后指纹
            由三个 CSV 的内容决定，不再使用含 ``created_at`` 的清单文件哈希。
        train_rows：实际参与训练的行数（不含验证集与测试集）。
        hyperparameters：完整到可复现训练的超参数。
        device：实际训练设备。
        notes：备注。

    返回：
        可直接序列化的元数据字典。

    异常：
        FileNotFoundError：三个划分文件不齐时抛出——此时无法证明数据版本，不应产出模型。
    """
    return {
        "model_name": str(model_name),
        "task": "category_classification",
        "framework": "scikit-learn",
        "framework_version": sklearn.__version__,
        "created_at": created_at,
        "random_seed": int(random_seed),
        "dataset_fingerprint": dataset_fingerprint(processed_dir),
        "train_rows": int(train_rows),
        # 契约 §2.3：max_length 仅 BERT 类模型填实际值，其余必须为 null
        "max_length": None,
        "device": str(device),
        "hyperparameters": dict(hyperparameters),
        "notes": str(notes),
    }


def build_metrics(
    *,
    model_name: str,
    categories: Sequence[str],
    dev_block: Mapping[str, Any],
    test_block: Mapping[str, Any] | None,
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """构造 ``metrics.json`` 的内容并校验结构。

    参数：
        model_name：模型名。
        categories：类别顺序，决定 ``per_class`` 的键与混淆矩阵行列。
        dev_block：验证集指标块，不得为 ``None``（契约要求验证集指标必须先产出）。
        test_block：测试集指标块；``None`` 表示尚未评估，序列化为 ``null``。
        cost：推理成本块。

    返回：
        可直接序列化的指标字典。

    异常：
        ValueError：缺字段、类别不一致或混淆矩阵尺寸不符时抛出。
    """
    if dev_block is None:
        raise ValueError("验证集指标块不得为空：契约要求 dev 必须先产出")

    category_list = [str(c) for c in categories]
    for name, block in (("dev", dev_block), ("test", test_block)):
        if block is None:
            continue
        missing = [key for key in METRICS_BLOCK_KEYS if key not in block]
        if missing:
            raise ValueError(f"{name} 指标块缺少字段：{missing}")
        if set(block["per_class"]) != set(category_list):
            raise ValueError(f"{name}.per_class 的类别与 categories 不一致")
        matrix = block["confusion_matrix"]
        size = len(category_list)
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError(f"{name}.confusion_matrix 必须是 {size}×{size}")

    missing_cost = [key for key in COST_KEYS if key not in cost]
    if missing_cost:
        raise ValueError(f"cost 缺少字段：{missing_cost}")

    return {
        "model_name": str(model_name),
        "categories": category_list,
        "dev": dict(dev_block),
        "test": dict(test_block) if test_block is not None else None,
        "cost": dict(cost),
    }


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """以 UTF-8 写出 JSON（缩进 2，保留中文）。

    参数：
        payload：待序列化内容。
        path：输出路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_model_body(
    *,
    models_dir: Path,
    model_name: str,
    estimator: Any,
    vectorizer: Any,
    label_mapping_source: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """写出模型本体、标签映射与元数据（不含 ``metrics.json``）。

    单独提供这一步是因为推理成本必须**通过统一入口加载后实测**
    （``docs/contracts.md`` §2.4 要求实测而非估计），而加载又要求产物已就位，
    因此先写出除指标外的部分，测完成本再补 ``metrics.json``。

    参数：
        models_dir：``models/`` 目录。
        model_name：模型名，用于子目录名。
        estimator：已训练模型。
        vectorizer：已拟合的向量器。
        label_mapping_source：``data/processed/label_mapping.json``。
        metadata：:func:`build_metadata` 的结果。

    返回：
        写出的模型目录路径。

    异常：
        ValueError：模型缺少 ``predict_proba``、类别与标签映射不一致、或元数据缺字段时抛出。
    """
    import joblib

    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"{type(estimator).__name__} 没有 predict_proba，无法满足契约 §2.5 的概率输出要求"
        )
    missing = [key for key in METADATA_REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata 缺少必需字段：{missing}")

    model_dir = Path(models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, model_dir / "model.joblib")
    save_vectorizer(vectorizer, model_dir / "vectorizer.joblib")

    mapping = json.loads(Path(label_mapping_source).read_text(encoding="utf-8"))
    classes = [str(value) for value in getattr(estimator, "classes_", [])]
    if classes and set(classes) != set(mapping["categories"]):
        raise ValueError(
            "模型的 classes_ 与 label_mapping 的类别不一致："
            f"缺少 {sorted(set(mapping['categories']) - set(classes))}，"
            f"多出 {sorted(set(classes) - set(mapping['categories']))}"
        )
    copy_label_mapping(Path(label_mapping_source), model_dir / "label_mapping.json")

    write_json(dict(metadata), model_dir / "metadata.json")
    return model_dir


def write_metrics_file(model_dir: Path, metrics: Mapping[str, Any]) -> Path:
    """写出 ``metrics.json``。

    参数：
        model_dir：模型目录。
        metrics：:func:`build_metrics` 的结果。

    返回：
        写出的 ``metrics.json`` 路径。
    """
    path = Path(model_dir) / "metrics.json"
    write_json(dict(metrics), path)
    return path


def write_model_artifact(
    *,
    models_dir: Path,
    model_name: str,
    estimator: Any,
    vectorizer: Any,
    label_mapping_source: Path,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """一次性写出一个模型目录的全部契约产物。

    等价于依次调用 :func:`write_model_body` 与 :func:`write_metrics_file`，
    便于不测量推理成本的场景（例如单元测试）使用。

    参数：
        models_dir：``models/`` 目录。
        model_name：模型名。
        estimator：已训练模型。
        vectorizer：已拟合的向量器。
        label_mapping_source：``data/processed/label_mapping.json``。
        metadata：元数据字典。
        metrics：指标字典。

    返回：
        写出的模型目录路径。
    """
    model_dir = write_model_body(
        models_dir=models_dir,
        model_name=model_name,
        estimator=estimator,
        vectorizer=vectorizer,
        label_mapping_source=label_mapping_source,
        metadata=metadata,
    )
    write_metrics_file(model_dir, metrics)
    return model_dir
