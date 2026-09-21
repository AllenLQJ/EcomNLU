"""校验 ``models/<model_name>/`` 是否符合 ``docs/contracts.md`` 中的模型产物契约。

只读校验，不修改任何文件。存在任何一项失败时以非零退出码结束。

结构类检查（文件、字段、标签顺序、指标形状）始终执行；真实推理检查依赖
``src/inference/`` 中尚未实现的 ``load_predictor``，未实现时该项标记为 SKIP，
不会被当作通过。

按 ``metadata.task`` 区分两类模型：

- ``category_classification``：10 个商品类。``label_mapping.json`` 必须与
  ``data/processed/label_mapping.json`` **逐字节相同**；
- ``sentiment_classification``（M5）：2 个情感类。``label_mapping.json`` 的
  ``categories`` 必须与 ``configs/base.yaml`` 的 ``task.sentiment_categories`` 一致
  （契约 v1.2 §2.1）。

两类的**统一推理入口都是** ``load_predictor``——适配器是任务无关的，
标签集取自模型目录自己的 ``label_mapping.json``（草案 §2.5）。

用法::

    python scripts/verify_model_contract.py models/tfidf_logreg
    python scripts/verify_model_contract.py models/tfidf_logreg models/bert_base_chinese
    python scripts/verify_model_contract.py            # 校验 models/ 下全部模型
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_JSON_FILES = ("label_mapping.json", "metadata.json", "metrics.json")
SPLIT_FILES = ("train.csv", "dev.csv", "test.csv")
METADATA_REQUIRED_KEYS = (
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
ALLOWED_TASKS = ("category_classification", "sentiment_classification")

# 情感任务的标签顺序来源（契约 v1.2 §2.1）：
# configs/base.yaml 的 task.sentiment_categories。刻意不在本脚本里硬编码，
# 也不在模块顶层 import src.inference——脚本要能在 src 包导入失败时仍给出结构类结论。
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
METRICS_BLOCK_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "per_class",
    "confusion_matrix",
)
COST_KEYS = (
    "model_size_bytes",
    "single_sample_latency_ms",
    "batch_32_latency_ms",
    "throughput_per_sec",
    "measured_on",
    "measured_at",
)

# framework -> 必需的模型本体文件（文件或目录）
MODEL_BODY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "scikit-learn": ("model.joblib", "vectorizer.joblib"),
    "fasttext": ("model.bin",),
    "transformers": ("model", "tokenizer"),
}

PROBE_TEXTS = ("手机很好用，电池耐用", "这本书内容不错，值得一读")


@dataclass
class CheckResult:
    """单项校验结果。``status`` 取值：ok / fail / skip。"""

    name: str
    status: str
    detail: str = ""


def sha256_of(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprint(processed_dir: Path) -> str | None:
    """按契约 §2.3 计算数据集指纹；划分文件不全时返回 ``None``。

    指纹由 train/dev/test 三个 CSV 的**文件内容**决定，顺序固定为 train:dev:test。
    刻意不采用 ``split_manifest.json`` 的整体哈希——该文件含每次运行都会变化的
    ``created_at``，会让同一份数据产生不同指纹（v1.0 的缺陷，见 reports/m3-validation.md）。
    """
    parts = []
    for name in SPLIT_FILES:
        path = processed_dir / name
        if not path.is_file():
            return None
        parts.append(sha256_of(path))
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def find_placeholders(node: Any, path: str = "") -> list[str]:
    """递归查找形如 ``<...>`` 的未替换占位符。"""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(find_placeholders(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(find_placeholders(value, f"{path}[{index}]"))
    elif isinstance(node, str) and node.startswith("<") and node.endswith(">"):
        found.append(f"{path} = {node}")
    return found


def prediction_problems(
    probas: Any, labels: Any, expected_categories: list[str]
) -> list[str]:
    """校验统一推理接口的返回值，返回所有发现的问题。"""
    problems: list[str] = []
    expected_count = len(PROBE_TEXTS)
    if not isinstance(probas, list):
        return ["predict_proba 返回值必须是 list"]
    if not isinstance(labels, list):
        return ["predict 返回值必须是 list"]
    if len(probas) != expected_count:
        problems.append(f"predict_proba 返回 {len(probas)} 条，期望 {expected_count} 条")
    if len(labels) != expected_count:
        problems.append(f"predict 返回 {len(labels)} 条，期望 {expected_count} 条")

    for index, item in enumerate(probas):
        if not isinstance(item, dict):
            problems.append(f"第 {index} 条概率不是字典")
            continue
        if set(item) != set(expected_categories):
            problems.append(
                f"第 {index} 条未覆盖全部 {len(expected_categories)} 个标签："
                f"期望 {sorted(expected_categories)}，实际 {sorted(item)}"
            )
            continue

        invalid_values = [
            f"{category}={value!r}"
            for category, value in item.items()
            if isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ]
        if invalid_values:
            problems.append(f"第 {index} 条含非法概率：{invalid_values[:3]}")
            continue

        probability_sum = sum(float(value) for value in item.values())
        if abs(probability_sum - 1.0) > 1e-3:
            problems.append(f"第 {index} 条概率和不为 1（{probability_sum:.4f}）")
            continue

        if index < len(labels):
            predicted_label = labels[index]
            highest_probability = max(float(value) for value in item.values())
            predicted_probability = (
                float(item[predicted_label]) if predicted_label in item else float("-inf")
            )
            if not math.isclose(
                predicted_probability, highest_probability, rel_tol=0.0, abs_tol=1e-12
            ):
                problems.append(
                    f"第 {index} 条 predict={predicted_label!r} 不是最高概率类别"
                )

    for index, label in enumerate(labels):
        if label not in expected_categories:
            problems.append(f"第 {index} 条预测 {label!r} 不在类别集合内")
    return problems


class ModelContractValidator:
    """按模型产物契约校验单个模型目录。"""

    def __init__(self, model_dir: Path, processed_dir: Path) -> None:
        self.model_dir = model_dir
        self.processed_dir = processed_dir
        self.checks: list[CheckResult] = []
        self.label_mapping: dict[str, Any] | None = None
        # 由 metadata.task 决定：类别任务比对 10 类商品映射，
        # 情感任务比对 2 类情感映射。见 resolve_expected_categories。
        self.task: str | None = None
        self.expected_categories: list[str] | None = None

    def record(self, name: str, status: str, detail: str = "") -> bool:
        """记录一项校验结果。"""
        self.checks.append(CheckResult(name, status, detail))
        return status == "ok"

    def load_json(self, filename: str) -> dict[str, Any] | None:
        """读取模型目录下的 JSON 文件。"""
        path = self.model_dir / filename
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.record(f"{filename} 可解析", "fail", f"{type(exc).__name__}: {exc}")
        return None

    # ---------------------------------------------------------------- 校验流程

    def run(self) -> list[CheckResult]:
        """执行全部校验。"""
        if not self.record(
            "模型目录存在",
            "ok" if self.model_dir.is_dir() else "fail",
            "" if self.model_dir.is_dir() else f"目录不存在：{self.model_dir}",
        ):
            return self.checks

        missing = [name for name in REQUIRED_JSON_FILES if not (self.model_dir / name).is_file()]
        if not self.record(
            "必需 JSON 文件齐全",
            "fail" if missing else "ok",
            f"缺少：{missing}" if missing else "",
        ):
            return self.checks

        self.label_mapping = self.load_json("label_mapping.json")
        metadata = self.load_json("metadata.json")
        metrics = self.load_json("metrics.json")

        self.task = (metadata or {}).get("task")
        # 标签集统一取自模型目录自己的 label_mapping.json（契约 v1.2 §2.5）。
        # 主任务那份是 10 个商品类，情感任务那份是 2 个情感类，规则完全相同。
        self.expected_categories = (self.label_mapping or {}).get("categories") or None

        self.check_label_mapping()
        if metadata is not None:
            self.check_metadata(metadata)
            self.check_model_body(metadata)
        if metrics is not None:
            self.check_metrics(metrics)
        self.check_inference()
        return self.checks

    # ---------------------------------------------------------------- 各项校验

    def load_sentiment_categories(self) -> list[str] | None:
        """从 ``configs/base.yaml`` 读取情感标签顺序。

        契约 v1.2 §2.1 把情感标签顺序的来源固定为配置项
        ``task.sentiment_categories``：标签顺序是契约的一部分，必须集中管理、
        能被校验器与 Web 层同时读取，不能散落在代码里。

        返回：
            情感标签列表；配置缺失或格式不符时返回 ``None``。
        """
        if not CONFIG_PATH.is_file():
            return None
        try:
            import yaml
        except ImportError:  # pragma: no cover - 依赖缺失时的兜底
            return None
        try:
            payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        task_section = payload.get("task")
        if not isinstance(task_section, dict):
            return None
        values = task_section.get("sentiment_categories")
        if not isinstance(values, list) or not values:
            return None
        return [str(value) for value in values]

    def check_label_mapping(self) -> None:
        """按任务校验 ``label_mapping.json``。

        契约 v1.2 §2.1 把现行单条规则按任务分列：

        - 主任务：与 ``data/processed/label_mapping.json`` **逐字节相同**（直接复制）；
        - 情感任务：``categories`` 必须等于 ``configs/base.yaml`` 的
          ``task.sentiment_categories``。

        分成两条是必要的：情感模型的标签集与 10 个商品类**必然不同**，
        套用字节比对会让任何情感模型在校验第一步就失败。
        """
        if self.task == "sentiment_classification":
            expected = self.load_sentiment_categories()
            if expected is None:
                self.record(
                    "label_mapping 与配置一致",
                    "skip",
                    f"{CONFIG_PATH.name} 缺少 task.sentiment_categories，无法判定情感标签",
                )
                return
            actual = (self.label_mapping or {}).get("categories")
            valid = actual == expected
            self.record(
                "label_mapping 与 task.sentiment_categories 一致",
                "ok" if valid else "fail",
                ""
                if valid
                else f"label_mapping.categories={actual}，配置={expected}"
                "；情感模型的 label_mapping.json 必须按该配置生成，不得复制商品映射",
            )
            return

        reference = self.processed_dir / "label_mapping.json"
        if not reference.is_file():
            self.record(
                "label_mapping 与数据契约一致",
                "skip",
                f"基准文件不存在：{reference}（需先完成 M2）",
            )
            return
        actual_bytes = (self.model_dir / "label_mapping.json").read_bytes()
        reference_bytes = reference.read_bytes()
        self.record(
            "label_mapping 与数据契约逐字节一致",
            "ok" if actual_bytes == reference_bytes else "fail",
            ""
            if actual_bytes == reference_bytes
            else "内容不同，必须直接复制 data/processed/label_mapping.json，不得重新生成",
        )

    def check_metadata(self, metadata: dict[str, Any]) -> None:
        """校验 metadata.json 的字段与数据集指纹。"""
        missing = [key for key in METADATA_REQUIRED_KEYS if key not in metadata]
        self.record(
            "metadata 必需字段齐全",
            "fail" if missing else "ok",
            f"缺少：{missing}" if missing else "",
        )

        placeholders = find_placeholders(metadata)
        self.record(
            "metadata 无残留占位符",
            "fail" if placeholders else "ok",
            f"仍为占位符：{placeholders[:5]}" if placeholders else "",
        )

        task = metadata.get("task")
        self.record(
            "metadata.task 合法",
            "ok" if task in ALLOWED_TASKS else "fail",
            "" if task in ALLOWED_TASKS else f"实际为 {task!r}，允许 {ALLOWED_TASKS}",
        )

        # max_length：BERT 类模型必须填实际值，其余必须为 null
        framework = metadata.get("framework")
        max_length = metadata.get("max_length")
        if framework == "transformers":
            valid = isinstance(max_length, int) and max_length > 0
            detail = "" if valid else f"transformers 模型必须填写正整数 max_length，实际为 {max_length!r}"
        else:
            valid = max_length is None
            detail = "" if valid else f"非 transformers 模型的 max_length 必须为 null，实际为 {max_length!r}"
        self.record("metadata.max_length 符合模型类型", "ok" if valid else "fail", detail)

        actual = compute_dataset_fingerprint(self.processed_dir)
        if actual is None:
            missing = [
                name for name in SPLIT_FILES if not (self.processed_dir / name).is_file()
            ]
            self.record(
                "数据集指纹与当前数据一致",
                "skip",
                f"data/processed 尚不完整，缺少 {missing}（需先完成 M2）",
            )
            return
        claimed = metadata.get("dataset_fingerprint")
        self.record(
            "数据集指纹与当前数据一致",
            "ok" if claimed == actual else "fail",
            ""
            if claimed == actual
            else f"metadata={claimed}，当前数据={actual}；该模型可能是用别的数据训练的",
        )

    def check_model_body(self, metadata: dict[str, Any]) -> None:
        """按 framework 确认模型本体文件存在。"""
        framework = metadata.get("framework")
        required = MODEL_BODY_REQUIREMENTS.get(framework)
        if required is None:
            self.record(
                "模型本体文件齐全",
                "fail",
                f"未知 framework {framework!r}，允许 {sorted(MODEL_BODY_REQUIREMENTS)}",
            )
            return
        missing = [name for name in required if not (self.model_dir / name).exists()]
        self.record(
            "模型本体文件齐全",
            "fail" if missing else "ok",
            f"framework={framework} 缺少：{missing}" if missing else f"framework={framework}",
        )

    def check_metrics_block(self, name: str, block: Any, categories: list[str]) -> list[str]:
        """校验 dev / test 指标块，返回问题列表。"""
        if block is None:
            if name == "dev":
                return ["dev 不得为 null（验证集指标必须先产出）"]
            return []

        problems = []
        if not isinstance(block, dict):
            return [f"{name} 不是对象"]

        missing = [key for key in METRICS_BLOCK_KEYS if key not in block]
        if missing:
            problems.append(f"{name} 缺少字段 {missing}")
            return problems

        per_class = block["per_class"]
        if not isinstance(per_class, dict) or set(per_class) != set(categories):
            problems.append(f"{name}.per_class 的类别与 categories 不一致")

        matrix = block["confusion_matrix"]
        size = len(categories)
        if not isinstance(matrix, list) or len(matrix) != size:
            problems.append(f"{name}.confusion_matrix 应为 {size}×{size} 二维数组")
        elif any(not isinstance(row, list) or len(row) != size for row in matrix):
            problems.append(f"{name}.confusion_matrix 每行长度应为 {size}")
        return problems

    def check_metrics(self, metrics: dict[str, Any]) -> None:
        """校验 metrics.json 的结构、类别顺序与测试集纪律。"""
        for key in ("model_name", "categories", "dev", "test", "cost"):
            if key not in metrics:
                self.record("metrics 必需字段齐全", "fail", f"缺少 {key}")
                return
        self.record("metrics 必需字段齐全", "ok")

        placeholders = find_placeholders(metrics)
        self.record(
            "metrics 无残留占位符",
            "fail" if placeholders else "ok",
            f"仍为占位符：{placeholders[:5]}" if placeholders else "",
        )

        categories = metrics["categories"]
        expected = self.expected_categories
        self.record(
            "metrics.categories 与 label_mapping 一致",
            "ok" if categories == expected else "fail",
            ""
            if categories == expected
            else f"metrics={categories}，label_mapping={expected}",
        )

        problems = self.check_metrics_block("dev", metrics["dev"], categories)
        problems.extend(self.check_metrics_block("test", metrics["test"], categories))
        self.record("dev / test 指标结构合法", "fail" if problems else "ok", "；".join(problems))

        cost = metrics["cost"]
        if not isinstance(cost, dict):
            self.record("cost 字段齐全", "fail", "cost 不是对象")
        else:
            missing = [key for key in COST_KEYS if key not in cost]
            self.record(
                "cost 字段齐全",
                "fail" if missing else "ok",
                f"缺少：{missing}" if missing else "",
            )

    def check_inference(self) -> None:
        """通过统一入口加载模型并执行真实推理。"""
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        try:
            inference_module = importlib.import_module("src.inference")
        except Exception as exc:
            self.record(
                "统一入口可加载并推理",
                "fail",
                f"src.inference 导入失败（{type(exc).__name__}: {exc}）",
            )
            return

        # 统一入口对两类任务是同一个（契约 v1.2 §2.5）：
        # 适配器是任务无关的，标签集取自模型目录自己的 label_mapping.json。
        entry_name = "load_predictor"
        load_predictor = getattr(inference_module, entry_name, None)
        if load_predictor is None:
            self.record(
                "统一入口可加载并推理",
                "skip",
                f"src.inference.{entry_name} 尚未实现（需先完成 M6）",
            )
            return
        if not callable(load_predictor):
            self.record(
                "统一入口可加载并推理",
                "fail",
                f"src.inference.{entry_name} 存在但不可调用",
            )
            return

        try:
            predictor = load_predictor(self.model_dir)
        except Exception as exc:
            self.record("统一入口可加载并推理", "fail", f"加载失败：{type(exc).__name__}: {exc}")
            return

        expected_categories = self.expected_categories
        if not expected_categories:
            self.record("统一入口可加载并推理", "skip", "缺少标签映射，无法判定标签集合")
            return

        try:
            probas = predictor.predict_proba(list(PROBE_TEXTS))
            labels = predictor.predict(list(PROBE_TEXTS))
        except Exception as exc:
            self.record("统一入口可加载并推理", "fail", f"推理失败：{type(exc).__name__}: {exc}")
            return

        problems = prediction_problems(probas, labels, expected_categories)

        self.record(
            "统一入口可加载并推理",
            "fail" if problems else "ok",
            "；".join(problems) if problems else f"已对 {len(PROBE_TEXTS)} 条探针文本完成推理",
        )


def discover_models(models_dir: Path) -> list[Path]:
    """列出 ``models/`` 下所有含 metadata.json 的子目录。"""
    if not models_dir.is_dir():
        return []
    return sorted(
        path for path in models_dir.iterdir() if path.is_dir() and (path / "metadata.json").is_file()
    )


def render(model_dir: Path, results: list[CheckResult]) -> tuple[str, int]:
    """渲染单个模型的校验结果，返回文本与失败项数。"""
    relative = model_dir.relative_to(PROJECT_ROOT).as_posix() if model_dir.is_relative_to(PROJECT_ROOT) else str(model_dir)
    lines = ["", "-" * 68, f"模型契约校验：{relative}", "-" * 68]
    markers = {"ok": "[OK]  ", "fail": "[FAIL]", "skip": "[SKIP]"}
    for result in results:
        lines.append(f"{markers[result.status]} {result.name}")
        if result.detail:
            lines.append(f"       {result.detail}")
    failed = sum(1 for result in results if result.status == "fail")
    skipped = sum(1 for result in results if result.status == "skip")
    lines.append(
        f"通过 {len(results) - failed - skipped} 项，失败 {failed} 项，跳过 {skipped} 项"
    )
    return "\n".join(lines), failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="校验模型产物是否符合模型契约")
    parser.add_argument("model_dirs", nargs="*", type=Path, help="模型目录，省略时校验 models/ 下全部")
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="处理后数据目录"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    args = parse_args(argv)
    model_dirs = args.model_dirs or discover_models(DEFAULT_MODELS_DIR)
    if not model_dirs:
        print("未找到任何模型目录（models/ 下没有含 metadata.json 的子目录）。")
        return 1

    total_failed = 0
    for model_dir in model_dirs:
        validator = ModelContractValidator(model_dir, args.processed_dir)
        text, failed = render(model_dir, validator.run())
        print(text)
        total_failed += failed

    print("=" * 68)
    if total_failed:
        print(f"结论：{total_failed} 项校验失败，相关模型不得用于部署或对比。")
        return 1
    print("结论：模型契约校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
