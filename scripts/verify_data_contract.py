"""校验 ``data/processed/`` 是否符合 ``docs/contracts.md`` 中的数据契约。

只读校验，不修改任何文件。存在任何一项失败时以非零退出码结束。

本脚本会**独立重算**泄漏检查，而不是复述 ``split_manifest.json`` 里的声明值——
否则校验就只是在重复被校验对象的说法。

用法::

    python scripts/verify_data_contract.py
    python scripts/verify_data_contract.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REQUIRED_FILES = ("train.csv", "dev.csv", "test.csv", "label_mapping.json", "split_manifest.json")
SPLITS = ("train", "dev", "test")
EXPECTED_COLUMNS = ["sample_id", "review", "cat", "label"]
EXPECTED_SOURCE_ROWS = 62_774
VALID_LABELS = {"0", "1"}
SPLIT_PAIRS = (("train", "dev"), ("train", "test"), ("dev", "test"))
EXAMPLE_LIMIT = 3
EXAMPLE_MAX_CHARS = 60


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


def truncate(text: str, limit: int = EXAMPLE_MAX_CHARS) -> str:
    """截断长文本用于报告展示。"""
    return text if len(text) <= limit else text[:limit] + "…"


def summarize(mask: pd.Series, series: pd.Series, limit: int = EXAMPLE_LIMIT) -> str:
    """把异常行概括为「数量 + 示例」。"""
    count = int(mask.sum())
    if count == 0:
        return ""
    examples = [truncate(str(value)) for value in series[mask].head(limit)]
    return f"{count} 条，例如：{examples}"


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


class DataContractValidator:
    """按数据契约逐项校验 ``data/processed/``。"""

    def __init__(self, processed_dir: Path, config: dict[str, Any]) -> None:
        self.processed_dir = processed_dir
        self.config = config
        self.expected_categories: list[str] = config["task"]["categories"]
        self.checks: list[CheckResult] = []

    # ---------------------------------------------------------------- 基础设施

    def record(self, name: str, status: str, detail: str = "") -> bool:
        """记录一项校验结果。"""
        self.checks.append(CheckResult(name, status, detail))
        return status == "ok"

    def read_csv(self, path: Path) -> pd.DataFrame:
        """按字符串读取 CSV，保留空值原貌。"""
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")

    # ---------------------------------------------------------------- 校验流程

    def run(self) -> list[CheckResult]:
        """执行全部校验。"""
        missing = [name for name in REQUIRED_FILES if not (self.processed_dir / name).is_file()]
        if not self.record(
            "必需文件齐全",
            "fail" if missing else "ok",
            f"缺少：{missing}" if missing else f"{len(REQUIRED_FILES)} 个文件均存在",
        ):
            return self.checks

        label_mapping = self.load_json("label_mapping.json")
        manifest = self.load_json("split_manifest.json")
        if label_mapping is None or manifest is None:
            return self.checks

        self.check_label_mapping(label_mapping)
        self.check_manifest_placeholders(manifest)
        self.check_manifest_contract(manifest)
        self.check_source_hash(manifest)

        frames = {}
        for split in SPLITS:
            frame = self.read_split(split)
            if frame is None:
                return self.checks
            frames[split] = frame

        self.check_file_hashes(manifest, frames)
        self.check_columns(frames)
        self.check_sample_ids(frames)
        self.check_values(frames)
        self.check_traceability(frames)
        self.check_leakage(frames, manifest)
        self.check_stratification(frames)
        return self.checks

    # ---------------------------------------------------------------- 各项校验

    def load_json(self, filename: str) -> dict[str, Any] | None:
        """读取 JSON 文件，失败时记录并返回 ``None``。"""
        path = self.processed_dir / filename
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.record(f"{filename} 可解析", "fail", f"{type(exc).__name__}: {exc}")
            return None
        if not isinstance(data, dict):
            self.record(f"{filename} 可解析", "fail", "JSON 顶层必须是对象")
            return None
        self.record(f"{filename} 可解析", "ok")
        return data

    def read_split(self, split: str) -> pd.DataFrame | None:
        """读取单个划分文件，并检查 BOM 与解析错误。"""
        path = self.processed_dir / f"{split}.csv"
        if path.read_bytes()[:3] == b"\xef\xbb\xbf":
            self.record(f"{split}.csv 无 BOM", "fail", "文件带有 UTF-8 BOM")
            return None
        try:
            frame = self.read_csv(path)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            self.record(f"{split}.csv 可读取", "fail", f"{type(exc).__name__}: {exc}")
            return None
        self.record(f"{split}.csv 无 BOM", "ok")
        return frame

    def check_label_mapping(self, label_mapping: dict[str, Any]) -> None:
        """校验标签映射与 ``configs/base.yaml`` 一致。"""
        categories = label_mapping.get("categories")
        if categories != self.expected_categories:
            self.record(
                "label_mapping 的 categories 与配置一致",
                "fail",
                f"期望 {self.expected_categories}，实际 {categories}",
            )
            return
        self.record("label_mapping 的 categories 与配置一致", "ok")

        label2id = label_mapping.get("label2id", {})
        id2label = label_mapping.get("id2label", {})
        expected_label2id = {name: index for index, name in enumerate(self.expected_categories)}
        expected_id2label = {str(index): name for index, name in enumerate(self.expected_categories)}

        problems = []
        if label2id != expected_label2id:
            problems.append(f"label2id 不符：{label2id}")
        if id2label != expected_id2label:
            problems.append(f"id2label 不符：{id2label}")
        self.record(
            "label2id / id2label 与 categories 顺序一致",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_manifest_placeholders(self, manifest: dict[str, Any]) -> None:
        """确认 ``split_manifest.json`` 中没有残留占位符。"""
        placeholders = find_placeholders(manifest)
        self.record(
            "split_manifest 无残留占位符",
            "fail" if placeholders else "ok",
            f"仍为占位符：{placeholders[:5]}" if placeholders else "",
        )

    def check_manifest_contract(self, manifest: dict[str, Any]) -> None:
        """校验 manifest 中不能仅靠文件哈希证明的固定规则。"""
        problems: list[str] = []

        created_at = manifest.get("created_at")
        if not isinstance(created_at, str):
            problems.append("created_at 必须是 ISO 8601 字符串")
        else:
            try:
                parsed = datetime.fromisoformat(created_at)
                if parsed.tzinfo is None:
                    problems.append("created_at 必须带时区")
            except ValueError:
                problems.append("created_at 不是合法 ISO 8601 时间")

        expected_seed = self.config["project"]["random_seed"]
        if manifest.get("random_seed") != expected_seed:
            problems.append(
                f"random_seed 应为 {expected_seed}，实际为 {manifest.get('random_seed')!r}"
            )

        source = manifest.get("source")
        expected_source_path = self.config["data"]["raw_path"]
        if not isinstance(source, dict) or source.get("path") != expected_source_path:
            actual = source.get("path") if isinstance(source, dict) else source
            problems.append(f"source.path 应为 {expected_source_path!r}，实际为 {actual!r}")

        cleaning = manifest.get("cleaning")
        required_rules = (
            "empty_review",
            "duplicate_review",
            "control_chars",
            "short_text_threshold",
        )
        if not isinstance(cleaning, dict):
            problems.append("cleaning 必须是对象")
        else:
            if cleaning.get("rows_before") != EXPECTED_SOURCE_ROWS:
                problems.append(
                    f"cleaning.rows_before 应为 {EXPECTED_SOURCE_ROWS}，"
                    f"实际为 {cleaning.get('rows_before')!r}"
                )
            rules = cleaning.get("rules")
            if not isinstance(rules, dict):
                problems.append("cleaning.rules 必须是对象")
            else:
                missing_rules = [
                    key for key in required_rules if key not in rules or rules[key] in (None, "")
                ]
                if missing_rules:
                    problems.append(f"cleaning.rules 缺少或留空：{missing_rules}")

        split = manifest.get("split")
        expected_ratios = {
            "train": self.config["data"]["train_ratio"],
            "dev": self.config["data"]["dev_ratio"],
            "test": self.config["data"]["test_ratio"],
        }
        if not isinstance(split, dict):
            problems.append("split 必须是对象")
        else:
            if split.get("strategy") != "stratified_by_cat":
                problems.append("split.strategy 必须为 'stratified_by_cat'")
            if split.get("ratios") != expected_ratios:
                problems.append(
                    f"split.ratios 应为 {expected_ratios}，实际为 {split.get('ratios')!r}"
                )
            if split.get("dedup_key") not in {"raw_text", "normalized_text"}:
                problems.append("split.dedup_key 只能是 raw_text 或 normalized_text")

        leakage = manifest.get("leakage_check")
        if not isinstance(leakage, dict) or leakage.get("based_on") != "normalized_text":
            problems.append("leakage_check.based_on 必须为 'normalized_text'")

        self.record(
            "split_manifest 固定规则合法",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_source_hash(self, manifest: dict[str, Any]) -> None:
        """确认 manifest 记录的原始数据哈希与实际文件一致。"""
        raw_path = PROJECT_ROOT / self.config["data"]["raw_path"]
        if not raw_path.is_file():
            self.record(
                "原始数据哈希一致", "skip", f"原始数据不存在：{raw_path}，跳过该检查"
            )
            return
        actual = sha256_of(raw_path)
        claimed = manifest.get("source", {}).get("sha256")
        self.record(
            "原始数据哈希一致",
            "ok" if actual == claimed else "fail",
            "" if actual == claimed else f"manifest={claimed}，实际={actual}",
        )

    def check_file_hashes(
        self, manifest: dict[str, Any], frames: dict[str, pd.DataFrame]
    ) -> None:
        """确认 manifest 记录的三个划分文件哈希与行数与实际一致。

        行数取自 pandas 解析结果而非文件物理行数：字段内含换行时，一条 CSV 记录会
        占用多个物理行，数行数会得到错误结果。
        """
        files = manifest.get("files", {})
        problems = []
        for split in SPLITS:
            filename = f"{split}.csv"
            entry = files.get(filename)
            if not isinstance(entry, dict):
                problems.append(f"{filename} 未记录")
                continue
            actual_hash = sha256_of(self.processed_dir / filename)
            if entry.get("sha256") != actual_hash:
                problems.append(f"{filename} sha256 不符")
            actual_rows = int(len(frames[split]))
            if entry.get("rows") != actual_rows:
                problems.append(f"{filename} rows 记录 {entry.get('rows')}，实际 {actual_rows}")
            if manifest.get("counts", {}).get(split) != actual_rows:
                problems.append(
                    f"counts.{split} 记录 {manifest.get('counts', {}).get(split)}，实际 {actual_rows}"
                )

        total = sum(int(len(frames[split])) for split in SPLITS)
        declared_total = manifest.get("cleaning", {}).get("rows_after")
        if declared_total != total:
            problems.append(f"cleaning.rows_after 记录 {declared_total}，三个划分合计 {total}")
        self.record(
            "manifest 的文件哈希与行数一致",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_columns(self, frames: dict[str, pd.DataFrame]) -> None:
        """校验列名与顺序。"""
        problems = [
            f"{split}.csv 列名为 {list(frame.columns)}"
            for split, frame in frames.items()
            if list(frame.columns) != EXPECTED_COLUMNS
        ]
        self.record(
            "列名与顺序为 sample_id,review,cat,label",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_sample_ids(self, frames: dict[str, pd.DataFrame]) -> None:
        """校验 sample_id 为整数、划分内唯一、且跨划分不重复。"""
        if any(list(frame.columns) != EXPECTED_COLUMNS for frame in frames.values()):
            self.record("sample_id 合法且不重复", "skip", "列名不符，跳过")
            return

        all_ids = []
        problems = []
        for split, frame in frames.items():
            ids = frame["sample_id"]
            bad = ~ids.str.fullmatch(r"\d+")
            if bad.any():
                problems.append(f"{split} 有非整数 sample_id：{summarize(bad, ids)}")
            if ids.duplicated().any():
                problems.append(f"{split} 内部 sample_id 重复 {int(ids.duplicated().sum())} 条")
            all_ids.extend(ids.tolist())

        duplicated_across = len(all_ids) - len(set(all_ids))
        if duplicated_across:
            problems.append(f"跨划分重复 sample_id {duplicated_across} 条")

        self.record(
            "sample_id 合法且不重复",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_values(self, frames: dict[str, pd.DataFrame]) -> None:
        """校验 review、cat、label 的取值。"""
        if any(list(frame.columns) != EXPECTED_COLUMNS for frame in frames.values()):
            self.record("review / cat / label 取值合法", "skip", "列名不符，跳过")
            return

        problems = []
        for split, frame in frames.items():
            review = frame["review"]
            empty = review.str.strip().eq("")
            if empty.any():
                problems.append(f"{split} 有空白 review {int(empty.sum())} 条")

            untrimmed = review.ne(review.str.strip())
            if untrimmed.any():
                problems.append(f"{split} 有首尾空白 {summarize(untrimmed, review)}")

            cat = frame["cat"]
            bad_cat = ~cat.isin(self.expected_categories)
            if bad_cat.any():
                problems.append(f"{split} 有非法类别 {summarize(bad_cat, cat)}")

            label = frame["label"]
            bad_label = ~label.isin(VALID_LABELS)
            if bad_label.any():
                problems.append(f"{split} 有非法情感标签 {summarize(bad_label, label)}")

        self.record(
            "review / cat / label 取值合法",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_traceability(self, frames: dict[str, pd.DataFrame]) -> None:
        """用原始数据核对 sample_id 指向的记录，确认没有错位。"""
        raw_path = PROJECT_ROOT / self.config["data"]["raw_path"]
        if not raw_path.is_file():
            self.record("sample_id 可追溯回原始数据", "skip", f"原始数据不存在：{raw_path}")
            return
        if any(list(frame.columns) != EXPECTED_COLUMNS for frame in frames.values()):
            self.record("sample_id 可追溯回原始数据", "skip", "列名不符，跳过")
            return

        raw = self.read_csv(raw_path).reset_index(drop=True)
        problems = []
        for split, frame in frames.items():
            ids = pd.to_numeric(frame["sample_id"], errors="coerce")
            out_of_range = ids.isna() | (ids < 0) | (ids >= len(raw))
            if out_of_range.any():
                problems.append(f"{split} 有 {int(out_of_range.sum())} 条 sample_id 超出原始数据范围")
                continue
            index = ids.astype(int)
            cat_mismatch = raw["cat"].iloc[index].to_numpy() != frame["cat"].to_numpy()
            label_mismatch = raw["label"].iloc[index].to_numpy() != frame["label"].to_numpy()
            if cat_mismatch.any():
                problems.append(f"{split} 有 {int(cat_mismatch.sum())} 条 cat 与原始数据不符")
            if label_mismatch.any():
                problems.append(f"{split} 有 {int(label_mismatch.sum())} 条 label 与原始数据不符")

        self.record(
            "sample_id 可追溯回原始数据",
            "fail" if problems else "ok",
            "；".join(problems),
        )

    def check_leakage(
        self, frames: dict[str, pd.DataFrame], manifest: dict[str, Any]
    ) -> None:
        """独立重算跨划分的文本重复，并与 manifest 声明值比对。"""
        if any(list(frame.columns) != EXPECTED_COLUMNS for frame in frames.values()):
            self.record("跨划分无文本泄漏", "skip", "列名不符，跳过")
            return

        normalized = {
            split: frame["review"].str.strip().str.replace(r"\s+", " ", regex=True)
            for split, frame in frames.items()
        }
        recomputed: dict[str, int] = {}
        for left, right in SPLIT_PAIRS:
            left_values = set(normalized[left][normalized[left].ne("")])
            right_values = set(normalized[right][normalized[right].ne("")])
            recomputed[f"{left}_{right}_overlap"] = len(left_values & right_values)

        leaked = {key: value for key, value in recomputed.items() if value}
        self.record(
            "跨划分无文本泄漏",
            "fail" if leaked else "ok",
            f"实测重复：{leaked}" if leaked else "三个划分两两无重复文本",
        )

        claimed = manifest.get("leakage_check", {})
        mismatched = {
            key: {"manifest": claimed.get(key), "actual": value}
            for key, value in recomputed.items()
            if claimed.get(key) != value
        }
        self.record(
            "manifest 的泄漏声明与实测一致",
            "fail" if mismatched else "ok",
            f"不一致：{mismatched}" if mismatched else "",
        )

    def check_stratification(self, frames: dict[str, pd.DataFrame]) -> None:
        """按类别反算各划分比例，防止 manifest 虚报分层抽样。"""
        if any(list(frame.columns) != EXPECTED_COLUMNS for frame in frames.values()):
            self.record("实测类别分层比例合法", "skip", "列名不符，跳过")
            return

        expected_ratios = {
            "train": float(self.config["data"]["train_ratio"]),
            "dev": float(self.config["data"]["dev_ratio"]),
            "test": float(self.config["data"]["test_ratio"]),
        }
        totals = pd.concat([frame["cat"] for frame in frames.values()]).value_counts()
        problems: list[str] = []
        for category in self.expected_categories:
            total = int(totals.get(category, 0))
            if total == 0:
                problems.append(f"{category}无样本")
                continue
            for split, ratio in expected_ratios.items():
                actual = int((frames[split]["cat"] == category).sum())
                if abs(actual - total * ratio) > 1.0:
                    problems.append(
                        f"{category}/{split}={actual}，期望约 {total * ratio:.1f}"
                    )
        self.record(
            "实测类别分层比例合法",
            "fail" if problems else "ok",
            "；".join(problems[:10]),
        )


def render(results: list[CheckResult]) -> str:
    """渲染校验结果。"""
    lines = [
        "",
        "=" * 68,
        "数据契约校验：data/processed",
        "=" * 68,
    ]
    markers = {"ok": "[OK]  ", "fail": "[FAIL]", "skip": "[SKIP]"}
    for result in results:
        lines.append(f"{markers[result.status]} {result.name}")
        if result.detail:
            lines.append(f"       {result.detail}")

    failed = sum(1 for result in results if result.status == "fail")
    skipped = sum(1 for result in results if result.status == "skip")
    lines.append("-" * 68)
    lines.append(
        f"通过 {len(results) - failed - skipped} 项，失败 {failed} 项，跳过 {skipped} 项"
    )
    lines.append("=" * 68)
    if failed:
        lines.append("结论：数据契约校验未通过，该数据不得用于训练。")
    else:
        lines.append("结论：数据契约校验通过。")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="校验 data/processed 是否符合数据契约")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="处理后数据目录"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"配置文件不存在：{args.config}", file=sys.stderr)
        return 2
    with args.config.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    validator = DataContractValidator(args.processed_dir, config)
    results = validator.run()
    print(render(results))
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
