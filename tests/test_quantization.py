"""BERT 动态量化产物的元数据规则测试。

只测**不需要加载真实模型**的部分：用极小的假模型/假分词器驱动
``save_quantized_model`` 的落盘逻辑，验证产出的 ``metadata.json`` 符合契约与
AGENTS.md 的路径约束。真实量化与推理由 ``scripts/quantize_bert.py`` 的运行记录覆盖。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.optimization.quantization import (  # noqa: E402
    QUANTIZED_WEIGHTS_NAME,
    _as_project_relative,
    save_quantized_model,
)


class FakeConfig:
    """只为 ``save_pretrained`` 落一个 config.json。"""

    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text('{"model_type": "fake"}', encoding="utf-8")


class FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeModel:
    """具备 ``config`` 与 ``state_dict`` 的最小对象，避免加载真实 BERT。"""

    def __init__(self) -> None:
        import torch

        self.config = FakeConfig()
        self._module = torch.nn.Linear(2, 2)

    def state_dict(self) -> dict:
        return self._module.state_dict()


def make_source(root: Path, model_name: str = "bert_base_chinese") -> Path:
    """构造一个最小的源模型目录。"""
    source = root / model_name
    source.mkdir(parents=True, exist_ok=True)
    (source / "label_mapping.json").write_text(
        json.dumps({"categories": ["甲", "乙"]}, ensure_ascii=False), encoding="utf-8"
    )
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "task": "category_classification",
                "framework": "transformers",
                "framework_version": "5.16.1",
                "created_at": "2026-09-22T10:00:00+08:00",
                "random_seed": 42,
                "dataset_fingerprint": "a" * 64,
                "train_rows": 1,
                "hyperparameters": {},
                "max_length": 128,
                "device": "cpu",
                "notes": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return source


class ProjectRelativePathTests(unittest.TestCase):
    """``_as_project_relative``：产物里不得出现个人电脑绝对路径（AGENTS.md §7.1）。"""

    def test_project_path_becomes_relative_posix(self) -> None:
        self.assertEqual(
            _as_project_relative(PROJECT_ROOT / "models" / "bert_base_chinese"),
            "models/bert_base_chinese",
        )

    def test_relative_input_is_resolved_against_cwd(self) -> None:
        result = _as_project_relative(Path("configs/base.yaml"))
        self.assertEqual(result, "configs/base.yaml")

    def test_path_outside_the_project_is_left_alone(self) -> None:
        """项目外的路径保留原值，不静默编造一个错误的相对路径。"""
        outside = Path(tempfile.gettempdir()) / "somewhere"
        self.assertEqual(_as_project_relative(outside), str(outside))


class SaveQuantizedModelMetadataTests(unittest.TestCase):
    """量化产物的 metadata 必须与目录名一致，且不含绝对路径。"""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.source = make_source(root)
        self.target = root / "bert_base_chinese_quantized"

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _save(self) -> dict:
        return save_quantized_model(
            FakeModel(), FakeTokenizer(), self.source, self.target
        )

    def test_model_name_follows_the_target_directory(self) -> None:
        """优化产物目录名是 ``<源模型>_<优化方式>``，model_name 必须跟着改。

        否则元数据与目录名不一致，对比报告和 Web 层会读出两个不同的模型名。
        """
        self._save()
        metadata = json.loads((self.target / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_name"], self.target.name)
        self.assertNotEqual(metadata["model_name"], self.source.name)

    def test_source_model_dir_is_recorded_relative_to_the_project_root(self) -> None:
        """源模型在项目内时，必须写成相对项目根的路径（AGENTS.md §7.1）。"""
        root = Path(self._temp.name)
        with patch("src.models.optimization.quantization.PROJECT_ROOT", root):
            self._save()
        metadata = json.loads((self.target / "metadata.json").read_text(encoding="utf-8"))
        source_dir = metadata["quantization"]["source_model_dir"]
        self.assertFalse(Path(source_dir).is_absolute(), source_dir)
        self.assertEqual(source_dir, self.source.name)

    def test_source_outside_the_project_keeps_its_original_path(self) -> None:
        """项目外的源模型保留原值：不静默编造一条错误的相对路径。"""
        self._save()
        metadata = json.loads((self.target / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["quantization"]["source_model_dir"], str(self.source))

    def test_quantization_block_records_the_weights_file(self) -> None:
        self._save()
        metadata = json.loads((self.target / "metadata.json").read_text(encoding="utf-8"))
        quantization = metadata["quantization"]
        self.assertEqual(quantization["dtype"], "qint8")
        self.assertEqual(quantization["target_device"], "cpu")
        self.assertEqual(quantization["weights_file"], f"model/{QUANTIZED_WEIGHTS_NAME}")
        self.assertTrue((self.target / quantization["weights_file"]).is_file())

    def test_label_mapping_is_copied_from_the_source(self) -> None:
        self._save()
        self.assertEqual(
            (self.target / "label_mapping.json").read_bytes(),
            (self.source / "label_mapping.json").read_bytes(),
        )

    def test_existing_metrics_json_is_preserved(self) -> None:
        """保存流程绝不整目录重建：已有 metrics.json 是耗时实测结果，不可丢失。"""
        self.target.mkdir(parents=True, exist_ok=True)
        metrics = self.target / "metrics.json"
        metrics.write_text('{"kept": true}', encoding="utf-8")
        self._save()
        self.assertTrue(metrics.is_file())
        self.assertEqual(json.loads(metrics.read_text(encoding="utf-8")), {"kept": True})


if __name__ == "__main__":
    unittest.main()
