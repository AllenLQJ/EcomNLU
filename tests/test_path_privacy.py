"""产物和报告不得写入本机绝对路径。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.report_environment import render_environment_report  # noqa: E402
from src.data.config import OUTSIDE_PROJECT, public_path  # noqa: E402


class PublicPathTests(unittest.TestCase):
    def test_project_file_is_relative_posix(self) -> None:
        self.assertEqual(public_path(PROJECT_ROOT / "configs" / "base.yaml"), "configs/base.yaml")

    def test_relative_input_is_resolved_from_the_project_root(self) -> None:
        self.assertEqual(public_path("configs/base.yaml"), "configs/base.yaml")

    def test_path_outside_the_project_hides_the_location(self) -> None:
        outside = Path(tempfile.gettempdir()) / "not-in-repo"
        self.assertEqual(public_path(outside), OUTSIDE_PROJECT)
        self.assertNotIn(":", public_path(outside))


class EnvironmentReportPrivacyTests(unittest.TestCase):
    def test_report_uses_python_and_omits_local_paths(self) -> None:
        report = render_environment_report(
            [
                {
                    "name": "pandas",
                    "specifier": ">=2.1,<3.0",
                    "version": "2.3.3",
                    "status": "符合",
                }
            ],
            python_version="3.11.15",
            implementation="CPython",
        )
        self.assertIn("- 解释器：`python`", report)
        self.assertIn("python scripts/clean_data.py", report)
        self.assertIn("Python 版本：3.11.15", report)
        self.assertNotIn("Users", report)
        self.assertNotIn("python.exe", report)
        self.assertNotIn(":\\", report)
        self.assertNotIn("操作系统", report)
