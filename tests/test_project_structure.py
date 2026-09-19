from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class ProjectInitializationTests(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        required_files = [
            "AGENTS.md",
            "README.md",
            "requirements.txt",
            "configs/base.yaml",
            "docs/team.md",
            "docs/development-plan.md",
            "scripts/download_data.py",
            "data/README.md",
        ]
        for relative_path in required_files:
            with self.subTest(path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).is_file())

    def test_required_directories_exist(self) -> None:
        required_directories = [
            "data/raw",
            "data/interim",
            "data/processed",
            "notebooks",
            "src/data",
            "src/features",
            "src/models/baseline",
            "src/models/fasttext",
            "src/models/bert",
            "src/models/optimization",
            "src/evaluation",
            "src/inference",
            "src/web",
            "reports/figures",
            "reports/meeting-notes",
            "reports/daily-reports",
        ]
        for relative_path in required_directories:
            with self.subTest(path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).is_dir())

    def test_raw_dataset_shape_when_downloaded(self) -> None:
        dataset_path = PROJECT_ROOT / "data/raw/online_shopping_10_cats.csv"
        if not dataset_path.exists():
            self.skipTest("Raw dataset has not been downloaded")

        with dataset_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.reader(file_obj)
            self.assertEqual(next(reader), ["cat", "label", "review"])
            self.assertEqual(sum(1 for _ in reader), 62_774)


if __name__ == "__main__":
    unittest.main()
