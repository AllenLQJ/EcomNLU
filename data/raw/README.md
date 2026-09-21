# Raw Data

本目录存放只读的原始数据，文件由 `python scripts/download_data.py` 下载并校验 SHA-256，不提交 Git，禁止手工修改。

本 README 的作用是让该目录在任何新克隆中都存在：`data/raw/` 下的数据文件被 `.gitignore` 排除，若此文件也不存在，目录就不会进入版本库，`tests/test_project_structure.py` 的目录检查会在新克隆上失败。

数据来源、字段、质量情况和许可提示见 `data/README.md`；原始数据质量检查报告见 `reports/data-quality/`。
