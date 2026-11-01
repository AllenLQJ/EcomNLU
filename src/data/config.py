"""数据流水线共用的配置加载工具。

模块职责：
    读取并校验 YAML 配置文件，并把项目内的相对路径统一解析到项目根目录，
    供数据检查、数据清洗、数据划分等脚本复用。

主要输入与输出：
    输入：配置文件路径（默认 ``configs/base.yaml``，可用参数覆盖）。
    输出：解析后的配置字典；以及以项目根目录为基准解析出的绝对路径对象。

关键设计理由与约束：
    所有路径都相对于项目根目录解析，禁止硬编码个人电脑上的绝对路径，
    这样同一份代码在不同机器、不同工作目录下都能正常运行。
    配置缺少必需分区时立即抛出异常，避免后续流程静默使用不完整的配置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

# 项目外路径的公开写法。产物、报告和日志不得回退成绝对路径。
OUTSIDE_PROJECT = "项目外"

REQUIRED_SECTIONS = ("project", "data", "task", "evaluation")


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """加载并校验 YAML 配置文件。

    Args:
        config_path: 可选的配置文件位置覆盖参数；传 None 时使用默认配置路径。

    Returns:
        解析后的配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 文件内容不是映射（字典）结构时抛出。
        KeyError: 缺少任一必需配置分区时抛出。
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {public_path(path)}")

    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise KeyError(f"Missing required config section '{section}' in {path}")
    return config


def public_path(path: str | Path, root: Path | None = None) -> str:
    """把路径写成相对项目根的 POSIX 形式，供写入产物和报告。

    相对路径按项目根解析，不按进程当前工作目录解析，因此同一条记录
    不随启动位置变化。项目根以外的路径一律写成「项目外」，不保留盘符、
    用户目录或绝对路径。

    Args:
        path: 待公开的路径，可以是相对路径或绝对路径。
        root: 可选的项目根覆盖；测试替换根目录时传入。

    Returns:
        相对项目根的 POSIX 路径，或 ``OUTSIDE_PROJECT``。
    """
    base = (root if root is not None else PROJECT_ROOT).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return OUTSIDE_PROJECT


def project_path(relative_path: str | Path) -> Path:
    """把项目内的相对路径解析到项目根目录，已是绝对路径时原样返回。

    Args:
        relative_path: 项目内的相对路径；如传入绝对路径则直接返回。

    Returns:
        解析后的绝对路径对象。
    """
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories(*paths: Path | str) -> None:
    """创建传入的每个目录（含父级目录），目录已存在时不做任何事。

    Args:
        *paths: 需要确保存在的目录路径，可以是相对项目根目录的路径。

    Returns:
        None。该函数只产生创建目录的副作用。
    """
    for path in paths:
        project_path(path).mkdir(parents=True, exist_ok=True)
