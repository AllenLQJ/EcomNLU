"""记录实际运行环境，并逐项与 requirements.txt 的约束比对。

本脚本属于工程规范中「记录依赖版本与运行环境」这一条要求的落地工具：
它把「本机到底装了什么」写成报告，从而让环境漂移在答辩前就暴露出来，
而不是等到演示当天才发现某个包缺失或版本不符。

Usage:
    python scripts/report_environment.py

输入文件：
    - ``requirements.txt``（路径可由 ``--requirements`` 覆盖），逐行读取
      直接依赖及其版本约束。

输出文件：
    - ``reports/environment.md``（路径可由 ``--output`` 覆盖），包含解释器
      版本、依赖比对表、越界/缺失项的结论与待确认事项、复现命令。

关键设计理由：
    1. 对每个需求项比对**本机实际安装的版本**与声明约束：既报告完全缺失的
       包，也报告「已安装但超出约束范围」的版本越界，这两类问题都需要在
       同步依赖时由维护 ``requirements.txt`` 的成员确认，否则不同成员会
       在不可比的环境上跑出不可比的指标；
    2. ``EXTRA_PACKAGES`` 用于把非直接依赖但会影响结果复现的传递依赖也一并
       记录，避免「requirements 里没有就当它不存在」的盲区；
    3. 报告只记录 Python 版本、实现和依赖比对。解释器写成 ``python``，
       复现命令不包含绝对路径、用户名、盘符或主机名。
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import ensure_directories, project_path, public_path  # noqa: E402


# 必须接受 ``uvicorn[standard]>=0.29`` 这类 extras 写法，
# 否则该需求会被静默跳过而不是出现在报告里。
REQUIREMENT_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?P<specifier>[<>=!~].*)?$"
)
# 这些包属于传递依赖而非直接需求，但对复现整条流水线同样重要，故一并记录。
# 之所以列出 pillow 与 wordcloud，是因为发生过一次真实事故：安装 wordcloud 时
# conda 顺带把 Pillow 与 matplotlib 换成了别的构建版本，导致 matplotlib 保存图片
# 时原生崩溃；而当时的检查只覆盖 requirements.txt 里的直接依赖，完全没有预警。
# pyarrow 是 pandas 3.x 字符串列的默认后端，其绑定的 RE2 是 .str 正则的实际执行引擎，
# 正则语义（\w / \s 只按 ASCII 解释、不接受 \u 转义）由它决定，因此必须记录。
EXTRA_PACKAGES = ("pyarrow", "pillow", "wordcloud")


def read_requirements(path: Path) -> list[tuple[str, str]]:
    """从 requirements.txt 中解析出 (包名, 版本约束) 列表，并忽略注释行。

    参数：
        path：requirements.txt 的绝对路径。

    返回：
        ``(name, specifier)`` 二元组列表；没有版本约束的条目其 specifier
        为空字符串。行内 ``#`` 之后的内容一律视为注释丢弃。
    """
    requirements: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = REQUIREMENT_PATTERN.match(line)
        if match:
            requirements.append((match.group("name"), (match.group("specifier") or "").strip()))
    return requirements


def installed_version(name: str) -> str | None:
    """返回某个发行包在本机实际安装的版本，未安装时返回 None。"""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_specifier(version: str, specifier: str) -> bool | None:
    """判断实际版本是否满足约束。

    返回：
        - ``True``：满足约束（或本就没有约束）；
        - ``False``：已安装但超出约束范围，属于环境漂移；
        - ``None``：无法校验（缺少 ``packaging`` 库，或约束字符串无法解析），
          此时报告中标记为「无法校验」而不是误报为符合。
    """
    if not specifier:
        return True
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        return None
    try:
        return Version(version) in Requirement(f"pkg{specifier}").specifier
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """解析命令行参数：requirements 文件路径与报告输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--output", default="reports/environment.md")
    return parser.parse_args()


def render_environment_report(
    rows: list[dict[str, object]],
    *,
    python_version: str,
    implementation: str,
) -> str:
    """生成环境报告正文。

    只写入解释器版本和依赖比对。不写入解释器绝对路径、用户目录、盘符或主机名。
    """
    missing = [row["name"] for row in rows if row["version"] == "-"]
    violations = [row for row in rows if str(row["status"]).startswith("**")]
    if python_version.startswith("3.11"):
        version_note = (
            f"> 说明：`README.md` 建议使用 Python 3.11。当前解释器为 Python {python_version}，"
            "主版本与建议一致。"
        )
    else:
        version_note = (
            f"> 说明：`README.md` 建议使用 Python 3.11。当前解释器为 Python {python_version}，"
            "主版本与建议不一致，引用指标时必须同时注明该版本。"
        )

    lines: list[str] = [
        "# 运行环境与依赖记录",
        "",
        "本文件由 `scripts/report_environment.py` 生成，记录实际使用的解释器与依赖版本，",
        "并与 `requirements.txt` 的约束逐项比对，用于发现环境漂移。",
        "",
        "## 1. 解释器",
        "",
        f"- Python 版本：{python_version}",
        "- 解释器：`python`",
        f"- 实现：{implementation}",
        "",
        version_note,
        "",
        "## 2. 依赖版本比对",
        "",
        "| 包 | requirements 约束 | 实际版本 | 状态 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['specifier']}` | `{row['version']}` | {row['status']} |"
        )

    lines.extend(["", "## 3. 结论与待确认事项", ""])
    if violations:
        lines.append("以下依赖的实际版本超出了 `requirements.txt` 的约束范围：")
        lines.append("")
        for row in violations:
            lines.append(
                f"- `{row['name']}`：约束 `{row['specifier']}`，实际 `{row['version']}`"
            )
        lines.append("")
        lines.append(
            "需要在下次同步时由维护 `requirements.txt` 的成员确认："
            "是把约束放宽到与实际环境一致，还是把环境降级到约束范围内。"
            "在确认之前，任何模型结果都必须连同本节记录的实际版本一起报告。"
        )
    else:
        lines.append("所有已安装依赖均符合 `requirements.txt` 的约束范围。")

    if missing:
        lines.extend(
            [
                "",
                f"以下依赖当前环境中**未安装**：{', '.join(f'`{name}`' for name in missing)}。",
                "",
                "影响说明：",
                "- `seaborn`：本项目 EDA 图表统一使用 `matplotlib` 实现，未依赖 seaborn，因此不影响 M2 流程。",
                "- `pytest`：本项目的测试全部基于标准库 `unittest`，使用 "
                "`python -m unittest discover -s tests` 运行，因此不影响测试执行。",
                "- 其余缺失项若属于后续阶段（FastText / BERT / Web）所需，必须在对应阶段开始前解决。",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. 复现命令",
            "",
            "```powershell",
            "# 在项目根目录、使用已确认的解释器运行全部数据流程",
            "python scripts/clean_data.py",
            "python scripts/split_data.py",
            "python scripts/run_eda.py",
            "python scripts/report_environment.py",
            "python -m unittest discover -s tests -v",
            "",
            "# 原始数据质量检查由独立脚本承担",
            "python src/data/check_raw.py",
            "```",
            "",
            "> 复现命令使用 `python`，不记录解释器的本机路径。脚本以项目根目录解析路径。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """比对依赖并写出环境报告。

    流程：读取 requirements.txt，对每个需求项（外加 ``EXTRA_PACKAGES``）
    查询本机实际安装版本，按「未安装 / 已安装但非直接依赖 / 无法校验 / 符合 /
    超出范围」归类，然后把解释器版本、比对表、缺失与越界结论、复现命令
    写入 ``reports/environment.md``。报告不记录解释器绝对路径。

    返回：
        进程退出码，成功为 0。
    """
    args = parse_args()
    requirements_path = project_path(args.requirements)
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements.txt not found: {public_path(requirements_path)}")

    requirements = read_requirements(requirements_path)
    rows: list[dict[str, object]] = []
    for name, specifier in requirements + [(name, "") for name in EXTRA_PACKAGES]:
        version = installed_version(name)
        if version is None:
            status = "未安装"
        elif not specifier:
            status = "已安装（非直接依赖）"
        else:
            satisfied = check_specifier(version, specifier)
            if satisfied is None:
                status = "无法校验"
            elif satisfied:
                status = "符合"
            else:
                status = "**超出 requirements 范围**"
        rows.append(
            {
                "name": name,
                "specifier": specifier or "-",
                "version": version or "-",
                "status": status,
            }
        )

    report = render_environment_report(
        rows,
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
    )
    report_path = project_path(args.output)
    ensure_directories(report_path.parent)
    report_path.write_text(report, encoding="utf-8")

    print(f"[env] python: {platform.python_version()}")
    for row in rows:
        print(f"[env]   {row['name']:<18} {row['version']:<14} {row['status']}")
    print(f"[env] report written: {public_path(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
