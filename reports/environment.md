# 运行环境与依赖记录

本文件由 `scripts/report_environment.py` 生成，记录实际使用的解释器与依赖版本，
并与 `requirements.txt` 的约束逐项比对，用于发现环境漂移。

## 1. 解释器与平台

- Python 版本：3.11.15
- 解释器：`python`
- 实现：CPython
- 操作系统：Windows 10（AMD64）

> 说明：`README.md` 建议使用 Python 3.11。当前实际使用的环境是 conda 环境 `AI`，
> 其解释器为 Python 3.11.15，与建议一致，因此该环境在解释器版本上没有偏差。

## 2. 依赖版本比对

| 包 | requirements 约束 | 实际版本 | 状态 |
| --- | --- | --- | --- |
| `pandas` | `>=2.1,<3.0` | `3.0.5` | **超出 requirements 范围** |
| `numpy` | `>=1.26,<3.0` | `2.4.6` | 符合 |
| `PyYAML` | `>=6.0,<7.0` | `6.0.3` | 符合 |
| `matplotlib` | `>=3.8,<4.0` | `3.11.2` | 符合 |
| `seaborn` | `>=0.13,<1.0` | `-` | 未安装 |
| `jieba` | `>=0.42,<1.0` | `0.42.1` | 符合 |
| `scikit-learn` | `>=1.4,<2.0` | `1.9.1` | 符合 |
| `joblib` | `>=1.3,<2.0` | `1.6.0` | 符合 |
| `fasttext-wheel` | `>=0.9.2,<1.0` | `0.9.2` | 符合 |
| `torch` | `>=2.2` | `2.6.0+cu124` | 符合 |
| `transformers` | `>=4.40,<6.0` | `5.17.0` | 符合 |
| `tqdm` | `>=4.66,<5.0` | `4.70.1` | 符合 |
| `fastapi` | `>=0.110,<1.0` | `0.141.1` | 符合 |
| `uvicorn` | `>=0.29,<1.0` | `0.53.0` | 符合 |
| `python-dotenv` | `>=1.0,<2.0` | `1.2.3` | 符合 |
| `pytest` | `>=8.0,<9.0` | `-` | 未安装 |
| `pyarrow` | `-` | `25.0.1` | 已安装（非直接依赖） |
| `pillow` | `-` | `12.3.0` | 已安装（非直接依赖） |
| `wordcloud` | `-` | `1.9.6` | 已安装（非直接依赖） |

## 3. 结论与复核结果

以下依赖的实际版本超出了 `requirements.txt` 的约束范围：

- `pandas`：约束 `>=2.1,<3.0`，实际 `3.0.5`

该差异现已完成团队复核：保留 `requirements.txt` 的 `pandas>=2.1,<3.0` 约束，不以成员个人 `AI` 环境中的 3.0.5 放宽项目依赖。最终验收采用 Conda `py310`，实测 `pandas==2.3.3`，符合项目约束。上表继续保留，作为 HuMingChang 当时运行环境发生漂移的历史记录。

以下依赖当前环境中**未安装**：`seaborn`, `pytest`。

影响说明：
- `seaborn`：本项目 EDA 图表统一使用 `matplotlib` 实现，未依赖 seaborn，因此不影响 M2 流程。
- `pytest`：本项目的测试全部基于标准库 `unittest`，使用 `python -m unittest discover -s tests` 运行，因此不影响测试执行。
- 其余缺失项若属于后续阶段（FastText / BERT / Web）所需，必须在对应阶段开始前解决。

## 4. 复现命令

```powershell
# 在项目根目录、使用已确认的解释器运行全部数据流程
python scripts/clean_data.py
python scripts/split_data.py
python scripts/run_eda.py
python scripts/report_environment.py
python -m unittest discover -s tests -v

# 原始数据质量检查由独立脚本承担
python src/data/check_raw.py
```

> 复现命令使用 `python`，不记录解释器的本机路径。脚本以项目根目录解析路径。
> 本节原先写有本机解释器绝对路径，归档时已去掉，依赖版本与复核结论保持原记录。
