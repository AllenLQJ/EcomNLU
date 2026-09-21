# 电商用户评论主题分类与情感分析系统

本项目使用 `online_shopping_10_cats` 中文电商评论数据集，构建可训练、可评估、可优化和可部署的文本分类系统。

## 项目任务

- 主任务：根据评论文本预测 10 个商品或服务类别。
- 扩展任务：判断评论的正向或负向情感。
- 最终交付：模型对比报告、可调用的预测 API 和 Web 演示页面。

固定类别为：书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店。

## 当前进度

- [x] M1：项目初始化
- [x] M2：数据清洗、分层划分与 EDA（待团队复核，见 `docs/development-plan.md`）
- [x] M3：TF-IDF 特征、线性基线与随机森林对比（待团队复核）
- [ ] M4：FastText 与 BERT
- [ ] M5：模型优化与情感分析扩展
- [ ] M6：Web 部署与答辩材料

M2 已产出可复现的数据闭环：原始 62,774 条 → 清洗后 62,752 条，
按类别分层划分为 50,206 / 6,274 / 6,272，跨集合重复文本为 0。
M3 已产出三个传统模型基线，选定 `tfidf_svm`（测试集 Macro-F1 = 0.8883）。
模型训练与端到端预测的演示页面尚未完成。

## 项目周期

- 正式开始：2026 年 9 月 21 日
- 项目截止：2026 年 9 月 22 日中午 12:00（北京时间）
- 9 月 22 日 10:00 开始最终集成验收，11:30 冻结代码和材料

## 项目结构

```text
.
├── AGENTS.md                  # 项目开发规范
├── configs/base.yaml          # 基础配置
├── data/
│   ├── raw/                   # 原始数据，只读且不提交 Git
│   ├── interim/               # 清洗后的中间数据
│   └── processed/             # train/dev/test
├── docs/                      # 团队分工与研发计划
├── notebooks/                 # 探索性分析
├── reports/                   # 图表、会议纪要、日报和答辩材料
├── scripts/                   # 可重复执行的命令行脚本
├── src/                       # 项目核心代码
├── tests/                     # 自动化检查
└── requirements.txt
```

## 环境准备

推荐使用 **Python 3.11**。各阶段必须在报告和模型元数据中记录实际
解释器及依赖版本，不得把单个成员的本地环境当作全队已统一的环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`.venv/` 与 `venv/` 都被 `.gitignore` 忽略，可按个人习惯任选其一；
是否可复现以受约束的 Python 版本、`requirements.txt` 和实际环境记录为准。

`fasttext-wheel`、PyTorch 和 Transformers 将在对应模型阶段实际验证；M1 不代表这些模型已经成功训练。

## 获取原始数据

```powershell
python scripts/download_data.py
```

脚本会把原始 CSV 写入：

```text
data/raw/online_shopping_10_cats.csv
```

脚本会校验文件 SHA-256、字段名与基础行数。原始 CSV 被 `.gitignore` 排除，团队成员应分别运行下载脚本获取相同数据。

数据来源、字段和使用限制见 [data/README.md](data/README.md)。

## 数据流程

四个脚本必须按顺序执行，每一步都拒绝跳过上游产物，从而保证流程可复现：

```powershell
python scripts/clean_data.py            # 清洗 -> data/interim/reviews_clean.csv
python scripts/split_data.py            # 8:1:1 分层划分 -> data/processed/
python scripts/run_eda.py               # EDA 图表与 max_length 评审报告
python scripts/report_environment.py    # 依赖与实际环境比对记录
```

原始数据质量检查由 `python src/data/check_raw.py` 负责，可独立运行，不属于上述链路。

`scripts/split_data.py` 除 `train/dev/test.csv` 外，还按接口契约
（[docs/contracts.md](docs/contracts.md)）生成 `label_mapping.json` 与
`split_manifest.json`，两者是下游训练的前置依赖。

产物位置：

| 产物 | 路径 |
| --- | --- |
| 数据契约产物 | `data/processed/{train,dev,test}.csv`、`label_mapping.json`、`split_manifest.json` |
| 原始数据质量报告 | `reports/data-quality/raw_quality_report.md` |
| 清洗报告 | `reports/data-cleaning-report.md` |
| 划分与泄漏检查报告 | `reports/data-split-report.md` |
| EDA 报告与图表 | `reports/eda-report.md`、`reports/figures/` |
| 环境与依赖记录 | `reports/environment.md` |

数据划分使用固定随机种子 42，并以评论文本为分组单位，确保同一文本不会跨越
训练集、验证集和测试集；划分脚本在检测到泄漏时会以非零退出码失败。

契约合规可用契约分支提供的校验器验收：

```powershell
python scripts/verify_data_contract.py
```

## 传统模型基线（M3）

训练 TF-IDF 基线与评估：

```powershell
python scripts/train_baseline.py                 # 完整训练，约 6-7 分钟
python scripts/train_baseline.py --quick         # 小样本冒烟，用于快速验证流程
python scripts/train_baseline.py --skip-ablation # 跳过特征方案对照，直接用默认配置
```

脚本会：只用训练集拟合向量器 → 在验证集上按 **Macro-F1** 选模 → 写出三个模型的
契约产物 → 通过统一入口实测推理成本 → 生成对比报告。

| 产物 | 路径 |
| --- | --- |
| 模型产物（每个模型 5 个文件） | `models/{tfidf_logreg,tfidf_svm,tfidf_rf}/` |
| 模型对比报告 | `reports/m3-baseline-comparison.md` |
| 机器可读结果（含调参与对照明细） | `reports/m3-baseline-results.json` |

模型契约用统一校验器验收（对每个模型逐项校验文件、字段、类别顺序与真实推理）：

```powershell
python scripts/verify_model_contract.py
```

实测结果（测试集 Macro-F1）：`tfidf_svm` 0.8883 ＞ `tfidf_logreg` 0.8777 ＞ `tfidf_rf` 0.8281。
**随机森林并未胜出**——它在三项指标上均垫底，体积 900 MB、单条延迟 72 ms，
不适合部署；结论与取舍依据见对比报告。

模型文件较大（合计约 1.05 GB），已被 `.gitignore` 排除，不进入版本库；
首次使用需运行上述训练命令或向队友索取。

## 运行环境

M2 数据流水线的已验证环境为 conda 环境 `AI`（Python 3.11.15）。
该记录仅说明 M2 的实际运行环境，不强制其他成员使用同一本地路径。
实际依赖版本及与 `requirements.txt` 的偏差见
[reports/environment.md](reports/environment.md)。

## 基础检查

```powershell
python -m unittest discover -s tests -v
```

## API 接口约定

Web 服务的统一接口、请求字段、响应结构和错误码见
[docs/contracts.md](docs/contracts.md)。单条评论的默认长度上限为 **2000 字符**，
由 `configs/base.yaml` 中的 `api.max_review_chars` 统一配置；超限请求返回
`REVIEW_TOO_LONG`，不会静默截断。

## 团队

- LanQvJian：项目统筹、工程集成、Web 部署与答辩组织
- HuMingChang：数据工程、EDA、TF-IDF 与传统模型
- HongSenYuan：FastText、BERT、模型优化与性能评估

详细职责和协作规则见 [docs/team.md](docs/team.md)，研发计划见 [docs/development-plan.md](docs/development-plan.md)。

## 评估原则

数据类别不均衡，主任务以验证集 `Macro-F1` 作为主要选模指标，同时报告 Accuracy、Macro Precision、Macro Recall、Weighted-F1、分类报告和混淆矩阵。

测试集只用于最终确认，禁止参与调参和模型选择。

## 安全和提交规则

- 不提交原始数据、模型权重、缓存、`.env` 或 API Key。
- 不使用个人电脑绝对路径，路径统一相对项目根目录解析。
- 不编造实验指标；未实际运行的结果必须标记为未验证。
- 每项功能在提交前记录验证命令和实际结果。

完整约束见 [AGENTS.md](AGENTS.md)。
