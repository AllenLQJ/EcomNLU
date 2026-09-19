# 电商用户评论主题分类与情感分析系统

本项目使用 `online_shopping_10_cats` 中文电商评论数据集，构建可训练、可评估、可优化和可部署的文本分类系统。

## 项目任务

- 主任务：根据评论文本预测 10 个商品或服务类别。
- 扩展任务：判断评论的正向或负向情感。
- 最终交付：模型对比报告、可调用的预测 API 和 Web 演示页面。

固定类别为：书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店。

## 当前进度

- [x] M1：项目初始化
- [ ] M2：数据清洗、分层划分与 EDA
- [ ] M3：TF-IDF 线性基线与随机森林
- [ ] M4：FastText 与 BERT
- [ ] M5：模型优化与情感分析扩展
- [ ] M6：Web 部署与答辩材料

当前仅完成项目初始化与原始数据获取。尚未完成数据清洗、模型训练或端到端预测。

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

推荐使用 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

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

## 基础检查

```powershell
python -m unittest discover -s tests -v
```

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
