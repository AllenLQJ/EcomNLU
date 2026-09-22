# 电商用户评论主题分类与情感分析系统

本项目使用 `online_shopping_10_cats` 中文电商评论数据集，构建可训练、可评估、可优化和可部署的文本分类系统。

## 项目任务

- 主任务：根据评论文本预测 10 个商品或服务类别。
- 扩展任务：判断评论的正向或负向情感。
- 最终交付：模型对比报告、可调用的预测 API 和 Web 演示页面。

固定类别为：书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店。

## 当前进度

- [x] M1：项目初始化
- [x] M2：数据清洗、分层划分与 EDA（已完成团队复核）
- [x] M3：TF-IDF 特征、线性基线与随机森林对比（已完成团队复核）
- [x] M4：FastText 与 BERT（代码、报告、模型契约、真实推理与 API 联调均已验收）
- [x] M5：模型优化与情感分析扩展（量化 + 情感二分类均已交付并验收）
- [x] M6：Web 部署与答辩材料（页面已接入真实模型完成端到端预测，答辩已完成）

M2 已产出可复现的数据闭环：原始 62,774 条 → 清洗后 62,752 条，
按类别分层划分为 50,206 / 6,274 / 6,272，跨集合重复文本为 0。
M3 已产出三个传统模型基线，选定 `tfidf_svm`（测试集 Macro-F1 = 0.8883）。
M4 已完成 4 个 FastText 变体与 1 个 BERT 中文微调模型，并依据验证集 Macro-F1
选定主部署模型 `models/bert_base_chinese`（见 `reports/model-selection.md`）。
M5 已交付 BERT 动态量化（体积 ↓62.7%，同一 CPU 口径下批 32 提速 1.65×）与情感二分类基线；
M6 页面可在同一次请求中返回类别与情感两个任务的预测结果。
五个模型族的完整横向对比见 `reports/model-comparison.md`（含数值来源与一处跨机器复现差异的说明）。

项目已于 2026 年 9 月 22 日完成答辩演示并进入结项归档阶段。最终状态、非 Git 产物位置、
已知限制和归档边界见 [reports/project-closure.md](reports/project-closure.md)。

各阶段交付物状态见 [docs/development-plan.md](docs/development-plan.md)，典型问题与排查过程见
[reports/typical-problems.md](reports/typical-problems.md)。

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

实测结果（测试集 Macro-F1）：`tfidf_svm` 0.8883 ＞ `tfidf_logreg` 0.8779 ＞ `tfidf_rf` 0.8240。
**随机森林并未胜出**——它在三项指标上均垫底，体积 898 MB、单条延迟 77 ms，
不适合部署；结论与取舍依据见对比报告。

> 以上为在本机以固定种子重训后的实测值。`tfidf_svm`（选定模型）与首次训练**完全一致（0.8883）**；
> `tfidf_logreg` 与 `tfidf_rf` 有 0.0002 / 0.0041 的差异，属随机森林固有方差与求解器数值差异范围。

模型文件较大（合计约 1.05 GB），已被 `.gitignore` 排除，不进入版本库；
首次使用需运行上述训练命令或向队友索取。

## 高级模型（M4：FastText 与 BERT）

M4 的两个模型族与 M3 基线使用同一套契约产物与统一评测入口，指标可以直接横向比较；
选模依据同为**验证集 Macro-F1**（AGENTS.md §6）。

### FastText

```powershell
# 语料生成 + 四个变体训练（字符级/词级 × 默认参数/自动调参）+ 验证集评测
python scripts/train_fasttext.py --autotune-duration 120

# 模型定稿后评测测试集
python scripts/evaluate_model.py --model-dir models/fasttext_word_tuned --split test
```

训练文件按 AGENTS.md §5.3 逐行校验制表符、换行符、空文本与非法标签；
推理时的切分方式记录在 `metadata.json` 的 `preprocess.tokenize`，由统一入口复用。

### BERT 微调

```powershell
python scripts/fetch_pretrained.py bert-base-chinese   # 经镜像下载预训练权重到 models/pretrained/
python scripts/train_bert.py --batch-size 32 --max-length 192 --epochs 3
```

`max_length` 由本数据集 token 长度分布实测确定（见 `reports/bert-token-lengths.json`），
默认 192 对应 96.3% 覆盖率，**不是**照搬其它项目新闻标题任务的 32。
训练超参、逐 epoch 曲线、`max_length` 依据与已知局限见 `reports/bert-training-report.md`。

### 统一评测与对比

所有模型都通过 `src.inference` 的统一入口评测，保证「离线评测口径 = 线上推理口径」：

```powershell
python scripts/evaluate_model.py --model-dir models/<model_name> --split test   # 单模型评测
python scripts/evaluate_model.py --all --split dev                              # 全部模型评测
python scripts/compare_models.py --error-analysis-model models/bert_base_chinese --split test
python scripts/verify_model_contract.py models/<model_name>
```

`scripts/compare_models.py` 只读取各模型目录里真实写出的 `metrics.json`，本机缺少对应
模型目录的模型族如实标注为「未提供」（例如没有 M3 的 `models/tfidf_*` 时）；同时输出
错误分析报告（最易混淆类别对 + 高置信度错误样本 + 结论）。

| 产物 | 路径 |
| --- | --- |
| FastText 模型 | `models/fasttext_{char,word}{,_tuned}/` |
| BERT 微调模型 | `models/bert_base_chinese/` |
| 训练报告 | `reports/fasttext-report.md`、`reports/bert-training-report.md` |
| 对比与错误分析 | `reports/model-comparison.md`、`reports/error-analysis.md` |
| 主部署模型决策 | `reports/model-selection.md` |

**训练环境**：BERT 微调在 conda 环境 `bert_gpu`（Python 3.12.14、`torch 2.14.0+cu126`、
NVIDIA RTX 4060 Laptop GPU、bf16 混合精度）上完成；FastText 训练与全部评测在项目 `venv`
（`torch 2.14.0+cpu`）上完成。实际环境与设备记录在 `models/bert_base_chinese/metadata.json`
的 `environment` 字段，`metrics.json` 的 `cost.measured_on` 标明延迟是在哪种设备上实测的。

## 预测与打分

三个模型各有一个独立脚本，功能完全一致，只是默认模型不同
（都是 `scripts/predict.py` 的薄封装，共用同一份实现）：

| 脚本 | 默认模型 |
| --- | --- |
| `scripts/predict_logreg.py` | `models/tfidf_logreg` |
| `scripts/predict_svm.py` | `models/tfidf_svm`（M3 选定的最优传统模型） |
| `scripts/predict_rf.py` | `models/tfidf_rf` |

单条评论预测，返回类别、置信度与 Top-K 概率：

```powershell
python scripts/predict_svm.py --text "这本书内容不错，作者水平很高，值得一读"
python scripts/predict_svm.py --text "手机很好用，电池耐用" --top-k 5
```

**不带任何参数时默认使用验证集 `data/processed/dev.csv`**：它自带真实标签，
因此一条命令即可同时看到逐条预测预览（✓/✗ 标注）与真实评估分数：

```powershell
python scripts/predict_svm.py                 # 预览前 10 条 + dev 全量分数
python scripts/predict_svm.py --preview 0     # 只看分数，不列预览
```

批量预测（CSV 中的一列评论；CSV 自带标签列时会一并显示预测对错与准确率）：

```powershell
python scripts/predict_svm.py --file data/processed/dev.csv --limit 20
python scripts/predict_svm.py --file reviews.csv --text-column review --limit 20
```

对某个划分跑评估分数（Macro-F1、Accuracy、逐类别指标；可存 JSON）：

```powershell
python scripts/predict_svm.py --eval test
python scripts/predict_svm.py --eval dev --output reports/score-svm-dev.json
```

三个脚本都支持 `--model` 临时覆盖默认模型，例如
`python scripts/predict_logreg.py --model models/tfidf_svm --text "..."`。

预测传入的是**原始评论文本**，清洗与分词由模型目录内的 `vectorizer.joblib`
在内部完成，因此不需要预先清洗。脚本本身不实现任何推理逻辑，
全部走 `src.inference.load_predictor`——也就是 M6 的 Web 服务将要调用的同一入口，
所以命令行与页面的分数必然一致。

> 已知失败模式：模型对**明确提到「手机」的短评论**可能高置信度判为「平板」。
> 原因是训练数据中「手机」出现在平板评论里的绝对次数多于手机评论（1,085 : 499），
> 词袋模型无法区分「手机很好用」与「比手机屏幕大」。详见对比报告第 6 节。

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

## 启动 Web 服务

模型未就绪时服务同样可以启动，此时健康检查返回 503 + `degraded`，预测接口返回
`MODEL_NOT_LOADED`，便于在模型产物就绪前先联调页面和错误处理。

```powershell
# 不带模型启动（页面可用，预测返回 MODEL_NOT_LOADED）
python -m src.web.app

# 只启用类别预测
python -m src.web.app --model-dir models/bert_base_chinese

# 类别 + 情感（M5 完成后的推荐用法，答辩演示用这条）
python -m src.web.app --model-dir models/bert_base_chinese `
                      --sentiment-model-dir models/sentiment_tfidf_svm
```

主任务与情感任务**各加载一个模型目录**（契约 v1.2 §2.1：一个目录只能承载一个任务），
二者都在服务启动时加载一次。未指定 `--sentiment-model-dir` 时情感功能关闭，
契约 §3.3 要求的 `sentiment` 字段返回 `null`，页面如实显示「服务未配置情感模型」，
不做任何推断；情感模型加载失败也不影响类别预测。

模型目录按 **`--model-dir` 参数 > `WEB_MODEL_DIR` 环境变量 > `configs/base.yaml`
的 `api.model_dir`** 顺序解析（见 `src/web/config.py`）。

注意 `WEB_MODEL_DIR` 必须在**启动进程之前**设置。`src/web/app.py` 末尾的
`app = create_app()` 在模块被导入时就完成配置解析，导入之后再设置环境变量不会生效：

```powershell
$env:WEB_MODEL_DIR = "models/bert_base_chinese"
python -m src.web.app                      # 正确：变量先于进程
```

```powershell
python -c "import os; os.environ['WEB_MODEL_DIR']='models/...'; import src.web.app"  # 错误：导入已发生
```

因此若在 Python 代码里（而非命令行）启动服务，请显式传参：
`create_app(model_dir="models/bert_base_chinese")`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-dir` | 无 | 主任务模型产物目录，须符合 [docs/contracts.md](docs/contracts.md) 第二章 |
| `--sentiment-model-dir` | 无 | 情感任务模型产物目录；不指定则情感功能关闭 |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--config` | `configs/base.yaml` | 配置文件路径 |

情感模型目录同样遵循「参数 > `WEB_SENTIMENT_MODEL_DIR` 环境变量 > 配置」的解析顺序。

启动后访问：

| 地址 | 说明 |
| --- | --- |
| http://127.0.0.1:8000/ | **演示页面**（智能分析） |
| http://127.0.0.1:8000/docs | 交互式接口文档 |
| http://127.0.0.1:8000/api/health | 健康检查 |
| http://127.0.0.1:8000/api/categories | 类别列表 |
| http://127.0.0.1:8000/api/predict | 单条评论预测（POST） |

模型在服务启动时加载**一次**，请求处理中不重复加载（契约 §3.5）。

### 页面使用流程

1. 在「输入电商评论」中输入一条中文评论（最多 2000 字符，右下角实时显示已用字数）。
2. 选择 Top-K（1–10，默认 3），点击「开始分析」，或按 `Ctrl + Enter` 提交。
3. 右侧依次显示：预测类别与置信度、使用模型与推理耗时、Top-K 概率条、情感分析状态。
4. 下方「建议处理方式」是**前端规则映射**（按预测类别确定性给出，非模型输出），并标注了优先级。
5. 展开「查看接口详情」可看到本次请求与响应的原始 JSON，便于解释前后端联调。
6. 「清空」恢复初始状态；顶部「帮助文档」「关于我们」为说明弹窗。

### 页面状态说明

| 状态 | 表现 |
| --- | --- |
| 模型未就绪 | 顶部状态胶囊显示橙色「模型未就绪」；页面仍可浏览，提交会提示 `MODEL_NOT_LOADED` |
| 服务正常 | 顶部显示绿色「服务正常」，提交后返回真实预测结果 |
| 服务异常 | 顶部显示红色「服务异常」，请求失败不保留上一次结果 |
| 输入不合法 | 输入框下方给出具体原因并聚焦到对应控件 |
| 情感分析 | 已配置情感模型时显示真实的正向/负向与置信度；未配置时显示「尚未启用 · 服务未配置情感模型」。未提交请求前显示「分析后显示」，三种状态区分明确，前端不做任何推断 |

页面为原生 HTML / CSS / JavaScript，无构建步骤、无 CDN 依赖，与后端同源，可在离线环境演示。
左侧「历史记录 / 数据统计 / 模型管理 / 系统设置」为本轮范围外的规划中模块，点击给出提示。

### 前端与接口的自动化检查

```powershell
python -m unittest discover -s tests -p "test_web*.py" -v   # 40 项
python -m unittest discover -s tests -v                     # 全量
```

Web 模块的已验证运行环境为 Python 3.10.20、FastAPI 0.141.1、pydantic 2.13.5、uvicorn 0.52.4。

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
