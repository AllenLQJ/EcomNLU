# 项目结项归档说明

- 结项日期：2026-09-22
- 项目状态：M1–M6 已完成，答辩演示已结束
- 当前结项分支：`fix/m5-quantization-final`
- 远端状态：结项提交 `4031ac5` 已快进合入受保护的 `main`，本地与 Gitee 远端 SHA 已核对一致
- 本轮说明：仅整理结项文档，未重新运行测试、训练或端到端服务

## 1. 完成范围

项目已形成数据处理、EDA、TF-IDF 传统基线、随机森林、FastText、BERT、BERT 动态量化、
情感二分类、统一推理接口、Web/API、模型对比、错误分析、答辩 PPT 和离线截图的完整闭环。

最终主部署模型为 `bert_base_chinese`，低成本备用为 `tfidf_svm`；情感任务使用
`sentiment_tfidf_svm`。所有最终数字以 `reports/model-comparison.md` 及对应阶段报告为准。

## 2. Git 内归档内容

- 源代码与配置：`src/`、`scripts/`、`configs/`
- 自动化检查代码：`tests/`
- 数据来源与复现说明：`data/README.md`
- 契约、计划与团队材料：`docs/`
- 实验结果、分析报告和答辩讲稿：`reports/`
- 最终演示截图：`reports/final-presentation/screenshots/`

原始数据、处理后 CSV、模型权重、生成图表和 PPT 按 `.gitignore` 留在本地，避免把大型产物、
许可不明确的数据或个人材料提交到团队仓库。

## 3. 本地归档产物

| 产物 | 路径 | 大小 | SHA-256 / 说明 |
| --- | --- | ---: | --- |
| 最终答辩 PPT | `reports/final-presentation/答辩演示.pptx` | 2,452,420 字节 | `88aa6a04b6294e0fe9fcd57cfc3aa584a4990e1ac121845a0697702a37c3af6b` |
| 原始数据 | `data/raw/online_shopping_10_cats.csv` | 11,299,695 字节 | `f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb` |
| 处理后数据 | `data/processed/` | 本地生成 | 可由清洗、划分脚本重新生成 |
| 模型产物 | `models/` | 本地生成或成员同步 | 共 11 个模型目录，均不提交 Git |
| EDA 图片 | `reports/figures/` | 本地生成 | 可由 `scripts/run_eda.py` 重新生成 |

`deliverables/` 中的简历属于个人材料，不是本项目交付物，继续由 `.gitignore` 排除。

## 4. 恢复顺序

1. 克隆仓库并按 `requirements.txt` 建立 Python 环境。
2. 运行 `python scripts/download_data.py` 获取并校验原始数据。
3. 依次运行 `clean_data.py`、`split_data.py` 和 `run_eda.py`。
4. 按 README 中的命令训练所需模型，或从团队离线归档复制对应模型目录。
5. 使用类别模型和情感模型参数启动 Web 服务。

模型与数据的实际复现命令、接口地址和异常处理说明以项目根目录 `README.md` 为准。

## 5. 已知限制

- 数据集上游未提供清晰、独立的许可证；不得据此推定可商业使用或公开重新分发原始数据。
- FastText 部分变体存在跨机器复现差异，原因尚未通过受控实验确认。
- BERT 动态量化结果会受 PyTorch 版本影响，引用指标时必须同时注明运行环境。
- “热水器”类别样本较少，当前各模型召回率均相对偏低。
- 未完成 BERT 类别加权对照实验，也未进行 CUDA 与 CPU 的跨设备性能比较。
- 原计划的两段演示录屏未交付；答辩最终使用现场演示、PPT 和真实运行截图完成。

以上均为明确记录的项目边界，不影响已交付功能，但后续对外展示时不得省略或改写为已验证结论。

## 6. 最终合并状态

结项文档已提交并快进合入受保护的远端 `main`。合并完成后已核对当前提交、
`origin/main`、本地 `main` 与 Gitee `refs/heads/main`，均指向
`4031ac5907ea40552cd9cda990df3a306a0fc07e`。
