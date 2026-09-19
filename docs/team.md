# 团队分工与协作约定

## 团队成员

| 成员 | 暂定角色 | 主要职责 | 主要交付物 |
| --- | --- | --- | --- |
| LanQvJian | 组长、项目统筹与集成 | 研发计划、任务协调、代码集成、Web 部署、答辩组织 | 项目计划、API/Web、集成测试、答辩材料 |
| HuMingChang | 数据与传统模型负责人 | 数据质量、清洗、EDA、TF-IDF、线性基线、随机森林 | 清洗数据、EDA 报告、传统模型与评估 |
| HongSenYuan | 深度学习与优化负责人 | FastText、BERT、调参、量化/蒸馏/剪枝、性能评估 | FastText/BERT 模型、优化实验、性能报告 |

以上为 M1 暂定分工。成员可以协作和互相评审，但每项任务只能有一个最终负责人；调整职责时更新本文件和研发计划。

## 共同责任

- 每人阅读并遵守根目录 `AGENTS.md`。
- 每天参加晨会并提交简短日报。
- 开发前同步主分支，在独立功能分支工作。
- 提交前运行与改动相关的检查，记录实际结果。
- 不提交数据文件、模型权重、密钥或个人 IDE 配置。
- 发现数据泄漏、错误标签或指标异常时立即记录，不能静默绕过。

## 代码评审建议

- 数据与传统模型改动：HuMingChang 主责，LanQvJian 复核可复现性。
- 深度学习和优化改动：HongSenYuan 主责，HuMingChang 复核评估公平性。
- API、Web 和发布材料：LanQvJian 主责，另外两位各完成一次功能验收。
- 合并主分支前至少由一名非作者成员检查。

## 分支建议

```text
feature/data-eda              # HuMingChang
feature/tfidf-baseline        # HuMingChang
feature/fasttext              # HongSenYuan
feature/bert                  # HongSenYuan
feature/model-optimization    # HongSenYuan
feature/web-app               # LanQvJian
docs/presentation             # LanQvJian，其他成员共同补充
```

## 沟通和记录

- 晨会纪要放在 `reports/meeting-notes/`。
- 日报放在 `reports/daily-reports/`，按日期和成员命名。
- 典型问题应记录现象、原因、解决方案、验证方式和负责人。
- 任务完成时同步更新 `docs/development-plan.md` 和 README 进度。
