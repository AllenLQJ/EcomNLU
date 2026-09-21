# 团队分工与协作约定

## 团队成员

| 成员 | 正式角色 | 主要职责 | 主要交付物 |
| --- | --- | --- | --- |
| LanQvJian | 组长、项目统筹与集成 | 研发计划、任务协调、代码集成、Web 部署、答辩组织 | 项目计划、API/Web、集成测试、答辩材料 |
| HuMingChang | 数据、传统模型与情感基线负责人 | 数据质量、清洗、EDA、TF-IDF、传统模型、M4 公平性复核、M5 情感分类基线 | 清洗数据、EDA 报告、传统模型、情感基线与错误分析 |
| HongSenYuan | 深度学习与优化负责人 | FastText、BERT、调参、量化/蒸馏/剪枝、性能评估 | FastText/BERT 模型、优化实验、性能报告 |

以上分工已由团队确认并正式生效。成员可以协作和互相评审，但每项任务只能有一个最终负责人；调整职责时更新本文件和研发计划。

## 仓库权限

- 远程仓库：`https://gitee.com/AllenLQJ/ecommerce-text-classification.git`
- LanQvJian：仓库所有者
- HuMingChang：开发者权限
- HongSenYuan：开发者权限
- `main`：保护分支，只有仓库管理员可以推送和合并
- 开发者在个人功能分支推送代码，通过 Pull Request 合并到 `main`。

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

## M4 验收责任

M4 采用三级验收，不由开发者自行宣布完成：

1. HongSenYuan 完成 FastText/BERT 开发并推送个人功能分支，提供训练配置、指标、模型产物说明和复现命令。
2. HuMingChang 进行技术复核，重点确认 M4 与 M3 使用相同的数据集指纹、标签顺序和评估口径；确认只用验证集选模，测试集未参与调参。
3. LanQvJian 进行最终检验，重点验证模型契约、磁盘重载、单条与批量真实推理、模型大小、推理延迟及 M6 部署可行性。
4. 问题修复并复验通过后，才允许合入 `main`。

## M5 并行分工

- HuMingChang 主责情感二分类基线：使用数据集 `label` 字段完成 TF-IDF + 逻辑回归，可选增加校准 LinearSVC；输出统一指标、混淆矩阵、按商品类别拆分的情感表现和典型错误分析；保存可重载产物并接入统一推理接口。
- HongSenYuan 主责高级模型优化：基于 M4 的 BERT 完成一种可落地的量化、蒸馏或剪枝方案；对比优化前后的 Macro-F1、模型大小、单条/批量延迟和吞吐量。时间允许时再增加 BERT 情感模型或类别与情感多任务模型。
- 两人交叉复核对方结果，LanQvJian 负责统一接口、Web/API 联调和最终验收。
- M5 所有实验必须使用稳定数据集指纹证明数据版本，并与 M3/M4 使用相同的测试集；不同成员不得同时无边界修改同一训练文件。

## 分支建议

```text
feature/data-eda              # HuMingChang
feature/tfidf-baseline        # HuMingChang
feature/fasttext              # HongSenYuan
feature/bert                  # HongSenYuan
feature/model-optimization    # HongSenYuan
feature/bert-optimization     # HongSenYuan，M5 BERT 优化
feature/sentiment-baseline    # HuMingChang，M5 情感分类基线
feature/web-app               # LanQvJian
docs/presentation             # LanQvJian，其他成员共同补充
```

## 沟通和记录

- 晨会纪要放在 `reports/meeting-notes/`。
- 日报放在 `reports/daily-reports/`，按日期和成员命名。
- 典型问题应记录现象、原因、解决方案、验证方式和负责人。
- 任务完成时同步更新 `docs/development-plan.md` 和 README 进度。
