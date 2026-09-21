# 契约确认记录：HuMingChang 确认 v1.0

- 确认人：HuMingChang（数据工程、传统模型，契约第一节「数据契约」的产出方）
- 被确认文件：`docs/contracts.md` v1.0（契约分支 `feature/api-contracts`，提交 `0049f75`）
- 确认日期：2026-09-21
- 结论：**确认，对 v1.0 条款无异议。** 数据契约已按契约完成改造并通过全部校验项；另附对 v1.1 三项提案的表决意见与确认后待办。
- 复核方式：由开发助手执行下列命令并整理结果，命令与输出均为实际运行结果；确认人 HuMingChang 对结论负责。

---

## 一、确认依据（实际运行证据）

| 验证项 | 命令 | 实际结果 |
| --- | --- | --- |
| 数据契约校验器 | `python scripts/verify_data_contract.py` | **通过 19 项，失败 0 项，跳过 0 项**，结论「数据契约校验通过」，退出码 0 |
| 数据流水线测试 | `python -m unittest discover -s tests` | `Ran 45 tests` → `OK`（无失败、无跳过），耗时 3.56 秒 |
| 全链路可复现 | 依次运行 `clean_data.py`、`split_data.py`、`run_eda.py`、`report_environment.py` | 四个脚本退出码均为 0 |
| 划分可复现性 | 固定种子 42 重跑 | 清洗与划分结果完全一致，泄漏计数恒为 0 |

校验器的运行方式：本分支（`hmc`）不含契约分支的校验脚本，因此用
`git worktree add <目录> origin/feature/api-contracts` 检出契约分支，
把本分支产出的 `data/processed/` 与原始数据复制进去后运行该校验器。
这样校验的是**契约分支的校验代码**与**本分支的真实数据产物**，两者未被混同。

### 1.1 校验器逐项结果

```text
[OK] 必需文件齐全（5 个文件均存在）
[OK] label_mapping.json 可解析 / split_manifest.json 可解析
[OK] label_mapping 的 categories 与配置一致
[OK] label2id / id2label 与 categories 顺序一致
[OK] split_manifest 无残留占位符 / 固定规则合法
[OK] 原始数据哈希一致
[OK] train.csv / dev.csv / test.csv 无 BOM
[OK] manifest 的文件哈希与行数一致
[OK] 列名与顺序为 sample_id,review,cat,label
[OK] sample_id 合法且不重复
[OK] review / cat / label 取值合法
[OK] sample_id 可追溯回原始数据
[OK] 跨划分无文本泄漏（三个划分两两无重复文本）
[OK] manifest 的泄漏声明与实测一致
[OK] 实测类别分层比例合法
```

### 1.2 数据契约产物的关键事实

| 项目 | 数值 |
| --- | --- |
| 原始行数 | 62,774 |
| 清洗后行数 | 62,752（删除 22 条，0.0350%） |
| 划分规模 | train 50,206 / dev 6,274 / test 6,272 |
| 跨集合重复文本 | train-dev 0、train-test 0、dev-test 0 |
| `split.dedup_key` | `normalized_text` |
| `cleaning.rules.duplicate_review` | `drop_within_category_only` |
| 原始数据 SHA-256 | `f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb` |

### 1.3 为达成契约合规所做的改造

首次用校验器核验本分支产物时结果为 **`[FAIL] 必需文件齐全 → 通过 0 项，失败 1 项`**。
据此逐项修复了 5 处不符合（全部在数据产出方一侧）：

| # | 契约要求 | 改造前 | 现在 |
| ---: | --- | --- | --- |
| 1 | `data/processed/label_mapping.json` | 不存在 | 已生成，类别顺序取自配置 |
| 2 | `data/processed/split_manifest.json` | 不存在 | 已生成，哈希与行数均从真实文件计算 |
| 3 | 列顺序 `sample_id,review,cat,label` | 原为 `cat,label,review` | 已按契约顺序，由 `write_contract_csv` 强制保证 |
| 4 | `sample_id`（原始 CSV 行号） | 完全没有 | 已在删行之前写入显式列，可追溯回原始数据 |
| 5 | 各类别分层比例容差 ±1.0 行 | 水果 +1.6、洗发水 +1.4 | 全部 ≤0.8（配额修复后） |

其中第 4 项的要点：`sample_id` **不能用 `reset_index` 充当**——删行后行号会整体前移，
追溯将全部错位；必须在任何筛选操作之前把原始行号落成独立列。

第 5 项的根因：为保证「同一评论文本不跨划分边界」，划分以文本组为单位整体投放，
当某类别含跨类别的重复文本组时会略微超出配额。修复方式是增加一轮**配额修复**，
仍只搬运整组，因此不破坏零泄漏约束；目标容差收紧到 ±0.5 行，严于契约的 ±1.0。

---

## 二、本次确认对 HuMingChang（M2/M3）的约束

| 要求 | 规定 | 现状 |
| --- | --- | --- |
| 数据产物 | `data/processed/{train,dev,test}.csv` + `label_mapping.json` + `split_manifest.json` | 已全部产出并通过校验 |
| 列顺序 | `sample_id,review,cat,label`，UTF-8 无 BOM，不写行索引 | 由 `src/data/raw_io.py` 的 `write_contract_csv` 强制 |
| `cat` 取值 | 中文类别名，取自 `configs/base.yaml` 的 `task.categories` | 符合 |
| 模型产物 | §2.1 目录 `models/<model_name>/`，四件套齐备；§2.2 scikit-learn 须保存 `model.joblib` 与 `vectorizer.joblib` | M3 按此产出，尚未开始 |
| `metadata.json` | 必填 `split_manifest_sha256`、`train_rows`、`random_seed`；`max_length` 对非 BERT 模型填 `null` | M3 按此产出 |
| `metrics.json` | 选模阶段 `test` 必须为 `null`；五项指标名与 `evaluation.report_metrics` 一致；混淆矩阵 10×10 行真实列预测 | M3 按此产出 |
| 测试集纪律 | 测试集只用于最终确认，禁止参与调参或选模 | 承诺遵守 |
| 统一加载接口 | `load_predictor(model_dir) -> CategoryPredictor`，`predict_proba` 覆盖全部 10 类 | M3 训练方按签名适配 |

---

## 三、对 v1.1 三项提案的表决意见

### 提案一：补充情感任务的模型产物约定

**立场：同意**，并建议补一条来自数据契约的约束。

情感二分类的标签语义在数据契约中已经固定：`label` 为 `0`（负向）或 `1`（正向）。
提案草案规定 `categories` 为 `["负向", "正向"]`，其下标恰好与 `label` 取值一致
（`负向`→0、`正向`→1）。**建议把这层对应关系在契约里写明**，因为一旦情感模型
把 `id2label` 写成 `["正向", "负向"]` 或训练时反转了标签方向，
混淆矩阵与 `per_class` 依然「结构合法」但方向全错，而现有校验器无法发现这类错误。

另需如实说明一项数据层面的限制（已记录在 `reports/data-quality/raw_quality_report.md`）：
原始数据中 **6 个类别（平板、水果、洗发水、衣服、计算机、酒店）的正负样本数完全相等**，
说明上游对情感标签做过**人工配平**。因此情感二分类是在配平子集上评估的，
比真实电商评论场景简单，**其高分不能直接解读为情感建模能力强**，
答辩与报告中必须如实说明这一限制。

### 提案二：修订 §2.3 中「本项目当前为纯 CPU」的表述

**立场：同意**，并补充两条实测事实。

1. **CUDA 版 torch 现成可用，M4 不需要额外装环境。** 本项目数据流水线所用的
   conda 环境 `AI` 实测：`torch 2.6.0+cu124`、`torch.cuda.is_available()` 为 `True`、
   设备 `NVIDIA GeForce RTX 4060 Laptop GPU`（显存 8.0 GB，可用 6.9 GB，算力 8.9）。
   该结果已记录在 `reports/environment.md`。因此「这台机器不能跑 GPU」不成立，
   真正的限制是**工具链不统一**，而非硬件缺失。

2. **更该写进契约的是环境一致性问题。** 目前项目实际有两个环境在跑：
   M4 方在用的项目 `venv`（`torch 2.14.0+cpu`）与本数据流水线所用的 conda 环境 `AI`
   （`torch 2.6.0+cu124`），两者的 torch 版本与构建都不同。
   AGENTS.md §7.2 要求「记录依赖版本和运行环境」；而 §2.3 的 `split_manifest_sha256`
   只保证**数据一致**，无法保证**环境一致**。
   **建议在 §2.3 增加 `environment` 字段**，至少记录解释器路径、`torch` 版本与
   `device`，使「所有模型在同一份数据、可比较的环境下评估」这一前提可被校验，
   而不只是靠约定。否则答辩现场无法解释两个环境产出的指标是否可比。

### 提案三：固定冲突重复文本的处理规则

**立场：不同意方案 A，主张维持当前实现。**

**一、提案对问题的量化前提需要更正。** 提案把「45 组、涉及 94 行」统称为冲突重复，
但按归一化文本口径精确拆解（实测 46 组 / 96 行），其中包含**三种性质完全不同**的情况：

| 情况 | 组数 | 行数 | 性质 |
| --- | ---: | ---: | --- |
| 同类别内、同情感标签 | 18 | 36 | 纯冗余，删除无信息损失 |
| 同类别内、**情感标签矛盾** | 17 | 34 | 真冲突：同文本同类别却既有 label=0 又有 label=1 |
| 跨类别歧义 | 28 | 60 | **不是冲突**：同一文本对应多个合法类别 |

被统称为「冲突」的 94 行里，**60 行（占 64%）并不是冲突**，而是同一文本对应多个
合法类别的有效配对。按方案 A 丢弃它们会损失有效样本。

**二、真正的冲突只有 17 组 34 行（0.054%），且冲突在 `label` 而非 `cat`。**
例如 `'3999抢购的 性价比高…'` 在「计算机」下同时存在 label=1 与 label=0。
需要指出：**对主任务 10 分类，两种副本的 `cat` 都是正确的**，冲突只影响扩展任务
情感二分类。因此方案 A 以「避免任意标签」为由丢弃 94 行，代价被高估、收益被低估。

**三、「方案 B/C 复杂度上升、周期内不划算」这一论据目前已不成立。**
按文本分组的划分与配额修复**已经实现并验证完毕**：上述校验器 19/19 通过，
跨划分文本泄漏为 0（由校验器独立重算），各类别分层比例全部落在 ±1.0 行内。
复杂度成本已经付掉，不存在「周期内不划算」。

**四、如实记录本方案残余的已知局限（这是本次表决的关键取舍）。**
当前规则为「按 `(review, cat)` 去重保留首条」，对这 17 组同类别内标签矛盾的记录而言，
**实质上就是任意保留了其中一个标签、任意丢弃了另一个**——这正是提案对
`drop_duplicates` 的批评，该批评**部分适用于当前实现**。
具体影响：17 行（0.027%）的样本带有任意选定的情感标签。
对主任务 10 分类无影响（`cat` 两种副本相同），对情感任务构成 17 组矛盾监督信号。

**结论**：在剩余开发周期内，维持当前实现（跨类别保留 + 分组划分零泄漏；
同类别内按 `(review, cat)` 去重）。理由是方案 A 会额外损失 60 行有效样本，
而当前实现的残余问题影响仅 17 行且已量化。
**建议把「同类别内标签矛盾整组删除」作为 v1.2 或 M3 之后的候选改进项记录在案**，
不与 v1.1 绑定。若 LanQvJian 坚持采用方案 A，请一并确认接受 60 行的额外损失。

---

## 四、确认后待办

| 编号 | 事项 | 负责人 | 说明 |
| ---: | --- | --- | --- |
| 待办 1 | `requirements.txt` 增加 `wordcloud>=1.9,<2.0`（以及传递依赖 `pillow`） | LanQvJian | 词云已成为 EDA 的实际依赖；该文件由 LanQvJian 维护，数据方未擅自改动 |
| 待办 2 | `src/data/check_raw.py` 第 270 行的归一化改用 Python 的 `re` | LanQvJian | 该行 `str.replace(r"\s+", " ", regex=True)` 走 RE2，其 `\s` 只识别 ASCII，**无法合并全角空格（U+3000）与不换行空格（U+00A0）**；实测 52 行的归一化结果与 Unicode 口径不同。当前重复统计数字恰好未受影响（两种口径均为 96 行/46 组），属**潜伏缺陷**，一旦全角空格出现在会构成重复对的位置即会静默少报 |
| 待办 3 | 修正数据质量报告中「冲突重复组：45（涉及 94 行）」的表述 | LanQvJian | 建议改为按第二节的三类拆分表述。原文「既无法直接用于训练」会使 28 组跨类别有效配对（60 行）被误删 |
| 待办 4 | 确认 `data/processed/` 下两个 JSON 是否仍需被 `.gitignore` 排除 | LanQvJian | 契约要求下游训练前必须存在这两个文件，当前需先运行 `scripts/split_data.py` 生成；若希望全新克隆即可训练，需调整忽略规则 |

---

## 五、记录

本确认记录由 HuMingChang 于 2026-09-21 出具，依据 `docs/contracts.md` §0 规定的变更流程留存于
`reports/meeting-notes/`。相关证据文件：`reports/data-split-report.md`（划分与泄漏）、
`reports/data-cleaning-report.md`（清洗规则与决策依据）、`reports/eda-report.md`（EDA 与 max_length 评审）、
`reports/environment.md`（依赖与环境比对）、`reports/daily-reports/2026-09-21-HuMingChang.md`（当日过程记录）。
