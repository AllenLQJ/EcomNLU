# 契约确认记录：HongSenYuan 确认 v1.0

- 确认人：HongSenYuan
- 被确认文件：`docs/contracts.md` v1.0（契约分支 `feature/api-contracts`，提交 `0049f75`）
- 确认日期：2026-09-21
- 结论：**确认，对 v1.0 条款无异议。** 附 6 项确认后待办，均为实现层或需按 §0 流程升版的事项，不构成本次确认的前提条件。
- 复核方式：由开发助手执行下列命令并整理结果，命令与输出均为实际运行结果；确认人 HongSenYuan 对结论负责。

## 一、确认依据（实际运行证据）

| 验证项 | 命令 | 实际结果 |
| --- | --- | --- |
| 契约分支单元测试 | `python -m pytest tests -q` | 本地存在 `data/raw` 时 **9 passed**；全新克隆下 `test_required_directories_exist` 失败，原因见待办 1 |
| 数据契约校验器 | `python scripts/verify_data_contract.py` | `data/processed` 为空时输出缺失文件清单、通过 0 项失败 1 项、退出码 1，**无 Python 堆栈泄漏** |
| 模型契约校验器 | `python scripts/verify_model_contract.py` | `models/` 下无模型时给出明确提示、退出码 1；`--help` 与 `--processed-dir` 可用 |
| 数据质量报告可复现性 | `python src/data/check_raw.py` 后与提交版逐字段对比 | **仅 3 处差异，全部为环境/时间字段**（`environment.platform`、`environment.python`、`generated_at`），所有数据统计完全一致 |

### 数据质量报告与独立测量的交叉核对

分支内 `reports/data-quality/raw_quality_report.md` 的数据与另一次独立测量结果逐项一致：

| 项目 | 报告值 | 独立测量值 |
| --- | --- | --- |
| 行数 / 字段 | 62,774 / `cat,label,review` | 一致 |
| 文件字节数 / SHA-256 | 11,299,695 / `f6c5b724…afdb` | 一致 |
| 类别分布 | 平板/水果/酒店/洗发水/衣服 各 10,000；计算机 3,992；书籍 3,851；手机 2,323；蒙牛 2,033；热水器 575 | 一致 |
| 情感标签 | `1`: 31,728，`0`: 31,046 | 一致 |
| 缺失 / 重复 / HTML 实体 | 1 / 49 行（45 组）/ 49 | 一致 |
| 长度中位数 / p95 / p99 / 最长 | 34 / 183 / 374.27 / 2,876 | 一致 |

控制字符一项报告的检查更严格：报告检出 1 条含 `\x7f` 的记录，独立测量使用的字符区间未覆盖 `\x7f`，以报告为准。

### 契约实现与文档的一致性抽查

以下条款经代码审查确认已按文档实现，而非仅写在文档里：

- §1.2 列名与顺序 `sample_id,review,cat,label` 逐项校验；`review` 非空、无首尾空白；`cat`、`label` 取值受白名单约束。
- §1.2 `sample_id` 校验为整数、划分内唯一、跨划分不重复，并可用原始数据核对是否错位（`check_sample_ids` / `check_traceability`）。
- §1.3 `label_mapping.json` 的 `categories` 与 `configs/base.yaml` 的 `task.categories` 比对。
- §1.4 manifest 占位符检测、源文件 SHA-256 校验、各 CSV 文件 SHA-256 校验、泄漏按归一化文本判定。
- §2.1/§2.2 `label_mapping.json` 与数据契约那份用 `read_bytes` 做**逐字节**比对；模型本体按类型校验。
- §2.3 `metadata.json` 必需字段含 `split_manifest_sha256`；`max_length` 对 transformers 模型要求正整数、其余必须为 `null`。
- §2.4 metrics 结构、`cost` 字段、统一入口推理结果（概率合法性、argmax 一致性）均在校验范围内。

## 二、本次确认对 HongSenYuan（M4/M5）的约束

已确认按以下规定产出模型，无需再逐条确认：

| 要求 | 规定 |
| --- | --- |
| 产物目录 | `models/<model_name>/`；BERT 用 `bert_base_chinese`，优化产物用 `bert_base_chinese_<优化方式>` |
| 四件套 | 模型本体 + `label_mapping.json`（从 `data/processed/` **逐字节复制**，不得重新生成）+ `metadata.json` + `metrics.json` |
| 模型本体 | Transformers 保存 `model/` 与 `tokenizer/` 两个 `save_pretrained` 目录 |
| metadata | `split_manifest_sha256` 必填；`max_length` 填本数据集 token 分布实测值；`hyperparameters` 完整到可复现 |
| metrics | dev 五指标 + `per_class` + 10×10 混淆矩阵（行真实、列预测）；**选模阶段 `test` 必须为 `null`**；`cost` 必须实测 |
| 统一入口 | 适配 `src/inference/` 的 `load_predictor` / `CategoryPredictor`；`predict_proba` 覆盖全部 10 类、键为中文类别名、缺失补 0 |
| 自检 | `python scripts/verify_model_contract.py models/<model_name>`，未通过不得用于部署或对比 |

## 三、确认后待办（6 项）

| 编号 | 问题 | 影响 | 建议负责人 | 处理方式 |
| ---: | --- | --- | --- | --- |
| 1 | `data/raw/` 未被 git 跟踪（`data/` 下只有三个 README 被跟踪），全新克隆不存在该目录 | README 记录的基础检查在新克隆必然失败 | 任一成员 | 补 `data/raw/README.md`，与 `interim/`、`processed/` 保持一致；不改契约条款 |
| 2 | 45 组冲突重复（94 行）的处理规则未定，契约 §1.4 只要求"如实记录"，数据质量报告已指出不能直接 `drop_duplicates` | 直接影响 M2 数据完整性与跨集合泄漏判定 | HuMingChang | 先定规则、写入 `split_manifest.cleaning.rules`；若需新增条款按 §0 升 v1.1 |
| 3 | 契约 §3.5 要求非法请求返回 `INVALID_JSON` + 400，而 FastAPI 默认对请求体校验失败返回 **422** 且结构不同 | M6 现场演示异常输入时会不符合契约 | LanQvJian | 实现层加异常处理器映射为统一错误结构；§3.5 本身不变 |
| 4 | Python 版本三处不一致：README 写 3.11、项目 venv 为 3.12.14、提交的报告记录 3.10.20 | 违反 AGENTS.md §7.2 对环境记录的要求，影响可复现性 | 全员 | 统一 README 与项目解释器口径，并用该解释器重新生成数据质量报告 |
| 5 | 情感任务在模型契约中是空白：未定义情感模型的 `model_name`、`task` 取值与 `metrics.categories`（应为 `["负向","正向"]`） | M5 做情感二分类时会各自即兴命名，与统一加载入口和指标校验对不上 | HongSenYuan 提出 | 按 §0 流程升 v1.1 补充 |
| 6 | §2.3 写有"本项目当前为纯 CPU，`device` 填 `cpu`" | 对项目 venv 成立（`torch 2.14.0+cpu`），但本机存在 RTX 4060（8 GB）且另有可用的 CUDA 环境；若 BERT 走 GPU，`metadata.device` 必须如实填 `cuda` | 全员讨论 | 若确定走 GPU 训练，按 §0 流程升 v1.1 修订该表述 |

待办 2、5、6 涉及条款文字，按 §0 规定须由提出人说明内容与原因、征得另两名成员同意、更新文件并提升版本号；本条记录不作为已生效的条款变更。

## 四、生效状态

| 成员 | 确认状态 | 日期 |
| --- | --- | --- |
| LanQvJian | 已确认（起草人） | 2026-09-21 |
| HuMingChang | 待确认 | |
| HongSenYuan | 已确认（本记录） | 2026-09-21 |

HuMingChang 确认后，`docs/contracts.md` v1.0 满足 §5 的正式生效条件。
