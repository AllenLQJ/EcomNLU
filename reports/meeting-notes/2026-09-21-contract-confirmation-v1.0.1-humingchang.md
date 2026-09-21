# 契约确认记录：HuMingChang 确认 v1.0.1（数据集指纹勘误）

- 确认人：HuMingChang（数据工程、传统模型；契约 §2.3 模型产物的产出方）
- 被确认文件：`docs/contracts.md` **v1.0.1**（契约分支 `fix/dataset-fingerprint`，提交 `63fd1f5`）
- 变更记录：`reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md`
- 确认日期：2026-09-21
- 结论：**确认，对 v1.0.1 无异议。** 已按新定义改造 M3 实现并重新通过全部校验。
- 复核方式：由开发助手执行下列命令并整理结果，命令与输出均为实际运行结果；确认人 HuMingChang 对结论负责。

---

## 一、确认依据（实际运行证据）

| 验证项 | 命令 | 实际结果 |
| --- | --- | --- |
| 模型契约校验 | `python scripts/verify_model_contract.py` | 三个模型各 **15 项通过、0 项失败、0 项跳过**，含新增的「数据集指纹与当前数据一致」 |
| 数据契约校验 | `python scripts/verify_data_contract.py` | **19 项通过、0 项失败、0 项跳过** |
| 自动化测试 | `python -m unittest discover -s tests` | `Ran 119 tests` → `OK` |
| 端到端重训 | `python scripts/train_baseline.py` | 退出码 0；三个模型指标与改造前**完全一致** |

### 1.1 指纹实现的交叉核对（关键证据）

变更记录第三节记录当前数据的数据集指纹为 `c9b76e66e05c96dd…`。
**M3 侧独立实现算出的值与之完全一致：**

```
我算出的指纹（src/models/baseline/artifacts.py）: c9b76e66e05c96ddde4923ffb3c5e39ae4ec074f1bbba6ee929a40e0681c95d5
变更记录记录的指纹                              : c9b76e66e05c96ddde4923ffb3c5e39ae4ec074f1bbba6ee929a40e0681c95d5
```

即契约文档给出的算法、`scripts/verify_model_contract.py` 的实现、
以及 M3 侧的实现在同一份真实数据上产出**同一个指纹**，
说明该定义不存在歧义、也**不依赖机器**（双方在不同环境中计算）。

三个划分文件各自的 SHA-256（供他人复核）：

| 文件 | SHA-256 |
| --- | --- |
| `train.csv` | `bdd8b161d887ccb71fac7024732e01d717b410d273f6ad5ad6bf8fb04a0bd172` |
| `dev.csv` | `757536c7559256aef0902f06407fbe156eef62298917de1716ca78f20207e079` |
| `test.csv` | `406750b7b7476a6fe9b09e6669544d294876ab9e2934add93c3894c0c5cfaa38` |

### 1.2 缺陷复现（确认 v1.0 确实不可满足）

在不改动任何清洗与划分逻辑的前提下连续运行两次 `scripts/split_data.py`
（划分由固定种子 42 保证确定性）：

| 文件 | 第一次 | 第二次 | 结论 |
| --- | --- | --- | --- |
| `train.csv` / `dev.csv` / `test.csv` | 见上表 | **完全相同** | 数据未变 |
| `split_manifest.json` | `10c21992152b2269…` | `32bb978b696032e2…` | **发生变化** |

三个 CSV 逐字节相同而清单哈希变化，唯一原因是 `created_at` 被更新。
因此 v1.0 的 `split_manifest_sha256` 条款**任何人都不可能稳定满足**，
属于规范缺陷而非实现问题——与变更记录的判断一致。
完整复现过程见 `reports/m3-validation.md` 第二节。

---

## 二、M3 侧所做改动

| 文件 | 改动 |
| --- | --- |
| `src/models/baseline/artifacts.py` | 新增 `dataset_fingerprint()`（按 §2.3 给定算法）；`METADATA_REQUIRED_KEYS` 以 `dataset_fingerprint` 取代 `split_manifest_sha256`；`build_metadata()` 改收 `processed_dir` |
| `scripts/train_baseline.py` | 元数据与对比报告改用数据集指纹 |
| `src/evaluation/metrics.py` | 对比报告第 1 节改为显示按 §2.3 计算的数据集指纹 |
| 三个 `models/*/metadata.json` | 字段改名并替换取值，其余字段（含 `created_at`、超参数、`device`）保持原值 |
| `tests/test_baseline_pipeline.py` | 新增 4 项指纹测试，含针对本缺陷的**回归保护** |
| `reports/m3-validation.md` | 新增，作为本勘误的 M3 验证记录 |

**回归保护**（防止他人回退到清单哈希）：

- `test_dataset_fingerprint_ignores_manifest_created_at`：改掉清单 `created_at` 后
  指纹必须不变；该测试同时断言清单哈希**本应**变化，以证明确实在测 v1.0 的那个缺陷。
- `test_dataset_fingerprint_survives_data_regeneration`：重新生成数据后指纹不变。
- `test_dataset_fingerprint_requires_all_three_splits`：缺少任一划分文件时明确报错。
- `test_metadata_records_the_current_dataset_fingerprint`：集成测试从 CSV 重算后比对，
  且断言 `metadata.json` 中**不得**再出现 `split_manifest_sha256`。

---

## 三、本次确认对 HuMingChang（M3）的约束

| 要求 | 规定 | 现状 |
| --- | --- | --- |
| `metadata.json` 字段 | 必填 `dataset_fingerprint`，不得使用 `split_manifest_sha256` | 三个模型已符合 |
| 指纹算法 | `sha256(":".join([sha256(train), sha256(dev), sha256(test)]))`，顺序固定 | 实现一致，已交叉核对 |
| 禁止事项 | 不得回退到 `split_manifest.json` 整体哈希 | 已加回归测试锁定 |
| 文档一致性 | M3 相关说明不得再引用 `split_manifest_sha256` 作为数据版本证据 | 已同步 README、研发计划、对比报告、日报 |

---

## 四、待办

| 编号 | 事项 | 负责人 | 说明 |
| ---: | --- | --- | --- |
| 待办 1 | 在 `reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md` 第七节的确认表中登记 HuMingChang 已确认 | LanQvJian | 本文件为数据方出具的确认记录；组长文件中的确认表由他维护，数据方未擅自改动 |
| 待办 2 | `README.md` 的「环境准备」一节仍写「推荐使用 Python 3.11」 | LanQvJian | 与 v1.0.1 无关，仅记录待确认；当前运行环境的实际版本见 `reports/environment.md` |
| 待办 3 | 变更记录第四节称「三个已训练模型更新元数据后重新验证，不需要重新训练」 | — | 数据方最终采取了**重训**：既因对比报告文本由训练脚本生成、需要一并刷新，也为了验证改动后的脚本仍可端到端跑通。重训后指标与改造前完全一致 |

---

## 五、记录

本确认记录由 HuMingChang 于 2026-09-21 出具，依据 `docs/contracts.md` §0 规定的变更流程
留存于 `reports/meeting-notes/`。

相关证据文件：`reports/m3-validation.md`（M3 独立验证与缺陷复现）、
`reports/m3-baseline-comparison.md`（模型对比）、
`reports/m3-baseline-results.json`（机器可读结果）、
`reports/daily-reports/2026-09-21-HuMingChang.md`（当日过程记录）。

v1.0 的确认记录（`2026-09-21-contract-confirmation-humingchang.md`）中
§2.3 相关描述由本版取代，原文保留作为历史记录不修改。
