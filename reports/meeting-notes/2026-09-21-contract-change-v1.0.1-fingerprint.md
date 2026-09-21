# 契约变更记录：v1.0 → v1.0.1（数据集指纹勘误）

日期：2026-09-21
提出人：LanQvJian
状态：**待 HuMingChang 与 HongSenYuan 确认后生效**
依据：`docs/contracts.md` §0 变更流程第 1–4 条

## 〇、为什么是 v1.0.1 而不是 v1.1

本次改动是对 v1.0 的一处**无法满足的缺陷**的勘误：v1.0 规定的
`split_manifest_sha256` 取自 `split_manifest.json` 整体，而该文件含 `created_at`，
任何人都**无法**产出一个能通过校验的值。这属于规范缺陷，不是政策选择，
因此**不占用 v1.1 版本号**。

`v1.1` 已由 HongSenYuan 的修订提案预定，用于情感任务产物约定、`device` 表述、
冲突重复处理规则三项修订（见 `2026-09-21-contract-v1.1-proposal.md`）。
两份变更互不影响，可分别表决、分别生效。

## 一、变更内容

仅改动 `docs/contracts.md` §2.3 一处，其余条款与 v1.0 完全相同。

| 项目 | v1.0 | v1.0.1 |
| --- | --- | --- |
| 字段名 | `split_manifest_sha256` | `dataset_fingerprint` |
| 取值来源 | `data/processed/split_manifest.json` 整体哈希 | train/dev/test 三个 CSV 的内容哈希 |
| 是否受 `created_at` 影响 | **是**（每次重新生成数据都会变） | 否 |

新指纹的计算方式（`train` → `dev` → `test` 顺序固定）：

```python
import hashlib
from pathlib import Path

def dataset_fingerprint(processed_dir: Path) -> str:
    parts = [
        hashlib.sha256((processed_dir / f"{split}.csv").read_bytes()).hexdigest()
        for split in ("train", "dev", "test")
    ]
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
```

## 二、变更原因

M3 独立验证（`reports/m3-validation.md`）发现阻断项：v1.0 的指纹取自
`split_manifest.json` 整体，而该文件含 `created_at`。即使重新生成的三个 CSV
字节完全一致，清单哈希仍会变化，模型被误判为"基于旧数据训练"。
由于 `data/processed/` 与模型产物都不提交 Git，其他成员按脚本复现数据后
**无法得到训练时的同一个清单哈希**，验证永远无法通过。

这不是实现方的错误，是 v1.0 契约给出的指纹定义本身不可满足。

## 三、实测证据

在真实数据上复现并验证（`created_at` 由 `14:35:19` 改为 `23:59:59`，
三个 CSV 字节未变）：

| 方法 | 变更前 | 变更后 | 结论 |
| --- | --- | --- | --- |
| 旧：清单哈希 | `72edce67c4083026…` | `d359bd47005706f2…` | 漂移，缺陷复现 |
| 新：数据集指纹 | `c9b76e66e05c96dd…` | `c9b76e66e05c96dd…` | 稳定，缺陷修复 |

当前 `data/processed/` 的数据集指纹为：

```
c9b76e66e05c96ddde4923ffb3c5e39ae4ec074f1bbba6ee929a40e0681c95d5
```

契约文档中的计算写法与 `scripts/verify_model_contract.py` 的实现，
在上述真实数据上产出**完全一致**的指纹。

## 四、影响范围与各人需做的改动

| 成员 | 需要改动 |
| --- | --- |
| HuMingChang（M3） | `metadata.json` 中的 `split_manifest_sha256` 改名为 `dataset_fingerprint`，取值改为按新方式计算。**三个已训练模型需更新元数据后重新验证，不需要重新训练** |
| HongSenYuan（M4/M5） | 按 v1.0.1 产出 `metadata.json`。尚未开始训练，无需返工 |
| LanQvJian（M6） | 已更新 `scripts/verify_model_contract.py`：改为从 `data/processed/` 的 CSV **重新计算**指纹再比对，不再读取 `split_manifest.json` |

`scripts/verify_model_contract.py` 现在会：

- 用新方式重算指纹并与模型记录值比对，不一致则判定失败；
- 把仍使用旧字段名的模型判定为「metadata 必需字段齐全」失败，并指出缺少 `dataset_fingerprint`。

## 五、验证方式

```powershell
python scripts/verify_model_contract.py models/<model_name>
```

## 六、对旧文档的处理

2026-09-21 的两份契约确认纪要（`2026-09-21-contract-confirmation-humingchang.md`、
`2026-09-21-contract-confirmation-hongsenyuan.md`）是对 **v1.0** 的确认，
其 §2.3 相关描述由本版取代，原文保留作为历史记录不修改。

## 七、确认

| 成员 | 确认状态 | 日期 |
| --- | --- | --- |
| LanQvJian | 已确认（提出人） | 2026-09-21 |
| HuMingChang | 待确认 | |
| HongSenYuan | 待确认 | |
