# M3 独立验证报告

- 验证对象：M3 传统模型基线（HuMingChang）
- 验证日期：2026-09-21
- 验证方式：由开发助手在真实数据上执行下列命令并整理结果，命令与输出均为实际运行结果
- 结论：**M3 交付通过契约校验；同时发现一处 v1.0 契约缺陷（阻断项），已由 v1.0.1 勘误解决**

本文件是 `docs/contracts.md` §2.3 与
`reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md`
所引用的 M3 验证记录。

---

## 一、验证项与结果

| 验证项 | 命令 | 实际结果 |
| --- | --- | --- |
| 模型契约校验 | `python scripts/verify_model_contract.py` | 三个模型各 **15 项通过、0 项失败、0 项跳过** |
| 数据契约校验 | `python scripts/verify_data_contract.py` | **19 项通过、0 项失败、0 项跳过** |
| 自动化测试 | `python -m unittest discover -s tests` | `Ran 119 tests` → `OK`（M3 相关 59 项） |
| 训练可复现性 | 两次独立运行 `scripts/train_baseline.py` | 三个模型的 Macro-F1 **完全一致**，耗时 389.6 秒 |
| 推理路径一致性 | 命令行与训练报告对比 | `scripts/predict_*.py --eval test` 与 `reports/m3-baseline-comparison.md` 数字一致 |
| 可加载性 | 统一入口 `src.inference.load_predictor` | 三个产物均可加载，概率和为 1 且覆盖全部 10 类 |

### 1.1 模型指标（测试集）

| 模型 | 验证集 Macro-F1 | 测试集 Macro-F1 | 模型大小 | 单条延迟 |
| --- | ---: | ---: | ---: | ---: |
| `tfidf_svm`（选定） | 0.8923 | 0.8883 | 105.98 MB | 3.56 ms |
| `tfidf_logreg` | 0.8721 | 0.8777 | 44.94 MB | 0.62 ms |
| `tfidf_rf` | 0.8165 | 0.8281 | 900.46 MB | 72.18 ms |

---

## 二、发现的阻断项：v1.0 的数据集指纹无法满足

### 2.1 现象

按 v1.0 §2.3 的要求，模型 `metadata.json` 记录 `split_manifest_sha256`
（`data/processed/split_manifest.json` 的整体哈希）。
**重新生成同一份数据后，该哈希会变化，导致已训练模型被判定为「基于旧数据训练」。**

### 2.2 复现过程与实测数据

在不改动任何清洗或划分逻辑的前提下连续运行两次 `scripts/split_data.py`，
划分由固定种子 42 保证确定性：

| 文件 | 第一次运行 | 第二次运行 | 结论 |
| --- | --- | --- | --- |
| `train.csv` | `bdd8b161d887ccb7…` | `bdd8b161d887ccb7…` | 字节相同 |
| `dev.csv` | `757536c7559256ae…` | `757536c7559256ae…` | 字节相同 |
| `test.csv` | `406750b7b7476a6f…` | `406750b7b7476a6f…` | 字节相同 |
| `split_manifest.json` | `10c21992152b2269…` | `32bb978b696032e2…` | **发生变化** |

三个划分文件逐字节相同，而清单哈希发生了变化——唯一原因是清单里的 `created_at`
被更新为新的运行时间。因此 v1.0 的指纹条款**任何人都不可能稳定满足**：
它衡量的不是数据内容，而是「清单文件最后一次生成的时间」。

后果：新成员按脚本复现数据后无法得到训练时的同一个清单哈希，
`scripts/verify_model_contract.py` 会持续报「该模型可能是在旧数据上训练的」，
**M3 无法通过验收**。M3 交付当时三个模型的 `metadata.json` 中该字段的实测值
（`32bb978b…`）与校验器实时重算值（`10c21992…`）确实不一致。

### 2.3 这不是实现方的错误

数据集指纹条款由 v1.0 规定，实现侧（`src/models/baseline/artifacts.py`）
是**按契约字面实现**的：在写出时从磁盘实时计算清单哈希。
缺陷在契约给出的指纹定义本身不可满足，故不应通过放宽实现或忽略校验来绕过。

### 2.4 处置

LanQvJian 提出 v1.0.1 勘误，把指纹改为由 train/dev/test 三个 CSV 的内容哈希计算，
并明确禁止回退到清单哈希。变更记录见
`reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md`。

---

## 三、v1.0.1 后的重新验证

### 3.1 实现改动

| 文件 | 改动 |
| --- | --- |
| `src/models/baseline/artifacts.py` | 新增 `dataset_fingerprint()`；`METADATA_REQUIRED_KEYS` 用 `dataset_fingerprint` 取代 `split_manifest_sha256`；`build_metadata()` 改收 `processed_dir` |
| `scripts/train_baseline.py` | 元数据与对比报告改用数据集指纹 |
| `src/evaluation/metrics.py` | 对比报告第 1 节显示按契约 §2.3 计算的数据集指纹 |
| 三个 `models/*/metadata.json` | 字段改名并替换取值（按变更记录第四节，**未重新训练**） |
| `tests/test_baseline_pipeline.py` | 新增 4 项指纹测试，含针对本缺陷的回归保护 |

### 3.2 指纹一致性与稳定性

| 检查 | 结果 |
| --- | --- |
| 实现与契约给定算法一致 | 同一份数据上，本实现算出 `c9b76e66e05c96dd…`，与变更记录第三节记录的值**完全一致** |
| 清单 `created_at` 改变后指纹是否漂移 | **不漂移**（`test_dataset_fingerprint_ignores_manifest_created_at`） |
| 重新生成同一份数据后指纹是否漂移 | **不漂移**（`test_dataset_fingerprint_survives_data_regeneration`） |
| 缺少任一划分文件时 | 明确报错，不产出含义不明的指纹 |

### 3.3 重新验收结果

```text
模型契约校验：models/tfidf_logreg  通过 15 项，失败 0 项，跳过 0 项
模型契约校验：models/tfidf_svm     通过 15 项，失败 0 项，跳过 0 项
模型契约校验：models/tfidf_rf      通过 15 项，失败 0 项，跳过 0 项
结论：模型契约校验通过。
```

其中「数据集指纹与当前数据一致」一项由校验器**从 CSV 重新计算**后比对，
不读取 `split_manifest.json`，与契约 §2.3 的要求一致。

---

## 四、本次验证未覆盖的范围

1. **单条评论的预测正确性不在校验范围内。** 契约 §2.5 只要求 `predict_proba`
   覆盖全部 10 类且 `predict` 与最高概率一致（结构合法），不校验预测是否判对。
   实测中契约自带的探针文本「手机很好用，电池耐用」被模型判为「平板」，
   这是已知的词袋模型局限，见第五节。
2. **`reports/figures/` 下的 EDA 图表未做人工目视检查**，仅做了程序化校验
   （文件有效性、无缺字形警告）。
3. **模型优化（量化/蒸馏/剪枝）与情感二分类属 M5 范围**，本报告不涉及。

---

## 五、M3 已知局限（报告中已如实说明）

1. **「蒙牛」类别的指标被词面捷径抬高。** 该类别 203/203 条测试评论全部直接包含
   「蒙牛」二字，而其他类别没有任何一条提及它，因此其 Precision/Recall/F1 全为 1.0000
   并不代表模型理解了「乳制品」类别。
2. **模型对明确提到「手机」的短评论会高置信度判为「平板」。** 根因是训练数据中
   「手机」出现在平板评论里的绝对次数多于手机评论（1,085 : 499），
   词袋模型无法区分「手机很好用」与「比手机屏幕大」。这是传统模型的固有局限，
   也是 M4 采用上下文表示的动机之一。
3. **字符级 2-3 gram 的词表触到 40 万上限**；随机森林产物 900 MB，不适合部署。
4. **scikit-learn 无 GPU 后端**，本机虽有可用 CUDA 设备，三个模型的
   `metadata.device` 仍须如实记为 `cpu`。
