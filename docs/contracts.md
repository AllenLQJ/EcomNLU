# 接口契约（强制）

版本：v1.0
生效日期：2026-09-21
维护人：LanQvJian

## 0. 本文件的效力

本项目三人并行开发，模块之间存在三处交接。本文件把这三处交接的格式**固定下来**，
使各模块可以独立开发而不必互相等待。

**本文件是强制约束，不是建议。** 任何与本文件不一致的产物都视为未完成，不得合并进 `main`。

变更流程：

1. 提出人说明改动内容和原因。
2. 通知另外两名成员并获得同意。
3. 更新本文件并提升版本号。
4. 在 `reports/meeting-notes/` 记录该次变更。

未经上述流程，任何人不得私自更改文件名、列名、字段名或 JSON 结构。

### 谁受哪份契约约束

| 契约 | 产出方 | 使用方 | 强制 |
| --- | --- | --- | --- |
| 一、数据契约 | HuMingChang（M2） | HongSenYuan（M4）、LanQvJian（M6） | 是 |
| 二、模型产物契约 | HuMingChang（M3）、HongSenYuan（M4/M5） | LanQvJian（M6）、HongSenYuan（M5 对比） | 是 |
| 三、API 契约 | LanQvJian（M6） | 前端页面、答辩演示 | 是 |

---

## 一、数据契约

**交接**：HuMingChang 的 M2 产物 → 所有下游模型训练。

### 1.1 目录与文件

```text
data/processed/
├── train.csv
├── dev.csv
├── test.csv
├── label_mapping.json
└── split_manifest.json
```

文件名固定，不得使用 `train_set.csv`、`training.csv` 等其它写法。

### 1.2 CSV 格式

| 项目 | 规定 |
| --- | --- |
| 编码 | UTF-8，**不带 BOM** |
| 表头 | 必须有，且列顺序固定为 `sample_id,review,cat,label` |
| 索引列 | 禁止写出 pandas 的行索引 |
| 换行 | 使用 `\n`；字段内的换行由 CSV 引号机制处理 |

各列含义：

| 列 | 类型 | 规定 |
| --- | --- | --- |
| `sample_id` | 整数 | 该记录在**原始 CSV** 中的行序号（不含表头，从 0 开始）。用于把任意处理后样本追溯回原始数据 |
| `review` | 字符串 | 已清洗的评论文本。不得有首尾空白，不得为纯空白，不得为空 |
| `cat` | 字符串 | **中文类别名**，取值必须是 `configs/base.yaml` 中 `task.categories` 的 10 类之一 |
| `label` | 整数 | 情感标签，只能是 `0`（负向）或 `1`（正向） |

**`cat` 使用中文类别名，不使用整数 id。** 整数映射交由 `label_mapping.json` 处理，
以保持处理后数据可读、可与原始数据直接对照。

模型训练代码必须**忽略 `sample_id` 列**，不得把它当作特征。

### 1.3 label_mapping.json

```json
{
  "categories": ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"],
  "label2id": {"书籍": 0, "平板": 1, "手机": 2, "水果": 3, "洗发水": 4, "热水器": 5, "蒙牛": 6, "衣服": 7, "计算机": 8, "酒店": 9},
  "id2label": {"0": "书籍", "1": "平板", "2": "手机", "3": "水果", "4": "洗发水", "5": "热水器", "6": "蒙牛", "7": "衣服", "8": "计算机", "9": "酒店"}
}
```

规定：

- `categories` 的顺序**必须与 `configs/base.yaml` 中 `task.categories` 完全一致**，顺序即 id 分配依据。
- id 从 0 到 9，与 `categories` 下标一一对应，即必须满足 `label2id[类别] == categories.index(类别)`。
- 本文件一旦生成即冻结。混淆矩阵、分类报告的类别顺序全部依赖它。

> 上例是标准内容，生成脚本必须输出与之完全一致的结果。若脚本输出与此不符，以本文件为准并检查脚本。

### 1.4 split_manifest.json

```json
{
  "created_at": "2026-09-21T10:05:00+08:00",
  "random_seed": 42,
  "source": {
    "path": "data/raw/online_shopping_10_cats.csv",
    "sha256": "f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb"
  },
  "cleaning": {
    "rules": {
      "empty_review": "drop",
      "duplicate_review": "<实际采用的规则>",
      "control_chars": "<实际采用的规则>",
      "short_text_threshold": "<实际采用的阈值>"
    },
    "rows_before": 62774,
    "rows_after": "<清洗后行数>"
  },
  "split": {
    "strategy": "stratified_by_cat",
    "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
    "dedup_key": "<raw_text 或 normalized_text>"
  },
  "counts": {"train": "<整数>", "dev": "<整数>", "test": "<整数>"},
  "files": {
    "train.csv": {"sha256": "<64位十六进制>", "rows": "<整数>"},
    "dev.csv": {"sha256": "<64位十六进制>", "rows": "<整数>"},
    "test.csv": {"sha256": "<64位十六进制>", "rows": "<整数>"}
  },
  "leakage_check": {
    "based_on": "normalized_text",
    "train_dev_overlap": 0,
    "train_test_overlap": 0,
    "dev_test_overlap": 0
  }
}
```

规定：

- `<...>` 是占位符，**必须替换为生成脚本输出的真实值，不得保留占位符**。
- 三个 `overlap` **必须是 `0`**（不是占位符，是硬性要求）。任何一项不为 0，说明划分存在跨集合泄漏，数据不合格，必须重做（AGENTS.md §3.1「避免同一文本同时进入训练集和测试集」、§12.4）。
- `files` 中的 `sha256` 是该 CSV 文件本身的哈希，用于证明所有模型使用了同一份数据。
- `cleaning.rules` 必须如实记录实际执行的规则。若某项未处理，写 `"none"`，不得留空或省略。

### 1.5 数据契约的验证方式

下游拿到 `data/processed/` 后，必须先验证再训练：

```powershell
python scripts/verify_data_contract.py
```

未通过验证的数据不得用于训练。

---

## 二、模型产物契约

**交接**：各模型训练方 → LanQvJian 的部署代码，以及 M5 的模型对比。

### 2.1 目录结构

每个模型产出一个独立目录，命名规范为小写字母加下划线：

```text
models/<model_name>/
```

允许的 `model_name`：

| model_name | 对应 |
| --- | --- |
| `tfidf_logreg` | TF-IDF + 逻辑回归 |
| `tfidf_svm` | TF-IDF + LinearSVC |
| `tfidf_rf` | TF-IDF + 随机森林 |
| `fasttext_char` | 字符级 FastText |
| `fasttext_word` | 词级 FastText |
| `bert_base_chinese` | BERT 微调 |
| `<model_name>_<优化方式>` | M5 优化产物，如 `bert_base_chinese_quantized` |

无论何种模型，目录下**必须**包含以下四个部分：

```text
models/<model_name>/
├── <模型本体>
├── label_mapping.json     # 内容与 data/processed/label_mapping.json 完全一致
├── metadata.json
└── metrics.json
```

`label_mapping.json` 必须与数据契约中的那份**逐字节相同**（直接复制，不得重新生成）。

### 2.2 模型本体的形式

| 模型类型 | 必须保存为 |
| --- | --- |
| scikit-learn 类 | `model.joblib`，以及特征处理器 `vectorizer.joblib` |
| FastText | `model.bin` |
| BERT / Transformers | `model/` 目录（`save_pretrained`）与 `tokenizer/` 目录（`save_pretrained`） |

**特征处理器必须与模型一起保存。** 推理阶段必须使用与训练阶段完全相同的预处理流程
（AGENTS.md §5.2）。只保存模型、推理时重新 fit 向量器，视为不合格。

### 2.3 metadata.json

```json
{
  "model_name": "tfidf_logreg",
  "task": "category_classification",
  "framework": "scikit-learn",
  "framework_version": "1.7.2",
  "created_at": "2026-09-21T14:30:00+08:00",
  "random_seed": 42,
  "split_manifest_sha256": "<data/processed/split_manifest.json 的 sha256>",
  "train_rows": 0,
  "hyperparameters": {},
  "max_length": null,
  "device": "cpu",
  "notes": ""
}
```

规定：

- 上例中的 `0`、`""`、`{}` 均为待填占位，必须替换为真实值。
- `split_manifest_sha256` 必填。它是「所有模型使用同一份无泄漏测试集」的**唯一硬证据**（AGENTS.md §12.4）。
- `train_rows` 填实际参与训练的行数，不含 dev 与 test。
- `max_length` 仅 BERT 类模型填写实际值，其余填 `null`。该值必须由本数据集 token 长度分布决定，不得照搬其它项目的 32（AGENTS.md §5.4）。
- `device` 记录实际训练设备。本项目当前为纯 CPU，填 `"cpu"`。
- `hyperparameters` 必须记录完整到可以复现训练。

### 2.4 metrics.json

```json
{
  "model_name": "tfidf_logreg",
  "categories": ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"],
  "dev": {
    "accuracy": 0.0,
    "macro_precision": 0.0,
    "macro_recall": 0.0,
    "macro_f1": 0.0,
    "weighted_f1": 0.0,
    "per_class": {
      "书籍": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    },
    "confusion_matrix": []
  },
  "test": null,
  "cost": {
    "model_size_bytes": 0,
    "single_sample_latency_ms": 0.0,
    "batch_32_latency_ms": 0.0,
    "throughput_per_sec": 0.0,
    "measured_on": "cpu",
    "measured_at": ""
  }
}
```

规定：

- 上例中的 `0`、`0.0`、`[]`、`""` 均为待填占位。**所有数值必须来自真实评估运行，不得编造或估计**（AGENTS.md §6、§7.4）。未评估的集合填 `null`。
- 五个指标名固定为 `accuracy`、`macro_precision`、`macro_recall`、`macro_f1`、`weighted_f1`，与 `configs/base.yaml` 的 `evaluation.report_metrics` 一致。
- `confusion_matrix` 为 10×10 二维数组，**行是真实类别、列是预测类别**，顺序严格按 `categories`。
- **`test` 在选模阶段必须保持为 `null`。** 只有最终选定模型后才允许评估测试集并填入。
  中途用测试集调参或选模违反 AGENTS.md §6，视为数据泄漏。
- `cost` 中的 `model_size_bytes` 与延迟为 AGENTS.md §6 对比表的必填项，不得事后估计，必须实测。

### 2.5 统一加载接口

所有模型必须能通过**同一个入口**加载，这是 M6 能用一段代码服务多个模型的前提。

> **状态说明（2026-09-21）**：`src/inference/` 由 LanQvJian 在 M6 阶段实现，当前尚未写出。
> 但**接口签名已经冻结**，模型训练方必须按下面的签名适配自己的模型，不要等待实现完成再动手。
> 接口实现后，`scripts/verify_model_contract.py` 的「统一入口可加载并推理」一项才会从
> SKIP 变为真实校验。

接口定义如下（实现位于 `src/inference/`）：

```python
def load_predictor(model_dir: Path) -> CategoryPredictor:
    """按模型目录加载为统一的预测器。"""

class CategoryPredictor(Protocol):
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """返回每个样本的 {中文类别名: 概率}，必须包含全部 10 类。"""

    def predict(self, texts: list[str]) -> list[str]:
        """返回每个样本的预测中文类别名。"""
```

规定：

- `predict_proba` 的返回**必须覆盖全部 10 个类别**，未出现的类别补 0，不得只返回 top-k。
- 键必须是中文类别名，不是整数 id。
- 每个模型负责把自己的原生输出适配到该接口。适配代码放在 `src/inference/`。

### 2.6 模型契约的验证方式

```powershell
python scripts/verify_model_contract.py models/<model_name>
```

校验必需文件、字段名、类别顺序、`predict_proba` 覆盖全部类别，并执行一次真实推理。
未通过验证的模型不得用于部署或对比。

---

## 三、API 契约

**交接**：LanQvJian 的后端 → 前端页面与答辩演示。

### 3.1 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/categories` | 返回 10 个类别，供前端展示 |
| POST | `/api/predict` | 单条评论预测 |

### 3.2 GET /api/health

成功（200）：

```json
{
  "status": "ok",
  "model_name": "bert_base_chinese",
  "model_loaded": true,
  "sentiment_enabled": false,
  "categories": ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]
}
```

模型未加载时返回 503，`status` 为 `"degraded"`，`model_loaded` 为 `false`。

### 3.3 GET /api/categories

成功（200）：

```json
{
  "categories": ["书籍", "平板", "手机", "水果", "洗发水", "热水器", "蒙牛", "衣服", "计算机", "酒店"]
}
```

- 响应顶层必须是 JSON 对象，唯一字段为 `categories`。
- 类别顺序必须与 `configs/base.yaml` 和 `label_mapping.json` 一致。
- 该端点不依赖模型是否已加载；只要服务可用就返回 200。

### 3.4 POST /api/predict

请求：

```json
{"review": "手机很好用，电池耐用", "top_k": 3}
```

- `review`：必填，字符串。
- `top_k`：可选，整数，默认 3，取值范围 1–10。

成功（200）：

```json
{
  "category": {"label": "手机", "confidence": 0.93},
  "top_k": [
    {"label": "手机", "probability": 0.93},
    {"label": "平板", "probability": 0.04},
    {"label": "计算机", "probability": 0.01}
  ],
  "sentiment": null,
  "elapsed_ms": 12.4,
  "model_name": "bert_base_chinese"
}
```

规定：

- `top_k` 数组按概率降序，长度等于请求的 `top_k`。
- **`sentiment` 在 M5 完成前固定为 `null`，不得省略该字段**，以保证前端结构稳定。
  M5 完成后其结构为 `{"label": "正向", "confidence": 0.97}`，取值只能是 `正向` 或 `负向`。
- `elapsed_ms` 为服务端推理耗时，不含网络时间。
- 任何时候都不得返回缺少 `category` 字段的 200 响应。

### 3.5 错误响应

所有错误使用统一结构：

```json
{"error": {"code": "EMPTY_REVIEW", "message": "评论内容不能为空"}}
```

错误码表：

| code | HTTP | 触发条件 |
| --- | ---: | --- |
| `INVALID_JSON` | 400 | 请求体不是合法 JSON 对象 |
| `MISSING_REVIEW` | 400 | 缺少 `review` 字段 |
| `EMPTY_REVIEW` | 400 | `review` 为空串或纯空白 |
| `REVIEW_TOO_LONG` | 400 | `review` 长度超过上限 |
| `INVALID_TOP_K` | 400 | `top_k` 不是 1–10 的整数 |
| `MODEL_NOT_LOADED` | 503 | 模型尚未加载完成 |
| `INTERNAL_ERROR` | 500 | 其它未预期异常 |

`review` 长度上限由配置项 `api.max_review_chars` 决定，并在 README 中写明。
超限时必须返回 `REVIEW_TOO_LONG`，不得静默截断。

### 3.6 服务端约定

- 模型在服务启动时加载**一次**，请求处理中不得重新加载（AGENTS.md §5.7）。
- 异常必须被捕获并转换为上表的错误码，不得把 Python 堆栈直接返回给前端。
- 启动方式与访问地址必须写入 `README.md`。

---

## 四、契约与文档的对应关系

| 本文件章节 | 对应的既有要求 |
| --- | --- |
| 1.2 `cat` 用中文 | 本文件新增约定 |
| 1.4 泄漏检查为 0 | AGENTS.md §3.1、§12.4 |
| 2.1 四件套 | AGENTS.md §7.3「标签映射必须与模型一起保存」 |
| 2.2 特征处理器一起保存 | AGENTS.md §5.2 |
| 2.3 `split_manifest_sha256` | AGENTS.md §12.4「相同且无泄漏的最终测试集」 |
| 2.4 `test` 保持 null | AGENTS.md §6「测试集只用于最终确认」 |
| 2.4 `cost` 字段 | AGENTS.md §5.6、§6 对比表 |
| 3.5 错误码 | AGENTS.md §5.7「对空输入、超长输入和异常请求给出清晰错误」 |
| 3.6 只加载一次模型 | AGENTS.md §5.7 |

## 五、生效确认

本契约由 LanQvJian 起草，**需经 HuMingChang 与 HongSenYuan 确认后正式生效**。
确认前如对任何条款有异议，应立即提出并记录在 `reports/meeting-notes/`。

| 成员 | 确认状态 | 日期 |
| --- | --- | --- |
| LanQvJian | 已确认（起草人） | 2026-09-21 |
| HuMingChang | 待确认 | |
| HongSenYuan | 待确认 | |
