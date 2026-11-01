# EcomNLU（智购感知）接口契约（强制）

版本：v1.3
首次生效：2026-09-21
本版生效：2026-09-21（三名成员已确认 v1.2 修订）
维护人：LanQvJian

> 2026-10-31：项目名称更新为 EcomNLU（智购感知），默认服务端口为 8001。
> 既有 v1.2 请求与响应契约继续有效。2026-11-01 起新增 v1.3 的 `POST /api/v1/nlu/analyze`（见 §3.6）。

## 变更记录

| 版本 | 日期 | 变更 | 原因 |
| --- | --- | --- | --- |
| v1.0 | 2026-09-21 | 首次发布 | — |
| v1.0.1 | 2026-09-21 | §2.3：`split_manifest_sha256` 改为 `dataset_fingerprint`，由 train/dev/test 三个 CSV 的内容哈希计算 | v1.0 的指纹取自 `split_manifest.json` 整体，该文件含 `created_at`，每次重新生成数据都会变化。即使三个 CSV 字节完全一致，模型仍会被判定为"基于旧数据训练"，导致 M3 无法通过验收（见 `reports/m3-validation.md` 阻断项） |
| v1.1 | 2026-09-21 | §2.3：`device` 改为记录实际使用的 `cpu` 或 `cuda` | M4 可使用已验证的 CUDA 环境训练，契约不得把设备错误固定为 CPU |
| v1.3 | 2026-11-01 | 新增 §3.6 `POST /api/v1/nlu/analyze`（子句级类别提示 + 整段情感）与错误码 `MISSING_TEXT` / `EMPTY_TEXT` / `TEXT_TOO_LONG`；映射配置见 `configs/base.yaml` 的 `analyze` 段 | EcomPilot / EcomOps 需要按子句取类别提示，而不是单条评论的 top-k；不改动 v1.2 的任何既有端点与字段 |
| v1.2 | 2026-09-21 | 固定 M2 冲突重复处理规则；新增情感任务的模型命名、标签映射、指标维度、统一推理与配置约定 | M5 情感模型必须能在不破坏主任务契约的前提下验收和部署 |

> **v1.0.1 是 v1.0 的勘误（erratum），不是功能扩展。**
> v1.0 规定的指纹条款任何人都无法稳定通过重新生成校验，属于规范缺陷，因此当时没有占用 v1.1 版本号。
>
> v1.1 已通过 HongSenYuan 提出、HuMingChang 同意、LanQvJian 批准的设备条款修订。
> v1.2 的冲突重复处理规则与情感任务产物约定已由 LanQvJian、HuMingChang、
> HongSenYuan 三人同意并正式生效，确认记录见
> `reports/meeting-notes/2026-09-21-contract-v1.2-clause-draft.md`。
>
> v1.0.1 仅改动 §2.3 的数据集指纹定义，其余条款与 v1.0 完全相同。
> 2026-09-21 的两份契约确认纪要以 v1.0 为准，其 §2.3 相关描述由本版取代。

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
- `cleaning.rules.duplicate_review` 固定为 `"drop_within_category_only"`：只在同一类别内去除重复文本；跨类别的相同文本保留。
- `split.dedup_key` 固定为 `"normalized_text"`，并且划分必须按该键分组；同一归一化文本的全部记录必须进入同一个划分。
- `leakage_check.based_on` 固定为 `"normalized_text"`，三组跨集合交集必须全部为 0。不得只做类别分层而忽略文本分组。

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
| `sentiment_tfidf_logreg` | 情感二分类 · TF-IDF + 逻辑回归 |
| `sentiment_tfidf_svm` | 情感二分类 · TF-IDF + 校准 LinearSVC |
| `sentiment_fasttext_word` | 情感二分类 · 词级 FastText |
| `sentiment_bert_base_chinese` | 情感二分类 · BERT 微调 |
| `<model_name>_<优化方式>` | M5 优化产物，如 `bert_base_chinese_quantized` |

无论何种模型，目录下**必须**包含以下四个部分：

```text
models/<model_name>/
├── <模型本体>
├── label_mapping.json     # 内容按 metadata.task 确定
├── metadata.json
└── metrics.json
```

`label_mapping.json` 按 `metadata.task` 分两种情况：

- `category_classification`：必须与 `data/processed/label_mapping.json` **逐字节相同**。
- `sentiment_classification`：`categories` 固定为 `["负向", "正向"]`，`label2id`
  固定为 `{"负向": 0, "正向": 1}`，`id2label` 固定为
  `{"0": "负向", "1": "正向"}`；取值来源为
  `configs/base.yaml` 的 `task.sentiment_categories`。原始 `label` 中 0 表示负向、1 表示正向，顺序不得调换。

一个模型目录只能承载一个任务。情感模型统一使用 `sentiment_` 前缀，多任务产物不得与单任务产物共用目录。

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
  "dataset_fingerprint": "<64位十六进制>",
  "train_rows": 0,
  "hyperparameters": {},
  "max_length": null,
  "device": "cpu",
  "notes": ""
}
```

规定：

- 上例中的 `0`、`""`、`{}` 均为待填占位，必须替换为真实值。
- **`dataset_fingerprint` 必填。** 它标识模型训练时所用的数据集，是「所有模型使用同一份无泄漏测试集」的**唯一硬证据**（AGENTS.md §12.4）。
- 计算方式固定如下，`train` → `dev` → `test` 的**顺序不得调换**：

  ```python
  import hashlib
  from pathlib import Path

  def dataset_fingerprint(processed_dir: Path) -> str:
      """按 docs/contracts.md §2.3 计算数据集指纹。"""
      parts = [
          hashlib.sha256((processed_dir / f"{split}.csv").read_bytes()).hexdigest()
          for split in ("train", "dev", "test")
      ]
      return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
  ```

  即：先分别取三个 CSV 文件的 SHA-256（小写十六进制），按 `train:dev:test` 用半角冒号连接，
  再对该字符串取 SHA-256。

- `scripts/verify_model_contract.py` 会用同样的方式，**从 `data/processed/` 的 CSV 文件重新计算**指纹，
  再与模型记录的值比对。校验不依赖 `split_manifest.json` 或任何其它元数据文件。
- **禁止改用 `split_manifest.json` 的整体哈希。** 该文件含 `created_at`，每次重新生成数据都会变化；
  即使三个 CSV 字节完全一致，指纹仍会漂移，模型会被误判为基于旧数据训练。
  这是 v1.0 的缺陷（见 `reports/m3-validation.md`），不得回退。
- `train_rows` 填实际参与训练的行数，不含 dev 与 test。
- `max_length` 仅 BERT 类模型填写实际值，其余填 `null`。该值必须由本数据集 token 长度分布决定，不得照搬其它项目的 32（AGENTS.md §5.4）。
- `device` 必须记录实际使用的训练设备，取 `"cpu"` 或 `"cuda"`；不得填写未实际
  使用的设备。若使用 CUDA，还必须在运行环境记录中注明 Python、PyTorch、CUDA
  版本和 GPU 型号，以便复现实验。
- `hyperparameters` 必须记录完整到可以复现训练。
- `task` 只能是 `category_classification` 或 `sentiment_classification`。

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
- `confusion_matrix` 的行列数必须等于 `categories` 长度：主任务为 10×10，情感任务为 2×2。**行是真实标签、列是预测标签**，顺序严格按 `categories`；`per_class` 的键也必须与 `categories` 一致。
- **`test` 在选模阶段必须保持为 `null`。** 只有最终选定模型后才允许评估测试集并填入。
  中途用测试集调参或选模违反 AGENTS.md §6，视为数据泄漏。
- `cost` 中的 `model_size_bytes` 与延迟为 AGENTS.md §6 对比表的必填项，不得事后估计，必须实测。
- 情感任务可在顶层增加 `per_product_category`，记录各商品类别内的情感指标；该字段为可选扩展，不属于四件套必填结构。

### 2.5 统一加载接口

所有模型必须能通过**同一个入口**加载，这是 M6 能用一段代码服务多个模型的前提。

> **状态说明（2026-09-21）**：`src/inference/` 已实现统一入口以及 scikit-learn、
> FastText、Transformers 三类适配器；模型训练方必须按下面的冻结签名交付模型。
> `scripts/verify_model_contract.py` 已通过统一入口执行真实加载与推理校验。

接口定义如下（实现位于 `src/inference/`）：

```python
def load_predictor(model_dir: Path) -> CategoryPredictor:
    """按模型目录加载为统一的预测器。"""

class CategoryPredictor(Protocol):
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """返回每个样本的 {中文标签名: 概率}，必须包含该模型全部标签。"""

    def predict(self, texts: list[str]) -> list[str]:
        """返回每个样本的预测中文标签名。"""
```

规定：

- `predict_proba` 的返回必须覆盖模型 `label_mapping.json` 中的**全部标签**（主任务 10 类、情感任务 2 类），未出现的标签补 0，不得只返回 top-k。
- 键必须是 `label_mapping.json` 中的中文标签名，不是整数 id。
- `load_predictor` 按目录加载，标签集取自该目录的 `label_mapping.json`；不新增仅名称不同的情感预测协议。
- 每个模型负责把自己的原生输出适配到该接口。适配代码放在 `src/inference/`。

`configs/base.yaml` 的 `task.sentiment_categories` 固定为 `["负向", "正向"]`，作为情感标签顺序的唯一配置来源。

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
| POST | `/api/v1/nlu/analyze` | 子句级类别提示 + 整段情感（v1.3，见 §3.7） |

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
| `MISSING_TEXT` | 400 | analyze：缺少 `text` 字段或类型错误（v1.3） |
| `EMPTY_TEXT` | 400 | analyze：`text` 为空串或纯空白（v1.3） |
| `TEXT_TOO_LONG` | 400 | analyze：`text` 超过 `analyze.max_text_chars`（v1.3） |
| `MODEL_NOT_LOADED` | 503 | 模型尚未加载完成 |
| `INTERNAL_ERROR` | 500 | 其它未预期异常 |

`review` 长度上限由配置项 `api.max_review_chars` 决定，并在 README 中写明。
超限时必须返回 `REVIEW_TOO_LONG`，不得静默截断。

### 3.6 服务端约定

- 模型在服务启动时加载**一次**，请求处理中不得重新加载（AGENTS.md §5.7）。
- 异常必须被捕获并转换为上表的错误码，不得把 Python 堆栈直接返回给前端。
- 启动方式与访问地址必须写入 `README.md`。

---


### 3.7 POST /api/v1/nlu/analyze（v1.3 新增）

供上游把一段用户文本拆成子句，逐句取类别提示，并取整段情感。

请求：

```json
{"text": "热水器漏水了，另外牛奶要一箱", "scene_type": "aftersales"}
```

- `text`：必填，字符串，去首尾空白后不能为空，长度上限见 `configs/base.yaml` 的 `analyze.max_text_chars`（默认 500）。
- `scene_type`：可选，默认 `AFTER_SALE_FEEDBACK`。`analyze.not_applicable_scenes`（默认 `PDF_BODY`、`CS_APOLOGY`）内的场景不调用模型，直接返回 `scene_not_applicable=true`；`analyze.priority_boost_scenes`（默认 `AFTER_SALE_FEEDBACK`、`aftersales`）内且情感为负向、`sentiment_score >= analyze.priority_boost_min_score`（默认 0.85）时 `priority_boost=true`；得分不够的负向只标 `NEGATIVE`，不加急。

成功（200）：

```json
{
  "clauses": [
    {"text": "热水器漏水了", "predicted_label": "热水器", "category_confidence": 0.93,
     "category_code": "WATER_HEATER", "catalog_category_hint": "HOME_APPLIANCE_HEATER",
     "brand_hint": null, "item_type": "PRODUCT"}
  ],
  "sentiment": "NEGATIVE", "sentiment_score": 0.97,
  "priority_boost": true, "scene_not_applicable": false, "model_name": "bert_base_chinese"
}
```

- 子句按 `，。！？；、和以及顺便还要` 与换行切分，最多 `analyze.max_clauses`（默认 5）个。整段和所有子句在同一批里推理（模型只前向一次）；情感只对整段做一次。
- 子句置信度低于 `analyze.confidence_threshold`（默认 0.65）时 `predicted_label="ABSTAIN"`，`category_code`、`catalog_category_hint`、`brand_hint` 均为 `null`；上游应把它当作"没有提示"。
- `category_code` / `catalog_category_hint` / `brand_hint` / `item_type` 来自 `analyze.categories` 的配置映射；类别没有对应目录品类时 `catalog_category_hint` 为 `null`。
- `sentiment` 取 `POSITIVE` / `NEGATIVE` / `NEUTRAL`；情感模型未配置时为 `NEUTRAL`、`sentiment_score=0`，**不猜**。情感针对整段文本，不是逐子句。
- 错误（400，同 §3.4 结构）：`MISSING_TEXT`（缺 `text` 或类型错误）、`EMPTY_TEXT`、`TEXT_TOO_LONG`、`INVALID_JSON`；模型未加载 `503 MODEL_NOT_LOADED`（不适用场景不需要模型，仍返回 200）。
- 底层是用电商评论训练的模型，对导购需求、售后吐槽这类文本的效果未经验证。

## 四、契约与文档的对应关系

| 本文件章节 | 对应的既有要求 |
| --- | --- |
| 1.2 `cat` 用中文 | 本文件新增约定 |
| 1.4 泄漏检查为 0 | AGENTS.md §3.1、§12.4 |
| 2.1 四件套 | AGENTS.md §7.3「标签映射必须与模型一起保存」 |
| 2.2 特征处理器一起保存 | AGENTS.md §5.2 |
| 2.3 `dataset_fingerprint` | AGENTS.md §12.4「相同且无泄漏的最终测试集」 |
| 2.4 `test` 保持 null | AGENTS.md §6「测试集只用于最终确认」 |
| 2.4 `cost` 字段 | AGENTS.md §5.6、§6 对比表 |
| 3.5 错误码 | AGENTS.md §5.7「对空输入、超长输入和异常请求给出清晰错误」 |
| 3.6 只加载一次模型 | AGENTS.md §5.7 |

## 五、生效确认

本契约由 LanQvJian 起草。v1.0.1 已经 HuMingChang 与 HongSenYuan 确认；v1.1
设备条款由 HongSenYuan 提出、HuMingChang 同意并经 LanQvJian 批准，于
2026-09-21 正式生效。后续如对条款有异议，应立即提出并记录在
`reports/meeting-notes/`。

| 成员 | 确认状态 | 日期 |
| --- | --- | --- |
| LanQvJian | 已确认 v1.1 设备条款（批准人） | 2026-09-21 |
| HuMingChang | 已确认 v1.0.1，并同意 v1.1 设备条款（见相关确认记录） | 2026-09-21 |
| HongSenYuan | 已确认 v1.0.1，并提出、同意 v1.1 设备条款；其余确认后待办继续有效 | 2026-09-21 |
