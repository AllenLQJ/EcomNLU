# FastText 模型报告

- 负责人：HongSenYuan
- 日期：2026-09-21
- 对应阶段：AGENTS.md §5.3、`docs/contracts.md` 第二章
- 数据：`data/processed/`（已通过数据契约校验 19/19，划分 train 50,206 / dev 6,274 / test 6,272，跨集合泄漏 0）

## 1. 语料生成与格式校验

字符级与词级各生成一份 FastText 训练文件，格式为 `__label__类别 切分后的文本`。
按 AGENTS.md §5.3 的要求逐行校验制表符、换行符、空文本与非法标签，实测统计：

| 切分方式 | 划分 | 总样本 | 写出 | 空文本跳过 | 非法标签跳过 | 需替换分隔符 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 字符级 | train | 50,206 | 50,206 | 0 | 0 | 0 |
| 字符级 | dev | 6,274 | 6,274 | 0 | 0 | 0 |
| 词级 | train | 50,206 | 50,206 | 0 | 0 | 0 |
| 词级 | dev | 6,274 | 6,274 | 0 | 0 | 0 |

校验器没有跳过或替换任何一行，说明上游 M2 清洗已经消除了空文本与异常分隔符；
校验逻辑仍然保留并生效（`src/models/fasttext/data_io.py`），以便将来数据变化时能立刻发现。

样例行：

```text
__label__书籍 做 父 母 一 定 要 有 刘 墉 这 样 的 心 态 ， 不 断 地 学 习 …      （字符级）
__label__书籍 做 父母 一定 要 有 刘墉 这样 的 心态 ， 不断 地 学习 …           （词级）
```

## 2. 训练结果与选模

四种组合（字符级/词级 × 默认参数/自动调参）全部训练并产出契约产物。
自动调参使用 `autotuneValidationFile=dev`，`autotuneDuration=120` 秒。

| 模型目录 | 切分 | 策略 | 训练耗时 | dev Macro-F1 | dev Acc | test Macro-F1 | test Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fasttext_char` | 字符级 | 默认参数 | 2.4s | 0.8290 | 0.8715 | 0.8378 | 0.8702 |
| `fasttext_char_tuned` | 字符级 | 自动调参 | 147.4s | 0.8546 | 0.8757 | 0.8539 | 0.8747 |
| `fasttext_word` | 词级 | 默认参数 | 1.7s | 0.8431 | 0.8757 | 0.8478 | 0.8795 |
| **`fasttext_word_tuned`** | 词级 | 自动调参 | 139.6s | **0.8607** | 0.8797 | **0.8655** | 0.8830 |

**选模结论：按验证集 Macro-F1 选定 `fasttext_word_tuned`（0.8607）作为 FastText 代表模型。**
测试集数值仅用于最终确认，未参与任何选择。

两条值得记录的结论：

1. **词级切分优于字符级**（同样策略下 dev Macro-F1 高约 1.2–1.4 个百分点）。
   数据里存在「蒙牛」「洗发水」这类成词信号，jieba 切分保留了它们，字符级则被打散。
2. **自动调参需要给足搜索预算，而且它优化的不是 Macro-F1。**
   搜索 20 秒时，两个调参模型都**低于**默认参数（dev Macro-F1 0.7991 / 0.8340）；
   给到 120 秒后反超（0.8546 / 0.8607）。原因是 FastText 的 autotune 目标是验证集上的
   P@1（准确率），在类别不均衡的数据上并不等价于 Macro-F1。选模必须自己用 Macro-F1 复核。

调参后的超参数（`fasttext_word_tuned`）：`dim=196`、`epoch=56`、`lr=0.0399`、
`word_ngrams=1`、`min_count=1`、`bucket=0`、`loss=softmax`。

## 3. 契约产物

每个模型目录包含契约 §2.1 要求的四件套：

```text
models/fasttext_word_tuned/
├── model.bin              # 模型本体
├── label_mapping.json     # 从 data/processed 逐字节复制
├── metadata.json          # §2.3 全字段，含 dataset_fingerprint
└── metrics.json           # §2.4 指标 + cost
```

`metadata.json` 额外记录 `preprocess.tokenize`（`chars` 或 `jieba_words`），
这是推理阶段必须复用的切分方式；`metadata.hyperparameters` 记录了实际使用的
`dim`/`epoch`/`lr`/`word_ngrams`/`min_count`/`bucket`/`loss` 与训练耗时。

## 4. 集成缺陷与修复（重要，涉及他人文件）

在第一次评测时发现 dev 准确率只有 **16.94%**（约等于全猜多数类），而 fastText 自带的
`model.test()` 在同一份 dev 文件上给出 **87.31%**。定位到统一入口的两个真实缺陷：

### 缺陷 1：推理切分与训练切分不一致

`src/inference/fasttext_adapter.py` 把**原始句子**直接交给 fastText，但字符级/词级模型
是在**空格分隔的 token** 上训练的。fastText 只按空格切词，因此整句退化成一个 token，
预测结果与训练分布完全不同。

修复：适配器按 `metadata.json` 的 `preprocess.tokenize` 复用训练时的切分方式
（`none` / `chars` / `jieba_words`），缺省 `none` 保持向后兼容。
修复后 dev 准确率由 0.1694 回到 0.8715，与 fastText 自带 `test()` 的 0.8731 一致。

### 缺陷 2：概率越界导致直接抛错

fastText 的概率来自 float32 softmax，实测会返回 **1.0000100135803223** 这类略大于 1 的值
（参考项目 `_03_fasttext` 的注释里也能看到 `1.0000093`）。适配器原来用严格的
`0.0 <= p <= 1.0` 判断，于是**任何 FastText 模型都会在预测时报错**，无法通过统一入口服务。

修复：允许 1e-4 的数值容差后夹紧到 `[0, 1]`，超出容差仍判为非法。

> **说明**：这两个修复改动了 `src/inference/fasttext_adapter.py`（LanQvJian 主责）。
> 改动是最小化的、向后兼容的，且不修则本阶段无法交付（统一入口无法服务任何 FastText 模型）。
> 请复核；如需回退，`git revert e5b0461` 即可，但 FastText 模型将无法通过统一入口评测。

## 5. 验证命令与结果

```powershell
# 训练（含语料生成、四变体训练、dev 评测）
python scripts/train_fasttext.py --autotune-duration 120 --evaluate-test

# 契约校验（四个模型目录）
python scripts/verify_model_contract.py models/fasttext_char
python scripts/verify_model_contract.py models/fasttext_char_tuned
python scripts/verify_model_contract.py models/fasttext_word
python scripts/verify_model_contract.py models/fasttext_word_tuned
```

实际结果：四个模型**各通过 15 项、失败 0 项、跳过 0 项**，结论均为「模型契约校验通过」。
校验包含：必需文件、label_mapping 与数据契约逐字节一致、metadata 全字段与无占位符、
`max_length` 为非 transformers 模型必须为 `null`、split 指纹与当前数据一致、
metrics 结构、以及**通过统一入口加载并完成真实推理**。

## 6. 局限与说明

1. **测试集协议**：四个变体都做了测试集评估，用于 AGENTS.md §6 要求的对比表。
   选模完全依据验证集，测试集数值未参与任何选择；这一点如实记录，不做粉饰。
2. **未使用类别加权**：FastText 的 `loss=softmax` 未加权，小类（热水器，574 条）
   的召回仍是 Macro-F1 的主要损失来源。类别加权留作可选改进。
3. **`reports/fasttext-summary.json` 的训练期快照**中，两个调参变体的 `test` 字段为空，
   因为测试集是在训练结束后用 `scripts/evaluate_model.py` 补评的；
   **权威数值以各模型目录的 `metrics.json` 与 `reports/model-comparison.md` 为准。**
