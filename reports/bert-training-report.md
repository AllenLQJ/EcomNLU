# BERT 中文微调报告

- 负责人：HongSenYuan
- 日期：2026-09-21
- 对应阶段：AGENTS.md §5.4、`docs/contracts.md` 第二章
- 数据：`data/processed/`（已通过数据契约校验 19/19；train 50,206 / dev 6,274 / test 6,272；
  `metadata.json` 里的 `dataset_fingerprint` 与所有其它模型一致）

## 1. `max_length` 的确定依据

AGENTS.md §5.4 明确要求 `max_length` 由**本数据集**的 token 长度分布决定，不得照搬参考项目
新闻标题任务的 `max_len=32`。因此先用本地 `bert-base-chinese` 分词器对 **train 划分 50,206 条**
逐条编码（含 `[CLS]`/`[SEP]`，不做截断）实测：

| 统计量 | p50 | p75 | p90 | p95 | p99 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token 数 | 36 | 64 | 130 | 182 | 366.95 | 2,878 | 59.17 |

| 候选截断长度 | 32 | 64 | 96 | 128 | 160 | **192** | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 覆盖率 | 44.88% | 75.10% | 85.45% | 89.85% | 92.68% | **96.29%** | 97.97% | 99.54% |

**结论：取 `max_length = 192`**，即覆盖率首次达到 95% 的最小候选长度（`token_stats.recommend_max_length`
的选取规则），实测覆盖 96.29%，超过 512 token 的样本只有 230 条（0.46%）。

对照参考项目的 `32`：它在本数据集只能覆盖 **44.88%**，一半以上评论会被截断，因此不能沿用。
完整原始数值见 `reports/bert-token-lengths.json`。

## 2. 数据管道（AGENTS.md §5.4）

`src/models/bert/dataset.py` 提供独立的 `Dataset`、`DataLoader` 与 `collate_fn`：

- `ReviewDataset`：先**整份编码一次**（`padding=False`，只截断到 `max_length`），编码按 4,096 条一块
  分批调用分词器，避免一次性占用过多内存；只使用 `cat` 列，`sample_id` 按契约 §1.2 忽略。
- `make_collate_fn`：**动态 padding**，每批补齐到该批最长样本的长度，同时生成 `attention_mask`。
- `build_dataloader`：train 批大小 32 且打乱（`torch.Generator` 固定种子 42），dev 批大小 64 不打乱。

动态 padding 的实际收益在本数据集上**有限**，如实记录：抽样 20 个训练批次，平均补齐长度 **181.6**、
最大 192。原因是 96.29% 覆盖在 192，而每批 32 条随机样本中几乎必然出现一条接近 192 的长评论，
只有 5.4% 的算力被省下；若用「按全量最大长度固定补齐」则会长期按 2,878 token 补齐，代价不可接受。

## 3. 训练设置与运行环境

| 项目 | 取值 |
| --- | --- |
| 预训练模型 | `bert-base-chinese`（本地目录 `models/pretrained/bert-base-chinese`，由 `scripts/fetch_pretrained.py` 经镜像下载） |
| 模型结构 | `AutoModelForSequenceClassification`（`num_labels=10`，`id2label` 与 `label_mapping.json` 顺序一致） |
| 可训练参数 | 102,275,338（约 102.3M） |
| `max_length` | 192（见第 1 节） |
| batch size | 32（dev 评估用 64） |
| 学习率 | 5e-5，AdamW（bias 与 LayerNorm 不做权重衰减，weight decay 0.01） |
| 调度器 | 线性预热（`warmup_ratio=0.1`）+ 线性衰减，共 3×1,569 步 |
| 梯度裁剪 | `clip_grad_norm_` 1.0 |
| epoch | 3 |
| 混合精度 | cuda 上 bf16 `autocast` |
| 类别加权 | 未启用（`--class-weights` 可选，本轮未使用） |
| 随机种子 | 42（python / numpy / torch / cuda 全部固定） |
| 设备 | cuda：NVIDIA GeForce RTX 4060 Laptop GPU（8 GB，CUDA 12.6） |
| 环境 | Python 3.12.14、torch 2.14.0+cu126、transformers 5.17.0（conda 环境 `bert_gpu`） |
| 权重下载 | `python scripts/fetch_pretrained.py bert-base-chinese`（走 `hf-mirror.com`，记录 SHA-256） |

选模纪律：**用 train 训练、用 dev 的 Macro-F1 选模，test 只在定稿后评测一次**（AGENTS.md §5.4/§6）。
每个 epoch 结束后在 dev 上评估，只保留 dev Macro-F1 最优的权重，并以 `save_pretrained` 落盘为
`models/bert_base_chinese/model/` 与 `tokenizer/`。

## 4. 训练曲线数据（逐 epoch）

数据来自 `reports/bert-training-summary.json` 的 `history`（AVG train loss 为 epoch 内批损失均值）：

| epoch | train loss | dev accuracy | dev Macro-F1 | dev loss | 本轮耗时 (s) | 学习率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4786 | 0.8966 | 0.8833 | 0.2949 | 443.2 | 5e-5 |
| 2 | 0.2247 | 0.9150 | 0.9080 | 0.2466 | 445.6 | 5e-5 |
| 3 | 0.1355 | 0.9208 | **0.9123** | 0.2591 | 445.2 | 5e-5 |

- 训练总耗时 **1,335.8 s（约 22.3 min）**，第 3 轮的 dev Macro-F1 最优，保存的即第 3 轮权重。
- train loss 单调下降，dev Macro-F1 与 accuracy 逐轮提升；**dev loss 在第 3 轮小幅回升**
  （0.2466 → 0.2591），是轻微过拟合的早期信号，但 Macro-F1 仍在提升，因此第 3 轮仍是最优轮。
- 本轮只跑了 3 个 epoch、单一超参组合（学习率、批大小、`max_length` 均未做网格搜索），
  原因是课程时间预算；若要继续提升，应先按 dev Macro-F1 决定是否加 epoch 与试 `--class-weights`，
  **不得用 test 挑参**。

## 5. 统一入口评测结果（AGENTS.md §6）

评测口径：`scripts/evaluate_model.py` → `src.inference.load_predictor` → 与线上同一个预测器，
因此「离线指标」与「线上推理」必然一致；FastText 与 BERT 使用同一份 test（同一 `split_manifest`）。

| 划分 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev（选模用） | 0.9208 | 0.9228 | 0.9044 | **0.9123** | 0.9207 |
| test（最终确认） | 0.9193 | 0.9286 | 0.9032 | **0.9138** | 0.9191 |

test 的每个类别指标、混淆矩阵与高置信度错误样本见自动生成的 `reports/error-analysis.md`；
与 FastText 各变体的对比见 `reports/model-comparison.md`。

要点：

1. dev 与 test 的 Macro-F1 相差 0.0016（0.9123 vs 0.9138），没有出现「dev 调过头」的迹象。
2. **热水器是唯一明显的短板**：test recall 0.6316（57 条里有 21 条漏判），F1 0.7347，
   是所有类别中最低的；它是数据集中样本量最小的类别（全量仅 574 条，占 0.91%，dev 与 test 各 57 条）。
   `reports/eda-report.md` 显示它的评论长度中位数只有 **22 字符**（10 类中最短）、正向占比 82.58%，
   内容多为「安装/售后」类短句，可供判别的信息本身很少——这与低召回一致，不是模型独有的问题。
3. 主要混淆集中在 **平板 ↔ 洗发水（54/58 条）**、**平板 → 衣服（43 条）**、**水果 → 洗发水（42 条）**，
   与 FastText 的混淆结构一致（见 `reports/error-analysis.md`），说明这不是某个模型的特有缺陷，
   而是类别体系本身存在边界模糊（洗发水/平板/衣服的评论里都大量出现促销、物流、售后等通用表述）。

### 5.1 针对性探针：模型更依赖关键词，而不是「读懂」评论

用统一入口对 4 条自造评论做单条预测（同一批 `data/processed` 权重，CPU）：
「苹果手机电池续航不错，信号也好」与「我买的华为手机很不错，拍照清晰」都被判成 **平板**
（BERT 置信度 0.973 / 0.999，`fasttext_word_tuned` 同样判平板，置信度 0.9999 / 0.9999），
而「这部手机用了三个月就坏了，售后也不管」被判成 **手机**（0.749）。

这不是两个模型同时出现的巧合，回到 train 划分统计就能解释：

| 类别 | 训练样本 | 含「手机」 | 含「平板」 | 含「华为」 |
| --- | ---: | ---: | ---: | ---: |
| 平板 | 8,000 | 1,085 | 1,020 | **1,096** |
| 手机 | 1,859 | 499 | 0 | **0** |

「手机」这一类里**一次都没有出现过「华为」**，而「平板」类里有 1,096 条提到「华为」（该类别实际是华为平板）。

> **结论与警告**：模型学到的是「华为 → 平板」这种**品牌/关键词关联**，而不是商品语义。
> 探针里「华为手机」的失败正好暴露了这一点。这也提示：(1) 线上遇到训练集中未出现的品牌组合时
> 置信度会虚高（0.999 并不等于可靠）；(2) 类别体系粒度不一致（品牌 vs 品类）会被模型当成捷径，
> 这是数据侧问题，换更大的模型或调参都无法根治。

> 方法学说明：这 4 条探针评论是人工构造的，统计结论（品牌词与类别的关联）来自真实 train 划分计数，
> 但探针本身**不构成测试集指标**，不能与上面的 dev/test 指标混用。

## 6. 与参考项目实现的差异及原因

| 方面 | 参考项目 `_04_bert` | 本项目 | 原因 |
| --- | --- | --- | --- |
| `max_len` | 32（新闻标题任务沿用） | **192**（本数据集实测覆盖 96.29%） | AGENTS.md §5.4 禁止照搬 |
| 分类模型 | `BertModel` + 自写 `nn.Linear` 头 | `AutoModelForSequenceClassification` | 契约 §2.2 要求 `save_pretrained` 的 `model/`+`tokenizer/` 目录 |
| 保存方式 | 裸 `state_dict` 的 `.pt` | `save_pretrained` | 需要能被统一入口与契约校验器加载 |
| 选模 | 每 200 步在 dev 上比 F1 | 每 epoch 在 dev 上比 **Macro-F1** | 类别不均衡下 Macro-F1 才是选模依据（AGENTS.md §6） |
| 补齐方式 | `padding='max_length'` 固定长度 | 动态 padding（`collate_fn` 按批补齐） | 减少长文本数据集上的无效算力 |
| 学习率调度 | 固定 lr | 线性预热 + 线性衰减、梯度裁剪 | 微调稳定性 |

## 7. 局限与未验证项

1. **未做超参搜索**：仅 `lr=5e-5 / batch=32 / epochs=3 / max_length=192` 一组配置；
   `--class-weights`（按类别频次倒数加权）已实现但本轮未运行，未验证能否提升热水器等小类的召回。
2. **`max_length` 只比过 192 一档**：256 的覆盖率为 97.97%，收益 1.68 个百分点、代价约 33% 算力，
   本轮因时间预算未做对照实验，属**未验证**项。
3. **延迟只在 CPU 上实测**：`metrics.json` 的 `cost.measured_on` 为 `cpu`（项目 venv，`torch 2.14.0+cpu`）：
   模型本体 390.2 MB、单条 183.80 ms、批量 32 条 8,191.78 ms、吞吐 3.9 条/秒。
   **GPU 上的推理延迟未测量**（属 M5 范围）。

   > 计时口径：与 M3 传统模型统一使用 ``src.evaluation.metrics.measure_inference_cost``
   > （1 次预热 + 20 次单条取**中位数**，批量 32 条重复 5 次取中位数），
   > 这样 M3 与 M4 的延迟、吞吐可以直接横向比较（AGENTS.md §12.4）。
   >
   > 更早的一次计时（单条 1,060 ms、批量 32 条 13,639 ms）是在**另一台 BERT 评测进程同时在跑**
   > 的情况下测出的，属资源争用，已作废；合并 main 后所有模型统一用上述函数在 CPU 空闲时复测，
   > 复测只刷新 `metrics.json` 的 `cost` 字段，`dev`/`test` 指标块（含混淆矩阵）原样保留。
4. **类别体系粒度不一致**（「蒙牛」是品牌、「酒店」是服务）是数据集固有限制，
   模型在「蒙牛」上 F1 = 1.0 说明它可能靠品牌关键词识别，不能推断模型理解了商品语义。

## 8. 复现命令

```powershell
python scripts/fetch_pretrained.py bert-base-chinese                 # 下载预训练权重（含 SHA-256 记录）
python scripts/train_bert.py --batch-size 32 --max-length 192 --epochs 3
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split dev
python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test
python scripts/verify_model_contract.py models/bert_base_chinese
```

训练摘要（超参、环境、逐 epoch 指标）写入 `reports/bert-training-summary.json`，
token 长度分布写入 `reports/bert-token-lengths.json`，模型契约信息写入
`models/bert_base_chinese/{metadata,label_mapping,metrics}.json`。

## 9. 验证记录（实际运行结果）

| 验证项 | 命令 | 实际结果 |
| --- | --- | --- |
| 自动化测试 | `python -m unittest discover -s tests -v` | **Ran 60 tests … OK**（数据流水线、契约校验器、推理适配器、项目结构） |
| 模型契约 | `python scripts/verify_model_contract.py models/bert_base_chinese` | 校验通过：必需文件、`label_mapping` 与数据契约逐字节一致、metadata 无占位符、`max_length=192`、split 指纹一致、**通过统一入口加载并完成真实推理** |
| 统一入口评测 | `python scripts/evaluate_model.py --model-dir models/bert_base_chinese --split test` | test 6,272 条，Macro-F1 0.913847、Acc 0.919324，指标与混淆矩阵写入 `metrics.json` |
| 单条预测 | 见第 5.1 节的探针 | 4 条评论全部返回合法类别名与覆盖 10 类的概率，接口行为符合契约 §2.5 |
| 全量 FastText 契约校验 | 对 4 个 `fasttext_*` 目录分别执行 | 4/4 通过 |

> 说明：在受限 shell（本机 DSH 的 workspace-write 沙箱）中直接跑测试时，有 4 个使用
> `tempfile.TemporaryDirectory` 的用例会因沙箱拒绝写入临时目录而报 `PermissionError`；
> 放开文件权限后同一套命令 **60 项全部通过**。这是运行环境的限制，不是代码缺陷。

