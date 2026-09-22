# 典型问题与排查过程

本文档对应 AGENTS.md §10「典型问题」评分项，收集项目开发中**真实发生**的问题，
每条按四要素记录：**问题现象 → 定位过程 → 解决方案 → 验证证据**。

**记录原则**

- 只写仓库中**真实记录过**的内容，不补全、不推断。
- 区分「已记录验证证据」与「只声称修复但无证据」——后者在本文档中明确标注，
  答辩时不得声称已验证。
- 无解的问题（如数据导致的固有局限）如实标注为**无解决方案**，不编造修复。

---

## 一、四要素齐全的问题（按答辩价值排序）

### 问题 1：契约 v1.0 的数据集指纹「任何人都不可能稳定满足」（规范缺陷，曾阻断 M3 验收）

**现象.** 连续两次运行 `scripts/split_data.py`（种子 42 保证确定性）后，三个 CSV 逐字节相同，
但清单哈希发生变化：

| 文件 | 两次运行 |
| --- | --- |
| `train.csv` | `bdd8b161d887ccb7…` 相同 |
| `dev.csv` | `757536c7559256ae…` 相同 |
| `test.csv` | `406750b7b7476a6f…` 相同 |
| `split_manifest.json` | `10c21992152b2269…` → `32bb978b696032e2…` **变化** |

后果：`scripts/verify_model_contract.py` 会持续报「该模型可能是在旧数据上训练的」，
**M3 无法通过验收**。

**定位过程.** 在**不改动任何清洗或划分逻辑**的前提下连续运行两次划分脚本，逐文件比对哈希，
定位到唯一变量是清单里的 `created_at`（时间戳）。随后做了受控实验：只把 `created_at`
由 `14:35:19` 改为 `23:59:59`（三个 CSV 字节未变），旧口径哈希从 `72edce67c4083026…`
漂移到 `d359bd47005706f2…`，新口径保持稳定。

**解决方案.** 判定为**规范本身的缺陷，不是实现错误**，因此升版为 **v1.0.1** 而非占用 v1.1 版本号。
字段 `split_manifest_sha256` 改名为 `dataset_fingerprint`，取值改为
`sha256(":".join([sha256(train), sha256(dev), sha256(test)]))`——只对数据内容取指纹，
不含时间戳。改动落在 `src/models/baseline/artifacts.py`（新增 `dataset_fingerprint()`）、
`scripts/train_baseline.py`、`src/evaluation/metrics.py`，并新增 4 项指纹测试。

**验证证据.**

- `python scripts/verify_model_contract.py` → 三个模型各 **15 项通过、0 失败、0 跳过**
- `python scripts/verify_data_contract.py` → **19 项通过、0 失败、0 跳过**
- 交叉核对：指纹 `c9b76e66e05c96dd…` 在「契约文档算法 / 校验器实现 / M3 实现」三处、
  **两台不同机器**上完全一致
- 回归保护：`test_dataset_fingerprint_ignores_manifest_created_at` 会改写 `created_at`
  并断言指纹不变，**同时断言清单哈希本应变化**——测试本身即证明它在测 v1.0 的那个缺陷

**出处.** `reports/m3-validation.md` 第 2、3 节；
`reports/meeting-notes/2026-09-21-contract-change-v1.0.1-fingerprint.md`；
`reports/daily-reports/2026-09-21-HuMingChang.md` M3-7 节。

---

### 问题 2：安装 wordcloud 导致 matplotlib 原生崩溃（影响全队）

**现象.** 安装 `wordcloud` 后，matplotlib **保存任何图片都会原生崩溃**，进程退出码
`0xC06D007F`，即使是一张没有标题的普通柱状图也无法保存。由于是原生崩溃而非 Python 异常，
`try/except` 无法捕获，**没有任何 Python 报错信息**。

**定位过程.** 做了 6 步二分，每步独立进程：

1. 全量分词与词频统计正常（50,206 条、10.3 秒）→ 排除 jieba 与数据
2. 纯 Pillow 存图 `PIL SAVE OK` → 排除 Pillow
3. `fig.canvas.draw()` 与 `savefig()` 均崩溃 → 锁定 matplotlib Agg 渲染
4. conda 版本快照：`rev 0` 里只有 Python 本身、无任何 matplotlib/numpy，
   说明这些包原本由 pip 安装；当天 `rev 1` 被 conda-forge 构建覆盖
5. 查包来源频道：`matplotlib-base`/`pillow`/`numpy`/`freetype`/`libpng` 来自 conda-forge，
   而 `libtiff` 来自 `repo.anaconda.com/pkgs/main`（defaults）——**频道混用**
6. 查 DLL：`Library\bin` 中 `libtiff.dll` 与 `tiff.dll` **并存**，`zlib.dll` 出现 3 份

**解决方案.** 改用 pip 从镜像重装整条图像栈（wheel 自带捆绑 DLL，不受 `Library\bin`
重复 DLL 影响）。附带改进：`scripts/report_environment.py` 的检查范围新增 `pillow` 与
`wordcloud` 两个传递依赖——本次之所以没能预警，正是因为该脚本只检查 `requirements.txt`
里的直接依赖，而 Pillow 不在其中。

**验证证据.** 极硬：

- `matplotlib savefig OK`，退出码 0
- 全链路 5 个脚本全部 exit 0
- **重新生成的 6 张图表字节数与故障前完全一致**
  （74,907 / 65,387 / 80,291 / 65,848 / 111,113 / 64,414 字节）
- 缺字形警告 `0`；`Ran 27 tests` → `OK`

**出处.** `reports/daily-reports/2026-09-21-HuMingChang.md`「遇到的问题 问题 9」。

---

### 问题 3：量化产物无法加载（`OSError` + 296 个未知键）

**现象.** 两个连续的失败模式：

1. `AutoModelForSequenceClassification.from_pretrained("model")` 抛
   `OSError: Error no file named model.safetensors`
2. 改在 fp32 骨架上直接 `load_state_dict(量化权重)`，报 **296 个未知键**，
   例如 `bert.encoder.layer.0.attention.self.query.scale`、`zero_point`、`_packed_params.dtype`

**定位过程.** 检查产物目录发现 `model/` 下只有 `quantized_state_dict.pt`、没有标准权重文件，
HF 加载流程找不到入口；进一步分析这些是 dynamic quant 模块的专有键，fp32 的 `nn.Linear`
没有对应容器——`strict=False` 也无法挽救，**因为容器类型根本不匹配**。

**解决方案.** 三步加载，顺序不可调换：
按 `config.json` 构造 fp32 骨架 → **先** `quantize_dynamic` 建立量化容器 →
**再** `load_state_dict` 灌权重。实现在 `src/models/optimization/quantization.py`；
`src/inference/transformers_adapter.py` 按 `metadata.json` 的 `quantization` 字段自动选路。

**验证证据.**

- 重新加载成功 `True`；恢复出 **74 个 dynamic quant Linear 层**（`0` 表示产物损坏）
- 三条探针预测正确：书籍 0.999244 / 平板 0.998584 / 酒店 0.999746
- `verify_model_contract.py models/bert_base_chinese_quantized` → **15 项通过、0 失败、0 跳过**
  （其中「统一入口可加载并推理」为真实执行，非跳过）

**出处.** `reports/daily-reports/2026-09-21-HongSenYuan.md` 问题 2、问题 3；
`reports/m5-quantization-report.md` 第 3、4 节。

---

### 问题 4：统一入口下 FastText dev 准确率只有 16.94%（切分不一致 + 概率越界）

**现象.** 接入统一推理入口后，FastText dev 准确率只有 **0.1694**（约等于全猜多数类），
而 fastText 自带 `model.test()` 在同一份 dev 文件上给出 **0.8731**。
另有第二处缺陷：fastText 概率来自 float32 softmax，实测返回 `1.0000100135803223`
这类略大于 1 的值，而适配器原本用严格 `0.0 <= p <= 1.0` 判断，
**于是任何 FastText 模型都会在预测时报错**。

**定位过程.** 把统一入口的结果与 fastText 自带 `test()` 的 0.8731 对比，定位到适配器把
**原始句子**直接交给 fastText，但字符级/词级模型是在**空格分隔的 token** 上训练的；
fastText 只按空格切词，因此整句退化成一个 token。

**解决方案.** 适配器按 `metadata.json` 的 `preprocess.tokenize` 复用训练切分
（`none` / `chars` / `jieba_words`，缺省 `none` 保持向后兼容）；概率允许 1e-4 容差后
夹紧到 `[0,1]`，超出容差仍判非法。

**注意这是改他人主责文件**：`src/inference/fasttext_adapter.py`（LanQvJian 主责）。
提交者在报告中主动写明「请复核；如需回退，`git revert e5b0461` 即可，
但 FastText 模型将无法通过统一入口评测」。

**验证证据.** 修复后 dev 准确率由 **0.1694 回到 0.8715**，与 fastText 自带 `test()`
的 0.8731 一致；四个 `fasttext_*` 目录契约校验各 **15 项通过、0 失败、0 跳过**，
校验含「通过统一入口加载并完成真实推理」。

**出处.** `reports/fasttext-report.md` 第 4 节（缺陷 1、2）、第 5 节。

---

### 问题 5：量化保存在 py310 环境必然失败（懒加载 → cv2 → NumPy ABI 不兼容）

**现象.** 报错堆栈链：

```text
torch.save(state_dict) → transformers/__init__.py 的模块级 __getattr__ 懒加载
  → 导入 transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3
  → import cv2 → RuntimeError: module compiled against ABI version 0x1000009
                 but this version of numpy is 0x2000000
```

根因：`opencv-python 4.5.5.64` 按 NumPy 1.x ABI 编译。
**注意卸载不是解法**——该 `import cv2` 无保护，卸载后会变成 `ModuleNotFoundError`，
同样失败。这一句排除了一个看似显然的错误答案。

**定位过程.** 顺报错堆栈逐层读出「`torch.save` → transformers 懒加载 → cv2 → ABI 版本不匹配」，
并核对 opencv 版本与 NumPy 版本的编译 ABI。

**解决方案.** 升级到 `opencv-python 4.14.0.94`（4.10 起才提供 NumPy 2 兼容 wheel），
并在 `requirements.txt` 显式加入 `opencv-python>=4.10,<5.0` 及原因注释，使其可复现。

**验证证据（部分记录）.** 验收环境记录为 Python 3.10.20、numpy 2.2.6、torch 2.12.0+cpu、
transformers 5.16.1、opencv-python 4.14.0，且该环境下量化保存、重载、契约 15/15、
预测、Web 联调 10/10 均成功。**但报告没有给出「修复前后对照」的独立命令与输出**，
属「环境修正 + 后续运行成功」型验证。

**出处.** `reports/m5-merge-test-report.md` 第 6.1 节。

---

### 问题 6：sklearn 静默忽略 `ngram_range`，让「词级 bigram」对照退化成 unigram

**现象.** 两次本该不同的对照实验得到**完全相同**的 Macro-F1（0.7899）与词表（4,144），
**异常一致才暴露了问题**。sklearn 只给了一条 UserWarning。

**定位过程.** 靠「两次对照结果完全相同」这一异常反查向量器构造参数。

**解决方案.** 用 `analyzer=<可调用对象>` 构造时 sklearn 会静默忽略 `ngram_range`；
改用 `tokenizer=` 并显式关闭 `token_pattern`，`ngram_range` 才真正生效。
新增回归测试 `test_ngram_range_actually_takes_effect` 固化该行为。

**验证证据（部分记录）.** 有回归测试固化；最终报告第 5 节的对照表显示两项词级方案结果已不同
（词级 unigram 0.8673 / 词表 26,461；词级 unigram+bigram 0.8595 / 词表 87,494），
词表规模差异可作旁证。**但报告未把该表明确标注为该缺陷的验证证据。**

**出处.** `reports/daily-reports/2026-09-21-HuMingChang.md` M3-2 节；
`reports/m3-baseline-comparison.md` 第 5 节。

---

### 问题 7：M2 数据产物 5 处不符合接口契约（含分层比例超差）

**现象.** 首次用契约校验器核验 M2 产物为 `[FAIL]`，5 处不符合全在数据产出方一侧：

| # | 契约要求 | 改造前实际 |
| ---: | --- | --- |
| 1 | `data/processed/label_mapping.json` | 不存在 |
| 2 | `data/processed/split_manifest.json` | 不存在 |
| 3 | 列顺序 `sample_id,review,cat,label` | 原为 `cat,label,review` |
| 4 | `sample_id` 列（原始 CSV 行号，可追溯） | 完全没有 |
| 5 | 各类别分层比例容差 ±1.0 行 | 水果 +1.6、洗发水 +1.4，超差 |

**定位过程.** 用 `git worktree add <目录> origin/feature/api-contracts` 检出契约分支，
把本分支产出的 `data/processed/` 与原始数据复制进去后运行该校验器。
这样校验的是**契约分支的校验代码**与**本分支的真实数据产物**，两者未被混同。

**解决方案.** 新建 `src/data/text_patterns.py`、`src/data/raw_io.py`（`write_contract_csv`
强制列顺序，且**不提供任何写入 `data/raw` 的函数**）、`src/data/manifest.py`；
`cleaning.py` 跟踪 `sample_id`（**不能用 `reset_index` 充当**——删行后行号会整体前移而错位）；
`splitting.py` 新增配额修复（只搬运整组，目标收紧到 ±0.5 行，严于契约 ±1.0）。

**验证证据.** 校验器输出 **通过 19 项、失败 0 项、跳过 0 项**；实测类别分层比例
全部 ≤0.8（配额修复后）；测试从 27 项扩充到 **45 项** → `OK`；
完整链路重跑两次样本量与划分结果完全一致。

**出处.** `reports/daily-reports/2026-09-21-HuMingChang.md` M2-10 节。

---

### 问题 8：跨类别重复文本——直接全局去重会删掉 32 行有效样本

**现象.** 初步统计显示「按评论文本全局去重可删除 49 行」，看似应直接去重。
但进一步实测发现：**45 种重复文本中有 28 种出现在多个类别下**（涉及 60 行），
全局去重会删除其中 **32 行**有效样本，且会给同一文本一个任意标签，并造成跨集合泄漏。

**定位过程.** 对重复文本按「是否跨类别」「情感标签是否一致」精确拆解，结果为
**46 组 / 96 行**：同类别内同情感标签 18 组 36 行（纯冗余）、同类别内**情感标签矛盾**
17 组 34 行（真冲突，冲突在 `label` 而非 `cat`）、跨类别歧义 28 组 60 行
（**不是冲突**——同一文本对应多个合法类别）。
即：被统称为「冲突」的 94 行里，**60 行（占 64%）并不是冲突**。

**解决方案.** 清洗阶段只删同类内完全重复（19 行）；跨类别相同文本保留，
改由划分阶段**按归一化文本分组**整体分配。契约 v1.2 §1.4 把规则固定为
`cleaning.rules.duplicate_review = drop_within_category_only`、
`split.dedup_key = normalized_text`，并要求校验器把它作为固定校验项。

**验证证据.** 泄漏检查 train/dev、train/test、dev/test 重复文本数均为 **0**；
`reports/data-split-stats.json`：`multi_category_groups: 28`、`group_count: 62720`、
`repair_moves: 1`、`leakage_free: true`；清洗前后 62,774 → 62,752；
测试 `test_same_text_under_different_categories_is_preserved` 与
`test_multi_category_text_stays_in_one_split` 固化；`verify_data_contract.py` 19/19。

**出处.** `reports/meeting-notes/2026-09-21-contract-v1.2-clause-draft.md` 第〇节；
`reports/meeting-notes/2026-09-21-contract-confirmation-humingchang.md` 第三节。

---

### 问题 9：pandas 3.x 下正则统计严重误判（89% 评论被判为「纯符号」）

**现象.** 质量报告一度显示「纯数字或纯符号评论」为 **55,799 条，占全体 89%**，明显不合理。
同一根因还导致 RE2 不接受 `\uXXXX` 转义，`r"[一-鿿]"` 直接抛 `ArrowInvalid`。

**定位过程.** pandas 3.0.5 字符串列默认使用 PyArrow 后端，`.str` 正则由 **RE2** 执行，
而 RE2 的 Perl 字符类只按 ASCII 解释——`\w` 等价于 `[0-9A-Za-z_]`，
因此 `[0-9\s\W]+` 会命中所有中文评论。

**解决方案.** (1) 中文范围改为**非 raw 字符串**，由 Python 先解析成真实字符再交给 RE2；
(2) 「纯符号文本」改用「既无中文也无拉丁字母」的显式定义，不再依赖 `\W`；
(3) 增加回归测试保护该行为。

**验证证据.** 修正后该指标为 **2 条**，与人工抽样一致。

**出处.** `reports/daily-reports/2026-09-21-HuMingChang.md`「问题 1」；
`src/data/text_patterns.py` 模块说明。

---

### 问题 10：量化延迟对比出现设备不公平（一度得出「慢 21 倍」的错误结论）

**现象.** 首版报告显示量化后「批 32 慢 **21 倍**」（237.7 ms → 5132.3 ms），
结论与「量化提速」完全相反。

**定位过程.** 发现 `evaluate_model_dir` 记录的 fp32 延迟来自 **CUDA**，
而 int8 动态量化只能跑 **CPU**（CUDA 无对应 int8 动态算子）。
拿两个设备互比，测到的是 **GPU 与 CPU 的差距**，不是量化的收益。

**解决方案.** 新增 `measure_cpu_latency`，把 fp32 也搬到 CPU 上重测；
`scripts/compare_models.py` 对比表新增「设备」列并给出跨设备不可比警告。

**验证证据.** 归一到 CPU 后批 32 **加速 1.65×**（721.22 → 463.02 ms）；
报告**同时保留两张表**并明确标注第一张**不能**用来评价量化效果；
独立复测（`reports/m5-merge-test-report.md` 第 9.4 节）CPU 单条 59.44 → 33.11 ms、
批 32 721.22 → 463.02 ms。`README.md` 与 `reports/model-comparison.md`
均加了跨设备警告。

**出处.** `reports/daily-reports/2026-09-21-HongSenYuan.md`「问题 4」；
`reports/m5-quantization-report.md` 第 2 节。

---

## 二、值得补充的问题（四要素略简，但均有原文支撑）

### 补充 1：BOM（U+FEFF）逃过清洗，成为 TF-IDF 垃圾 token

- **现象.** 进入 M3 分词时，第一条训练评论被 jieba 切出的第一个 token 是 `'﻿'`；
  原始数据中有 **20 条**评论内部嵌入 U+FEFF。此前清洗规则只覆盖 ASCII 控制字符
  `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`，而该字符**不可见，人工检查看不出来**。
- **解决方案.** 质量检查新增「含 Unicode 格式字符 / 空白字符」两项统计（实测 20 条 / 52 条）；
  清洗新增 `strip_unicode_format_chars`（按 Cf 类删字符保留样本，一并覆盖零宽连接符、软连字符）；
  报告样例把不可见字符渲染为 `[U+FEFF]` 使证据可核查。
- **验证证据.** 修复后清洗产物中 U+FEFF 残留 **0 条**；该步骤修改 20 行，
  并因 BOM 去除产生 1 条新的同类别重复而多删 1 行，最终 62,753 → 62,752。
- **出处.** `reports/daily-reports/2026-09-21-HuMingChang.md`「问题 7」。

### 补充 2：情感模型的 `label_mapping.json` 必然校验失败（契约 v1.2 的真正阻断点）

- **现象.** 情感模型的标签集是 `["负向", "正向"]`，与主任务 10 个商品类别必然不同，
  因此**任何情感模型都会在校验的第一步就失败**。「这一条不补，提案一的其余四项改了也没用。」
- **定位过程.** 起草 v1.2 时核对 M2 产物 + 代码审查 `scripts/verify_model_contract.py` 的
  `check_label_mapping`；同时核对出三点**不需要改**：适配器已任务无关、
  `metadata.task` 已允许 `sentiment_classification`、`check_metrics_block` 用
  `len(categories)` 而非写死 10。
- **解决方案.** v1.2 §2.1 让 `label_mapping.json` 按 `metadata.task` 分两套规则；
  `configs/base.yaml` 新增 `task.sentiment_categories`；**不新增** `SentimentPredictor` 协议。
- **验证证据.** 合并后实测 `verify_model_contract.py` → `sentiment_tfidf_logreg`
  与 `sentiment_tfidf_svm` 各通过 **15 项、失败 0**；全量 unittest **223 项 OK（跳过 5）**。
- **出处.** `reports/meeting-notes/2026-09-21-contract-v1.2-clause-draft.md` 第〇节；
  `reports/m5-merge-test-report.md` 第 5.2 节。

### 补充 3：继承分支时选错了基点（`Already on 'feature/fasttext'` 有误导性）

- **现象.** 新建 `feature/bert-optimization` 后，`scripts/` 下 `evaluate_model.py`、
  `train_bert.py`、`compare_models.py` **全部缺失**。原因是 `checkout feature/fasttext`
  输出 `Already on 'feature/fasttext'`，**但实际 HEAD 仍在 `main`（`54e7360`）**，
  该输出具有误导性。
- **定位过程.** `git rev-list --left-right --count` 与 `git ls-files scripts/` **双重核对**。
- **解决方案.** 删除错误分支，改用 `git checkout -b feature/bert-optimization feature/fasttext`
  显式指定基点。
- **验证证据.** 新分支基点为 `729999e`，M4 脚本齐全。
- **出处.** `reports/daily-reports/2026-09-21-HongSenYuan.md`「问题 1」。

### 补充 4：量化产物的两处元数据缺陷（独立验收时查出）

- **现象.** `metadata.model_name` 为 `bert_base_chinese` 而目录名是
  `bert_base_chinese_quantized`（契约 §2.3 要求一致）；
  `quantization.source_model_dir` 写入了**本机绝对路径**（AGENTS.md §7.1 禁止）。
- **解决方案.** `save_quantized_model` 改为取目标目录名；新增 `_as_project_relative()`
  转相对项目根的 POSIX 路径。
- **验证证据.** 新增 `tests/test_quantization.py`（**9 项**）——量化模块此前**无任何测试覆盖**。
- **出处.** `reports/m5-merge-test-report.md` 第 9.6 节；提交 `2c8e9e4`。

### 补充 5：手机被误判为平板（**只有三要素：无解决方案，是如实标注的固有局限**）

- **现象.** 实测：契约自带探针文本「手机很好用，电池耐用」被判为**平板**；
  BERT 与 `fasttext_word_tuned` 对「苹果手机电池续航不错，信号也好」
  「我买的华为手机很不错，拍照清晰」都判成**平板**（BERT 置信度 **0.973 / 0.999**，
  fastText **0.9999 / 0.9999**）。
- **定位过程.** 回到 train 划分统计：平板类 8,000 条中含「手机」**1,085** 条、
  含「华为」**1,096** 条；手机类 1,859 条中含「手机」**499** 条、含「华为」**0** 条。
  结论：模型学到的是「华为 → 平板」这种**品牌/关键词关联**，而不是商品语义；
  **置信度会虚高（0.999 并不等于可靠）**。
- **解决方案.** **无。** 报告明确标注为「传统模型的固有局限，也是 M4 采用上下文表示的动机之一」
  「这是数据侧问题，换更大的模型或调参都无法根治」。
- **验证证据.** 属**实测复现**（探针 + train 计数），不是「修好了」的验证。
  `scripts/predict_svm.py` 与 `README.md` 均把它作为已知失败模式提示使用者。
- **出处.** `reports/m3-baseline-comparison.md` 第 6 节；
  `reports/m3-validation.md` 第 4、5.2 节；`README.md`。

---

## 三、四要素不全的问题（如实标注，**答辩时不得补全或声称已验证**）

| 问题 | 缺什么 |
| --- | --- |
| **FastText NumPy 2 垫片**（见下方警示） | **缺验证证据**——无测试、无前后对比、无命令记录；只有代码注释级的定位分析 |
| 问题 5（cv2/NumPy ABI） | 有报错原文、根因、修复动作和环境记录，但**无修复前后对照命令** |
| 问题 6（sklearn `ngram_range`） | 定位过程短；修复后的直接验证只有回归测试名，无对照数字被明确标注 |
| pandas 3.x 可空 Int64 列 `clip` 抛错 | 记录为「已解决」，但**无任何命令或数字** |
| `uvicorn[standard]` 被依赖脚本静默跳过 | 有现象与解决，**无独立验证记录** |
| PyPI 无法访问 | **网络问题本身未解决**，靠改用预装 conda 环境绕过 |
| EDA 报告结论与数字张冠李戴 | 根因清楚（取值代码写错），验证只有「报告已重新生成并核对」，**无数字证据** |
| M4 验收中的两处（浮点相等判断、旧字段名） | **均无独立验证记录** |
| LinearSVC 无 `predict_proba` | 用 `CalibratedClassifierCV` 包装解决，有契约校验但**无该修复的专项对照** |
| FastAPI 默认 422 与契约 400 冲突 | 有实现与测试断言，但**无独立对照记录** |

### ⚠ 警示：已选入 PPT 的「FastText NumPy 2 垫片」缺验证证据

`src/inference/fasttext_adapter.py` 的 `apply_numpy2_compat` 文档字符串记录了一个真实缺陷：
NumPy 2.0 起 `np.array(probs, copy=False)` 语义变为「禁止复制」，而 `probs` 是元组必然需要复制，
因此 `predict()` 一定抛 `ValueError`——**训练能跑、预测必崩**；且上游 `fasttext-wheel 0.9.2`
与 `fasttext 0.9.3` 都带同一缺陷，**无法靠升级解决**。

但该问题的**验证证据未记录**：`tests/` 下检索 `numpy2|compat|asarray` **零命中**；
`reports/fasttext-report.md` 第 4 节记录的两个缺陷是另外两个，**没有**该垫片的前后对比数字。

该问题已出现在 PPT 与演讲稿中。**若评委追问「怎么确认修好了」，目前仓库里答不上来。**
建议二选一：补一次真实复现/验证记录，或答辩时不将其作为「已修复并验证」的案例讲述。

---

## 四、答辩提示

1. **最有说服力的两条是问题 2 与问题 1**：前者有 6 步二分定位 + 字节级验证
   （6 张图字节数完全一致）；后者是「规范缺陷不是实现错误」的典型，
   且三人交叉核对同一指纹，能体现团队协作。
2. **问题 4 是唯一一个「必须改他人主责文件」的案例**，且提交者主动写了回退方案与
   `git revert` 命令，适合回答「你们怎么处理跨成员修改」这类追问。
3. **问题 10（设备不公平）体现了「对结论的自我怀疑」**——报告主动保留了两张表并标注
   第一张不可用，比只给一个漂亮数字更可信。
4. **补充 5（手机→平板）适合回答「你们模型的局限是什么」**——有统计根因、无解、
   如实标注，比声称「已解决」更站得住。
5. 引用本文档任一问题时，**请连同「验证证据」一节的命令与数字一起引用**；
   标注为「部分记录」或「未记录」的，答辩时请如实说明。
