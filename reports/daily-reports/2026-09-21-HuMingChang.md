# 日报：2026-09-21 - HuMingChang

## 今日完成

### 1. 运行环境确认

- 完成内容：确认使用 conda 环境 `AI`（Python 3.11.15，与 README 建议的 3.11 一致）作为本项目统一运行环境，
  并生成依赖比对报告。
- 涉及文件：`scripts/report_environment.py`、`reports/environment.md`

### 2. M2-1 原始数据质量检查（已按团队决定移交给组长）

- 完成内容：编写只读质量检查模块与脚本，产出质量报告。
- 涉及文件（**已按团队决定删除**，见「问题 10」）：原 `src/data/quality.py`、
  `scripts/check_data_quality.py`、`reports/data-quality-report.md`、
  `reports/data-quality-stats.json`。
- 现状：原始数据质量检查由组长的 `src/data/check_raw.py` 承担，
  报告位于 `reports/data-quality/raw_quality_report.md`；
  两者共同的文本判定标准保留在 `src/data/text_patterns.py`。

### 3. M2-2 清洗规则确定（基于实测统计）

| 规则 | 实测数量 | 处理方式 | 依据 |
| --- | ---: | --- | --- |
| 缺失 / 空白评论 | 1 | 删除 | 无文本无法分类 |
| 控制字符 | 1 | 删除字符，保留样本 | 编码噪声，样本有效 |
| Unicode 格式字符（BOM / 零宽字符） | 20 | 删除字符，保留样本 | 不可见但会被 jieba 切成垃圾 token |
| HTML 实体 | 49 | 还原为真实字符 | 采集编码残留，可无损修复 |
| 空白 / 换行 / 制表符 / 全角空格 | 726 | 归一化为单个空格 | 避免同文本被当成不同样本 |
| 纯符号 / 纯数字文本 | 2 | 删除 | 无任何中文也无字母 |
| 同类别内重复文本 | 19 | 删除 | 冗余记录会放大样本权重 |
| 超短文本 | 37 条 ≤3 字符 | **不按长度硬删** | 「挺好」等短评仍是有效信号 |
| 超长文本 | 最长 2,876 字符 | 训练时截断，不删样本 | 长评论含类别与情感关键信息 |
| URL | 17 | 保留 | 无规则要求删除，避免信息丢失 |

- 涉及文件：`src/data/cleaning.py`、`scripts/clean_data.py`、`reports/data-cleaning-report.md`

### 4. M2-3 文本清洗

- 完成内容：实现 7 步可复现清洗流程，每步记录输入、输出、删除量与修改量。
- 实测结果：62,774 → 62,752 条，删除 22 条（0.0350%），另有 796 条仅被修正而未删除。

### 5. M2-4 分层划分与 M2-5 泄漏检查

- 完成内容：以评论文本为分组单位、以类别为分层依据，按固定种子 42 完成 8:1:1 划分。
- 实测结果：

| 集合 | 样本数 | 实际占比 | 目标占比 | 偏差 |
| --- | ---: | ---: | ---: | ---: |
| train | 50,206 | 80.0070% | 80% | +0.0070% |
| dev | 6,274 | 9.9981% | 10% | -0.0019% |
| test | 6,272 | 9.9949% | 10% | -0.0051% |

- 每个类别的分层比例全部落在接口契约的 ±1.0 行容差内（最紧的是水果与洗发水，
  经配额修复后偏差降到 0.2–0.8 行）。
- 泄漏检查：train/dev、train/test、dev/test 的重复文本数均为 **0**，出现在多个集合的文本数为 **0**。
- 涉及文件：`src/data/splitting.py`、`scripts/split_data.py`、`reports/data-split-report.md`

### 6. M2-6 EDA 图表与结论

- 完成内容：生成 6 张图表，每张图都有标题、坐标名称、图例，并配有文字结论。
- 涉及文件：`src/data/eda.py`、`scripts/run_eda.py`、`reports/eda-report.md`、
  `reports/figures/`（category_distribution、category_sentiment_stacked、review_length_histogram、
  review_length_boxplot、review_length_cdf、cleaning_funnel）

主要结论：

1. 类别不均衡：最大「平板」10,000 条（15.94%），最小「热水器」574 条（0.91%），差距 17.4:1。
   忽略占比 0.91% 的小类对 Accuracy 的影响不到 0.9%，因此必须用 Macro-F1 选模。
2. 情感分布：整体正向 50.56%，接近均衡；但「热水器」正向占比达 82.58%，类别间差异明显。
3. 文本长度：中位数 34 字符，均值 58.4，p95 = 183，p99 = 374，最长 2,876，长尾明显。
4. 类别间长度差异大：「书籍」中位数 145 字符，「热水器」仅 22 字符。

### 7. M2-7 BERT 候选 max_length 评审

- 实测覆盖率（在训练集 50,206 条上统计，以字符长度近似 token 长度）：

| max_length | 完整保留 | 被截断 |
| ---: | ---: | ---: |
| 32（原投满分项目取值） | 47.03% | 52.97%（26,596 条） |
| 128（推荐默认） | 89.73% | 10.27% |
| 256（高覆盖备选） | 97.88% | 2.12% |

- 结论：**不能照搬原项目的 `max_len=32`**，该值会截断超过一半的评论；
  推荐默认 `max_length=128`，高覆盖备选 `max_length=256`。

### 8. 自动化测试

- 完成内容：新增 24 项测试（清洗规则 11 项、划分与泄漏 5 项、真实数据 2 项、产物校验 5 项等），
  与原有 3 项合计 27 项。

### 9. 分词与词云分析（新增）

- 完成内容：新增 jieba 分词 + 停用词过滤 + 词频统计，并生成词云图与各类别高频词对比图。
  按 AGENTS.md 5.1「词云不能代替严谨统计」的要求，同时输出定量的高频词明细表。
- 涉及文件：`src/data/eda.py`（`collect_word_frequencies`、`plot_wordcloud`、
  `plot_top_words_by_category`、`cjk_font_file`、`is_meaningful_token`）、
  `scripts/run_eda.py`、`reports/figures/wordcloud_train.png`、
  `reports/figures/top_words_by_category.png`、`reports/eda-report.md` 第 6 节

- 实测结果：
  - 分词范围：训练集 50,206 条；分词与统计耗时 8.8 秒，词云渲染 2.1 秒；
    保留 735,747 个词次，词云展示 200 个词。
  - 停用词表：自建 110 个中文停用词（jieba 未对外暴露内置停用词表），
    并过滤长度小于 2 的词、以及既无中文也无拉丁字母的纯标点或纯数字串。
  - 结果与影响：6 张原图 + 2 张新图，共 **8 张图**；原报告的第 6 节顺延为第 7 节。
  - 分词方式为 jieba 精确模式；词云刻意不经 matplotlib 画布
    （wordcloud 直接输出位图，用 Pillow 叠加标题即可），因此它在问题 9 的
    matplotlib 故障期间仍能正常产出。

- 分析结论（已写入 EDA 报告 6.2 节）：
  用「该词进入多少个类别的 Top-15」衡量区分度，比「与第二名的词频倍数」更有说服力：
  「不错」进入 **10/10** 个类别的高频词表，属于完全通用的评价词；
  而「酒店 / 苹果 / 本书 / 功能 / 蒙牛 / 安装」只出现在 **1/10** 个类别中，
  是强类别信号。这一对比直接指导后续 TF-IDF 特征工程：
  通用评价词在各类别中分布接近，需检查其实际权重，必要时并入停用词表。

### 10. 接口契约合规改造（新增）

- 背景：组长建立了 `feature/api-contracts` 分支并起草 `docs/contracts.md`（v1.0），
  其中第一节「数据契约」规定 M2 必须产出特定格式的数据，并要求 HuMingChang 确认。
  用组长的 `scripts/verify_data_contract.py` 实测我的 M2 产物，结果为**未通过**。
- 发现的不符合项（5 处，全部在我这边）：

  | # | 契约要求 | 改造前实际 |
  | ---: | --- | --- |
  | 1 | `data/processed/label_mapping.json` | 不存在 |
  | 2 | `data/processed/split_manifest.json` | 不存在 |
  | 3 | 列顺序 `sample_id,review,cat,label` | 原为 `cat,label,review` |
  | 4 | `sample_id` 列（原始 CSV 行号，可追溯） | 完全没有 |
  | 5 | 各类别分层比例容差 ±1.0 行 | 水果 +1.6、洗发水 +1.4，超差 |

- 改造内容：
  1. 新建 `src/data/text_patterns.py`：集中文本判定正则与工具函数，
     并把「CJK 正则须用非 raw 字符串」「不可用 `\w`/`\W`/`\s` 判断中文」两条踩坑经验
     写进模块说明，供全项目复用；
  2. 新建 `src/data/raw_io.py`：原始数据读取、SHA-256，以及强制契约列顺序的
     `write_contract_csv`；该模块**不提供任何写入 `data/raw` 的函数**，
     从接口上杜绝误覆盖原始数据；
  3. 新建 `src/data/manifest.py`：生成 `label_mapping.json` 与 `split_manifest.json`，
     文件哈希与行数全部从真实文件重新计算；
  4. `cleaning.py` 跟踪 `sample_id`：在任何删行操作之前写入原始行号，
     因此删行、去重、划分之后依然能正确追溯（不能用 `reset_index` 充当，
     删行后行号会整体前移而错位）；
  5. `splitting.py` 新增配额修复：贪心分配以整组为单位，当类别含跨类别重复文本组时
     会略微超配，因此增加一轮「只搬运整组」的回填，把分层比例压进 ±0.5 行
     （严于契约的 ±1.0，留出余量）；
  6. 删除我重复实现的质量检查：`src/data/quality.py`、`scripts/check_data_quality.py`
     及对应报告，统一采用组长的 `src/data/check_raw.py`；
  7. 测试从 27 项扩充到 **45 项**，新增契约合规测试组，
     独立重算分层比例、泄漏计数与 `sample_id` 可追溯性后再与清单比对。

- 验收结果：组长的 `scripts/verify_data_contract.py` 输出
  **通过 19 项，失败 0 项，跳过 0 项 → 结论：数据契约校验通过**。

## 验证结果

- 执行命令：

```powershell
$py = "C:\Users\hmc45\Miniconda3\envs\AI\python.exe"
& $py scripts/clean_data.py
& $py scripts/split_data.py
& $py scripts/run_eda.py
& $py scripts/report_environment.py
& $py -m unittest discover -s tests -v
```

原始数据质量检查由组长实现的脚本承担：`python src/data/check_raw.py`
（产出 `reports/data-quality/raw_quality_report.md`）。

- 实际结果：
  - 四个数据脚本全部正常退出（exit code 0），报告与数据文件均已生成。
  - 测试结果：`Ran 45 tests` → `OK`（无失败、无跳过）。
  - 完整链路重跑两次，样本量与划分结果完全一致，流程可复现。
  - 图表中文渲染已用 matplotlib 缺字形警告检测验证：`missing-glyph warn: 0`。

- 尚未验证：
  - 图表仅做了程序化校验（文件大小 65–111 KB、无缺字形警告），**未做人工目视检查**，
    请 LanQvJian 与 HongSenYuan 打开 `reports/figures/` 确认排版与可读性。
  - BERT 覆盖率使用字符长度近似 token 长度，尚未用真实 tokenizer 复核（当前网络无法下载预训练模型），
    已在 `reports/eda-report.md` 第 6.3 节明确标注此项局限。
  - 传统模型与 BERT 尚未开始，属于 M3/M4 范围。

## 遇到的问题

### 问题 1：pandas 3.x 下正则统计出现严重误判

- 现象：质量报告一度显示「纯数字或纯符号评论」为 55,799 条，占全体 89%，明显不合理。
- 原因：环境中的 pandas 版本为 3.0.5，字符串正则被委托给 PyArrow/RE2 执行，
  而 RE2 的 `\w` / `\W` 只识别 ASCII，导致中文全部被 `\W` 匹配。
  同时 RE2 不支持 `\uXXXX` 转义，`r"[\u4e00-\u9fff]"` 直接抛出 `ArrowInvalid`。
- 解决方案：
  1. 中文范围改为非 raw 字符串，由 Python 先解析成真实字符再交给引擎；
  2. 「纯符号文本」改用「既无中文也无拉丁字母」的显式定义，不再依赖 `\W`；
  3. 增加回归测试 `test_meaningless_metric_is_not_inflated`，防止该问题再次出现。
- 是否已解决：已解决。修正后该指标为 2 条，与人工抽样一致。

### 问题 2：pandas 3.x 对可空 Int64 列执行 clip 抛错

- 现象：长度直方图生成时报 `TypeError: Invalid value '504.24' for dtype 'Int64'`。
- 原因：分位数是浮点数，而 `.str.len()` 在 pandas 3.x 下返回可空 `Int64`，`clip(upper=float)` 不兼容。
- 解决方案：绘图前显式转换为 `float64`。
- 是否已解决：已解决。

### 问题 3：重复文本跨类别，直接全局去重会丢数据

- 现象：初步统计「按评论文本全局去重可删除 49 行」，看似应直接去重。
- 原因：进一步实测发现 45 种重复文本中，**28 种出现在多个类别下**（涉及 60 行）。
  这类记录不是冗余数据，而是同一文本对应多个合法标签，全局去重会删除其中 32 行有效样本。
- 解决方案：清洗阶段只删除同一类别内的完全重复记录（19 行）；
  跨类别的相同文本保留，改由划分阶段以文本为单位整体分配，从而既保住样本又保证零泄漏。
- 是否已解决：已解决，并已由测试 `test_same_text_under_different_categories_is_preserved`
  与 `test_multi_category_text_stays_in_one_split` 固化。

### 问题 4：EDA 报告出现结论与数字不匹配

- 现象：报告一度写成「删除量最大的是 drop_missing_or_blank（18 条）」，
  而 18 条实际来自 `deduplicate_within_category`，步骤名与数字张冠李戴。
- 原因：取最大值的代码写成了取「第一个有删除的步骤」，却用 `max()` 取数字。
- 解决方案：改为对 (步骤名, 删除量) 整体取最大值，并补充无删除时的分支。
- 是否已解决：已解决，报告已重新生成并核对。

### 问题 5：`uvicorn[standard]` 被依赖比对脚本静默跳过

- 现象：依赖报告首次只有 15 行，缺少 requirements.txt 中的 `uvicorn`。
- 原因：解析正则不支持 `[extras]` 写法，带 extras 的行整体匹配失败后被静默忽略。
- 解决方案：正则增加可选 extras 分组，并校验「非注释行数 = 解析行数」（16 = 16）。
- 是否已解决：已解决。

### 问题 6：PyPI 无法访问

- 现象：`pip download` 超时，`Invoke-WebRequest` 报 `The SSL connection could not be established`，
  三个常用镜像源均失败；TCP 443 端口可连通。
- 影响：当前环境**无法安装新依赖**，只能使用已有 conda 环境。
- 现状：已改用预装了全部依赖的 conda 环境 `AI`，M2 未受影响。
  但后续若需安装新包（例如模型优化阶段的 `optimum`、`onnxruntime`），必须先解决网络或改用离线包。
- 是否已解决：M2 已绕过，网络问题本身**未解决**，需要团队确认。

### 问题 7：BOM 等不可见字符逃过清洗，成为 TF-IDF 垃圾 token

- 现象：进入 M3 做分词时，第一条训练评论被 jieba 切出的第一个 token 是 `'\ufeff'`。
- 原因：原始数据中有 **20 条评论文本内部嵌入了 U+FEFF（BOM / 零宽不换行空格）**。
  这类字符属于 Unicode 格式字符（Cf 类），而先前的清洗规则只覆盖 ASCII 控制字符
  （`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`），因此完全没有处理。
  它不可见，人工检查看不出来，但会被分词器切成一个独立 token 进入词表。
- 解决方案：
  1. 质量检查增加「含 Unicode 格式字符」与「含 Unicode 空白字符」两项统计，
     让问题可见而不是被静默修掉（实测分别为 20 条和 52 条）。
  2. 清洗新增 `strip_unicode_format_chars` 步骤，按 Unicode Cf 类删除字符并保留样本，
     因此零宽连接符、软连字符等同类问题一并覆盖。
  3. 报告样例中把不可见字符渲染为 `[U+FEFF]`，使证据可核查。
- 是否已解决：已解决。修复后清洗产物中 U+FEFF 残留为 **0 条**；
  该步骤修改 20 行，并因 BOM 去除后产生 1 条新的同类别重复而多删 1 行，
  最终样本量由 62,753 变为 62,752。

### 问题 8：pandas 的空白归一化不认全角空格（潜在缺陷）

- 现象：排查问题 7 时发现，`DataFrame.str.replace(r"\s+", " ")` 对
  `"a\u3000b"` 与 `"a\u00a0b"`（全角空格、不换行空格）**完全无效**。
- 原因：与问题 1 同源。pandas 3.x 把字符串正则交给 PyArrow/RE2，RE2 的 `\s`
  只等价于 ASCII 的 `[\t\n\f\r ]`，不匹配任何 Unicode 空白字符。
- 影响：原始数据中 52 行含全角空格，恰好都出现在文本首尾，被 `.str.strip()`
  兜住了，所以当前结果没有出错；但一旦全角空格出现在文本中间就会漏掉，
  造成「同一文本因排版差异被当成两个样本」的隐患。
- 解决方案：空白归一化改为在 Python 侧执行（`re` 是 Unicode 感知的），
  实测该改动使现有数据 0 行发生变化，属于无副作用的健壮性修复；
  并新增回归测试 `test_unicode_spaces_are_normalised` 固化该行为。
- 是否已解决：已解决。

### 问题 9：安装 wordcloud 导致 matplotlib 全部崩溃（影响全队）

- 现象：为生成词云安装 wordcloud 后，matplotlib 保存任何图片都会**原生崩溃**，
  进程退出码 `0xC06D007F`，即使是一张没有标题的普通柱状图也无法保存。
  由于是原生崩溃而非 Python 异常，`try/except` 无法捕获，也没有任何 Python 报错信息。
- 定位过程（逐步二分，每步都在独立进程中验证）：
  1. 全量分词与词频统计**正常**（50,206 条、10.3 秒）→ 排除 jieba 与数据问题；
  2. 纯 Pillow 存图 `PIL SAVE OK` → 排除 Pillow 本身损坏；
  3. `fig.canvas.draw()` 崩溃、`savefig()` 崩溃 → 锁定在 matplotlib 的 Agg 渲染；
  4. 查 conda 版本快照：`rev 0` 里**只有 Python 本身**，没有任何 matplotlib/numpy，
     说明这些 ML 包原本是 pip 安装的；当天 `rev 1` 用 conda-forge 构建覆盖了它们；
  5. 查包来源频道：`matplotlib-base`、`pillow`、`numpy`、`freetype`、`libpng` 来自
     conda-forge，而 `libtiff` 来自 `repo.anaconda.com/pkgs/main`（defaults 频道），
     属于频道混用；
  6. 查 DLL：`Library\bin` 中 `libtiff.dll` 与 `tiff.dll` **并存**，
     `zlib.dll` 出现 3 份 → 确认存在重复/冲突的动态库。
- 根本原因：conda 安装 wordcloud 时，把原本由 pip 安装的 matplotlib/numpy/pillow
  覆盖成 conda-forge 构建，同时混入 defaults 频道的 libtiff，产生互不兼容的重复 DLL，
  导致 conda-forge 版 matplotlib 的 Agg 渲染扩展（`_backend_agg.pyd`）加载时崩溃。
- 解决方案：改用 pip 从清华镜像重装图像栈，让整条链路来自同一来源。
  pip 的 wheel 自带捆绑 DLL，不受 `Library\bin` 中重复 DLL 的影响：

  ```powershell
  & C:\Users\hmc45\Miniconda3\envs\AI\python.exe -m pip install --force-reinstall --no-deps `
      matplotlib==3.11.2 pillow numpy `
      -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
  ```

- 验证证据（全部实际执行）：
  - `matplotlib savefig OK`，退出码 0；
  - 全链路 5 个脚本全部 exit 0；
  - **重新生成的 6 张图表字节数与故障前完全一致**
    （74,907 / 65,387 / 80,291 / 65,848 / 111,113 / 64,414 字节），
    说明图像栈已恢复到与故障前完全一致的状态；
  - 中文字形检查：`缺字形警告: 0`；
  - 测试套件 `Ran 27 tests` → `OK`；
  - 故障期间已生成的 6 张图经验证**完好无损**（崩溃发生在写文件之前）。
- 附带改进：`scripts/report_environment.py` 的检查范围新增 `pillow` 与 `wordcloud`
  两个传递依赖。本次之所以没能预警，正是因为该脚本只检查 `requirements.txt` 里的
  直接依赖，而 Pillow 不在其中。
- 是否已解决：已解决。但需注意这是一次**环境层面的教训**：
  该 conda 环境本身是 pip 与 conda 混装的，今后再往该环境安装任何包，
  都可能再次覆盖 pip 管理的包，安装前应先确认 conda 的变更计划。
- 遗留事项：`requirements.txt` 需新增 `wordcloud>=1.9,<2.0`（该文件由 LanQvJian 维护，
  我未改动）。

### 问题 10：与组长重复实现了原始数据质量检查

- 现象：组长在 `feature/api-contracts` 分支新建了 `src/data/check_raw.py`（770 行）
  与 `reports/data-quality/raw_quality_report.*`，与我的 `src/data/quality.py`（465 行）
  和 `reports/data-quality-report.md` 功能完全重复——两者都是只读的原始数据质量检查，
  都产出 Markdown + JSON 报告。而 `src/data/` 按 `docs/team.md` 属于数据工程职责。

- 原因：**我没有提交 M2。** 此前讨论 M2 提交方式时决定「暂不提交，等复核」，
  组长只能从只有 M1 的 `main` 出发，看不到我的实现，于是自己写了一份。
  这不是他的问题，是提交策略造成的可见性问题。

- 处置：按 HuMingChang 决定，**以组长的 `check_raw.py` 为准，我方退让**。
  已删除我的 `src/data/quality.py`、`scripts/check_data_quality.py` 及两份报告；
  两者共用的文本判定标准抽到 `src/data/text_patterns.py`，避免今后再次各写一份。

- 交叉验证（重复实现的意外收获）：两份独立实现的关键数字**完全一致**——
  行数 62,774、缺失 1、完全重复行 0、重复文本 49 行/45 组、控制字符 1、
  HTML 实体 49、最长 2,876、中位数 34.0。这等于对 M2 数据事实做了一次独立复核。

- 审查中发现的问题（`check_raw.py` 已成为唯一实现，故一并记录）：
  1. 第 270 行 `frame[text].str.strip().str.replace(r"\s+", " ", regex=True)` 走 RE2，
     而 RE2 的 `\s` 只识别 ASCII，**无法合并全角空格（U+3000）与不换行空格（U+00A0）**。
     实测有 52 行的归一化结果因此与 Python Unicode 口径不同。
     当前重复统计数字恰好未受影响（两种口径都是 96 行 / 46 组 / 可删 50），
     属于**潜伏缺陷**，一旦全角空格出现在会构成重复对的位置就会静默少报。建议改用 Python 的 `re`。
  2. 其报告称「冲突重复组：45（涉及 94 行）……既无法直接用于训练」，
     表述不够精确：45 种重复文本中只有 **28 种跨类别**，另 **17 种是同类别内纯冗余**。
     两者处理方式不同（同类冗余删除；跨类别的相同文本是同一文本对应多个合法标签，
     应保留并靠分组划分防泄漏）。按其表述会把 28 种有效配对误删。
  3. 他做对的地方：CJK 正则用字面量字符（`[一-鿿]`）书写，成功绕开 RE2 的 `\u` 陷阱，
     且未使用 `\W` 判断中文噪声。

- 是否已解决：重复实现已收敛。上述两条建议需转告组长，由他决定是否修改
  （该文件现由他主责，我不擅自改动）。

## 明日计划

- 进入 M3：TF-IDF 特征工程 + 逻辑回归 / LinearSVC 基线 + 随机森林对比。
- 保存向量器、标签映射与模型产物，确保推理阶段复用同一套预处理。
- 在验证集上以 Macro-F1 选模，输出分类报告与混淆矩阵。
- 与 HongSenYuan 对齐统一评估接口，保证三个模型使用同一测试集可比。

## 需要协助

1. **（LanQvJian）`requirements.txt` 版本约束与实际环境不一致**：
   约束为 `pandas>=2.1,<3.0`，实际为 3.0.5。
   需要确认是放宽约束，还是调整环境。详见 `reports/environment.md`。
2. **（LanQvJian）`reports/figures/*` 被 `.gitignore` 排除**：
   图表由脚本可复现生成，但答辩 PPT 需要图片文件。
   请确认是保留该规则（答辩前现场重跑脚本），还是单独放开必要图片。
3. **（全网）PyPI 不可达**：请确认是否统一走镜像源或使用离线 wheel 包。
4. **（LanQvJian）分支命名**：当前本地分支为 `hmc`，
   而 `docs/team.md` 建议 `feature/data-eda` / `feature/tfidf-baseline`。
   请确认是否按规范重命名或在推送时使用规范分支名。
