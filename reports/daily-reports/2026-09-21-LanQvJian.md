# 日报：2026-09-21 - LanQvJian

> 本文件由 Git 提交记录与仓库产物整理，事实与哈希可核对。
> 标注 `<!-- 待补充 -->` 处需本人填写。

## 今日完成

当天 18 次提交，主线是**契约治理 + 统一推理入口 + Web 骨架**，为 M3/M4/M5 并行开发提供共同接口。

- 完成内容（按主线归纳）：
  1. **契约从 v1.0 演进到 v1.2**：定义共享数据模型与 API 接口（v1.0）→ 勘误数据集指纹算法为 v1.0.1 →
     启用 M4 设备记录规则（v1.1）→ 起草 v1.2 的数据与情感规则。
  2. **统一推理入口**：新增模型适配器层，让传统模型、FastText、BERT 走同一个 `src.inference.load_predictor`，
     保证「离线评测口径 = 线上推理口径」。
  3. **M6 Web 骨架**：按说明书实现同源演示页面，并补上契约合规的预测 API。
  4. **M4 验收闭环**：合并已复核的 M4 实现、处理验收阻塞项、记录验收完成。
  5. **M5 并行启动**：划分 M4 复核与 M5 并行任务，收口情感契约与验收口径。
- 涉及文件：
  - `docs/contracts.md`（v1.0 → v1.0.1 → v1.1 → v1.2 连续修订）
  - `src/inference/`（统一入口与各模型适配器）、`scripts/verify_data_contract.py`、`src/data/check_raw.py`
  - `src/web/`（`app.py`、页面与静态资源）
  - `docs/development-plan.md`（M2/M4/M5 状态同步）
- Git 分支/提交（真实哈希，按时间）：

  | 时间 | 提交 | 说明 |
  | --- | --- | --- |
  | 09:20 | `9a74398` | test(baseline): 路径解析用例改用必然存在的目录，去掉对模型产物的依赖 |
  | 10:13 | `0049f75` | feat(contract): define shared data model and API interfaces |
  | 11:33 | `6148bb2` | feat(inference): add unified model predictor adapters |
  | 12:00 | `42f79a1` | fix(docs): align M2 environment and EDA dependencies |
  | 12:03 | `daf37ca` | merge: integrate inference adapters into M2 baseline |
  | 13:10 | `99ee20f` | feat(web): 按说明书实现 M6 同源演示页面 |
  | 16:40 | `f73bfdb` | docs(contract): activate v1.0.1 after team confirmation |
  | 17:02 | `2eb6a32` | docs(contract): activate v1.1 device rule for M4 |
  | 17:02 | `9ad148f` | feat(web): add contract-compliant prediction API |
  | 17:36 | `c569c1c` | docs(contract): draft v1.2 data and sentiment rules |
  | 19:38 | `54e7360` | docs(plan): define M4 review and M5 parallel tasks |
  | 20:00 | `f709352` | merge: integrate reviewed M4 implementation |
  | 20:03 | `2b3526e` | fix(m4): resolve final acceptance blockers |
  | 20:35 | `7a3a1c2` | docs(validation): complete M4 final acceptance |
  | 20:41 | `01a8fba` | docs(plan): mark M4 complete and M5 in progress |
  | 20:46 | `b911fd1` | docs(plan): sync M4 merge and M5 progress |
  | 20:50 | `9454abe` | docs(plan): mark M2 review complete |
  | 21:36 | `6bc3552` | fix(m5): finalize sentiment contract and acceptance |

## 验证结果

- 执行命令：`python scripts/verify_data_contract.py` / `python -m unittest discover -s tests`
- 实际结果：数据契约校验 **19 项通过、0 失败、0 跳过**（该结果由 HuMingChang 在 M2 侧实测记录）；
  当日全量测试通过，Web 前端与接口测试覆盖新增的预测 API。
- 尚未验证：
  1. M6 页面当次只验证了同源加载与接口连通，**尚未接入真实 BERT 模型**；
  2. 情感模型尚未接入页面（等 M5 交付）。

## 遇到的问题

**问题 1：v1.0 的数据集指纹任何人都无法稳定满足（规范缺陷）**

- 现象：连续两次运行 `scripts/split_data.py`（种子 42），三个 CSV 逐字节相同，但
  `split_manifest.json` 哈希发生变化（`10c21992…` → `32bb978b…`），导致
  `scripts/verify_model_contract.py` 持续报「该模型可能是在旧数据上训练的」。
- 原因：v1.0 规定的 `split_manifest_sha256` 把清单里的 `created_at` 一并纳入哈希，
  重新生成数据后**必然漂移**——这是**规范本身的缺陷，不是实现错误**。
- 解决方案：勘误为 **v1.0.1**（明确「属规范缺陷，不占用 v1.1 版本号」），字段改名
  `dataset_fingerprint`，取值改为 `sha256(":".join([sha256(train), sha256(dev), sha256(test)]))`。
- 是否已解决：**已解决**。交叉核对确认同一指纹在「契约文档算法 / 校验器实现 / M3 实现」
  三处、两台不同机器上完全一致。

**问题 2：FastText 在统一入口下 dev 准确率只有 16.94%**

- 现象：接入统一入口后 FastText dev 准确率 **0.1694**（约等于全猜多数类），而 fastText 自带
  `model.test()` 在同一份 dev 文件上是 **0.8731**。
- 原因：适配器把**原始句子**直接交给 fastText，但模型是在**空格分隔的 token** 上训练的；
  fastText 只按空格切词，整句退化成一个 token。另有一处：fastText 概率来自 float32 softmax，
  实测会出现 `1.0000100135803223` 这类略大于 1 的值，原严格校验会让**任何 FastText 模型预测时报错**。
- 解决方案：适配器按 `metadata.json` 的 `preprocess.tokenize` 复用训练切分
  （`none` / `chars` / `jieba_words`）；概率允许 1e-4 容差后夹紧到 `[0,1]`。
  该修改落在 `src/inference/fasttext_adapter.py`（**本人主责文件**），由 HongSenYuan 提出并留了回退方案。
- 是否已解决：**已解决**。dev 准确率由 0.1694 回到 0.8715，与 fastText 自带 `test()` 的 0.8731 一致。

<!-- 待补充：当天是否还有其他你认为值得记录的问题 -->

## 明日计划

- 把真实 BERT 模型接入 Web 服务，完成类别 + 情感的端到端预测。
- 修复量化产物的环境依赖与元数据问题（HongSenYuan 已提交量化实现，待集成验证）。
- 准备答辩材料的离线备用方案。

## 需要协助

- 需要 HongSenYuan 确认量化模型接入 Web 的启动参数与设备要求（int8 动态量化只能跑 CPU）。
- 需要 HuMingChang 提供情感模型目录的最终路径与标签顺序，供页面展示。
