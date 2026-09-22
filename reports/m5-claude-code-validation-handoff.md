# M5 最终检查与测试交接单（Claude Code）

## 1. 任务目标

请对尚未合入主线的 M5 BERT 动态量化交付做独立检查、修复和复验。

当前结论是：**M5 组合代码基本通过，但还不能合并到 `main`。** 请不要只复述现有报告，也不要在阻断项未解除前直接合并主线。

本次检查对象：

- HongSenYuan 开发分支：`origin/feature/bert-optimization`
- HuMingChang 组合复核分支：`origin/feature/sentiment-baseline`
- 当前主线：`origin/main`
- 指定运行环境：Conda `py310`
- Python：`D:\My_Software\Anaconda3\envs\py310\python.exe`

开始前必须执行 `git fetch origin --prune`，重新读取远端 SHA；本文记录的 SHA 只代表 2026-09-22 检查时的状态：

- `origin/feature/bert-optimization`：`ad9197d`
- `origin/feature/sentiment-baseline`：`e351c76`
- `origin/main`：`4bd2e0f`

## 2. 已确认通过的内容

在组合分支 `e351c76` 上、使用 Conda `py310`，已取得以下实际结果：

- `pytest`：**175 passed, 3 skipped, 0 failed**。
- `compileall`：通过。
- M5 情感基线已单独完成验收并合入 `main`。
- BERT 动态量化实现、报告和对比 JSON 已提交。
- 组合分支相对当时主线的 `git diff --check` 没有空白错误。
- HuMingChang 的合并报告记录其环境下 `unittest` 178 项通过，但报告同时明确承认没有量化模型产物，不能替代本次独立产物验收。

现有量化报告声称：

- 动态量化 74 个 Linear 层；
- 模型体积 390.17 MB → 145.58 MB；
- dev Macro-F1 0.912276 → 0.904659；
- test Macro-F1 0.913847 → 0.900508；
- 量化模型曾重新加载并执行探针推理。

以上数字有报告和 JSON，但当前验收机没有 HongSenYuan 的原始量化产物，因此尚未独立核验。

## 3. 当前两个阻断项

### 阻断项一：缺少完整量化模型产物

当前共享 `models/` 中存在：

```text
models/bert_base_chinese/
models/sentiment_tfidf_logreg/
models/sentiment_tfidf_svm/
```

但不存在可验收的：

```text
models/bert_base_chinese_quantized/
```

HongSenYuan 应提供完整目录，至少包含：

```text
models/bert_base_chinese_quantized/
├── model/
│   ├── config.json
│   └── quantized_state_dict.pt
├── tokenizer/
├── label_mapping.json
├── metadata.json
└── metrics.json
```

模型文件受 `.gitignore` 管理，不应提交 Git，但必须通过团队共享目录交给最终验收人。

### 阻断项二：组合分支落后最新主线

检查时 `origin/feature/sentiment-baseline` 相对 `origin/main` 为：

```text
behind 2 / ahead 9
```

缺少的主线提交为：

```text
99ee20f feat(web): 按说明书实现 M6 同源演示页面
4bd2e0f feat(web): complete frontend and sentiment integration
```

这两项涉及 Web 与情感集成，不能用“只是文档更新”跳过。应先把最新 `origin/main` 合入一个新的 M5 验收修复分支，再运行完整测试和 Web/API 联调。不要直接在受保护的 `main` 上试合并。

## 4. `py310` 中已复现的量化失败

使用以下命令尝试从已验收的 fp32 BERT 重建量化产物：

```powershell
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' `
  scripts\quantize_bert.py `
  --skip-eval `
  --target models\bert_base_chinese_quantized_validation
```

实际流程完成了模型加载和 74 个 Linear 层量化，但在保存 `state_dict` 时失败：

```text
RuntimeError: module compiled against ABI version 0x1000009
but this version of numpy is 0x2000000

ImportError: numpy.core.multiarray failed to import
```

失败栈经过：

```text
torch.save(model.state_dict())
→ transformers lazy import
→ import cv2
→ NumPy / OpenCV ABI 不兼容
```

中断后目标目录只留下约 713 字节的 `quantized_state_dict.pt`，属于损坏文件；同时缺少 `label_mapping.json`、`metadata.json`、`metrics.json`。后续结果为：

- 模型契约失败；
- `src.inference.load_predictor` 加载失败；
- 真实预测因缺少 `metadata.json` 失败。

请先判断并修复以下两种可能性：

1. `py310` 中 OpenCV 是按 NumPy 1.x ABI 编译，而当前安装的是 NumPy 2.x；需要升级兼容的 OpenCV，或将 NumPy 调整到兼容版本并同步依赖约束。
2. 量化保存实现不应在序列化纯 PyTorch `state_dict` 时触发无关的 Transformers/OpenCV 延迟导入；如能通过更安全、任务无关的序列化方式避免该依赖，应补代码和回归测试。

不得仅删除错误提示或跳过保存。修复后必须生成正常大小的量化权重并重新加载。

## 5. 必须执行的检查顺序

### 5.1 创建隔离验收分支

```powershell
git fetch origin --prune
git switch -c fix/m5-quantization-final origin/feature/sentiment-baseline
git merge origin/main
```

若出现冲突，应按 v1.2 契约和最新 Web/情感集成语义解决，不得覆盖其他成员的新代码。

### 5.2 检查环境

```powershell
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' -c `
  "import sys,numpy,torch,transformers; print(sys.version); print('numpy',numpy.__version__); print('torch',torch.__version__); print('transformers',transformers.__version__)"

& 'D:\My_Software\Anaconda3\envs\py310\python.exe' -c `
  "import cv2; print('cv2',cv2.__version__)"
```

把最终采用的 NumPy、OpenCV、PyTorch、Transformers 版本写入验收记录。

### 5.3 重新生成或接收量化产物

优先使用 HongSenYuan 提供的原始完整产物进行独立校验。如果必须重建，先运行快速路径：

```powershell
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' `
  scripts\quantize_bert.py --skip-eval
```

快速路径结束后必须确认：

- 量化权重不是 0 字节或几百字节的中断文件；
- `label_mapping.json`、`metadata.json` 已写出；
- `verify_quantized_model` 的三条探针真实执行成功。

如果 `--skip-eval` 按设计不写 `metrics.json`，文档和帮助文本必须明确“该目录尚不满足完整模型契约”；不得把快速产物描述成可部署产物。

完整 dev/test 评测耗时很长。如直接采用 HongSenYuan 的完整产物，可以不重复跑两个完整划分，但必须核对其 `metrics.json`、数据集指纹及报告 JSON 一致，并至少完成真实重载和探针预测。

### 5.4 模型契约与真实推理

```powershell
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' `
  scripts\verify_model_contract.py models\bert_base_chinese_quantized

& 'D:\My_Software\Anaconda3\envs\py310\python.exe' `
  scripts\predict.py --model models\bert_base_chinese_quantized `
  --text '这本书内容很好，印刷清晰，非常值得推荐' --top-k 3

& 'D:\My_Software\Anaconda3\envs\py310\python.exe' `
  scripts\predict.py --model models\bert_base_chinese_quantized `
  --text '酒店的床单有异味，前台态度也不好' --top-k 3
```

预期：模型契约 15/15 通过，两条预测均返回合法的 10 类概率、概率和约为 1，并能从磁盘重新加载。

### 5.5 数据、测试与编译

```powershell
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' scripts\verify_data_contract.py
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' -m pytest -q
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' -m unittest discover -s tests
& 'D:\My_Software\Anaconda3\envs\py310\python.exe' -m compileall -q scripts src tests
git diff --check
```

### 5.6 Web/API 联调

最新 `main` 已包含前端和情感集成。至少验证：

- 类别 fp32 BERT 能正常加载；
- 情感模型能正常加载并返回正负情感；
- 将主类别模型切换为量化 BERT 后，健康检查仍为可用状态；
- `/api/predict` 返回合法类别、置信度、Top-K、情感和耗时；
- 空输入、超长输入仍按契约报错。

## 6. 文档修正要求

`reports/m5-merge-test-report.md` 当前第五节已过期：它写着分支尚未包含 `4c2ddc3`，但后续提交 `e351c76` 已同步该提交；与此同时，当前分支又落后更新的 `99ee20f` 和 `4bd2e0f`。完成同步和复验后，应更新：

- 当前 `main` SHA；
- 组合分支 SHA；
- 实际测试数量和结果；
- 量化模型契约与真实推理结果；
- Web/API 联调结果；
- 仍未验证的项目。

不得保留已经完成的事项为“未验证”，也不得把没有实际运行的项目写成通过。

## 7. 最终通过标准

只有同时满足以下条件，才能回复“可以合并 M5”：

1. 组合分支包含最新 `origin/main`，落后数为 0。
2. 提供完整且可读取的 `bert_base_chinese_quantized` 产物。
3. 量化模型契约 15/15 通过。
4. 量化模型从磁盘重新加载并完成真实预测。
5. 数据集指纹与 M3/M4/M5 其它模型一致。
6. 完整自动化测试无失败。
7. 最新 Web/API 与量化类别模型、情感模型联调通过。
8. 合并测试报告更新为当前真实状态。
9. 模型权重继续由 `.gitignore` 排除，不提交 Git。

完成后请提供：改动文件清单、验证命令、实际结果、仍未解决事项、提交 SHA 和建议合并分支；不要自行合并 `main`。
