# 数据说明

## 数据集

- 名称：`online_shopping_10_cats`
- 来源仓库：`SophonPlus/ChineseNlpCorpus`
- 下载地址：`https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/datasets/online_shopping_10_cats/online_shopping_10_cats.zip`
- 本项目固定文件 SHA-256：`f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb`
- 原始 CSV 大小：11,299,695 字节
- 原始行数：62,774
- 文件编码：UTF-8

## 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `cat` | 字符串 | 商品或服务类别，主任务标签 |
| `label` | 整数 | 情感标签，0 为负向、1 为正向 |
| `review` | 字符串 | 中文评论文本 |

## 已知质量情况

M1 只做原始文件级校验，已知原始数据至少包含：

- 1 条缺失评论
- 49 条重复评论文本
- 明显的类别不均衡
- 超短和超长评论

这些问题将在 M2 中通过可复现脚本处理。不得手工直接修改 `data/raw` 中的文件。

## 使用和许可提示

上游仓库将该数据集整理为教学和 NLP 研究语料，但当前数据目录未提供清晰、独立的数据许可证。本项目将其用于课程实战和学习研究；若需要公开发布、商业使用或重新分发原始数据，必须再次核对上游许可和原始语料权利，不得仅依据本说明推定已获得授权。

原始数据不提交 Git。团队成员运行：

```powershell
python scripts/download_data.py
```

即可得到相同文件。
