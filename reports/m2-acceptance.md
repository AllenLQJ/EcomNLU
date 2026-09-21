# M2 数据闭环正式验收记录

## 验收结论

M2 于 2026-09-21 由 LanQvJian 以非作者身份完成复核，验收结论为：**通过**。

## 验收环境

- Conda 环境：`py310`
- Python：3.10.20
- pandas：2.3.3（满足 `>=2.1,<3.0`）
- NumPy：2.2.6
- scikit-learn：1.7.2
- jieba：0.42.1
- wordcloud：1.9.6
- Pillow：12.2.0

## 实际复验结果

1. 从原始数据重新执行清洗：62,774 条变为 62,752 条，共删除 22 条。
2. 使用随机种子 42 重新执行 8:1:1 分层划分：训练集 50,206 条、验证集 6,274 条、测试集 6,272 条。
3. train/dev/test 两两重复的规范化评论文本均为 0。
4. `scripts/verify_data_contract.py`：19 项通过、0 项失败、0 项跳过。
5. `python -m unittest discover -s tests -v`：60 项通过、0 项失败、0 项跳过。
6. 重新生成并目视检查 8 张 EDA 图：类别分布、类别内情感、长度直方图、长度箱线图、长度 CDF、清洗漏斗、训练集词云和分类别高频词均可读。
7. BERT 候选截断长度覆盖率重新核对：128 为 89.73%，256 为 97.88%，因此保留“推荐 128、高覆盖备选 256、不得照搬 32”的结论。

## 复现命令

```powershell
& D:\My_Software\Anaconda3\envs\py310\python.exe scripts/clean_data.py
& D:\My_Software\Anaconda3\envs\py310\python.exe scripts/split_data.py
& D:\My_Software\Anaconda3\envs\py310\python.exe scripts/run_eda.py
& D:\My_Software\Anaconda3\envs\py310\python.exe scripts/verify_data_contract.py
& D:\My_Software\Anaconda3\envs\py310\python.exe -m unittest discover -s tests -v
```

原始数据、处理中间数据和生成图表按 `.gitignore` 管理，不直接提交；上述产物可由脚本重复生成。
