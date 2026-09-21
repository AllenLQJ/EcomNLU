"""渲染情感二分类基线的对比报告。

模块职责：
    把训练脚本产出的模型结果、逐类别指标与错误分析统计渲染成一份 Markdown 报告，
    覆盖 M5 任务要求报告的全部内容：Accuracy、Macro Precision、Macro Recall、
    Macro-F1、混淆矩阵，以及各商品类别内的情感表现。

主要输入与输出：
    输入：``{模型名: {"dev":…, "test":…, "cost":…}}``、选定模型名、情感标签顺序、
        数据集摘要、特征配置、特征方案对照、逐商品类别指标、错误分析统计。
    输出：完整的 Markdown 报告文本。

关键设计理由与约束：

1. **选模依据与最终确认分开呈现。** 报告第 3 节把「验证集选模」与「测试集确认」
   分成两列，避免读者误以为测试集参与了选模（AGENTS.md §6）。

2. **不臆测错误的语义原因。** 报告只陈述从真实误分类样本中能观察到的事实，
   典型错误分类的解读放在独立的错误分析文档里，并逐条附上原始评论。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def render_sentiment_report(
    *,
    model_results: Mapping[str, Mapping[str, Any]],
    selected_model: str,
    labels: Sequence[str],
    selection_metric: str,
    dataset_summary: Mapping[str, Any],
    vectorizer_config: Mapping[str, Any],
    ablation_results: Sequence[Mapping[str, Any]] = (),
    per_product_category: Mapping[str, Mapping[str, Any]] | None = None,
    error_summary: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> str:
    """渲染情感二分类基线报告。

    Args:
        model_results: ``{模型名: {"dev": 指标块, "test": 指标块, "cost": 成本块,
            "hyperparameters": …, "description": …}}``。
        selected_model: 依据验证集指标选出的模型名。
        labels: 情感标签顺序，固定为 ``["负向", "正向"]``。
        selection_metric: 选模依据的指标名，本项目为 ``macro_f1``。
        dataset_summary: 数据规模摘要，需含 ``train_rows`` / ``dev_rows`` /
            ``test_rows`` / ``random_seed`` / ``dataset_fingerprint`` /
            ``positive_rate``。
        vectorizer_config: 特征配置摘要。
        ablation_results: 特征方案对照结果，每项含 ``name``、``macro_f1``、``note``。
        per_product_category: 选定模型在各商品类别内的情感指标块。
        error_summary: 错误分析统计，由训练脚本按启发式标记汇总。
        notes: 追加到「结论与已知局限」的条目。

    Returns:
        完整的 Markdown 报告文本。
    """
    label_list = [str(label) for label in labels]
    negative_name, positive_name = label_list[0], label_list[1]

    lines: list[str] = [
        "# 情感二分类基线报告（M5）",
        "",
        "本报告由 `scripts/train_sentiment.py` 生成，所有指标均来自真实运行结果，",
        "未运行过的实验一律标注为「未验证」，不作估计。",
        "",
        "## 1. 评估口径",
        "",
        f"- 数据规模：训练集 {dataset_summary['train_rows']:,} 条、"
        f"验证集 {dataset_summary['dev_rows']:,} 条、测试集 {dataset_summary['test_rows']:,} 条",
        "- 标签来源：**直接取数据集自带的数字标签列 `label`（取值只有 0 和 1）**，"
        "未重新制作标签、未做任何人工标注；本任务只把该列按固定对应关系重命名以便阅读：",
        "",
        "  | 原始 `label` 值 | 含义 | 情感类名 |",
        "  | ---: | --- | --- |",
        f"  | `0` | 差评 | {negative_name} |",
        f"  | `1` | 好评 | {positive_name} |",
        "",
        "  该对应关系是 1 对 1 且不可调换的（`0` 为负向、`1` 为正向），"
        "定义在 `configs/base.yaml` 的 `task.sentiment_categories`，"
        "训练脚本、契约校验器与 Web 层都读同一份配置，不各自硬编码。",
        "",
        f"- 标签顺序：{'、'.join(label_list)}（即原始 `label` 的 0/1 顺序）",
        f"- 整体正例占比（`label == 1`）：训练集 {dataset_summary['positive_rate']['train']:.2%}、"
        f"验证集 {dataset_summary['positive_rate']['dev']:.2%}、"
        f"测试集 {dataset_summary['positive_rate']['test']:.2%}",
        f"- **选模依据：验证集 {selection_metric}**，测试集在选模完成后才评估，"
        "仅用于确认与横向对比",
        f"- 训练集划分随机种子：{dataset_summary['random_seed']}",
        f"- 数据集指纹：`{dataset_summary['dataset_fingerprint'][:16]}…`"
        "（按契约 §2.3 由 train/dev/test 三个 CSV 的内容哈希算出）",
        f"- 与 M3 类别任务的数据集指纹一致：**{'是' if dataset_summary.get('fingerprint_matches_m3') else '否'}**"
        "（不一致说明两者使用的不是同一份划分，对比无效）",
        "",
        "## 2. 特征配置",
        "",
        "| 项目 | 取值 |",
        "| --- | --- |",
    ]
    for key, value in vectorizer_config.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## 3. 模型对比",
            "",
            f"按验证集 {selection_metric} 排序，**最优为 `{selected_model}`**：",
            "",
            "| 模型 | 验证集 Macro-F1 | 验证集 Accuracy | 测试集 Macro-F1 | "
            "测试集 Accuracy | 模型大小 | 单条延迟 | 批量32延迟 | 吞吐量 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    ordered = sorted(
        model_results.items(), key=lambda item: item[1]["dev"]["macro_f1"], reverse=True
    )
    for name, result in ordered:
        dev = result["dev"]
        test = result.get("test")
        cost = result["cost"]
        marker = " ← 选定" if name == selected_model else ""
        test_f1 = f"{test['macro_f1']:.4f}" if test else "未评估"
        test_acc = f"{test['accuracy']:.4f}" if test else "未评估"
        lines.append(
            f"| `{name}`{marker} | {dev['macro_f1']:.4f} | {dev['accuracy']:.4f} | "
            f"{test_f1} | {test_acc} | {cost['model_size_bytes'] / 1024 / 1024:.2f} MB | "
            f"{cost['single_sample_latency_ms']:.2f} ms | "
            f"{cost['batch_32_latency_ms']:.2f} ms | "
            f"{cost['throughput_per_sec']:.0f} 条/秒 |"
        )

    lines.extend(
        [
            "",
            "推理成本通过 `src.inference.load_predictor` 实测（情感模型与类别模型",
            "共用同一个统一入口，适配器是任务无关的），测量范围是",
            "「加载产物 -> 清洗 -> 分词 -> 向量化 -> 预测」的端到端耗时。",
            "",
            "## 4. 超参数调参过程",
            "",
            "| 模型 | 超参数 | 验证集 Macro-F1 | 验证集 Accuracy |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, result in model_results.items():
        for entry in result.get("tuning", ()):  # type: ignore[union-attr]
            parameters = "、".join(f"{k}={v}" for k, v in entry["hyperparameters"].items())
            lines.append(
                f"| `{name}` | {parameters} | {entry['dev_macro_f1']:.4f} | "
                f"{entry['dev_accuracy']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 5. 选定模型的逐情感类别指标",
            "",
            f"模型 `{selected_model}` 在**测试集**上的表现：",
            "",
            f"| 情感 | Precision | Recall | F1 | 支持数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    selected = model_results[selected_model]
    block = selected.get("test") or selected["dev"]
    set_name = "测试集" if selected.get("test") else "验证集"
    for label in label_list:
        values = block["per_class"][label]
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']:,} |"
        )
    lines.extend(
        [
            "",
            f"以上为{set_name}结果。",
            "",
            f"- Accuracy：**{block['accuracy']:.4f}**",
            f"- Macro Precision：**{block['macro_precision']:.4f}**",
            f"- Macro Recall：**{block['macro_recall']:.4f}**",
            f"- Macro-F1：**{block['macro_f1']:.4f}**",
            f"- Weighted F1：**{block['weighted_f1']:.4f}**",
            "",
            "### 混淆矩阵",
            "",
            "行为真实情感，列为预测情感，顺序严格按 `metrics.json` 的 `categories`：",
            "",
            "| 真实 \\ 预测 | " + " | ".join(label_list) + " |",
            "| --- | " + " | ".join("---:" for _ in label_list) + " |",
        ]
    )
    matrix = block["confusion_matrix"]
    for index, label in enumerate(label_list):
        row = " | ".join(str(int(value)) for value in matrix[index])
        lines.append(f"| **{label}** | {row} |")

    if per_product_category:
        lines.extend(
            [
                "",
                "## 6. 各商品类别内的情感表现",
                "",
                "整体指标会把 10 个商品类别混在一起，掩盖类别间的差异，因此按类别拆开。",
                f"「正例率」是该类别中真实为{positive_name}的比例，"
                "用于判断该类别的指标是否被类别先验抬高。",
                "",
                f"| 商品类别 | 样本数 | 正例率 | Accuracy | Macro-F1 | "
                f"{negative_name} F1 | {positive_name} F1 | {positive_name} Precision | "
                f"{positive_name} Recall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        # 按 Macro-F1 升序，让表现最差的类别排在最前面，便于答辩时直接定位问题。
        ordered_categories = sorted(
            per_product_category.items(), key=lambda item: item[1]["macro_f1"]
        )
        for category, stats in ordered_categories:
            lines.append(
                f"| {category} | {stats['support']:,} | {stats['positive_rate']:.2%} | "
                f"{stats['accuracy']:.4f} | {stats['macro_f1']:.4f} | "
                f"{stats['per_class'][negative_name]['f1']:.4f} | "
                f"{stats['per_class'][positive_name]['f1']:.4f} | "
                f"{stats['per_class'][positive_name]['precision']:.4f} | "
                f"{stats['per_class'][positive_name]['recall']:.4f} |"
            )
        weakest = ordered_categories[0] if ordered_categories else None
        strongest = ordered_categories[-1] if ordered_categories else None
        if weakest and strongest:
            lines.extend(
                [
                    "",
                    f"表现最好的是 **{strongest[0]}**（Macro-F1 {strongest[1]['macro_f1']:.4f}），"
                    f"最差的是 **{weakest[0]}**（Macro-F1 {weakest[1]['macro_f1']:.4f}），"
                    f"两者相差 {strongest[1]['macro_f1'] - weakest[1]['macro_f1']:.4f}。",
                    "该差异是实测结果；语义原因的解读见第 8 节与错误分析文档。",
                ]
            )

    if ablation_results:
        lines.extend(
            [
                "",
                "## 7. 特征方案对照（验证集实测）",
                "",
                "为避免凭直觉选择分词方案，先用最快的模型在验证集上实测比较：",
                "",
                "| 方案 | 验证集 Macro-F1 | 验证集 Accuracy | 词表大小 | 说明 |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in ablation_results:
            vocabulary = row.get("vocabulary_size")
            vocabulary_text = f"{vocabulary:,}" if isinstance(vocabulary, int) else "—"
            lines.append(
                f"| {row['name']} | {row['macro_f1']:.4f} | {row.get('accuracy', 0.0):.4f} | "
                f"{vocabulary_text} | {row['note']} |"
            )

    if error_summary:
        lines.extend(
            [
                "",
                "## 8. 典型错误分布",
                "",
                f"在测试集 {error_summary['test_rows']:,} 条评论中共有 "
                f"**{error_summary['error_count']}** 条判错"
                f"（错误率 {error_summary['error_rate']:.2%}）："
                f"真实{positive_name}被判为{negative_name} {error_summary['false_negative']} 条，"
                f"真实{negative_name}被判为{positive_name} {error_summary['false_positive']} 条。",
                "",
                "误分类样本按**可观察的语言现象**（启发式线索词命中）分组如下。",
                "分组只是检索手段，不代表已经证明因果；逐条实例与解读见"
                f"`{error_summary.get('cases_path', 'reports/m5-sentiment-error-cases.json')}` "
                "与 `reports/m5-sentiment-error-analysis.md`。",
                "",
                "| 现象分组 | 命中条数 | 占全部误分类比例 |",
                "| --- | ---: | ---: |",
            ]
        )
        for tag, info in sorted(
            error_summary["by_tag"].items(), key=lambda item: -item[1]["count"]
        ):
            share = info["count"] / error_summary["error_count"] if error_summary["error_count"] else 0.0
            lines.append(f"| {tag} | {info['count']} | {share:.1%} |")

        lines.extend(
            [
                "",
                "代表性误分类样本（按绝对置信度降序取前 5 条，完整清单见 JSON）：",
                "",
                "| 评论（截断） | 真实 | 预测 | 置信度 | 现象分组 |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for case in error_summary.get("top_examples", ()):
            text = str(case["text"]).replace("|", "／").replace("\n", " ")
            if len(text) > 60:
                text = text[:60] + "…"
            lines.append(
                f"| {text} | {case['true']} | {case['predicted']} | "
                f"{case['confidence']:.4f} | {case.get('tag', '—')} |"
            )

    if notes:
        lines.extend(["", "## 9. 结论与已知局限", ""])
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)
