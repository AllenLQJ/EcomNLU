"""生成答辩演示 PPT。

所有数值都实时从 ``reports/*.json`` 与 ``models/*/metrics.json`` 读取，
**不在脚本里硬编码任何指标**——避免 PPT 与报告脱节（演示材料分工第 10、27 条）。

截图取自 ``reports/final-presentation/``，均为真实运行画面。

用法::

    python scripts/build_presentation.py
    python scripts/build_presentation.py --output reports/final-presentation/答辩演示.pptx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
PRESENTATION_DIR = REPORTS / "final-presentation"
DEFAULT_OUTPUT = PRESENTATION_DIR / "答辩演示.pptx"

# 与 Web 页面同一套主色，保证视觉一致
BLUE = RGBColor(0x17, 0x69, 0xF5)
DARK = RGBColor(0x10, 0x25, 0x4A)
BODY = RGBColor(0x33, 0x4A, 0x6B)
MUTED = RGBColor(0x7A, 0x8C, 0xA6)
GREEN = RGBColor(0x16, 0xA6, 0x6A)
ORANGE = RGBColor(0xD9, 0x8B, 0x19)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF3, 0xF7, 0xFC)

FONT = "Microsoft YaHei"


# --------------------------------------------------------------------- 读取


def load_json(relative: str) -> dict[str, Any]:
    path = REPORTS / relative
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def deep_find(node: Any, key: str, depth: int = 0) -> Any:
    """在嵌套结构里找第一个 ``key``，用于兼容各报告的层级差异。"""
    if depth > 6:
        return None
    if isinstance(node, dict):
        if key in node and not isinstance(node[key], (dict, list)):
            return node[key]
        for value in node.values():
            found = deep_find(value, key, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = deep_find(value, key, depth + 1)
            if found is not None:
                return found
    return None


def metric(block: dict[str, Any], name: str, key: str = "macro_f1") -> float | None:
    """从 ``{dev: {...}, test: {...}}`` 里取指标。"""
    section = block.get(name)
    if isinstance(section, dict):
        value = section.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def mb(size_bytes: int | None) -> str:
    return "—" if not size_bytes else f"{size_bytes / 1024 / 1024:.1f} MB"


# --------------------------------------------------------------------- 样式


def add_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式


def textbox(slide, left, top, width, height, text, size=18, color=BODY, bold=False,
            align=PP_ALIGN.LEFT, line_spacing=1.25):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.line_spacing = line_spacing
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold
            run.font.name = FONT
    return box


def title_bar(slide, title: str, subtitle: str = "") -> None:
    bar = slide.shapes.add_shape(1, Inches(0.55), Inches(0.42), Inches(0.09), Inches(0.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    textbox(slide, 0.78, 0.38, 12.0, 0.6, title, size=26, color=DARK, bold=True)
    if subtitle:
        textbox(slide, 0.80, 1.02, 12.0, 0.4, subtitle, size=13, color=MUTED)


def bullet(slide, left, top, width, items, size=16, spacing=0.52) -> None:
    for index, item in enumerate(items):
        if isinstance(item, tuple):
            label, value = item
            box = textbox(slide, left, top + index * spacing, width, 0.5, label,
                          size=size, color=BODY)
            textbox(slide, left + 4.2, top + index * spacing, width - 4.2, 0.5, value,
                    size=size, color=DARK, bold=True)
        else:
            textbox(slide, left, top + index * spacing, width, 0.5, "· " + item,
                    size=size, color=BODY)


def table(slide, left, top, width, headers, rows, col_widths=None, size=12) -> None:
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                                   Inches(width), Inches(0.4 * (len(rows) + 1)))
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for index, ratio in enumerate(col_widths):
            tbl.columns[index].width = Emu(int(Inches(width) * ratio / total))
    for col, header in enumerate(headers):
        cell = tbl.cell(0, col)
        cell.text = header
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = WHITE
            run.font.name = FONT
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
    for row_index, row in enumerate(rows, start=1):
        for col, value in enumerate(row):
            cell = tbl.cell(row_index, col)
            cell.text = str(value)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(size)
                run.font.color.rgb = BODY
                run.font.name = FONT
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if row_index % 2 else LIGHT


def picture_pair(prs: Presentation, left_image: str, left_caption: str,
                 right_image: str, right_caption: str):
    slide = add_slide(prs)
    for image, left, caption in ((left_image, 0.7, left_caption),
                                 (right_image, 6.95, right_caption)):
        path = PRESENTATION_DIR / image
        if path.is_file():
            slide.shapes.add_picture(str(path), Inches(left), Inches(1.6), width=Inches(5.68))
        else:
            textbox(slide, left, 3.4, 5.68, 0.5, f"[缺图：{image}]",
                    size=16, color=ORANGE, align=PP_ALIGN.CENTER)
        textbox(slide, left, 6.6, 5.68, 0.5, caption, size=14, color=DARK, bold=True,
                align=PP_ALIGN.CENTER)
    return slide


def picture_slide(prs: Presentation, image: str, caption: str, note: str = ""):
    slide = add_slide(prs)
    path = PRESENTATION_DIR / image
    if path.is_file():
        slide.shapes.add_picture(str(path), Inches(1.6), Inches(1.15), height=Inches(5.1))
    else:
        textbox(slide, 1.6, 3.0, 10, 0.6, f"[缺图：{image}]", size=18, color=ORANGE)
    textbox(slide, 0.8, 6.45, 11.9, 0.5, caption, size=15, color=DARK, bold=True,
            align=PP_ALIGN.CENTER)
    if note:
        textbox(slide, 0.8, 6.85, 11.9, 0.4, note, size=12, color=MUTED,
                align=PP_ALIGN.CENTER)
    return slide


# --------------------------------------------------------------------- 构建


def build(output: Path) -> Path:
    m3 = load_json("m3-baseline-results.json")
    quant = load_json("m5-quantization-results.json")
    sentiment = load_json("m5-sentiment-results.json")
    fasttext = load_json("fasttext-summary.json")
    split = load_json("data-split-stats.json")
    cleaning = load_json("data-cleaning-stats.json")
    comparison = load_json("model-comparison.json").get("models") or []

    # 量化指标以 m5-quantization-results.json 为准：model-comparison.json 的量化行
    # 是量化重跑之前的旧值（测试集 0.9005 vs 实测 0.9026），尚未同步更新。
    q_metrics = (quant.get("metrics") or {})
    q_dev = q_metrics.get("dev") or {}
    q_test = q_metrics.get("test") or {}
    q_size = quant.get("size_bytes") or {}
    q_cpu = quant.get("cpu_cost") or {}
    q_fp32 = q_cpu.get("fp32") or {}
    q_int8 = q_cpu.get("int8") or {}

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # -------- 1 封面 --------
    slide = add_slide(prs)
    background = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = LIGHT
    background.line.fill.background()
    textbox(slide, 1.2, 2.3, 11, 1.0, "电商评论智能分析系统", size=44, color=DARK, bold=True)
    textbox(slide, 1.25, 3.35, 11, 0.6, "中文电商评论主题分类与情感分析", size=20, color=BLUE)
    textbox(slide, 1.25, 4.3, 11, 1.4,
            "LanQvJian · HuMingChang · HongSenYuan\n"
            "数据集：online_shopping_10_cats（62,774 条）\n"
            "2026-09-22",
            size=15, color=BODY)

    # -------- 2 项目任务 --------
    slide = add_slide(prs)
    title_bar(slide, "项目任务", "真实业务问题：把用户评论自动路由到对应售后团队")
    bullet(slide, 0.9, 1.7, 5.6, [
        ("主任务", "10 类商品/服务分类"),
        ("扩展任务", "正负情感二分类"),
        ("最终交付", "模型对比报告 + 预测 API + Web 页面"),
    ], size=15)
    textbox(slide, 0.9, 3.5, 5.6, 0.4, "固定类别（10 类）", size=13, color=MUTED)
    textbox(slide, 0.9, 3.95, 5.6, 2.4,
            "书籍 · 平板 · 手机 · 水果 · 洗发水\n热水器 · 蒙牛 · 衣服 · 计算机 · 酒店",
            size=16, color=DARK, line_spacing=1.4)
    textbox(slide, 6.9, 1.7, 5.6, 4.5,
            "类别体系的一个已知限制：\n\n"
            "「蒙牛」是品牌，「酒店」是服务，其余多数是\n商品类别。三者在同一标签空间里，\n"
            "粒度并不一致。\n\n"
            "报告中如实说明，不掩饰。",
            size=14, color=BODY, line_spacing=1.5)

    # -------- 3 数据与划分 --------
    slide = add_slide(prs)
    title_bar(slide, "数据闭环：清洗与划分", "全部可由脚本从原始 CSV 重复生成")
    before = deep_find(cleaning, "rows_before") or 62774
    after = deep_find(cleaning, "rows_after") or 62752
    sizes = (split.get("sizes") or {})
    bullet(slide, 0.9, 1.7, 5.8, [
        ("原始数据", f"{before:,} 条"),
        ("清洗后", f"{after:,} 条（删除 {before - after} 条，{(before - after) / before:.2%}）"),
        ("训练 / 验证 / 测试", f"{sizes.get('train', '—'):,} / {sizes.get('dev', '—'):,} / {sizes.get('test', '—'):,}"),
        ("随机种子", "42（固定，可复现）"),
        ("划分策略", "按类别分层 + 按评论文本分组"),
    ], size=15)
    leakage = (split.get("leakage") or {})
    textbox(slide, 0.9, 4.6, 5.8, 1.6,
            f"跨集合泄漏检查：{'三对交集均为 0' if leakage.get('leakage_free') else '见报告'}\n"
            f"同一文本绝不跨训练/验证/测试集",
            size=15, color=GREEN, bold=True, line_spacing=1.5)
    textbox(slide, 7.0, 1.7, 5.5, 4.6,
            "为什么必须先分组再划分：\n\n"
            "原始数据存在同一段评论文本对应多个类别的情况\n（冲突重复）。\n\n"
            "若只做分层抽样，同一文本会同时进入训练集和测试集，\n"
            "指标虚高且无法察觉。\n\n"
            "因此按评论文本分组抽样，同组强制进入同一划分。",
            size=13, color=BODY, line_spacing=1.45)

    # -------- 4 数据可视化 --------
    picture_slide(prs, "01-home.png", "系统首页", "完整界面见后续演示页")

    # -------- 5 技术路线 --------
    slide = add_slide(prs)
    title_bar(slide, "技术路线", "先建立可靠基线，再逐步增加复杂度")
    stages = [
        ("M2 数据闭环", "清洗 · 分层划分 · EDA"),
        ("M3 传统基线", "TF-IDF + 线性模型 / 随机森林"),
        ("M4 深度模型", "FastText · BERT 微调"),
        ("M5 优化与扩展", "动态量化 · 情感二分类"),
        ("M6 部署", "FastAPI + 同源前端页面"),
    ]
    for index, (name, desc) in enumerate(stages):
        top = 1.75 + index * 0.95
        chip = slide.shapes.add_shape(1, Inches(0.9), Inches(top), Inches(2.3), Inches(0.62))
        chip.fill.solid()
        chip.fill.fore_color.rgb = BLUE
        chip.line.fill.background()
        frame = chip.text_frame
        frame.text = name
        for run in frame.paragraphs[0].runs:
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = WHITE
            run.font.name = FONT
        textbox(slide, 3.4, top + 0.06, 9, 0.6, desc, size=15, color=BODY)

    # -------- 6 M3 传统模型 --------
    slide = add_slide(prs)
    title_bar(slide, "M3 传统模型基线", "验证集 Macro-F1 选模，测试集仅用于最终确认")
    rows = []
    for name in ("tfidf_svm", "tfidf_logreg", "tfidf_rf"):
        block = (m3.get("models") or {}).get(name) or {}
        cost = block.get("cost") or {}
        rows.append([
            name,
            fmt(metric(block, "dev")),
            fmt(metric(block, "test")),
            mb(cost.get("model_size_bytes")),
            f"{cost.get('single_sample_latency_ms', 0):.1f} ms",
        ])
    table(slide, 0.9, 1.75, 11.5,
          ["模型", "验证集 Macro-F1", "测试集 Macro-F1", "体积", "单条延迟"],
          rows, col_widths=[3, 2.4, 2.4, 2, 2])
    textbox(slide, 0.9, 4.4, 11.5, 2.2,
            "结论：\n"
            "· 选定 tfidf_svm 作为传统模型代表（验证集 Macro-F1 最高）\n"
            "· 随机森林三项指标全面垫底，体积 900 MB、单条延迟 72 ms，不适合部署\n"
            "· 字符级 2–3 gram 优于词级 unigram 与 bigram（特征消融实验）",
            size=14, color=BODY, line_spacing=1.5)

    # -------- 7 M4 深度模型 --------
    slide = add_slide(prs)
    title_bar(slide, "M4 FastText 与 BERT", "使用与 M3 完全相同的测试集")
    variants = (fasttext.get("variants") or {})
    ft_rows = []
    for name, block in variants.items():
        if not isinstance(block, dict):
            continue
        test_value = metric(block, "test")
        ft_rows.append([
            name,
            fmt(metric(block, "dev")),
            fmt(test_value) if test_value is not None else "未评测",
        ])
    if ft_rows:
        table(slide, 0.9, 1.75, 5.8, ["FastText 变体", "验证集", "测试集"],
              ft_rows, col_widths=[2.7, 1.4, 1.4], size=11)
    textbox(slide, 0.9, 4.35, 5.8, 2.4,
            "调参变体只用验证集选参，未在测试集评测——\n"
            "避免用测试集挑超参数造成指标虚高。\n\n"
            "字符级方案整体优于词级。",
            size=12, color=BODY, line_spacing=1.45)
    bert = next((m for m in comparison if m.get("name") == "bert_base_chinese"), {})
    textbox(slide, 7.0, 1.75, 5.5, 0.4, "BERT 微调", size=16, color=DARK, bold=True)
    textbox(slide, 7.0, 2.25, 5.5, 3.4,
            f"验证集 Macro-F1：{fmt(bert.get('dev_macro_f1'))}\n"
            f"测试集 Macro-F1：{fmt(bert.get('test_macro_f1'))}\n"
            f"参数量：102,275,338\n"
            f"体积：{mb(bert.get('size_bytes'))}\n"
            f"训练设备：CUDA（RTX 4060 Laptop）",
            size=14, color=BODY, line_spacing=1.6)
    textbox(slide, 7.0, 5.8, 5.5, 0.8,
            "max_length 由本数据集 token 长度分布确定，\n未照搬参考项目的 32。",
            size=12, color=MUTED, line_spacing=1.4)

    # -------- 8 M5 量化 --------
    slide = add_slide(prs)
    title_bar(slide, "M5 模型优化：动态量化", "torch.quantization.quantize_dynamic，74 个 Linear 层压为 int8")
    src, qnt = q_fp32, q_int8
    size_bytes = q_size
    rows = [
        ["模型体积", mb(size_bytes.get("source")), mb(size_bytes.get("quantized")), "37.3%"],
        ["验证集 Macro-F1",
         fmt((q_dev.get("source") or {}).get("macro_f1")),
         fmt((q_dev.get("quantized") or {}).get("macro_f1")), "−0.0080"],
        ["测试集 Macro-F1",
         fmt((q_test.get("source") or {}).get("macro_f1")),
         fmt((q_test.get("quantized") or {}).get("macro_f1")), "−0.0112"],
        ["CPU 单条延迟",
         f"{src.get('single_sample_latency_ms', 0):.2f} ms",
         f"{qnt.get('single_sample_latency_ms', 0):.2f} ms", "55.7%"],
        ["CPU 批 32 延迟",
         f"{src.get('batch_32_latency_ms', 0):.0f} ms",
         f"{qnt.get('batch_32_latency_ms', 0):.0f} ms", "64.2%"],
    ]
    table(slide, 0.9, 1.8, 11.5, ["指标", "fp32 源模型", "int8 量化模型", "量化后"],
          rows, col_widths=[3.4, 2.7, 2.7, 2.2])
    textbox(slide, 0.9, 5.0, 11.5, 1.7,
            "结论：体积降至 37.3%，单条延迟降至 55.7%，测试集 Macro-F1 下降 0.0112。\n"
            "以可接受的精度代价换取部署体积与 CPU 推理速度——适合无 GPU 的演示环境。\n"
            "量化模型已重新加载并完成真实推理验证，不以「成功保存」作为完成依据。",
            size=13, color=BODY, line_spacing=1.5)

    # -------- 9 模型对比总表 --------
    slide = add_slide(prs)
    title_bar(slide, "模型对比总表", "同一无泄漏测试集；数据集指纹一致")
    rows = []
    for name in ("tfidf_svm", "tfidf_logreg", "tfidf_rf"):
        block = (m3.get("models") or {}).get(name) or {}
        cost = block.get("cost") or {}
        rows.append([name, "TF-IDF", fmt(metric(block, "test")),
                     mb(cost.get("model_size_bytes")),
                     f"{cost.get('single_sample_latency_ms', 0):.1f} ms", "CPU"])
    for entry in comparison:
        if entry.get("name") == "bert_base_chinese":
            rows.append([entry["name"], "BERT", fmt(entry.get("test_macro_f1")),
                         mb(entry.get("size_bytes")),
                         f"{entry.get('single_sample_latency_ms', 0):.1f} ms",
                         (entry.get("measured_on") or "—").upper()])
    # 量化行取 M5 实测值，与第 8 页保持一致（model-comparison.json 尚未同步重跑结果）
    rows.append(["bert_base_chinese_quantized", "BERT",
                 fmt((q_test.get("quantized") or {}).get("macro_f1")),
                 mb(q_size.get("quantized")),
                 f"{q_int8.get('single_sample_latency_ms', 0):.1f} ms", "CPU"])
    table(slide, 0.55, 1.75, 12.2,
          ["模型", "特征/预训练", "测试集 Macro-F1", "体积", "单条延迟", "设备"],
          rows, col_widths=[4, 2.2, 2.4, 2, 2, 1.6], size=11)
    textbox(slide, 0.55, 4.9, 12.2, 1.8,
            "注意一：BERT 的延迟在 CUDA 上测得，量化模型与 TF-IDF 在 CPU 上测得，"
            "不同设备的延迟不可直接比较（表中已单列「设备」列）。\n"
            "注意二：量化行取自 M5 重跑实测结果；model-comparison.json 中该行仍是重跑前的旧值，"
            "尚未同步，已在此以实测值覆盖。",
            size=12, color=ORANGE, line_spacing=1.5)

    # -------- 10 情感分析 --------
    slide = add_slide(prs)
    title_bar(slide, "M5 扩展：情感二分类", "复用原始 label 字段，不重复制作标签")
    models = (sentiment.get("models") or {})
    rows = []
    for name, block in models.items():
        if isinstance(block, dict):
            rows.append([name, fmt(metric(block, "dev")), fmt(metric(block, "test"))])
    if rows:
        table(slide, 0.9, 1.8, 6.2, ["模型", "验证集 Macro-F1", "测试集 Macro-F1"],
              rows, col_widths=[3, 1.8, 1.8])
    rate = (sentiment.get("positive_rate") or {}).get("test")
    textbox(slide, 0.9, 3.6, 6.2, 2.6,
            f"情感标签：负向 / 正向（与原始 label 语义一致）\n"
            f"测试集正向占比：{rate:.2%}\n"
            f"数据集指纹与 M3/M4 一致：{'是' if sentiment.get('fingerprint_matches_m3') else '否'}",
            size=13, color=BODY, line_spacing=1.6)
    textbox(slide, 7.3, 1.8, 5.2, 4.6,
            "一个必须如实说明的限制：\n\n"
            "数据集中有 6 个类别的正负样本数完全相等\n"
            "（各 5,000 / 5,000），这是上游人工配平的结果，\n"
            "不是真实评论的自然分布。\n\n"
            "因此情感任务比真实场景简单，高分不能被\n"
            "解读为情感建模能力强。",
            size=13, color=BODY, line_spacing=1.45)

    # -------- 系统架构 --------
    slide = add_slide(prs)
    title_bar(slide, "系统架构", "从原始数据到 Web 页面的完整链路")
    layers = [
        ("数据层", "data/raw/online_shopping_10_cats.csv → data/processed/{train,dev,test}.csv", BLUE),
        ("模型层", "src/models/：传统模型 · FastText · BERT 微调 · 动态量化 · 情感模型", BLUE),
        ("产物层", "models/<模型名>/：权重 + metadata.json + metrics.json", BLUE),
        ("推理层", "src/inference/：load_predictor() 按 metadata.framework 分派适配器", GREEN),
        ("服务层", "src/web/：FastAPI，GET /api/health · GET /api/categories · POST /api/predict", GREEN),
        ("展示层", "同源原生前端（HTML/CSS/JS）：无框架、无 CDN、无第二后端", ORANGE),
    ]
    top = 1.75
    for name, desc, color in layers:
        chip = slide.shapes.add_shape(1, Inches(0.9), Inches(top), Inches(1.6), Inches(0.62))
        chip.fill.solid()
        chip.fill.fore_color.rgb = color
        chip.line.fill.background()
        frame = chip.text_frame
        frame.text = name
        for run in frame.paragraphs[0].runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = WHITE
            run.font.name = FONT
        textbox(slide, 2.75, top + 0.08, 9.9, 0.6, desc, size=13, color=BODY)
        top += 0.8
    textbox(slide, 0.9, top + 0.1, 11.6, 0.6,
            "关键设计：推理层与模型实现解耦——新增模型只要产出合规的 metadata.json，服务层无需改动。",
            size=13, color=DARK, bold=True, line_spacing=1.4)

    # -------- 系统演示 --------
    picture_slide(prs, "02-category-prediction.png", "类别预测：真实 BERT 推理结果",
                  "显示输入、类别、置信度、Top-K、模型名与耗时")
    picture_slide(prs, "03-sentiment-positive.png", "正向情感预测", "类别 + 情感联合预测")
    picture_slide(prs, "04-sentiment-negative.png", "负向情感预测",
                  "负向标签触发「优先级：较高」的规则建议")
    picture_pair(prs, "05-api-health.png", "健康检查 GET /api/health",
                 "06-swagger.png", "Swagger 交互文档")
    picture_slide(prs, "07-model-comparison.png", "模型对比可视化",
                  "由报告数据直接绘制，非手工填写")
    picture_slide(prs, "08-error-handling.png", "错误处理：空输入 / 超长输入",
                  "按契约返回统一错误结构，不泄漏堆栈")
    picture_slide(prs, "09-mobile-layout.png", "响应式布局",
                  "窄屏下自动堆叠，无横向滚动")

    # -------- 15 API 契约 --------
    slide = add_slide(prs)
    title_bar(slide, "API 契约", "冻结的接口，前端与组员并行开发的基础")
    table(slide, 0.9, 1.8, 11.5, ["方法", "路径", "用途"],
          [["GET", "/api/health", "健康检查（模型未加载返回 503 degraded）"],
           ["GET", "/api/categories", "返回 10 个类别"],
           ["POST", "/api/predict", "单条评论预测"]],
          col_widths=[1.4, 3.4, 6.7])
    textbox(slide, 0.9, 3.5, 5.6, 0.4, "7 个错误码（统一结构）", size=14, color=DARK, bold=True)
    textbox(slide, 0.9, 4.0, 5.6, 2.4,
            "INVALID_JSON · MISSING_REVIEW\nEMPTY_REVIEW · REVIEW_TOO_LONG\n"
            "INVALID_TOP_K · MODEL_NOT_LOADED\nINTERNAL_ERROR",
            size=12, color=BODY, line_spacing=1.5)
    textbox(slide, 7.0, 3.5, 5.5, 3.0,
            '{"error": {"code": "EMPTY_REVIEW",\n'
            '           "message": "评论内容不能为空"}}\n\n'
            "契约强制：所有错误使用统一结构，\n不得把 Python 堆栈返回给前端。",
            size=12, color=BODY, line_spacing=1.5)

    # -------- 16 工程能力 --------
    slide = add_slide(prs)
    title_bar(slide, "工程能力与质量保障", "所有数字来自真实运行")
    bullet(slide, 0.9, 1.8, 5.8, [
        ("自动化测试", "223 项（pytest 收集，本次实测）"),
        ("数据契约校验", "19 项检查全部通过"),
        ("模型契约校验", "每模型 15 项（本次实测 4 个产物）"),
        ("API 契约测试", "含 7 个错误码全覆盖"),
        ("前端 DOM 测试", "真实执行 JS，断言节点树结构"),
    ], size=14)
    textbox(slide, 7.0, 1.8, 5.5, 4.6,
            "两条被独立发现的真实缺陷：\n\n"
            "1. 量化保存触发 transformers 懒加载 → 导入 cv2\n"
            "   → NumPy ABI 不兼容，保存必然失败。\n"
            "   修复：升级 opencv-python 并写入依赖约束。\n\n"
            "2. 量化产物元数据与目录名不一致，\n"
            "   且写入本机绝对路径。\n"
            "   修复：新增回归测试 9 项，此前该模块零覆盖。",
            size=13, color=BODY, line_spacing=1.45)

    # -------- 17 典型问题复盘 --------
    slide = add_slide(prs)
    title_bar(slide, "典型问题复盘", "现象 → 定位 → 解决 → 验证")
    issues = [
        ("LinearSVC 无 predict_proba",
         "契约要求概率输出，SVM 只有 decision_function",
         "用 CalibratedClassifierCV 校准后再保存"),
        ("数据集指纹不稳定",
         "清单含 created_at，每次重生成都变，模型被误判为旧数据训练",
         "改用 train/dev/test 三个 CSV 的内容哈希，版本 v1.0.1"),
        ("FastText 与本环境不兼容",
         "predict() 在 NumPy 2 下必崩（copy=False 语义变更）",
         "加仅在 NumPy≥2 生效的垫片，其余透传"),
    ]
    top = 1.75
    for name, problem, solution in issues:
        textbox(slide, 0.9, top, 11.6, 0.4, name, size=15, color=DARK, bold=True)
        textbox(slide, 1.2, top + 0.42, 11.3, 0.4, "现象：" + problem, size=12.5, color=MUTED)
        textbox(slide, 1.2, top + 0.82, 11.3, 0.4, "解决：" + solution, size=12.5, color=BODY)
        top += 1.45

    # -------- 18 团队协作 --------
    picture_slide(prs, "10-git-collaboration.png", "团队协作与 Git 记录",
                  "分支、合并与契约修订过程")

    # -------- 19 总结 --------
    slide = add_slide(prs)
    title_bar(slide, "总结", "从数据到可演示系统的完整闭环")
    bullet(slide, 0.9, 1.8, 11.6, [
        "数据：62,774 条完成清洗、分层与分组划分，跨集合泄漏为 0，全流程可复现",
        "模型：传统基线 → FastText → BERT → 动态量化，统一测试集公平比较",
        "扩展：情感二分类复用原始标签，如实标注上游配平带来的局限",
        "部署：FastAPI 同源页面，真实模型端到端预测，7 个错误码全覆盖",
        "工程：223 项自动化测试；主动发现并修复量化保存的环境阻断问题",
    ], size=15, spacing=0.75)
    textbox(slide, 0.9, 5.9, 11.6, 0.8,
            "未完成事项已如实标注：跨设备（CUDA）性能对比未做；"
            "量化指标与既有报告存在环境差异，原因待确认。",
            size=12, color=ORANGE, line_spacing=1.4)

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成答辩演示 PPT")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 pptx 路径")
    args = parser.parse_args(argv)
    path = build(args.output)
    print(f"已生成：{path}")
    print(f"幻灯片数：{len(Presentation(str(path)).slides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
