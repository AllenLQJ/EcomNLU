"""把「答辩演示-简约高级版-v4.pptx」复制为 v5，并按评分标准补齐 5 页。

对照《投满分分类项目实战》的问卷星评分标准（10 个维度），v4 缺少或太薄：
  一、团队风采（分工/会议纪要/讨论照片/git）—— 补「02 团队风采」
  二、数据处理（缺失/异常/重复）—— 补「03 数据处理」明细页
  三、探索性数据分析（直方图/箱线图等）—— 补「04 探索性数据分析」
  四、特征工程（转换/编码/提取）—— 补「05 特征工程」
  九、项目创新以及未来展望 —— 补「21 创新与展望」

所有内容取自仓库文档：
  docs/team.md、docs/development-plan.md、reports/meeting-notes/、
  reports/daily-reports/、reports/data-quality/raw_quality_report.md、
  reports/data-cleaning-report.md、reports/m3-baseline-results.json、
  reports/eda-report.md 与 reports/figures/ 的成品图。

风格完全对齐 v4：13.333×7.5、Microsoft YaHei、
主色 2E7D6B / 深色 13233A / 蓝 497DAA / 橙 D88A4B / 灰 6F7B88 / 浅底 E8EEEA。

用法::

    python scripts/extend_presentation_v5.py

输出：reports/final-presentation/答辩演示-简约高级版-v5.pptx
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Inches, Pt

try:  # 虚线边框（讨论照片占位框）
    from pptx.enum.line import MSO_LINE_DASH_STYLE
except ImportError:  # 旧版 python-pptx 没有该枚举时退回实线灰框
    MSO_LINE_DASH_STYLE = None

PROJECT = Path(__file__).resolve().parents[1]
PRES_DIR = PROJECT / "reports" / "final-presentation"
FIG_DIR = PROJECT / "reports" / "figures"
SRC = PRES_DIR / "答辩演示-简约高级版-v4.pptx"
DST = PRES_DIR / "答辩演示-简约高级版-v5.pptx"

# v4 配色与版式常量
TEAL = RGBColor(0x2E, 0x7D, 0x6B)
DARK = RGBColor(0x13, 0x23, 0x3A)
BLUE = RGBColor(0x49, 0x7D, 0xAA)
ORANGE = RGBColor(0xD8, 0x8A, 0x4B)
MUTED = RGBColor(0x6F, 0x7B, 0x88)
DIV = RGBColor(0xD8, 0xDE, 0xD9)
LIGHT = RGBColor(0xE8, 0xEE, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

LABEL_RE = re.compile(r"^\d{2}\s+\S")

# 旧章节号 → 新章节号（插入 4 个新章节后顺延）
LABEL_MAP = {
    "02 数据处理": "03 数据处理",
    "03 系统展示": "06 系统展示",
    "04 方法": "07 方法",
    "05 模型结果": "08 模型结果",
    "06 优化": "09 优化",
    "07 最终选择": "10 最终选择",
    "08 扩展任务": "11 扩展任务",
    "09 工程实现": "12 工程实现",
    "10 端到端演示": "13 端到端演示",
    "11 API 演示": "14 API 演示",
    "12 结果证据": "15 结果证据",
    "13 稳定性": "16 稳定性",
    "14 多端展示": "17 多端展示",
    "15 协作基础": "18 协作基础",
    "16 验证": "19 验证",
    "17 问题解决": "20 问题解决",
}


# --------------------------------------------------------------------- 基础图形


def rect(slide, x, y, w, h, fill=None, line=None, rounded=False, dash=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.2)
        if dash is not None and MSO_LINE_DASH_STYLE is not None:
            try:
                shape.line.dash_style = dash
            except (TypeError, ValueError):
                pass
    return shape


def text(slide, x, y, w, h, lines, size=12.5, color=DARK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15, name=None):
    """lines: str 或 [(text, size, color, bold), ...] 多行。"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(lines if isinstance(lines, list) else lines.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if isinstance(line, tuple):
            parts, spec = [line[0]], (line[1], line[2], line[3])
        else:
            parts, spec = [line], (size, color, bold)
        for part in parts:
            run = p.add_run()
            run.text = part
            run.font.name = FONT
            run.font.size = Pt(spec[0])
            run.font.color.rgb = spec[1]
            run.font.bold = spec[2]
    return box


def chrome(slide, label, title, subtitle=None):
    """v4 内容页公共骨架：左竖条 + 章节标签 + 标题 + 分隔线 + 页脚 + 页码。"""
    rect(slide, 0, 0, 0.19, 7.5, fill=TEAL)
    text(slide, 0.77, 0.44, 2.6, 0.25, label, size=9.75, color=TEAL, bold=True)
    text(slide, 0.75, 0.75, 11.56, 0.6, title, size=25.5, color=DARK, bold=True)
    rect(slide, 0.75, 1.45, 11.67, 0.02, fill=DIV)
    if subtitle:
        text(slide, 0.77, 1.72, 11.0, 0.5, subtitle, size=15, color=MUTED)
    text(slide, 0.77, 7.10, 4.38, 0.21, "EcomNLU · 智购感知", size=8.25, color=MUTED)
    page = text(slide, 11.77, 7.08, 0.65, 0.23, "", size=9, color=TEAL, bold=True)
    page.name = "页码"
    return page


def stat(slide, x, y, w, number, label, color=TEAL, number_size=22.5):
    text(slide, x, y, w, 0.52, number, size=number_size, color=color, bold=True)
    text(slide, x, y + 0.5, w, 0.31, label, size=11.25, color=MUTED)


def picture_fit(slide, path, x_zone, zone_w, y, height):
    """按高度等比放图并在区域内居中。"""
    im = Image.open(str(path))
    w = height * im.size[0] / im.size[1]
    x = x_zone + (zone_w - w) / 2
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(height))
    return x, w


def chip(slide, x, y, num, color=TEAL, size=0.42):
    c = rect(slide, x, y, size, size, fill=color, rounded=True)
    tf = c.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = str(num)
    run.font.name = FONT
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = WHITE
    return c


def new_table(slide, x, y, w, headers, rows, col_widths):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   Inches(x), Inches(y), Inches(w),
                                   Inches(0.5 * (len(rows) + 1)))
    tbl = shape.table
    total = sum(col_widths)
    for i, r in enumerate(col_widths):
        tbl.columns[i].width = Emu(int(Inches(w) * r / total))
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = h
        run.font.name = FONT
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = WHITE
    for ri, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = tbl.cell(ri, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = Pt(14.25)
            run.font.color.rgb = DARK
    return shape


# --------------------------------------------------------------------- 新页面


def slide_team(prs, layout):
    s = prs.slides.add_slide(layout)
    chrome(s, "02 团队风采", "团队风采", "三人并行 · 契约对齐 · 全程留痕")
    members = [
        ("LanQvJian", "组长 · 统筹与集成", TEAL,
         ["研发计划与任务协调", "API / Web 部署", "集成测试与答辩组织"]),
        ("HuMingChang", "数据 · 传统模型 · 情感", BLUE,
         ["数据质量、清洗与 EDA", "TF-IDF 传统模型", "情感二分类基线"]),
        ("HongSenYuan", "深度学习与优化", ORANGE,
         ["FastText · BERT 微调", "动态量化与性能评估", "演示录屏"]),
    ]
    for i, (name, role, color, duties) in enumerate(members):
        x = 0.85 + i * 3.68
        rect(s, x, 2.18, 3.45, 2.42, fill=LIGHT, rounded=True)
        text(s, x + 0.25, 2.4, 3.0, 0.4, name, size=18, color=color, bold=True)
        text(s, x + 0.25, 2.82, 3.0, 0.3, role, size=11.5, color=MUTED)
        text(s, x + 0.25, 3.28, 3.0, 1.2, "\n".join("· " + d for d in duties),
             size=11.5, color=DARK, spacing=1.3)
    rect(s, 0.85, 4.95, 3.45, 1.62, fill=WHITE, rounded=True)
    stat(s, 1.1, 5.15, 3.0, "8 篇", "会议纪要（09-19 启动 → 09-21 契约 v1.2）", color=BLUE)
    text(s, 1.1, 5.95, 3.0, 0.5, "指纹修订、提案、表决全程留痕\nreports/meeting-notes/",
         size=10.5, color=MUTED, spacing=1.25)
    rect(s, 4.53, 4.95, 3.45, 1.62, fill=WHITE, rounded=True)
    stat(s, 4.78, 5.15, 3.0, "每日", "晨会 + 邮件日报", color=ORANGE)
    text(s, 4.78, 5.95, 3.0, 0.5, "日报归档 reports/daily-reports/（2 篇）\n研发计划 docs/development-plan.md",
         size=10.5, color=MUTED, spacing=1.25)
    rect(s, 8.21, 4.95, 3.45, 1.62, fill=WHITE, line=MUTED, rounded=True,
         dash=MSO_LINE_DASH_STYLE.DASH if MSO_LINE_DASH_STYLE else None)
    text(s, 8.21, 5.45, 3.45, 0.4, "三人讨论照片", size=13, color=MUTED, bold=True,
         align=PP_ALIGN.CENTER)
    text(s, 8.21, 5.85, 3.45, 0.4, "（补拍后替换此处）\nGit 分支协作见第 27 页",
         size=10.5, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.25)
    return s


def slide_data_detail(prs, layout):
    s = prs.slides.add_slide(layout)
    chrome(s, "03 数据处理", "数据质量与清洗明细",
           "缺失、异常、重复、不均衡——每一条处理都有规则依据")
    items = [
        ("缺失", TEAL, "仅 1 条空白评论；cat / label 字段 0 缺失"),
        ("异常", ORANGE, "清洗 7 类操作：控制字符、Unicode 格式字符、纯符号/数字样本、HTML 实体还原等"),
        ("重复", BLUE, "完全重复行 0；冲突重复 45 组 94 行（同文本多标签，先定规则再处理，不能直接去重）"),
        ("不均衡", DARK, "热水器 575 条（0.92%）vs 平板 10,000 条（15.93%）→ 分层划分 + Macro-F1 选模应对"),
    ]
    rect(s, 0.85, 1.9, 6.0, 4.7, fill=WHITE, rounded=True)
    for i, (head, color, body) in enumerate(items):
        y = 2.12 + i * 1.1
        text(s, 1.1, y, 1.15, 0.35, head, size=15, color=color, bold=True)
        text(s, 2.35, y - 0.03, 4.35, 0.95, body, size=11.5, color=DARK, spacing=1.25)
    picture_fit(s, FIG_DIR / "cleaning_funnel.png", 7.05, 5.4, 1.98, 2.75)
    text(s, 7.05, 4.95, 5.4, 0.35, "清洗漏斗：62,774 → 62,752（删除 22 条，0.035%）",
         size=12, color=DARK, bold=True, align=PP_ALIGN.CENTER)
    text(s, 7.05, 5.4, 5.4, 0.9,
         "全部过程可复现（种子 42）\n"
         "报告：reports/data-quality/raw_quality_report.md\n"
         "      reports/data-cleaning-report.md · data-cleaning-stats.json",
         size=10.5, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.3)
    text(s, 0.85, 6.62, 11.6, 0.35,
         "结论：删除比例仅 0.035%，数据基本干净；真正的风险是类别不均衡与冲突重复，均已写入报告。",
         size=12, color=DARK, bold=True)
    return s


def slide_eda(prs, layout):
    s = prs.slides.add_slide(layout)
    chrome(s, "04 探索性数据分析", "探索性数据分析",
           "直方图 · 箱线图 · 分布堆叠——结论全部出自 reports/eda-report.md")
    figures = [
        ("category_distribution.png", "类别分布：5 类各 1 万条，热水器仅 575 → 严重不均衡"),
        ("category_sentiment_stacked.png", "类别×情感：6 类正负各 5,000（上游人工配平）"),
        ("review_length_histogram.png", "文本长度直方图：短评为主 → max_length 推荐 128，备选 256，否定 32"),
        ("review_length_boxplot.png", "长度箱线图：极长评论为离群点，截断策略有依据"),
    ]
    for i, (fname, caption) in enumerate(figures):
        col, row = i % 2, i // 2
        x_zone = 0.95 if col == 0 else 6.75
        y_img = 1.95 if row == 0 else 4.35
        picture_fit(s, FIG_DIR / fname, x_zone, 5.6, y_img, 1.9)
        text(s, x_zone, y_img + 1.95, 5.6, 0.4, caption, size=11, color=MUTED,
             align=PP_ALIGN.CENTER)
    return s


def slide_features(prs, layout):
    s = prs.slides.add_slide(layout)
    chrome(s, "05 特征工程", "特征工程：TF-IDF 字符级方案",
           "特征转换、编码与提取——实测选型，不靠直觉")
    rect(s, 0.85, 1.95, 4.75, 4.55, fill=LIGHT, rounded=True)
    text(s, 1.1, 2.15, 4.3, 0.4, "字符级 2–3 gram（实测选定）", size=16.5, color=TEAL, bold=True)
    text(s, 1.1, 2.68, 4.3, 1.5,
         "词表上限 400,000\nmin_df = 2（过滤罕见字符串）\n子线性 TF 缩放 + 停用词过滤\n最小词长 2；jieba 分词仅用于词级对照",
         size=11.5, color=DARK, spacing=1.35)
    text(s, 1.1, 4.35, 4.3, 1.0,
         "向量器只在训练集 fit——防止特征泄漏；推理复用同一预处理（src/features/text_features.py）",
         size=11, color=BLUE, bold=True, spacing=1.3)
    text(s, 1.1, 5.5, 4.3, 0.8,
         "标签编码：cat 中文类别名直接作标签，label2id 随模型产物保存（契约 §2）",
         size=11, color=ORANGE, spacing=1.3)
    new_table(s, 5.85, 1.95, 6.7,
              ["特征方案（消融）", "验证集 Macro-F1"],
              [["词级 unigram（jieba 分词）", "0.8673"],
               ["词级 unigram + bigram", "0.8595"],
               ["字符级 2–3 gram  ✓ 选定", "0.8852"]],
              col_widths=[4.3, 2.4])
    text(s, 5.85, 4.05, 6.7, 0.6,
         "字符级不依赖分词，对错别字、口语化表达更稳健。",
         size=11.5, color=MUTED)
    text(s, 0.85, 6.55, 11.6, 0.4,
         "超参调优：LogReg C=1→10（0.8345→0.8721）· SVM C=0.5 最优（0.8923）· RF 200 棵树（0.8165，900 MB 淘汰）",
         size=12, color=DARK, bold=True)
    return s


def slide_future(prs, layout):
    s = prs.slides.add_slide(layout)
    chrome(s, "21 创新与展望", "项目创新与未来展望", "如实说明当前边界，同时给出下一步方向")
    rect(s, 0.85, 1.95, 5.75, 4.85, fill=WHITE, rounded=True)
    text(s, 1.1, 2.1, 5.0, 0.4, "项目创新", size=18, color=TEAL, bold=True)
    innovations = [
        "数据集指纹：三份划分 CSV 的内容哈希，模型与训练数据强绑定（契约 v1.0.1）",
        "契约驱动并行：接口冻结 + 推理层按 metadata 分派，新模型零改动接入",
        "INT8 离线方案：无 GPU 环境完整演示，体积降至 37.3%",
        "情感联动业务：负向评论自动提升售后处理优先级",
    ]
    for i, item in enumerate(innovations):
        y = 2.65 + i * 0.98
        chip(s, 1.1, y, i + 1)
        text(s, 1.68, y - 0.05, 4.75, 0.9, item, size=11.5, color=DARK, spacing=1.2)
    rect(s, 6.85, 1.95, 5.75, 4.85, fill=WHITE, rounded=True)
    text(s, 7.1, 2.1, 5.0, 0.4, "未来展望", size=18, color=BLUE, bold=True)
    futures = [
        "大模型对比：LLM 分类与蒸馏实验，与现方案形成基线对照",
        "数据不均衡：重采样 / class weight（热水器仅 575 条）",
        "多模态扩展：评论图片、视频内容的联合建模",
        "工程化落地：Docker 容器、CI/CD 自动验收与云端部署",
    ]
    for i, item in enumerate(futures):
        y = 2.65 + i * 0.98
        chip(s, 7.1, y, i + 1, color=BLUE)
        text(s, 7.68, y - 0.05, 4.75, 0.9, item, size=11.5, color=DARK, spacing=1.2)
    return s


# --------------------------------------------------------------------- 组装


def renumber(prs):
    """按最终顺序统一刷新章节标签与右下角页码。"""
    for pos, slide in enumerate(prs.slides, start=1):
        page_num = f"{pos:02d}"
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            tf = shape.text_frame
            joined = tf.text.strip()
            is_label = (bool(joined) and LABEL_RE.match(joined)
                        and shape.top is not None and shape.top < Inches(0.6))
            is_page = (shape.left is not None and shape.top is not None
                       and shape.left > Inches(11.5) and shape.top > Inches(6.9)
                       and (joined == "" or joined.isdigit()))
            if is_label:
                for p in tf.paragraphs:
                    if LABEL_RE.match(p.text):
                        new_label = LABEL_MAP.get(p.text.strip(), p.text.strip())
                        for ri, run in enumerate(p.runs):
                            run.text = new_label if ri == 0 else ""
            elif is_page:
                p = tf.paragraphs[0]
                if p.runs:
                    for ri, run in enumerate(p.runs):
                        run.text = page_num if ri == 0 else ""
                else:
                    run = p.add_run()
                    run.text = page_num
                    run.font.name = FONT
                    run.font.size = Pt(9)
                    run.font.bold = True
                    run.font.color.rgb = TEAL


def main() -> int:
    if not SRC.is_file():
        raise SystemExit(f"找不到源文件：{SRC}")
    shutil.copy2(SRC, DST)

    prs = Presentation(str(DST))
    layout = prs.slides[1].slide_layout  # 与内容页同版式

    # 新页先加到末尾，随后按顺序重排
    team = slide_team(prs, layout)
    detail = slide_data_detail(prs, layout)
    eda = slide_eda(prs, layout)
    features = slide_features(prs, layout)
    future = slide_future(prs, layout)

    xml_slides = prs.slides._sldIdLst
    slides = list(xml_slides)
    for sld in slides:
        xml_slides.remove(sld)
    # 原 23 页（索引 0–22）＋ 新 5 页（索引 23–27，创建顺序）
    order = [0, 1, 23, 2, 24, 25, 26, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 21, 22]
    for i in order:
        xml_slides.append(slides[i])

    renumber(prs)
    prs.save(str(DST))

    # 校验
    check = Presentation(str(DST))
    print(f"已生成：{DST}")
    print(f"幻灯片数：{len(check.slides)}（应为 28）")
    labels = []
    for i, slide in enumerate(check.slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame and LABEL_RE.match(shape.text_frame.text.strip()) \
                    and shape.top is not None and shape.top < Inches(0.6):
                labels.append((i, shape.text_frame.text.strip()))
    for i, label in labels:
        print(f"  第 {i:>2} 页  {label}")
    pictures = sum(1 for s in check.slides for sh in s.shapes if sh.shape_type == 13)
    print(f"图片数：{pictures}（应为 15：10 张系统截图 + 4 张 EDA + 1 张漏斗）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
