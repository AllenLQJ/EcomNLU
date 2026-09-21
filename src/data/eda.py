"""电商评论分类项目的探索性数据分析（EDA）图表生成模块。

本模块负责把清洗后的中文电商评论数据画成 EDA 图表，并为每张图表写出一段可直接
放进报告的中文结论文字，最终由 ``scripts/run_eda.py`` 组装成完整的 EDA 报告。

主要输入与输出：
    输入：清洗后的评论数据表（DataFrame）或单列评论文本，以及数据质量检查阶段
        产出的清洗步骤统计（原始行数、每一步删除与修正的行数等）。
    输出：每张图保存到 ``reports/figures/``（具体文件名由调用方通过
        ``output_path`` 传入）；每个绘图函数除写文件外还会**返回**一段中文结论
        字符串，由 :func:`render_eda_report` 拼接进报告正文，这样报告里的数字与
        图表始终来自同一次计算，不会出现「图更新了、报告没同步」的脱节问题。
        其中 :func:`plot_length_cdf` 额外返回覆盖率明细表，供报告渲染 Markdown 表格。

关键约束：
    只依赖 matplotlib（不引入 seaborn），以减少答辩与部署环境的依赖负担；
    图表标题、坐标轴名称、图例文字全部为中文，因此必须先通过
    :func:`setup_chinese_font` 配置一个可用的 CJK 字体，
    否则图中的中文会渲染成方框（豆腐块），图表将不可读。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

# 下面这组导入必须放在 matplotlib.use("Agg") 之后，否则后端切换不会生效，
# 因此用 noqa: E402 抑制「模块级导入未放在文件顶部」的检查告警。
import jieba  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 复用统一的文本判定标准，保证「什么算有效文本」在整条流水线上是同一套判断。
# 停用词表同样放在 text_patterns 中，供 EDA 与 TF-IDF 特征提取共用，避免口径分叉。
from src.data.text_patterns import (  # noqa: E402
    CHINESE_STOPWORDS,
    CJK_PATTERN,
    LATIN_LETTER_PATTERN,
)


CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DengXian",
    "KaiTi",
)
MAX_LENGTH_CANDIDATES: tuple[int, ...] = (32, 64, 96, 128, 160, 192, 256, 384, 512)
FIGURE_DPI = 150
NEGATIVE_COLOR = "#c0392b"
POSITIVE_COLOR = "#27ae60"

# 词云与高频词统计的相关参数。
WORDCLOUD_MIN_LENGTH = 2  # 分词后只保留长度不小于该值的词，过滤单字噪声
WORDCLOUD_TOP_N = 200  # 词云最多展示的词数
CATEGORY_TOP_N = 15  # 每个类别展示的高频词条数
WORDCLOUD_WIDTH = 1600
WORDCLOUD_HEIGHT = 900
WORDCLOUD_SEED = 42  # 词云布局的随机种子，固定后每次生成同一张图

_FONT_NAME: str | None = None


def setup_chinese_font() -> str | None:
    """配置可用的中文字体（CJK），避免图里的中文显示成方框。

    按 CJK_FONT_CANDIDATES 的顺序在系统已安装字体中挑选第一个可用项，并写入
    matplotlib 的全局 rcParams；同时把负号设置为 Unicode 正常显示，
    避免坐标轴负号变成方框。

    Returns:
        str | None: 实际生效的字体名称；若系统里没有任何候选 CJK 字体则返回
        ``None``（此时仍然会出图，但图中的中文不可读，属于需要人工确认的降级状态）。
    """
    global _FONT_NAME
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            _FONT_NAME = candidate
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_NAME = None
    return None


def _save(figure: plt.Figure, output_path: Path) -> None:
    """把图形写入磁盘并释放画布，同时保证输出目录已经存在。

    Args:
        figure: 待保存的 matplotlib 图形对象。
        output_path: 图片输出路径；其父目录不存在时会自动创建。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def cjk_font_file() -> str | None:
    """把中文字体族名称解析成真实的字体文件路径。

    wordcloud 与 Pillow 都需要直接读取字体文件，而不是 matplotlib 的字体族名称，
    因此这里借助 matplotlib 的字体管理器完成解析。

    优先返回 ``.ttf`` 单字体文件：``.ttc`` 是字体集合（TrueType Collection），
    部分图像库对其支持不完整，``.ttf`` 的兼容性最好。

    Returns:
        str | None: 找到的中文字体文件绝对路径；系统里没有中文字体时返回 None。
    """
    resolved_files: list[str] = []
    for candidate in CJK_FONT_CANDIDATES:
        try:
            resolved = fm.findfont(
                fm.FontProperties(family=candidate), fallback_to_default=False
            )
        except Exception:
            # findfont 在找不到字体且禁止回退时会抛异常，此处按「没找到」处理，
            # 继续尝试下一个候选字体。
            continue
        if resolved and Path(resolved).is_file() and resolved not in resolved_files:
            resolved_files.append(resolved)

    # 先返回 .ttf，找不到再退回到 .ttc
    for path in resolved_files:
        if Path(path).suffix.lower() == ".ttf":
            return path
    return resolved_files[0] if resolved_files else None


def is_meaningful_token(token: str, min_length: int = WORDCLOUD_MIN_LENGTH) -> bool:
    """判断一个分词结果是否值得计入词频统计。

    过滤三类噪声：过短的词（默认长度小于 2，主要是「的 / 了 / 是」等单字虚词）、
    停用词表中的虚词、以及既不含中文也不含拉丁字母的纯标点或纯数字串。

    Args:
        token: jieba 切分出的单个词。
        min_length: 允许保留的最小词长（按字符数计）。

    Returns:
        bool: 该词是否应当保留。注意这里用 Python 的 ``re`` 直接匹配字符串，
        而不是 pandas 的 ``.str`` 方法，因为 pandas 3.x 会把正则交给 RE2 执行，
        而 RE2 的字符类行为与 Python 原生正则并不完全一致。
    """
    if len(token) < min_length:
        return False
    if token in CHINESE_STOPWORDS:
        return False
    if not (CJK_PATTERN.search(token) or LATIN_LETTER_PATTERN.search(token)):
        return False
    return True


def collect_word_frequencies(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    category_column: str,
    min_length: int = WORDCLOUD_MIN_LENGTH,
    top_n: int = WORDCLOUD_TOP_N,
    category_top_n: int = CATEGORY_TOP_N,
) -> tuple[list[tuple[str, int]], dict[str, list[tuple[str, int]]], int]:
    """用 jieba 对评论文本分词，一次性统计整体与各类别的高频词。

    只遍历一遍数据，同时累加「整体词频」和「按类别分组的词频」，避免重复分词
    （对 5 万条评论分词约需 40 秒，重复一遍的成本不可接受）。

    Args:
        dataframe: 待统计的评论数据表，通常传入训练集，避免统计口径与特征工程不一致。
        text_column: 评论文本列名（本项目为 ``review``）。
        category_column: 类别列名（本项目为 ``cat``）。
        min_length: 保留的最小词长。
        top_n: 整体词频返回的词数上限。
        category_top_n: 每个类别返回的词数上限。

    Returns:
        tuple: ``(整体高频词列表, 各类别高频词字典, 保留词次总数)``。
        列表元素为 ``(词, 出现次数)`` 元组，按出现次数从高到低排列。
    """
    overall: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = {}
    kept_token_total = 0

    texts = dataframe[text_column].astype("string").fillna("")
    categories = dataframe[category_column].astype("string").fillna("")
    for text, category in zip(texts, categories):
        bucket = per_category.setdefault(str(category), Counter())
        for token in jieba.cut(str(text)):
            if not is_meaningful_token(token, min_length):
                continue
            overall[token] += 1
            bucket[token] += 1
            kept_token_total += 1

    overall_top = overall.most_common(top_n)
    category_top = {
        category: bucket.most_common(category_top_n)
        for category, bucket in per_category.items()
    }
    return overall_top, category_top, kept_token_total


def plot_wordcloud(
    frequencies: Sequence[tuple[str, int]],
    *,
    output_path: Path,
    title: str = "训练集评论分词词云（已过滤停用词）",
) -> str:
    """用 wordcloud 库渲染分词后的词云图。

    本函数刻意**不经过 matplotlib 画布**：wordcloud 生成的本身就是一张完整的位图，
    只需用 Pillow 在其上下叠加标题与读图说明即可落盘，少一层画布转换。
    本项目其余统计图仍统一使用 matplotlib。

    wordcloud 与 Pillow 均采用延迟导入：这样运行环境即使缺少这两个库，
    EDA 的其余图表仍能正常生成，只有调用本函数时才明确报错，不会连累整个流程。

    Args:
        frequencies: ``(词, 出现次数)`` 序列，通常来自 :func:`collect_word_frequencies`。
        output_path: 图片输出路径，通常位于 ``reports/figures/`` 下。
        title: 图表标题。

    Returns:
        str: 中文结论文字，列出出现次数最高的若干词及其次数，并说明词云图的读图方式。

    Raises:
        ImportError: 运行环境未安装 wordcloud 或 Pillow 时抛出。
        RuntimeError: 系统缺少中文字体文件，无法渲染中文词云时抛出。
        ValueError: 传入的词频序列为空时抛出。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        from wordcloud import WordCloud
    except ImportError as exc:  # pragma: no cover - 取决于运行环境是否装了这些库
        raise ImportError(
            "生成词云需要 wordcloud 与 Pillow 库，请先安装："
            "conda install -n AI -c conda-forge wordcloud"
        ) from exc

    if not frequencies:
        raise ValueError("词频序列为空，无法生成词云；请检查分词与停用词过滤结果")

    font_file = cjk_font_file()
    if font_file is None:
        raise RuntimeError(
            "未找到中文字体文件，wordcloud 无法渲染中文，请先安装任一中文字体"
        )

    cloud = WordCloud(
        font_path=font_file,
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color="white",
        max_words=len(frequencies),
        colormap="tab10",
        prefer_horizontal=0.9,
        relative_scaling=0.5,
        # 固定随机种子，保证每次运行生成完全相同的图，满足可复现要求
        random_state=WORDCLOUD_SEED,
    ).generate_from_frequencies(dict(frequencies))

    image = Image.fromarray(cloud.to_array())

    # 词云本身没有坐标轴与图例，因此用 Pillow 在图像上下各留一条留白带：
    # 上方写标题，下方写读图说明，起到与其他图表「标题 + 图例」等价的作用。
    title_band = 95
    note_band = 55
    canvas = Image.new(
        "RGB", (image.width, image.height + title_band + note_band), "white"
    )
    canvas.paste(image, (0, title_band))

    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(font_file, 34)
    note_font = ImageFont.truetype(font_file, 20)
    draw.text(
        (image.width // 2, title_band // 2),
        title,
        font=title_font,
        fill="#222222",
        anchor="mm",
    )
    draw.text(
        (image.width // 2, image.height + title_band + note_band // 2),
        f"词的大小表示词频高低（jieba 分词，已过滤 {len(CHINESE_STOPWORDS)} 个停用词）；"
        "颜色仅用于区分，不含额外含义",
        font=note_font,
        fill="#555555",
        anchor="mm",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="png")

    top_words = "、".join(f"{word}（{count:,} 次）" for word, count in frequencies[:10])
    return (
        f"对训练集评论做 jieba 精确模式分词并过滤停用词后，出现次数最高的词依次为：{top_words}。"
        f"共统计到 {len(frequencies):,} 个有效词进入词云。"
        "词云只能反映词频高低，无法体现词的分布差异，"
        "因此它只作为语料特征的直观展示，不参与任何特征选择或模型决策；"
        "定量的高频词统计见下一节的各类别高频词对比图。"
    )


def plot_top_words_by_category(
    category_frequencies: Mapping[str, Sequence[tuple[str, int]]],
    *,
    output_path: Path,
) -> str:
    """绘制各类别高频词的横向柱状图。

    词云擅长直观展示，但不便于量化比较。这张图用 2×5 的子图逐一列出 10 个类别的
    Top-N 高频词及其出现次数，作为严谨统计证据与词云互补。

    Args:
        category_frequencies: ``{类别: [(词, 次数), ...]}`` 映射。
        output_path: 图片输出路径，通常位于 ``reports/figures/`` 下。

    Returns:
        str: 中文结论文字，指出各类别最具区分度的高频词及其出现次数。
    """
    if not category_frequencies:
        raise ValueError("类别词频为空，无法绘制各类别高频词图")

    # 按整体词频从高到低排列类别，让样本量大的类别排在前面，便于对照上一节的分布图
    ordered = sorted(
        category_frequencies.items(),
        key=lambda item: sum(count for _, count in item[1]),
        reverse=True,
    )
    columns = 5
    rows = int(np.ceil(len(ordered) / columns))

    figure, axes_grid = plt.subplots(rows, columns, figsize=(20, 3.1 * rows))
    axes_list = np.atleast_1d(axes_grid).ravel()

    for axes, (category, words) in zip(axes_list, ordered):
        # 横向柱状图需要把词频最高的词放在最上方，因此反转顺序后绘制
        words = list(reversed(words))
        positions = np.arange(len(words))
        axes.barh(positions, [count for _, count in words], color="#4c72b0")
        axes.set_yticks(positions)
        axes.set_yticklabels([word for word, _ in words], fontsize=8)
        axes.set_title(f"{category}", fontsize=11)
        axes.set_xlabel("出现次数", fontsize=8)
        axes.grid(axis="x", linestyle="--", alpha=0.4)
        axes.set_axisbelow(True)
        axes.tick_params(axis="x", labelsize=7)
        for position, (_, count) in zip(positions, words):
            axes.annotate(
                f"{count:,}",
                xy=(count, position),
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )

    # 子图数量可能多于类别数，多出来的坐标轴要隐藏，否则会出现空白格
    for axes in axes_list[len(ordered) :]:
        axes.axis("off")

    figure.suptitle(
        f"各类别 Top-{CATEGORY_TOP_N} 高频词对比（jieba 分词，已过滤停用词）",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save(figure, output_path)

    # 统计每个词出现在多少个类别的 Top-N 中。这是衡量「跨类别通用程度」的量化指标：
    # 覆盖的类别越多，说明该词越通用、对类别判定越没有区分能力。用它替代
    # 「第一名与第二名的词频倍数」这类指标——后者只能说明词频衰减快慢，
    # 无法说明该词是否真的集中在这一个类别里。
    coverage: dict[str, int] = {}
    for words in category_frequencies.values():
        for word, _ in words:
            coverage[word] = coverage.get(word, 0) + 1

    total_categories = len(category_frequencies)
    generic_threshold = max(2, total_categories // 2)
    generic_words = sorted(
        (word for word, count in coverage.items() if count >= generic_threshold),
        key=lambda word: (-coverage[word], word),
    )[:5]

    # 只出现在单一类别 Top-N 里的词，是最具类别指示性的词
    exclusive_words: list[tuple[str, str]] = [
        (category, words[0][0])
        for category, words in ordered
        if words and coverage.get(words[0][0], 0) == 1
    ]

    generic_text = (
        "、".join(f"「{word}」（{coverage[word]}/{total_categories} 个类别的 Top-N）" for word in generic_words)
        or "无"
    )
    exclusive_text = "、".join(f"{category}→「{word}」" for category, word in exclusive_words) or "无"

    return (
        f"各类别高频词分布如下：{exclusive_text} 等词只出现在单一类别的 Top-{CATEGORY_TOP_N} 中，"
        "带有明确的领域或品类特征，对类别判定有直接指示作用。"
        f"与此相对，{generic_text} 等评价类词会同时进入多数类别的高频词表，"
        "属于跨类别的通用评价语言，区分能力很弱。"
        "这一对比对后续 TF-IDF 特征工程有直接指导意义：通用评价词在各类别中的分布接近，"
        "TF-IDF 的 IDF 项本应自动压低它们的权重，但若其在文档中占比过高仍会稀释有效特征；"
        "因此特征工程阶段需要检查这类词的实际权重，必要时补充进停用词表。"
        "需注意本图只统计词频，未考虑词在类别间的相对分布，定量结论以 TF-IDF 权重诊断为准。"
    )


def plot_category_distribution(
    dataframe: pd.DataFrame, *, category_column: str, output_path: Path
) -> str:
    """绘制各类别样本数量与占比的柱状图。

    Args:
        dataframe: 已清洗的评论数据表。
        category_column: 类别列名（本项目为 ``cat``）。
        output_path: 图片输出路径，通常位于 ``reports/figures/`` 下。

    Returns:
        str: 中文结论文字，说明样本最多与最少的类别、最大最小样本量之比，
        并解释本任务为什么必须用 Macro-F1 选模而不能只看 Accuracy。
    """
    counts = dataframe[category_column].value_counts()
    total = int(counts.sum())
    shares = counts / total

    figure, axes = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(counts))
    bars = axes.bar(positions, counts.values, color="#2c7fb8", edgecolor="white")
    axes.set_title(f"各类别样本数量分布（共 {total:,} 条评论，10 个类别）", fontsize=13)
    axes.set_xlabel("商品或服务类别")
    axes.set_ylabel("样本数量（条）")
    axes.set_xticks(positions)
    axes.set_xticklabels(counts.index, rotation=0)
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)

    for bar, count, share in zip(bars, counts.values, shares.values):
        axes.annotate(
            f"{count:,}\n{share:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes.set_ylim(0, counts.max() * 1.18)
    axes.legend(
        [bars[0]],
        [f"样本量（最大/最小 = {counts.max() / counts.min():.1f} 倍）"],
        loc="upper right",
    )
    _save(figure, output_path)

    smallest = counts.idxmin()
    largest = counts.idxmax()
    return (
        f"10 个类别共 {total:,} 条评论。样本最多的类别是「{largest}」"
        f"（{counts.max():,} 条，占 {shares.max():.2%}），最少的是「{smallest}」"
        f"（{counts.min():,} 条，占 {shares.min():.2%}），最大与最小之比为 "
        f"{counts.max() / counts.min():.1f}:1，属于明显的类别不均衡。"
        f"Accuracy 对本任务的小类完全不敏感：即使模型彻底忽略只占 {shares.min():.2%} 的"
        f"「{smallest}」，整体 Accuracy 的下降也不到 {shares.min():.1%}，"
        "模型仍可能看起来「表现良好」。因此本项目以验证集 Macro-F1 作为主要选模指标，"
        "同时报告每类的 precision / recall / F1 与混淆矩阵，"
        "确保少数类的表现不被平均掉。"
    )


def plot_category_sentiment(
    dataframe: pd.DataFrame,
    *,
    category_column: str,
    sentiment_column: str,
    output_path: Path,
) -> str:
    """绘制每个类别内部负向 / 正向情感的堆叠柱状图。

    Args:
        dataframe: 已清洗的评论数据表。
        category_column: 类别列名。
        sentiment_column: 情感标签列名（0 表示负向，1 表示正向）。
        output_path: 图片输出路径。

    Returns:
        str: 中文结论文字，给出整体正负情感比例、正负最倾斜与最均衡的类别，
        并提示情感任务在不同类别上的难度并不一致，且该倾斜可能与数据采集方式有关。
    """
    pivot = (
        dataframe.groupby([category_column, sentiment_column], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for label in (0, 1):
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[[0, 1]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    totals = pivot.sum(axis=1)
    positive_share = pivot[1] / totals

    figure, axes = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(pivot))
    negative_bars = axes.bar(
        positions, pivot[0].values, color=NEGATIVE_COLOR, label="负向 (label=0)"
    )
    positive_bars = axes.bar(
        positions,
        pivot[1].values,
        bottom=pivot[0].values,
        color=POSITIVE_COLOR,
        label="正向 (label=1)",
    )
    axes.set_title("各类别内部情感分布（负向 / 正向堆叠）", fontsize=13)
    axes.set_xlabel("商品或服务类别")
    axes.set_ylabel("样本数量（条）")
    axes.set_xticks(positions)
    axes.set_xticklabels(pivot.index)
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)
    axes.legend(loc="upper right")

    for negative_bar, positive_bar, share, total in zip(
        negative_bars, positive_bars, positive_share.values, totals.values
    ):
        axes.annotate(
            f"正向 {share:.1%}",
            xy=(positive_bar.get_x() + positive_bar.get_width() / 2, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axes.set_ylim(0, totals.max() * 1.15)
    _save(figure, output_path)

    overall = dataframe[sentiment_column].value_counts(normalize=True)
    most_skewed = positive_share.idxmax()
    most_balanced = (positive_share - 0.5).abs().idxmin()
    return (
        f"整体情感分布接近均衡：正向 {overall.get(1, 0):.2%}、负向 "
        f"{overall.get(0, 0):.2%}。但类别之间差异明显：「{most_skewed}」的正向占比为 "
        f"{positive_share.max():.2%}，正负样本严重倾斜；而「{most_balanced}」的正向占比为 "
        f"{positive_share[most_balanced]:.2%}，接近均衡。"
        "这意味着扩展任务（情感二分类）的难度在不同类别上并不一致，"
        "评估时必须按类别分别查看召回率，不能只看整体指标；同时提示「"
        f"{most_skewed}」类别的情感标签可能受采集方式影响，需要在报告中如实说明。"
    )


def plot_length_histogram(
    texts: pd.Series, *, output_path: Path, clip_quantile: float = 0.995
) -> str:
    """绘制评论长度直方图，并在高分位处截断以便观察主体分布。

    Args:
        texts: 评论文本列。
        output_path: 图片输出路径。
        clip_quantile: 用于截断显示的分位数，默认 0.995；避免极少数超长评论
            把横轴拉得过长，导致主体分布被压缩而看不清。

    Returns:
        str: 中文结论文字，包含中位数、平均值、95% / 99% 分位数与最长长度，
        并说明分布右偏时应优先参考中位数和分位数而不是平均值。
    """
    # 统一转换为 float64：pandas 3.x 对可空 Int64 序列以小数作为截断边界时会
    # 抛出 TypeError，因此先转成字符串取长度、再落到浮点类型上做分位数计算。
    lengths = texts.astype("string").str.len().dropna().astype("float64")
    upper = float(lengths.quantile(clip_quantile))
    overflow = int((lengths > upper).sum())

    figure, axes = plt.subplots(figsize=(10, 5.5))
    axes.hist(
        lengths.clip(upper=upper).values,
        bins=60,
        color="#4c72b0",
        edgecolor="white",
        label=f"评论长度分布（{upper:.0f} 字符以上已截断显示，共 {overflow:,} 条）",
    )
    for value, color, name in (
        (float(lengths.median()), "#e67e22", "中位数"),
        (float(lengths.mean()), "#8e44ad", "平均值"),
        (float(lengths.quantile(0.95)), "#c0392b", "95% 分位"),
    ):
        axes.axvline(value, color=color, linestyle="--", linewidth=1.6, label=f"{name} = {value:.1f}")
    axes.set_title(f"评论文本长度分布（n = {len(lengths):,}）", fontsize=13)
    axes.set_xlabel("评论长度（字符数）")
    axes.set_ylabel("样本数量（条）")
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)
    axes.legend(loc="upper right")
    _save(figure, output_path)

    return (
        f"评论长度呈典型的长尾分布：中位数 {lengths.median():.0f} 字符，"
        f"平均值 {lengths.mean():.1f}，95% 分位 {lengths.quantile(0.95):.0f}，"
        f"99% 分位 {lengths.quantile(0.99):.0f}，最长 {int(lengths.max()):,} 字符。"
        "分布高度右偏，说明少数超长评论会把平均值拉高，"
        "决策时应优先参考中位数和分位数而不是平均值。"
    )


def plot_length_boxplot(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    category_column: str,
    output_path: Path,
) -> str:
    """绘制各类别评论长度的箱线图（纵轴采用对数刻度）。

    Args:
        dataframe: 已清洗的评论数据表。
        text_column: 评论文本列名。
        category_column: 类别列名。
        output_path: 图片输出路径。

    Returns:
        str: 中文结论文字，指出长度中位数最长与最短的类别，并说明截断长度必须
        对所有类别统一取值，否则短文本类别会在特征上获得不公平的优势。
    """
    working = dataframe[[category_column, text_column]].copy()
    working["_length"] = working[text_column].astype("string").str.len()
    working = working.dropna(subset=["_length"])
    order = working[category_column].value_counts().index
    grouped = [working.loc[working[category_column] == category, "_length"].values for category in order]

    figure, axes = plt.subplots(figsize=(10.5, 5.5))
    boxplot = axes.boxplot(
        grouped,
        tick_labels=list(order),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#c0392b", "linewidth": 1.8},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#a6cee3")
        patch.set_edgecolor("#1f4e79")
    axes.set_yscale("log")
    axes.set_title("各类别评论文本长度分布（对数刻度，箱线图，异常值点已隐藏）", fontsize=13)
    axes.set_xlabel("商品或服务类别")
    axes.set_ylabel("评论长度（字符数，对数刻度）")
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)
    axes.legend(
        [boxplot["medians"][0]],
        ["箱体为四分位区间，红线为中位数"],
        loc="upper right",
    )
    _save(figure, output_path)

    medians = {category: float(np.median(values)) for category, values in zip(order, grouped)}
    longest = max(medians, key=medians.get)
    shortest = min(medians, key=medians.get)
    return (
        f"各类别的长度中位数差异明显：「{longest}」最长（{medians[longest]:.0f} 字符），"
        f"「{shortest}」最短（{medians[shortest]:.0f} 字符）。"
        "所有类别都存在超长尾部，说明长度差异是类别间的固有差异，"
        "截断策略必须对所有类别采用统一长度，不能按类别分别截断，"
        "否则会让短文本类别在特征上获得不公平优势。"
    )


def plot_length_cdf(
    texts: pd.Series, *, output_path: Path, candidates: Sequence[int] = MAX_LENGTH_CANDIDATES
) -> tuple[str, list[dict[str, Any]]]:
    """绘制评论长度的经验累积分布，并标注各候选 max_length 的覆盖率。

    Args:
        texts: 评论文本列。
        output_path: 图片输出路径。
        candidates: 候选截断长度集合，默认取 MAX_LENGTH_CANDIDATES。

    Returns:
        tuple[str, list[dict[str, Any]]]: 二元组，返回值结构如下：
        第一个元素是中文结论文字，说明 max_length=128 与 max_length=256 的
        覆盖率，以及继续加长带来的收益递减；
        第二个元素是覆盖率明细表，列表中每个字典的字段含义为：
        ``max_length``（候选截断长度）、``coverage``（完整保留占比）、
        ``truncated``（被截断占比）、``truncated_rows``（被截断样本数），
        该表由 :func:`render_eda_report` 渲染成报告里的 Markdown 表格。
    """
    lengths = np.sort(
        texts.astype("string").str.len().dropna().astype("float64").to_numpy()
    )
    total = len(lengths)
    cumulative = np.arange(1, total + 1) / total

    table: list[dict[str, Any]] = []
    for candidate in candidates:
        covered = float((lengths <= candidate).mean())
        table.append(
            {
                "max_length": int(candidate),
                "coverage": covered,
                "truncated": 1.0 - covered,
                "truncated_rows": int(total - (lengths <= candidate).sum()),
            }
        )

    figure, axes = plt.subplots(figsize=(10.5, 5.5))
    axes.plot(lengths, cumulative, color="#2c7fb8", linewidth=2, label="评论长度累积分布")
    for row in table:
        axes.axvline(row["max_length"], color="#7f8c8d", linestyle=":", linewidth=1.1)
        axes.annotate(
            f"{row['max_length']}\n{row['coverage']:.1%}",
            xy=(row["max_length"], row["coverage"]),
            xytext=(0, -32),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2c3e50",
        )
        axes.plot([row["max_length"]], [row["coverage"]], marker="o", color="#c0392b", markersize=4)

    axes.axhline(0.95, color="#e67e22", linestyle="--", linewidth=1.3, label="95% 覆盖率参考线")
    axes.set_title("评论长度累积分布与候选 max_length 覆盖率", fontsize=13)
    axes.set_xlabel("截断长度 max_length（字符数）")
    axes.set_ylabel("不超过该长度的样本累计占比")
    axes.set_xlim(0, min(int(lengths.max()), 1024))
    axes.set_ylim(0, 1.02)
    axes.grid(linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)
    axes.legend(loc="lower right")
    _save(figure, output_path)

    row_128 = next(row for row in table if row["max_length"] == 128)
    row_256 = next(row for row in table if row["max_length"] == 256)
    conclusion = (
        f"以字符长度近似 token 长度（中文 BERT 的词表对汉字基本一字一 token），"
        f"max_length=128 可完整覆盖 {row_128['coverage']:.2%} 的评论，"
        f"max_length=256 可覆盖 {row_256['coverage']:.2%}。"
        "覆盖率随长度增加而快速趋缓，说明继续加长收益有限但计算成本线性上升。"
    )
    return conclusion, table


def plot_cleaning_funnel(
    steps: Sequence[Mapping[str, Any]], *, output_path: Path
) -> str:
    """绘制数据清洗各步骤样本量变化的柱状图（清洗漏斗）。

    Args:
        steps: 清洗步骤统计序列，每一项为映射，需包含 ``name``（步骤名称）、
            ``rows_before`` / ``rows_after``（该步骤执行前后的行数）、
            ``rows_removed``（删除的行数）与 ``rows_modified``（仅修正未删除的行数）。
        output_path: 图片输出路径。

    Returns:
        str: 中文结论文字，给出清洗前后样本量、合计删除量、删除量最大的步骤，
        并强调本流程「修复而非丢弃」这一关键取舍。
    """
    labels = ["原始数据"] + [str(step["name"]) for step in steps]
    counts = [int(steps[0]["rows_before"])] + [int(step["rows_after"]) for step in steps]

    figure, axes = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(counts))
    bars = axes.bar(positions, counts, color="#5b8c5a", edgecolor="white")
    axes.set_title("数据清洗各步骤样本量变化", fontsize=13)
    axes.set_xlabel("处理步骤（按执行顺序）")
    axes.set_ylabel("样本数量（条）")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    axes.set_ylim(min(counts) * 0.995, max(counts) * 1.002)
    axes.grid(axis="y", linestyle="--", alpha=0.4)
    axes.set_axisbelow(True)

    removed_total = counts[0] - counts[-1]
    for bar, count in zip(bars, counts):
        axes.annotate(
            f"{count:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axes.legend(
        [bars[0]],
        [f"清洗共删除 {removed_total:,} 条（{removed_total / counts[0]:.4%}）"],
        loc="lower right",
    )
    _save(figure, output_path)

    removed_steps = [
        (str(step["name"]), int(step["rows_removed"]))
        for step in steps
        if int(step["rows_removed"]) > 0
    ]
    if removed_steps:
        largest_name, largest_count = max(removed_steps, key=lambda item: item[1])
        detail = "、".join(f"{name} 删除 {count} 条" for name, count in removed_steps)
        removal_sentence = (
            f"删除量最大的是「{largest_name}」（{largest_count} 条），具体为：{detail}。"
        )
    else:
        removal_sentence = "本次清洗没有任何步骤删除样本。"

    modified_total = sum(int(step["rows_modified"]) for step in steps)
    return (
        f"清洗前 {counts[0]:,} 条，清洗后 {counts[-1]:,} 条，合计删除 {removed_total:,} 条"
        f"（{removed_total / counts[0]:.4%}）。{removal_sentence}"
        "删除比例低于千分之一，对类别分布和样本规模没有实质影响；"
        f"另有 {modified_total:,} 条文本仅被修正（控制字符、HTML 实体、空白归一化）而未被删除，"
        "这一类「修复而非丢弃」的处理保留了有效样本，是本流程的关键取舍。"
    )


def render_eda_report(
    *,
    conclusions: Mapping[str, str],
    figure_paths: Mapping[str, str],
    length_table: Sequence[Mapping[str, Any]],
    font_name: str | None,
    dataset_summary: Mapping[str, Any],
    max_length_recommendation: str,
    word_frequencies: Sequence[tuple[str, int]] = (),
) -> str:
    """把各图表的结论文本、图片路径与覆盖率表格渲染成完整的 EDA 报告。

    Args:
        conclusions: 图表名称到中文结论文字的映射，键包括 ``category``、
            ``sentiment``、``length_hist``、``length_box``、``length_cdf``、
            ``cleaning``、``wordcloud``、``top_words``，分别对应各绘图函数的返回值。
        figure_paths: 图表名称到图片相对路径的映射，键与 ``conclusions`` 一致。
        length_table: :func:`plot_length_cdf` 返回的覆盖率明细表。
        font_name: 实际生效的绘图中文字体名，``None`` 表示未找到中文字体。
        dataset_summary: 数据集规模摘要，需包含 ``raw_rows``、``clean_rows``、
            ``train_rows``、``dev_rows``、``test_rows``。
        max_length_recommendation: 关于 BERT ``max_length`` 的中文评审建议文字。
        word_frequencies: 整体高频词列表，用于在报告中附上高频词 Top-N 表格。

    Returns:
        str: 完整的 Markdown 报告文本，由上层脚本写入 ``reports/eda-report.md``。
    """
    lines: list[str] = [
        "# 探索性数据分析报告（EDA）",
        "",
        "本报告由 `scripts/run_eda.py` 生成。所有图表保存在 `reports/figures/`，",
        "结论中的数字均由脚本从真实数据计算得到。",
        "",
        f"- 绘图中文字体：{font_name if font_name else '未找到中文字体，图中中文可能显示为方框'}",
        f"- 原始样本量：{dataset_summary['raw_rows']:,}",
        f"- 清洗后样本量：{dataset_summary['clean_rows']:,}",
        f"- 训练集 / 验证集 / 测试集：{dataset_summary['train_rows']:,} / "
        f"{dataset_summary['dev_rows']:,} / {dataset_summary['test_rows']:,}",
        "",
        "## 1. 图表清单",
        "",
        "| 图表 | 文件 | 说明 |",
        "| --- | --- | --- |",
        f"| 类别分布柱状图 | `{figure_paths['category']}` | 10 个类别的样本量与占比 |",
        f"| 类别内情感堆叠图 | `{figure_paths['sentiment']}` | 各类别正负情感构成 |",
        f"| 文本长度直方图 | `{figure_paths['length_hist']}` | 长度分布与分位数参考线 |",
        f"| 文本长度箱线图 | `{figure_paths['length_box']}` | 各类别长度四分位对比 |",
        f"| 长度累积分布图 | `{figure_paths['length_cdf']}` | 候选 max_length 覆盖率 |",
        f"| 清洗步骤变化图 | `{figure_paths['cleaning']}` | 各步骤样本量变化 |",
        f"| 分词词云图 | `{figure_paths['wordcloud']}` | jieba 分词后的整体词云 |",
        f"| 各类别高频词图 | `{figure_paths['top_words']}` | 10 个类别的 Top-N 高频词 |",
        "",
        "## 2. 类别分布结论",
        "",
        conclusions["category"],
        "",
        "## 3. 类别内情感分布结论",
        "",
        conclusions["sentiment"],
        "",
        "## 4. 文本长度结论",
        "",
        conclusions["length_hist"],
        "",
        conclusions["length_box"],
        "",
        "## 5. 数据清洗前后对比结论",
        "",
        conclusions["cleaning"],
        "",
        "## 6. 分词与词云分析",
        "",
        "分词方式：jieba 精确模式。停用词过滤规则：过滤长度小于 "
        f"{WORDCLOUD_MIN_LENGTH} 的词、{len(CHINESE_STOPWORDS)} 个中文停用词，"
        "以及既不含中文也不含拉丁字母的纯标点或纯数字串。",
        "统计范围：**训练集**，与后续 TF-IDF 特征拟合的口径保持一致，",
        "避免任何测试集信息影响特征工程决策。",
        "",
        "### 6.1 整体词云",
        "",
        conclusions["wordcloud"],
        "",
        "### 6.2 各类别高频词",
        "",
        conclusions["top_words"],
        "",
        "### 6.3 整体高频词 Top-20（定量统计）",
        "",
        "词云只能反映视觉上的词频高低，**不能代替严谨统计**，因此附上高频词明细表。",
        "",
        "| 排名 | 词 | 出现次数 |",
        "| ---: | --- | ---: |",
    ]
    for rank, (word, count) in enumerate(list(word_frequencies)[:20], start=1):
        lines.append(f"| {rank} | {word} | {count:,} |")

    lines.extend(
        [
            "",
            "## 7. BERT 候选 max_length 评审",
            "",
            conclusions["length_cdf"],
            "",
            "### 7.1 覆盖率明细",
            "",
            "| 候选 max_length | 完整保留占比 | 被截断占比 | 被截断样本数 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in length_table:
        lines.append(
            f"| {row['max_length']} | {row['coverage']:.2%} | {row['truncated']:.2%} | "
            f"{row['truncated_rows']:,} |"
        )

    lines.extend(
        [
            "",
            "### 7.2 结论与建议",
            "",
            max_length_recommendation,
            "",
            "### 7.3 本评审的局限",
            "",
            "- 上表用**字符长度**近似 token 长度。中文 BERT 的 WordPiece 词表对汉字基本是一字一 token，",
            "  因此该近似在中文评论上偏差很小，但仍是近似值。",
            "- 本项目当前环境无法下载预训练分词器，因此未能用真实 tokenizer 复核。",
            "  在 BERT 阶段拿到本地分词器后，必须用真实 token 长度重新计算一次覆盖率，",
            "  并把结果补充到本节；在此之前，本节的数字应标注为「字符长度近似」。",
            "",
        ]
    )
    return "\n".join(lines)
