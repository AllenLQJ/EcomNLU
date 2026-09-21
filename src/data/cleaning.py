"""中文电商评论数据集的数据清洗模块。

模块职责：
    对原始评论数据按固定顺序执行 7 步清洗规则，并逐步记录每一步的输入样本量、
    输出样本量、删除量与修改量，供 `scripts/clean_data.py` 生成可核查的清洗报告。

主要输入与输出：
    输入：至少包含评论、类别、情感三列的 pandas DataFrame。
    输出：CleaningResult，内含清洗后的 DataFrame 与按执行顺序排列的 CleaningStep 列表。

关键设计理由与约束：
    清洗规则按以下固定顺序执行，每一步都会记录输入/输出/删除量/修改量，
    使清洗报告能够展示数据实际发生了什么：

    1. 删除评论缺失或为空的样本（没有文本就无法分类）。
    2. 删除控制字符（保留样本，只修复文本）。
    3. 删除 Unicode 格式字符：BOM、零宽空格与零宽连接符、软连字符（保留样本）。
    4. 将 ``&#183;`` 之类的 HTML 实体还原为真实字符。
    5. 空白归一化：换行、制表符、连续空格以及全角表意空格等 Unicode 空白统一为单个空格。
    6. 删除既不含中文字符也不含拉丁字母的样本，即纯标点或纯数字噪声。
       英文与拼音评论予以保留，因为它们依然携带类别与情感信号。
    7. 删除重复的 ``(评论, 类别)`` 组合记录。

    URL 有意保留：原始数据中仅 17 条含 URL，它们携带信息，且没有规则要求删除。
    报告会给出该计数，使这一决定保持可见，而不是被静默执行。

    本模块显式处理两个 pandas 3.x 陷阱：

    * 字符串正则会交给 PyArrow/RE2 执行，而 RE2 的 ``\\s`` 只识别 ASCII。
      因此 ``.str.replace(r"\\s+")`` 永远无法合并全角表意空格，
      所以空白归一化改用 Python 中支持 Unicode 的 ``re`` 完成。
    * Unicode 格式字符使用 ``unicodedata`` 类别判断，而不是写一个巨大的正则字符类，
      从而不会因为字符类写漏而意外放过某些格式字符。

    本模块只读原始数据，绝不修改 ``data/raw`` 目录下的任何文件。
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from src.data.text_patterns import (
    CJK_PATTERN,
    CONTROL_CHAR_PATTERN,
    LATIN_LETTER_PATTERN,
    UNICODE_FORMAT_PATTERN,
    UNICODE_SPACE_PATTERN,
    WHITESPACE_PATTERN,
    normalize_whitespace,
)


EXAMPLE_LIMIT = 3
EXAMPLE_WIDTH = 60

# 数据契约要求产出的每一行都能追溯回原始数据，因此清洗阶段必须保留这一列。
SAMPLE_ID_COLUMN = "sample_id"


@dataclass(frozen=True)
class CleaningStep:
    """单个清洗步骤的执行结果。

    Attributes:
        name: 步骤标识，供报告与调用方引用。
        description: 步骤的中文规则说明，直接写入清洗报告。
        rows_before: 该步骤执行前的样本量。
        rows_after: 该步骤执行后的样本量。
        rows_modified: 仅修改文本而保留样本的行数，默认为 0。
        examples: 该步骤修改或删除的样例文本，最多 EXAMPLE_LIMIT 条。
    """

    name: str
    description: str
    rows_before: int
    rows_after: int
    rows_modified: int = 0
    examples: tuple[str, ...] = ()

    @property
    def rows_removed(self) -> int:
        """返回该步骤删除的样本量。

        Returns:
            输入样本量减去输出样本量的差值。
        """
        return self.rows_before - self.rows_after


@dataclass
class CleaningResult:
    """清洗后的数据表，以及按执行顺序记录的步骤结果。

    Attributes:
        dataframe: 清洗后的数据表，索引已重置。
        steps: 已执行步骤的有序列表，用于生成清洗报告。
    """

    dataframe: pd.DataFrame
    steps: list[CleaningStep] = field(default_factory=list)

    @property
    def rows_before(self) -> int:
        """返回清洗前的样本量。

        Returns:
            第一步记录的输入样本量；没有任何步骤时返回 0。
        """
        return self.steps[0].rows_before if self.steps else 0

    @property
    def rows_after(self) -> int:
        """返回清洗后的样本量。

        Returns:
            最后一步记录的输出样本量；没有任何步骤时返回 0。
        """
        return self.steps[-1].rows_after if self.steps else 0


def _escape_control_char(match: re.Match[str]) -> str:
    """把不可见的控制字符转成可读的 ``\\x7f`` 形式。

    Args:
        match: 正则匹配对象，``match.group()`` 为被匹配到的控制字符。

    Returns:
        形如 ``\\x7f`` 的可读转义文本。
    """
    return "\\x" + format(ord(match.group()), "02x")


def _escape_format_char(match: re.Match[str]) -> str:
    """把不可见的 Unicode 格式字符渲染为 ``[U+FEFF]``。

    Args:
        match: 正则匹配对象，``match.group()`` 为被匹配到的格式字符。

    Returns:
        形如 ``[U+FEFF]`` 的可读文本。
    """
    return f"[U+{ord(match.group()):04X}]"


def _strip_unicode_format(value: str) -> str:
    """删除单个文本中的 Unicode 格式字符（类别 Cf）。

    Args:
        value: 待处理的评论文本。

    Returns:
        删除格式字符之后的文本。
    """
    return UNICODE_FORMAT_PATTERN.sub("", value)


def _preview(text: str) -> str:
    """为报告缩短文本，并让不可见字符变得可见。

    如果没有这些转义步骤，不可见字符不会在报告中留下任何痕迹，
    那些清洗规则的证据也就无法核查。

    Args:
        text: 待预览的文本。

    Returns:
        不超过 EXAMPLE_WIDTH 个字符的预览文本，超长时以 ``...`` 结尾。
    """
    collapsed = WHITESPACE_PATTERN.sub(" ", str(text)).strip()
    collapsed = CONTROL_CHAR_PATTERN.sub(_escape_control_char, collapsed)
    collapsed = UNICODE_FORMAT_PATTERN.sub(_escape_format_char, collapsed)
    if len(collapsed) <= EXAMPLE_WIDTH:
        return collapsed
    return collapsed[:EXAMPLE_WIDTH] + "..."


def _record(
    steps: list[CleaningStep],
    *,
    name: str,
    description: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    modified_mask: pd.Series | None = None,
    examples: Sequence[str] = (),
) -> None:
    """向步骤列表追加一条清洗记录。

    Args:
        steps: 已执行步骤的记录列表，本函数就地追加，不返回新列表。
        name: 步骤标识。
        description: 步骤的中文规则说明。
        before: 该步骤执行前的数据表。
        after: 该步骤执行后的数据表。
        modified_mask: 标记文本被修改的布尔掩码，None 表示该步骤只删除样本。
        examples: 该步骤的样例文本，最多保留 EXAMPLE_LIMIT 条。
    """
    steps.append(
        CleaningStep(
            name=name,
            description=description,
            rows_before=int(before.shape[0]),
            rows_after=int(after.shape[0]),
            rows_modified=int(modified_mask.sum()) if modified_mask is not None else 0,
            examples=tuple(examples[:EXAMPLE_LIMIT]),
        )
    )


def clean_reviews(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    category_column: str,
    sentiment_column: str,
    sample_id_column: str = SAMPLE_ID_COLUMN,
) -> CleaningResult:
    """应用模块说明中约定的清洗规则，并记录每一步的结果。

    规则严格按模块说明中的固定顺序执行，每一步的输入样本量、输出样本量、
    删除量与修改量都会被记录，供 :func:`render_cleaning_report` 汇总成报告。

    清洗过程会生成并保留 ``sample_id`` 列，其值为该记录在**原始 CSV 中从 0 开始的行号**。
    这是接口契约的硬性要求：任何处理后样本都必须能追溯回原始数据，
    否则无法用原始数据核对 ``cat`` 与 ``label`` 是否被意外改动。因此这里不使用
    ``reset_index`` 作为标识，而是在步骤开始时就把原始行号落成显式列，
    使它能在删行、去重、划分之后依然正确。

    Args:
        dataframe: 待清洗的原始数据表，必须与原始 CSV 的行序一致（未经筛选或重排）。
        text_column: 评论文本列名。
        category_column: 类别列名。
        sentiment_column: 情感标签列名。
        sample_id_column: 追溯用行号列名，默认 ``sample_id``。

    Returns:
        一个 :class:`CleaningResult`，包含清洗后的数据表（列顺序为
        ``sample_id, review, cat, label``），以及用于生成报告的、
        按执行顺序排列的 :class:`CleaningStep` 列表。

    Raises:
        KeyError: 当输入数据表缺少任一必需列时抛出。
        ValueError: 当清洗后出现空评论文本或非法情感标签时抛出。
    """
    for column in (text_column, category_column, sentiment_column):
        if column not in dataframe.columns:
            raise KeyError(f"Missing required column: {column}")

    steps: list[CleaningStep] = []
    working = dataframe[[category_column, sentiment_column, text_column]].copy()
    # 原始 CSV 行号（不含表头，从 0 开始）。必须在任何删行操作之前写入，
    # 这样后续步骤删掉的行不会让其余记录的行号发生错位。
    working[sample_id_column] = range(len(working))

    # --- 步骤 1：评论缺失或为空白 -----------------------------------------
    texts = working[text_column].astype("string")
    blank_mask = texts.isna() | (texts.str.strip() == "")
    before = working
    working = working.loc[~blank_mask].copy()
    _record(
        steps,
        name="drop_missing_or_blank",
        description="删除评论缺失或清洗后为空的样本",
        before=before,
        after=working,
    )

    # --- 步骤 2：ASCII 控制字符 -------------------------------------------
    original = working[text_column].astype("string")

    def _remove_control(value: str) -> str:
        """删除单个文本中的 ASCII 控制字符。

        Args:
            value: 待处理的评论文本。

        Returns:
            删除控制字符之后的文本。
        """
        return CONTROL_CHAR_PATTERN.sub("", value)

    without_control = original.map(_remove_control)
    control_mask = without_control != original
    working[text_column] = without_control
    _record(
        steps,
        name="remove_control_chars",
        description="删除异常控制字符（保留样本）",
        before=working,
        after=working,
        modified_mask=control_mask,
        examples=[
            _preview(original.loc[index]) for index in original.index[control_mask]
        ],
    )

    # --- 步骤 3：Unicode 格式字符（BOM、零宽连接符） ----------------------
    without_format = without_control.map(_strip_unicode_format)
    format_mask = without_format != without_control
    working[text_column] = without_format
    _record(
        steps,
        name="strip_unicode_format_chars",
        description="删除 Unicode 格式字符（BOM、零宽字符、软连字符），保留样本",
        before=working,
        after=working,
        modified_mask=format_mask,
        examples=[
            _preview(without_control.loc[index])
            for index in without_control.index[format_mask]
        ],
    )

    # --- 步骤 4：HTML 实体 ------------------------------------------------
    unescaped = without_format.map(html.unescape)
    entity_mask = unescaped != without_format
    working[text_column] = unescaped
    _record(
        steps,
        name="unescape_html_entities",
        description="将 HTML 实体（如 &#183;）还原为真实字符",
        before=working,
        after=working,
        modified_mask=entity_mask,
        examples=[
            _preview(without_format.loc[index]) for index in without_format.index[entity_mask]
        ],
    )

    # --- 步骤 5：空白归一化 ----------------------------------------------
    normalized = unescaped.map(normalize_whitespace)
    whitespace_mask = normalized != unescaped
    working[text_column] = normalized
    _record(
        steps,
        name="normalize_whitespace",
        description="合并换行、制表符、连续空格与全角空格，并去除首尾空白",
        before=working,
        after=working,
        modified_mask=whitespace_mask,
        examples=[
            _preview(unescaped.loc[index]) for index in unescaped.index[whitespace_mask]
        ],
    )

    # --- 步骤 6：无意义文本 ----------------------------------------------
    texts = working[text_column].astype("string")
    has_cjk = texts.str.contains(CJK_PATTERN, regex=True, na=False)
    has_letter = texts.str.contains(LATIN_LETTER_PATTERN, regex=True, na=False)
    meaningful_mask = has_cjk | has_letter
    before = working
    dropped_examples = [_preview(value) for value in texts.loc[~meaningful_mask]]
    working = working.loc[meaningful_mask].copy()
    _record(
        steps,
        name="drop_meaningless_text",
        description="删除既无中文字符也无拉丁字母的纯符号或纯数字样本",
        before=before,
        after=working,
        examples=dropped_examples,
    )

    # --- 步骤 7：重复的（评论, 类别）组合 ---------------------------------
    before = working
    duplicate_mask = working.duplicated(
        subset=[text_column, category_column], keep="first"
    )
    duplicate_examples = [
        _preview(working.loc[index, text_column]) for index in working.index[duplicate_mask]
    ]
    working = working.loc[~duplicate_mask].copy()
    _record(
        steps,
        name="deduplicate_within_category",
        description="删除同一类别内完全重复的评论文本",
        before=before,
        after=working,
        examples=duplicate_examples,
    )

    # --- 最终校验 ---------------------------------------------------------
    if working[text_column].isna().any() or (working[text_column] == "").any():
        raise ValueError("Cleaning produced empty review text")
    if not set(working[sentiment_column].unique()).issubset({0, 1}):
        raise ValueError("Cleaning produced an invalid sentiment label")
    if working[sample_id_column].duplicated().any():
        raise ValueError("Cleaning produced duplicate sample_id values")

    # 按数据契约固定列顺序输出：sample_id, review, cat, label
    working = working[
        [sample_id_column, text_column, category_column, sentiment_column]
    ].reset_index(drop=True)

    return CleaningResult(dataframe=working, steps=steps)


def render_cleaning_report(
    result: CleaningResult,
    *,
    source_path: str,
    output_path: str,
) -> str:
    """把已执行的清洗步骤渲染为 Markdown 报告。

    报告正文属于程序输出，其内容与措辞保持原样，便于与历史报告对比。

    Args:
        result: :func:`clean_reviews` 返回的清洗结果。
        source_path: 输入文件路径，写入报告的「输入输出」小节。
        output_path: 输出文件路径，写入报告的「输入输出」小节。

    Returns:
        完整的 Markdown 报告文本。
    """
    lines: list[str] = [
        "# 数据清洗报告",
        "",
        "本报告由 `scripts/clean_data.py` 生成，记录实际执行的每一步清洗规则及其影响。",
        "规则按顺序应用，每一步的样本量变化如下。",
        "",
        "## 1. 输入输出",
        "",
        f"- 输入文件：`{source_path}`",
        f"- 输出文件：`{output_path}`",
        f"- 清洗前样本量：{result.rows_before:,}",
        f"- 清洗后样本量：{result.rows_after:,}",
        f"- 合计删除：{result.rows_before - result.rows_after:,}"
        f"（{(result.rows_before - result.rows_after) / result.rows_before * 100:.4f}%）",
        "",
        "## 2. 清洗步骤与影响",
        "",
        "| 步骤 | 规则 | 输入样本 | 输出样本 | 删除 | 修改文本 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for step in result.steps:
        lines.append(
            f"| `{step.name}` | {step.description} | {step.rows_before:,} | "
            f"{step.rows_after:,} | {step.rows_removed:,} | {step.rows_modified:,} |"
        )

    lines.extend(
        [
            "",
            "## 3. 规则说明与决策依据",
            "",
            "1. **缺失与空白评论直接删除**：没有文本就无法分类，也无法通过插值修复，",
            "   且原始数据中此类样本数量极少，删除不会改变类别分布。",
            "2. **控制字符删除而非删除样本**：控制字符是编码噪声，样本本身仍有内容，",
            "   删除字符即可保留有效样本。",
            "3. **Unicode 格式字符删除而非删除样本**：原始数据中有 20 条评论文本内部嵌入了",
            "   U+FEFF（BOM / 零宽不换行空格）。这类字符不可见，但会被 jieba 切分成",
            "   独立 token，污染 TF-IDF 特征。属于同一类编码噪声，因此与规则 2 一样",
            "   只删除字符、保留样本。零宽连接符、软连字符等 Unicode 格式字符（Cf 类）",
            "   一并处理，避免遗漏。",
            "4. **HTML 实体还原为真实字符**：数据集中的 `&#183;` 等实体是采集合入的编码残留，",
            "   还原后文本可读且不丢失信息。",
            "5. **空白归一化**：合并换行、制表符、连续空格以及全角空格（U+3000）、",
            "   不换行空格（U+00A0）等 Unicode 空白，避免同一文本因排版差异被当成不同样本。",
            "   注意：本步骤使用 Python 的 `re` 而非 pandas 的 `.str.replace`，",
            "   因为 pandas 3.x 会把字符串正则交给 PyArrow/RE2 执行，而 RE2 的 `\\s` 只识别",
            "   ASCII 空白，无法匹配全角空格。",
            "6. **不做固定长度硬删除**：短评如「挺好的」「有点卡」依然是有效的类别与情感信号，",
            "   按长度阈值删除会人为引入分布偏差；只删除既无中文也无字母的纯符号或纯数字噪声。",
            "7. **同类别的重复文本删除**：同一类别内完全相同的文本属于冗余记录，",
            "   保留会使训练集重复计数并放大该样本权重。",
            "8. **URL 保留**：原始数据仅 17 条含 URL，且无规则要求删除，保留以免丢失信息。",
            "",
            "### 3.1 跨类别相同文本的处理（重要）",
            "",
            "原始数据中存在同一评论文本出现在多个类别的情况。这类记录**不是冗余数据**，",
            "而是同一文本对应多个合法标签，因此清洗阶段保留，改由划分阶段以文本为单位整体分配，",
            "确保同一文本不会同时出现在训练集和测试集。具体数量见划分报告。",
            "",
        ]
    )

    example_steps = [step for step in result.steps if step.examples]
    if example_steps:
        lines.extend(["## 4. 修改与删除样例", ""])
        for step in example_steps:
            lines.extend([f"### `{step.name}`", ""])
            for example in step.examples:
                lines.append(f"- {example}")
            lines.append("")

    return "\n".join(lines)
