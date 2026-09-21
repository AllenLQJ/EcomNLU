"""数据流水线共用的文本正则与判定标准。

模块职责：
    集中定义「什么样的文本算异常」这一判断标准，供清洗、EDA、测试等模块复用，
    避免同一套正则在多处各写一遍而逐渐走样。

主要输入与输出：
    输入：无（本模块只提供常量）。
    输出：一组已编译的正则对象，以及基于它们的小工具函数。

关键设计理由与约束（本项目踩过的坑，务必遵守）：

1. **CJK 正则必须用非 raw 字符串书写。**
   pandas 3.x 会把 ``Series.str`` 上的正则交给 PyArrow/RE2 执行，而 RE2 不支持
   ``\\uXXXX`` 转义，写 ``r"[\\u4e00-\\u9fff]"`` 会直接抛 ``ArrowInvalid``。
   因此这里写成 ``"[\\u4e00-\\u9fff]"``（不加 raw 前缀），让 Python 先把转义
   解析成真实字符，再交给正则引擎。

2. **不要用 ``\\w`` / ``\\W`` / ``\\s`` 判断中文文本。**
   RE2 的这几个字符类只识别 ASCII：``\\w`` 不含汉字，``\\s`` 不含全角空格
   （U+3000）与不换行空格（U+00A0）。曾经因此把「纯数字或纯符号评论」
   误报成 55,799 条（占全体 89%）。需要 Unicode 语义时，请用 Python 的 ``re``
   直接处理字符串，而不要走 pandas 的 ``.str`` 方法。

3. **本模块不承载任何 I/O**，读取原始数据请使用 :mod:`src.data.raw_io`。

说明：原始数据质量的**正式检查与报告**由 ``src/data/check_raw.py`` 负责
（该模块由 LanQvJian 编写，属于接口契约验收的一部分）。本模块只提供两者共同的
判定标准，不重复实现报告逻辑。
"""

from __future__ import annotations

import re


# ASCII 控制字符，以及 C1 控制字符区间。
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")

# HTML 实体，例如 &#183; 与 &quot;。
HTML_ENTITY_PATTERN = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,10}|#\d{1,6});")

# HTML 标签，例如 <br> 与 </p>。
HTML_TAG_PATTERN = re.compile(r"<[^<>\n]{0,40}>")

# URL。
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# 汉字区间 U+4E00–U+9FFF。
# 刻意不使用 raw 字符串：Python 会先把 \u 转义解析成真实字符，
# 这样 RE2 收到的是字面字符而不是它不认识的 \u 转义。
CJK_PATTERN = re.compile("[\u4e00-\u9fff]")

# 拉丁字母，用于识别英文或拼音评论。
LATIN_LETTER_PATTERN = re.compile("[A-Za-z]")

# Unicode 格式字符（Unicode 类别 Cf）：BOM、零宽空格与连接符、双向控制符、软连字符。
# 这些字符不可见，但会被 jieba 切成独立 token，污染 TF-IDF 词表；
# 原始数据中有 20 条评论文本内部嵌入了 U+FEFF。
UNICODE_FORMAT_PATTERN = re.compile(
    "["
    "\u00ad"  # 软连字符
    "\u200b-\u200f"  # 零宽空格 至 从右到左标记
    "\u202a-\u202e"  # 双向嵌入与覆盖
    "\u2060-\u2064"  # 词连接符、不可见运算符
    "\u2066-\u206f"  # 双向隔离符、不可见运算符
    "\ufeff"  # BOM / 零宽不换行空格
    "\ufff9-\ufffb"  # 行间注释符
    "]"
)

# ASCII 之外的 Unicode 空白：不换行空格、en/em 空格、全角空格、行分隔符等。
UNICODE_SPACE_PATTERN = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]"
)

# 合并任意空白用。**必须用 Python 的 re 处理字符串，不要传给 pandas 的 .str**，
# 否则会走 RE2 的 ASCII 版 \s，全角空格无法合并。
WHITESPACE_PATTERN = re.compile(r"\s+")


def has_meaningful_character(text: str) -> bool:
    """判断文本是否含有中文或拉丁字母。

    用于识别「既无中文也无字母」的纯标点或纯数字噪声。刻意不使用 ``\\W``
    判断——RE2 的 ``\\W`` 会把汉字当作非单词字符，导致中文文本被误判为噪声。

    参数：
        text：待判断的文本。

    返回：
        文本含中文或拉丁字母时返回 True。
    """
    return bool(CJK_PATTERN.search(text) or LATIN_LETTER_PATTERN.search(text))


def normalize_whitespace(text: str) -> str:
    """归一化文本中的空白。

    合并换行、制表符、连续空格，以及全角空格（U+3000）和不换行空格（U+00A0）
    等 Unicode 空白，并去除首尾空白。使用 Python 的 ``re`` 以保证 Unicode 语义。

    参数：
        text：待归一化的文本。

    返回：
        归一化后的文本。
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()
