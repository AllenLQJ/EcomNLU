"""中文评论的分词与 TF-IDF 特征提取。

模块职责：
    把**原始评论文本**转换成 TF-IDF 稀疏特征矩阵。分词使用 jieba，文本清洗复用
    ``src.data.cleaning`` 的规则，两部分都封装在向量器内部。

主要输入与输出：
    输入：原始评论文本（未清洗的字符串序列或 DataFrame 列）。
    输出：已拟合的 :class:`sklearn.feature_extraction.text.TfidfVectorizer`；
        以及 :func:`vectorize` 返回的稀疏特征矩阵。

关键设计理由与约束：

1. **向量器必须吃原始文本，并在内部完成清洗与分词。**
   ``src/inference/sklearn_adapter.py`` 是把原始文本直接交给 ``vectorizer.joblib`` 的，
   它不做任何预处理。因此清洗与分词必须封装进向量器，否则推理阶段的预处理会与训练
   阶段不一致，表现为「没有报错但准确率下降」（AGENTS.md §5.2、契约 §2.2）。
   本模块用 ``analyzer=`` 传入分词回调，把「清洗 -> 分词 -> 过滤」全部收进向量器。

2. **分词回调必须是模块级函数**，参数通过 :func:`functools.partial` 绑定。
   向量器要用 joblib 序列化，lambda 或嵌套函数无法反序列化，会导致推理阶段加载失败。

3. **只用训练集拟合向量器。** 词表与 IDF 权重都属于「从数据学到的参数」，
   在验证集或测试集上拟合会造成特征泄漏。因此 :func:`vectorize` 把「拟合训练集」
   与「变换其余集合」分开，调用方无法误用。

4. 字符级表示作为对照项保留：中文没有天然词边界，字符 n-gram 是常见的替代方案，
   本模块支持 ``analyzer_mode="char"``，用于在验证集上实测比较而不是凭直觉取舍。
"""

from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import jieba
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.cleaning import normalize_review
from src.data.config import public_path
from src.data.text_patterns import CHINESE_STOPWORDS, CJK_PATTERN, LATIN_LETTER_PATTERN


# 分词后保留的最小词长。设为 2 可过滤「的 / 了 / 是」等单字虚词；
# 单字实义词（如「好 / 差」）会随之丢失，这属于本项目的已知取舍，记录在对比报告中。
DEFAULT_MIN_TOKEN_LENGTH = 2

# 向量器的默认超参数。集中在此处，便于写入 metadata.json 复现训练。
DEFAULT_VECTORIZER_CONFIG: dict[str, Any] = {
    "analyzer_mode": "word",
    "ngram_range": (1, 1),
    "min_df": 2,
    "max_df": 1.0,
    "max_features": None,
    "sublinear_tf": True,
    "min_token_length": DEFAULT_MIN_TOKEN_LENGTH,
    "use_stopwords": True,
}


def is_meaningful_token(token: str, min_length: int) -> bool:
    """判断一个分词结果是否应当保留。

    过滤三类噪声：过短的词、停用词、以及既不含中文也不含拉丁字母的纯标点或纯数字串。

    参数：
        token：jieba 切分出的单个词。
        min_length：保留的最小词长（按字符数计）。

    返回：
        该词是否保留。判断使用 Python 的 ``re`` 直接处理字符串，
        而不是 pandas 的 ``.str`` 方法——后者会把正则交给 RE2，而 RE2 的字符类
        只按 ASCII 解释（详见 ``src/data/text_patterns.py`` 的模块说明）。
    """
    if len(token) < min_length:
        return False
    if not (CJK_PATTERN.search(token) or LATIN_LETTER_PATTERN.search(token)):
        return False
    return True


def tokenize_review(
    text: str,
    *,
    min_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    use_stopwords: bool = True,
) -> list[str]:
    """把一条原始评论切成词序列。

    处理顺序：先调用 :func:`src.data.cleaning.normalize_review` 完成文本清洗
    （控制字符、Unicode 格式字符、HTML 实体、空白归一化），再用 jieba 精确模式分词，
    最后过滤噪声词与停用词。

    该函数是**模块级函数**，因此可以被 :func:`functools.partial` 绑定参数后交给
    ``TfidfVectorizer(analyzer=...)``，并随向量器一起被 joblib 序列化。

    参数：
        text：原始评论文本。
        min_length：保留的最小词长。
        use_stopwords：是否过滤中文停用词表。

    返回：
        过滤后的词列表；文本清洗后为空时返回空列表。
    """
    cleaned = normalize_review(text)
    if not cleaned:
        return []
    tokens: list[str] = []
    for token in jieba.cut(cleaned):
        if not is_meaningful_token(token, min_length):
            continue
        if use_stopwords and token in CHINESE_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def char_preprocess(text: str) -> str:
    """字符级表示的预处理：清洗并去除空白。

    字符 n-gram 会把空格也当作字符，产生大量无意义的空格组合，因此需要去掉空白。
    同样是模块级函数，以保证向量器可被序列化。

    参数：
        text：原始评论文本。

    返回：
        清洗并去除空白后的文本。
    """
    return normalize_review(text).replace(" ", "")


def build_tfidf_vectorizer(
    *,
    analyzer_mode: str = "word",
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_df: float = 1.0,
    max_features: int | None = None,
    sublinear_tf: bool = True,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    use_stopwords: bool = True,
) -> TfidfVectorizer:
    """构造尚未拟合的 TF-IDF 向量器。

    ``word`` 模式用 jieba 分词，``char`` 模式用字符 n-gram。两种模式都把文本清洗
    放在向量器内部完成，因此向量器可以直接接收原始文本。

    参数：
        analyzer_mode：``"word"`` 或 ``"char"``。
        ngram_range：n-gram 范围。词级常用 (1, 1) 或 (1, 2)；字符级常用 (2, 3)。
        min_df：词频低于该值的词直接丢弃（整数表示绝对文档数）。
        max_df：文档频率高于该比例的词丢弃，可用于剔除跨类别通用词。
        max_features：词表上限，``None`` 表示不限制。
        sublinear_tf：是否用 ``1 + log(tf)`` 替代原始词频，文本任务常用。
        min_token_length：词级模式下的最小词长。
        use_stopwords：词级模式下是否过滤停用词。

    返回：
        未拟合的 :class:`TfidfVectorizer`。

    异常：
        ValueError：``analyzer_mode`` 不是 ``word`` 或 ``char`` 时抛出。
    """
    if analyzer_mode == "word":
        # 用 tokenizer= 而不是 analyzer=：``analyzer`` 传入可调用对象时，
        # sklearn 会**忽略 ngram_range**（会给出 UserWarning），
        # 于是「词级 bigram」这类配置会静默退化成 unigram。
        # 改用 tokenizer= 并显式关闭 token_pattern，ngram_range 才会真正生效。
        tokenizer = functools.partial(
            tokenize_review,
            min_length=min_token_length,
            use_stopwords=use_stopwords,
        )
        return TfidfVectorizer(
            analyzer="word",
            tokenizer=tokenizer,
            token_pattern=None,
            # 清洗与大小写处理都在 tokenize_review 内部完成，这里关闭默认预处理器，
            # 避免同一文本被处理两遍、也避免 lowercase 改变中文以外的行为预期。
            lowercase=False,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float64,
        )
    if analyzer_mode == "char":
        return TfidfVectorizer(
            analyzer="char",
            preprocessor=char_preprocess,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float64,
        )
    raise ValueError(f"analyzer_mode 必须是 'word' 或 'char'，实际为 {analyzer_mode!r}")


def vectorize(
    vectorizer: TfidfVectorizer,
    train_texts: Sequence[str],
    other_texts: Mapping[str, Sequence[str]] | None = None,
) -> tuple[TfidfVectorizer, spmatrix, dict[str, spmatrix]]:
    """只在训练集上拟合向量器，并变换其余集合。

    把「拟合」与「变换」绑定在同一个函数里，是为了让调用方**无法意外地**
    在验证集或测试集上拟合——那会造成特征泄漏。

    参数：
        vectorizer：尚未拟合的向量器。
        train_texts：训练集原始评论文本。
        other_texts：``{集合名: 文本序列}``，例如 ``{"dev": ..., "test": ...}``。

    返回：
        ``(已拟合的向量器, 训练集特征矩阵, {集合名: 特征矩阵})``。

    异常：
        ValueError：训练集为空时抛出。
    """
    if len(train_texts) == 0:
        raise ValueError("训练集为空，无法拟合向量器")

    train_matrix = vectorizer.fit_transform(list(train_texts))
    matrices: dict[str, spmatrix] = {}
    for name, texts in (other_texts or {}).items():
        matrices[name] = vectorizer.transform(list(texts))
    return vectorizer, train_matrix, matrices


def vectorizer_signature(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    """提取向量器的可复现超参数，供写入 ``metadata.json``。

    参数：
        vectorizer：已拟合或未拟合的向量器。

    返回：
        含关键超参数与词表规模的字典。含 ``vocabulary_size``，
        该值只有在拟合之后才有意义。
    """
    analyzer = vectorizer.analyzer
    tokenizer = getattr(vectorizer, "tokenizer", None)
    if callable(analyzer):
        mode = "word（自定义 analyzer）"
    elif analyzer == "word" and callable(tokenizer):
        mode = "word（自定义 tokenizer）"
    else:
        mode = str(analyzer)
    signature: dict[str, Any] = {
        "analyzer_mode": mode,
        "ngram_range": list(vectorizer.ngram_range),
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
        "max_features": vectorizer.max_features,
        "sublinear_tf": bool(vectorizer.sublinear_tf),
        "vocabulary_size": len(vectorizer.vocabulary_) if vectorizer.vocabulary_ else 0,
    }
    return signature


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    """把向量器序列化到磁盘。

    文件名固定为 ``vectorizer.joblib``（契约 §2.2），由调用方传入完整路径。
    序列化前会做一次自检：未拟合的向量器没有 ``vocabulary_``，保存它等于交付一个
    不可用的产物，因此这里直接报错而不是静默写出。

    参数：
        vectorizer：已拟合的向量器。
        path：输出路径。

    异常：
        ValueError：向量器尚未拟合时抛出。
    """
    if not getattr(vectorizer, "vocabulary_", None):
        raise ValueError("向量器尚未拟合（vocabulary_ 为空），拒绝保存不可用的产物")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path) -> TfidfVectorizer:
    """从磁盘加载向量器。

    参数：
        path：``vectorizer.joblib`` 的路径。

    返回：
        已拟合的向量器。

    异常：
        FileNotFoundError：文件不存在时抛出。
        TypeError：文件内容不是 TfidfVectorizer 时抛出，避免下游拿到错误类型
            而出现难以定位的报错。
    """
    if not path.is_file():
        raise FileNotFoundError(f"向量器文件不存在：{public_path(path)}")
    vectorizer = joblib.load(path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(
            f"{path} 的内容是 {type(vectorizer).__name__}，不是 TfidfVectorizer"
        )
    return vectorizer


def copy_label_mapping(source: Path, destination: Path) -> str:
    """把数据契约的 ``label_mapping.json`` 逐字节复制到模型目录。

    契约 §2.1 要求模型目录下的 ``label_mapping.json`` 与 ``data/processed/`` 中的那份
    **逐字节相同**，因此这里用字节复制而不是「读出来再写回去」——
    后者会因键序或缩进格式差异而产生不相同的文件。

    参数：
        source：``data/processed/label_mapping.json``。
        destination：模型目录下的目标路径。

    返回：
        复制后文件的 SHA-256，供报告留证。

    异常：
        FileNotFoundError：源文件不存在时抛出。
        ValueError：源文件不是合法 JSON 对象时抛出。
    """
    if not source.is_file():
        raise FileNotFoundError(f"缺少数据契约的标签映射：{public_path(source)}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "categories" not in payload:
        raise ValueError(f"{source} 不是合法的标签映射文件")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

    import hashlib

    digest = hashlib.sha256()
    with destination.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(split_path: Path, *, text_column: str, category_column: str) -> pd.DataFrame:
    """读取一个划分文件并校验必需列。

    参数：
        split_path：``data/processed/{train,dev,test}.csv``。
        text_column：评论文本列名。
        category_column：类别列名。

    返回：
        该划分的 DataFrame。

    异常：
        FileNotFoundError：文件不存在时抛出。
        KeyError：缺少必需列时抛出。
    """
    if not split_path.is_file():
        raise FileNotFoundError(
            f"缺少划分文件：{split_path}。请先运行 'python scripts/split_data.py'。"
        )
    frame = pd.read_csv(split_path, encoding="utf-8")
    missing = [c for c in (text_column, category_column) if c not in frame.columns]
    if missing:
        raise KeyError(f"{split_path} 缺少必需列：{missing}")
    return frame
