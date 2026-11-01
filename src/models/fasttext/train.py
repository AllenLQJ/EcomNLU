"""FastText 模型的训练、产物落盘与评测。

对应 AGENTS.md §5.3（字符级与词级、默认参数与调参后的模型）与
``docs/contracts.md`` 第二章（四件套、metadata.json、metrics.json）。

设计要点：

* 训练数据只来自 ``data/processed`` 的契约划分，字符级与词级各生成一份训练文件；
* 评测一律通过 ``src.inference`` 的统一入口进行，保证"评测口径 = 线上推理口径"，
  而不是自己再写一套预测逻辑；
* 选模只看验证集 Macro-F1；测试集在模型定稿后才评估一次（契约 §2.4）。
"""

from __future__ import annotations

import importlib.metadata
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from src.data.config import public_path
from src.evaluation.predictor_eval import evaluate_model_dir, summarize
from src.models.artifact import (
    MODELS_DIR,
    PROCESSED_DIR,
    copy_label_mapping,
    read_split,
    write_json,
    write_metadata,
)
from src.models.fasttext.data_io import (
    BuildStats,
    build_fasttext_file,
    read_fasttext_file,
    tokenize_chars,
    tokenize_words,
)

TOKENIZERS: dict[str, Callable[[str], str]] = {
    "chars": tokenize_chars,
    "words": tokenize_words,
}


def fasttext_version() -> str:
    """返回 fasttext 发行版版本号，写入 metadata 的 framework_version。"""
    for distribution in ("fasttext-wheel", "fasttext"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


# fasttext 的属性命名在不同版本间不一致（``wordNgrams`` / ``word_ngrams``），
# 这里按候选名依次探测，避免因为一个属性名就让整条训练链路失败。
_MODEL_ATTRS: dict[str, tuple[str, ...]] = {
    "lr": ("lr",),
    "epoch": ("epoch",),
    "word_ngrams": ("wordNgrams", "word_ngrams"),
    "min_count": ("minCount", "min_count"),
    "bucket": ("bucket",),
    "dim": ("dim",),
    "loss": ("loss",),
    "minn": ("minn",),
    "maxn": ("maxn",),
}


def _json_safe(value: Any) -> Any:
    """把 fasttext 的枚举类型（如 ``loss_name``）转成可写入 JSON 的基本类型。"""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def model_hyperparameters(model: Any) -> dict[str, Any]:
    """读出 fasttext 实际使用的超参数，保证 metadata 可复现训练。"""
    actual: dict[str, Any] = {}
    for key, candidates in _MODEL_ATTRS.items():
        for candidate in candidates:
            if hasattr(model, candidate):
                actual[key] = _json_safe(getattr(model, candidate))
                break
    if "dim" not in actual:
        try:
            actual["dim"] = int(model.get_dimension())
        except Exception:  # 取不到维度不影响训练结果，只是少一个记录字段
            pass
    return actual



@dataclass
class PreparedCorpus:
    """一个切分口径下的训练/验证文件路径与生成统计。"""

    mode: str
    train_path: Path
    dev_path: Path
    stats: dict[str, BuildStats] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)


def prepare_corpus(
    mode: str,
    categories: Sequence[str],
    scratch_dir: Path,
    processed_dir: Path | None = None,
) -> PreparedCorpus:
    """生成指定切分口径（chars / words）的 FastText 训练与验证文件。"""
    if mode not in TOKENIZERS:
        raise ValueError(f"mode 必须是 {sorted(TOKENIZERS)} 之一，实际为 {mode!r}")
    tokenize = TOKENIZERS[mode]
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    corpus = PreparedCorpus(
        mode=mode,
        train_path=scratch_dir / f"train_{mode}.txt",
        dev_path=scratch_dir / f"dev_{mode}.txt",
    )
    for split, path in (("train", corpus.train_path), ("dev", corpus.dev_path)):
        rows = read_split(split, categories, processed_dir)
        corpus.stats[split] = build_fasttext_file(rows, categories, path, tokenize)
        corpus.rows[split] = corpus.stats[split].written
    return corpus


def train_default(input_path: Path) -> Any:
    """默认参数训练（与参考项目 ``_03_快fasttext/_03_...默认参数方式`` 对应）。"""
    import fasttext

    return fasttext.train_supervised(input=str(input_path), verbose=0)


def train_autotune(input_path: Path, valid_path: Path, duration: int) -> Any:
    """自动调参训练，用验证集做搜索（对应参考项目的 ``_04_...自动调参数方式``）。"""
    import fasttext

    return fasttext.train_supervised(
        input=str(input_path),
        autotuneValidationFile=str(valid_path),
        autotuneDuration=int(duration),
        verbose=0,
    )


def materialize_model(
    model: Any,
    model_name: str,
    model_dir: Path,
    mode: str,
    train_rows: int,
    hyperparameters: dict[str, Any],
    notes: str,
    processed_dir: Path | None = None,
) -> Path:
    """把训练好的 FastText 模型落成契约规定的模型目录。

    产出：``model.bin`` + ``label_mapping.json``（逐字节复制）+ ``metadata.json``。
    ``metrics.json`` 由 :func:`evaluate_model_dir` 在真实评测后写出。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.bin"))
    copy_label_mapping(model_dir, processed_dir)
    tokenize_mode = "chars" if mode == "chars" else "jieba_words"
    write_metadata(
        model_dir=model_dir,
        model_name=model_name,
        framework="fasttext",
        framework_version=fasttext_version(),
        train_rows=train_rows,
        hyperparameters={"mode": mode, **hyperparameters},
        device="cpu",
        max_length=None,
        notes=notes,
        # 推理时必须复用训练时的切分方式；统一入口据此对原始文本切分（契约 §2.5）。
        extra={"preprocess": {"tokenize": tokenize_mode}},
        processed_dir=processed_dir,
    )
    return model_dir


def run(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    scratch_dir: Path | None = None,
    autotune_duration: int = 120,
    seed: int = 42,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    """训练全部 FastText 变体、落盘契约产物并评测。

    Returns:
        汇总字典，包含各变体的验证集指标与产物路径，供报告脚本使用。
    """
    import fasttext

    from src.data.config import load_config
    from src.models.artifact import load_categories

    fasttext.FastText.eprint = lambda *args, **kwargs: None  # 关闭 autotune 的噪声输出

    config = load_config()
    categories = load_categories(config)
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    scratch_dir = Path(scratch_dir) if scratch_dir else models_dir / ".fasttext_scratch"

    summary: dict[str, Any] = {"variants": {}, "corpora": {}, "categories": categories}

    variants = (
        ("fasttext_char", "chars", "default"),
        ("fasttext_char_tuned", "chars", "autotune"),
        ("fasttext_word", "words", "default"),
        ("fasttext_word_tuned", "words", "autotune"),
    )

    corpora: dict[str, PreparedCorpus] = {}
    for mode in ("chars", "words"):
        corpora[mode] = prepare_corpus(mode, categories, scratch_dir, processed_dir)
        corpus = corpora[mode]
        summary["corpora"][mode] = {
            "train_file": public_path(corpus.train_path),
            "dev_file": public_path(corpus.dev_path),
            "rows": corpus.rows,
            "stats": {name: stats.as_dict() for name, stats in corpus.stats.items()},
            "sample_lines": read_fasttext_file(corpus.train_path, limit=2),
        }
        print(
            f"[data] {mode}: train={corpus.rows['train']:,} dev={corpus.rows['dev']:,} "
            f"-> {corpus.train_path.name}"
        )

    for model_name, mode, strategy in variants:
        corpus = corpora[mode]
        started = time.perf_counter()
        if strategy == "default":
            model = train_default(corpus.train_path)
            hyperparameters = {"strategy": "default", "autotune_duration": None}
        else:
            model = train_autotune(corpus.train_path, corpus.dev_path, autotune_duration)
            hyperparameters = {
                "strategy": "autotune",
                "autotune_duration": int(autotune_duration),
                "autotune_validation_file": "dev",
            }
        train_seconds = time.perf_counter() - started

        model_dir = materialize_model(
            model=model,
            model_name=model_name,
            model_dir=models_dir / model_name,
            mode=mode,
            train_rows=corpus.rows["train"],
            hyperparameters={
                **hyperparameters,
                **model_hyperparameters(model),
                "training_seconds": round(train_seconds, 1),
            },
            notes=(
                "FastText 监督分类；字符级切分" if mode == "chars" else "FastText 监督分类；jieba 词级切分"
            )
            + (
                "；默认参数"
                if strategy == "default"
                else f"；autotuneValidationFile=dev，autotuneDuration={autotune_duration}s"
            ),
            processed_dir=processed_dir,
        )

        dev_block = evaluate_model_dir(model_dir, categories, "dev", processed_dir)
        print(
            f"[train] {model_name}: {train_seconds:.1f}s "
            f"dev_macro_f1={dev_block['macro_f1']:.4f} dev_acc={dev_block['accuracy']:.4f}"
        )
        summary["variants"][model_name] = {
            "mode": mode,
            "strategy": strategy,
            "train_seconds": round(train_seconds, 1),
            "dev": summarize(dev_block),
            "model_dir": public_path(model_dir),
        }

    if evaluate_test:
        # 选模只依据验证集 Macro-F1（AGENTS.md §6）；测试集在这里对各候选模型
        # 各评估一次，仅用于最终确认与对比表，不参与任何选择。
        for model_name in list(summary["variants"]):
            model_dir = models_dir / model_name
            test_block = evaluate_model_dir(model_dir, categories, "test", processed_dir)
            summary["variants"][model_name]["test"] = summarize(test_block)
            print(
                f"[test] {model_name}: macro_f1={test_block['macro_f1']:.4f} "
                f"acc={test_block['accuracy']:.4f}"
            )

    if scratch_dir.is_dir():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    summary["seed"] = seed
    return summary
