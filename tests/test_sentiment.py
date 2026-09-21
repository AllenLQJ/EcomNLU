"""M5 情感二分类基线的测试：标签空间、估计器、契约产物与统一推理接口。

本文件覆盖从「数据集自带的数字标签列 -> 中文情感名 -> TF-IDF -> 训练 ->
契约产物 -> 统一接口推理」的关键规则，是「情感任务可复现且符合接口契约」
这一交付要求的自动化证据。

Usage:
    python -m unittest discover -s tests

输入文件：
    - 单元测试：不读取磁盘文件，在内存里构造小型合成数据与临时目录；
    - 集成测试：读取 ``models/sentiment_*`` 下真实训练产出的模型目录，
      不存在时自动跳过（未训练时不应该让测试变红）。

关键设计理由：
    1. **不依赖先跑完整训练。** 单元测试用极小样本，提交前即可快速自检；
    2. **标签对应关系必须被锁死。** 数据集自带的 ``label`` 列里 ``0`` 是差评、
       ``1`` 是好评，这条对应关系一旦颠倒，正负情感会整体反掉且不会抛异常。
       测试用已知极性的文本断言方向正确，而不是只断言「跑通了」；
    3. **标签顺序的唯一定义处是 ``configs/base.yaml``。**
       ``scripts/verify_model_contract.py`` 与训练脚本都从那里读，
       测试断言配置的取值与顺序，任何漂移都会立刻变红（契约 v1.2 草案 §2.6）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.features.text_features import build_tfidf_vectorizer, vectorize  # noqa: E402
from src.inference import PredictorError, load_predictor  # noqa: E402
from src.models.baseline.artifacts import METADATA_REQUIRED_KEYS  # noqa: E402
from src.models.sentiment.artifacts import (  # noqa: E402
    SENTIMENT_METADATA_REQUIRED_KEYS,
    SENTIMENT_TASK,
    build_per_product_category_metrics,
    build_sentiment_label_mapping,
    build_sentiment_metadata,
    build_sentiment_metrics,
    write_sentiment_artifact,
    write_sentiment_model_body,
)
from src.models.sentiment.estimators import (  # noqa: E402
    SENTIMENT_MODEL_NAMES,
    build_sentiment_estimator,
    select_best_per_model,
    select_overall,
    train_sentiment_candidate,
    translate_labels,
)


# 情感标签顺序从配置读取，不在测试里另写一份——测试要能抓到配置漂移。
LABELS = [str(value) for value in load_config()["task"]["sentiment_categories"]]
NEGATIVE_NAME, POSITIVE_NAME = LABELS

# 合成语料：正负样本用互不重叠的词汇，保证小样本下即可线性可分，
# 便于断言「预测方向正确」而不是断言一个不稳定的准确率数字。
POSITIVE_TEXTS = [
    "质量很好非常满意",
    "用起来很舒服值得购买",
    "效果不错会再次购买",
    "快递很快服务态度好",
]
NEGATIVE_TEXTS = [
    "质量很差非常失望",
    "用起来很难受退货了",
    "效果不好不会再买",
    "发货很慢客服态度差",
]


def build_tiny_dataset(*, repeats: int = 12):
    """构造一个极小的可分词中文情感数据集。

    Args:
        repeats: 每个模板重复的次数，用于让向量器的 ``min_df=2`` 有统计基础。

    Returns:
        ``(texts, labels)``，labels 为中文情感类名。
    """
    texts: list[str] = []
    labels: list[str] = []
    for _ in range(repeats):
        texts.extend(POSITIVE_TEXTS)
        labels.extend([POSITIVE_NAME] * len(POSITIVE_TEXTS))
        texts.extend(NEGATIVE_TEXTS)
        labels.extend([NEGATIVE_NAME] * len(NEGATIVE_TEXTS))
    return texts, labels


class SentimentLabelTests(unittest.TestCase):
    """情感标签顺序与其唯一来源。"""

    def test_config_defines_two_labels_negative_first(self) -> None:
        """配置必须定义恰好 2 个标签，且负向在前（对应原始 label 值 0）。"""
        self.assertEqual(len(LABELS), 2)
        self.assertEqual(LABELS[0], "负向")
        self.assertEqual(LABELS[1], "正向")

    def test_validator_reads_the_same_config(self) -> None:
        """契约校验器必须从同一份配置读取情感标签，不得另存副本。"""
        from scripts.verify_model_contract import ModelContractValidator

        validator = ModelContractValidator(PROJECT_ROOT / "models" / "nonexistent", PROJECT_ROOT / "data" / "processed")
        self.assertEqual(validator.load_sentiment_categories(), LABELS)

    def test_label_mapping_structure(self) -> None:
        """标签映射的结构必须与契约 §1.3 同形，且 id 与顺序一一对应。"""
        mapping = build_sentiment_label_mapping(LABELS)
        self.assertEqual(mapping["categories"], LABELS)
        self.assertEqual(mapping["label2id"], {NEGATIVE_NAME: 0, POSITIVE_NAME: 1})
        self.assertEqual(mapping["id2label"], {"0": NEGATIVE_NAME, "1": POSITIVE_NAME})

    def test_label_mapping_rejects_wrong_count(self) -> None:
        """标签数不为 2 必须报错，避免写出含义不明的映射。"""
        with self.assertRaises(ValueError):
            build_sentiment_label_mapping(["正向"])


class TranslateLabelTests(unittest.TestCase):
    """原始 label 列 0/1 -> 中文情感类名。"""

    def test_zero_is_negative_one_is_positive(self) -> None:
        """0 是差评（负向）、1 是好评（正向），顺序不得颠倒。"""
        self.assertEqual(
            translate_labels([0, 1, 0, 1], LABELS),
            [NEGATIVE_NAME, POSITIVE_NAME, NEGATIVE_NAME, POSITIVE_NAME],
        )

    def test_accepts_string_digits(self) -> None:
        """CSV 读出的标签若被当成字符串也应能处理。"""
        self.assertEqual(
            translate_labels(["1", "0"], LABELS), [POSITIVE_NAME, NEGATIVE_NAME]
        )

    def test_rejects_out_of_range(self) -> None:
        """出现 0/1 之外的值必须报错，不得静默丢弃样本。"""
        with self.assertRaises(ValueError):
            translate_labels([0, 2], LABELS)

    def test_rejects_non_numeric(self) -> None:
        """非数值标签必须报错。"""
        with self.assertRaises(ValueError):
            translate_labels(["好评"], LABELS)

    def test_rejects_wrong_label_count(self) -> None:
        """标签列表长度必须为 2。"""
        with self.assertRaises(ValueError):
            translate_labels([0, 1], ["仅一个"])


class SentimentEstimatorTests(unittest.TestCase):
    """候选模型构造、训练与验证集选模。"""

    def test_build_estimators(self) -> None:
        """两个候选模型按名字构造出正确的类型。"""
        logreg = build_sentiment_estimator(
            "sentiment_tfidf_logreg", {"C": 1.0}, random_seed=42
        )
        self.assertIsInstance(logreg, LogisticRegression)
        self.assertTrue(hasattr(logreg, "predict_proba"))

        svm = build_sentiment_estimator(
            "sentiment_tfidf_svm", {"C": 1.0}, random_seed=42
        )
        self.assertIsInstance(svm, CalibratedClassifierCV)
        self.assertTrue(hasattr(svm, "predict_proba"))

    def test_unknown_model_name_raises(self) -> None:
        """未知模型名必须报错，避免写出含义不明的产物。"""
        with self.assertRaises(ValueError):
            build_sentiment_estimator("sentiment_bert", {}, random_seed=42)

    def test_model_names_are_two(self) -> None:
        """M5 任务规定两个候选：逻辑回归与校准后的线性 SVM。"""
        self.assertEqual(
            set(SENTIMENT_MODEL_NAMES),
            {"sentiment_tfidf_logreg", "sentiment_tfidf_svm"},
        )

    def test_train_and_select_by_dev_macro_f1(self) -> None:
        """训练后在验证集上评估，并按 Macro-F1 选出唯一最优配置。"""
        texts, labels = build_tiny_dataset()
        vectorizer = build_tfidf_vectorizer(
            analyzer_mode="char", ngram_range=(1, 2), min_df=1
        )
        vectorizer, matrix, others = vectorize(vectorizer, texts, {"dev": texts})

        candidates = [
            train_sentiment_candidate(
                "sentiment_tfidf_logreg",
                {"C": C},
                train_matrix=matrix,
                train_labels=labels,
                dev_matrix=others["dev"],
                dev_labels=labels,
                labels=LABELS,
                random_seed=42,
            )
            for C in (0.5, 2.0)
        ]
        for candidate in candidates:
            # 指标块必须覆盖两个情感类，混淆矩阵必须是 2×2
            self.assertEqual(set(candidate.dev_metrics["per_class"]), set(LABELS))
            matrix_block = candidate.dev_metrics["confusion_matrix"]
            self.assertEqual(len(matrix_block), 2)
            self.assertTrue(all(len(row) == 2 for row in matrix_block))

        best = select_best_per_model(candidates)
        self.assertEqual(set(best), {"sentiment_tfidf_logreg"})
        overall = select_overall(best)
        self.assertIs(overall, best["sentiment_tfidf_logreg"])
        self.assertGreaterEqual(
            overall.dev_macro_f1, min(c.dev_macro_f1 for c in candidates)
        )

    def test_select_best_per_model_keeps_higher_score(self) -> None:
        """同一模型名下多个配置只保留验证集 Macro-F1 最高的一个。"""
        texts, labels = build_tiny_dataset()
        vectorizer = build_tfidf_vectorizer(
            analyzer_mode="char", ngram_range=(1, 2), min_df=1
        )
        vectorizer, matrix, others = vectorize(vectorizer, texts, {"dev": texts})
        candidates = [
            train_sentiment_candidate(
                "sentiment_tfidf_logreg",
                {"C": C},
                train_matrix=matrix,
                train_labels=labels,
                dev_matrix=others["dev"],
                dev_labels=labels,
                labels=LABELS,
                random_seed=42,
            )
            for C in (1.0, 5.0)
        ]
        best = select_best_per_model(candidates)
        self.assertEqual(
            best["sentiment_tfidf_logreg"].dev_macro_f1,
            max(c.dev_macro_f1 for c in candidates),
        )


class SentimentArtifactTests(unittest.TestCase):
    """契约产物的结构、校验与统一推理接口。"""

    def _build_model_dir(self, root: Path) -> tuple[Path, Path]:
        """在临时目录里训练并写出一个完整的情感模型产物。

        Returns:
            ``(模型目录, processed 目录)``。
        """
        texts, labels = build_tiny_dataset()
        vectorizer = build_tfidf_vectorizer(
            analyzer_mode="char", ngram_range=(1, 2), min_df=1
        )
        vectorizer, matrix, others = vectorize(vectorizer, texts, {"dev": texts})
        candidate = train_sentiment_candidate(
            "sentiment_tfidf_logreg",
            {"C": 1.0},
            train_matrix=matrix,
            train_labels=labels,
            dev_matrix=others["dev"],
            dev_labels=labels,
            labels=LABELS,
            random_seed=42,
        )

        # 指纹计算需要三个划分 CSV 存在
        processed = root / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            (processed / f"{split}.csv").write_text(
                f"sample_id,review,cat,label\n0,{split},类0,1\n", encoding="utf-8"
            )

        model_dir = write_sentiment_artifact(
            models_dir=root / "models",
            model_name="sentiment_tfidf_logreg",
            estimator=candidate.estimator,
            vectorizer=vectorizer,
            labels=LABELS,
            metadata=build_sentiment_metadata(
                model_name="sentiment_tfidf_logreg",
                created_at="2026-09-21T20:00:00+08:00",
                random_seed=42,
                processed_dir=processed,
                train_rows=len(texts),
                hyperparameters={"C": 1.0},
            ),
            metrics=build_sentiment_metrics(
                model_name="sentiment_tfidf_logreg",
                labels=LABELS,
                dev_block=candidate.dev_metrics,
                test_block=None,
                cost={
                    "model_size_bytes": 1,
                    "single_sample_latency_ms": 1.0,
                    "batch_32_latency_ms": 1.0,
                    "throughput_per_sec": 1.0,
                    "measured_on": "cpu",
                    "measured_at": "2026-09-21T20:00:00+08:00",
                },
            ),
        )
        return model_dir, processed

    def test_artifact_contains_required_files(self) -> None:
        """契约四件套必须齐全：模型、向量器、标签映射、元数据、指标。"""
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, _ = self._build_model_dir(Path(temporary))
            for filename in (
                "model.joblib",
                "vectorizer.joblib",
                "label_mapping.json",
                "metadata.json",
                "metrics.json",
            ):
                self.assertTrue((model_dir / filename).is_file(), f"缺少 {filename}")

    def test_label_mapping_is_sentiment_space(self) -> None:
        """情感模型的 label_mapping.json 必须是情感标签，不得复制商品映射。

        契约 v1.2 草案 §2.2：情感任务的标签集与 10 个商品类必然不同，
        套用「与数据契约逐字节相同」会让任何情感模型在校验第一步就失败。
        """
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, _ = self._build_model_dir(Path(temporary))
            mapping = json.loads(
                (model_dir / "label_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping["categories"], LABELS)
            self.assertEqual(set(mapping), {"categories", "label2id", "id2label"})

    def test_metadata_records_sentiment_task(self) -> None:
        """metadata 必须标识情感任务，且字段齐全、max_length 为 null。"""
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, _ = self._build_model_dir(Path(temporary))
            metadata = json.loads(
                (model_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["task"], SENTIMENT_TASK)
            self.assertIsNone(metadata["max_length"])
            self.assertEqual(metadata["framework"], "scikit-learn")
            for key in SENTIMENT_METADATA_REQUIRED_KEYS:
                self.assertIn(key, metadata)
            self.assertEqual(SENTIMENT_METADATA_REQUIRED_KEYS, METADATA_REQUIRED_KEYS)

    def test_metadata_fingerprint_matches_recomputation(self) -> None:
        """数据集指纹必须能由三个划分 CSV 重新算出（契约 §2.3）。"""
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, processed = self._build_model_dir(Path(temporary))
            metadata = json.loads(
                (model_dir / "metadata.json").read_text(encoding="utf-8")
            )
            from src.models.baseline.artifacts import dataset_fingerprint

            self.assertEqual(
                metadata["dataset_fingerprint"], dataset_fingerprint(processed)
            )

    def test_metrics_uses_sentiment_label_space(self) -> None:
        """metrics.categories 必须是情感标签，混淆矩阵为 2×2。"""
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, _ = self._build_model_dir(Path(temporary))
            metrics = json.loads(
                (model_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["categories"], LABELS)
            self.assertIsNone(metrics["test"], "选模阶段 test 必须为 null")
            self.assertEqual(len(metrics["dev"]["confusion_matrix"]), 2)

    def test_write_body_rejects_model_without_predict_proba(self) -> None:
        """缺 predict_proba 的模型必须拒绝写出，而不是交付不可用产物。"""
        from sklearn.svm import LinearSVC

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts, labels = build_tiny_dataset()
            vectorizer = build_tfidf_vectorizer(
                analyzer_mode="char", ngram_range=(1, 2), min_df=1
            )
            vectorizer, matrix, _ = vectorize(vectorizer, texts)
            estimator = LinearSVC(dual=True, random_state=42).fit(matrix, labels)
            with self.assertRaises(ValueError):
                write_sentiment_model_body(
                    models_dir=root / "models",
                    model_name="sentiment_tfidf_svm",
                    estimator=estimator,
                    vectorizer=vectorizer,
                    labels=LABELS,
                    metadata={},
                )

    def test_unified_entry_returns_sentiment_and_confidence(self) -> None:
        """统一入口必须返回正负情感及置信度，且概率和为 1。

        这是契约 v1.2 草案 §2.5 的关键点：**不新增 SentimentPredictor 协议**，
        情感模型与类别模型共用 ``load_predictor``，标签集由模型目录自己决定。
        """
        with tempfile.TemporaryDirectory() as temporary:
            model_dir, _ = self._build_model_dir(Path(temporary))
            predictor = load_predictor(model_dir)

            probas = predictor.predict_proba(POSITIVE_TEXTS[:2] + NEGATIVE_TEXTS[:2])
            self.assertEqual(len(probas), 4)
            for item in probas:
                self.assertEqual(set(item), set(LABELS))
                self.assertAlmostEqual(sum(item.values()), 1.0, places=6)

            # 预测方向必须正确：概率列顺序若被写反，这一条会立刻失败。
            # sklearn 按 classes_ 排序排概率列，中文排序后「正向」在「负向」之前，
            # 因此适配器必须按 classes_ 解析列位置，不能假定「第 0 列是负向」。
            for probabilities in probas[:2]:
                self.assertGreater(
                    probabilities[POSITIVE_NAME], probabilities[NEGATIVE_NAME]
                )
            for probabilities in probas[2:]:
                self.assertGreater(
                    probabilities[NEGATIVE_NAME], probabilities[POSITIVE_NAME]
                )

            labels_out = predictor.predict(POSITIVE_TEXTS[:1] + NEGATIVE_TEXTS[:1])
            self.assertEqual(labels_out, [POSITIVE_NAME, NEGATIVE_NAME])

            # API 契约 §3.4 的 {"label": …, "confidence": …} 即取最高概率项
            probabilities = probas[0]
            best = max(probabilities, key=lambda name: probabilities[name])
            self.assertEqual(best, POSITIVE_NAME)
            self.assertGreater(probabilities[best], 0.5)

    def test_unified_entry_rejects_missing_directory(self) -> None:
        """目录不存在时报面向使用者的错误。"""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PredictorError):
                load_predictor(Path(temporary) / "nope")


class PerProductCategoryTests(unittest.TestCase):
    """各商品类别内的情感表现统计。"""

    def test_groups_and_support(self) -> None:
        """按商品类别分组，support 之和等于总样本数。"""
        import pandas as pd

        frame = pd.DataFrame(
            {"review": ["a", "b", "c", "d"], "cat": ["手机", "手机", "书籍", "书籍"]}
        )
        block = build_per_product_category_metrics(
            frame,
            true_labels=[POSITIVE_NAME, NEGATIVE_NAME, POSITIVE_NAME, POSITIVE_NAME],
            predicted_labels=[
                POSITIVE_NAME,
                POSITIVE_NAME,
                POSITIVE_NAME,
                POSITIVE_NAME,
            ],
            labels=LABELS,
            category_column="cat",
        )
        self.assertEqual(set(block), {"手机", "书籍"})
        self.assertEqual(sum(item["support"] for item in block.values()), 4)
        self.assertEqual(block["手机"]["support"], 2)
        self.assertAlmostEqual(block["手机"]["accuracy"], 0.5)
        self.assertAlmostEqual(block["书籍"]["positive_rate"], 1.0)
        self.assertEqual(set(block["手机"]["per_class"]), set(LABELS))

    def test_length_mismatch_raises(self) -> None:
        """长度不一致必须报错，而不是错位统计。"""
        import pandas as pd

        frame = pd.DataFrame({"review": ["a", "b"], "cat": ["手机", "书籍"]})
        with self.assertRaises(ValueError):
            build_per_product_category_metrics(
                frame,
                true_labels=[POSITIVE_NAME],
                predicted_labels=[POSITIVE_NAME, POSITIVE_NAME],
                labels=LABELS,
                category_column="cat",
            )

    def test_missing_category_column_raises(self) -> None:
        """缺少商品类别列必须报错。"""
        import pandas as pd

        frame = pd.DataFrame({"review": ["a", "b"]})
        with self.assertRaises(KeyError):
            build_per_product_category_metrics(
                frame,
                true_labels=[POSITIVE_NAME, NEGATIVE_NAME],
                predicted_labels=[POSITIVE_NAME, NEGATIVE_NAME],
                labels=LABELS,
                category_column="cat",
            )


class MetricsValidationTests(unittest.TestCase):
    """metrics 结构校验。"""

    def _dev_block(self) -> dict:
        return {
            "accuracy": 0.9,
            "macro_precision": 0.9,
            "macro_recall": 0.9,
            "macro_f1": 0.9,
            "weighted_f1": 0.9,
            "per_class": {
                NEGATIVE_NAME: {
                    "precision": 0.9,
                    "recall": 0.9,
                    "f1": 0.9,
                    "support": 10,
                },
                POSITIVE_NAME: {
                    "precision": 0.9,
                    "recall": 0.9,
                    "f1": 0.9,
                    "support": 10,
                },
            },
            "confusion_matrix": [[9, 1], [1, 9]],
        }

    def _cost(self) -> dict:
        return {
            "model_size_bytes": 1,
            "single_sample_latency_ms": 1.0,
            "batch_32_latency_ms": 1.0,
            "throughput_per_sec": 1.0,
            "measured_on": "cpu",
            "measured_at": "2026-09-21T20:00:00+08:00",
        }

    def test_accepts_valid_block(self) -> None:
        """合法指标块应通过并带出 per_product_category 字段。"""
        payload = build_sentiment_metrics(
            model_name="sentiment_tfidf_logreg",
            labels=LABELS,
            dev_block=self._dev_block(),
            test_block=None,
            cost=self._cost(),
            per_product_category={"手机": self._dev_block()},
        )
        self.assertEqual(payload["categories"], LABELS)
        self.assertIn("per_product_category", payload)

    def test_rejects_wrong_matrix_size(self) -> None:
        """混淆矩阵尺寸不符必须报错。"""
        block = self._dev_block()
        block["confusion_matrix"] = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            build_sentiment_metrics(
                model_name="sentiment_tfidf_logreg",
                labels=LABELS,
                dev_block=block,
                test_block=None,
                cost=self._cost(),
            )

    def test_rejects_wrong_per_class_keys(self) -> None:
        """per_class 的键必须是情感标签。"""
        block = self._dev_block()
        block["per_class"] = {"好评": block["per_class"][POSITIVE_NAME]}
        with self.assertRaises(ValueError):
            build_sentiment_metrics(
                model_name="sentiment_tfidf_logreg",
                labels=LABELS,
                dev_block=block,
                test_block=None,
                cost=self._cost(),
            )

    def test_rejects_missing_cost_field(self) -> None:
        """cost 缺字段必须报错。"""
        cost = self._cost()
        cost.pop("throughput_per_sec")
        with self.assertRaises(ValueError):
            build_sentiment_metrics(
                model_name="sentiment_tfidf_logreg",
                labels=LABELS,
                dev_block=self._dev_block(),
                test_block=None,
                cost=cost,
            )


class ErrorTagTests(unittest.TestCase):
    """误分类现象分组（错误分析用）。"""

    def test_tag_priority_and_categories(self) -> None:
        """分组应落在 ERROR_TAGS 内，且「否定与转折并存」优先级高于「含否定词」。"""
        from scripts.train_sentiment import ERROR_TAGS, classify_error_tag, SHORT_TEXT_MAX_CHARS

        self.assertEqual(
            classify_error_tag("质量不错，但是用不了多久就坏了"), "否定与转折并存"
        )
        self.assertEqual(classify_error_tag("不好用"), "含否定词")
        # 阈值仅覆盖极短且无线索词的文本；阈值本身也必须是低值，
        # 否则会误伤「玩游戏发热快」这类带评价对象的短句。
        self.assertLessEqual(SHORT_TEXT_MAX_CHARS, 5)
        self.assertEqual(classify_error_tag("非常满意"), "超短文本（无明确评价对象）")
        self.assertEqual(classify_error_tag("整体感觉一般般"), "其他（无明显线索）")
        for text in ("呵呵，质量真好啊", "不错但是不好用", "不好用", "很好", "还行"):
            self.assertIn(classify_error_tag(text), ERROR_TAGS)


class SentimentPredictScriptTests(unittest.TestCase):
    """``scripts/predict_sentiment_svm.py`` 与共享入口任务分流的守卫测试。

    该脚本是 ``scripts/predict.py`` 的薄封装（与 M3 的 ``predict_svm.py`` 同构）。
    这里锁住四件事：脚本固定的模型目录、它声明的任务、共享入口确实接受
    ``default_task``、以及**任务分流读的是产物自述而不是模型名**。
    最后一条最要紧：情感与类别任务共用同一套预测代码，唯一的差别就是
    标签空间和真实标签列；一旦分流失效，脚本不会报错，而是给出恒为 0 的准确率，
    这种静默错误若到演示现场才发现代价很高。
    """

    def test_wrapper_pins_model_dir_and_task(self) -> None:
        """封装脚本固定的模型目录与任务必须与契约允许的命名对应。"""
        import importlib

        module = importlib.import_module("scripts.predict_sentiment_svm")
        self.assertEqual(module.MODEL_DIR, "models/sentiment_tfidf_svm")
        self.assertEqual(module.TASK, SENTIMENT_TASK)
        self.assertIn("sentiment_tfidf_svm", SENTIMENT_MODEL_NAMES)

    def test_shared_entry_accepts_a_default_task(self) -> None:
        """共享入口必须接受 ``default_task``，否则封装脚本无法固定任务。"""
        import inspect

        from scripts.predict import main

        signature = inspect.signature(main)
        self.assertIn("default_task", signature.parameters)
        self.assertIsNone(signature.parameters["default_task"].default)

    def test_task_resolution_prefers_metadata_over_model_name(self) -> None:
        """任务分流必须读产物自述：显式值优先，否则取 metadata.task。"""
        config = load_config()
        model_dir = PROJECT_ROOT / "models" / "sentiment_tfidf_svm"
        if not (model_dir / "metadata.json").is_file():
            self.skipTest("情感模型尚未训练")

        from scripts.predict import TASK_CATEGORY, resolve_task

        self.assertEqual(resolve_task(config, model_dir, None), SENTIMENT_TASK)
        self.assertEqual(
            resolve_task(config, model_dir, SENTIMENT_TASK), SENTIMENT_TASK
        )
        # 显式值应当能覆盖产物自述（用于临时按另一种口径读同一个模型）
        self.assertEqual(
            resolve_task(config, model_dir, TASK_CATEGORY), TASK_CATEGORY
        )

    def test_label_space_switches_with_task(self) -> None:
        """情感任务取情感标签与 label 列，类别任务取 10 类与 cat 列。"""
        from scripts.predict import TASK_CATEGORY, TASK_SENTIMENT, build_label_space

        config = load_config()
        self.assertEqual(
            build_label_space(config, TASK_SENTIMENT),
            (LABELS, config["data"]["sentiment_column"]),
        )
        labels, column = build_label_space(config, TASK_CATEGORY)
        self.assertEqual(len(labels), 10)
        self.assertEqual(column, config["data"]["category_column"])

    def test_true_labels_are_translated_from_raw_zero_one(self) -> None:
        """真实标签必须从原始 0/1 翻成中文情感名。

        这是本任务最容易静默出错的一步：若拿 ``"0"`` 去比预测出的「负向」，
        准确率会恒为 0 而不抛任何异常，因此用已知极性的序列直接断言方向。
        """
        import pandas as pd

        from scripts.predict import TASK_SENTIMENT, read_truth_labels

        frame = pd.DataFrame({"label": [0, 1, 1, 0]})
        truth = read_truth_labels(
            frame, truth_column="label", task=TASK_SENTIMENT, labels=LABELS
        )
        self.assertEqual(
            truth, [NEGATIVE_NAME, POSITIVE_NAME, POSITIVE_NAME, NEGATIVE_NAME]
        )

    def test_missing_truth_column_raises_a_clear_error(self) -> None:
        """缺少真实标签列时必须报出列名，而不是静默按「无标签」处理。"""
        import pandas as pd

        from scripts.predict import TASK_SENTIMENT, read_truth_labels

        with self.assertRaises(KeyError) as context:
            read_truth_labels(
                pd.DataFrame({"review": ["很好"]}),
                truth_column="label",
                task=TASK_SENTIMENT,
                labels=LABELS,
            )
        self.assertIn("label", str(context.exception))


@unittest.skipUnless(
    (PROJECT_ROOT / "models" / "sentiment_tfidf_svm" / "metadata.json").is_file(),
    "情感模型尚未训练，跳过集成测试",
)
class TrainedSentimentArtifactTests(unittest.TestCase):
    """对真实训练产物做集成校验（未训练时自动跳过）。"""

    def setUp(self) -> None:
        self.model_dir = PROJECT_ROOT / "models" / "sentiment_tfidf_svm"

    def test_real_artifact_passes_structure_checks(self) -> None:
        """真实产物必须字段齐全、标签空间正确、测试集已评估。"""
        metrics = json.loads(
            (self.model_dir / "metrics.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (self.model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["task"], SENTIMENT_TASK)
        self.assertEqual(metrics["categories"], LABELS)
        self.assertIsNotNone(metrics["test"], "最终模型的测试集结果必须已填入")
        self.assertEqual(len(metrics["test"]["confusion_matrix"]), 2)
        self.assertIn("per_product_category", metrics)
        self.assertEqual(len(metrics["per_product_category"]), 10)

    def test_real_artifact_label_mapping_matches_config(self) -> None:
        """真实产物的 label_mapping.json 必须与配置的情感标签一致。"""
        mapping = json.loads(
            (self.model_dir / "label_mapping.json").read_text(encoding="utf-8")
        )
        self.assertEqual(mapping["categories"], LABELS)

    def test_real_artifact_shared_fingerprint_with_m3(self) -> None:
        """情感模型与 M3 类别模型必须使用同一个数据集指纹。"""
        m3_metadata = PROJECT_ROOT / "models" / "tfidf_svm" / "metadata.json"
        if not m3_metadata.is_file():
            self.skipTest("M3 模型产物不存在")
        sentiment_fingerprint = json.loads(
            (self.model_dir / "metadata.json").read_text(encoding="utf-8")
        )["dataset_fingerprint"]
        m3_fingerprint = json.loads(m3_metadata.read_text(encoding="utf-8"))[
            "dataset_fingerprint"
        ]
        self.assertEqual(sentiment_fingerprint, m3_fingerprint)

    def test_real_artifact_serves_via_unified_entry(self) -> None:
        """真实产物必须能通过统一入口加载并返回正负情感及置信度。"""
        predictor = load_predictor(self.model_dir)
        texts = ["质量太差了，用两天就坏了", "非常好用，推荐购买"]
        probabilities = predictor.predict_proba(texts)
        labels_out = predictor.predict(texts)
        self.assertEqual(labels_out, [NEGATIVE_NAME, POSITIVE_NAME])
        for item, label in zip(probabilities, labels_out):
            confidence = item[label]
            self.assertEqual(label, max(item, key=lambda name: item[name]))
            self.assertGreaterEqual(confidence, 0.5)
            self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
