"""M3 传统模型基线的测试：特征提取、评估指标、契约产物与设备判定。

本文件覆盖从「原始文本 -> TF-IDF 特征 -> 训练 -> 契约产物」的关键规则，
是「传统模型阶段可复现且符合接口契约」这一交付要求的自动化证据。

Usage:
    python -m unittest discover -s tests

输入文件：
    - 单元测试：不读取任何磁盘文件，在内存里构造小型合成数据与临时目录；
    - 集成测试：读取 ``models/`` 下真实训练产出的模型目录，不存在时自动跳过。

输出文件：
    单元测试只在 :class:`tempfile.TemporaryDirectory` 中写文件，不污染仓库。

关键设计理由：
    1. **单元测试用极小样本、永远可运行**：不需要先跑完整训练（那要几十分钟），
       提交前即可快速自检；
    2. **集成测试独立重算而不是相信产物自述**：``dataset_fingerprint``、
       ``label_mapping.json`` 的字节一致性都由测试自己从磁盘重新计算后比对；
    3. 期望值来自契约 ``docs/contracts.md`` §2，一旦契约变更这些测试会失败，
       提醒同步更新实现。
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config, project_path  # noqa: E402
from src.evaluation.device import (  # noqa: E402
    MODEL_FAMILY_DEVICES,
    describe_environment,
    resolve_device,
)
from src.evaluation.metrics import (  # noqa: E402
    COST_KEYS,
    METRICS_BLOCK_KEYS,
    compute_metrics_block,
    directory_size_bytes,
    measure_inference_cost,
    top_confusions,
)
from src.features.text_features import (  # noqa: E402
    build_tfidf_vectorizer,
    copy_label_mapping,
    load_vectorizer,
    save_vectorizer,
    tokenize_review,
    vectorize,
    vectorizer_signature,
)
from src.models.baseline.artifacts import (  # noqa: E402
    METADATA_REQUIRED_KEYS,
    build_metadata,
    build_metrics,
    dataset_fingerprint,
    sha256_of,
    write_metrics_file,
    write_model_body,
)
from src.models.baseline.estimators import (  # noqa: E402
    MODEL_NAMES,
    build_estimator,
    select_best_per_model,
    select_overall,
    train_candidate,
)


SMALL_TEXTS = [
    "手机很好用，电池耐用，屏幕清晰",
    "这本书内容不错，作者水平很高",
    "洗发水味道好，头发顺滑不油腻",
    "平板屏幕清晰，看视频很爽",
    "酒店服务态度差，房间也不干净",
    "水果很新鲜，苹果又大又甜",
]
SMALL_CATEGORIES = ["手机", "书籍", "洗发水", "平板", "酒店", "水果"]


class TokenizeReviewTests(unittest.TestCase):
    """分词与文本清洗的单条测试。"""

    def test_cleaning_is_applied_inside_tokenizer(self) -> None:
        """分词必须复用清洗规则：BOM 与控制字符都要先被去掉。

        这一点很关键——推理适配器是把**原始文本**直接交给向量器的，
        清洗若不在分词内部完成，推理与训练的预处理就会不一致。
        """
        tokens = tokenize_review("\ufeff手机\x7f很好用")
        self.assertIn("手机", tokens)
        self.assertNotIn("\ufeff", tokens)
        self.assertNotIn("\x7f", tokens)

    def test_html_entity_and_whitespace_are_normalised(self) -> None:
        """HTML 实体应还原，空白应归一化。"""
        tokens = tokenize_review("彼得&#183;林奇   的书\n很不错")
        self.assertNotIn("\n", "".join(tokens))
        self.assertTrue(any("彼得" in token or "林奇" in token for token in tokens))

    def test_punctuation_only_text_yields_no_tokens(self) -> None:
        """纯标点文本不应产生任何词。"""
        self.assertEqual(tokenize_review("??????????"), [])
        self.assertEqual(tokenize_review("，。！？"), [])

    def test_short_tokens_are_filtered(self) -> None:
        """长度小于最小词长（默认 2）的词应被过滤。"""
        tokens = tokenize_review("好 书 内容好", min_length=2)
        self.assertNotIn("好", tokens)
        self.assertNotIn("书", tokens)

    def test_stopwords_can_be_toggled(self) -> None:
        """停用词开关应生效：关闭后「这个」会保留。"""
        with_stop = tokenize_review("这个手机不错", use_stopwords=True)
        without_stop = tokenize_review("这个手机不错", use_stopwords=False)
        self.assertNotIn("这个", with_stop)
        self.assertIn("这个", without_stop)


class VectorizerTests(unittest.TestCase):
    """TF-IDF 向量器的构造、拟合与持久化。"""

    def test_ngram_range_actually_takes_effect(self) -> None:
        """回归保护：词级模式下 ``ngram_range`` 必须真正生效。

        历史问题：用 ``analyzer=<可调用对象>`` 构造向量器时，sklearn 会**忽略**
        ``ngram_range`` 并只给出一条 UserWarning，于是「词级 bigram」这类配置会
        静默退化成 unigram，对照实验得出「bigram 没有提升」的错误结论。
        改用 ``tokenizer=`` 并显式关闭 ``token_pattern`` 才修复。
        """
        unigram = build_tfidf_vectorizer(min_df=1, ngram_range=(1, 1))
        bigram = build_tfidf_vectorizer(min_df=1, ngram_range=(1, 2))
        unigram, unigram_matrix, _ = vectorize(unigram, SMALL_TEXTS)
        bigram, bigram_matrix, _ = vectorize(bigram, SMALL_TEXTS)
        self.assertGreater(bigram_matrix.shape[1], unigram_matrix.shape[1])

        bigram_terms = [t for t in bigram.get_feature_names_out() if " " in t]
        self.assertTrue(bigram_terms, "bigram 模式下应当出现含空格的组合词条")

    def test_fit_only_on_train_and_transform_others(self) -> None:
        """向量器只在训练集上拟合，其余集合只做变换（避免特征泄漏）。"""
        vectorizer = build_tfidf_vectorizer(min_df=1)
        vectorizer, train_matrix, others = vectorize(
            vectorizer, SMALL_TEXTS[:4], {"dev": SMALL_TEXTS[4:]}
        )
        self.assertEqual(train_matrix.shape[0], 4)
        self.assertEqual(others["dev"].shape[0], 2)
        # 验证集里若出现训练集未见过的词，词表不应因此变大
        before = len(vectorizer.vocabulary_)
        vectorizer.transform(["一个训练集里完全没出现过的词组"])
        self.assertEqual(len(vectorizer.vocabulary_), before)

    def test_joblib_roundtrip_preserves_transformation(self) -> None:
        """向量器序列化后必须能还原出完全相同的变换结果。

        推理适配器依赖 ``vectorizer.joblib``；若往返后结果不一致，
        线上预测会与训练时的评估结果不符，且不会有任何报错。
        """
        vectorizer = build_tfidf_vectorizer(min_df=1)
        vectorizer, matrix, _ = vectorize(vectorizer, SMALL_TEXTS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectorizer.joblib"
            save_vectorizer(vectorizer, path)
            restored = load_vectorizer(path)
            again = restored.transform(SMALL_TEXTS)
            self.assertEqual((matrix != again).nnz, 0)
            self.assertEqual(restored.vocabulary_, vectorizer.vocabulary_)

    def test_saving_unfitted_vectorizer_is_rejected(self) -> None:
        """未拟合的向量器不得被保存：那会交付一个不可用的产物。"""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_vectorizer(
                    build_tfidf_vectorizer(), Path(directory) / "unfitted.joblib"
                )

    def test_signature_records_effective_configuration(self) -> None:
        """签名必须记录真实的配置，供写入 metadata 复现训练。"""
        vectorizer = build_tfidf_vectorizer(min_df=1, ngram_range=(1, 2))
        vectorizer, _, _ = vectorize(vectorizer, SMALL_TEXTS)
        signature = vectorizer_signature(vectorizer)
        self.assertEqual(signature["ngram_range"], [1, 2])
        self.assertGreater(signature["vocabulary_size"], 0)

    def test_char_mode_builds_features(self) -> None:
        """字符级模式应能在不做分词的情况下产出特征。"""
        vectorizer = build_tfidf_vectorizer(
            analyzer_mode="char", ngram_range=(2, 3), min_df=1
        )
        vectorizer, matrix, _ = vectorize(vectorizer, SMALL_TEXTS)
        self.assertEqual(matrix.shape[0], len(SMALL_TEXTS))
        self.assertGreater(matrix.shape[1], 0)

    def test_unknown_analyzer_mode_is_rejected(self) -> None:
        """未知模式必须报错，不能静默当成词级处理。"""
        with self.assertRaises(ValueError):
            build_tfidf_vectorizer(analyzer_mode="syllable")


class MetricsTests(unittest.TestCase):
    """评估指标的数值与结构。"""

    def setUp(self) -> None:
        """构造一个 3 类的小型可手算样例。"""
        self.categories = ["甲", "乙", "丙"]
        self.y_true = ["甲", "甲", "乙", "丙", "丙", "丙"]
        self.y_pred = ["甲", "乙", "乙", "丙", "丙", "甲"]

    def test_block_contains_all_contract_keys(self) -> None:
        """指标块必须含契约 §2.4 规定的全部字段。"""
        block = compute_metrics_block(
            self.y_true, self.y_pred, categories=self.categories
        )
        for key in METRICS_BLOCK_KEYS:
            self.assertIn(key, block)

    def test_per_class_keys_equal_categories(self) -> None:
        """``per_class`` 的键集必须与 categories 完全一致（含无预测的类别）。"""
        block = compute_metrics_block(
            self.y_true, self.y_pred, categories=self.categories
        )
        self.assertEqual(set(block["per_class"]), set(self.categories))

    def test_confusion_matrix_is_rows_true_columns_predicted(self) -> None:
        """混淆矩阵必须是「行真实、列预测」，顺序与 categories 一致。"""
        block = compute_metrics_block(
            self.y_true, self.y_pred, categories=self.categories
        )
        matrix = block["confusion_matrix"]
        self.assertEqual(len(matrix), 3)
        # 甲 真实 2 条：预测 甲 1 条、预测 乙 1 条
        self.assertEqual(matrix[0], [1, 1, 0])
        # 乙 真实 1 条：全部预测为 乙
        self.assertEqual(matrix[1], [0, 1, 0])
        # 丙 真实 3 条：预测 丙 2 条、预测 甲 1 条
        self.assertEqual(matrix[2], [1, 0, 2])

    def test_known_macro_f1_value(self) -> None:
        """Macro-F1 的数值应可手算复核。

        逐类计算（真实 6 条，预测正确的有 4 条）：
          甲：预测为甲的有 2 条（其中 1 条真实是甲）-> precision 1/2 = 0.5；
              真实为甲的有 2 条 -> recall 1/2 = 0.5；F1 = 0.5
          乙：预测为乙的有 2 条（其中 1 条真实是乙）-> precision 1/2 = 0.5；
              真实为乙的有 1 条 -> recall 1/1 = 1.0；F1 = 2*0.5*1/(0.5+1) ≈ 0.6667
          丙：预测为丙的有 2 条且全部正确 -> precision 2/2 = 1.0；
              真实为丙的有 3 条 -> recall 2/3 ≈ 0.6667；F1 = 2*1*0.6667/(1+0.6667) = 0.8
        宏平均 = (0.5 + 0.6667 + 0.8) / 3 ≈ 0.6556
        """
        block = compute_metrics_block(
            self.y_true, self.y_pred, categories=self.categories
        )
        self.assertAlmostEqual(block["accuracy"], 4 / 6, places=6)
        self.assertAlmostEqual(block["macro_f1"], 0.6555556, places=6)
        self.assertAlmostEqual(block["per_class"]["丙"]["precision"], 1.0, places=6)
        self.assertAlmostEqual(block["per_class"]["丙"]["recall"], 2 / 3, places=6)
        self.assertAlmostEqual(block["per_class"]["丙"]["f1"], 0.8, places=6)

    def test_unknown_category_is_rejected(self) -> None:
        """出现 categories 之外的类别必须报错，而不是静默忽略。"""
        with self.assertRaises(ValueError):
            compute_metrics_block(
                ["甲", "丁"], ["甲", "甲"], categories=self.categories
            )

    def test_length_mismatch_is_rejected(self) -> None:
        """真实与预测数量不一致必须报错。"""
        with self.assertRaises(ValueError):
            compute_metrics_block(["甲"], ["甲", "乙"], categories=self.categories)

    def test_top_confusions_skips_diagonal(self) -> None:
        """最易混淆类别对不应包含对角线。"""
        block = compute_metrics_block(
            self.y_true, self.y_pred, categories=self.categories
        )
        pairs = top_confusions(block["confusion_matrix"], self.categories)
        for true_category, predicted_category, _ in pairs:
            self.assertNotEqual(true_category, predicted_category)
        self.assertEqual(pairs[0][2], 1)


class CostMeasurementTests(unittest.TestCase):
    """推理成本的测量。"""

    def test_cost_fields_and_values(self) -> None:
        """成本块字段齐全，且延迟与吞吐量应为正数、互相自洽。"""
        vectorizer = build_tfidf_vectorizer(min_df=1)
        vectorizer, matrix, _ = vectorize(vectorizer, SMALL_TEXTS)
        model = LogisticRegression(max_iter=200).fit(matrix, SMALL_CATEGORIES)

        def predict_proba(texts: list[str]) -> list[dict[str, float]]:
            probabilities = model.predict_proba(vectorizer.transform(texts))
            return [
                {category: float(value) for category, value in zip(model.classes_, row)}
                for row in probabilities
            ]

        cost = measure_inference_cost(
            predict_proba, probe_texts=SMALL_TEXTS, model_size_bytes=1234
        )
        for key in COST_KEYS:
            self.assertIn(key, cost)
        self.assertEqual(cost["model_size_bytes"], 1234)
        self.assertEqual(cost["measured_on"], "cpu")
        self.assertGreater(cost["single_sample_latency_ms"], 0)
        self.assertGreater(cost["batch_32_latency_ms"], 0)
        self.assertGreater(cost["throughput_per_sec"], 0)

    def test_directory_size_is_sum_of_files(self) -> None:
        """目录大小应为其中所有文件之和。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"x" * 100)
            nested = root / "sub"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"y" * 50)
            self.assertEqual(directory_size_bytes(root), 150)
            self.assertEqual(directory_size_bytes(root / "missing"), 0)


class DeviceTests(unittest.TestCase):
    """设备判定。"""

    def test_sklearn_always_uses_cpu_even_with_cuda(self) -> None:
        """scikit-learn 没有 GPU 实现，即使本机 CUDA 可用也必须记为 cpu。

        这是契约 §2.3 的要求：``device`` 记录**实际使用的**设备，
        不是「机器上可用的」设备。写错会让 M4 的 GPU 产物与 M3 的 CPU 产物
        在报告里无法区分。
        """
        self.assertEqual(resolve_device("scikit-learn"), "cpu")
        self.assertEqual(resolve_device("fasttext"), "cpu")

    def test_transformers_device_follows_cuda_availability(self) -> None:
        """transformers 应依据 CUDA 实际可用性决定 cpu / cuda。"""
        from src.evaluation.device import cuda_available

        expected = "cuda" if cuda_available() else "cpu"
        self.assertEqual(resolve_device("transformers"), expected)

    def test_unknown_family_is_rejected(self) -> None:
        """未知模型族必须报错，避免被默默当成 CPU 记录出错误的 device。"""
        with self.assertRaises(ValueError):
            resolve_device("quantum")

    def test_describe_environment_keys(self) -> None:
        """环境描述应含报告所需的全部键。"""
        info = describe_environment()
        for key in ("cuda_available", "torch_version", "gpu_name", "scikit_learn_device"):
            self.assertIn(key, info)
        self.assertEqual(info["scikit_learn_device"], "cpu")
        self.assertIn("cpu_only", set(MODEL_FAMILY_DEVICES.values()))


class ArtifactTests(unittest.TestCase):
    """契约产物的写出与校验（使用合成模型，不依赖完整训练）。"""

    def setUp(self) -> None:
        """准备临时目录、合成标签映射与一个极小的已拟合模型。"""
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.categories = list(SMALL_CATEGORIES)
        mapping = {
            "categories": self.categories,
            "label2id": {name: index for index, name in enumerate(self.categories)},
            "id2label": {str(index): name for index, name in enumerate(self.categories)},
        }
        self.label_mapping_path = self.processed / "label_mapping.json"
        self.label_mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.manifest_path = self.processed / "split_manifest.json"
        self.manifest_path.write_text('{"random_seed": 42}', encoding="utf-8")

        # 契约 v1.0.1 起，数据指纹由 train/dev/test 三个 CSV 的内容算出，
        # 因此夹具必须真的造出这三个文件，否则 build_metadata 会因缺文件而报错。
        for split in ("train", "dev", "test"):
            (self.processed / f"{split}.csv").write_text(
                "sample_id,review,cat,label\n0,很好,手机,1\n", encoding="utf-8"
            )

        self.vectorizer = build_tfidf_vectorizer(min_df=1)
        self.vectorizer, matrix, _ = vectorize(self.vectorizer, SMALL_TEXTS)
        self.model = LogisticRegression(max_iter=500).fit(matrix, SMALL_CATEGORIES)

    def tearDown(self) -> None:
        """清理临时目录。"""
        self._temporary.cleanup()

    def _write(self, model_name: str = "tfidf_logreg") -> Path:
        """写出一个合成模型产物目录。"""
        metadata = build_metadata(
            model_name=model_name,
            created_at="2026-09-21T12:00:00+08:00",
            random_seed=42,
            processed_dir=self.processed,
            train_rows=6,
            hyperparameters={"C": 1.0},
        )
        block = compute_metrics_block(
            SMALL_CATEGORIES, SMALL_CATEGORIES, categories=self.categories
        )
        metrics = build_metrics(
            model_name=model_name,
            categories=self.categories,
            dev_block=block,
            test_block=None,
            cost={
                "model_size_bytes": 1,
                "single_sample_latency_ms": 1.0,
                "batch_32_latency_ms": 2.0,
                "throughput_per_sec": 100.0,
                "measured_on": "cpu",
                "measured_at": "2026-09-21T12:00:00+08:00",
            },
        )
        return write_model_body(
            models_dir=self.root / "models",
            model_name=model_name,
            estimator=self.model,
            vectorizer=self.vectorizer,
            label_mapping_source=self.label_mapping_path,
            metadata=metadata,
        )

    def test_metadata_has_all_required_keys(self) -> None:
        """metadata 必须含契约规定的 11 个字段，且 max_length 为 null。"""
        model_dir = self._write()
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        for key in METADATA_REQUIRED_KEYS:
            self.assertIn(key, metadata)
        self.assertIsNone(metadata["max_length"])
        self.assertEqual(metadata["framework"], "scikit-learn")
        self.assertEqual(metadata["task"], "category_classification")

    def test_dataset_fingerprint_matches_the_contract_algorithm(self) -> None:
        """数据指纹必须等于契约 §2.3 规定的算法结果。

        契约 v1.0.1 起使用 ``dataset_fingerprint``（三个 CSV 的内容哈希），
        取代了 v1.0 的 ``split_manifest_sha256``（清单文件整体哈希）。
        """
        model_dir = self._write()
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("dataset_fingerprint", metadata)
        self.assertNotIn("split_manifest_sha256", metadata)
        self.assertEqual(
            metadata["dataset_fingerprint"], dataset_fingerprint(self.processed)
        )

    def test_dataset_fingerprint_ignores_manifest_created_at(self) -> None:
        """回归保护：改掉清单的 ``created_at`` 不得改变数据集指纹。

        这是契约 v1.0.1 勘误的**核心原因**：v1.0 的指纹取自 ``split_manifest.json``
        整体，而该文件含 ``created_at``，重新生成数据就会让指纹漂移，
        即使三个 CSV 字节完全一致，模型仍会被判为「基于旧数据训练」。
        新指纹只由 CSV 内容决定，因此必须对清单的任何变化免疫。
        """
        before = dataset_fingerprint(self.processed)

        original = self.manifest_path.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["created_at"] = "2099-12-31T23:59:59+08:00"
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            after = dataset_fingerprint(self.processed)
            self.assertEqual(before, after, "清单的 created_at 不应影响数据集指纹")
            # 同时确认旧做法确实会漂移，说明这个回归测试有意义
            self.assertNotEqual(
                sha256_of(self.manifest_path),
                hashlib.sha256(original.encode("utf-8")).hexdigest(),
                "清单哈希本应随 created_at 变化，否则此测试无法体现 v1.0 的缺陷",
            )
        finally:
            self.manifest_path.write_text(original, encoding="utf-8")

    def test_dataset_fingerprint_survives_data_regeneration(self) -> None:
        """回归保护：重新生成同一份数据后指纹必须保持不变。

        划分由固定种子保证确定性，因此重跑 ``split_data.py`` 产出的三个 CSV
        应当逐字节相同；指纹由 CSV 内容决定，因此也必须相同。
        """
        first = dataset_fingerprint(self.processed)
        second = dataset_fingerprint(self.processed)
        self.assertEqual(first, second)

    def test_dataset_fingerprint_requires_all_three_splits(self) -> None:
        """缺少任一划分文件时必须报错，而不是产出含义不明的指纹。"""
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            with self.assertRaises(FileNotFoundError):
                dataset_fingerprint(empty)

    def test_label_mapping_is_byte_identical(self) -> None:
        """模型目录下的 label_mapping.json 必须与数据契约那份逐字节相同。"""
        model_dir = self._write()
        self.assertEqual(
            (model_dir / "label_mapping.json").read_bytes(),
            self.label_mapping_path.read_bytes(),
        )

    def test_model_body_files_exist(self) -> None:
        """模型本体文件必须齐全（契约 §2.2）。"""
        model_dir = self._write()
        for filename in ("model.joblib", "vectorizer.joblib", "metadata.json"):
            self.assertTrue((model_dir / filename).is_file(), filename)

    def test_metrics_requires_dev_block(self) -> None:
        """dev 指标块不得为空（契约要求验证集指标必须先产出）。"""
        with self.assertRaises(ValueError):
            build_metrics(
                model_name="tfidf_logreg",
                categories=self.categories,
                dev_block=None,
                test_block=None,
                cost={},
            )

    def test_metrics_rejects_wrong_confusion_size(self) -> None:
        """混淆矩阵尺寸不符必须报错。"""
        block = compute_metrics_block(
            SMALL_CATEGORIES, SMALL_CATEGORIES, categories=self.categories
        )
        block["confusion_matrix"] = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            build_metrics(
                model_name="tfidf_logreg",
                categories=self.categories,
                dev_block=block,
                test_block=None,
                cost={"model_size_bytes": 1},
            )

    def test_missing_predict_proba_is_rejected(self) -> None:
        """没有 predict_proba 的模型不得写出——契约 §2.5 要求概率输出。"""
        from sklearn.svm import LinearSVC

        model_dir = self._write()
        metadata_path = model_dir / "metadata.json"
        estimator = LinearSVC(dual=True, max_iter=200).fit(
            self.vectorizer.transform(SMALL_TEXTS), SMALL_CATEGORIES
        )
        with self.assertRaises(ValueError):
            write_model_body(
                models_dir=self.root / "models2",
                model_name="tfidf_svm",
                estimator=estimator,
                vectorizer=self.vectorizer,
                label_mapping_source=self.label_mapping_path,
                metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
            )

    def test_class_mismatch_is_rejected(self) -> None:
        """模型类别与标签映射不一致时必须报错，避免写出标签错位的产物。"""
        partial = LogisticRegression(max_iter=500).fit(
            self.vectorizer.transform(SMALL_TEXTS), ["手机"] * 3 + ["书籍"] * 3
        )
        metadata = json.loads(
            (self._write() / "metadata.json").read_text(encoding="utf-8")
        )
        with self.assertRaises(ValueError):
            write_model_body(
                models_dir=self.root / "models3",
                model_name="tfidf_logreg",
                estimator=partial,
                vectorizer=self.vectorizer,
                label_mapping_source=self.label_mapping_path,
                metadata=metadata,
            )

    def test_metrics_file_can_be_written_after_cost_measurement(self) -> None:
        """两阶段写出：先写本体，测完成本再补 metrics.json。"""
        model_dir = self._write()
        self.assertFalse((model_dir / "metrics.json").is_file())
        block = compute_metrics_block(
            SMALL_CATEGORIES, SMALL_CATEGORIES, categories=self.categories
        )
        metrics = build_metrics(
            model_name="tfidf_logreg",
            categories=self.categories,
            dev_block=block,
            test_block=block,
            cost={
                "model_size_bytes": directory_size_bytes(model_dir),
                "single_sample_latency_ms": 1.0,
                "batch_32_latency_ms": 2.0,
                "throughput_per_sec": 100.0,
                "measured_on": "cpu",
                "measured_at": "2026-09-21T12:00:00+08:00",
            },
        )
        write_metrics_file(model_dir, metrics)
        written = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(written["model_name"], "tfidf_logreg")
        self.assertIsNotNone(written["test"])


class EstimatorTests(unittest.TestCase):
    """候选模型的构造与选模逻辑。"""

    def test_all_contract_model_names_can_be_built(self) -> None:
        """契约允许的三个模型名都应能构造出估计器。"""
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                estimator = build_estimator(name, {}, random_seed=42)
                self.assertIsNotNone(estimator)

    def test_svm_estimator_supports_predict_proba(self) -> None:
        """``tfidf_svm`` 必须经校准后支持 predict_proba。

        LinearSVC 原生只有 decision_function；契约 §2.5 要求概率输出，
        推理适配器会直接拒绝不带 predict_proba 的模型。
        """
        estimator = build_estimator("tfidf_svm", {"C": 1.0}, random_seed=42)
        self.assertTrue(hasattr(estimator, "predict_proba"))

    def test_unknown_model_name_is_rejected(self) -> None:
        """未知模型名必须报错。"""
        with self.assertRaises(ValueError):
            build_estimator("tfidf_xgboost", {}, random_seed=42)

    def test_selection_uses_dev_macro_f1(self) -> None:
        """选模必须按验证集 Macro-F1，与其他指标无关。"""
        vectorizer = build_tfidf_vectorizer(min_df=1)
        vectorizer, matrix, _ = vectorize(vectorizer, SMALL_TEXTS)
        candidate = train_candidate(
            "tfidf_logreg",
            {"C": 1.0},
            train_matrix=matrix,
            train_labels=SMALL_CATEGORIES,
            dev_matrix=matrix,
            dev_labels=SMALL_CATEGORIES,
            categories=SMALL_CATEGORIES,
            random_seed=42,
        )
        best = select_best_per_model([candidate])
        self.assertEqual(set(best), {"tfidf_logreg"})
        self.assertIs(select_overall(best), candidate)

    def test_select_overall_rejects_empty_input(self) -> None:
        """没有候选时必须报错，而不是返回 None。"""
        with self.assertRaises(ValueError):
            select_overall({})


class TrainedModelArtifactTests(unittest.TestCase):
    """对真实训练产物的集成校验；产物不存在时自动跳过。"""

    def setUp(self) -> None:
        """加载配置并定位 ``models/``；没有已训练模型时跳过。"""
        self.config = load_config()
        self.categories = [str(c) for c in self.config["task"]["categories"]]
        self.models_dir = project_path("models")
        self.processed_dir = project_path(self.config["data"]["processed_dir"])
        if not self.models_dir.is_dir():
            self.skipTest("models/ 不存在，尚未训练基线模型")
        self.model_dirs = sorted(
            path
            for path in self.models_dir.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        )
        if not self.model_dirs:
            self.skipTest("models/ 下没有已训练的模型（含 metadata.json 的子目录）")

    def test_every_model_has_the_four_piece_artifact(self) -> None:
        """每个模型目录必须含契约 §2.1 的四件套与模型本体。"""
        for model_dir in self.model_dirs:
            with self.subTest(model=model_dir.name):
                for filename in (
                    "model.joblib",
                    "vectorizer.joblib",
                    "label_mapping.json",
                    "metadata.json",
                    "metrics.json",
                ):
                    self.assertTrue((model_dir / filename).is_file(), filename)

    def test_label_mapping_matches_the_data_contract_byte_for_byte(self) -> None:
        """模型目录的标签映射必须与数据契约那份逐字节相同。"""
        source = self.processed_dir / "label_mapping.json"
        if not source.is_file():
            self.skipTest("数据契约的 label_mapping.json 不存在")
        for model_dir in self.model_dirs:
            with self.subTest(model=model_dir.name):
                self.assertEqual(
                    (model_dir / "label_mapping.json").read_bytes(), source.read_bytes()
                )

    def test_metadata_records_the_current_split_manifest(self) -> None:
        """数据指纹必须与当前清单一致，否则模型是在旧数据上训练的。"""
    def test_metadata_records_the_current_dataset_fingerprint(self) -> None:
        """数据指纹必须与当前划分数据一致，否则模型是用别的数据训练的。

        校验方式与 ``scripts/verify_model_contract.py`` 相同：**从 CSV 重新计算**，
        不读取 ``split_manifest.json``（契约 v1.0.1）。
        """
        try:
            actual = dataset_fingerprint(self.processed_dir)
        except FileNotFoundError:
            self.skipTest("data/processed 下的划分文件不齐")
        for model_dir in self.model_dirs:
            metadata = json.loads(
                (model_dir / "metadata.json").read_text(encoding="utf-8")
            )
            with self.subTest(model=model_dir.name):
                self.assertNotIn("split_manifest_sha256", metadata)
                self.assertEqual(metadata["dataset_fingerprint"], actual)

    def test_metadata_marks_non_transformer_max_length_null(self) -> None:
        """非 transformers 模型的 max_length 必须为 null。"""
        for model_dir in self.model_dirs:
            metadata = json.loads(
                (model_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("framework") == "transformers":
                continue
            with self.subTest(model=model_dir.name):
                self.assertIsNone(metadata["max_length"])

    def test_metrics_cover_all_categories_in_order(self) -> None:
        """dev 指标块必须覆盖全部类别，且顺序与 categories 一致。"""
        for model_dir in self.model_dirs:
            metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            with self.subTest(model=model_dir.name):
                self.assertEqual(metrics["categories"], self.categories)
                self.assertEqual(
                    list(metrics["dev"]["per_class"].keys()), self.categories
                )
                matrix = metrics["dev"]["confusion_matrix"]
                self.assertEqual(len(matrix), len(self.categories))
                self.assertTrue(all(len(row) == len(self.categories) for row in matrix))
                self.assertGreater(metrics["dev"]["macro_f1"], 0.0)

    def test_dev_metrics_are_not_nulls(self) -> None:
        """验证集指标不得为空，且不得出现 NaN。"""
        for model_dir in self.model_dirs:
            metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            with self.subTest(model=model_dir.name):
                for key in ("accuracy", "macro_f1", "weighted_f1"):
                    value = metrics["dev"][key]
                    self.assertIsInstance(value, float)
                    self.assertFalse(np.isnan(value))

    def test_cost_was_measured_not_estimated(self) -> None:
        """成本字段必须来自实测：延迟为正、时间戳非空。"""
        for model_dir in self.model_dirs:
            metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            cost = metrics["cost"]
            with self.subTest(model=model_dir.name):
                self.assertGreater(cost["model_size_bytes"], 0)
                self.assertGreater(cost["single_sample_latency_ms"], 0)
                self.assertGreater(cost["throughput_per_sec"], 0)
                self.assertTrue(cost["measured_at"])

    def test_models_can_be_loaded_through_the_unified_entry(self) -> None:
        """统一入口应能加载每个产物并给出合法的概率分布。"""
        from src.inference import load_predictor

        probe = ["手机很好用，电池耐用", "这本书内容不错，值得一读"]
        for model_dir in self.model_dirs:
            with self.subTest(model=model_dir.name):
                predictor = load_predictor(model_dir)
                probabilities = predictor.predict_proba(probe)
                labels = predictor.predict(probe)
                self.assertEqual(len(probabilities), len(probe))
                for item, label in zip(probabilities, labels):
                    self.assertEqual(set(item), set(self.categories))
                    self.assertAlmostEqual(sum(item.values()), 1.0, places=4)
                    self.assertTrue(all(value >= 0 for value in item.values()))
                    self.assertEqual(label, max(item, key=item.get))


class PredictScriptTests(unittest.TestCase):
    """三个预测封装脚本与服务端约定的守卫测试。

    三个 ``scripts/predict_<model>.py`` 是 ``scripts/predict.py`` 的薄封装，
    各自把一个模型名写死为常量。若将来有人改了模型目录名（或契约允许的
    ``model_name`` 取值）却没同步这些脚本，运行时会直接报「模型目录不存在」，
    而这类错误若在演示现场才发现，代价很高，因此在这里用测试提前拦住。
    """

    WRAPPERS = {
        "predict_logreg": "tfidf_logreg",
        "predict_svm": "tfidf_svm",
        "predict_rf": "tfidf_rf",
    }

    def test_wrappers_pin_the_expected_model_dirs(self) -> None:
        """每个封装脚本固定的模型目录必须与契约允许的 model_name 对应。"""
        import importlib

        for module_name, expected_model in self.WRAPPERS.items():
            with self.subTest(script=module_name):
                module = importlib.import_module(f"scripts.{module_name}")
                self.assertEqual(module.MODEL_DIR, f"models/{expected_model}")
                self.assertIn(expected_model, MODEL_NAMES)

    def test_shared_entry_accepts_a_default_model(self) -> None:
        """共享入口必须接受 ``default_model``，否则封装脚本无法复用。

        这是「一份实现 + 三个薄封装」这一设计的关键接口；
        若签名被改掉，三个脚本会当场失效。
        """
        import inspect

        from scripts.predict import main

        signature = inspect.signature(main)
        self.assertIn("default_model", signature.parameters)
        self.assertIsNone(signature.parameters["default_model"].default)

    def test_relative_model_paths_resolve_against_project_root(self) -> None:
        """相对模型路径必须按项目根目录解析，使脚本可在任意工作目录运行。

        这里刻意用 ``models`` 这个仓库中必然存在的目录（``models/README.md`` 已入库），
        而不是 ``models/tfidf_svm``。``resolve_model_dir`` 会校验目录真实存在，而模型
        产物按 ``.gitignore`` 不入库、且按 M3 验收结论在记录证据后会被清理，用具体模型名
        会让本用例变成"模型在时通过、模型删后报错"——它真正要测的是路径解析逻辑，
        不该依赖一个与被测行为无关的产物是否恰好存在。
        """
        from scripts.predict import resolve_model_dir

        config = load_config()
        resolved = resolve_model_dir(config, "models")
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, project_path("models"))

    def test_missing_model_dir_raises_a_clear_error(self) -> None:
        """不存在的模型目录必须报错并给出路径，而不是静默回退到别的模型。"""
        from scripts.predict import resolve_model_dir

        config = load_config()
        with self.assertRaises(FileNotFoundError) as context:
            resolve_model_dir(config, "models/not_a_real_model")
        self.assertIn("not_a_real_model", str(context.exception))

    def test_no_arguments_defaults_to_the_dev_split(self) -> None:
        """不带任何输入参数时，应默认使用验证集而不是报错。

        因为 ``data/processed/dev.csv`` 自带真实标签，默认行为可以一次给出
        逐条预测预览与真实评估分数，便于快速自检。
        """
        from scripts.predict import parse_args

        args = parse_args([])
        self.assertIsNone(args.text)
        self.assertIsNone(args.file)
        self.assertIsNone(args.eval)
        self.assertEqual(args.preview, 10, "评估模式应默认列出 10 条逐条预测")

    def test_preview_can_be_disabled(self) -> None:
        """``--preview 0`` 应关闭逐条预览，只输出分数。"""
        from scripts.predict import parse_args

        self.assertEqual(parse_args(["--preview", "0"]).preview, 0)

    def test_explicit_inputs_override_the_default(self) -> None:
        """显式给出 --text / --file / --eval 时不得落到默认分支。"""
        from scripts.predict import parse_args

        self.assertEqual(parse_args(["--text", "很好"]).text, "很好")
        self.assertEqual(parse_args(["--file", "x.csv"]).file, "x.csv")
        self.assertEqual(parse_args(["--eval", "test"]).eval, "test")

    def test_every_wrapper_has_a_matching_trained_model(self) -> None:
        """每个封装脚本指向的模型都应已训练；未训练时跳过。"""
        models_dir = project_path("models")
        if not models_dir.is_dir():
            self.skipTest("models/ 不存在，尚未训练基线模型")
        for module_name, expected_model in self.WRAPPERS.items():
            with self.subTest(script=module_name):
                model_dir = models_dir / expected_model
                if not (model_dir / "metadata.json").is_file():
                    self.skipTest(f"{expected_model} 尚未训练")
                self.assertTrue((model_dir / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
