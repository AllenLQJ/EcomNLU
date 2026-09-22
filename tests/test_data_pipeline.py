"""M2 数据流水线的测试：清洗规则、数据划分、文本泄漏与接口契约合规。

本文件覆盖从原始 CSV 到 ``data/processed/`` 全部契约产物的关键规则，
是「数据处理阶段可复现且符合接口契约」这一交付要求的自动化证据。

Usage:
    python -m unittest discover -s tests

输入文件：
    - 单元测试：不读取任何磁盘文件，直接在内存里构造一个小型合成 DataFrame；
    - 集成测试：读取真实的原始 CSV（``data.raw_path``）、清洗产物
      ``data/interim/reviews_clean.csv`` 以及 ``data/processed/`` 下的
      三个划分文件与两个清单文件。

输出文件：
    无。测试只做断言，不写任何文件，避免污染仓库中的真实数据产物。

关键设计理由：
    1. **单元测试使用小型合成数据、永远可运行**：清洗与划分的每条规则都用
       几行手工构造的样本精确覆盖，因此不依赖 6 万余条的原始数据是否已下载，
       在任意机器上都能立即执行，适合提交前快速自检；
    2. **集成测试校验真实产物、产物不存在时自动跳过**：``setUp`` 中先检查
       文件是否存在，不存在就 ``skipTest``，这样在尚未运行流水线的环境下
       测试套件不会整体失败，而在产物齐备时又能真正验证行数、标签合法性、
       契约字段与跨集合泄漏；
    3. **契约合规测试独立重算，不读取 manifest 的自我声明**：分层比例、泄漏计数、
       ``sample_id`` 可追溯性都由测试自己从文件重新计算后再与清单比对，
       否则清单写错了测试也会跟着错（与 ``scripts/verify_data_contract.py`` 同样的思路，
       两者互为独立校验）；
    4. 集成测试里的期望数值（``62774``、缺失 1 条、重复 49 条、U+FEFF 20 条）
       是本数据集实测并记录在案的事实，一旦上游数据或清洗逻辑变化就会失败，
       属于有意的回归保护，不能为了「让测试通过」而放宽。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_reviews  # noqa: E402
from src.data.config import load_config, project_path  # noqa: E402
from src.data.manifest import (  # noqa: E402
    LEAKAGE_KEYS,
    build_label_mapping,
    leakage_for_manifest,
)
from src.data.raw_io import (  # noqa: E402
    CONTRACT_COLUMNS,
    load_raw_dataframe,
    sha256_of_file,
    write_contract_csv,
)
from src.data.splitting import (  # noqa: E402
    QUOTA_TOLERANCE,
    check_text_leakage,
    split_by_group,
)
from src.data.text_patterns import (  # noqa: E402
    UNICODE_FORMAT_PATTERN,
    has_meaningful_character,
    normalize_whitespace,
)


TEXT = "review"
CATEGORY = "cat"
SENTIMENT = "label"
SAMPLE_ID = "sample_id"

# 契约校验器使用的分层容差（行）。
CONTRACT_STRATIFICATION_TOLERANCE = 1.0


def make_frame(rows: list[list[object]]) -> pd.DataFrame:
    """把行列表包装成带有本项目三个约定列的数据框。

    参数：
        rows：每行按 ``[类别, 情感标签, 评论文本]`` 顺序排列的合成数据。

    返回：
        列名为 ``cat, label, review`` 的 DataFrame。
    """
    return pd.DataFrame(rows, columns=[CATEGORY, SENTIMENT, TEXT])


def clean(rows: list[list[object]]):
    """在内存中的合成数据上运行正式清洗流水线。

    参数：
        rows：合成数据行。

    返回：
        :class:`src.data.cleaning.CleaningResult`。
    """
    return clean_reviews(
        make_frame(rows),
        text_column=TEXT,
        category_column=CATEGORY,
        sentiment_column=SENTIMENT,
    )


class CleaningRuleTests(unittest.TestCase):
    """每条测试锁定一条已记录在案的清洗规则。"""

    def test_missing_and_blank_reviews_are_dropped(self) -> None:
        """缺失与纯空白评论应被删除，其余样本保留。"""
        result = clean(
            [
                ["书籍", 1, None],
                ["书籍", 0, "   "],
                ["书籍", 1, "内容很好"],
            ]
        )
        self.assertEqual(result.dataframe.shape[0], 1)
        self.assertEqual(result.dataframe.iloc[0][TEXT], "内容很好")

    def test_control_characters_are_removed_but_sample_kept(self) -> None:
        """控制字符应被删除，但样本本身必须保留。"""
        result = clean([["书籍", 1, "好书\x7f值得推荐"]])
        self.assertEqual(result.dataframe.shape[0], 1)
        self.assertEqual(result.dataframe.iloc[0][TEXT], "好书值得推荐")

    def test_bom_and_zero_width_characters_are_removed(self) -> None:
        """BOM 与零宽字符应被删除。

        原始数据中有 20 条评论文本内部嵌入了 U+FEFF，若不清除，
        它会被 jieba 切成一个独立 token，污染 TF-IDF 词表。
        """
        result = clean([["书籍", 1, "\ufeff好书\u200b推荐"]])
        self.assertEqual(result.dataframe.shape[0], 1)
        self.assertEqual(result.dataframe.iloc[0][TEXT], "好书推荐")

    def test_html_entities_are_unescaped(self) -> None:
        """HTML 实体应还原为真实字符。"""
        result = clean([["书籍", 1, "彼得&#183;林奇的书"]])
        self.assertEqual(result.dataframe.iloc[0][TEXT], "彼得·林奇的书")

    def test_whitespace_is_normalised(self) -> None:
        """换行与连续空格应被合并为单个空格。"""
        result = clean([["书籍", 1, "好书\n\n推荐   很快  "]])
        self.assertEqual(result.dataframe.iloc[0][TEXT], "好书 推荐 很快")

    def test_unicode_spaces_are_normalised(self) -> None:
        """全角空格与不换行空格也应被归一化。

        回归保护：pandas 3.x 的字符串列默认使用 PyArrow 后端（``ArrowStringArray``），
        ``.str.replace(r"\\s+")`` 因此由 PyArrow 绑定的 RE2 执行，
        而 RE2 的 ``\\s`` 等价于 ``[\\t\\n\\f\\r ]``，只覆盖 ASCII 空白，
        匹配不到全角空格（U+3000）与不换行空格（U+00A0），必须改用 Python 的 ``re``。
        """
        result = clean([["书籍", 1, "好书\u3000推荐\u00a0很快"]])
        self.assertEqual(result.dataframe.iloc[0][TEXT], "好书 推荐 很快")

    def test_pure_symbol_text_is_dropped(self) -> None:
        """既无中文也无字母的纯符号文本应被删除。"""
        result = clean(
            [
                ["洗发水", 1, "??????????"],
                ["洗发水", 1, "？？？？？？"],
                ["洗发水", 1, "真的很好用"],
            ]
        )
        self.assertEqual(result.dataframe.shape[0], 1)

    def test_non_chinese_text_with_letters_is_kept(self) -> None:
        """英文与拼音评论仍携带信号，不得被「无意义文本」规则删除。"""
        result = clean(
            [
                ["书籍", 0, "waste of my money"],
                ["手机", 1, "keyi"],
            ]
        )
        self.assertEqual(result.dataframe.shape[0], 2)

    def test_duplicate_text_inside_one_category_is_dropped(self) -> None:
        """同一类别内完全重复的文本属于冗余记录，应被删除。"""
        result = clean(
            [
                ["书籍", 1, "非常好的一本书"],
                ["书籍", 1, "非常好的一本书"],
            ]
        )
        self.assertEqual(result.dataframe.shape[0], 1)

    def test_same_text_under_different_categories_is_preserved(self) -> None:
        """跨类别的相同文本是两个合法的「文本-标签」配对，必须保留。

        这类记录不是冗余数据，若被全局去重会丢失有效样本；
        跨集合泄漏改由「以文本为单位整体划分」来防止。
        """
        result = clean(
            [
                ["书籍", 1, "还不错"],
                ["手机", 1, "还不错"],
            ]
        )
        self.assertEqual(result.dataframe.shape[0], 2)

    def test_labels_stay_legal_after_cleaning(self) -> None:
        """清洗后情感标签必须仍然只取 0 或 1。"""
        result = clean(
            [
                ["书籍", 1, "很好"],
                ["书籍", 0, "很差"],
            ]
        )
        self.assertTrue(set(result.dataframe[SENTIMENT].unique()).issubset({0, 1}))

    def test_every_step_is_recorded(self) -> None:
        """七步清洗规则必须按固定顺序执行且全部留有记录。"""
        result = clean([["书籍", 1, "很好"]])
        names = [step.name for step in result.steps]
        self.assertEqual(
            names,
            [
                "drop_missing_or_blank",
                "remove_control_chars",
                "strip_unicode_format_chars",
                "unescape_html_entities",
                "normalize_whitespace",
                "drop_meaningless_text",
                "deduplicate_within_category",
            ],
        )
        self.assertEqual(result.rows_before, result.steps[0].rows_before)
        self.assertEqual(result.rows_after, result.steps[-1].rows_after)

    def test_sample_id_records_the_original_row_number(self) -> None:
        """sample_id 必须记录原始行号，且在删除样本后保持正确。

        契约要求任意处理后样本都能追溯回原始数据，因此这个编号不能用
        ``reset_index`` 充当：删行之后行号会整体前移，追溯就会错位。
        """
        # 第 0 行缺失会被删除，第 1、3 行被保留，因此它们的 sample_id 应为 1 和 3
        result = clean(
            [
                ["书籍", 1, None],
                ["书籍", 1, "第一本好书"],
                ["书籍", 0, "   "],
                ["手机", 1, "第一台好手机"],
            ]
        )
        self.assertEqual(list(result.dataframe[SAMPLE_ID]), [1, 3])

    def test_output_columns_follow_the_contract_order(self) -> None:
        """清洗产物的列顺序必须严格符合契约。"""
        result = clean([["书籍", 1, "很好"]])
        self.assertEqual(list(result.dataframe.columns), list(CONTRACT_COLUMNS))


class SplittingTests(unittest.TestCase):
    """划分必须满足分层、可复现且无泄漏。"""

    @staticmethod
    def build_frame() -> pd.DataFrame:
        """构造含重复文本与极小类别的合成数据。

        返回：
            用于划分测试的 DataFrame。
        """
        rows: list[list[object]] = []
        # 两个样本量较大的类别，加上一个样本量很小的类别
        for index in range(200):
            rows.append(["书籍", index % 2, f"书籍评论{index}"])
        for index in range(100):
            rows.append(["手机", index % 2, f"手机评论{index}"])
        # 刻意制造重复文本
        rows.append(["书籍", 1, "通用好评"])
        rows.append(["手机", 1, "通用好评"])
        rows.append(["书籍", 0, "重复评论"])
        rows.append(["书籍", 0, "重复评论"])
        # 极小类别也必须能进入每一个集合
        for index in range(10):
            rows.append(["热水器", 1, f"热水器评论{index}"])
        return make_frame(rows)

    def test_no_text_crosses_splits(self) -> None:
        """任何评论文本都不得跨越划分边界。"""
        result = split_by_group(
            self.build_frame(),
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        leakage = check_text_leakage(result.splits, text_column=TEXT)
        self.assertTrue(leakage["leakage_free"], leakage)
        self.assertEqual(leakage["total_overlapping_texts"], 0)
        for count in leakage["pairwise_overlaps"].values():
            self.assertEqual(count, 0)

    def test_split_is_deterministic_for_a_fixed_seed(self) -> None:
        """固定随机种子下，两次划分必须得到完全相同的结果。"""
        frame = self.build_frame()
        first = split_by_group(
            frame,
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        second = split_by_group(
            frame,
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        for name in first.splits:
            pd.testing.assert_frame_equal(first.splits[name], second.splits[name])

    def test_all_rows_are_kept_and_sizes_follow_ratios(self) -> None:
        """划分不得丢行，且各集合规模应接近目标比例。"""
        frame = self.build_frame()
        result = split_by_group(
            frame,
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        total = sum(frame.shape[0] for frame in result.splits.values())
        self.assertEqual(total, frame.shape[0])
        for name, ratio in result.ratios.items():
            actual = result.splits[name].shape[0] / total
            self.assertAlmostEqual(actual, ratio, places=2)

    def test_every_category_meets_the_contract_tolerance(self) -> None:
        """每个类别的分层比例都必须落在契约容差内。

        回归保护：分组划分以「整组投放」为单位，当类别含跨类别的重复文本组时
        会略微超出配额，因此需要一轮配额修复。契约的校验标准是 ±1.0 行。
        """
        frame = self.build_frame()
        result = split_by_group(
            frame,
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        combined = pd.concat(result.splits.values())
        totals = combined[CATEGORY].value_counts()
        for category, total in totals.items():
            for name, ratio in result.ratios.items():
                actual = int((result.splits[name][CATEGORY] == category).sum())
                with self.subTest(category=category, split=name):
                    self.assertLessEqual(
                        abs(actual - total * ratio),
                        CONTRACT_STRATIFICATION_TOLERANCE,
                    )

    def test_multi_category_text_stays_in_one_split(self) -> None:
        """跨类别的相同文本必须整体落在同一个集合内。"""
        result = split_by_group(
            self.build_frame(),
            text_column=TEXT,
            category_column=CATEGORY,
            ratios={"train": 0.8, "dev": 0.1, "test": 0.1},
            seed=42,
        )
        for name, frame in result.splits.items():
            hits = frame.loc[frame[TEXT] == "通用好评"]
            if not hits.empty:
                self.assertEqual(
                    hits.shape[0],
                    2,
                    f"重复文本在集合 {name} 内被拆开了",
                )

    def test_ratios_must_sum_to_one(self) -> None:
        """比例之和不为 1 时必须报错。"""
        with self.assertRaises(ValueError):
            split_by_group(
                self.build_frame(),
                text_column=TEXT,
                category_column=CATEGORY,
                ratios={"train": 0.8, "dev": 0.1},
                seed=42,
            )


class TextPatternTests(unittest.TestCase):
    """文本判定标准的单元测试。"""

    def test_has_meaningful_character_rejects_symbols_only(self) -> None:
        """纯符号与纯数字不含中文或字母，应被判为无意义。"""
        self.assertFalse(has_meaningful_character("????????"))
        self.assertFalse(has_meaningful_character("12345.6"))
        self.assertFalse(has_meaningful_character("，。！"))

    def test_has_meaningful_character_accepts_chinese_letters_and_pinyin(self) -> None:
        """中文、英文、拼音都应被判定为有意义。"""
        self.assertTrue(has_meaningful_character("手机很好"))
        self.assertTrue(has_meaningful_character("waste of my money"))
        self.assertTrue(has_meaningful_character("keyi"))

    def test_normalize_whitespace_handles_unicode_spaces(self) -> None:
        """归一化必须同时处理 ASCII 空白与 Unicode 空白。"""
        self.assertEqual(normalize_whitespace("a\u3000b\u00a0c"), "a b c")
        self.assertEqual(normalize_whitespace("  a\n\tb  "), "a b")

    def test_unicode_format_pattern_matches_bom_and_zero_width(self) -> None:
        """Unicode 格式字符正则必须能匹配 BOM 与零宽字符。"""
        for character in ("\ufeff", "\u200b", "\u200d", "\u00ad"):
            with self.subTest(character=repr(character)):
                self.assertIsNotNone(UNICODE_FORMAT_PATTERN.search(character))


class RawDatasetTests(unittest.TestCase):
    """针对真实原始文件的集成校验。"""

    def setUp(self) -> None:
        """加载配置并定位原始数据；文件缺失时跳过整组测试。"""
        self.config = load_config()
        self.raw_path = project_path(self.config["data"]["raw_path"])
        if not self.raw_path.is_file():
            self.skipTest("原始数据集尚未下载")

    def test_raw_data_matches_documented_facts(self) -> None:
        """原始数据必须与 ``data/README.md`` 记录与实测确认的事实一致。

        期望值全部为实测结果，属于有意的回归保护：
        行数 62,774、缺失评论 1 条、评论文本重复可删 49 行、完全重复行 0 行、
        含 U+FEFF 的评论 20 条；且类别与情感标签取值合法。
        """
        frame = load_raw_dataframe(self.raw_path)
        self.assertEqual(frame.shape[0], 62_774)
        self.assertEqual([str(column) for column in frame.columns], ["cat", "label", "review"])

        texts = frame["review"].astype("string")
        self.assertEqual(int(texts.isna().sum()), 1)
        self.assertEqual(int(frame.duplicated().sum()), 0)
        self.assertEqual(int(texts.duplicated(keep="first").sum()), 49)
        self.assertEqual(
            int(texts.fillna("").str.contains(UNICODE_FORMAT_PATTERN, regex=True).sum()),
            20,
        )
        self.assertTrue(set(frame["label"].unique()).issubset({0, 1}))
        self.assertEqual(
            sorted(frame["cat"].unique()), sorted(self.config["task"]["categories"])
        )


class ManifestBuilderTests(unittest.TestCase):
    """清单构造函数的单元测试（不依赖磁盘上的真实产物）。"""

    def test_label_mapping_ids_follow_category_order(self) -> None:
        """id 必须等于类别在列表中的下标。"""
        categories = ["甲", "乙", "丙"]
        mapping = build_label_mapping(categories)
        self.assertEqual(mapping["categories"], categories)
        self.assertEqual(mapping["label2id"], {"甲": 0, "乙": 1, "丙": 2})
        self.assertEqual(mapping["id2label"], {"0": "甲", "1": "乙", "2": "丙"})

    def test_label_mapping_rejects_duplicate_categories(self) -> None:
        """类别重复时必须报错，否则 id 映射会有歧义。"""
        with self.assertRaises(ValueError):
            build_label_mapping(["甲", "乙", "甲"])

    def test_leakage_for_manifest_renames_keys(self) -> None:
        """泄漏结果必须转换成契约规定的键名，且 based_on 固定为 normalized_text。"""
        converted = leakage_for_manifest(
            {
                "pairwise_overlaps": {
                    "train__dev": 0,
                    "train__test": 0,
                    "dev__test": 0,
                }
            }
        )
        self.assertEqual(converted["based_on"], "normalized_text")
        for key in LEAKAGE_KEYS:
            self.assertEqual(converted[key], 0)


class ContractArtifactTests(unittest.TestCase):
    """接口契约产物的合规校验。

    这些断言与 ``scripts/verify_data_contract.py`` 相互独立：
    校验脚本由 LanQvJian 维护、在契约分支上运行；本组测试由数据方维护、
    随流水线一起运行。两者都能发现对方的疏漏。
    """

    def setUp(self) -> None:
        """定位契约产物；文件缺失时跳过整组测试。"""
        self.config = load_config()
        self.categories = [str(c) for c in self.config["task"]["categories"]]
        self.processed_dir = project_path(self.config["data"]["processed_dir"])
        self.split_paths = {
            name: self.processed_dir / f"{name}.csv"
            for name in ("train", "dev", "test")
        }
        self.required = [
            *self.split_paths.values(),
            self.processed_dir / "label_mapping.json",
            self.processed_dir / "split_manifest.json",
        ]
        if not all(path.is_file() for path in self.required):
            self.skipTest("请先运行 clean_data.py 与 split_data.py 生成契约产物")

        self.frames = {
            name: pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
            for name, path in self.split_paths.items()
        }
        self.label_mapping = json.loads(
            (self.processed_dir / "label_mapping.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (self.processed_dir / "split_manifest.json").read_text(encoding="utf-8")
        )

    def test_label_mapping_matches_config_order(self) -> None:
        """label_mapping 的类别顺序必须与配置完全一致。"""
        self.assertEqual(self.label_mapping["categories"], self.categories)
        self.assertEqual(
            self.label_mapping["label2id"],
            {name: index for index, name in enumerate(self.categories)},
        )
        self.assertEqual(
            self.label_mapping["id2label"],
            {str(index): name for index, name in enumerate(self.categories)},
        )

    def test_split_columns_and_order(self) -> None:
        """三个划分文件的列名与顺序必须为 sample_id,review,cat,label。"""
        for name, frame in self.frames.items():
            with self.subTest(split=name):
                self.assertEqual(list(frame.columns), list(CONTRACT_COLUMNS))

    def test_review_values_are_clean(self) -> None:
        """review 不得为空或纯空白，且不得有首尾空白。"""
        for name, frame in self.frames.items():
            review = frame[TEXT]
            with self.subTest(split=name):
                self.assertFalse(review.str.strip().eq("").any())
                self.assertFalse(review.ne(review.str.strip()).any())

    def test_categories_and_labels_are_legal(self) -> None:
        """类别必须属于配置的 10 类，情感标签只能是 0 或 1。"""
        for name, frame in self.frames.items():
            with self.subTest(split=name):
                self.assertTrue(set(frame[CATEGORY].unique()).issubset(set(self.categories)))
                self.assertTrue(set(frame[SENTIMENT].unique()).issubset({"0", "1"}))

    def test_sample_ids_are_unique_and_across_splits_disjoint(self) -> None:
        """sample_id 必须是整数、集合内唯一、且跨集合不重复。"""
        all_ids: list[str] = []
        for name, frame in self.frames.items():
            ids = frame[SAMPLE_ID]
            with self.subTest(split=name):
                self.assertTrue(ids.str.fullmatch(r"\d+").all())
                self.assertFalse(ids.duplicated().any())
            all_ids.extend(ids.tolist())
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_sample_ids_trace_back_to_raw_rows(self) -> None:
        """用原始数据核对 sample_id 指向的行，确认 cat 与 label 未被改动。

        这是契约里「可追溯」要求的实质校验：若清洗误改了标签或错位，这里会失败。
        """
        raw_path = project_path(self.config["data"]["raw_path"])
        if not raw_path.is_file():
            self.skipTest("原始数据集尚未下载")
        raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, encoding="utf-8")

        for name, frame in self.frames.items():
            index = frame[SAMPLE_ID].astype(int)
            with self.subTest(split=name):
                self.assertTrue((index < len(raw)).all())
                self.assertTrue(
                    (raw[CATEGORY].iloc[index].to_numpy() == frame[CATEGORY].to_numpy()).all()
                )
                self.assertTrue(
                    (raw[SENTIMENT].iloc[index].to_numpy() == frame[SENTIMENT].to_numpy()).all()
                )

    def test_splits_cover_the_cleaned_dataset_exactly(self) -> None:
        """三个划分行数之和必须等于清单记录的清洗后行数。"""
        total = sum(frame.shape[0] for frame in self.frames.values())
        self.assertEqual(self.manifest["cleaning"]["rows_after"], total)
        self.assertEqual(sum(self.manifest["counts"].values()), total)

    def test_manifest_has_no_unreplaced_placeholders(self) -> None:
        """清单不得残留形如 ``<...>`` 的未替换占位符。"""

        def walk(node: object) -> list[str]:
            found: list[str] = []
            if isinstance(node, dict):
                for value in node.values():
                    found.extend(walk(value))
            elif isinstance(node, list):
                for value in node:
                    found.extend(walk(value))
            elif isinstance(node, str) and node.startswith("<") and node.endswith(">"):
                found.append(node)
            return found

        self.assertEqual(walk(self.manifest), [])

    def test_manifest_fixed_rules(self) -> None:
        """清单中的固定规则字段必须符合契约。"""
        from datetime import datetime

        created_at = self.manifest["created_at"]
        self.assertIsNotNone(datetime.fromisoformat(created_at).tzinfo)
        self.assertEqual(
            self.manifest["random_seed"], self.config["project"]["random_seed"]
        )
        self.assertEqual(self.manifest["source"]["path"], self.config["data"]["raw_path"])
        self.assertEqual(self.manifest["cleaning"]["rows_before"], 62_774)
        for key in (
            "empty_review",
            "duplicate_review",
            "control_chars",
            "short_text_threshold",
        ):
            with self.subTest(rule=key):
                self.assertNotIn(self.manifest["cleaning"]["rules"].get(key), (None, ""))
        self.assertEqual(
            self.manifest["cleaning"]["rules"]["duplicate_review"],
            "drop_within_category_only",
        )
        self.assertEqual(self.manifest["split"]["strategy"], "stratified_by_cat")
        self.assertEqual(
            self.manifest["split"]["ratios"],
            {
                "train": float(self.config["data"]["train_ratio"]),
                "dev": float(self.config["data"]["dev_ratio"]),
                "test": float(self.config["data"]["test_ratio"]),
            },
        )
        self.assertEqual(self.manifest["split"]["dedup_key"], "normalized_text")
        self.assertEqual(self.manifest["leakage_check"]["based_on"], "normalized_text")

    def test_manifest_hashes_match_the_real_files(self) -> None:
        """清单记录的文件哈希必须与磁盘上的真实文件一致。"""
        raw_path = project_path(self.config["data"]["raw_path"])
        if raw_path.is_file():
            self.assertEqual(
                self.manifest["source"]["sha256"], sha256_of_file(raw_path)
            )
        for name, path in self.split_paths.items():
            entry = self.manifest["files"][f"{name}.csv"]
            with self.subTest(split=name):
                self.assertEqual(entry["sha256"], sha256_of_file(path))
                self.assertEqual(entry["rows"], self.frames[name].shape[0])

    def test_leakage_is_independently_zero(self) -> None:
        """独立重算跨集合重复文本，并与清单声明值比对。

        不直接相信清单里的 0，而是自己从文件重新计算。
        """
        frames = {
            name: pd.read_csv(path, encoding="utf-8")
            for name, path in self.split_paths.items()
        }
        leakage = check_text_leakage(frames, text_column=TEXT)
        self.assertTrue(leakage["leakage_free"], leakage)
        declared = leakage_for_manifest(leakage)
        for key in LEAKAGE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(self.manifest["leakage_check"][key], declared[key])

    def test_stratification_within_contract_tolerance(self) -> None:
        """独立按类别反算分层比例，确认落在契约容差内。"""
        combined = pd.concat([frame[CATEGORY] for frame in self.frames.values()])
        totals = combined.value_counts()
        ratios = {
            "train": float(self.config["data"]["train_ratio"]),
            "dev": float(self.config["data"]["dev_ratio"]),
            "test": float(self.config["data"]["test_ratio"]),
        }
        for category in self.categories:
            total = int(totals.get(category, 0))
            self.assertGreater(total, 0, f"{category} 无样本")
            for name, ratio in ratios.items():
                actual = int((self.frames[name][CATEGORY] == category).sum())
                with self.subTest(category=category, split=name):
                    self.assertLessEqual(
                        abs(actual - total * ratio),
                        CONTRACT_STRATIFICATION_TOLERANCE,
                    )

    def test_quota_tolerance_headroom(self) -> None:
        """配额修复的目标容差必须严于契约校验标准，留出余量。"""
        self.assertLess(QUOTA_TOLERANCE, CONTRACT_STRATIFICATION_TOLERANCE)

    def test_contract_csv_roundtrip_writes_expected_columns(self) -> None:
        """write_contract_csv 必须按契约列顺序写出，且缺少列时报错。"""
        import tempfile

        frame = pd.DataFrame(
            {
                SENTIMENT: [1],
                CATEGORY: ["书籍"],
                TEXT: ["很好"],
                SAMPLE_ID: [7],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            write_contract_csv(frame, path)
            written = pd.read_csv(path, encoding="utf-8")
            self.assertEqual(list(written.columns), list(CONTRACT_COLUMNS))
            # 文件不得带 BOM
            self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))

            with self.assertRaises(KeyError):
                write_contract_csv(frame.drop(columns=[SAMPLE_ID]), path)


if __name__ == "__main__":
    unittest.main()
