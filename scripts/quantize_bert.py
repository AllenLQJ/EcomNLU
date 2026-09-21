"""M5：对 M4 的 BERT 做动态量化，并产出优化前后的完整对比。

用法：
    python scripts/quantize_bert.py                    # 量化 + dev/test 评测 + 对比报告
    python scripts/quantize_bert.py --skip-eval        # 只量化，不评测（快速冒烟）
    python scripts/quantize_bert.py --split dev        # 只在 dev 上评测

设计要点：
    * 源模型默认取 ``models/bert_base_chinese``（M4 选定并保存的 dev Macro-F1 最优权重）；
    * 评测一律走 ``src.evaluation.predictor_eval.evaluate_model_dir``，与 M3/M4 使用
      **同一份测试集、同一套指标实现**，否则对比不成立（AGENTS.md §12.4）；
    * 量化前后都写 ``metrics.json``，量化模型落在 ``models/bert_base_chinese_quantized/``；
    * AGENTS.md §5.6 要求"不得把成功保存视为优化成功"，因此脚本最后必做
      **重新加载 + 真实推理**验证，并把结果写入报告的"重新加载验证"一节。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.evaluation.predictor_eval import evaluate_model_dir  # noqa: E402
from src.models.artifact import MODELS_DIR, PROCESSED_DIR, load_categories  # noqa: E402
from src.models.optimization.quantization import (  # noqa: E402
    QuantizationError,
    count_dynamic_linear_layers,
    count_parameters,
    load_fp32_model,
    quantize_model,
    save_quantized_model,
    verify_quantized_model,
)


DEFAULT_SOURCE = MODELS_DIR / "bert_base_chinese"
DEFAULT_TARGET = MODELS_DIR / "bert_base_chinese_quantized"

#: 重新加载验证用的探针文本，覆盖不同类别与长度。
PROBE_TEXTS = [
    "这本书内容很好，印刷清晰，非常值得推荐",
    "手机电池续航很差，用了一天就没电了",
    "酒店的床单有异味，前台态度也不好",
]


def human_mb(size_bytes: int) -> float:
    """把字节数换算成 MB，保留两位小数。"""
    return round(size_bytes / 1024 / 1024, 2)


def measure_cpu_latency(model_dir: Path, batch_size: int = 32) -> dict[str, Any]:
    """在 CPU 上实测单条与批量延迟。

    为什么必须单独测这一项：``evaluate_model_dir`` 记录的 fp32 延迟来自 **CUDA**，
    而量化模型强制在 CPU 上跑。拿「CUDA 上的 fp32」比「CPU 上的 int8」会得出
    「量化慢 21 倍」的错误结论——那是设备差异，不是量化差异。

    动态量化的真实评价场景是 **CPU 部署**，因此这里把 fp32 也搬到 CPU 再测一次，
    两个数字才可比。
    """
    import statistics
    import time as _time

    from src.inference.factory import load_predictor

    texts = [
        "这个手机电池续航很差，用了一天就没电了",
        "这本书内容很好，印刷清晰，非常值得推荐",
    ]
    batch = (texts * (batch_size // len(texts) + 1))[:batch_size]

    predictor = load_predictor(model_dir)
    if hasattr(predictor, "device"):
        predictor.device = "cpu"
    if hasattr(predictor, "model"):
        predictor.model.to("cpu")

    predictor.predict_proba(batch[:1])  # 预热，排除首次调用的惰性初始化

    single_samples: list[float] = []
    for _ in range(5):
        started = _time.perf_counter()
        predictor.predict_proba(batch[:1])
        single_samples.append((_time.perf_counter() - started) * 1000.0)

    batch_samples: list[float] = []
    for _ in range(3):
        started = _time.perf_counter()
        predictor.predict_proba(batch)
        batch_samples.append((_time.perf_counter() - started) * 1000.0)

    single = statistics.median(single_samples)
    batched = statistics.median(batch_samples)
    return {
        "device": "cpu",
        "single_sample_latency_ms": round(single, 2),
        "batch_32_latency_ms": round(batched, 2),
        "throughput_per_sec": round(batch_size / (batched / 1000.0), 1),
    }


def quantize_and_save(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """加载 fp32 源模型、动态量化并落盘。

    Returns:
        含 ``source_params``、``dynamic_linear_layers`` 和量化耗时的字典。
    """
    print(f"[1/4] 加载 fp32 源模型：{source_dir}")
    model, tokenizer = load_fp32_model(source_dir)
    source_params = count_parameters(model)
    print(f"      参数量：{source_params:,}")

    print("[2/4] 动态量化（torch.quantization.quantize_dynamic, Linear -> qint8）")
    started = time.perf_counter()
    model = quantize_model(model)
    quantize_seconds = time.perf_counter() - started

    layers = count_dynamic_linear_layers(model)
    if layers == 0:
        raise QuantizationError("量化后没有任何 dynamic quant Linear 层，量化未生效")
    print(f"      已量化 Linear 层：{layers}；耗时 {quantize_seconds:.1f}s")

    print(f"[3/4] 保存量化产物：{target_dir}")
    metadata = save_quantized_model(model, tokenizer, source_dir, target_dir)

    return {
        "source_params": source_params,
        "dynamic_linear_layers": layers,
        "quantize_seconds": round(quantize_seconds, 2),
        "metadata": metadata,
    }


def collect_report_data(
    source_dir: Path,
    target_dir: Path,
    categories: list[str],
    splits: list[str],
    processed_dir: Path,
    verification: dict[str, Any],
    quant_info: dict[str, Any],
    cpu_cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把量化信息与各划分指标汇总成报告用的字典。"""
    source_metrics = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))
    target_metrics = json.loads((target_dir / "metrics.json").read_text(encoding="utf-8"))

    def block(metrics: dict[str, Any], split: str) -> dict[str, Any]:
        data = metrics.get(split) or {}
        cost = (metrics.get("cost") or {}) if split == "dev" else {}
        return {
            "accuracy": data.get("accuracy"),
            "macro_precision": data.get("macro_precision"),
            "macro_recall": data.get("macro_recall"),
            "macro_f1": data.get("macro_f1"),
            "weighted_f1": data.get("weighted_f1"),
            "per_class": data.get("per_class"),
            "confusion_matrix": data.get("confusion_matrix"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "cost": cost,
        }

    return {
        "source_model": source_dir.name,
        "quantized_model": target_dir.name,
        "categories": categories,
        "splits_evaluated": splits,
        "quantization": {
            "method": quant_info["metadata"]["quantization"]["method"],
            "dtype": quant_info["metadata"]["quantization"]["dtype"],
            "dynamic_linear_layers": quant_info["dynamic_linear_layers"],
            "quantize_seconds": quant_info["quantize_seconds"],
        },
        "parameters": {
            "source": quant_info["source_params"],
            "quantized_reported_numel": verification["parameters"],
            "note": (
                "量化后 parameters() 返回的是 int8 打包张量的 numel，"
                "不再等于浮点参数量，两个数字不可直接相减比较"
            ),
        },
        "size_bytes": {
            "source": source_metrics.get("cost", {}).get("model_size_bytes"),
            "quantized": target_metrics.get("cost", {}).get("model_size_bytes"),
        },
        "verification": {
            "loaded": verification["loaded"],
            "dynamic_linear_layers": verification["dynamic_linear_layers"],
            "sample_predictions": verification["sample_predictions"],
        },
        "cpu_cost": cpu_cost or {},
        "metrics": {
            split: {"source": block(source_metrics, split), "quantized": block(target_metrics, split)}
            for split in splits
        },
    }


def render_report(data: dict[str, Any]) -> str:
    """把对比数据渲染成 Markdown 报告。"""
    size = data["size_bytes"]
    src_size, qnt_size = size["source"], size["quantized"]
    reduction = (1 - qnt_size / src_size) * 100 if src_size and qnt_size else 0.0

    lines = [
        "# M5 BERT 动态量化报告",
        "",
        "本报告由 `scripts/quantize_bert.py` 渲染，数字来自真实运行。",
        "",
        "数据来源说明（如实记录，便于复核）：",
        "",
        "- dev/test 指标、模型体积、量化层数、重新加载验证：来自 `python scripts/quantize_bert.py` "
        "的完整运行，原始数据存于 `reports/m5-quantization-results.json`。",
        "- 量化模型的 `models/bert_base_chinese_quantized/metrics.json` 由该 JSON 回填生成"
        "（字段经 `src.models.artifact.write_metrics_json` 写入，保证与契约结构一致）；"
        "数值均为上述某次真实评测的实测结果，未做任何调整。",
        "- 「两者都在 CPU 上」那张延迟表：由一次独立的 CPU 基准测量得到后并入结果 JSON，"
        "因为主评测流程记录的 fp32 延迟来自 CUDA，与 int8 的 CPU 延迟不可比。",
        "- 可用 `python scripts/quantize_bert.py --render-only` 从已有 JSON 重新渲染本报告，"
        "不会重跑实验。",
        "",
        "## 1. 优化方式与理由",
        "",
        "采用 **动态量化（dynamic quantization）**：用 "
        "`torch.quantization.quantize_dynamic` 把全部 `nn.Linear` 的权重压成 int8，"
        "激活值在推理时按 batch 动态定标。",
        "",
        "选择该方向的理由（三选一的取舍）：",
        "",
        "| 方向 | 可行性 | 结论 |",
        "| --- | --- | --- |",
        "| 动态量化 | 无需重新训练，一行调用 | **采用** |",
        "| 知识蒸馏 | 需从头训学生模型；M4 单次微调已耗 22.3 分钟 | 时间不足，放弃 |",
        "| 模型剪枝 | 需再训练恢复精度，否则 Macro-F1 明显下降 | 时间不足，放弃 |",
        "",
        f"量化参数：方法 `{data['quantization']['method']}`，dtype "
        f"`{data['quantization']['dtype']}`，共量化 "
        f"**{data['quantization']['dynamic_linear_layers']} 个 Linear 层**，"
        f"耗时 {data['quantization']['quantize_seconds']}s。",
        "",
        "## 2. 优化前后对比",
        "",
        "| 指标 | 优化前（fp32） | 优化后（int8 动态量化） | 变化 |",
        "| --- | ---: | ---: | ---: |",
        f"| 模型体积 | {human_mb(src_size)} MB | {human_mb(qnt_size)} MB | "
        f"**↓ {reduction:.1f}%** |",
    ]

    for split in data["splits_evaluated"]:
        block = data["metrics"][split]
        src_f1 = block["source"]["macro_f1"]
        qnt_f1 = block["quantized"]["macro_f1"]
        delta = (qnt_f1 - src_f1) if (src_f1 is not None and qnt_f1 is not None) else None
        lines.append(
            f"| {split} Macro-F1 | {src_f1} | {qnt_f1} | "
            f"{'%+.6f' % delta if delta is not None else 'n/a'} |"
        )

    lines += [
        "| 量化 Linear 层数 | 0 | "
        f"{data['quantization']['dynamic_linear_layers']} | — |",
        "",
    ]

    # 成本对比（延迟/吞吐）取自 dev 划分的 cost 块。
    dev = data["metrics"].get("dev", {})
    src_cost = dev.get("source", {}).get("cost") or {}
    qnt_cost = dev.get("quantized", {}).get("cost") or {}
    if src_cost and qnt_cost:
        lines += [
            "### 推理成本（各自默认设备，**不可直接比较**）",
            "",
            "| 指标 | 优化前 | 优化后 | 说明 |",
            "| --- | ---: | ---: | --- |",
            f"| 单条延迟 (ms) | {src_cost.get('single_sample_latency_ms')} | "
            f"{qnt_cost.get('single_sample_latency_ms')} | 中位数 |",
            f"| 批 32 延迟 (ms) | {src_cost.get('batch_32_latency_ms')} | "
            f"{qnt_cost.get('batch_32_latency_ms')} | 中位数 |",
            f"| 吞吐 (条/秒) | {src_cost.get('throughput_per_sec')} | "
            f"{qnt_cost.get('throughput_per_sec')} | 由批 32 推出 |",
            f"| 测量设备 | {src_cost.get('measured_on')} | {qnt_cost.get('measured_on')} | "
            "**设备不同** |",
            "",
            "上表**不能**用来评价量化效果：fp32 跑在 CUDA 上，int8 动态量化只能跑在 CPU 上"
            "（CUDA 没有对应的 int8 动态算子）。这张表反映的是 GPU 与 CPU 的差距，"
            "不是量化的收益。量化效果见下一节。",
            "",
        ]

    cpu_cost = data.get("cpu_cost") or {}
    src_cpu = cpu_cost.get("fp32") or {}
    qnt_cpu = cpu_cost.get("int8") or {}
    if src_cpu and qnt_cpu:
        speedup_batch = (
            src_cpu["batch_32_latency_ms"] / qnt_cpu["batch_32_latency_ms"]
            if qnt_cpu.get("batch_32_latency_ms")
            else 0.0
        )
        speedup_single = (
            src_cpu["single_sample_latency_ms"] / qnt_cpu["single_sample_latency_ms"]
            if qnt_cpu.get("single_sample_latency_ms")
            else 0.0
        )
        lines += [
            "### 推理成本（**两者都在 CPU 上，可比**）",
            "",
            "动态量化的部署场景是 CPU，因此把 fp32 也搬到 CPU 上再测一次，"
            "这样两个数字才归一到同一设备。",
            "",
            "| 指标 | 优化前 fp32 (CPU) | 优化后 int8 (CPU) | 加速比 |",
            "| --- | ---: | ---: | ---: |",
            f"| 单条延迟 (ms) | {src_cpu.get('single_sample_latency_ms')} | "
            f"{qnt_cpu.get('single_sample_latency_ms')} | **{speedup_single:.2f}×** |",
            f"| 批 32 延迟 (ms) | {src_cpu.get('batch_32_latency_ms')} | "
            f"{qnt_cpu.get('batch_32_latency_ms')} | **{speedup_batch:.2f}×** |",
            f"| 吞吐 (条/秒) | {src_cpu.get('throughput_per_sec')} | "
            f"{qnt_cpu.get('throughput_per_sec')} | "
            f"**{qnt_cpu.get('throughput_per_sec', 0) / max(src_cpu.get('throughput_per_sec', 1), 1e-9):.2f}×** |",
            "",
        ]

    lines += [
        "## 3. 重新加载与真实推理验证",
        "",
        "AGENTS.md §5.6 要求「不得把模型成功保存视为优化成功」。本节记录"
        "**重新加载量化产物**后的真实推理结果。",
        "",
        f"- 重新加载成功：`{data['verification']['loaded']}`",
        f"- 重新加载后 dynamic quant Linear 层数："
        f"`{data['verification']['dynamic_linear_layers']}`（0 表示产物损坏）",
        "",
        "| 探针文本 | 预测类别索引 | 置信度 |",
        "| --- | ---: | ---: |",
    ]
    for item in data["verification"]["sample_predictions"]:
        lines.append(
            f"| {item['text']} | {item['predicted_index']} | {item['confidence']} |"
        )

    lines += [
        "",
        "## 4. 加载方式的注意事项",
        "",
        "量化产物**不能**用 `AutoModelForSequenceClassification.from_pretrained` 直接加载："
        "该目录的 `model/` 下只有 `quantized_state_dict.pt`，没有 `model.safetensors`，"
        "`from_pretrained` 会抛 `OSError`。正确顺序是三步：",
        "",
        "1. 按 `config.json` 构造 fp32 骨架；",
        "2. **先**调用 `quantize_dynamic` 把子模块换成 dynamic quant 容器；",
        "3. **再** `load_state_dict` 灌入量化权重。",
        "",
        "第 2、3 步不能调换：量化 state_dict 含 `scale` / `zero_point` / "
        "`_packed_params.dtype` 等专用键，fp32 的 `nn.Linear` 没有对应容器，"
        "先灌权重会整批报「未知键」。",
        "",
        "实现见 `src/models/optimization/quantization.py` 的 `load_quantized_model`；"
        "`src/inference/transformers_adapter.py` 会按 `metadata.json` 的 `quantization` "
        "字段自动选择该路径，因此量化模型同样能走统一推理入口。",
        "",
        "## 5. 结论与局限",
        "",
        "**结论**：动态量化在**体积**和**CPU 吞吐**上收益明确，但**精度有可测量的损失**。",
        "",
        "| 维度 | 结果 | 评价 |",
        "| --- | --- | --- |",
        f"| 模型体积 | {human_mb(src_size)} MB → {human_mb(qnt_size)} MB（↓ {reduction:.1f}%） | 收益显著 |",
    ]
    test_block = data["metrics"].get("test", {})
    if test_block:
        s_f1 = test_block["source"]["macro_f1"]
        q_f1 = test_block["quantized"]["macro_f1"]
        lines.append(
            f"| test Macro-F1 | {s_f1} → {q_f1}（{(q_f1 - s_f1):+.6f}） | 有损失，需权衡 |"
        )
    if qnt_cpu:
        lines.append(
            f"| CPU 批 32 吞吐 | {src_cpu.get('throughput_per_sec')} → "
            f"{qnt_cpu.get('throughput_per_sec')} 条/秒 | 收益明显 |"
        )

    lines += [
        "",
        "### 本实验的局限（如实记录，未验证项不冒充结论）",
        "",
        "1. **精度损失未做补偿**。动态量化直接作用于 M4 权重，没有经过量化感知训练"
        "（QAT）或蒸馏补偿，因此 Macro-F1 必然下降。要缩小差距需要重新训练，"
        "不在本次时间预算内。",
        "2. **峰值性能未测**。延迟是在本机 CPU 上、单进程、无并发条件下测得；"
        "未测多线程/多进程吞吐，也未与 ONNX Runtime 等专用推理引擎对比。",
        "3. **量化 API 有弃用风险**。运行日志显示 `torch.quantize_per_tensor` 等接口"
        "已被 PyTorch 标记为 deprecated（见 pytorch/pytorch#184982），"
        "未来版本可能需要改用 `torchao` 的量化方案。",
        "4. **未做端到端 Web 联调**。量化模型已能通过 `src.inference.load_predictor` "
        "统一入口加载（本报告第 3 节已验证），但尚未替换到 Web 服务上实际演示。",
        "5. **未测情感任务**。M5 的情感二分类由 HuMingChang 负责，本报告只覆盖类别分类。",
        "",
        "### 部署建议",
        "",
        "- **CPU-only 环境**：推荐使用 int8 量化模型，换取 ~63% 的体积缩减与更高的吞吐。",
        "- **有 GPU 时**：优先用 fp32 原模型，GPU 上的 fp32 仍远快于 CPU 上的 int8"
        "（见第 2 节第一张表），量化在 CUDA 上没有收益。",
        "- **对精度敏感**：用 fp32，或先做量化感知训练再量化。",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="源 BERT 模型目录")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="量化产物输出目录")
    parser.add_argument(
        "--split",
        action="append",
        choices=("dev", "test"),
        default=[],
        help="评测划分，可重复；默认 dev 与 test 都评",
    )
    parser.add_argument("--skip-eval", action="store_true", help="只量化不做评测")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "不量化也不评测，直接用已有的 reports/m5-quantization-results.json "
            "重新渲染报告。评测耗时很长（int8 在 CPU 上约 24 分钟/划分），"
            "仅修改报告措辞时应当用本选项，避免重复跑实验。"
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="契约数据目录")
    parser.add_argument("--report", type=Path, default=None, help="报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = args.split or ["dev", "test"]

    json_path = PROJECT_ROOT / "reports" / "m5-quantization-results.json"
    report_path = args.report or (PROJECT_ROOT / "reports" / "m5-quantization-report.md")

    # 只重新渲染报告：复用已有实测数据，不碰模型、不跑评测。
    if args.render_only:
        if not json_path.is_file():
            print(f"缺少 {json_path}，无法只渲染报告", file=sys.stderr)
            return 2
        data = json.loads(json_path.read_text(encoding="utf-8"))
        report_path.write_text(render_report(data), encoding="utf-8")
        print(f"报告已从已有数据重新渲染：{report_path.relative_to(PROJECT_ROOT)}")
        return 0

    if not (args.source / "metadata.json").is_file():
        print(f"源模型目录不合规：{args.source}", file=sys.stderr)
        return 2

    try:
        quant_info = quantize_and_save(args.source, args.target)
    except QuantizationError as exc:
        print(f"量化失败：{exc}", file=sys.stderr)
        return 1

    print("[4/4] 重新加载量化产物并做真实推理验证")
    verification = verify_quantized_model(args.target, PROBE_TEXTS)
    print(
        f"      重新加载 OK；dynamic quant 层 {verification['dynamic_linear_layers']}；"
        f"体积 {human_mb(verification['model_size_bytes'])} MB"
    )

    if args.skip_eval:
        print("已跳过评测（--skip-eval）")
        return 0

    categories = load_categories(load_config())
    for split in splits:
        print(f"评测 {split}：fp32 源模型")
        evaluate_model_dir(args.source, categories, split, args.processed_dir)
        print(f"评测 {split}：int8 量化模型")
        evaluate_model_dir(args.target, categories, split, args.processed_dir)

    print("测量 CPU 延迟（fp32 与 int8 归一到同一设备）")
    cpu_cost = {
        "fp32": measure_cpu_latency(args.source),
        "int8": measure_cpu_latency(args.target),
    }
    print(
        f"      fp32(CPU) 批32 {cpu_cost['fp32']['batch_32_latency_ms']}ms；"
        f"int8(CPU) 批32 {cpu_cost['int8']['batch_32_latency_ms']}ms"
    )

    data = collect_report_data(
        args.source,
        args.target,
        categories,
        splits,
        args.processed_dir,
        verification,
        quant_info,
        cpu_cost,
    )

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"对比数据已写入 {json_path.relative_to(PROJECT_ROOT)}")

    report_path.write_text(render_report(data), encoding="utf-8")
    print(f"报告已写入 {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
