"""BERT 动态量化：把 Linear 层权重压成 int8，换取更小的体积与更快的 CPU 推理。

为什么选动态量化
----------------
``docs/development-plan.md`` 的 M5 要求「至少完成动态量化、蒸馏或剪枝中的一种」。
三者的取舍：

* **动态量化**：``torch.quantization.quantize_dynamic`` 一行完成，**不需要重新训练**，
  也不需要校准数据集；只量化 ``nn.Linear`` 的权重，激活值在运行时按 batch 动态定标。
  代价是 CPU 上才有加速（CUDA 没有对应的 int8 动态算子）。
* 蒸馏：收益最大，但需要从头训一个学生模型；本项目 BERT 单次微调已耗时 22.3 分钟，
  在 9/22 12:00 截止前无法完成可靠的学生模型训练与调优。
* 剪枝：需要再训练/微调才能恢复精度，否则 Macro-F1 会明显下降；同样受时间约束。

因此本项目选择动态量化，并把「不得只以成功保存作为完成依据」（AGENTS.md §5.6）
落实为 :func:`verify_quantized_model` —— 必须重新加载产物并跑真实推理。

关键设计约束
------------
1. **量化产物不能再用 ``from_pretrained`` 直接加载。** 标准 HF 加载流程会重建
   fp32 的 ``nn.Linear``，量化后的 ``weight``（``torch.qint8`` 打包格式）塞不回去。
   本模块的做法是三步：按 ``config`` 构造骨架 → **先调用** ``quantize_dynamic``
   把子模块换成 dynamic quant 容器 → 再 ``load_state_dict`` 灌入权重。
   顺序不能调换：量化 state_dict 里的 ``scale`` / ``zero_point`` /
   ``_packed_params.dtype`` 在 fp32 模块上没有对应容器，先灌权重会整批报未知键。
2. **量化只对 CPU 有意义。** :func:`quantize_model` 强制在 CPU 上进行，
   测量对比也必须用 CPU，否则 GPU 上的 fp32 与 CPU 上的 int8 不可比。
3. **量化不改标签顺序与 max_length。** ``label_mapping.json`` 与 ``metadata.json``
   原样复制，保证与统一推理接口、契约 §2.2 完全兼容。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.data.config import PROJECT_ROOT
from src.evaluation.metrics import directory_size_bytes


#: 量化权重文件名。放在 ``model/`` 之外，避免被 ``from_pretrained`` 误读为标准权重。
QUANTIZED_WEIGHTS_NAME = "quantized_state_dict.pt"

#: 需要量化的模块类型。只量化 Linear：BERT 的算力与参数几乎都在 Linear 上，
#: 而对 embedding / LayerNorm 做动态量化收益很小、精度风险反而更高。
QUANTIZED_MODULE_TYPES = ("Linear",)


class QuantizationError(RuntimeError):
    """量化流程中的可预期错误。"""


def _as_project_relative(path: Path) -> str:
    """把路径表示为相对项目根目录的 POSIX 形式。

    AGENTS.md §7.1 禁止在产物里写入 ``D:\\...`` 这类个人电脑绝对路径：它会随产物
    被队友拿到，既无法复现、也泄露本机目录结构。路径不在项目根目录内时保留原值，
    避免静默写出一条错误的相对路径。
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _require_torch() -> Any:
    """导入 torch，缺失时给出可执行的提示。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境缺失时才会走到
        raise QuantizationError(
            "未安装 torch，请先执行 pip install -r requirements.txt"
        ) from exc
    return torch


def load_fp32_model(model_dir: str | Path) -> tuple[Any, Any]:
    """加载源模型的 fp32 模型与分词器。

    Args:
        model_dir: 符合契约 §2.2 的模型目录，需含 ``model/`` 与 ``tokenizer/``。

    Returns:
        ``(model, tokenizer)``，模型已 ``eval()`` 但**未**搬到 GPU——
        动态量化只在 CPU 上有意义。

    Raises:
        QuantizationError: 目录结构不合规，或 transformers 加载失败。
    """
    directory = Path(model_dir)
    model_path = directory / "model"
    tokenizer_path = directory / "tokenizer"
    for path in (model_path, tokenizer_path):
        if not path.is_dir():
            raise QuantizationError(
                f"缺少目录 {path.name}/：{path}（见 docs/contracts.md §2.2）"
            )

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise QuantizationError(
            "未安装 transformers，请先执行 pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return model, tokenizer


def quantize_model(model: Any) -> Any:
    """对 ``nn.Linear`` 层做动态量化（int8 权重）。

    Args:
        model: fp32 的 ``BertForSequenceClassification``。

    Returns:
        量化后的模型；**它与入参是同一对象**（``quantize_dynamic`` 原地替换子模块），
        因此调用方不必重新绑定变量。
    """
    torch = _require_torch()
    if torch.cuda.is_available():
        # 明确记录：量化本身在 CPU 上完成，与是否有 GPU 无关。
        pass
    model.to("cpu")
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized


def count_dynamic_linear_layers(model: Any) -> int:
    """统计已被替换为 dynamic quant 的 Linear 层数量。"""
    torch = _require_torch()
    return sum(
        1
        for _, module in model.named_modules()
        if isinstance(module, torch.nn.quantized.dynamic.Linear)
    )


def count_parameters(model: Any) -> int:
    """统计模型参数量（含量化前的等价参数）。"""
    return int(sum(p.numel() for p in model.parameters()))


def save_quantized_model(
    model: Any,
    tokenizer: Any,
    source_dir: str | Path,
    target_dir: str | Path,
) -> dict[str, Any]:
    """把量化模型落盘为一份符合契约的模型产物目录。

    目录结构与源模型一致（``model/``、``tokenizer/``、``label_mapping.json``、
    ``metadata.json``），额外写入 :data:`QUANTIZED_WEIGHTS_NAME`。
    ``metadata.json`` 会追加 ``quantization`` 字段说明优化方式。

    Args:
        model: 已量化的模型。
        tokenizer: 与源模型一致的分词器。
        source_dir: 源模型目录，用于复制标签映射与元数据。
        target_dir: 目标目录；已存在的权重与元数据会被覆盖，**其余文件保持不动**。

    Returns:
        写入的 metadata 字典，便于调用方直接记录。
    """
    source = Path(source_dir)
    target = Path(target_dir)
    if not source.is_dir():
        raise QuantizationError(f"源模型目录不存在：{source}")

    # 关键：只覆盖权重与元数据，**绝不整目录删除**。
    # 目标目录里可能已有 metrics.json —— 那是实测得到的评测结果，耗时且不可再生。
    # 早期实现用 shutil.rmtree(target) 重建目录，一次运行中断就把评测结果永久删掉了。
    (target / "model").mkdir(parents=True, exist_ok=True)
    (target / "tokenizer").mkdir(parents=True, exist_ok=True)

    torch = _require_torch()

    # 先保存结构与超参（此时 config 仍是标准 BERT 配置），权重另存。
    model.config.save_pretrained(str(target / "model"))
    tokenizer.save_pretrained(str(target / "tokenizer"))
    torch.save(model.state_dict(), str(target / "model" / QUANTIZED_WEIGHTS_NAME))

    for name in ("label_mapping.json", "metadata.json"):
        shutil.copy2(source / name, target / name)

    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    # 优化产物的目录名是 <源模型>_<优化方式>，metadata 里的 model_name 必须跟着改，
    # 否则元数据与目录名不一致，会让对比报告和 Web 层读出两个不同的模型名。
    metadata["model_name"] = target.name
    metadata["quantization"] = {
        "method": "torch.quantization.quantize_dynamic",
        "quantized_module_types": list(QUANTIZED_MODULE_TYPES),
        "dtype": "qint8",
        "target_device": "cpu",
        "weights_file": f"model/{QUANTIZED_WEIGHTS_NAME}",
        "source_model_dir": _as_project_relative(source),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_quantized_model(model_dir: str | Path) -> tuple[Any, Any]:
    """重新加载量化产物，并用它完成后续真实推理。

    这是 AGENTS.md §5.6「必须重新加载产物并完成真实推理验证」的落地步骤。

    加载分三步：按 config 构建 fp32 骨架 → 灌入量化 ``state_dict`` →
    再次 ``quantize_dynamic`` 恢复动态量化模块结构。

    Args:
        model_dir: :func:`save_quantized_model` 产出的目录。

    Returns:
        ``(model, tokenizer)``，模型在 CPU 上且已 ``eval()``。
    """
    directory = Path(model_dir)
    weights = directory / "model" / QUANTIZED_WEIGHTS_NAME
    if not weights.is_file():
        raise QuantizationError(
            f"未找到量化权重 {weights}；请先运行 scripts/quantize_bert.py"
        )

    torch = _require_torch()
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise QuantizationError(
            "未安装 transformers，请先执行 pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(directory / "tokenizer"))

    # 加载顺序是本函数的关键，三步都不能调换：
    #
    # 1. 这里**不能**用 from_pretrained("model")：该目录下只有量化权重，没有
    #    model.safetensors / pytorch_model.bin，from_pretrained 会直接抛 OSError。
    #    因此先按 config 构造出同结构的 fp32 骨架。
    config = AutoConfig.from_pretrained(str(directory / "model"))
    model = AutoModelForSequenceClassification.from_config(config)

    # 2. 必须**先量化骨架、再灌权重**。量化后的 state_dict 里除了浮点参数，还有
    #    ``scale`` / ``zero_point`` / ``_packed_params.dtype`` 等 dynamic quant 专有键，
    #    fp32 的 nn.Linear 没有对应容器，直接 load_state_dict 会整批报「未知键」。
    #    先把子模块替换成 torch.nn.quantized.dynamic.Linear，这些键才有归宿。
    model = quantize_model(model)

    # weights_only=True 是现代 torch 的默认安全策略，量化权重是纯张量字典，可安全加载。
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise QuantizationError(
            f"量化权重含 {len(unexpected)} 个未知键，例如 {unexpected[:3]}"
        )
    model.eval()
    return model, tokenizer


def verify_quantized_model(model_dir: str | Path, sample_texts: list[str]) -> dict[str, Any]:
    """重新加载量化产物并对给定文本做真实推理。

    Args:
        model_dir: 量化产物目录。
        sample_texts: 探针文本，至少一条。

    Returns:
        含 ``loaded``、``dynamic_linear_layers``、``sample_predictions`` 的字典。

    Raises:
        QuantizationError: 产物缺失、加载失败或推理输出维度不符。
    """
    if not sample_texts:
        raise QuantizationError("verify_quantized_model 需要至少一条探针文本")

    torch = _require_torch()
    model, tokenizer = load_quantized_model(model_dir)
    layers = count_dynamic_linear_layers(model)
    if layers == 0:
        raise QuantizationError(
            "重新加载后没有任何 dynamic quant Linear 层，量化产物可能已损坏"
        )

    num_labels = int(model.config.num_labels)
    predictions: list[dict[str, Any]] = []
    for text in sample_texts:
        encoded = tokenizer(
            text, padding=True, truncation=True, max_length=192, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        if logits.shape[-1] != num_labels:
            raise QuantizationError(
                f"推理输出维度 {logits.shape[-1]} 与类别数 {num_labels} 不一致"
            )
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        best = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        predictions.append(
            {
                "text": text,
                "predicted_index": best,
                "confidence": round(float(probabilities[best]), 6),
            }
        )

    return {
        "loaded": True,
        "dynamic_linear_layers": layers,
        "parameters": count_parameters(model),
        "model_size_bytes": directory_size_bytes(Path(model_dir)),
        "sample_predictions": predictions,
    }
