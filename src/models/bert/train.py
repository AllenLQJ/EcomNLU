"""BERT 微调训练、选模与产物落盘。

训练纪律（AGENTS.md §5.4、契约 §2.4）：

* 用训练集训练、**验证集 Macro-F1 选模**、测试集只做最终确认；
* 固定随机种子，记录 batch size、学习率、epoch、``max_length`` 与运行设备；
* 保存验证集 Macro-F1 最优的模型，并且以 ``save_pretrained`` 的形式落盘。

保存最优模型采用"按 epoch 评估、只保留历史最优"的策略，同时把每个 epoch 的
指标写进训练历史，供报告画出训练曲线（AGENTS.md §10 训练曲线要求）。
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.models.artifact import (
    MODELS_DIR,
    PROCESSED_DIR,
    copy_label_mapping,
    read_split,
    write_json,
    write_metadata,
)


@dataclass
class BertTrainConfig:
    """BERT 微调的超参数。取值集中在此，便于复现与写入 metadata。"""

    pretrained_dir: Path
    model_name: str = "bert_base_chinese"
    max_length: int = 192
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float | None = None
    seed: int = 42
    device: str = "auto"
    use_bf16: bool = True
    class_weights: bool = False
    log_every_steps: int = 100
    num_workers: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def hyperparameters(self) -> dict[str, Any]:
        """写入 ``metadata.json`` 的超参记录（可复现训练所需的最小集合）。"""
        payload = asdict(self)
        payload["pretrained_dir"] = str(self.pretrained_dir)
        return payload


def resolve_device(requested: str = "auto") -> str:
    """把设备请求解析为实际设备名。"""
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 的随机种子。"""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_environment(device: str) -> dict[str, Any]:
    """记录运行环境，写入 metadata 与训练历史（AGENTS.md §7.2）。"""
    import platform

    import torch
    import transformers

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info["gpu_name"] = properties.name
        info["gpu_total_memory_gb"] = round(properties.total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info


def build_optimizer_and_scheduler(
    model: Any,
    learning_rate: float,
    total_steps: int,
    warmup_ratio: float,
    weight_decay: float,
) -> tuple[Any, Any]:
    """AdamW + 线性预热后线性衰减。

    不依赖 transformers 的调度器工厂：该 API 在 5.x 有变动，自己实现更稳定，
    并且调度形状完全可复现。
    """
    import torch

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(key in name for key in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(key in name for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=learning_rate)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: Any,
    loader: Any,
    device: str,
    categories: Sequence[str],
    loss_fn: Any | None = None,
) -> dict[str, Any]:
    """在给定 DataLoader 上评估，返回准确率、Macro-F1 与平均损失。"""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predictions: list[int] = []
    truths: list[int] = []
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            if loss_fn is not None:
                total_loss += float(loss_fn(outputs.logits, inputs["labels"]).item())
                batches += 1
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            truths.extend(inputs["labels"].cpu().tolist())

    result = {
        "accuracy": round(float(accuracy_score(truths, predictions)), 6),
        "macro_f1": round(
            float(f1_score(truths, predictions, average="macro", zero_division=0)), 6
        ),
        "samples": len(truths),
    }
    if batches:
        result["loss"] = round(total_loss / batches, 6)
    return result


def _class_weight_tensor(
    rows: Sequence[tuple[str, int]], num_labels: int, device: str
) -> Any:
    """按训练集类别频次的倒数生成权重，缓和类别不均衡对 Macro-F1 的影响。"""
    import torch

    counts = [0] * num_labels
    for _, label in rows:
        counts[int(label)] += 1
    total = sum(counts)
    weights = [
        (total / (num_labels * count)) if count else 0.0 for count in counts
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_contract_model(
    model: Any,
    tokenizer: Any,
    model_dir: Path,
    config: BertTrainConfig,
    train_rows: int,
    environment: dict[str, Any],
    processed_dir: Path | None = None,
) -> Path:
    """按契约 §2.1/§2.2 落盘：模型本体、分词器、标签映射与 metadata。"""
    model_dir = Path(model_dir)
    (model_dir / "model").mkdir(parents=True, exist_ok=True)
    (model_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir / "model"))
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    copy_label_mapping(model_dir, processed_dir)
    write_metadata(
        model_dir=model_dir,
        model_name=config.model_name,
        framework="transformers",
        framework_version=str(environment.get("transformers", "unknown")),
        train_rows=train_rows,
        hyperparameters=config.hyperparameters(),
        device=str(environment.get("device", "cpu")),
        max_length=config.max_length,
        notes=config.notes,
        extra={
            "pretrained_model": str(config.pretrained_dir),
            "environment": environment,
            **config.extra,
        },
        processed_dir=processed_dir,
    )
    return model_dir


def run(
    config: BertTrainConfig,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """执行一次完整的 BERT 微调，返回训练与选模摘要。"""
    import torch

    from src.data.config import load_config
    from src.models.artifact import load_categories
    from src.models.bert.dataset import batch_length_stats, build_dataloader
    from src.models.bert.modeling import build_model, parameter_count
    from transformers import AutoTokenizer

    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    device = resolve_device(config.device)
    set_seed(config.seed)

    categories = load_categories(load_config())
    environment = describe_environment(device)
    print(
        f"[env] device={device} torch={environment['torch']} "
        f"transformers={environment['transformers']}"
        + (f" gpu={environment.get('gpu_name')}" if "gpu_name" in environment else "")
    )

    train_rows = read_split("train", categories, processed_dir)
    dev_rows = read_split("dev", categories, processed_dir)
    print(f"[data] train={len(train_rows):,} dev={len(dev_rows):,} max_length={config.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(str(config.pretrained_dir))
    train_loader = build_dataloader(
        train_rows, tokenizer, config.max_length, config.batch_size, True, config.num_workers, config.seed
    )
    dev_loader = build_dataloader(
        dev_rows, tokenizer, config.max_length, config.batch_size * 2, False, config.num_workers, config.seed
    )
    print(f"[data] 动态 padding 实测：{batch_length_stats(train_loader)}")

    model = build_model(config.pretrained_dir, categories, config.dropout).to(device)
    parameters = parameter_count(model)
    print(f"[model] 可训练参数 {parameters/1e6:.1f}M")

    loss_fn = torch.nn.CrossEntropyLoss()
    if config.class_weights:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=_class_weight_tensor(train_rows, len(categories), device)
        )
        print("[loss] 已启用按类别频次倒数加权的 CrossEntropyLoss")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config.learning_rate, total_steps, config.warmup_ratio, config.weight_decay
    )

    model_dir = models_dir / config.model_name
    history: list[dict[str, Any]] = []
    best = {"macro_f1": -1.0, "epoch": 0}
    use_bf16 = config.use_bf16 and device.startswith("cuda")
    training_started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        running_loss = 0.0
        seen_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**inputs).logits
                loss = loss_fn(logits, inputs["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            seen_batches += 1
            if step % config.log_every_steps == 0 or step == steps_per_epoch:
                print(
                    f"[train] epoch {epoch}/{config.epochs} step {step}/{steps_per_epoch} "
                    f"loss={running_loss/seen_batches:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{time.perf_counter()-epoch_started:.0f}s"
                )

        dev_result = evaluate(model, dev_loader, device, categories, loss_fn)
        record = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, seen_batches), 6),
            "dev": dev_result,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
            "learning_rate": config.learning_rate,
        }
        history.append(record)
        print(
            f"[dev] epoch {epoch}: macro_f1={dev_result['macro_f1']:.4f} "
            f"acc={dev_result['accuracy']:.4f} loss={dev_result.get('loss')}"
        )

        if dev_result["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": dev_result["macro_f1"], "epoch": epoch}
            save_contract_model(
                model, tokenizer, model_dir, config, len(train_rows), environment, processed_dir
            )
            print(f"[save] 验证集 Macro-F1 刷新为 {best['macro_f1']:.4f}，已保存到 {model_dir}")

    summary = {
        "model_name": config.model_name,
        "device": device,
        "environment": environment,
        "parameters": parameters,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "class_weights": config.class_weights,
        "bf16": use_bf16,
        "best_dev_macro_f1": best["macro_f1"],
        "best_epoch": best["epoch"],
        "total_seconds": round(time.perf_counter() - training_started, 1),
        "history": history,
        "model_dir": str(model_dir),
    }
    if history_path is not None:
        write_json(Path(history_path), summary)
    return summary
