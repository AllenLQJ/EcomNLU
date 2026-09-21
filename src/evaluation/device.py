"""训练与推理的设备探测，供全队共用。

模块职责：
    统一回答「这台机器上能用什么计算设备」，并把结论整理成可直接写入
    模型 ``metadata.json`` 的 ``device`` 字段（``docs/contracts.md`` §2.3）
    与报告中的环境说明。

主要输入与输出：
    输入：无（探测本机环境）。
    输出：设备字符串（``"cpu"`` / ``"cuda"``）与设备信息字典。

关键设计理由与约束：

1. **``device`` 必须记录「实际使用的」设备，而不是「可用的」设备。**
   契约 §2.3 要求不得填写未实际使用的设备。因此本模块把两件事分开：
   :func:`describe_environment` 报告能力（有无 GPU），
   :func:`resolve_device` 决定本次实际用什么，调用方把后者写进 ``metadata.json``。

2. **不同模型族的可用设备不同。** scikit-learn 的线性模型与随机森林只有 CPU 实现，
   即使机器上有可用的 CUDA 也必须记 ``"cpu"``；PyTorch / Transformers 可以走 GPU。
   :func:`resolve_device` 按模型族判断，避免出现「有 GPU 就无脑写 cuda」这种与事实
   不符的记录。

3. **导入 torch 的成本被隔离。** torch 是重量级依赖，只有确实需要判断 PyTorch
   设备时才导入；纯 CPU 的 sklearn 训练路径不应该因为探测而付出加载 torch 的代价，
   也不应该因为环境没装 torch 而无法运行。
"""

from __future__ import annotations

from typing import Any

# 各模型族的可用设备。"cpu_gpu" 表示同时支持 CPU 与 CUDA；
# "cpu_only" 表示该族在当前依赖下没有 GPU 实现。
MODEL_FAMILY_DEVICES: dict[str, str] = {
    "scikit-learn": "cpu_only",
    "fasttext": "cpu_only",  # fasttext-wheel 为 CPU 实现
    "transformers": "cpu_gpu",
    "torch": "cpu_gpu",
}


def cuda_available() -> bool:
    """检测当前环境是否可以实际使用 CUDA。

    Returns:
        ``True`` 表示 torch 已安装、为 CUDA 构建，且 ``torch.cuda.is_available()``
        为真。任何一项不满足都返回 ``False``（不抛异常）——
        探测失败应当降级到 CPU，而不是让训练无法启动。
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(model_family: str) -> str:
    """按模型族决定本次实际使用的设备。

    参数：
        model_family：``scikit-learn`` / ``fasttext`` / ``transformers`` / ``torch``。

    返回：
        ``"cuda"`` 或 ``"cpu"``。scikit-learn 与 fasttext 恒为 ``"cpu"``，
        因为它们没有 GPU 实现；transformers / torch 在 CUDA 可用时为 ``"cuda"``。

    异常：
        ValueError：模型族不在 :data:`MODEL_FAMILY_DEVICES` 中时抛出，
            避免未知模型族被默默当成 CPU 而记录出错误的 ``device``。
    """
    if model_family not in MODEL_FAMILY_DEVICES:
        raise ValueError(
            f"未知模型族 {model_family!r}，允许：{sorted(MODEL_FAMILY_DEVICES)}"
        )
    if MODEL_FAMILY_DEVICES[model_family] == "cpu_only":
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def describe_environment() -> dict[str, Any]:
    """汇总本机的计算设备情况，用于报告与答辩环境说明。

    Returns:
        含以下键的字典：

        - ``cuda_available``：CUDA 是否可用；
        - ``torch_version``：torch 版本，未安装时为 ``None``；
        - ``gpu_name``：GPU 型号，无可用 GPU 时为 ``None``；
        - ``gpu_total_memory_gb`` / ``gpu_free_memory_gb``：显存总量与当前可用量；
        - ``compute_capability``：算力等级，如 ``"8.9"``；
        - ``sklearn_device`` / ``fasttext_device`` / ``transformers_device``：
          各族在本机实际会使用的设备。
    """
    info: dict[str, Any] = {
        "cuda_available": False,
        "torch_version": None,
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_free_memory_gb": None,
        "compute_capability": None,
    }
    try:
        import torch
    except ImportError:
        pass
    else:
        info["torch_version"] = str(torch.__version__)
        if cuda_available():
            info["cuda_available"] = True
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                info["gpu_total_memory_gb"] = round(total / 1024**3, 1)
                info["gpu_free_memory_gb"] = round(free / 1024**3, 1)
                major, minor = torch.cuda.get_device_capability(0)
                info["compute_capability"] = f"{major}.{minor}"
            except Exception:
                # 显存查询等细节失败不影响「CUDA 可用」这个主结论
                pass

    for family in ("scikit-learn", "fasttext", "transformers"):
        key = family.replace("-", "_") + "_device"
        info[key] = resolve_device(family)
    return info


def render_environment_note() -> str:
    """把设备情况渲染成一段可直接放进报告的中文说明。

    Returns:
        Markdown 段落。
    """
    info = describe_environment()
    if info["cuda_available"]:
        capability = (
            f"，算力 {info['compute_capability']}" if info["compute_capability"] else ""
        )
        return (
            f"本机检测到可用的 CUDA 设备：{info['gpu_name']}"
            f"（显存 {info['gpu_total_memory_gb']} GB，当前可用 "
            f"{info['gpu_free_memory_gb']} GB{capability}），"
            f"torch 版本 {info['torch_version']}。"
            "但本报告涉及的三个模型均为 scikit-learn 实现，**该库没有 GPU 后端**，"
            "因此 `metadata.device` 按实际情况记为 `cpu`；"
            "CUDA 将用于 M4 的 FastText 与 BERT（PyTorch）阶段。"
        )
    return (
        "本机未检测到可用的 CUDA 设备，M3 与 M4 均在 CPU 上运行，"
        "`metadata.device` 记为 `cpu`。"
    )
