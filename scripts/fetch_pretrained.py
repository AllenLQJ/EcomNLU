"""下载 BERT 中文预训练模型到本地目录。

用法：
    python scripts/fetch_pretrained.py
    python scripts/fetch_pretrained.py bert-base-chinese

为什么要单独写这个脚本：

* 本网络环境无法直连 ``huggingface.co``，需要走镜像 ``hf-mirror.com``；
* 镜像不实现 Xet 存储后端的 ``xet-read-token`` 接口，对 Xet 仓库直接返回 404，
  因此必须设置 ``HF_HUB_DISABLE_XET=1`` 回退到普通 HTTP 下载；
* Windows 未开启开发者模式时不能创建符号链接，默认缓存机制会失败，因此用
  ``snapshot_download(local_dir=...)`` 直接把真实文件写进项目目录。

下载结果写入 ``models/pretrained/<名称>/``（已被 ``.gitignore`` 排除），
并记录各文件的 SHA-256 以便复现与追溯（AGENTS.md §7.3）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
MANIFEST_NAME = "download_manifest.json"
BEIJING = timezone(timedelta(hours=8))

# 名称 -> (HuggingFace 仓库 id, 除权重外的必需文件)
CANDIDATES: dict[str, tuple[str, list[str]]] = {
    "bert-base-chinese": (
        "bert-base-chinese",
        [
            "config.json",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    ),
    "rbt3": (
        "hfl/rbt3",
        [
            "config.json",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    ),
}

# 一个可用模型至少需要这些文件；权重允许 safetensors 或 bin 两种格式之一。
REQUIRED_FILES = ("config.json", "vocab.txt")
# 权重格式按优先级尝试：先 safetensors，缺失时再回退到 bin。
# 不要一次请求两种格式——那是同一份权重的两份拷贝，白白多下几百 MB。
WEIGHT_PREFERENCES = ("model.safetensors", "pytorch_model.bin")


def configure_environment(endpoint: str | None = None) -> str:
    """在导入 huggingface_hub 之前设置好镜像与缓存环境变量。"""
    resolved = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    os.environ["HF_ENDPOINT"] = resolved
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_home"))
    return resolved


def sha256_of(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_files(target_dir: Path) -> list[str]:
    """返回缺失的必需文件；权重文件完全不存在时也算缺。"""
    missing = [name for name in REQUIRED_FILES if not (target_dir / name).is_file()]
    if not any((target_dir / name).is_file() for name in WEIGHT_PREFERENCES):
        missing.append(" 或 ".join(WEIGHT_PREFERENCES))
    return missing


def fetch_model(name: str, force: bool = False) -> Path:
    """下载并校验指定模型，返回本地目录。"""
    if name not in CANDIDATES:
        raise KeyError(f"未知模型 {name!r}，可选：{sorted(CANDIDATES)}")
    repo_id, extra_patterns = CANDIDATES[name]
    target_dir = PRETRAINED_DIR / name

    if not force and not missing_files(target_dir):
        print(f"[skip] {name} 已完整：{target_dir}")
        return target_dir

    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 先只请求 safetensors；若该仓库没有 safetensors，再回退请求 bin。
    for position, weight_file in enumerate(WEIGHT_PREFERENCES):
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=[*extra_patterns, weight_file],
            max_workers=4,
        )
        if not missing_files(target_dir):
            break
        if position + 1 < len(WEIGHT_PREFERENCES):
            print(f"[info] {name} 未获得 {weight_file}，尝试下一种权重格式")

    missing = missing_files(target_dir)
    if missing:
        raise RuntimeError(f"{name} 下载后仍缺少：{missing}")

    files = {
        item.name: {"bytes": item.stat().st_size, "sha256": sha256_of(item)}
        for item in sorted(target_dir.iterdir())
        if item.is_file() and item.name != MANIFEST_NAME
    }
    manifest = {
        "name": name,
        "repo_id": repo_id,
        "endpoint": os.environ.get("HF_ENDPOINT"),
        "downloaded_at": datetime.now(BEIJING).replace(microsecond=0).isoformat(),
        "files": files,
    }
    (target_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    size_mb = (
        sum(item.stat().st_size for item in target_dir.rglob("*") if item.is_file())
        / 1024**2
    )
    print(f"[ok] {name} <- {repo_id} | {size_mb:.1f} MB | {time.time() - started:.1f}s")
    return target_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names", nargs="*", default=[], help=f"要下载的模型名，默认全部：{sorted(CANDIDATES)}"
    )
    parser.add_argument("--endpoint", default=None, help="HF 镜像地址覆盖")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args(argv)

    endpoint = configure_environment(args.endpoint)
    print(f"endpoint: {endpoint}")
    print(f"目标目录: {PRETRAINED_DIR}")

    selected = args.names or list(CANDIDATES)
    unknown = [name for name in selected if name not in CANDIDATES]
    if unknown:
        parser.error(f"未知模型名：{unknown}；可选：{sorted(CANDIDATES)}")

    failures = 0
    for name in selected:
        try:
            fetch_model(name, force=args.force)
        except Exception as exc:  # 网络与镜像问题种类多，记录后继续
            failures += 1
            print(f"[fail] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
