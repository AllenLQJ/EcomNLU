"""Download and validate the immutable raw dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "online_shopping_10_cats.csv"
DATASET_URL = (
    "https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/"
    "datasets/online_shopping_10_cats/online_shopping_10_cats.zip"
)
EXPECTED_SHA256 = "f6c5b724122d8547b700ac4524898eb7e28abbee63f798e95a8faa408829afdb"
EXPECTED_COLUMNS = ["cat", "label", "review"]
EXPECTED_ROWS = 62_774
DOWNLOAD_ATTEMPTS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv(path: Path) -> int:
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        columns = next(reader)
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns: {columns}")
        row_count = sum(1 for _ in reader)

    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {row_count}")
    return row_count


def fetch_archive() -> bytes:
    """Download the archive with bounded retries for transient network failures."""
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                DATASET_URL,
                headers={"User-Agent": "ecommerce-text-classification/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                chunks = []
                while chunk := response.read(1024 * 1024):
                    chunks.append(chunk)
                return b"".join(chunks)
        except Exception as exc:  # Network errors vary across Python/Windows versions.
            last_error = exc
            print(f"Download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {exc}")
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(attempt)
    raise RuntimeError("Dataset download failed after retries") from last_error


def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and not force:
        rows = validate_csv(OUTPUT_PATH)
        print(f"Dataset already valid: {OUTPUT_PATH} ({rows:,} rows)")
        return OUTPUT_PATH

    archive_bytes = fetch_archive()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in archive, found: {csv_names}")
        with archive.open(csv_names[0]) as source, tempfile.NamedTemporaryFile(
            dir=RAW_DIR, suffix=".csv.tmp", delete=False
        ) as target:
            temporary_path = Path(target.name)
            shutil.copyfileobj(source, target)

    try:
        rows = validate_csv(temporary_path)
        temporary_path.replace(OUTPUT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Downloaded and validated: {OUTPUT_PATH} ({rows:,} rows)")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_dataset()
