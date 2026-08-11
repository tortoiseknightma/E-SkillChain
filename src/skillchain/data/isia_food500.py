"""ISIA Food-500 → Utility 食物查询图。

官方数据是 10 个 4 GiB 分卷加一个约 607 MiB 的末卷；为遵守“够用即止”，本脚本只下载
末卷，并解析其中落在同一卷内的完整 local ZIP records（当前实测 5,663 张、7 类）。若末卷
结构变化或合规图片不足，备用源固定为 Food-101，不伪造或文生食物图。发布方页面未声明
图片许可证，因此 manifest 明示 ``not-specified-by-publisher``，产物仅用于本研究复现。
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import io
import json
import mmap
import struct
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath

import imagehash
import requests
from PIL import Image, ImageOps

from skillchain import config
from skillchain.data import (
    discard_staging_directory,
    prepare_staging_directory,
    publish_staged_directory,
)
from skillchain.data.muge import write_preview

SOURCE_PAGE = "http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html"
ARCHIVE_URL = (
    "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/ISIA_Food500.zip"
)
RAW_DIR = config.DATA_DIR / "raw" / "isia_food500"
ARCHIVE_PATH = RAW_DIR / "ISIA_Food500.zip"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "utility_food"
ARCHIVE_SIZE = 636_880_932
# 官方只提供 HTTP 且未发布校验和；这是 2026-07-10 两次完整传输后固定的 TOFU 摘要，
# 用于阻止后续静默变化。它不能替代发布方签名，计划中因此禁止分发这批图片。
ARCHIVE_SHA256 = "3e68b472a0fa2bcd23d354c1fffcc101a54bb50e80dbd641d1b365d4a6f4109c"
MAX_COMPRESSED_SIZE = 32 << 20
MAX_UNCOMPRESSED_SIZE = 64 << 20
MAX_COMPRESSION_RATIO = 100
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_SIGNATURE = b"PK\x03\x04"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class LocalZipRecord:
    name: str
    category: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    data_offset: int
    data_end: int


@dataclass(frozen=True)
class FoodCleanReport:
    candidates: int
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    image_paths: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_at(data: mmap.mmap, offset: int) -> LocalZipRecord | None:
    if offset + LOCAL_HEADER.size > len(data):
        return None
    (
        signature,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        crc32,
        compressed_size,
        size,
        name_length,
        extra_length,
    ) = LOCAL_HEADER.unpack_from(data, offset)
    if (
        signature != 0x04034B50
        or flags & 0x08
        or method not in (0, 8)
        or not 1 <= name_length <= 1024
        or compressed_size > MAX_COMPRESSED_SIZE
        or size > MAX_UNCOMPRESSED_SIZE
        or (size > 0 and compressed_size == 0)
        or size / max(compressed_size, 1) > MAX_COMPRESSION_RATIO
    ):
        return None
    name_start = offset + LOCAL_HEADER.size
    name_end = name_start + name_length
    data_offset = name_end + extra_length
    data_end = data_offset + compressed_size
    if data_end > len(data):
        return None
    try:
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = bytes(data[name_start:name_end]).decode(encoding).replace("\\", "/")
    except UnicodeDecodeError:
        return None
    parts = PurePosixPath(name).parts
    if (
        len(parts) != 4
        or parts[0] != "ISIA_Food500"
        or parts[1] != "images"
        or PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES
    ):
        return None
    return LocalZipRecord(
        name=name,
        category=parts[2],
        method=method,
        crc32=crc32,
        compressed_size=compressed_size,
        size=size,
        data_offset=data_offset,
        data_end=data_end,
    )


def discover_local_records(archive_path: Path) -> list[LocalZipRecord]:
    """扫描末卷内完整 local records；不依赖缺失的前 10 卷或中央目录。"""
    records: list[LocalZipRecord] = []
    with Path(archive_path).open("rb") as source, mmap.mmap(
        source.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        position = 0
        while True:
            offset = data.find(LOCAL_SIGNATURE, position)
            if offset < 0:
                break
            record = _record_at(data, offset)
            if record is None:
                position = offset + len(LOCAL_SIGNATURE)
                continue
            records.append(record)
            position = record.data_end
    return records


def _read_record(data: mmap.mmap, record: LocalZipRecord) -> bytes:
    compressed = bytes(data[record.data_offset : record.data_end])
    if record.method == 0:
        raw = compressed
    else:
        decompressor = zlib.decompressobj(-15)
        raw = decompressor.decompress(compressed, MAX_UNCOMPRESSED_SIZE + 1)
        if len(raw) > MAX_UNCOMPRESSED_SIZE or decompressor.unconsumed_tail:
            raise ValueError(f"解压输出超限: {record.name}")
        remaining = MAX_UNCOMPRESSED_SIZE + 1 - len(raw)
        raw += decompressor.flush(remaining)
        if len(raw) > MAX_UNCOMPRESSED_SIZE or not decompressor.eof:
            raise ValueError(f"解压流不完整或输出超限: {record.name}")
    if len(raw) != record.size or (binascii.crc32(raw) & 0xFFFFFFFF) != record.crc32:
        raise ValueError(f"CRC/长度校验失败: {record.name}")
    return raw


def _balanced_records(records: list[LocalZipRecord]) -> list[LocalZipRecord]:
    groups: dict[str, list[LocalZipRecord]] = defaultdict(list)
    for record in records:
        groups[record.category].append(record)
    ordered: list[LocalZipRecord] = []
    for items in zip_longest(*(groups[name] for name in sorted(groups))):
        ordered.extend(item for item in items if item is not None)
    return ordered


def balanced_local_records(records: list[LocalZipRecord]) -> list[LocalZipRecord]:
    """Return the stable category-interleaved order used by materialization.

    Portfolio-only selected-member adapters use this public helper so they do
    not have to duplicate the archive ordering rule or unpack the ZIP tree.
    """

    return _balanced_records(records)


def read_local_record(data: mmap.mmap, record: LocalZipRecord) -> bytes:
    """Read, decompress, and CRC-verify one discovered local ZIP member."""

    return _read_record(data, record)


def materialize(
    archive_path: Path,
    destination: Path,
    *,
    limit: int = 600,
    min_side: int = 200,
) -> FoodCleanReport:
    records = _balanced_records(discover_local_records(archive_path))
    destination = Path(destination)
    staging = prepare_staging_directory(destination)
    seen_hashes: set[str] = set()
    kept_names: list[str] = []
    duplicates = too_small = damaged = processed = 0
    try:
        with Path(archive_path).open("rb") as source, mmap.mmap(
            source.fileno(), 0, access=mmap.ACCESS_READ
        ) as data, (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for record in records:
                processed += 1
                try:
                    raw = _read_record(data, record)
                    with Image.open(io.BytesIO(raw)) as opened:
                        opened.load()
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                except (OSError, ValueError, zlib.error):
                    damaged += 1
                    continue
                if min(image.size) < min_side:
                    too_small += 1
                    continue
                fingerprint = str(imagehash.phash(image))
                if fingerprint in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(fingerprint)
                filename = f"isia-{len(kept_names) + 1:04d}.jpg"
                image.save(staging / filename, format="JPEG", quality=90, optimize=True)
                kept_names.append(filename)
                manifest.write(
                    json.dumps(
                        {
                            "image": filename,
                            "category": record.category,
                            "source": "isia-food500",
                            "source_member": record.name,
                            "source_page": SOURCE_PAGE,
                            "source_archive": ARCHIVE_URL,
                            "source_archive_sha256": ARCHIVE_SHA256,
                            "source_volume_index": 10,
                            "license": "not-specified-by-publisher",
                            "license_note": (
                                "发布方截至 2026-07-10 未声明图片许可证；"
                                "本地研究复现使用，不随仓库、Demo 或交付物分发"
                            ),
                            "distribution": "local-only-not-redistributed",
                            "cloud_upload_allowed": True,
                            "public_demo_allowed": True,
                            "retrieved_at": "2026-07-10",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if len(kept_names) >= limit:
                    break
        if len(kept_names) != limit:
            raise RuntimeError(f"ISIA Food-500 合规图片不足：期望 {limit}，实际 {len(kept_names)}")
        if len(list(staging.glob("isia-*.jpg"))) != len(kept_names):
            raise RuntimeError("ISIA Food-500 staging 数量校验失败")
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return FoodCleanReport(
        candidates=processed,
        kept=len(kept_names),
        duplicates=duplicates,
        too_small=too_small,
        damaged=damaged,
        image_paths=tuple(destination / name for name in kept_names),
    )


def _download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int = ARCHIVE_SIZE,
    expected_sha256: str = ARCHIVE_SHA256,
    attempts: int = 5,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = None
    for attempt in range(attempts):
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            head.raise_for_status()
            remote_size = int(head.headers["Content-Length"])
            break
        except (KeyError, ValueError, requests.RequestException):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    if remote_size != expected_size:
        raise RuntimeError(
            f"ISIA Food-500 末卷远端长度变化：期望 {expected_size}，实际 {remote_size}"
        )
    if destination.exists() and destination.stat().st_size == expected_size:
        if sha256_file(destination) == expected_sha256:
            print(f"[skip] {destination.name} {expected_size} bytes sha256=ok")
            return
        print("[warn] 已有末卷 SHA-256 不匹配，将从头覆盖")
        have = 0
    else:
        have = destination.stat().st_size if destination.exists() else 0
    for attempt in range(attempts):
        try:
            headers = {"Range": f"bytes={have}-"} if 0 < have < expected_size else {}
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                mode = "ab" if have and response.status_code == 206 else "wb"
                with destination.open(mode) as target:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            target.write(chunk)
            have = destination.stat().st_size
            if have == expected_size:
                break
        except (OSError, requests.RequestException):
            have = destination.stat().st_size if destination.exists() else 0
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    if destination.stat().st_size != expected_size:
        raise RuntimeError("ISIA Food-500 末卷下载不完整，可重跑 download 断点续传")
    actual_sha256 = sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"ISIA Food-500 末卷 SHA-256 不匹配：{actual_sha256}"
        )


def download() -> None:
    _download_file(ARCHIVE_URL, ARCHIVE_PATH)


def probe(limit: int = 5) -> None:
    records = discover_local_records(ARCHIVE_PATH)
    categories = Counter(record.category for record in records)
    print(f"local_records={len(records)} categories={dict(sorted(categories.items()))}")
    with ARCHIVE_PATH.open("rb") as source, mmap.mmap(
        source.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        for index, record in enumerate(records[:limit]):
            raw = _read_record(data, record)
            with Image.open(io.BytesIO(raw)) as image:
                print(
                    f"image[{index}] member={record.name} format={image.format} "
                    f"size={image.size} mode={image.mode} bytes={len(raw)}"
                )


def clean(limit: int = 600) -> FoodCleanReport:
    report = materialize(ARCHIVE_PATH, DESTINATION, limit=limit)
    write_preview(report.image_paths, config.DATA_DIR / "clean" / "preview_isia_food500.jpg")
    print(
        f"candidates={report.candidates} kept={report.kept} duplicates={report.duplicates} "
        f"too_small={report.too_small} damaged={report.damaged}"
    )
    return report


def stats() -> None:
    rows = [
        json.loads(line)
        for line in (DESTINATION / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories = Counter(row["category"] for row in rows)
    print(f"query_images={len(rows)}")
    print(f"categories={json.dumps(dict(sorted(categories.items())), ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--limit", type=int, default=5)
    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--limit", type=int, default=600)
    subparsers.add_parser("stats")
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "probe":
        probe(args.limit)
    elif args.command == "clean":
        clean(args.limit)
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
