"""iNaturalist 开放许可图片 → Encyclopedia 查询图。

通过官方 observations API 取得少量研究级元数据，再直接从官方 Open Data S3
下载 500px medium 图；不遍历网站页面。只接受 CC0、CC-BY、CC-BY-SA，并把
许可、署名、观察与物种信息写入 manifest。备用源为 Wikimedia Commons。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

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

API_URL = "https://api.inaturalist.org/v1/observations"
RAW_FILE = config.DATA_DIR / "raw" / "inaturalist" / "candidates.jsonl"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "encyclopedia"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
LICENSE_URLS = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
}
OPEN_DATA_HOSTS = {"inaturalist-open-data.s3.amazonaws.com"}
ICONIC_TAXA = ("Plantae", "Animalia", "Fungi")


@dataclass(frozen=True)
class INatReport:
    candidates: int
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    image_paths: tuple[Path, ...]


def _is_open_data_url(url: str) -> bool:
    parsed_url = urlsplit(url)
    return (
        parsed_url.scheme == "https"
        and parsed_url.hostname in OPEN_DATA_HOSTS
        and parsed_url.path.startswith("/photos/")
    )


def _photo_variant(url: str, variant: str) -> str:
    return re.sub(r"/(?:square|medium|original)\.", f"/{variant}.", url)


def normalize_candidate(candidate: dict) -> dict:
    """补齐旧版 metadata 的审计字段，同时拒绝非官方开放数据域。"""
    row = dict(candidate)
    license_code = str(row.get("license") or "").lower()
    source_url = str(row.get("source_url") or "")
    if license_code not in LICENSE_URLS or not _is_open_data_url(source_url):
        raise ValueError("iNaturalist 候选缺少合规许可或不来自官方 Open Data 域")
    attribution = str(row.get("attribution") or "").strip()
    if license_code in {"cc-by", "cc-by-sa"} and not attribution:
        raise ValueError("iNaturalist CC BY/CC BY-SA 候选必须提供署名")
    row["license"] = license_code
    row["license_url"] = LICENSE_URLS[license_code]
    row["attribution"] = attribution
    row.setdefault(
        "observation_url",
        f"https://www.inaturalist.org/observations/{int(row['observation_id'])}",
    )
    api_photo_url = str(
        row.pop("photo_url", None)
        or row.get("api_photo_url")
        or _photo_variant(source_url, "square")
    )
    original_url = str(row.get("original_url") or _photo_variant(source_url, "original"))
    if not _is_open_data_url(api_photo_url) or not _is_open_data_url(original_url):
        raise ValueError("iNaturalist 照片 URL 不来自官方 Open Data 域")
    row["api_photo_url"] = api_photo_url
    row["original_url"] = original_url
    return row


def normalize_candidates(candidates: Iterable[dict]) -> tuple[list[dict], int]:
    accepted: list[dict] = []
    rejected = 0
    for candidate in candidates:
        try:
            accepted.append(normalize_candidate(candidate))
        except (KeyError, TypeError, ValueError):
            rejected += 1
    return accepted, rejected


def extract_candidates(payload: dict, min_original_side: int = 200) -> list[dict]:
    """从已探查的 API payload 中每个观察确定性选第一张合规照片。"""
    candidates = []
    for observation in payload.get("results", []):
        if observation.get("quality_grade") != "research":
            continue
        taxon = observation.get("taxon") or {}
        for photo in observation.get("photos") or []:
            license_code = str(photo.get("license_code") or "").lower()
            dimensions = photo.get("original_dimensions") or {}
            if license_code not in ALLOWED_LICENSES:
                continue
            width = int(dimensions.get("width") or 0)
            height = int(dimensions.get("height") or 0)
            if min(width, height) < min_original_side:
                continue
            url = str(photo.get("url") or "")
            if not _is_open_data_url(url):
                continue
            attribution = str(photo.get("attribution") or "").strip()
            if license_code in {"cc-by", "cc-by-sa"} and not attribution:
                continue
            candidates.append(
                {
                    "observation_id": int(observation["id"]),
                    "photo_id": int(photo["id"]),
                    "scientific_name": str(taxon.get("name") or ""),
                    "common_name": taxon.get("preferred_common_name"),
                    "iconic_taxon": str(taxon.get("iconic_taxon_name") or ""),
                    "license": license_code,
                    "license_url": LICENSE_URLS[license_code],
                    "attribution": attribution,
                    "observation_url": (
                        f"https://www.inaturalist.org/observations/{observation['id']}"
                    ),
                    "api_photo_url": url,
                    "original_url": _photo_variant(url, "original"),
                    "source_url": _photo_variant(url, "medium"),
                }
            )
            break
    return candidates


def _round_robin(groups: Iterable[list[dict]]) -> list[dict]:
    result = []
    for items in zip_longest(*groups):
        result.extend(item for item in items if item is not None)
    return result


def _get_json(
    session,
    url: str,
    *,
    params: dict,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """对瞬时 SSL/连接错误做有限指数退避，最终失败仍抛出原异常。"""
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            sleep(2**attempt)
    raise AssertionError("unreachable")


def download_metadata(per_taxon: int = 400, page_size: int = 200) -> list[dict]:
    """按植物/动物/真菌分别分页，轮转保存，避免单一大类挤占 800 图配额。"""
    groups: list[list[dict]] = []
    with requests.Session() as session:
        for iconic_taxon in ICONIC_TAXA:
            group: list[dict] = []
            page = 1
            while len(group) < per_taxon:
                params = {
                    "quality_grade": "research",
                    "photos": "true",
                    "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
                    "iconic_taxa": iconic_taxon,
                    "per_page": page_size,
                    "page": page,
                    "order": "asc",
                    "order_by": "id",
                }
                batch = extract_candidates(_get_json(session, API_URL, params=params))
                if not batch:
                    break
                group.extend(batch)
                page += 1
                time.sleep(1.0)
            groups.append(group[:per_taxon])
    candidates = _round_robin(groups)
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_FILE.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for candidate in candidates:
            target.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, RAW_FILE)
    print(f"candidates={len(candidates)} output={RAW_FILE}")
    return candidates


def _fetch_image(candidate: dict) -> bytes | None:
    for attempt in range(3):
        try:
            response = requests.get(candidate["source_url"], timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(0.5 * (2**attempt))
    return None


def materialize(
    candidates: list[dict],
    destination: Path,
    *,
    limit: int = 800,
    min_side: int = 200,
    workers: int = 8,
    fetcher: Callable[[dict], bytes | None] = _fetch_image,
) -> INatReport:
    if workers < 1:
        raise ValueError("workers 必须至少为 1")
    destination = Path(destination)
    staging = prepare_staging_directory(destination)
    seen_hashes: set[str] = set()
    kept_names: list[str] = []
    duplicates = too_small = damaged = processed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor, (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            # Python 3.12 的 executor.map 会一次提交整个 iterable；按 worker 数量分批，
            # 避免凑够 limit 后仍下载并等待所有未使用候选。
            for offset in range(0, len(candidates), workers):
                batch = candidates[offset : offset + workers]
                for candidate, raw in zip(batch, executor.map(fetcher, batch)):
                    processed += 1
                    if raw is None:
                        damaged += 1
                        continue
                    try:
                        with Image.open(io.BytesIO(raw)) as opened:
                            opened.load()
                            image = ImageOps.exif_transpose(opened).convert("RGB")
                    except (OSError, ValueError):
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
                    filename = f"inat-{candidate['photo_id']}.jpg"
                    image.save(staging / filename, format="JPEG", quality=90, optimize=True)
                    kept_names.append(filename)
                    row = dict(candidate)
                    row["image"] = filename
                    row["cloud_upload_allowed"] = True
                    row["public_demo_allowed"] = True
                    manifest.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    if len(kept_names) >= limit:
                        break
                if len(kept_names) >= limit:
                    break
        if len(list(staging.glob("inat-*.jpg"))) != len(kept_names):
            raise RuntimeError("iNaturalist staging 数量校验失败")
        if len(kept_names) != limit:
            raise RuntimeError(
                f"iNaturalist 合规图片不足：期望 {limit}，实际 {len(kept_names)}"
            )
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return INatReport(
        candidates=processed,
        kept=len(kept_names),
        duplicates=duplicates,
        too_small=too_small,
        damaged=damaged,
        image_paths=tuple(destination / name for name in kept_names),
    )


def repair_existing_collection(
    destination: Path,
    candidates: Iterable[dict],
    *,
    workers: int = 8,
    fetcher: Callable[[dict], bytes | None] = _fetch_image,
) -> INatReport:
    """保留合规缓存图，丢弃旧版非法来源记录，并从候选尾部确定性回填。"""
    destination = Path(destination)
    manifest_path = destination / "manifest.jsonl"
    existing = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized, rejected = normalize_candidates(candidates)
    by_photo_id = {int(row["photo_id"]): row for row in normalized}
    preserved: list[dict] = []
    for old_row in existing:
        metadata = by_photo_id.get(int(old_row["photo_id"]))
        if metadata is not None:
            preserved.append(metadata)
    preserved_ids = {int(row["photo_id"]) for row in preserved}
    fallback = [
        row
        for row in reversed(normalized)
        if int(row["photo_id"]) not in preserved_ids
    ]

    def fetch_with_existing_cache(candidate: dict) -> bytes | None:
        cached = destination / f"inat-{candidate['photo_id']}.jpg"
        if cached.is_file():
            try:
                return cached.read_bytes()
            except OSError:
                pass
        return fetcher(candidate)

    report = materialize(
        [*preserved, *fallback],
        destination,
        limit=len(existing),
        workers=workers,
        fetcher=fetch_with_existing_cache,
    )
    print(
        f"preserved={len(preserved)} backfilled={report.kept - len(preserved)} "
        f"invalid_metadata={rejected}"
    )
    return report


def refresh_existing_manifest(destination: Path, candidates: Iterable[dict]) -> int:
    """不重新下载图片，原子补齐既有 manifest 的来源与许可审计字段。"""
    destination = Path(destination)
    manifest_path = destination / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少既有 iNaturalist manifest: {manifest_path}")
    normalized, rejected = normalize_candidates(candidates)
    by_photo_id = {int(row["photo_id"]): row for row in normalized}
    existing = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    staging = prepare_staging_directory(destination)
    try:
        with (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for old_row in existing:
                photo_id = int(old_row["photo_id"])
                metadata = by_photo_id.get(photo_id)
                if metadata is None:
                    raise RuntimeError(f"既有照片 {photo_id} 在合规 metadata 中不存在")
                image_name = str(old_row["image"])
                if Path(image_name).name != image_name:
                    raise ValueError(f"非法图片文件名: {image_name}")
                source_image = destination / image_name
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                shutil.copy2(source_image, staging / image_name)
                row = dict(metadata)
                row["image"] = image_name
                row["cloud_upload_allowed"] = True
                row["public_demo_allowed"] = True
                manifest.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        if len(list(staging.glob("inat-*.jpg"))) != len(existing):
            raise RuntimeError("iNaturalist manifest 刷新时图片数量校验失败")
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    if rejected:
        print(f"invalid_metadata={rejected}")
    return len(existing)


def refresh_manifest() -> int:
    with RAW_FILE.open(encoding="utf-8") as source:
        candidates = [json.loads(line) for line in source if line.strip()]
    report = repair_existing_collection(DESTINATION, candidates, workers=24)
    refreshed = report.kept
    print(f"refreshed={refreshed} output={DESTINATION / 'manifest.jsonl'}")
    return refreshed


def clean(limit: int = 800, workers: int = 8) -> INatReport:
    with RAW_FILE.open(encoding="utf-8") as source:
        candidates, invalid_metadata = normalize_candidates(
            json.loads(line) for line in source if line.strip()
        )

    def fetch_with_existing_cache(candidate: dict) -> bytes | None:
        cached = DESTINATION / f"inat-{candidate['photo_id']}.jpg"
        if cached.is_file():
            try:
                return cached.read_bytes()
            except OSError:
                pass
        return _fetch_image(candidate)

    report = materialize(
        candidates,
        DESTINATION,
        limit=limit,
        workers=workers,
        fetcher=fetch_with_existing_cache,
    )
    write_preview(report.image_paths, config.DATA_DIR / "clean" / "preview_inaturalist.jpg")
    print(
        f"candidates={report.candidates} invalid_metadata={invalid_metadata} "
        f"kept={report.kept} duplicates={report.duplicates} "
        f"too_small={report.too_small} damaged={report.damaged}"
    )
    return report


def probe(limit: int = 5) -> None:
    with RAW_FILE.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= limit:
                break
            row = json.loads(line)
            print(f"candidate[{index}] fields={{{', '.join(f'{k}:{type(v).__name__}' for k, v in row.items())}}}")
            print(row)


def stats() -> None:
    manifest = DESTINATION / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    taxa: dict[str, int] = {}
    licenses: dict[str, int] = {}
    for row in rows:
        taxa[row["iconic_taxon"]] = taxa.get(row["iconic_taxon"], 0) + 1
        licenses[row["license"]] = licenses.get(row["license"], 0) + 1
    print(f"query_images={len(rows)}")
    print(f"iconic_taxa={json.dumps(taxa, ensure_ascii=False, sort_keys=True)}")
    print(f"licenses={json.dumps(licenses, ensure_ascii=False, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--per-taxon", type=int, default=400)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--limit", type=int, default=5)
    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--limit", type=int, default=800)
    clean_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("refresh-manifest")
    subparsers.add_parser("stats")
    args = parser.parse_args()
    if args.command == "download":
        download_metadata(per_taxon=args.per_taxon)
    elif args.command == "probe":
        probe(args.limit)
    elif args.command == "clean":
        clean(args.limit, args.workers)
    elif args.command == "refresh-manifest":
        refresh_manifest()
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
