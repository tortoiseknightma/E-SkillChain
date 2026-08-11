"""Wikimedia Commons 开放许可文档图 → Utility 文档查询图。

按 Receipts、Letters、Certificates、Passports、Documents、Business cards、Tickets 七类获取，
只接受 Public Domain、CC0、CC BY、CC BY-SA，并逐图保存许可、署名和稳定描述页。主源
RVL-CDIP 端点不可用时使用本源；再不可用才退回 Tobacco3482，不使用文生图或页面爬虫。
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
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

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "ECommerceSkillChain/0.1 "
    "(academic research reproduction; https://github.com/)"
)
CATEGORIES = (
    "Receipts",
    "Letters",
    "Certificates",
    "Passports",
    "Documents",
    "Business cards",
    "Tickets",
)
RAW_FILE = config.DATA_DIR / "raw" / "wikimedia_documents" / "candidates.jsonl"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "utility_docs"
CC_LICENSE = re.compile(
    r"^CC (?P<kind>BY|BY-SA) (?P<version>[1-4]\.\d)(?: [a-z]{2})?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommonsCleanReport:
    candidates: int
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    image_paths: tuple[Path, ...]


def _plain_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _valid_license(license_name: str, license_url: str | None, attribution: str) -> bool:
    if license_name.casefold() == "public domain":
        return True
    parsed = urlsplit(str(license_url or ""))
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "creativecommons.org":
        return False
    if license_name.casefold() == "cc0":
        return parsed.path.startswith("/publicdomain/zero/1.0")
    match = CC_LICENSE.fullmatch(license_name)
    if not match or not attribution:
        return False
    kind = match.group("kind").lower()
    version = match.group("version")
    return parsed.path.startswith(f"/licenses/{kind}/{version}")


def normalize_candidate(candidate: dict) -> dict:
    row = dict(candidate)
    license_name = str(row.get("license") or "").strip()
    license_url = row.get("license_url")
    attribution = _plain_text(str(row.get("attribution") or ""))
    description = urlsplit(str(row.get("description_url") or ""))
    source = urlsplit(str(row.get("source_url") or ""))
    original = urlsplit(str(row.get("original_url") or ""))
    if not _valid_license(license_name, license_url, attribution):
        raise ValueError("Wikimedia Commons 许可/署名不完整或不一致")
    if (
        description.scheme != "https"
        or description.hostname != "commons.wikimedia.org"
        or not description.path.startswith("/wiki/File:")
        or source.scheme != "https"
        or source.hostname != "upload.wikimedia.org"
        or original.scheme != "https"
        or original.hostname != "upload.wikimedia.org"
    ):
        raise ValueError("Wikimedia Commons 来源 URL 不合规")
    row["license"] = license_name
    row["attribution"] = attribution
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


def extract_candidates(
    payload: dict,
    category: str,
    min_original_side: int = 200,
) -> list[dict]:
    candidates = []
    for page in payload.get("query", {}).get("pages", []):
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        info = imageinfo[0]
        metadata = info.get("extmetadata") or {}
        license_name = str((metadata.get("LicenseShortName") or {}).get("value") or "")
        license_url = (metadata.get("LicenseUrl") or {}).get("value")
        attribution = _plain_text(str((metadata.get("Artist") or {}).get("value") or ""))
        source_url = str(info.get("thumburl") or info.get("url") or "")
        original_url = str(info.get("url") or source_url)
        source_parts = urlsplit(source_url)
        original_parts = urlsplit(original_url)
        width = int(info.get("thumbwidth") or info.get("width") or 0)
        height = int(info.get("thumbheight") or info.get("height") or 0)
        description_url = str(info.get("descriptionurl") or "")
        if (
            source_parts.scheme != "https"
            or source_parts.hostname != "upload.wikimedia.org"
            or original_parts.scheme != "https"
            or original_parts.hostname != "upload.wikimedia.org"
            or min(width, height) < min_original_side
            or not _valid_license(license_name, license_url, attribution)
            or urlsplit(description_url).scheme != "https"
            or urlsplit(description_url).hostname != "commons.wikimedia.org"
            or not urlsplit(description_url).path.startswith("/wiki/File:")
        ):
            continue
        candidates.append(
            {
                "page_id": int(page["pageid"]),
                "title": str(page["title"]),
                "document_category": category,
                "width": width,
                "height": height,
                "mime": str(info.get("mime") or ""),
                "license": license_name,
                "license_url": license_url,
                "attribution": attribution,
                "description_url": description_url,
                "original_url": original_url,
                "source_url": source_url,
            }
        )
    return candidates


def _get_json(
    session,
    *,
    params: dict,
    attempts: int = 8,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    for attempt in range(attempts):
        try:
            response = session.get(
                API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == attempts - 1:
                raise
            sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def download_metadata(per_category: int = 50) -> list[dict]:
    candidates: list[dict] = []
    with requests.Session() as session:
        for category in CATEGORIES:
            payload = _get_json(
                session,
                params={
                    "action": "query",
                    "generator": "categorymembers",
                    "gcmtitle": f"Category:{category}",
                    "gcmtype": "file",
                    "gcmlimit": per_category,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|size|mime",
                    "iiurlwidth": 1200,
                    "format": "json",
                    "formatversion": 2,
                },
            )
            selected = extract_candidates(payload, category)
            candidates.extend(selected)
            print(f"category={category!r} accepted={len(selected)}")
            time.sleep(0.2)
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_FILE.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for candidate in candidates:
            target.write(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, RAW_FILE)
    print(f"candidates={len(candidates)} output={RAW_FILE}")
    return candidates


def _balanced_candidates(candidates: Iterable[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate["document_category"]].append(candidate)
    ordered: list[dict] = []
    for items in zip_longest(*(groups[name] for name in sorted(groups))):
        ordered.extend(item for item in items if item is not None)
    return ordered


def _fetch_image(candidate: dict) -> bytes | None:
    for attempt in range(3):
        try:
            response = requests.get(
                candidate["source_url"],
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
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
    limit: int = 200,
    min_side: int = 200,
    workers: int = 8,
    fetcher: Callable[[dict], bytes | None] = _fetch_image,
) -> CommonsCleanReport:
    if workers < 1:
        raise ValueError("workers 必须至少为 1")
    ordered = _balanced_candidates(candidates)
    destination = Path(destination)
    staging = prepare_staging_directory(destination)
    seen_hashes: set[str] = set()
    kept_names: list[str] = []
    duplicates = too_small = damaged = processed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor, (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for offset in range(0, len(ordered), workers):
                batch = ordered[offset : offset + workers]
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
                    filename = f"commons-{candidate['page_id']}.jpg"
                    image.save(staging / filename, format="JPEG", quality=90, optimize=True)
                    kept_names.append(filename)
                    row = dict(candidate)
                    row.update({"image": filename, "source": "wikimedia-commons"})
                    row["cloud_upload_allowed"] = True
                    row["public_demo_allowed"] = True
                    manifest.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    if len(kept_names) >= limit:
                        break
                if len(kept_names) >= limit:
                    break
        if len(kept_names) != limit:
            raise RuntimeError(
                f"Wikimedia Commons 合规文档图不足：期望 {limit}，实际 {len(kept_names)}"
            )
        if len(list(staging.glob("commons-*.jpg"))) != len(kept_names):
            raise RuntimeError("Wikimedia Commons staging 数量校验失败")
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return CommonsCleanReport(
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
) -> CommonsCleanReport:
    """保留 provenance 合规的缓存图，原子回填旧 manifest 中的不合规记录。"""
    destination = Path(destination)
    existing = [
        json.loads(line)
        for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized, rejected = normalize_candidates(candidates)
    by_page_id = {int(row["page_id"]): row for row in normalized}
    preserved = [
        by_page_id[int(row["page_id"])]
        for row in existing
        if int(row["page_id"]) in by_page_id
    ]
    preserved_ids = {int(row["page_id"]) for row in preserved}
    fallback = [
        row for row in reversed(normalized) if int(row["page_id"]) not in preserved_ids
    ]

    def fetch_with_existing_cache(candidate: dict) -> bytes | None:
        cached = destination / f"commons-{candidate['page_id']}.jpg"
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


def repair_manifest() -> CommonsCleanReport:
    with RAW_FILE.open(encoding="utf-8") as source:
        candidates = [json.loads(line) for line in source if line.strip()]
    report = repair_existing_collection(DESTINATION, candidates)
    write_preview(
        report.image_paths,
        config.DATA_DIR / "clean" / "preview_wikimedia_documents.jpg",
    )
    return report


def clean(limit: int = 200, workers: int = 8) -> CommonsCleanReport:
    with RAW_FILE.open(encoding="utf-8") as source:
        candidates, invalid_metadata = normalize_candidates(
            json.loads(line) for line in source if line.strip()
        )
    report = materialize(candidates, DESTINATION, limit=limit, workers=workers)
    write_preview(
        report.image_paths,
        config.DATA_DIR / "clean" / "preview_wikimedia_documents.jpg",
    )
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
            print(f"candidate[{index}] {json.loads(line)}")


def stats() -> None:
    rows = [
        json.loads(line)
        for line in (DESTINATION / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    categories = Counter(row["document_category"] for row in rows)
    licenses = Counter(row["license"] for row in rows)
    print(f"query_images={len(rows)}")
    print(f"categories={json.dumps(dict(sorted(categories.items())), ensure_ascii=False)}")
    print(f"licenses={json.dumps(dict(sorted(licenses.items())), ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--per-category", type=int, default=50)
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--limit", type=int, default=5)
    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--limit", type=int, default=200)
    clean_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("repair-manifest")
    subparsers.add_parser("stats")
    args = parser.parse_args()
    if args.command == "download":
        download_metadata(args.per_category)
    elif args.command == "probe":
        probe(args.limit)
    elif args.command == "clean":
        clean(args.limit, args.workers)
    elif args.command == "repair-manifest":
        repair_manifest()
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
