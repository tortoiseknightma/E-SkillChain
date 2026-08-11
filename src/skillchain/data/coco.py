"""COCO 2017 → Multi-Product 查询图。

筛选规则基于已探查的 instances JSON：每张图至少包含 3 个非 crowd、边框短边
不小于 20px 的商品类实例。val2017 不足 800 时才扩展 train2017。备用源为
Open Images，避免把下载规模无条件扩大。
"""

from __future__ import annotations

import argparse
import binascii
import io
import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

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

RAW_DIR = config.DATA_DIR / "raw" / "coco"
ANNOTATION_PATH = RAW_DIR / "extracted" / "annotations" / "instances_val2017.json"
ANNOTATION_ARCHIVE = RAW_DIR / "annotations_trainval2017.zip"
IMAGE_ARCHIVE = RAW_DIR / "val2017.zip"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "multi_product"

DOWNLOADS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}

PRODUCT_CATEGORIES = {
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


@dataclass(frozen=True)
class CocoCleanReport:
    candidates: int
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    image_paths: tuple[Path, ...]


def select_candidates(
    data: dict,
    *,
    min_instances: int = 3,
    min_bbox_side: float = 20,
    limit: int | None = 800,
) -> list[dict]:
    """返回按 image_id 稳定排序、带类别计数的候选图片。"""
    category_names = {item["id"]: item["name"] for item in data["categories"]}
    image_metadata = {item["id"]: item for item in data["images"]}
    counts: dict[int, Counter] = defaultdict(Counter)
    for annotation in data["annotations"]:
        name = category_names.get(annotation["category_id"])
        bbox = annotation.get("bbox", [])
        if (
            name not in PRODUCT_CATEGORIES
            or annotation.get("iscrowd", 0)
            or len(bbox) != 4
            or min(float(bbox[2]), float(bbox[3])) < min_bbox_side
        ):
            continue
        counts[annotation["image_id"]][name] += 1

    selected = []
    for image_id in sorted(counts):
        category_counts = counts[image_id]
        instance_count = sum(category_counts.values())
        if instance_count < min_instances:
            continue
        image = image_metadata[image_id]
        selected.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "instance_count": instance_count,
                "categories": dict(sorted(category_counts.items())),
            }
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected


def materialize(
    data: dict,
    image_archive: Path,
    destination: Path,
    *,
    min_instances: int = 3,
    min_bbox_side: float = 20,
    limit: int = 800,
    min_image_side: int = 200,
) -> CocoCleanReport:
    candidates = select_candidates(
        data,
        min_instances=min_instances,
        min_bbox_side=min_bbox_side,
        limit=None,
    )
    destination = Path(destination)
    staging = prepare_staging_directory(destination)
    seen_hashes: set[str] = set()
    kept_names: list[str] = []
    duplicates = too_small = damaged = 0
    processed = 0

    try:
        with zipfile.ZipFile(image_archive) as archive, (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for candidate in candidates:
                processed += 1
                member = f"val2017/{candidate['file_name']}"
                try:
                    raw = archive.read(member)
                    with Image.open(io.BytesIO(raw)) as opened:
                        opened.load()
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                except (KeyError, OSError, ValueError, binascii.Error):
                    damaged += 1
                    continue
                if min(image.size) < min_image_side:
                    too_small += 1
                    continue
                fingerprint = str(imagehash.phash(image))
                if fingerprint in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(fingerprint)
                filename = f"coco-{candidate['image_id']}.jpg"
                output = staging / filename
                image.save(output, format="JPEG", quality=90, optimize=True)
                kept_names.append(filename)
                row = dict(candidate)
                row.update({"image": filename, "source": "coco-val2017"})
                manifest.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                if len(kept_names) >= limit:
                    break
        if len(list(staging.glob("coco-*.jpg"))) != len(kept_names):
            raise RuntimeError("COCO staging 数量校验失败")
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise

    return CocoCleanReport(
        candidates=processed,
        kept=len(kept_names),
        duplicates=duplicates,
        too_small=too_small,
        damaged=damaged,
        image_paths=tuple(destination / name for name in kept_names),
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    have = destination.stat().st_size if destination.exists() else 0
    if have == expected:
        print(f"[skip] {destination.name} {have} bytes")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        mode = "ab" if have and response.status_code == 206 else "wb"
        with destination.open(mode) as target:
            for chunk in response.iter_content(1 << 20):
                target.write(chunk)
    if destination.stat().st_size != expected:
        raise RuntimeError(f"{destination.name} 下载不完整，可重跑 download 断点续传")


def download() -> None:
    for name, url in DOWNLOADS.items():
        _download_file(url, RAW_DIR / name)
    extract_annotations(ANNOTATION_ARCHIVE, ANNOTATION_PATH)


def extract_annotations(archive_path: Path, output_path: Path) -> None:
    """安全、原子地解出 probe/clean 所需的 val instances 标注。"""
    member = "annotations/instances_val2017.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    try:
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, temporary.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target)
        with temporary.open(encoding="utf-8") as source:
            data = json.load(source)
        required = {"images", "categories", "annotations"}
        if not required.issubset(data):
            raise ValueError(f"COCO 标注缺少字段：{sorted(required.difference(data))}")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def probe(limit: int = 5) -> None:
    data = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    print(f"top={{{', '.join(f'{key}:{type(value).__name__}' for key, value in data.items())}}}")
    for section in ("images", "categories", "annotations"):
        print(f"{section}: count={len(data[section])}")
        for item in data[section][:limit]:
            print({key: type(value).__name__ for key, value in item.items()}, item)


def clean() -> CocoCleanReport:
    data = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    report = materialize(data, IMAGE_ARCHIVE, DESTINATION)
    write_preview(report.image_paths, config.DATA_DIR / "clean" / "preview_coco.jpg")
    print(
        f"candidates={report.candidates} kept={report.kept} duplicates={report.duplicates} "
        f"too_small={report.too_small} damaged={report.damaged}"
    )
    if report.kept < 800:
        print("[next] val2017 不足 800，需下载 train2017 扩展")
    return report


def stats() -> None:
    rows = [json.loads(line) for line in (DESTINATION / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    distribution = Counter()
    for row in rows:
        distribution.update(row["categories"])
    print(f"query_images={len(rows)}")
    print(f"category_instances={json.dumps(dict(distribution.most_common()), ensure_ascii=False)}")
    print(f"preview={config.DATA_DIR / 'clean' / 'preview_coco.jpg'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--limit", type=int, default=5)
    subparsers.add_parser("clean")
    subparsers.add_parser("stats")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "download":
        download()
    elif args.command == "probe":
        probe(args.limit)
    elif args.command == "clean":
        clean()
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
