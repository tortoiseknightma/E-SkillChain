"""合并 Utility 食物图与文档图，生成统一的 800 张查询图目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from skillchain import config
from skillchain.data import (
    discard_staging_directory,
    prepare_staging_directory,
    publish_staged_directory,
)
from skillchain.data.muge import write_preview
from skillchain.data.wikimedia_documents import normalize_candidate as normalize_commons

FOOD_SOURCE = config.DATA_DIR / "clean" / "query_images" / "utility_food"
DOCUMENT_SOURCE = config.DATA_DIR / "clean" / "query_images" / "utility_docs"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "utility"


@dataclass(frozen=True)
class UtilityReport:
    food: int
    documents: int
    total: int
    image_paths: tuple[Path, ...]


def _validate_provenance(row: dict, kind: str) -> None:
    if not str(row.get("source") or "") or not str(row.get("license") or ""):
        raise ValueError(f"{kind} provenance 缺少 source/license")
    if kind == "food":
        if (
            row["source"] != "isia-food500"
            or not str(row.get("source_member") or "").startswith(
                "ISIA_Food500/images/"
            )
            or urlsplit(str(row.get("source_page") or "")).hostname != "123.57.42.89"
        ):
            raise ValueError("food provenance 来源不符合 ISIA Food-500 schema")
        if row["license"] != "not-specified-by-publisher":
            raise ValueError("food provenance 未知许可状态不得改写为其他许可证")
        if (
            not str(row.get("license_note") or "")
            or row.get("distribution") != "local-only-not-redistributed"
        ):
            raise ValueError("food provenance 未披露未知许可与不分发约束")
    elif kind == "document":
        if row["source"] != "wikimedia-commons":
            raise ValueError("document provenance 来源不符合 Wikimedia Commons schema")
        try:
            normalize_commons(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"document provenance 许可或来源无效: {error}") from error
    else:
        raise ValueError(f"未知 Utility provenance kind: {kind}")


def _load_manifest(source: Path, expected: int, kind: str) -> list[dict]:
    manifest = source / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise RuntimeError(f"{source.name} 数量不符：期望 {expected}，实际 {len(rows)}")
    for row in rows:
        _validate_provenance(row, kind)
    return rows


def compose_sources(
    food_source: Path,
    document_source: Path,
    destination: Path,
    *,
    food_count: int = 600,
    document_count: int = 200,
) -> UtilityReport:
    food_source = Path(food_source)
    document_source = Path(document_source)
    destination = Path(destination)
    groups = (
        ("food", food_source, _load_manifest(food_source, food_count, "food")),
        (
            "document",
            document_source,
            _load_manifest(document_source, document_count, "document"),
        ),
    )
    staging = prepare_staging_directory(destination)
    image_names: list[str] = []
    seen_names: set[str] = set()
    try:
        with (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for kind, source, rows in groups:
                for source_row in rows:
                    image_name = str(source_row["image"])
                    if Path(image_name).name != image_name:
                        raise ValueError(f"非法图片文件名: {image_name}")
                    if image_name in seen_names:
                        raise RuntimeError(f"Utility 图片文件名冲突: {image_name}")
                    source_image = source / image_name
                    if not source_image.is_file():
                        raise FileNotFoundError(source_image)
                    with Image.open(source_image) as opened:
                        opened.verify()
                    shutil.copy2(source_image, staging / image_name)
                    seen_names.add(image_name)
                    image_names.append(image_name)
                    row = dict(source_row)
                    row["utility_kind"] = kind
                    manifest.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        expected_total = food_count + document_count
        if len(image_names) != expected_total:
            raise RuntimeError(
                f"Utility staging 数量不符：期望 {expected_total}，实际 {len(image_names)}"
            )
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return UtilityReport(
        food=food_count,
        documents=document_count,
        total=len(image_names),
        image_paths=tuple(destination / name for name in image_names),
    )


def compose() -> UtilityReport:
    report = compose_sources(FOOD_SOURCE, DOCUMENT_SOURCE, DESTINATION)
    preview_images = (
        *report.image_paths[:5],
        *report.image_paths[report.food : report.food + 4],
    )
    write_preview(preview_images, config.DATA_DIR / "clean" / "preview_utility.jpg")
    print(f"food={report.food} documents={report.documents} total={report.total}")
    return report


def stats() -> None:
    rows = [
        json.loads(line)
        for line in (DESTINATION / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kinds = Counter(row["utility_kind"] for row in rows)
    print(f"query_images={len(rows)}")
    print(f"kinds={json.dumps(dict(sorted(kinds.items())), ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compose", "stats"))
    args = parser.parse_args()
    if args.command == "compose":
        compose()
    else:
        stats()


if __name__ == "__main__":
    main()
