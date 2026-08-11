"""XiaChuFang Recipe Corpus → recipe KB。

官方研究语料包含 1,520,327 条中文食谱。归档内成员虽名为 ``.json``，实际为
JSONL；本模块直接从 ZIP 流式读取，只保留 100,000 个唯一菜名，不解压 2GB
中间文件。备用源为公开中文菜谱数据集；英文菜谱翻译只作最后预案，并须记录
Fable 或 GPT 5.6 Sol 的实际模型来源。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import requests

from skillchain import config
from skillchain.data import normalize_line_separators
from skillchain.data._kb_source_lock import (
    KBSourceLock,
    KBSourceLockError,
    SourceFileSnapshot,
    VerifiedKBSourceLock,
    canonical_json_bytes,
    citation_uri,
    load_verified_kb_source_lock,
    prepare_formal_output_parent,
    publish_staged_file_create_only,
    snapshot_regular_file,
    verify_source_snapshot,
)
from skillchain.data.kb_catalog import KBEntryV2
from skillchain.data.wiki_zh import console_safe

RAW_ARCHIVE = config.DATA_DIR / "raw" / "recipes" / "xiachufang_recipe_corpus_full.zip"
MEMBER = "recipe_corpus_full.json"
OUTPUT_FILE = config.DATA_DIR / "kb" / "recipes.jsonl"
DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1HDUHNDHUxKJilfKr3D_9lA7K_RtJ6ewY&export=download&confirm=t"
)
EXPECTED_SIZE = 663_982_342
SOURCE_DATASET = "xiachufang-recipe-corpus-full"
SOURCE_REVISION = "xiachufang-recipe-corpus-full-unpinned"
SOURCE_URI = "https://drive.google.com/open?id=1HDUHNDHUxKJilfKr3D_9lA7K_RtJ6ewY"
LICENSE_ID = "unknown-unverified"


@dataclass(frozen=True)
class RecipeCleanReport:
    kept: int
    duplicates: int
    incomplete: int
    malformed: int


@dataclass(frozen=True)
class _RecipeEntrySource:
    lock: KBSourceLock
    lock_sha256: str


def iter_records(archive_path: Path, member: str = MEMBER) -> Iterator[dict]:
    """逐行读取 ZIP 内 JSONL，格式错误时报告成员行号。"""
    for _, _, record in _iter_records_with_identity(
        archive_path, label=str(archive_path), member=member
    ):
        yield record


def _iter_records_with_identity(
    archive_source: Path | BinaryIO,
    *,
    label: str,
    member: str = MEMBER,
) -> Iterator[tuple[int, str, dict]]:
    with zipfile.ZipFile(archive_source) as archive:
        matching = [info for info in archive.infolist() if info.filename == member]
        if len(matching) != 1:
            raise ValueError(f"{label} must contain {member} exactly once")
        member_info = matching[0]
        unix_mode = member_info.external_attr >> 16
        if member_info.is_dir() or stat.S_ISLNK(unix_mode):
            raise ValueError(f"{label}!{member} must be a regular ZIP member")
        with archive.open(member_info) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(
                        f"{label}!{member}:{line_number} JSON 格式错误"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"{label}!{member}:{line_number} 必须是 JSON 对象")
                yield line_number, hashlib.sha256(line).hexdigest(), record


def _canonical_title(record: dict) -> str:
    dish = normalize_line_separators(str(record.get("dish", ""))).strip()
    name = normalize_line_separators(str(record.get("name", ""))).strip()
    return dish if dish and dish.casefold() != "unknown" else name


def recipe_to_kb_entry(record: dict) -> dict:
    """Convert a development record to an explicitly unverified KB entry."""

    return _recipe_to_kb_entry(record, source=None)


def _recipe_to_kb_entry(
    record: dict,
    *,
    source: _RecipeEntrySource | None,
    record_number: int | None = None,
    raw_record_sha256: str | None = None,
) -> dict:
    title = _canonical_title(record)
    name = normalize_line_separators(str(record["name"])).strip()
    description = normalize_line_separators(str(record.get("description", ""))).strip()
    ingredients = [
        normalize_line_separators(str(item)).strip()
        for item in record["recipeIngredient"]
        if str(item).strip()
    ]
    instructions = [
        normalize_line_separators(str(item)).strip()
        for item in record["recipeInstructions"]
        if str(item).strip()
    ]
    sections = [f"菜谱：{name}"]
    if description:
        sections.append(f"简介：{description}")
    sections.append("食材：\n" + "\n".join(f"- {item}" for item in ingredients))
    sections.append(
        "步骤：\n"
        + "\n".join(f"{index}. {item}" for index, item in enumerate(instructions, 1))
    )
    normalized = re.sub(r"\s+", "", title).casefold()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    text = "\n\n".join(sections)
    author = normalize_line_separators(str(record.get("author", ""))).strip()
    if source is None:
        source_dataset = SOURCE_DATASET
        source_revision = SOURCE_REVISION
        source_record_id = f"canonical-title-{digest}"
        source_uri = f"{SOURCE_URI}#entry-{digest}"
        license_id = LICENSE_ID
        attribution = author or None
        verification_status = "unverified"
    else:
        if record_number is None or raw_record_sha256 is None:
            raise AssertionError("formal recipe entry is missing its record identity")
        source_dataset = source.lock.source_dataset
        source_revision = source.lock.source_revision
        source_record_id = (
            f"{MEMBER}:line:{record_number}:fields:recipe-core-v1:sha256:"
            f"{raw_record_sha256}"
        )
        source_uri = citation_uri(
            source.lock.source_uri,
            source_sha256=source.lock.source_sha256,
            source_lock_sha256=source.lock_sha256,
            locator=(
                f"{MEMBER}-line-{record_number}-fields-recipe-core-v1-sha256-"
                f"{raw_record_sha256}"
            ),
        )
        license_id = source.lock.license_id
        attribution = source.lock.attribution
        if author:
            attribution = f"{attribution}; record author: {author}"
        verification_status = "source_verified"
    return {
        "schema_version": 2,
        "entry_id": f"recipe-{digest}",
        "title": title,
        "text": text,
        "kind": "recipe",
        "origin": "dump",
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_record_id": source_record_id,
        "source_uri": source_uri,
        "license_id": license_id,
        "attribution": attribution,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "entity_group_id": f"recipe-title-{digest}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": verification_status,
    }


def clean_recipes(
    archive_path: Path,
    output_path: Path,
    *,
    max_entries: int = 100_000,
) -> RecipeCleanReport:
    """Diagnostic cleaner; output is always explicitly unverified."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as target:
            report = _clean_recipe_records(
                _iter_records_with_identity(
                    archive_path, label=str(archive_path), member=MEMBER
                ),
                target,
                max_entries=max_entries,
                source=None,
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def clean_recipes_formal(
    archive_path: Path,
    output_path: Path,
    *,
    source_lock_path: Path,
    expected_source_lock_sha256: str,
    max_entries: int = 100_000,
) -> RecipeCleanReport:
    """Create source-verified recipe entries from externally locked ZIP bytes."""

    if max_entries <= 0:
        raise KBSourceLockError(
            "formal recipe cleaning requires a positive max_entries"
        )

    verified_lock = load_verified_kb_source_lock(
        source_lock_path,
        expected_lock_sha256=expected_source_lock_sha256,
        expected_adapter_id="xiachufang-recipe-zip-v1",
    )
    source_snapshot = snapshot_regular_file(archive_path, "recipe raw archive")
    if source_snapshot.sha256 != verified_lock.lock.source_sha256:
        raise KBSourceLockError(
            "recipe raw archive does not match the external source lock"
        )
    output_path = prepare_formal_output_parent(output_path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_path.parent,
            prefix=f".{output_path.name}.formal-",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            report = _clean_recipe_records(
                _iter_records_with_identity(
                    io.BytesIO(source_snapshot.content),
                    label=str(source_snapshot.path),
                    member=MEMBER,
                ),
                target,
                max_entries=max_entries,
                source=_RecipeEntrySource(
                    lock=verified_lock.lock,
                    lock_sha256=verified_lock.lock_sha256,
                ),
            )
            target.flush()
            os.fsync(target.fileno())
        if report.kept == 0:
            raise KBSourceLockError("formal recipe cleaning produced no KB entries")
        _verify_formal_inputs(verified_lock, source_snapshot)
        publish_staged_file_create_only(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def _clean_recipe_records(
    records: Iterator[tuple[int, str, dict]],
    target: BinaryIO,
    *,
    max_entries: int,
    source: _RecipeEntrySource | None,
) -> RecipeCleanReport:
    seen_titles: set[str] = set()
    kept = duplicates = incomplete = malformed = 0

    for record_number, raw_record_sha256, record in records:
        required = ("name", "recipeIngredient", "recipeInstructions")
        if any(key not in record for key in required):
            malformed += 1
            continue
        ingredients = record["recipeIngredient"]
        instructions = record["recipeInstructions"]
        if not isinstance(ingredients, list) or not isinstance(instructions, list):
            malformed += 1
            continue
        ingredients = [
            normalize_line_separators(str(item)).strip()
            for item in ingredients
            if str(item).strip()
        ]
        instructions = [
            normalize_line_separators(str(item)).strip()
            for item in instructions
            if str(item).strip()
        ]
        title = _canonical_title(record)
        if not title or len(ingredients) < 2 or not instructions:
            incomplete += 1
            continue
        normalized = re.sub(r"\s+", "", title).casefold()
        if normalized in seen_titles:
            duplicates += 1
            continue
        seen_titles.add(normalized)
        normalized_record = dict(record)
        normalized_record["recipeIngredient"] = ingredients
        normalized_record["recipeInstructions"] = instructions
        row = _recipe_to_kb_entry(
            normalized_record,
            source=source,
            record_number=record_number,
            raw_record_sha256=raw_record_sha256,
        )
        entry = KBEntryV2.model_validate(row)
        target.write(canonical_json_bytes(entry))
        kept += 1
        if kept >= max_entries:
            break
    return RecipeCleanReport(kept, duplicates, incomplete, malformed)


def _verify_formal_inputs(
    verified_lock: VerifiedKBSourceLock, source_snapshot: SourceFileSnapshot
) -> None:
    verify_source_snapshot(source_snapshot, "recipe raw archive")
    verify_source_snapshot(verified_lock.lock_snapshot, "KB source lock")


def download() -> None:
    RAW_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    have = RAW_ARCHIVE.stat().st_size if RAW_ARCHIVE.exists() else 0
    if have == EXPECTED_SIZE:
        print(f"[skip] {RAW_ARCHIVE.name} 已完整 ({have} bytes)")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(
        DOWNLOAD_URL, headers=headers, stream=True, timeout=120
    ) as response:
        response.raise_for_status()
        mode = "ab" if have and response.status_code == 206 else "wb"
        with RAW_ARCHIVE.open(mode) as target:
            for chunk in response.iter_content(1 << 20):
                target.write(chunk)
    actual = RAW_ARCHIVE.stat().st_size
    if actual != EXPECTED_SIZE:
        raise RuntimeError(
            f"下载不完整：期望 {EXPECTED_SIZE} 字节，实际 {actual}；可重跑续传"
        )


def probe(limit: int = 5) -> None:
    for index, record in enumerate(iter_records(RAW_ARCHIVE)):
        if index >= limit:
            break
        types = {key: type(value).__name__ for key, value in record.items()}
        preview = {
            key: (value[:3] if isinstance(value, list) else str(value)[:120])
            for key, value in record.items()
        }
        print(console_safe(f"record[{index}] fields={types} value={preview}"))


def clean() -> RecipeCleanReport:
    report = clean_recipes(RAW_ARCHIVE, OUTPUT_FILE)
    print(
        f"kept={report.kept} duplicates={report.duplicates} "
        f"incomplete={report.incomplete} malformed={report.malformed}"
    )
    return report


def stats() -> None:
    count = total_chars = 0
    with OUTPUT_FILE.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            count += 1
            total_chars += len(row["text"])
    average = total_chars / count if count else 0
    print(f"entries={count}")
    print(f"average_chars={average:.1f}")
    print('origins={"dump":' + str(count) + "}")
    print(f"output={OUTPUT_FILE}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--limit", type=int, default=5)
    subparsers.add_parser("clean")
    formal_parser = subparsers.add_parser("clean-formal")
    formal_parser.add_argument("--source-lock", type=Path, required=True)
    formal_parser.add_argument("--expected-source-lock-sha256", required=True)
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
    elif args.command == "clean-formal":
        report = clean_recipes_formal(
            RAW_ARCHIVE,
            OUTPUT_FILE,
            source_lock_path=args.source_lock,
            expected_source_lock_sha256=args.expected_source_lock_sha256,
        )
        print(
            f"kept={report.kept} duplicates={report.duplicates} "
            f"incomplete={report.incomplete} malformed={report.malformed}"
        )
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
