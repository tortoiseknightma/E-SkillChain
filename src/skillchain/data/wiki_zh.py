"""中文维基百科质量过滤版 → encyclopedia KB。

实际格式（2026-07-10 已探查）：顶层 JSON 数组；每项仅含 ``completion`` 与
``source``。``completion`` 有时以首行标题开头，有时直接是导语，因此标题需按
确定性规则提取。

数据源备用顺序：pleisto/wikipedia-cn-20230720-filtered（当前）→ 0xDing
同名镜像 → fjcanyue/wikipedia-zh-cn 全量 dump。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

import ijson

from skillchain import config
from skillchain.data import normalize_line_separators
from skillchain.data._hf import download_file
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

RAW_FILE = config.DATA_DIR / "raw" / "wiki_zh" / "wikipedia-cn-20230720-filtered.json"
OUTPUT_FILE = config.DATA_DIR / "kb" / "encyclopedia.jsonl"
SOURCE_DATASET = "pleisto/wikipedia-cn-20230720-filtered"
SOURCE_REVISION = "20230720-filtered-unpinned"
SOURCE_URI = "https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered"
LICENSE_ID = "unknown-unverified"

# 面向视觉电商百科的宽召回词表。Phase 2 的 BM25 负责精排，这里只剔除明显无关条目。
RELEVANCE_KEYWORDS = (
    "植物",
    "动物",
    "昆虫",
    "鸟类",
    "鱼类",
    "哺乳",
    "爬行",
    "两栖",
    "物种",
    "科植物",
    "属植物",
    "材料",
    "材质",
    "纤维",
    "纺织",
    "面料",
    "棉",
    "麻",
    "丝绸",
    "羊毛",
    "皮革",
    "木材",
    "金属",
    "合金",
    "塑料",
    "玻璃",
    "陶瓷",
    "宝石",
    "矿物",
    "品牌",
    "商标",
    "服装",
    "鞋",
    "珠宝",
    "首饰",
    "化妆品",
    "护肤",
    "食品",
    "食材",
    "香料",
    "水果",
    "蔬菜",
)


@dataclass(frozen=True)
class WikiCleanReport:
    kept: int
    duplicates: int
    irrelevant: int
    too_short: int
    malformed: int


@dataclass(frozen=True)
class _WikiEntrySource:
    lock: KBSourceLock
    lock_sha256: str


def download() -> None:
    download_file(
        "pleisto/wikipedia-cn-20230720-filtered",
        "wikipedia-cn-20230720-filtered.json",
        RAW_FILE,
    )


def iter_records(path: Path) -> Iterator[dict]:
    """流式读取 JSON 数组或 JSONL，不把 524MB dump 整体载入内存。"""
    path = Path(path)
    with path.open("rb") as source:
        yield from _iter_records_stream(source, str(path))


def _iter_records_stream(source: BinaryIO, label: str) -> Iterator[dict]:
    first = b""
    while not first:
        first = source.read(1)
        if not first:
            return
        if first.isspace():
            first = b""
    source.seek(0)
    if first == b"[":
        yield from ijson.items(source, "item")
        return
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{label}:{line_number} 不是合法 JSON") from error


def derive_title(completion: str) -> str:
    """从首行标题或百科导语中确定性提取短标题。"""
    text = completion.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and len(lines[0]) <= 80:
        return lines[0]
    lead = lines[0]
    candidates = []
    for marker in ("（", "(", "，", "。"):
        position = lead.find(marker)
        if 0 < position <= 80:
            candidates.append(position)
    for match in re.finditer(r"(?:是|为)(?:一|中国|位于|指|由)", lead):
        if 0 < match.start() <= 80:
            candidates.append(match.start())
    end = min(candidates) if candidates else min(len(lead), 80)
    return lead[:end].strip(" ：:，,。")


def console_safe(value: str, encoding: str | None = None) -> str:
    """让包含终端编码不支持字符的探查文本仍可打印。"""
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def clean_wikipedia(
    source_path: Path,
    output_path: Path,
    *,
    keywords: Iterable[str] = RELEVANCE_KEYWORDS,
    min_chars: int = 80,
    max_entries: int = 200_000,
) -> WikiCleanReport:
    """Diagnostic cleaner; output is always explicitly unverified."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as target:
            report = _clean_wikipedia_records(
                enumerate(iter_records(source_path), start=1),
                target,
                keywords=keywords,
                min_chars=min_chars,
                max_entries=max_entries,
                source=None,
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def clean_wikipedia_formal(
    source_path: Path,
    output_path: Path,
    *,
    source_lock_path: Path,
    expected_source_lock_sha256: str,
    keywords: Iterable[str] = RELEVANCE_KEYWORDS,
    min_chars: int = 80,
    max_entries: int = 200_000,
) -> WikiCleanReport:
    """Create source-verified entries from externally locked immutable bytes.

    The raw dump and canonical source-lock files are snapshotted before parsing
    and rechecked before a create-only publish.  The caller-owned expected lock
    digest is mandatory; there is no boolean or model-object formal shortcut.
    """

    if min_chars <= 0 or max_entries <= 0:
        raise KBSourceLockError(
            "formal wiki cleaning requires positive min_chars and max_entries"
        )

    verified_lock = load_verified_kb_source_lock(
        source_lock_path,
        expected_lock_sha256=expected_source_lock_sha256,
        expected_adapter_id="wiki-zh-filtered-v1",
    )
    source_snapshot = snapshot_regular_file(source_path, "wiki raw source")
    if source_snapshot.sha256 != verified_lock.lock.source_sha256:
        raise KBSourceLockError(
            "wiki raw source does not match the external source lock"
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
            report = _clean_wikipedia_records(
                enumerate(
                    _iter_records_stream(
                        io.BytesIO(source_snapshot.content), str(source_snapshot.path)
                    ),
                    start=1,
                ),
                target,
                keywords=keywords,
                min_chars=min_chars,
                max_entries=max_entries,
                source=_WikiEntrySource(
                    lock=verified_lock.lock,
                    lock_sha256=verified_lock.lock_sha256,
                ),
            )
            target.flush()
            os.fsync(target.fileno())
        if report.kept == 0:
            raise KBSourceLockError("formal wiki cleaning produced no KB entries")
        _verify_formal_inputs(verified_lock, source_snapshot)
        publish_staged_file_create_only(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def _clean_wikipedia_records(
    records: Iterable[tuple[int, dict]],
    target: BinaryIO,
    *,
    keywords: Iterable[str],
    min_chars: int,
    max_entries: int,
    source: _WikiEntrySource | None,
) -> WikiCleanReport:
    keywords = tuple(keywords)
    seen_titles: set[str] = set()
    kept = duplicates = irrelevant = too_short = malformed = 0

    for record_number, record in records:
        completion = record.get("completion") if isinstance(record, dict) else None
        if not isinstance(completion, str):
            malformed += 1
            continue
        text = normalize_line_separators(completion).strip()
        if len(text) < min_chars:
            too_short += 1
            continue
        if keywords and not any(keyword in text for keyword in keywords):
            irrelevant += 1
            continue
        title = derive_title(text)
        if not title:
            malformed += 1
            continue
        normalized_title = re.sub(r"\s+", "", title).casefold()
        if normalized_title in seen_titles:
            duplicates += 1
            continue
        seen_titles.add(normalized_title)
        digest = hashlib.sha1(normalized_title.encode("utf-8")).hexdigest()[:16]
        row = _wiki_entry(
            record,
            record_number=record_number,
            digest=digest,
            title=title,
            text=text,
            source=source,
        )
        entry = KBEntryV2.model_validate(row)
        target.write(canonical_json_bytes(entry))
        kept += 1
        if kept >= max_entries:
            break
    return WikiCleanReport(kept, duplicates, irrelevant, too_short, malformed)


def _wiki_entry(
    record: dict,
    *,
    record_number: int,
    digest: str,
    title: str,
    text: str,
    source: _WikiEntrySource | None,
) -> dict:
    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if source is None:
        raw_source = record.get("source")
        attribution = (
            normalize_line_separators(str(raw_source)).strip()
            if raw_source is not None
            else None
        )
        source_dataset = SOURCE_DATASET
        source_revision = SOURCE_REVISION
        source_record_id = f"filtered-title-{digest}"
        source_uri = f"{SOURCE_URI}#entry-{digest}"
        license_id = LICENSE_ID
        verification_status = "unverified"
    else:
        raw_record_sha256 = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
        source_record_id = (
            f"item:{record_number}:field:completion:canonical-sha256:"
            f"{raw_record_sha256}"
        )
        source_dataset = source.lock.source_dataset
        source_revision = source.lock.source_revision
        source_uri = citation_uri(
            source.lock.source_uri,
            source_sha256=source.lock.source_sha256,
            source_lock_sha256=source.lock_sha256,
            locator=(
                f"item-{record_number}-field-completion-canonical-sha256-"
                f"{raw_record_sha256}"
            ),
        )
        license_id = source.lock.license_id
        attribution = source.lock.attribution
        verification_status = "source_verified"
    return {
        "schema_version": 2,
        "entry_id": f"wiki-{digest}",
        "title": title,
        "text": text,
        "kind": "encyclopedia",
        "origin": "dump",
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_record_id": source_record_id,
        "source_uri": source_uri,
        "license_id": license_id,
        "attribution": attribution or None,
        "content_sha256": content_sha256,
        "entity_group_id": f"wiki-title-{digest}",
        "near_duplicate_cluster_id": None,
        "derivation_parent_entry_ids": [],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_prompt_sha256": None,
        "verification_status": verification_status,
    }


def _verify_formal_inputs(
    verified_lock: VerifiedKBSourceLock, source_snapshot: SourceFileSnapshot
) -> None:
    verify_source_snapshot(source_snapshot, "wiki raw source")
    verify_source_snapshot(verified_lock.lock_snapshot, "KB source lock")


def probe(limit: int = 5) -> None:
    for index, record in enumerate(iter_records(RAW_FILE)):
        if index >= limit:
            break
        types = {key: type(value).__name__ for key, value in record.items()}
        preview = {key: str(value)[:120] for key, value in record.items()}
        print(console_safe(f"record[{index}] fields={types} value={preview}"))


def clean() -> WikiCleanReport:
    report = clean_wikipedia(RAW_FILE, OUTPUT_FILE)
    print(
        f"kept={report.kept} duplicates={report.duplicates} irrelevant={report.irrelevant} "
        f"too_short={report.too_short} malformed={report.malformed}"
    )
    return report


def stats() -> None:
    count = total_chars = 0
    origins: dict[str, int] = {}
    with OUTPUT_FILE.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            count += 1
            total_chars += len(row["text"])
            origins[row["origin"]] = origins.get(row["origin"], 0) + 1
    average = total_chars / count if count else 0
    print(f"entries={count}")
    print(f"average_chars={average:.1f}")
    print(f"origins={json.dumps(origins, ensure_ascii=False, sort_keys=True)}")
    print(f"output={OUTPUT_FILE}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    probe_parser = subparsers.add_parser("probe", aliases=["inspect"])
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
    elif args.command in {"probe", "inspect"}:
        probe(args.limit)
    elif args.command == "clean":
        clean()
    elif args.command == "clean-formal":
        report = clean_wikipedia_formal(
            RAW_FILE,
            OUTPUT_FILE,
            source_lock_path=args.source_lock,
            expected_source_lock_sha256=args.expected_source_lock_sha256,
        )
        print(
            f"kept={report.kept} duplicates={report.duplicates} "
            f"irrelevant={report.irrelevant} too_short={report.too_short} "
            f"malformed={report.malformed}"
        )
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
