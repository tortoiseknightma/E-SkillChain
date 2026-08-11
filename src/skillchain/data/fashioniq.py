"""Resumable downloader for the frozen FashionIQ image URL inventory."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Callable, Iterable, Literal
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.data.source_lock import (
    RequiredSourceLock,
    load_and_verify_required_source_lock,
    load_required_source_lock,
)
from skillchain.data.source_review import VerifiedSourceReviewLedger
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)
from skillchain.tools.serialization import (
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
)

FROZEN_METADATA_REVISION = "3bb39d7d9a024d92f1b0fd0929b38d73dff24a0a"
CATEGORIES = ("dress", "shirt", "toptee")
ALLOWED_HOSTS = {"ecx.images-amazon.com", "g-ecx.images-amazon.com"}
ASIN_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_ROOT = config.DATA_DIR / "raw" / "fashioniq"
DEFAULT_METADATA_ROOT = DEFAULT_ROOT / f"fashion-iq-metadata-{FROZEN_METADATA_REVISION}"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "images"
DEFAULT_STATE_ROOT = DEFAULT_ROOT / "download-state"
DEFAULT_EXCLUSIONS = DEFAULT_STATE_ROOT / "exclusions.jsonl"
MAX_IMAGE_BYTES = 50 * 1024 * 1024
SOURCE_DATASET = "fashioniq"
ADAPTER_ID = "fashioniq-images-exact-lock-v1"
TRANSFORM_POLICY_VERSION = "fashioniq-original-byte-reference-v1"
LICENSE_ATTRIBUTION = (
    "FashionIQ academic-research dataset; original FashionIQ source URL retained"
)
REQUIRED_ADAPTER_PURPOSES = (
    "capability_gold",
    "product_gallery",
    "tool_gold",
)
REQUIRED_ADAPTER_PERMISSIONS = (
    "download_allowed",
    "local_embedding_allowed",
    "local_research_allowed",
    "public_demo_allowed",
    "remote_embedding_allowed",
)
_ASSET_FILE = "dataset-assets.jsonl"
_DISPOSITION_FILE = "asset-dispositions.jsonl"
_MANIFEST_FILE = "manifest.json"
_MAX_ADAPTER_METADATA_BYTES = 64 * 1024 * 1024
_MAX_LICENSE_EVIDENCE_BYTES = 2 * 1024 * 1024
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FashionIQProvenanceError(ValueError):
    """FashionIQ formal adapter inputs or outputs are not exactly authorized."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class FashionIQAssetDisposition(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    source_record_id: str = Field(min_length=1)
    category: Literal["dress", "shirt", "toptee"]
    asin: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    source_url: str = Field(min_length=1)
    disposition: Literal["accepted_asset", "excluded_locked"]
    reason: str = Field(min_length=1)

    @field_validator("source_record_id", "source_url", "reason")
    @classmethod
    def validate_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("FashionIQ disposition text must be canonical")
        return value

    @model_validator(mode="after")
    def validate_identity(self):
        if self.source_record_id != f"{self.category}:{self.asin}":
            raise ValueError(
                "FashionIQ disposition source_record_id must bind category and ASIN"
            )
        return self


class FashionIQAdapterFile(_StrictFrozenModel):
    path: Literal["asset-dispositions.jsonl", "dataset-assets.jsonl"]
    sha256: Sha256
    bytes: int = Field(ge=1)
    rows: int = Field(ge=1)


class FashionIQAdapterManifest(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    adapter_id: Literal["fashioniq-images-exact-lock-v1"] = ADAPTER_ID
    source_dataset: Literal["fashioniq"] = SOURCE_DATASET
    source_revision: str = Field(min_length=1)
    source_lock_sha256: Sha256
    license_id: str = Field(min_length=1)
    license_evidence_sha256: Sha256
    source_review_policy_sha256: Sha256
    source_review_ledger_sha256: Sha256
    source_review_record_sha256: Sha256
    authorized_purposes: tuple[
        Literal["capability_gold", "product_gallery", "tool_gold"], ...
    ]
    authorized_permissions: tuple[
        Literal[
            "download_allowed",
            "local_embedding_allowed",
            "local_research_allowed",
            "public_demo_allowed",
            "remote_embedding_allowed",
        ],
        ...,
    ]
    remote_embedding_allowed: bool
    redistribution_allowed: bool
    public_demo_allowed: bool
    inventory_total: int = Field(ge=1)
    accepted_assets: int = Field(ge=1)
    excluded_locked: int = Field(ge=1)
    files: tuple[FashionIQAdapterFile, ...]

    @field_validator("authorized_purposes", "authorized_permissions", mode="before")
    @classmethod
    def coerce_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("files", mode="before")
    @classmethod
    def coerce_files(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_closure(self):
        if self.authorized_purposes != REQUIRED_ADAPTER_PURPOSES:
            raise ValueError("FashionIQ manifest purpose binding is incomplete")
        if self.authorized_permissions != REQUIRED_ADAPTER_PERMISSIONS:
            raise ValueError("FashionIQ manifest permission binding is incomplete")
        if self.redistribution_allowed:
            raise ValueError(
                "FashionIQ manifest cannot authorize redistribution"
            )
        if self.accepted_assets + self.excluded_locked != self.inventory_total:
            raise ValueError("FashionIQ manifest does not close the inventory")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted({_ASSET_FILE, _DISPOSITION_FILE})):
            raise ValueError("FashionIQ manifest file set is invalid")
        return self


@dataclass(frozen=True)
class FashionIQAdapterBuildResult:
    output_dir: Path
    drafts: tuple[DatasetAssetDraft, ...]
    dispositions: tuple[FashionIQAssetDisposition, ...]
    manifest: FashionIQAdapterManifest
    manifest_sha256: str


@dataclass(frozen=True)
class VerifiedFashionIQAdapterBundle:
    root: Path
    drafts: tuple[DatasetAssetDraft, ...]
    dispositions: tuple[FashionIQAssetDisposition, ...]
    manifest: FashionIQAdapterManifest
    manifest_sha256: str


@dataclass(frozen=True)
class ImageRecord:
    category: str
    asin: str
    source_url: str


@dataclass(frozen=True)
class ImageResult:
    record: ImageRecord
    status: str
    error: str | None = None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _validate_source_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"unexpected FashionIQ image host: {parsed.hostname!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "FashionIQ inventory URL contains unexpected credentials or parameters"
        )


def transport_url(source_url: str) -> str:
    """Upgrade the frozen inventory URL to TLS without changing its identity."""
    _validate_source_url(source_url)
    parsed = urlsplit(source_url)
    return urlunsplit(("https", parsed.netloc, parsed.path, "", ""))


def read_inventory(metadata_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen: set[tuple[str, str]] = set()
    image_url_root = Path(metadata_root) / "image_url"
    for category in CATEGORIES:
        inventory = image_url_root / f"asin2url.{category}.txt"
        if not inventory.is_file():
            raise FileNotFoundError(f"missing FashionIQ inventory: {inventory}")
        for line_number, raw_line in enumerate(
            inventory.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            fields = raw_line.split()
            if len(fields) != 2:
                raise ValueError(f"{inventory}:{line_number}: expected ASIN and URL")
            asin, source_url = fields
            if not ASIN_PATTERN.fullmatch(asin):
                raise ValueError(f"{inventory}:{line_number}: unsafe ASIN {asin!r}")
            _validate_source_url(source_url)
            key = (category, asin)
            if key in seen:
                raise ValueError(f"{inventory}:{line_number}: duplicate record {key!r}")
            seen.add(key)
            records.append(ImageRecord(category, asin, source_url))
    return records


def read_exclusions(
    path: Path,
    records: Iterable[ImageRecord],
) -> dict[tuple[str, str], dict[str, str]]:
    if not Path(path).is_file():
        return {}
    inventory = {(record.category, record.asin): record for record in records}
    exclusions: dict[tuple[str, str], dict[str, str]] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict) or row.get("decision") != "exclude":
            raise ValueError(f"{path}:{line_number}: expected an exclusion decision")
        key = (str(row.get("category", "")), str(row.get("asin", "")))
        record = inventory.get(key)
        if record is None:
            raise ValueError(
                f"{path}:{line_number}: exclusion is not in inventory: {key}"
            )
        if row.get("source_url") != record.source_url:
            raise ValueError(f"{path}:{line_number}: source URL differs from inventory")
        if key in exclusions:
            raise ValueError(f"{path}:{line_number}: duplicate exclusion: {key}")
        exclusions[key] = {str(key): str(value) for key, value in row.items()}
    return exclusions


def is_valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _publish_bytes(content: bytes, destination: Path) -> None:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"invalid image size: {len(content)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        with temporary.open("wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        if not is_valid_image(temporary):
            raise ValueError("response is not a decodable image")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_image(url: str, *, attempts: int = 4) -> bytes:
    headers = {"User-Agent": "ECommerceSkillChain-FashionIQ/1.0"}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=(15, 60),
                stream=True,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and not (
                    content_type.startswith("image/")
                    or content_type == "application/octet-stream"
                ):
                    raise ValueError(f"unexpected content type: {content_type}")
                expected = int(response.headers.get("Content-Length", 0) or 0)
                if expected > MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_IMAGE_BYTES:
                        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


def _download_one(
    record: ImageRecord,
    *,
    output_root: Path,
    replacement_root: Path,
    fetcher: Callable[[str], bytes],
) -> ImageResult:
    destination = output_root / record.category / f"{record.asin}.jpg"
    if is_valid_image(destination):
        return ImageResult(record, "skipped")
    destination.unlink(missing_ok=True)

    replacement = replacement_root / f"{record.asin}.jpg"
    try:
        if replacement.is_file():
            _publish_bytes(replacement.read_bytes(), destination)
            return ImageResult(record, "replacement")
        _publish_bytes(fetcher(transport_url(record.source_url)), destination)
        return ImageResult(record, "downloaded")
    except Exception as error:  # one bad remote object must not stop the inventory
        return ImageResult(record, "failed", f"{type(error).__name__}: {error}")


class DownloadLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def __enter__(self) -> "DownloadLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
        }
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    current = json.loads(self.path.read_text(encoding="utf-8"))
                    same_host = current.get("hostname") == socket.gethostname()
                    alive = same_host and self._pid_is_alive(int(current["pid"]))
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    alive = False
                if alive:
                    raise RuntimeError(
                        f"FashionIQ downloader already owns {self.path} "
                        f"(pid={current['pid']})"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
                json.dump(value, target, sort_keys=True)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.path.unlink(missing_ok=True)


def download_inventory(
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    *,
    workers: int = 6,
    progress_every: int = 100,
    exclusions_path: Path | None = None,
    fetcher: Callable[[str], bytes] = _fetch_image,
) -> dict[str, object]:
    if workers < 1 or workers > 64:
        raise ValueError("workers must be between 1 and 64")
    records = read_inventory(metadata_root)
    exclusions_path = (
        Path(exclusions_path)
        if exclusions_path is not None
        else Path(state_root) / "exclusions.jsonl"
    )
    exclusions = read_exclusions(exclusions_path, records)
    active_records = [
        record for record in records if (record.category, record.asin) not in exclusions
    ]
    replacement_root = Path(metadata_root) / "image_url" / "broken_links"
    output_root = Path(output_root)
    state_root = Path(state_root)
    counts = {
        "downloaded": 0,
        "replacement": 0,
        "skipped": 0,
        "failed": 0,
        "excluded": len(exclusions),
    }
    failures: list[dict[str, str]] = []
    started_at = time.time()

    def snapshot(status: str, completed: int) -> dict[str, object]:
        return {
            "status": status,
            "pid": os.getpid(),
            "metadata_revision": FROZEN_METADATA_REVISION,
            "total": len(records),
            "completed_this_run": completed,
            "counts": dict(counts),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "output_root": str(output_root.resolve()),
            "exclusions_path": str(exclusions_path.resolve()),
            "updated_at": time.time(),
        }

    with DownloadLock(state_root / "download.lock"):
        for category, asin in exclusions:
            if is_valid_image(output_root / category / f"{asin}.jpg"):
                raise RuntimeError(
                    f"excluded FashionIQ record has a valid image: {category}/{asin}"
                )
        completed = len(exclusions)
        _atomic_json(state_root / "summary.json", snapshot("running", completed))
        batch_size = max(workers * 8, 32)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(active_records), batch_size):
                futures = [
                    executor.submit(
                        _download_one,
                        record,
                        output_root=output_root,
                        replacement_root=replacement_root,
                        fetcher=fetcher,
                    )
                    for record in active_records[offset : offset + batch_size]
                ]
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    counts[result.status] += 1
                    if result.error is not None:
                        failures.append(
                            {
                                **asdict(result.record),
                                "error": result.error,
                            }
                        )
                    if completed % progress_every == 0:
                        _atomic_json(
                            state_root / "summary.json", snapshot("running", completed)
                        )
                        _atomic_jsonl(state_root / "failures.jsonl", failures)
                        print(
                            f"progress={completed}/{len(records)} "
                            f"downloaded={counts['downloaded']} "
                            f"replacement={counts['replacement']} "
                            f"skipped={counts['skipped']} failed={counts['failed']}",
                            flush=True,
                        )
        _atomic_jsonl(state_root / "failures.jsonl", failures)
        final_status = (
            "complete_with_failures"
            if failures
            else "complete_with_exclusions"
            if exclusions
            else "complete"
        )
        report = snapshot(final_status, completed)
        _atomic_json(state_root / "summary.json", report)
        print(
            f"status={final_status} total={len(records)} "
            f"downloaded={counts['downloaded']} replacement={counts['replacement']} "
            f"skipped={counts['skipped']} failed={counts['failed']} "
            f"excluded={counts['excluded']}",
            flush=True,
        )
        return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_repeated_failures(
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> dict[str, object]:
    state_root = Path(state_root)
    output_root = Path(output_root)
    summary_path = state_root / "summary.json"
    failures_path = state_root / "failures.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete_with_failures" or summary.get(
        "completed_this_run"
    ) != summary.get("total"):
        raise RuntimeError("current FashionIQ run is not a complete failed pass")
    baselines = sorted(
        state_root.glob("complete-run-*.failures.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not baselines:
        raise RuntimeError("no previous complete FashionIQ failure baseline exists")
    baseline_path = baselines[0]

    records = read_inventory(metadata_root)
    inventory = {(record.category, record.asin): record for record in records}

    def load_failures(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        values: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(raw_line)
            key = (str(row["category"]), str(row["asin"]))
            if key in values:
                raise RuntimeError(f"{path}:{line_number}: duplicate failure: {key}")
            record = inventory.get(key)
            if record is None or row.get("source_url") != record.source_url:
                raise RuntimeError(f"{path}:{line_number}: failure identity mismatch")
            values[key] = {str(field): str(value) for field, value in row.items()}
        return values

    current = load_failures(failures_path)
    baseline = load_failures(baseline_path)
    if set(current) != set(baseline):
        raise RuntimeError("the last two complete FashionIQ failure sets differ")
    if not current:
        raise RuntimeError("there are no repeated failures to exclude")
    for category, asin in current:
        if is_valid_image(output_root / category / f"{asin}.jpg"):
            raise RuntimeError(f"failure now has a valid image: {category}/{asin}")

    decided_at = datetime.now(UTC).isoformat()
    reason = "owner_directed_exclusion_after_two_identical_complete_retries"
    exclusion_rows = [
        {
            "category": category,
            "asin": asin,
            "source_url": current[(category, asin)]["source_url"],
            "decision": "exclude",
            "reason": reason,
            "last_error": current[(category, asin)].get("error", ""),
            "decided_at": decided_at,
        }
        for category, asin in sorted(current)
    ]
    exclusions_path = state_root / "exclusions.jsonl"
    _atomic_jsonl(exclusions_path, exclusion_rows)
    exclusion_summary = {
        "schema_version": 1,
        "status": "complete_with_exclusions",
        "decision": reason,
        "decided_at": decided_at,
        "inventory_total": len(records),
        "accepted_images": len(records) - len(current),
        "excluded": len(current),
        "current_failures_file": failures_path.name,
        "current_failures_sha256": _file_sha256(failures_path),
        "baseline_failures_file": baseline_path.name,
        "baseline_failures_sha256": _file_sha256(baseline_path),
        "failure_sets_identical": True,
        "formal_ready": False,
        "remaining_gates": [
            "license_version_and_use_scope",
            "external_source_lock",
            "selection_disposition_review",
            "asset_catalog",
        ],
    }
    _atomic_json(state_root / "exclusions-summary.json", exclusion_summary)

    final_summary = dict(summary)
    final_summary["status"] = "complete_with_exclusions"
    final_summary["counts"] = {
        "downloaded": 0,
        "replacement": 0,
        "skipped": len(records) - len(current),
        "failed": 0,
        "excluded": len(current),
    }
    final_summary["completed_this_run"] = len(records)
    final_summary["exclusions_path"] = str(exclusions_path.resolve())
    final_summary["exclusions_summary_path"] = str(
        (state_root / "exclusions-summary.json").resolve()
    )
    final_summary["updated_at"] = time.time()
    _atomic_json(summary_path, final_summary)
    return exclusion_summary


def verify_finalized_acquisition(
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
) -> dict[str, object]:
    """Verify acquisition coverage including the downloader's mutable summary."""

    return _verify_finalized_acquisition_coverage(
        metadata_root,
        output_root,
        state_root,
        require_runtime_summary=True,
    )


def _verify_finalized_acquisition_coverage(
    metadata_root: Path,
    output_root: Path,
    state_root: Path,
    *,
    require_runtime_summary: bool,
) -> dict[str, object]:
    records = read_inventory(metadata_root)
    inventory = {(record.category, record.asin) for record in records}
    state_root = Path(state_root)
    output_root = Path(output_root)
    exclusion_summary = json.loads(
        (state_root / "exclusions-summary.json").read_text(encoding="utf-8")
    )
    exclusions = set(read_exclusions(state_root / "exclusions.jsonl", records))
    images: set[tuple[str, str]] = set()
    for category in CATEGORIES:
        category_root = output_root / category
        if not category_root.is_dir():
            continue
        for path in category_root.glob("*.jpg"):
            key = (category, path.stem)
            if key in images:
                raise RuntimeError(f"duplicate FashionIQ image identity: {key}")
            images.add(key)
    partials = list(output_root.rglob("*.part")) if output_root.is_dir() else []
    if partials:
        raise RuntimeError(f"FashionIQ still has {len(partials)} partial files")
    if images & exclusions:
        raise RuntimeError("FashionIQ accepted and excluded sets overlap")
    if images | exclusions != inventory:
        missing = inventory - images - exclusions
        unexpected = (images | exclusions) - inventory
        raise RuntimeError(
            f"FashionIQ finalized coverage differs: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if (
        require_runtime_summary
        and json.loads(
            (state_root / "summary.json").read_text(encoding="utf-8")
        ).get("status")
        != "complete_with_exclusions"
    ):
        raise RuntimeError("FashionIQ summary is not finalized with exclusions")
    if exclusion_summary.get("status") != "complete_with_exclusions":
        raise RuntimeError(
            "FashionIQ exclusion summary is not finalized with exclusions"
        )
    expected_counts = {
        "inventory_total": len(inventory),
        "accepted_images": len(images),
        "excluded": len(exclusions),
    }
    for key, value in expected_counts.items():
        if exclusion_summary.get(key) != value:
            raise RuntimeError(f"FashionIQ exclusion summary differs at {key}")
    return {
        "status": "complete_with_exclusions",
        **expected_counts,
        "formal_ready": False,
        "exclusions_sha256": _file_sha256(state_root / "exclusions.jsonl"),
    }


def build_fashioniq_image_adapter(
    *,
    raw_root: Path | str,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    license_evidence_path: Path | str,
    expected_license_evidence_sha256: str,
    source_review_ledger: VerifiedSourceReviewLedger,
    output_dir: Path | str,
) -> FashionIQAdapterBuildResult:
    """Publish exact-lock-bound FashionIQ asset drafts and full dispositions.

    The generic source lock is re-hashed before and after materialization.  The
    owner review must match that exact lock, revision, licence evidence, purposes,
    and local-use permissions.  Frozen exclusions are represented in the
    disposition ledger but can never become ``DatasetAssetDraft`` rows.
    """

    if not isinstance(source_review_ledger, VerifiedSourceReviewLedger):
        raise TypeError("FashionIQ adapter requires a verified source review ledger")
    raw_root = Path(raw_root).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"FashionIQ adapter output already exists: {output_dir}")
    pinned_lock = load_required_source_lock(
        source_lock_path,
        expected_lock_file_sha256=expected_source_lock_sha256,
    )
    metadata_root, images_root, state_root = _validate_fashioniq_lock_layout(
        pinned_lock, raw_root
    )
    verified_lock = load_and_verify_required_source_lock(
        source_lock_path,
        raw_root,
        expected_lock_file_sha256=expected_source_lock_sha256,
    )
    if verified_lock.lock != pinned_lock:
        raise FashionIQProvenanceError(
            "FashionIQ source lock changed between preflight and RAW verification"
        )
    lock = verified_lock.lock
    evidence_bytes, evidence = _load_license_evidence(
        license_evidence_path,
        expected_sha256=expected_license_evidence_sha256,
    )
    if evidence.get("source_id") != SOURCE_DATASET:
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence targets a different source"
        )
    license_id = evidence.get("license_id")
    if not isinstance(license_id, str) or not license_id.strip():
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence has no canonical license_id"
        )
    if license_id != license_id.strip():
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence license_id is not canonical"
        )
    approval = source_review_ledger.require_approval(
        SOURCE_DATASET,
        source_revision=lock.source_revision,
        source_lock_sha256=verified_lock.lock_file_sha256,
        license_id=license_id,
        license_evidence_sha256=sha256_bytes(evidence_bytes),
        purposes=REQUIRED_ADAPTER_PURPOSES,
        permissions=REQUIRED_ADAPTER_PERMISSIONS,
    )
    if approval.permissions.redistribution_allowed:
        raise FashionIQProvenanceError(
            "FashionIQ adapter forbids redistribution"
        )

    acquisition = _verify_finalized_acquisition_coverage(
        metadata_root,
        images_root,
        state_root,
        require_runtime_summary=False,
    )
    records = sorted(
        read_inventory(metadata_root), key=lambda row: (row.category, row.asin)
    )
    exclusions = read_exclusions(state_root / "exclusions.jsonl", records)
    drafts: list[DatasetAssetDraft] = []
    dispositions: list[FashionIQAssetDisposition] = []
    images_scope = _scope_by_id(lock, "accepted_images")
    assert images_scope.root is not None
    for record in records:
        source_record_id = f"{record.category}:{record.asin}"
        exclusion = exclusions.get((record.category, record.asin))
        if exclusion is not None:
            reason = exclusion.get("reason", "")
            if not reason or reason != reason.strip():
                raise FashionIQProvenanceError(
                    f"FashionIQ exclusion has no canonical reason: {source_record_id}"
                )
            dispositions.append(
                FashionIQAssetDisposition(
                    source_record_id=source_record_id,
                    category=record.category,
                    asin=record.asin,
                    source_url=record.source_url,
                    disposition="excluded_locked",
                    reason=reason,
                )
            )
            continue
        image_path = images_root / record.category / f"{record.asin}.jpg"
        if not is_valid_image(image_path):
            raise FashionIQProvenanceError(
                f"FashionIQ accepted image is not decodable: {source_record_id}"
            )
        local_path = (
            Path(images_scope.root) / record.category / f"{record.asin}.jpg"
        ).as_posix()
        drafts.append(
            DatasetAssetDraft(
                source_dataset=SOURCE_DATASET,
                source_revision=lock.source_revision,
                source_record_id=source_record_id,
                transform_policy_version=TRANSFORM_POLICY_VERSION,
                local_path=local_path,
                product_id=f"fashioniq:{record.asin}",
                derivation_parent_asset_ids=[],
                license_id=license_id,
                source_url=record.source_url,
                attribution=LICENSE_ATTRIBUTION,
                cloud_upload_allowed=approval.permissions.remote_embedding_allowed,
                public_demo_allowed=approval.permissions.public_demo_allowed,
            )
        )
        dispositions.append(
            FashionIQAssetDisposition(
                source_record_id=source_record_id,
                category=record.category,
                asin=record.asin,
                source_url=record.source_url,
                disposition="accepted_asset",
                reason="accepted_by_exact_source_lock_and_owner_review",
            )
        )
    if len(drafts) != acquisition["accepted_images"]:
        raise FashionIQProvenanceError(
            "FashionIQ accepted draft count differs from finalized acquisition"
        )
    if len(dispositions) != acquisition["inventory_total"]:
        raise FashionIQProvenanceError(
            "FashionIQ dispositions do not close the frozen inventory"
        )

    draft_rows = tuple(drafts)
    disposition_rows = tuple(dispositions)
    draft_bytes = canonical_jsonl_bytes(draft_rows)
    disposition_bytes = canonical_jsonl_bytes(disposition_rows)
    files = tuple(
        sorted(
            (
                FashionIQAdapterFile(
                    path=_ASSET_FILE,
                    sha256=sha256_bytes(draft_bytes),
                    bytes=len(draft_bytes),
                    rows=len(draft_rows),
                ),
                FashionIQAdapterFile(
                    path=_DISPOSITION_FILE,
                    sha256=sha256_bytes(disposition_bytes),
                    bytes=len(disposition_bytes),
                    rows=len(disposition_rows),
                ),
            ),
            key=lambda item: item.path,
        )
    )
    approval_bytes = canonical_json_bytes(approval.model_dump(mode="json"))
    manifest = FashionIQAdapterManifest(
        source_revision=lock.source_revision,
        source_lock_sha256=verified_lock.lock_file_sha256,
        license_id=license_id,
        license_evidence_sha256=sha256_bytes(evidence_bytes),
        source_review_policy_sha256=source_review_ledger.policy_file_sha256,
        source_review_ledger_sha256=source_review_ledger.ledger_file_sha256,
        source_review_record_sha256=sha256_bytes(approval_bytes),
        authorized_purposes=REQUIRED_ADAPTER_PURPOSES,
        authorized_permissions=REQUIRED_ADAPTER_PERMISSIONS,
        remote_embedding_allowed=approval.permissions.remote_embedding_allowed,
        redistribution_allowed=approval.permissions.redistribution_allowed,
        public_demo_allowed=approval.permissions.public_demo_allowed,
        inventory_total=len(disposition_rows),
        accepted_assets=len(draft_rows),
        excluded_locked=len(exclusions),
        files=files,
    )
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    manifest_sha256 = sha256_bytes(manifest_bytes)

    staging = new_staging_directory(output_dir)
    try:
        (staging / _ASSET_FILE).write_bytes(draft_bytes)
        (staging / _DISPOSITION_FILE).write_bytes(disposition_bytes)
        (staging / _MANIFEST_FILE).write_bytes(manifest_bytes)
        load_verified_fashioniq_adapter_bundle(
            staging, expected_manifest_sha256=manifest_sha256
        )
        refreshed = load_and_verify_required_source_lock(
            source_lock_path,
            raw_root,
            expected_lock_file_sha256=expected_source_lock_sha256,
        )
        if refreshed != verified_lock:
            raise FashionIQProvenanceError(
                "FashionIQ source lock identity changed during adapter build"
            )
        refreshed_evidence, _ = _load_license_evidence(
            license_evidence_path,
            expected_sha256=expected_license_evidence_sha256,
        )
        if refreshed_evidence != evidence_bytes:
            raise FashionIQProvenanceError(
                "FashionIQ licence evidence changed during adapter build"
            )
        source_review_ledger.require_approval(
            SOURCE_DATASET,
            source_revision=lock.source_revision,
            source_lock_sha256=verified_lock.lock_file_sha256,
            license_id=license_id,
            license_evidence_sha256=sha256_bytes(evidence_bytes),
            purposes=REQUIRED_ADAPTER_PURPOSES,
            permissions=REQUIRED_ADAPTER_PERMISSIONS,
        )
        atomic_publish_new_directory(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return FashionIQAdapterBuildResult(
        output_dir=output_dir,
        drafts=draft_rows,
        dispositions=disposition_rows,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def load_verified_fashioniq_adapter_bundle(
    root: Path | str,
    *,
    expected_manifest_sha256: str,
) -> VerifiedFashionIQAdapterBundle:
    """Verify a create-only adapter bundle under an external manifest digest."""

    _require_sha256(expected_manifest_sha256, "FashionIQ adapter manifest")
    root = Path(root).absolute()
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise FashionIQProvenanceError(
            "FashionIQ adapter bundle root is unavailable"
        ) from error
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or _path_is_junction(root)
    ):
        raise FashionIQProvenanceError(
            "FashionIQ adapter bundle root must be a real directory"
        )
    manifest_bytes = read_stable_regular_file(
        root / _MANIFEST_FILE,
        label="FashionIQ adapter manifest",
        max_bytes=_MAX_ADAPTER_METADATA_BYTES,
    )
    if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise FashionIQProvenanceError(
            "FashionIQ adapter manifest external digest mismatch"
        )
    try:
        manifest = FashionIQAdapterManifest.model_validate_json(
            manifest_bytes, strict=True
        )
    except ValueError as error:
        raise FashionIQProvenanceError(
            "FashionIQ adapter manifest schema is invalid"
        ) from error
    if manifest_bytes != canonical_json_bytes(manifest.model_dump(mode="json")):
        raise FashionIQProvenanceError(
            "FashionIQ adapter manifest must be canonical JSON"
        )
    expected_files = {_MANIFEST_FILE, *(item.path for item in manifest.files)}
    actual_files: set[str] = set()
    for path in root.iterdir():
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or _path_is_junction(path)
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise FashionIQProvenanceError(
                "FashionIQ adapter bundle contains a non-regular payload"
            )
        actual_files.add(path.name)
    if actual_files != expected_files:
        raise FashionIQProvenanceError(
            "FashionIQ adapter bundle file set differs from manifest"
        )
    payloads: dict[str, bytes] = {}
    for descriptor in manifest.files:
        content = read_stable_regular_file(
            root / descriptor.path,
            label=f"FashionIQ adapter payload {descriptor.path}",
            max_bytes=_MAX_ADAPTER_METADATA_BYTES,
        )
        if (
            len(content) != descriptor.bytes
            or sha256_bytes(content) != descriptor.sha256
        ):
            raise FashionIQProvenanceError(
                f"FashionIQ adapter payload drifted: {descriptor.path}"
            )
        payloads[descriptor.path] = content
    drafts = _parse_canonical_rows(
        payloads[_ASSET_FILE], DatasetAssetDraft, "FashionIQ dataset assets"
    )
    dispositions = _parse_canonical_rows(
        payloads[_DISPOSITION_FILE],
        FashionIQAssetDisposition,
        "FashionIQ asset dispositions",
    )
    file_by_path = {item.path: item for item in manifest.files}
    if (
        file_by_path[_ASSET_FILE].rows != len(drafts)
        or file_by_path[_DISPOSITION_FILE].rows != len(dispositions)
        or manifest.accepted_assets != len(drafts)
        or manifest.inventory_total != len(dispositions)
    ):
        raise FashionIQProvenanceError(
            "FashionIQ adapter row counts differ from manifest"
        )
    if any(
        draft.source_dataset != SOURCE_DATASET
        or draft.source_revision != manifest.source_revision
        or draft.license_id != manifest.license_id
        or draft.cloud_upload_allowed != manifest.remote_embedding_allowed
        or draft.public_demo_allowed != manifest.public_demo_allowed
        for draft in drafts
    ):
        raise FashionIQProvenanceError(
            "FashionIQ adapter drafts differ from manifest authorization"
        )
    accepted_ids = {
        row.source_record_id
        for row in dispositions
        if row.disposition == "accepted_asset"
    }
    excluded_ids = {
        row.source_record_id
        for row in dispositions
        if row.disposition == "excluded_locked"
    }
    draft_ids = {row.source_record_id for row in drafts}
    disposition_by_id = {row.source_record_id: row for row in dispositions}
    if (
        len(accepted_ids) + len(excluded_ids) != len(dispositions)
        or accepted_ids & excluded_ids
        or len(draft_ids) != len(drafts)
        or draft_ids != accepted_ids
        or manifest.excluded_locked != len(excluded_ids)
    ):
        raise FashionIQProvenanceError(
            "FashionIQ excluded or accepted disposition closure is invalid"
        )
    for draft in drafts:
        disposition = disposition_by_id[draft.source_record_id]
        expected_path = (
            f"fashioniq/images/{disposition.category}/{disposition.asin}.jpg"
        )
        if (
            draft.local_path != expected_path
            or draft.product_id != f"fashioniq:{disposition.asin}"
            or draft.source_url != disposition.source_url
            or draft.transform_policy_version != TRANSFORM_POLICY_VERSION
            or draft.attribution != LICENSE_ATTRIBUTION
            or draft.derivation_parent_asset_ids
            or draft.derivation_parent_asset_id is not None
        ):
            raise FashionIQProvenanceError(
                "FashionIQ draft identity differs from accepted disposition"
            )
    return VerifiedFashionIQAdapterBundle(
        root=root,
        drafts=drafts,
        dispositions=dispositions,
        manifest=manifest,
        manifest_sha256=expected_manifest_sha256,
    )


def _validate_fashioniq_lock_layout(
    lock: RequiredSourceLock, raw_root: Path
) -> tuple[Path, Path, Path]:
    if lock.source_id != SOURCE_DATASET:
        raise FashionIQProvenanceError(
            "FashionIQ adapter received a source lock for another source"
        )
    expected_scope_ids = (
        "accepted_images",
        "annotations",
        "exclusion_ledger",
        "url_inventory",
    )
    scope_ids = tuple(scope.scope_id for scope in lock.artifact_scopes)
    if scope_ids != expected_scope_ids:
        raise FashionIQProvenanceError(
            "FashionIQ source lock does not contain the exact required scopes"
        )
    accepted = _scope_by_id(lock, "accepted_images")
    annotations = _scope_by_id(lock, "annotations")
    inventory = _scope_by_id(lock, "url_inventory")
    exclusions = _scope_by_id(lock, "exclusion_ledger")
    if (
        accepted.mode != "recursive_tree"
        or accepted.root != "fashioniq/images"
        or annotations.mode != "recursive_tree"
        or annotations.root is None
        or not annotations.root.startswith("fashioniq/fashion-iq-")
        or inventory.mode != "recursive_tree"
        or inventory.root is None
        or not inventory.root.startswith("fashioniq/fashion-iq-metadata-")
        or exclusions.mode != "explicit_files"
    ):
        raise FashionIQProvenanceError(
            "FashionIQ source lock scope layout is not adapter-compatible"
        )
    required_exclusion_names = {
        "exclusions-summary.json",
        "exclusions.jsonl",
        "failures.jsonl",
    }
    state_parents = {Path(path).parent.as_posix() for path in exclusions.paths}
    names = {Path(path).name for path in exclusions.paths}
    if state_parents != {
        "fashioniq/download-state"
    } or not required_exclusion_names.issubset(names):
        raise FashionIQProvenanceError(
            "FashionIQ source lock does not bind finalized exclusion state"
        )
    return (
        raw_root / Path(inventory.root),
        raw_root / Path(accepted.root),
        raw_root / "fashioniq" / "download-state",
    )


def _scope_by_id(lock: RequiredSourceLock, scope_id: str):
    for scope in lock.artifact_scopes:
        if scope.scope_id == scope_id:
            return scope
    raise FashionIQProvenanceError(f"FashionIQ source lock is missing {scope_id}")


def _load_license_evidence(
    path: Path | str, *, expected_sha256: str
) -> tuple[bytes, dict[str, object]]:
    _require_sha256(expected_sha256, "FashionIQ licence evidence")
    content = read_stable_regular_file(
        path,
        label="FashionIQ licence evidence",
        max_bytes=_MAX_LICENSE_EVIDENCE_BYTES,
    )
    if sha256_bytes(content) != expected_sha256:
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence external digest mismatch"
        )
    value = parse_canonical_json(content, label="FashionIQ licence evidence")
    if not isinstance(value, dict):
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence root must be an object"
        )
    if content != canonical_json_bytes(value):
        raise FashionIQProvenanceError(
            "FashionIQ licence evidence must be canonical JSON"
        )
    return content, value


def _parse_canonical_rows(content: bytes, model, label: str):
    rows = parse_canonical_jsonl(content, label=label)
    try:
        values = tuple(
            model.model_validate_json(canonical_json_bytes(row), strict=True)
            for row in rows
        )
    except ValueError as error:
        raise FashionIQProvenanceError(f"{label} schema is invalid") from error
    if not values or content != canonical_jsonl_bytes(values):
        raise FashionIQProvenanceError(f"{label} must be nonempty canonical JSONL")
    return values


def _require_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FashionIQProvenanceError(
            f"{label} SHA-256 must be 64 lowercase hex characters"
        )


def _path_is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--exclusions", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = download_inventory(
        args.metadata_root,
        args.output_root,
        args.state_root,
        workers=args.workers,
        progress_every=args.progress_every,
        exclusions_path=args.exclusions,
    )
    if report["status"] not in {"complete", "complete_with_exclusions"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
