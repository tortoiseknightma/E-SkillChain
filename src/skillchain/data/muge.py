"""MUGE 电商图文检索数据集：下载、探查、清洗与统计。

实际格式（2026-07-10 已探查）：

* ``*_texts.jsonl``：``text_id``、``text``、``image_ids``；
* ``*_imgs.tsv``：``image_id<TAB>base64-jpeg``。

MUGE 不提供商品类目；清洗时 ``category_l1`` 明确写为 ``unknown``。只有独立分类器
可以后续补齐这些 MUGE 标签；MEP-3M 只新增自带类目的独立商品，不能为现有 MUGE
商品补标签。

数据源备用顺序：hf-mirror 的 JieJieJ/muge 分卷包（当前）→ 天池 MUGE
（需登录）→ MEP-3M 承担商品库。上游不可用时应报出 URL 与恢复方式。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Iterable, Iterator, Literal, TextIO
from urllib.parse import urlsplit

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain.data import (
    discard_staging_directory,
    prepare_staging_directory,
    publish_staged_directory_and_file,
    publish_staged_directory,
    recover_directory_and_file_publish,
)
from skillchain.data.fashion_queries import select_exact_match_products
from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.synthesis.store import atomic_create_file

RAW_DIR = config.DATA_DIR / "raw" / "muge"
EXTRACT_DIR = RAW_DIR / "extracted"
CLEAN_DIR = config.DATA_DIR / "clean"

_MIRROR = "https://hf-mirror.com/datasets/JieJieJ/muge/resolve/main"
_PARTS = ["MUGE1.z01", "MUGE1.z02", "MUGE1.z03", "MUGE1.z04", "MUGE1.zip"]
_PROVENANCE_FILE = "muge-provenance.jsonl"
_DIAGNOSTIC_MARKER = "muge-diagnostic.json"
_EXACT_QUERY_MARKER = "muge-exact-query-diagnostic.json"
_TRANSFORM_POLICY_VERSION = "muge-exif-rgb-jpeg-q90-optimize-v1"
_LICENSE_REVIEW_POLICY_VERSION = "muge-human-license-review-v1"
_SELECTION_POLICY_VERSION = "muge-source-order-selection-v1"
_RETAINED_GALLERY_POLICY_VERSION = "muge-retained-gallery-dependency-v1"
_DISPOSITION_POLICY_VERSION = "muge-deterministic-disposition-review-v1"
_VERIFIED_HANDLE_TOKEN = object()
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PRODUCT_SCHEMA = pa.schema(
    [
        ("product_id", pa.string()),
        ("title", pa.string()),
        ("category_l1", pa.string()),
        ("category_l2", pa.string()),
        ("category_l3", pa.string()),
        ("ocr_text", pa.string()),
        ("image_path", pa.string()),
        ("source", pa.string()),
    ]
)


class MUGEProvenanceError(ValueError):
    """Raised when MUGE provenance is absent, mutable, or contradictory."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MUGEPermissionMatrix(_StrictFrozenModel):
    """Conservative permissions; a local claim cannot enable remote/public use."""

    local_noncommercial_research_allowed: Literal[True] = True
    local_noncommercial_embedding_allowed: Literal[True] = True
    remote_embedding_allowed: Literal[False] = False
    cloud_upload_allowed: Literal[False] = False
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False


class MUGESourceLock(_StrictFrozenModel):
    """Externally supplied identity and usage policy for one immutable MUGE dump."""

    schema_version: Literal[1] = 1
    source_dataset: Literal["muge"] = "muge"
    source_revision: str
    texts_sha256: Sha256
    images_sha256: Sha256
    license_id: str
    license_evidence_sha256: Sha256
    license_evidence_url: str
    source_url: str
    attribution: str | None = None
    cloud_upload_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        value = value.strip()
        normalized = value.casefold()
        mutable = {"main", "master", "head", "latest", "unknown", "unpinned"}
        if (
            not value
            or normalized in mutable
            or normalized.endswith(("/main", "/master", "/head", "/latest"))
        ):
            raise ValueError("source_revision must identify an immutable revision")
        return value

    @field_validator("license_id")
    @classmethod
    def validate_license(cls, value: str) -> str:
        value = value.strip()
        if not value or value.casefold() in {
            "unknown",
            "unknown-unverified",
            "none",
            "unlicensed",
            "tbd",
        }:
            raise ValueError("license_id must be explicit and verified")
        return value

    @field_validator("source_url", "license_evidence_url")
    @classmethod
    def validate_source_url(cls, value: str, info) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if (
            not value
            or not parsed.scheme
            or any(character.isspace() for character in value)
        ):
            raise ValueError(f"{info.field_name} must be an absolute URI")
        if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname is None:
            raise ValueError(f"HTTP {info.field_name} must contain a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{info.field_name} must not contain credentials")
        mutable_segments = {"main", "master", "head", "latest", "unknown"}
        if any(
            segment.casefold() in mutable_segments
            for segment in parsed.path.split("/")
            if segment
        ):
            raise ValueError(f"{info.field_name} must not point at a mutable revision")
        return value

    @field_validator("attribution")
    @classmethod
    def validate_attribution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("attribution must not be blank")
        return value


class MUGELicenseReview(_StrictFrozenModel):
    """Externally pinned human review of the exact lock and licence bytes."""

    schema_version: Literal[1] = 1
    review_policy_version: Literal["muge-human-license-review-v1"] = (
        _LICENSE_REVIEW_POLICY_VERSION
    )
    source_lock_sha256: Sha256
    source_revision: str
    license_id: str
    license_evidence_sha256: Sha256
    decision: Literal["approved_local_noncommercial_research_only"]
    permissions: MUGEPermissionMatrix = Field(default_factory=MUGEPermissionMatrix)
    reviewer_kind: Literal["human"]
    reviewer_id: str
    reviewed_at: str

    @field_validator("source_revision", "license_id", "reviewer_id")
    @classmethod
    def validate_review_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        if info.field_name == "reviewer_id":
            return _validate_accountable_human(value, "reviewer_id")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
            raise ValueError("reviewed_at must be a second-precision UTC timestamp")
        return value


def _validate_accountable_human(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    if re.search(
        r"(?:^|[-_. ])(?:llm|gpt|chatgpt|claude|qwen|gemini|model)(?:$|[-_. ])",
        value.casefold(),
    ):
        raise ValueError(f"{field_name} must identify an accountable human")
    return value


def _validate_utc_timestamp(value: str, field_name: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a valid second-precision UTC timestamp"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{field_name} must be a valid second-precision UTC timestamp")
    return value


class MUGESelectionPlan(_StrictFrozenModel):
    """Human-owned policy registered against source bytes before formal cleaning."""

    schema_version: Literal[1] = 1
    selection_policy_version: Literal["muge-source-order-selection-v1"] = (
        _SELECTION_POLICY_VERSION
    )
    source_dataset: Literal["muge"] = "muge"
    source_lock_sha256: Sha256
    license_review_sha256: Sha256
    source_revision: str
    source_texts_sha256: Sha256
    source_images_sha256: Sha256
    candidate_definition: Literal[
        "first_n_unique_text_referenced_images_in_locked_text_order"
    ] = "first_n_unique_text_referenced_images_in_locked_text_order"
    candidate_count: int = Field(gt=0)
    candidate_universe_sha256: Sha256
    retained_gallery_policy_version: Literal["muge-retained-gallery-dependency-v1"] = (
        _RETAINED_GALLERY_POLICY_VERSION
    )
    retained_gallery_row_count: int = Field(ge=0)
    retained_gallery_manifest_sha256: Sha256
    minimum_image_side: int = Field(gt=0)
    transform_policy_version: Literal["muge-exif-rgb-jpeg-q90-optimize-v1"] = (
        _TRANSFORM_POLICY_VERSION
    )
    duplicate_policy: Literal[
        "normalized_jpeg_phash_first_seen_in_locked_image_order"
    ] = "normalized_jpeg_phash_first_seen_in_locked_image_order"
    disposition_policy_version: Literal["muge-deterministic-disposition-review-v1"] = (
        _DISPOSITION_POLICY_VERSION
    )
    owner_kind: Literal["human"]
    owner_id: str
    preregistered_at: str

    @field_validator("source_revision")
    @classmethod
    def validate_source_revision(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_revision must not be blank")
        return value

    @field_validator("owner_id")
    @classmethod
    def validate_owner_id(cls, value: str) -> str:
        return _validate_accountable_human(value, "owner_id")

    @field_validator("preregistered_at")
    @classmethod
    def validate_preregistered_at(cls, value: str) -> str:
        return _validate_utc_timestamp(value, "preregistered_at")


MUGEDispositionReason = Literal[
    "accepted_by_preregistered_policy",
    "missing_source_image",
    "invalid_source_image",
    "below_preregistered_minimum_side",
    "normalized_perceptual_duplicate",
]


class MUGECandidateDisposition(_StrictFrozenModel):
    source_record_id: str = Field(pattern=r"^image:[0-9]+$")
    product_id: str = Field(pattern=r"^muge-[0-9]+$")
    disposition: Literal["accepted", "rejected"]
    reason: MUGEDispositionReason

    @model_validator(mode="after")
    def validate_identity_and_reason(self):
        image_id = self.source_record_id.removeprefix("image:")
        if self.product_id != f"muge-{image_id}":
            raise ValueError(
                "candidate product_id must map directly from source_record_id"
            )
        accepted_reason = self.reason == "accepted_by_preregistered_policy"
        if accepted_reason != (self.disposition == "accepted"):
            raise ValueError("candidate disposition and reason are inconsistent")
        return self


class MUGEDispositionLedger(_StrictFrozenModel):
    """External human-reviewed, complete decision ledger for the locked universe."""

    schema_version: Literal[1] = 1
    disposition_policy_version: Literal["muge-deterministic-disposition-review-v1"] = (
        _DISPOSITION_POLICY_VERSION
    )
    selection_plan_sha256: Sha256
    source_lock_sha256: Sha256
    license_review_sha256: Sha256
    source_revision: str
    candidate_universe_sha256: Sha256
    candidate_count: int = Field(gt=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    dispositions: tuple[MUGECandidateDisposition, ...]
    reviewer_kind: Literal["human"]
    reviewer_id: str
    reviewed_at: str

    @field_validator("dispositions", mode="before")
    @classmethod
    def accept_json_dispositions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, value: str) -> str:
        return _validate_accountable_human(value, "reviewer_id")

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return _validate_utc_timestamp(value, "reviewed_at")

    @model_validator(mode="after")
    def validate_complete_canonical_partition(self):
        keys = [
            int(value.source_record_id.removeprefix("image:"))
            for value in self.dispositions
        ]
        if keys != sorted(set(keys)):
            raise ValueError(
                "candidate dispositions must be canonically sorted and unique"
            )
        if len(keys) != self.candidate_count:
            raise ValueError("every candidate requires exactly one disposition")
        accepted = sum(value.disposition == "accepted" for value in self.dispositions)
        rejected = sum(value.disposition == "rejected" for value in self.dispositions)
        if (accepted, rejected) != (self.accepted_count, self.rejected_count):
            raise ValueError("candidate disposition counts are inconsistent")
        if accepted + rejected != self.candidate_count:
            raise ValueError("accepted and rejected rows must partition the universe")
        return self


class MUGEAssetProvenance(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    asset_role: Literal["product_gallery_candidate"]
    product_id: str = Field(pattern=r"^muge-[0-9]+$")
    filename: str = Field(pattern=r"^muge-[0-9]+\.jpg$")
    source_dataset: Literal["muge"]
    source_revision: str
    source_record_id: str = Field(pattern=r"^image:[0-9]+$")
    source_lock_sha256: Sha256
    selection_plan_sha256: Sha256
    retained_gallery_manifest_sha256: Sha256
    retained_gallery_row_count: int = Field(ge=0)
    source_texts_sha256: Sha256
    source_images_sha256: Sha256
    source_image_sha256: Sha256
    output_asset_sha256: Sha256
    transform_policy_version: Literal["muge-exif-rgb-jpeg-q90-optimize-v1"]
    derivation_parent_asset_ids: tuple[str, ...] = ()
    license_id: str
    license_evidence_sha256: Sha256
    license_review_sha256: Sha256
    license_review_policy_version: Literal["muge-human-license-review-v1"] = (
        _LICENSE_REVIEW_POLICY_VERSION
    )
    license_evidence_url: str
    source_url: str
    attribution: str | None = None
    cloud_upload_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False

    @field_validator("source_revision", "license_id", "source_url")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance values must not be blank")
        return value

    @field_validator("derivation_parent_asset_ids")
    @classmethod
    def reject_unregistered_derivation_parents(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if value:
            raise ValueError(
                "raw MUGE source records are external records, not catalog asset parents"
            )
        return value

    @field_validator("derivation_parent_asset_ids", mode="before")
    @classmethod
    def coerce_json_parent_ids(cls, value):
        return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class _SourceSnapshot:
    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class _RetainedGalleryBinding:
    """Exact non-MUGE rows and image bytes that influence duplicate selection."""

    rows: tuple[dict, ...]
    row_count: int
    manifest_sha256: str
    fingerprints: tuple[str, ...]
    image_snapshots: tuple[_SourceSnapshot, ...]


@dataclass(frozen=True)
class VerifiedMUGESourceLock:
    """Canonical lock bytes matched to a caller-owned digest trust root."""

    lock: MUGESourceLock
    lock_sha256: str
    snapshot: _SourceSnapshot
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedMUGELicenseReview:
    review: MUGELicenseReview
    review_sha256: str
    snapshot: _SourceSnapshot
    evidence_snapshot: _SourceSnapshot
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedMUGESelectionPlan:
    """Plan bytes independently pinned and re-derived from locked source inputs."""

    plan: MUGESelectionPlan
    plan_sha256: str
    snapshot: _SourceSnapshot
    text_snapshot: _SourceSnapshot
    image_snapshot: _SourceSnapshot
    retained_gallery_parquet_path: Path
    candidate_image_ids: tuple[int, ...]
    source_lock: VerifiedMUGESourceLock
    license_review: VerifiedMUGELicenseReview
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedMUGESelection:
    """Plan plus complete reviewed disposition ledger, safe for formal export."""

    plan: VerifiedMUGESelectionPlan
    ledger: MUGEDispositionLedger
    ledger_sha256: str
    ledger_snapshot: _SourceSnapshot
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedMUGEDraftBundle:
    """Drafts re-derived from verified source, licence, provenance, and assets."""

    drafts: tuple[DatasetAssetDraft, ...]
    bundle_sha256: str
    snapshot: _SourceSnapshot
    source_lock: VerifiedMUGESourceLock
    license_review: VerifiedMUGELicenseReview
    selection: VerifiedMUGESelection
    _verification_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CleanReport:
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    missing_title: int
    image_paths: tuple[Path, ...]


def download() -> None:
    """断点续传全部分卷；可反复运行，完整分卷会跳过。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in _PARTS:
        url = f"{_MIRROR}/{name}"
        destination = RAW_DIR / name
        response = requests.head(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        remote_size = int(response.headers["Content-Length"])
        have = destination.stat().st_size if destination.exists() else 0
        if have == remote_size:
            print(f"[skip] {name} 已完整 ({have / 1e6:.0f} MB)")
            continue
        if have > remote_size:
            raise RuntimeError(f"{destination} 比远端文件更大，请人工检查后重试")
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"[get ] {name} {have / 1e6:.0f}/{remote_size / 1e6:.0f} MB")
        with requests.get(url, headers=headers, stream=True, timeout=120) as stream:
            stream.raise_for_status()
            mode = "ab" if have and stream.status_code == 206 else "wb"
            with destination.open(mode) as target:
                for chunk in stream.iter_content(chunk_size=1 << 20):
                    target.write(chunk)
        actual_size = destination.stat().st_size
        if actual_size != remote_size:
            raise RuntimeError(
                f"{name} 下载不完整：期望 {remote_size} 字节，实际 {actual_size} 字节"
            )
        print(f"[done] {name} {actual_size / 1e6:.0f} MB")


def extract(members: Iterable[str] | None = None) -> None:
    """用 7-Zip 直接解开 PKZIP 分卷，避免额外生成 2.6GB 拼接副本。"""
    executable = _find_7zip()
    archive = RAW_DIR / "MUGE1.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"缺少 {archive}；请先运行 download")
    selected = list(members or ["MUGE/*"])
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    command = [executable, "x", str(archive), *selected, f"-o{EXTRACT_DIR}", "-y"]
    subprocess.run(command, check=True)


def _find_7zip() -> str:
    candidates = [shutil.which(name) for name in ("7z", "7zz", "7za")]
    if os.name == "nt":
        candidates.append(r"C:\Program Files\7-Zip\7z.exe")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("未找到 7-Zip；安装 7-Zip 后重试 MUGE 分卷解压")


def read_text_records(path: Path) -> Iterator[dict]:
    """流式读取 MUGE 文本记录，并在格式漂移时给出精确行号。"""
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            expected = {"text_id", "text", "image_ids"}
            missing = expected.difference(record)
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段 {sorted(missing)}")
            if not isinstance(record["image_ids"], list):
                raise ValueError(f"{path}:{line_number} image_ids 必须是列表")
            yield record


def build_titles(text_paths: Iterable[Path], limit: int) -> dict[int, str]:
    """按稳定的文件/行顺序，为前 ``limit`` 个唯一图片选择首个查询文本作标题。"""
    titles: dict[int, str] = {}
    for path in text_paths:
        for record in read_text_records(path):
            title = str(record["text"]).strip() or "无标题商品"
            for raw_image_id in record["image_ids"]:
                image_id = int(raw_image_id)
                titles.setdefault(image_id, title)
                if len(titles) >= limit:
                    return titles
    return titles


def _iter_image_rows(source: TextIO) -> Iterator[tuple[int, str]]:
    for line_number, line in enumerate(source, start=1):
        try:
            raw_image_id, encoded = line.rstrip("\r\n").split("\t", maxsplit=1)
            yield int(raw_image_id), encoded
        except ValueError as error:
            raise ValueError(f"图片 TSV 第 {line_number} 行格式错误") from error


def _build_titles_from_snapshot(
    snapshot: _SourceSnapshot, limit: int
) -> dict[int, str]:
    try:
        lines = snapshot.content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise MUGEProvenanceError("locked MUGE texts source must be UTF-8") from error
    titles: dict[int, str] = {}
    seen_text_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_source_lock_constant,
            )
        except (json.JSONDecodeError, MUGEProvenanceError) as error:
            raise MUGEProvenanceError(
                f"locked MUGE texts source has invalid JSON at line {line_number}"
            ) from error
        if not isinstance(record, dict) or set(record) != {
            "text_id",
            "text",
            "image_ids",
        }:
            raise MUGEProvenanceError(
                f"locked MUGE texts source has invalid fields at line {line_number}"
            )
        if (
            not isinstance(record["text_id"], int)
            or isinstance(record["text_id"], bool)
            or record["text_id"] < 0
            or record["text_id"] in seen_text_ids
        ):
            raise MUGEProvenanceError(
                f"locked MUGE text_id must be a unique non-negative integer at line {line_number}"
            )
        seen_text_ids.add(record["text_id"])
        if not isinstance(record["text"], str):
            raise MUGEProvenanceError(
                f"locked MUGE text must be a string at line {line_number}"
            )
        if not isinstance(record["image_ids"], list):
            raise MUGEProvenanceError(
                f"locked MUGE image_ids must be a list at line {line_number}"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in record["image_ids"]
        ) or len(record["image_ids"]) != len(set(record["image_ids"])):
            raise MUGEProvenanceError(
                f"locked MUGE image_ids must be unique non-negative integers at line {line_number}"
            )
        title = record["text"].strip()
        if not title:
            raise MUGEProvenanceError(
                f"locked MUGE text must not be blank at line {line_number}"
            )
        for raw_image_id in record["image_ids"]:
            image_id = int(raw_image_id)
            titles.setdefault(image_id, title)
            if len(titles) >= limit:
                return titles
    return titles


def _write_normalized_jpeg(image: Image.Image, destination: Path) -> str:
    image.save(destination, format="JPEG", quality=90, optimize=True)
    with Image.open(destination) as normalized:
        normalized.load()
        return str(imagehash.phash(normalized.convert("RGB")))


def _normalize_product_row(row: dict) -> dict:
    return {field.name: row.get(field.name) for field in _PRODUCT_SCHEMA}


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _path_is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _reject_link_ancestors(path: Path, label: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            continue
        if _path_is_link_or_junction(current, metadata):
            raise MUGEProvenanceError(
                f"{label} must not traverse a symlink or junction"
            )


def _snapshot_regular_file(path: Path, label: str) -> _SourceSnapshot:
    path = Path(path).absolute()
    _reject_link_ancestors(path, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise MUGEProvenanceError(f"unable to inspect {label}") from error
    if _path_is_link_or_junction(path, before) or not stat.S_ISREG(before.st_mode):
        raise MUGEProvenanceError(f"{label} must be a regular non-symlink file")
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise MUGEProvenanceError(f"{label} changed before it could be read")
            content = source.read()
            after_open = os.fstat(source.fileno())
    except OSError as error:
        raise MUGEProvenanceError(f"unable to read {label}") from error
    if _file_identity(opened) != _file_identity(after_open):
        raise MUGEProvenanceError(f"{label} changed while it was read")
    try:
        after_path = path.lstat()
    except OSError as error:
        raise MUGEProvenanceError(f"{label} changed after it was read") from error
    if _file_identity(after_open) != _file_identity(after_path):
        raise MUGEProvenanceError(f"{label} changed while it was read")
    return _SourceSnapshot(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        identity=_file_identity(after_open),
    )


def _verify_snapshot(snapshot: _SourceSnapshot, label: str) -> None:
    current = _snapshot_regular_file(snapshot.path, label)
    if (
        current.identity != snapshot.identity
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise MUGEProvenanceError(f"{label} changed during MUGE cleaning")


def _reject_source_lock_constant(token: str) -> None:
    raise MUGEProvenanceError(f"invalid source-lock constant: {token}")


def load_verified_muge_source_lock(
    path: Path | str, *, expected_lock_sha256: str
) -> VerifiedMUGESourceLock:
    """Load canonical lock bytes under an independent caller-owned digest."""

    if not isinstance(expected_lock_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_lock_sha256
    ):
        raise MUGEProvenanceError(
            "formal MUGE use requires an external expected source-lock SHA-256"
        )
    snapshot = _snapshot_regular_file(Path(path), "MUGE source lock")
    if snapshot.sha256 != expected_lock_sha256:
        raise MUGEProvenanceError(
            "MUGE source lock does not match external expected digest"
        )
    try:
        json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_source_lock_constant,
        )
        lock = MUGESourceLock.model_validate_json(snapshot.content, strict=True)
    except Exception as error:
        raise MUGEProvenanceError(
            "MUGE source lock violates its strict schema"
        ) from error
    if snapshot.content != _canonical_json_bytes(lock.model_dump(mode="json")):
        raise MUGEProvenanceError("MUGE source lock must be canonical JSON")
    _verify_snapshot(snapshot, "MUGE source lock")
    return VerifiedMUGESourceLock(
        lock, snapshot.sha256, snapshot, _VERIFIED_HANDLE_TOKEN
    )


def load_verified_muge_license_review(
    path: Path | str,
    *,
    expected_review_sha256: str,
    source_lock: VerifiedMUGESourceLock,
    license_evidence_path: Path | str,
) -> VerifiedMUGELicenseReview:
    """Load an external human decision bound to exact lock and evidence bytes."""

    if (
        not isinstance(source_lock, VerifiedMUGESourceLock)
        or source_lock._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise MUGEProvenanceError(
            "formal MUGE licence review requires a verified source lock"
        )
    if not isinstance(expected_review_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_review_sha256
    ):
        raise MUGEProvenanceError(
            "formal MUGE use requires an external licence-review SHA-256"
        )
    evidence_snapshot = _snapshot_regular_file(
        Path(license_evidence_path), "MUGE license evidence"
    )
    if not evidence_snapshot.content.strip():
        raise MUGEProvenanceError("MUGE license evidence must not be empty")
    if evidence_snapshot.sha256 != source_lock.lock.license_evidence_sha256:
        raise MUGEProvenanceError(
            "MUGE license evidence does not match the external source lock"
        )
    snapshot = _snapshot_regular_file(Path(path), "MUGE human license review")
    if snapshot.sha256 != expected_review_sha256:
        raise MUGEProvenanceError(
            "MUGE license review does not match the external expected digest"
        )
    try:
        json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_source_lock_constant,
        )
        review = MUGELicenseReview.model_validate_json(snapshot.content, strict=True)
    except Exception as error:
        raise MUGEProvenanceError(
            "MUGE license review violates its strict schema"
        ) from error
    if snapshot.content != _canonical_json_bytes(review.model_dump(mode="json")):
        raise MUGEProvenanceError("MUGE license review must be canonical JSON")
    expected_binding = (
        source_lock.lock_sha256,
        source_lock.lock.source_revision,
        source_lock.lock.license_id,
        evidence_snapshot.sha256,
        MUGEPermissionMatrix(),
    )
    actual_binding = (
        review.source_lock_sha256,
        review.source_revision,
        review.license_id,
        review.license_evidence_sha256,
        review.permissions,
    )
    if actual_binding != expected_binding:
        raise MUGEProvenanceError(
            "MUGE license review is not bound to the verified source evidence"
        )
    _verify_snapshot(source_lock.snapshot, "MUGE source lock")
    _verify_snapshot(evidence_snapshot, "MUGE license evidence")
    _verify_snapshot(snapshot, "MUGE human license review")
    return VerifiedMUGELicenseReview(
        review=review,
        review_sha256=snapshot.sha256,
        snapshot=snapshot,
        evidence_snapshot=evidence_snapshot,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _refresh_verified_muge_source_evidence(
    source_lock: VerifiedMUGESourceLock,
    license_review: VerifiedMUGELicenseReview,
) -> tuple[VerifiedMUGESourceLock, VerifiedMUGELicenseReview]:
    if (
        not isinstance(source_lock, VerifiedMUGESourceLock)
        or source_lock._verification_token is not _VERIFIED_HANDLE_TOKEN
        or not isinstance(license_review, VerifiedMUGELicenseReview)
        or license_review._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise MUGEProvenanceError(
            "formal MUGE selection requires verified source and licence handles"
        )
    refreshed_lock = load_verified_muge_source_lock(
        source_lock.snapshot.path,
        expected_lock_sha256=source_lock.lock_sha256,
    )
    refreshed_review = load_verified_muge_license_review(
        license_review.snapshot.path,
        expected_review_sha256=license_review.review_sha256,
        source_lock=refreshed_lock,
        license_evidence_path=license_review.evidence_snapshot.path,
    )
    if (
        source_lock.lock != refreshed_lock.lock
        or source_lock.lock_sha256 != refreshed_lock.lock_sha256
        or license_review.review != refreshed_review.review
        or license_review.review_sha256 != refreshed_review.review_sha256
    ):
        raise MUGEProvenanceError(
            "MUGE source or licence handle differs from source-bound recomputation"
        )
    return refreshed_lock, refreshed_review


def _selection_candidate_payloads(
    text_snapshot: _SourceSnapshot, candidate_count: int
) -> tuple[tuple[int, ...], bytes, str]:
    titles = _build_titles_from_snapshot(text_snapshot, candidate_count)
    if len(titles) != candidate_count:
        raise MUGEProvenanceError(
            "MUGE selection plan candidate_count exceeds the locked source universe"
        )
    candidate_ids = tuple(sorted(titles))
    content = b"".join(
        _canonical_json_bytes(
            {
                "product_id": f"muge-{image_id}",
                "source_record_id": f"image:{image_id}",
                "title_sha256": hashlib.sha256(
                    titles[image_id].encode("utf-8")
                ).hexdigest(),
            }
        )
        for image_id in candidate_ids
    )
    return candidate_ids, content, hashlib.sha256(content).hexdigest()


def _load_canonical_strict_model(
    path: Path | str,
    *,
    expected_sha256: str,
    label: str,
    model_type,
):
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise MUGEProvenanceError(f"formal {label} requires an external SHA-256")
    snapshot = _snapshot_regular_file(Path(path), label)
    if snapshot.sha256 != expected_sha256:
        raise MUGEProvenanceError(
            f"{label} does not match the external expected digest"
        )
    try:
        json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_source_lock_constant,
        )
        value = model_type.model_validate_json(snapshot.content, strict=True)
    except Exception as error:
        raise MUGEProvenanceError(f"{label} violates its strict schema") from error
    if snapshot.content != _canonical_json_bytes(value.model_dump(mode="json")):
        raise MUGEProvenanceError(f"{label} must be canonical JSON")
    _verify_snapshot(snapshot, label)
    return value, snapshot


def load_verified_muge_selection_plan(
    path: Path | str,
    *,
    expected_plan_sha256: str,
    text_source_path: Path | str,
    image_source_path: Path | str,
    retained_gallery_parquet_path: Path | str,
    source_lock: VerifiedMUGESourceLock,
    license_review: VerifiedMUGELicenseReview,
) -> VerifiedMUGESelectionPlan:
    """Verify a caller-pinned plan before any formal MUGE output is produced."""

    refreshed_lock, refreshed_review = _refresh_verified_muge_source_evidence(
        source_lock, license_review
    )
    plan, snapshot = _load_canonical_strict_model(
        path,
        expected_sha256=expected_plan_sha256,
        label="MUGE preregistered selection plan",
        model_type=MUGESelectionPlan,
    )
    text_snapshot = _snapshot_regular_file(Path(text_source_path), "MUGE texts source")
    image_snapshot = _snapshot_regular_file(
        Path(image_source_path), "MUGE images source"
    )
    expected_binding = (
        refreshed_lock.lock_sha256,
        refreshed_review.review_sha256,
        refreshed_lock.lock.source_revision,
        refreshed_lock.lock.texts_sha256,
        refreshed_lock.lock.images_sha256,
    )
    actual_binding = (
        plan.source_lock_sha256,
        plan.license_review_sha256,
        plan.source_revision,
        plan.source_texts_sha256,
        plan.source_images_sha256,
    )
    if actual_binding != expected_binding:
        raise MUGEProvenanceError(
            "MUGE selection plan is not bound to the verified source and review"
        )
    if (
        text_snapshot.sha256 != refreshed_lock.lock.texts_sha256
        or image_snapshot.sha256 != refreshed_lock.lock.images_sha256
    ):
        raise MUGEProvenanceError(
            "MUGE selection inputs do not match the verified source lock"
        )
    candidate_ids, _, universe_sha256 = _selection_candidate_payloads(
        text_snapshot, plan.candidate_count
    )
    if plan.candidate_universe_sha256 != universe_sha256:
        raise MUGEProvenanceError(
            "MUGE selection plan does not match the source-derived candidate universe"
        )
    retained_gallery_path = Path(retained_gallery_parquet_path).absolute()
    retained_gallery = _snapshot_retained_gallery(retained_gallery_path)
    _validate_retained_gallery_binding(plan, retained_gallery)
    for retained_snapshot in retained_gallery.image_snapshots:
        _verify_snapshot(retained_snapshot, "MUGE retained gallery image")
    for source_snapshot, label in (
        (snapshot, "MUGE preregistered selection plan"),
        (text_snapshot, "MUGE texts source"),
        (image_snapshot, "MUGE images source"),
        (refreshed_lock.snapshot, "MUGE source lock"),
        (refreshed_review.snapshot, "MUGE human license review"),
        (refreshed_review.evidence_snapshot, "MUGE license evidence"),
    ):
        _verify_snapshot(source_snapshot, label)
    return VerifiedMUGESelectionPlan(
        plan=plan,
        plan_sha256=snapshot.sha256,
        snapshot=snapshot,
        text_snapshot=text_snapshot,
        image_snapshot=image_snapshot,
        retained_gallery_parquet_path=retained_gallery_path,
        candidate_image_ids=candidate_ids,
        source_lock=refreshed_lock,
        license_review=refreshed_review,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _refresh_verified_muge_selection_plan(
    selection_plan: VerifiedMUGESelectionPlan,
) -> VerifiedMUGESelectionPlan:
    if (
        not isinstance(selection_plan, VerifiedMUGESelectionPlan)
        or selection_plan._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise MUGEProvenanceError(
            "formal MUGE use requires a verified preregistered selection plan"
        )
    refreshed = load_verified_muge_selection_plan(
        selection_plan.snapshot.path,
        expected_plan_sha256=selection_plan.plan_sha256,
        text_source_path=selection_plan.text_snapshot.path,
        image_source_path=selection_plan.image_snapshot.path,
        retained_gallery_parquet_path=selection_plan.retained_gallery_parquet_path,
        source_lock=selection_plan.source_lock,
        license_review=selection_plan.license_review,
    )
    if (
        selection_plan.plan != refreshed.plan
        or selection_plan.candidate_image_ids != refreshed.candidate_image_ids
        or selection_plan.plan_sha256 != refreshed.plan_sha256
        or selection_plan.retained_gallery_parquet_path
        != refreshed.retained_gallery_parquet_path
    ):
        raise MUGEProvenanceError(
            "MUGE selection-plan handle differs from source-bound recomputation"
        )
    return refreshed


def load_verified_muge_selection(
    path: Path | str,
    *,
    expected_ledger_sha256: str,
    selection_plan: VerifiedMUGESelectionPlan,
) -> VerifiedMUGESelection:
    """Load a complete external disposition ledger over a verified plan universe."""

    refreshed_plan = _refresh_verified_muge_selection_plan(selection_plan)
    ledger, snapshot = _load_canonical_strict_model(
        path,
        expected_sha256=expected_ledger_sha256,
        label="MUGE candidate disposition ledger",
        model_type=MUGEDispositionLedger,
    )
    expected_binding = (
        refreshed_plan.plan_sha256,
        refreshed_plan.source_lock.lock_sha256,
        refreshed_plan.license_review.review_sha256,
        refreshed_plan.source_lock.lock.source_revision,
        refreshed_plan.plan.candidate_universe_sha256,
        refreshed_plan.plan.candidate_count,
    )
    actual_binding = (
        ledger.selection_plan_sha256,
        ledger.source_lock_sha256,
        ledger.license_review_sha256,
        ledger.source_revision,
        ledger.candidate_universe_sha256,
        ledger.candidate_count,
    )
    if actual_binding != expected_binding:
        raise MUGEProvenanceError(
            "MUGE disposition ledger is not bound to the verified selection plan"
        )
    ledger_ids = tuple(
        int(value.source_record_id.removeprefix("image:"))
        for value in ledger.dispositions
    )
    if ledger_ids != refreshed_plan.candidate_image_ids:
        raise MUGEProvenanceError(
            "MUGE disposition ledger must cover every source-derived candidate exactly once"
        )
    if ledger.reviewed_at < refreshed_plan.plan.preregistered_at:
        raise MUGEProvenanceError(
            "MUGE disposition review cannot predate the preregistered selection plan"
        )
    _verify_snapshot(snapshot, "MUGE candidate disposition ledger")
    _verify_snapshot(refreshed_plan.snapshot, "MUGE preregistered selection plan")
    return VerifiedMUGESelection(
        plan=refreshed_plan,
        ledger=ledger,
        ledger_sha256=snapshot.sha256,
        ledger_snapshot=snapshot,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _refresh_verified_muge_selection(
    selection: VerifiedMUGESelection,
) -> VerifiedMUGESelection:
    if (
        not isinstance(selection, VerifiedMUGESelection)
        or selection._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise MUGEProvenanceError(
            "formal MUGE export requires a verified selection handle"
        )
    refreshed = load_verified_muge_selection(
        selection.ledger_snapshot.path,
        expected_ledger_sha256=selection.ledger_sha256,
        selection_plan=selection.plan,
    )
    if (
        selection.ledger != refreshed.ledger
        or selection.ledger_sha256 != refreshed.ledger_sha256
        or selection.plan.plan != refreshed.plan.plan
    ):
        raise MUGEProvenanceError(
            "MUGE selection handle differs from source-bound recomputation"
        )
    return refreshed


def _require_real_directory_tree(root: Path, descendant: Path) -> None:
    try:
        relative = descendant.relative_to(root)
    except ValueError:
        raise MUGEProvenanceError(
            "product_image_root must stay below asset_root"
        ) from None
    current = root
    for part in (".", *relative.parts):
        if part != ".":
            current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise MUGEProvenanceError(
                f"MUGE asset directory is missing: {current}"
            ) from error
        if _path_is_link_or_junction(current, metadata) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise MUGEProvenanceError(
                f"MUGE asset directory must be a real directory: {current}"
            )


def _existing_image_path(row: dict, parquet_path: Path) -> Path:
    path = Path(str(row["image_path"]))
    return path if path.is_absolute() else parquet_path.parent / path


def _snapshot_retained_gallery_rows(
    rows: Iterable[dict], parquet_path: Path
) -> _RetainedGalleryBinding:
    retained_rows = tuple(
        _normalize_product_row(row) for row in rows if row.get("source") != "muge"
    )
    payload = _canonical_json_bytes(
        {"policy_version": _RETAINED_GALLERY_POLICY_VERSION}
    )
    fingerprints: list[str] = []
    snapshots: list[_SourceSnapshot] = []
    for index, row in enumerate(retained_rows):
        image_path = _existing_image_path(row, parquet_path)
        snapshot = _snapshot_regular_file(
            image_path, f"retained gallery image {row['product_id']}"
        )
        try:
            with Image.open(io.BytesIO(snapshot.content)) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
                fingerprint = str(imagehash.phash(image))
        except (OSError, ValueError) as error:
            raise MUGEProvenanceError(
                f"unreadable retained gallery image for {row['product_id']}"
            ) from error
        payload += _canonical_json_bytes(
            {
                "dependency_index": index,
                "image_sha256": snapshot.sha256,
                "product_row": row,
                "selection_phash": fingerprint,
            }
        )
        fingerprints.append(fingerprint)
        snapshots.append(snapshot)
    return _RetainedGalleryBinding(
        rows=retained_rows,
        row_count=len(retained_rows),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        fingerprints=tuple(fingerprints),
        image_snapshots=tuple(snapshots),
    )


def _snapshot_retained_gallery(parquet_path: Path) -> _RetainedGalleryBinding:
    parquet_path = Path(parquet_path)
    if not os.path.lexists(parquet_path):
        return _snapshot_retained_gallery_rows((), parquet_path)
    parquet_snapshot = _snapshot_regular_file(
        parquet_path, "MUGE retained-gallery product parquet"
    )
    try:
        rows = pq.read_table(pa.BufferReader(parquet_snapshot.content)).to_pylist()
    except Exception as error:
        raise MUGEProvenanceError(
            "unable to read the retained-gallery product parquet"
        ) from error
    binding = _snapshot_retained_gallery_rows(rows, parquet_path)
    _verify_snapshot(parquet_snapshot, "MUGE retained-gallery product parquet")
    return binding


def compute_muge_retained_gallery_binding(parquet_path: Path | str) -> tuple[int, str]:
    """Compute preregistration material without creating or claiming an approval."""

    binding = _snapshot_retained_gallery(Path(parquet_path))
    for snapshot in binding.image_snapshots:
        _verify_snapshot(snapshot, "MUGE retained gallery image")
    return binding.row_count, binding.manifest_sha256


def _validate_retained_gallery_binding(
    plan: MUGESelectionPlan, binding: _RetainedGalleryBinding
) -> None:
    if (
        plan.retained_gallery_policy_version != _RETAINED_GALLERY_POLICY_VERSION
        or plan.retained_gallery_row_count != binding.row_count
        or plan.retained_gallery_manifest_sha256 != binding.manifest_sha256
    ):
        raise MUGEProvenanceError(
            "MUGE retained gallery differs from the preregistered selection dependency"
        )


def clean_dataset(
    *,
    image_tsv: Path,
    titles: dict[int, str],
    output_dir: Path,
    parquet_path: Path,
    min_side: int = 200,
    text_source: Path | None = None,
    source_lock_path: Path | None = None,
    expected_source_lock_sha256: str | None = None,
    license_evidence_path: Path | None = None,
    license_review_path: Path | None = None,
    expected_license_review_sha256: str | None = None,
    selection_plan_path: Path | None = None,
    expected_selection_plan_sha256: str | None = None,
    source_lock: MUGESourceLock | None = None,
) -> CleanReport:
    """清洗被 ``titles`` 选中的图片并写统一 Product Parquet。

    去重键使用感知哈希（pHash）精确值；输入顺序决定重复组保留哪一张，因而输出可复现。
    """
    output_dir = Path(output_dir)
    parquet_path = Path(parquet_path)
    formal_requested = any(
        value is not None
        for value in (
            text_source,
            source_lock_path,
            expected_source_lock_sha256,
            license_evidence_path,
            license_review_path,
            expected_license_review_sha256,
            selection_plan_path,
            expected_selection_plan_sha256,
            source_lock,
        )
    )
    if source_lock is not None:
        raise MUGEProvenanceError(
            "in-process source_lock objects cannot establish formal identity; use a "
            "canonical source_lock_path and external expected_source_lock_sha256"
        )
    if formal_requested and (
        text_source is None
        or source_lock_path is None
        or expected_source_lock_sha256 is None
        or license_evidence_path is None
        or license_review_path is None
        or expected_license_review_sha256 is None
        or selection_plan_path is None
        or expected_selection_plan_sha256 is None
    ):
        raise MUGEProvenanceError(
            "formal MUGE cleaning requires text_source, source_lock_path, and "
            "expected_source_lock_sha256, license_evidence_path, license_review_path, "
            "expected_license_review_sha256, selection_plan_path, and "
            "expected_selection_plan_sha256 together"
        )
    text_snapshot: _SourceSnapshot | None = None
    image_snapshot: _SourceSnapshot | None = None
    lock_snapshot: _SourceSnapshot | None = None
    license_snapshot: _SourceSnapshot | None = None
    lock_sha256: str | None = None
    verified_lock: MUGESourceLock | None = None
    verified_license: VerifiedMUGELicenseReview | None = None
    verified_plan: VerifiedMUGESelectionPlan | None = None
    if formal_requested:
        verified = load_verified_muge_source_lock(
            source_lock_path, expected_lock_sha256=expected_source_lock_sha256
        )
        verified_lock = verified.lock
        lock_snapshot = verified.snapshot
        lock_sha256 = verified.lock_sha256
        if not titles:
            raise MUGEProvenanceError("formal MUGE cleaning requires non-empty titles")
        text_snapshot = _snapshot_regular_file(Path(text_source), "MUGE texts source")
        image_snapshot = _snapshot_regular_file(Path(image_tsv), "MUGE images source")
        verified_license = load_verified_muge_license_review(
            license_review_path,
            expected_review_sha256=expected_license_review_sha256,
            source_lock=verified,
            license_evidence_path=license_evidence_path,
        )
        license_snapshot = verified_license.evidence_snapshot
        if text_snapshot.sha256 != verified_lock.texts_sha256:
            raise MUGEProvenanceError(
                "MUGE texts source does not match the external source lock"
            )
        if image_snapshot.sha256 != verified_lock.images_sha256:
            raise MUGEProvenanceError(
                "MUGE images source does not match the external source lock"
            )
        verified_plan = load_verified_muge_selection_plan(
            selection_plan_path,
            expected_plan_sha256=expected_selection_plan_sha256,
            text_source_path=text_source,
            image_source_path=image_tsv,
            retained_gallery_parquet_path=parquet_path,
            source_lock=verified,
            license_review=verified_license,
        )
        if min_side != verified_plan.plan.minimum_image_side:
            raise MUGEProvenanceError(
                "MUGE minimum image side differs from the preregistered selection plan"
            )
        locked_titles = _build_titles_from_snapshot(
            text_snapshot, verified_plan.plan.candidate_count
        )
        if locked_titles != titles:
            raise MUGEProvenanceError(
                "titles do not match the preregistered source-derived candidate universe"
            )
    recover_directory_and_file_publish(output_dir, parquet_path)
    retained_binding: _RetainedGalleryBinding | None = None
    if verified_plan is not None:
        retained_binding = _snapshot_retained_gallery(parquet_path)
        _validate_retained_gallery_binding(verified_plan.plan, retained_binding)
        retained_rows = list(retained_binding.rows)
        seen_hashes = set(retained_binding.fingerprints)
    else:
        existing_rows = (
            pq.read_table(parquet_path).to_pylist() if parquet_path.is_file() else []
        )
        retained_rows = [
            _normalize_product_row(row)
            for row in existing_rows
            if row.get("source") != "muge"
        ]
        seen_hashes: set[str] = set()
        for row in retained_rows:
            image_path = _existing_image_path(row, parquet_path)
            try:
                with Image.open(image_path) as opened:
                    opened.load()
                    normalized = ImageOps.exif_transpose(opened).convert("RGB")
                    seen_hashes.add(str(imagehash.phash(normalized)))
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"unreadable retained product image for {row['product_id']}: {image_path}"
                ) from error

    staging = prepare_staging_directory(output_dir)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    rows: list[dict] = []
    image_paths: list[Path] = []
    provenance_records: list[MUGEAssetProvenance] = []
    duplicates = too_small = damaged = 0
    found_ids: set[int] = set()
    source_ids: set[int] = set()

    try:
        image_source: TextIO
        if image_snapshot is None:
            image_source = Path(image_tsv).open(encoding="ascii")
        else:
            try:
                image_source = io.StringIO(image_snapshot.content.decode("ascii"))
            except UnicodeDecodeError as error:
                raise MUGEProvenanceError(
                    "locked MUGE images source must be ASCII TSV"
                ) from error
        with image_source as source:
            for image_id, encoded in _iter_image_rows(source):
                if verified_lock is not None:
                    if image_id < 0 or image_id in source_ids:
                        raise MUGEProvenanceError(
                            "locked MUGE image ids must be unique non-negative integers"
                        )
                    source_ids.add(image_id)
                if image_id not in titles:
                    continue
                found_ids.add(image_id)
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                    with Image.open(io.BytesIO(image_bytes)) as opened:
                        opened.load()
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                except (binascii.Error, OSError, ValueError):
                    damaged += 1
                    continue
                if min(image.size) < min_side:
                    too_small += 1
                    continue
                filename = f"muge-{image_id}.jpg"
                staged_destination = staging / filename
                fingerprint = _write_normalized_jpeg(image, staged_destination)
                if fingerprint in seen_hashes:
                    staged_destination.unlink()
                    duplicates += 1
                    continue
                seen_hashes.add(fingerprint)

                destination = output_dir / filename
                try:
                    stored_path = destination.relative_to(
                        parquet_path.parent
                    ).as_posix()
                except ValueError:
                    stored_path = destination.as_posix()
                rows.append(
                    {
                        "product_id": f"muge-{image_id}",
                        "title": titles[image_id],
                        "category_l1": "unknown",
                        "category_l2": None,
                        "category_l3": None,
                        "ocr_text": None,
                        "image_path": stored_path,
                        "source": "muge",
                    }
                )
                image_paths.append(destination)
                if verified_lock is not None:
                    assert lock_sha256 is not None
                    assert verified_plan is not None
                    source_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
                    output_asset_sha256 = hashlib.sha256(
                        staged_destination.read_bytes()
                    ).hexdigest()
                    provenance_records.append(
                        MUGEAssetProvenance(
                            asset_role="product_gallery_candidate",
                            product_id=f"muge-{image_id}",
                            filename=filename,
                            source_dataset=verified_lock.source_dataset,
                            source_revision=verified_lock.source_revision,
                            source_record_id=f"image:{image_id}",
                            source_lock_sha256=lock_sha256,
                            selection_plan_sha256=verified_plan.plan_sha256,
                            retained_gallery_manifest_sha256=(
                                verified_plan.plan.retained_gallery_manifest_sha256
                            ),
                            retained_gallery_row_count=(
                                verified_plan.plan.retained_gallery_row_count
                            ),
                            source_texts_sha256=verified_lock.texts_sha256,
                            source_images_sha256=verified_lock.images_sha256,
                            source_image_sha256=source_image_sha256,
                            output_asset_sha256=output_asset_sha256,
                            transform_policy_version=_TRANSFORM_POLICY_VERSION,
                            derivation_parent_asset_ids=(),
                            license_id=verified_lock.license_id,
                            license_evidence_sha256=(
                                verified_lock.license_evidence_sha256
                            ),
                            license_review_sha256=(verified_license.review_sha256),
                            license_evidence_url=verified_lock.license_evidence_url,
                            source_url=verified_lock.source_url,
                            attribution=verified_lock.attribution,
                            cloud_upload_allowed=(
                                verified_license.review.permissions.cloud_upload_allowed
                            ),
                            public_demo_allowed=(
                                verified_license.review.permissions.public_demo_allowed
                            ),
                        )
                    )

        merged_rows = rows + retained_rows
        table = pa.Table.from_pylist(merged_rows, schema=_PRODUCT_SCHEMA)
        pq.write_table(table, temporary_parquet, compression="zstd")
        if pq.read_table(temporary_parquet).num_rows != len(merged_rows):
            raise RuntimeError("MUGE Parquet 写入后的行数校验失败")
        if len(list(staging.glob("muge-*.jpg"))) != len(rows):
            raise RuntimeError("MUGE staging 图片数量校验失败")
        if verified_lock is not None and not provenance_records:
            raise MUGEProvenanceError(
                "formal MUGE cleaning produced no exportable product assets"
            )
        if verified_lock is None:
            (staging / _DIAGNOSTIC_MARKER).write_bytes(
                _canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "mode": "diagnostic",
                        "eligible_for_formal_export": False,
                        "reason": "missing externally locked MUGE provenance",
                    }
                )
            )
        else:
            (staging / _PROVENANCE_FILE).write_bytes(
                b"".join(
                    _canonical_json_bytes(record.model_dump(mode="json"))
                    for record in provenance_records
                )
            )
            assert text_snapshot is not None and image_snapshot is not None
            assert lock_snapshot is not None
            assert license_snapshot is not None
            assert verified_license is not None
            _verify_snapshot(text_snapshot, "MUGE texts source")
            _verify_snapshot(image_snapshot, "MUGE images source")
            _verify_snapshot(lock_snapshot, "MUGE source lock")
            _verify_snapshot(license_snapshot, "MUGE license evidence")
            _verify_snapshot(verified_license.snapshot, "MUGE human license review")
            assert verified_plan is not None
            _verify_snapshot(
                verified_plan.snapshot, "MUGE preregistered selection plan"
            )
            assert retained_binding is not None
            for retained_snapshot in retained_binding.image_snapshots:
                _verify_snapshot(retained_snapshot, "MUGE retained gallery image")
        publish_staged_directory_and_file(
            staging, output_dir, temporary_parquet, parquet_path
        )
    except Exception:
        discard_staging_directory(staging)
        temporary_parquet.unlink(missing_ok=True)
        raise
    return CleanReport(
        kept=len(rows),
        duplicates=duplicates,
        too_small=too_small,
        damaged=damaged,
        missing_title=len(set(titles).difference(found_ids)),
        image_paths=tuple(image_paths),
    )


def _canonical_json_bytes(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MUGEProvenanceError(f"duplicate provenance JSON key: {key}")
        result[key] = value
    return result


def _load_provenance_records(
    path: Path, *, expected_provenance_sha256: str
) -> tuple[MUGEAssetProvenance, ...]:
    if not isinstance(expected_provenance_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_provenance_sha256
    ):
        raise MUGEProvenanceError(
            "formal MUGE export requires an external provenance SHA-256"
        )
    snapshot = _snapshot_regular_file(path, "MUGE provenance sidecar")
    if snapshot.sha256 != expected_provenance_sha256:
        raise MUGEProvenanceError(
            "MUGE provenance sidecar does not match the external expected digest"
        )
    records: list[MUGEAssetProvenance] = []
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MUGEProvenanceError("MUGE provenance sidecar must be UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MUGEProvenanceError("MUGE provenance sidecar has a blank row")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_source_lock_constant,
            )
            record = MUGEAssetProvenance.model_validate(raw)
        except Exception as error:
            raise MUGEProvenanceError(
                f"MUGE provenance sidecar row {line_number} is invalid"
            ) from error
        if _canonical_json_bytes(record.model_dump(mode="json")) != (
            line + "\n"
        ).encode("utf-8"):
            raise MUGEProvenanceError("MUGE provenance sidecar must be canonical JSONL")
        records.append(record)
    if not records:
        raise MUGEProvenanceError("MUGE provenance sidecar must not be empty")
    _verify_snapshot(snapshot, "MUGE provenance sidecar")
    return tuple(records)


def _normalized_jpeg_phash(image: Image.Image) -> str:
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=90, optimize=True)
    encoded.seek(0)
    with Image.open(encoded) as normalized:
        normalized.load()
        return str(imagehash.phash(normalized.convert("RGB")))


def _derive_preregistered_dispositions(
    *,
    selection: VerifiedMUGESelection,
    parquet_rows: list[dict],
    parquet_path: Path,
) -> tuple[MUGECandidateDisposition, ...]:
    plan = selection.plan.plan
    candidate_ids = set(selection.plan.candidate_image_ids)
    outcomes: dict[int, MUGEDispositionReason] = {
        image_id: "missing_source_image" for image_id in candidate_ids
    }
    seen_hashes: set[str] = set()
    for row in parquet_rows:
        if row.get("source") == "muge":
            continue
        image_path = _existing_image_path(row, parquet_path)
        try:
            with Image.open(image_path) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
                seen_hashes.add(str(imagehash.phash(image)))
        except (OSError, ValueError) as error:
            raise MUGEProvenanceError(
                "unable to revalidate retained catalog image for MUGE selection"
            ) from error

    try:
        image_source = io.StringIO(
            selection.plan.image_snapshot.content.decode("ascii")
        )
    except UnicodeDecodeError as error:
        raise MUGEProvenanceError(
            "locked MUGE images source must be ASCII TSV"
        ) from error
    seen_source_ids: set[int] = set()
    try:
        for image_id, encoded in _iter_image_rows(image_source):
            if image_id < 0 or image_id in seen_source_ids:
                raise MUGEProvenanceError(
                    "locked MUGE image ids must be unique non-negative integers"
                )
            seen_source_ids.add(image_id)
            if image_id not in candidate_ids:
                continue
            try:
                source_bytes = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(source_bytes)) as opened:
                    opened.load()
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except (binascii.Error, OSError, ValueError):
                outcomes[image_id] = "invalid_source_image"
                continue
            if min(image.size) < plan.minimum_image_side:
                outcomes[image_id] = "below_preregistered_minimum_side"
                continue
            fingerprint = _normalized_jpeg_phash(image)
            if fingerprint in seen_hashes:
                outcomes[image_id] = "normalized_perceptual_duplicate"
                continue
            seen_hashes.add(fingerprint)
            outcomes[image_id] = "accepted_by_preregistered_policy"
    except ValueError as error:
        raise MUGEProvenanceError(
            "locked MUGE images source has invalid TSV"
        ) from error

    return tuple(
        MUGECandidateDisposition(
            source_record_id=f"image:{image_id}",
            product_id=f"muge-{image_id}",
            disposition=(
                "accepted"
                if outcomes[image_id] == "accepted_by_preregistered_policy"
                else "rejected"
            ),
            reason=outcomes[image_id],
        )
        for image_id in sorted(candidate_ids)
    )


def export_dataset_asset_drafts(
    *,
    parquet_path: Path,
    asset_root: Path,
    product_image_root: Path,
    output_path: Path,
    expected_provenance_sha256: str,
    verified_selection: VerifiedMUGESelection,
) -> tuple[DatasetAssetDraft, ...]:
    """Export formal drafts only from locked product assets, never Exact copies."""

    selection = _refresh_verified_muge_selection(verified_selection)
    verified = selection.plan.source_lock
    source_lock = verified.lock
    verified_license = selection.plan.license_review
    license_snapshot = verified_license.evidence_snapshot
    parquet_path = Path(parquet_path)
    if (
        selection.plan.retained_gallery_parquet_path.absolute()
        != parquet_path.absolute()
    ):
        raise MUGEProvenanceError(
            "MUGE export parquet differs from the preregistered retained-gallery path"
        )
    asset_root = Path(asset_root).absolute()
    product_image_root = Path(product_image_root).absolute()
    _require_real_directory_tree(asset_root, product_image_root)
    if os.path.lexists(product_image_root / _DIAGNOSTIC_MARKER):
        raise MUGEProvenanceError(
            "diagnostic MUGE cleaning cannot be used for formal export"
        )
    if os.path.lexists(product_image_root / _EXACT_QUERY_MARKER):
        raise MUGEProvenanceError(
            "Exact query copies are diagnostic-only and cannot be formally exported"
        )
    records = _load_provenance_records(
        product_image_root / _PROVENANCE_FILE,
        expected_provenance_sha256=expected_provenance_sha256,
    )
    parquet_snapshot = _snapshot_regular_file(parquet_path, "MUGE product parquet")
    try:
        rows = pq.read_table(pa.BufferReader(parquet_snapshot.content)).to_pylist()
    except Exception as error:
        raise MUGEProvenanceError("unable to read MUGE product parquet") from error
    retained_gallery = _snapshot_retained_gallery_rows(rows, parquet_path)
    _validate_retained_gallery_binding(selection.plan.plan, retained_gallery)
    muge_row_list = [row for row in rows if row.get("source") == "muge"]
    muge_product_ids = [str(row["product_id"]) for row in muge_row_list]
    if len(muge_product_ids) != len(set(muge_product_ids)):
        raise MUGEProvenanceError("MUGE product ids must be unique")
    muge_rows = {str(row["product_id"]): row for row in muge_row_list}
    if len(muge_rows) != len(records):
        raise MUGEProvenanceError(
            "MUGE provenance sidecar does not cover every cleaned product exactly once"
        )
    expected_dispositions = _derive_preregistered_dispositions(
        selection=selection,
        parquet_rows=rows,
        parquet_path=parquet_path,
    )
    if selection.ledger.dispositions != expected_dispositions:
        raise MUGEProvenanceError(
            "MUGE disposition ledger differs from deterministic source-bound selection"
        )
    accepted_product_ids = {
        value.product_id
        for value in expected_dispositions
        if value.disposition == "accepted"
    }
    if accepted_product_ids != set(muge_rows):
        raise MUGEProvenanceError(
            "MUGE accepted dispositions must exactly match formally exported products"
        )
    drafts: list[DatasetAssetDraft] = []
    seen_product_ids: set[str] = set()
    for record in records:
        if record.product_id in seen_product_ids:
            raise MUGEProvenanceError("MUGE provenance product ids must be unique")
        seen_product_ids.add(record.product_id)
        row = muge_rows.get(record.product_id)
        if row is None:
            raise MUGEProvenanceError("MUGE provenance references an unknown product")
        expected_record_id = f"image:{record.product_id.removeprefix('muge-')}"
        expected = {
            "source_dataset": source_lock.source_dataset,
            "source_revision": source_lock.source_revision,
            "source_record_id": expected_record_id,
            "source_lock_sha256": verified.lock_sha256,
            "selection_plan_sha256": selection.plan.plan_sha256,
            "retained_gallery_manifest_sha256": (
                selection.plan.plan.retained_gallery_manifest_sha256
            ),
            "retained_gallery_row_count": (
                selection.plan.plan.retained_gallery_row_count
            ),
            "source_texts_sha256": source_lock.texts_sha256,
            "source_images_sha256": source_lock.images_sha256,
            "license_id": source_lock.license_id,
            "license_evidence_sha256": source_lock.license_evidence_sha256,
            "license_review_sha256": verified_license.review_sha256,
            "license_review_policy_version": _LICENSE_REVIEW_POLICY_VERSION,
            "license_evidence_url": source_lock.license_evidence_url,
            "source_url": source_lock.source_url,
            "attribution": source_lock.attribution,
            "cloud_upload_allowed": (
                verified_license.review.permissions.cloud_upload_allowed
            ),
            "public_demo_allowed": (
                verified_license.review.permissions.public_demo_allowed
            ),
        }
        actual = {key: getattr(record, key) for key in expected}
        if actual != expected:
            raise MUGEProvenanceError(
                f"MUGE provenance does not match the external source lock: {record.product_id}"
            )
        if record.transform_policy_version != _TRANSFORM_POLICY_VERSION:
            raise MUGEProvenanceError("MUGE transform policy is not formally supported")
        if record.derivation_parent_asset_ids:
            raise MUGEProvenanceError(
                "MUGE provenance must not invent unregistered derivation parents"
            )

        image_path = product_image_root / record.filename
        try:
            metadata = image_path.lstat()
        except OSError as error:
            raise MUGEProvenanceError(
                f"cleaned MUGE product image is missing: {record.filename}"
            ) from error
        if _path_is_link_or_junction(image_path, metadata) or not stat.S_ISREG(
            metadata.st_mode
        ):
            raise MUGEProvenanceError(
                "cleaned MUGE product image must be a regular file"
            )
        stored_path = Path(str(row["image_path"]))
        resolved_stored_path = (
            stored_path.absolute()
            if stored_path.is_absolute()
            else (parquet_path.parent / stored_path).absolute()
        )
        if resolved_stored_path != image_path:
            raise MUGEProvenanceError(
                "MUGE product row points outside the locked product image directory"
            )
        image_snapshot = _snapshot_regular_file(
            image_path, f"cleaned MUGE image {record.filename}"
        )
        if image_snapshot.sha256 != record.output_asset_sha256:
            raise MUGEProvenanceError(
                "cleaned MUGE image does not match provenance final-byte digest"
            )
        local_path = image_path.relative_to(asset_root).as_posix()
        drafts.append(
            DatasetAssetDraft(
                source_dataset=record.source_dataset,
                source_revision=record.source_revision,
                source_record_id=record.source_record_id,
                transform_policy_version=record.transform_policy_version,
                local_path=local_path,
                product_id=record.product_id,
                derivation_parent_asset_ids=[],
                license_id=record.license_id,
                source_url=record.source_url,
                attribution=record.attribution,
                cloud_upload_allowed=record.cloud_upload_allowed,
                public_demo_allowed=record.public_demo_allowed,
            )
        )
    drafts.sort(key=lambda item: (item.product_id or "", item.local_path))
    _verify_snapshot(parquet_snapshot, "MUGE product parquet")
    _verify_snapshot(verified.snapshot, "MUGE source lock")
    _verify_snapshot(license_snapshot, "MUGE license evidence")
    _verify_snapshot(verified_license.snapshot, "MUGE human license review")
    _verify_snapshot(selection.plan.snapshot, "MUGE preregistered selection plan")
    _verify_snapshot(selection.ledger_snapshot, "MUGE candidate disposition ledger")
    for retained_snapshot in retained_gallery.image_snapshots:
        _verify_snapshot(retained_snapshot, "MUGE retained gallery image")
    atomic_create_file(
        output_path,
        b"".join(
            _canonical_json_bytes(draft.model_dump(mode="json")) for draft in drafts
        ),
    )
    return tuple(drafts)


def load_verified_muge_draft_bundle(
    path: Path | str,
    *,
    expected_bundle_sha256: str,
    parquet_path: Path | str,
    asset_root: Path | str,
    product_image_root: Path | str,
    expected_provenance_sha256: str,
    verified_selection: VerifiedMUGESelection,
) -> VerifiedMUGEDraftBundle:
    """Re-derive canonical drafts from every source-bound formal input."""

    if not isinstance(expected_bundle_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_bundle_sha256
    ):
        raise MUGEProvenanceError(
            "formal MUGE draft loading requires an external bundle SHA-256"
        )
    snapshot = _snapshot_regular_file(Path(path), "MUGE dataset asset drafts")
    if snapshot.sha256 != expected_bundle_sha256:
        raise MUGEProvenanceError(
            "MUGE draft bundle does not match the external expected digest"
        )
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MUGEProvenanceError("MUGE draft bundle must be UTF-8") from error
    if not text or not text.endswith("\n"):
        raise MUGEProvenanceError(
            "MUGE draft bundle must be non-empty newline-terminated JSONL"
        )
    drafts: list[DatasetAssetDraft] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MUGEProvenanceError("MUGE draft bundle contains a blank row")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_source_lock_constant,
            )
            draft = DatasetAssetDraft.model_validate(raw, strict=True)
        except Exception as error:
            raise MUGEProvenanceError(
                f"MUGE draft bundle row {line_number} is invalid"
            ) from error
        if _canonical_json_bytes(draft.model_dump(mode="json")) != (line + "\n").encode(
            "utf-8"
        ):
            raise MUGEProvenanceError("MUGE draft bundle must be canonical JSONL")
        drafts.append(draft)
    if len({draft.local_path for draft in drafts}) != len(drafts):
        raise MUGEProvenanceError("MUGE draft bundle paths must be unique")
    if tuple(
        sorted(drafts, key=lambda item: (item.product_id or "", item.local_path))
    ) != tuple(drafts):
        raise MUGEProvenanceError("MUGE draft bundle must use canonical row order")
    selection = _refresh_verified_muge_selection(verified_selection)
    verified_lock = selection.plan.source_lock
    verified_license = selection.plan.license_review
    with tempfile.TemporaryDirectory(
        prefix=".muge-draft-reverify-", dir=str(snapshot.path.parent)
    ) as temporary:
        expected_path = Path(temporary) / "dataset-assets.jsonl"
        expected_drafts = export_dataset_asset_drafts(
            parquet_path=Path(parquet_path),
            asset_root=Path(asset_root),
            product_image_root=Path(product_image_root),
            output_path=expected_path,
            expected_provenance_sha256=expected_provenance_sha256,
            verified_selection=selection,
        )
        if (
            expected_drafts != tuple(drafts)
            or expected_path.read_bytes() != snapshot.content
        ):
            raise MUGEProvenanceError(
                "MUGE draft bundle differs from source-bound recomputation"
            )
    _verify_snapshot(snapshot, "MUGE dataset asset drafts")
    _verify_snapshot(verified_lock.snapshot, "MUGE source lock")
    _verify_snapshot(verified_license.snapshot, "MUGE human license review")
    _verify_snapshot(verified_license.evidence_snapshot, "MUGE license evidence")
    return VerifiedMUGEDraftBundle(
        tuple(drafts),
        snapshot.sha256,
        snapshot,
        verified_lock,
        verified_license,
        selection,
        _VERIFIED_HANDLE_TOKEN,
    )


def write_preview(
    image_paths: Iterable[Path], output: Path, tile_size: int = 256
) -> None:
    """把前 9 张图写成确定性的 3×3 JPEG 拼贴。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (tile_size * 3, tile_size * 3), "white")
    for index, path in enumerate(list(image_paths)[:9]):
        with Image.open(path) as opened:
            tile = ImageOps.fit(opened.convert("RGB"), (tile_size, tile_size))
        canvas.paste(tile, ((index % 3) * tile_size, (index // 3) * tile_size))
    canvas.save(output, format="JPEG", quality=90)


def materialize_query_images(
    image_paths: Iterable[Path], destination: Path, limit: int = 2_000
) -> int:
    """从清洗商品图中确定性抽取 Exact Match 查询图；优先硬链接以节省空间。"""
    destination = Path(destination)
    staging = prepare_staging_directory(destination)
    count = 0
    diagnostic_relations: list[dict[str, object]] = []
    try:
        for source in image_paths:
            if count >= limit:
                break
            target = staging / source.name
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
            diagnostic_relations.append(
                {
                    "query_filename": target.name,
                    "parent_product_id": source.stem,
                    "relation": "exact-byte-copy",
                }
            )
            count += 1
        if len(list(staging.glob("muge-*.jpg"))) != count:
            raise RuntimeError("Exact Match staging 数量校验失败")
        (staging / _EXACT_QUERY_MARKER).write_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "mode": "diagnostic",
                    "eligible_for_formal_export": False,
                    "eligible_as_independent_positive": False,
                    "relations": diagnostic_relations,
                }
            )
        )
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return count


def probe(limit: int = 5) -> None:
    """打印真实字段类型和图片规格；解析/清洗逻辑以此输出为依据。"""
    text_path = EXTRACT_DIR / "MUGE" / "train_texts.jsonl"
    image_path = EXTRACT_DIR / "MUGE" / "train_imgs.tsv"
    if not text_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(
            "缺少解压后的 train_texts.jsonl/train_imgs.tsv；先运行 extract"
        )
    for index, record in enumerate(read_text_records(text_path)):
        if index >= limit:
            break
        types = {key: type(value).__name__ for key, value in record.items()}
        print(f"text[{index}] fields={types} value={record}")
    with image_path.open(encoding="ascii") as source:
        for index, (image_id, encoded) in enumerate(_iter_image_rows(source)):
            if index >= limit:
                break
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                print(
                    f"image[{index}] image_id={image_id} format={image.format} "
                    f"mode={image.mode} size={image.size}"
                )


def clean(product_limit: int = 30_000, query_limit: int = 2_000) -> CleanReport:
    text_path = EXTRACT_DIR / "MUGE" / "train_texts.jsonl"
    image_path = EXTRACT_DIR / "MUGE" / "train_imgs.tsv"
    candidate_limit = product_limit + max(1_000, product_limit // 20)
    titles = build_titles([text_path], limit=candidate_limit)
    report = clean_dataset(
        image_tsv=image_path,
        titles=titles,
        output_dir=CLEAN_DIR / "product_images" / "muge",
        parquet_path=CLEAN_DIR / "products.parquet",
    )
    product_rows = pq.read_table(CLEAN_DIR / "products.parquet").to_pylist()
    exact_rows = select_exact_match_products(product_rows, limit=query_limit)
    exact_paths = [(CLEAN_DIR / row["image_path"]) for row in exact_rows]
    query_count = materialize_query_images(
        exact_paths,
        CLEAN_DIR / "query_images" / "exact_match",
        limit=query_limit,
    )
    write_preview(report.image_paths, CLEAN_DIR / "preview_muge.jpg")
    print(
        f"kept={report.kept} duplicates={report.duplicates} too_small={report.too_small} "
        f"damaged={report.damaged} missing_title={report.missing_title} "
        f"query_images={query_count}"
    )
    return report


def stats() -> None:
    parquet_path = CLEAN_DIR / "products.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    muge_rows = [row for row in rows if row["source"] == "muge"]
    categories: dict[str, int] = {}
    for row in muge_rows:
        categories[row["category_l1"]] = categories.get(row["category_l1"], 0) + 1
    print(f"products={len(muge_rows)}")
    print(f"category_l1={json.dumps(categories, ensure_ascii=False, sort_keys=True)}")
    print(f"preview={CLEAN_DIR / 'preview_muge.jpg'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "--member", action="append", dest="members", help="可重复指定 7-Zip 成员通配符"
    )
    probe_parser = subparsers.add_parser("probe", aliases=["inspect"])
    probe_parser.add_argument("--limit", type=int, default=5)
    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--product-limit", type=int, default=30_000)
    clean_parser.add_argument("--query-limit", type=int, default=2_000)
    subparsers.add_parser("stats")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "download":
        download()
    elif args.command == "extract":
        extract(args.members)
    elif args.command in {"probe", "inspect"}:
        probe(args.limit)
    elif args.command == "clean":
        clean(args.product_limit, args.query_limit)
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
