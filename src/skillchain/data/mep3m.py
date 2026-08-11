"""MEP-3M Phase 1 product pipeline.

Only the pinned annotation parquet and a small, product-oriented archive subset are
downloaded.  Every remote artifact is verified before it can enter the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, Iterable, Iterator, Literal, Mapping, Sequence
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
    publish_staged_directory,
    publish_staged_directory_and_file,
    recover_directory_and_file_publish,
)
from skillchain.data.asset_catalog import DatasetAssetDraft
from skillchain.synthesis.store import atomic_create_file

REPOSITORY = "chendelong/MEP-3M"
REVISION = "4f38a7404a7c78d849169a480f77af593fb78d03"
SELECTED_SUBCLASS_IDS = (
    34,
    63,
    100,
    431,
    546,
    405,
    570,
    367,
    414,
    555,
    550,
    490,
    26,
    502,
    421,
    1,
)
REMOTE_ARCHIVE_COUNT = 599
REMOTE_ARCHIVE_BYTES = 71_869_244_985
SELECTED_ARCHIVE_BYTES = 519_487_095

RAW_DIR = config.DATA_DIR / "raw" / "mep3m"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
ARCHIVES_DIR = RAW_DIR / "archives"
EXTRACTED_DIR = RAW_DIR / "extracted"
CLEAN_DIR = config.DATA_DIR / "clean"

PRODUCT_SCHEMA = pa.schema(
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

_PROVENANCE_FILE = "mep3m-provenance.jsonl"
_DIAGNOSTIC_MARKER = "mep3m-diagnostic.json"
_TRANSFORM_POLICY_VERSION = "mep3m-exif-rgb-jpeg-q90-optimize-v1"
_LICENSE_REVIEW_POLICY_VERSION = "mep3m-human-license-use-review-v1"
_EXACT_EXCLUSION_REASON = "annotations_have_no_stable_product_or_view_group_identity"
_VERIFIED_HANDLE_TOKEN = object()
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class MEP3MProvenanceError(ValueError):
    """Raised when formal MEP-3M provenance cannot be proven."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MEP3MSourceLock(_StrictFrozenModel):
    """External identity lock for source artifacts and licence evidence bytes.

    This model deliberately carries no permission decision.  A source publisher or
    lock author cannot make formal use eligible by asserting a boolean here; formal
    use additionally requires an independently pinned ``MEP3MLicenseUseReview``.
    """

    schema_version: Literal[1] = 1
    source_dataset: Literal["mep3m"] = "mep3m"
    source_revision: str
    annotation_sha256: Sha256
    archive_sha256_by_subclass: dict[int, Sha256]
    license_id: str
    license_evidence_sha256: Sha256
    license_evidence_url: str
    source_url: str
    attribution: str | None = None

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if value != REVISION:
            raise ValueError("source_revision must equal the pinned MEP-3M revision")
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
            raise ValueError("license_id must be explicit")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if not value or not parsed.scheme or parsed.hostname is None:
            raise ValueError("source_url must be an absolute URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("source_url must not contain credentials")
        if REVISION not in parsed.path:
            raise ValueError("source_url must include the pinned revision")
        return value

    @field_validator("license_evidence_url")
    @classmethod
    def validate_license_evidence_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if not value or not parsed.scheme or parsed.hostname is None:
            raise ValueError("license_evidence_url must be an absolute URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("license_evidence_url must not contain credentials")
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

    @model_validator(mode="after")
    def validate_pinned_artifacts(self):
        if self.annotation_sha256 != ANNOTATION_SPEC.sha256:
            raise ValueError("annotation SHA-256 must equal the pinned package hash")
        if not self.archive_sha256_by_subclass:
            raise ValueError("at least one pinned archive SHA-256 is required")
        for subclass_id, digest in self.archive_sha256_by_subclass.items():
            spec = ARCHIVE_SPECS.get(subclass_id)
            if spec is None or digest != spec.sha256:
                raise ValueError(
                    f"archive SHA-256 does not match the pinned package: {subclass_id}"
                )
        return self


class MEP3MPermissionMatrix(_StrictFrozenModel):
    """Fail-closed permissions granted only by an accountable human review."""

    local_noncommercial_research_allowed: Literal[True] = True
    local_noncommercial_embedding_allowed: Literal[True] = True
    remote_embedding_allowed: Literal[False] = False
    cloud_upload_allowed: Literal[False] = False
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False


class MEP3MLicenseUseReview(_StrictFrozenModel):
    """Externally pinned human review of exact evidence bytes and their revision."""

    schema_version: Literal[1] = 1
    review_policy_version: Literal["mep3m-human-license-use-review-v1"] = (
        _LICENSE_REVIEW_POLICY_VERSION
    )
    source_lock_sha256: Sha256
    source_revision: str
    license_id: str
    license_evidence_sha256: Sha256
    license_evidence_uri: str
    license_evidence_revision: str
    decision: Literal["approved_local_noncommercial_research_only"]
    permissions: MEP3MPermissionMatrix = Field(default_factory=MEP3MPermissionMatrix)
    reviewer_kind: Literal["human"]
    reviewer_id: str
    reviewed_at: str

    @field_validator(
        "source_revision", "license_id", "license_evidence_revision", "reviewer_id"
    )
    @classmethod
    def validate_review_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be blank")
        if info.field_name == "license_evidence_revision":
            mutable = {"main", "master", "head", "latest", "unknown", "unpinned"}
            normalized = value.casefold().replace("\\", "/")
            if normalized in mutable or any(
                normalized.endswith(f"/{segment}") for segment in mutable
            ):
                raise ValueError("license_evidence_revision must be immutable")
        if info.field_name == "reviewer_id" and re.search(
            r"(?:^|[-_. ])(?:llm|gpt|chatgpt|claude|qwen|gemini|model)(?:$|[-_. ])",
            value.casefold(),
        ):
            raise ValueError("reviewer_id must identify an accountable human")
        return value

    @field_validator("license_evidence_uri")
    @classmethod
    def validate_evidence_uri(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if (
            not value
            or not parsed.scheme
            or any(character.isspace() for character in value)
        ):
            raise ValueError("license_evidence_uri must be an absolute URI")
        if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname is None:
            raise ValueError("HTTP license_evidence_uri must contain a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("license_evidence_uri must not contain credentials")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        try:
            parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError(
                "reviewed_at must be a valid second-precision UTC timestamp"
            ) from error
        if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != value:
            raise ValueError(
                "reviewed_at must be a valid second-precision UTC timestamp"
            )
        return value


class MEP3MAssetProvenance(_StrictFrozenModel):
    schema_version: Literal[3] = 3
    asset_role: Literal["product_gallery_candidate"]
    identity_scope: Literal["single_asset_only"]
    exact_eligibility_candidate: Literal[False]
    exact_exclusion_reason: Literal[
        "annotations_have_no_stable_product_or_view_group_identity"
    ]
    product_id: str = Field(pattern=r"^mep3m-[0-9]+-[0-9]+$")
    filename: str = Field(pattern=r"^mep3m-[0-9]+-[0-9]+\.jpg$")
    source_dataset: Literal["mep3m"]
    source_revision: str
    source_record_id: str
    source_lock_sha256: Sha256
    annotation_sha256: Sha256
    archive_subclass_id: int = Field(ge=0)
    archive_sha256: Sha256
    extraction_receipt_sha256: Sha256
    archive_member_path: str
    archive_member_uncompressed_size: int = Field(ge=1)
    archive_member_crc32: str = Field(pattern=r"^[0-9a-f]{8}$")
    source_image_sha256: Sha256
    normalized_asset_sha256: Sha256
    transform_policy_version: Literal["mep3m-exif-rgb-jpeg-q90-optimize-v1"]
    derivation_parent_asset_ids: tuple[str, ...] = ()
    license_id: str
    license_evidence_sha256: Sha256
    license_review_sha256: Sha256
    license_review_policy_version: Literal["mep3m-human-license-use-review-v1"] = (
        _LICENSE_REVIEW_POLICY_VERSION
    )
    license_evidence_url: str
    license_evidence_revision: str
    source_url: str
    attribution: str | None = None
    local_noncommercial_research_allowed: Literal[True]
    local_noncommercial_embedding_allowed: Literal[True]
    remote_embedding_allowed: Literal[False]
    cloud_upload_allowed: Literal[False]
    redistribution_allowed: Literal[False]
    public_demo_allowed: Literal[False]

    @field_validator(
        "source_revision",
        "source_record_id",
        "archive_member_path",
        "license_id",
        "source_url",
    )
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance fields must not be blank")
        return value

    @field_validator("derivation_parent_asset_ids", mode="before")
    @classmethod
    def coerce_json_parent_ids(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("derivation_parent_asset_ids")
    @classmethod
    def reject_fabricated_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError("MEP-3M external records are not catalog asset parents")
        return value


class MEP3MExtractionMember(_StrictFrozenModel):
    """One archive member recorded by an independently frozen extraction run."""

    member_path: str
    uncompressed_size: int = Field(ge=1)
    crc32: str = Field(pattern=r"^[0-9a-f]{8}$")
    member_sha256: Sha256

    @field_validator("member_path")
    @classmethod
    def validate_member_path(cls, value: str) -> str:
        if "\\" in value or not re.fullmatch(
            r"[0-9]+/[0-9]+\.(?:jpg|JPG|png|PNG)", value
        ):
            raise ValueError("archive member path must be a canonical image path")
        return value


class MEP3MExtractionReceipt(_StrictFrozenModel):
    """Canonical evidence joining verified archive bytes to extracted members."""

    schema_version: Literal[1] = 1
    source_dataset: Literal["mep3m"] = "mep3m"
    source_revision: str
    archive_subclass_id: int = Field(ge=0)
    archive_filename: str
    archive_size: int = Field(ge=1)
    archive_sha256: Sha256
    members: tuple[MEP3MExtractionMember, ...]

    @field_validator("members", mode="before")
    @classmethod
    def coerce_json_members(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_receipt(self):
        if self.source_revision != REVISION:
            raise ValueError("receipt revision must equal the pinned MEP-3M revision")
        if self.archive_filename != f"{self.archive_subclass_id}.rar":
            raise ValueError("receipt archive filename is inconsistent")
        if not self.members:
            raise ValueError("extraction receipt must contain at least one member")
        paths = [member.member_path for member in self.members]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("extraction receipt members must be unique and sorted")
        expected_prefix = f"{self.archive_subclass_id}/"
        if any(not path.startswith(expected_prefix) for path in paths):
            raise ValueError("receipt member belongs to a different subclass")
        return self


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    content: bytes
    sha256: str
    identity: tuple[int, ...]
    ancestor_identity: tuple[tuple[str, int, int, int, int], ...]


@dataclass(frozen=True)
class _DigestSnapshot:
    path: Path
    size: int
    sha256: str
    identity: tuple[int, ...]
    ancestor_identity: tuple[tuple[str, int, int, int, int], ...]


@dataclass(frozen=True)
class VerifiedMEP3MSourceLock:
    """Canonical source-identity bytes; this handle grants no usage permission."""

    lock: MEP3MSourceLock
    lock_sha256: str
    snapshot: _FileSnapshot
    _verification_token: object | None = None


@dataclass(frozen=True)
class VerifiedMEP3MLicenseUseReview:
    """Human review independently pinned to source, evidence, and permissions."""

    review: MEP3MLicenseUseReview
    review_sha256: str
    snapshot: _FileSnapshot
    evidence_snapshot: _FileSnapshot
    _verification_token: object | None = None


@dataclass(frozen=True)
class VerifiedMEP3MExtractionReceipt:
    receipt: MEP3MExtractionReceipt
    receipt_sha256: str
    snapshot: _FileSnapshot


@dataclass(frozen=True)
class CanonicalMEP3MDraftBundle:
    """Canonical draft bytes only; this object never grants formal identity."""

    drafts: tuple[DatasetAssetDraft, ...]
    bundle_sha256: str
    snapshot: _FileSnapshot
    formal_eligible: Literal[False] = False


@dataclass(frozen=True)
class ExactEligibilityCoverage:
    total_assets: int
    eligible_assets: int
    stable_product_groups: int
    exclusion_reason: str


@dataclass(frozen=True)
class FileSpec:
    path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
            f"{self.path}"
        )


@dataclass(frozen=True)
class AnnotationSpec(FileSpec):
    source_path: str
    source_size: int
    source_sha256: str
    converter_url: str


ANNOTATION_SPEC = AnnotationSpec(
    path="annotations/0000.parquet",
    size=233_520_842,
    sha256="46f26558baa8343b3cafef64944b2221342811d6fdaad1a768c7e1d8d97e3de9",
    source_path="annotations.json",
    source_size=1_698_563_323,
    source_sha256="930ace6fa1fd920252b5d285cac227b9ffd176e376b16c759df4fd620d72b451",
    converter_url=(
        "https://huggingface.co/datasets/chendelong/MEP-3M/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    ),
)

_ARCHIVE_METADATA = {
    1: (29_172_310, "baad307f29d18475130dd2da4116fb76062b57b705eb95a6b8e7e5b2305d6682"),
    26: (
        101_540_940,
        "8d4aeee9e3ed490034a33be60e0caa84511e4b84bef87fddbe94c9a232e5184d",
    ),
    34: (
        34_477_466,
        "3ac4d30b9228962c0fb6faa7a7b7a3e856bc2feda07a6014930c48273c3dc0ac",
    ),
    63: (
        24_133_422,
        "7a9cf4d6d25e5d8b9a27ebc6d98163277fa6045618ff376ce4fcdc4d26149589",
    ),
    100: (
        32_504_747,
        "1795e2f28b918733a7243a400c30410eadddb39959beb4a1648244de910b855c",
    ),
    367: (
        62_746_459,
        "185a402e96d91ac18cd6a268d023a2241b444b973fc4a965087a3611dfb773c0",
    ),
    405: (
        37_600_989,
        "29040ebf642dd7f2018aa06b6ccf360455237254cc013e8fc1dca806de3adf43",
    ),
    414: (
        30_272_558,
        "a3f240bca5e89c5b8ad2d72afabc6bb8c89ddf56d87cf5d9e14dcb61553bbcde",
    ),
    421: (
        10_522_225,
        "6a7ea32e16b86290874991c0d5bb53f90bebfcea45dbefcf5b2b734c83850664",
    ),
    431: (
        4_992_505,
        "4eb973ab3082456684bd60c89c633208bf99576ab103b9d90bc8a8d406f6fad2",
    ),
    490: (
        42_089_434,
        "765baf6fcb7a1e90e39cfaa23ab4c922c8436c6189cf3d92064e4751db0c5591",
    ),
    502: (
        17_652_152,
        "6f25582fe655348112e88c448de0e71d7fbc2621d2108cb4b65f2ae19f0e7949",
    ),
    546: (
        48_229_518,
        "a2d5da8c032e6493d6326f3e1caf25a68e2c9ea92cc476567aaa78a0895a6474",
    ),
    550: (
        14_848_839,
        "172162d0538cefb6161bd2e9c0a95bd968ea60cf342ccb4529446f20a8f33201",
    ),
    555: (
        6_186_685,
        "e67505f7fe190a5506f615b033d6793598452b5acb842bc80f30b7fa8b5267e7",
    ),
    570: (
        22_516_846,
        "9c91e186f752c636b79f61a98ab048f2e429d7fd27e2039f34c11bf063bbce3e",
    ),
}
ARCHIVE_SPECS = {
    subclass_id: FileSpec(path=f"Images/{subclass_id}.rar", size=size, sha256=sha256)
    for subclass_id, (size, sha256) in _ARCHIVE_METADATA.items()
}


@dataclass(frozen=True)
class CleanReport:
    kept: int
    duplicates: int
    too_small: int
    damaged: int
    image_paths: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_WINDOWS_REPARSE_POINT = 0x400


def _file_attributes(metadata: os.stat_result) -> int:
    return int(getattr(metadata, "st_file_attributes", 0))


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        _file_attributes(metadata) & _WINDOWS_REPARSE_POINT
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        _file_attributes(metadata),
    )


def _ancestor_identity(
    path: Path, label: str, *, include_leaf: bool = False
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Reject symlink/junction/reparse traversal and snapshot every existing ancestor."""

    absolute = Path(path).absolute()
    target = absolute if include_leaf else absolute.parent
    ordered = list(reversed((target, *target.parents)))
    identities: list[tuple[str, int, int, int, int]] = []
    missing_seen = False
    for current in ordered:
        if not os.path.lexists(current):
            missing_seen = True
            continue
        if missing_seen:
            raise MEP3MProvenanceError(
                f"{label} has an inconsistent missing ancestor chain"
            )
        try:
            metadata = current.lstat()
        except OSError as error:
            raise MEP3MProvenanceError(
                f"unable to inspect {label} ancestors"
            ) from error
        if _is_link_or_reparse(metadata):
            raise MEP3MProvenanceError(
                f"{label} must not traverse a symlink, junction, or reparse point"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise MEP3MProvenanceError(f"{label} ancestor must be a real directory")
        identities.append(
            (
                os.fspath(current),
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_mode),
                _file_attributes(metadata),
            )
        )
    return tuple(identities)


def _verify_ancestor_identity(
    path: Path,
    expected: tuple[tuple[str, int, int, int, int], ...],
    label: str,
    *,
    include_leaf: bool = False,
) -> None:
    if _ancestor_identity(path, label, include_leaf=include_leaf) != expected:
        raise MEP3MProvenanceError(f"{label} ancestor chain changed during processing")


def _snapshot_regular_file(path: Path, label: str) -> _FileSnapshot:
    path = Path(path).absolute()
    digest_snapshot = _digest_regular_file(path, label, keep_content=True)
    assert isinstance(digest_snapshot, _FileSnapshot)
    return digest_snapshot


def _digest_regular_file(
    path: Path, label: str, *, keep_content: bool = False
) -> _DigestSnapshot | _FileSnapshot:
    path = Path(path).absolute()
    ancestors = _ancestor_identity(path, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise MEP3MProvenanceError(f"unable to inspect {label}") from error
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise MEP3MProvenanceError(
            f"{label} must be a regular non-symlink/non-reparse file"
        )
    if int(before.st_nlink) != 1:
        raise MEP3MProvenanceError(f"{label} must not be a hard-linked file")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_content else None
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if _file_identity(before) != _file_identity(opened):
                raise MEP3MProvenanceError(f"{label} changed before it could be read")
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after_open = os.fstat(source.fileno())
    except OSError as error:
        raise MEP3MProvenanceError(f"unable to read {label}") from error
    if _file_identity(opened) != _file_identity(after_open):
        raise MEP3MProvenanceError(f"{label} changed while it was read")
    try:
        after_path = path.lstat()
    except OSError as error:
        raise MEP3MProvenanceError(f"{label} changed after it was read") from error
    if _file_identity(after_open) != _file_identity(after_path):
        raise MEP3MProvenanceError(f"{label} changed while it was read")
    _verify_ancestor_identity(path, ancestors, label)
    identity = _file_identity(after_open)
    if chunks is not None:
        return _FileSnapshot(
            path=path,
            content=b"".join(chunks),
            sha256=digest.hexdigest(),
            identity=identity,
            ancestor_identity=ancestors,
        )
    return _DigestSnapshot(
        path=path,
        size=int(after_open.st_size),
        sha256=digest.hexdigest(),
        identity=identity,
        ancestor_identity=ancestors,
    )


def _verify_file_snapshot(snapshot: _FileSnapshot, label: str) -> None:
    current = _snapshot_regular_file(snapshot.path, label)
    if (
        current.identity != snapshot.identity
        or current.ancestor_identity != snapshot.ancestor_identity
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise MEP3MProvenanceError(f"{label} changed during MEP-3M processing")


def _verify_digest_snapshot(snapshot: _DigestSnapshot, label: str) -> None:
    current = _digest_regular_file(snapshot.path, label)
    assert isinstance(current, _DigestSnapshot)
    if (
        current.identity != snapshot.identity
        or current.ancestor_identity != snapshot.ancestor_identity
        or current.size != snapshot.size
        or current.sha256 != snapshot.sha256
    ):
        raise MEP3MProvenanceError(f"{label} changed during MEP-3M processing")


def _canonical_json_bytes(value: Mapping) -> bytes:
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


def _reject_json_constant(token: str) -> None:
    raise MEP3MProvenanceError(f"non-finite JSON constant is forbidden: {token}")


def _require_external_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MEP3MProvenanceError(f"{label} requires an external SHA-256 trust root")


def load_verified_mep3m_source_lock(
    path: Path | str, *, expected_lock_sha256: str
) -> VerifiedMEP3MSourceLock:
    """Load canonical lock bytes under an independent caller-owned digest."""

    _require_external_sha256(
        expected_lock_sha256, "formal MEP-3M source-lock verification"
    )
    snapshot = _snapshot_regular_file(Path(path), "MEP-3M source lock")
    if snapshot.sha256 != expected_lock_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M source lock does not match external expected digest"
        )
    try:
        raw = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(raw, dict) or not isinstance(
            raw.get("archive_sha256_by_subclass"), dict
        ):
            raise MEP3MProvenanceError("source-lock archive map must be an object")
        archive_map: dict[int, object] = {}
        for key, value in raw["archive_sha256_by_subclass"].items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"(?:0|[1-9][0-9]*)", key)
                or str(int(key)) != key
            ):
                raise MEP3MProvenanceError(
                    "source-lock archive keys must be canonical decimal strings"
                )
            archive_map[int(key)] = value
        raw = {**raw, "archive_sha256_by_subclass": archive_map}
        lock = MEP3MSourceLock.model_validate(raw, strict=True)
    except Exception as error:
        raise MEP3MProvenanceError(
            "MEP-3M source lock violates its strict schema"
        ) from error
    if snapshot.content != _canonical_json_bytes(lock.model_dump(mode="json")):
        raise MEP3MProvenanceError("MEP-3M source lock must be canonical JSON")
    _verify_file_snapshot(snapshot, "MEP-3M source lock")
    return VerifiedMEP3MSourceLock(
        lock, snapshot.sha256, snapshot, _VERIFIED_HANDLE_TOKEN
    )


def _load_verified_license_evidence(
    path: Path | str, source_lock: MEP3MSourceLock
) -> _FileSnapshot:
    snapshot = _snapshot_regular_file(Path(path), "MEP-3M license evidence")
    if not snapshot.content.strip():
        raise MEP3MProvenanceError("MEP-3M license evidence must not be empty")
    if snapshot.sha256 != source_lock.license_evidence_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M license evidence bytes do not match the external source lock"
        )
    _verify_file_snapshot(snapshot, "MEP-3M license evidence")
    return snapshot


def load_verified_mep3m_license_use_review(
    path: Path | str,
    *,
    expected_review_sha256: str,
    source_lock: VerifiedMEP3MSourceLock,
    license_evidence_path: Path | str,
) -> VerifiedMEP3MLicenseUseReview:
    """Load a human decision under a caller-owned digest trust root.

    A verified source lock is only an identity prerequisite.  It cannot substitute
    for this independently authored and pinned licence/use decision.
    """

    if (
        not isinstance(source_lock, VerifiedMEP3MSourceLock)
        or source_lock._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise MEP3MProvenanceError(
            "formal MEP-3M licence review requires a verified source lock"
        )
    _require_external_sha256(expected_review_sha256, "formal MEP-3M licence/use review")
    evidence_snapshot = _load_verified_license_evidence(
        license_evidence_path, source_lock.lock
    )
    snapshot = _snapshot_regular_file(Path(path), "MEP-3M human licence/use review")
    if snapshot.sha256 != expected_review_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M licence/use review does not match external expected digest"
        )
    try:
        json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        review = MEP3MLicenseUseReview.model_validate_json(
            snapshot.content, strict=True
        )
    except Exception as error:
        raise MEP3MProvenanceError(
            "MEP-3M licence/use review violates its strict schema"
        ) from error
    if snapshot.content != _canonical_json_bytes(review.model_dump(mode="json")):
        raise MEP3MProvenanceError("MEP-3M licence/use review must be canonical JSON")
    expected_binding = (
        source_lock.lock_sha256,
        source_lock.lock.source_revision,
        source_lock.lock.license_id,
        evidence_snapshot.sha256,
        source_lock.lock.license_evidence_url,
        MEP3MPermissionMatrix(),
    )
    actual_binding = (
        review.source_lock_sha256,
        review.source_revision,
        review.license_id,
        review.license_evidence_sha256,
        review.license_evidence_uri,
        review.permissions,
    )
    if actual_binding != expected_binding:
        raise MEP3MProvenanceError(
            "MEP-3M licence/use review is not bound to the verified source evidence"
        )
    _verify_file_snapshot(source_lock.snapshot, "MEP-3M source lock")
    _verify_file_snapshot(evidence_snapshot, "MEP-3M license evidence")
    _verify_file_snapshot(snapshot, "MEP-3M human licence/use review")
    return VerifiedMEP3MLicenseUseReview(
        review=review,
        review_sha256=snapshot.sha256,
        snapshot=snapshot,
        evidence_snapshot=evidence_snapshot,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def load_verified_mep3m_extraction_receipt(
    path: Path | str,
    *,
    expected_receipt_sha256: str,
    source_lock: MEP3MSourceLock,
    archive_subclass_id: int,
    archive_snapshot: _DigestSnapshot,
) -> VerifiedMEP3MExtractionReceipt:
    """Load an extraction receipt only under a caller-owned digest trust root."""

    _require_external_sha256(
        expected_receipt_sha256, "formal MEP-3M extraction receipt"
    )
    snapshot = _snapshot_regular_file(
        Path(path), f"MEP-3M extraction receipt {archive_subclass_id}"
    )
    if snapshot.sha256 != expected_receipt_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M extraction receipt does not match external expected digest"
        )
    try:
        raw = json.loads(
            snapshot.content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        receipt = MEP3MExtractionReceipt.model_validate(raw, strict=True)
    except Exception as error:
        raise MEP3MProvenanceError(
            "MEP-3M extraction receipt violates its strict schema"
        ) from error
    if snapshot.content != _canonical_json_bytes(receipt.model_dump(mode="json")):
        raise MEP3MProvenanceError("MEP-3M extraction receipt must be canonical JSON")
    expected_archive_sha = source_lock.archive_sha256_by_subclass.get(
        archive_subclass_id
    )
    if (
        receipt.archive_subclass_id != archive_subclass_id
        or archive_snapshot.path.name != receipt.archive_filename
        or receipt.archive_size != archive_snapshot.size
        or receipt.archive_sha256 != archive_snapshot.sha256
        or receipt.archive_sha256 != expected_archive_sha
    ):
        raise MEP3MProvenanceError(
            "MEP-3M extraction receipt does not identify the verified archive"
        )
    _verify_archive_receipt_members(archive_snapshot, receipt)
    _verify_file_snapshot(snapshot, f"MEP-3M extraction receipt {archive_subclass_id}")
    return VerifiedMEP3MExtractionReceipt(receipt, snapshot.sha256, snapshot)


def _validate_local_artifact(path: Path, spec: FileSpec) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing local artifact: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise RuntimeError(
            f"local artifact size mismatch for {path.name}: "
            f"expected {spec.size}, got {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != spec.sha256:
        raise RuntimeError(
            f"local artifact SHA256 mismatch for {path.name}: "
            f"expected {spec.sha256}, got {actual_sha256}"
        )


def validate_remote_archive_index(
    items: Sequence[Mapping], *, require_full_index: bool = True
) -> None:
    files = {
        str(item.get("path")): item for item in items if item.get("type") == "file"
    }
    if require_full_index:
        archive_items = [
            item for item in items if str(item.get("path", "")).endswith(".rar")
        ]
        if len(archive_items) != REMOTE_ARCHIVE_COUNT:
            raise RuntimeError("remote archive index has an unexpected file count")
        if (
            sum(int(item.get("size", -1)) for item in archive_items)
            != REMOTE_ARCHIVE_BYTES
        ):
            raise RuntimeError("remote archive index has an unexpected total size")
    for spec in ARCHIVE_SPECS.values():
        item = files.get(spec.path)
        lfs = item.get("lfs", {}) if item else {}
        if (
            item is None
            or int(item.get("size", -1)) != spec.size
            or int(lfs.get("size", -1)) != spec.size
            or lfs.get("oid") != spec.sha256
        ):
            raise RuntimeError(f"remote archive index mismatch for {spec.path}")
    if sum(spec.size for spec in ARCHIVE_SPECS.values()) != SELECTED_ARCHIVE_BYTES:
        raise RuntimeError("selected archive manifest total is inconsistent")


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    session=requests,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 4,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    have = destination.stat().st_size if destination.exists() else 0
    if have > expected_size:
        raise RuntimeError(f"{destination.name} length exceeds pinned size")
    if have == expected_size:
        if _sha256(destination) != expected_sha256:
            raise RuntimeError(f"{destination.name} SHA256 mismatch")
        return

    for attempt in range(max_attempts):
        have = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(
                url, headers=headers, stream=True, timeout=120, allow_redirects=True
            ) as response:
                response.raise_for_status()
                mode = "ab" if have and response.status_code == 206 else "wb"
                with destination.open(mode) as target:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            target.write(chunk)
            break
        except (OSError, requests.RequestException):
            if attempt + 1 == max_attempts:
                raise
            sleep(float(2**attempt))

    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{destination.name} length mismatch: expected {expected_size}, got {actual_size}"
        )
    if _sha256(destination) != expected_sha256:
        raise RuntimeError(f"{destination.name} SHA256 mismatch")


def download(*, session=requests, sleep: Callable[[float], None] = time.sleep) -> None:
    """Download only the pinned parquet and selected RAR files."""
    api_url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/Images"
    response = session.get(api_url, timeout=60)
    response.raise_for_status()
    validate_remote_archive_index(response.json())

    _download_verified_file(
        ANNOTATION_SPEC.converter_url,
        ANNOTATIONS_DIR / "0000.parquet",
        expected_size=ANNOTATION_SPEC.size,
        expected_sha256=ANNOTATION_SPEC.sha256,
        session=session,
        sleep=sleep,
    )
    for subclass_id in SELECTED_SUBCLASS_IDS:
        spec = ARCHIVE_SPECS[subclass_id]
        _download_verified_file(
            spec.url,
            ARCHIVES_DIR / f"{subclass_id}.rar",
            expected_size=spec.size,
            expected_sha256=spec.sha256,
            session=session,
            sleep=sleep,
        )


def _find_7zip() -> str:
    candidates = [shutil.which(name) for name in ("7z", "7zz", "7za")]
    if os.name == "nt":
        candidates.append(r"C:\Program Files\7-Zip\7z.exe")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("7-Zip was not found")


def validate_rar_members(
    subclass_id: int,
    members: Iterable[str],
    *,
    encrypted: bool = False,
    volumes: int = 1,
) -> tuple[str, ...]:
    if encrypted:
        raise ValueError("RAR archive is encrypted")
    if volumes != 1:
        raise ValueError("RAR archive is a multi-volume archive")
    normalized: list[str] = []
    canonical_paths: set[str] = set()
    pattern = re.compile(rf"{subclass_id}/[0-9]+\.(?:jpg|png)", re.IGNORECASE)
    for member in members:
        clean = member.replace("\\", "/")
        if not pattern.fullmatch(clean):
            raise ValueError(f"unsafe or unexpected RAR member: {member}")
        canonical_key = clean.casefold()
        if canonical_key in canonical_paths:
            raise ValueError(f"duplicate RAR member path: {member}")
        canonical_paths.add(canonical_key)
        normalized.append(clean)
    if not normalized:
        raise ValueError("RAR archive contains no image members")
    return tuple(normalized)


def _parse_7zip_listing(subclass_id: int, output: str) -> tuple[str, ...]:
    output = output.replace("\r\n", "\n")
    header, separator, member_listing = output.partition("----------")
    if not separator:
        header = ""
        member_listing = output
    archive_type = re.search(r"^Type = (.+)$", header, re.MULTILINE)
    if archive_type and archive_type.group(1).strip() != "Rar5":
        raise ValueError("RAR archive is not Rar5")
    if re.search(r"^Solid = \+$", header, re.MULTILINE):
        raise ValueError("RAR archive is unexpectedly solid")
    encrypted = bool(re.search(r"^Encrypted = \+$", output, re.MULTILINE))
    multivolume = bool(re.search(r"^Multivolume = \+$", header, re.MULTILINE))
    volume_values = re.findall(r"^Volumes = (\d+)$", header, re.MULTILINE)
    volumes = max((int(value) for value in volume_values), default=1)
    if multivolume:
        volumes = max(volumes, 2)
    members: list[str] = []
    for block in re.split(r"\n\s*\n", member_listing.strip()):
        fields = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", maxsplit=1)
                fields[key] = value
        member = fields.get("Path")
        if member is None:
            continue
        if fields.get("Folder") == "+":
            normalized_directory = member.replace("\\", "/").rstrip("/")
            if normalized_directory != str(subclass_id):
                raise ValueError(f"unsafe or unexpected RAR directory: {member}")
            continue
        members.append(member)
    return validate_rar_members(
        subclass_id, members, encrypted=encrypted, volumes=volumes
    )


def _run_7zip_for_archive_verification(command: list[str], label: str):
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise MEP3MProvenanceError(f"7-Zip failed while verifying {label}") from error
    if not isinstance(completed.stdout, str) or not isinstance(completed.stderr, str):
        raise MEP3MProvenanceError(
            f"7-Zip returned invalid text output while verifying {label}"
        )
    return completed


def _discard_archive_verification_tree(root: Path) -> None:
    if not os.path.lexists(root):
        return
    try:
        _require_safe_tree(root, "MEP-3M archive verification directory")
    except MEP3MProvenanceError:
        # Preserve an unexpected filesystem shape for manual inspection instead
        # of traversing it during cleanup.
        return
    shutil.rmtree(root)


def _verify_archive_receipt_members(
    archive_snapshot: _DigestSnapshot,
    receipt: MEP3MExtractionReceipt,
) -> None:
    """Re-read every receipt member from the locked archive with 7-Zip."""

    label = f"MEP-3M archive {receipt.archive_subclass_id}"
    _verify_digest_snapshot(archive_snapshot, label)
    try:
        executable = _find_7zip()
    except RuntimeError as error:
        raise MEP3MProvenanceError(
            "formal MEP-3M archive verification requires 7-Zip"
        ) from error

    listing = _run_7zip_for_archive_verification(
        [executable, "l", "-slt", str(archive_snapshot.path)],
        f"{label} listing",
    )
    try:
        archive_members = _parse_7zip_listing(
            receipt.archive_subclass_id, listing.stdout
        )
    except ValueError as error:
        raise MEP3MProvenanceError(
            "MEP-3M archive listing output is invalid"
        ) from error
    archive_member_keys = {value.casefold() for value in archive_members}
    requested = tuple(member.member_path for member in receipt.members)
    missing = [
        member_path
        for member_path in requested
        if member_path.casefold() not in archive_member_keys
    ]
    if missing:
        raise MEP3MProvenanceError(
            "MEP-3M receipt members are absent from the verified archive: "
            f"{missing[:5]}"
        )

    verification_root = Path(
        tempfile.mkdtemp(prefix=".mep3m-archive-verification-")
    ).absolute()
    try:
        _require_real_directory(
            verification_root, "MEP-3M archive verification directory"
        )
        _run_7zip_for_archive_verification(
            [
                executable,
                "x",
                str(archive_snapshot.path),
                *requested,
                f"-o{verification_root}",
                "-y",
                "-bb0",
                "-bd",
            ],
            f"{label} selected-member extraction",
        )
        _require_safe_tree(verification_root, "MEP-3M archive verification directory")
        actual_paths = {
            path.relative_to(verification_root).as_posix()
            for path in verification_root.rglob("*")
            if path.is_file()
        }
        if actual_paths != set(requested):
            raise MEP3MProvenanceError(
                "MEP-3M archive extraction output does not exactly match receipt "
                "members"
            )
        for member in receipt.members:
            extracted = _snapshot_regular_file(
                verification_root.joinpath(*member.member_path.split("/")),
                f"MEP-3M archive member {member.member_path}",
            )
            crc32 = f"{zlib.crc32(extracted.content) & 0xFFFFFFFF:08x}"
            if (
                len(extracted.content) != member.uncompressed_size
                or extracted.sha256 != member.member_sha256
                or crc32 != member.crc32
            ):
                raise MEP3MProvenanceError(
                    "MEP-3M archive member bytes do not match extraction receipt: "
                    f"{member.member_path}"
                )
            _verify_file_snapshot(
                extracted, f"MEP-3M archive member {member.member_path}"
            )
        _require_safe_tree(verification_root, "MEP-3M archive verification directory")
        _verify_digest_snapshot(archive_snapshot, label)
    finally:
        _discard_archive_verification_tree(verification_root)


def _validate_extracted_staging(
    subclass_id: int, staging: Path, members: Iterable[str]
) -> None:
    for child in tuple(staging.iterdir()):
        if child.is_symlink():
            raise RuntimeError(f"unexpected extracted symlink: {child.name}")
        if child.is_dir():
            if child.name != str(subclass_id):
                raise RuntimeError(f"unexpected extracted directory: {child.name}")
            if any(child.iterdir()):
                raise RuntimeError(
                    f"extracted subclass root directory is not empty: {child.name}"
                )
            child.rmdir()
        elif not child.is_file():
            raise RuntimeError(f"unexpected extracted filesystem entry: {child.name}")

    expected_names = sorted(Path(member).name.lower() for member in members)
    actual_names = sorted(path.name.lower() for path in staging.iterdir())
    if actual_names != expected_names:
        raise RuntimeError(
            f"extracted member validation failed for subclass {subclass_id}"
        )


def extract(selected_ids: Iterable[int] = SELECTED_SUBCLASS_IDS) -> None:
    selected_ids = tuple(selected_ids)
    for subclass_id in selected_ids:
        _validate_local_artifact(
            ARCHIVES_DIR / f"{subclass_id}.rar", ARCHIVE_SPECS[subclass_id]
        )
    executable = _find_7zip()
    for subclass_id in selected_ids:
        archive = ARCHIVES_DIR / f"{subclass_id}.rar"
        listing = subprocess.run(
            [executable, "l", "-slt", str(archive)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        members = _parse_7zip_listing(subclass_id, listing.stdout)
        destination = EXTRACTED_DIR / str(subclass_id)
        staging = prepare_staging_directory(destination)
        try:
            subprocess.run(
                [executable, "e", str(archive), f"-o{staging}", "-y"], check=True
            )
            _validate_extracted_staging(subclass_id, staging, members)
            publish_staged_directory(staging, destination)
        except Exception:
            discard_staging_directory(staging)
            raise


def _false_to_none(value):
    if value is None or (isinstance(value, str) and value.strip().upper() == "FALSE"):
        return None
    return value


def iter_selected_annotations(
    parquet_path: Path,
    *,
    selected_ids: Iterable[int] = SELECTED_SUBCLASS_IDS,
    per_subclass: int | None = 1_000,
    limit: int | None = 10_000,
    batch_size: int = 1_024,
) -> Iterator[dict]:
    """Stream parquet row groups, retain selected subclasses, then round-robin."""
    selected = tuple(int(value) for value in selected_ids)
    selected_set = set(selected)
    queues = {subclass_id: deque() for subclass_id in selected}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            subclass_id = int(row["sub_class_id"])
            queue = queues.get(subclass_id)
            if subclass_id not in selected_set or (
                per_subclass is not None and len(queue) >= per_subclass
            ):
                continue
            normalized = dict(row)
            for field in ("title", "OCR", "subsub_class_name"):
                normalized[field] = _false_to_none(normalized.get(field))
            queue.append(normalized)

    emitted = 0
    while any(queues.values()) and (limit is None or emitted < limit):
        for subclass_id in selected:
            if limit is not None and emitted >= limit:
                return
            if queues[subclass_id]:
                yield queues[subclass_id].popleft()
                emitted += 1


def _normalize_product_row(row: Mapping) -> dict:
    return {field.name: row.get(field.name) for field in PRODUCT_SCHEMA}


def _stored_image_path(path: Path, parquet_path: Path) -> str:
    try:
        return path.relative_to(parquet_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _existing_image_path(row: Mapping, parquet_path: Path) -> Path:
    path = Path(str(row["image_path"]))
    return path if path.is_absolute() else parquet_path.parent / path


def _source_image_path(annotation: Mapping, extracted_dir: Path) -> Path:
    subclass_id = int(annotation["sub_class_id"])
    raw = str(annotation["img_path"]).replace("\\", "/").rstrip("/")
    stem = Path(raw).stem
    if not stem.isdigit():
        raise ValueError(f"unexpected annotation image path: {raw}")
    folder = extracted_dir / str(subclass_id)
    for suffix in (".jpg", ".JPG", ".png", ".PNG"):
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return folder / f"{stem}.jpg"


def _annotation_record_key(annotation: Mapping) -> tuple[int, str]:
    subclass_id = int(annotation["sub_class_id"])
    raw = str(annotation["img_path"]).replace("\\", "/").rstrip("/")
    stem = Path(raw).stem
    if not stem.isdigit():
        raise MEP3MProvenanceError(f"invalid MEP-3M annotation image path: {raw}")
    return subclass_id, stem


def _normalized_annotation(annotation: Mapping) -> dict:
    normalized = dict(annotation)
    for field in ("title", "OCR", "subsub_class_name"):
        normalized[field] = _false_to_none(normalized.get(field))
    return normalized


def _locked_annotation_rows(snapshot: _FileSnapshot) -> dict[tuple[int, str], dict]:
    try:
        rows = pq.read_table(pa.BufferReader(snapshot.content)).to_pylist()
    except Exception as error:
        raise MEP3MProvenanceError("locked MEP-3M annotations are invalid") from error
    indexed: dict[tuple[int, str], dict] = {}
    for row in rows:
        normalized = _normalized_annotation(row)
        key = _annotation_record_key(normalized)
        if key in indexed:
            raise MEP3MProvenanceError(
                f"locked MEP-3M annotation identity is duplicated: {key}"
            )
        indexed[key] = normalized
    return indexed


def _require_real_directory(
    path: Path, label: str
) -> tuple[tuple[str, int, int, int, int], ...]:
    identity = _ancestor_identity(Path(path), label, include_leaf=True)
    if not identity or Path(identity[-1][0]) != Path(path).absolute():
        raise MEP3MProvenanceError(f"unable to inspect {label}")
    return identity


def _require_safe_tree(path: Path, label: str) -> None:
    """Reject reparse traversal, hard-linked files, and special files in a tree."""

    root = Path(path).absolute()
    _require_real_directory(root, label)
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as error:
            raise MEP3MProvenanceError(f"unable to inspect {label}") from error
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                metadata = entry_path.lstat()
            except OSError as error:
                raise MEP3MProvenanceError(f"unable to inspect {label}") from error
            if _is_link_or_reparse(metadata):
                raise MEP3MProvenanceError(
                    f"{label} must not contain a symlink, junction, or reparse point"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(entry_path)
            elif stat.S_ISREG(metadata.st_mode):
                if int(metadata.st_nlink) != 1:
                    raise MEP3MProvenanceError(
                        f"{label} must not contain a hard-linked file"
                    )
            else:
                raise MEP3MProvenanceError(
                    f"{label} must not contain a special filesystem object"
                )


def _write_normalized_jpeg(image: Image.Image, destination: Path) -> str:
    image.save(destination, format="JPEG", quality=90, optimize=True)
    with Image.open(destination) as normalized:
        normalized.load()
        return str(imagehash.phash(normalized.convert("RGB")))


def _validate_selected_coverage(
    required_subclass_ids: set[int],
    accepted_subclass_ids: set[int],
    final_subclass_ids: set[int],
    final_class_ids: set[int],
) -> None:
    missing_accepted = required_subclass_ids - accepted_subclass_ids
    missing_final = required_subclass_ids - final_subclass_ids
    if missing_accepted or missing_final:
        raise RuntimeError(
            "MEP-3M subclass coverage failed: "
            f"no_valid={sorted(missing_accepted)} not_published={sorted(missing_final)}"
        )
    if (
        required_subclass_ids == set(SELECTED_SUBCLASS_IDS)
        and len(final_class_ids) < 14
    ):
        raise RuntimeError(
            "MEP-3M top-level class coverage failed: "
            f"expected at least 14, got {len(final_class_ids)}"
        )


def clean_dataset(
    *,
    annotations: Iterable[Mapping],
    extracted_dir: Path,
    output_dir: Path,
    parquet_path: Path,
    limit: int = 10_000,
    per_subclass: int = 1_000,
    min_side: int = 200,
    required_subclass_ids: Iterable[int] = SELECTED_SUBCLASS_IDS,
    annotation_source: Path | None = None,
    archive_paths: Mapping[int, Path] | None = None,
    extraction_receipt_paths: Mapping[int, Path] | None = None,
    expected_extraction_receipt_sha256_by_subclass: Mapping[int, str] | None = None,
    license_evidence_path: Path | None = None,
    license_review_path: Path | None = None,
    expected_license_review_sha256: str | None = None,
    source_lock_path: Path | None = None,
    expected_source_lock_sha256: str | None = None,
    source_lock: MEP3MSourceLock | None = None,
) -> CleanReport:
    """Filter, balance, and atomically merge MEP products into products.parquet."""
    output_dir = Path(output_dir)
    parquet_path = Path(parquet_path)
    extracted_dir = Path(extracted_dir)
    required_subclass_ids = {int(value) for value in required_subclass_ids}
    formal_requested = any(
        value is not None
        for value in (
            annotation_source,
            archive_paths,
            extraction_receipt_paths,
            expected_extraction_receipt_sha256_by_subclass,
            license_evidence_path,
            license_review_path,
            expected_license_review_sha256,
            source_lock_path,
            expected_source_lock_sha256,
            source_lock,
        )
    )
    if source_lock is not None:
        raise MEP3MProvenanceError(
            "in-process source_lock objects cannot establish formal identity; use a "
            "canonical source_lock_path and external expected_source_lock_sha256"
        )
    annotation_snapshot: _FileSnapshot | None = None
    archive_snapshots: dict[int, _DigestSnapshot] = {}
    receipt_by_subclass: dict[int, VerifiedMEP3MExtractionReceipt] = {}
    license_snapshot: _FileSnapshot | None = None
    verified_license: VerifiedMEP3MLicenseUseReview | None = None
    locked_annotations: dict[tuple[int, str], dict] | None = None
    lock_snapshot: _FileSnapshot | None = None
    lock_sha256: str | None = None
    verified_lock: MEP3MSourceLock | None = None
    if formal_requested:
        if (
            annotation_source is None
            or archive_paths is None
            or extraction_receipt_paths is None
            or expected_extraction_receipt_sha256_by_subclass is None
            or license_evidence_path is None
            or license_review_path is None
            or expected_license_review_sha256 is None
            or source_lock_path is None
            or expected_source_lock_sha256 is None
        ):
            raise MEP3MProvenanceError(
                "formal MEP-3M cleaning requires annotation_source, archive_paths, "
                "externally locked extraction_receipt_paths, license_evidence_path, "
                "an independently pinned license_review_path, source_lock_path, and "
                "all expected external SHA-256 values"
            )
        verified = load_verified_mep3m_source_lock(
            source_lock_path, expected_lock_sha256=expected_source_lock_sha256
        )
        verified_lock = verified.lock
        lock_snapshot = verified.snapshot
        lock_sha256 = verified.lock_sha256
        normalized_archive_paths = {
            int(key): Path(value) for key, value in archive_paths.items()
        }
        normalized_receipt_paths = {
            int(key): Path(value) for key, value in extraction_receipt_paths.items()
        }
        normalized_receipt_digests = {
            int(key): value
            for key, value in expected_extraction_receipt_sha256_by_subclass.items()
        }
        if set(normalized_archive_paths) != required_subclass_ids:
            raise MEP3MProvenanceError(
                "archive_paths must exactly cover required_subclass_ids"
            )
        if (
            set(normalized_receipt_paths) != required_subclass_ids
            or set(normalized_receipt_digests) != required_subclass_ids
        ):
            raise MEP3MProvenanceError(
                "extraction receipt paths and external digests must exactly cover "
                "required_subclass_ids"
            )
        if set(verified_lock.archive_sha256_by_subclass) != required_subclass_ids:
            raise MEP3MProvenanceError(
                "source lock archives must exactly cover required_subclass_ids"
            )
        annotation_snapshot = _snapshot_regular_file(
            Path(annotation_source), "MEP-3M annotation parquet"
        )
        if (
            len(annotation_snapshot.content) != ANNOTATION_SPEC.size
            or annotation_snapshot.sha256 != ANNOTATION_SPEC.sha256
            or annotation_snapshot.sha256 != verified_lock.annotation_sha256
        ):
            raise MEP3MProvenanceError(
                "annotation parquet does not match the pinned external source lock"
            )
        locked_annotations = _locked_annotation_rows(annotation_snapshot)
        verified_license = load_verified_mep3m_license_use_review(
            license_review_path,
            expected_review_sha256=expected_license_review_sha256,
            source_lock=verified,
            license_evidence_path=license_evidence_path,
        )
        license_snapshot = verified_license.evidence_snapshot
        annotation_values = tuple(
            _normalized_annotation(value) for value in annotations
        )
        for annotation in annotation_values:
            key = _annotation_record_key(annotation)
            if locked_annotations.get(key) != annotation:
                raise MEP3MProvenanceError(
                    f"annotation row does not match pinned parquet: {key}"
                )
        annotations = annotation_values
        for subclass_id, archive_path in normalized_archive_paths.items():
            snapshot = _digest_regular_file(
                archive_path, f"MEP-3M archive {subclass_id}"
            )
            assert isinstance(snapshot, _DigestSnapshot)
            spec = ARCHIVE_SPECS[subclass_id]
            if (
                snapshot.size != spec.size
                or snapshot.sha256 != spec.sha256
                or snapshot.sha256
                != verified_lock.archive_sha256_by_subclass[subclass_id]
            ):
                raise MEP3MProvenanceError(
                    f"archive {subclass_id} does not match the pinned external source lock"
                )
            archive_snapshots[subclass_id] = snapshot
        for subclass_id in sorted(required_subclass_ids):
            receipt_by_subclass[subclass_id] = load_verified_mep3m_extraction_receipt(
                normalized_receipt_paths[subclass_id],
                expected_receipt_sha256=normalized_receipt_digests[subclass_id],
                source_lock=verified_lock,
                archive_subclass_id=subclass_id,
                archive_snapshot=archive_snapshots[subclass_id],
            )
    if formal_requested:
        if os.path.lexists(output_dir):
            _require_safe_tree(output_dir, "MEP-3M output directory")
        else:
            _ancestor_identity(output_dir, "MEP-3M output directory")
        if os.path.lexists(parquet_path):
            _snapshot_regular_file(parquet_path, "MEP-3M existing product parquet")
        else:
            _ancestor_identity(parquet_path, "MEP-3M product parquet output")
        directory_staging = output_dir.with_name(f".{output_dir.name}.staging")
        directory_backup = output_dir.with_name(f".{output_dir.name}.backup")
        file_backup = parquet_path.with_name(f".{parquet_path.name}.backup")
        for residue, label in (
            (directory_staging, "MEP-3M staging residue"),
            (directory_backup, "MEP-3M directory backup"),
        ):
            if os.path.lexists(residue):
                _require_safe_tree(residue, label)
        if os.path.lexists(file_backup):
            _snapshot_regular_file(file_backup, "MEP-3M product parquet backup")
    recover_directory_and_file_publish(output_dir, parquet_path)
    if formal_requested and parquet_path.is_file():
        existing_parquet_snapshot = _snapshot_regular_file(
            parquet_path, "MEP-3M existing product parquet"
        )
        existing_rows = pq.read_table(
            pa.BufferReader(existing_parquet_snapshot.content)
        ).to_pylist()
    else:
        existing_parquet_snapshot = None
        existing_rows = (
            pq.read_table(parquet_path).to_pylist() if parquet_path.is_file() else []
        )
    retained_rows = [
        _normalize_product_row(row)
        for row in existing_rows
        if row.get("source") != "mep3m"
    ]
    seen_hashes: set[str] = set()
    retained_image_snapshots: list[_FileSnapshot] = []
    for row in retained_rows:
        image_path = _existing_image_path(row, parquet_path)
        try:
            if formal_requested:
                retained_snapshot = _snapshot_regular_file(
                    image_path, f"retained product image {row['product_id']}"
                )
                retained_image_snapshots.append(retained_snapshot)
                opened_image = Image.open(io.BytesIO(retained_snapshot.content))
            else:
                opened_image = Image.open(image_path)
            with opened_image as opened:
                opened.load()
                seen_hashes.add(str(imagehash.phash(ImageOps.exif_transpose(opened))))
        except MEP3MProvenanceError:
            raise
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"unreadable retained product image for {row['product_id']}: {image_path}"
            ) from error

    if formal_requested:
        _require_real_directory(extracted_dir, "MEP-3M extracted root")
    for subclass_id in sorted(required_subclass_ids):
        subclass_directory = extracted_dir / str(subclass_id)
        if formal_requested:
            _require_real_directory(
                subclass_directory, f"MEP-3M extracted subclass {subclass_id}"
            )
        elif not subclass_directory.is_dir():
            raise RuntimeError(
                f"missing extracted subclass directory for MEP-3M: {subclass_directory}"
            )

    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    if formal_requested and os.path.lexists(temporary_parquet):
        raise MEP3MProvenanceError(
            "MEP-3M temporary product parquet must not already exist"
        )
    staging = prepare_staging_directory(output_dir)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if formal_requested:
        _require_real_directory(staging, "MEP-3M staging directory")
        _require_real_directory(parquet_path.parent, "MEP-3M product parquet parent")
    if formal_requested:
        _ancestor_identity(temporary_parquet, "MEP-3M temporary product parquet")
    queues = {subclass_id: deque() for subclass_id in SELECTED_SUBCLASS_IDS}
    accepted_subclass_ids: set[int] = set()
    product_subclass_ids: dict[str, int] = {}
    product_class_ids: dict[str, int] = {}
    provenance_by_product_id: dict[str, MEP3MAssetProvenance] = {}
    source_image_snapshots: list[_FileSnapshot] = []
    normalized_image_snapshots: dict[str, _FileSnapshot] = {}
    temporary_parquet_snapshot: _FileSnapshot | None = None
    provenance_snapshot: _FileSnapshot | None = None
    duplicates = too_small = damaged = 0
    try:
        for annotation in annotations:
            subclass_id = int(annotation["sub_class_id"])
            queue = queues.setdefault(subclass_id, deque())
            source = _source_image_path(annotation, extracted_dir)
            if not source.is_file():
                raise RuntimeError(
                    f"missing extracted image for MEP-3M annotation: {source}"
                )
            source_snapshot: _FileSnapshot | None = None
            try:
                if verified_lock is None:
                    opened_source = Image.open(source)
                else:
                    source_snapshot = _snapshot_regular_file(
                        source, f"MEP-3M extracted image {subclass_id}"
                    )
                    member_path = f"{subclass_id}/{source.name}"
                    receipt = receipt_by_subclass[subclass_id]
                    members = {
                        member.member_path: member for member in receipt.receipt.members
                    }
                    member = members.get(member_path)
                    if member is None:
                        raise MEP3MProvenanceError(
                            f"extracted image is not a locked archive member: {member_path}"
                        )
                    actual_crc32 = (
                        f"{zlib.crc32(source_snapshot.content) & 0xFFFFFFFF:08x}"
                    )
                    if (
                        len(source_snapshot.content) != member.uncompressed_size
                        or source_snapshot.sha256 != member.member_sha256
                        or actual_crc32 != member.crc32
                    ):
                        raise MEP3MProvenanceError(
                            f"extracted image bytes do not match archive member: {member_path}"
                        )
                    source_image_snapshots.append(source_snapshot)
                    opened_source = Image.open(io.BytesIO(source_snapshot.content))
                with opened_source as opened:
                    opened.load()
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except MEP3MProvenanceError:
                raise
            except (OSError, ValueError):
                damaged += 1
                continue
            if min(image.size) < min_side:
                too_small += 1
                continue
            if len(queue) >= per_subclass:
                continue
            stem = Path(str(annotation["img_path"]).replace("\\", "/")).stem
            filename = f"mep3m-{subclass_id}-{stem}.jpg"
            staged_destination = staging / filename
            fingerprint = _write_normalized_jpeg(image, staged_destination)
            if fingerprint in seen_hashes:
                staged_destination.unlink()
                duplicates += 1
                continue
            seen_hashes.add(fingerprint)
            normalized_snapshot = _snapshot_regular_file(
                staged_destination, f"MEP-3M normalized image {subclass_id}/{stem}"
            )

            destination = output_dir / filename
            product_id = f"mep3m-{subclass_id}-{stem}"
            queue.append(
                {
                    "product_id": product_id,
                    "title": str(_false_to_none(annotation.get("title")) or "untitled"),
                    "category_l1": str(
                        annotation.get("class_name") or annotation.get("class_id")
                    ),
                    "category_l2": str(
                        annotation.get("sub_class_name")
                        or annotation.get("sub_class_id")
                    ),
                    "category_l3": _false_to_none(annotation.get("subsub_class_name")),
                    "ocr_text": _false_to_none(annotation.get("OCR")),
                    "image_path": _stored_image_path(destination, parquet_path),
                    "source": "mep3m",
                }
            )
            accepted_subclass_ids.add(subclass_id)
            product_subclass_ids[product_id] = subclass_id
            product_class_ids[product_id] = int(annotation["class_id"])
            if verified_lock is not None:
                assert source_snapshot is not None
                assert lock_sha256 is not None
                assert verified_license is not None
                receipt = receipt_by_subclass[subclass_id]
                member_path = f"{subclass_id}/{source.name}"
                member = next(
                    item
                    for item in receipt.receipt.members
                    if item.member_path == member_path
                )
                normalized_image_snapshots[product_id] = normalized_snapshot
                provenance_by_product_id[product_id] = MEP3MAssetProvenance(
                    asset_role="product_gallery_candidate",
                    identity_scope="single_asset_only",
                    exact_eligibility_candidate=False,
                    exact_exclusion_reason=_EXACT_EXCLUSION_REASON,
                    product_id=product_id,
                    filename=filename,
                    source_dataset=verified_lock.source_dataset,
                    source_revision=verified_lock.source_revision,
                    source_record_id=f"image:{subclass_id}/{stem}",
                    source_lock_sha256=lock_sha256,
                    annotation_sha256=verified_lock.annotation_sha256,
                    archive_subclass_id=subclass_id,
                    archive_sha256=verified_lock.archive_sha256_by_subclass[
                        subclass_id
                    ],
                    extraction_receipt_sha256=receipt.receipt_sha256,
                    archive_member_path=member.member_path,
                    archive_member_uncompressed_size=member.uncompressed_size,
                    archive_member_crc32=member.crc32,
                    source_image_sha256=source_snapshot.sha256,
                    normalized_asset_sha256=normalized_snapshot.sha256,
                    transform_policy_version=_TRANSFORM_POLICY_VERSION,
                    derivation_parent_asset_ids=(),
                    license_id=verified_lock.license_id,
                    license_evidence_sha256=verified_lock.license_evidence_sha256,
                    license_review_sha256=verified_license.review_sha256,
                    license_evidence_url=verified_lock.license_evidence_url,
                    license_evidence_revision=(
                        verified_license.review.license_evidence_revision
                    ),
                    source_url=verified_lock.source_url,
                    attribution=verified_lock.attribution,
                    local_noncommercial_research_allowed=(
                        verified_license.review.permissions.local_noncommercial_research_allowed
                    ),
                    local_noncommercial_embedding_allowed=(
                        verified_license.review.permissions.local_noncommercial_embedding_allowed
                    ),
                    remote_embedding_allowed=(
                        verified_license.review.permissions.remote_embedding_allowed
                    ),
                    cloud_upload_allowed=(
                        verified_license.review.permissions.cloud_upload_allowed
                    ),
                    redistribution_allowed=(
                        verified_license.review.permissions.redistribution_allowed
                    ),
                    public_demo_allowed=(
                        verified_license.review.permissions.public_demo_allowed
                    ),
                )

        mep_rows: list[dict] = []
        subclass_order = [
            *SELECTED_SUBCLASS_IDS,
            *sorted(set(queues) - set(SELECTED_SUBCLASS_IDS)),
        ]
        while any(queues.values()) and len(mep_rows) < limit:
            for subclass_id in subclass_order:
                if len(mep_rows) >= limit:
                    break
                if queues[subclass_id]:
                    mep_rows.append(queues[subclass_id].popleft())
        if len(mep_rows) < limit:
            raise RuntimeError(
                f"insufficient valid MEP-3M images: need {limit}, got {len(mep_rows)}"
            )
        final_product_ids = {row["product_id"] for row in mep_rows}
        _validate_selected_coverage(
            required_subclass_ids,
            accepted_subclass_ids,
            {product_subclass_ids[product_id] for product_id in final_product_ids},
            {product_class_ids[product_id] for product_id in final_product_ids},
        )

        selected_names = {Path(row["image_path"]).name for row in mep_rows}
        for path in tuple(staging.iterdir()):
            if path.name not in selected_names:
                path.unlink()
        rows = retained_rows + mep_rows
        table = pa.Table.from_pylist(rows, schema=PRODUCT_SCHEMA)
        pq.write_table(table, temporary_parquet, compression="zstd")
        if formal_requested:
            temporary_parquet_snapshot = _snapshot_regular_file(
                temporary_parquet, "MEP-3M staged product parquet"
            )
            staged_row_count = pq.read_table(
                pa.BufferReader(temporary_parquet_snapshot.content)
            ).num_rows
        else:
            staged_row_count = pq.read_table(temporary_parquet).num_rows
        if staged_row_count != len(rows):
            raise RuntimeError("MEP-3M parquet row validation failed")
        if len(list(staging.glob("mep3m-*.jpg"))) != len(mep_rows):
            raise RuntimeError("MEP-3M staged image validation failed")
        if verified_lock is None:
            (staging / _DIAGNOSTIC_MARKER).write_bytes(
                _canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "mode": "diagnostic",
                        "eligible_for_formal_export": False,
                        "exact_eligibility_coverage": 0,
                        "reason": "missing externally locked MEP-3M provenance",
                    }
                )
            )
        else:
            selected_provenance = tuple(
                provenance_by_product_id[row["product_id"]] for row in mep_rows
            )
            if not selected_provenance:
                raise MEP3MProvenanceError(
                    "formal MEP-3M cleaning produced no exportable assets"
                )
            (staging / _PROVENANCE_FILE).write_bytes(
                b"".join(
                    _canonical_json_bytes(record.model_dump(mode="json"))
                    for record in selected_provenance
                )
            )
            provenance_snapshot = _snapshot_regular_file(
                staging / _PROVENANCE_FILE, "MEP-3M staged provenance sidecar"
            )
            assert annotation_snapshot is not None
            assert lock_snapshot is not None
            if existing_parquet_snapshot is not None:
                _verify_file_snapshot(
                    existing_parquet_snapshot, "MEP-3M existing product parquet"
                )
            for snapshot in retained_image_snapshots:
                _verify_file_snapshot(snapshot, "retained product image")
            _verify_file_snapshot(annotation_snapshot, "MEP-3M annotation parquet")
            for subclass_id, snapshot in archive_snapshots.items():
                _verify_digest_snapshot(snapshot, f"MEP-3M archive {subclass_id}")
            for subclass_id, receipt in receipt_by_subclass.items():
                _verify_file_snapshot(
                    receipt.snapshot, f"MEP-3M extraction receipt {subclass_id}"
                )
            for snapshot in source_image_snapshots:
                _verify_file_snapshot(snapshot, "MEP-3M extracted image")
            for product_id in final_product_ids:
                _verify_file_snapshot(
                    normalized_image_snapshots[product_id],
                    f"MEP-3M normalized image {product_id}",
                )
            assert license_snapshot is not None
            _verify_file_snapshot(license_snapshot, "MEP-3M license evidence")
            assert verified_license is not None
            _verify_file_snapshot(
                verified_license.snapshot, "MEP-3M human licence/use review"
            )
            _verify_file_snapshot(lock_snapshot, "MEP-3M source lock")
            assert temporary_parquet_snapshot is not None
            assert provenance_snapshot is not None
            _verify_file_snapshot(
                temporary_parquet_snapshot, "MEP-3M staged product parquet"
            )
            _verify_file_snapshot(
                provenance_snapshot, "MEP-3M staged provenance sidecar"
            )
            _require_safe_tree(staging, "MEP-3M staging directory")
            if os.path.lexists(directory_backup) or os.path.lexists(file_backup):
                raise MEP3MProvenanceError(
                    "MEP-3M publish backup appeared during formal cleaning"
                )
        publish_staged_directory_and_file(
            staging, output_dir, temporary_parquet, parquet_path
        )
    except Exception:
        if os.path.lexists(staging):
            try:
                _require_safe_tree(staging, "MEP-3M failed staging directory")
            except MEP3MProvenanceError:
                # Leave an unsafe residue untouched; the next formal run also
                # refuses it instead of traversing an attacker-controlled tree.
                pass
            else:
                discard_staging_directory(staging)
        if os.path.lexists(temporary_parquet):
            try:
                _snapshot_regular_file(
                    temporary_parquet, "MEP-3M failed temporary product parquet"
                )
            except MEP3MProvenanceError:
                pass
            else:
                temporary_parquet.unlink()
        raise

    if verified_lock is not None:
        assert temporary_parquet_snapshot is not None
        assert provenance_snapshot is not None
        published_parquet = _digest_regular_file(
            parquet_path, "MEP-3M published product parquet"
        )
        assert isinstance(published_parquet, _DigestSnapshot)
        if published_parquet.sha256 != temporary_parquet_snapshot.sha256:
            raise MEP3MProvenanceError(
                "published product parquet changed after atomic publish"
            )
        published_provenance = _digest_regular_file(
            output_dir / _PROVENANCE_FILE, "MEP-3M published provenance sidecar"
        )
        assert isinstance(published_provenance, _DigestSnapshot)
        if published_provenance.sha256 != provenance_snapshot.sha256:
            raise MEP3MProvenanceError(
                "published provenance changed after atomic publish"
            )
        for record in selected_provenance:
            published = _digest_regular_file(
                output_dir / record.filename,
                f"MEP-3M published image {record.product_id}",
            )
            assert isinstance(published, _DigestSnapshot)
            if published.sha256 != record.normalized_asset_sha256:
                raise MEP3MProvenanceError(
                    f"published image changed after atomic publish: {record.product_id}"
                )

    return CleanReport(
        kept=len(mep_rows),
        duplicates=duplicates,
        too_small=too_small,
        damaged=damaged,
        image_paths=tuple(
            output_dir / Path(row["image_path"]).name for row in mep_rows
        ),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MEP3MProvenanceError(f"duplicate provenance JSON key: {key}")
        result[key] = value
    return result


def _load_provenance_records(
    path: Path, *, expected_provenance_sha256: str
) -> tuple[MEP3MAssetProvenance, ...]:
    _require_external_sha256(
        expected_provenance_sha256, "formal MEP-3M provenance sidecar"
    )
    snapshot = _snapshot_regular_file(path, "MEP-3M provenance sidecar")
    if snapshot.sha256 != expected_provenance_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M provenance sidecar does not match external expected digest"
        )
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MEP3MProvenanceError("MEP-3M provenance must be UTF-8") from error
    records: list[MEP3MAssetProvenance] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MEP3MProvenanceError("MEP-3M provenance contains a blank row")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            record = MEP3MAssetProvenance.model_validate(raw, strict=True)
        except Exception as error:
            raise MEP3MProvenanceError(
                f"MEP-3M provenance row {line_number} is invalid"
            ) from error
        if _canonical_json_bytes(record.model_dump(mode="json")) != (
            line + "\n"
        ).encode("utf-8"):
            raise MEP3MProvenanceError("MEP-3M provenance must be canonical JSONL")
        records.append(record)
    if not records:
        raise MEP3MProvenanceError("MEP-3M provenance must not be empty")
    _verify_file_snapshot(snapshot, "MEP-3M provenance sidecar")
    return tuple(records)


def load_verified_mep3m_provenance(
    path: Path | str, *, expected_provenance_sha256: str
) -> tuple[MEP3MAssetProvenance, ...]:
    """Re-open a provenance sidecar under a caller-owned digest trust root."""

    return _load_provenance_records(
        Path(path), expected_provenance_sha256=expected_provenance_sha256
    )


def _validate_record_against_lock(
    record: MEP3MAssetProvenance,
    source_lock: MEP3MSourceLock,
    source_lock_sha256: str,
    license_review: VerifiedMEP3MLicenseUseReview,
) -> None:
    expected_archive_sha = source_lock.archive_sha256_by_subclass.get(
        record.archive_subclass_id
    )
    if (
        record.source_revision != source_lock.source_revision
        or record.source_lock_sha256 != source_lock_sha256
        or record.annotation_sha256 != source_lock.annotation_sha256
        or expected_archive_sha is None
        or record.archive_sha256 != expected_archive_sha
        or record.license_id != source_lock.license_id
        or record.license_evidence_sha256 != source_lock.license_evidence_sha256
        or record.license_review_sha256 != license_review.review_sha256
        or record.license_review_policy_version != _LICENSE_REVIEW_POLICY_VERSION
        or record.license_evidence_url != source_lock.license_evidence_url
        or record.license_evidence_revision
        != license_review.review.license_evidence_revision
        or record.source_url != source_lock.source_url
        or record.attribution != source_lock.attribution
        or record.local_noncommercial_research_allowed
        != license_review.review.permissions.local_noncommercial_research_allowed
        or record.local_noncommercial_embedding_allowed
        != license_review.review.permissions.local_noncommercial_embedding_allowed
        or record.remote_embedding_allowed
        != license_review.review.permissions.remote_embedding_allowed
        or record.cloud_upload_allowed
        != license_review.review.permissions.cloud_upload_allowed
        or record.redistribution_allowed
        != license_review.review.permissions.redistribution_allowed
        or record.public_demo_allowed
        != license_review.review.permissions.public_demo_allowed
    ):
        raise MEP3MProvenanceError(
            f"MEP-3M provenance does not match the external source lock: {record.product_id}"
        )
    parts = record.product_id.split("-")
    expected_record_id = f"image:{parts[1]}/{parts[2]}"
    expected_member_prefix = f"{parts[1]}/{parts[2]}."
    if (
        record.filename != f"{record.product_id}.jpg"
        or record.source_record_id != expected_record_id
        or record.archive_subclass_id != int(parts[1])
        or not record.archive_member_path.startswith(expected_member_prefix)
        or Path(record.archive_member_path).suffix
        not in {".jpg", ".JPG", ".png", ".PNG"}
    ):
        raise MEP3MProvenanceError("MEP-3M source record identity is inconsistent")
    if (
        record.identity_scope != "single_asset_only"
        or record.exact_eligibility_candidate is not False
        or record.exact_exclusion_reason != _EXACT_EXCLUSION_REASON
    ):
        raise MEP3MProvenanceError(
            "MEP-3M annotations do not prove an Exact multi-view identity"
        )


def _require_product_directory(asset_root: Path, product_root: Path) -> None:
    try:
        relative = product_root.relative_to(asset_root)
    except ValueError:
        raise MEP3MProvenanceError(
            "product_image_root must stay below asset_root"
        ) from None
    current = asset_root
    _require_real_directory(current, "MEP-3M asset root")
    for part in relative.parts:
        current = current / part
        _require_real_directory(current, "MEP-3M product directory")


def audit_exact_eligibility(
    *,
    product_image_root: Path,
    source_lock_path: Path,
    expected_source_lock_sha256: str,
    license_evidence_path: Path,
    license_review_path: Path,
    expected_license_review_sha256: str,
    expected_provenance_sha256: str,
) -> ExactEligibilityCoverage:
    """Report the fail-closed Exact coverage implied by the actual annotations."""

    verified = load_verified_mep3m_source_lock(
        source_lock_path, expected_lock_sha256=expected_source_lock_sha256
    )
    source_lock = verified.lock
    license_review = load_verified_mep3m_license_use_review(
        license_review_path,
        expected_review_sha256=expected_license_review_sha256,
        source_lock=verified,
        license_evidence_path=license_evidence_path,
    )
    product_image_root = Path(product_image_root).absolute()
    _require_real_directory(product_image_root, "MEP-3M product directory")
    if os.path.lexists(product_image_root / _DIAGNOSTIC_MARKER):
        raise MEP3MProvenanceError(
            "diagnostic MEP-3M cleaning has no formal Exact eligibility"
        )
    records = _load_provenance_records(
        product_image_root / _PROVENANCE_FILE,
        expected_provenance_sha256=expected_provenance_sha256,
    )
    for record in records:
        _validate_record_against_lock(
            record, source_lock, verified.lock_sha256, license_review
        )
    _verify_file_snapshot(verified.snapshot, "MEP-3M source lock")
    _verify_file_snapshot(license_review.evidence_snapshot, "MEP-3M license evidence")
    _verify_file_snapshot(license_review.snapshot, "MEP-3M human licence/use review")
    return ExactEligibilityCoverage(
        total_assets=len(records),
        eligible_assets=0,
        stable_product_groups=0,
        exclusion_reason=_EXACT_EXCLUSION_REASON,
    )


def export_dataset_asset_drafts(
    *,
    parquet_path: Path,
    asset_root: Path,
    product_image_root: Path,
    output_path: Path,
    source_lock_path: Path,
    expected_source_lock_sha256: str,
    license_evidence_path: Path,
    license_review_path: Path,
    expected_license_review_sha256: str,
    expected_provenance_sha256: str,
) -> tuple[DatasetAssetDraft, ...]:
    """Export locked assets without claiming unsupported multi-view identity."""

    verified = load_verified_mep3m_source_lock(
        source_lock_path, expected_lock_sha256=expected_source_lock_sha256
    )
    source_lock = verified.lock
    license_review = load_verified_mep3m_license_use_review(
        license_review_path,
        expected_review_sha256=expected_license_review_sha256,
        source_lock=verified,
        license_evidence_path=license_evidence_path,
    )
    parquet_path = Path(parquet_path)
    asset_root = Path(asset_root).absolute()
    product_image_root = Path(product_image_root).absolute()
    _require_product_directory(asset_root, product_image_root)
    if os.path.lexists(product_image_root / _DIAGNOSTIC_MARKER):
        raise MEP3MProvenanceError(
            "diagnostic MEP-3M cleaning cannot be formally exported"
        )
    records = _load_provenance_records(
        product_image_root / _PROVENANCE_FILE,
        expected_provenance_sha256=expected_provenance_sha256,
    )
    parquet_snapshot = _snapshot_regular_file(parquet_path, "MEP-3M product parquet")
    try:
        rows = pq.read_table(pa.BufferReader(parquet_snapshot.content)).to_pylist()
    except Exception as error:
        raise MEP3MProvenanceError("MEP-3M product parquet is invalid") from error
    mep_rows = [row for row in rows if row.get("source") == "mep3m"]
    product_ids = [str(row["product_id"]) for row in mep_rows]
    if len(product_ids) != len(set(product_ids)):
        raise MEP3MProvenanceError("MEP-3M product ids must be unique")
    rows_by_product_id = {str(row["product_id"]): row for row in mep_rows}
    if len(rows_by_product_id) != len(records):
        raise MEP3MProvenanceError(
            "MEP-3M provenance does not cover every cleaned asset exactly once"
        )

    drafts: list[DatasetAssetDraft] = []
    image_snapshots: list[_FileSnapshot] = []
    seen: set[str] = set()
    for record in records:
        if record.product_id in seen:
            raise MEP3MProvenanceError("MEP-3M provenance product ids must be unique")
        seen.add(record.product_id)
        _validate_record_against_lock(
            record, source_lock, verified.lock_sha256, license_review
        )
        row = rows_by_product_id.get(record.product_id)
        if row is None:
            raise MEP3MProvenanceError("MEP-3M provenance references an unknown row")
        image_path = product_image_root / record.filename
        image_snapshot = _snapshot_regular_file(
            image_path, f"MEP-3M cleaned image {record.product_id}"
        )
        image_snapshots.append(image_snapshot)
        if image_snapshot.sha256 != record.normalized_asset_sha256:
            raise MEP3MProvenanceError(
                f"cleaned image bytes do not match frozen provenance: {record.product_id}"
            )
        stored = Path(str(row["image_path"]))
        stored_path = (
            stored.absolute()
            if stored.is_absolute()
            else (parquet_path.parent / stored).absolute()
        )
        if stored_path != image_path:
            raise MEP3MProvenanceError(
                "MEP-3M product row points outside the locked product directory"
            )
        drafts.append(
            DatasetAssetDraft(
                source_dataset=record.source_dataset,
                source_revision=record.source_revision,
                source_record_id=record.source_record_id,
                transform_policy_version=record.transform_policy_version,
                local_path=image_path.relative_to(asset_root).as_posix(),
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
    _verify_file_snapshot(parquet_snapshot, "MEP-3M product parquet")
    for snapshot in image_snapshots:
        _verify_file_snapshot(snapshot, "MEP-3M cleaned image")
    _verify_file_snapshot(verified.snapshot, "MEP-3M source lock")
    _verify_file_snapshot(license_review.evidence_snapshot, "MEP-3M license evidence")
    _verify_file_snapshot(license_review.snapshot, "MEP-3M human licence/use review")
    output_path = Path(output_path).absolute()
    _ancestor_identity(output_path, "MEP-3M draft bundle output")
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"MEP-3M draft bundle already exists; refusing overwrite: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(output_path.parent, "MEP-3M draft bundle parent")
    atomic_create_file(
        output_path,
        b"".join(
            _canonical_json_bytes(draft.model_dump(mode="json")) for draft in drafts
        ),
    )
    bundle_snapshot = _snapshot_regular_file(output_path, "MEP-3M draft bundle")
    _verify_file_snapshot(bundle_snapshot, "MEP-3M draft bundle")
    return tuple(drafts)


def load_canonical_mep3m_draft_bundle(
    path: Path | str, *, expected_bundle_sha256: str
) -> CanonicalMEP3MDraftBundle:
    """Check canonical draft bytes without granting formal dataset identity."""

    _require_external_sha256(expected_bundle_sha256, "MEP-3M draft bundle")
    snapshot = _snapshot_regular_file(Path(path), "MEP-3M draft bundle")
    if snapshot.sha256 != expected_bundle_sha256:
        raise MEP3MProvenanceError(
            "MEP-3M draft bundle does not match external expected digest"
        )
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MEP3MProvenanceError("MEP-3M draft bundle must be UTF-8") from error
    drafts: list[DatasetAssetDraft] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MEP3MProvenanceError("MEP-3M draft bundle contains a blank row")
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            draft = DatasetAssetDraft.model_validate(raw, strict=True)
        except Exception as error:
            raise MEP3MProvenanceError(
                f"MEP-3M draft bundle row {line_number} is invalid"
            ) from error
        if (
            draft.source_dataset != "mep3m"
            or draft.source_revision != REVISION
            or draft.transform_policy_version != _TRANSFORM_POLICY_VERSION
            or draft.cloud_upload_allowed is not False
            or draft.public_demo_allowed is not False
        ):
            raise MEP3MProvenanceError(
                "MEP-3M canonical draft row contradicts the provisional source policy"
            )
        if _canonical_json_bytes(draft.model_dump(mode="json")) != (line + "\n").encode(
            "utf-8"
        ):
            raise MEP3MProvenanceError("MEP-3M draft bundle must be canonical JSONL")
        drafts.append(draft)
    if not drafts:
        raise MEP3MProvenanceError("MEP-3M draft bundle must not be empty")
    identities = [
        (draft.product_id or "", draft.local_path, draft.source_record_id)
        for draft in drafts
    ]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise MEP3MProvenanceError(
            "MEP-3M draft bundle identities must be unique and sorted"
        )
    _verify_file_snapshot(snapshot, "MEP-3M draft bundle")
    return CanonicalMEP3MDraftBundle(tuple(drafts), snapshot.sha256, snapshot)


def load_verified_mep3m_draft_bundle(
    path: Path | str, *, expected_bundle_sha256: str
) -> CanonicalMEP3MDraftBundle:
    """Retired fail-closed shim for the former source-unbound formal loader."""

    del path, expected_bundle_sha256
    raise MEP3MProvenanceError(
        "MEP-3M draft JSONL is source-unbound and cannot receive formal verified "
        "identity; use load_canonical_mep3m_draft_bundle for provisional inspection"
    )


def write_preview(
    image_paths: Iterable[Path], output: Path, tile_size: int = 256
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (tile_size * 3, tile_size * 3), "white")
    for index, path in enumerate(list(image_paths)[:9]):
        with Image.open(path) as opened:
            tile = ImageOps.fit(opened.convert("RGB"), (tile_size, tile_size))
        canvas.paste(tile, ((index % 3) * tile_size, (index // 3) * tile_size))
    canvas.save(output, format="JPEG", quality=90)


def probe(limit: int = 5) -> None:
    annotation_path = ANNOTATIONS_DIR / "0000.parquet"
    _validate_local_artifact(annotation_path, ANNOTATION_SPEC)
    for subclass_id in SELECTED_SUBCLASS_IDS:
        _validate_local_artifact(
            ARCHIVES_DIR / f"{subclass_id}.rar", ARCHIVE_SPECS[subclass_id]
        )
    parquet = pq.ParquetFile(annotation_path)
    print(f"schema={parquet.schema_arrow}")
    head: list[dict] = []
    counts = {subclass_id: 0 for subclass_id in SELECTED_SUBCLASS_IDS}
    for batch in parquet.iter_batches(batch_size=1_024):
        for row in batch.to_pylist():
            if len(head) < limit:
                head.append(row)
            subclass_id = int(row["sub_class_id"])
            if subclass_id in counts:
                counts[subclass_id] += 1
    for index, row in enumerate(head):
        print(f"row[{index}]={row}")
    print(f"selected_counts={json.dumps(counts, sort_keys=True)}")
    executable = _find_7zip()
    for subclass_id in SELECTED_SUBCLASS_IDS:
        archive = ARCHIVES_DIR / f"{subclass_id}.rar"
        listing = subprocess.run(
            [executable, "l", "-slt", str(archive)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        members = _parse_7zip_listing(subclass_id, listing.stdout)
        print(f"archive[{subclass_id}] members={len(members)} first={members[0]}")
        extracted = next((EXTRACTED_DIR / str(subclass_id)).glob("*"), None)
        if extracted:
            with Image.open(extracted) as image:
                print(
                    f"image[{subclass_id}] format={image.format} mode={image.mode} size={image.size}"
                )


def clean(limit: int = 10_000) -> CleanReport:
    annotation_path = ANNOTATIONS_DIR / "0000.parquet"
    _validate_local_artifact(annotation_path, ANNOTATION_SPEC)
    annotations = iter_selected_annotations(
        annotation_path, per_subclass=None, limit=None
    )
    report = clean_dataset(
        annotations=annotations,
        extracted_dir=EXTRACTED_DIR,
        output_dir=CLEAN_DIR / "product_images" / "mep3m",
        parquet_path=CLEAN_DIR / "products.parquet",
        limit=limit,
        required_subclass_ids=SELECTED_SUBCLASS_IDS,
    )
    write_preview(report.image_paths, CLEAN_DIR / "preview_mep3m.jpg")
    print(
        f"kept={report.kept} duplicates={report.duplicates} "
        f"too_small={report.too_small} damaged={report.damaged}"
    )
    return report


def stats() -> None:
    rows = pq.read_table(CLEAN_DIR / "products.parquet").to_pylist()
    sources: dict[str, int] = {}
    mep_l1: dict[str, int] = {}
    mep_l2: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        sources[source] = sources.get(source, 0) + 1
        if source == "mep3m":
            mep_l1[row["category_l1"]] = mep_l1.get(row["category_l1"], 0) + 1
            mep_l2[row["category_l2"]] = mep_l2.get(row["category_l2"], 0) + 1
    muge_unknown = sum(
        row["source"] == "muge" and row["category_l1"] == "unknown" for row in rows
    )
    print(f"products={len(rows)}")
    print(f"sources={json.dumps(sources, ensure_ascii=False, sort_keys=True)}")
    print(f"mep_category_l1={json.dumps(mep_l1, ensure_ascii=False, sort_keys=True)}")
    print(f"mep_category_l2={json.dumps(mep_l2, ensure_ascii=False, sort_keys=True)}")
    print(f"muge_unknown={muge_unknown}")
    print(f"preview={CLEAN_DIR / 'preview_mep3m.jpg'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    subparsers.add_parser("extract")
    probe_parser = subparsers.add_parser("probe", aliases=["inspect"])
    probe_parser.add_argument("--limit", type=int, default=5)
    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--limit", type=int, default=10_000)
    subparsers.add_parser("stats")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "download":
        download()
    elif args.command == "extract":
        extract()
    elif args.command in {"probe", "inspect"}:
        probe(args.limit)
    elif args.command == "clean":
        clean(args.limit)
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
