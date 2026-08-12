"""Self-contained, deterministic contracts for static Skill authoring.

The author sees one canonical request containing every allowed byte.  The
in-process transport implemented here is intentionally *never* considered a
mechanically isolated formal author.  It is useful for deterministic tests and
diagnostic runs; callers requesting formal eligibility fail before invocation.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Lock
import time
from typing import Annotated, Any, Literal, Mapping, Protocol, Self
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain import config
from skillchain import llm as llm_module
from skillchain.llm import LLMResponse, Provider
from skillchain.schemas import Skill
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.task_spec import CapabilityTaskSpec, TaskSpecification, ToolName
from skillchain.taxonomy import TaxonomyRegistry
from skillchain.tools.registry import (
    RegistryError,
    RegistryManifest,
    ToolExecutionContext,
    ToolInvocationResult,
    ToolRegistry,
    ToolSpec,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_FORBIDDEN_STRONG = re.compile(
    r"(?<![a-z0-9])(?:corpus|eval(?:uation)?|bank|trajector(?:y|ies)|labels?|"
    r"rubrics?|gold|judges?)(?![a-z0-9])",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_WORD_PATTERN = r"(?:responses?|gates?|tests?|results?)"
_FORBIDDEN_PATH = re.compile(
    rf"(?<![a-z0-9]){_FORBIDDEN_PATH_WORD_PATTERN}(?![a-z0-9])",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_SEPARATOR = re.compile(
    rf"(?:"
    rf"(?<![a-z0-9]){_FORBIDDEN_PATH_WORD_PATTERN}(?![a-z0-9])\s*[/\\]"
    rf"|[/\\]\s*{_FORBIDDEN_PATH_WORD_PATTERN}(?![a-z0-9])"
    rf")",
    re.IGNORECASE,
)
_PATH_MARKERS = ("/", "\\", ".json", ".jsonl", ".parquet")
_VERIFIED_HANDLE_TOKEN = object()
_VERIFIED_SANDBOX_PROFILE_TOKEN = object()
_VERIFIED_CALL_AUTHORIZATION_TOKEN = object()
_CONSUMED_CALL_AUTHORIZATION_TOKEN = object()
_FORMAL_AUTHORING_INVOCATION_TOKEN = object()
_SKILL_MARKDOWN_PREFIX = b"<!-- skillchain-static-skill-v1\n"
_SKILL_MARKDOWN_SEPARATOR = b"-->\n"
CANONICAL_AUTHORING_RESPONSE_FORMAT = "canonical_authoring_payload_v1"
FORCED_SUBMISSION_RESPONSE_FORMAT = "forced_submission_function_v1"
AUTHORING_SUBMISSION_TOOL_NAME = "submit_authoring_payload"
PROVIDER_JSON_CONTENT_RESPONSE_FORMAT = "provider_json_content_v2"
AUTHORING_CONTENT_SCHEMA_NAME = "skillchain_authoring_content_v2"
_LEGACY_AUTHORING_TOOL_NAMES = (
    "image_product_search",
    "text_product_search",
    "style_similar_search",
    "encyclopedia_lookup",
    "recipe_lookup",
    "object_detect",
    "document_ocr",
)


class AuthoringContractError(ValueError):
    """An authoring input, output, or audit artifact failed closed."""


class FormalAuthoringJobError(AuthoringContractError):
    """The isolated worker failed after the parent captured its process audit."""

    def __init__(
        self,
        *,
        exit_code: int,
        stdout: bytes,
        stderr: bytes,
        elapsed_ms: int,
        command_sha256: str,
    ) -> None:
        super().__init__("isolated authoring job returned failure")
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed_ms = elapsed_ms
        self.command_sha256 = command_sha256


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _nonblank(value: str, name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-blank without edge whitespace")
    return value


def _digest_without(payload: dict[str, object], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_json_bytes(unsigned))


def _scan_untrusted_text(value: object, label: str) -> None:
    """Reject private-input names and path-like indirection in authored text."""

    def contains_forbidden_path_reference(item: str) -> bool:
        # Keep the path marker and its forbidden component local to the same
        # token.  A sentence may legitimately discuss a tool ``result`` and,
        # elsewhere, use a slash in domain prose such as ``product/evidence``.
        # Treating those distant fragments as one path rejected safe authored
        # instructions while providing no additional path-leak protection.
        return _FORBIDDEN_PATH_SEPARATOR.search(item) is not None or any(
            any(marker in token.casefold() for marker in _PATH_MARKERS)
            and _FORBIDDEN_PATH.search(token) is not None
            for token in item.split()
        )

    def visit(item: object) -> None:
        if isinstance(item, str):
            if _FORBIDDEN_STRONG.search(item) or contains_forbidden_path_reference(
                item
            ):
                raise AuthoringContractError(
                    f"{label} references forbidden experiment-derived information"
                )
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)

    visit(value)


def _scan_untrusted_bytes(value: bytes, label: str) -> None:
    """Scan the exact UTF-8 text exposed to an author, not only metadata."""

    _scan_untrusted_text(_decode_locked_utf8(value, label), label)


def _decode_locked_utf8(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AuthoringContractError(
            f"{label} must be a locked UTF-8 text excerpt"
        ) from error


def _canonical_model_bytes(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json"))


class CanonicalSpecification(_StrictFrozenModel):
    specification_kind: Literal["taxonomy", "task_specification", "tool_registry"]
    version: str
    identity_sha256: Sha256
    canonical_json: str
    bytes_sha256: Sha256

    @field_validator("version")
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("canonical_json")
    @classmethod
    def require_content(cls, value: str) -> str:
        if not value:
            raise ValueError("canonical_json must not be empty")
        return value

    @model_validator(mode="after")
    def verify_canonical_bytes(self) -> Self:
        content = self.canonical_json.encode("utf-8")
        if sha256_bytes(content) != self.bytes_sha256:
            raise ValueError("specification bytes_sha256 mismatch")
        try:
            parsed = parse_canonical_json(
                content, label=f"{self.specification_kind} specification"
            )
        except ArtifactFormatError as error:
            raise ValueError("specification bytes must be canonical JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("specification must contain an object")
        return self

    @property
    def content(self) -> bytes:
        return self.canonical_json.encode("utf-8")


class PromptIdentity(_StrictFrozenModel):
    prompt_id: str
    prompt_version: str
    template: str
    prompt_sha256: Sha256

    @field_validator("prompt_id", "prompt_version", "template")
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        if self.prompt_sha256 != sha256_bytes(self.template.encode("utf-8")):
            raise ValueError("prompt_sha256 mismatch")
        _scan_untrusted_text(self.template, "authoring prompt")
        return self


class CompilerIdentity(_StrictFrozenModel):
    compiler_id: Literal["skillchain.static-authoring"] = "skillchain.static-authoring"
    compiler_version: Literal["3.0.0", "4.0.0"] = "3.0.0"


class ModelIdentity(_StrictFrozenModel):
    provider: Provider
    endpoint: str
    model: str
    revision: str

    @field_validator("endpoint", "model", "revision")
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        if self.endpoint != config.PROVIDER_ENDPOINTS[self.provider]:
            raise ValueError("model endpoint does not match provider configuration")
        if self.provider == "claude":
            raise ValueError(
                "static authoring requires provider seed support; claude is unsupported"
            )
        if self.revision.casefold() in {
            "current",
            "head",
            "latest",
            "main",
            "master",
            "unknown",
            "unpinned",
        }:
            raise ValueError("author model revision must be immutable")
        if not self.model.endswith(f"-{self.revision}"):
            raise ValueError("author model ID must end with its immutable revision")
        return self


class FixedDecoding(_StrictFrozenModel):
    temperature_milli: Literal[0] = 0
    top_p_milli: Literal[1000] = 1000
    seed: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    response_format: Literal[
        "canonical_authoring_payload_v1",
        "forced_submission_function_v1",
        "provider_json_content_v2",
    ] = CANONICAL_AUTHORING_RESPONSE_FORMAT


class PriceSchedule(_StrictFrozenModel):
    schedule_id: str
    provider: Provider
    model: str
    input_microusd_per_million_tokens: int = Field(ge=0)
    output_microusd_per_million_tokens: int = Field(ge=0)
    source_currency: Literal["CNY"] = "CNY"
    source_input_microunits_per_million_tokens: int = Field(ge=0)
    source_output_microunits_per_million_tokens: int = Field(ge=0)
    conversion_microusd_per_source_unit: int = Field(gt=0)
    pricing_scope: Literal["china-beijing"] = "china-beijing"
    pricing_mode: Literal["non-thinking-realtime"] = "non-thinking-realtime"
    tier_max_input_tokens: int = Field(ge=1)
    source_url: str
    source_revision: str

    @field_validator("schedule_id", "model", "source_url", "source_revision")
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_conversion(self) -> Self:
        if not self.source_url.startswith("https://"):
            raise ValueError("price source URL must use HTTPS")
        expected_input = (
            self.source_input_microunits_per_million_tokens
            * self.conversion_microusd_per_source_unit
            + 999_999
        ) // 1_000_000
        expected_output = (
            self.source_output_microunits_per_million_tokens
            * self.conversion_microusd_per_source_unit
            + 999_999
        ) // 1_000_000
        if (
            self.input_microusd_per_million_tokens != expected_input
            or self.output_microusd_per_million_tokens != expected_output
        ):
            raise ValueError("micro-USD prices do not match the frozen conversion")
        return self


@dataclass(frozen=True)
class VerifiedPriceSchedule:
    """Price bytes loaded from a caller-pinned, stable external file."""

    value: PriceSchedule
    file_sha256: str
    path: Path
    _verification_token: object = field(repr=False, compare=False)


def load_verified_price_schedule(
    path: str | Path, *, expected_file_sha256: str
) -> VerifiedPriceSchedule:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise AuthoringContractError("expected price-schedule SHA-256 is invalid")
    try:
        content = read_stable_regular_file(path, label="price schedule")
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError("price-schedule file digest mismatch")
        raw = parse_canonical_json(content, label="price schedule")
        if not isinstance(raw, dict):
            raise AuthoringContractError("price schedule must contain an object")
        value = PriceSchedule.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError(
            "price schedule is not canonical and valid"
        ) from error
    return VerifiedPriceSchedule(
        value=value,
        file_sha256=expected_file_sha256,
        path=Path(path).resolve(strict=True),
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


class AuthoringBudgets(_StrictFrozenModel):
    max_successful_calls: Literal[1] = 1
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_total_tokens: int = Field(ge=2)
    max_cost_microusd: int = Field(ge=0)
    max_human_review_minutes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.max_total_tokens > self.max_input_tokens + self.max_output_tokens:
            raise ValueError("max_total_tokens exceeds component token ceilings")
        return self


class PublicSourceLock(_StrictFrozenModel):
    """External acquisition/license trust root; contents remain separate files."""

    schema_version: Literal[1] = 1
    source_id: str
    url: str
    revision: str
    media_type: str
    content_sha256: Sha256
    acquisition_record_sha256: Sha256
    license_id: str
    license_evidence_url: str
    license_evidence_sha256: Sha256
    lock_sha256: Sha256

    @field_validator(
        "source_id",
        "url",
        "revision",
        "media_type",
        "license_id",
        "license_evidence_url",
    )
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        if self.lock_sha256 != _digest_without(
            self.model_dump(mode="json"), "lock_sha256"
        ):
            raise ValueError("public-source lock SHA-256 mismatch")
        return self


class PublicSourceMaterial(_StrictFrozenModel):
    """Locked public UTF-8 excerpts and license evidence visible to the author."""

    source_id: str
    url: str
    revision: str
    media_type: str
    text_transform: Literal["utf8-identity-v1"] = "utf8-identity-v1"
    content_text: str
    content_base64: str
    content_sha256: Sha256
    acquisition_record_base64: str | None = None
    acquisition_record_sha256: Sha256 | None = None
    license_id: str
    license_evidence_url: str
    license_evidence_text: str
    license_evidence_base64: str
    license_evidence_sha256: Sha256
    source_lock_file_sha256: Sha256 | None = None

    @field_validator(
        "source_id",
        "url",
        "revision",
        "media_type",
        "license_id",
        "license_evidence_url",
    )
    @classmethod
    def clean_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_material(self) -> Self:
        if not self.url.startswith(
            "https://"
        ) or not self.license_evidence_url.startswith("https://"):
            raise ValueError("public source and license evidence must use HTTPS")
        if self.revision.casefold() in {
            "main",
            "master",
            "head",
            "latest",
            "current",
            "unknown",
            "unpinned",
        }:
            raise ValueError("public source revision must be immutable")
        if self.revision not in self.url and self.content_sha256 not in self.url:
            raise ValueError("public source URL must bind revision or content hash")
        content = _decode_canonical_base64(self.content_base64, "public source")
        evidence = _decode_canonical_base64(
            self.license_evidence_base64, "license evidence"
        )
        acquisition = (
            _decode_canonical_base64(
                self.acquisition_record_base64, "acquisition record"
            )
            if self.acquisition_record_base64 is not None
            else None
        )
        if not content or not evidence:
            raise ValueError("public source and license evidence must not be empty")
        acquisition_fields = (
            self.acquisition_record_base64,
            self.acquisition_record_sha256,
            self.source_lock_file_sha256,
        )
        if any(item is None for item in acquisition_fields) != all(
            item is None for item in acquisition_fields
        ):
            raise ValueError(
                "acquisition record and public-source lock must be present together"
            )
        if acquisition is not None and not acquisition:
            raise ValueError("acquisition record must not be empty")
        if self.license_id.casefold() in {
            "unknown",
            "none",
            "unlicensed",
            "tbd",
        }:
            raise ValueError("license_id must be explicit")
        if sha256_bytes(content) != self.content_sha256:
            raise ValueError("public source content_sha256 mismatch")
        if sha256_bytes(evidence) != self.license_evidence_sha256:
            raise ValueError("license evidence SHA-256 mismatch")
        if self.content_text != _decode_locked_utf8(content, "public source content"):
            raise ValueError("public source readable text differs from locked bytes")
        if self.license_evidence_text != _decode_locked_utf8(
            evidence, "license evidence"
        ):
            raise ValueError("license evidence readable text differs from locked bytes")
        if acquisition is not None and (
            sha256_bytes(acquisition) != self.acquisition_record_sha256
        ):
            raise ValueError("acquisition record SHA-256 mismatch")
        _scan_untrusted_text(
            {
                "source_id": self.source_id,
                "url": self.url,
                "revision": self.revision,
                "license_evidence_url": self.license_evidence_url,
            },
            "public source metadata",
        )
        _scan_untrusted_bytes(content, "public source content")
        _scan_untrusted_bytes(evidence, "license evidence")
        if acquisition is not None:
            _scan_untrusted_bytes(acquisition, "acquisition record")
        return self


@dataclass(frozen=True)
class VerifiedPublicSourceMaterial:
    """Material whose lock and three referenced byte streams were verified."""

    value: PublicSourceMaterial
    source_lock: PublicSourceLock
    source_lock_file_sha256: str
    source_lock_path: Path
    content_path: Path
    acquisition_record_path: Path
    license_evidence_path: Path
    _verification_token: object = field(repr=False, compare=False)


def _decode_canonical_base64(value: str, label: str) -> bytes:
    try:
        content = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as error:
        raise ValueError(f"{label} must use strict base64") from error
    if base64.b64encode(content).decode("ascii") != value:
        raise ValueError(f"{label} base64 must be canonical")
    return content


def build_public_source_material(
    *,
    source_id: str,
    url: str,
    revision: str,
    media_type: str,
    content: bytes,
    license_id: str,
    license_evidence_url: str,
    license_evidence: bytes,
) -> PublicSourceMaterial:
    return PublicSourceMaterial(
        source_id=source_id,
        url=url,
        revision=revision,
        media_type=media_type,
        content_text=_decode_locked_utf8(content, "public source content"),
        content_base64=base64.b64encode(content).decode("ascii"),
        content_sha256=sha256_bytes(content),
        license_id=license_id,
        license_evidence_url=license_evidence_url,
        license_evidence_text=_decode_locked_utf8(license_evidence, "license evidence"),
        license_evidence_base64=base64.b64encode(license_evidence).decode("ascii"),
        license_evidence_sha256=sha256_bytes(license_evidence),
    )


def build_public_source_lock(
    *,
    source_id: str,
    url: str,
    revision: str,
    media_type: str,
    content_sha256: str,
    acquisition_record_sha256: str,
    license_id: str,
    license_evidence_url: str,
    license_evidence_sha256: str,
) -> PublicSourceLock:
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "url": url,
        "revision": revision,
        "media_type": media_type,
        "content_sha256": content_sha256,
        "acquisition_record_sha256": acquisition_record_sha256,
        "license_id": license_id,
        "license_evidence_url": license_evidence_url,
        "license_evidence_sha256": license_evidence_sha256,
    }
    return PublicSourceLock.model_validate(
        {**payload, "lock_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def load_verified_public_source_material(
    *,
    source_lock_path: str | Path,
    expected_source_lock_file_sha256: str,
    content_path: str | Path,
    acquisition_record_path: str | Path,
    license_evidence_path: str | Path,
) -> VerifiedPublicSourceMaterial:
    """Load a public source only through an independently pinned external lock."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_lock_file_sha256):
        raise AuthoringContractError("expected public-source lock SHA-256 is invalid")
    try:
        lock_bytes = read_stable_regular_file(
            source_lock_path, label="public-source lock"
        )
        if sha256_bytes(lock_bytes) != expected_source_lock_file_sha256:
            raise AuthoringContractError("public-source lock file digest mismatch")
        raw = parse_canonical_json(lock_bytes, label="public-source lock")
        if not isinstance(raw, dict):
            raise AuthoringContractError("public-source lock must contain an object")
        source_lock = PublicSourceLock.model_validate(raw, strict=True)
        content = read_stable_regular_file(content_path, label="public source content")
        acquisition = read_stable_regular_file(
            acquisition_record_path, label="public source acquisition record"
        )
        evidence = read_stable_regular_file(
            license_evidence_path, label="public source license evidence"
        )
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError(
            "public-source lock is not canonical and valid"
        ) from error
    actual = {
        "content": sha256_bytes(content),
        "acquisition record": sha256_bytes(acquisition),
        "license evidence": sha256_bytes(evidence),
    }
    expected = {
        "content": source_lock.content_sha256,
        "acquisition record": source_lock.acquisition_record_sha256,
        "license evidence": source_lock.license_evidence_sha256,
    }
    for label, digest in actual.items():
        if digest != expected[label]:
            raise AuthoringContractError(f"public source {label} digest mismatch")
    try:
        material = PublicSourceMaterial(
            source_id=source_lock.source_id,
            url=source_lock.url,
            revision=source_lock.revision,
            media_type=source_lock.media_type,
            content_text=_decode_locked_utf8(content, "public source content"),
            content_base64=base64.b64encode(content).decode("ascii"),
            content_sha256=source_lock.content_sha256,
            acquisition_record_base64=base64.b64encode(acquisition).decode("ascii"),
            acquisition_record_sha256=source_lock.acquisition_record_sha256,
            license_id=source_lock.license_id,
            license_evidence_url=source_lock.license_evidence_url,
            license_evidence_text=_decode_locked_utf8(evidence, "license evidence"),
            license_evidence_base64=base64.b64encode(evidence).decode("ascii"),
            license_evidence_sha256=source_lock.license_evidence_sha256,
            source_lock_file_sha256=expected_source_lock_file_sha256,
        )
    except ValidationError as error:
        raise AuthoringContractError("verified public source is invalid") from error
    return VerifiedPublicSourceMaterial(
        value=material,
        source_lock=source_lock,
        source_lock_file_sha256=expected_source_lock_file_sha256,
        source_lock_path=Path(source_lock_path).resolve(strict=True),
        content_path=Path(content_path).resolve(strict=True),
        acquisition_record_path=Path(acquisition_record_path).resolve(strict=True),
        license_evidence_path=Path(license_evidence_path).resolve(strict=True),
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


class ReferenceSkillBundle(_StrictFrozenModel):
    bundle_id: str
    version: str
    public_source_ids: tuple[str, ...] = Field(min_length=1)
    guidance: tuple[str, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @field_validator("public_source_ids", "guidance", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        _nonblank(self.bundle_id, "bundle_id")
        _nonblank(self.version, "version")
        if self.public_source_ids != tuple(sorted(set(self.public_source_ids))):
            raise ValueError("public_source_ids must be sorted and unique")
        for item in self.guidance:
            _nonblank(item, "guidance")
        _scan_untrusted_text(
            {
                "bundle_id": self.bundle_id,
                "public_source_ids": self.public_source_ids,
                "guidance": self.guidance,
            },
            "reference bundle",
        )
        if self.bundle_sha256 != _digest_without(
            self.model_dump(mode="json"), "bundle_sha256"
        ):
            raise ValueError("bundle_sha256 mismatch")
        return self


class AuthoringInput(_StrictFrozenModel):
    """Byte-identical, self-contained common input for LLMStatic and future S1."""

    schema_version: Literal[3, 4] = 3
    status: Literal["frozen"] = "frozen"
    consumers: tuple[Literal["llm_static", "s1"], ...]
    taxonomy: CanonicalSpecification
    task_specification: CanonicalSpecification
    tool_registry: CanonicalSpecification
    tool_registry_runtime_binding: Literal["frozen", "deferred_until_bank_compile"]
    tool_registry_runtime_sha256: Sha256 | None
    prompt: PromptIdentity
    compiler: CompilerIdentity
    model: ModelIdentity
    decoding: FixedDecoding
    price_schedule: PriceSchedule
    price_schedule_file_sha256: Sha256 | None
    budgets: AuthoringBudgets
    public_sources: tuple[PublicSourceMaterial, ...] = ()
    reference_skill_bundle: ReferenceSkillBundle | None
    external_verification: Literal["provisional", "externally_verified"]
    input_sha256: Sha256

    @field_validator("consumers", "public_sources", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        content_submission = (
            self.decoding.response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        )
        if (
            self.schema_version == 4,
            self.compiler.compiler_version == "4.0.0",
            content_submission,
        ).count(True) not in {0, 3}:
            raise ValueError(
                "authoring input schema, compiler, and response contract generations "
                "must advance together"
            )
        if self.consumers != ("llm_static", "s1"):
            raise ValueError("consumers must be exactly llm_static and s1")
        if self.decoding.max_output_tokens > self.budgets.max_output_tokens:
            raise ValueError("decoding max_output_tokens exceeds authoring budget")
        if (self.tool_registry_runtime_binding == "frozen") != (
            self.tool_registry_runtime_sha256 is not None
        ):
            raise ValueError("tool registry runtime binding is inconsistent")
        if (
            self.price_schedule.provider != self.model.provider
            or self.price_schedule.model != self.model.model
        ):
            raise ValueError("price schedule is not bound to author model")
        source_ids = tuple(item.source_id for item in self.public_sources)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("public sources must be sorted and unique")
        if self.reference_skill_bundle is not None:
            missing = set(self.reference_skill_bundle.public_source_ids) - set(
                source_ids
            )
            if missing:
                raise ValueError("reference bundle cites an unknown public source")
        sources_are_locked = all(
            item.source_lock_file_sha256 is not None for item in self.public_sources
        )
        if self.external_verification == "externally_verified":
            if self.price_schedule_file_sha256 is None or not sources_are_locked:
                raise ValueError("externally verified input is missing external locks")
        elif self.price_schedule_file_sha256 is not None or (
            self.public_sources and sources_are_locked
        ):
            raise ValueError("provisional input must not claim partial external locks")
        if self.input_sha256 != _digest_without(
            self.model_dump(mode="json"), "input_sha256"
        ):
            raise ValueError("input_sha256 mismatch")
        return self

    @property
    def packet_sha256(self) -> str:
        """Compatibility alias for the v1 field name."""

        return self.input_sha256

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)


AuthoringPacket = AuthoringInput


@dataclass(frozen=True)
class VerifiedAuthoringInput:
    """Opaque formal handle rooted in independent file/protocol/live-runtime locks."""

    value: AuthoringInput
    file_sha256: str
    path: Path
    expected_taxonomy_sha256: str
    expected_task_specification_sha256: str
    expected_tool_registry_sha256: str
    expected_tool_registry_runtime_sha256: str
    public_sources: tuple[VerifiedPublicSourceMaterial, ...]
    price_schedule: VerifiedPriceSchedule
    registry: ToolRegistry = field(repr=False, compare=False)
    _verification_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedAuthoringInvocationInput:
    """Externally verified packet handle that deliberately omits tool runtime.

    Schema v3 freezes ToolSpec identity for the one-shot author call while
    deferring machine-specific registry runtime binding until Bank compile.
    This handle can authorize only the invocation phase; finalization still
    requires :class:`VerifiedAuthoringInput`.
    """

    value: AuthoringInput
    file_sha256: str
    path: Path
    expected_taxonomy_sha256: str
    expected_task_specification_sha256: str
    expected_tool_registry_sha256: str
    public_sources: tuple[VerifiedPublicSourceMaterial, ...]
    price_schedule: VerifiedPriceSchedule
    _verification_token: object = field(repr=False, compare=False)


def _require_formal_runtime_registry(registry: ToolRegistry) -> None:
    """Reject diagnostic registries at every formally eligible entry point."""

    try:
        registry.require_formal_runtime()
    except RegistryError as error:
        raise AuthoringContractError(
            "formal SpecBaseline requires explicit runtime bindings for every tool"
        ) from error


def _specification(
    kind: Literal["taxonomy", "task_specification", "tool_registry"],
    version: str,
    identity_sha256: str,
    value: BaseModel,
) -> CanonicalSpecification:
    content = _canonical_model_bytes(value)
    return CanonicalSpecification(
        specification_kind=kind,
        version=version,
        identity_sha256=identity_sha256,
        canonical_json=content.decode("utf-8"),
        bytes_sha256=sha256_bytes(content),
    )


def build_reference_skill_bundle(
    *,
    bundle_id: str,
    version: str,
    public_source_ids: tuple[str, ...],
    guidance: tuple[str, ...],
) -> ReferenceSkillBundle:
    payload = {
        "bundle_id": bundle_id,
        "version": version,
        "public_source_ids": list(public_source_ids),
        "guidance": list(guidance),
    }
    return ReferenceSkillBundle.model_validate(
        {**payload, "bundle_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def build_authoring_packet(
    *,
    taxonomy: TaxonomyRegistry,
    task_specification: TaskSpecification,
    tool_registry: ToolRegistry,
    prompt: PromptIdentity,
    model: ModelIdentity,
    decoding: FixedDecoding,
    price_schedule: PriceSchedule | VerifiedPriceSchedule,
    budgets: AuthoringBudgets,
    public_sources: tuple[PublicSourceMaterial | VerifiedPublicSourceMaterial, ...],
    reference_skill_bundle: ReferenceSkillBundle | None = None,
    defer_tool_registry_runtime: bool = False,
) -> AuthoringInput:
    v5_contract_flags = (
        decoding.response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
        task_specification.task_spec_version == "ecommerce-task-spec-v1",
        tool_registry.manifest.schema_version == 2,
    )
    if any(v5_contract_flags) and not all(v5_contract_flags):
        raise AuthoringContractError(
            "authoring content v2, TaskSpec v1, and registry v2 must be selected "
            "together"
        )
    is_v5_contract = all(v5_contract_flags)
    price_verified = isinstance(price_schedule, VerifiedPriceSchedule)
    if price_verified:
        if price_schedule._verification_token is not _VERIFIED_HANDLE_TOKEN:
            raise AuthoringContractError("price-schedule verified handle is invalid")
        schedule_value = price_schedule.value
        price_file_sha256 = price_schedule.file_sha256
    else:
        schedule_value = price_schedule
        price_file_sha256 = None
    source_values: list[PublicSourceMaterial] = []
    source_verification: list[bool] = []
    for item in public_sources:
        if isinstance(item, VerifiedPublicSourceMaterial):
            if item._verification_token is not _VERIFIED_HANDLE_TOKEN:
                raise AuthoringContractError("public-source verified handle is invalid")
            source_values.append(item.value)
            source_verification.append(True)
        else:
            source_values.append(item)
            source_verification.append(False)
    all_verified = price_verified and all(source_verification)
    if any(source_verification) and not all(source_verification):
        raise AuthoringContractError(
            "public sources cannot mix verified and provisional"
        )
    if source_verification and price_verified != all(source_verification):
        raise AuthoringContractError(
            "price schedule and public sources must share verification state"
        )
    payload = {
        "schema_version": 4 if is_v5_contract else 3,
        "status": "frozen",
        "consumers": ["llm_static", "s1"],
        "taxonomy": _specification(
            "taxonomy",
            taxonomy.taxonomy_version,
            taxonomy.taxonomy_sha256,
            taxonomy,
        ).model_dump(mode="json"),
        "task_specification": _specification(
            "task_specification",
            task_specification.task_spec_version,
            task_specification.task_spec_sha256,
            task_specification,
        ).model_dump(mode="json"),
        "tool_registry": _specification(
            "tool_registry",
            f"mvp-tool-registry-v{tool_registry.manifest.schema_version}",
            tool_registry.registry_sha256,
            tool_registry.manifest,
        ).model_dump(mode="json"),
        "tool_registry_runtime_binding": (
            "deferred_until_bank_compile" if defer_tool_registry_runtime else "frozen"
        ),
        "tool_registry_runtime_sha256": (
            None
            if defer_tool_registry_runtime
            else tool_registry.registry_runtime_sha256
        ),
        "prompt": prompt.model_dump(mode="json"),
        "compiler": CompilerIdentity(
            compiler_version="4.0.0" if is_v5_contract else "3.0.0"
        ).model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
        "decoding": decoding.model_dump(mode="json"),
        "price_schedule": schedule_value.model_dump(mode="json"),
        "price_schedule_file_sha256": price_file_sha256,
        "budgets": budgets.model_dump(mode="json"),
        "public_sources": [item.model_dump(mode="json") for item in source_values],
        "reference_skill_bundle": (
            reference_skill_bundle.model_dump(mode="json")
            if reference_skill_bundle is not None
            else None
        ),
        "external_verification": (
            "externally_verified" if all_verified else "provisional"
        ),
    }
    return AuthoringInput.model_validate(
        {**payload, "input_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def _revalidate_input(value: AuthoringInput) -> AuthoringInput:
    try:
        validated = AuthoringInput.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise AuthoringContractError(
            "authoring input violates frozen schema"
        ) from error
    _materialize_specifications(validated)
    return validated


def load_authoring_packet(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_registry: ToolRegistry | None = None,
) -> AuthoringInput:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise AuthoringContractError("expected authoring-input SHA-256 is invalid")
    try:
        content = read_stable_regular_file(
            path, label="authoring input", max_bytes=_MAX_INPUT_BYTES
        )
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError("authoring-input file digest mismatch")
        raw = parse_canonical_json(content, label="authoring input")
        if not isinstance(raw, dict):
            raise AuthoringContractError("authoring input must be an object")
        value = AuthoringInput.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError(
            "authoring input is not canonical and valid"
        ) from error
    value = _revalidate_input(value)
    if expected_registry is not None and (
        value.tool_registry.identity_sha256 != expected_registry.registry_sha256
        or (
            value.tool_registry_runtime_sha256 is not None
            and value.tool_registry_runtime_sha256
            != expected_registry.registry_runtime_sha256
        )
    ):
        raise AuthoringContractError("authoring input registry binding mismatch")
    return value


def _materialize_specifications(
    value: AuthoringInput,
) -> tuple[TaxonomyRegistry, TaskSpecification, RegistryManifest]:
    try:
        taxonomy_raw = parse_canonical_json(
            value.taxonomy.content, label="embedded taxonomy"
        )
        task_raw = parse_canonical_json(
            value.task_specification.content, label="embedded Task Specification"
        )
        registry_raw = parse_canonical_json(
            value.tool_registry.content, label="embedded tool registry"
        )
        if not all(
            isinstance(item, dict) for item in (taxonomy_raw, task_raw, registry_raw)
        ):
            raise ValueError("embedded specifications must be objects")
        taxonomy = TaxonomyRegistry.model_validate(taxonomy_raw, strict=True)
        task_specification = TaskSpecification.model_validate(task_raw, strict=True)
        registry_input = dict(registry_raw)
        tools = registry_input.get("tools")
        if not isinstance(tools, list):
            raise ValueError("embedded registry tools must be an array")
        registry_input["tools"] = tuple(
            ToolSpec.model_validate(item, strict=True) for item in tools
        )
        registry = RegistryManifest.model_validate(registry_input, strict=True)
    except (ArtifactFormatError, ValidationError, ValueError) as error:
        raise AuthoringContractError(
            "embedded specification validation failed"
        ) from error
    identities = (
        (
            value.taxonomy,
            "taxonomy",
            taxonomy.taxonomy_version,
            taxonomy.taxonomy_sha256,
        ),
        (
            value.task_specification,
            "task_specification",
            task_specification.task_spec_version,
            task_specification.task_spec_sha256,
        ),
        (
            value.tool_registry,
            "tool_registry",
            f"mvp-tool-registry-v{registry.schema_version}",
            registry.registry_sha256,
        ),
    )
    for artifact, expected_kind, expected_version, expected_hash in identities:
        if (
            artifact.specification_kind != expected_kind
            or artifact.version != expected_version
            or artifact.identity_sha256 != expected_hash
        ):
            raise AuthoringContractError("embedded specification identity mismatch")
    if task_specification.taxonomy_sha256 != taxonomy.taxonomy_sha256:
        raise AuthoringContractError("Task Specification taxonomy binding mismatch")
    if set(task_specification.capabilities_by_id) != set(taxonomy.capabilities_by_id):
        raise AuthoringContractError("embedded capability set mismatch")
    expected_generation = (
        (3, "ecommerce-task-spec-v0", 1, "3.0.0")
        if value.schema_version == 3
        else (4, "ecommerce-task-spec-v1", 2, "4.0.0")
    )
    observed_generation = (
        value.schema_version,
        task_specification.task_spec_version,
        registry.schema_version,
        value.compiler.compiler_version,
    )
    if observed_generation != expected_generation:
        raise AuthoringContractError(
            "embedded TaskSpec, registry, packet, and compiler generations differ"
        )
    registered = {item.name: item for item in registry.tools}
    for task in task_specification.capabilities:
        if not set(task.allowed_tools) <= set(registered):
            raise AuthoringContractError(
                f"TaskSpec allows an unregistered tool: {task.capability_id}"
            )
        rules = (
            *task.input_preconditions,
            *task.success_criteria,
            *task.failure_conditions,
            *task.acceptable_answer_rules,
            *task.safety_constraints,
            *task.fallback.trigger_rules,
            *task.fallback.response_rules,
            *task.indeterminate_conditions,
        )
        for rule in rules:
            if rule.provenance.kind != "tool_contract":
                continue
            match = re.fullmatch(
                r"tool:([a-z_]+)@([0-9]+(?:\.[0-9]+){2})",
                rule.provenance.source_ref,
            )
            if match is None:
                raise AuthoringContractError(
                    "TaskSpec tool provenance is not parseable"
                )
            tool_name, tool_version = match.groups()
            tool = registered.get(tool_name)
            if (
                tool is None
                or tool.tool_version != tool_version
                or tool_name not in task.allowed_tools
            ):
                raise AuthoringContractError(
                    f"TaskSpec tool provenance is not executable: "
                    f"{task.capability_id}/{rule.rule_id}"
                )
    return taxonomy, task_specification, registry


def _refresh_verified_public_source(
    value: VerifiedPublicSourceMaterial,
) -> VerifiedPublicSourceMaterial:
    if value._verification_token is not _VERIFIED_HANDLE_TOKEN:
        raise AuthoringContractError("public-source verified handle is invalid")
    return load_verified_public_source_material(
        source_lock_path=value.source_lock_path,
        expected_source_lock_file_sha256=value.source_lock_file_sha256,
        content_path=value.content_path,
        acquisition_record_path=value.acquisition_record_path,
        license_evidence_path=value.license_evidence_path,
    )


def _refresh_verified_price_schedule(
    value: VerifiedPriceSchedule,
) -> VerifiedPriceSchedule:
    if value._verification_token is not _VERIFIED_HANDLE_TOKEN:
        raise AuthoringContractError("price-schedule verified handle is invalid")
    return load_verified_price_schedule(
        value.path, expected_file_sha256=value.file_sha256
    )


def load_verified_authoring_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    registry: ToolRegistry,
    expected_taxonomy_sha256: str,
    expected_task_specification_sha256: str,
    expected_tool_registry_sha256: str,
    expected_tool_registry_runtime_sha256: str,
    public_sources: tuple[VerifiedPublicSourceMaterial, ...],
    price_schedule: VerifiedPriceSchedule,
) -> VerifiedAuthoringInput:
    """Create the only handle accepted for a formally eligible SpecBaseline."""

    _require_formal_runtime_registry(registry)
    protocol_locks = (
        expected_taxonomy_sha256,
        expected_task_specification_sha256,
        expected_tool_registry_sha256,
        expected_tool_registry_runtime_sha256,
    )
    if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in protocol_locks):
        raise AuthoringContractError("expected protocol lock SHA-256 is invalid")
    if (
        registry.registry_sha256 != expected_tool_registry_sha256
        or registry.registry_runtime_sha256 != expected_tool_registry_runtime_sha256
    ):
        raise AuthoringContractError("live registry does not match independent locks")
    invocation_input = _load_verified_authoring_invocation_input(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_task_specification_sha256=expected_task_specification_sha256,
        expected_tool_registry_sha256=expected_tool_registry_sha256,
        public_sources=public_sources,
        price_schedule=price_schedule,
        require_deferred_runtime=False,
    )
    value = invocation_input.value
    if (
        value.tool_registry_runtime_sha256 is not None
        and value.tool_registry_runtime_sha256 != expected_tool_registry_runtime_sha256
    ):
        raise AuthoringContractError("authoring input does not match protocol locks")
    return VerifiedAuthoringInput(
        value=value,
        file_sha256=expected_file_sha256,
        path=Path(path).resolve(strict=True),
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_task_specification_sha256=expected_task_specification_sha256,
        expected_tool_registry_sha256=expected_tool_registry_sha256,
        expected_tool_registry_runtime_sha256=expected_tool_registry_runtime_sha256,
        public_sources=invocation_input.public_sources,
        price_schedule=invocation_input.price_schedule,
        registry=registry,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def _load_verified_authoring_invocation_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_taxonomy_sha256: str,
    expected_task_specification_sha256: str,
    expected_tool_registry_sha256: str,
    public_sources: tuple[VerifiedPublicSourceMaterial, ...],
    price_schedule: VerifiedPriceSchedule,
    require_deferred_runtime: bool,
) -> VerifiedAuthoringInvocationInput:
    """Verify every public authoring input without inventing a tool runtime."""

    protocol_locks = (
        expected_taxonomy_sha256,
        expected_task_specification_sha256,
        expected_tool_registry_sha256,
    )
    if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in protocol_locks):
        raise AuthoringContractError("expected protocol lock SHA-256 is invalid")
    refreshed_price = _refresh_verified_price_schedule(price_schedule)
    refreshed_sources = tuple(
        _refresh_verified_public_source(item) for item in public_sources
    )
    value = load_authoring_packet(path, expected_file_sha256=expected_file_sha256)
    taxonomy, tasks, embedded_registry = _materialize_specifications(value)
    if (
        taxonomy.taxonomy_sha256 != expected_taxonomy_sha256
        or tasks.task_spec_sha256 != expected_task_specification_sha256
        or embedded_registry.registry_sha256 != expected_tool_registry_sha256
    ):
        raise AuthoringContractError("authoring input does not match protocol locks")
    if require_deferred_runtime and (
        value.tool_registry_runtime_binding != "deferred_until_bank_compile"
        or value.tool_registry_runtime_sha256 is not None
    ):
        raise AuthoringContractError(
            "invocation-only packet must defer tool runtime until Bank compile"
        )
    expected_sources = tuple(
        item.value
        for item in sorted(refreshed_sources, key=lambda item: item.value.source_id)
    )
    if (
        value.external_verification != "externally_verified"
        or value.price_schedule != refreshed_price.value
        or value.price_schedule_file_sha256 != refreshed_price.file_sha256
        or value.public_sources != expected_sources
    ):
        raise AuthoringContractError("authoring input external-lock binding mismatch")
    return VerifiedAuthoringInvocationInput(
        value=value,
        file_sha256=expected_file_sha256,
        path=Path(path).resolve(strict=True),
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_task_specification_sha256=expected_task_specification_sha256,
        expected_tool_registry_sha256=expected_tool_registry_sha256,
        public_sources=refreshed_sources,
        price_schedule=refreshed_price,
        _verification_token=_VERIFIED_HANDLE_TOKEN,
    )


def load_verified_authoring_invocation_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_taxonomy_sha256: str,
    expected_task_specification_sha256: str,
    expected_tool_registry_sha256: str,
    public_sources: tuple[VerifiedPublicSourceMaterial, ...],
    price_schedule: VerifiedPriceSchedule,
) -> VerifiedAuthoringInvocationInput:
    """Verify one deferred-runtime packet for the isolated invocation phase."""

    return _load_verified_authoring_invocation_input(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_task_specification_sha256=expected_task_specification_sha256,
        expected_tool_registry_sha256=expected_tool_registry_sha256,
        public_sources=public_sources,
        price_schedule=price_schedule,
        require_deferred_runtime=True,
    )


def _refresh_verified_authoring_invocation_input(
    value: VerifiedAuthoringInvocationInput,
) -> VerifiedAuthoringInvocationInput:
    if value._verification_token is not _VERIFIED_HANDLE_TOKEN:
        raise AuthoringContractError(
            "authoring invocation-input verified handle is invalid"
        )
    return load_verified_authoring_invocation_input(
        value.path,
        expected_file_sha256=value.file_sha256,
        expected_taxonomy_sha256=value.expected_taxonomy_sha256,
        expected_task_specification_sha256=value.expected_task_specification_sha256,
        expected_tool_registry_sha256=value.expected_tool_registry_sha256,
        public_sources=value.public_sources,
        price_schedule=value.price_schedule,
    )


def _refresh_verified_authoring_input(
    value: VerifiedAuthoringInput,
) -> VerifiedAuthoringInput:
    if value._verification_token is not _VERIFIED_HANDLE_TOKEN:
        raise AuthoringContractError("verified authoring-input handle is invalid")
    _require_formal_runtime_registry(value.registry)
    return load_verified_authoring_input(
        value.path,
        expected_file_sha256=value.file_sha256,
        registry=value.registry,
        expected_taxonomy_sha256=value.expected_taxonomy_sha256,
        expected_task_specification_sha256=value.expected_task_specification_sha256,
        expected_tool_registry_sha256=value.expected_tool_registry_sha256,
        expected_tool_registry_runtime_sha256=(
            value.expected_tool_registry_runtime_sha256
        ),
        public_sources=value.public_sources,
        price_schedule=value.price_schedule,
    )


class RuleCoverage(_StrictFrozenModel):
    precondition_rule_ids: tuple[str, ...]
    success_rule_ids: tuple[str, ...]
    failure_rule_ids: tuple[str, ...]
    acceptable_answer_rule_ids: tuple[str, ...]
    safety_rule_ids: tuple[str, ...]
    fallback_trigger_rule_ids: tuple[str, ...]
    fallback_response_rule_ids: tuple[str, ...]
    indeterminate_rule_ids: tuple[str, ...]

    @field_validator("*", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_sets(self) -> Self:
        for name, value in self:
            if value != tuple(sorted(set(value))) or not value:
                raise ValueError(f"{name} must be non-empty, sorted, and unique")
        return self


class OutputCoverage(_StrictFrozenModel):
    response_kind: Literal["product_cards", "grounded_text"]
    required_sections: tuple[str, ...]
    card_requirement: Literal["required", "forbidden", "optional"]
    card_fields: tuple[str, ...]
    evidence_requirement: Literal[
        "product_retrieval_evidence",
        "cited_knowledge_evidence",
        "typed_tool_evidence",
    ]

    @field_validator("required_sections", "card_fields", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)


class DraftToolStep(_StrictFrozenModel):
    step_id: str
    instruction: str
    tool_name: ToolName
    success_rule_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("success_rule_ids", mode="before")
    @classmethod
    def coerce_rules(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        _nonblank(self.step_id, "step_id")
        _nonblank(self.instruction, "instruction")
        if self.success_rule_ids != tuple(sorted(set(self.success_rule_ids))):
            raise ValueError("success_rule_ids must be sorted and unique")
        _scan_untrusted_text(self.instruction, "draft instruction")
        return self


class CapabilityDraft(_StrictFrozenModel):
    capability_id: str
    objective: str
    steps: tuple[DraftToolStep, ...] = Field(min_length=1)
    fallback_instruction: str
    fallback_may_request_clarification: bool
    fallback_must_state_uncertainty: Literal[True]
    citation_source_ids: tuple[str, ...]
    rule_coverage: RuleCoverage
    output_coverage: OutputCoverage
    draft_sha256: Sha256

    @field_validator("steps", "citation_source_ids", mode="before")
    @classmethod
    def coerce_steps(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_draft(self) -> Self:
        _nonblank(self.capability_id, "capability_id")
        _nonblank(self.objective, "objective")
        _nonblank(self.fallback_instruction, "fallback_instruction")
        _scan_untrusted_text(
            {
                "objective": self.objective,
                "steps": [item.model_dump(mode="json") for item in self.steps],
                "fallback_instruction": self.fallback_instruction,
            },
            "capability draft",
        )
        step_ids = tuple(item.step_id for item in self.steps)
        if step_ids != tuple(sorted(set(step_ids))):
            raise ValueError("draft steps must be sorted and unique")
        if self.citation_source_ids != tuple(sorted(set(self.citation_source_ids))):
            raise ValueError("citation_source_ids must be sorted and unique")
        if self.draft_sha256 != _digest_without(
            self.model_dump(mode="json"), "draft_sha256"
        ):
            raise ValueError("draft_sha256 mismatch")
        return self


class CapabilityDraftPayload(_StrictFrozenModel):
    """Hash-free author content; the trusted runner adds all identities."""

    capability_id: str
    objective: str
    steps: tuple[DraftToolStep, ...] = Field(min_length=1)
    fallback_instruction: str
    fallback_may_request_clarification: bool
    fallback_must_state_uncertainty: Literal[True]
    citation_source_ids: tuple[str, ...]
    rule_coverage: RuleCoverage
    output_coverage: OutputCoverage

    @field_validator("steps", "citation_source_ids", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        _nonblank(self.capability_id, "capability_id")
        _nonblank(self.objective, "objective")
        _nonblank(self.fallback_instruction, "fallback_instruction")
        _scan_untrusted_text(
            {
                "objective": self.objective,
                "steps": [item.model_dump(mode="json") for item in self.steps],
                "fallback_instruction": self.fallback_instruction,
            },
            "capability draft payload",
        )
        step_ids = tuple(item.step_id for item in self.steps)
        if step_ids != tuple(sorted(set(step_ids))):
            raise ValueError("payload steps must be sorted and unique")
        if self.citation_source_ids != tuple(sorted(set(self.citation_source_ids))):
            raise ValueError("citation_source_ids must be sorted and unique")
        return self


class AuthoringDraftPayload(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    drafts: tuple[CapabilityDraftPayload, ...] = Field(min_length=1)

    @field_validator("drafts", mode="before")
    @classmethod
    def coerce_drafts(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        ids = tuple(item.capability_id for item in self.drafts)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("payload drafts must be sorted and unique")
        return self


class DraftToolStepContent(_StrictFrozenModel):
    """Model-authored step semantics without compiler-owned representation."""

    instruction: str
    tool_name: ToolName
    success_rule_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("success_rule_ids", mode="before")
    @classmethod
    def coerce_rules(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        _nonblank(self.instruction, "instruction")
        if len(self.success_rule_ids) != len(set(self.success_rule_ids)):
            raise ValueError("success_rule_ids must be unique")
        if any(not item or item != item.strip() for item in self.success_rule_ids):
            raise ValueError(
                "success_rule_ids must be non-blank without edge whitespace"
            )
        _scan_untrusted_text(self.instruction, "authoring step instruction")
        return self


class CapabilityAuthoringContent(_StrictFrozenModel):
    """Only fields that require author judgment; TaskSpec fields are omitted."""

    capability_id: str
    objective: str
    steps: tuple[DraftToolStepContent, ...] = Field(min_length=1, max_length=16)
    fallback_instruction: str
    citation_source_ids: tuple[str, ...]

    @field_validator("steps", "citation_source_ids", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        _nonblank(self.capability_id, "capability_id")
        _nonblank(self.objective, "objective")
        _nonblank(self.fallback_instruction, "fallback_instruction")
        if len(self.citation_source_ids) != len(set(self.citation_source_ids)):
            raise ValueError("citation_source_ids must be unique")
        if any(not item or item != item.strip() for item in self.citation_source_ids):
            raise ValueError(
                "citation_source_ids must be non-blank without edge whitespace"
            )
        _scan_untrusted_text(
            {
                "objective": self.objective,
                "steps": [item.model_dump(mode="json") for item in self.steps],
                "fallback_instruction": self.fallback_instruction,
            },
            "capability authoring content",
        )
        return self


class AuthoringContentPayload(_StrictFrozenModel):
    """Prospective v2 wire payload with compiler-owned TaskSpec projection."""

    # This is a wire-level discriminator, not a convenience default.  A
    # provider response that omits it must fail instead of being silently
    # upgraded by Pydantic.
    schema_version: Literal[2]
    drafts: tuple[CapabilityAuthoringContent, ...] = Field(min_length=1)

    @field_validator("drafts", mode="before")
    @classmethod
    def coerce_drafts(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        ids = tuple(item.capability_id for item in self.drafts)
        if len(ids) != len(set(ids)):
            raise ValueError("authoring content capability ids must be unique")
        return self


def _inline_local_schema_refs(
    value: object, definitions: Mapping[str, object]
) -> object:
    """Inline Pydantic-local refs for conservative function-calling providers."""

    if isinstance(value, list):
        return [_inline_local_schema_refs(item, definitions) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if reference is not None:
        if (
            not isinstance(reference, str)
            or not reference.startswith("#/$defs/")
            or len(value) != 1
        ):
            raise AuthoringContractError(
                "authoring submission schema contains an unsupported reference"
            )
        name = reference.removeprefix("#/$defs/")
        target = definitions.get(name)
        if not isinstance(target, dict):
            raise AuthoringContractError(
                "authoring submission schema reference is unresolved"
            )
        return _inline_local_schema_refs(target, definitions)
    return {
        key: _inline_local_schema_refs(item, definitions)
        for key, item in value.items()
        if key != "$defs"
    }


def authoring_submission_tool_definition() -> dict[str, object]:
    """Return the exact non-executable function envelope frozen for v4 calls."""

    raw_schema = AuthoringDraftPayload.model_json_schema()
    definitions = raw_schema.get("$defs")
    if not isinstance(definitions, dict):
        raise AuthoringContractError("authoring draft schema lacks local definitions")
    schema = _inline_local_schema_refs(raw_schema, definitions)
    if not isinstance(schema, dict):
        raise AuthoringContractError("authoring submission schema must be an object")
    # Pydantic treats a literal field with a model-side default as optional in
    # JSON Schema.  The provider submission must state its wire version
    # explicitly so the raw response remains self-describing.
    schema["required"] = ["schema_version", "drafts"]
    schema_version = (schema.get("properties") or {}).get("schema_version")
    if isinstance(schema_version, dict):
        schema_version.pop("default", None)
    # This definition is a frozen v4 protocol artifact.  ToolName may grow in
    # later registry generations, but that must not rewrite historical v1
    # bytes or broaden the already-consumed submission contract.
    try:
        tool_name_schema = schema["properties"]["drafts"]["items"]["properties"][
            "steps"
        ]["items"]["properties"]["tool_name"]
    except (KeyError, TypeError) as error:
        raise AuthoringContractError(
            "authoring submission schema tool path drifted"
        ) from error
    if not isinstance(tool_name_schema, dict):
        raise AuthoringContractError("authoring submission tool-name schema is invalid")
    tool_name_schema["enum"] = list(_LEGACY_AUTHORING_TOOL_NAMES)
    return {
        "type": "function",
        "function": {
            "name": AUTHORING_SUBMISSION_TOOL_NAME,
            "description": (
                "Submit exactly one complete static-authoring payload. "
                "This is an output envelope only and is never executed."
            ),
            "parameters": schema,
        },
    }


def authoring_content_json_schema(
    value: AuthoringInput,
) -> dict[str, object]:
    """Build the packet-bound v2 author-content JSON Schema.

    Unlike v1, the model cannot submit TaskSpec-owned coverage, output, or
    fallback flag fields.  The schema is specialized to the embedded
    capability, tool, success-rule, and public-source identities.
    """

    value = _revalidate_input(value)
    _, task_specification, _ = _materialize_specifications(value)
    source_ids = [item.source_id for item in value.public_sources]
    capability_variants: list[dict[str, object]] = []
    for task in task_specification.capabilities:
        success_ids = [item.rule_id for item in task.success_criteria]
        citation_items: dict[str, object] = {"type": "string"}
        if source_ids:
            citation_items["enum"] = source_ids
        capability_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "capability_id",
                    "objective",
                    "steps",
                    "fallback_instruction",
                    "citation_source_ids",
                ],
                "properties": {
                    "capability_id": {"const": task.capability_id},
                    "objective": {"type": "string", "minLength": 1},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "instruction",
                                "tool_name",
                                "success_rule_ids",
                            ],
                            "properties": {
                                "instruction": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "tool_name": {
                                    "type": "string",
                                    "enum": list(task.allowed_tools),
                                },
                                "success_rule_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "enum": success_ids,
                                    },
                                },
                            },
                        },
                    },
                    "fallback_instruction": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "citation_source_ids": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": len(source_ids),
                        "uniqueItems": True,
                        "items": citation_items,
                    },
                },
            }
        )
    capability_count = len(capability_variants)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": AUTHORING_CONTENT_SCHEMA_NAME,
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "drafts"],
        "properties": {
            "schema_version": {"const": 2, "type": "integer"},
            "drafts": {
                "type": "array",
                "minItems": capability_count,
                "maxItems": capability_count,
                "uniqueItems": True,
                "items": {"oneOf": capability_variants},
            },
        },
    }


class AuthoringOutputContract(_StrictFrozenModel):
    """Exact provider framing embedded in request-v2 and its request digest."""

    schema_version: Literal[1] = 1
    mode: Literal["forced_single_function_submission"] = (
        "forced_single_function_submission"
    )
    tool_name: Literal["submit_authoring_payload"] = AUTHORING_SUBMISSION_TOOL_NAME
    tool_definition_canonical_json: str
    tool_definition_sha256: Sha256
    tool_choice: Literal["forced_named_function"] = "forced_named_function"
    parallel_tool_calls: Literal[False] = False
    assistant_text_allowed: Literal[False] = False
    tool_execution: Literal[False] = False
    contract_sha256: Sha256

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        try:
            content = self.tool_definition_canonical_json.encode("utf-8")
            raw = parse_canonical_json(
                content,
                label="authoring output-contract tool definition",
            )
        except (ArtifactFormatError, UnicodeEncodeError) as error:
            raise ValueError(
                "authoring output-contract tool definition is invalid"
            ) from error
        if (
            raw != authoring_submission_tool_definition()
            or self.tool_definition_sha256 != sha256_bytes(content)
            or self.contract_sha256
            != _digest_without(
                self.model_dump(mode="json"),
                "contract_sha256",
            )
        ):
            raise ValueError("authoring output contract drifted")
        return self


def build_authoring_output_contract() -> AuthoringOutputContract:
    tool_bytes = canonical_json_bytes(authoring_submission_tool_definition())
    payload = {
        "schema_version": 1,
        "mode": "forced_single_function_submission",
        "tool_name": AUTHORING_SUBMISSION_TOOL_NAME,
        "tool_definition_canonical_json": tool_bytes.decode("utf-8"),
        "tool_definition_sha256": sha256_bytes(tool_bytes),
        "tool_choice": "forced_named_function",
        "parallel_tool_calls": False,
        "assistant_text_allowed": False,
        "tool_execution": False,
    }
    return AuthoringOutputContract.model_validate(
        {
            **payload,
            "contract_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


class AuthoringContentOutputContract(_StrictFrozenModel):
    """Packet-bound output contract for prospective author-content v2 calls."""

    schema_version: Literal[2] = 2
    mode: Literal["provider_json_object_content_submission"] = (
        "provider_json_object_content_submission"
    )
    authoring_input_sha256: Sha256
    schema_name: Literal["skillchain_authoring_content_v2"] = (
        AUTHORING_CONTENT_SCHEMA_NAME
    )
    json_schema_canonical_json: str
    json_schema_sha256: Sha256
    provider_response_format: Literal["json_object"] = "json_object"
    provider_guarantee: Literal["json_syntax_only"] = "json_syntax_only"
    json_schema_enforcement: Literal["prompt_and_audit_only"] = "prompt_and_audit_only"
    runner_packet_contract_validation: Literal[True] = True
    runner_validation_engine: Literal["pydantic_and_trusted_compiler"] = (
        "pydantic_and_trusted_compiler"
    )
    tools_supplied: Literal[False] = False
    compiler_owned_fields: tuple[
        Literal[
            "citation_order",
            "fallback_flags",
            "output_coverage",
            "rule_coverage",
            "step_ids",
            "success_rule_order",
        ],
        ...,
    ]
    contract_sha256: Sha256

    @field_validator("compiler_owned_fields", mode="before")
    @classmethod
    def coerce_compiler_fields(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.compiler_owned_fields != (
            "citation_order",
            "fallback_flags",
            "output_coverage",
            "rule_coverage",
            "step_ids",
            "success_rule_order",
        ):
            raise ValueError("compiler-owned authoring fields drifted")
        try:
            content = self.json_schema_canonical_json.encode("utf-8")
            raw = parse_canonical_json(
                content,
                label="authoring content output-contract JSON Schema",
            )
        except (ArtifactFormatError, UnicodeEncodeError) as error:
            raise ValueError(
                "authoring content output-contract JSON Schema is invalid"
            ) from error
        if (
            not isinstance(raw, dict)
            or raw.get("title") != self.schema_name
            or self.json_schema_sha256 != sha256_bytes(content)
            or self.contract_sha256
            != _digest_without(
                self.model_dump(mode="json"),
                "contract_sha256",
            )
        ):
            raise ValueError("authoring content output contract drifted")
        return self


def build_authoring_content_output_contract(
    value: AuthoringInput,
) -> AuthoringContentOutputContract:
    schema_bytes = canonical_json_bytes(authoring_content_json_schema(value))
    payload = {
        "schema_version": 2,
        "mode": "provider_json_object_content_submission",
        "authoring_input_sha256": value.input_sha256,
        "schema_name": AUTHORING_CONTENT_SCHEMA_NAME,
        "json_schema_canonical_json": schema_bytes.decode("utf-8"),
        "json_schema_sha256": sha256_bytes(schema_bytes),
        "provider_response_format": "json_object",
        "provider_guarantee": "json_syntax_only",
        "json_schema_enforcement": "prompt_and_audit_only",
        "runner_packet_contract_validation": True,
        "runner_validation_engine": "pydantic_and_trusted_compiler",
        "tools_supplied": False,
        "compiler_owned_fields": [
            "citation_order",
            "fallback_flags",
            "output_coverage",
            "rule_coverage",
            "step_ids",
            "success_rule_order",
        ],
    }
    return AuthoringContentOutputContract.model_validate(
        {
            **payload,
            "contract_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


class AuthoringDraftBundle(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    authoring_input_sha256: Sha256
    drafts: tuple[CapabilityDraft, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @field_validator("drafts", mode="before")
    @classmethod
    def coerce_drafts(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        ids = tuple(item.capability_id for item in self.drafts)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("drafts must be sorted and unique")
        if self.bundle_sha256 != _digest_without(
            self.model_dump(mode="json"), "bundle_sha256"
        ):
            raise ValueError("draft bundle SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)


def _rule_ids(rules) -> tuple[str, ...]:
    return tuple(sorted(item.rule_id for item in rules))


def _coverage_for(task: CapabilityTaskSpec) -> RuleCoverage:
    return RuleCoverage(
        precondition_rule_ids=_rule_ids(task.input_preconditions),
        success_rule_ids=_rule_ids(task.success_criteria),
        failure_rule_ids=_rule_ids(task.failure_conditions),
        acceptable_answer_rule_ids=_rule_ids(task.acceptable_answer_rules),
        safety_rule_ids=_rule_ids(task.safety_constraints),
        fallback_trigger_rule_ids=_rule_ids(task.fallback.trigger_rules),
        fallback_response_rule_ids=_rule_ids(task.fallback.response_rules),
        indeterminate_rule_ids=_rule_ids(task.indeterminate_conditions),
    )


def _output_for(task: CapabilityTaskSpec) -> OutputCoverage:
    return OutputCoverage.model_validate(task.output_contract.model_dump(mode="python"))


def build_capability_draft(
    *,
    capability_id: str,
    objective: str,
    steps: tuple[DraftToolStep, ...],
    fallback_instruction: str,
    fallback_may_request_clarification: bool,
    fallback_must_state_uncertainty: Literal[True],
    citation_source_ids: tuple[str, ...],
    rule_coverage: RuleCoverage,
    output_coverage: OutputCoverage,
) -> CapabilityDraft:
    payload = {
        "capability_id": capability_id,
        "objective": objective,
        "steps": [item.model_dump(mode="json") for item in steps],
        "fallback_instruction": fallback_instruction,
        "fallback_may_request_clarification": fallback_may_request_clarification,
        "fallback_must_state_uncertainty": fallback_must_state_uncertainty,
        "citation_source_ids": list(citation_source_ids),
        "rule_coverage": rule_coverage.model_dump(mode="json"),
        "output_coverage": output_coverage.model_dump(mode="json"),
    }
    return CapabilityDraft.model_validate(
        {**payload, "draft_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def build_authoring_draft_bundle(
    *, authoring_input_sha256: str, drafts: tuple[CapabilityDraft, ...]
) -> AuthoringDraftBundle:
    payload = {
        "schema_version": 2,
        "authoring_input_sha256": authoring_input_sha256,
        "drafts": [item.model_dump(mode="json") for item in drafts],
    }
    return AuthoringDraftBundle.model_validate(
        {**payload, "bundle_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def normalize_authoring_draft_payload(
    value: AuthoringDraftPayload,
    *,
    authoring_input_sha256: str,
) -> AuthoringDraftBundle:
    """Add content identities outside the untrusted model boundary."""

    drafts = tuple(
        build_capability_draft(
            capability_id=item.capability_id,
            objective=item.objective,
            steps=item.steps,
            fallback_instruction=item.fallback_instruction,
            fallback_may_request_clarification=(
                item.fallback_may_request_clarification
            ),
            fallback_must_state_uncertainty=item.fallback_must_state_uncertainty,
            citation_source_ids=item.citation_source_ids,
            rule_coverage=item.rule_coverage,
            output_coverage=item.output_coverage,
        )
        for item in value.drafts
    )
    return build_authoring_draft_bundle(
        authoring_input_sha256=authoring_input_sha256,
        drafts=drafts,
    )


def normalize_authoring_content_payload(
    value: AuthoringContentPayload,
    *,
    authoring_input: AuthoringInput,
) -> AuthoringDraftBundle:
    """Inject only deterministic TaskSpec-owned fields into author content."""

    authoring_input = _revalidate_input(authoring_input)
    _, task_specification, _ = _materialize_specifications(authoring_input)
    content_by_capability = {item.capability_id: item for item in value.drafts}
    expected_capabilities = set(task_specification.capabilities_by_id)
    if set(content_by_capability) != expected_capabilities:
        raise AuthoringContractError(
            "authoring content must cover every frozen capability exactly once"
        )
    allowed_source_ids = {item.source_id for item in authoring_input.public_sources}
    drafts: list[CapabilityDraft] = []
    for task in task_specification.capabilities:
        item = content_by_capability[task.capability_id]
        if not set(item.citation_source_ids) <= allowed_source_ids:
            raise AuthoringContractError(
                f"authoring content cites an unknown source: {task.capability_id}"
            )
        allowed_tools = set(task.allowed_tools)
        expected_success_ids = {rule.rule_id for rule in task.success_criteria}
        observed_success_ids: set[str] = set()
        steps: list[DraftToolStep] = []
        for index, content_step in enumerate(item.steps, start=1):
            if content_step.tool_name not in allowed_tools:
                raise AuthoringContractError(
                    f"authoring content uses a forbidden tool: {task.capability_id}"
                )
            step_success_ids = set(content_step.success_rule_ids)
            if not step_success_ids <= expected_success_ids:
                raise AuthoringContractError(
                    f"authoring content maps an unknown success rule: "
                    f"{task.capability_id}"
                )
            observed_success_ids.update(step_success_ids)
            steps.append(
                DraftToolStep(
                    step_id=f"{index:02d}.{content_step.tool_name}",
                    instruction=content_step.instruction,
                    tool_name=content_step.tool_name,
                    success_rule_ids=tuple(sorted(step_success_ids)),
                )
            )
        if observed_success_ids != expected_success_ids:
            raise AuthoringContractError(
                f"authoring content steps do not cover every success rule: "
                f"{task.capability_id}"
            )
        drafts.append(
            build_capability_draft(
                capability_id=task.capability_id,
                objective=item.objective,
                steps=tuple(steps),
                fallback_instruction=item.fallback_instruction,
                fallback_may_request_clarification=(
                    task.fallback.may_request_clarification
                ),
                fallback_must_state_uncertainty=(task.fallback.must_state_uncertainty),
                citation_source_ids=tuple(sorted(item.citation_source_ids)),
                rule_coverage=_coverage_for(task),
                output_coverage=_output_for(task),
            )
        )
    return build_authoring_draft_bundle(
        authoring_input_sha256=authoring_input.input_sha256,
        drafts=tuple(drafts),
    )


def _spec_draft(
    task: CapabilityTaskSpec,
    tools: dict[str, ToolSpec],
) -> CapabilityDraft:
    success_ids = _rule_ids(task.success_criteria)
    steps = tuple(
        DraftToolStep(
            step_id=f"{index:02d}.{tool_name}",
            instruction=(
                f"Invoke {tool_name} version {tools[tool_name].tool_version}; "
                f"treat its output as {tools[tool_name].output_trust}."
            ),
            tool_name=tool_name,
            success_rule_ids=success_ids,
        )
        for index, tool_name in enumerate(task.allowed_tools, start=1)
    )
    return build_capability_draft(
        capability_id=task.capability_id,
        objective=" ".join(item.statement for item in task.acceptable_answer_rules),
        steps=steps,
        fallback_instruction=" ".join(
            item.statement for item in task.fallback.response_rules
        ),
        fallback_may_request_clarification=task.fallback.may_request_clarification,
        fallback_must_state_uncertainty=task.fallback.must_state_uncertainty,
        # SpecBaseline is derived only from frozen TaskSpec/ToolSpec content.
        # Per-rule provenance is rendered from TaskSpec below; AuthoringPacket
        # public-source ids must never affect this diagnostic Bank.
        citation_source_ids=(),
        rule_coverage=_coverage_for(task),
        output_coverage=_output_for(task),
    )


def build_spec_draft_bundle(value: AuthoringInput) -> AuthoringDraftBundle:
    """Build a valid structured fixture solely from self-contained input bytes."""

    value = _revalidate_input(value)
    _, task_specification, registry = _materialize_specifications(value)
    tools = {item.name: item for item in registry.tools}
    drafts = tuple(_spec_draft(task, tools) for task in task_specification.capabilities)
    return build_authoring_draft_bundle(
        authoring_input_sha256=value.input_sha256, drafts=drafts
    )


class StrictSkillArtifact(_StrictFrozenModel):
    """Strict superset of the repository's real six-field ``Skill`` model."""

    slug: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    version: int = Field(ge=1)
    description: str
    body: str
    static_refs: tuple[str, ...]
    operators: tuple[ToolName, ...]
    capability_id: str
    parent_skill_sha256: Sha256 | None
    skill_sha256: Sha256

    @field_validator("static_refs", "operators", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_skill(self) -> Self:
        _nonblank(self.description, "description")
        if (
            not self.body
            or not self.body.endswith("\n")
            or self.body != self.body.lstrip()
        ):
            raise ValueError("body must be non-empty canonical Markdown")
        _nonblank(self.capability_id, "capability_id")
        if self.static_refs != tuple(sorted(set(self.static_refs))):
            raise ValueError("static_refs must be sorted and unique")
        if self.operators != tuple(sorted(set(self.operators))):
            raise ValueError("operators must be sorted and unique")
        if any(
            Path(item).is_absolute() or ".." in Path(item).parts
            for item in self.static_refs
        ):
            raise ValueError("static_refs must be safe relative paths")
        if self.skill_sha256 != _digest_without(
            self.model_dump(mode="json"), "skill_sha256"
        ):
            raise ValueError("skill_sha256 mismatch")
        return self

    def to_runtime_skill(self) -> Skill:
        return Skill(
            slug=self.slug,
            version=self.version,
            description=self.description,
            body=self.body,
            static_refs=list(self.static_refs),
            operators=list(self.operators),
        )


def render_skill_markdown(skill: StrictSkillArtifact) -> bytes:
    """Render a deterministic, lossless SKILL.md representation."""

    try:
        skill = StrictSkillArtifact.model_validate(
            skill.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise AuthoringContractError("Skill artifact is invalid") from error
    metadata = skill.model_dump(mode="json")
    body = metadata.pop("body")
    metadata["format"] = "skillchain-static-skill-v1"
    rendered = (
        _SKILL_MARKDOWN_PREFIX
        + canonical_json_bytes(metadata)
        + _SKILL_MARKDOWN_SEPARATOR
        + body.encode("utf-8")
    )
    if parse_skill_markdown(rendered) != skill:
        raise AuthoringContractError("SKILL.md render did not round-trip")
    return rendered


def parse_skill_markdown(content: bytes) -> StrictSkillArtifact:
    """Parse only the canonical SKILL.md format produced above."""

    if not content.startswith(_SKILL_MARKDOWN_PREFIX):
        raise AuthoringContractError("SKILL.md header is invalid")
    remainder = content[len(_SKILL_MARKDOWN_PREFIX) :]
    if _SKILL_MARKDOWN_SEPARATOR not in remainder:
        raise AuthoringContractError("SKILL.md metadata boundary is invalid")
    metadata_bytes, body_bytes = remainder.split(_SKILL_MARKDOWN_SEPARATOR, 1)
    try:
        metadata = parse_canonical_json(metadata_bytes, label="SKILL.md metadata")
        if not isinstance(metadata, dict):
            raise AuthoringContractError("SKILL.md metadata must be an object")
        if metadata.pop("format", None) != "skillchain-static-skill-v1":
            raise AuthoringContractError("SKILL.md format identity is invalid")
        body = body_bytes.decode("utf-8", errors="strict")
        skill = StrictSkillArtifact.model_validate(
            {**metadata, "body": body}, strict=True
        )
    except (ArtifactFormatError, UnicodeDecodeError, ValidationError) as error:
        raise AuthoringContractError("SKILL.md is not canonical and valid") from error
    canonical = (
        _SKILL_MARKDOWN_PREFIX
        + canonical_json_bytes({**metadata, "format": "skillchain-static-skill-v1"})
        + _SKILL_MARKDOWN_SEPARATOR
        + body_bytes
    )
    if canonical != content:
        raise AuthoringContractError("SKILL.md bytes are not canonical")
    return skill


def publish_skill_markdown_bank(
    bank: StaticBankArtifact, artifact_dir: str | Path
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for skill in bank.skills:
        path = Path(artifact_dir) / "skills" / skill.slug / "SKILL.md"
        atomic_create_file(path, render_skill_markdown(skill))
        paths.append(path)
    return tuple(paths)


class BankCapabilityBinding(_StrictFrozenModel):
    capability_id: str
    skill_slug: str


class StaticBankArtifact(_StrictFrozenModel):
    schema_version: Literal[2, 3] = 2
    baseline_kind: Literal["spec", "llm_static"]
    construction_identity_sha256: Sha256
    construction_identity_policy: Literal["spec-content-v1", "reviewed-draft-v1"]
    runtime_binding_policy: Literal["registry-runtime-v1", "registry-runtime-v2"] = (
        "registry-runtime-v1"
    )
    compiler: CompilerIdentity
    tool_registry_sha256: Sha256
    tool_registry_runtime_sha256: Sha256
    skills: tuple[StrictSkillArtifact, ...]
    capability_map: tuple[BankCapabilityBinding, ...]
    bank_sha256: Sha256

    @field_validator("skills", "capability_map", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        expected_generation = (
            (2, "3.0.0", "registry-runtime-v1")
            if self.schema_version == 2
            else (3, "4.0.0", "registry-runtime-v2")
        )
        if (
            self.schema_version,
            self.compiler.compiler_version,
            self.runtime_binding_policy,
        ) != expected_generation:
            raise ValueError("Bank schema/compiler/runtime generations differ")
        expected_policy = (
            "spec-content-v1" if self.baseline_kind == "spec" else "reviewed-draft-v1"
        )
        if self.construction_identity_policy != expected_policy:
            raise ValueError("construction identity policy mismatches baseline kind")
        slugs = tuple(item.slug for item in self.skills)
        if slugs != tuple(sorted(set(slugs))):
            raise ValueError("skills must be sorted and unique")
        capabilities = tuple(item.capability_id for item in self.capability_map)
        mapped_slugs = tuple(item.skill_slug for item in self.capability_map)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("capability map must be sorted and unique")
        if len(self.capability_map) != len(self.skills) or set(mapped_slugs) != set(
            slugs
        ):
            raise ValueError("capability map must cover every skill exactly once")
        skill_capability_by_slug = {
            item.slug: item.capability_id for item in self.skills
        }
        if any(
            skill_capability_by_slug.get(item.skill_slug) != item.capability_id
            for item in self.capability_map
        ):
            raise ValueError("capability map disagrees with Skill capability identity")
        if self.bank_sha256 != _digest_without(
            self.model_dump(mode="json"), "bank_sha256"
        ):
            raise ValueError("bank_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)

    def runtime_skills(self) -> tuple[Skill, ...]:
        return tuple(item.to_runtime_skill() for item in self.skills)


def _expected_coverage(task: CapabilityTaskSpec) -> dict[str, tuple[str, ...]]:
    return _coverage_for(task).model_dump(mode="python")


def _assert_complete_draft(
    draft: CapabilityDraft,
    task: CapabilityTaskSpec,
    allowed_source_ids: set[str],
    *,
    baseline_kind: Literal["spec", "llm_static"],
) -> None:
    actual = draft.rule_coverage.model_dump(mode="python")
    if actual != _expected_coverage(task):
        raise AuthoringContractError(
            f"draft does not cover every frozen rule: {task.capability_id}"
        )
    if draft.output_coverage != _output_for(task):
        raise AuthoringContractError(
            f"draft does not cover the complete output contract: {task.capability_id}"
        )
    if (
        draft.fallback_may_request_clarification
        != task.fallback.may_request_clarification
        or draft.fallback_must_state_uncertainty != task.fallback.must_state_uncertainty
    ):
        raise AuthoringContractError("draft fallback flags differ from frozen contract")
    if baseline_kind == "spec" and draft.citation_source_ids:
        raise AuthoringContractError(
            "SpecBaseline must use frozen TaskSpec provenance, not public-source ids"
        )
    if (
        baseline_kind == "llm_static"
        and allowed_source_ids
        and not draft.citation_source_ids
    ):
        raise AuthoringContractError(
            "LLMStatic draft must cite at least one verified public source"
        )
    if not set(draft.citation_source_ids) <= allowed_source_ids:
        raise AuthoringContractError("draft cites an unverified public source")
    allowed = set(task.allowed_tools)
    used = {item.tool_name for item in draft.steps}
    if not used <= allowed:
        raise AuthoringContractError("draft uses a tool outside capability allowlist")
    success = set(draft.rule_coverage.success_rule_ids)
    if any(not set(item.success_rule_ids) <= success for item in draft.steps):
        raise AuthoringContractError("draft step cites an unknown success rule")
    if set().union(*(set(item.success_rule_ids) for item in draft.steps)) != success:
        raise AuthoringContractError(
            "draft steps do not collectively cover every success rule"
        )


def _render_rules(title: str, rules) -> list[str]:
    return [
        f"## {title}",
        *(
            f"- [{item.rule_id}] {item.statement} "
            f"(source: {item.provenance.kind}:{item.provenance.source_ref})"
            for item in rules
        ),
    ]


def _render_body(draft: CapabilityDraft, task: CapabilityTaskSpec) -> str:
    output = task.output_contract
    lines = ["# Objective", draft.objective]
    lines += _render_rules("Input preconditions", task.input_preconditions)
    lines += _render_rules("Success criteria", task.success_criteria)
    lines += ["## Tool procedure"]
    lines += [
        f"{index}. [{step.tool_name}] {step.instruction}"
        for index, step in enumerate(draft.steps, start=1)
    ]
    lines += _render_rules("Acceptable answer rules", task.acceptable_answer_rules)
    lines += [
        "## Output contract",
        f"- response_kind: {output.response_kind}",
        f"- required_sections: {','.join(output.required_sections)}",
        f"- card_requirement: {output.card_requirement}",
        f"- card_fields: {','.join(output.card_fields)}",
        f"- evidence_requirement: {output.evidence_requirement}",
    ]
    lines += _render_rules("Failure conditions", task.failure_conditions)
    lines += _render_rules("Safety constraints", task.safety_constraints)
    lines += _render_rules("Fallback triggers", task.fallback.trigger_rules)
    lines += _render_rules("Fallback responses", task.fallback.response_rules)
    lines += _render_rules("Indeterminate conditions", task.indeterminate_conditions)
    lines += [
        "## Fallback flags",
        (
            "- may_request_clarification: "
            f"{str(draft.fallback_may_request_clarification).lower()}"
        ),
        "- must_state_uncertainty: true",
        "## Authored public source citations",
        *(
            (f"- {source_id}" for source_id in draft.citation_source_ids)
            if draft.citation_source_ids
            else ("- None; this SpecBaseline uses frozen TaskSpec provenance.",)
        ),
        "## Authored fallback instruction",
        draft.fallback_instruction,
    ]
    return "\n\n".join(lines) + "\n"


def _spec_baseline_identity(value: AuthoringInput) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": "spec-baseline-identity-v1",
                "taxonomy_bytes_sha256": value.taxonomy.bytes_sha256,
                "task_specification_bytes_sha256": (
                    value.task_specification.bytes_sha256
                ),
                "tool_registry_bytes_sha256": value.tool_registry.bytes_sha256,
                "prompt_sha256": value.prompt.prompt_sha256,
                "compiler": value.compiler.model_dump(mode="json"),
            }
        )
    )


def _validated_bank_materials(
    value: AuthoringInput,
    bundle: AuthoringDraftBundle,
    *,
    baseline_kind: Literal["spec", "llm_static"],
) -> tuple[
    tuple[StrictSkillArtifact, ...],
    tuple[BankCapabilityBinding, ...],
    str,
]:
    """Validate and render content without claiming a machine runtime binding."""

    _, tasks, registry = _materialize_specifications(value)
    if bundle.authoring_input_sha256 != value.input_sha256:
        raise AuthoringContractError("draft bundle input binding mismatch")
    by_id = tasks.capabilities_by_id
    if {item.capability_id for item in bundle.drafts} != set(by_id):
        raise AuthoringContractError("draft capability set mismatch")
    registered = {item.name for item in registry.tools}
    allowed_source_ids = {item.source_id for item in value.public_sources}
    skills: list[StrictSkillArtifact] = []
    bindings: list[BankCapabilityBinding] = []
    for draft in sorted(bundle.drafts, key=lambda item: item.capability_id):
        task = by_id[draft.capability_id]
        _assert_complete_draft(
            draft,
            task,
            allowed_source_ids,
            baseline_kind=baseline_kind,
        )
        operators = tuple(sorted({item.tool_name for item in draft.steps}))
        if not set(operators) <= registered:
            raise AuthoringContractError("draft references an unregistered tool")
        slug = "static-" + re.sub(r"[^a-z0-9]+", "-", draft.capability_id).strip("-")
        skill_payload = {
            "slug": slug,
            "version": 1,
            "description": draft.objective,
            "body": _render_body(draft, task),
            "static_refs": [],
            "operators": list(operators),
            "capability_id": draft.capability_id,
            "parent_skill_sha256": None,
        }
        skill = StrictSkillArtifact.model_validate(
            {
                **skill_payload,
                "skill_sha256": sha256_bytes(canonical_json_bytes(skill_payload)),
            }
        )
        skills.append(skill)
        bindings.append(
            BankCapabilityBinding(
                capability_id=draft.capability_id, skill_slug=skill.slug
            )
        )
    construction_identity = (
        _spec_baseline_identity(value)
        if baseline_kind == "spec"
        else bundle.bundle_sha256
    )
    return tuple(skills), tuple(bindings), construction_identity


def _compile(
    value: AuthoringInput,
    bundle: AuthoringDraftBundle,
    *,
    baseline_kind: Literal["spec", "llm_static"],
    tool_registry_runtime_sha256: str | None = None,
) -> StaticBankArtifact:
    skills, bindings, construction_identity = _validated_bank_materials(
        value,
        bundle,
        baseline_kind=baseline_kind,
    )
    _, _, registry = _materialize_specifications(value)
    runtime_sha256 = tool_registry_runtime_sha256 or value.tool_registry_runtime_sha256
    if runtime_sha256 is None:
        raise AuthoringContractError(
            "Bank compilation requires an independently verified tool runtime"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", runtime_sha256):
        raise AuthoringContractError("tool runtime SHA-256 is invalid")
    payload = {
        "schema_version": 2 if registry.schema_version == 1 else 3,
        "baseline_kind": baseline_kind,
        "construction_identity_sha256": construction_identity,
        "construction_identity_policy": (
            "spec-content-v1" if baseline_kind == "spec" else "reviewed-draft-v1"
        ),
        "runtime_binding_policy": (f"registry-runtime-v{registry.schema_version}"),
        "compiler": value.compiler.model_dump(mode="json"),
        "tool_registry_sha256": value.tool_registry.identity_sha256,
        "tool_registry_runtime_sha256": runtime_sha256,
        "skills": [item.model_dump(mode="json") for item in skills],
        "capability_map": [item.model_dump(mode="json") for item in bindings],
    }
    return StaticBankArtifact.model_validate(
        {**payload, "bank_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def load_static_bank(
    path: str | Path,
    *,
    expected_file_sha256: str,
    registry: ToolRegistry,
) -> StaticBankArtifact:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise AuthoringContractError("expected Bank file SHA-256 is invalid")
    try:
        content = read_stable_regular_file(path, label="static Bank")
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError("static Bank file digest mismatch")
        raw = parse_canonical_json(content, label="static Bank")
        if not isinstance(raw, dict):
            raise AuthoringContractError("static Bank must contain an object")
        bank = StaticBankArtifact.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError(
            "static Bank is not canonical and valid"
        ) from error
    if (
        bank.tool_registry_sha256 != registry.registry_sha256
        or bank.tool_registry_runtime_sha256 != registry.registry_runtime_sha256
    ):
        raise AuthoringContractError("static Bank live registry binding mismatch")
    registered = {item.name for item in registry.specs()}
    if any(not set(item.operators) <= registered for item in bank.skills):
        raise AuthoringContractError("static Bank contains an unavailable operator")
    return bank


def runtime_skill_for_capability(
    bank: StaticBankArtifact, capability_id: str, registry: ToolRegistry
) -> Skill:
    try:
        bank = StaticBankArtifact.model_validate(
            bank.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise AuthoringContractError("runtime Bank violates frozen schema") from error
    if bank.tool_registry_runtime_sha256 != registry.registry_runtime_sha256:
        raise AuthoringContractError("runtime registry does not match Bank")
    mapping = {item.capability_id: item.skill_slug for item in bank.capability_map}
    slug = mapping.get(capability_id)
    if slug is None:
        raise AuthoringContractError("capability is absent from Bank")
    artifact = next(item for item in bank.skills if item.slug == slug)
    if not set(artifact.operators) <= {item.name for item in registry.specs()}:
        raise AuthoringContractError("runtime Skill operator is unavailable")
    return artifact.to_runtime_skill()


def load_skill_markdown(
    path: str | Path, *, expected_file_sha256: str
) -> StrictSkillArtifact:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise AuthoringContractError("expected SKILL.md SHA-256 is invalid")
    try:
        content = read_stable_regular_file(path, label="SKILL.md")
    except ArtifactFormatError as error:
        raise AuthoringContractError("SKILL.md file is unsafe") from error
    if sha256_bytes(content) != expected_file_sha256:
        raise AuthoringContractError("SKILL.md file digest mismatch")
    return parse_skill_markdown(content)


def invoke_skill_operator(
    skill: Skill,
    operator: ToolName,
    arguments: Mapping[str, object],
    context: ToolExecutionContext,
    registry: ToolRegistry,
) -> ToolInvocationResult:
    """Dispatch a runtime Skill operator through the live typed registry."""

    try:
        skill = Skill.model_validate(skill.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise AuthoringContractError("runtime Skill is invalid") from error
    if operator not in skill.operators:
        raise AuthoringContractError("operator is not declared by runtime Skill")
    return registry.invoke(operator, arguments, context)


class CanonicalAuthoringRequest(_StrictFrozenModel):
    schema_version: Literal[1, 2, 3] = 1
    authoring_input: AuthoringInput
    decoding: FixedDecoding
    price_schedule: PriceSchedule
    output_contract: AuthoringOutputContract | AuthoringContentOutputContract | None = (
        None
    )
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.decoding != self.authoring_input.decoding:
            raise ValueError("request decoding differs from frozen input")
        if self.price_schedule != self.authoring_input.price_schedule:
            raise ValueError("request price schedule differs from frozen input")
        if self.schema_version == 1:
            if (
                self.output_contract is not None
                or self.decoding.response_format != CANONICAL_AUTHORING_RESPONSE_FORMAT
            ):
                raise ValueError("request-v1 cannot carry a structured output contract")
        elif self.schema_version == 2:
            if (
                self.output_contract != build_authoring_output_contract()
                or self.decoding.response_format != FORCED_SUBMISSION_RESPONSE_FORMAT
            ):
                raise ValueError("request-v2 output contract drifted")
        elif (
            self.output_contract
            != build_authoring_content_output_contract(self.authoring_input)
            or self.decoding.response_format != PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        ):
            raise ValueError("request-v3 output contract drifted")
        if self.request_sha256 != sha256_bytes(
            canonical_json_bytes(self._unsigned_payload())
        ):
            raise ValueError("request_sha256 mismatch")
        return self

    def _unsigned_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "authoring_input": self.authoring_input.model_dump(mode="json"),
            "decoding": self.decoding.model_dump(mode="json"),
            "price_schedule": self.price_schedule.model_dump(mode="json"),
        }
        if self.schema_version in {2, 3}:
            if self.output_contract is None:  # guarded by the model validator
                raise ValueError("structured request output contract is missing")
            payload["output_contract"] = self.output_contract.model_dump(mode="json")
        return payload

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                **self._unsigned_payload(),
                "request_sha256": self.request_sha256,
            }
        )


class IsolationAttestation(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    execution_mode: Literal["in_process_test"] = "in_process_test"
    process_isolated: Literal[False] = False
    network_mechanically_disabled: Literal[False] = False
    filesystem_mechanically_allowlisted: Literal[False] = False
    readable_paths: tuple[str, ...] = ()
    writable_paths: tuple[str, ...] = ()
    formal_eligible: Literal[False] = False
    reason: Literal["python_transport_is_not_mechanically_isolated"] = (
        "python_transport_is_not_mechanically_isolated"
    )
    attestation_sha256: Sha256

    @field_validator("readable_paths", "writable_paths", mode="before")
    @classmethod
    def coerce_paths(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_attestation(self) -> Self:
        if self.readable_paths or self.writable_paths:
            raise ValueError("in-process test transport must not claim path isolation")
        if self.attestation_sha256 != _digest_without(
            self.model_dump(mode="json"), "attestation_sha256"
        ):
            raise ValueError("attestation_sha256 mismatch")
        return self


def in_process_test_attestation() -> IsolationAttestation:
    payload = {
        "schema_version": 1,
        "execution_mode": "in_process_test",
        "process_isolated": False,
        "network_mechanically_disabled": False,
        "filesystem_mechanically_allowlisted": False,
        "readable_paths": [],
        "writable_paths": [],
        "formal_eligible": False,
        "reason": "python_transport_is_not_mechanically_isolated",
    }
    return IsolationAttestation.model_validate(
        {
            **payload,
            "attestation_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
    )


class AuthoringSandboxProfile(_StrictFrozenModel):
    """Externally pinned controls for the isolated authoring container.

    The network name refers to an operator-managed network whose firewall/proxy
    receipt is independently committed by ``network_policy_sha256``.  Docker or
    Podman flags alone cannot prove an egress allowlist, so a profile without
    that external commitment is deliberately not loadable as a formal handle.
    """

    schema_version: Literal[2] = 2
    engine: Literal["docker"]
    engine_path: str
    engine_binary_sha256: Sha256
    image_reference: str
    image_digest: Sha256
    network_name: str
    network_id: Sha256
    network_policy_sha256: Sha256
    proxy_url: str
    proxy_container_name: str
    proxy_image_digest: Sha256
    proxy_external_network_name: str
    allowed_provider_endpoint: str
    credential_env_name: str
    memory_megabytes: int = Field(ge=256, le=32768)
    cpu_count_milli: int = Field(ge=100, le=16000)
    pids_limit: int = Field(ge=16, le=1024)
    timeout_seconds: int = Field(ge=30, le=7200)
    profile_sha256: Sha256

    @field_validator(
        "engine_path",
        "image_reference",
        "network_name",
        "network_id",
        "proxy_url",
        "proxy_container_name",
        "proxy_external_network_name",
        "allowed_provider_endpoint",
        "credential_env_name",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        engine_path = Path(self.engine_path)
        if not engine_path.is_absolute() or engine_path.name.casefold() not in {
            self.engine,
            f"{self.engine}.exe",
        }:
            raise ValueError("engine_path must be an absolute matching executable")
        repository_digest = self.image_reference.endswith(
            f"@sha256:{self.image_digest}"
        )
        local_image_id = self.image_reference == f"sha256:{self.image_digest}"
        if not (repository_digest or local_image_id):
            raise ValueError("image_reference must bind the immutable image digest")
        endpoint = urlparse(self.allowed_provider_endpoint)
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username
            or endpoint.port not in {None, 443}
        ):
            raise ValueError(
                "allowed provider endpoint must be credential-free HTTPS on port 443"
            )
        proxy = urlparse(self.proxy_url)
        if (
            proxy.scheme != "http"
            or proxy.hostname != self.proxy_container_name
            or proxy.port != 3128
            or proxy.username
            or proxy.path not in {"", "/"}
        ):
            raise ValueError(
                "proxy_url must be an unauthenticated http://<proxy-container>:3128 URL"
            )
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.credential_env_name):
            raise ValueError("credential_env_name must be an environment variable name")
        if self.profile_sha256 != _digest_without(
            self.model_dump(mode="json"), "profile_sha256"
        ):
            raise ValueError("sandbox profile SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)


@dataclass(frozen=True)
class VerifiedAuthoringSandboxProfile:
    value: AuthoringSandboxProfile
    path: Path
    file_sha256: str
    _verification_token: object = field(repr=False, compare=False)


def load_verified_authoring_sandbox_profile(
    path: str | Path, *, expected_file_sha256: str
) -> VerifiedAuthoringSandboxProfile:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise AuthoringContractError("expected sandbox-profile SHA-256 is invalid")
    try:
        content = read_stable_regular_file(path, label="authoring sandbox profile")
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError("sandbox-profile external digest mismatch")
        raw = parse_canonical_json(content, label="authoring sandbox profile")
        if not isinstance(raw, dict):
            raise AuthoringContractError("sandbox profile must contain an object")
        profile = AuthoringSandboxProfile.model_validate(raw, strict=True)
        engine_path = Path(profile.engine_path)
        if engine_path.is_symlink():
            raise AuthoringContractError("sandbox engine must not be a symlink")
        engine = engine_path.resolve(strict=True)
        if not engine.is_file():
            raise AuthoringContractError("sandbox engine must be a regular file")
        before = engine.stat()
        engine_bytes = engine.read_bytes()
        after = engine.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AuthoringContractError("sandbox engine changed during verification")
        if sha256_bytes(engine_bytes) != profile.engine_binary_sha256:
            raise AuthoringContractError("sandbox engine binary digest mismatch")
    except (ArtifactFormatError, OSError, ValidationError) as error:
        raise AuthoringContractError(
            "sandbox profile is not safely verifiable"
        ) from error
    return VerifiedAuthoringSandboxProfile(
        value=profile,
        path=Path(path).resolve(strict=True),
        file_sha256=expected_file_sha256,
        _verification_token=_VERIFIED_SANDBOX_PROFILE_TOKEN,
    )


@dataclass(frozen=True)
class _AuthoringCallAuthorizationSource:
    repository_root: Path
    freeze_lock_path: Path
    expected_freeze_lock_file_sha256: str
    invocation_input: VerifiedAuthoringInvocationInput
    sandbox_profile: VerifiedAuthoringSandboxProfile
    variant: str
    run_id: str
    output_directory: Path
    receipt_path: Path


@dataclass(frozen=True)
class _AuthoringCallAuthorizationFacts:
    authorization_id: str
    approved_by: str
    freeze_lock_file_sha256: str
    protocol_deviation_file_sha256: str
    authoring_input_file_sha256: str
    authoring_input_sha256: str
    authorized_budgets_sha256: str
    output_contract_sha256: str
    runtime_lock_bundle_sha256: str
    runtime_file_sha256s: tuple[tuple[str, str], ...]
    authorized_variant: str
    authorized_model: str
    required_run_id: str
    required_output_directory: Path
    required_output_directory_reference: str
    required_claim_file: Path
    required_claim_file_reference: str
    required_receipt_file: Path
    required_receipt_file_reference: str


@dataclass
class _AuthoringCallAuthorizationState:
    lock: Any = field(default_factory=Lock, repr=False, compare=False)
    consumed: bool = False


@dataclass(frozen=True)
class VerifiedAuthoringCallAuthorization:
    """Opaque, one-call authority rooted in an independently pinned freeze.

    The handle deliberately carries mutable consumption state in addition to a
    create-only claim path.  The in-memory state stops reuse of one handle,
    while the claim prevents a second loader, gateway, or process from
    consuming the same owner-issued authorization.
    """

    _facts: _AuthoringCallAuthorizationFacts = field(repr=False)
    _source: _AuthoringCallAuthorizationSource = field(repr=False, compare=False)
    _state: _AuthoringCallAuthorizationState = field(repr=False, compare=False)
    _verification_token: object = field(repr=False, compare=False)

    @property
    def authorization_id(self) -> str:
        return self._facts.authorization_id

    @property
    def required_run_id(self) -> str:
        return self._facts.required_run_id

    @property
    def required_output_directory(self) -> Path:
        return self._facts.required_output_directory

    @property
    def required_claim_file(self) -> Path:
        return self._facts.required_claim_file

    @property
    def required_receipt_file(self) -> Path:
        return self._facts.required_receipt_file


def _authority_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise AuthoringContractError(f"{label} SHA-256 is invalid")
    return value


def _authority_nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuthoringContractError(f"{label} is missing")
    return value


def _authority_path(
    repository_root: Path,
    value: object,
    label: str,
    *,
    must_exist: bool,
) -> tuple[Path, str]:
    reference = _authority_nonblank(value, label)
    relative = Path(reference)
    if relative.is_absolute() or relative.drive:
        raise AuthoringContractError(f"{label} must be repository-relative")
    try:
        resolved = (repository_root / relative).resolve(strict=must_exist)
    except OSError as error:
        raise AuthoringContractError(f"{label} cannot be resolved") from error
    if resolved != repository_root and repository_root not in resolved.parents:
        raise AuthoringContractError(f"{label} escapes the repository root")
    if not must_exist and os.path.lexists(resolved):
        raise AuthoringContractError(f"{label} already exists")
    return resolved, reference.replace("\\", "/")


def _authority_object(
    path: Path,
    expected_file_sha256: str,
    label: str,
) -> dict[str, object]:
    try:
        content = read_stable_regular_file(
            path, label=label, max_bytes=_MAX_INPUT_BYTES
        )
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError(f"{label} file digest mismatch")
        raw = parse_canonical_json(content, label=label)
    except ArtifactFormatError as error:
        raise AuthoringContractError(f"{label} is not safely verifiable") from error
    if not isinstance(raw, dict):
        raise AuthoringContractError(f"{label} must contain an object")
    return raw


def _runtime_authority_bindings(
    repository_root: Path,
    bundle: object,
    *,
    sandbox_profile: VerifiedAuthoringSandboxProfile,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(bundle, dict):
        raise AuthoringContractError("authoring runtime lock bundle is missing")
    required = (
        ("deployment_receipt_file", "deployment_receipt_file_sha256"),
        ("lock_manifest_file", "lock_manifest_file_sha256"),
        ("network_policy_file", "network_policy_file_sha256"),
        ("sandbox_profile_file", "sandbox_profile_file_sha256"),
    )
    bindings: list[tuple[str, str]] = []
    objects: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    profile_path: Path | None = None
    for path_key, digest_key in required:
        path, _ = _authority_path(
            repository_root,
            bundle.get(path_key),
            f"runtime {path_key}",
            must_exist=True,
        )
        digest = _authority_sha256(
            bundle.get(digest_key),
            f"runtime {digest_key}",
        )
        objects[path_key] = _authority_object(path, digest, f"runtime {path_key}")
        paths[path_key] = path
        bindings.append((digest_key, digest))
        if path_key == "sandbox_profile_file":
            profile_path = path
    if (
        profile_path != sandbox_profile.path
        or dict(bindings)["sandbox_profile_file_sha256"] != sandbox_profile.file_sha256
    ):
        raise AuthoringContractError(
            "call authorization runtime does not match the verified sandbox profile"
        )
    runtime_version = _authority_nonblank(
        bundle.get("runtime_version"), "runtime version"
    )
    binding_map = dict(bindings)
    deployment = objects["deployment_receipt_file"]
    manifest = objects["lock_manifest_file"]
    network = objects["network_policy_file"]
    profile = sandbox_profile.value
    manifest_root = paths["lock_manifest_file"].parent
    expected_paths = {
        "deployment_receipt_file": manifest_root / "deployment-receipt.json",
        "network_policy_file": manifest_root / "network-policy.json",
        "sandbox_profile_file": manifest_root / "sandbox-profile.json",
    }
    probes = deployment.get("probes")
    required_probes = {
        "allowlisted_connect_succeeds",
        "non_allowlisted_connect_denied",
        "direct_ip_egress_blocked",
        "external_dns_blocked",
    }
    if (
        any(paths[key] != expected for key, expected in expected_paths.items())
        or manifest.get("deployment_receipt_file_sha256")
        != binding_map["deployment_receipt_file_sha256"]
        or manifest.get("network_policy_file_sha256")
        != binding_map["network_policy_file_sha256"]
        or manifest.get("sandbox_profile_file_sha256")
        != binding_map["sandbox_profile_file_sha256"]
        or network.get("policy_sha256") != _digest_without(network, "policy_sha256")
        or deployment.get("receipt_sha256")
        != _digest_without(deployment, "receipt_sha256")
        or not isinstance(probes, dict)
        or set(probes) != required_probes
        or any(
            not isinstance(item, dict) or item.get("exit_code") != 0
            for item in probes.values()
        )
        or deployment.get("runtime_version") != runtime_version
        or deployment.get("engine_binary_sha256") != profile.engine_binary_sha256
        or deployment.get("image_digest") != profile.image_digest
        or deployment.get("network_id") != profile.network_id
        or deployment.get("network_policy_file_sha256") != profile.network_policy_sha256
        or deployment.get("sandbox_profile_file_sha256") != sandbox_profile.file_sha256
        or deployment.get("proxy_image_digest") != profile.proxy_image_digest
        or network.get("allowed_hostname")
        != urlparse(profile.allowed_provider_endpoint).hostname
        or network.get("allowed_port") != 443
        or network.get("authoring_network_id") != profile.network_id
        or network.get("authoring_network_name") != profile.network_name
        or network.get("authoring_network_internal") is not True
        or network.get("proxy_container_name") != profile.proxy_container_name
        or network.get("proxy_external_network_name")
        != profile.proxy_external_network_name
        or network.get("proxy_image_digest") != profile.proxy_image_digest
        or network.get("proxy_image_digest") != profile.image_digest
        or network.get("proxy_url") != profile.proxy_url
        or profile.allowed_provider_endpoint
        != f"https://{network.get('allowed_hostname')}/compatible-mode/v1"
        or network.get("denies_direct_egress") is not True
        or network.get("denies_non_allowlisted_proxy_targets") is not True
        or network.get("denies_ip_literal_proxy_targets") is not True
    ):
        raise AuthoringContractError("authoring runtime lock bundle cross-check failed")
    runtime_sha256 = sha256_bytes(canonical_json_bytes(bundle))
    return runtime_sha256, tuple(sorted(bindings))


def _verify_canonical_request_authority_binding(
    repository_root: Path,
    binding: object,
    *,
    authoring_input: AuthoringInput,
) -> None:
    if not isinstance(binding, dict):
        raise AuthoringContractError("v5 freeze lacks its canonical-request binding")
    required_fields = {
        "file",
        "file_sha256",
        "byte_size",
        "semantic_request_sha256",
    }
    if set(binding) != required_fields:
        raise AuthoringContractError("v5 canonical-request binding fields drifted")
    request_path, _ = _authority_path(
        repository_root,
        binding.get("file"),
        "canonical authoring request",
        must_exist=True,
    )
    expected_file_sha256 = _authority_sha256(
        binding.get("file_sha256"),
        "canonical authoring request file",
    )
    expected_semantic_sha256 = _authority_sha256(
        binding.get("semantic_request_sha256"),
        "canonical authoring request semantic identity",
    )
    expected_byte_size = binding.get("byte_size")
    if type(expected_byte_size) is not int or expected_byte_size <= 0:
        raise AuthoringContractError("canonical authoring request byte size is invalid")
    try:
        persisted = read_stable_regular_file(
            request_path,
            label="canonical authoring request",
            max_bytes=_MAX_INPUT_BYTES,
        )
    except ArtifactFormatError as error:
        raise AuthoringContractError(
            "canonical authoring request is not safely verifiable"
        ) from error
    request = build_canonical_authoring_request(authoring_input)
    canonical = request.canonical_bytes()
    if (
        len(persisted) != expected_byte_size
        or len(canonical) != expected_byte_size
        or sha256_bytes(persisted) != expected_file_sha256
        or persisted != canonical
        or request.request_sha256 != expected_semantic_sha256
    ):
        raise AuthoringContractError("canonical authoring request authority drifted")


def _verify_authoring_call_authorization(
    source: _AuthoringCallAuthorizationSource,
) -> _AuthoringCallAuthorizationFacts:
    invocation_input = _refresh_verified_authoring_invocation_input(
        source.invocation_input
    )
    value = invocation_input.value
    if (
        value.schema_version != 4
        or value.decoding.response_format != PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    ):
        raise AuthoringContractError(
            "call authorization handles are reserved for authoring schema v4"
        )
    profile = load_verified_authoring_sandbox_profile(
        source.sandbox_profile.path,
        expected_file_sha256=source.sandbox_profile.file_sha256,
    )
    freeze = _authority_object(
        source.freeze_lock_path,
        source.expected_freeze_lock_file_sha256,
        "authoring freeze lock",
    )
    if freeze.get("status") != "frozen":
        raise AuthoringContractError(
            "owner-issued frozen call authorization is required; "
            "candidate authority is not executable"
        )
    approved_by = _authority_nonblank(
        freeze.get("approved_by"), "freeze approving owner"
    )
    variants = freeze.get("variants")
    common = freeze.get("common")
    authorization = freeze.get("call_authorization")
    if not all(isinstance(item, dict) for item in (variants, common, authorization)):
        raise AuthoringContractError("authoring freeze lacks formal call authority")
    variants_dict = variants if isinstance(variants, dict) else {}
    common_dict = common if isinstance(common, dict) else {}
    authorization_dict = authorization if isinstance(authorization, dict) else {}
    selected = variants_dict.get(source.variant)
    if not isinstance(selected, dict):
        raise AuthoringContractError("authorized authoring variant is missing")

    packet_path, _ = _authority_path(
        source.repository_root,
        selected.get("packet_file"),
        "authorized authoring packet",
        must_exist=True,
    )
    packet_file_sha256 = _authority_sha256(
        selected.get("packet_file_sha256"), "authorized authoring packet"
    )
    if (
        packet_path != invocation_input.path
        or packet_file_sha256 != invocation_input.file_sha256
        or selected.get("input_sha256") != value.input_sha256
        or selected.get("model") != value.model.model
        or selected.get("model_revision") != value.model.revision
    ):
        raise AuthoringContractError(
            "call authorization does not bind the verified authoring packet"
        )
    price_path, _ = _authority_path(
        source.repository_root,
        selected.get("price_schedule_file"),
        "authorized price schedule",
        must_exist=True,
    )
    price_file_sha256 = _authority_sha256(
        selected.get("price_schedule_file_sha256"),
        "authorized price schedule",
    )
    if (
        price_path != invocation_input.price_schedule.path
        or price_file_sha256 != invocation_input.price_schedule.file_sha256
    ):
        raise AuthoringContractError(
            "call authorization does not bind the verified price schedule"
        )

    budgets = value.budgets.model_dump(mode="json")
    budgets_sha256 = sha256_bytes(canonical_json_bytes(budgets))
    if (
        common_dict.get("taxonomy_sha256") != invocation_input.expected_taxonomy_sha256
        or common_dict.get("task_specification_sha256")
        != invocation_input.expected_task_specification_sha256
        or common_dict.get("tool_registry_sha256")
        != invocation_input.expected_tool_registry_sha256
        or common_dict.get("budgets") != budgets
        or common_dict.get("decoding") != value.decoding.model_dump(mode="json")
    ):
        raise AuthoringContractError(
            "call authorization protocol or budget binding drifted"
        )
    _verify_canonical_request_authority_binding(
        source.repository_root,
        common_dict.get("canonical_request"),
        authoring_input=value,
    )
    output_contract = build_authoring_content_output_contract(value)
    output_channel = common_dict.get("output_channel")
    if not isinstance(output_channel, dict):
        raise AuthoringContractError("v5 freeze lacks its output-channel lock")
    schema_path, _ = _authority_path(
        source.repository_root,
        output_channel.get("json_schema_file"),
        "authoring content JSON Schema",
        must_exist=True,
    )
    schema_sha256 = _authority_sha256(
        output_channel.get("json_schema_file_sha256"),
        "authoring content JSON Schema",
    )
    schema_bytes = read_stable_regular_file(
        schema_path,
        label="authoring content JSON Schema",
        max_bytes=_MAX_INPUT_BYTES,
    )
    if (
        sha256_bytes(schema_bytes) != schema_sha256
        or schema_bytes != output_contract.json_schema_canonical_json.encode("utf-8")
        or output_channel.get("mode") != output_contract.mode
        or output_channel.get("schema_name") != output_contract.schema_name
        or output_channel.get("output_contract_sha256")
        != output_contract.contract_sha256
        or output_channel.get("provider_response_format")
        != {"type": output_contract.provider_response_format}
        or output_channel.get("provider_guarantee")
        != output_contract.provider_guarantee
        or output_channel.get("json_schema_enforcement")
        != output_contract.json_schema_enforcement
        or output_channel.get("runner_packet_contract_validation") is not True
        or output_channel.get("runner_validation_engine")
        != output_contract.runner_validation_engine
        or output_channel.get("tools_supplied") is not False
    ):
        raise AuthoringContractError("v5 output-channel authority drifted")

    runtime_sha256, runtime_bindings = _runtime_authority_bindings(
        source.repository_root,
        common_dict.get("runtime_lock_bundle"),
        sandbox_profile=profile,
    )
    required_output, output_reference = _authority_path(
        source.repository_root,
        authorization_dict.get("required_output_directory"),
        "authorized output directory",
        must_exist=False,
    )
    required_claim, claim_reference = _authority_path(
        source.repository_root,
        authorization_dict.get("required_claim_file"),
        "authorized attempt claim",
        must_exist=False,
    )
    required_receipt, receipt_reference = _authority_path(
        source.repository_root,
        authorization_dict.get("required_receipt_file"),
        "authorized invocation receipt",
        must_exist=False,
    )
    authorized_paths = (required_output, required_claim, required_receipt)
    paths_are_nested = any(
        left in right.parents or right in left.parents
        for index, left in enumerate(authorized_paths)
        for right in authorized_paths[index + 1 :]
    )
    if (
        len(set(authorized_paths)) != len(authorized_paths)
        or required_output == source.repository_root
        or paths_are_nested
    ):
        raise AuthoringContractError(
            "authorized output paths must be pairwise distinct and non-nested"
        )
    deviation_path, _ = _authority_path(
        source.repository_root,
        authorization_dict.get("protocol_deviation_file"),
        "protocol deviation",
        must_exist=True,
    )
    deviation_sha256 = _authority_sha256(
        authorization_dict.get("protocol_deviation_file_sha256"),
        "protocol deviation",
    )
    deviation = _authority_object(
        deviation_path,
        deviation_sha256,
        "protocol deviation",
    )
    call_budget = deviation.get("call_budget")
    if (
        deviation.get("status")
        != "approved_for_one_prospective_interface_repair_attempt"
        or deviation.get("approved_by") != approved_by
        or not isinstance(call_budget, dict)
    ):
        raise AuthoringContractError(
            "protocol deviation is not an owner-approved prospective authority"
        )
    call_budget_dict = call_budget if isinstance(call_budget, dict) else {}
    budget_keys = (
        "authorized_model",
        "authorized_variant",
        "required_run_id",
        "max_additional_provider_attempts",
        "consume_on",
        "reissue_policy",
        "retry_fallback_repair_policy",
    )
    if any(
        call_budget_dict.get(key) != authorization_dict.get(key) for key in budget_keys
    ):
        raise AuthoringContractError(
            "protocol deviation and call budget authority differ"
        )
    if (
        authorization_dict.get("authority_issued") is not True
        or authorization_dict.get("provider_call_authorized") is not True
        or authorization_dict.get("budget_authorized") is not True
        or authorization_dict.get("runtime_lock_frozen") is not True
        or authorization_dict.get("approved_by") != approved_by
        or authorization_dict.get("authorized_variant") != source.variant
        or authorization_dict.get("authorized_model") != value.model.model
        or authorization_dict.get("authoring_input_file_sha256")
        != invocation_input.file_sha256
        or authorization_dict.get("authoring_input_sha256") != value.input_sha256
        or authorization_dict.get("authorized_budgets_sha256") != budgets_sha256
        or authorization_dict.get("output_contract_sha256")
        != output_contract.contract_sha256
        or authorization_dict.get("runtime_lock_bundle_sha256") != runtime_sha256
        or authorization_dict.get("sandbox_profile_file_sha256") != profile.file_sha256
        or authorization_dict.get("required_run_id") != source.run_id
        or authorization_dict.get("max_additional_provider_attempts") != 1
        or authorization_dict.get("consume_on")
        != "atomic-create-claim-before-container-launch"
        or authorization_dict.get("reissue_policy")
        != "new-owner-approved-deviation-only"
        or authorization_dict.get("retry_fallback_repair_policy") != "forbidden"
        or required_output != source.output_directory
        or required_receipt != source.receipt_path
    ):
        raise AuthoringContractError(
            "owner-issued call authorization does not match this v5 invocation"
        )
    authorization_id = _authority_nonblank(
        authorization_dict.get("authorization_id"),
        "call authorization id",
    )
    return _AuthoringCallAuthorizationFacts(
        authorization_id=authorization_id,
        approved_by=approved_by,
        freeze_lock_file_sha256=source.expected_freeze_lock_file_sha256,
        protocol_deviation_file_sha256=deviation_sha256,
        authoring_input_file_sha256=invocation_input.file_sha256,
        authoring_input_sha256=value.input_sha256,
        authorized_budgets_sha256=budgets_sha256,
        output_contract_sha256=output_contract.contract_sha256,
        runtime_lock_bundle_sha256=runtime_sha256,
        runtime_file_sha256s=runtime_bindings,
        authorized_variant=source.variant,
        authorized_model=value.model.model,
        required_run_id=source.run_id,
        required_output_directory=required_output,
        required_output_directory_reference=output_reference,
        required_claim_file=required_claim,
        required_claim_file_reference=claim_reference,
        required_receipt_file=required_receipt,
        required_receipt_file_reference=receipt_reference,
    )


def load_verified_authoring_call_authorization(
    freeze_lock_path: str | Path,
    *,
    expected_freeze_lock_file_sha256: str,
    repository_root: str | Path,
    invocation_input: VerifiedAuthoringInvocationInput,
    sandbox_profile: VerifiedAuthoringSandboxProfile,
    variant: str,
    run_id: str,
    output_directory: str | Path,
    receipt_path: str | Path,
) -> VerifiedAuthoringCallAuthorization:
    """Verify, but do not yet consume, one owner-issued v5 call authority."""

    freeze_sha256 = _authority_sha256(
        expected_freeze_lock_file_sha256,
        "expected authoring freeze lock",
    )
    if (
        not isinstance(invocation_input, VerifiedAuthoringInvocationInput)
        or invocation_input._verification_token is not _VERIFIED_HANDLE_TOKEN
    ):
        raise TypeError(
            "authoring call authority requires VerifiedAuthoringInvocationInput"
        )
    if (
        not isinstance(sandbox_profile, VerifiedAuthoringSandboxProfile)
        or sandbox_profile._verification_token is not _VERIFIED_SANDBOX_PROFILE_TOKEN
    ):
        raise TypeError(
            "authoring call authority requires VerifiedAuthoringSandboxProfile"
        )
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise AuthoringContractError(
            "call-authority repository root is not a directory"
        )
    freeze_path = Path(freeze_lock_path).resolve(strict=True)
    if freeze_path != root and root not in freeze_path.parents:
        raise AuthoringContractError("authoring freeze lock is outside repository root")
    source = _AuthoringCallAuthorizationSource(
        repository_root=root,
        freeze_lock_path=freeze_path,
        expected_freeze_lock_file_sha256=freeze_sha256,
        invocation_input=invocation_input,
        sandbox_profile=sandbox_profile,
        variant=_authority_nonblank(variant, "authorized variant"),
        run_id=_authority_nonblank(run_id, "authorized run id"),
        output_directory=Path(output_directory).resolve(strict=False),
        receipt_path=Path(receipt_path).resolve(strict=False),
    )
    facts = _verify_authoring_call_authorization(source)
    return VerifiedAuthoringCallAuthorization(
        _facts=facts,
        _source=source,
        _state=_AuthoringCallAuthorizationState(),
        _verification_token=_VERIFIED_CALL_AUTHORIZATION_TOKEN,
    )


def _consume_authoring_call_authorization(
    value: VerifiedAuthoringCallAuthorization,
    *,
    request: CanonicalAuthoringRequest,
    sandbox_profile: VerifiedAuthoringSandboxProfile,
    output_directory: Path,
) -> str:
    if (
        not isinstance(value, VerifiedAuthoringCallAuthorization)
        or value._verification_token is not _VERIFIED_CALL_AUTHORIZATION_TOKEN
    ):
        raise AuthoringContractError(
            "v5 formal authoring requires a verified owner-issued call authorization"
        )
    if (
        sandbox_profile.path != value._source.sandbox_profile.path
        or sandbox_profile.file_sha256 != value._source.sandbox_profile.file_sha256
        or output_directory.resolve(strict=False)
        != value._facts.required_output_directory
        or request.authoring_input.input_sha256 != value._facts.authoring_input_sha256
    ):
        raise AuthoringContractError(
            "verified call authorization does not match the formal gateway request"
        )
    with value._state.lock:
        if value._state.consumed:
            raise AuthoringContractError(
                "owner-issued authoring call authorization was already consumed"
            )
        refreshed = _verify_authoring_call_authorization(value._source)
        if refreshed != value._facts:
            raise AuthoringContractError(
                "owner-issued authoring call authorization changed before use"
            )
        payload = {
            "schema_version": 1,
            "authorization_id": refreshed.authorization_id,
            "status": "consumed_before_container_launch",
            "consume_on": "atomic-create-claim-before-container-launch",
            "freeze_lock_file_sha256": refreshed.freeze_lock_file_sha256,
            "protocol_deviation_file_sha256": (
                refreshed.protocol_deviation_file_sha256
            ),
            "authoring_input_file_sha256": refreshed.authoring_input_file_sha256,
            "authoring_input_sha256": refreshed.authoring_input_sha256,
            "authorized_budgets_sha256": refreshed.authorized_budgets_sha256,
            "output_contract_sha256": refreshed.output_contract_sha256,
            "runtime_lock_bundle_sha256": refreshed.runtime_lock_bundle_sha256,
            "runtime_file_sha256s": dict(refreshed.runtime_file_sha256s),
            "request_sha256": request.request_sha256,
            "run_id": refreshed.required_run_id,
            "variant": refreshed.authorized_variant,
            "model": refreshed.authorized_model,
            "required_output_directory": (
                refreshed.required_output_directory_reference
            ),
            "required_claim_file": refreshed.required_claim_file_reference,
            "required_receipt_file": refreshed.required_receipt_file_reference,
        }
        claim_bytes = canonical_json_bytes(
            {
                **payload,
                "claim_sha256": sha256_bytes(canonical_json_bytes(payload)),
            }
        )
        atomic_create_file(refreshed.required_claim_file, claim_bytes)
        value._state.consumed = True
    return sha256_bytes(claim_bytes)


class FormalIsolationAttestation(_StrictFrozenModel):
    schema_version: Literal[2] = 2
    execution_mode: Literal["container_job"] = "container_job"
    process_isolated: Literal[True] = True
    network_mechanically_allowlisted: Literal[True] = True
    filesystem_mechanically_allowlisted: Literal[True] = True
    root_filesystem_read_only: Literal[True] = True
    capabilities_dropped: Literal[True] = True
    no_new_privileges: Literal[True] = True
    input_mount: Literal["/input:ro"] = "/input:ro"
    output_mount: Literal["/output:rw"] = "/output:rw"
    sandbox_profile_file_sha256: Sha256
    sandbox_profile_sha256: Sha256
    engine_binary_sha256: Sha256
    image_digest: Sha256
    network_id: Sha256
    network_policy_sha256: Sha256
    proxy_image_digest: Sha256
    request_sha256: Sha256
    response_file_sha256: Sha256
    command_sha256: Sha256
    stdout_sha256: Sha256
    stderr_sha256: Sha256
    elapsed_ms: int = Field(ge=0)
    formal_eligible: Literal[True] = True
    attestation_sha256: Sha256

    @model_validator(mode="after")
    def validate_attestation(self) -> Self:
        if self.attestation_sha256 != _digest_without(
            self.model_dump(mode="json"), "attestation_sha256"
        ):
            raise ValueError("formal isolation attestation SHA-256 mismatch")
        return self


IsolationEvidence = IsolationAttestation | FormalIsolationAttestation


def _parse_isolation_evidence(raw: dict[str, object]) -> IsolationEvidence:
    model = (
        FormalIsolationAttestation
        if raw.get("execution_mode") == "container_job"
        else IsolationAttestation
    )
    return model.model_validate(raw, strict=True)


class ControlledLLMTransport(Protocol):
    """Testable transport boundary: the only input is one canonical request."""

    def complete(self, canonical_request: bytes) -> LLMResponse: ...


class UnifiedChatTransport:
    """Production transport through the repository's single ``llm.chat`` entry."""

    def complete(self, canonical_request: bytes) -> LLMResponse:
        try:
            raw = parse_canonical_json(
                canonical_request, label="controlled authoring request"
            )
            if not isinstance(raw, dict):
                raise AuthoringContractError(
                    "controlled authoring request must contain an object"
                )
            request = CanonicalAuthoringRequest.model_validate(raw, strict=True)
        except (ArtifactFormatError, ValidationError) as error:
            raise AuthoringContractError(
                "controlled authoring request is invalid"
            ) from error
        value = request.authoring_input
        transport_kwargs: dict[str, Any]
        if value.decoding.response_format in {
            CANONICAL_AUTHORING_RESPONSE_FORMAT,
            PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
        }:
            transport_kwargs = {
                "json_mode": True,
            }
        else:
            contract = request.output_contract
            if contract is None:
                raise AuthoringContractError(
                    "structured authoring request lacks its output contract"
                )
            if not isinstance(contract, AuthoringOutputContract):
                raise AuthoringContractError(
                    "forced-function authoring request has the wrong output contract"
                )
            raw_tool = parse_canonical_json(
                contract.tool_definition_canonical_json.encode("utf-8"),
                label="controlled authoring output tool",
            )
            if not isinstance(raw_tool, dict):
                raise AuthoringContractError(
                    "controlled authoring output tool must be an object"
                )
            transport_kwargs = {
                "json_mode": False,
                "tools": [raw_tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": contract.tool_name},
                },
                "parallel_tool_calls": contract.parallel_tool_calls,
            }
        return llm_module.chat(
            value.model.provider,
            [
                {"role": "system", "content": value.prompt.template},
                {"role": "user", "content": canonical_request.decode("utf-8")},
            ],
            model=value.model.model,
            temperature=value.decoding.temperature_milli / 1000,
            top_p=value.decoding.top_p_milli / 1000,
            seed=value.decoding.seed,
            max_tokens=value.decoding.max_output_tokens,
            thinking=False,
            max_attempts=1,
            record_usage=False,
            **transport_kwargs,
        )


class ControlledAuthoringGateway:
    """One-use gateway.  Python transports can never claim formal isolation."""

    def __init__(self, transport: ControlledLLMTransport) -> None:
        if not hasattr(transport, "complete"):
            raise TypeError("transport must implement complete(canonical_request)")
        self._transport = transport
        self._used = False
        self._attestation = in_process_test_attestation()

    @property
    def attestation(self) -> IsolationAttestation:
        return self._attestation

    def complete_once(self, canonical_request: bytes) -> LLMResponse:
        if self._used:
            raise AuthoringContractError("authoring gateway permits only one call")
        self._used = True
        try:
            raw = parse_canonical_json(
                canonical_request,
                label="controlled diagnostic authoring request",
            )
            if not isinstance(raw, dict):
                raise AuthoringContractError(
                    "controlled diagnostic authoring request must contain an object"
                )
            request = CanonicalAuthoringRequest.model_validate(raw, strict=True)
        except (ArtifactFormatError, ValidationError) as error:
            raise AuthoringContractError(
                "controlled diagnostic authoring request is invalid"
            ) from error
        if (
            request.authoring_input.schema_version == 4
            and request.authoring_input.decoding.response_format
            == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        ):
            raise AuthoringContractError(
                "v5 authoring cannot execute through a diagnostic transport; use "
                "sealed offline replay or verified formal authorization"
            )
        return self._transport.complete(canonical_request)


class OfflineAuthoringReplayGateway:
    """Replay one pre-materialized response without owning or calling a transport."""

    __slots__ = ("_attestation", "_request", "_response", "_used")

    def __init__(self, value: AuthoringInput, response: LLMResponse) -> None:
        if type(value) is not AuthoringInput:
            raise TypeError("offline replay requires one exact AuthoringInput")
        input_value = _revalidate_input(value)
        if (
            input_value.schema_version != 4
            or input_value.decoding.response_format
            != PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        ):
            raise TypeError("offline replay is reserved for v5 diagnostic authoring")
        if type(response) is not LLMResponse:
            raise TypeError("offline replay requires one exact LLMResponse")
        try:
            response = LLMResponse.model_validate(
                response.model_dump(mode="python"),
                strict=True,
            )
        except (AttributeError, ValidationError) as error:
            raise TypeError("offline replay requires one strict LLMResponse") from error
        self._request = build_canonical_authoring_request(input_value)
        self._response = response
        self._used = False
        self._attestation = in_process_test_attestation()

    @property
    def attestation(self) -> IsolationAttestation:
        return self._attestation

    def complete_once(self, canonical_request: bytes) -> LLMResponse:
        if self._used:
            raise AuthoringContractError("authoring replay permits only one use")
        self._used = True
        if canonical_request != self._request.canonical_bytes():
            raise AuthoringContractError(
                "offline replay request differs from its materialized response binding"
            )
        return self._response


_OFFLINE_AUTHORING_REPLAY_COMPLETE_ONCE = OfflineAuthoringReplayGateway.complete_once


class FormalContainerAuthoringGateway:
    """One-use Docker/Podman boundary with a parent-owned execution receipt."""

    __slots__ = ("_profile", "_used", "_attestation", "_authorized_v5_request")

    def __init__(self, profile: VerifiedAuthoringSandboxProfile) -> None:
        if (
            not isinstance(profile, VerifiedAuthoringSandboxProfile)
            or profile._verification_token is not _VERIFIED_SANDBOX_PROFILE_TOKEN
        ):
            raise TypeError("formal gateway requires an externally verified profile")
        self._profile = profile
        self._used = False
        self._attestation: FormalIsolationAttestation | None = None
        self._authorized_v5_request: tuple[str, str, object] | None = None

    @property
    def attestation(self) -> FormalIsolationAttestation:
        if self._attestation is None:
            raise AuthoringContractError(
                "formal isolation receipt is not available yet"
            )
        return self._attestation

    def _refresh_profile(self) -> AuthoringSandboxProfile:
        verified = load_verified_authoring_sandbox_profile(
            self._profile.path,
            expected_file_sha256=self._profile.file_sha256,
        )
        if verified.value != self._profile.value:
            raise AuthoringContractError("verified sandbox profile changed")
        return verified.value

    @staticmethod
    def _inspect_json(profile: AuthoringSandboxProfile, *arguments: str) -> object:
        try:
            completed = subprocess.run(
                [profile.engine_path, *arguments],
                check=False,
                capture_output=True,
                timeout=30,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AuthoringContractError("sandbox runtime inspection failed") from error
        if completed.returncode != 0:
            raise AuthoringContractError("sandbox runtime inspection returned failure")
        try:
            return json.loads(completed.stdout)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuthoringContractError(
                "sandbox runtime inspection returned invalid JSON"
            ) from error

    @classmethod
    def _verify_runtime_network(cls, profile: AuthoringSandboxProfile) -> None:
        network_raw = cls._inspect_json(
            profile, "network", "inspect", profile.network_name
        )
        if not isinstance(network_raw, list) or len(network_raw) != 1:
            raise AuthoringContractError("sandbox network inspection is ambiguous")
        network = network_raw[0]
        if (
            not isinstance(network, dict)
            or network.get("Id") != profile.network_id
            or network.get("Internal") is not True
        ):
            raise AuthoringContractError("sandbox network identity/policy drifted")

        proxy_raw = cls._inspect_json(
            profile, "container", "inspect", profile.proxy_container_name
        )
        if not isinstance(proxy_raw, list) or len(proxy_raw) != 1:
            raise AuthoringContractError("sandbox proxy inspection is ambiguous")
        proxy = proxy_raw[0]
        if not isinstance(proxy, dict):
            raise AuthoringContractError("sandbox proxy inspection is invalid")
        state = proxy.get("State")
        config_raw = proxy.get("Config")
        host_config = proxy.get("HostConfig")
        network_settings = proxy.get("NetworkSettings")
        if not all(
            isinstance(item, dict)
            for item in (state, config_raw, host_config, network_settings)
        ):
            raise AuthoringContractError("sandbox proxy metadata is incomplete")
        config_dict = config_raw if isinstance(config_raw, dict) else {}
        host_dict = host_config if isinstance(host_config, dict) else {}
        settings_dict = network_settings if isinstance(network_settings, dict) else {}
        expected_image = f"sha256:{profile.proxy_image_digest}"
        expected_hostname = urlparse(profile.allowed_provider_endpoint).hostname
        env = config_dict.get("Env")
        command = config_dict.get("Cmd")
        networks = settings_dict.get("Networks")
        cap_drop = host_dict.get("CapDrop")
        security_opt = host_dict.get("SecurityOpt")
        if (
            state.get("Running") is not True
            or proxy.get("Image") != expected_image
            or command != ["python", "-I", "-m", "skillchain.runners.egress_proxy"]
            or not isinstance(env, list)
            or f"SKILLCHAIN_ALLOWED_HOST={expected_hostname}" not in env
            or "SKILLCHAIN_ALLOWED_PORT=443" not in env
            or any(item.startswith(f"{profile.credential_env_name}=") for item in env)
            or host_dict.get("ReadonlyRootfs") is not True
            or not isinstance(cap_drop, list)
            or "ALL" not in cap_drop
            or not isinstance(security_opt, list)
            or "no-new-privileges" not in security_opt
            or not isinstance(networks, dict)
            or set(networks)
            != {profile.network_name, profile.proxy_external_network_name}
        ):
            raise AuthoringContractError("sandbox proxy identity/policy drifted")
        attached = network.get("Containers")
        if not isinstance(attached, dict) or set(attached) != {proxy.get("Id")}:
            raise AuthoringContractError("sandbox network has an unexpected peer")

    def _authorize_v5_call(
        self,
        request: CanonicalAuthoringRequest,
        authorization: VerifiedAuthoringCallAuthorization,
        output_directory: Path,
    ) -> None:
        """Consume one owner grant before any container/runtime/provider action."""

        if self._used or self._authorized_v5_request is not None:
            raise AuthoringContractError(
                "formal authoring gateway was already prepared"
            )
        if (
            request.authoring_input.schema_version != 4
            or request.authoring_input.decoding.response_format
            != PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
        ):
            raise AuthoringContractError(
                "v5 call authorization cannot be attached to a legacy request"
            )
        claim_sha256 = _consume_authoring_call_authorization(
            authorization,
            request=request,
            sandbox_profile=self._profile,
            output_directory=output_directory,
        )
        self._authorized_v5_request = (
            request.request_sha256,
            claim_sha256,
            _CONSUMED_CALL_AUTHORIZATION_TOKEN,
        )

    def complete_once(self, canonical_request: bytes) -> LLMResponse:
        if self._used:
            raise AuthoringContractError("authoring gateway permits only one call")
        self._used = True
        try:
            raw = parse_canonical_json(
                canonical_request, label="formal authoring request"
            )
            if not isinstance(raw, dict):
                raise AuthoringContractError(
                    "formal authoring request must be an object"
                )
            request = CanonicalAuthoringRequest.model_validate(raw, strict=True)
        except (ArtifactFormatError, ValidationError) as error:
            raise AuthoringContractError(
                "formal authoring request is invalid"
            ) from error
        value = request.authoring_input
        if value.schema_version == 4:
            authorized = self._authorized_v5_request
            if (
                authorized is None
                or authorized[0] != request.request_sha256
                or authorized[2] is not _CONSUMED_CALL_AUTHORIZATION_TOKEN
            ):
                raise AuthoringContractError(
                    "v5 formal authoring reached the gateway without a consumed "
                    "owner-issued call authorization"
                )
        elif self._authorized_v5_request is not None:
            raise AuthoringContractError(
                "legacy formal authoring cannot consume a v5 call authorization"
            )
        profile = self._refresh_profile()
        self._verify_runtime_network(profile)
        if value.model.endpoint != profile.allowed_provider_endpoint:
            raise AuthoringContractError(
                "sandbox egress profile does not match provider"
            )
        expected_env = config.PROVIDER_API_KEY_ENV[value.model.provider]
        if profile.credential_env_name != expected_env:
            raise AuthoringContractError(
                "sandbox credential name does not match provider"
            )
        if not os.environ.get(expected_env):
            raise AuthoringContractError(
                f"required temporary credential is absent: {expected_env}"
            )

        with tempfile.TemporaryDirectory(prefix="skillchain-author-") as temp:
            root = Path(temp)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            output_root.mkdir()
            atomic_create_file(input_root / "authoring-request.json", canonical_request)
            for mount_path in (input_root, output_root):
                if "," in str(mount_path):
                    raise AuthoringContractError(
                        "container mount path contains a comma"
                    )
            command = [
                profile.engine_path,
                "run",
                "--rm",
                "--pull=never",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                f"--pids-limit={profile.pids_limit}",
                f"--memory={profile.memory_megabytes}m",
                f"--cpus={profile.cpu_count_milli / 1000:g}",
                f"--network={profile.network_name}",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                "--mount",
                f"type=bind,src={input_root},dst=/input,readonly",
                "--mount",
                f"type=bind,src={output_root},dst=/output",
                "--env",
                profile.credential_env_name,
                "--env",
                f"HTTPS_PROXY={profile.proxy_url}",
                "--env",
                f"https_proxy={profile.proxy_url}",
                "--env",
                "NO_PROXY=",
                "--env",
                "no_proxy=",
                profile.image_reference,
                "python",
                "-I",
                "-m",
                "skillchain.runners.authoring_worker",
            ]
            command_sha256 = sha256_bytes(canonical_json_bytes(command))
            started = time.perf_counter_ns()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=profile.timeout_seconds,
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise AuthoringContractError("isolated authoring job failed") from error
            elapsed_ms = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            if completed.returncode != 0:
                raise FormalAuthoringJobError(
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    elapsed_ms=elapsed_ms,
                    command_sha256=command_sha256,
                )
            entries = tuple(sorted(item.name for item in output_root.iterdir()))
            if entries != ("authoring-response.json",):
                raise AuthoringContractError("isolated job output file set is invalid")
            try:
                response_bytes = read_stable_regular_file(
                    output_root / "authoring-response.json",
                    label="isolated authoring response",
                    max_bytes=_MAX_RESPONSE_BYTES,
                )
                response_raw = parse_canonical_json(
                    response_bytes, label="isolated authoring response"
                )
                if not isinstance(response_raw, dict):
                    raise AuthoringContractError(
                        "isolated authoring response must contain an object"
                    )
                response = LLMResponse.model_validate(response_raw)
            except (ArtifactFormatError, ValidationError) as error:
                raise AuthoringContractError(
                    "isolated authoring response is invalid"
                ) from error
            payload = {
                "schema_version": 2,
                "execution_mode": "container_job",
                "process_isolated": True,
                "network_mechanically_allowlisted": True,
                "filesystem_mechanically_allowlisted": True,
                "root_filesystem_read_only": True,
                "capabilities_dropped": True,
                "no_new_privileges": True,
                "input_mount": "/input:ro",
                "output_mount": "/output:rw",
                "sandbox_profile_file_sha256": self._profile.file_sha256,
                "sandbox_profile_sha256": profile.profile_sha256,
                "engine_binary_sha256": profile.engine_binary_sha256,
                "image_digest": profile.image_digest,
                "network_id": profile.network_id,
                "network_policy_sha256": profile.network_policy_sha256,
                "proxy_image_digest": profile.proxy_image_digest,
                "request_sha256": request.request_sha256,
                "response_file_sha256": sha256_bytes(response_bytes),
                "command_sha256": command_sha256,
                "stdout_sha256": sha256_bytes(completed.stdout),
                "stderr_sha256": sha256_bytes(completed.stderr),
                "elapsed_ms": elapsed_ms,
                "formal_eligible": True,
            }
            self._attestation = FormalIsolationAttestation.model_validate(
                {
                    **payload,
                    "attestation_sha256": sha256_bytes(canonical_json_bytes(payload)),
                },
                strict=True,
            )
            return response


_FORMAL_CONTAINER_AUTHORIZE_V5_CALL = FormalContainerAuthoringGateway._authorize_v5_call
_FORMAL_CONTAINER_COMPLETE_ONCE = FormalContainerAuthoringGateway.complete_once


@dataclass(frozen=True)
class AuthoringInvocation:
    authoring_input_sha256: str
    request: CanonicalAuthoringRequest
    response: LLMResponse
    request_file_sha256: str
    response_file_sha256: str
    pre_review_draft_file_sha256: str
    pre_review_draft: AuthoringDraftBundle
    cost_microusd: int
    isolation_attestation: IsolationEvidence
    isolation_attestation_file_sha256: str
    _formal_token: object | None = field(default=None, repr=False, compare=False)


def build_canonical_authoring_request(
    value: AuthoringInput,
) -> CanonicalAuthoringRequest:
    """Build the exact canonical request passed to an authoring transport."""

    value = _revalidate_input(value)
    response_format = value.decoding.response_format
    structured = response_format != CANONICAL_AUTHORING_RESPONSE_FORMAT
    content_submission = response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    payload: dict[str, object] = {
        "schema_version": 3 if content_submission else (2 if structured else 1),
        "authoring_input": value.model_dump(mode="json"),
        "decoding": value.decoding.model_dump(mode="json"),
        "price_schedule": value.price_schedule.model_dump(mode="json"),
    }
    if content_submission:
        payload["output_contract"] = build_authoring_content_output_contract(
            value
        ).model_dump(mode="json")
    elif structured:
        payload["output_contract"] = build_authoring_output_contract().model_dump(
            mode="json"
        )
    return CanonicalAuthoringRequest.model_validate(
        {**payload, "request_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def _calculated_cost(value: AuthoringInput, response: LLMResponse) -> int:
    schedule = value.price_schedule
    numerator = (
        response.usage.input_tokens * schedule.input_microusd_per_million_tokens
        + response.usage.output_tokens * schedule.output_microusd_per_million_tokens
    )
    return (numerator + 999_999) // 1_000_000


def _normalize_author_response(
    response_text: str, *, authoring_input_sha256: str
) -> AuthoringDraftBundle:
    try:
        parsed = parse_canonical_json(
            response_text.encode("utf-8"), label="author draft payload"
        )
        if not isinstance(parsed, dict):
            raise AuthoringContractError("author draft payload must contain an object")
        payload = AuthoringDraftPayload.model_validate(parsed, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError(
            "author draft payload is not canonical and valid"
        ) from error
    return normalize_authoring_draft_payload(
        payload,
        authoring_input_sha256=authoring_input_sha256,
    )


def authoring_response_envelope_is_complete(
    decoding: FixedDecoding, response: LLMResponse
) -> bool:
    """Return whether provider framing matches the frozen output channel."""

    if decoding.response_format in {
        CANONICAL_AUTHORING_RESPONSE_FORMAT,
        PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    }:
        return response.finish_reason == "stop" and not response.tool_calls
    return (
        response.finish_reason == "tool_calls"
        and response.text == ""
        and len(response.tool_calls) == 1
        and response.tool_calls[0].name == AUTHORING_SUBMISSION_TOOL_NAME
    )


def _normalize_authoring_response(
    response: LLMResponse,
    *,
    decoding: FixedDecoding,
    authoring_input: AuthoringInput,
) -> AuthoringDraftBundle:
    if decoding.response_format == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT:
        if response.finish_reason != "stop":
            raise AuthoringContractError(
                "JSON author response is truncated or did not finish with stop"
            )
        if response.tool_calls:
            raise AuthoringContractError(
                "JSON author response must not contain tool calls"
            )
        try:
            parsed = parse_strict_json(
                response.text.encode("utf-8"),
                label="author content JSON",
            )
            if not isinstance(parsed, dict):
                raise AuthoringContractError(
                    "author content JSON must contain an object"
                )
            content_payload = AuthoringContentPayload.model_validate(
                parsed, strict=True
            )
        except (ArtifactFormatError, UnicodeEncodeError, ValidationError) as error:
            raise AuthoringContractError("author content JSON is not valid") from error
        return normalize_authoring_content_payload(
            content_payload,
            authoring_input=authoring_input,
        )

    if decoding.response_format == CANONICAL_AUTHORING_RESPONSE_FORMAT:
        if response.finish_reason != "stop":
            raise AuthoringContractError(
                "author response is truncated or did not finish with stop"
            )
        if response.tool_calls:
            raise AuthoringContractError("static author must not issue tool calls")
        return _normalize_author_response(
            response.text,
            authoring_input_sha256=authoring_input.input_sha256,
        )

    if response.finish_reason != "tool_calls":
        raise AuthoringContractError(
            "structured author response did not finish with tool_calls"
        )
    if response.text != "":
        raise AuthoringContractError(
            "structured author response must not include assistant text"
        )
    if len(response.tool_calls) != 1:
        raise AuthoringContractError(
            "structured author response must contain exactly one submission"
        )
    submission = response.tool_calls[0]
    if submission.name != AUTHORING_SUBMISSION_TOOL_NAME:
        raise AuthoringContractError(
            "structured author response used an unexpected submission name"
        )
    try:
        parsed = parse_strict_json(
            submission.arguments_json.encode("utf-8"),
            label="author submission arguments",
        )
        if not isinstance(parsed, dict):
            raise AuthoringContractError(
                "author submission arguments must contain an object"
            )
        payload = AuthoringDraftPayload.model_validate(parsed, strict=True)
    except (ArtifactFormatError, UnicodeEncodeError, ValidationError) as error:
        raise AuthoringContractError(
            "author submission arguments are not valid"
        ) from error
    return normalize_authoring_draft_payload(
        payload,
        authoring_input_sha256=authoring_input.input_sha256,
    )


def _authoring_response_payload_bytes(
    decoding: FixedDecoding, response: LLMResponse
) -> bytes:
    """Return raw authored payload bytes for the legacy manifest digest field."""

    if decoding.response_format in {
        CANONICAL_AUTHORING_RESPONSE_FORMAT,
        PROVIDER_JSON_CONTENT_RESPONSE_FORMAT,
    }:
        return response.text.encode("utf-8")
    if not authoring_response_envelope_is_complete(decoding, response):
        raise AuthoringContractError("author response envelope is incomplete")
    return response.tool_calls[0].arguments_json.encode("utf-8")


def invoke_llm_static(
    value: AuthoringInput | VerifiedAuthoringInput | VerifiedAuthoringInvocationInput,
    gateway: (
        ControlledAuthoringGateway
        | OfflineAuthoringReplayGateway
        | FormalContainerAuthoringGateway
    ),
    artifact_dir: str | Path,
    *,
    require_formal_eligibility: bool = True,
    call_authorization: VerifiedAuthoringCallAuthorization | None = None,
) -> AuthoringInvocation:
    """Persist one request/response audit; never retry the controlled transport."""

    verified_input: VerifiedAuthoringInput | None = None
    verified_invocation_input: VerifiedAuthoringInvocationInput | None = None
    if isinstance(value, VerifiedAuthoringInput):
        verified_input = _refresh_verified_authoring_input(value)
        input_value = verified_input.value
    elif isinstance(value, VerifiedAuthoringInvocationInput):
        verified_invocation_input = _refresh_verified_authoring_invocation_input(value)
        input_value = verified_invocation_input.value
    else:
        input_value = _revalidate_input(value)
    is_v5_call = (
        input_value.schema_version == 4
        and input_value.decoding.response_format
        == PROVIDER_JSON_CONTENT_RESPONSE_FORMAT
    )
    if not isinstance(
        gateway,
        (
            ControlledAuthoringGateway,
            OfflineAuthoringReplayGateway,
            FormalContainerAuthoringGateway,
        ),
    ):
        raise AuthoringContractError("runner requires a controlled authoring gateway")
    if require_formal_eligibility:
        if (
            (verified_input is None and verified_invocation_input is None)
            or type(gateway) is not FormalContainerAuthoringGateway
            or FormalContainerAuthoringGateway._authorize_v5_call
            is not _FORMAL_CONTAINER_AUTHORIZE_V5_CALL
            or FormalContainerAuthoringGateway.complete_once
            is not _FORMAL_CONTAINER_COMPLETE_ONCE
        ):
            raise AuthoringContractError(
                "formal isolation is not mechanically proven without verified input "
                "and the container gateway; "
                "refusing author call"
            )
    elif is_v5_call:
        if (
            type(gateway) is not OfflineAuthoringReplayGateway
            or OfflineAuthoringReplayGateway.complete_once
            is not _OFFLINE_AUTHORING_REPLAY_COMPLETE_ONCE
        ):
            raise AuthoringContractError(
                "v5 diagnostic authoring accepts only a sealed offline response "
                "replay; transports require formal owner authorization"
            )
    elif type(gateway) is not ControlledAuthoringGateway:
        raise AuthoringContractError(
            "formal container gateway cannot be downgraded to diagnostic execution"
        )
    if require_formal_eligibility and is_v5_call:
        if (
            not isinstance(
                call_authorization,
                VerifiedAuthoringCallAuthorization,
            )
            or call_authorization._verification_token
            is not _VERIFIED_CALL_AUTHORIZATION_TOKEN
        ):
            raise AuthoringContractError(
                "v5 formal authoring requires a verified owner-issued "
                "call authorization"
            )
    elif call_authorization is not None:
        raise AuthoringContractError(
            "call authorization can only be consumed by a v5 formal invocation"
        )
    request = build_canonical_authoring_request(input_value)
    root = Path(artifact_dir)
    if require_formal_eligibility and os.path.lexists(root):
        raise FileExistsError(
            f"formal authoring destination already exists; refusing overwrite: {root}"
        )
    input_bytes = input_value.canonical_bytes()
    request_bytes = request.canonical_bytes()
    if not require_formal_eligibility:
        root.mkdir(parents=True, exist_ok=True)
        atomic_create_file(root / "authoring-input.json", input_bytes)
        atomic_create_file(root / "authoring-request.json", request_bytes)
    elif is_v5_call:
        if not isinstance(gateway, FormalContainerAuthoringGateway):
            raise AuthoringContractError(
                "v5 formal authoring requires the formal container gateway"
            )
        if not isinstance(call_authorization, VerifiedAuthoringCallAuthorization):
            raise AuthoringContractError(
                "v5 formal authoring requires a verified owner-issued "
                "call authorization"
            )
        gateway._authorize_v5_call(request, call_authorization, root)
    try:
        response = gateway.complete_once(request_bytes)
    except FormalAuthoringJobError as error:
        failure_payload = {
            "schema_version": 1,
            "status": "isolated_job_failed",
            "formal_eligible": False,
            "request_sha256": request.request_sha256,
            "command_sha256": error.command_sha256,
            "exit_code": error.exit_code,
            "stdout_sha256": sha256_bytes(error.stdout),
            "stderr_sha256": sha256_bytes(error.stderr),
            "elapsed_ms": error.elapsed_ms,
        }
        failure_artifacts = {
            "authoring-input.json": input_bytes,
            "authoring-request.json": request_bytes,
            "container-stdout.bin": error.stdout,
            "container-stderr.bin": error.stderr,
            "job-failure.json": canonical_json_bytes(
                {
                    **failure_payload,
                    "failure_sha256": sha256_bytes(
                        canonical_json_bytes(failure_payload)
                    ),
                }
            ),
        }
        staging = new_staging_directory(root)
        try:
            for name, content in failure_artifacts.items():
                atomic_create_file(staging / name, content)
            atomic_publish_new_directory(staging, root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        raise
    attestation = gateway.attestation
    attestation_bytes = _canonical_model_bytes(attestation)
    if not require_formal_eligibility:
        atomic_create_file(root / "isolation-attestation.json", attestation_bytes)
    if not isinstance(response, LLMResponse):
        raise AuthoringContractError("transport returned a non-LLMResponse value")
    try:
        response = LLMResponse.model_validate(
            response.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise AuthoringContractError(
            "transport returned an invalid LLMResponse"
        ) from error
    response_bytes = _canonical_model_bytes(response)
    if len(response_bytes) > _MAX_RESPONSE_BYTES:
        raise AuthoringContractError("full LLMResponse exceeds byte limit")
    if require_formal_eligibility:
        # Persist the provider call before interpreting authored content. A bad
        # schema response still consumes the one-call budget and must remain an
        # auditable formal outcome instead of disappearing with an exception.
        call_artifacts = {
            "authoring-input.json": input_bytes,
            "authoring-request.json": request_bytes,
            "authoring-response.json": response_bytes,
            "isolation-attestation.json": attestation_bytes,
        }
        staging = new_staging_directory(root)
        try:
            for name, content in call_artifacts.items():
                atomic_create_file(staging / name, content)
            atomic_publish_new_directory(staging, root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    else:
        atomic_create_file(root / "authoring-response.json", response_bytes)
    if (
        response.provider != input_value.model.provider
        or response.endpoint != input_value.model.endpoint
        or response.requested_model != input_value.model.model
        or response.response_model != input_value.model.model
    ):
        raise AuthoringContractError("author response identity drift")
    if response.request_id == request.request_sha256:
        raise AuthoringContractError(
            "provider request id must be independently assigned"
        )
    cost = _calculated_cost(input_value, response)
    usage = response.usage
    if usage.input_tokens <= 0 or usage.output_tokens <= 0:
        raise AuthoringContractError("author response usage must be positive")
    if (
        usage.input_tokens > input_value.budgets.max_input_tokens
        or usage.output_tokens > input_value.budgets.max_output_tokens
        or usage.total_tokens > input_value.budgets.max_total_tokens
        or cost > input_value.budgets.max_cost_microusd
    ):
        raise AuthoringContractError("author call exceeded frozen budget")
    bundle = _normalize_authoring_response(
        response,
        decoding=input_value.decoding,
        authoring_input=input_value,
    )
    # Validate every rule and ToolSpec before a human may review the draft.
    # Machine-specific runtime identity is deliberately deferred to finalize.
    _validated_bank_materials(
        input_value,
        bundle,
        baseline_kind="llm_static",
    )
    draft_bytes = bundle.canonical_bytes()
    atomic_create_file(root / "pre-review-draft.json", draft_bytes)
    return AuthoringInvocation(
        authoring_input_sha256=input_value.input_sha256,
        request=request,
        response=response,
        request_file_sha256=sha256_bytes(request_bytes),
        response_file_sha256=sha256_bytes(response_bytes),
        pre_review_draft_file_sha256=sha256_bytes(draft_bytes),
        pre_review_draft=bundle,
        cost_microusd=cost,
        isolation_attestation=attestation,
        isolation_attestation_file_sha256=sha256_bytes(attestation_bytes),
        _formal_token=(
            _FORMAL_AUTHORING_INVOCATION_TOKEN if require_formal_eligibility else None
        ),
    )


def run_llm_static(
    value: AuthoringInput | VerifiedAuthoringInput | VerifiedAuthoringInvocationInput,
    gateway: ControlledAuthoringGateway | FormalContainerAuthoringGateway,
    artifact_dir: str | Path,
    *,
    require_formal_eligibility: bool = True,
    call_authorization: VerifiedAuthoringCallAuthorization | None = None,
) -> AuthoringInvocation:
    """Compatibility name for the request phase; human finalize is separate."""

    return invoke_llm_static(
        value,
        gateway,
        artifact_dir,
        require_formal_eligibility=require_formal_eligibility,
        call_authorization=call_authorization,
    )


class ReviewChecklist(_StrictFrozenModel):
    no_private_inputs: Literal[True] = True
    safety_checked: Literal[True] = True
    schema_valid: Literal[True] = True
    source_citations_checked: Literal[True] = True
    tool_permissions_checked: Literal[True] = True
    edit_scope_checked: Literal[True] = True


def _review_locked_projection(bundle: AuthoringDraftBundle) -> object:
    """Fields a reviewer may verify but must not structurally rewrite."""

    return {
        "schema_version": bundle.schema_version,
        "authoring_input_sha256": bundle.authoring_input_sha256,
        "drafts": [
            {
                "capability_id": draft.capability_id,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "success_rule_ids": list(step.success_rule_ids),
                    }
                    for step in draft.steps
                ],
                "fallback_may_request_clarification": (
                    draft.fallback_may_request_clarification
                ),
                "fallback_must_state_uncertainty": (
                    draft.fallback_must_state_uncertainty
                ),
                "rule_coverage": draft.rule_coverage.model_dump(mode="json"),
                "output_coverage": draft.output_coverage.model_dump(mode="json"),
            }
            for draft in bundle.drafts
        ],
    }


class HumanReviewArtifact(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    reviewer_id: str
    reviewer_kind: Literal["human"] = "human"
    checklist: ReviewChecklist
    review_minutes: int = Field(gt=0)
    edit_scope: Literal["prose_and_citations_only"]
    change_reason: str
    pre_review_canonical_json: str
    pre_review_sha256: Sha256
    post_review_canonical_json: str
    post_review_sha256: Sha256
    changed: bool
    review_sha256: Sha256

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        _nonblank(self.reviewer_id, "reviewer_id")
        _nonblank(self.change_reason, "change_reason")
        pre = _parse_draft_text(self.pre_review_canonical_json, "pre-review draft")
        post = _parse_draft_text(self.post_review_canonical_json, "post-review draft")
        if sha256_bytes(pre.canonical_bytes()) != self.pre_review_sha256:
            raise ValueError("pre-review SHA-256 mismatch")
        if sha256_bytes(post.canonical_bytes()) != self.post_review_sha256:
            raise ValueError("post-review SHA-256 mismatch")
        if self.changed != (pre.canonical_bytes() != post.canonical_bytes()):
            raise ValueError("human review changed flag does not match canonical diff")
        if _review_locked_projection(pre) != _review_locked_projection(post):
            raise ValueError("human review changed fields outside allowed edit scope")
        _scan_untrusted_text(
            {
                "reviewer_id": self.reviewer_id,
                "change_reason": self.change_reason,
            },
            "human review",
        )
        if self.review_sha256 != _digest_without(
            self.model_dump(mode="json"), "review_sha256"
        ):
            raise ValueError("review_sha256 mismatch")
        return self

    @property
    def post_review_bundle(self) -> AuthoringDraftBundle:
        return _parse_draft_text(self.post_review_canonical_json, "post-review draft")

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)


def _parse_draft_text(text: str, label: str) -> AuthoringDraftBundle:
    try:
        raw = parse_canonical_json(text.encode("utf-8"), label=label)
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must contain an object")
        return AuthoringDraftBundle.model_validate(raw, strict=True)
    except (ArtifactFormatError, ValidationError) as error:
        raise ValueError(f"{label} is not canonical and valid") from error


def build_human_review(
    *,
    pre_review: AuthoringDraftBundle,
    post_review: AuthoringDraftBundle,
    reviewer_id: str,
    review_minutes: int,
    change_reason: str,
    checklist: ReviewChecklist | None = None,
) -> HumanReviewArtifact:
    pre_text = pre_review.canonical_bytes().decode("utf-8")
    post_text = post_review.canonical_bytes().decode("utf-8")
    checklist_model = ReviewChecklist.model_validate(
        (checklist or ReviewChecklist()).model_dump(mode="python")
        if isinstance(checklist, ReviewChecklist) or checklist is None
        else checklist,
        strict=True,
    )
    payload = {
        "schema_version": 1,
        "reviewer_id": reviewer_id,
        "reviewer_kind": "human",
        "checklist": checklist_model.model_dump(mode="json"),
        "review_minutes": review_minutes,
        "edit_scope": "prose_and_citations_only",
        "change_reason": change_reason,
        "pre_review_canonical_json": pre_text,
        "pre_review_sha256": sha256_bytes(pre_text.encode("utf-8")),
        "post_review_canonical_json": post_text,
        "post_review_sha256": sha256_bytes(post_text.encode("utf-8")),
        "changed": pre_text != post_text,
    }
    return HumanReviewArtifact.model_validate(
        {**payload, "review_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


class BaselineAuthoringManifest(_StrictFrozenModel):
    schema_version: Literal[2, 3] = 2
    baseline_kind: Literal["spec", "llm_static"]
    formal_eligible: bool
    formal_ineligibility_reason: (
        Literal[
            "unverified_in_memory_authoring_input",
            "python_transport_is_not_mechanically_isolated",
        ]
        | None
    )
    authoring_input_verification: Literal["provisional", "externally_verified"]
    authoring_input_sha256: Sha256
    authoring_input_file_sha256: Sha256
    construction_identity_sha256: Sha256
    construction_identity_policy: Literal["spec-content-v1", "reviewed-draft-v1"]
    runtime_binding_policy: Literal["registry-runtime-v1", "registry-runtime-v2"]
    prompt_sha256: Sha256
    request_sha256: Sha256 | None
    request_file_sha256: Sha256 | None
    response_file_sha256: Sha256 | None
    pre_review_draft_file_sha256: Sha256 | None
    draft_text_sha256: Sha256 | None
    provider: Provider | None
    endpoint: str | None
    requested_model: str | None
    response_model: str | None
    provider_request_id: str | None
    finish_reason: str | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_microusd: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    successful_call_count: int = Field(ge=0, le=1)
    decoding_sha256: Sha256 | None
    price_schedule_id: str | None
    price_schedule_file_sha256: Sha256 | None
    public_source_lock_file_sha256s: tuple[Sha256, ...]
    isolation_attestation_sha256: Sha256 | None
    isolation_attestation_file_sha256: Sha256 | None
    human_review_sha256: Sha256 | None
    human_review_file_sha256: Sha256 | None
    human_review_minutes: int = Field(ge=0)
    compiler_id: Literal["skillchain.static-authoring"]
    compiler_version: Literal["3.0.0", "4.0.0"]
    tool_registry_sha256: Sha256
    tool_registry_runtime_sha256: Sha256
    bank_sha256: Sha256
    bank_file_sha256: Sha256
    skill_markdown_file_sha256s: tuple[Sha256, ...] = Field(min_length=1)
    manifest_sha256: Sha256

    @field_validator(
        "public_source_lock_file_sha256s",
        "skill_markdown_file_sha256s",
        mode="before",
    )
    @classmethod
    def coerce_hash_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_generation = (
            (2, "3.0.0", "registry-runtime-v1")
            if self.schema_version == 2
            else (3, "4.0.0", "registry-runtime-v2")
        )
        if (
            self.schema_version,
            self.compiler_version,
            self.runtime_binding_policy,
        ) != expected_generation:
            raise ValueError("manifest schema/compiler/runtime generations differ")
        if self.baseline_kind == "spec":
            llm_fields = (
                self.request_sha256,
                self.request_file_sha256,
                self.response_file_sha256,
                self.pre_review_draft_file_sha256,
                self.draft_text_sha256,
                self.provider,
                self.endpoint,
                self.requested_model,
                self.response_model,
                self.provider_request_id,
                self.finish_reason,
                self.decoding_sha256,
                self.price_schedule_id,
                self.isolation_attestation_sha256,
                self.isolation_attestation_file_sha256,
                self.human_review_sha256,
                self.human_review_file_sha256,
            )
            if any(item is not None for item in llm_fields) or any(
                (
                    self.input_tokens,
                    self.output_tokens,
                    self.cost_microusd,
                    self.latency_ms,
                    self.successful_call_count,
                    self.human_review_minutes,
                )
            ):
                raise ValueError("SpecBaseline must contain no LLM/review activity")
            if not self.formal_eligible and self.formal_ineligibility_reason != (
                "unverified_in_memory_authoring_input"
            ):
                raise ValueError("provisional SpecBaseline has an invalid reason")
        else:
            if self.successful_call_count != 1:
                raise ValueError("LLMStatic must record exactly one successful call")
            if self.formal_eligible:
                if self.formal_ineligibility_reason is not None:
                    raise ValueError(
                        "formal LLMStatic cannot have an ineligibility reason"
                    )
            elif self.formal_ineligibility_reason != (
                "python_transport_is_not_mechanically_isolated"
            ):
                raise ValueError("LLMStatic has an invalid isolation reason")
            required_llm_fields = (
                self.request_sha256,
                self.request_file_sha256,
                self.response_file_sha256,
                self.pre_review_draft_file_sha256,
                self.draft_text_sha256,
                self.provider,
                self.endpoint,
                self.requested_model,
                self.response_model,
                self.provider_request_id,
                self.finish_reason,
                self.decoding_sha256,
                self.price_schedule_id,
                self.isolation_attestation_sha256,
                self.isolation_attestation_file_sha256,
                self.human_review_sha256,
                self.human_review_file_sha256,
            )
            if any(item is None for item in required_llm_fields):
                raise ValueError("LLMStatic must bind every call/review audit field")
            if self.input_tokens <= 0 or self.output_tokens <= 0:
                raise ValueError("LLMStatic token usage must be positive")
            if self.human_review_minutes <= 0:
                raise ValueError("LLMStatic must record positive human review time")
        expected_policy = (
            "spec-content-v1" if self.baseline_kind == "spec" else "reviewed-draft-v1"
        )
        if self.construction_identity_policy != expected_policy:
            raise ValueError("manifest construction identity policy mismatch")
        if self.formal_eligible and (
            self.authoring_input_verification != "externally_verified"
            or self.price_schedule_file_sha256 is None
        ):
            raise ValueError("formal baseline lacks external verification locks")
        if self.authoring_input_verification == "externally_verified":
            if self.price_schedule_file_sha256 is None:
                raise ValueError("verified input lacks price-schedule lock")
        elif self.price_schedule_file_sha256 is not None or (
            self.public_source_lock_file_sha256s
        ):
            raise ValueError("provisional input must not contain external lock claims")
        if self.formal_eligible == (self.formal_ineligibility_reason is not None):
            raise ValueError("formal eligibility reason is inconsistent")
        if self.formal_ineligibility_reason is not None:
            _nonblank(self.formal_ineligibility_reason, "formal_ineligibility_reason")
        if self.manifest_sha256 != _digest_without(
            self.model_dump(mode="json"), "manifest_sha256"
        ):
            raise ValueError("manifest_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_model_bytes(self)


@dataclass(frozen=True)
class AuthoringRunResult:
    bank: StaticBankArtifact
    manifest: BaselineAuthoringManifest
    runtime_skills: tuple[Skill, ...]


@dataclass(frozen=True)
class VerifiedAuthoringRun:
    result: AuthoringRunResult
    authoring_input: VerifiedAuthoringInput
    manifest_file_sha256: str


def _manifest(payload: dict[str, object]) -> BaselineAuthoringManifest:
    return BaselineAuthoringManifest.model_validate(
        {**payload, "manifest_sha256": sha256_bytes(canonical_json_bytes(payload))}
    )


def run_spec_baseline(
    value: AuthoringInput | VerifiedAuthoringInput,
    artifact_dir: str | Path,
    *,
    require_formal_eligibility: bool = True,
) -> AuthoringRunResult:
    """Compile only embedded specs; model, budgets, and sources do not affect Bank."""

    if isinstance(value, VerifiedAuthoringInput):
        _require_formal_runtime_registry(value.registry)
        verified = _refresh_verified_authoring_input(value)
        input_value = verified.value
        formal_eligible = True
        formal_reason = None
    else:
        input_value = _revalidate_input(value)
        if require_formal_eligibility:
            raise AuthoringContractError(
                "formal SpecBaseline requires VerifiedAuthoringInput"
            )
        formal_eligible = False
        formal_reason = "unverified_in_memory_authoring_input"
    bundle = build_spec_draft_bundle(input_value)
    bank = _compile(
        input_value,
        bundle,
        baseline_kind="spec",
        tool_registry_runtime_sha256=(
            verified.registry.registry_runtime_sha256
            if isinstance(value, VerifiedAuthoringInput)
            else None
        ),
    )
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_bytes = input_value.canonical_bytes()
    bank_bytes = bank.canonical_bytes()
    skill_markdown_bytes = tuple(render_skill_markdown(item) for item in bank.skills)
    source_lock_hashes = tuple(
        item.source_lock_file_sha256
        for item in input_value.public_sources
        if item.source_lock_file_sha256 is not None
    )
    manifest_payload = {
        "schema_version": bank.schema_version,
        "baseline_kind": "spec",
        "formal_eligible": formal_eligible,
        "formal_ineligibility_reason": formal_reason,
        "authoring_input_verification": input_value.external_verification,
        "authoring_input_sha256": input_value.input_sha256,
        "authoring_input_file_sha256": sha256_bytes(input_bytes),
        "construction_identity_sha256": bank.construction_identity_sha256,
        "construction_identity_policy": bank.construction_identity_policy,
        "runtime_binding_policy": bank.runtime_binding_policy,
        "prompt_sha256": input_value.prompt.prompt_sha256,
        "request_sha256": None,
        "request_file_sha256": None,
        "response_file_sha256": None,
        "pre_review_draft_file_sha256": None,
        "draft_text_sha256": None,
        "provider": None,
        "endpoint": None,
        "requested_model": None,
        "response_model": None,
        "provider_request_id": None,
        "finish_reason": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_microusd": 0,
        "latency_ms": 0,
        "successful_call_count": 0,
        "decoding_sha256": None,
        "price_schedule_id": None,
        "price_schedule_file_sha256": input_value.price_schedule_file_sha256,
        "public_source_lock_file_sha256s": list(source_lock_hashes),
        "isolation_attestation_sha256": None,
        "isolation_attestation_file_sha256": None,
        "human_review_sha256": None,
        "human_review_file_sha256": None,
        "human_review_minutes": 0,
        "compiler_id": input_value.compiler.compiler_id,
        "compiler_version": input_value.compiler.compiler_version,
        "tool_registry_sha256": input_value.tool_registry.identity_sha256,
        "tool_registry_runtime_sha256": bank.tool_registry_runtime_sha256,
        "bank_sha256": bank.bank_sha256,
        "bank_file_sha256": sha256_bytes(bank_bytes),
        "skill_markdown_file_sha256s": [
            sha256_bytes(content) for content in skill_markdown_bytes
        ],
    }
    manifest = _manifest(manifest_payload)
    if isinstance(value, VerifiedAuthoringInput):
        final_verified = _refresh_verified_authoring_input(verified)
        if final_verified.value != input_value:
            raise AuthoringContractError(
                "verified authoring input changed during SpecBaseline construction"
            )
    atomic_create_file(root / "authoring-input.json", input_bytes)
    atomic_create_file(root / "static-bank.json", bank_bytes)
    publish_skill_markdown_bank(bank, root)
    atomic_create_file(root / "authoring-manifest.json", manifest.canonical_bytes())
    return AuthoringRunResult(bank, manifest, bank.runtime_skills())


def _verify_invocation_artifacts(
    value: AuthoringInput,
    invocation: AuthoringInvocation,
    root: Path,
) -> tuple[
    CanonicalAuthoringRequest,
    LLMResponse,
    AuthoringDraftBundle,
    IsolationEvidence,
]:
    """Re-read the persisted audit instead of trusting a forgeable dataclass."""

    try:
        input_bytes = read_stable_regular_file(
            root / "authoring-input.json", label="saved authoring input"
        )
        request_bytes = read_stable_regular_file(
            root / "authoring-request.json", label="saved authoring request"
        )
        response_bytes = read_stable_regular_file(
            root / "authoring-response.json", label="saved authoring response"
        )
        draft_bytes = read_stable_regular_file(
            root / "pre-review-draft.json", label="saved pre-review draft"
        )
        attestation_bytes = read_stable_regular_file(
            root / "isolation-attestation.json", label="saved isolation attestation"
        )
        request_raw = parse_canonical_json(request_bytes, label="saved request")
        response_raw = parse_canonical_json(response_bytes, label="saved response")
        draft_raw = parse_canonical_json(draft_bytes, label="saved pre-review draft")
        attestation_raw = parse_canonical_json(
            attestation_bytes, label="saved isolation attestation"
        )
        if not all(
            isinstance(item, dict)
            for item in (request_raw, response_raw, draft_raw, attestation_raw)
        ):
            raise AuthoringContractError("saved invocation artifacts must be objects")
        request = CanonicalAuthoringRequest.model_validate(request_raw, strict=True)
        # LLMResponse's canonical JSON represents its tuple of tool calls as an
        # array; its own contract validators still reject extra/identity drift.
        response = LLMResponse.model_validate(response_raw)
        draft = AuthoringDraftBundle.model_validate(draft_raw, strict=True)
        attestation = _parse_isolation_evidence(attestation_raw)
    except (ArtifactFormatError, ValidationError) as error:
        raise AuthoringContractError("saved invocation audit is invalid") from error
    if input_bytes != value.canonical_bytes():
        raise AuthoringContractError("saved authoring input changed before finalize")
    if (
        sha256_bytes(request_bytes) != invocation.request_file_sha256
        or sha256_bytes(response_bytes) != invocation.response_file_sha256
        or sha256_bytes(draft_bytes) != invocation.pre_review_draft_file_sha256
        or sha256_bytes(attestation_bytes)
        != invocation.isolation_attestation_file_sha256
        or request != invocation.request
        or response != invocation.response
        or draft != invocation.pre_review_draft
        or attestation != invocation.isolation_attestation
    ):
        raise AuthoringContractError("saved invocation audit does not match invocation")
    if request.authoring_input != value or request.request_sha256 != (
        invocation.request.request_sha256
    ):
        raise AuthoringContractError("saved request input identity mismatch")
    if isinstance(attestation, FormalIsolationAttestation) and (
        attestation.request_sha256 != request.request_sha256
        or attestation.response_file_sha256 != sha256_bytes(response_bytes)
    ):
        raise AuthoringContractError("formal isolation receipt binding mismatch")
    if isinstance(attestation, FormalIsolationAttestation) != (
        invocation._formal_token is _FORMAL_AUTHORING_INVOCATION_TOKEN
    ):
        raise AuthoringContractError("authoring invocation authority mismatch")
    if (
        response.provider != value.model.provider
        or response.endpoint != value.model.endpoint
        or response.requested_model != value.model.model
        or response.response_model != value.model.model
    ):
        raise AuthoringContractError("saved response no longer satisfies call contract")
    if not authoring_response_envelope_is_complete(value.decoding, response):
        raise AuthoringContractError("saved response no longer satisfies call contract")
    cost = _calculated_cost(value, response)
    if cost != invocation.cost_microusd:
        raise AuthoringContractError("saved response cost does not match invocation")
    usage = response.usage
    if usage.input_tokens <= 0 or usage.output_tokens <= 0:
        raise AuthoringContractError("saved response usage must be positive")
    if (
        usage.input_tokens > value.budgets.max_input_tokens
        or usage.output_tokens > value.budgets.max_output_tokens
        or usage.total_tokens > value.budgets.max_total_tokens
        or cost > value.budgets.max_cost_microusd
    ):
        raise AuthoringContractError("saved response exceeds frozen budget")
    normalized = _normalize_authoring_response(
        response,
        decoding=value.decoding,
        authoring_input=value,
    )
    if normalized != draft:
        raise AuthoringContractError(
            "saved response no longer matches the pre-review draft"
        )
    return request, response, draft, attestation


def finalize_llm_static(
    value: AuthoringInput | VerifiedAuthoringInput,
    invocation: AuthoringInvocation,
    review: HumanReviewArtifact,
    artifact_dir: str | Path,
) -> AuthoringRunResult:
    """Compile the reviewed draft and persist review, Bank, and full manifest."""

    verified_input: VerifiedAuthoringInput | None = None
    if isinstance(value, VerifiedAuthoringInput):
        verified_input = _refresh_verified_authoring_input(value)
        input_value = verified_input.value
    else:
        input_value = _revalidate_input(value)
    try:
        review = HumanReviewArtifact.model_validate(
            review.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise AuthoringContractError("human review artifact is invalid") from error
    if invocation.authoring_input_sha256 != input_value.input_sha256:
        raise AuthoringContractError("invocation input identity mismatch")
    try:
        raw_attestation = invocation.isolation_attestation.model_dump(mode="python")
        attestation = _parse_isolation_evidence(raw_attestation)
    except (AttributeError, ValidationError) as error:
        raise AuthoringContractError("isolation attestation is invalid") from error
    root = Path(artifact_dir)
    request, response, pre_review, saved_attestation = _verify_invocation_artifacts(
        input_value, invocation, root
    )
    if saved_attestation != attestation:
        raise AuthoringContractError("saved attestation differs from invocation")
    formal_eligible = isinstance(attestation, FormalIsolationAttestation)
    if formal_eligible and (
        verified_input is None
        or invocation._formal_token is not _FORMAL_AUTHORING_INVOCATION_TOKEN
    ):
        raise AuthoringContractError(
            "formal LLMStatic finalize requires verified input and invocation authority"
        )
    if review.review_minutes > input_value.budgets.max_human_review_minutes:
        raise AuthoringContractError("human review budget exceeded")
    if review.pre_review_canonical_json != pre_review.canonical_bytes().decode("utf-8"):
        raise AuthoringContractError("human review pre-draft does not match invocation")
    post = review.post_review_bundle
    if post.authoring_input_sha256 != input_value.input_sha256:
        raise AuthoringContractError("post-review draft input identity mismatch")
    bank = _compile(
        input_value,
        post,
        baseline_kind="llm_static",
        tool_registry_runtime_sha256=(
            verified_input.registry.registry_runtime_sha256
            if verified_input is not None
            else None
        ),
    )
    input_bytes = input_value.canonical_bytes()
    decoding_sha = sha256_bytes(_canonical_model_bytes(input_value.decoding))
    review_bytes = review.canonical_bytes()
    bank_bytes = bank.canonical_bytes()
    skill_markdown_bytes = tuple(render_skill_markdown(item) for item in bank.skills)
    source_lock_hashes = tuple(
        item.source_lock_file_sha256
        for item in input_value.public_sources
        if item.source_lock_file_sha256 is not None
    )
    manifest_payload = {
        "schema_version": bank.schema_version,
        "baseline_kind": "llm_static",
        "formal_eligible": formal_eligible,
        "formal_ineligibility_reason": (
            None if formal_eligible else attestation.reason
        ),
        "authoring_input_verification": input_value.external_verification,
        "authoring_input_sha256": input_value.input_sha256,
        "authoring_input_file_sha256": sha256_bytes(input_bytes),
        "construction_identity_sha256": bank.construction_identity_sha256,
        "construction_identity_policy": bank.construction_identity_policy,
        "runtime_binding_policy": bank.runtime_binding_policy,
        "prompt_sha256": input_value.prompt.prompt_sha256,
        "request_sha256": request.request_sha256,
        "request_file_sha256": invocation.request_file_sha256,
        "response_file_sha256": invocation.response_file_sha256,
        "pre_review_draft_file_sha256": (invocation.pre_review_draft_file_sha256),
        # Field name is retained for manifest schema-v2 compatibility.  Under
        # forced_submission_function_v1 it binds the raw function arguments,
        # not the necessarily-empty assistant text channel.
        "draft_text_sha256": sha256_bytes(
            _authoring_response_payload_bytes(input_value.decoding, response)
        ),
        "provider": response.provider,
        "endpoint": response.endpoint,
        "requested_model": response.requested_model,
        "response_model": response.response_model,
        "provider_request_id": response.request_id,
        "finish_reason": response.finish_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_microusd": invocation.cost_microusd,
        "latency_ms": response.latency_ms,
        "successful_call_count": 1,
        "decoding_sha256": decoding_sha,
        "price_schedule_id": input_value.price_schedule.schedule_id,
        "price_schedule_file_sha256": input_value.price_schedule_file_sha256,
        "public_source_lock_file_sha256s": list(source_lock_hashes),
        "isolation_attestation_sha256": (attestation.attestation_sha256),
        "isolation_attestation_file_sha256": (
            invocation.isolation_attestation_file_sha256
        ),
        "human_review_sha256": review.review_sha256,
        "human_review_file_sha256": sha256_bytes(review_bytes),
        "human_review_minutes": review.review_minutes,
        "compiler_id": input_value.compiler.compiler_id,
        "compiler_version": input_value.compiler.compiler_version,
        "tool_registry_sha256": input_value.tool_registry.identity_sha256,
        "tool_registry_runtime_sha256": bank.tool_registry_runtime_sha256,
        "bank_sha256": bank.bank_sha256,
        "bank_file_sha256": sha256_bytes(bank_bytes),
        "skill_markdown_file_sha256s": [
            sha256_bytes(content) for content in skill_markdown_bytes
        ],
    }
    manifest = _manifest(manifest_payload)
    if verified_input is not None:
        refreshed = _refresh_verified_authoring_input(verified_input)
        if refreshed.value != input_value:
            raise AuthoringContractError(
                "verified authoring input changed during LLMStatic finalize"
            )
    atomic_create_file(root / "human-review.json", review_bytes)
    atomic_create_file(root / "static-bank.json", bank_bytes)
    publish_skill_markdown_bank(bank, root)
    atomic_create_file(root / "authoring-manifest.json", manifest.canonical_bytes())
    return AuthoringRunResult(bank, manifest, bank.runtime_skills())


def _load_canonical_object(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str,
    max_bytes: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    try:
        content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
        if sha256_bytes(content) != expected_file_sha256:
            raise AuthoringContractError(f"{label} file digest mismatch")
        raw = parse_canonical_json(content, label=label)
        if not isinstance(raw, dict):
            raise AuthoringContractError(f"{label} must contain an object")
    except ArtifactFormatError as error:
        raise AuthoringContractError(f"{label} is unsafe or non-canonical") from error
    return content, raw


def _verify_skill_markdown_files(
    root: Path,
    bank: StaticBankArtifact,
    expected_file_sha256s: tuple[str, ...],
) -> None:
    if len(expected_file_sha256s) != len(bank.skills):
        raise AuthoringContractError("SKILL.md digest count does not match Bank")
    for skill, expected_digest in zip(bank.skills, expected_file_sha256s, strict=True):
        loaded = load_skill_markdown(
            root / "skills" / skill.slug / "SKILL.md",
            expected_file_sha256=expected_digest,
        )
        if loaded != skill:
            raise AuthoringContractError("SKILL.md disagrees with static Bank")


def load_verified_authoring_run(
    artifact_dir: str | Path,
    *,
    expected_manifest_file_sha256: str,
    registry: ToolRegistry,
    expected_taxonomy_sha256: str,
    expected_task_specification_sha256: str,
    expected_tool_registry_sha256: str,
    expected_tool_registry_runtime_sha256: str,
    public_sources: tuple[VerifiedPublicSourceMaterial, ...],
    price_schedule: VerifiedPriceSchedule,
    sandbox_profile: VerifiedAuthoringSandboxProfile | None = None,
) -> VerifiedAuthoringRun:
    """Reload and cross-check every artifact from one externally pinned manifest."""

    _require_formal_runtime_registry(registry)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_file_sha256):
        raise AuthoringContractError("expected authoring-manifest SHA-256 is invalid")
    root = Path(artifact_dir)
    manifest_bytes, manifest_raw = _load_canonical_object(
        root / "authoring-manifest.json",
        label="authoring manifest",
        expected_file_sha256=expected_manifest_file_sha256,
    )
    try:
        manifest = BaselineAuthoringManifest.model_validate(manifest_raw, strict=True)
    except ValidationError as error:
        raise AuthoringContractError("authoring manifest violates schema") from error
    verified_input = load_verified_authoring_input(
        root / "authoring-input.json",
        expected_file_sha256=manifest.authoring_input_file_sha256,
        registry=registry,
        expected_taxonomy_sha256=expected_taxonomy_sha256,
        expected_task_specification_sha256=expected_task_specification_sha256,
        expected_tool_registry_sha256=expected_tool_registry_sha256,
        expected_tool_registry_runtime_sha256=(expected_tool_registry_runtime_sha256),
        public_sources=public_sources,
        price_schedule=price_schedule,
    )
    value = verified_input.value
    expected_source_locks = tuple(
        item.source_lock_file_sha256 for item in value.public_sources
    )
    if (
        manifest.authoring_input_verification != "externally_verified"
        or manifest.authoring_input_sha256 != value.input_sha256
        or manifest.prompt_sha256 != value.prompt.prompt_sha256
        or manifest.price_schedule_file_sha256 != value.price_schedule_file_sha256
        or manifest.public_source_lock_file_sha256s != expected_source_locks
        or manifest.compiler_id != value.compiler.compiler_id
        or manifest.compiler_version != value.compiler.compiler_version
        or manifest.tool_registry_sha256 != registry.registry_sha256
        or manifest.tool_registry_runtime_sha256 != registry.registry_runtime_sha256
    ):
        raise AuthoringContractError("authoring manifest input/lock binding mismatch")
    bank = load_static_bank(
        root / "static-bank.json",
        expected_file_sha256=manifest.bank_file_sha256,
        registry=registry,
    )
    if (
        bank.baseline_kind != manifest.baseline_kind
        or bank.bank_sha256 != manifest.bank_sha256
        or bank.construction_identity_sha256 != manifest.construction_identity_sha256
        or bank.construction_identity_policy != manifest.construction_identity_policy
        or bank.runtime_binding_policy != manifest.runtime_binding_policy
    ):
        raise AuthoringContractError("authoring manifest Bank binding mismatch")
    _verify_skill_markdown_files(root, bank, manifest.skill_markdown_file_sha256s)
    if manifest.baseline_kind == "spec":
        rebuilt = _compile(
            value,
            build_spec_draft_bundle(value),
            baseline_kind="spec",
            tool_registry_runtime_sha256=registry.registry_runtime_sha256,
        )
        if rebuilt.canonical_bytes() != bank.canonical_bytes():
            raise AuthoringContractError("SpecBaseline Bank is not reproducible")
    else:
        _verify_loaded_llm_static_run(
            root, value, bank, manifest, sandbox_profile=sandbox_profile
        )
    result = AuthoringRunResult(bank, manifest, bank.runtime_skills())
    # Re-read trust roots and all derived files after cross-checking to narrow
    # replacement races; the caller-pinned manifest digest remains authoritative.
    try:
        final_manifest = read_stable_regular_file(
            root / "authoring-manifest.json",
            label="authoring manifest final recheck",
        )
    except ArtifactFormatError as error:
        raise AuthoringContractError(
            "authoring manifest is unsafe during final recheck"
        ) from error
    if (
        final_manifest != manifest_bytes
        or sha256_bytes(final_manifest) != expected_manifest_file_sha256
    ):
        raise AuthoringContractError("authoring manifest changed during verification")
    final_input = _refresh_verified_authoring_input(verified_input)
    if final_input.value != value:
        raise AuthoringContractError("authoring input changed during verification")
    final_bank = load_static_bank(
        root / "static-bank.json",
        expected_file_sha256=manifest.bank_file_sha256,
        registry=registry,
    )
    if final_bank != bank:
        raise AuthoringContractError("static Bank changed during verification")
    _verify_skill_markdown_files(root, final_bank, manifest.skill_markdown_file_sha256s)
    if manifest.baseline_kind == "llm_static":
        _verify_loaded_llm_static_run(
            root,
            final_input.value,
            final_bank,
            manifest,
            sandbox_profile=sandbox_profile,
        )
    return VerifiedAuthoringRun(
        result=result,
        authoring_input=verified_input,
        manifest_file_sha256=expected_manifest_file_sha256,
    )


def _verify_loaded_llm_static_run(
    root: Path,
    value: AuthoringInput,
    bank: StaticBankArtifact,
    manifest: BaselineAuthoringManifest,
    *,
    sandbox_profile: VerifiedAuthoringSandboxProfile | None,
) -> None:
    required_hashes = (
        manifest.request_file_sha256,
        manifest.response_file_sha256,
        manifest.pre_review_draft_file_sha256,
        manifest.isolation_attestation_file_sha256,
        manifest.human_review_file_sha256,
    )
    if any(item is None for item in required_hashes):
        raise AuthoringContractError("LLMStatic manifest omits an audit file digest")
    request_bytes, request_raw = _load_canonical_object(
        root / "authoring-request.json",
        label="authoring request",
        expected_file_sha256=manifest.request_file_sha256,
    )
    response_bytes, response_raw = _load_canonical_object(
        root / "authoring-response.json",
        label="authoring response",
        expected_file_sha256=manifest.response_file_sha256,
        max_bytes=_MAX_RESPONSE_BYTES,
    )
    draft_bytes, draft_raw = _load_canonical_object(
        root / "pre-review-draft.json",
        label="pre-review draft",
        expected_file_sha256=manifest.pre_review_draft_file_sha256,
    )
    attestation_bytes, attestation_raw = _load_canonical_object(
        root / "isolation-attestation.json",
        label="isolation attestation",
        expected_file_sha256=manifest.isolation_attestation_file_sha256,
    )
    review_bytes, review_raw = _load_canonical_object(
        root / "human-review.json",
        label="human review",
        expected_file_sha256=manifest.human_review_file_sha256,
    )
    try:
        request = CanonicalAuthoringRequest.model_validate(request_raw, strict=True)
        response = LLMResponse.model_validate(response_raw)
        draft = AuthoringDraftBundle.model_validate(draft_raw, strict=True)
        attestation = _parse_isolation_evidence(attestation_raw)
        review = HumanReviewArtifact.model_validate(review_raw, strict=True)
    except ValidationError as error:
        raise AuthoringContractError("LLMStatic audit violates schema") from error
    if (
        request.authoring_input != value
        or request.request_sha256 != manifest.request_sha256
        or request_bytes != request.canonical_bytes()
        or response_bytes != _canonical_model_bytes(response)
        or draft_bytes != draft.canonical_bytes()
        or attestation_bytes != _canonical_model_bytes(attestation)
        or review_bytes != review.canonical_bytes()
    ):
        raise AuthoringContractError("LLMStatic canonical audit cross-check failed")
    usage = response.usage
    cost = _calculated_cost(value, response)
    formal_attestation = isinstance(attestation, FormalIsolationAttestation)
    if formal_attestation:
        if sandbox_profile is None:
            raise AuthoringContractError(
                "formal LLMStatic reload requires the verified sandbox profile"
            )
        refreshed_profile = load_verified_authoring_sandbox_profile(
            sandbox_profile.path,
            expected_file_sha256=sandbox_profile.file_sha256,
        )
        if (
            refreshed_profile.value != sandbox_profile.value
            or attestation.sandbox_profile_file_sha256 != sandbox_profile.file_sha256
            or attestation.sandbox_profile_sha256
            != sandbox_profile.value.profile_sha256
            or attestation.engine_binary_sha256
            != sandbox_profile.value.engine_binary_sha256
            or attestation.image_digest != sandbox_profile.value.image_digest
            or attestation.network_policy_sha256
            != sandbox_profile.value.network_policy_sha256
            or attestation.request_sha256 != request.request_sha256
            or attestation.response_file_sha256 != sha256_bytes(response_bytes)
        ):
            raise AuthoringContractError("formal sandbox receipt/profile mismatch")
    elif sandbox_profile is not None:
        raise AuthoringContractError(
            "diagnostic LLMStatic cannot consume a formal sandbox profile"
        )
    if (
        response.provider != value.model.provider
        or response.endpoint != value.model.endpoint
        or response.requested_model != value.model.model
        or response.response_model != value.model.model
        or response.request_id == request.request_sha256
        or not authoring_response_envelope_is_complete(value.decoding, response)
        or usage.input_tokens <= 0
        or usage.output_tokens <= 0
        or usage.input_tokens > value.budgets.max_input_tokens
        or usage.output_tokens > value.budgets.max_output_tokens
        or usage.total_tokens > value.budgets.max_total_tokens
        or cost > value.budgets.max_cost_microusd
    ):
        raise AuthoringContractError("LLMStatic response violates frozen call contract")
    normalized_response = _normalize_authoring_response(
        response,
        decoding=value.decoding,
        authoring_input=value,
    )
    if (
        normalized_response != draft
        or draft.authoring_input_sha256 != value.input_sha256
        or review.pre_review_canonical_json != draft_bytes.decode("utf-8")
        or review.review_minutes > value.budgets.max_human_review_minutes
    ):
        raise AuthoringContractError("LLMStatic draft/review binding mismatch")
    rebuilt = _compile(
        value,
        review.post_review_bundle,
        baseline_kind="llm_static",
        tool_registry_runtime_sha256=bank.tool_registry_runtime_sha256,
    )
    if rebuilt.canonical_bytes() != bank.canonical_bytes():
        raise AuthoringContractError("LLMStatic reviewed Bank is not reproducible")
    if (
        manifest.draft_text_sha256
        != sha256_bytes(_authoring_response_payload_bytes(value.decoding, response))
        or manifest.provider != response.provider
        or manifest.endpoint != response.endpoint
        or manifest.requested_model != response.requested_model
        or manifest.response_model != response.response_model
        or manifest.provider_request_id != response.request_id
        or manifest.finish_reason != response.finish_reason
        or manifest.input_tokens != usage.input_tokens
        or manifest.output_tokens != usage.output_tokens
        or manifest.cost_microusd != cost
        or manifest.latency_ms != response.latency_ms
        or manifest.decoding_sha256
        != sha256_bytes(_canonical_model_bytes(value.decoding))
        or manifest.price_schedule_id != value.price_schedule.schedule_id
        or manifest.isolation_attestation_sha256 != attestation.attestation_sha256
        or manifest.human_review_sha256 != review.review_sha256
        or manifest.human_review_minutes != review.review_minutes
        or manifest.formal_eligible != formal_attestation
        or manifest.formal_ineligibility_reason
        != (None if formal_attestation else attestation.reason)
    ):
        raise AuthoringContractError("LLMStatic manifest audit values mismatch")


__all__ = [
    "AUTHORING_CONTENT_SCHEMA_NAME",
    "AUTHORING_SUBMISSION_TOOL_NAME",
    "CANONICAL_AUTHORING_RESPONSE_FORMAT",
    "PROVIDER_JSON_CONTENT_RESPONSE_FORMAT",
    "FORCED_SUBMISSION_RESPONSE_FORMAT",
    "AuthoringBudgets",
    "AuthoringContentOutputContract",
    "AuthoringContentPayload",
    "AuthoringContractError",
    "AuthoringDraftBundle",
    "AuthoringDraftPayload",
    "AuthoringInput",
    "AuthoringInvocation",
    "AuthoringOutputContract",
    "AuthoringPacket",
    "AuthoringRunResult",
    "VerifiedAuthoringRun",
    "BankCapabilityBinding",
    "BaselineAuthoringManifest",
    "CanonicalAuthoringRequest",
    "CanonicalSpecification",
    "CapabilityDraft",
    "CapabilityDraftPayload",
    "CapabilityAuthoringContent",
    "CompilerIdentity",
    "ControlledAuthoringGateway",
    "ControlledLLMTransport",
    "OfflineAuthoringReplayGateway",
    "DraftToolStep",
    "DraftToolStepContent",
    "FixedDecoding",
    "HumanReviewArtifact",
    "AuthoringSandboxProfile",
    "FormalContainerAuthoringGateway",
    "FormalIsolationAttestation",
    "IsolationAttestation",
    "ModelIdentity",
    "OutputCoverage",
    "PriceSchedule",
    "PromptIdentity",
    "PublicSourceMaterial",
    "PublicSourceLock",
    "ReferenceSkillBundle",
    "RuleCoverage",
    "ReviewChecklist",
    "StaticBankArtifact",
    "StrictSkillArtifact",
    "UnifiedChatTransport",
    "VerifiedAuthoringInput",
    "VerifiedAuthoringInvocationInput",
    "VerifiedAuthoringCallAuthorization",
    "VerifiedAuthoringSandboxProfile",
    "VerifiedPriceSchedule",
    "VerifiedPublicSourceMaterial",
    "build_authoring_draft_bundle",
    "build_authoring_content_output_contract",
    "build_canonical_authoring_request",
    "build_authoring_packet",
    "build_authoring_output_contract",
    "build_capability_draft",
    "build_human_review",
    "build_public_source_lock",
    "build_public_source_material",
    "build_reference_skill_bundle",
    "build_spec_draft_bundle",
    "authoring_response_envelope_is_complete",
    "authoring_content_json_schema",
    "authoring_submission_tool_definition",
    "finalize_llm_static",
    "in_process_test_attestation",
    "invoke_llm_static",
    "invoke_skill_operator",
    "load_authoring_packet",
    "load_skill_markdown",
    "load_static_bank",
    "load_verified_authoring_input",
    "load_verified_authoring_invocation_input",
    "load_verified_authoring_call_authorization",
    "load_verified_authoring_sandbox_profile",
    "load_verified_authoring_run",
    "load_verified_price_schedule",
    "load_verified_public_source_material",
    "normalize_authoring_draft_payload",
    "normalize_authoring_content_payload",
    "parse_skill_markdown",
    "publish_skill_markdown_bank",
    "render_skill_markdown",
    "run_spec_baseline",
    "run_llm_static",
    "runtime_skill_for_capability",
]
