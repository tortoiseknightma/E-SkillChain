"""Offline, fail-closed Wikimedia revision adapter for the mini KB.

Acquisition and formal ingestion are deliberately separate.  An acquisition
client may fetch one immutable MediaWiki revision and persist the raw response,
an HTTP receipt, and its deterministic canonical snapshot.  Formal ingestion
never uses the network: it re-derives every snapshot from the raw bytes and
requires independent digests for the selection lock, license evidence files,
and a canonical page-level human review ledger.

The two supported tracks are deliberately narrow:

* ``zh.wikipedia.org`` main-namespace text for encyclopedia entries;
* ``zh.wikibooks.org`` ``食谱/`` or ``食譜/`` main-namespace text for recipe
  entries.

No LLM-produced text is accepted as factual evidence and an automated model
cannot be the approving reviewer.  Every KB entry is an exact character span
of one fixed wikitext revision and cites its page id, revision id, span, and
span hash.  Published bundles are re-parsed canonically and formal consumers
must load them through another externally supplied bundle digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Annotated, Any, Literal, Mapping, Self
from urllib.parse import parse_qs, urlencode, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data._kb_source_lock import (
    SourceFileSnapshot,
    prepare_formal_output_parent,
    publish_staged_file_create_only,
    snapshot_regular_file,
    verify_source_snapshot,
)
from skillchain.data.kb_catalog import KBEntryV2, canonical_jsonl_bytes
from skillchain.synthesis.store import atomic_publish_new_directory

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MediaWikiSha1 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
KBKind = Literal["encyclopedia", "recipe"]
ProjectDomain = Literal["zh.wikipedia.org", "zh.wikibooks.org"]
ReviewerKind = Literal["human", "publisher"]

ADAPTER_ID = "wikimedia-revision-api-v1"
SELECTION_POLICY_VERSION = "wikimedia-mini-selection-v1"
ACQUISITION_CONTRACT = "mediawiki-revision-response-v1"
LICENSE_ID = "CC-BY-SA-4.0"
LICENSE_URI = "https://creativecommons.org/licenses/by-sa/4.0/"
TERMS_REVISION_URI = (
    "https://foundation.wikimedia.org/w/index.php?"
    "title=Policy:Terms_of_Use/en&oldid=554852"
)
_PROJECT_BY_KIND: dict[KBKind, ProjectDomain] = {
    "encyclopedia": "zh.wikipedia.org",
    "recipe": "zh.wikibooks.org",
}
_EXPECTED_POLICY_URI: dict[ProjectDomain, str] = {
    "zh.wikipedia.org": TERMS_REVISION_URI,
    "zh.wikibooks.org": TERMS_REVISION_URI,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SNAPSHOT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*\.json$")
_SAFE_EVIDENCE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*\.(?:html|json|txt)$")
_RECIPE_PREFIXES = ("\u98df\u8c31/", "\u98df\u8b5c/")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_REQUEST_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
VALIDATOR_ABSENT_REASON = "origin response did not include ETag or Last-Modified"
REVIEW_ATTESTATION = (
    "I performed this review without delegating the approval decision to an "
    "automated model."
)
_LLM_REVIEWER = re.compile(
    r"(?:^|[-_./ ])(?:llm|gpt|chatgpt|claude|gemini|codex|fable|model|"
    r"openai|anthropic|copilot|assistant|agent|bot|automation|machine)"
    r"(?:$|[-_./ ])",
    flags=re.IGNORECASE,
)


class WikimediaKBError(ValueError):
    """A Wikimedia acquisition snapshot or formal selection is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonblank(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned


def _parse_aware_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field_name} must be ISO-8601") from error
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _absolute_https_uri(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{field_name} must be a credential-free HTTPS URI")
    return value


def _safe_header_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    value = _nonblank(value, field_name)
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{field_name} must be one HTTP header value")
    return value


def _safe_root_file(value: str, field_name: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"{field_name} must be one safe root-level file name")
    return value


def canonical_json_document(value: BaseModel | dict[str, Any]) -> bytes:
    """Return the only accepted on-disk encoding for locks and snapshots."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def revision_api_uri(project_domain: ProjectDomain, revision_id: int) -> str:
    """Return the canonical acquisition URI for exactly one revision."""

    if revision_id <= 0:
        raise ValueError("revision_id must be positive")
    query = urlencode(
        (
            ("action", "query"),
            ("format", "json"),
            ("formatversion", "2"),
            ("prop", "revisions"),
            ("revids", str(revision_id)),
            ("rvprop", "ids|timestamp|sha1|contentmodel|content"),
            ("rvslots", "main"),
        )
    )
    return f"https://{project_domain}/w/api.php?{query}"


def revision_source_uri(project_domain: ProjectDomain, revision_id: int) -> str:
    return f"https://{project_domain}/w/index.php?oldid={revision_id}"


def page_history_uri(project_domain: ProjectDomain, page_id: int) -> str:
    return f"https://{project_domain}/w/index.php?curid={page_id}&action=history"


class WikimediaRevisionSnapshot(_StrictFrozenModel):
    """Canonical offline envelope produced from one official API response."""

    schema_version: Literal[1] = 1
    acquisition_contract: Literal["mediawiki-revision-response-v1"] = (
        ACQUISITION_CONTRACT
    )
    project_domain: ProjectDomain
    request_uri: str
    acquired_at: datetime
    page_id: int = Field(gt=0)
    namespace: Literal[0]
    title: str
    revision_id: int = Field(gt=0)
    parent_revision_id: int = Field(ge=0)
    revision_timestamp: datetime
    revision_sha1: MediaWikiSha1
    content_model: Literal["wikitext"]
    content_format: Literal["text/x-wiki"]
    content: str
    content_sha256: Sha256

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _nonblank(value, "title")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("acquired_at", "revision_timestamp", mode="before")
    @classmethod
    def parse_datetimes(cls, value: Any, info) -> datetime:
        return _parse_aware_datetime(value, info.field_name)

    @field_validator("request_uri")
    @classmethod
    def validate_request_uri(cls, value: str) -> str:
        return _absolute_https_uri(value, "request_uri")

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if self.request_uri != revision_api_uri(self.project_domain, self.revision_id):
            raise ValueError("request_uri is not the canonical fixed-revision API URI")
        expected_content_sha256 = hashlib.sha256(
            self.content.encode("utf-8")
        ).hexdigest()
        if self.content_sha256 != expected_content_sha256:
            raise ValueError("content_sha256 does not match revision content")
        if self.revision_sha1 != hashlib.sha1(self.content.encode("utf-8")).hexdigest():
            raise ValueError("revision_sha1 does not match revision content")
        if self.project_domain == "zh.wikibooks.org" and not self.title.startswith(
            _RECIPE_PREFIXES
        ):
            raise ValueError("recipe snapshots must be under an approved recipe prefix")
        return self


class WikimediaAcquisitionReceipt(_StrictFrozenModel):
    """Transport receipt binding raw API bytes to one canonical snapshot."""

    schema_version: Literal[1] = 1
    acquisition_contract: Literal["mediawiki-revision-response-v1"] = (
        ACQUISITION_CONTRACT
    )
    project_domain: ProjectDomain
    revision_id: int = Field(gt=0)
    request_uri: str
    final_uri: str
    http_status: Literal[200]
    http_date: str
    x_request_id: str
    validator_kind: Literal["etag", "last_modified", "etag_and_last_modified", "none"]
    etag: str | None = None
    last_modified: str | None = None
    validator_absent_reason: (
        Literal["origin response did not include ETag or Last-Modified"] | None
    ) = None
    acquired_at: datetime
    raw_response_file: str
    raw_response_bytes: int = Field(gt=0)
    raw_response_sha256: Sha256
    response_file: str
    response_sha256: Sha256

    @field_validator("request_uri", "final_uri")
    @classmethod
    def validate_uris(cls, value: str, info) -> str:
        return _absolute_https_uri(value, info.field_name)

    @field_validator(
        "http_date",
        "x_request_id",
        "etag",
        "last_modified",
        "validator_absent_reason",
    )
    @classmethod
    def validate_headers(cls, value: str | None, info) -> str | None:
        return _safe_header_value(value, info.field_name)

    @field_validator("http_date")
    @classmethod
    def validate_http_date(cls, value: str) -> str:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError) as error:
            raise ValueError("http_date must be one RFC 5322 HTTP Date") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("http_date must include a timezone")
        return value

    @field_validator("x_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not _REQUEST_ID_RE.fullmatch(value):
            raise ValueError("x_request_id must be the response UUID")
        return value.lower()

    @field_validator("acquired_at", mode="before")
    @classmethod
    def parse_acquired_at(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "acquired_at")

    @field_validator("raw_response_file", "response_file")
    @classmethod
    def validate_files(cls, value: str, info) -> str:
        return _safe_root_file(value, info.field_name, _SAFE_SNAPSHOT_NAME)

    @model_validator(mode="after")
    def validate_transport_binding(self) -> Self:
        expected_uri = revision_api_uri(self.project_domain, self.revision_id)
        if self.request_uri != expected_uri or self.final_uri != expected_uri:
            raise ValueError("receipt URIs must equal the canonical fixed-revision URI")
        expected_kind = (
            "etag_and_last_modified"
            if self.etag is not None and self.last_modified is not None
            else "etag"
            if self.etag is not None
            else "last_modified"
            if self.last_modified is not None
            else "none"
        )
        if self.validator_kind != expected_kind:
            raise ValueError("validator_kind does not match captured headers")
        if expected_kind == "none":
            if self.validator_absent_reason != VALIDATOR_ABSENT_REASON:
                raise ValueError(
                    "validator absence requires the exact auditable absence reason"
                )
        elif self.validator_absent_reason is not None:
            raise ValueError(
                "validator_absent_reason is forbidden when a validator was captured"
            )
        if self.raw_response_file == self.response_file:
            raise ValueError("raw response and canonical snapshot files must differ")
        return self


class WikimediaPermissionMatrix(_StrictFrozenModel):
    """Conservative CC BY-SA permissions and redistribution obligations."""

    local_research_allowed: Literal[True]
    cloud_processing_allowed: Literal[True]
    redistribution_allowed: Literal[True]
    public_demo_allowed: Literal[True]
    attribution_required: Literal[True]
    license_notice_required: Literal[True]
    indicate_changes_required: Literal[True]
    share_alike_on_adaptations: Literal[True]
    preserve_additional_attribution_notices: Literal[True]
    no_endorsement: Literal[True]


class WikimediaLicenseEvidence(_StrictFrozenModel):
    applicable_project: ProjectDomain
    license_id: Literal["CC-BY-SA-4.0"]
    license_uri: Literal["https://creativecommons.org/licenses/by-sa/4.0/"]
    policy_revision_uri: str
    evidence_file: str
    evidence_sha256: Sha256
    acquired_at: datetime
    http_status: Literal[200]
    final_uri: str
    etag: str | None = None
    last_modified: str | None = None

    @field_validator("policy_revision_uri", "final_uri")
    @classmethod
    def validate_policy_uri(cls, value: str) -> str:
        return _absolute_https_uri(value, "policy_revision_uri")

    @field_validator("evidence_file")
    @classmethod
    def validate_evidence_file(cls, value: str) -> str:
        return _safe_root_file(value, "evidence_file", _SAFE_EVIDENCE_NAME)

    @field_validator("acquired_at", mode="before")
    @classmethod
    def parse_acquired_at(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "acquired_at")

    @field_validator("etag", "last_modified")
    @classmethod
    def validate_headers(cls, value: str | None, info) -> str | None:
        return _safe_header_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_policy_binding(self) -> Self:
        if self.policy_revision_uri != _EXPECTED_POLICY_URI[self.applicable_project]:
            raise ValueError("license policy must use the reviewed immutable revision")
        if self.final_uri != self.policy_revision_uri:
            raise ValueError(
                "license evidence final URI must equal its policy revision"
            )
        if self.etag is None and self.last_modified is None:
            raise ValueError("license evidence requires ETag or Last-Modified")
        return self


class WikimediaSelectionEntry(_StrictFrozenModel):
    """Human-reviewed choice of an exact evidence span from one revision."""

    kind: KBKind
    project_domain: ProjectDomain
    raw_response_file: str
    raw_response_sha256: Sha256
    acquisition_receipt_file: str
    acquisition_receipt_sha256: Sha256
    response_file: str
    response_sha256: Sha256
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    revision_timestamp: datetime
    revision_sha1: MediaWikiSha1
    canonical_title: str
    evidence_start_char: int = Field(ge=0)
    evidence_end_char: int = Field(gt=0)
    evidence_sha256: Sha256
    source_uri: str
    history_uri: str
    license_id: Literal["CC-BY-SA-4.0"]
    license_uri: Literal["https://creativecommons.org/licenses/by-sa/4.0/"]
    attribution: str
    additional_attribution_notices: tuple[str, ...] = ()
    page_footer_history_and_talk_reviewed: Literal[True]
    unresolved_rights_notice: Literal[False]

    @field_validator("revision_timestamp", mode="before")
    @classmethod
    def parse_revision_timestamp(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "revision_timestamp")

    @field_validator("canonical_title", "attribution")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("source_uri", "history_uri")
    @classmethod
    def validate_source_uris(cls, value: str, info) -> str:
        return _absolute_https_uri(value, info.field_name)

    @field_validator("raw_response_file", "acquisition_receipt_file", "response_file")
    @classmethod
    def validate_response_file(cls, value: str, info) -> str:
        return _safe_root_file(value, info.field_name, _SAFE_SNAPSHOT_NAME)

    @field_validator("additional_attribution_notices", mode="before")
    @classmethod
    def coerce_notices(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("additional_attribution_notices")
    @classmethod
    def validate_notices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_nonblank(item, "attribution notice") for item in value)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("additional attribution notices must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_selection_bindings(self) -> Self:
        if self.project_domain != _PROJECT_BY_KIND[self.kind]:
            raise ValueError("kind does not match the approved Wikimedia project")
        if self.kind == "recipe" and not self.canonical_title.startswith(
            _RECIPE_PREFIXES
        ):
            raise ValueError("recipe title must use an approved recipe prefix")
        acquisition_files = {
            self.raw_response_file,
            self.acquisition_receipt_file,
            self.response_file,
        }
        if len(acquisition_files) != 3:
            raise ValueError("raw response, receipt, and snapshot files must differ")
        if self.evidence_end_char <= self.evidence_start_char:
            raise ValueError("evidence span must be non-empty")
        if self.source_uri != revision_source_uri(
            self.project_domain, self.revision_id
        ):
            raise ValueError("source_uri must identify the exact page revision")
        if self.history_uri != page_history_uri(self.project_domain, self.page_id):
            raise ValueError("history_uri must identify the exact page history")
        if self.source_uri not in self.attribution:
            raise ValueError("attribution must link to the selected page revision")
        if "contributors" not in self.attribution.casefold():
            raise ValueError("attribution must credit the page contributors")
        return self


class WikimediaReviewDecision(_StrictFrozenModel):
    """One human approval bound to the exact selected evidence characters."""

    kind: KBKind
    project_domain: ProjectDomain
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    raw_response_sha256: Sha256
    acquisition_receipt_sha256: Sha256
    response_sha256: Sha256
    evidence_start_char: int = Field(ge=0)
    evidence_end_char: int = Field(gt=0)
    evidence_sha256: Sha256
    decision: Literal["approved"]
    reviewer_kind: ReviewerKind
    reviewer_id: str
    reviewed_at: datetime
    evidence_reviewed_in_revision_context: Literal[True]
    page_rights_and_attribution_reviewed: Literal[True]
    automated_approval_delegated: Literal[False]
    decision_basis: str
    attestation: Literal[
        "I performed this review without delegating the approval decision to an "
        "automated model."
    ]

    @field_validator("reviewer_id", "decision_basis")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "reviewer_id" and _LLM_REVIEWER.search(value):
            raise ValueError("an LLM cannot approve a Wikimedia selection")
        return value

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def parse_reviewed_at(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "reviewed_at")

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.project_domain != _PROJECT_BY_KIND[self.kind]:
            raise ValueError("review decision kind does not match project")
        if self.evidence_end_char <= self.evidence_start_char:
            raise ValueError("review decision evidence span must be non-empty")
        return self


class WikimediaReviewLedger(_StrictFrozenModel):
    """Canonical, externally digest-locked page-level human review ledger."""

    schema_version: Literal[1] = 1
    selection_id: str
    decisions: tuple[WikimediaReviewDecision, ...]

    @field_validator("selection_id")
    @classmethod
    def validate_selection_id(cls, value: str) -> str:
        return _nonblank(value, "selection_id")

    @field_validator("decisions", mode="before")
    @classmethod
    def coerce_decisions(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_decisions(self) -> Self:
        expected = tuple(
            sorted(
                self.decisions,
                key=lambda item: (item.kind, item.page_id, item.revision_id),
            )
        )
        if self.decisions != expected:
            raise ValueError("review decisions must be canonically sorted")
        identities = [
            (item.project_domain, item.page_id, item.revision_id)
            for item in self.decisions
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("review ledger contains duplicate page revisions")
        return self


class WikimediaSelectionLock(_StrictFrozenModel):
    """External trust root for a 10--20 entry-per-track mini selection."""

    schema_version: Literal[1] = 1
    adapter_id: Literal["wikimedia-revision-api-v1"] = ADAPTER_ID
    policy_version: Literal["wikimedia-mini-selection-v1"] = SELECTION_POLICY_VERSION
    selection_id: str
    created_at: datetime
    permission_matrix: WikimediaPermissionMatrix
    license_evidence: tuple[WikimediaLicenseEvidence, ...]
    entries: tuple[WikimediaSelectionEntry, ...]
    reviewer_kind: ReviewerKind
    reviewer_id: str
    reviewed_at: datetime
    review_record_file: str
    review_record_uri: str
    review_record_sha256: Sha256

    @field_validator("selection_id", "reviewer_id")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("reviewer_id")
    @classmethod
    def reject_model_reviewer(cls, value: str) -> str:
        if _LLM_REVIEWER.search(value):
            raise ValueError("an LLM cannot attest page-level licensing")
        return value

    @field_validator("created_at", "reviewed_at", mode="before")
    @classmethod
    def parse_lock_datetimes(cls, value: Any, info) -> datetime:
        return _parse_aware_datetime(value, info.field_name)

    @field_validator("review_record_uri")
    @classmethod
    def validate_review_record_uri(cls, value: str) -> str:
        value = _nonblank(value, "review_record_uri")
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "urn", "ipfs"}:
            raise ValueError("review_record_uri must be immutable-capable")
        return value

    @field_validator("review_record_file")
    @classmethod
    def validate_review_record_file(cls, value: str) -> str:
        return _safe_root_file(value, "review_record_file", _SAFE_SNAPSHOT_NAME)

    @field_validator("license_evidence", "entries", mode="before")
    @classmethod
    def coerce_json_tuples(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_selection_set(self) -> Self:
        if self.review_record_sha256 not in self.review_record_uri.casefold():
            raise ValueError("review record URI must be content-addressed")
        evidence_projects = [item.applicable_project for item in self.license_evidence]
        if evidence_projects != sorted(_PROJECT_BY_KIND.values()):
            raise ValueError("license_evidence must contain both projects in order")
        ordering = tuple(
            sorted(
                self.entries,
                key=lambda item: (item.kind, item.page_id, item.revision_id),
            )
        )
        if self.entries != ordering:
            raise ValueError("selection entries must be canonically sorted")
        kind_counts = {
            kind: sum(entry.kind == kind for entry in self.entries)
            for kind in _PROJECT_BY_KIND
        }
        if any(count < 10 or count > 20 for count in kind_counts.values()):
            raise ValueError("each KB track must select between 10 and 20 entries")
        unique_fields = {
            "acquisition files": [
                name
                for entry in self.entries
                for name in (
                    entry.raw_response_file,
                    entry.acquisition_receipt_file,
                    entry.response_file,
                )
            ],
            "raw response hashes": [
                entry.raw_response_sha256 for entry in self.entries
            ],
            "receipt hashes": [
                entry.acquisition_receipt_sha256 for entry in self.entries
            ],
            "response hashes": [entry.response_sha256 for entry in self.entries],
            "page revisions": [
                (entry.project_domain, entry.page_id, entry.revision_id)
                for entry in self.entries
            ],
            "canonical page titles": [
                (entry.project_domain, entry.canonical_title.casefold())
                for entry in self.entries
            ],
            "evidence spans": [
                (
                    entry.project_domain,
                    entry.page_id,
                    entry.revision_id,
                    entry.evidence_start_char,
                    entry.evidence_end_char,
                )
                for entry in self.entries
            ],
        }
        for label, values in unique_fields.items():
            if len(values) != len(set(values)):
                raise ValueError(f"selection contains duplicate {label}")
        if self.review_record_file in {
            item.evidence_file for item in self.license_evidence
        }:
            raise ValueError("review ledger and license evidence files must differ")
        return self


class WikimediaArtifactDescriptor(_StrictFrozenModel):
    path: str
    bytes: int = Field(ge=0)
    sha256: Sha256
    row_count: int = Field(ge=1)


class WikimediaKBBundleManifest(_StrictFrozenModel):
    schema_version: Literal[1]
    adapter_id: Literal["wikimedia-revision-api-v1"]
    selection_lock_sha256: Sha256
    response_set_sha256: Sha256
    review_ledger_sha256: Sha256
    license_evidence_sha256: dict[ProjectDomain, Sha256]
    permission_matrix: WikimediaPermissionMatrix
    license_evidence: tuple[WikimediaLicenseEvidence, ...]
    redistribution_notice: Literal[
        "Preserve attribution, page-specific notices, and the CC BY-SA 4.0 "
        "license notice; mark changes and release adaptations under CC BY-SA "
        "4.0 or a compatible later version."
    ]
    entry_counts: dict[KBKind, int]
    artifacts: dict[str, WikimediaArtifactDescriptor]
    bundle_sha256: Sha256

    @field_validator("license_evidence", mode="before")
    @classmethod
    def coerce_license_evidence(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_names = {"encyclopedia.jsonl", "recipes.jsonl"}
        if set(self.artifacts) != expected_names:
            raise ValueError("bundle artifact set is invalid")
        if set(self.entry_counts) != set(_PROJECT_BY_KIND):
            raise ValueError("bundle entry counts must contain both tracks")
        if set(self.license_evidence_sha256) != set(_PROJECT_BY_KIND.values()):
            raise ValueError(
                "bundle license-evidence digests must contain both projects"
            )
        if [item.applicable_project for item in self.license_evidence] != sorted(
            _PROJECT_BY_KIND.values()
        ):
            raise ValueError("bundle license evidence must contain both projects")
        if (
            self.artifacts["encyclopedia.jsonl"].row_count
            != self.entry_counts["encyclopedia"]
        ):
            raise ValueError("encyclopedia count mismatch")
        if self.artifacts["recipes.jsonl"].row_count != self.entry_counts["recipe"]:
            raise ValueError("recipe count mismatch")
        unsigned = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if (
            self.bundle_sha256
            != hashlib.sha256(canonical_json_document(unsigned)).hexdigest()
        ):
            raise ValueError("bundle self-hash mismatch")
        return self


@dataclass(frozen=True)
class VerifiedWikimediaSelection:
    lock: WikimediaSelectionLock
    lock_sha256: str
    lock_snapshot: SourceFileSnapshot


@dataclass(frozen=True)
class VerifiedWikimediaKBBundle:
    output_dir: Path
    manifest: WikimediaKBBundleManifest
    manifest_snapshot: SourceFileSnapshot
    artifact_snapshots: dict[str, SourceFileSnapshot]


@dataclass(frozen=True)
class WikimediaAcquisitionWriteReport:
    raw_response_sha256: str
    acquisition_receipt_sha256: str
    response_sha256: str


@dataclass(frozen=True)
class WikimediaKBBuildReport:
    output_dir: Path
    selection_lock_sha256: str
    encyclopedia_entries: int
    recipe_entries: int
    manifest: WikimediaKBBundleManifest


def canonicalize_revision_api_response(
    raw_response: bytes,
    *,
    request_uri: str,
    acquired_at: datetime,
) -> bytes:
    """Normalize one official API response into a strict offline snapshot.

    This is an acquisition helper, not a provenance attestation.  The resulting
    bytes become formal only after a human reviews the page and an externally
    hashed selection lock pins the snapshot and its evidence span.
    """

    project_domain, requested_revision_id = _parse_canonical_request_uri(request_uri)
    raw = _parse_strict_json_object(raw_response, "MediaWiki API response")
    if set(raw) != {"batchcomplete", "query"} or raw["batchcomplete"] is not True:
        raise WikimediaKBError("API response top-level shape is not canonical")
    query = raw["query"]
    if not isinstance(query, dict) or set(query) != {"pages"}:
        raise WikimediaKBError("API response query shape is not canonical")
    pages = query["pages"]
    if not isinstance(pages, list) or len(pages) != 1:
        raise WikimediaKBError("API response must contain exactly one page")
    page = pages[0]
    if not isinstance(page, dict) or set(page) != {
        "pageid",
        "ns",
        "title",
        "revisions",
    }:
        raise WikimediaKBError("API page shape is not canonical")
    revisions = page["revisions"]
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise WikimediaKBError("API response must contain exactly one revision")
    revision = revisions[0]
    if not isinstance(revision, dict) or set(revision) != {
        "revid",
        "parentid",
        "timestamp",
        "sha1",
        "slots",
    }:
        raise WikimediaKBError("API revision shape is not canonical")
    slots = revision["slots"]
    if not isinstance(slots, dict) or set(slots) != {"main"}:
        raise WikimediaKBError("API response must contain only the main slot")
    main = slots["main"]
    if not isinstance(main, dict) or set(main) != {
        "contentmodel",
        "contentformat",
        "content",
    }:
        raise WikimediaKBError("API main-slot shape is not canonical")
    if revision["revid"] != requested_revision_id:
        raise WikimediaKBError("API response revision does not match request_uri")
    try:
        snapshot = WikimediaRevisionSnapshot(
            project_domain=project_domain,
            request_uri=request_uri,
            acquired_at=acquired_at,
            page_id=page["pageid"],
            namespace=page["ns"],
            title=page["title"],
            revision_id=revision["revid"],
            parent_revision_id=revision["parentid"],
            revision_timestamp=revision["timestamp"],
            revision_sha1=revision["sha1"],
            content_model=main["contentmodel"],
            content_format=main["contentformat"],
            content=main["content"],
            content_sha256=hashlib.sha256(
                str(main["content"]).encode("utf-8")
            ).hexdigest(),
        )
    except Exception as error:
        raise WikimediaKBError("API response violates the revision contract") from error
    return canonical_json_document(snapshot)


def write_revision_api_snapshot(
    raw_response: bytes,
    destination: Path | str,
    *,
    raw_response_destination: Path | str,
    acquisition_receipt_destination: Path | str,
    request_uri: str,
    final_uri: str,
    http_status: int,
    http_date: str,
    x_request_id: str,
    acquired_at: datetime,
    etag: str | None = None,
    last_modified: str | None = None,
    validator_absent_reason: str | None = None,
) -> WikimediaAcquisitionWriteReport:
    """Create raw bytes, their HTTP receipt, and the canonical snapshot.

    The caller performs the HTTP GET independently and supplies the exact raw
    response and transport facts.  This helper does not contact Wikimedia and
    does not make the result formally trusted.  Formal ingestion later requires
    all three files, an externally pinned selection, and an external human
    review ledger.
    """

    response_content = canonicalize_revision_api_response(
        raw_response, request_uri=request_uri, acquired_at=acquired_at
    )
    project_domain, revision_id = _parse_canonical_request_uri(request_uri)
    destinations = {
        "raw_response_file": Path(raw_response_destination).absolute(),
        "acquisition_receipt_file": Path(acquisition_receipt_destination).absolute(),
        "response_file": Path(destination).absolute(),
    }
    if len(set(destinations.values())) != 3:
        raise WikimediaKBError("acquisition output paths must be distinct")
    if len({path.parent for path in destinations.values()}) != 1:
        raise WikimediaKBError("acquisition outputs must share one directory")
    for label, path in destinations.items():
        _safe_root_file(path.name, label, _SAFE_SNAPSHOT_NAME)

    response_sha256 = hashlib.sha256(response_content).hexdigest()
    raw_sha256 = hashlib.sha256(raw_response).hexdigest()
    try:
        validator_kind = (
            "etag_and_last_modified"
            if etag is not None and last_modified is not None
            else "etag"
            if etag is not None
            else "last_modified"
            if last_modified is not None
            else "none"
        )
        receipt = WikimediaAcquisitionReceipt(
            project_domain=project_domain,
            revision_id=revision_id,
            request_uri=request_uri,
            final_uri=final_uri,
            http_status=http_status,
            http_date=http_date,
            x_request_id=x_request_id,
            validator_kind=validator_kind,
            etag=etag,
            last_modified=last_modified,
            validator_absent_reason=validator_absent_reason,
            acquired_at=acquired_at,
            raw_response_file=destinations["raw_response_file"].name,
            raw_response_bytes=len(raw_response),
            raw_response_sha256=raw_sha256,
            response_file=destinations["response_file"].name,
            response_sha256=response_sha256,
        )
    except Exception as error:
        raise WikimediaKBError("HTTP acquisition receipt is invalid") from error
    receipt_content = canonical_json_document(receipt)
    payloads = {
        destinations["raw_response_file"]: raw_response,
        destinations["acquisition_receipt_file"]: receipt_content,
        destinations["response_file"]: response_content,
    }
    for path in payloads:
        if os.path.lexists(path):
            raise FileExistsError(
                f"acquisition output already exists; refusing overwrite: {path}"
            )
    for path, content in payloads.items():
        _write_create_only_bytes(path, content, "Wikimedia acquisition")
    return WikimediaAcquisitionWriteReport(
        raw_response_sha256=raw_sha256,
        acquisition_receipt_sha256=hashlib.sha256(receipt_content).hexdigest(),
        response_sha256=response_sha256,
    )


def load_verified_selection_lock(
    path: Path | str, *, expected_selection_lock_sha256: str
) -> VerifiedWikimediaSelection:
    """Load a canonical selection only through a caller-owned SHA-256 root."""

    if not _SHA256_RE.fullmatch(expected_selection_lock_sha256):
        raise WikimediaKBError(
            "formal Wikimedia ingestion requires an external selection-lock SHA-256"
        )
    try:
        snapshot = _snapshot_secure_file(path, "Wikimedia selection lock")
    except Exception as error:
        raise WikimediaKBError("unable to snapshot Wikimedia selection lock") from error
    if snapshot.sha256 != expected_selection_lock_sha256:
        raise WikimediaKBError("selection lock does not match external expected digest")
    raw = _parse_strict_json_object(snapshot.content, "Wikimedia selection lock")
    try:
        lock = WikimediaSelectionLock.model_validate(raw)
    except Exception as error:
        raise WikimediaKBError("Wikimedia selection lock is invalid") from error
    if snapshot.content != canonical_json_document(lock):
        raise WikimediaKBError("Wikimedia selection lock must be canonical JSON")
    try:
        _verify_secure_snapshot(snapshot, "Wikimedia selection lock")
    except Exception as error:
        raise WikimediaKBError(
            "Wikimedia selection lock changed during load"
        ) from error
    return VerifiedWikimediaSelection(lock, snapshot.sha256, snapshot)


def build_wikimedia_kb_bundle(
    snapshot_root: Path | str,
    evidence_root: Path | str,
    output_dir: Path | str,
    *,
    selection_lock_path: Path | str,
    expected_selection_lock_sha256: str,
    expected_license_evidence_sha256: Mapping[ProjectDomain, str],
    expected_review_ledger_sha256: str,
) -> WikimediaKBBuildReport:
    """Build the two JSONL sources from externally locked offline evidence."""

    verified = load_verified_selection_lock(
        selection_lock_path,
        expected_selection_lock_sha256=expected_selection_lock_sha256,
    )
    expected_license_digests = _validate_external_license_digests(
        expected_license_evidence_sha256
    )
    if not _SHA256_RE.fullmatch(expected_review_ledger_sha256):
        raise WikimediaKBError(
            "formal Wikimedia ingestion requires an external review-ledger SHA-256"
        )

    snapshot_root = Path(snapshot_root).absolute()
    acquisition_names = {
        name
        for entry in verified.lock.entries
        for name in (
            entry.raw_response_file,
            entry.acquisition_receipt_file,
            entry.response_file,
        )
    }
    root_identity = _snapshot_exact_directory(
        snapshot_root,
        acquisition_names,
        "Wikimedia snapshot root",
    )

    source_snapshots: dict[str, SourceFileSnapshot] = {}
    parsed_snapshots: dict[str, WikimediaRevisionSnapshot] = {}
    for selected in verified.lock.entries:
        raw_snapshot = _snapshot_secure_file(
            snapshot_root / selected.raw_response_file,
            f"Wikimedia raw response {selected.raw_response_file}",
        )
        receipt_snapshot = _snapshot_secure_file(
            snapshot_root / selected.acquisition_receipt_file,
            f"Wikimedia acquisition receipt {selected.acquisition_receipt_file}",
        )
        response_snapshot = _snapshot_secure_file(
            snapshot_root / selected.response_file,
            f"Wikimedia canonical response {selected.response_file}",
        )
        if raw_snapshot.sha256 != selected.raw_response_sha256:
            raise WikimediaKBError("raw response hash does not match selection")
        if receipt_snapshot.sha256 != selected.acquisition_receipt_sha256:
            raise WikimediaKBError("acquisition receipt hash does not match selection")
        if response_snapshot.sha256 != selected.response_sha256:
            raise WikimediaKBError("canonical response hash does not match selection")

        receipt = _load_canonical_acquisition_receipt(
            receipt_snapshot, selected.acquisition_receipt_file
        )
        _validate_receipt_against_selection(receipt, selected)
        if (
            receipt.raw_response_sha256 != raw_snapshot.sha256
            or receipt.raw_response_bytes != len(raw_snapshot.content)
        ):
            raise WikimediaKBError("receipt does not bind the raw response bytes")
        if receipt.response_sha256 != response_snapshot.sha256:
            raise WikimediaKBError("receipt does not bind the canonical response")
        rebuilt = canonicalize_revision_api_response(
            raw_snapshot.content,
            request_uri=receipt.request_uri,
            acquired_at=receipt.acquired_at,
        )
        if rebuilt != response_snapshot.content:
            raise WikimediaKBError(
                "canonical response is not derived from the receipt-bound raw bytes"
            )
        parsed = _load_canonical_revision_snapshot(
            response_snapshot, selected.response_file
        )
        _validate_snapshot_against_selection(parsed, selected)
        for item in (raw_snapshot, receipt_snapshot, response_snapshot):
            source_snapshots[item.path.name] = item
        parsed_snapshots[selected.response_file] = parsed

    evidence_root = Path(evidence_root).absolute()
    evidence_names = {item.evidence_file for item in verified.lock.license_evidence} | {
        verified.lock.review_record_file
    }
    evidence_root_identity = _snapshot_exact_directory(
        evidence_root, evidence_names, "Wikimedia evidence root"
    )
    evidence_snapshots: dict[str, SourceFileSnapshot] = {}
    for reference in verified.lock.license_evidence:
        expected_digest = expected_license_digests[reference.applicable_project]
        if reference.evidence_sha256 != expected_digest:
            raise WikimediaKBError(
                "selection license evidence does not match external expected digest"
            )
        evidence_snapshot = _snapshot_secure_file(
            evidence_root / reference.evidence_file,
            f"Wikimedia license evidence {reference.applicable_project}",
        )
        if evidence_snapshot.sha256 != expected_digest:
            raise WikimediaKBError(
                "license evidence file does not match external expected digest"
            )
        _validate_license_evidence_bytes(evidence_snapshot.content, reference)
        evidence_snapshots[reference.evidence_file] = evidence_snapshot

    if verified.lock.review_record_sha256 != expected_review_ledger_sha256:
        raise WikimediaKBError(
            "selection review ledger does not match external expected digest"
        )
    review_snapshot = _snapshot_secure_file(
        evidence_root / verified.lock.review_record_file,
        "Wikimedia human review ledger",
    )
    if review_snapshot.sha256 != expected_review_ledger_sha256:
        raise WikimediaKBError(
            "review ledger file does not match external expected digest"
        )
    review_ledger = _load_canonical_review_ledger(review_snapshot)
    _validate_review_ledger(review_ledger, verified.lock)
    evidence_snapshots[verified.lock.review_record_file] = review_snapshot

    entries_by_kind: dict[KBKind, list[KBEntryV2]] = {
        "encyclopedia": [],
        "recipe": [],
    }
    for selected in verified.lock.entries:
        snapshot = parsed_snapshots[selected.response_file]
        evidence = snapshot.content[
            selected.evidence_start_char : selected.evidence_end_char
        ]
        entry = _build_entry(selected, evidence)
        entries_by_kind[selected.kind].append(entry)

    output_dir = Path(output_dir).absolute()
    output_parent_identity = _prepare_create_only_parent(output_dir)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        artifact_bytes = {
            "encyclopedia.jsonl": canonical_jsonl_bytes(
                entries_by_kind["encyclopedia"]
            ),
            "recipes.jsonl": canonical_jsonl_bytes(entries_by_kind["recipe"]),
        }
        for name, content in artifact_bytes.items():
            (staging / name).write_bytes(content)
        response_set_sha256 = hashlib.sha256(
            canonical_json_document(
                {
                    name: source_snapshots[name].sha256
                    for name in sorted(source_snapshots)
                }
            )
        ).hexdigest()
        artifact_payload = {
            name: {
                "path": name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "row_count": len(
                    entries_by_kind[
                        "encyclopedia" if name == "encyclopedia.jsonl" else "recipe"
                    ]
                ),
            }
            for name, content in artifact_bytes.items()
        }
        unsigned = {
            "schema_version": 1,
            "adapter_id": ADAPTER_ID,
            "selection_lock_sha256": verified.lock_sha256,
            "response_set_sha256": response_set_sha256,
            "review_ledger_sha256": review_snapshot.sha256,
            "license_evidence_sha256": {
                project: expected_license_digests[project]
                for project in sorted(expected_license_digests)
            },
            "permission_matrix": verified.lock.permission_matrix.model_dump(
                mode="json"
            ),
            "license_evidence": [
                item.model_dump(mode="json") for item in verified.lock.license_evidence
            ],
            "redistribution_notice": (
                "Preserve attribution, page-specific notices, and the CC BY-SA "
                "4.0 license notice; mark changes and release adaptations under "
                "CC BY-SA 4.0 or a compatible later version."
            ),
            "entry_counts": {
                "encyclopedia": len(entries_by_kind["encyclopedia"]),
                "recipe": len(entries_by_kind["recipe"]),
            },
            "artifacts": artifact_payload,
        }
        manifest = WikimediaKBBundleManifest.model_validate(
            {
                **unsigned,
                "bundle_sha256": hashlib.sha256(
                    canonical_json_document(unsigned)
                ).hexdigest(),
            }
        )
        (staging / "manifest.json").write_bytes(canonical_json_document(manifest))

        _verify_bundle_directory(staging, manifest.bundle_sha256)

        _verify_exact_directory(
            snapshot_root,
            acquisition_names,
            root_identity,
            "Wikimedia snapshot root",
        )
        _verify_exact_directory(
            evidence_root,
            evidence_names,
            evidence_root_identity,
            "Wikimedia evidence root",
        )
        try:
            _verify_secure_snapshot(verified.lock_snapshot, "Wikimedia selection lock")
            for name, source_snapshot in source_snapshots.items():
                _verify_secure_snapshot(
                    source_snapshot, f"Wikimedia acquisition input {name}"
                )
            for name, evidence_snapshot in evidence_snapshots.items():
                _verify_secure_snapshot(
                    evidence_snapshot, f"Wikimedia review evidence {name}"
                )
        except Exception as error:
            raise WikimediaKBError(
                "formal Wikimedia inputs changed before publication"
            ) from error
        if (
            _directory_identity(output_dir.parent, "Wikimedia bundle output parent")
            != output_parent_identity
        ):
            raise WikimediaKBError("Wikimedia bundle output parent changed")
        atomic_publish_new_directory(staging, output_dir)
        _verify_bundle_directory(output_dir, manifest.bundle_sha256)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return WikimediaKBBuildReport(
        output_dir=output_dir,
        selection_lock_sha256=verified.lock_sha256,
        encyclopedia_entries=len(entries_by_kind["encyclopedia"]),
        recipe_entries=len(entries_by_kind["recipe"]),
        manifest=manifest,
    )


def load_verified_wikimedia_kb_bundle(
    output_dir: Path | str, *, expected_bundle_sha256: str
) -> VerifiedWikimediaKBBundle:
    """Reparse a published bundle through a caller-owned bundle digest.

    A digest copied from the builder's return value in the same process is not
    an independent trust root.  Formal consumers must obtain this argument from
    reviewed configuration, release metadata, or another caller-owned channel.
    """

    if not _SHA256_RE.fullmatch(expected_bundle_sha256):
        raise WikimediaKBError(
            "formal bundle loading requires an external expected bundle SHA-256"
        )
    return _verify_bundle_directory(Path(output_dir).absolute(), expected_bundle_sha256)


def _validate_external_license_digests(
    value: Mapping[ProjectDomain, str],
) -> dict[ProjectDomain, str]:
    expected_projects = set(_PROJECT_BY_KIND.values())
    if set(value) != expected_projects:
        raise WikimediaKBError(
            "external license-evidence digests must contain exactly both projects"
        )
    result: dict[ProjectDomain, str] = {}
    for project in sorted(expected_projects):
        digest = value[project]  # type: ignore[index]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise WikimediaKBError(
                "formal ingestion requires external license-evidence SHA-256 values"
            )
        result[project] = digest  # type: ignore[index]
    return result


def _load_canonical_acquisition_receipt(
    snapshot: SourceFileSnapshot, label: str
) -> WikimediaAcquisitionReceipt:
    raw = _parse_strict_json_object(
        snapshot.content, f"Wikimedia acquisition receipt {label}"
    )
    try:
        receipt = WikimediaAcquisitionReceipt.model_validate(raw)
    except Exception as error:
        raise WikimediaKBError(f"invalid acquisition receipt: {label}") from error
    if snapshot.content != canonical_json_document(receipt):
        raise WikimediaKBError(f"acquisition receipt must be canonical JSON: {label}")
    return receipt


def _validate_receipt_against_selection(
    receipt: WikimediaAcquisitionReceipt, selected: WikimediaSelectionEntry
) -> None:
    bindings = {
        "project_domain": (receipt.project_domain, selected.project_domain),
        "revision_id": (receipt.revision_id, selected.revision_id),
        "raw_response_file": (
            receipt.raw_response_file,
            selected.raw_response_file,
        ),
        "raw_response_sha256": (
            receipt.raw_response_sha256,
            selected.raw_response_sha256,
        ),
        "response_file": (receipt.response_file, selected.response_file),
        "response_sha256": (receipt.response_sha256, selected.response_sha256),
    }
    mismatched = [name for name, pair in bindings.items() if pair[0] != pair[1]]
    if mismatched:
        raise WikimediaKBError(
            "acquisition receipt does not match selection: " + ", ".join(mismatched)
        )


def _validate_license_evidence_bytes(
    content: bytes, reference: WikimediaLicenseEvidence
) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WikimediaKBError("license evidence must be UTF-8 text") from error
    normalized = " ".join(text.casefold().split())
    if len(content) < 100:
        raise WikimediaKBError("license evidence is too short")
    if "creativecommons.org/licenses/by-sa/4.0" not in normalized:
        raise WikimediaKBError("license evidence does not identify CC BY-SA 4.0")
    if not any(
        marker in normalized
        for marker in ("attribution-sharealike 4.0", "cc by-sa 4.0")
    ):
        raise WikimediaKBError("license evidence lacks the reviewed license label")
    if reference.policy_revision_uri.casefold() not in normalized:
        raise WikimediaKBError("license evidence lacks its immutable policy URI")


def _load_canonical_review_ledger(
    snapshot: SourceFileSnapshot,
) -> WikimediaReviewLedger:
    raw = _parse_strict_json_object(snapshot.content, "Wikimedia human review ledger")
    try:
        ledger = WikimediaReviewLedger.model_validate(raw)
    except Exception as error:
        raise WikimediaKBError("Wikimedia human review ledger is invalid") from error
    if snapshot.content != canonical_json_document(ledger):
        raise WikimediaKBError("Wikimedia review ledger must be canonical JSON")
    return ledger


def _validate_review_ledger(
    ledger: WikimediaReviewLedger, lock: WikimediaSelectionLock
) -> None:
    if ledger.selection_id != lock.selection_id:
        raise WikimediaKBError("review ledger targets a different selection")
    if len(ledger.decisions) != len(lock.entries):
        raise WikimediaKBError("review ledger must contain one row per selection")
    for selected, decision in zip(lock.entries, ledger.decisions, strict=True):
        bindings = {
            "kind": (decision.kind, selected.kind),
            "project_domain": (decision.project_domain, selected.project_domain),
            "page_id": (decision.page_id, selected.page_id),
            "revision_id": (decision.revision_id, selected.revision_id),
            "raw_response_sha256": (
                decision.raw_response_sha256,
                selected.raw_response_sha256,
            ),
            "acquisition_receipt_sha256": (
                decision.acquisition_receipt_sha256,
                selected.acquisition_receipt_sha256,
            ),
            "response_sha256": (decision.response_sha256, selected.response_sha256),
            "evidence_start_char": (
                decision.evidence_start_char,
                selected.evidence_start_char,
            ),
            "evidence_end_char": (
                decision.evidence_end_char,
                selected.evidence_end_char,
            ),
            "evidence_sha256": (
                decision.evidence_sha256,
                selected.evidence_sha256,
            ),
            "reviewer_kind": (decision.reviewer_kind, lock.reviewer_kind),
            "reviewer_id": (decision.reviewer_id, lock.reviewer_id),
            "reviewed_at": (decision.reviewed_at, lock.reviewed_at),
        }
        mismatched = [name for name, pair in bindings.items() if pair[0] != pair[1]]
        if mismatched:
            raise WikimediaKBError(
                "review ledger row does not match selection: " + ", ".join(mismatched)
            )


def _verify_bundle_directory(
    output_dir: Path, expected_bundle_sha256: str
) -> VerifiedWikimediaKBBundle:
    expected_names = {"encyclopedia.jsonl", "recipes.jsonl", "manifest.json"}
    root_identity = _snapshot_exact_directory(
        output_dir, expected_names, "Wikimedia KB bundle"
    )
    manifest_snapshot = _snapshot_secure_file(
        output_dir / "manifest.json", "Wikimedia KB bundle manifest"
    )
    raw_manifest = _parse_strict_json_object(
        manifest_snapshot.content, "Wikimedia KB bundle manifest"
    )
    try:
        manifest = WikimediaKBBundleManifest.model_validate(raw_manifest)
    except Exception as error:
        raise WikimediaKBError("Wikimedia KB bundle manifest is invalid") from error
    if manifest_snapshot.content != canonical_json_document(manifest):
        raise WikimediaKBError("Wikimedia KB bundle manifest must be canonical JSON")
    if manifest.bundle_sha256 != expected_bundle_sha256:
        raise WikimediaKBError("bundle does not match external expected digest")

    artifacts: dict[str, SourceFileSnapshot] = {}
    for name in ("encyclopedia.jsonl", "recipes.jsonl"):
        snapshot = _snapshot_secure_file(
            output_dir / name, f"Wikimedia KB artifact {name}"
        )
        descriptor = manifest.artifacts[name]
        if descriptor.path != name:
            raise WikimediaKBError("bundle artifact descriptor path mismatch")
        if (
            descriptor.bytes != len(snapshot.content)
            or descriptor.sha256 != snapshot.sha256
        ):
            raise WikimediaKBError("bundle artifact bytes do not match manifest")
        entries: list[KBEntryV2] = []
        for row_index, line in enumerate(snapshot.content.splitlines(), start=1):
            raw_entry = _parse_strict_json_object(
                line, f"Wikimedia KB artifact {name} row {row_index}"
            )
            try:
                entries.append(KBEntryV2.model_validate(raw_entry))
            except Exception as error:
                raise WikimediaKBError(
                    f"Wikimedia KB artifact row is invalid: {name}:{row_index}"
                ) from error
        if len(entries) != descriptor.row_count:
            raise WikimediaKBError("bundle artifact row count mismatch")
        if snapshot.content != canonical_jsonl_bytes(entries):
            raise WikimediaKBError("bundle artifact must be canonical JSONL")
        artifacts[name] = snapshot

    if (
        len(artifacts["encyclopedia.jsonl"].content.splitlines())
        != manifest.entry_counts["encyclopedia"]
    ):
        raise WikimediaKBError("encyclopedia entry count mismatch")
    if (
        len(artifacts["recipes.jsonl"].content.splitlines())
        != manifest.entry_counts["recipe"]
    ):
        raise WikimediaKBError("recipe entry count mismatch")
    _verify_exact_directory(
        output_dir, expected_names, root_identity, "Wikimedia KB bundle"
    )
    _verify_secure_snapshot(manifest_snapshot, "Wikimedia KB bundle manifest")
    for name, snapshot in artifacts.items():
        _verify_secure_snapshot(snapshot, f"Wikimedia KB artifact {name}")
    return VerifiedWikimediaKBBundle(
        output_dir=output_dir,
        manifest=manifest,
        manifest_snapshot=manifest_snapshot,
        artifact_snapshots=artifacts,
    )


def _parse_canonical_request_uri(value: str) -> tuple[ProjectDomain, int]:
    try:
        value = _absolute_https_uri(value, "request_uri")
    except ValueError as error:
        raise WikimediaKBError(str(error)) from error
    parsed = urlsplit(value)
    if parsed.hostname not in set(_PROJECT_BY_KIND.values()):
        raise WikimediaKBError("request_uri uses an unapproved Wikimedia project")
    project_domain: ProjectDomain = parsed.hostname  # type: ignore[assignment]
    if parsed.port is not None or parsed.path != "/w/api.php" or parsed.fragment:
        raise WikimediaKBError("request_uri must target the canonical Action API path")
    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    if any(len(values) != 1 for values in query.values()):
        raise WikimediaKBError("request_uri must not contain duplicate query keys")
    expected_keys = {
        "action",
        "format",
        "formatversion",
        "prop",
        "revids",
        "rvprop",
        "rvslots",
    }
    if set(query) != expected_keys:
        raise WikimediaKBError("request_uri parameter set is not canonical")
    fixed = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "rvprop": "ids|timestamp|sha1|contentmodel|content",
        "rvslots": "main",
    }
    if any(query[key][0] != expected for key, expected in fixed.items()):
        raise WikimediaKBError("request_uri parameter values are not canonical")
    try:
        revision_id = int(query["revids"][0])
    except ValueError as error:
        raise WikimediaKBError("request_uri revids must be an integer") from error
    if revision_id <= 0 or value != revision_api_uri(project_domain, revision_id):
        raise WikimediaKBError("request_uri encoding/order is not canonical")
    return project_domain, revision_id


def _load_canonical_revision_snapshot(
    source_snapshot: SourceFileSnapshot, label: str
) -> WikimediaRevisionSnapshot:
    raw = _parse_strict_json_object(
        source_snapshot.content, f"Wikimedia response {label}"
    )
    try:
        parsed = WikimediaRevisionSnapshot.model_validate(raw)
    except Exception as error:
        raise WikimediaKBError(f"Wikimedia response is invalid: {label}") from error
    if source_snapshot.content != canonical_json_document(parsed):
        raise WikimediaKBError(f"Wikimedia response must be canonical JSON: {label}")
    return parsed


def _validate_snapshot_against_selection(
    snapshot: WikimediaRevisionSnapshot, selected: WikimediaSelectionEntry
) -> None:
    bindings = {
        "project_domain": (snapshot.project_domain, selected.project_domain),
        "page_id": (snapshot.page_id, selected.page_id),
        "revision_id": (snapshot.revision_id, selected.revision_id),
        "revision_timestamp": (
            snapshot.revision_timestamp,
            selected.revision_timestamp,
        ),
        "revision_sha1": (snapshot.revision_sha1, selected.revision_sha1),
        "canonical_title": (snapshot.title, selected.canonical_title),
    }
    mismatched = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatched:
        raise WikimediaKBError(
            "response does not match selection fields: " + ", ".join(mismatched)
        )
    if selected.evidence_end_char > len(snapshot.content):
        raise WikimediaKBError("selected evidence span exceeds revision content")
    evidence = snapshot.content[
        selected.evidence_start_char : selected.evidence_end_char
    ]
    if not evidence.strip() or len(re.findall(r"\w", evidence, re.UNICODE)) < 40:
        raise WikimediaKBError("selected evidence span is too weak for a KB entry")
    if hashlib.sha256(evidence.encode("utf-8")).hexdigest() != selected.evidence_sha256:
        raise WikimediaKBError("selected evidence span hash does not match")


def _build_entry(selected: WikimediaSelectionEntry, evidence: str) -> KBEntryV2:
    identity = (
        f"{selected.project_domain}:{selected.page_id}:{selected.revision_id}:"
        f"{selected.evidence_start_char}:{selected.evidence_end_char}:"
        f"{selected.evidence_sha256}"
    )
    entry_id = "wmf-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    citation = (
        f"{selected.source_uri}#wikitext-char-"
        f"{selected.evidence_start_char}-{selected.evidence_end_char}"
    )
    attribution = selected.attribution
    if selected.additional_attribution_notices:
        attribution += "; additional notices: " + " | ".join(
            selected.additional_attribution_notices
        )
    return KBEntryV2(
        schema_version=2,
        entry_id=entry_id,
        title=selected.canonical_title,
        text=evidence,
        kind=selected.kind,
        origin="dump",
        source_dataset=f"wikimedia-action-api:{selected.project_domain}",
        source_revision=(f"page-{selected.page_id}-revision-{selected.revision_id}"),
        source_record_id=(
            f"pageid:{selected.page_id}:revid:{selected.revision_id}:"
            f"wikitext-char:{selected.evidence_start_char}-"
            f"{selected.evidence_end_char}:sha256:{selected.evidence_sha256}"
        ),
        source_uri=citation,
        license_id=selected.license_id,
        attribution=attribution,
        content_sha256=selected.evidence_sha256,
        entity_group_id=f"wmf:{selected.project_domain}:page:{selected.page_id}",
        near_duplicate_cluster_id=None,
        derivation_parent_entry_ids=(),
        synth_provider=None,
        synth_model=None,
        synthesis_prompt_sha256=None,
        verification_status="source_verified",
    )


def _parse_strict_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: _raise_nonfinite(constant),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, WikimediaKBError) as error:
        raise WikimediaKBError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise WikimediaKBError(f"{label} must contain one JSON object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WikimediaKBError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raise_nonfinite(value: str):
    raise WikimediaKBError(f"non-finite JSON value: {value}")


def _path_is_link_or_junction(path: Path, metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(path, "is_junction", lambda: False)())
        or bool(attributes & _REPARSE_POINT)
    )


def _reject_link_path(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError:
            continue
        if _path_is_link_or_junction(current, metadata):
            raise WikimediaKBError(f"{label} must not traverse a symlink or junction")


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    _reject_link_path(path, label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise WikimediaKBError(f"{label} is missing") from error
    if _path_is_link_or_junction(path, metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise WikimediaKBError(f"{label} must be a real directory")
    return metadata.st_dev, metadata.st_ino


def _snapshot_secure_file(path: Path | str, label: str) -> SourceFileSnapshot:
    """Snapshot one single-link file with full ancestor/reparse checks."""

    absolute = Path(path).absolute()
    _reject_link_path(absolute, label)
    try:
        before = absolute.lstat()
    except OSError as error:
        raise WikimediaKBError(f"unable to inspect {label}") from error
    if _path_is_link_or_junction(absolute, before) or not stat.S_ISREG(before.st_mode):
        raise WikimediaKBError(f"{label} must be a regular non-reparse file")
    if before.st_nlink != 1:
        raise WikimediaKBError(f"{label} must not be hard-linked")
    try:
        snapshot = snapshot_regular_file(absolute, label)
    except Exception as error:
        raise WikimediaKBError(f"unable to snapshot {label}") from error
    try:
        after = absolute.lstat()
    except OSError as error:
        raise WikimediaKBError(f"{label} changed after snapshot") from error
    if (
        _path_is_link_or_junction(absolute, after)
        or after.st_nlink != 1
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != snapshot.identity
    ):
        raise WikimediaKBError(f"{label} changed during snapshot")
    return snapshot


def _verify_secure_snapshot(snapshot: SourceFileSnapshot, label: str) -> None:
    current = _snapshot_secure_file(snapshot.path, label)
    if (
        current.identity != snapshot.identity
        or current.sha256 != snapshot.sha256
        or current.content != snapshot.content
    ):
        raise WikimediaKBError(f"{label} changed before publication")
    try:
        verify_source_snapshot(snapshot, label)
    except Exception as error:
        raise WikimediaKBError(f"{label} changed before publication") from error


def _write_create_only_bytes(destination: Path, content: bytes, label: str) -> None:
    try:
        destination = prepare_formal_output_parent(destination)
        _reject_link_path(destination.parent, f"{label} parent")
    except Exception as error:
        if isinstance(error, FileExistsError):
            raise
        raise WikimediaKBError(f"invalid {label} destination") from error
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.staging-",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        _reject_link_path(temporary, f"{label} staging file")
        temporary_metadata = temporary.lstat()
        if (
            _path_is_link_or_junction(temporary, temporary_metadata)
            or not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
        ):
            raise WikimediaKBError(f"{label} staging file is unsafe")
        publish_staged_file_create_only(temporary, destination)
        temporary.unlink()
        temporary = None
        published = _snapshot_secure_file(destination, label)
        if published.content != content:
            raise WikimediaKBError(f"{label} changed during create-only publication")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _snapshot_exact_directory(
    path: Path, expected_names: set[str], label: str
) -> tuple[int, int]:
    identity = _directory_identity(path, label)
    try:
        items = list(path.iterdir())
    except OSError as error:
        raise WikimediaKBError(f"unable to enumerate {label}") from error
    names = {item.name for item in items}
    if len(names) != len(items) or names != expected_names:
        raise WikimediaKBError(f"{label} artifact set does not match selection")
    for item in items:
        metadata = item.lstat()
        if _path_is_link_or_junction(item, metadata) or not stat.S_ISREG(
            metadata.st_mode
        ):
            raise WikimediaKBError(f"{label} must contain only regular files")
        if metadata.st_nlink != 1:
            raise WikimediaKBError(f"{label} must not contain hard-linked files")
    return identity


def _verify_exact_directory(
    path: Path,
    expected_names: set[str],
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    if _snapshot_exact_directory(path, expected_names, label) != expected_identity:
        raise WikimediaKBError(f"{label} changed during formal ingestion")


def _prepare_create_only_parent(destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = _directory_identity(destination.parent, "Wikimedia bundle output parent")
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Wikimedia KB output already exists; refusing overwrite: {destination}"
        )
    return identity


__all__ = [
    "ACQUISITION_CONTRACT",
    "ADAPTER_ID",
    "LICENSE_ID",
    "LICENSE_URI",
    "SELECTION_POLICY_VERSION",
    "TERMS_REVISION_URI",
    "VALIDATOR_ABSENT_REASON",
    "VerifiedWikimediaSelection",
    "VerifiedWikimediaKBBundle",
    "WikimediaAcquisitionReceipt",
    "WikimediaAcquisitionWriteReport",
    "WikimediaKBError",
    "WikimediaKBBuildReport",
    "WikimediaKBBundleManifest",
    "WikimediaLicenseEvidence",
    "WikimediaPermissionMatrix",
    "WikimediaReviewDecision",
    "WikimediaReviewLedger",
    "WikimediaRevisionSnapshot",
    "WikimediaSelectionEntry",
    "WikimediaSelectionLock",
    "build_wikimedia_kb_bundle",
    "canonical_json_document",
    "canonicalize_revision_api_response",
    "load_verified_selection_lock",
    "load_verified_wikimedia_kb_bundle",
    "page_history_uri",
    "revision_api_uri",
    "revision_source_uri",
    "write_revision_api_snapshot",
]
