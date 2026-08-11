"""Externally pinned human review ledger for dataset permissions and PII.

Download state and adapter-generated hashes are evidence, not authorization.
This module keeps the human decision in a separate canonical ledger and binds
each approval to an externally locked source snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    read_stable_regular_file,
    sha256_bytes,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Purpose = Literal[
    "capability_gold",
    "challenge",
    "interaction_pattern",
    "knowledge_evidence",
    "language_style",
    "product_gallery",
    "tool_gold",
]
PermissionName = Literal[
    "download_allowed",
    "local_research_allowed",
    "local_embedding_allowed",
    "remote_embedding_allowed",
    "redistribution_allowed",
    "public_demo_allowed",
]

_MAX_POLICY_BYTES = 2 * 1024 * 1024
_MAX_LEDGER_BYTES = 8 * 1024 * 1024


class SourceReviewError(ValueError):
    """A source review policy or ledger is missing, invalid, or incomplete."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourcePermissions(_StrictFrozenModel):
    """Explicit permissions; absent legal evidence is represented as false."""

    download_allowed: bool
    local_research_allowed: bool
    local_embedding_allowed: bool
    remote_embedding_allowed: bool
    redistribution_allowed: bool
    public_demo_allowed: bool

    @model_validator(mode="after")
    def validate_implications(self) -> Self:
        if self.local_embedding_allowed and not self.local_research_allowed:
            raise ValueError("local embedding requires local research permission")
        if self.remote_embedding_allowed and not self.local_research_allowed:
            raise ValueError("remote embedding requires local research permission")
        if self.public_demo_allowed and not self.local_research_allowed:
            raise ValueError("public demo requires local research permission")
        return self


class SourceReviewRequirement(_StrictFrozenModel):
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    required: bool
    purposes: tuple[Purpose, ...]
    required_permissions: tuple[PermissionName, ...]
    pii_review: Literal["not_applicable", "required"]

    @field_validator("purposes", "required_permissions", mode="before")
    @classmethod
    def coerce_json_arrays(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_sets(self) -> Self:
        if not self.purposes:
            raise ValueError("source requirement purposes must not be empty")
        if self.purposes != tuple(sorted(set(self.purposes))):
            raise ValueError("source requirement purposes must be sorted and unique")
        if self.required_permissions != tuple(sorted(set(self.required_permissions))):
            raise ValueError("required permissions must be sorted and unique")
        return self


class SourceReviewPolicy(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    portfolio_sha256: Sha256
    requirements: tuple[SourceReviewRequirement, ...]

    @field_validator("requirements", mode="before")
    @classmethod
    def coerce_json_requirements(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_requirements(self) -> Self:
        if not self.requirements:
            raise ValueError("source review policy must not be empty")
        source_ids = tuple(item.source_id for item in self.requirements)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("source requirements must be sorted and unique")
        return self


class SourceReviewRecord(_StrictFrozenModel):
    """One human decision bound to immutable source and licence evidence."""

    schema_version: Literal[1] = 1
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_revision: str = Field(min_length=1)
    source_lock_sha256: Sha256
    license_id: str = Field(min_length=1)
    license_evidence_sha256: Sha256
    decision: Literal["approved", "deferred", "rejected"]
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    purposes: tuple[Purpose, ...]
    permissions: SourcePermissions
    pii_status: Literal["not_applicable", "reviewed_no_pii", "restricted", "redacted"]
    redaction_policy_sha256: Sha256 | None = None
    notes: str | None = None

    @field_validator("purposes", mode="before")
    @classmethod
    def coerce_json_purposes(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "source_revision",
        "license_id",
        "reviewer_id",
    )
    @classmethod
    def validate_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("review text fields must not have surrounding whitespace")
        return value

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("notes must be absent or canonical nonblank text")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        if self.purposes != tuple(sorted(set(self.purposes))):
            raise ValueError("review purposes must be sorted and unique")
        if self.decision == "approved":
            if not self.permissions.download_allowed:
                raise ValueError("approved source must permit download")
            if not self.permissions.local_research_allowed:
                raise ValueError("approved source must permit local research")
            if not self.purposes:
                raise ValueError("approved source must declare at least one purpose")
        if self.pii_status == "redacted":
            if self.redaction_policy_sha256 is None:
                raise ValueError("redacted source requires a redaction policy digest")
        elif self.redaction_policy_sha256 is not None:
            raise ValueError("redaction policy is valid only for redacted sources")
        if self.pii_status == "restricted" and (
            self.permissions.remote_embedding_allowed
            or self.permissions.redistribution_allowed
            or self.permissions.public_demo_allowed
        ):
            raise ValueError("restricted PII cannot leave the local research boundary")
        return self


class SourceReviewProposal(_StrictFrozenModel):
    """AI/preparer recommendation that cannot impersonate an owner signature."""

    schema_version: Literal[1] = 1
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_revision: str = Field(min_length=1)
    source_lock_sha256: Sha256
    license_id: str = Field(min_length=1)
    license_evidence_sha256: Sha256
    proposed_decision: Literal["approved", "deferred", "rejected"]
    prepared_by: str = Field(min_length=1)
    prepared_at: datetime
    owner_confirmation_required: Literal[True] = True
    purposes: tuple[Purpose, ...]
    permissions: SourcePermissions
    pii_status: Literal[
        "not_applicable", "reviewed_no_pii", "restricted", "redacted"
    ]
    redaction_policy_sha256: Sha256 | None = None
    rationale: str = Field(min_length=1)

    @field_validator("purposes", mode="before")
    @classmethod
    def coerce_json_purposes(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "source_revision",
        "license_id",
        "prepared_by",
        "rationale",
    )
    @classmethod
    def validate_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("proposal text fields must be canonical")
        return value

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        if self.prepared_at.tzinfo is None or self.prepared_at.utcoffset() is None:
            raise ValueError("prepared_at must include a timezone")
        if self.purposes != tuple(sorted(set(self.purposes))):
            raise ValueError("proposal purposes must be sorted and unique")
        if self.proposed_decision == "approved":
            if not self.permissions.download_allowed:
                raise ValueError("proposed approval must permit download")
            if not self.permissions.local_research_allowed:
                raise ValueError("proposed approval must permit local research")
            if not self.purposes:
                raise ValueError("proposed approval requires a purpose")
        if self.pii_status == "redacted":
            if self.redaction_policy_sha256 is None:
                raise ValueError("redacted proposal requires a redaction policy")
        elif self.redaction_policy_sha256 is not None:
            raise ValueError("redaction policy is valid only for redacted data")
        if self.pii_status == "restricted" and (
            self.permissions.remote_embedding_allowed
            or self.permissions.redistribution_allowed
            or self.permissions.public_demo_allowed
        ):
            raise ValueError("restricted PII cannot leave the local boundary")
        return self

    def owner_record(
        self,
        *,
        reviewer_id: str,
        reviewed_at: datetime,
    ) -> SourceReviewRecord:
        """Materialize this exact proposal only after explicit owner confirmation."""

        return SourceReviewRecord(
            source_id=self.source_id,
            source_revision=self.source_revision,
            source_lock_sha256=self.source_lock_sha256,
            license_id=self.license_id,
            license_evidence_sha256=self.license_evidence_sha256,
            decision=self.proposed_decision,
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
            purposes=self.purposes,
            permissions=self.permissions,
            pii_status=self.pii_status,
            redaction_policy_sha256=self.redaction_policy_sha256,
            notes=self.rationale,
        )


@dataclass(frozen=True)
class LoadedSourceReviewLedger:
    """A canonical ledger that may honestly retain deferred/rejected blockers."""

    policy: SourceReviewPolicy
    policy_file_sha256: str
    records: tuple[SourceReviewRecord, ...]
    ledger_file_sha256: str
    blockers: tuple[str, ...]

    @property
    def approved(self) -> dict[str, SourceReviewRecord]:
        return {
            record.source_id: record
            for record in self.records
            if record.decision == "approved"
        }


@dataclass(frozen=True)
class VerifiedSourceReviewLedger:
    policy: SourceReviewPolicy
    policy_file_sha256: str
    records: tuple[SourceReviewRecord, ...]
    ledger_file_sha256: str

    @property
    def approved(self) -> dict[str, SourceReviewRecord]:
        return {
            record.source_id: record
            for record in self.records
            if record.decision == "approved"
        }

    def require_approval(
        self,
        source_id: str,
        *,
        source_revision: str,
        source_lock_sha256: str,
        license_id: str,
        license_evidence_sha256: str,
        purposes: tuple[Purpose, ...] = (),
        permissions: tuple[PermissionName, ...] = (),
    ) -> SourceReviewRecord:
        """Require the approval for the exact bytes consumed by an adapter."""

        record = self.approved.get(source_id)
        if record is None:
            raise SourceReviewError(f"{source_id}: no approved source review")
        expected = {
            "source_revision": (record.source_revision, source_revision),
            "source_lock_sha256": (
                record.source_lock_sha256,
                source_lock_sha256,
            ),
            "license_id": (record.license_id, license_id),
            "license_evidence_sha256": (
                record.license_evidence_sha256,
                license_evidence_sha256,
            ),
        }
        mismatches = [
            field_name
            for field_name, (reviewed, observed) in expected.items()
            if reviewed != observed
        ]
        if mismatches:
            raise SourceReviewError(
                f"{source_id}: adapter source differs from reviewed evidence: "
                + ", ".join(mismatches)
            )
        missing_purposes = sorted(set(purposes).difference(record.purposes))
        if missing_purposes:
            raise SourceReviewError(
                f"{source_id}: review does not authorize purposes: "
                + ", ".join(missing_purposes)
            )
        missing_permissions = sorted(
            permission
            for permission in set(permissions)
            if getattr(record.permissions, permission) is not True
        )
        if missing_permissions:
            raise SourceReviewError(
                f"{source_id}: review does not grant permissions: "
                + ", ".join(missing_permissions)
            )
        return record


def load_source_review_policy(
    path: str | Path,
    *,
    expected_policy_file_sha256: str,
    expected_portfolio_sha256: str,
) -> SourceReviewPolicy:
    _require_sha256(expected_policy_file_sha256, "policy")
    _require_sha256(expected_portfolio_sha256, "portfolio")
    content = read_stable_regular_file(
        path,
        label="source review policy",
        max_bytes=_MAX_POLICY_BYTES,
    )
    if sha256_bytes(content) != expected_policy_file_sha256:
        raise SourceReviewError("source review policy external digest mismatch")
    value = parse_canonical_json(content, label="source review policy")
    if not isinstance(value, dict):
        raise SourceReviewError("source review policy root must be an object")
    try:
        policy = SourceReviewPolicy.model_validate_json(content, strict=True)
    except ValueError as error:
        raise SourceReviewError("source review policy schema is invalid") from error
    if content != canonical_json_bytes(policy.model_dump(mode="json")):
        raise SourceReviewError("source review policy must be canonical JSON")
    if policy.portfolio_sha256 != expected_portfolio_sha256:
        raise SourceReviewError("source review policy binds a different portfolio")
    return policy


def load_verified_source_review_ledger(
    path: str | Path,
    policy: SourceReviewPolicy,
    *,
    policy_file_sha256: str,
    expected_ledger_file_sha256: str,
) -> VerifiedSourceReviewLedger:
    loaded = load_source_review_ledger(
        path,
        policy,
        policy_file_sha256=policy_file_sha256,
        expected_ledger_file_sha256=expected_ledger_file_sha256,
    )
    if loaded.blockers:
        raise SourceReviewError(
            "source review prerequisites are incomplete: "
            + ", ".join(loaded.blockers)
        )
    return VerifiedSourceReviewLedger(
        policy=loaded.policy,
        policy_file_sha256=loaded.policy_file_sha256,
        records=loaded.records,
        ledger_file_sha256=loaded.ledger_file_sha256,
    )


def load_source_review_ledger(
    path: str | Path,
    policy: SourceReviewPolicy,
    *,
    policy_file_sha256: str,
    expected_ledger_file_sha256: str,
) -> LoadedSourceReviewLedger:
    """Load and bind a ledger without pretending deferred sources are approved."""

    _require_sha256(policy_file_sha256, "policy")
    _require_sha256(expected_ledger_file_sha256, "ledger")
    content = read_stable_regular_file(
        path,
        label="source review ledger",
        max_bytes=_MAX_LEDGER_BYTES,
    )
    if sha256_bytes(content) != expected_ledger_file_sha256:
        raise SourceReviewError("source review ledger external digest mismatch")
    rows = parse_canonical_jsonl(content, label="source review ledger")
    try:
        records = tuple(
            SourceReviewRecord.model_validate_json(
                canonical_json_bytes(row),
                strict=True,
            )
            for row in rows
        )
    except ValueError as error:
        raise SourceReviewError("source review ledger schema is invalid") from error
    if not records:
        raise SourceReviewError("source review ledger must not be empty")
    canonical = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) for record in records
    )
    if content != canonical:
        raise SourceReviewError("source review ledger must be canonical JSONL")
    source_ids = tuple(record.source_id for record in records)
    if source_ids != tuple(sorted(set(source_ids))):
        raise SourceReviewError("source review records must be sorted and unique")
    blockers = _required_approval_blockers(policy, records)
    return LoadedSourceReviewLedger(
        policy=policy,
        policy_file_sha256=policy_file_sha256,
        records=records,
        ledger_file_sha256=expected_ledger_file_sha256,
        blockers=blockers,
    )


def _required_approval_blockers(
    policy: SourceReviewPolicy,
    records: tuple[SourceReviewRecord, ...],
) -> tuple[str, ...]:
    by_source = {record.source_id: record for record in records}
    known = {item.source_id for item in policy.requirements}
    unknown = sorted(set(by_source).difference(known))
    if unknown:
        raise SourceReviewError(
            "source review ledger contains sources outside the policy: "
            + ", ".join(unknown)
        )
    blockers: list[str] = []
    for requirement in policy.requirements:
        record = by_source.get(requirement.source_id)
        if record is None:
            if requirement.required:
                blockers.append(f"{requirement.source_id}:missing")
            continue
        if record.decision != "approved":
            if requirement.required:
                blockers.append(f"{requirement.source_id}:decision={record.decision}")
            continue
        if not set(requirement.purposes).issubset(record.purposes):
            blockers.append(f"{requirement.source_id}:purpose")
        for permission in requirement.required_permissions:
            if getattr(record.permissions, permission) is not True:
                blockers.append(f"{requirement.source_id}:{permission}")
        if (
            requirement.pii_review == "required"
            and record.pii_status == "not_applicable"
        ):
            blockers.append(f"{requirement.source_id}:pii_review")
    return tuple(blockers)


def _require_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SourceReviewError(
            f"{label} expected SHA-256 must be 64 lowercase hex characters"
        )


__all__ = [
    "SourcePermissions",
    "LoadedSourceReviewLedger",
    "SourceReviewError",
    "SourceReviewPolicy",
    "SourceReviewProposal",
    "SourceReviewRecord",
    "SourceReviewRequirement",
    "VerifiedSourceReviewLedger",
    "load_source_review_policy",
    "load_source_review_ledger",
    "load_verified_source_review_ledger",
]
