"""External human-review receipt for an immutable Codex authoring bundle.

The Codex v5 canonical run directory has an exact file-set commitment.  Human
review therefore lives outside that directory and binds the immutable terminal
receipt and pre-review draft by digest.  Bank compilation remains a later,
separate operation.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.static_authoring import HumanReviewArtifact
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _relative_file(value: str, label: str) -> str:
    if not value or value != value.strip() or "\\" in value:
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


class CodexAuthoringHumanReviewReceipt(BaseModel):
    """Positive, unchanged human review bound to one canonical Codex run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    artifact_kind: Literal["codex_authoring_human_review"] = (
        "codex_authoring_human_review"
    )
    candidate_id: str
    run_id: str
    decision: Literal["accepted_unchanged"]
    canonical_invocation_receipt_file: str
    canonical_invocation_receipt_file_sha256: Sha256
    canonical_invocation_receipt_payload_sha256: Sha256
    pre_review_file: str
    pre_review_file_sha256: Sha256
    pre_review_bundle_sha256: Sha256
    authoring_input_sha256: Sha256
    max_human_review_minutes: int = Field(gt=0)
    human_review: HumanReviewArtifact
    human_review_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator(
        "candidate_id",
        "run_id",
    )
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("identity fields must be nonblank and trimmed")
        return value

    @field_validator(
        "canonical_invocation_receipt_file",
        "pre_review_file",
    )
    @classmethod
    def validate_relative_file(cls, value: str) -> str:
        return _relative_file(value, "bound file")

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        review = self.human_review
        draft = review.post_review_bundle
        if (
            review.changed
            or review.pre_review_sha256 != self.pre_review_file_sha256
            or review.post_review_sha256 != self.pre_review_file_sha256
            or draft.bundle_sha256 != self.pre_review_bundle_sha256
            or draft.authoring_input_sha256 != self.authoring_input_sha256
            or review.review_minutes > self.max_human_review_minutes
            or review.review_sha256 != self.human_review_sha256
        ):
            raise ValueError("human review does not match the accepted unchanged draft")
        unsigned = self.model_dump(mode="json")
        observed = unsigned.pop("receipt_sha256")
        if observed != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ValueError("human-review receipt self-hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_codex_authoring_human_review_receipt(
    *,
    candidate_id: str,
    run_id: str,
    canonical_invocation_receipt_file: str,
    canonical_invocation_receipt_file_sha256: str,
    canonical_invocation_receipt_payload_sha256: str,
    pre_review_file: str,
    pre_review_file_sha256: str,
    pre_review_bundle_sha256: str,
    authoring_input_sha256: str,
    max_human_review_minutes: int,
    human_review: HumanReviewArtifact,
) -> CodexAuthoringHumanReviewReceipt:
    payload = {
        "schema_version": 1,
        "artifact_kind": "codex_authoring_human_review",
        "candidate_id": candidate_id,
        "run_id": run_id,
        "decision": "accepted_unchanged",
        "canonical_invocation_receipt_file": canonical_invocation_receipt_file,
        "canonical_invocation_receipt_file_sha256": (
            canonical_invocation_receipt_file_sha256
        ),
        "canonical_invocation_receipt_payload_sha256": (
            canonical_invocation_receipt_payload_sha256
        ),
        "pre_review_file": pre_review_file,
        "pre_review_file_sha256": pre_review_file_sha256,
        "pre_review_bundle_sha256": pre_review_bundle_sha256,
        "authoring_input_sha256": authoring_input_sha256,
        "max_human_review_minutes": max_human_review_minutes,
        "human_review": human_review.model_dump(mode="json"),
        "human_review_sha256": human_review.review_sha256,
    }
    return CodexAuthoringHumanReviewReceipt.model_validate(
        {
            **payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


__all__ = [
    "CodexAuthoringHumanReviewReceipt",
    "build_codex_authoring_human_review_receipt",
]
