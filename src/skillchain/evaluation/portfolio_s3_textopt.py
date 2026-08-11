"""Deterministic, fail-closed rule patches for Portfolio S3 Skill Bodies.

The model-facing proposal in this module is deliberately not a complete Skill
Body.  A trusted compiler binds the proposal to one failure cluster and one
parent Skill, resolves exact rule-line hashes, protects compiler-owned Body
sections, and only then emits the existing :class:`PortfolioSkillMutation`
carrier used by the treatment pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.evaluation.portfolio_treatments import (
    PortfolioSkillMutation,
    PortfolioTreatmentError,
    parse_portfolio_skill_output_contract,
)
from skillchain.static_authoring import StaticBankArtifact, StrictSkillArtifact
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RuleId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]

PORTFOLIO_S3_TEXTOPT_POLICY_VERSION = "portfolio-s3-textopt-v1"
PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION = (
    "portfolio-s3-textopt-proposal-v1"
)
PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION = (
    "portfolio-s3-rejected-edit-buffer-v1"
)

PortfolioS3FailureClassification = Literal["SKILL_DEFECT", "EXECUTION_LAPSE"]
PortfolioS3OptimizationMode = Literal["diagnostic_l1", "confirmed_l3"]

_EDITABLE_HEADINGS = frozenset(
    {
        "## Success criteria",
        "## Acceptable answer rules",
        "## Failure conditions",
        "## Fallback triggers",
        "## Fallback responses",
        "## Indeterminate conditions",
    }
)
_OUTPUT_CONTRACT_HEADING = "## Output contract"
_OUTPUT_CONTRACT_FIELDS = (
    "response_kind",
    "required_sections",
    "card_requirement",
    "card_fields",
    "evidence_requirement",
)
_RULE_LINE_RE = re.compile(
    r"^- \[(?P<rule_id>[A-Za-z0-9][A-Za-z0-9_.:-]*)\] "
    r"(?P<statement>.+) (?P<provenance>\(source: (?P<source>[^()\s]+)\))$"
)
_HEADING_RE = re.compile(r"^#{1,2} .+$")
_S3_RULE_ID_PREFIX = "s3.textopt."
_S3_SOURCE_PREFIX = "s3_textopt:"
_S3_INSERTED_RULE_ID_RE = re.compile(
    r"^s3\.textopt\.(?P<fingerprint_prefix>[0-9a-f]{16})\.(?P<index>[1-9][0-9]*)$"
)
_S3_INSERTED_PROVENANCE_RE = re.compile(
    r"^\(source: s3_textopt:(?P<fingerprint>[0-9a-f]{64})\)$"
)


class PortfolioS3TextOptError(PortfolioTreatmentError):
    """An S3 TextOpt proposal, binding, or deterministic compile is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonblank without edge whitespace")
    return value


def _sorted_unique_text(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not values or any(not item or item != item.strip() for item in values):
        raise ValueError(f"{label} must contain nonblank values")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")
    return values


def _self_hash(model: BaseModel, field_name: str) -> str:
    payload = model.model_dump(mode="json", exclude={field_name})
    return sha256_bytes(canonical_json_bytes(payload))


class PortfolioS3FailureClusterBinding(_StrictFrozenModel):
    """Immutable classifier output authorizing (or blocking) one S3 patch."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s3-failure-cluster-binding"] = (
        "portfolio-s3-failure-cluster-binding"
    )
    policy_version: Literal[PORTFOLIO_S3_TEXTOPT_POLICY_VERSION] = (
        PORTFOLIO_S3_TEXTOPT_POLICY_VERSION
    )
    capability_id: str
    failure_cluster_id: str
    classification: PortfolioS3FailureClassification
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    addressed_dimensions: tuple[str, ...] = Field(min_length=1)
    support_count: int = Field(ge=1)
    minimum_support_count: int = Field(ge=1)
    cluster_sha256: Sha256

    @field_validator("evidence_ids", "addressed_dimensions", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @field_validator("capability_id", "failure_cluster_id")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("evidence_ids", "addressed_dimensions")
    @classmethod
    def validate_text_tuples(
        cls, value: tuple[str, ...], info
    ) -> tuple[str, ...]:
        return _sorted_unique_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.support_count < len(self.evidence_ids):
            raise ValueError("support_count cannot be smaller than evidence count")
        if self.cluster_sha256 != _self_hash(self, "cluster_sha256"):
            raise ValueError("cluster_sha256 mismatch")
        return self

    @property
    def is_eligible_skill_defect(self) -> bool:
        return (
            self.classification == "SKILL_DEFECT"
            and self.support_count >= self.minimum_support_count
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_s3_failure_cluster_binding(
    *,
    capability_id: str,
    failure_cluster_id: str,
    classification: PortfolioS3FailureClassification,
    evidence_ids: tuple[str, ...],
    addressed_dimensions: tuple[str, ...],
    support_count: int,
    minimum_support_count: int,
) -> PortfolioS3FailureClusterBinding:
    """Build a canonical self-hashed cluster binding."""

    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s3-failure-cluster-binding",
        "policy_version": PORTFOLIO_S3_TEXTOPT_POLICY_VERSION,
        "capability_id": capability_id,
        "failure_cluster_id": failure_cluster_id,
        "classification": classification,
        "evidence_ids": list(evidence_ids),
        "addressed_dimensions": list(addressed_dimensions),
        "support_count": support_count,
        "minimum_support_count": minimum_support_count,
    }
    return _validate_model(
        PortfolioS3FailureClusterBinding,
        {**payload, "cluster_sha256": sha256_bytes(canonical_json_bytes(payload))},
        "failure cluster binding",
    )


class _PortfolioS3TextPatchEdit(_StrictFrozenModel):
    rule_id: RuleId
    expected_text_sha256: Sha256
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    support_count: int = Field(ge=1)
    addressed_dimensions: tuple[str, ...] = Field(min_length=1)
    rationale: str

    @field_validator("evidence_ids", "addressed_dimensions", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @field_validator("evidence_ids", "addressed_dimensions")
    @classmethod
    def validate_text_tuples(
        cls, value: tuple[str, ...], info
    ) -> tuple[str, ...]:
        return _sorted_unique_text(value, info.field_name)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _nonblank(value, "rationale")

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.support_count < len(self.evidence_ids):
            raise ValueError("support_count cannot be smaller than evidence count")
        return self


def _validate_statement(value: str) -> str:
    _nonblank(value, "replacement")
    if (
        "\n" in value
        or "\r" in value
        or value.startswith("#")
        or value.startswith("- [")
        or "(source:" in value
    ):
        raise ValueError("replacement must be one plain rule statement")
    return value


class PortfolioS3ReplaceRuleEdit(_PortfolioS3TextPatchEdit):
    op: Literal["replace_rule"] = "replace_rule"
    replacement: str

    @field_validator("replacement")
    @classmethod
    def validate_replacement(cls, value: str) -> str:
        return _validate_statement(value)


class PortfolioS3InsertAfterRuleEdit(_PortfolioS3TextPatchEdit):
    op: Literal["insert_after_rule"] = "insert_after_rule"
    replacement: str

    @field_validator("replacement")
    @classmethod
    def validate_replacement(cls, value: str) -> str:
        return _validate_statement(value)


class PortfolioS3DeleteRuleEdit(_PortfolioS3TextPatchEdit):
    op: Literal["delete_rule"] = "delete_rule"
    replacement: None

    @field_validator("replacement", mode="before")
    @classmethod
    def normalize_empty_replacement(cls, value: object) -> object:
        return None if value == "" else value


PortfolioS3TextPatchEdit: TypeAlias = Annotated[
    PortfolioS3ReplaceRuleEdit
    | PortfolioS3InsertAfterRuleEdit
    | PortfolioS3DeleteRuleEdit,
    Field(discriminator="op"),
]


class PortfolioS3TextPatchProposal(_StrictFrozenModel):
    """Strict model-facing patch envelope; it is not yet trusted to execute."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s3-text-patch-proposal"] = (
        "portfolio-s3-text-patch-proposal"
    )
    policy_version: Literal[PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION] = (
        PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION
    )
    capability_id: str
    failure_cluster_id: str
    failure_cluster_sha256: Sha256
    edits: tuple[PortfolioS3TextPatchEdit, ...] = Field(min_length=1, max_length=3)
    reconsideration_rationale: str | None

    @field_validator("edits", mode="before")
    @classmethod
    def coerce_edits(cls, value: object) -> object:
        return _tuple(value)

    @field_validator("capability_id", "failure_cluster_id")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("reconsideration_rationale")
    @classmethod
    def validate_reconsideration(cls, value: object) -> object:
        # Codex's frozen Structured Outputs subset has no JSON-null type.  The
        # provider schema therefore uses an empty-string sentinel for the
        # normal, first-attempt case; normalize it before strict validation.
        if value == "":
            return None
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("reconsideration_rationale must be text or null")
            _nonblank(value, "reconsideration_rationale")
        return value

    @model_validator(mode="after")
    def validate_unique_anchors(self) -> Self:
        anchors = tuple(edit.rule_id for edit in self.edits)
        if len(set(anchors)) != len(anchors):
            raise ValueError("patch edits must target unique rule IDs")
        return self


def portfolio_s3_text_patch_proposal_json_schema(
    *, max_edits: Literal[1, 3] = 3
) -> dict[str, object]:
    """Return a closed schema in the frozen Codex Structured Outputs subset."""

    if max_edits not in {1, 3}:
        raise PortfolioS3TextOptError("S3 proposal schema max_edits must be 1 or 3")

    def edit_schema(op: str) -> dict[str, object]:
        replacement: dict[str, object] = {"type": "string"}
        if op == "delete_rule":
            replacement["enum"] = [""]
        properties: dict[str, object] = {
            "op": {"type": "string", "enum": [op]},
            "rule_id": {"type": "string"},
            "expected_text_sha256": {"type": "string"},
            "replacement": replacement,
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "support_count": {"type": "integer"},
            "addressed_dimensions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }

    properties: dict[str, object] = {
        "schema_version": {"type": "integer", "enum": [1]},
        "artifact_kind": {
            "type": "string",
            "enum": ["portfolio-s3-text-patch-proposal"],
        },
        "policy_version": {
            "type": "string",
            "enum": [PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION],
        },
        "capability_id": {"type": "string"},
        "failure_cluster_id": {"type": "string"},
        "failure_cluster_sha256": {"type": "string"},
        "edits": {
            "type": "array",
            "minItems": 1,
            "maxItems": max_edits,
            "items": {
                "anyOf": [
                    edit_schema("replace_rule"),
                    edit_schema("insert_after_rule"),
                    edit_schema("delete_rule"),
                ]
            },
        },
        # The CLI subset cannot spell nullable.  Empty means no prior rejection.
        "reconsideration_rationale": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def parse_portfolio_s3_text_patch_proposal(
    content: bytes | str | Mapping[str, object],
) -> PortfolioS3TextPatchProposal:
    """Strictly parse provider output without requiring provider key ordering."""

    try:
        if isinstance(content, str):
            raw: object = parse_strict_json(
                content.encode("utf-8"), label="S3 text patch proposal"
            )
        elif isinstance(content, bytes):
            raw = parse_strict_json(content, label="S3 text patch proposal")
        elif isinstance(content, Mapping):
            raw = dict(content)
        else:
            raise PortfolioS3TextOptError(
                "S3 text patch proposal must be JSON bytes, text, or an object"
            )
        return PortfolioS3TextPatchProposal.model_validate(raw, strict=True)
    except (ArtifactFormatError, UnicodeEncodeError, ValidationError) as error:
        raise PortfolioS3TextOptError("S3 text patch proposal is invalid") from error


class PortfolioS3NormalizedTextPatchEdit(_StrictFrozenModel):
    op: Literal["replace_rule", "insert_after_rule", "delete_rule"]
    rule_id: RuleId
    expected_text_sha256: Sha256
    replacement: str | None
    generated_rule_id: RuleId | None
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    support_count: int = Field(ge=1)
    addressed_dimensions: tuple[str, ...] = Field(min_length=1)
    rationale: str

    @field_validator("evidence_ids", "addressed_dimensions", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        _sorted_unique_text(self.evidence_ids, "evidence_ids")
        _sorted_unique_text(self.addressed_dimensions, "addressed_dimensions")
        _nonblank(self.rationale, "rationale")
        if self.support_count < len(self.evidence_ids):
            raise ValueError("support_count cannot be smaller than evidence count")
        if self.op == "insert_after_rule":
            if self.replacement is None or self.generated_rule_id is None:
                raise ValueError("insert edit requires replacement and generated ID")
        elif self.op == "replace_rule":
            if self.replacement is None or self.generated_rule_id is not None:
                raise ValueError("replace edit shape is invalid")
        elif self.replacement is not None or self.generated_rule_id is not None:
            raise ValueError("delete edit shape is invalid")
        if self.replacement is not None:
            _validate_statement(self.replacement)
        return self


class PortfolioS3TextPatchArtifact(_StrictFrozenModel):
    """Trusted normalized patch artifact written beside the stage mutation."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s3-text-patch"] = "portfolio-s3-text-patch"
    policy_version: Literal[PORTFOLIO_S3_TEXTOPT_POLICY_VERSION] = (
        PORTFOLIO_S3_TEXTOPT_POLICY_VERSION
    )
    optimization_mode: PortfolioS3OptimizationMode
    capability_id: str
    failure_cluster_id: str
    failure_cluster_sha256: Sha256
    parent_skill_sha256: Sha256
    origin_skill_sha256: Sha256
    patch_fingerprint: Sha256
    edits: tuple[PortfolioS3NormalizedTextPatchEdit, ...] = Field(
        min_length=1, max_length=3
    )
    reconsideration_rationale: str | None = None
    patch_sha256: Sha256

    @field_validator("edits", mode="before")
    @classmethod
    def coerce_edits(cls, value: object) -> object:
        return _tuple(value)

    @field_validator("capability_id", "failure_cluster_id")
    @classmethod
    def validate_names(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("reconsideration_rationale")
    @classmethod
    def validate_reconsideration(cls, value: str | None) -> str | None:
        if value is not None:
            _nonblank(value, "reconsideration_rationale")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        expected_count = 1 if self.optimization_mode == "diagnostic_l1" else None
        if expected_count is not None and len(self.edits) != expected_count:
            raise ValueError("diagnostic_l1 requires exactly one edit")
        anchors = tuple(edit.rule_id for edit in self.edits)
        if len(set(anchors)) != len(anchors):
            raise ValueError("normalized edits must target unique rule IDs")
        if self.patch_fingerprint != _patch_fingerprint(
            self.capability_id, self.edits
        ):
            raise ValueError("patch_fingerprint mismatch")
        generated_ids = tuple(
            edit.generated_rule_id
            for edit in self.edits
            if edit.generated_rule_id is not None
        )
        if len(set(generated_ids)) != len(generated_ids):
            raise ValueError("generated S3 rule IDs must be unique")
        for index, edit in enumerate(self.edits, start=1):
            if edit.op == "insert_after_rule" and edit.generated_rule_id != (
                f"{_S3_RULE_ID_PREFIX}{self.patch_fingerprint[:16]}.{index}"
            ):
                raise ValueError("generated S3 rule ID does not match patch fingerprint")
        if self.patch_sha256 != _self_hash(self, "patch_sha256"):
            raise ValueError("patch_sha256 mismatch")
        return self

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item for edit in self.edits for item in edit.evidence_ids}))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS3RejectedEditRecord(_StrictFrozenModel):
    capability_id: str
    patch_fingerprint: Sha256
    rejected_patch_sha256: Sha256
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    rejection_reason: str
    record_sha256: Sha256

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def coerce_evidence(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _nonblank(self.capability_id, "capability_id")
        _nonblank(self.rejection_reason, "rejection_reason")
        _sorted_unique_text(self.evidence_ids, "evidence_ids")
        if self.record_sha256 != _self_hash(self, "record_sha256"):
            raise ValueError("record_sha256 mismatch")
        return self


class PortfolioS3RejectedEditBuffer(_StrictFrozenModel):
    """Immutable canonical set of previously gate-rejected patch semantics."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s3-rejected-edit-buffer"] = (
        "portfolio-s3-rejected-edit-buffer"
    )
    policy_version: Literal[PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION] = (
        PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION
    )
    records: tuple[PortfolioS3RejectedEditRecord, ...]
    buffer_sha256: Sha256

    @field_validator("records", mode="before")
    @classmethod
    def coerce_records(cls, value: object) -> object:
        return _tuple(value)

    @model_validator(mode="after")
    def validate_buffer(self) -> Self:
        keys = tuple(
            (record.capability_id, record.patch_fingerprint, record.record_sha256)
            for record in self.records
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("rejected edit records must be sorted and unique")
        if self.buffer_sha256 != _self_hash(self, "buffer_sha256"):
            raise ValueError("buffer_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_empty_portfolio_s3_rejected_edit_buffer() -> PortfolioS3RejectedEditBuffer:
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s3-rejected-edit-buffer",
        "policy_version": PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION,
        "records": [],
    }
    return _validate_model(
        PortfolioS3RejectedEditBuffer,
        {**payload, "buffer_sha256": sha256_bytes(canonical_json_bytes(payload))},
        "empty rejected edit buffer",
    )


def append_portfolio_s3_rejected_edit(
    buffer: PortfolioS3RejectedEditBuffer,
    *,
    patch: PortfolioS3TextPatchArtifact,
    rejection_reason: str,
) -> PortfolioS3RejectedEditBuffer:
    """Return a new buffer containing one deterministic rejected-patch record."""

    buffer = _revalidate_instance(
        buffer, PortfolioS3RejectedEditBuffer, "rejected edit buffer"
    )
    patch = _revalidate_instance(patch, PortfolioS3TextPatchArtifact, "text patch")
    record_payload = {
        "capability_id": patch.capability_id,
        "patch_fingerprint": patch.patch_fingerprint,
        "rejected_patch_sha256": patch.patch_sha256,
        "evidence_ids": list(patch.evidence_ids),
        "rejection_reason": rejection_reason,
    }
    record = _validate_model(
        PortfolioS3RejectedEditRecord,
        {
            **record_payload,
            "record_sha256": sha256_bytes(canonical_json_bytes(record_payload)),
        },
        "rejected edit record",
    )
    records = tuple(
        sorted(
            (*buffer.records, record),
            key=lambda item: (
                item.capability_id,
                item.patch_fingerprint,
                item.record_sha256,
            ),
        )
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s3-rejected-edit-buffer",
        "policy_version": PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION,
        "records": [item.model_dump(mode="json") for item in records],
    }
    return _validate_model(
        PortfolioS3RejectedEditBuffer,
        {**payload, "buffer_sha256": sha256_bytes(canonical_json_bytes(payload))},
        "rejected edit buffer",
    )


def parse_portfolio_s3_rejected_edit_buffer(
    content: bytes | str,
) -> PortfolioS3RejectedEditBuffer:
    """Load only canonical persisted rejected-buffer bytes."""

    raw, data = _parse_canonical_artifact(content, "S3 rejected edit buffer")
    buffer = _validate_model(
        PortfolioS3RejectedEditBuffer, raw, "S3 rejected edit buffer"
    )
    if buffer.canonical_bytes() != data:
        raise PortfolioS3TextOptError(
            "S3 rejected edit buffer is not canonical model bytes"
        )
    return buffer


def parse_portfolio_s3_text_patch_artifact(
    content: bytes | str,
) -> PortfolioS3TextPatchArtifact:
    """Load only canonical persisted normalized-patch bytes."""

    raw, data = _parse_canonical_artifact(content, "S3 text patch artifact")
    patch = _validate_model(
        PortfolioS3TextPatchArtifact,
        raw,
        "S3 text patch artifact",
    )
    if patch.canonical_bytes() != data:
        raise PortfolioS3TextOptError(
            "S3 text patch artifact is not canonical model bytes"
        )
    return patch


class _BodyRule:
    __slots__ = ("line_index", "rule_id", "statement", "provenance", "section")

    def __init__(
        self,
        *,
        line_index: int,
        rule_id: str,
        statement: str,
        provenance: str,
        section: str,
    ) -> None:
        self.line_index = line_index
        self.rule_id = rule_id
        self.statement = statement
        self.provenance = provenance
        self.section = section


class _ParsedBody:
    __slots__ = ("lines", "rules", "sections")

    def __init__(
        self,
        *,
        lines: list[str],
        rules: dict[str, _BodyRule],
        sections: dict[str, str],
    ) -> None:
        self.lines = lines
        self.rules = rules
        self.sections = sections


def _is_compiler_inserted_rule(rule_id: str, provenance: str) -> bool:
    rule_match = _S3_INSERTED_RULE_ID_RE.fullmatch(rule_id)
    provenance_match = _S3_INSERTED_PROVENANCE_RE.fullmatch(provenance)
    return bool(
        rule_match is not None
        and provenance_match is not None
        and rule_match.group("fingerprint_prefix")
        == provenance_match.group("fingerprint")[:16]
    )


def _parse_body(body: str) -> _ParsedBody:
    if (
        not body
        or "\r" in body
        or not body.endswith("\n")
        or body.endswith("\n\n")
        or body != body.lstrip()
    ):
        raise PortfolioS3TextOptError(
            "Skill Body must use LF and exactly one terminal newline"
        )
    lines = body[:-1].split("\n")
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if _HEADING_RE.fullmatch(line):
            headings.append((index, line))
    if not headings or headings[0] != (0, "# Objective"):
        raise PortfolioS3TextOptError("Skill Body must start with # Objective")
    heading_names = tuple(name for _, name in headings)
    if len(set(heading_names)) != len(heading_names):
        raise PortfolioS3TextOptError("Skill Body headings must be unique")

    sections: dict[str, str] = {}
    section_by_line: dict[int, str] = {}
    for position, (start, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        sections[heading] = "\n".join(lines[start:end]) + "\n"
        for line_index in range(start, end):
            section_by_line[line_index] = heading

    rules: dict[str, _BodyRule] = {}
    for index, line in enumerate(lines):
        match = _RULE_LINE_RE.fullmatch(line)
        if match is None:
            continue
        rule_id = match.group("rule_id")
        if rule_id in rules:
            raise PortfolioS3TextOptError("Skill Body rule IDs must be globally unique")
        rules[rule_id] = _BodyRule(
            line_index=index,
            rule_id=rule_id,
            statement=match.group("statement"),
            provenance=match.group("provenance"),
            section=section_by_line[index],
        )

    _validate_output_contract(body, sections)
    return _ParsedBody(lines=lines, rules=rules, sections=sections)


def _validate_output_contract(body: str, sections: Mapping[str, str]) -> None:
    block = sections.get(_OUTPUT_CONTRACT_HEADING)
    if block is None:
        raise PortfolioS3TextOptError("Skill Body lacks a unique output contract")
    fields: list[str] = []
    values: dict[str, str] = {}
    for line in block.splitlines()[1:]:
        if not line:
            continue
        match = re.fullmatch(r"- ([a-z_]+): ?(.*)", line)
        if match is None:
            raise PortfolioS3TextOptError("output contract contains an invalid line")
        key, value = match.groups()
        if key in values:
            raise PortfolioS3TextOptError("output contract repeats a field")
        fields.append(key)
        values[key] = value
    if tuple(fields) != _OUTPUT_CONTRACT_FIELDS:
        raise PortfolioS3TextOptError(
            "output contract must contain exactly the frozen five fields"
        )
    if not values["response_kind"] or not values["evidence_requirement"]:
        raise PortfolioS3TextOptError("output contract values are incomplete")
    try:
        parse_portfolio_skill_output_contract(body)
    except PortfolioTreatmentError as error:
        raise PortfolioS3TextOptError("output contract semantics are invalid") from error


def build_portfolio_s3_editable_rule_catalog(
    *,
    parent_bank: StaticBankArtifact,
    capability_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    """Expose exact editable rule lines and hashes for copy-only model use."""

    parent_bank = _revalidate_instance(parent_bank, StaticBankArtifact, "parent Bank")
    if (
        not capability_ids
        or capability_ids != tuple(sorted(set(capability_ids)))
    ):
        raise PortfolioS3TextOptError(
            "editable rule catalog capability IDs must be sorted and unique"
        )
    catalog: list[dict[str, object]] = []
    for capability_id in capability_ids:
        skill = _skill_for_capability(parent_bank, capability_id, "parent Bank")
        parsed = _parse_body(skill.body)
        rules = []
        for rule in sorted(parsed.rules.values(), key=lambda item: item.line_index):
            if rule.section not in _EDITABLE_HEADINGS:
                continue
            full_line = parsed.lines[rule.line_index]
            rules.append(
                {
                    "rule_id": rule.rule_id,
                    "section": rule.section,
                    "full_line": full_line,
                    "expected_text_sha256": sha256_bytes(full_line.encode("utf-8")),
                    "delete_allowed": _is_compiler_inserted_rule(
                        rule.rule_id,
                        rule.provenance,
                    ),
                }
            )
        if not rules:
            raise PortfolioS3TextOptError(
                f"parent Skill has no editable rules: {capability_id}"
            )
        catalog.append(
            {
                "capability_id": capability_id,
                "rules": rules,
            }
        )
    return catalog


def _skill_for_capability(
    bank: StaticBankArtifact, capability_id: str, label: str
) -> StrictSkillArtifact:
    bank = _revalidate_instance(bank, StaticBankArtifact, label)
    matches = tuple(
        skill for skill in bank.skills if skill.capability_id == capability_id
    )
    if len(matches) != 1:
        raise PortfolioS3TextOptError(
            f"{label} must contain exactly one Skill for {capability_id}"
        )
    return matches[0]


def _patch_fingerprint(
    capability_id: str,
    edits: tuple[
        PortfolioS3TextPatchEdit | PortfolioS3NormalizedTextPatchEdit, ...
    ],
) -> str:
    semantics = []
    for edit in edits:
        semantics.append(
            {
                "op": edit.op,
                "rule_id": edit.rule_id,
                "expected_text_sha256": edit.expected_text_sha256,
                "replacement": edit.replacement,
            }
        )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": PORTFOLIO_S3_TEXTOPT_POLICY_VERSION,
                "capability_id": capability_id,
                "edits": semantics,
            }
        )
    )


def _validate_binding_and_proposal(
    failure_cluster: PortfolioS3FailureClusterBinding,
    proposal: PortfolioS3TextPatchProposal,
) -> None:
    if (
        proposal.capability_id != failure_cluster.capability_id
        or proposal.failure_cluster_id != failure_cluster.failure_cluster_id
        or proposal.failure_cluster_sha256 != failure_cluster.cluster_sha256
    ):
        raise PortfolioS3TextOptError(
            "proposal does not match its failure-cluster identity"
        )
    if not failure_cluster.is_eligible_skill_defect:
        raise PortfolioS3TextOptError(
            "execution lapse or under-supported cluster cannot modify a Skill Body"
        )
    cluster_evidence = set(failure_cluster.evidence_ids)
    cluster_dimensions = set(failure_cluster.addressed_dimensions)
    for edit in proposal.edits:
        if not set(edit.evidence_ids) <= cluster_evidence:
            raise PortfolioS3TextOptError("edit cites evidence outside its cluster")
        if not set(edit.addressed_dimensions) <= cluster_dimensions:
            raise PortfolioS3TextOptError("edit addresses a dimension outside its cluster")
        if (
            edit.support_count < failure_cluster.minimum_support_count
            or edit.support_count > failure_cluster.support_count
        ):
            raise PortfolioS3TextOptError("edit support count is outside cluster bounds")


def _validate_binding_and_patch(
    failure_cluster: PortfolioS3FailureClusterBinding,
    patch: PortfolioS3TextPatchArtifact,
) -> None:
    if (
        patch.capability_id != failure_cluster.capability_id
        or patch.failure_cluster_id != failure_cluster.failure_cluster_id
        or patch.failure_cluster_sha256 != failure_cluster.cluster_sha256
    ):
        raise PortfolioS3TextOptError("normalized patch cluster binding mismatch")
    if not failure_cluster.is_eligible_skill_defect:
        raise PortfolioS3TextOptError(
            "execution lapse or under-supported cluster cannot modify a Skill Body"
        )
    cluster_evidence = set(failure_cluster.evidence_ids)
    cluster_dimensions = set(failure_cluster.addressed_dimensions)
    for edit in patch.edits:
        if not set(edit.evidence_ids) <= cluster_evidence:
            raise PortfolioS3TextOptError("edit cites evidence outside its cluster")
        if not set(edit.addressed_dimensions) <= cluster_dimensions:
            raise PortfolioS3TextOptError("edit addresses a dimension outside its cluster")
        if (
            edit.support_count < failure_cluster.minimum_support_count
            or edit.support_count > failure_cluster.support_count
        ):
            raise PortfolioS3TextOptError("edit support count is outside cluster bounds")


def _validate_reconsideration(
    *,
    fingerprint: str,
    evidence_ids: tuple[str, ...],
    rationale: str | None,
    buffer: PortfolioS3RejectedEditBuffer,
) -> None:
    previous = tuple(
        record for record in buffer.records if record.patch_fingerprint == fingerprint
    )
    if not previous:
        return
    prior_evidence = {
        evidence_id for record in previous for evidence_id in record.evidence_ids
    }
    new_evidence = set(evidence_ids)
    if not prior_evidence < new_evidence or rationale is None:
        raise PortfolioS3TextOptError(
            "rejected patch requires strictly more evidence and reconsideration rationale"
        )


def normalize_portfolio_s3_text_patch(
    *,
    parent_bank: StaticBankArtifact,
    failure_cluster: PortfolioS3FailureClusterBinding,
    proposal: PortfolioS3TextPatchProposal,
    optimization_mode: PortfolioS3OptimizationMode = "diagnostic_l1",
    origin_bank: StaticBankArtifact | None = None,
    rejected_buffer: PortfolioS3RejectedEditBuffer | None = None,
) -> PortfolioS3TextPatchArtifact:
    """Bind raw model output to exact parent bytes and create a trusted patch."""

    failure_cluster = _revalidate_instance(
        failure_cluster,
        PortfolioS3FailureClusterBinding,
        "failure cluster binding",
    )
    proposal = _revalidate_instance(
        proposal, PortfolioS3TextPatchProposal, "S3 text patch proposal"
    )
    _validate_binding_and_proposal(failure_cluster, proposal)
    if optimization_mode == "diagnostic_l1" and len(proposal.edits) != 1:
        raise PortfolioS3TextOptError("diagnostic_l1 requires exactly one edit")
    if optimization_mode == "confirmed_l3" and not 1 <= len(proposal.edits) <= 3:
        raise PortfolioS3TextOptError("confirmed_l3 permits one to three edits")

    parent_skill = _skill_for_capability(
        parent_bank, proposal.capability_id, "parent Bank"
    )
    origin_skill = parent_skill
    if optimization_mode == "confirmed_l3":
        if origin_bank is None:
            raise PortfolioS3TextOptError("confirmed_l3 requires its frozen origin Bank")
        origin_skill = _skill_for_capability(
            origin_bank, proposal.capability_id, "origin Bank"
        )
    elif origin_bank is not None:
        origin_skill = _skill_for_capability(
            origin_bank, proposal.capability_id, "origin Bank"
        )

    parsed = _parse_body(parent_skill.body)
    fingerprint = _patch_fingerprint(proposal.capability_id, proposal.edits)
    patch_evidence = tuple(
        sorted({item for edit in proposal.edits for item in edit.evidence_ids})
    )
    buffer = rejected_buffer or build_empty_portfolio_s3_rejected_edit_buffer()
    buffer = _revalidate_instance(
        buffer, PortfolioS3RejectedEditBuffer, "rejected edit buffer"
    )
    _validate_reconsideration(
        fingerprint=fingerprint,
        evidence_ids=patch_evidence,
        rationale=proposal.reconsideration_rationale,
        buffer=buffer,
    )

    sections: set[str] = set()
    normalized_edits: list[PortfolioS3NormalizedTextPatchEdit] = []
    generated_ids: set[str] = set()
    for index, edit in enumerate(proposal.edits, start=1):
        target = parsed.rules.get(edit.rule_id)
        if target is None:
            raise PortfolioS3TextOptError(f"target rule is missing: {edit.rule_id}")
        line = parsed.lines[target.line_index]
        if sha256_bytes(line.encode("utf-8")) != edit.expected_text_sha256:
            raise PortfolioS3TextOptError(f"target rule hash drifted: {edit.rule_id}")
        if target.section not in _EDITABLE_HEADINGS:
            raise PortfolioS3TextOptError(
                f"target rule is in a protected section: {edit.rule_id}"
            )
        sections.add(target.section)
        generated_rule_id: str | None = None
        if edit.op == "replace_rule":
            if edit.replacement == target.statement:
                raise PortfolioS3TextOptError("replace_rule must change its statement")
        elif edit.op == "insert_after_rule":
            generated_rule_id = f"{_S3_RULE_ID_PREFIX}{fingerprint[:16]}.{index}"
            if generated_rule_id in parsed.rules or generated_rule_id in generated_ids:
                raise PortfolioS3TextOptError("generated S3 rule ID collides")
            generated_ids.add(generated_rule_id)
        elif not _is_compiler_inserted_rule(
            target.rule_id,
            target.provenance,
        ):
            raise PortfolioS3TextOptError(
                "delete_rule may remove only a rule inserted by S3 TextOpt"
            )
        normalized_edits.append(
            PortfolioS3NormalizedTextPatchEdit(
                op=edit.op,
                rule_id=edit.rule_id,
                expected_text_sha256=edit.expected_text_sha256,
                replacement=edit.replacement,
                generated_rule_id=generated_rule_id,
                evidence_ids=edit.evidence_ids,
                support_count=edit.support_count,
                addressed_dimensions=edit.addressed_dimensions,
                rationale=edit.rationale,
            )
        )
    if len(sections) != 1:
        raise PortfolioS3TextOptError("all edits must target the same editable section")

    artifact_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s3-text-patch",
        "policy_version": PORTFOLIO_S3_TEXTOPT_POLICY_VERSION,
        "optimization_mode": optimization_mode,
        "capability_id": proposal.capability_id,
        "failure_cluster_id": proposal.failure_cluster_id,
        "failure_cluster_sha256": proposal.failure_cluster_sha256,
        "parent_skill_sha256": parent_skill.skill_sha256,
        "origin_skill_sha256": origin_skill.skill_sha256,
        "patch_fingerprint": fingerprint,
        "edits": [edit.model_dump(mode="json") for edit in normalized_edits],
        "reconsideration_rationale": proposal.reconsideration_rationale,
    }
    artifact = _validate_model(
        PortfolioS3TextPatchArtifact,
        {
            **artifact_payload,
            "patch_sha256": sha256_bytes(canonical_json_bytes(artifact_payload)),
        },
        "normalized S3 text patch",
    )
    _apply_patch(parent_skill.body, origin_skill.body, artifact)
    return artifact


def compile_portfolio_s3_text_patch(
    *,
    parent_bank: StaticBankArtifact,
    failure_cluster: PortfolioS3FailureClusterBinding,
    proposal_or_patch: PortfolioS3TextPatchProposal | PortfolioS3TextPatchArtifact,
    optimization_mode: PortfolioS3OptimizationMode = "diagnostic_l1",
    origin_bank: StaticBankArtifact | None = None,
    rejected_buffer: PortfolioS3RejectedEditBuffer | None = None,
) -> PortfolioSkillMutation:
    """Compile a raw or normalized patch into the existing full-Body carrier."""

    failure_cluster = _revalidate_instance(
        failure_cluster,
        PortfolioS3FailureClusterBinding,
        "failure cluster binding",
    )
    if isinstance(proposal_or_patch, PortfolioS3TextPatchProposal):
        patch = normalize_portfolio_s3_text_patch(
            parent_bank=parent_bank,
            failure_cluster=failure_cluster,
            proposal=proposal_or_patch,
            optimization_mode=optimization_mode,
            origin_bank=origin_bank,
            rejected_buffer=rejected_buffer,
        )
    else:
        patch = _revalidate_instance(
            proposal_or_patch, PortfolioS3TextPatchArtifact, "S3 text patch"
        )
        if patch.optimization_mode != optimization_mode:
            raise PortfolioS3TextOptError("patch optimization mode mismatch")
        _validate_binding_and_patch(failure_cluster, patch)
        buffer = rejected_buffer or build_empty_portfolio_s3_rejected_edit_buffer()
        buffer = _revalidate_instance(
            buffer, PortfolioS3RejectedEditBuffer, "rejected edit buffer"
        )
        _validate_reconsideration(
            fingerprint=patch.patch_fingerprint,
            evidence_ids=patch.evidence_ids,
            rationale=patch.reconsideration_rationale,
            buffer=buffer,
        )

    parent_skill = _skill_for_capability(
        parent_bank, patch.capability_id, "parent Bank"
    )
    if parent_skill.skill_sha256 != patch.parent_skill_sha256:
        raise PortfolioS3TextOptError("normalized patch parent Skill binding mismatch")
    origin_skill = parent_skill
    if patch.optimization_mode == "confirmed_l3":
        if origin_bank is None:
            raise PortfolioS3TextOptError("confirmed_l3 requires its frozen origin Bank")
        origin_skill = _skill_for_capability(
            origin_bank, patch.capability_id, "origin Bank"
        )
    elif origin_bank is not None:
        origin_skill = _skill_for_capability(
            origin_bank, patch.capability_id, "origin Bank"
        )
    if origin_skill.skill_sha256 != patch.origin_skill_sha256:
        raise PortfolioS3TextOptError("normalized patch origin Skill binding mismatch")

    body = _apply_patch(parent_skill.body, origin_skill.body, patch)
    try:
        return PortfolioSkillMutation(
            capability_id=patch.capability_id,
            parent_skill_sha256=parent_skill.skill_sha256,
            body=body,
        )
    except ValidationError as error:
        raise PortfolioS3TextOptError("compiled Body mutation is invalid") from error


def _apply_patch(
    parent_body: str,
    origin_body: str,
    patch: PortfolioS3TextPatchArtifact,
) -> str:
    parsed = _parse_body(parent_body)
    origin = _parse_body(origin_body)
    working = list(parsed.lines)
    positions = {rule_id: rule.line_index for rule_id, rule in parsed.rules.items()}

    anchor_sections: set[str] = set()
    for index, edit in enumerate(patch.edits, start=1):
        target = parsed.rules.get(edit.rule_id)
        if target is None:
            raise PortfolioS3TextOptError(f"target rule is missing: {edit.rule_id}")
        target_line = parsed.lines[target.line_index]
        if sha256_bytes(target_line.encode("utf-8")) != edit.expected_text_sha256:
            raise PortfolioS3TextOptError(f"target rule hash drifted: {edit.rule_id}")
        if target.section not in _EDITABLE_HEADINGS:
            raise PortfolioS3TextOptError(
                f"target rule is in a protected section: {edit.rule_id}"
            )
        anchor_sections.add(target.section)
        if edit.op == "replace_rule" and edit.replacement == target.statement:
            raise PortfolioS3TextOptError("replace_rule must change its statement")
        if edit.op == "insert_after_rule":
            expected_id = f"{_S3_RULE_ID_PREFIX}{patch.patch_fingerprint[:16]}.{index}"
            if edit.generated_rule_id != expected_id:
                raise PortfolioS3TextOptError("generated S3 rule ID is invalid")
            if edit.generated_rule_id in parsed.rules:
                raise PortfolioS3TextOptError("generated S3 rule ID collides")
        if edit.op == "delete_rule" and not _is_compiler_inserted_rule(
            target.rule_id,
            target.provenance,
        ):
            raise PortfolioS3TextOptError(
                "delete_rule may remove only a rule inserted by S3 TextOpt"
            )
    if len(anchor_sections) != 1:
        raise PortfolioS3TextOptError("all edits must target the same editable section")

    for edit in patch.edits:
        line_index = positions.get(edit.rule_id)
        if line_index is None:
            raise PortfolioS3TextOptError(f"target rule is missing: {edit.rule_id}")
        line = working[line_index]
        if sha256_bytes(line.encode("utf-8")) != edit.expected_text_sha256:
            raise PortfolioS3TextOptError(f"target rule hash drifted: {edit.rule_id}")
        match = _RULE_LINE_RE.fullmatch(line)
        if match is None:
            raise PortfolioS3TextOptError("target is not a canonical rule line")
        if edit.op == "replace_rule":
            assert edit.replacement is not None
            working[line_index] = (
                f"- [{edit.rule_id}] {edit.replacement} {match.group('provenance')}"
            )
        elif edit.op == "insert_after_rule":
            assert edit.replacement is not None and edit.generated_rule_id is not None
            inserted = (
                f"- [{edit.generated_rule_id}] {edit.replacement} "
                f"(source: {_S3_SOURCE_PREFIX}{patch.patch_fingerprint})"
            )
            # Canonical Bodies separate every logical line with one blank line.
            working[line_index + 1 : line_index + 1] = ["", inserted]
            for rule_id, position in tuple(positions.items()):
                if position > line_index:
                    positions[rule_id] = position + 2
            positions[edit.generated_rule_id] = line_index + 2
        else:
            if not _is_compiler_inserted_rule(
                edit.rule_id,
                match.group("provenance"),
            ):
                raise PortfolioS3TextOptError(
                    "delete_rule may remove only a rule inserted by S3 TextOpt"
                )
            start = line_index - 1 if line_index > 0 and working[line_index - 1] == "" else line_index
            removed = line_index - start + 1
            del working[start : line_index + 1]
            del positions[edit.rule_id]
            for rule_id, position in tuple(positions.items()):
                if position > line_index:
                    positions[rule_id] = position - removed

    body = "\n".join(working) + "\n"
    if body == parent_body:
        raise PortfolioS3TextOptError("S3 text patch produced no Body change")
    result = _parse_body(body)

    for heading, source_bytes in parsed.sections.items():
        if heading not in _EDITABLE_HEADINGS and result.sections.get(heading) != source_bytes:
            raise PortfolioS3TextOptError(f"protected Body section drifted: {heading}")
    if set(result.sections) != set(parsed.sections):
        raise PortfolioS3TextOptError("Body section set drifted")
    if (
        result.sections[_OUTPUT_CONTRACT_HEADING]
        != parsed.sections[_OUTPUT_CONTRACT_HEADING]
        or sha256_bytes(result.sections[_OUTPUT_CONTRACT_HEADING].encode("utf-8"))
        != sha256_bytes(parsed.sections[_OUTPUT_CONTRACT_HEADING].encode("utf-8"))
    ):
        raise PortfolioS3TextOptError("output contract bytes drifted")

    result_size = len(body.encode("utf-8"))
    if patch.optimization_mode == "diagnostic_l1":
        if result_size * 100 > len(parent_body.encode("utf-8")) * 110:
            raise PortfolioS3TextOptError("diagnostic_l1 Body growth exceeds 10 percent")
    elif result_size * 100 > len(origin_body.encode("utf-8")) * 120:
        raise PortfolioS3TextOptError("confirmed_l3 Body growth exceeds 20 percent")
    if patch.optimization_mode == "confirmed_l3" and set(origin.sections) != set(
        parsed.sections
    ):
        raise PortfolioS3TextOptError("origin and parent Body section sets differ")
    if patch.optimization_mode == "confirmed_l3":
        for heading, origin_bytes in origin.sections.items():
            if (
                heading not in _EDITABLE_HEADINGS
                and parsed.sections[heading] != origin_bytes
            ):
                raise PortfolioS3TextOptError(
                    f"protected parent section drifted from frozen origin: {heading}"
                )
    return body


def _parse_canonical_artifact(
    content: bytes | str,
    label: str,
) -> tuple[object, bytes]:
    try:
        data = content.encode("utf-8") if isinstance(content, str) else content
        return parse_canonical_json(data, label=label), data
    except (ArtifactFormatError, UnicodeEncodeError) as error:
        raise PortfolioS3TextOptError(f"{label} is not canonical JSON") from error


def _validate_model(model_type, value: object, label: str):
    try:
        return model_type.model_validate(value, strict=True)
    except ValidationError as error:
        raise PortfolioS3TextOptError(f"{label} is invalid") from error


def _revalidate_instance(value, model_type, label: str):
    try:
        if not isinstance(value, model_type):
            raise PortfolioS3TextOptError(f"{label} has the wrong type")
        validated = model_type.model_validate(value.model_dump(mode="python"), strict=True)
    except ValidationError as error:
        raise PortfolioS3TextOptError(f"{label} is invalid") from error
    if hasattr(value, "canonical_bytes") and validated.canonical_bytes() != value.canonical_bytes():
        raise PortfolioS3TextOptError(f"{label} canonical bytes drifted")
    return validated


__all__ = [
    "PORTFOLIO_S3_REJECTED_EDIT_BUFFER_POLICY_VERSION",
    "PORTFOLIO_S3_TEXTOPT_POLICY_VERSION",
    "PORTFOLIO_S3_TEXTOPT_PROPOSAL_POLICY_VERSION",
    "PortfolioS3DeleteRuleEdit",
    "PortfolioS3FailureClusterBinding",
    "PortfolioS3InsertAfterRuleEdit",
    "PortfolioS3NormalizedTextPatchEdit",
    "PortfolioS3RejectedEditBuffer",
    "PortfolioS3RejectedEditRecord",
    "PortfolioS3ReplaceRuleEdit",
    "PortfolioS3TextOptError",
    "PortfolioS3TextPatchArtifact",
    "PortfolioS3TextPatchProposal",
    "append_portfolio_s3_rejected_edit",
    "build_empty_portfolio_s3_rejected_edit_buffer",
    "build_portfolio_s3_editable_rule_catalog",
    "build_portfolio_s3_failure_cluster_binding",
    "compile_portfolio_s3_text_patch",
    "normalize_portfolio_s3_text_patch",
    "parse_portfolio_s3_rejected_edit_buffer",
    "parse_portfolio_s3_text_patch_artifact",
    "parse_portfolio_s3_text_patch_proposal",
    "portfolio_s3_text_patch_proposal_json_schema",
]
