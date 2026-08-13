"""Parent-protected sparse compilation for Portfolio S1 Creator output.

The model is allowed to choose ``inherit`` or supply author prose for a
capability.  Everything else remains owned by the trusted compiler.  In
particular, inherited Skill objects are copied from the parent Bank without
being rendered again, while patched Skill objects are compiled through the
normal AuthoringInput compiler and checked against the parent's immutable
metadata and body sections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain import static_authoring as static_authoring_module
from skillchain.evaluation.portfolio_treatments import (
    compile_portfolio_s1_creator_bank,
)
from skillchain.static_authoring import (
    AuthoringContentPayload,
    AuthoringDraftBundle,
    AuthoringInput,
    CapabilityAuthoringContent,
    DraftToolStepContent,
    StaticBankArtifact,
    StrictSkillArtifact,
    normalize_authoring_content_payload,
)
from skillchain.runners.assistant_deterministic_contract import (
    DETERMINISTIC_SEMANTIC_POLICY_VERSION,
    SEMANTIC_POLICY_CAPABILITIES,
    DeterministicSemanticPolicy,
    parse_deterministic_semantic_policy,
    render_deterministic_semantic_policy,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    parse_strict_json,
    read_stable_regular_file,
    sha256_bytes,
)


S1_SPARSE_PATCH_POLICY_VERSION = "portfolio-s1-sparse-patch-v1"
S1_SPARSE_COMPILATION_POLICY_VERSION = "portfolio-s1-sparse-compilation-v1"
S1_SPARSE_SCREENED_BANK_POLICY_VERSION = "portfolio-s1-screened-sparse-bank-v1"
ENCYCLOPEDIA_CAPABILITY = "knowledge.visual_encyclopedia"
ENCYCLOPEDIA_FALLBACK_MARKER = "not enough evidence"
ENCYCLOPEDIA_TOOL_SEQUENCE = ("object_detect", "encyclopedia_lookup")
FORBIDDEN_CONTROL_TOKENS = ("<|begin|>", "<|im_end|>")
_CONTROL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_-]+\|>")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_STEP_RE = re.compile(r"^(\d+)\. \[([a-z][a-z0-9_]*)\] (.+)$")
_MUTABLE_BODY_SECTIONS = frozenset(
    {
        "# Objective",
        "## Tool procedure",
        "## Authored public source citations",
        "## Authored fallback instruction",
    }
)
S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS = (
    "bank",
    "corpus",
    "eval",
    "evaluation",
    "gold",
    "judge",
    "judges",
    "label",
    "labels",
    "rubric",
    "rubrics",
    "trajectories",
    "trajectory",
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class S1SparsePatchError(ValueError):
    """Sparse S1 output or its compiled Bank violated the frozen contract."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def sparse_author_content_lexical_guard() -> dict[str, object]:
    """Project the exact downstream authored-prose scanner for the Creator."""

    trusted = static_authoring_module._FORBIDDEN_STRONG  # noqa: SLF001
    if any(
        trusted.search(f"safe {word} prose") is None
        for word in S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS
    ):
        raise S1SparsePatchError("trusted authored-prose scanner drifted")
    return {
        "scope": [
            "skills[].patch.objective",
            "skills[].patch.steps[].instruction",
            "skills[].patch.fallback_instruction",
        ],
        "matching": "python_re_ignorecase_exact_pattern",
        "forbidden_whole_words": list(S1_AUTHOR_CONTENT_FORBIDDEN_WHOLE_WORDS),
        "trusted_validator_regex": trusted.pattern,
        "trusted_validator_regex_sha256": sha256_bytes(trusted.pattern.encode("utf-8")),
        "forbidden_path_reference_rule": (
            "response/gate/test/result whole word and path marker must occur "
            "in the same path token or as adjacent path components"
        ),
        "required_detector_prediction_phrase": "predicted class name",
    }


class SparseSemanticPolicyV1(_StrictFrozenModel):
    """Creator-owned semantic selector consumed by deterministic runtime."""

    schema_version: Literal[1] = 1
    policy_version: Literal["core-fast-semantic-policy-v2"] = (
        DETERMINISTIC_SEMANTIC_POLICY_VERSION
    )
    evidence_terms: tuple[str, ...] = Field(default=(), max_length=16)
    require_all_terms: bool = False
    abstain_when_no_evidence: Literal[True] = True
    ocr_extraction_plan: Literal["all-lines", "literal-material-spans"] = "all-lines"

    @field_validator("evidence_terms", mode="before")
    @classmethod
    def _coerce_terms(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            # Ordering is representation-only for a term set and cannot be
            # expressed by JSON Schema. Canonicalize it at the compiler edge;
            # the validator below still rejects duplicates and malformed terms.
            return tuple(sorted(value, key=str.casefold))
        return value

    @model_validator(mode="after")
    def _validate_terms(self) -> Self:
        if self.evidence_terms != tuple(
            sorted(set(self.evidence_terms), key=str.casefold)
        ) or any(not item or item != item.strip() for item in self.evidence_terms):
            raise ValueError("semantic evidence terms must be canonical and unique")
        return self


class SparseSkillContentPatchV1(_StrictFrozenModel):
    """Only the author-judgment fields accepted for one patched Skill."""

    objective: str
    steps: tuple[DraftToolStepContent, ...] = Field(min_length=1, max_length=16)
    fallback_instruction: str
    citation_source_ids: tuple[str, ...]
    semantic_policy: SparseSemanticPolicyV1

    @field_validator("steps", "citation_source_ids", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        if not self.objective.strip() or not self.fallback_instruction.strip():
            raise ValueError("sparse patch prose must be non-blank")
        if self.citation_source_ids != tuple(sorted(set(self.citation_source_ids))):
            raise ValueError("sparse patch citation ids must be sorted and unique")
        authored = canonical_json_bytes(self.model_dump(mode="json")).decode("utf-8")
        token = _CONTROL_TOKEN_RE.search(authored)
        if token is not None:
            raise ValueError(f"forbidden control token: {token.group(0)}")
        return self


class SparseSkillDraftV1(_StrictFrozenModel):
    capability_id: str
    action: Literal["inherit", "patch"]
    parent_skill_sha256: Sha256
    patch: SparseSkillContentPatchV1 | None = None

    @model_validator(mode="after")
    def _validate_action(self) -> Self:
        if (self.action == "inherit") != (self.patch is None):
            raise ValueError("inherit requires null patch and patch requires content")
        return self


class SparsePatchDraftPayloadV1(_StrictFrozenModel):
    """The hash-free, structured model output for a sparse S1 proposal."""

    schema_version: Literal[1]
    skills: tuple[SparseSkillDraftV1, ...] = Field(min_length=1)

    @field_validator("skills", mode="before")
    @classmethod
    def _coerce_skills(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_skills(self) -> Self:
        capabilities = tuple(item.capability_id for item in self.skills)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("sparse Skill drafts must be sorted and unique")
        if not any(item.action == "patch" for item in self.skills):
            raise ValueError("a sparse S1 proposal must patch at least one Skill")
        return self


class SparsePatchDraftV1(_StrictFrozenModel):
    """Trusted, lineage-bound sparse draft after parsing model output."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-sparse-patch-draft"] = (
        "portfolio-s1-sparse-patch-draft"
    )
    policy_version: Literal[S1_SPARSE_PATCH_POLICY_VERSION] = (
        S1_SPARSE_PATCH_POLICY_VERSION
    )
    parent_bank_sha256: Sha256
    authoring_input_sha256: Sha256
    feedback_bundle_sha256: Sha256
    skills: tuple[SparseSkillDraftV1, ...] = Field(min_length=1)
    draft_sha256: Sha256

    @field_validator("skills", mode="before")
    @classmethod
    def _coerce_skills(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_draft(self) -> Self:
        capabilities = tuple(item.capability_id for item in self.skills)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("bound sparse Skill drafts must be sorted and unique")
        if not any(item.action == "patch" for item in self.skills):
            raise ValueError("a bound sparse S1 draft must contain a patch")
        payload = self.model_dump(mode="json", exclude={"draft_sha256"})
        if self.draft_sha256 != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("sparse S1 draft SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class SparseSkillCompilationBindingV1(_StrictFrozenModel):
    capability_id: str
    action: Literal["inherit", "patch"]
    parent_skill_sha256: Sha256
    parent_skill_bytes_sha256: Sha256
    candidate_skill_sha256: Sha256
    candidate_skill_bytes_sha256: Sha256
    inherited_bytes_exact: bool

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        expected = self.action == "inherit"
        if self.inherited_bytes_exact != expected:
            raise ValueError("sparse Skill byte-preservation disposition drifted")
        if expected and (
            self.parent_skill_sha256 != self.candidate_skill_sha256
            or self.parent_skill_bytes_sha256 != self.candidate_skill_bytes_sha256
        ):
            raise ValueError("inherited Skill bytes are not identical")
        return self


class SparseCompilationReceiptV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-sparse-compilation-receipt"] = (
        "portfolio-s1-sparse-compilation-receipt"
    )
    policy_version: Literal[S1_SPARSE_COMPILATION_POLICY_VERSION] = (
        S1_SPARSE_COMPILATION_POLICY_VERSION
    )
    sparse_draft_sha256: Sha256
    sparse_draft_file_sha256: Sha256
    parent_bank_sha256: Sha256
    feedback_bundle_sha256: Sha256
    authoring_input_sha256: Sha256
    tool_registry_sha256: Sha256
    tool_registry_runtime_sha256: Sha256
    compiler_identity_sha256: Sha256
    sparse_compiler_file_sha256: Sha256
    compiler_owned_sections_sha256: Sha256
    bindings: tuple[SparseSkillCompilationBindingV1, ...]
    candidate_bank_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("bindings", mode="before")
    @classmethod
    def _coerce_bindings(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        capabilities = tuple(item.capability_id for item in self.bindings)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("sparse compilation bindings must be sorted and unique")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("sparse compilation receipt SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class SparseScreenedSkillBindingV1(_StrictFrozenModel):
    capability_id: str
    disposition: Literal["retained_patch", "reverted_to_parent", "unchanged_parent"]
    parent_skill_sha256: Sha256
    creator_candidate_skill_sha256: Sha256
    screened_skill_sha256: Sha256
    parent_bytes_restored: bool

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        restored = self.disposition != "retained_patch"
        if self.parent_bytes_restored != restored:
            raise ValueError("screened sparse restoration flag drifted")
        if restored and self.screened_skill_sha256 != self.parent_skill_sha256:
            raise ValueError("screened sparse parent restoration is not exact")
        if (
            self.disposition == "retained_patch"
            and self.screened_skill_sha256 != self.creator_candidate_skill_sha256
        ):
            raise ValueError("retained sparse patch differs from Creator candidate")
        return self


class SparseScreenedBankReceiptV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-screened-sparse-bank-receipt"] = (
        "portfolio-s1-screened-sparse-bank-receipt"
    )
    policy_version: Literal[S1_SPARSE_SCREENED_BANK_POLICY_VERSION] = (
        S1_SPARSE_SCREENED_BANK_POLICY_VERSION
    )
    parent_bank_sha256: Sha256
    creator_candidate_bank_sha256: Sha256
    creator_compilation_receipt_sha256: Sha256
    development_screen_sha256: Sha256
    retained_capability_ids: tuple[str, ...]
    bindings: tuple[SparseScreenedSkillBindingV1, ...]
    screened_bank_sha256: Sha256
    receipt_sha256: Sha256

    @field_validator("retained_capability_ids", "bindings", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.retained_capability_ids != tuple(
            sorted(set(self.retained_capability_ids))
        ):
            raise ValueError("retained capability ids must be sorted and unique")
        capabilities = tuple(item.capability_id for item in self.bindings)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("screened sparse bindings must be sorted and unique")
        retained = tuple(
            item.capability_id
            for item in self.bindings
            if item.disposition == "retained_patch"
        )
        if retained != self.retained_capability_ids:
            raise ValueError("screened sparse retained bindings disagree")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != sha256_bytes(canonical_json_bytes(payload)):
            raise ValueError("screened sparse receipt SHA-256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CompiledSparseS1Candidate:
    bank: StaticBankArtifact
    draft: SparsePatchDraftV1
    semantic_draft_bundle: AuthoringDraftBundle
    receipt: SparseCompilationReceiptV1


@dataclass(frozen=True)
class ScreenedSparseS1Candidate:
    bank: StaticBankArtifact
    receipt: SparseScreenedBankReceiptV1


def sparse_patch_output_json_schema(
    *,
    parent_skill_sha256_by_capability: Mapping[str, str],
    frozen_objective_by_capability: Mapping[str, str] | None = None,
    frozen_content_by_capability: Mapping[str, Mapping[str, object]] | None = None,
    patch_capabilities: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return the strict structured-output envelope for one sparse proposal."""

    capability_ids = tuple(sorted(parent_skill_sha256_by_capability))
    if capability_ids != tuple(sorted(set(capability_ids))) or not capability_ids:
        raise S1SparsePatchError("capability ids must be non-empty, sorted, and unique")
    if any(
        not _SHA_RE.fullmatch(parent_skill_sha256_by_capability[item])
        for item in capability_ids
    ):
        raise S1SparsePatchError("parent Skill SHA-256 mapping is invalid")
    allowed_patches = (
        tuple(sorted(SEMANTIC_POLICY_CAPABILITIES))
        if patch_capabilities is None
        else patch_capabilities
    )
    if (
        allowed_patches != tuple(sorted(set(allowed_patches)))
        or not set(allowed_patches) <= set(capability_ids)
        or not set(allowed_patches) <= set(SEMANTIC_POLICY_CAPABILITIES)
    ):
        raise S1SparsePatchError("semantic patch capabilities are invalid")
    if frozen_objective_by_capability is not None and (
        set(frozen_objective_by_capability) != set(capability_ids)
        or any(
            not isinstance(frozen_objective_by_capability[item], str)
            or not frozen_objective_by_capability[item].strip()
            for item in capability_ids
        )
    ):
        raise S1SparsePatchError("frozen sparse objectives are invalid")
    if frozen_content_by_capability is not None and (
        set(frozen_content_by_capability) != set(capability_ids)
        or any(
            set(frozen_content_by_capability[item])
            != {
                "objective",
                "steps",
                "fallback_instruction",
                "citation_source_ids",
            }
            for item in capability_ids
        )
    ):
        raise S1SparsePatchError("frozen sparse content is invalid")
    tool_step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["instruction", "tool_name", "success_rule_ids"],
        "properties": {
            "instruction": {"type": "string"},
            "tool_name": {"type": "string"},
            "success_rule_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
    }
    patch = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "objective",
            "steps",
            "fallback_instruction",
            "citation_source_ids",
            "semantic_policy",
        ],
        "properties": {
            "objective": {"type": "string"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": tool_step,
            },
            "fallback_instruction": {"type": "string"},
            "citation_source_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "semantic_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema_version",
                    "policy_version",
                    "evidence_terms",
                    "require_all_terms",
                    "abstain_when_no_evidence",
                    "ocr_extraction_plan",
                ],
                "properties": {
                    "schema_version": {"type": "integer", "enum": [1]},
                    "policy_version": {
                        "type": "string",
                        "enum": [DETERMINISTIC_SEMANTIC_POLICY_VERSION],
                    },
                    "evidence_terms": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1, "maxLength": 80},
                    },
                    "require_all_terms": {"type": "boolean"},
                    "abstain_when_no_evidence": {"type": "boolean", "enum": [True]},
                    "ocr_extraction_plan": {
                        "type": "string",
                        "enum": ["all-lines", "literal-material-spans"],
                    },
                },
            },
        },
    }
    branches: list[dict[str, object]] = []
    for capability_id in capability_ids:
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "capability_id",
                    "action",
                    "parent_skill_sha256",
                ],
                "properties": {
                    "capability_id": {
                        "type": "string",
                        "enum": [capability_id],
                    },
                    "action": {"type": "string", "enum": ["inherit"]},
                    "parent_skill_sha256": {
                        "type": "string",
                        "enum": [parent_skill_sha256_by_capability[capability_id]],
                    },
                },
            }
        )
        if capability_id not in allowed_patches:
            continue
        capability_patch = patch
        if frozen_content_by_capability is not None:
            frozen = frozen_content_by_capability[capability_id]
            capability_patch = {
                **patch,
                "properties": {
                    **patch["properties"],
                    # Structured Outputs accepts primitive enums but rejects
                    # enums whose values are arrays/objects. The parent-bound
                    # binder below still checks steps/citations byte-exactly.
                    "objective": {
                        "type": "string",
                        "enum": [frozen["objective"]],
                    },
                    "fallback_instruction": {
                        "type": "string",
                        "enum": [frozen["fallback_instruction"]],
                    },
                },
            }
        elif frozen_objective_by_capability is not None:
            capability_patch = {
                **patch,
                "properties": {
                    **patch["properties"],
                    "objective": {
                        "type": "string",
                        "enum": [frozen_objective_by_capability[capability_id]],
                    },
                },
            }
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "capability_id",
                    "action",
                    "parent_skill_sha256",
                    "patch",
                ],
                "properties": {
                    "capability_id": {
                        "type": "string",
                        "enum": [capability_id],
                    },
                    "action": {"type": "string", "enum": ["patch"]},
                    "parent_skill_sha256": {
                        "type": "string",
                        "enum": [parent_skill_sha256_by_capability[capability_id]],
                    },
                    "patch": capability_patch,
                },
            }
        )
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "skills"],
        "properties": {
            "schema_version": {"type": "integer", "enum": [1]},
            "skills": {
                "type": "array",
                "minItems": len(capability_ids),
                "maxItems": len(capability_ids),
                "items": {"anyOf": branches},
            },
        },
    }
    return schema


def decode_sparse_parent_content(
    parent_bank: StaticBankArtifact,
    authoring_input: AuthoringInput,
) -> tuple[CapabilityAuthoringContent, ...]:
    """Expose the byte-proven parent authoring projection to sparse callers."""

    return _decode_parent_authoring_content(parent_bank, authoring_input)


def bind_sparse_patch_draft(
    raw_final: bytes,
    *,
    parent_bank: StaticBankArtifact,
    authoring_input: AuthoringInput,
    feedback_bundle_sha256: str,
) -> SparsePatchDraftV1:
    """Parse one model response and bind every entry to trusted parent bytes."""

    if not _SHA_RE.fullmatch(feedback_bundle_sha256):
        raise S1SparsePatchError("Feedback bundle SHA-256 is invalid")
    try:
        raw = parse_strict_json(raw_final, label="S1 sparse Creator final output")
        if not isinstance(raw, dict):
            raise S1SparsePatchError("S1 sparse Creator output must be an object")
        payload = SparsePatchDraftPayloadV1.model_validate(raw, strict=True)
    except ValidationError as error:
        if "forbidden control token" in str(error):
            raise S1SparsePatchError(
                "S1 sparse Creator output contains a forbidden control token"
            ) from error
        raise S1SparsePatchError("S1 sparse Creator output is invalid") from error
    except ArtifactFormatError as error:
        raise S1SparsePatchError("S1 sparse Creator output is invalid") from error
    parent_by_capability = _skill_by_capability(parent_bank)
    if tuple(item.capability_id for item in payload.skills) != tuple(
        sorted(parent_by_capability)
    ):
        raise S1SparsePatchError(
            "sparse Creator output must cover every parent capability exactly once"
        )
    for item in payload.skills:
        parent = parent_by_capability[item.capability_id]
        if item.parent_skill_sha256 != parent.skill_sha256:
            raise S1SparsePatchError(
                f"parent Skill binding drifted: {item.capability_id}"
            )
        if item.action == "patch":
            assert item.patch is not None
            _validate_patch_contract(item.capability_id, item.patch, parent)
    unsigned = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-sparse-patch-draft",
        "policy_version": S1_SPARSE_PATCH_POLICY_VERSION,
        "parent_bank_sha256": parent_bank.bank_sha256,
        "authoring_input_sha256": authoring_input.input_sha256,
        "feedback_bundle_sha256": feedback_bundle_sha256,
        "skills": [item.model_dump(mode="json") for item in payload.skills],
    }
    return SparsePatchDraftV1.model_validate(
        {
            **unsigned,
            "draft_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
        },
        strict=True,
    )


def load_sparse_patch_draft(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> SparsePatchDraftV1:
    """Load one canonical, externally SHA-bound sparse draft artifact."""

    return _load_canonical_sparse_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=SparsePatchDraftV1,
        label="S1 sparse patch draft",
    )


def load_sparse_compilation_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> SparseCompilationReceiptV1:
    """Load one canonical, externally SHA-bound sparse compilation receipt."""

    return _load_canonical_sparse_model(
        path,
        expected_file_sha256=expected_file_sha256,
        model_type=SparseCompilationReceiptV1,
        label="S1 sparse compilation receipt",
    )


def _load_canonical_sparse_model(
    path: str | Path,
    *,
    expected_file_sha256: str,
    model_type: type[SparsePatchDraftV1] | type[SparseCompilationReceiptV1],
    label: str,
):
    if not _SHA_RE.fullmatch(expected_file_sha256):
        raise S1SparsePatchError(f"{label} expected SHA-256 is invalid")
    content = read_stable_regular_file(
        path,
        label=label,
        max_bytes=8 * 1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise S1SparsePatchError(f"{label} file SHA-256 drifted")
    try:
        value = model_type.model_validate_json(content, strict=True)
    except ValidationError as error:
        raise S1SparsePatchError(f"{label} is invalid") from error
    if value.canonical_bytes() != content:
        raise S1SparsePatchError(f"{label} is not canonical")
    return value


def compile_sparse_s1_candidate(
    *,
    parent_bank: StaticBankArtifact,
    authoring_input: AuthoringInput,
    sparse_draft: SparsePatchDraftV1,
    tool_registry_runtime_sha256: str,
) -> CompiledSparseS1Candidate:
    """Compile a sparse draft and prove all inherited/frozen bytes are stable."""

    try:
        parent_bank = StaticBankArtifact.model_validate(
            parent_bank.model_dump(mode="python"), strict=True
        )
        authoring_input = AuthoringInput.model_validate(
            authoring_input.model_dump(mode="python"), strict=True
        )
        sparse_draft = SparsePatchDraftV1.model_validate(
            sparse_draft.model_dump(mode="python"), strict=True
        )
    except (AttributeError, ValidationError) as error:
        raise S1SparsePatchError("sparse compiler inputs are invalid") from error
    _validate_parent_lineage(
        parent_bank=parent_bank,
        authoring_input=authoring_input,
        sparse_draft=sparse_draft,
        tool_registry_runtime_sha256=tool_registry_runtime_sha256,
    )
    parent_content = _decode_parent_authoring_content(parent_bank, authoring_input)
    patch_by_capability = {
        item.capability_id: item.patch
        for item in sparse_draft.skills
        if item.action == "patch"
    }
    composed_content = tuple(
        _as_authoring_content(
            capability_id=content.capability_id,
            patch=patch_by_capability.get(content.capability_id),
            parent_content=content,
        )
        for content in parent_content
    )
    try:
        semantic_bundle = normalize_authoring_content_payload(
            AuthoringContentPayload(schema_version=2, drafts=composed_content),
            authoring_input=authoring_input,
        )
        compiled_full = compile_portfolio_s1_creator_bank(
            authoring_input,
            semantic_bundle,
            tool_registry_runtime_sha256=tool_registry_runtime_sha256,
        )
    except Exception as error:
        raise S1SparsePatchError(
            "sparse patch failed trusted AuthoringInput compilation"
        ) from error

    parent_by_capability = _skill_by_capability(parent_bank)
    compiled_by_capability = _skill_by_capability(compiled_full)
    entry_by_capability = {item.capability_id: item for item in sparse_draft.skills}
    selected_skills: list[StrictSkillArtifact] = []
    bindings: list[SparseSkillCompilationBindingV1] = []
    frozen_section_records: list[dict[str, Any]] = []
    for capability_id in sorted(parent_by_capability):
        parent = parent_by_capability[capability_id]
        compiled = compiled_by_capability[capability_id]
        entry = entry_by_capability[capability_id]
        parent_bytes = canonical_json_bytes(parent.model_dump(mode="json"))
        if entry.action == "inherit":
            selected = parent
            if canonical_json_bytes(selected.model_dump(mode="json")) != parent_bytes:
                raise S1SparsePatchError(
                    f"inherited Skill bytes drifted: {capability_id}"
                )
        else:
            _assert_patch_only_changed_mutable_fields(parent, compiled)
            assert entry.patch is not None
            parent_content_item = next(
                item for item in parent_content if item.capability_id == capability_id
            )
            if (
                entry.patch.objective != parent_content_item.objective
                or entry.patch.steps != parent_content_item.steps
                or entry.patch.fallback_instruction
                != parent_content_item.fallback_instruction
                or entry.patch.citation_source_ids
                != parent_content_item.citation_source_ids
            ):
                raise S1SparsePatchError(
                    f"patch changed runtime-owned prose: {capability_id}"
                )
            parent_policy = parse_deterministic_semantic_policy(
                parent.body, capability_id=capability_id
            )
            if (
                parent_policy.evidence_terms
                or parent_policy.ocr_extraction_plan != "all-lines"
            ):
                raise S1SparsePatchError(
                    f"parent already contains a semantic treatment: {capability_id}"
                )
            policy = _semantic_policy_for_patch(capability_id, entry.patch)
            if not policy.evidence_terms and policy.ocr_extraction_plan == "all-lines":
                raise S1SparsePatchError(
                    f"patch semantic policy is a no-op: {capability_id}"
                )
            selected_payload = parent.model_dump(mode="json")
            selected_payload["version"] = parent.version + 1
            selected_payload["parent_skill_sha256"] = parent.skill_sha256
            selected_payload["body"] = (
                parent.body + render_deterministic_semantic_policy(policy)
            )
            selected_payload["skill_sha256"] = sha256_bytes(
                canonical_json_bytes(
                    {
                        key: value
                        for key, value in selected_payload.items()
                        if key != "skill_sha256"
                    }
                )
            )
            selected = StrictSkillArtifact.model_validate(selected_payload, strict=True)
        selected_bytes = canonical_json_bytes(selected.model_dump(mode="json"))
        frozen_sections = _body_sections(parent.body)
        frozen_section_records.append(
            {
                "capability_id": capability_id,
                "sections": {
                    heading: body
                    for heading, body in sorted(frozen_sections.items())
                    if heading not in _MUTABLE_BODY_SECTIONS
                },
            }
        )
        selected_skills.append(selected)
        bindings.append(
            SparseSkillCompilationBindingV1(
                capability_id=capability_id,
                action=entry.action,
                parent_skill_sha256=parent.skill_sha256,
                parent_skill_bytes_sha256=sha256_bytes(parent_bytes),
                candidate_skill_sha256=selected.skill_sha256,
                candidate_skill_bytes_sha256=sha256_bytes(selected_bytes),
                inherited_bytes_exact=entry.action == "inherit",
            )
        )

    bank_payload = {
        "schema_version": parent_bank.schema_version,
        "baseline_kind": parent_bank.baseline_kind,
        "construction_identity_sha256": sparse_draft.draft_sha256,
        "construction_identity_policy": parent_bank.construction_identity_policy,
        "runtime_binding_policy": parent_bank.runtime_binding_policy,
        "compiler": parent_bank.compiler.model_dump(mode="json"),
        "tool_registry_sha256": parent_bank.tool_registry_sha256,
        "tool_registry_runtime_sha256": parent_bank.tool_registry_runtime_sha256,
        "skills": [item.model_dump(mode="json") for item in selected_skills],
        "capability_map": [
            item.model_dump(mode="json") for item in parent_bank.capability_map
        ],
    }
    candidate = StaticBankArtifact.model_validate(
        {
            **bank_payload,
            "bank_sha256": sha256_bytes(canonical_json_bytes(bank_payload)),
        },
        strict=True,
    )
    compiler_identity_sha256 = sha256_bytes(
        canonical_json_bytes(parent_bank.compiler.model_dump(mode="json"))
    )
    sparse_compiler_file_sha256 = sha256_bytes(
        read_stable_regular_file(Path(__file__), label="S1 sparse compiler source")
    )
    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-sparse-compilation-receipt",
        "policy_version": S1_SPARSE_COMPILATION_POLICY_VERSION,
        "sparse_draft_sha256": sparse_draft.draft_sha256,
        "sparse_draft_file_sha256": sha256_bytes(sparse_draft.canonical_bytes()),
        "parent_bank_sha256": parent_bank.bank_sha256,
        "feedback_bundle_sha256": sparse_draft.feedback_bundle_sha256,
        "authoring_input_sha256": authoring_input.input_sha256,
        "tool_registry_sha256": parent_bank.tool_registry_sha256,
        "tool_registry_runtime_sha256": parent_bank.tool_registry_runtime_sha256,
        "compiler_identity_sha256": compiler_identity_sha256,
        "sparse_compiler_file_sha256": sparse_compiler_file_sha256,
        "compiler_owned_sections_sha256": sha256_bytes(
            canonical_json_bytes(frozen_section_records)
        ),
        "bindings": [item.model_dump(mode="json") for item in bindings],
        "candidate_bank_sha256": candidate.bank_sha256,
    }
    receipt = SparseCompilationReceiptV1.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        },
        strict=True,
    )
    return CompiledSparseS1Candidate(
        bank=candidate,
        draft=sparse_draft,
        semantic_draft_bundle=semantic_bundle,
        receipt=receipt,
    )


def compose_screened_sparse_bank(
    *,
    parent_bank: StaticBankArtifact,
    creator_candidate_bank: StaticBankArtifact,
    creator_compilation_receipt: SparseCompilationReceiptV1,
    development_screen_sha256: str,
    retained_capability_ids: tuple[str, ...],
) -> ScreenedSparseS1Candidate:
    """Revert every failed development-screen capability byte-exactly.

    The caller owns the metric decision and supplies the already frozen set of
    capability ids that met the development-screen predicates.  This function
    owns only deterministic Bank composition and refuses to retain an unchanged
    Skill or a capability absent from the Creator receipt.
    """

    if not _SHA_RE.fullmatch(development_screen_sha256):
        raise S1SparsePatchError("development screen SHA-256 is invalid")
    retained = tuple(sorted(set(retained_capability_ids)))
    if retained != retained_capability_ids:
        raise S1SparsePatchError(
            "retained capability ids must already be sorted and unique"
        )
    parent_by_capability = _skill_by_capability(parent_bank)
    candidate_by_capability = _skill_by_capability(creator_candidate_bank)
    receipt_by_capability = {
        item.capability_id: item for item in creator_compilation_receipt.bindings
    }
    if (
        creator_compilation_receipt.parent_bank_sha256 != parent_bank.bank_sha256
        or creator_compilation_receipt.candidate_bank_sha256
        != creator_candidate_bank.bank_sha256
        or set(parent_by_capability) != set(candidate_by_capability)
        or set(parent_by_capability) != set(receipt_by_capability)
        or not set(retained) <= set(parent_by_capability)
        or any(receipt_by_capability[item].action != "patch" for item in retained)
    ):
        raise S1SparsePatchError("screened sparse Bank lineage or retain set drifted")
    immutable_parent = parent_bank.model_dump(
        mode="json",
        exclude={"construction_identity_sha256", "skills", "bank_sha256"},
    )
    immutable_candidate = creator_candidate_bank.model_dump(
        mode="json",
        exclude={"construction_identity_sha256", "skills", "bank_sha256"},
    )
    if immutable_parent != immutable_candidate:
        raise S1SparsePatchError("Creator candidate changed immutable Bank metadata")

    selected_skills: list[StrictSkillArtifact] = []
    bindings: list[SparseScreenedSkillBindingV1] = []
    for capability_id in sorted(parent_by_capability):
        parent = parent_by_capability[capability_id]
        creator = candidate_by_capability[capability_id]
        creator_binding = receipt_by_capability[capability_id]
        parent_bytes = canonical_json_bytes(parent.model_dump(mode="json"))
        creator_bytes = canonical_json_bytes(creator.model_dump(mode="json"))
        if (
            creator_binding.parent_skill_sha256 != parent.skill_sha256
            or creator_binding.parent_skill_bytes_sha256 != sha256_bytes(parent_bytes)
            or creator_binding.candidate_skill_sha256 != creator.skill_sha256
            or creator_binding.candidate_skill_bytes_sha256
            != sha256_bytes(creator_bytes)
        ):
            raise S1SparsePatchError(
                f"screened sparse Skill receipt drifted: {capability_id}"
            )
        if capability_id in retained:
            selected = creator
            disposition = "retained_patch"
            restored = False
        else:
            selected = parent
            disposition = (
                "reverted_to_parent"
                if creator_binding.action == "patch"
                else "unchanged_parent"
            )
            restored = True
            if canonical_json_bytes(selected.model_dump(mode="json")) != parent_bytes:
                raise S1SparsePatchError(
                    f"failed capability was not restored byte-exactly: {capability_id}"
                )
        selected_skills.append(selected)
        bindings.append(
            SparseScreenedSkillBindingV1(
                capability_id=capability_id,
                disposition=disposition,
                parent_skill_sha256=parent.skill_sha256,
                creator_candidate_skill_sha256=creator.skill_sha256,
                screened_skill_sha256=selected.skill_sha256,
                parent_bytes_restored=restored,
            )
        )

    construction_identity = sha256_bytes(
        canonical_json_bytes(
            {
                "policy_version": S1_SPARSE_SCREENED_BANK_POLICY_VERSION,
                "parent_bank_sha256": parent_bank.bank_sha256,
                "creator_candidate_bank_sha256": creator_candidate_bank.bank_sha256,
                "creator_compilation_receipt_sha256": (
                    creator_compilation_receipt.receipt_sha256
                ),
                "development_screen_sha256": development_screen_sha256,
                "retained_capability_ids": list(retained),
            }
        )
    )
    bank_payload = {
        "schema_version": parent_bank.schema_version,
        "baseline_kind": parent_bank.baseline_kind,
        "construction_identity_sha256": construction_identity,
        "construction_identity_policy": parent_bank.construction_identity_policy,
        "runtime_binding_policy": parent_bank.runtime_binding_policy,
        "compiler": parent_bank.compiler.model_dump(mode="json"),
        "tool_registry_sha256": parent_bank.tool_registry_sha256,
        "tool_registry_runtime_sha256": parent_bank.tool_registry_runtime_sha256,
        "skills": [item.model_dump(mode="json") for item in selected_skills],
        "capability_map": [
            item.model_dump(mode="json") for item in parent_bank.capability_map
        ],
    }
    screened = StaticBankArtifact.model_validate(
        {
            **bank_payload,
            "bank_sha256": sha256_bytes(canonical_json_bytes(bank_payload)),
        },
        strict=True,
    )
    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-screened-sparse-bank-receipt",
        "policy_version": S1_SPARSE_SCREENED_BANK_POLICY_VERSION,
        "parent_bank_sha256": parent_bank.bank_sha256,
        "creator_candidate_bank_sha256": creator_candidate_bank.bank_sha256,
        "creator_compilation_receipt_sha256": (
            creator_compilation_receipt.receipt_sha256
        ),
        "development_screen_sha256": development_screen_sha256,
        "retained_capability_ids": list(retained),
        "bindings": [item.model_dump(mode="json") for item in bindings],
        "screened_bank_sha256": screened.bank_sha256,
    }
    receipt = SparseScreenedBankReceiptV1.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": sha256_bytes(canonical_json_bytes(receipt_payload)),
        },
        strict=True,
    )
    return ScreenedSparseS1Candidate(bank=screened, receipt=receipt)


def _skill_by_capability(
    bank: StaticBankArtifact,
) -> dict[str, StrictSkillArtifact]:
    result = {item.capability_id: item for item in bank.skills}
    if len(result) != len(bank.skills):
        raise S1SparsePatchError("parent Bank capability identities are not unique")
    return result


def _validate_patch_contract(
    capability_id: str,
    patch: SparseSkillContentPatchV1,
    parent: StrictSkillArtifact,
) -> None:
    _semantic_policy_for_patch(capability_id, patch)
    try:
        static_authoring_module._scan_untrusted_text(
            patch.semantic_policy.evidence_terms,
            "S1 typed semantic policy",
        )
    except static_authoring_module.AuthoringContractError as error:
        raise S1SparsePatchError(
            "typed semantic policy references forbidden experiment information"
        ) from error
    sequence = tuple(item.tool_name for item in patch.steps)
    parent_sequence = tuple(_parse_tool_steps(parent.body)[0])
    if sequence != parent_sequence:
        raise S1SparsePatchError(
            f"patch may not change the tool sequence: {capability_id}"
        )
    if capability_id == ENCYCLOPEDIA_CAPABILITY:
        if sequence != ENCYCLOPEDIA_TOOL_SEQUENCE:
            raise S1SparsePatchError("Encyclopedia tool sequence is not fail-closed")
        if ENCYCLOPEDIA_FALLBACK_MARKER not in patch.fallback_instruction.lower():
            raise S1SparsePatchError(
                "Encyclopedia patch lacks the exact fallback evidence marker"
            )


def _semantic_policy_for_patch(
    capability_id: str,
    patch: SparseSkillContentPatchV1,
) -> DeterministicSemanticPolicy:
    if capability_id not in SEMANTIC_POLICY_CAPABILITIES:
        raise S1SparsePatchError(
            f"capability has no S1-consumed semantic policy: {capability_id}"
        )
    try:
        return DeterministicSemanticPolicy(
            capability_id=capability_id,  # type: ignore[arg-type]
            evidence_terms=patch.semantic_policy.evidence_terms,
            require_all_terms=patch.semantic_policy.require_all_terms,
            abstain_when_no_evidence=patch.semantic_policy.abstain_when_no_evidence,
            ocr_extraction_plan=patch.semantic_policy.ocr_extraction_plan,
        )
    except (TypeError, ValueError) as error:
        raise S1SparsePatchError("typed semantic policy is invalid") from error


def _validate_parent_lineage(
    *,
    parent_bank: StaticBankArtifact,
    authoring_input: AuthoringInput,
    sparse_draft: SparsePatchDraftV1,
    tool_registry_runtime_sha256: str,
) -> None:
    if (
        sparse_draft.parent_bank_sha256 != parent_bank.bank_sha256
        or sparse_draft.authoring_input_sha256 != authoring_input.input_sha256
        or parent_bank.baseline_kind != "llm_static"
        or parent_bank.compiler != authoring_input.compiler
        or parent_bank.tool_registry_sha256
        != authoring_input.tool_registry.identity_sha256
        or parent_bank.tool_registry_runtime_sha256 != tool_registry_runtime_sha256
        or tuple(item.capability_id for item in sparse_draft.skills)
        != tuple(sorted(_skill_by_capability(parent_bank)))
    ):
        raise S1SparsePatchError("sparse S1 parent/compiler/registry lineage drifted")


def _as_authoring_content(
    *,
    capability_id: str,
    patch: SparseSkillContentPatchV1 | None,
    parent_content: CapabilityAuthoringContent,
) -> CapabilityAuthoringContent:
    if patch is None:
        return parent_content
    return CapabilityAuthoringContent(
        capability_id=capability_id,
        objective=patch.objective,
        steps=patch.steps,
        fallback_instruction=patch.fallback_instruction,
        citation_source_ids=patch.citation_source_ids,
    )


def _body_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    heading: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            if line in sections:
                raise S1SparsePatchError(f"duplicate compiled Skill heading: {line}")
            heading = line
            sections[heading] = []
        elif heading is None:
            if line:
                raise S1SparsePatchError("compiled Skill has prose before its heading")
        else:
            sections[heading].append(line)
    if "# Objective" not in sections or "## Output contract" not in sections:
        raise S1SparsePatchError("compiled Skill lacks required frozen headings")
    return {key: "\n".join(value) for key, value in sections.items()}


def _parse_tool_steps(body: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    section = _body_sections(body).get("## Tool procedure")
    if section is None:
        raise S1SparsePatchError("parent Skill lacks its Tool procedure")
    tools: list[str] = []
    instructions: list[str] = []
    ordinals: list[int] = []
    for line in section.splitlines():
        if not line:
            continue
        match = _TOOL_STEP_RE.fullmatch(line)
        if match is None:
            raise S1SparsePatchError("parent Skill tool procedure is not canonical")
        ordinals.append(int(match.group(1)))
        tools.append(match.group(2))
        instructions.append(match.group(3))
    if ordinals != list(range(1, len(ordinals) + 1)) or not tools:
        raise S1SparsePatchError("parent Skill tool ordinals drifted")
    return tuple(tools), tuple(instructions)


def _decode_parent_authoring_content(
    parent_bank: StaticBankArtifact,
    authoring_input: AuthoringInput,
) -> tuple[CapabilityAuthoringContent, ...]:
    try:
        task_raw = parse_canonical_json(
            authoring_input.task_specification.canonical_json.encode("utf-8"),
            label="S1 sparse TaskSpec",
        )
    except ArtifactFormatError as error:
        raise S1SparsePatchError("S1 sparse TaskSpec is invalid") from error
    capabilities = task_raw.get("capabilities") if isinstance(task_raw, dict) else None
    if not isinstance(capabilities, list):
        raise S1SparsePatchError("S1 sparse TaskSpec capabilities are invalid")
    success_ids: dict[str, tuple[str, ...]] = {}
    for task in capabilities:
        if not isinstance(task, dict) or not isinstance(
            task.get("success_criteria"), list
        ):
            raise S1SparsePatchError("S1 sparse TaskSpec rule shape drifted")
        success_ids[str(task.get("capability_id"))] = tuple(
            sorted(str(rule["rule_id"]) for rule in task["success_criteria"])
        )
    allowed_sources = {item.source_id for item in authoring_input.public_sources}
    contents: list[CapabilityAuthoringContent] = []
    for skill in sorted(parent_bank.skills, key=lambda item: item.capability_id):
        sections = _body_sections(skill.body)
        objective_lines = [
            line for line in sections["# Objective"].splitlines() if line
        ]
        fallback_lines = [
            line
            for line in sections.get(
                "## Authored fallback instruction", ""
            ).splitlines()
            if line
        ]
        if objective_lines != [skill.description] or len(fallback_lines) != 1:
            raise S1SparsePatchError(
                f"parent authored prose cannot be losslessly decoded: {skill.capability_id}"
            )
        tools, instructions = _parse_tool_steps(skill.body)
        ids = success_ids.get(skill.capability_id)
        if not ids:
            raise S1SparsePatchError(
                f"parent capability has no frozen success rules: {skill.capability_id}"
            )
        citation_lines = [
            line[2:]
            for line in sections.get(
                "## Authored public source citations", ""
            ).splitlines()
            if line.startswith("- ") and line[2:] in allowed_sources
        ]
        contents.append(
            CapabilityAuthoringContent(
                capability_id=skill.capability_id,
                objective=skill.description,
                steps=tuple(
                    DraftToolStepContent(
                        instruction=instruction,
                        tool_name=tool,
                        success_rule_ids=ids,
                    )
                    for tool, instruction in zip(tools, instructions, strict=True)
                ),
                fallback_instruction=fallback_lines[0],
                citation_source_ids=tuple(sorted(citation_lines)),
            )
        )
    # Prove the decoder itself does not introduce a new parent representation.
    try:
        bundle = normalize_authoring_content_payload(
            AuthoringContentPayload(schema_version=2, drafts=tuple(contents)),
            authoring_input=authoring_input,
        )
        rebuilt = compile_portfolio_s1_creator_bank(
            authoring_input,
            bundle,
            tool_registry_runtime_sha256=parent_bank.tool_registry_runtime_sha256,
        )
    except Exception as error:
        raise S1SparsePatchError(
            "parent Skill decoder failed trusted compilation"
        ) from error
    rebuilt_by_capability = _skill_by_capability(rebuilt)
    for capability_id, parent in _skill_by_capability(parent_bank).items():
        if canonical_json_bytes(parent.model_dump(mode="json")) != canonical_json_bytes(
            rebuilt_by_capability[capability_id].model_dump(mode="json")
        ):
            raise S1SparsePatchError(
                f"parent Skill decoder is not byte-exact: {capability_id}"
            )
    return tuple(contents)


def _assert_patch_only_changed_mutable_fields(
    parent: StrictSkillArtifact,
    candidate: StrictSkillArtifact,
) -> None:
    immutable_parent = parent.model_dump(
        mode="json", exclude={"description", "body", "skill_sha256"}
    )
    immutable_candidate = candidate.model_dump(
        mode="json", exclude={"description", "body", "skill_sha256"}
    )
    if immutable_parent != immutable_candidate:
        raise S1SparsePatchError(
            f"patch changed immutable Skill metadata: {parent.capability_id}"
        )
    parent_sections = _body_sections(parent.body)
    candidate_sections = _body_sections(candidate.body)
    if tuple(parent_sections) != tuple(candidate_sections):
        raise S1SparsePatchError(
            f"patch changed compiled body headings: {parent.capability_id}"
        )
    for heading in parent_sections:
        if (
            heading not in _MUTABLE_BODY_SECTIONS
            and parent_sections[heading] != candidate_sections[heading]
        ):
            raise S1SparsePatchError(
                f"patch changed compiler-owned section {heading}: "
                f"{parent.capability_id}"
            )


__all__ = [
    "CompiledSparseS1Candidate",
    "ENCYCLOPEDIA_CAPABILITY",
    "ENCYCLOPEDIA_FALLBACK_MARKER",
    "FORBIDDEN_CONTROL_TOKENS",
    "S1SparsePatchError",
    "S1_SPARSE_COMPILATION_POLICY_VERSION",
    "S1_SPARSE_PATCH_POLICY_VERSION",
    "S1_SPARSE_SCREENED_BANK_POLICY_VERSION",
    "ScreenedSparseS1Candidate",
    "SparseCompilationReceiptV1",
    "SparsePatchDraftPayloadV1",
    "SparsePatchDraftV1",
    "SparseScreenedBankReceiptV1",
    "SparseSemanticPolicyV1",
    "SparseSkillContentPatchV1",
    "SparseSkillDraftV1",
    "bind_sparse_patch_draft",
    "compile_sparse_s1_candidate",
    "compose_screened_sparse_bank",
    "decode_sparse_parent_content",
    "load_sparse_compilation_receipt",
    "load_sparse_patch_draft",
    "sparse_author_content_lexical_guard",
    "sparse_patch_output_json_schema",
]
