"""Strict Task Specification v0 bound to the frozen MVP taxonomy.

The specification contains only public/project inputs.  It defines task
success and failure before any experiment output exists, and exposes tool
permissions as an allowlist.  The loader binds both taxonomy identity and an
optional independent expected specification digest.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.taxonomy import (
    DEFAULT_TAXONOMY_SHA256,
    PROJECT_ROOT,
    TASK_SPEC_VERSION,
    TaxonomyRegistry,
    _digest_without_field,
    _load_canonical_json_object,
    assert_public_inputs_only,
    load_default_taxonomy_registry,
)

TASK_SPEC_SCHEMA_VERSION = 1
DEFAULT_TASK_SPEC_PATH = (
    PROJECT_ROOT / "specs" / "task_specs" / f"{TASK_SPEC_VERSION}.json"
)
DEFAULT_TASK_SPEC_SHA256 = (
    "5ee0a127bb1d498cc0b7dc223625a13125be058e20425ec528691dc1bbf495da"
)
MVP_TASK_SPEC_V1_PATH = (
    PROJECT_ROOT / "specs" / "task_specs" / "ecommerce-task-spec-v1.json"
)
MVP_TASK_SPEC_V1_SHA256 = (
    "8b7b1ea59766e6de4e8c0310e9403f0ea7e550fc2aab39427adf77aa7682aa6b"
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RuleSourceKind = Literal["public_source", "tool_contract", "project_choice"]
ToolName = Literal[
    "image_product_search",
    "text_product_search",
    "style_similar_search",
    "encyclopedia_lookup",
    "recipe_lookup",
    "object_detect",
    "document_ocr",
    "multi_product_search",
]
CardRequirement = Literal["required", "forbidden", "optional"]
ResponseKind = Literal["product_cards", "grounded_text"]
EvidenceRequirement = Literal[
    "product_retrieval_evidence",
    "cited_knowledge_evidence",
    "typed_tool_evidence",
]

_TOOL_CONTRACT_VERSIONS = {
    "document_ocr": "1.0.0",
    "encyclopedia_lookup": "1.0.0",
    "image_product_search": "2.1.0",
    "multi_product_search": "1.0.0",
    "object_detect": "1.0.0",
    "recipe_lookup": "1.0.0",
    "style_similar_search": "2.3.0",
    "text_product_search": "2.1.0",
}
_TOOL_NAMES = frozenset(_TOOL_CONTRACT_VERSIONS)
_TOOL_CONTRACT_VERSION_HISTORY = {
    **{name: frozenset({version}) for name, version in _TOOL_CONTRACT_VERSIONS.items()},
    "style_similar_search": frozenset({"2.1.0", "2.2.0", "2.3.0"}),
}
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class TaskSpecError(ValueError):
    """The Task Specification is malformed, contaminated, or not bound."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _clean_text(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank without edge whitespace")
    return value


def _identifier(value: str, field_name: str) -> str:
    value = _clean_text(value, field_name)
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase identifier")
    return value


def _coerce_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class RuleProvenance(_StrictFrozenModel):
    """Evidence class and stable locator for one frozen rule."""

    kind: RuleSourceKind
    source_ref: str

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str, info) -> str:
        return _clean_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_ref_namespace(self) -> Self:
        if self.kind == "public_source":
            if not self.source_ref.startswith("https://"):
                raise ValueError("public_source must use an HTTPS locator")
        elif self.kind == "tool_contract":
            match = re.fullmatch(
                r"tool:([a-z_]+)@([0-9]+(?:\.[0-9]+){2})", self.source_ref
            )
            if (
                match is None
                or match.group(1) not in _TOOL_NAMES
                or match.group(2) not in _TOOL_CONTRACT_VERSION_HISTORY[match.group(1)]
            ):
                raise ValueError(
                    "tool_contract source_ref is not the frozen tool version"
                )
        elif not re.fullmatch(r"project-choice:[a-z0-9][a-z0-9._-]*", self.source_ref):
            raise ValueError("project_choice must use the project-choice namespace")
        return self


class TaskRule(_StrictFrozenModel):
    rule_id: str
    statement: str
    provenance: RuleProvenance

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        return _identifier(value, "rule_id")

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        return _clean_text(value, "statement")


class OutputContract(_StrictFrozenModel):
    """Answer shape for a positive supported answer.

    ``required`` means every returned product must use a card with the required
    fields.  A fallback/no-supported-product response may contain zero cards;
    it must follow the separate fallback contract instead.
    """

    response_kind: ResponseKind
    required_sections: tuple[str, ...] = Field(min_length=1)
    card_requirement: CardRequirement
    card_fields: tuple[str, ...]
    evidence_requirement: EvidenceRequirement

    @field_validator("required_sections", "card_fields", mode="before")
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator("required_sections", "card_fields")
    @classmethod
    def validate_fields(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        cleaned = tuple(_identifier(item, info.field_name) for item in value)
        if cleaned != tuple(sorted(cleaned)) or len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_card_shape(self) -> Self:
        required_product_fields = {"evidence_reference", "product_id", "title"}
        if self.card_requirement == "required":
            if self.response_kind != "product_cards":
                raise ValueError("required cards need product_cards response_kind")
            if not required_product_fields <= set(self.card_fields):
                raise ValueError("required product card fields are incomplete")
        elif self.card_requirement == "forbidden" and self.card_fields:
            raise ValueError("forbidden cards must not declare card_fields")
        return self


class FallbackContract(_StrictFrozenModel):
    trigger_rules: tuple[TaskRule, ...] = Field(min_length=1)
    response_rules: tuple[TaskRule, ...] = Field(min_length=1)
    may_request_clarification: bool
    must_state_uncertainty: Literal[True]

    @field_validator("trigger_rules", "response_rules", mode="before")
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return _coerce_tuple(value)


class CapabilityTaskSpec(_StrictFrozenModel):
    capability_id: str
    input_preconditions: tuple[TaskRule, ...] = Field(min_length=1)
    success_criteria: tuple[TaskRule, ...] = Field(min_length=1)
    failure_conditions: tuple[TaskRule, ...] = Field(min_length=1)
    acceptable_answer_rules: tuple[TaskRule, ...] = Field(min_length=1)
    allowed_tools: tuple[ToolName, ...] = Field(min_length=1)
    output_contract: OutputContract
    safety_constraints: tuple[TaskRule, ...] = Field(min_length=1)
    fallback: FallbackContract
    indeterminate_conditions: tuple[TaskRule, ...] = Field(min_length=1)

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        return _identifier(value, "capability_id")

    @field_validator(
        "input_preconditions",
        "success_criteria",
        "failure_conditions",
        "acceptable_answer_rules",
        "allowed_tools",
        "safety_constraints",
        "indeterminate_conditions",
        mode="before",
    )
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(
        cls, value: tuple[ToolName, ...]
    ) -> tuple[ToolName, ...]:
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError("allowed_tools must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_rule_ids(self) -> Self:
        rules = (
            *self.input_preconditions,
            *self.success_criteria,
            *self.failure_conditions,
            *self.acceptable_answer_rules,
            *self.safety_constraints,
            *self.fallback.trigger_rules,
            *self.fallback.response_rules,
            *self.indeterminate_conditions,
        )
        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule_id values must be unique within a capability")
        return self


class TaskSpecification(_StrictFrozenModel):
    """Frozen task contract for all selected taxonomy capabilities."""

    schema_version: Literal[1]
    task_spec_version: Literal[
        "ecommerce-task-spec-v0",
        "ecommerce-task-spec-v1",
    ]
    status: Literal["frozen"]
    taxonomy_version: Literal["ecommerce-mvp-taxonomy-v0"]
    taxonomy_sha256: Sha256
    capabilities: tuple[CapabilityTaskSpec, ...] = Field(min_length=5, max_length=7)
    task_spec_sha256: Sha256

    @field_validator("capabilities", mode="before")
    @classmethod
    def coerce_capabilities(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_specification(self) -> Self:
        capability_ids = tuple(item.capability_id for item in self.capabilities)
        if capability_ids != tuple(sorted(capability_ids)):
            raise ValueError("capability task specs must be sorted by capability_id")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability task specs must be unique")
        dumped = self.model_dump(mode="json")
        assert_public_inputs_only(dumped, "task specification")
        expected_versions = dict(_TOOL_CONTRACT_VERSIONS)
        if self.task_spec_version == "ecommerce-task-spec-v0":
            allowed_style_versions = frozenset({"2.1.0"})
        else:
            # TaskSpec v1 has several immutable, hash-distinguished historical
            # snapshots.  Accept one internally consistent Style generation so
            # those exact artifacts remain readable; the active v1 loader still
            # pins the current file and digest (2.3.0) independently.
            allowed_style_versions = _TOOL_CONTRACT_VERSION_HISTORY[
                "style_similar_search"
            ]
        observed_style_versions: set[str] = set()

        def validate_tool_refs(value: object) -> None:
            if isinstance(value, dict):
                if value.get("kind") == "tool_contract":
                    source_ref = value.get("source_ref")
                    match = (
                        re.fullmatch(
                            r"tool:([a-z_]+)@([0-9]+(?:\.[0-9]+){2})",
                            source_ref,
                        )
                        if isinstance(source_ref, str)
                        else None
                    )
                    if match is None:
                        raise ValueError(
                            "tool_contract source_ref differs from TaskSpec generation"
                        )
                    tool_name, tool_version = match.groups()
                    if tool_name == "style_similar_search":
                        if tool_version not in allowed_style_versions:
                            raise ValueError(
                                "tool_contract source_ref differs from TaskSpec generation"
                            )
                        observed_style_versions.add(tool_version)
                    elif expected_versions.get(tool_name) != tool_version:
                        raise ValueError(
                            "tool_contract source_ref differs from TaskSpec generation"
                        )
                for nested in value.values():
                    validate_tool_refs(nested)
            elif isinstance(value, list):
                for nested in value:
                    validate_tool_refs(nested)

        validate_tool_refs(dumped)
        if len(observed_style_versions) > 1:
            raise ValueError("TaskSpec mixes Style tool contract generations")
        expected_hash = _digest_without_field(dumped, "task_spec_sha256")
        if self.task_spec_sha256 != expected_hash:
            raise ValueError("task_spec_sha256 mismatch")
        return self

    @property
    def capabilities_by_id(self):
        from types import MappingProxyType

        return MappingProxyType(
            {item.capability_id: item for item in self.capabilities}
        )


def load_task_specification(
    path: str | Path,
    *,
    taxonomy: TaxonomyRegistry,
    expected_sha256: str | None = None,
) -> TaskSpecification:
    """Load and bind one canonical Task Specification to a taxonomy registry."""

    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise TaskSpecError("expected Task Specification SHA-256 is invalid")
    try:
        raw = _load_canonical_json_object(path, label="task specification")
    except ValueError as error:
        raise TaskSpecError(str(error)) from error
    try:
        specification = TaskSpecification.model_validate(raw, strict=True)
    except ValidationError as error:
        raise TaskSpecError("task specification violates schema") from error
    if (
        expected_sha256 is not None
        and specification.task_spec_sha256 != expected_sha256
    ):
        raise TaskSpecError("task specification does not match expected SHA-256")
    _bind_to_taxonomy(specification, taxonomy)
    return specification


def load_default_task_specification() -> TaskSpecification:
    """Load both tracked frozen artifacts under independent digest locks."""

    taxonomy = load_default_taxonomy_registry()
    if taxonomy.taxonomy_sha256 != DEFAULT_TAXONOMY_SHA256:
        raise TaskSpecError("default taxonomy identity is inconsistent")
    return load_task_specification(
        DEFAULT_TASK_SPEC_PATH,
        taxonomy=taxonomy,
        expected_sha256=DEFAULT_TASK_SPEC_SHA256,
    )


def load_mvp_task_specification_v1() -> TaskSpecification:
    """Load the v1 contract whose multi-product path is executable."""

    taxonomy = load_default_taxonomy_registry()
    if taxonomy.taxonomy_sha256 != DEFAULT_TAXONOMY_SHA256:
        raise TaskSpecError("default taxonomy identity is inconsistent")
    return load_task_specification(
        MVP_TASK_SPEC_V1_PATH,
        taxonomy=taxonomy,
        expected_sha256=MVP_TASK_SPEC_V1_SHA256,
    )


def _bind_to_taxonomy(
    specification: TaskSpecification,
    taxonomy: TaxonomyRegistry,
) -> None:
    if specification.taxonomy_version != taxonomy.taxonomy_version:
        raise TaskSpecError("Task Specification taxonomy version mismatch")
    if specification.taxonomy_sha256 != taxonomy.taxonomy_sha256:
        raise TaskSpecError("Task Specification taxonomy SHA-256 mismatch")
    taxonomy_capabilities = taxonomy.capabilities_by_id
    if set(specification.capabilities_by_id) != set(taxonomy_capabilities):
        raise TaskSpecError("Task Specification capability set mismatch")
    for capability_id, task in specification.capabilities_by_id.items():
        expected_cards = taxonomy_capabilities[capability_id].requires_card
        actual_cards = task.output_contract.card_requirement == "required"
        if expected_cards != actual_cards:
            raise TaskSpecError(
                f"Task Specification card requirement mismatch for {capability_id}"
            )


__all__ = [
    "DEFAULT_TASK_SPEC_PATH",
    "DEFAULT_TASK_SPEC_SHA256",
    "MVP_TASK_SPEC_V1_PATH",
    "MVP_TASK_SPEC_V1_SHA256",
    "TASK_SPEC_SCHEMA_VERSION",
    "CapabilityTaskSpec",
    "FallbackContract",
    "OutputContract",
    "RuleProvenance",
    "TaskRule",
    "TaskSpecError",
    "TaskSpecification",
    "load_default_task_specification",
    "load_mvp_task_specification_v1",
    "load_task_specification",
]
