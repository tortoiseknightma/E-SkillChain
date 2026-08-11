"""Frozen, bank-independent intent and capability taxonomy.

The tracked taxonomy artifact is the reviewable authority for v0.  Runtime
helpers retain the original one-intent-to-one-primary-capability interface so
existing query planning remains deterministic.  Loaders reject non-canonical
JSON, coordinated content changes without an expected digest, and references
to experiment-derived information.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from types import MappingProxyType
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

from skillchain.schemas import Intent
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
    validate_json_value,
)

TAXONOMY_VERSION = "ecommerce-mvp-taxonomy-v0"
TASK_SPEC_VERSION = "ecommerce-task-spec-v0"
TAXONOMY_SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "specs" / "taxonomy" / f"{TAXONOMY_VERSION}.json"

# Filled from the reviewed tracked artifact.  This is intentionally separate
# from its self-hash so a coordinated artifact edit cannot silently become the
# default runtime taxonomy.
DEFAULT_TAXONOMY_SHA256 = (
    "af23dbe76e8c0cffdbb056b44c6595af6c508eaa0158f96227cbc92e244ca315"
)

INTENT_IDS: tuple[Intent, ...] = (
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")
_MAX_ARTIFACT_BYTES = 512 * 1024
_FORBIDDEN_REFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bskill\b", re.IGNORECASE),
    re.compile(r"\bbank\b", re.IGNORECASE),
    re.compile(r"\brubrics?\b", re.IGNORECASE),
    re.compile(r"\bgold\b", re.IGNORECASE),
    re.compile(
        r"\b(?:evaluation|eval|judge|metric|benchmark|pilot|assistant)"
        r"[ _-]*(?:result|score|output|response)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:training[ _-]*)?trajector(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"候选\s*技能|技能库|评分准则|金标|评测结果|评价结果|裁判结果|训练轨迹"),
)


class TaxonomyError(ValueError):
    """The taxonomy artifact is malformed, contaminated, or not trusted."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _require_clean_text(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank without edge whitespace")
    return value


def _require_identifier(value: str, field_name: str) -> str:
    value = _require_clean_text(value, field_name)
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase dotted identifier")
    return value


def _coerce_json_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class CapabilityDefinition(_StrictFrozenModel):
    """One selected MVP capability, independent of any generated solution."""

    capability_id: str
    intent_id: Intent
    title: str
    definition: str
    requires_card: bool
    selection_rationale: str

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        return _require_identifier(value, "capability_id")

    @field_validator("title", "definition", "selection_rationale")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_clean_text(value, info.field_name)


class IntentDefinition(_StrictFrozenModel):
    """Intent boundary plus its pre-selection candidate coverage matrix row."""

    intent_id: Intent
    title: str
    definition: str
    candidate_capability_ids: tuple[str, ...] = Field(min_length=8, max_length=12)
    mvp_capability_ids: tuple[str, ...] = Field(min_length=1)
    primary_capability_id: str

    @field_validator("candidate_capability_ids", "mvp_capability_ids", mode="before")
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return _coerce_json_tuple(value)

    @field_validator("title", "definition")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_clean_text(value, info.field_name)

    @field_validator("primary_capability_id")
    @classmethod
    def validate_primary_id(cls, value: str) -> str:
        return _require_identifier(value, "primary_capability_id")

    @field_validator("candidate_capability_ids", "mvp_capability_ids")
    @classmethod
    def validate_capability_ids(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        validated = tuple(_require_identifier(item, info.field_name) for item in value)
        if len(validated) != len(set(validated)):
            raise ValueError(f"{info.field_name} must be unique")
        if validated != tuple(sorted(validated)):
            raise ValueError(f"{info.field_name} must be sorted")
        return validated

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        candidate_ids = set(self.candidate_capability_ids)
        selected_ids = set(self.mvp_capability_ids)
        if not selected_ids <= candidate_ids:
            raise ValueError("MVP capabilities must be selected from the candidate row")
        if self.primary_capability_id not in selected_ids:
            raise ValueError("primary capability must be an MVP capability")
        return self


class TaxonomyRegistry(_StrictFrozenModel):
    """Canonical content-addressed taxonomy registry v0."""

    schema_version: Literal[1]
    taxonomy_version: Literal["ecommerce-mvp-taxonomy-v0"]
    status: Literal["frozen"]
    intents: tuple[IntentDefinition, ...]
    capabilities: tuple[CapabilityDefinition, ...] = Field(min_length=5, max_length=7)
    taxonomy_sha256: Sha256

    @field_validator("intents", "capabilities", mode="before")
    @classmethod
    def coerce_tuple_fields(cls, value: object) -> object:
        return _coerce_json_tuple(value)

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        intent_ids = tuple(item.intent_id for item in self.intents)
        if intent_ids != INTENT_IDS:
            raise ValueError("taxonomy must contain the five intents in frozen order")

        capability_ids = tuple(item.capability_id for item in self.capabilities)
        if capability_ids != tuple(sorted(capability_ids)):
            raise ValueError("capabilities must be sorted by capability_id")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability_id values must be unique")

        by_intent: dict[Intent, set[str]] = {intent: set() for intent in INTENT_IDS}
        for capability in self.capabilities:
            by_intent[capability.intent_id].add(capability.capability_id)
        for intent in self.intents:
            if set(intent.mvp_capability_ids) != by_intent[intent.intent_id]:
                raise ValueError(
                    f"MVP capability mapping is inconsistent for {intent.intent_id}"
                )

        assert_public_inputs_only(self.model_dump(mode="json"), "taxonomy")
        expected_hash = _digest_without_field(
            self.model_dump(mode="json"), "taxonomy_sha256"
        )
        if self.taxonomy_sha256 != expected_hash:
            raise ValueError("taxonomy_sha256 mismatch")
        return self

    @property
    def capabilities_by_id(self) -> MappingProxyType[str, CapabilityDefinition]:
        return MappingProxyType(
            {item.capability_id: item for item in self.capabilities}
        )

    @property
    def intents_by_id(self) -> MappingProxyType[Intent, IntentDefinition]:
        return MappingProxyType({item.intent_id: item for item in self.intents})


_CAPABILITY_DEFINITIONS: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        capability_id="knowledge.visual_encyclopedia",
        intent_id="encyclopedia",
        title="Grounded visual encyclopedia",
        definition="Explain a visually indicated entity using cited encyclopedia evidence and explicit uncertainty.",
        requires_card=False,
        selection_rationale="Selected because object detection and encyclopedia lookup provide a grounded visual-to-text path.",
    ),
    CapabilityDefinition(
        capability_id="product.exact_match",
        intent_id="exact_match",
        title="Exact product match",
        definition="Identify the same catalog product from grounded product-retrieval evidence.",
        requires_card=True,
        selection_rationale="Selected because the product gallery and image/text retrieval contracts can provide product-level evidence.",
    ),
    CapabilityDefinition(
        capability_id="product.multi_search",
        intent_id="multi_product",
        title="Multi-product decomposition and search",
        definition="Separate multiple visible products and return grounded candidates for each requested item.",
        requires_card=True,
        selection_rationale="Selected because object detection and product retrieval form an implemented composition path.",
    ),
    CapabilityDefinition(
        capability_id="product.style_recommendation",
        intent_id="divergent_rec",
        title="Style-diverse recommendation",
        definition="Return relevant but meaningfully diverse style alternatives grounded in the product gallery.",
        requires_card=True,
        selection_rationale="Selected because the style-similarity tool exposes a diversity-aware retrieval contract.",
    ),
    CapabilityDefinition(
        capability_id="utility.document_reading",
        intent_id="utility",
        title="Grounded document reading",
        definition="Transcribe or extract requested fields from an approved document image while retaining line evidence.",
        requires_card=False,
        selection_rationale="Selected because document OCR exposes line-level evidence and an approval-gated runtime contract.",
    ),
    CapabilityDefinition(
        capability_id="utility.recipe_guidance",
        intent_id="utility",
        title="Grounded recipe guidance",
        definition="Provide recipe guidance for an identified dish using cited recipe evidence and explicit uncertainty.",
        requires_card=False,
        selection_rationale="Selected because recipe lookup provides a separate cited evidence path with distinct safety constraints.",
    ),
)

CAPABILITY_DEFINITIONS_BY_ID = MappingProxyType(
    {item.capability_id: item for item in _CAPABILITY_DEFINITIONS}
)
CAPABILITIES_BY_INTENT = MappingProxyType(
    {
        intent: tuple(
            item for item in _CAPABILITY_DEFINITIONS if item.intent_id == intent
        )
        for intent in INTENT_IDS
    }
)
PRIMARY_CAPABILITY_ID_BY_INTENT = MappingProxyType(
    {
        "exact_match": "product.exact_match",
        "multi_product": "product.multi_search",
        "divergent_rec": "product.style_recommendation",
        "encyclopedia": "knowledge.visual_encyclopedia",
        # Compatibility fallback for the current intent-only planner.  New
        # utility planning must select an explicit subtype from
        # capabilities_for_intent instead of relying on this default.
        "utility": "utility.document_reading",
    }
)


def capability_for_intent(intent: Intent) -> str:
    """Return the primary compatibility fallback for an intent-only caller."""

    return PRIMARY_CAPABILITY_ID_BY_INTENT[intent]


def capabilities_for_intent(intent: Intent) -> tuple[CapabilityDefinition, ...]:
    """Return all frozen MVP capabilities available under one intent."""

    return CAPABILITIES_BY_INTENT[intent]


def requires_card_for_intent(intent: Intent) -> bool:
    """Return the frozen product-card requirement for one intent."""

    requirements = {item.requires_card for item in capabilities_for_intent(intent)}
    if len(requirements) != 1:
        raise TaxonomyError(
            f"intent {intent} mixes card requirements; select a capability explicitly"
        )
    return requirements.pop()


def load_taxonomy_registry(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> TaxonomyRegistry:
    """Load one stable canonical registry and optionally bind an expected hash."""

    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise TaxonomyError("expected taxonomy SHA-256 is invalid")
    raw = _load_canonical_json_object(path, label="taxonomy registry")
    try:
        registry = TaxonomyRegistry.model_validate(raw, strict=True)
    except ValidationError as error:
        raise TaxonomyError("taxonomy registry violates schema") from error
    if expected_sha256 is not None and registry.taxonomy_sha256 != expected_sha256:
        raise TaxonomyError("taxonomy registry does not match expected SHA-256")
    return registry


def load_default_taxonomy_registry() -> TaxonomyRegistry:
    """Load the tracked v0 registry using the independent runtime digest lock."""

    registry = load_taxonomy_registry(
        DEFAULT_TAXONOMY_PATH,
        expected_sha256=DEFAULT_TAXONOMY_SHA256,
    )
    expected_mapping = dict(PRIMARY_CAPABILITY_ID_BY_INTENT)
    actual_mapping = {
        intent.intent_id: intent.primary_capability_id for intent in registry.intents
    }
    if actual_mapping != expected_mapping:
        raise TaxonomyError("tracked taxonomy does not match runtime primary mapping")
    expected_mvp_mapping = {
        intent: tuple(item.capability_id for item in capabilities)
        for intent, capabilities in CAPABILITIES_BY_INTENT.items()
    }
    actual_mvp_mapping = {
        intent.intent_id: intent.mvp_capability_ids for intent in registry.intents
    }
    if actual_mvp_mapping != expected_mvp_mapping:
        raise TaxonomyError("tracked taxonomy does not match runtime MVP mapping")
    for capability in registry.capabilities:
        runtime = CAPABILITY_DEFINITIONS_BY_ID[capability.capability_id]
        if capability != runtime:
            raise TaxonomyError("tracked taxonomy does not match runtime capability")
    return registry


def assert_public_inputs_only(value: object, label: str) -> None:
    """Reject information that could make a frozen input result-dependent."""

    validate_json_value(value)

    def visit(item: object) -> None:
        if isinstance(item, str):
            for pattern in _FORBIDDEN_REFERENCE_PATTERNS:
                if pattern.search(item):
                    raise ValueError(
                        f"{label} references forbidden experiment-derived information"
                    )
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)

    visit(value)


def _digest_without_field(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256_bytes(canonical_json_bytes(payload))


def _canonical_pretty_json_bytes(value: object) -> bytes:
    validate_json_value(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _object_without_duplicates(
    pairs: list[tuple[str, Any]], label: str
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TaxonomyError(f"{label} contains duplicate key {key}")
        value[key] = item
    return value


def _reject_constant(token: str, label: str) -> None:
    raise TaxonomyError(f"{label} contains non-finite number {token}")


def _load_canonical_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        content = read_stable_regular_file(
            path,
            label=label,
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=lambda pairs: _object_without_duplicates(pairs, label),
            parse_constant=lambda token: _reject_constant(token, label),
        )
    except TaxonomyError:
        raise
    except ArtifactFormatError as error:
        raise TaxonomyError(f"{label} cannot be read safely") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TaxonomyError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise TaxonomyError(f"{label} must be a JSON object")
    try:
        if content != _canonical_pretty_json_bytes(raw):
            raise TaxonomyError(f"{label} must use canonical pretty JSON")
    except (ArtifactFormatError, TypeError, ValueError) as error:
        if isinstance(error, TaxonomyError):
            raise
        raise TaxonomyError(f"{label} contains an unsupported JSON value") from error
    return raw


__all__ = [
    "CAPABILITIES_BY_INTENT",
    "CAPABILITY_DEFINITIONS_BY_ID",
    "DEFAULT_TAXONOMY_PATH",
    "DEFAULT_TAXONOMY_SHA256",
    "INTENT_IDS",
    "PRIMARY_CAPABILITY_ID_BY_INTENT",
    "TASK_SPEC_VERSION",
    "TAXONOMY_SCHEMA_VERSION",
    "TAXONOMY_VERSION",
    "CapabilityDefinition",
    "IntentDefinition",
    "TaxonomyError",
    "TaxonomyRegistry",
    "assert_public_inputs_only",
    "capability_for_intent",
    "capabilities_for_intent",
    "load_default_taxonomy_registry",
    "load_taxonomy_registry",
    "requires_card_for_intent",
]
