"""Deterministic Grounded Contract Success scoring for the Portfolio track.

The module deliberately owns no model calls and performs no artifact writes.  It
scores runner-owned Assistant artifacts against a frozen Task Specification and
an optional public-safe scorer sidecar.  A missing or unresolved component is a
zero; a missing terminal query/config row is a population error.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.evaluation.assistant_runs import (
    MAIN_CONFIG_ORDER,
    AssistantBackendResponse,
    AssistantExecutionReceipt,
    AssistantRequestSnapshot,
    build_assistant_query_input,
    build_legacy_assistant_query_input,
)
from skillchain.evaluation.packets import (
    AssistantResult,
    AssistantRunConfig,
    AssistantToolTrace,
    VisibleCard,
    VisibleToolEvidence,
)
from skillchain.evaluation.portfolio_launch import PortfolioLaunchInstance
from skillchain.evaluation.portfolio_treatments import (
    PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
    PortfolioExecutionArtifactAlias,
)
from skillchain.evaluation.portfolio_gcs_evidence import (
    GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
    PublicScorerCallEvidenceV2,
    PublicScorerEvidenceIntegrityError,
    PublicScorerEvidenceV2,
    ScorerStylePayloadV2,
    StyleSupportStatus,
    classify_style_query_v2,
    require_public_scorer_evidence_v2,
)
from skillchain.schemas import Query
from skillchain.synthesis.splitting import (
    GROUP_FIELDS,
    GROUPING_POLICY_VERSION,
    build_atomic_groups,
)
from skillchain.task_spec import CapabilityTaskSpec, TaskSpecification
from skillchain.tools.contracts import JSONValue, validate_json_value
from skillchain.tools.serialization import (
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Binary = Literal[0, 1]
AnswerMode = Literal["supported", "fallback", "unresolved"]
StyleOutcomeStatus = StyleSupportStatus | Literal["unresolved"]
ScorerPayloadKind = Literal[
    "product_candidates_v1",
    "multi_mapping_v1",
    "knowledge_sources_v1",
    "ocr_lines_v1",
    "detections_v1",
]
PhysicalAssistantRunConfig = Literal["noskill", "llm_static", "s1", "s1s2"]
GCSProjectionKind = Literal["direct", "execution_artifact_alias"]
GCSPhysicalSidecarStatus = Literal["available", "missing", "invalid"]

GCS_POLICY_VERSION = "portfolio-grounded-contract-success-v1"
GCS_SCORER_EVIDENCE_POLICY_VERSION = "portfolio-gcs-scorer-evidence-v1"
GCS_V2_POLICY_VERSION = "portfolio-grounded-contract-success-v2"
GCS_V2_SIDECAR_MAPPING_POLICY_VERSION = (
    "portfolio-gcs-five-config-physical-sidecar-map-v2"
)
GCS_V2_EXPECTED_POLICY_SHA256 = (
    "ccb838617c857719e8a864f1638b3aa01f2e2aacbb7344629bcd9087b9e20651"
)
GCS_BOOTSTRAP_POLICY_VERSION = "portfolio-gcs-component-bootstrap-v1"
GCS_BOOTSTRAP_ROOT_SEED = 2026080601
GCS_BOOTSTRAP_REPLICATES = 10_000
GCS_CAPABILITY_ORDER: tuple[str, ...] = (
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)
GCS_COMPONENTS: tuple[str, ...] = (
    "route_acceptable",
    "no_hard_error",
    "tool_contract_pass",
    "evidence_grounded",
    "output_contract_pass",
)
GCS_V2_PHYSICAL_CONFIG_ORDER: tuple[PhysicalAssistantRunConfig, ...] = (
    "noskill",
    "llm_static",
    "s1",
    "s1s2",
)
GCS_V2_FULL_ALIAS_SOURCE_CONFIG: Literal["s1s2"] = "s1s2"
SYSTEM_GAIN_MACRO_DELTA_PP_MIN = 2.0
SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN = 0.0
SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX = 1.0
SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN = -3.0

_PRODUCT_EVIDENCE_RE = re.compile(r"^tool-call-(\d+)-evidence-(\d+)$")
_PRODUCT_ID_RE = re.compile(r"^tool-call-(\d+)-product-(\d+)$")
_SOURCE_RE = re.compile(r"^tool-call-(\d+)-source-(\d+)$")
_LINE_RE = re.compile(r"^tool-call-(\d+)-line-(\d+)$")
_STYLE_EVIDENCE_RE = re.compile(r"^tool-call-(\d+)-style-evidence-(\d+)-(\d+)$")
_ANY_HANDLE_RE = re.compile(
    r"tool-call-\d+-(?:evidence|product|source|line)-\d+",
    re.IGNORECASE,
)
_ITEM_REF_RE = re.compile(r"^item-(\d{3})$")
_HIGH_RISK_ATOM_RE = re.compile(
    r"(?:[$€£¥￥]\s*\d[\d,.]*|\d+(?:[.,:]\d+)*(?:%|°[CF]|℃|℉|g|kg|mg|ml|l|分钟|小时|秒|元)?|(?=[A-Za-z0-9_-]{8,}\b)(?=[A-Za-z0-9_-]*[0-9_-])[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)
_MATERIAL_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_FORBIDDEN_SIDECAR_KEYS = frozenset(
    {
        "asset_id",
        "image_path",
        "local_path",
        "source_dataset",
        "source_record_id",
        "registry_sha256",
        "registry_runtime_sha256",
        "runtime_binding_sha256",
    }
)
_FALLBACK_MARKERS: dict[str, tuple[str, ...]] = {
    "knowledge.visual_encyclopedia": (
        "not enough evidence",
        "unable to verify",
        "ambiguous",
        "证据不足",
        "无法验证",
        "有歧义",
    ),
    "product.exact_match": (
        "no supported match",
        "no exact match",
        "unable to verify",
        "没有受支持的匹配",
        "没有同款",
        "无法验证",
    ),
    "product.multi_search": (
        "no supported item",
        "unresolved",
        "unable to identify",
        "没有受支持的商品",
        "未解决",
        "无法识别",
    ),
    "product.style_recommendation": (
        "no supported alternative",
        "unable to recommend",
        "没有受支持的替代",
        "无法推荐",
    ),
    "utility.document_reading": (
        "unreadable",
        "insufficient text",
        "unable to read",
        "无法辨认",
        "文字不足",
        "无法读取",
    ),
    "utility.recipe_guidance": (
        "no supported recipe",
        "dish is unresolved",
        "not enough evidence",
        "没有受支持的食谱",
        "菜品未确定",
        "证据不足",
    ),
}
_UNTRUSTED_DOCUMENT_MARKERS = (
    "untrusted",
    "transcription only",
    "not independently verified",
    "不可信",
    "仅作转录",
    "未经独立验证",
)
_UNSUPPORTED_PRODUCT_CLAIMS = (
    "price",
    "stock",
    "availability",
    "价格",
    "售价",
    "库存",
    "有货",
)
_UNSUPPORTED_RECIPE_GUARANTEES = (
    "allergy-free",
    "guaranteed safe",
    "safe for everyone",
    "无过敏",
    "保证安全",
    "绝对安全",
)

_ASSET_ARGUMENT_PROJECTION: dict[str, JSONValue] = {"asset_handle": "query_asset"}
_EXACT_ALLOWED_SEQUENCES = frozenset(
    {
        ("image_product_search",),
        ("text_product_search",),
        ("image_product_search", "text_product_search"),
        ("text_product_search", "image_product_search"),
    }
)
_MULTI_ALLOWED_SEQUENCES = frozenset({("multi_product_search",)})
_ENCYCLOPEDIA_ALLOWED_SEQUENCES = frozenset(
    {
        ("encyclopedia_lookup",),
        ("object_detect", "encyclopedia_lookup"),
    }
)
_RECIPE_ALLOWED_SEQUENCES = frozenset(
    {
        ("recipe_lookup",),
        ("object_detect", "recipe_lookup"),
    }
)
_DOCUMENT_ALLOWED_SEQUENCES = frozenset({("document_ocr",)})
_STYLE_ALLOWED_SEQUENCES = frozenset({("style_similar_search",)})
_SECTION_WITH_COLON_RE = re.compile(
    r"^\s*(?:\*\*|__)?(?P<label>[a-z][a-z0-9_]*)(?:\*\*|__)?\s*:\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_MARKDOWN_SECTION_RE = re.compile(
    r"^\s*#{1,6}\s*(?P<label>[a-z][a-z0-9_]*)\s*:?\s*$", re.IGNORECASE
)

GCS_REASON_CODES: tuple[str, ...] = (
    "assistant_hard_error",
    "card_contract_failed",
    "evidence_card_union_mismatch",
    "evidence_handle_unknown",
    "fallback_contract_failed",
    "input_binding_invalid",
    "material_claim_uncited",
    "multi_mapping_invalid",
    "output_section_invalid",
    "receipt_trace_mismatch",
    "requested_item_coverage_unresolved",
    "route_unacceptable",
    "scorer_sidecar_invalid",
    "scorer_sidecar_missing",
    "semantic_claim_support_unresolved",
    "style_oracle_unavailable",
    "tool_argument_invalid",
    "tool_contract_failed",
    "tool_evidence_mismatch",
    "unsupported_claim",
)
GCS_V2_REASON_CODES: tuple[str, ...] = tuple(
    sorted(
        {
            *(item for item in GCS_REASON_CODES if item != "style_oracle_unavailable"),
            "style_evidence_invalid",
            "style_mode_mismatch",
            "style_status_invalid",
            "style_target_family_mismatch",
        }
    )
)

_GCS_POLICY_PAYLOAD: dict[str, JSONValue] = {
    "policy_version": GCS_POLICY_VERSION,
    "capabilities": list(GCS_CAPABILITY_ORDER),
    "components": list(GCS_COMPONENTS),
    "formula": "int(all_five_components)",
    "fixed_full_denominator": True,
    "missing_invalid_or_unresolved_component_value": 0,
    "noskill_route_policy": "not_applicable_counts_as_pass_only_with_empty_route_identity",
    "section_grammar": "canonical_ascii_identifier_line_heading-v1",
    "section_grammar_contract": {
        "colon_heading_pattern": _SECTION_WITH_COLON_RE.pattern,
        "markdown_heading_pattern": _MARKDOWN_SECTION_RE.pattern,
        "requirements": [
            "every TaskSpec required section occurs exactly once",
            "each required section is non-empty",
            "non-empty content before the first required section is invalid",
        ],
    },
    "semantic_policy": (
        "citation_completeness_handle_binding_and_exact_high_risk_atom_support"
    ),
    "oracle_contract": {
        "public_asset_argument_projection": _ASSET_ARGUMENT_PROJECTION,
        "private_argument_hash_binding": "sha256 canonical {asset_id: authoritative query asset id}",
        "successful_trace_evidence_binding": (
            "zip successful trace subsequence to visible evidence, then bind sidecar by call identity"
        ),
        "public_payload_closure": (
            "product payloads close through cards; knowledge sources and OCR lines close through visible text; detection labels/confidences close through visible detections"
        ),
        "handle_patterns": {
            "product_evidence": _PRODUCT_EVIDENCE_RE.pattern,
            "product_id": _PRODUCT_ID_RE.pattern,
            "source": _SOURCE_RE.pattern,
            "ocr_line": _LINE_RE.pattern,
            "multi_item": _ITEM_REF_RE.pattern,
            "matching": "ASCII case-insensitive; full-match except response handle scan",
        },
        "high_risk_atom_pattern": _HIGH_RISK_ATOM_RE.pattern,
        "material_statement_split_pattern": _MATERIAL_SPLIT_RE.pattern,
        "unsupported_product_claim_markers": list(_UNSUPPORTED_PRODUCT_CLAIMS),
        "unsupported_recipe_guarantee_markers": list(_UNSUPPORTED_RECIPE_GUARANTEES),
        "untrusted_document_markers": list(_UNTRUSTED_DOCUMENT_MARKERS),
        "capabilities": {
            "knowledge.visual_encyclopedia": {
                "allowed_tool_sequences": [
                    list(item) for item in sorted(_ENCYCLOPEDIA_ALLOWED_SEQUENCES)
                ],
                "positive": "non-empty bound source set; every material statement cites a known source handle",
                "fallback": "empty source set plus capability fallback grammar",
                "payload_kinds": ["detections_v1", "knowledge_sources_v1"],
            },
            "product.exact_match": {
                "allowed_tool_sequences": [
                    list(item) for item in sorted(_EXACT_ALLOWED_SEQUENCES)
                ],
                "positive": "one or more eligible candidates; only eligible candidates participate in exact card/title/product/evidence closure",
                "fallback": "zero eligible candidates, including an all-ineligible raw candidate set, plus capability fallback grammar",
                "payload_kinds": ["product_candidates_v1"],
            },
            "product.multi_search": {
                "allowed_tool_sequences": [
                    list(item) for item in sorted(_MULTI_ALLOWED_SEQUENCES)
                ],
                "positive": "ExpectedMultiItemSet required; exact item/status/candidate/card/item_refs/quantity closure",
                "fallback": "resolved expected set with zero matched items plus capability fallback grammar",
                "payload_kinds": ["multi_mapping_v1"],
            },
            "product.style_recommendation": {
                "allowed_tool_sequences": [],
                "positive": "unavailable in v1 until the public-safe Style adapter is bound",
                "fallback": "fail closed",
                "payload_kinds": [],
                "oracle_coverage_available": False,
            },
            "utility.document_reading": {
                "allowed_tool_sequences": [
                    list(item) for item in sorted(_DOCUMENT_ALLOWED_SEQUENCES)
                ],
                "positive": "exactly one OCR; every material statement cites an untrusted line whose normalized text supports each high-risk atom",
                "fallback": "empty line set plus capability fallback grammar",
                "payload_kinds": ["ocr_lines_v1"],
                "required_content_trust": "untrusted_document_text",
            },
            "utility.recipe_guidance": {
                "allowed_tool_sequences": [
                    list(item) for item in sorted(_RECIPE_ALLOWED_SEQUENCES)
                ],
                "positive": "non-empty bound source set; every material statement and high-risk atom is source-supported; no safety guarantee",
                "fallback": "empty source set plus capability fallback grammar",
                "payload_kinds": ["detections_v1", "knowledge_sources_v1"],
            },
        },
    },
    "fallback_markers": {
        key: list(value) for key, value in sorted(_FALLBACK_MARKERS.items())
    },
    "reason_codes": list(GCS_REASON_CODES),
    "bootstrap": {
        "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
        "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
        "replicates": GCS_BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "method": "paired_component_percentile",
        "percentile_definition": "Hyndman-Fan type 7 linear interpolation",
        "unit": "query_connected_component",
        "group_fields": list(GROUP_FIELDS),
        "draw_stream_key": [
            "policy_version",
            "root_seed",
            "scope",
            "population_mapping_sha256",
        ],
        "derived_seed": "unsigned big-endian integer from first 16 SHA-256 bytes",
        "prng": "Python random.Random MT19937 with randrange over sorted component ids",
        "draw_shape": "K component ids sampled with replacement for each of 10000 replicates, where K is the component count",
        "draw_stream_hash": "SHA-256 over concatenated canonical JSON bytes of each ordered draw",
        "contrast_orientation": "lexicographic config pair; reverse by exact CI endpoint swap and negation",
        "missing_capability_rule": "any missing capability replicate makes that capability and macro CI unavailable",
        "precision_warning_component_count_lt": 20,
        "zero_variance_rule": "all non-empty reported and partially observed replicate delta series are constant",
        "descriptive_only_rule": "component_count equals one or zero_variance is true",
    },
    "system_gain_thresholds_pp": {
        "macro_delta_min": SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
        "macro_ci95_low_min": SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
        "hard_error_delta_max": SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
        "per_capability_delta_min": SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN,
        "availability": "both baseline and treatment oracle coverage complete; treatment headline available; macro CI available",
    },
}
validate_json_value(_GCS_POLICY_PAYLOAD)
GCS_POLICY_SHA256 = sha256_bytes(canonical_json_bytes(_GCS_POLICY_PAYLOAD))

_GCS_V2_POLICY_PAYLOAD: dict[str, JSONValue] = {
    "policy_version": GCS_V2_POLICY_VERSION,
    "predecessor_policy_version": GCS_POLICY_VERSION,
    "capabilities": list(GCS_CAPABILITY_ORDER),
    "components": list(GCS_COMPONENTS),
    "formula": "int(all_five_components)",
    "fixed_full_denominator": True,
    "row_failure_semantics": {
        "valid_terminal_model_or_tool_failure": "five-component row with failed components equal to zero",
        "semantic_claim_support_unresolved": "row zero; does not make oracle coverage unavailable",
        "oracle_coverage": "implementation availability only",
        "missing_or_invalid_sidecar": "fatal population integrity error",
        "missing_terminal_query_config_row": "fatal population integrity error",
    },
    "scorer_evidence": {
        "policy_version": GCS_SCORER_EVIDENCE_V2_POLICY_VERSION,
        "schema_version": 2,
        "atomic_checkpoint_required": True,
        "private_fields_forbidden": True,
        "successful_call_identity_binding": [
            "call_index",
            "tool_name",
            "arguments_sha256",
            "result_sha256",
        ],
        "assistant_binding": [
            "matrix_run_id",
            "instance_id",
            "request_sha256",
            "query_id",
            "config",
            "query_artifact_sha256",
            "assistant_result_sha256",
            "assistant_receipt_sha256",
        ],
    },
    "oracle_contract": {
        "style": {
            "allowed_tool_sequences": [["style_similar_search"]],
            "argument_binding": (
                "sha256 canonical {asset_id: authoritative query asset id, "
                "query: frozen Query.text}"
            ),
            "payload_kind": "style_candidates_v2",
            "support_statuses": ["candidates", "no_result", "unsupported"],
            "submodes": [
                "same_category_alternative",
                "cross_category_coordination",
            ],
            "candidate_closure": (
                "exact cards, product/evidence handles, Style facet handles, "
                "mode, family, category, facet value, confidence, provenance"
            ),
            "fallback": (
                "no_result or unsupported with zero candidates/cards/handles and "
                "an explicit evidence-limitation marker"
            ),
        },
        "non_style_payload_kinds": [
            "product_candidates_v1",
            "multi_mapping_v1",
            "knowledge_sources_v1",
            "ocr_lines_v1",
            "detections_v1",
        ],
        "ocr_line_handle_pattern": _LINE_RE.pattern,
        "style_evidence_handle_pattern": _STYLE_EVIDENCE_RE.pattern,
    },
    "reason_codes": list(GCS_V2_REASON_CODES),
    "bootstrap": {
        "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
        "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
        "replicates": GCS_BOOTSTRAP_REPLICATES,
        "unit": "query_connected_component",
        "algorithm_unchanged_from_v1": True,
    },
    "system_gain_thresholds_pp": {
        "macro_delta_min": SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
        "macro_ci95_low_min": SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
        "hard_error_delta_max": SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
        "per_capability_delta_min": SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN,
    },
}
validate_json_value(_GCS_V2_POLICY_PAYLOAD)
GCS_V2_POLICY_SHA256 = sha256_bytes(canonical_json_bytes(_GCS_V2_POLICY_PAYLOAD))
if GCS_V2_POLICY_SHA256 != GCS_V2_EXPECTED_POLICY_SHA256:
    raise RuntimeError("tracked GCS v2 policy changed without an identity revision")

_GCS_V2_ALIAS_MAPPING_POLICY_PAYLOAD: dict[str, JSONValue] = {
    "policy_version": GCS_V2_SIDECAR_MAPPING_POLICY_VERSION,
    "gcs_policy_version": GCS_V2_POLICY_VERSION,
    "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
    "logical_config_order": list(MAIN_CONFIG_ORDER),
    "physical_config_order": list(GCS_V2_PHYSICAL_CONFIG_ORDER),
    "fixed_logical_denominator": 5,
    "fixed_physical_sidecar_denominator": 4,
    "full_projection": {
        "logical_config": "full",
        "physical_config": GCS_V2_FULL_ALIAS_SOURCE_CONFIG,
        "required_alias_policy_version": (
            PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
        ),
        "required_stage_decision": "rolled_back",
        "required_reuse_scope": "assistant_and_evaluator_query_artifacts",
        "required_provider_model_call_count": 0,
        "required_rejected_candidate_use": "diagnostic_only",
        "sidecar_rule": (
            "reuse the canonical public_scorer_evidence subdocument embedded in "
            "the physical s1s2 Assistant checkpoint; never create, copy, rewrite, "
            "or re-hash a Full sidecar"
        ),
        "score_rule": (
            "score the physical s1s2 artifacts under canonical GCS v2, then "
            "change only the logical score config to full"
        ),
    },
    "matrix_invariants": [
        "five strict logical launch instances per query",
        "four strict physical sidecar bindings per query",
        "one complete rolled-back full-to-s1s2 execution alias inventory",
        "one Full alias receipt per physical s1s2 shard",
        "one matrix-wide query artifact and split manifest",
        "Full and s1s2 share byte-exact physical evidence and score semantics",
    ],
    "physical_binding": {
        "checkpoint": (
            "canonical row with exact launch, request, response, receipt, runtime, "
            "treatment Bank, Query projection, and result closure"
        ),
        "paths": [
            "shards/<physical-shard>/assistant/<query-id>.json",
        ],
        "sidecar_state": (
            "available canonical GCS v2 sidecar required for scoring; missing or "
            "invalid evidence is a fatal population-integrity error"
        ),
    },
    "hash_envelopes": [
        "physical_binding_sha256",
        "alias_binding_sha256",
        "mapping_sha256",
        "score_matrix_sha256",
        "summary_sha256",
        "bootstrap_sha256",
        "decision_sha256",
    ],
    "bootstrap": {
        "metric_schema": "portfolio-gcs-component-bootstrap-v1",
        "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "coverage": (
            "all six capabilities present and every row oracle_available; semantic "
            "claim support unresolved remains a valid terminal zero under GCS v2"
        ),
    },
}
validate_json_value(_GCS_V2_ALIAS_MAPPING_POLICY_PAYLOAD)
GCS_V2_ALIAS_MAPPING_POLICY_SHA256 = sha256_bytes(
    canonical_json_bytes(_GCS_V2_ALIAS_MAPPING_POLICY_PAYLOAD)
)


class PortfolioGCSError(ValueError):
    """A GCS contract or fixed population is inconsistent."""


class PortfolioGCSIntegrityError(PortfolioGCSError):
    """A v2 sidecar/checkpoint binding is fatally inconsistent."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _hash_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    validate_json_value(value)  # type: ignore[arg-type]
    return sha256_bytes(canonical_json_bytes(value))


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_json(model.model_dump(mode="json", exclude={field_name}))


def _coerce_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _reject_private_sidecar_values(value: JSONValue, *, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_SIDECAR_KEYS:
                raise ValueError(f"{path} contains forbidden private key {key}")
            _reject_private_sidecar_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_sidecar_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if (
            lowered.startswith("asset.v")
            or re.match(r"^[a-z]:[\\/]", value, re.IGNORECASE)
            or lowered.startswith("query_images/")
            or "../" in lowered
        ):
            raise ValueError(f"{path} contains a private identity or local path")


def gcs_policy_payload() -> dict[str, JSONValue]:
    """Return a detached canonical policy payload suitable for manifest binding."""

    return copy.deepcopy(_GCS_POLICY_PAYLOAD)


def gcs_v2_policy_payload() -> dict[str, JSONValue]:
    """Return the detached canonical GCS v2 policy payload."""

    return copy.deepcopy(_GCS_V2_POLICY_PAYLOAD)


def gcs_v2_alias_mapping_policy_payload() -> dict[str, JSONValue]:
    """Return the detached five-config physical-sidecar mapping policy."""

    return copy.deepcopy(_GCS_V2_ALIAS_MAPPING_POLICY_PAYLOAD)


class ExpectedMultiItemV1(_StrictFrozenModel):
    item_ref: str
    label: str

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("expected multi item_ref is invalid")
        return value


class ScorerProductCandidateV1(_StrictFrozenModel):
    candidate_ordinal: int = Field(ge=1, le=12)
    evidence_reference: str
    product_id: str
    title: str
    eligible: bool
    public_attributes: tuple[tuple[str, str], ...] = ()

    @field_validator("public_attributes", mode="before")
    @classmethod
    def coerce_attributes(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @field_validator("evidence_reference", "product_id", "title")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_handles(self) -> Self:
        evidence = _PRODUCT_EVIDENCE_RE.fullmatch(self.evidence_reference)
        product = _PRODUCT_ID_RE.fullmatch(self.product_id)
        if evidence is None or product is None or evidence.groups() != product.groups():
            raise ValueError("product candidate handles are inconsistent")
        if int(evidence.group(2)) != self.candidate_ordinal:
            raise ValueError("product candidate ordinal differs from its handles")
        if self.public_attributes != tuple(sorted(self.public_attributes)) or len(
            {key for key, _ in self.public_attributes}
        ) != len(self.public_attributes):
            raise ValueError("public product attributes must be sorted and unique")
        for key, value in self.public_attributes:
            _nonblank(key, "public attribute key")
            _nonblank(value, "public attribute value")
            if key in _FORBIDDEN_SIDECAR_KEYS:
                raise ValueError("public product attributes contain a private key")
        return self


class ScorerProductPayloadV1(_StrictFrozenModel):
    candidates: tuple[ScorerProductCandidateV1, ...] = ()

    @field_validator("candidates", mode="before")
    @classmethod
    def coerce_candidates(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        ordinals = tuple(item.candidate_ordinal for item in self.candidates)
        if ordinals != tuple(range(1, len(ordinals) + 1)):
            raise ValueError("product candidate ordinals must be contiguous")
        if len({item.evidence_reference for item in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("product candidate handles must be unique")
        return self


class ScorerMultiItemV1(_StrictFrozenModel):
    item_ref: str
    label: str
    status: Literal["matched", "unresolved"]
    candidate_ordinal: int | None = Field(default=None, ge=1, le=12)

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("multi item_ref is invalid")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "matched") != (self.candidate_ordinal is not None):
            raise ValueError("multi item status and candidate must be paired")
        return self


class ScorerMultiPayloadV1(_StrictFrozenModel):
    items: tuple[ScorerMultiItemV1, ...]
    candidates: tuple[ScorerProductCandidateV1, ...] = ()

    @field_validator("items", "candidates", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        expected_refs = tuple(
            f"item-{index:03d}" for index in range(1, len(self.items) + 1)
        )
        if tuple(item.item_ref for item in self.items) != expected_refs:
            raise ValueError("multi item refs must be contiguous")
        ordinal_sequence = tuple(item.candidate_ordinal for item in self.candidates)
        if ordinal_sequence != tuple(range(1, len(self.candidates) + 1)):
            raise ValueError("multi candidate ordinals must be contiguous")
        candidate_ordinals = set(ordinal_sequence)
        if any(not item.eligible for item in self.candidates):
            raise ValueError("multi candidates must all be eligible")
        if any(
            item.candidate_ordinal not in candidate_ordinals
            for item in self.items
            if item.status == "matched"
        ):
            raise ValueError("multi matched item references an absent candidate")
        referenced = {
            item.candidate_ordinal
            for item in self.items
            if item.candidate_ordinal is not None
        }
        if candidate_ordinals != referenced:
            raise ValueError("multi candidates must exactly cover matched items")
        return self


class ScorerKnowledgeSourceV1(_StrictFrozenModel):
    evidence_reference: str
    title: str
    text: str
    uri: str | None = None

    @field_validator("evidence_reference", "title", "text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _nonblank(value, "uri")
        if not value.startswith("https://") or "@" in value.split("/", 3)[2]:
            raise ValueError("knowledge source URI must be credential-free HTTPS")
        return value

    @field_validator("evidence_reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        if _SOURCE_RE.fullmatch(value) is None:
            raise ValueError("knowledge source handle is invalid")
        return value


class ScorerKnowledgePayloadV1(_StrictFrozenModel):
    sources: tuple[ScorerKnowledgeSourceV1, ...] = ()

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if len({item.evidence_reference for item in self.sources}) != len(self.sources):
            raise ValueError("knowledge source handles must be unique")
        return self


class ScorerOCRLineV1(_StrictFrozenModel):
    line_reference: str
    text: str
    content_trust: Literal["untrusted_document_text"]
    truncated: bool = False
    fields: tuple[tuple[str, str], ...] = ()

    @field_validator("fields", mode="before")
    @classmethod
    def coerce_fields(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @field_validator("line_reference", "text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        if _LINE_RE.fullmatch(self.line_reference) is None:
            raise ValueError("OCR line handle is invalid")
        if self.fields != tuple(sorted(self.fields)) or len(
            {key for key, _ in self.fields}
        ) != len(self.fields):
            raise ValueError("OCR fields must be sorted and unique")
        if any(key in _FORBIDDEN_SIDECAR_KEYS for key, _value in self.fields):
            raise ValueError("OCR fields contain a private key")
        return self


class ScorerOCRPayloadV1(_StrictFrozenModel):
    lines: tuple[ScorerOCRLineV1, ...] = ()

    @field_validator("lines", mode="before")
    @classmethod
    def coerce_lines(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        if len({item.line_reference for item in self.lines}) != len(self.lines):
            raise ValueError("OCR line handles must be unique")
        return self


class ScorerDetectionV1(_StrictFrozenModel):
    item_ref: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("item_ref", "label")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        value = _nonblank(value, info.field_name)
        if info.field_name == "item_ref" and _ITEM_REF_RE.fullmatch(value) is None:
            raise ValueError("detection item_ref is invalid")
        return value


class ScorerDetectionPayloadV1(_StrictFrozenModel):
    detections: tuple[ScorerDetectionV1, ...] = ()

    @field_validator("detections", mode="before")
    @classmethod
    def coerce_detections(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_detections(self) -> Self:
        refs = tuple(item.item_ref for item in self.detections)
        if refs != tuple(f"item-{index:03d}" for index in range(1, len(refs) + 1)):
            raise ValueError("detection item refs must be contiguous")
        return self


class PublicScorerCallEvidenceV1(_StrictFrozenModel):
    call_index: int = Field(ge=1)
    tool_name: str
    arguments_sha256: Sha256
    result_sha256: Sha256
    argument_projection: dict[str, JSONValue]
    payload_kind: ScorerPayloadKind
    payload: dict[str, JSONValue]
    payload_sha256: Sha256

    @field_validator("tool_name")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_public_payload(self) -> Self:
        validate_json_value(self.argument_projection)
        validate_json_value(self.payload)
        _reject_private_sidecar_values(self.argument_projection, path="arguments")
        _reject_private_sidecar_values(self.payload)
        if self.argument_projection == _ASSET_ARGUMENT_PROJECTION:
            pass
        elif set(self.argument_projection) not in (
            {"query"},
            {"entity"},
            {"dish"},
        ) or not all(
            isinstance(item, str) and bool(item.strip())
            for item in self.argument_projection.values()
        ):
            raise ValueError("scorer argument projection is not public-safe")
        payload_model = {
            "product_candidates_v1": ScorerProductPayloadV1,
            "multi_mapping_v1": ScorerMultiPayloadV1,
            "knowledge_sources_v1": ScorerKnowledgePayloadV1,
            "ocr_lines_v1": ScorerOCRPayloadV1,
            "detections_v1": ScorerDetectionPayloadV1,
        }[self.payload_kind]
        payload_model.model_validate(self.payload, strict=True)
        if self.payload_sha256 != _hash_json(self.payload):
            raise ValueError("scorer payload hash mismatch")
        return self


class PublicScorerEvidenceV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-gcs-scorer-evidence-v1"] = (
        GCS_SCORER_EVIDENCE_POLICY_VERSION
    )
    query_id: str
    config: AssistantRunConfig
    query_artifact_sha256: Sha256
    assistant_receipt_sha256: Sha256
    expected_multi_items: tuple[ExpectedMultiItemV1, ...] | None = None
    calls: tuple[PublicScorerCallEvidenceV1, ...]
    evidence_sha256: Sha256

    @field_validator("expected_multi_items", "calls", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return None if value is None else _coerce_tuple(value)

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if tuple(item.call_index for item in self.calls) != tuple(
            sorted(item.call_index for item in self.calls)
        ) or len({item.call_index for item in self.calls}) != len(self.calls):
            raise ValueError("scorer call indices must be sorted and unique")
        if self.expected_multi_items is not None:
            refs = tuple(item.item_ref for item in self.expected_multi_items)
            if not refs or refs != tuple(
                f"item-{index:03d}" for index in range(1, len(refs) + 1)
            ):
                raise ValueError(
                    "expected multi item refs must be non-empty and contiguous"
                )
        if self.evidence_sha256 != _self_hash(self, "evidence_sha256"):
            raise ValueError("public scorer evidence self hash mismatch")
        return self


def make_public_scorer_call_evidence(
    *,
    call_index: int,
    tool_name: str,
    arguments_sha256: str,
    result_sha256: str,
    argument_projection: Mapping[str, JSONValue],
    payload_kind: ScorerPayloadKind,
    payload: Mapping[str, JSONValue],
) -> PublicScorerCallEvidenceV1:
    """Create one self-bound, public-safe scorer call observation."""

    payload_dict = dict(payload)
    return PublicScorerCallEvidenceV1.model_validate(
        {
            "call_index": call_index,
            "tool_name": tool_name,
            "arguments_sha256": arguments_sha256,
            "result_sha256": result_sha256,
            "argument_projection": dict(argument_projection),
            "payload_kind": payload_kind,
            "payload": payload_dict,
            "payload_sha256": _hash_json(payload_dict),
        },
        strict=True,
    )


def make_public_scorer_evidence(
    *,
    query_id: str,
    config: AssistantRunConfig,
    query_artifact_sha256: str,
    assistant_receipt_sha256: str,
    calls: Sequence[PublicScorerCallEvidenceV1],
    expected_multi_items: Sequence[ExpectedMultiItemV1] | None = None,
) -> PublicScorerEvidenceV1:
    """Create one self-authenticating scorer sidecar."""

    unsigned = {
        "schema_version": 1,
        "policy_version": GCS_SCORER_EVIDENCE_POLICY_VERSION,
        "query_id": query_id,
        "config": config,
        "query_artifact_sha256": query_artifact_sha256,
        "assistant_receipt_sha256": assistant_receipt_sha256,
        "expected_multi_items": (
            None if expected_multi_items is None else tuple(expected_multi_items)
        ),
        "calls": tuple(calls),
    }
    return PublicScorerEvidenceV1.model_validate(
        {**unsigned, "evidence_sha256": _hash_json(_jsonable(unsigned))},
        strict=True,
    )


def _jsonable(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-serializable: {type(value)!r}")


def _canonical_model_file_sha256(value: BaseModel) -> str:
    return sha256_bytes(canonical_json_bytes(value.model_dump(mode="json")))


def _validate_assistant_checkpoint_binding_v2(
    content: bytes,
    *,
    launch_instance: PortfolioLaunchInstance,
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    query_artifact_sha256: str,
    split_manifest_sha256: str,
) -> tuple[AssistantRequestSnapshot, PublicScorerEvidenceV2]:
    """Prove that the typed artifacts were derived from these checkpoint bytes."""

    try:
        raw = parse_canonical_json(content, label="GCS v2 Assistant checkpoint")
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "kind",
            "instance_sha256",
            "query_ordinal",
            "request",
            "response",
            "receipt",
            "public_scorer_evidence",
            "row_sha256",
        }:
            raise ValueError("Assistant checkpoint fields are not exact")
        unsigned = dict(raw)
        supplied_row_sha256 = unsigned.pop("row_sha256")
        if (
            raw.get("schema_version") != 2
            or raw.get("kind") != "portfolio-assistant-checkpoint"
            or supplied_row_sha256 != _hash_json(unsigned)
        ):
            raise ValueError("Assistant checkpoint row hash is invalid")
        request = AssistantRequestSnapshot.model_validate_json(
            canonical_json_bytes(raw["request"]), strict=True
        )
        response = AssistantBackendResponse.model_validate_json(
            canonical_json_bytes(raw["response"]), strict=True
        )
        stored_receipt = AssistantExecutionReceipt.model_validate_json(
            canonical_json_bytes(raw["receipt"]), strict=True
        )
        scorer_evidence = require_public_scorer_evidence_v2(
            raw["public_scorer_evidence"]
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError("physical Assistant checkpoint is invalid") from exc

    receipt_response = response
    if response.error_code in {
        "hidden_evaluation_identity",
        "final_packet_isolation_error",
    }:
        receipt_response = response.model_copy(update={"error_code": None})
        expected_outcome = "success"
    else:
        expected_outcome = (
            "success"
            if response.error_code is None
            else "timeout"
            if response.error_code == "timeout"
            else "runtime_error"
        )
    expected_result_fields = {
        "run_id": launch_instance.matrix_run_id,
        "query_id": request.query.query_id,
        "config": request.config,
        "response_text": response.response_text,
        "visible_cards": response.visible_cards,
        "visible_tool_evidence": response.visible_tool_evidence,
        "tool_trace": response.tool_trace,
        "selected_capability": response.selected_capability,
        "skill_slug": response.skill_slug,
        "bank_sha256": (
            request.treatment.bank_sha256
            if response.selected_capability is not None
            else None
        ),
        "route_trace_sha256": response.route_trace_sha256,
        "query_artifact_sha256": query_artifact_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "registry_sha256": response.registry_sha256,
        "registry_runtime_sha256": response.registry_runtime_sha256,
        "backbone_provider": response.backbone_provider,
        "backbone_model": response.backbone_model,
        "backbone_request_id": response.backbone_request_id,
        "usage": response.usage,
        "latency_ms": response.latency_ms,
        "error_code": response.error_code,
    }
    expected_query_input = (
        build_assistant_query_input(query)
        if launch_instance.asset_token is not None
        else build_legacy_assistant_query_input(query)
    )
    if (
        raw["instance_sha256"] != launch_instance.instance_sha256
        or raw["query_ordinal"] != launch_instance.query_ordinal
        or request.matrix_run_id != launch_instance.matrix_run_id
        or request.config != launch_instance.config
        or request.query_ordinal != launch_instance.query_ordinal
        or request.query.query_id != launch_instance.query_id
        or request.query.query_id != query.query_id
        or request.query != expected_query_input
        or request.query.query_sha256 != launch_instance.query_sha256
        or request.query.public_input_sha256 != launch_instance.public_input_sha256
        or response.request_sha256 != request.request_sha256
        or response.backbone_provider != request.backbone.provider
        or response.backbone_model != request.backbone.model
        or response.backbone_endpoint != request.backbone.endpoint
        or response.backbone_identity_sha256 != request.backbone.identity_sha256
        or response.registry_sha256 != request.registry.registry_sha256
        or response.registry_runtime_sha256 != request.registry.registry_runtime_sha256
        or response.budget_sha256 != request.budget.budget_sha256
        or stored_receipt != receipt
        or receipt.request_sha256 != request.request_sha256
        or receipt.response_sha256
        != _hash_json(receipt_response.model_dump(mode="json"))
        or receipt.aggregate_usage != response.usage
        or receipt.tool_trace != response.tool_trace
        or receipt.outcome != expected_outcome
        or (
            receipt.outcome == "success"
            and receipt.query_asset_sha256 != launch_instance.image_sha256
        )
        or (
            receipt.outcome != "success"
            and receipt.query_asset_sha256 not in {None, launch_instance.image_sha256}
        )
        or any(
            getattr(result, field_name) != expected
            for field_name, expected in expected_result_fields.items()
        )
        or scorer_evidence.matrix_run_id != launch_instance.matrix_run_id
        or scorer_evidence.instance_id != launch_instance.instance_sha256
        or scorer_evidence.request_sha256 != request.request_sha256
        or scorer_evidence.query_id != query.query_id
        or scorer_evidence.config != launch_instance.config
        or scorer_evidence.query_artifact_sha256 != query_artifact_sha256
        or scorer_evidence.assistant_result_sha256 != _hash_json(result)
        or scorer_evidence.assistant_receipt_sha256 != receipt.receipt_sha256
    ):
        raise PortfolioGCSError(
            "Assistant checkpoint/launch/result/receipt binding is inconsistent"
        )
    asset_binding = request.query.asset_binding
    if launch_instance.asset_token is not None:
        if (
            asset_binding is None
            or asset_binding.asset_token != launch_instance.asset_token
            or asset_binding.binding_sha256 != launch_instance.asset_binding_sha256
        ):
            raise PortfolioGCSError(
                "Assistant checkpoint asset binding differs from Core launch"
            )
    elif (
        asset_binding is not None
        or launch_instance.asset_id != query.asset_id
        or launch_instance.image_path != query.image_path
    ):
        raise PortfolioGCSError(
            "Assistant checkpoint asset binding differs from dev launch"
        )
    return request, scorer_evidence


class GCSPhysicalSidecarBindingV2(_StrictFrozenModel):
    """One immutable physical scorer sidecar in a non-aliased shard."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-gcs-five-config-physical-sidecar-map-v2"] = (
        GCS_V2_SIDECAR_MAPPING_POLICY_VERSION
    )
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    matrix_run_id: str
    accepted_batch_id: str
    shard_id: str
    config: PhysicalAssistantRunConfig
    query_ordinal: int = Field(ge=0)
    query_id: str
    evaluation_query_sha256: Sha256
    instance_sha256: Sha256
    query_sha256: Sha256
    public_input_sha256: Sha256
    image_sha256: Sha256
    assistant_output_relpath: str
    scorer_checkpoint_relpath: str
    query_artifact_sha256: Sha256
    split_manifest_sha256: Sha256
    assistant_request_sha256: Sha256
    treatment_bank_sha256: Sha256 | None
    assistant_receipt_sha256: Sha256
    assistant_checkpoint_file_sha256: Sha256
    sidecar_status: GCSPhysicalSidecarStatus
    scorer_evidence_sha256: Sha256 | None = None
    scorer_evidence_document_sha256: Sha256 | None = None
    sidecar_failure_code: (
        Literal["scorer_sidecar_missing", "scorer_sidecar_invalid"] | None
    ) = None
    runtime_lock_sha256: Sha256
    launch_plan_sha256: Sha256
    binding_sha256: Sha256

    @field_validator("matrix_run_id", "accepted_batch_id", "shard_id", "query_id")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("assistant_output_relpath", "scorer_checkpoint_relpath")
    @classmethod
    def validate_relative_path(cls, value: str, info) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError(f"{info.field_name} is not normalized relative POSIX")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if (self.config == "noskill") != (self.treatment_bank_sha256 is None):
            raise ValueError("physical GCS treatment Bank identity is inconsistent")
        if (
            self.assistant_output_relpath
            != f"shards/{self.shard_id}/assistant/{self.query_id}.json"
            or self.scorer_checkpoint_relpath != self.assistant_output_relpath
        ):
            raise ValueError("physical GCS sidecar is not embedded in its checkpoint")
        if self.sidecar_status == "available":
            if (
                self.scorer_evidence_sha256 is None
                or self.scorer_evidence_document_sha256 is None
                or self.sidecar_failure_code is not None
            ):
                raise ValueError("available physical sidecar lacks exact hashes")
        elif self.sidecar_status == "missing":
            if (
                self.scorer_evidence_sha256 is not None
                or self.scorer_evidence_document_sha256 is not None
                or self.sidecar_failure_code != "scorer_sidecar_missing"
            ):
                raise ValueError("missing physical sidecar binding is inconsistent")
        elif (
            self.scorer_evidence_sha256 is not None
            or self.scorer_evidence_document_sha256 is None
            or self.sidecar_failure_code != "scorer_sidecar_invalid"
        ):
            raise ValueError("invalid physical sidecar binding is inconsistent")
        if self.binding_sha256 != _self_hash(self, "binding_sha256"):
            raise ValueError("physical scorer sidecar binding hash mismatch")
        return self


def make_gcs_physical_sidecar_binding_v2(
    *,
    launch_instance: PortfolioLaunchInstance,
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV2,
    query_artifact_sha256: str,
    split_manifest_sha256: str,
    assistant_checkpoint_file_bytes: bytes,
    runtime_lock_sha256: str,
    launch_plan_sha256: str,
) -> GCSPhysicalSidecarBindingV2:
    """Bind one embedded canonical v2 sidecar to its physical checkpoint."""

    if (
        not _strict_model_revalidates(launch_instance, PortfolioLaunchInstance)
        or not _strict_model_revalidates(result, AssistantResult)
        or not _strict_model_revalidates(receipt, AssistantExecutionReceipt)
        or not _strict_model_revalidates(sidecar, PublicScorerEvidenceV2)
        or not _sidecar_binding_valid(query, result, receipt, sidecar)
    ):
        raise PortfolioGCSError("physical scorer sidecar contract is invalid")
    if sidecar.config == "full" or launch_instance.config == "full":
        raise PortfolioGCSError("a physical Full scorer sidecar is forbidden")
    if (
        result.run_id != launch_instance.matrix_run_id
        or result.query_id != launch_instance.query_id
        or result.config != launch_instance.config
        or result.query_artifact_sha256 != query_artifact_sha256
        or query.query_id != launch_instance.query_id
    ):
        raise PortfolioGCSError("physical launch/result/sidecar identity drifted")
    verified_request, embedded_sidecar = _validate_assistant_checkpoint_binding_v2(
        assistant_checkpoint_file_bytes,
        launch_instance=launch_instance,
        query=query,
        result=result,
        receipt=receipt,
        query_artifact_sha256=query_artifact_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    if embedded_sidecar != sidecar:
        raise PortfolioGCSError("physical scorer sidecar differs from its checkpoint")
    canonical_sidecar_bytes = canonical_json_bytes(sidecar.model_dump(mode="json"))
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_SIDECAR_MAPPING_POLICY_VERSION,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "matrix_run_id": launch_instance.matrix_run_id,
        "accepted_batch_id": launch_instance.accepted_batch_id,
        "shard_id": launch_instance.shard_id,
        "config": sidecar.config,
        "query_ordinal": launch_instance.query_ordinal,
        "query_id": sidecar.query_id,
        "evaluation_query_sha256": _hash_json(query.model_dump(mode="json")),
        "instance_sha256": launch_instance.instance_sha256,
        "query_sha256": launch_instance.query_sha256,
        "public_input_sha256": launch_instance.public_input_sha256,
        "image_sha256": launch_instance.image_sha256,
        "assistant_output_relpath": launch_instance.assistant_output_relpath,
        "scorer_checkpoint_relpath": launch_instance.assistant_output_relpath,
        "query_artifact_sha256": sidecar.query_artifact_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "assistant_request_sha256": verified_request.request_sha256,
        "treatment_bank_sha256": verified_request.treatment.bank_sha256,
        "assistant_receipt_sha256": sidecar.assistant_receipt_sha256,
        "assistant_checkpoint_file_sha256": sha256_bytes(
            assistant_checkpoint_file_bytes
        ),
        "sidecar_status": "available",
        "scorer_evidence_sha256": sidecar.evidence_sha256,
        "scorer_evidence_document_sha256": sha256_bytes(canonical_sidecar_bytes),
        "sidecar_failure_code": None,
        "runtime_lock_sha256": runtime_lock_sha256,
        "launch_plan_sha256": launch_plan_sha256,
    }
    return GCSPhysicalSidecarBindingV2.model_validate(
        {**unsigned, "binding_sha256": _hash_json(unsigned)}, strict=True
    )


def make_gcs_unavailable_physical_sidecar_binding_v2(
    *,
    launch_instance: PortfolioLaunchInstance,
    config: PhysicalAssistantRunConfig,
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    query_artifact_sha256: str,
    split_manifest_sha256: str,
    assistant_checkpoint_file_bytes: bytes,
    sidecar_status: Literal["missing", "invalid"],
    observed_sidecar_file_sha256: str | None = None,
    runtime_lock_sha256: str,
    launch_plan_sha256: str,
) -> GCSPhysicalSidecarBindingV2:
    """Reject a missing/invalid v2 sidecar before a score population is built."""

    raise PortfolioGCSIntegrityError(
        "canonical GCS v2 requires an embedded valid scorer sidecar"
    )


class GCSFullArtifactAliasReceiptV2(_StrictFrozenModel):
    """Typed view of the existing create-only Full shard alias receipt."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-shard-artifact-alias"] = "portfolio-shard-artifact-alias"
    policy_version: Literal[PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION] = (
        PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
    )
    matrix_run_id: str
    accepted_batch_id: str
    target_shard_id: str
    target_config: Literal["full"]
    source_shard_id: str
    source_config: Literal["s1s2"]
    query_ids: tuple[str, ...]
    treatment_alias_sha256: Sha256
    source_bank_sha256: Sha256
    target_bank_sha256: Sha256
    source_shard_summary_file_sha256: Sha256
    source_shard_summary_sha256: Sha256
    source_shard_audit_file_sha256: Sha256
    source_shard_audit_sha256: Sha256
    reuse_scope: Literal["assistant_and_evaluator_query_artifacts"] = (
        "assistant_and_evaluator_query_artifacts"
    )
    provider_model_call_count: Literal[0] = 0
    rejected_candidate_use: Literal["diagnostic_only"] = "diagnostic_only"
    runtime_lock_sha256: Sha256
    launch_plan_sha256: Sha256
    alias_receipt_sha256: Sha256

    @field_validator("query_ids", mode="before")
    @classmethod
    def coerce_query_ids(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator(
        "matrix_run_id",
        "accepted_batch_id",
        "target_shard_id",
        "source_shard_id",
    )
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if (
            not self.query_ids
            or len(set(self.query_ids)) != len(self.query_ids)
            or any(
                not query_id or query_id != query_id.strip()
                for query_id in self.query_ids
            )
        ):
            raise ValueError("Full alias query inventory is empty or duplicated")
        if self.target_shard_id == self.source_shard_id:
            raise ValueError("Full alias target and source shard must differ")
        if self.source_bank_sha256 != self.target_bank_sha256:
            raise ValueError("Full alias does not reuse the accepted parent Bank")
        if self.alias_receipt_sha256 != _self_hash(self, "alias_receipt_sha256"):
            raise ValueError("Full alias receipt hash mismatch")
        return self


class GCSFullAliasBindingV2(_StrictFrozenModel):
    treatment_alias: PortfolioExecutionArtifactAlias
    receipt: GCSFullArtifactAliasReceiptV2
    alias_receipt_file_sha256: Sha256
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if (
            self.treatment_alias.target_config != "full"
            or self.treatment_alias.source_config != "s1s2"
            or self.treatment_alias.stage_decision != "rolled_back"
            or self.treatment_alias.alias_sha256 != self.receipt.treatment_alias_sha256
            or self.treatment_alias.source_bank_sha256
            != self.receipt.source_bank_sha256
            or self.treatment_alias.target_bank_sha256
            != self.receipt.target_bank_sha256
        ):
            raise ValueError("Full treatment alias differs from its shard receipt")
        if self.alias_receipt_file_sha256 != _canonical_model_file_sha256(self.receipt):
            raise ValueError("Full alias receipt file hash mismatch")
        if self.binding_sha256 != _self_hash(self, "binding_sha256"):
            raise ValueError("Full alias binding hash mismatch")
        return self


def make_gcs_full_alias_binding_v2(
    receipt: GCSFullArtifactAliasReceiptV2 | Mapping[str, object],
    treatment_alias: PortfolioExecutionArtifactAlias | Mapping[str, object],
    *,
    alias_receipt_file_bytes: bytes,
) -> GCSFullAliasBindingV2:
    """Strictly parse and bind one canonical ``artifact-alias.json``."""

    try:
        typed = (
            receipt
            if isinstance(receipt, GCSFullArtifactAliasReceiptV2)
            else GCSFullArtifactAliasReceiptV2.model_validate(receipt, strict=True)
        )
        if not _strict_model_revalidates(typed, GCSFullArtifactAliasReceiptV2):
            raise ValueError("alias receipt failed strict revalidation")
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError("Full artifact alias receipt is invalid") from exc
    try:
        typed_treatment_alias = (
            treatment_alias
            if isinstance(treatment_alias, PortfolioExecutionArtifactAlias)
            else PortfolioExecutionArtifactAlias.model_validate(
                treatment_alias, strict=True
            )
        )
        if not _strict_model_revalidates(
            typed_treatment_alias, PortfolioExecutionArtifactAlias
        ):
            raise ValueError("treatment alias failed strict revalidation")
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError("Full treatment alias is invalid") from exc
    canonical_receipt_bytes = canonical_json_bytes(typed.model_dump(mode="json"))
    if alias_receipt_file_bytes != canonical_receipt_bytes:
        raise PortfolioGCSError("Full artifact alias receipt file is not canonical")
    unsigned = {
        "treatment_alias": typed_treatment_alias,
        "receipt": typed,
        "alias_receipt_file_sha256": sha256_bytes(alias_receipt_file_bytes),
    }
    try:
        return GCSFullAliasBindingV2.model_validate(
            {**unsigned, "binding_sha256": _hash_json(_jsonable(unsigned))},
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError(
            "Full artifact/treatment alias binding is inconsistent"
        ) from exc


class GCSLogicalSidecarProjectionV2(_StrictFrozenModel):
    """One logical GCS row projected from one physical scorer sidecar."""

    query_id: str
    logical_config: AssistantRunConfig
    logical_shard_id: str
    physical_config: PhysicalAssistantRunConfig
    physical_shard_id: str
    physical_query_ordinal: int = Field(ge=0)
    accepted_batch_id: str
    evaluation_query_sha256: Sha256
    logical_instance_sha256: Sha256
    physical_instance_sha256: Sha256
    query_sha256: Sha256
    public_input_sha256: Sha256
    image_sha256: Sha256
    logical_assistant_output_relpath: str
    physical_assistant_output_relpath: str
    scorer_checkpoint_relpath: str
    projection_kind: GCSProjectionKind
    physical_binding_sha256: Sha256
    query_artifact_sha256: Sha256
    split_manifest_sha256: Sha256
    assistant_request_sha256: Sha256
    treatment_bank_sha256: Sha256 | None
    assistant_receipt_sha256: Sha256
    assistant_checkpoint_file_sha256: Sha256
    sidecar_status: GCSPhysicalSidecarStatus
    scorer_evidence_sha256: Sha256 | None = None
    scorer_evidence_document_sha256: Sha256 | None = None
    sidecar_failure_code: (
        Literal["scorer_sidecar_missing", "scorer_sidecar_invalid"] | None
    ) = None
    treatment_alias_sha256: Sha256 | None = None
    alias_receipt_sha256: Sha256 | None = None
    alias_receipt_file_sha256: Sha256 | None = None
    projection_sha256: Sha256

    @field_validator(
        "query_id",
        "logical_shard_id",
        "physical_shard_id",
        "accepted_batch_id",
        "logical_assistant_output_relpath",
        "physical_assistant_output_relpath",
        "scorer_checkpoint_relpath",
    )
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "logical_assistant_output_relpath",
        "physical_assistant_output_relpath",
        "scorer_checkpoint_relpath",
    )
    @classmethod
    def validate_projection_path(cls, value: str, info) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError(f"{info.field_name} is not normalized relative POSIX")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.sidecar_status == "available":
            if (
                self.scorer_evidence_sha256 is None
                or self.scorer_evidence_document_sha256 is None
                or self.sidecar_failure_code is not None
            ):
                raise ValueError("available logical sidecar projection is incomplete")
        elif self.sidecar_status == "missing":
            if (
                self.scorer_evidence_sha256 is not None
                or self.scorer_evidence_document_sha256 is not None
                or self.sidecar_failure_code != "scorer_sidecar_missing"
            ):
                raise ValueError("missing logical sidecar projection is inconsistent")
        elif (
            self.scorer_evidence_sha256 is not None
            or self.scorer_evidence_document_sha256 is None
            or self.sidecar_failure_code != "scorer_sidecar_invalid"
        ):
            raise ValueError("invalid logical sidecar projection is inconsistent")
        alias_fields = (
            self.treatment_alias_sha256,
            self.alias_receipt_sha256,
            self.alias_receipt_file_sha256,
        )
        if self.projection_kind == "direct":
            if (
                self.logical_config == "full"
                or self.logical_config != self.physical_config
                or self.logical_shard_id != self.physical_shard_id
                or self.logical_instance_sha256 != self.physical_instance_sha256
                or self.logical_assistant_output_relpath
                != self.physical_assistant_output_relpath
                or any(value is not None for value in alias_fields)
            ):
                raise ValueError("direct GCS sidecar projection is inconsistent")
        elif (
            self.logical_config != "full"
            or self.physical_config != GCS_V2_FULL_ALIAS_SOURCE_CONFIG
            or self.logical_shard_id == self.physical_shard_id
            or self.logical_instance_sha256 == self.physical_instance_sha256
            or self.logical_assistant_output_relpath
            == self.physical_assistant_output_relpath
            or any(value is None for value in alias_fields)
        ):
            raise ValueError("Full GCS sidecar projection is inconsistent")
        if self.projection_sha256 != _self_hash(self, "projection_sha256"):
            raise ValueError("logical GCS sidecar projection hash mismatch")
        return self


def _make_gcs_logical_projection_v2(
    physical: GCSPhysicalSidecarBindingV2,
    *,
    logical_instance: PortfolioLaunchInstance,
    alias: GCSFullAliasBindingV2 | None,
) -> GCSLogicalSidecarProjectionV2:
    receipt = None if alias is None else alias.receipt
    if (
        logical_instance.matrix_run_id != physical.matrix_run_id
        or logical_instance.query_id != physical.query_id
        or logical_instance.query_ordinal != physical.query_ordinal
        or logical_instance.accepted_batch_id != physical.accepted_batch_id
        or logical_instance.query_sha256 != physical.query_sha256
        or logical_instance.public_input_sha256 != physical.public_input_sha256
        or logical_instance.image_sha256 != physical.image_sha256
    ):
        raise ValueError("logical launch instance differs from physical query identity")
    if receipt is None:
        if (
            logical_instance.config != physical.config
            or logical_instance.shard_id != physical.shard_id
            or logical_instance.instance_sha256 != physical.instance_sha256
        ):
            raise ValueError("direct logical launch instance differs from physical")
    elif (
        logical_instance.config != "full"
        or logical_instance.shard_id != receipt.target_shard_id
        or physical.config != "s1s2"
        or physical.shard_id != receipt.source_shard_id
    ):
        raise ValueError("Full logical launch instance differs from alias receipt")
    unsigned = {
        "query_id": physical.query_id,
        "logical_config": physical.config if receipt is None else "full",
        "logical_shard_id": (
            physical.shard_id if receipt is None else receipt.target_shard_id
        ),
        "physical_config": physical.config,
        "physical_shard_id": physical.shard_id,
        "physical_query_ordinal": physical.query_ordinal,
        "accepted_batch_id": physical.accepted_batch_id,
        "evaluation_query_sha256": physical.evaluation_query_sha256,
        "logical_instance_sha256": logical_instance.instance_sha256,
        "physical_instance_sha256": physical.instance_sha256,
        "query_sha256": physical.query_sha256,
        "public_input_sha256": physical.public_input_sha256,
        "image_sha256": physical.image_sha256,
        "logical_assistant_output_relpath": (logical_instance.assistant_output_relpath),
        "physical_assistant_output_relpath": physical.assistant_output_relpath,
        "scorer_checkpoint_relpath": physical.scorer_checkpoint_relpath,
        "projection_kind": (
            "direct" if receipt is None else "execution_artifact_alias"
        ),
        "physical_binding_sha256": physical.binding_sha256,
        "query_artifact_sha256": physical.query_artifact_sha256,
        "split_manifest_sha256": physical.split_manifest_sha256,
        "assistant_request_sha256": physical.assistant_request_sha256,
        "treatment_bank_sha256": physical.treatment_bank_sha256,
        "assistant_receipt_sha256": physical.assistant_receipt_sha256,
        "assistant_checkpoint_file_sha256": (physical.assistant_checkpoint_file_sha256),
        "sidecar_status": physical.sidecar_status,
        "scorer_evidence_sha256": physical.scorer_evidence_sha256,
        "scorer_evidence_document_sha256": physical.scorer_evidence_document_sha256,
        "sidecar_failure_code": physical.sidecar_failure_code,
        "treatment_alias_sha256": (
            None if receipt is None else receipt.treatment_alias_sha256
        ),
        "alias_receipt_sha256": (
            None if receipt is None else receipt.alias_receipt_sha256
        ),
        "alias_receipt_file_sha256": (
            None if alias is None else alias.alias_receipt_file_sha256
        ),
    }
    return GCSLogicalSidecarProjectionV2.model_validate(
        {**unsigned, "projection_sha256": _hash_json(unsigned)}, strict=True
    )


class GCSFiveConfigSidecarMapV2(_StrictFrozenModel):
    """Complete N x 5 logical map backed by exactly N x 4 sidecars."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_version: Literal[
        "portfolio-gcs-five-config-physical-sidecar-map-v2"
    ] = GCS_V2_SIDECAR_MAPPING_POLICY_VERSION
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    base_gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    matrix_run_id: str
    runtime_lock_sha256: Sha256
    launch_plan_sha256: Sha256
    population_mapping_sha256: Sha256
    query_count: int = Field(ge=1)
    physical_sidecar_count: int = Field(ge=4)
    logical_row_count: int = Field(ge=5)
    query_ids: tuple[str, ...]
    config_order: tuple[AssistantRunConfig, ...]
    physical_config_order: tuple[PhysicalAssistantRunConfig, ...]
    launch_instances: tuple[PortfolioLaunchInstance, ...]
    physical_bindings: tuple[GCSPhysicalSidecarBindingV2, ...]
    execution_artifact_aliases: tuple[PortfolioExecutionArtifactAlias, ...]
    full_aliases: tuple[GCSFullAliasBindingV2, ...]
    rows: tuple[GCSLogicalSidecarProjectionV2, ...]
    mapping_sha256: Sha256

    @field_validator(
        "query_ids",
        "config_order",
        "physical_config_order",
        "launch_instances",
        "physical_bindings",
        "execution_artifact_aliases",
        "full_aliases",
        "rows",
        mode="before",
    )
    @classmethod
    def coerce_v2_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @field_validator("matrix_run_id")
    @classmethod
    def validate_matrix_run_id(cls, value: str) -> str:
        return _nonblank(value, "matrix_run_id")

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        if self.config_order != MAIN_CONFIG_ORDER:
            raise ValueError("GCS v2 logical config order is not frozen")
        if self.physical_config_order != GCS_V2_PHYSICAL_CONFIG_ORDER:
            raise ValueError("GCS v2 physical config order is not frozen")
        if (
            len(self.execution_artifact_aliases) != 1
            or self.execution_artifact_aliases[0].target_config != "full"
            or self.execution_artifact_aliases[0].source_config != "s1s2"
            or self.execution_artifact_aliases[0].stage_decision != "rolled_back"
        ):
            raise ValueError("GCS v2 requires exactly the Full rollback alias")
        if (
            self.query_ids != tuple(sorted(self.query_ids))
            or len(set(self.query_ids)) != len(self.query_ids)
            or len(self.query_ids) != self.query_count
        ):
            raise ValueError("GCS v2 query inventory is not sorted and unique")
        if (
            self.physical_sidecar_count != self.query_count * 4
            or len(self.physical_bindings) != self.physical_sidecar_count
            or self.logical_row_count != self.query_count * 5
            or len(self.rows) != self.logical_row_count
        ):
            raise ValueError("GCS v2 physical/logical denominator is inconsistent")

        logical_config_index = {
            config: index for index, config in enumerate(MAIN_CONFIG_ORDER)
        }
        if len(self.launch_instances) != self.query_count * len(MAIN_CONFIG_ORDER):
            raise ValueError("GCS v2 launch instance denominator is inconsistent")
        if self.launch_instances != tuple(
            sorted(
                self.launch_instances,
                key=lambda item: (
                    item.query_id,
                    logical_config_index[item.config],
                ),
            )
        ):
            raise ValueError("GCS v2 launch instances are not canonical")
        launch_by_key: dict[
            tuple[str, AssistantRunConfig], PortfolioLaunchInstance
        ] = {}
        query_id_by_ordinal: dict[int, str] = {}
        observed_instance_ordinals: set[int] = set()
        observed_instance_sha256: set[str] = set()
        for instance in self.launch_instances:
            key = (instance.query_id, instance.config)
            if key in launch_by_key:
                raise ValueError("GCS v2 contains duplicate launch instances")
            launch_by_key[key] = instance
            if instance.matrix_run_id != self.matrix_run_id:
                raise ValueError("GCS v2 launch instance matrix identity drifted")
            prior_query_id = query_id_by_ordinal.setdefault(
                instance.query_ordinal, instance.query_id
            )
            if prior_query_id != instance.query_id:
                raise ValueError("GCS v2 global query ordinal maps to multiple queries")
            if instance.instance_ordinal in observed_instance_ordinals:
                raise ValueError("GCS v2 launch instance ordinal is duplicated")
            observed_instance_ordinals.add(instance.instance_ordinal)
            if instance.instance_sha256 in observed_instance_sha256:
                raise ValueError("GCS v2 launch instance hash is duplicated")
            observed_instance_sha256.add(instance.instance_sha256)
        expected_launch_keys = {
            (query_id, config)
            for query_id in self.query_ids
            for config in MAIN_CONFIG_ORDER
        }
        if set(launch_by_key) != expected_launch_keys:
            raise ValueError("GCS v2 launch instances are not rectangular")
        for query_id in self.query_ids:
            instances = tuple(
                launch_by_key[(query_id, config)] for config in MAIN_CONFIG_ORDER
            )
            if (
                len({item.accepted_batch_id for item in instances}) != 1
                or len({item.query_ordinal for item in instances}) != 1
                or len({item.query_sha256 for item in instances}) != 1
                or len({item.public_input_sha256 for item in instances}) != 1
                or len({item.image_sha256 for item in instances}) != 1
                or len(
                    {
                        (
                            item.asset_id,
                            item.image_path,
                            item.asset_token,
                            item.asset_binding_sha256,
                        )
                        for item in instances
                    }
                )
                != 1
            ):
                raise ValueError("GCS v2 logical query identity differs across configs")

        config_index = {
            config: index for index, config in enumerate(GCS_V2_PHYSICAL_CONFIG_ORDER)
        }
        expected_physical_order = tuple(
            sorted(
                self.physical_bindings,
                key=lambda item: (item.query_id, config_index[item.config]),
            )
        )
        if self.physical_bindings != expected_physical_order:
            raise ValueError("GCS v2 physical bindings are not canonical")
        physical_by_key: dict[
            tuple[str, PhysicalAssistantRunConfig], GCSPhysicalSidecarBindingV2
        ] = {}
        for binding in self.physical_bindings:
            key = (binding.query_id, binding.config)
            if key in physical_by_key:
                raise ValueError("GCS v2 contains duplicate physical sidecars")
            physical_by_key[key] = binding
            if (
                binding.matrix_run_id != self.matrix_run_id
                or binding.runtime_lock_sha256 != self.runtime_lock_sha256
                or binding.launch_plan_sha256 != self.launch_plan_sha256
            ):
                raise ValueError("GCS v2 physical sidecar root identity drifted")
            if binding.sidecar_status != "available":
                raise ValueError("GCS v2 mapping contains unavailable scorer evidence")
            launch_instance = launch_by_key[key]
            if (
                binding.instance_sha256 != launch_instance.instance_sha256
                or binding.shard_id != launch_instance.shard_id
                or binding.query_ordinal != launch_instance.query_ordinal
                or binding.accepted_batch_id != launch_instance.accepted_batch_id
                or binding.query_sha256 != launch_instance.query_sha256
                or binding.public_input_sha256 != launch_instance.public_input_sha256
                or binding.image_sha256 != launch_instance.image_sha256
                or binding.assistant_output_relpath
                != launch_instance.assistant_output_relpath
            ):
                raise ValueError("GCS v2 physical binding differs from launch instance")
        expected_physical_keys = {
            (query_id, config)
            for query_id in self.query_ids
            for config in GCS_V2_PHYSICAL_CONFIG_ORDER
        }
        if set(physical_by_key) != expected_physical_keys:
            raise ValueError("GCS v2 physical sidecars are not rectangular")
        if (
            len({item.query_artifact_sha256 for item in self.physical_bindings}) != 1
            or len({item.split_manifest_sha256 for item in self.physical_bindings}) != 1
        ):
            raise ValueError("GCS v2 physical sidecars span query or split artifacts")
        for query_id in self.query_ids:
            query_bindings = tuple(
                physical_by_key[(query_id, config)]
                for config in GCS_V2_PHYSICAL_CONFIG_ORDER
            )
            if (
                len({item.accepted_batch_id for item in query_bindings}) != 1
                or len({item.query_ordinal for item in query_bindings}) != 1
                or len({item.query_artifact_sha256 for item in query_bindings}) != 1
                or len({item.split_manifest_sha256 for item in query_bindings}) != 1
                or len({item.evaluation_query_sha256 for item in query_bindings}) != 1
                or len({item.query_sha256 for item in query_bindings}) != 1
                or len({item.public_input_sha256 for item in query_bindings}) != 1
                or len({item.image_sha256 for item in query_bindings}) != 1
            ):
                raise ValueError(
                    "GCS v2 query identity differs across physical configs"
                )

        shard_bindings: dict[str, list[GCSPhysicalSidecarBindingV2]] = defaultdict(list)
        for binding in self.physical_bindings:
            shard_bindings[binding.shard_id].append(binding)
        for members in shard_bindings.values():
            if (
                len({item.config for item in members}) != 1
                or len({item.accepted_batch_id for item in members}) != 1
            ):
                raise ValueError("GCS v2 physical shard identity is inconsistent")
            ordered_members = sorted(members, key=lambda item: item.query_ordinal)
            ordinals = tuple(item.query_ordinal for item in ordered_members)
            if len(set(ordinals)) != len(ordinals):
                raise ValueError("GCS v2 shard query ordinals are duplicated")
        batch_config_shards: dict[tuple[str, PhysicalAssistantRunConfig], str] = {}
        batch_config_members: dict[
            tuple[str, PhysicalAssistantRunConfig], tuple[tuple[int, str], ...]
        ] = {}
        for shard_id, members in shard_bindings.items():
            batch = members[0].accepted_batch_id
            config = members[0].config
            key = (batch, config)
            if key in batch_config_shards:
                raise ValueError("GCS v2 batch/config is split across physical shards")
            batch_config_shards[key] = shard_id
            batch_config_members[key] = tuple(
                (item.query_ordinal, item.query_id)
                for item in sorted(members, key=lambda item: item.query_ordinal)
            )
        batches = {batch for batch, _config in batch_config_shards}
        for batch in batches:
            memberships = tuple(
                batch_config_members.get((batch, config))
                for config in GCS_V2_PHYSICAL_CONFIG_ORDER
            )
            if (
                any(membership is None for membership in memberships)
                or len(
                    {membership for membership in memberships if membership is not None}
                )
                != 1
            ):
                raise ValueError(
                    "GCS v2 batch physical configs do not share query membership"
                )

        if self.full_aliases != tuple(
            sorted(
                self.full_aliases,
                key=lambda item: item.receipt.target_shard_id,
            )
        ):
            raise ValueError("GCS v2 Full aliases are not canonical")
        launch_shard_members: dict[str, list[PortfolioLaunchInstance]] = defaultdict(
            list
        )
        for instance in self.launch_instances:
            launch_shard_members[instance.shard_id].append(instance)
        alias_by_source: dict[str, GCSFullAliasBindingV2] = {}
        target_shards: set[str] = set()
        for alias in self.full_aliases:
            receipt = alias.receipt
            if alias.treatment_alias != self.execution_artifact_aliases[0]:
                raise ValueError("GCS v2 Full receipt uses an unbound treatment alias")
            if (
                receipt.source_shard_id in alias_by_source
                or receipt.target_shard_id in target_shards
            ):
                raise ValueError("GCS v2 Full alias shard is duplicated")
            alias_by_source[receipt.source_shard_id] = alias
            target_shards.add(receipt.target_shard_id)
            members = shard_bindings.get(receipt.source_shard_id)
            if members is None or any(item.config != "s1s2" for item in members):
                raise ValueError("GCS v2 Full alias source is not physical S1+S2")
            ordered_members = sorted(members, key=lambda item: item.query_ordinal)
            target_members = sorted(
                launch_shard_members.get(receipt.target_shard_id, ()),
                key=lambda item: item.query_ordinal,
            )
            if (
                receipt.matrix_run_id != self.matrix_run_id
                or receipt.runtime_lock_sha256 != self.runtime_lock_sha256
                or receipt.launch_plan_sha256 != self.launch_plan_sha256
                or receipt.accepted_batch_id != members[0].accepted_batch_id
                or receipt.query_ids != tuple(item.query_id for item in ordered_members)
                or any(
                    item.treatment_bank_sha256 != receipt.source_bank_sha256
                    for item in ordered_members
                )
                or receipt.target_shard_id in shard_bindings
                or not target_members
                or any(
                    item.config != "full"
                    or item.accepted_batch_id != receipt.accepted_batch_id
                    for item in target_members
                )
                or receipt.query_ids != tuple(item.query_id for item in target_members)
                or tuple(item.query_ordinal for item in target_members)
                != tuple(item.query_ordinal for item in ordered_members)
            ):
                raise ValueError("GCS v2 Full alias receipt/source binding drifted")
        s1s2_shards = {
            binding.shard_id
            for binding in self.physical_bindings
            if binding.config == "s1s2"
        }
        if set(alias_by_source) != s1s2_shards:
            raise ValueError("GCS v2 does not map every S1+S2 source shard to Full")

        logical_index = {
            config: index for index, config in enumerate(MAIN_CONFIG_ORDER)
        }
        if self.rows != tuple(
            sorted(
                self.rows,
                key=lambda item: (item.query_id, logical_index[item.logical_config]),
            )
        ):
            raise ValueError("GCS v2 logical projections are not canonical")
        row_by_key: dict[
            tuple[str, AssistantRunConfig], GCSLogicalSidecarProjectionV2
        ] = {}
        for row in self.rows:
            key = (row.query_id, row.logical_config)
            if key in row_by_key:
                raise ValueError("GCS v2 contains duplicate logical projections")
            row_by_key[key] = row
        expected_logical_keys = {
            (query_id, config)
            for query_id in self.query_ids
            for config in MAIN_CONFIG_ORDER
        }
        if set(row_by_key) != expected_logical_keys:
            raise ValueError("GCS v2 logical projections are not rectangular")
        for query_id in self.query_ids:
            for config in GCS_V2_PHYSICAL_CONFIG_ORDER:
                expected = _make_gcs_logical_projection_v2(
                    physical_by_key[(query_id, config)],
                    logical_instance=launch_by_key[(query_id, config)],
                    alias=None,
                )
                if row_by_key[(query_id, config)] != expected:
                    raise ValueError("GCS v2 direct projection differs from sidecar")
            source = physical_by_key[(query_id, "s1s2")]
            expected_full = _make_gcs_logical_projection_v2(
                source,
                logical_instance=launch_by_key[(query_id, "full")],
                alias=alias_by_source[source.shard_id],
            )
            if row_by_key[(query_id, "full")] != expected_full:
                raise ValueError("GCS v2 Full projection differs from physical S1+S2")
        if self.mapping_sha256 != _self_hash(self, "mapping_sha256"):
            raise ValueError("GCS v2 sidecar map hash mismatch")
        return self


def build_gcs_five_config_sidecar_map_v2(
    queries: Sequence[Query],
    launch_instances: Sequence[PortfolioLaunchInstance],
    physical_bindings: Sequence[GCSPhysicalSidecarBindingV2],
    full_aliases: Sequence[GCSFullAliasBindingV2],
    execution_artifact_aliases: Sequence[PortfolioExecutionArtifactAlias],
) -> GCSFiveConfigSidecarMapV2:
    """Build the fixed five-config logical map over four physical sidecars."""

    if any(
        not _strict_model_revalidates(item, PortfolioLaunchInstance)
        for item in launch_instances
    ):
        raise PortfolioGCSError("GCS v2 contains an invalid launch instance")
    if any(
        not _strict_model_revalidates(item, GCSPhysicalSidecarBindingV2)
        for item in physical_bindings
    ):
        raise PortfolioGCSError("GCS v2 contains an invalid physical binding")
    if any(
        not _strict_model_revalidates(item, GCSFullAliasBindingV2)
        for item in full_aliases
    ):
        raise PortfolioGCSError("GCS v2 contains an invalid Full alias binding")
    if any(
        not _strict_model_revalidates(item, PortfolioExecutionArtifactAlias)
        for item in execution_artifact_aliases
    ):
        raise PortfolioGCSError("GCS v2 contains an invalid alias inventory")
    population = build_gcs_population_v2(queries)
    query_ids = tuple(item.query_id for item in population.bindings)
    query_by_id = {query.query_id: query for query in queries}
    logical_order = {config: index for index, config in enumerate(MAIN_CONFIG_ORDER)}
    ordered_launch_instances = tuple(
        sorted(
            launch_instances,
            key=lambda item: (item.query_id, logical_order[item.config]),
        )
    )
    launch_by_key = {
        (item.query_id, item.config): item for item in ordered_launch_instances
    }
    expected_launch_keys = {
        (query_id, config) for query_id in query_ids for config in MAIN_CONFIG_ORDER
    }
    if (
        len(launch_by_key) != len(ordered_launch_instances)
        or set(launch_by_key) != expected_launch_keys
    ):
        raise PortfolioGCSError("GCS v2 launch instances are not rectangular")
    physical_order = {
        config: index for index, config in enumerate(GCS_V2_PHYSICAL_CONFIG_ORDER)
    }
    ordered_physical = tuple(
        sorted(
            physical_bindings,
            key=lambda item: (item.query_id, physical_order[item.config]),
        )
    )
    if any(
        item.query_id not in query_by_id
        or item.evaluation_query_sha256
        != _hash_json(query_by_id[item.query_id].model_dump(mode="json"))
        for item in ordered_physical
    ):
        raise PortfolioGCSError("GCS v2 physical binding differs from Query artifact")
    if not ordered_physical:
        raise PortfolioGCSError("GCS v2 physical sidecar population is empty")
    roots = {
        (
            item.matrix_run_id,
            item.runtime_lock_sha256,
            item.launch_plan_sha256,
        )
        for item in ordered_physical
    }
    if len(roots) != 1:
        raise PortfolioGCSError("GCS v2 physical sidecars span execution roots")
    matrix_run_id, runtime_lock_sha256, launch_plan_sha256 = next(iter(roots))
    ordered_aliases = tuple(
        sorted(full_aliases, key=lambda item: item.receipt.target_shard_id)
    )
    ordered_execution_aliases = tuple(
        sorted(execution_artifact_aliases, key=lambda item: item.target_config)
    )
    alias_by_source = {item.receipt.source_shard_id: item for item in ordered_aliases}
    rows: list[GCSLogicalSidecarProjectionV2] = []
    for physical in ordered_physical:
        rows.append(
            _make_gcs_logical_projection_v2(
                physical,
                logical_instance=launch_by_key[(physical.query_id, physical.config)],
                alias=None,
            )
        )
        if physical.config == "s1s2":
            alias = alias_by_source.get(physical.shard_id)
            if alias is None:
                raise PortfolioGCSError(
                    "GCS v2 S1+S2 sidecar lacks a Full alias receipt"
                )
            rows.append(
                _make_gcs_logical_projection_v2(
                    physical,
                    logical_instance=launch_by_key[(physical.query_id, "full")],
                    alias=alias,
                )
            )
    ordered_rows = tuple(
        sorted(
            rows, key=lambda item: (item.query_id, logical_order[item.logical_config])
        )
    )
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_version": GCS_V2_SIDECAR_MAPPING_POLICY_VERSION,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "base_gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        "matrix_run_id": matrix_run_id,
        "runtime_lock_sha256": runtime_lock_sha256,
        "launch_plan_sha256": launch_plan_sha256,
        "population_mapping_sha256": population.population_mapping_sha256,
        "query_count": len(query_ids),
        "physical_sidecar_count": len(ordered_physical),
        "logical_row_count": len(ordered_rows),
        "query_ids": query_ids,
        "config_order": MAIN_CONFIG_ORDER,
        "physical_config_order": GCS_V2_PHYSICAL_CONFIG_ORDER,
        "launch_instances": ordered_launch_instances,
        "physical_bindings": ordered_physical,
        "execution_artifact_aliases": ordered_execution_aliases,
        "full_aliases": ordered_aliases,
        "rows": ordered_rows,
    }
    try:
        return GCSFiveConfigSidecarMapV2.model_validate(
            {**unsigned, "mapping_sha256": _hash_json(_jsonable(unsigned))},
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError("GCS v2 sidecar mapping is invalid") from exc


class GCSQueryBinding(_StrictFrozenModel):
    query_id: str
    canonical_capability: str
    split: str
    component_id: str
    group_values: tuple[tuple[str, str | None], ...]

    @field_validator("group_values", mode="before")
    @classmethod
    def coerce_group_values(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value


class GCSPopulation(_StrictFrozenModel):
    policy_version: Literal["portfolio-grounded-contract-success-v1"] = (
        GCS_POLICY_VERSION
    )
    grouping_policy_version: Literal["query-connected-components-v1"] = (
        GROUPING_POLICY_VERSION
    )
    query_count: int = Field(ge=1)
    component_count: int = Field(ge=1)
    bindings: tuple[GCSQueryBinding, ...]
    component_sizes: tuple[tuple[str, int], ...]
    population_mapping_sha256: Sha256

    @field_validator("bindings", "component_sizes", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(
                tuple(item) if isinstance(item, list) else item for item in value
            )
        return value

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        if len(self.bindings) != self.query_count:
            raise ValueError("GCS population query count mismatch")
        if len(self.component_sizes) != self.component_count:
            raise ValueError("GCS population component count mismatch")
        if tuple(item.query_id for item in self.bindings) != tuple(
            sorted(item.query_id for item in self.bindings)
        ):
            raise ValueError("GCS population bindings must be query-sorted")
        if sum(size for _component, size in self.component_sizes) != self.query_count:
            raise ValueError("GCS population component sizes do not cover queries")
        if self.component_sizes != tuple(sorted(self.component_sizes)) or len(
            {component for component, _size in self.component_sizes}
        ) != len(self.component_sizes):
            raise ValueError("GCS population components must be sorted and unique")
        observed_sizes = Counter(item.component_id for item in self.bindings)
        if self.component_sizes != tuple(sorted(observed_sizes.items())):
            raise ValueError("GCS population component membership is inconsistent")
        mapping_payload = {
            "grouping_policy_version": self.grouping_policy_version,
            "group_fields": list(GROUP_FIELDS),
            "bindings": [item.model_dump(mode="json") for item in self.bindings],
        }
        if self.population_mapping_sha256 != _hash_json(mapping_payload):
            raise ValueError("GCS population mapping hash is inconsistent")
        return self


class GCSPopulationV2(GCSPopulation):
    """The unchanged connected-component population bound to GCS v2."""

    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )


def build_gcs_population(queries: Sequence[Query]) -> GCSPopulation:
    """Build the authoritative connected-component mapping for one population."""

    ordered = tuple(sorted(queries, key=lambda item: item.query_id))
    if not ordered:
        raise PortfolioGCSError("GCS population is empty")
    if len({item.query_id for item in ordered}) != len(ordered):
        raise PortfolioGCSError("GCS population contains duplicate query_id")
    for query in ordered:
        if not _strict_model_revalidates(query, Query):
            raise PortfolioGCSError("GCS population contains an invalid Query")
        if query.canonical_capability not in GCS_CAPABILITY_ORDER:
            raise PortfolioGCSError("GCS population contains unknown capability")
        if query.requires_card is None:
            raise PortfolioGCSError("GCS population contains unresolved card policy")
        if query.canonical_capability not in query.acceptable_capabilities:
            raise PortfolioGCSError("GCS population acceptable set is inconsistent")

    groups = build_atomic_groups(list(ordered))
    component_by_query = {
        query.query_id: group.group_id for group in groups for query in group.queries
    }
    bindings = tuple(
        GCSQueryBinding(
            query_id=query.query_id,
            canonical_capability=query.canonical_capability,
            split=query.split,
            component_id=component_by_query[query.query_id],
            group_values=tuple(
                (field_name, getattr(query, field_name)) for field_name in GROUP_FIELDS
            ),
        )
        for query in ordered
    )
    component_sizes = tuple(
        sorted((group.group_id, len(group.queries)) for group in groups)
    )
    mapping_payload = {
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "group_fields": list(GROUP_FIELDS),
        "bindings": [item.model_dump(mode="json") for item in bindings],
    }
    return GCSPopulation(
        query_count=len(bindings),
        component_count=len(component_sizes),
        bindings=bindings,
        component_sizes=component_sizes,
        population_mapping_sha256=_hash_json(mapping_payload),
    )


def build_gcs_population_v2(queries: Sequence[Query]) -> GCSPopulationV2:
    """Build the same immutable grouping under the explicit v2 identity."""

    v1 = build_gcs_population(queries)
    return GCSPopulationV2.model_validate(
        {**v1.model_dump(mode="json"), "policy_version": GCS_V2_POLICY_VERSION},
        strict=True,
    )


class GCSToolObservation(_StrictFrozenModel):
    trace: AssistantToolTrace
    visible_evidence: VisibleToolEvidence | None = None
    scorer_call: PublicScorerCallEvidenceV1 | PublicScorerCallEvidenceV2 | None = None


class CapabilityOracleDecision(_StrictFrozenModel):
    answer_mode: AnswerMode
    oracle_available: bool
    semantic_claim_support_resolved: bool
    tool_contract_pass: Binary
    evidence_grounded: Binary
    output_contract_pass: Binary
    reason_codes: tuple[str, ...] = ()

    @field_validator("reason_codes", mode="before")
    @classmethod
    def coerce_reasons(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("oracle reason codes must be sorted and unique")
        if any(item not in GCS_REASON_CODES for item in self.reason_codes):
            raise ValueError("oracle reason code is outside the frozen vocabulary")
        if not self.semantic_claim_support_resolved and self.evidence_grounded:
            raise ValueError("unresolved semantic support cannot be grounded")
        return self


class CapabilityOracleDecisionV2(_StrictFrozenModel):
    answer_mode: AnswerMode
    oracle_available: Literal[True] = True
    semantic_claim_support_resolved: bool
    tool_contract_pass: Binary
    evidence_grounded: Binary
    output_contract_pass: Binary
    style_support_status: StyleOutcomeStatus | None = None
    reason_codes: tuple[str, ...] = ()

    @field_validator("reason_codes", mode="before")
    @classmethod
    def coerce_reasons(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("v2 oracle reason codes must be sorted and unique")
        if any(item not in GCS_V2_REASON_CODES for item in self.reason_codes):
            raise ValueError("v2 oracle reason code is outside the frozen vocabulary")
        if not self.semantic_claim_support_resolved and self.evidence_grounded:
            raise ValueError("unresolved semantic support cannot be grounded")
        return self


class GCSQueryScore(_StrictFrozenModel):
    policy_version: Literal["portfolio-grounded-contract-success-v1"] = (
        GCS_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_POLICY_SHA256] = GCS_POLICY_SHA256
    query_id: str
    config: AssistantRunConfig
    canonical_capability: str
    component_id: str
    route_disposition: Literal["pass", "fail", "not_applicable"]
    answer_mode: AnswerMode
    oracle_available: bool
    semantic_claim_support_resolved: bool
    route_acceptable: Binary
    no_hard_error: Binary
    tool_contract_pass: Binary
    evidence_grounded: Binary
    output_contract_pass: Binary
    hard_error: Binary
    gcs: Binary
    reason_codes: tuple[str, ...]

    @field_validator("reason_codes", mode="before")
    @classmethod
    def coerce_reasons(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_score(self) -> Self:
        expected = int(
            all(
                (
                    self.route_acceptable,
                    self.no_hard_error,
                    self.tool_contract_pass,
                    self.evidence_grounded,
                    self.output_contract_pass,
                )
            )
        )
        if self.gcs != expected:
            raise ValueError("GCS score differs from its five components")
        if self.hard_error != 1 - self.no_hard_error:
            raise ValueError("GCS hard-error flag differs from no_hard_error")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("GCS reason codes must be sorted and unique")
        allowed_reason_codes = (
            GCS_V2_REASON_CODES
            if self.policy_version == GCS_V2_POLICY_VERSION
            else GCS_REASON_CODES
        )
        if any(item not in allowed_reason_codes for item in self.reason_codes):
            raise ValueError("GCS reason code is outside the frozen vocabulary")
        return self


class GCSQueryScoreV2(GCSQueryScore):
    """One fixed-denominator row under the v2 six-oracle policy."""

    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    evaluated_capability: str
    style_support_status: StyleOutcomeStatus | None = None

    @model_validator(mode="after")
    def validate_style_status(self) -> Self:
        if self.evaluated_capability not in GCS_CAPABILITY_ORDER:
            raise ValueError("v2 evaluated capability is unknown")
        if (self.evaluated_capability == "product.style_recommendation") != (
            self.style_support_status is not None
        ):
            raise ValueError("Style support status is required exactly for Style rows")
        return self


@dataclass(frozen=True)
class _OracleContext:
    query: Query
    result: AssistantResult
    receipt: AssistantExecutionReceipt
    task: CapabilityTaskSpec
    observations: tuple[GCSToolObservation, ...]
    sidecar: PublicScorerEvidenceV1 | PublicScorerEvidenceV2 | None
    sections: Mapping[str, str]
    section_valid: bool
    base_tool_valid: bool


class CapabilityGCSOracle(Protocol):
    capability_id: str

    def evaluate(
        self, context: _OracleContext
    ) -> CapabilityOracleDecision | CapabilityOracleDecisionV2: ...


def _parse_sections(
    response_text: str, required_sections: Sequence[str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    required = frozenset(item.casefold() for item in required_sections)
    collected: dict[str, list[str]] = {}
    current: str | None = None
    invalid = False
    for raw_line in response_text.splitlines():
        colon = _SECTION_WITH_COLON_RE.fullmatch(raw_line)
        markdown = _MARKDOWN_SECTION_RE.fullmatch(raw_line)
        match = colon or markdown
        label = match.group("label").casefold() if match is not None else None
        if label in required:
            if label in collected:
                invalid = True
            else:
                collected[label] = []
            current = label
            if colon is not None and colon.group("body").strip():
                collected[label].append(colon.group("body").strip())
            continue
        if raw_line.strip():
            if current is None:
                invalid = True
            else:
                collected[current].append(raw_line.strip())

    sections = {label: "\n".join(lines).strip() for label, lines in collected.items()}
    if set(sections) != required or any(not sections[label] for label in required):
        invalid = True
    return sections, (("output_section_invalid",) if invalid else ())


def _card_bytes(card: VisibleCard) -> bytes:
    return canonical_json_bytes(card.model_dump(mode="json"))


def _stable_evidence_card_union(
    evidence: Sequence[VisibleToolEvidence],
) -> tuple[VisibleCard, ...]:
    seen: set[bytes] = set()
    cards: list[VisibleCard] = []
    for item in evidence:
        for card in item.cards:
            encoded = _card_bytes(card)
            if encoded not in seen:
                seen.add(encoded)
                cards.append(card)
    return tuple(cards)


def _sidecar_binding_valid(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV1 | PublicScorerEvidenceV2 | None,
) -> bool:
    sidecar_type: type[BaseModel]
    if type(sidecar) is PublicScorerEvidenceV1:
        sidecar_type = PublicScorerEvidenceV1
    elif type(sidecar) is PublicScorerEvidenceV2:
        sidecar_type = PublicScorerEvidenceV2
    else:
        return False
    if not _strict_model_revalidates(sidecar, sidecar_type):
        return False
    assert sidecar is not None
    successful = tuple(item for item in result.tool_trace if item.status == "success")
    if (
        sidecar.query_id != query.query_id
        or sidecar.config != result.config
        or sidecar.query_artifact_sha256 != result.query_artifact_sha256
        or sidecar.assistant_receipt_sha256 != receipt.receipt_sha256
        or len(sidecar.calls) != len(successful)
    ):
        return False
    if isinstance(sidecar, PublicScorerEvidenceV2) and (
        sidecar.request_sha256 != receipt.request_sha256
        or sidecar.assistant_result_sha256 != _hash_json(result)
    ):
        return False
    for trace, call in zip(successful, sidecar.calls, strict=True):
        if (
            call.call_index != trace.call_index
            or call.tool_name != trace.tool_name
            or call.arguments_sha256 != trace.arguments_sha256
            or call.result_sha256 != trace.result_sha256
        ):
            return False
    return True


def _build_observations(
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV1 | PublicScorerEvidenceV2 | None,
) -> tuple[tuple[GCSToolObservation, ...], bool, tuple[str, ...]]:
    reasons: set[str] = set()
    if tuple(item.call_index for item in result.tool_trace) != tuple(
        range(1, len(result.tool_trace) + 1)
    ):
        reasons.add("receipt_trace_mismatch")
    if tuple(result.tool_trace) != tuple(receipt.tool_trace):
        reasons.add("receipt_trace_mismatch")

    successful_evidence = iter(result.visible_tool_evidence)
    scorer_by_index = (
        {} if sidecar is None else {item.call_index: item for item in sidecar.calls}
    )
    observations: list[GCSToolObservation] = []
    evidence_count = 0
    for trace in result.tool_trace:
        evidence: VisibleToolEvidence | None = None
        if trace.status == "success":
            evidence = next(successful_evidence, None)
            evidence_count += int(evidence is not None)
            if (
                evidence is None
                or evidence.tool_name != trace.tool_name
                or evidence.status != "success"
            ):
                reasons.add("tool_evidence_mismatch")
        observations.append(
            GCSToolObservation(
                trace=trace,
                visible_evidence=evidence,
                scorer_call=scorer_by_index.get(trace.call_index),
            )
        )
    if next(successful_evidence, None) is not None or evidence_count != len(
        result.visible_tool_evidence
    ):
        reasons.add("tool_evidence_mismatch")
    if _stable_evidence_card_union(result.visible_tool_evidence) != tuple(
        result.visible_cards
    ):
        reasons.add("evidence_card_union_mismatch")
    if any(item.status != "success" for item in result.tool_trace):
        reasons.add("tool_contract_failed")
    base_valid = not reasons
    return tuple(observations), base_valid, tuple(sorted(reasons))


def _expected_argument_hash(query: Query, tool_name: str) -> str | None:
    if tool_name in {
        "image_product_search",
        "style_similar_search",
        "object_detect",
        "document_ocr",
        "multi_product_search",
    }:
        return _hash_json({"asset_id": query.asset_id})
    if tool_name == "text_product_search":
        return _hash_json({"query": query.text})
    return None


def _argument_projection_valid(query: Query, observation: GCSToolObservation) -> bool:
    call = observation.scorer_call
    if call is None:
        return False
    if (
        isinstance(call, PublicScorerCallEvidenceV2)
        and observation.trace.tool_name == "style_similar_search"
    ):
        return call.arguments_sha256 == _hash_json(
            {"asset_id": query.asset_id, "query": query.text}
        ) and call.argument_projection == {
            "asset_handle": "query_asset",
            "query": query.text,
        }
    expected = _expected_argument_hash(query, observation.trace.tool_name)
    if expected is not None:
        if observation.trace.tool_name == "text_product_search":
            return call.arguments_sha256 == expected and call.argument_projection == {
                "query": query.text
            }
        return (
            call.arguments_sha256 == expected
            and call.argument_projection == _ASSET_ARGUMENT_PROJECTION
        )
    expected_key = {
        "encyclopedia_lookup": "entity",
        "recipe_lookup": "dish",
    }.get(observation.trace.tool_name)
    if expected_key is None or set(call.argument_projection) != {expected_key}:
        return False
    value = call.argument_projection[expected_key]
    return (
        isinstance(value, str)
        and bool(value.strip())
        and call.arguments_sha256 == _hash_json({expected_key: value})
    )


def _common_tool_contract(
    context: _OracleContext,
    allowed_sequences: frozenset[tuple[str, ...]],
) -> tuple[bool, set[str]]:
    reasons: set[str] = set()
    names = tuple(item.trace.tool_name for item in context.observations)
    if not context.base_tool_valid or names not in allowed_sequences:
        reasons.add("tool_contract_failed")
    if any(name not in context.task.allowed_tools for name in names):
        reasons.add("tool_contract_failed")
    if context.sidecar is None:
        reasons.update({"scorer_sidecar_missing", "semantic_claim_support_unresolved"})
    elif not _sidecar_binding_valid(
        context.query, context.result, context.receipt, context.sidecar
    ):
        reasons.update({"scorer_sidecar_invalid", "semantic_claim_support_unresolved"})
    if (
        context.sidecar is not None
        and context.task.capability_id != "product.multi_search"
        and context.sidecar.expected_multi_items is not None
    ):
        reasons.add("scorer_sidecar_invalid")
    for observation in context.observations:
        if observation.trace.status != "success":
            reasons.add("tool_contract_failed")
        if not _argument_projection_valid(context.query, observation):
            reasons.add("tool_argument_invalid")
    return not reasons, reasons


def _card_contract(
    context: _OracleContext, answer_mode: AnswerMode
) -> tuple[bool, set[str]]:
    reasons: set[str] = set()
    contract = context.task.output_contract
    cards = context.result.visible_cards
    if not context.section_valid:
        reasons.add("output_section_invalid")
    if answer_mode == "unresolved":
        reasons.add("card_contract_failed")
    elif contract.card_requirement == "forbidden":
        if cards:
            reasons.add("card_contract_failed")
    elif answer_mode == "supported" and not cards:
        reasons.add("card_contract_failed")
    elif answer_mode == "fallback" and cards:
        reasons.add("card_contract_failed")

    seen_references: set[str] = set()
    seen_products: set[str] = set()
    for card in cards:
        fields = dict(card.fields)
        if "title" in contract.card_fields and not card.title.strip():
            reasons.add("card_contract_failed")
        for required in set(contract.card_fields) - {"title"}:
            if not fields.get(required):
                reasons.add("card_contract_failed")
        reference = fields.get("evidence_reference")
        product_id = fields.get("product_id")
        if reference is not None and (
            reference in seen_references
            or _PRODUCT_EVIDENCE_RE.fullmatch(reference) is None
        ):
            reasons.add("card_contract_failed")
        if product_id is not None and (
            product_id in seen_products or _PRODUCT_ID_RE.fullmatch(product_id) is None
        ):
            reasons.add("card_contract_failed")
        if reference is not None:
            seen_references.add(reference)
        if product_id is not None:
            seen_products.add(product_id)
    return not reasons, reasons


def _fallback_pass(context: _OracleContext) -> bool:
    lowered = context.result.response_text.casefold()
    markers = _FALLBACK_MARKERS[context.query.canonical_capability]
    return (
        not context.result.visible_cards
        and not _ANY_HANDLE_RE.findall(context.result.response_text)
        and any(marker.casefold() in lowered for marker in markers)
        and not _HIGH_RISK_ATOM_RE.findall(context.sections.get("answer", ""))
    )


def _material_statements(text: str) -> tuple[str, ...]:
    return tuple(
        item.strip().lstrip("-*• ").strip()
        for item in _MATERIAL_SPLIT_RE.split(text)
        if re.search(r"[A-Za-z0-9\u3400-\u9fff]", item)
    )


def _normalized_atom(value: str) -> str:
    return re.sub(r"[\s,，]", "", value.casefold())


def _claims_are_grounded(
    sections: Mapping[str, str],
    evidence_by_handle: Mapping[str, str],
    *,
    section_names: Sequence[str] = ("answer", "evidence"),
) -> tuple[bool, set[str]]:
    reasons: set[str] = set()
    for section_name in section_names:
        for statement in _material_statements(sections.get(section_name, "")):
            handles = tuple(dict.fromkeys(_ANY_HANDLE_RE.findall(statement)))
            if not handles:
                reasons.add("material_claim_uncited")
                continue
            if any(handle not in evidence_by_handle for handle in handles):
                reasons.add("evidence_handle_unknown")
                continue
            cited = _normalized_atom(
                "\n".join(evidence_by_handle[handle] for handle in handles)
            )
            for atom in _HIGH_RISK_ATOM_RE.findall(_ANY_HANDLE_RE.sub("", statement)):
                if _normalized_atom(atom) not in cited:
                    reasons.add("unsupported_claim")
    return not reasons, reasons


def _unknown_response_handles(response_text: str, valid_handles: set[str]) -> set[str]:
    return {
        handle
        for handle in _ANY_HANDLE_RE.findall(response_text)
        if handle not in valid_handles
    }


def _visible_evidence_text(observation: GCSToolObservation) -> str:
    evidence = observation.visible_evidence
    if evidence is None:
        return ""
    values: list[str] = []
    if evidence.visible_text is not None:
        values.append(evidence.visible_text)
    for card in evidence.cards:
        values.extend((card.title, card.body))
        values.extend(item for pair in card.fields for item in pair)
    for citation in evidence.citations:
        values.extend((citation.title, citation.uri, citation.excerpt))
    for detection in evidence.detections:
        values.append(detection.label)
    return "\n".join(values)


def _public_values_are_visible(
    observation: GCSToolObservation, values: Sequence[str]
) -> bool:
    visible = _visible_evidence_text(observation)
    return bool(visible) and all(value in visible for value in values)


def _normalized_document_value(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)


def _document_extractions_are_grounded(
    sections: Mapping[str, str], line_text: Mapping[str, str]
) -> tuple[bool, set[str]]:
    """Validate the frozen ``field: value line-handle`` extraction grammar."""

    reasons: set[str] = set()
    statement_count = 0
    for section_name in ("answer", "evidence"):
        for statement in _material_statements(sections.get(section_name, "")):
            statement_count += 1
            exact_handles = tuple(
                handle
                for handle in dict.fromkeys(_ANY_HANDLE_RE.findall(statement))
                if _LINE_RE.fullmatch(handle) is not None
            )
            if len(exact_handles) != 1:
                reasons.add("material_claim_uncited")
                continue
            handle = exact_handles[0]
            if handle not in line_text:
                reasons.add("evidence_handle_unknown")
                continue
            without_handle = _ANY_HANDLE_RE.sub("", statement)
            without_handle = re.sub(r"[\[\](){}【】]", " ", without_handle)
            parts = re.split(r"[:：]", without_handle, maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                reasons.add("unsupported_claim")
                continue
            value = parts[1].strip().strip(" .。;；,，")
            normalized_value = _normalized_document_value(value)
            normalized_line = _normalized_document_value(line_text[handle])
            if not normalized_value or normalized_value not in normalized_line:
                reasons.add("unsupported_claim")
    if statement_count == 0:
        reasons.add("material_claim_uncited")
    return not reasons, reasons


def _decision(
    *,
    answer_mode: AnswerMode,
    oracle_available: bool,
    semantic_resolved: bool,
    tool_pass: bool,
    evidence_pass: bool,
    output_pass: bool,
    reasons: set[str],
) -> CapabilityOracleDecision:
    return CapabilityOracleDecision(
        answer_mode=answer_mode,
        oracle_available=oracle_available,
        semantic_claim_support_resolved=semantic_resolved,
        tool_contract_pass=int(tool_pass),
        evidence_grounded=int(evidence_pass and semantic_resolved),
        output_contract_pass=int(output_pass),
        reason_codes=tuple(sorted(reasons)),
    )


class ExactMatchGCSOracle:
    capability_id = "product.exact_match"

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecision:
        tool_pass, reasons = _common_tool_contract(context, _EXACT_ALLOWED_SEQUENCES)
        candidates: list[ScorerProductCandidateV1] = []
        if tool_pass:
            for observation in context.observations:
                call = observation.scorer_call
                try:
                    if call is None or call.payload_kind != "product_candidates_v1":
                        raise ValueError
                    payload = ScorerProductPayloadV1.model_validate(
                        call.payload, strict=True
                    )
                    if any(
                        int(
                            _PRODUCT_EVIDENCE_RE.fullmatch(
                                item.evidence_reference
                            ).group(1)
                        )
                        != call.call_index
                        for item in payload.candidates
                    ):
                        raise ValueError
                    candidates.extend(payload.candidates)
                except (AttributeError, ValueError):
                    tool_pass = False
                    reasons.add("scorer_sidecar_invalid")

        eligible = tuple(item for item in candidates if item.eligible)
        answer_mode: AnswerMode = (
            "supported" if eligible else "fallback" if tool_pass else "unresolved"
        )
        evidence_pass = tool_pass
        valid_handles = {
            handle
            for item in eligible
            for handle in (item.evidence_reference, item.product_id)
        }
        expected_cards = {
            (item.evidence_reference, item.product_id, item.title) for item in eligible
        }
        actual_cards = {
            (
                dict(card.fields).get("evidence_reference"),
                dict(card.fields).get("product_id"),
                card.title,
            )
            for card in context.result.visible_cards
        }
        if expected_cards != actual_cards:
            evidence_pass = False
            reasons.add("card_contract_failed")
        if _unknown_response_handles(context.result.response_text, valid_handles):
            evidence_pass = False
            reasons.add("evidence_handle_unknown")
        if answer_mode == "supported":
            card_section = context.sections.get("product_cards", "")
            for item in eligible:
                if not all(
                    value in card_section
                    for value in (
                        item.evidence_reference,
                        item.product_id,
                        item.title,
                    )
                ):
                    evidence_pass = False
                    reasons.add("card_contract_failed")
            claim_text = "\n".join(
                context.sections.get(name, "") for name in ("answer", "product_cards")
            ).casefold()
            public_attributes = " ".join(
                f"{key} {value}"
                for item in eligible
                for key, value in item.public_attributes
            ).casefold()
            if any(
                marker in claim_text and marker not in public_attributes
                for marker in _UNSUPPORTED_PRODUCT_CLAIMS
            ):
                evidence_pass = False
                reasons.add("unsupported_claim")
        elif answer_mode == "fallback" and not _fallback_pass(context):
            evidence_pass = False
            reasons.add("fallback_contract_failed")

        output_pass, output_reasons = _card_contract(context, answer_mode)
        reasons.update(output_reasons)
        return _decision(
            answer_mode=answer_mode,
            oracle_available=context.sidecar is not None,
            semantic_resolved=tool_pass,
            tool_pass=tool_pass,
            evidence_pass=evidence_pass,
            output_pass=output_pass,
            reasons=reasons,
        )


class MultiSearchGCSOracle:
    capability_id = "product.multi_search"

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecision:
        tool_pass, reasons = _common_tool_contract(context, _MULTI_ALLOWED_SEQUENCES)
        payload: ScorerMultiPayloadV1 | None = None
        call = context.observations[0].scorer_call if context.observations else None
        if tool_pass:
            try:
                if call is None or call.payload_kind != "multi_mapping_v1":
                    raise ValueError
                payload = ScorerMultiPayloadV1.model_validate(call.payload, strict=True)
                if any(
                    int(
                        _PRODUCT_EVIDENCE_RE.fullmatch(item.evidence_reference).group(1)
                    )
                    != call.call_index
                    for item in payload.candidates
                ):
                    raise ValueError
            except ValueError:
                tool_pass = False
                reasons.add("scorer_sidecar_invalid")
        if context.sidecar is None or context.sidecar.expected_multi_items is None:
            tool_pass = False
            reasons.update(
                {
                    "requested_item_coverage_unresolved",
                    "semantic_claim_support_unresolved",
                }
            )
        elif payload is not None:
            expected = tuple(
                (item.item_ref, item.label)
                for item in context.sidecar.expected_multi_items
            )
            observed = tuple((item.item_ref, item.label) for item in payload.items)
            if expected != observed:
                tool_pass = False
                reasons.add("multi_mapping_invalid")

        matched = (
            ()
            if payload is None
            else tuple(item for item in payload.items if item.status == "matched")
        )
        answer_mode: AnswerMode = (
            "supported"
            if tool_pass and matched
            else "fallback"
            if tool_pass
            else "unresolved"
        )
        evidence_pass = tool_pass
        valid_handles: set[str] = set()
        if payload is not None:
            candidates_by_ordinal = {
                item.candidate_ordinal: item for item in payload.candidates
            }
            item_refs_by_ordinal: dict[int, list[str]] = defaultdict(list)
            for item in matched:
                assert item.candidate_ordinal is not None
                item_refs_by_ordinal[item.candidate_ordinal].append(item.item_ref)
            cards_by_ordinal: dict[int, VisibleCard] = {}
            for card in context.result.visible_cards:
                fields = dict(card.fields)
                evidence_match = _PRODUCT_EVIDENCE_RE.fullmatch(
                    fields.get("evidence_reference", "")
                )
                product_match = _PRODUCT_ID_RE.fullmatch(fields.get("product_id", ""))
                if (
                    evidence_match is None
                    or product_match is None
                    or evidence_match.groups() != product_match.groups()
                ):
                    evidence_pass = False
                    reasons.add("multi_mapping_invalid")
                    continue
                ordinal = int(evidence_match.group(2))
                candidate = candidates_by_ordinal.get(ordinal)
                if ordinal in cards_by_ordinal or candidate is None:
                    evidence_pass = False
                    reasons.add("multi_mapping_invalid")
                    continue
                cards_by_ordinal[ordinal] = card
                refs = tuple(
                    item.strip()
                    for item in fields.get("item_refs", "").split(",")
                    if item.strip()
                )
                if (
                    refs != tuple(item_refs_by_ordinal.get(ordinal, ()))
                    or fields.get("quantity") != str(len(refs))
                    or card.title != candidate.title
                    or fields.get("evidence_reference") != candidate.evidence_reference
                    or fields.get("product_id") != candidate.product_id
                ):
                    evidence_pass = False
                    reasons.add("multi_mapping_invalid")
            expected_ordinals = set(item_refs_by_ordinal)
            if set(cards_by_ordinal) != expected_ordinals:
                evidence_pass = False
                reasons.add("multi_mapping_invalid")
            for ordinal, candidate in candidates_by_ordinal.items():
                if ordinal in expected_ordinals:
                    valid_handles.update(
                        (candidate.evidence_reference, candidate.product_id)
                    )
            mapping = context.sections.get("item_mapping", "")
            cards_section = context.sections.get("product_cards", "")
            for item in payload.items:
                item_lines = tuple(
                    line for line in mapping.splitlines() if item.item_ref in line
                )
                if len(item_lines) != 1 or item.status not in item_lines[0]:
                    evidence_pass = False
                    reasons.add("multi_mapping_invalid")
                    continue
                if item.candidate_ordinal is None:
                    if "candidate-" in item_lines[0] or any(
                        _PRODUCT_ID_RE.fullmatch(handle) is not None
                        for handle in _ANY_HANDLE_RE.findall(item_lines[0])
                    ):
                        evidence_pass = False
                        reasons.add("multi_mapping_invalid")
                else:
                    candidate = candidates_by_ordinal[item.candidate_ordinal]
                    if not (
                        f"candidate-{item.candidate_ordinal}" in item_lines[0]
                        or candidate.product_id in item_lines[0]
                    ):
                        evidence_pass = False
                        reasons.add("multi_mapping_invalid")
            for ordinal in expected_ordinals:
                candidate = candidates_by_ordinal[ordinal]
                if not all(
                    value in cards_section
                    for value in (
                        candidate.evidence_reference,
                        candidate.product_id,
                        candidate.title,
                    )
                ):
                    evidence_pass = False
                    reasons.add("card_contract_failed")
        if _unknown_response_handles(context.result.response_text, valid_handles):
            evidence_pass = False
            reasons.add("evidence_handle_unknown")
        if answer_mode == "fallback" and not _fallback_pass(context):
            evidence_pass = False
            reasons.add("fallback_contract_failed")

        output_pass, output_reasons = _card_contract(context, answer_mode)
        reasons.update(output_reasons)
        return _decision(
            answer_mode=answer_mode,
            oracle_available=context.sidecar is not None,
            semantic_resolved=tool_pass,
            tool_pass=tool_pass,
            evidence_pass=evidence_pass,
            output_pass=output_pass,
            reasons=reasons,
        )


class _KnowledgeGCSOracle:
    capability_id: str
    lookup_tool: Literal["encyclopedia_lookup", "recipe_lookup"]

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecision:
        sequences = (
            _ENCYCLOPEDIA_ALLOWED_SEQUENCES
            if self.lookup_tool == "encyclopedia_lookup"
            else _RECIPE_ALLOWED_SEQUENCES
        )
        tool_pass, reasons = _common_tool_contract(context, sequences)
        sources: tuple[ScorerKnowledgeSourceV1, ...] = ()
        if tool_pass:
            for observation in context.observations:
                call = observation.scorer_call
                try:
                    if call is None:
                        raise ValueError
                    if observation.trace.tool_name == "object_detect":
                        if call.payload_kind != "detections_v1":
                            raise ValueError
                        detection_payload = ScorerDetectionPayloadV1.model_validate(
                            call.payload, strict=True
                        )
                        visible_detections = (
                            ()
                            if observation.visible_evidence is None
                            else observation.visible_evidence.detections
                        )
                        if tuple(
                            (item.label, item.confidence)
                            for item in detection_payload.detections
                        ) != tuple(
                            (item.label, item.confidence) for item in visible_detections
                        ):
                            raise ValueError
                    else:
                        if call.payload_kind != "knowledge_sources_v1":
                            raise ValueError
                        payload = ScorerKnowledgePayloadV1.model_validate(
                            call.payload, strict=True
                        )
                        if any(
                            int(_SOURCE_RE.fullmatch(item.evidence_reference).group(1))
                            != call.call_index
                            for item in payload.sources
                        ):
                            raise ValueError
                        source_ordinals = tuple(
                            int(_SOURCE_RE.fullmatch(item.evidence_reference).group(2))
                            for item in payload.sources
                        )
                        if source_ordinals != tuple(
                            range(1, len(source_ordinals) + 1)
                        ) or any(
                            not _public_values_are_visible(
                                observation,
                                (
                                    item.evidence_reference,
                                    item.title,
                                    item.text,
                                ),
                            )
                            for item in payload.sources
                        ):
                            raise ValueError
                        sources = payload.sources
                except (AttributeError, ValueError):
                    tool_pass = False
                    reasons.add("scorer_sidecar_invalid")
        answer_mode: AnswerMode = (
            "supported"
            if tool_pass and sources
            else "fallback"
            if tool_pass
            else "unresolved"
        )
        evidence_pass = tool_pass
        source_text = {
            item.evidence_reference: f"{item.title}\n{item.text}" for item in sources
        }
        if _unknown_response_handles(context.result.response_text, set(source_text)):
            evidence_pass = False
            reasons.add("evidence_handle_unknown")
        if answer_mode == "supported":
            claims_pass, claim_reasons = _claims_are_grounded(
                context.sections, source_text
            )
            evidence_pass &= claims_pass
            reasons.update(claim_reasons)
            if self.lookup_tool == "recipe_lookup":
                lowered = "\n".join(
                    context.sections.get(name, "") for name in ("answer", "evidence")
                ).casefold()
                if any(marker in lowered for marker in _UNSUPPORTED_RECIPE_GUARANTEES):
                    evidence_pass = False
                    reasons.add("unsupported_claim")
        elif answer_mode == "fallback" and not _fallback_pass(context):
            evidence_pass = False
            reasons.add("fallback_contract_failed")
        output_pass, output_reasons = _card_contract(context, answer_mode)
        reasons.update(output_reasons)
        return _decision(
            answer_mode=answer_mode,
            oracle_available=context.sidecar is not None,
            semantic_resolved=tool_pass,
            tool_pass=tool_pass,
            evidence_pass=evidence_pass,
            output_pass=output_pass,
            reasons=reasons,
        )


class EncyclopediaGCSOracle(_KnowledgeGCSOracle):
    capability_id = "knowledge.visual_encyclopedia"
    lookup_tool = "encyclopedia_lookup"


class RecipeGCSOracle(_KnowledgeGCSOracle):
    capability_id = "utility.recipe_guidance"
    lookup_tool = "recipe_lookup"


class DocumentGCSOracle:
    capability_id = "utility.document_reading"

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecision:
        tool_pass, reasons = _common_tool_contract(context, _DOCUMENT_ALLOWED_SEQUENCES)
        lines: tuple[ScorerOCRLineV1, ...] = ()
        call = context.observations[0].scorer_call if context.observations else None
        if tool_pass:
            try:
                if call is None or call.payload_kind != "ocr_lines_v1":
                    raise ValueError
                payload = ScorerOCRPayloadV1.model_validate(call.payload, strict=True)
                if any(
                    int(_LINE_RE.fullmatch(item.line_reference).group(1))
                    != call.call_index
                    for item in payload.lines
                ):
                    raise ValueError
                line_ordinals = tuple(
                    int(_LINE_RE.fullmatch(item.line_reference).group(2))
                    for item in payload.lines
                )
                observation = context.observations[0]
                if line_ordinals != tuple(range(1, len(line_ordinals) + 1)) or any(
                    not _public_values_are_visible(
                        observation, (item.line_reference, item.text)
                    )
                    for item in payload.lines
                ):
                    raise ValueError
                lines = payload.lines
            except (AttributeError, ValueError):
                tool_pass = False
                reasons.add("scorer_sidecar_invalid")
        answer_mode: AnswerMode = (
            "supported"
            if tool_pass and lines
            else "fallback"
            if tool_pass
            else "unresolved"
        )
        evidence_pass = tool_pass
        line_text = {item.line_reference: item.text for item in lines}
        if _unknown_response_handles(context.result.response_text, set(line_text)):
            evidence_pass = False
            reasons.add("evidence_handle_unknown")
        if answer_mode == "supported":
            claims_pass, claim_reasons = _document_extractions_are_grounded(
                context.sections, line_text
            )
            evidence_pass &= claims_pass
            reasons.update(claim_reasons)
            lowered = context.result.response_text.casefold()
            if not any(
                marker.casefold() in lowered for marker in _UNTRUSTED_DOCUMENT_MARKERS
            ):
                evidence_pass = False
                reasons.add("unsupported_claim")
        elif answer_mode == "fallback" and not _fallback_pass(context):
            evidence_pass = False
            reasons.add("fallback_contract_failed")
        output_pass, output_reasons = _card_contract(context, answer_mode)
        reasons.update(output_reasons)
        return _decision(
            answer_mode=answer_mode,
            oracle_available=context.sidecar is not None,
            semantic_resolved=tool_pass,
            tool_pass=tool_pass,
            evidence_pass=evidence_pass,
            output_pass=output_pass,
            reasons=reasons,
        )


_UNSUPPORTED_STYLE_CLAIMS = (
    *_UNSUPPORTED_PRODUCT_CLAIMS,
    "material",
    "fabric",
    "suitable",
    "suitability",
    "perfect for",
    "guaranteed",
    "材质",
    "面料",
    "适合",
    "保证",
)


def _style_response_handles(response_text: str) -> set[str]:
    return {match.group(0) for match in _STYLE_EVIDENCE_RE.finditer(response_text)}


class StyleGCSOracleV2:
    """Deterministic same/cross-category Style oracle for GCS v2."""

    capability_id = "product.style_recommendation"

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecisionV2:
        tool_pass, reasons = _common_tool_contract(context, _STYLE_ALLOWED_SEQUENCES)
        payload: ScorerStylePayloadV2 | None = None
        call = context.observations[0].scorer_call if context.observations else None
        if tool_pass:
            try:
                if (
                    not isinstance(call, PublicScorerCallEvidenceV2)
                    or call.payload_kind != "style_candidates_v2"
                ):
                    raise ValueError
                payload = ScorerStylePayloadV2.model_validate(call.payload, strict=True)
                candidate_calls = {
                    int(
                        _PRODUCT_EVIDENCE_RE.fullmatch(item.evidence_reference).group(1)
                    )
                    for item in payload.candidates
                }
                facet_calls = {
                    int(_STYLE_EVIDENCE_RE.fullmatch(facet.evidence_reference).group(1))
                    for item in payload.candidates
                    for facet in item.style_evidence
                }
                if candidate_calls.union(facet_calls).difference({call.call_index}):
                    raise ValueError
            except (AttributeError, ValueError):
                tool_pass = False
                reasons.add("style_evidence_invalid")

        status: StyleOutcomeStatus = (
            "unresolved" if payload is None else payload.support_status
        )
        classification = classify_style_query_v2(context.query.text)
        if payload is not None and (
            payload.style_submode != classification.style_submode
            or payload.requested_target_families
            != classification.requested_target_families
            or payload.allowed_target_families != classification.allowed_target_families
        ):
            tool_pass = False
            reasons.add("style_mode_mismatch")

        candidates = () if payload is None else payload.candidates
        if payload is not None and candidates:
            candidate_categories = {item.category for item in candidates}
            anchor = payload.anchor_category
            if anchor is None:
                tool_pass = False
                reasons.add("style_evidence_invalid")
            elif payload.style_submode == "same_category_alternative":
                if any(item.category != anchor for item in candidates):
                    tool_pass = False
                    reasons.add("style_mode_mismatch")
            else:
                if any(
                    item.category == anchor
                    or item.category not in payload.allowed_target_families
                    for item in candidates
                ):
                    tool_pass = False
                    reasons.add("style_target_family_mismatch")
                if not set(payload.requested_target_families).issubset(
                    candidate_categories
                ):
                    tool_pass = False
                    reasons.add("style_target_family_mismatch")

        answer_mode: AnswerMode = (
            "supported"
            if tool_pass and status == "candidates"
            else "fallback"
            if tool_pass and status in {"no_result", "unsupported"}
            else "unresolved"
        )
        evidence_pass = tool_pass
        valid_product_handles = {
            handle
            for item in candidates
            for handle in (item.evidence_reference, item.product_id)
        }
        valid_style_handles = {
            facet.evidence_reference
            for item in candidates
            for facet in item.style_evidence
        }
        expected_cards = {
            (item.evidence_reference, item.product_id, item.title)
            for item in candidates
        }
        actual_cards = {
            (
                dict(card.fields).get("evidence_reference"),
                dict(card.fields).get("product_id"),
                card.title,
            )
            for card in context.result.visible_cards
        }
        if expected_cards != actual_cards:
            evidence_pass = False
            reasons.add("card_contract_failed")
        if _unknown_response_handles(
            context.result.response_text, valid_product_handles
        ) or _style_response_handles(context.result.response_text).difference(
            valid_style_handles
        ):
            evidence_pass = False
            reasons.add("evidence_handle_unknown")

        if answer_mode == "supported":
            cards_by_product = {
                dict(card.fields).get("product_id"): card
                for card in context.result.visible_cards
            }
            product_cards = context.sections.get("product_cards", "")
            rationale = context.sections.get("diversity_rationale", "")
            for candidate in candidates:
                card = cards_by_product.get(candidate.product_id)
                if card is None:
                    evidence_pass = False
                    reasons.add("card_contract_failed")
                    continue
                fields = dict(card.fields)
                expected_facet_fields = {
                    f"style_evidence_{ordinal}": facet.evidence_reference
                    for ordinal, facet in enumerate(candidate.style_evidence, 1)
                }
                observed_facet_fields = {
                    key: value
                    for key, value in fields.items()
                    if key.startswith("style_evidence_")
                }
                if (
                    fields.get("style_submode") != candidate.style_submode
                    or observed_facet_fields != expected_facet_fields
                    or (
                        candidate.style_submode == "same_category_alternative"
                        and fields.get("similarity_source")
                        != candidate.similarity_source
                    )
                    or (
                        candidate.style_submode == "cross_category_coordination"
                        and "similarity_source" in fields
                    )
                    or candidate.category not in card.body
                ):
                    evidence_pass = False
                    reasons.add("style_evidence_invalid")
                if not all(
                    facet.evidence_reference in card.body
                    or (
                        facet.facet in card.body
                        and facet.value in card.body
                        and fields.get(f"style_evidence_{ordinal}")
                        == facet.evidence_reference
                    )
                    for ordinal, facet in enumerate(candidate.style_evidence, 1)
                ):
                    evidence_pass = False
                    reasons.add("style_evidence_invalid")
                if not all(
                    value in product_cards
                    for value in (
                        candidate.title,
                        candidate.evidence_reference,
                        candidate.product_id,
                    )
                ):
                    evidence_pass = False
                    reasons.add("card_contract_failed")
                for facet in candidate.style_evidence:
                    if not all(
                        value in rationale
                        for value in (
                            facet.evidence_reference,
                            facet.facet,
                            facet.value,
                        )
                    ):
                        evidence_pass = False
                        reasons.add("style_evidence_invalid")

            claim_text = "\n".join(
                context.sections.get(name, "")
                for name in ("answer", "diversity_rationale", "product_cards")
            ).casefold()
            public_style_text = " ".join(
                f"{facet.facet} {facet.value}"
                for item in candidates
                for facet in item.style_evidence
            ).casefold()
            if any(
                marker.casefold() in claim_text
                and marker.casefold() not in public_style_text
                for marker in _UNSUPPORTED_STYLE_CLAIMS
            ):
                evidence_pass = False
                reasons.add("unsupported_claim")
        elif answer_mode == "fallback":
            if candidates or not _fallback_pass(context):
                evidence_pass = False
                reasons.add("fallback_contract_failed")

        output_pass, output_reasons = _card_contract(context, answer_mode)
        reasons.update(output_reasons)
        return CapabilityOracleDecisionV2(
            answer_mode=answer_mode,
            semantic_claim_support_resolved=tool_pass,
            tool_contract_pass=int(tool_pass),
            evidence_grounded=int(evidence_pass and tool_pass),
            output_contract_pass=int(output_pass),
            style_support_status=status,
            reason_codes=tuple(sorted(reasons)),
        )


class FailClosedStyleOracle:
    capability_id = "product.style_recommendation"

    def evaluate(self, context: _OracleContext) -> CapabilityOracleDecision:
        structural_mode: AnswerMode = (
            "supported" if context.result.visible_cards else "fallback"
        )
        output_pass, output_reasons = _card_contract(context, structural_mode)
        return CapabilityOracleDecision(
            answer_mode="unresolved",
            oracle_available=False,
            semantic_claim_support_resolved=False,
            tool_contract_pass=0,
            evidence_grounded=0,
            output_contract_pass=int(output_pass),
            reason_codes=tuple(
                sorted(
                    {
                        "semantic_claim_support_unresolved",
                        "style_oracle_unavailable",
                        *output_reasons,
                    }
                )
            ),
        )


def portfolio_gcs_oracles_v1() -> dict[str, CapabilityGCSOracle]:
    """Return the explicit v1 registry with a fail-closed Style placeholder."""

    oracles: tuple[CapabilityGCSOracle, ...] = (
        EncyclopediaGCSOracle(),
        ExactMatchGCSOracle(),
        MultiSearchGCSOracle(),
        FailClosedStyleOracle(),
        DocumentGCSOracle(),
        RecipeGCSOracle(),
    )
    return {item.capability_id: item for item in oracles}


def portfolio_gcs_oracles_v2() -> dict[str, CapabilityGCSOracle]:
    """Return the complete six-capability v2 deterministic oracle registry."""

    oracles: tuple[CapabilityGCSOracle, ...] = (
        EncyclopediaGCSOracle(),
        ExactMatchGCSOracle(),
        MultiSearchGCSOracle(),
        StyleGCSOracleV2(),
        DocumentGCSOracle(),
        RecipeGCSOracle(),
    )
    return {item.capability_id: item for item in oracles}


def _population_binding(population: GCSPopulation, query: Query) -> GCSQueryBinding:
    binding = next(
        (item for item in population.bindings if item.query_id == query.query_id), None
    )
    if binding is None:
        raise PortfolioGCSError("query is absent from the GCS population")
    expected_groups = tuple(
        (field_name, getattr(query, field_name)) for field_name in GROUP_FIELDS
    )
    if (
        binding.canonical_capability != query.canonical_capability
        or binding.split != query.split
        or binding.group_values != expected_groups
    ):
        raise PortfolioGCSError("query metadata drifted from the GCS population")
    return binding


def _strict_model_revalidates(value: BaseModel, model_type: type[BaseModel]) -> bool:
    if type(value) is not model_type:
        return False
    try:
        model_type.model_validate_json(
            canonical_json_bytes(value.model_dump(mode="json")), strict=True
        )
    except (TypeError, ValueError):
        return False
    return True


def score_portfolio_gcs(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV1 | None,
    task_spec: TaskSpecification,
    oracles: Mapping[str, CapabilityGCSOracle],
    *,
    population: GCSPopulation,
) -> GCSQueryScore:
    """Score one terminal query/config row without dropping failed components."""

    if not _strict_model_revalidates(population, GCSPopulation):
        raise PortfolioGCSError("GCS population contract is invalid")
    binding = _population_binding(population, query)
    if set(oracles) != set(GCS_CAPABILITY_ORDER):
        raise PortfolioGCSError(
            "GCS oracle registry must contain exactly six capabilities"
        )
    if any(key != oracle.capability_id for key, oracle in oracles.items()):
        raise PortfolioGCSError("GCS oracle registry key differs from oracle identity")

    reasons: set[str] = set()
    input_valid = True
    canonical_task = task_spec.capabilities_by_id.get(query.canonical_capability)
    if (
        task_spec.task_spec_version != "ecommerce-task-spec-v1"
        or canonical_task is None
        or result.query_id != query.query_id
        or not _strict_model_revalidates(result, AssistantResult)
        or not _strict_model_revalidates(receipt, AssistantExecutionReceipt)
        or (receipt.outcome == "success" and receipt.query_asset_id != query.asset_id)
    ):
        input_valid = False
        reasons.add("input_binding_invalid")
    if canonical_task is None:
        raise PortfolioGCSError("Task Specification lacks the query capability")

    effective_capability = query.canonical_capability
    if (
        result.config != "noskill"
        and result.selected_capability in query.acceptable_capabilities
        and result.selected_capability in task_spec.capabilities_by_id
    ):
        assert result.selected_capability is not None
        effective_capability = result.selected_capability
    task = task_spec.capabilities_by_id[effective_capability]

    observations, base_tool_valid, observation_reasons = _build_observations(
        result, receipt, sidecar
    )
    reasons.update(observation_reasons)
    sections, section_reasons = _parse_sections(
        result.response_text, task.output_contract.required_sections
    )
    reasons.update(section_reasons)

    route_disposition: Literal["pass", "fail", "not_applicable"] = "fail"
    route_acceptable = 0
    attempt = receipt.route_attempt
    if result.config == "noskill":
        if (
            attempt is not None
            and attempt.status == "not_applicable"
            and result.selected_capability is None
            and result.skill_slug is None
            and result.route_trace_sha256 is None
            and result.bank_sha256 is None
        ):
            route_disposition = "not_applicable"
            route_acceptable = 1
    elif (
        attempt is not None
        and attempt.status == "selected"
        and attempt.selected_capability == result.selected_capability
        and attempt.skill_slug == result.skill_slug
        and attempt.route_trace_sha256 == result.route_trace_sha256
        and result.selected_capability in query.acceptable_capabilities
    ):
        route_disposition = "pass"
        route_acceptable = 1
    if not route_acceptable:
        reasons.add("route_unacceptable")

    no_hard_error = int(
        input_valid
        and result.error_code is None
        and receipt.outcome == "success"
        and base_tool_valid
        and all(item.trace.status == "success" for item in observations)
    )
    if not no_hard_error:
        reasons.add("assistant_hard_error")

    context = _OracleContext(
        query=query,
        result=result,
        receipt=receipt,
        task=task,
        observations=observations,
        sidecar=sidecar,
        sections=sections,
        section_valid=not section_reasons,
        base_tool_valid=base_tool_valid and input_valid,
    )
    decision = oracles[effective_capability].evaluate(context)
    reasons.update(decision.reason_codes)
    components = (
        route_acceptable,
        no_hard_error,
        decision.tool_contract_pass,
        decision.evidence_grounded,
        decision.output_contract_pass,
    )
    return GCSQueryScore(
        query_id=query.query_id,
        config=result.config,
        canonical_capability=query.canonical_capability,
        component_id=binding.component_id,
        route_disposition=route_disposition,
        answer_mode=decision.answer_mode,
        oracle_available=decision.oracle_available,
        semantic_claim_support_resolved=(decision.semantic_claim_support_resolved),
        route_acceptable=route_acceptable,
        no_hard_error=no_hard_error,
        tool_contract_pass=decision.tool_contract_pass,
        evidence_grounded=decision.evidence_grounded,
        output_contract_pass=decision.output_contract_pass,
        hard_error=1 - no_hard_error,
        gcs=int(all(components)),
        reason_codes=tuple(sorted(reasons)),
    )


class GCSMappedQueryScoreV2(_StrictFrozenModel):
    """Logical score plus the immutable physical artifact provenance."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    sidecar_mapping_sha256: Sha256
    projection: GCSLogicalSidecarProjectionV2
    physical_score: GCSQueryScoreV2
    logical_score: GCSQueryScoreV2
    score_sha256: Sha256

    @model_validator(mode="after")
    def validate_v2_score(self) -> Self:
        if (
            self.physical_score.query_id != self.projection.query_id
            or self.physical_score.config != self.projection.physical_config
            or self.logical_score.query_id != self.projection.query_id
            or self.logical_score.config != self.projection.logical_config
        ):
            raise ValueError("GCS v2 score identity differs from its projection")
        expected_logical = GCSQueryScoreV2.model_validate(
            {
                **self.physical_score.model_dump(mode="python"),
                "config": self.projection.logical_config,
            },
            strict=True,
        )
        if self.logical_score != expected_logical:
            raise ValueError("GCS v2 logical score changed physical score semantics")
        if self.projection.sidecar_status != "available":
            raise ValueError("mapped GCS v2 score requires an available sidecar")
        if self.score_sha256 != _self_hash(self, "score_sha256"):
            raise ValueError("GCS v2 score hash mismatch")
        return self


def _gcs_v2_alias_projection(
    sidecar_map: GCSFiveConfigSidecarMapV2,
    *,
    query_id: str,
    logical_config: AssistantRunConfig,
) -> GCSLogicalSidecarProjectionV2:
    matches = tuple(
        row
        for row in sidecar_map.rows
        if row.query_id == query_id and row.logical_config == logical_config
    )
    if len(matches) != 1:
        raise PortfolioGCSError("GCS v2 logical row lacks one physical projection")
    return matches[0]


def score_portfolio_gcs_mapped_v2(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV2,
    task_spec: TaskSpecification,
    oracles: Mapping[str, CapabilityGCSOracle],
    *,
    logical_config: AssistantRunConfig,
    assistant_checkpoint_file_bytes: bytes,
    population: GCSPopulationV2,
    sidecar_map: GCSFiveConfigSidecarMapV2,
) -> GCSMappedQueryScoreV2:
    """Score physical artifacts and project them into one logical v2 row."""

    if not _strict_model_revalidates(sidecar_map, GCSFiveConfigSidecarMapV2):
        raise PortfolioGCSError("GCS v2 sidecar map contract is invalid")
    if population.population_mapping_sha256 != sidecar_map.population_mapping_sha256:
        raise PortfolioGCSError("GCS v2 sidecar map differs from the population")
    projection = _gcs_v2_alias_projection(
        sidecar_map, query_id=query.query_id, logical_config=logical_config
    )
    physical_launch_matches = tuple(
        item
        for item in sidecar_map.launch_instances
        if item.query_id == query.query_id and item.config == projection.physical_config
    )
    if len(physical_launch_matches) != 1:
        raise PortfolioGCSError("GCS v2 physical launch instance is unavailable")
    if (
        result.query_id != projection.query_id
        or result.config != projection.physical_config
        or result.query_artifact_sha256 != projection.query_artifact_sha256
        or result.split_manifest_sha256 != projection.split_manifest_sha256
        or _hash_json(query.model_dump(mode="json"))
        != projection.evaluation_query_sha256
        or receipt.receipt_sha256 != projection.assistant_receipt_sha256
        or sha256_bytes(assistant_checkpoint_file_bytes)
        != projection.assistant_checkpoint_file_sha256
    ):
        raise PortfolioGCSError("GCS v2 physical Assistant artifact binding drifted")
    _request, embedded_sidecar = _validate_assistant_checkpoint_binding_v2(
        assistant_checkpoint_file_bytes,
        launch_instance=physical_launch_matches[0],
        query=query,
        result=result,
        receipt=receipt,
        query_artifact_sha256=projection.query_artifact_sha256,
        split_manifest_sha256=projection.split_manifest_sha256,
    )
    canonical_sidecar_bytes = canonical_json_bytes(sidecar.model_dump(mode="json"))
    if (
        projection.sidecar_status != "available"
        or embedded_sidecar != sidecar
        or sidecar.evidence_sha256 != projection.scorer_evidence_sha256
        or sha256_bytes(canonical_sidecar_bytes)
        != projection.scorer_evidence_document_sha256
    ):
        raise PortfolioGCSIntegrityError(
            "mapped GCS v2 physical scorer sidecar binding drifted"
        )

    physical_score = score_portfolio_gcs_v2(
        query,
        result,
        receipt,
        sidecar,
        task_spec,
        oracles,
        population=population,
    )
    logical_score = GCSQueryScoreV2.model_validate(
        {
            **physical_score.model_dump(mode="python"),
            "config": logical_config,
        },
        strict=True,
    )
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "sidecar_mapping_sha256": sidecar_map.mapping_sha256,
        "projection": projection,
        "physical_score": physical_score,
        "logical_score": logical_score,
    }
    return GCSMappedQueryScoreV2.model_validate(
        {**unsigned, "score_sha256": _hash_json(_jsonable(unsigned))}, strict=True
    )


class GCSFiveConfigScoresV2(_StrictFrozenModel):
    """A strict N x 5 logical score rectangle under one sidecar map."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    sidecar_mapping_sha256: Sha256
    population_mapping_sha256: Sha256
    query_count: int = Field(ge=1)
    logical_row_count: int = Field(ge=5)
    config_order: tuple[AssistantRunConfig, ...]
    rows: tuple[GCSMappedQueryScoreV2, ...]
    scores_sha256: Sha256

    @field_validator("config_order", "rows", mode="before")
    @classmethod
    def coerce_score_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_scores(self) -> Self:
        if self.config_order != MAIN_CONFIG_ORDER:
            raise ValueError("GCS v2 score config order is not frozen")
        if (
            self.logical_row_count != self.query_count * len(MAIN_CONFIG_ORDER)
            or len(self.rows) != self.logical_row_count
        ):
            raise ValueError("GCS v2 score denominator is inconsistent")
        logical_index = {
            config: index for index, config in enumerate(MAIN_CONFIG_ORDER)
        }
        if self.rows != tuple(
            sorted(
                self.rows,
                key=lambda item: (
                    item.logical_score.query_id,
                    logical_index[item.logical_score.config],
                ),
            )
        ):
            raise ValueError("GCS v2 scores are not canonical")
        by_key: dict[tuple[str, AssistantRunConfig], GCSMappedQueryScoreV2] = {}
        for row in self.rows:
            key = (row.logical_score.query_id, row.logical_score.config)
            if key in by_key:
                raise ValueError("GCS v2 contains duplicate logical scores")
            by_key[key] = row
            if (
                row.sidecar_mapping_sha256 != self.sidecar_mapping_sha256
                or row.logical_score.component_id is None
            ):
                raise ValueError("GCS v2 score mapping identity drifted")
        query_ids = sorted({query_id for query_id, _config in by_key})
        if len(query_ids) != self.query_count or set(by_key) != {
            (query_id, config) for query_id in query_ids for config in MAIN_CONFIG_ORDER
        }:
            raise ValueError("GCS v2 scores are not rectangular")
        for query_id in query_ids:
            source = by_key[(query_id, "s1s2")]
            full = by_key[(query_id, "full")]
            if (
                full.physical_score != source.physical_score
                or full.projection.physical_binding_sha256
                != source.projection.physical_binding_sha256
                or full.logical_score.model_dump(mode="python", exclude={"config"})
                != source.logical_score.model_dump(mode="python", exclude={"config"})
            ):
                raise ValueError("GCS v2 Full score is not an exact S1+S2 tie")
        if self.scores_sha256 != _self_hash(self, "scores_sha256"):
            raise ValueError("GCS v2 score matrix hash mismatch")
        return self


def build_gcs_five_config_scores_v2(
    rows: Sequence[GCSMappedQueryScoreV2],
    queries: Sequence[Query],
    sidecar_map: GCSFiveConfigSidecarMapV2,
) -> GCSFiveConfigScoresV2:
    """Validate and freeze all five logical configurations for one population."""

    if not _strict_model_revalidates(sidecar_map, GCSFiveConfigSidecarMapV2):
        raise PortfolioGCSError("GCS v2 sidecar map contract is invalid")
    if any(not _strict_model_revalidates(item, GCSMappedQueryScoreV2) for item in rows):
        raise PortfolioGCSError("GCS v2 contains an invalid logical score")
    population = build_gcs_population_v2(queries)
    if population.population_mapping_sha256 != sidecar_map.population_mapping_sha256:
        raise PortfolioGCSError("GCS v2 score population differs from sidecar map")
    logical_order = {config: index for index, config in enumerate(MAIN_CONFIG_ORDER)}
    ordered = tuple(
        sorted(
            rows,
            key=lambda item: (
                item.logical_score.query_id,
                logical_order[item.logical_score.config],
            ),
        )
    )
    map_rows = {(row.query_id, row.logical_config): row for row in sidecar_map.rows}
    for row in ordered:
        projection = map_rows.get(
            (row.logical_score.query_id, row.logical_score.config)
        )
        if projection is None or row.projection != projection:
            raise PortfolioGCSError("GCS v2 score projection differs from sidecar map")
    for config in MAIN_CONFIG_ORDER:
        _validate_score_population_v2(
            tuple(
                row.logical_score
                for row in ordered
                if row.logical_score.config == config
            ),
            queries,
            population,
            expected_config=config,
            label=f"GCS v2 logical {config}",
        )
    for config in GCS_V2_PHYSICAL_CONFIG_ORDER:
        _validate_score_population_v2(
            tuple(
                row.physical_score
                for row in ordered
                if row.logical_score.config == config
            ),
            queries,
            population,
            expected_config=config,
            label=f"GCS v2 physical {config}",
        )
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "sidecar_mapping_sha256": sidecar_map.mapping_sha256,
        "population_mapping_sha256": population.population_mapping_sha256,
        "query_count": len(population.bindings),
        "logical_row_count": len(ordered),
        "config_order": MAIN_CONFIG_ORDER,
        "rows": ordered,
    }
    try:
        return GCSFiveConfigScoresV2.model_validate(
            {**unsigned, "scores_sha256": _hash_json(_jsonable(unsigned))},
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioGCSError("GCS v2 score matrix is invalid") from exc


def gcs_five_config_logical_scores_v2(
    scores: GCSFiveConfigScoresV2, config: AssistantRunConfig
) -> tuple[GCSQueryScoreV2, ...]:
    """Expose one logical config from the five-config mapped score matrix."""

    if not _strict_model_revalidates(scores, GCSFiveConfigScoresV2):
        raise PortfolioGCSError("GCS v2 score matrix contract is invalid")
    rows = tuple(
        row.logical_score for row in scores.rows if row.logical_score.config == config
    )
    if len(rows) != scores.query_count:
        raise PortfolioGCSError("GCS v2 logical config is not rectangular")
    return rows


def score_portfolio_gcs_v2(
    query: Query,
    result: AssistantResult,
    receipt: AssistantExecutionReceipt,
    sidecar: PublicScorerEvidenceV2,
    task_spec: TaskSpecification,
    oracles: Mapping[str, CapabilityGCSOracle],
    *,
    population: GCSPopulationV2,
) -> GCSQueryScoreV2:
    """Score one v2 row; sidecar/checkpoint integrity errors are always fatal."""

    if not _strict_model_revalidates(population, GCSPopulationV2):
        raise PortfolioGCSError("GCS v2 population contract is invalid")
    binding = _population_binding(population, query)
    if set(oracles) != set(GCS_CAPABILITY_ORDER):
        raise PortfolioGCSError(
            "GCS v2 oracle registry must contain exactly six capabilities"
        )
    if any(key != oracle.capability_id for key, oracle in oracles.items()):
        raise PortfolioGCSError(
            "GCS v2 oracle registry key differs from oracle identity"
        )
    try:
        sidecar = require_public_scorer_evidence_v2(sidecar)
    except PublicScorerEvidenceIntegrityError as error:
        raise PortfolioGCSIntegrityError(str(error)) from error
    if not _strict_model_revalidates(result, AssistantResult) or not (
        _strict_model_revalidates(receipt, AssistantExecutionReceipt)
    ):
        raise PortfolioGCSIntegrityError(
            "Assistant result or execution receipt failed strict revalidation"
        )
    if not _sidecar_binding_valid(query, result, receipt, sidecar):
        raise PortfolioGCSIntegrityError(
            "v2 scorer evidence differs from Assistant/receipt/call identity"
        )

    reasons: set[str] = set()
    canonical_task = task_spec.capabilities_by_id.get(query.canonical_capability)
    if (
        task_spec.task_spec_version != "ecommerce-task-spec-v1"
        or canonical_task is None
        or result.query_id != query.query_id
        or (receipt.outcome == "success" and receipt.query_asset_id != query.asset_id)
    ):
        raise PortfolioGCSIntegrityError(
            "v2 query, TaskSpec, result, or receipt binding is invalid"
        )

    effective_capability = query.canonical_capability
    if (
        result.config != "noskill"
        and result.selected_capability in query.acceptable_capabilities
        and result.selected_capability in task_spec.capabilities_by_id
    ):
        assert result.selected_capability is not None
        effective_capability = result.selected_capability
    task = task_spec.capabilities_by_id[effective_capability]

    observations, base_tool_valid, observation_reasons = _build_observations(
        result, receipt, sidecar
    )
    reasons.update(observation_reasons)
    sections, section_reasons = _parse_sections(
        result.response_text, task.output_contract.required_sections
    )
    reasons.update(section_reasons)

    route_disposition: Literal["pass", "fail", "not_applicable"] = "fail"
    route_acceptable = 0
    attempt = receipt.route_attempt
    if result.config == "noskill":
        if (
            attempt is not None
            and attempt.status == "not_applicable"
            and result.selected_capability is None
            and result.skill_slug is None
            and result.route_trace_sha256 is None
            and result.bank_sha256 is None
        ):
            route_disposition = "not_applicable"
            route_acceptable = 1
    elif (
        attempt is not None
        and attempt.status == "selected"
        and attempt.selected_capability == result.selected_capability
        and attempt.skill_slug == result.skill_slug
        and attempt.route_trace_sha256 == result.route_trace_sha256
        and result.selected_capability in query.acceptable_capabilities
    ):
        route_disposition = "pass"
        route_acceptable = 1
    if not route_acceptable:
        reasons.add("route_unacceptable")

    no_hard_error = int(
        result.error_code is None
        and receipt.outcome == "success"
        and base_tool_valid
        and all(item.trace.status == "success" for item in observations)
    )
    if not no_hard_error:
        reasons.add("assistant_hard_error")

    context = _OracleContext(
        query=query,
        result=result,
        receipt=receipt,
        task=task,
        observations=observations,
        sidecar=sidecar,
        sections=sections,
        section_valid=not section_reasons,
        base_tool_valid=base_tool_valid,
    )
    decision = oracles[effective_capability].evaluate(context)
    reasons.update(decision.reason_codes)
    components = (
        route_acceptable,
        no_hard_error,
        decision.tool_contract_pass,
        decision.evidence_grounded,
        decision.output_contract_pass,
    )
    return GCSQueryScoreV2(
        query_id=query.query_id,
        config=result.config,
        canonical_capability=query.canonical_capability,
        evaluated_capability=effective_capability,
        component_id=binding.component_id,
        route_disposition=route_disposition,
        answer_mode=decision.answer_mode,
        oracle_available=True,
        semantic_claim_support_resolved=(decision.semantic_claim_support_resolved),
        route_acceptable=route_acceptable,
        no_hard_error=no_hard_error,
        tool_contract_pass=decision.tool_contract_pass,
        evidence_grounded=decision.evidence_grounded,
        output_contract_pass=decision.output_contract_pass,
        hard_error=1 - no_hard_error,
        gcs=int(all(components)),
        style_support_status=getattr(decision, "style_support_status", None),
        reason_codes=tuple(sorted(reasons)),
    )


class GCSCapabilitySummary(_StrictFrozenModel):
    capability_id: str
    query_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    gcs_rate: float | None
    hard_error_count: int = Field(ge=0)
    hard_error_rate: float | None
    component_failure_counts: tuple[tuple[str, int], ...]
    oracle_coverage_complete: bool

    @field_validator("component_failure_counts", mode="before")
    @classmethod
    def coerce_component_counts(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.capability_id not in GCS_CAPABILITY_ORDER:
            raise ValueError("unknown GCS capability summary")
        if (
            self.success_count > self.query_count
            or self.hard_error_count > self.query_count
        ):
            raise ValueError("GCS capability counts exceed their denominator")
        expected_rate = (
            None if self.query_count == 0 else self.success_count / self.query_count
        )
        expected_error_rate = (
            None if self.query_count == 0 else self.hard_error_count / self.query_count
        )
        if (
            self.gcs_rate != expected_rate
            or self.hard_error_rate != expected_error_rate
        ):
            raise ValueError("GCS capability rates differ from their counts")
        if (
            tuple(name for name, _count in self.component_failure_counts)
            != GCS_COMPONENTS
        ):
            raise ValueError("GCS capability component order is not frozen")
        if any(
            count < 0 or count > self.query_count
            for _name, count in self.component_failure_counts
        ):
            raise ValueError("GCS capability component count is invalid")
        return self


class GCSConfigSummary(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-grounded-contract-success-v1"] = (
        GCS_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_POLICY_SHA256] = GCS_POLICY_SHA256
    config: AssistantRunConfig
    scope: str
    population_mapping_sha256: Sha256
    query_count: int = Field(ge=1)
    component_count: int = Field(ge=1)
    component_size_histogram: tuple[tuple[int, int], ...]
    capabilities: tuple[GCSCapabilitySummary, ...]
    query_micro_rate: float = Field(ge=0.0, le=1.0)
    headline_macro_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_error_rate: float = Field(ge=0.0, le=1.0)
    component_failure_counts: tuple[tuple[str, int], ...]
    failure_reason_counts: tuple[tuple[str, int], ...]
    oracle_coverage_complete: bool
    headline_available: bool

    @field_validator(
        "component_size_histogram",
        "capabilities",
        "component_failure_counts",
        "failure_reason_counts",
        mode="before",
    )
    @classmethod
    def coerce_summary_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(
                tuple(item) if isinstance(item, list) else item for item in value
            )
        return value

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        return _nonblank(value, "scope")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if (
            tuple(item.capability_id for item in self.capabilities)
            != GCS_CAPABILITY_ORDER
        ):
            raise ValueError("GCS summary capability universe/order is not frozen")
        if sum(item.query_count for item in self.capabilities) != self.query_count:
            raise ValueError("GCS capability summaries do not cover the population")
        if (
            self.component_size_histogram
            != tuple(sorted(self.component_size_histogram))
            or len({size for size, _count in self.component_size_histogram})
            != len(self.component_size_histogram)
            or any(
                size <= 0 or count <= 0 for size, count in self.component_size_histogram
            )
        ):
            raise ValueError("GCS component histogram is invalid")
        if (
            sum(size * count for size, count in self.component_size_histogram)
            != self.query_count
        ):
            raise ValueError("GCS component histogram does not cover the population")
        if (
            sum(count for _size, count in self.component_size_histogram)
            != self.component_count
        ):
            raise ValueError("GCS component histogram count is inconsistent")
        if (
            tuple(name for name, _count in self.component_failure_counts)
            != GCS_COMPONENTS
        ):
            raise ValueError("GCS summary component order is not frozen")
        if self.failure_reason_counts != tuple(sorted(self.failure_reason_counts)):
            raise ValueError("GCS failure reasons must be sorted")
        if len({reason for reason, _count in self.failure_reason_counts}) != len(
            self.failure_reason_counts
        ):
            raise ValueError("GCS failure reasons must be unique")
        if any(count <= 0 for _reason, count in self.failure_reason_counts):
            raise ValueError("GCS failure reason counts must be positive")
        expected_micro = (
            sum(item.success_count for item in self.capabilities) / self.query_count
        )
        expected_hard_error = (
            sum(item.hard_error_count for item in self.capabilities) / self.query_count
        )
        if self.query_micro_rate != expected_micro:
            raise ValueError("GCS query micro differs from capability counts")
        if self.hard_error_rate != expected_hard_error:
            raise ValueError("GCS hard-error rate differs from capability counts")
        expected_component_failures = tuple(
            (
                component,
                sum(
                    dict(item.component_failure_counts)[component]
                    for item in self.capabilities
                ),
            )
            for component in GCS_COMPONENTS
        )
        if self.component_failure_counts != expected_component_failures:
            raise ValueError(
                "GCS global component failures differ from capability counts"
            )
        expected_coverage = all(
            item.oracle_coverage_complete for item in self.capabilities
        )
        if self.oracle_coverage_complete != expected_coverage:
            raise ValueError("GCS oracle coverage differs from capability coverage")
        rates = tuple(item.gcs_rate for item in self.capabilities)
        expected_macro = (
            None
            if any(rate is None for rate in rates)
            else sum(rate for rate in rates if rate is not None)
            / len(GCS_CAPABILITY_ORDER)
        )
        if self.headline_macro_rate != expected_macro:
            raise ValueError("GCS headline macro differs from the six-capability mean")
        if self.headline_available != (
            self.oracle_coverage_complete and expected_macro is not None
        ):
            raise ValueError("GCS headline availability is inconsistent")
        return self


class GCSConfigSummaryV2(GCSConfigSummary):
    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    style_support_status_counts: tuple[tuple[str, int], ...]

    @field_validator("style_support_status_counts", mode="before")
    @classmethod
    def coerce_style_counts(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_style_counts(self) -> Self:
        expected = ("candidates", "no_result", "unsupported", "unresolved")
        if tuple(key for key, _count in self.style_support_status_counts) != expected:
            raise ValueError("v2 Style status count order is frozen")
        if any(
            count < 0 or count > self.query_count
            for _key, count in self.style_support_status_counts
        ):
            raise ValueError("v2 Style status count is invalid")
        return self


def _component_size_histogram(population: GCSPopulation) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(Counter(size for _component, size in population.component_sizes).items())
    )


def _validate_score_population(
    scores: Sequence[GCSQueryScore],
    queries: Sequence[Query],
    population: GCSPopulation,
    *,
    expected_config: AssistantRunConfig | None,
    label: str,
) -> tuple[AssistantRunConfig, dict[str, GCSQueryScore]]:
    if not scores:
        raise PortfolioGCSError(f"{label} GCS scores are empty")
    by_query: dict[str, GCSQueryScore] = {}
    configs: set[AssistantRunConfig] = set()
    for score in scores:
        if not _strict_model_revalidates(score, GCSQueryScore):
            raise PortfolioGCSError(f"{label} contains an invalid terminal score")
        if score.query_id in by_query:
            raise PortfolioGCSError(f"{label} contains duplicate terminal rows")
        by_query[score.query_id] = score
        configs.add(score.config)
    if len(configs) != 1:
        raise PortfolioGCSError(f"{label} must contain exactly one config")
    config = next(iter(configs))
    if expected_config is not None and config != expected_config:
        raise PortfolioGCSError(f"{label} config differs from the requested config")

    query_by_id = {query.query_id: query for query in queries}
    if set(by_query) != set(query_by_id):
        raise PortfolioGCSError(f"{label} is not a rectangular terminal population")
    binding_by_id = {item.query_id: item for item in population.bindings}
    for query_id, score in by_query.items():
        binding = binding_by_id[query_id]
        if (
            score.canonical_capability != query_by_id[query_id].canonical_capability
            or score.canonical_capability != binding.canonical_capability
            or score.component_id != binding.component_id
        ):
            raise PortfolioGCSError(f"{label} score metadata drifted from population")
    return config, by_query


def _validate_score_population_v2(
    scores: Sequence[GCSQueryScoreV2],
    queries: Sequence[Query],
    population: GCSPopulationV2,
    *,
    expected_config: AssistantRunConfig | None,
    label: str,
) -> tuple[AssistantRunConfig, dict[str, GCSQueryScoreV2]]:
    if not scores:
        raise PortfolioGCSError(f"{label} GCS v2 scores are empty")
    by_query: dict[str, GCSQueryScoreV2] = {}
    configs: set[AssistantRunConfig] = set()
    for score in scores:
        if not _strict_model_revalidates(score, GCSQueryScoreV2):
            raise PortfolioGCSError(f"{label} contains an invalid v2 terminal score")
        if score.query_id in by_query:
            raise PortfolioGCSError(f"{label} contains duplicate terminal rows")
        by_query[score.query_id] = score
        configs.add(score.config)
    if len(configs) != 1:
        raise PortfolioGCSError(f"{label} must contain exactly one config")
    config = next(iter(configs))
    if expected_config is not None and config != expected_config:
        raise PortfolioGCSError(f"{label} config differs from the requested config")
    query_by_id = {query.query_id: query for query in queries}
    if set(by_query) != set(query_by_id):
        raise PortfolioGCSError(f"{label} is not a rectangular terminal population")
    binding_by_id = {item.query_id: item for item in population.bindings}
    for query_id, score in by_query.items():
        binding = binding_by_id[query_id]
        if (
            score.canonical_capability != query_by_id[query_id].canonical_capability
            or score.canonical_capability != binding.canonical_capability
            or score.component_id != binding.component_id
        ):
            raise PortfolioGCSError(f"{label} score metadata drifted from population")
    return config, by_query


def summarize_gcs(
    scores: Sequence[GCSQueryScore],
    queries: Sequence[Query],
    config: AssistantRunConfig,
    scope: str,
) -> GCSConfigSummary:
    """Aggregate one complete config over the immutable six-capability universe."""

    scope = _nonblank(scope, "scope")
    population = build_gcs_population(queries)
    _actual_config, score_by_id = _validate_score_population(
        scores,
        queries,
        population,
        expected_config=config,
        label="summary",
    )
    query_by_id = {query.query_id: query for query in queries}

    capability_summaries: list[GCSCapabilitySummary] = []
    for capability in GCS_CAPABILITY_ORDER:
        rows = tuple(
            score_by_id[query_id]
            for query_id in sorted(score_by_id)
            if query_by_id[query_id].canonical_capability == capability
        )
        count = len(rows)
        successes = sum(item.gcs for item in rows)
        hard_errors = sum(item.hard_error for item in rows)
        component_failures = tuple(
            (
                component,
                sum(1 - int(getattr(item, component)) for item in rows),
            )
            for component in GCS_COMPONENTS
        )
        capability_summaries.append(
            GCSCapabilitySummary(
                capability_id=capability,
                query_count=count,
                success_count=successes,
                gcs_rate=None if count == 0 else successes / count,
                hard_error_count=hard_errors,
                hard_error_rate=None if count == 0 else hard_errors / count,
                component_failure_counts=component_failures,
                oracle_coverage_complete=bool(rows)
                and all(
                    item.oracle_available and item.semantic_claim_support_resolved
                    for item in rows
                ),
            )
        )

    all_rows = tuple(score_by_id[query_id] for query_id in sorted(score_by_id))
    rates = tuple(item.gcs_rate for item in capability_summaries)
    macro_rate = (
        None
        if any(rate is None for rate in rates)
        else sum(rate for rate in rates if rate is not None) / len(GCS_CAPABILITY_ORDER)
    )
    oracle_coverage_complete = all(
        item.oracle_coverage_complete for item in capability_summaries
    )
    component_failure_counts = tuple(
        (
            component,
            sum(1 - int(getattr(item, component)) for item in all_rows),
        )
        for component in GCS_COMPONENTS
    )
    reason_counts = Counter(reason for item in all_rows for reason in item.reason_codes)
    return GCSConfigSummary(
        config=config,
        scope=scope,
        population_mapping_sha256=population.population_mapping_sha256,
        query_count=len(all_rows),
        component_count=population.component_count,
        component_size_histogram=_component_size_histogram(population),
        capabilities=tuple(capability_summaries),
        query_micro_rate=sum(item.gcs for item in all_rows) / len(all_rows),
        headline_macro_rate=macro_rate,
        hard_error_rate=sum(item.hard_error for item in all_rows) / len(all_rows),
        component_failure_counts=component_failure_counts,
        failure_reason_counts=tuple(sorted(reason_counts.items())),
        oracle_coverage_complete=oracle_coverage_complete,
        headline_available=oracle_coverage_complete and macro_rate is not None,
    )


class GCSFiveConfigSummaryEnvelopeV2(_StrictFrozenModel):
    """Canonical v2 summary plus the five-config physical-map provenance."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    sidecar_mapping_sha256: Sha256
    score_matrix_sha256: Sha256
    summary: GCSConfigSummaryV2
    summary_sha256: Sha256

    @model_validator(mode="after")
    def validate_five_config_summary(self) -> Self:
        if self.summary_sha256 != _self_hash(self, "summary_sha256"):
            raise ValueError("GCS v2 five-config summary hash mismatch")
        return self


def summarize_gcs_five_config_v2(
    scores: GCSFiveConfigScoresV2,
    queries: Sequence[Query],
    config: AssistantRunConfig,
    scope: str,
) -> GCSFiveConfigSummaryEnvelopeV2:
    """Aggregate one logical config without dropping its v2 physical binding."""

    if not _strict_model_revalidates(scores, GCSFiveConfigScoresV2):
        raise PortfolioGCSError("GCS v2 score matrix contract is invalid")
    population = build_gcs_population_v2(queries)
    if scores.population_mapping_sha256 != population.population_mapping_sha256:
        raise PortfolioGCSError("GCS v2 score matrix population is inconsistent")
    logical_scores = gcs_five_config_logical_scores_v2(scores, config)
    summary = summarize_gcs_v2(logical_scores, queries, config, scope)
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "sidecar_mapping_sha256": scores.sidecar_mapping_sha256,
        "score_matrix_sha256": scores.scores_sha256,
        "summary": summary,
    }
    return GCSFiveConfigSummaryEnvelopeV2.model_validate(
        {**unsigned, "summary_sha256": _hash_json(_jsonable(unsigned))}, strict=True
    )


def summarize_gcs_v2(
    scores: Sequence[GCSQueryScoreV2],
    queries: Sequence[Query],
    config: AssistantRunConfig,
    scope: str,
) -> GCSConfigSummaryV2:
    """Aggregate v2 rows without treating ordinary row-zero as unavailable."""

    scope = _nonblank(scope, "scope")
    population = build_gcs_population_v2(queries)
    _actual_config, score_by_id = _validate_score_population_v2(
        scores,
        queries,
        population,
        expected_config=config,
        label="v2 summary",
    )
    query_by_id = {query.query_id: query for query in queries}
    capability_summaries: list[GCSCapabilitySummary] = []
    for capability in GCS_CAPABILITY_ORDER:
        rows = tuple(
            score_by_id[query_id]
            for query_id in sorted(score_by_id)
            if query_by_id[query_id].canonical_capability == capability
        )
        count = len(rows)
        successes = sum(item.gcs for item in rows)
        hard_errors = sum(item.hard_error for item in rows)
        capability_summaries.append(
            GCSCapabilitySummary(
                capability_id=capability,
                query_count=count,
                success_count=successes,
                gcs_rate=None if count == 0 else successes / count,
                hard_error_count=hard_errors,
                hard_error_rate=None if count == 0 else hard_errors / count,
                component_failure_counts=tuple(
                    (
                        component,
                        sum(1 - int(getattr(item, component)) for item in rows),
                    )
                    for component in GCS_COMPONENTS
                ),
                oracle_coverage_complete=bool(rows)
                and all(item.oracle_available for item in rows),
            )
        )
    all_rows = tuple(score_by_id[query_id] for query_id in sorted(score_by_id))
    rates = tuple(item.gcs_rate for item in capability_summaries)
    macro_rate = (
        None
        if any(rate is None for rate in rates)
        else sum(rate for rate in rates if rate is not None) / len(GCS_CAPABILITY_ORDER)
    )
    oracle_coverage_complete = all(
        item.oracle_coverage_complete for item in capability_summaries
    )
    reason_counts = Counter(reason for item in all_rows for reason in item.reason_codes)
    status_order: tuple[StyleOutcomeStatus, ...] = (
        "candidates",
        "no_result",
        "unsupported",
        "unresolved",
    )
    return GCSConfigSummaryV2(
        config=config,
        scope=scope,
        population_mapping_sha256=population.population_mapping_sha256,
        query_count=len(all_rows),
        component_count=population.component_count,
        component_size_histogram=_component_size_histogram(population),
        capabilities=tuple(capability_summaries),
        query_micro_rate=sum(item.gcs for item in all_rows) / len(all_rows),
        headline_macro_rate=macro_rate,
        hard_error_rate=sum(item.hard_error for item in all_rows) / len(all_rows),
        component_failure_counts=tuple(
            (
                component,
                sum(1 - int(getattr(item, component)) for item in all_rows),
            )
            for component in GCS_COMPONENTS
        ),
        failure_reason_counts=tuple(sorted(reason_counts.items())),
        oracle_coverage_complete=oracle_coverage_complete,
        headline_available=oracle_coverage_complete and macro_rate is not None,
        style_support_status_counts=tuple(
            (
                status,
                sum(item.style_support_status == status for item in all_rows),
            )
            for status in status_order
        ),
    )


class GCSBootstrapInterval(_StrictFrozenModel):
    low_pp: float
    high_pp: float

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not math.isfinite(self.low_pp) or not math.isfinite(self.high_pp):
            raise ValueError("GCS bootstrap interval must be finite")
        if self.low_pp > self.high_pp:
            raise ValueError("GCS bootstrap interval is reversed")
        return self


class GCSCapabilityBootstrapDelta(_StrictFrozenModel):
    capability_id: str
    point_delta_pp: float | None
    ci95: GCSBootstrapInterval | None
    available_replicates: int = Field(ge=0, le=GCS_BOOTSTRAP_REPLICATES)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        if self.capability_id not in GCS_CAPABILITY_ORDER:
            raise ValueError("unknown GCS bootstrap capability")
        if self.point_delta_pp is not None and not math.isfinite(self.point_delta_pp):
            raise ValueError("GCS capability delta must be finite")
        if self.point_delta_pp is None and (
            self.ci95 is not None or self.available_replicates != 0
        ):
            raise ValueError("missing GCS capability point cannot have replicates")
        if (self.ci95 is not None) != (
            self.available_replicates == GCS_BOOTSTRAP_REPLICATES
        ):
            raise ValueError("GCS capability CI availability is inconsistent")
        return self


class PairedGCSBootstrap(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-gcs-component-bootstrap-v1"] = (
        GCS_BOOTSTRAP_POLICY_VERSION
    )
    gcs_policy_sha256: Literal[GCS_POLICY_SHA256] = GCS_POLICY_SHA256
    baseline_config: AssistantRunConfig
    treatment_config: AssistantRunConfig
    scope: str
    population_mapping_sha256: Sha256
    query_count: int = Field(ge=1)
    component_count: int = Field(ge=1)
    component_size_histogram: tuple[tuple[int, int], ...]
    root_seed: Literal[2026080601] = GCS_BOOTSTRAP_ROOT_SEED
    derived_seed: int = Field(ge=0)
    replicates: Literal[10000] = GCS_BOOTSTRAP_REPLICATES
    draw_stream_sha256: Sha256
    macro_delta_pp: float | None
    macro_ci95: GCSBootstrapInterval | None
    macro_available_replicates: int = Field(ge=0, le=GCS_BOOTSTRAP_REPLICATES)
    query_micro_delta_pp: float
    query_micro_ci95: GCSBootstrapInterval
    hard_error_delta_pp: float
    hard_error_ci95: GCSBootstrapInterval
    per_capability: tuple[GCSCapabilityBootstrapDelta, ...]
    baseline_oracle_coverage_complete: bool
    treatment_oracle_coverage_complete: bool
    zero_variance: bool
    descriptive_only: bool
    precision_warnings: tuple[
        Literal[
            "single_component_descriptive_only",
            "low_component_count_precision",
            "zero_variance_descriptive_only",
            "replicate_capability_missing",
        ],
        ...,
    ] = ()

    @field_validator(
        "component_size_histogram",
        "per_capability",
        "precision_warnings",
        mode="before",
    )
    @classmethod
    def coerce_bootstrap_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(
                tuple(item) if isinstance(item, list) else item for item in value
            )
        return value

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        return _nonblank(value, "scope")

    @model_validator(mode="after")
    def validate_bootstrap(self) -> Self:
        if self.baseline_config == self.treatment_config:
            raise ValueError("paired GCS contrast requires two configs")
        if (
            self.component_size_histogram
            != tuple(sorted(self.component_size_histogram))
            or len({size for size, _count in self.component_size_histogram})
            != len(self.component_size_histogram)
            or any(
                size <= 0 or count <= 0 for size, count in self.component_size_histogram
            )
        ):
            raise ValueError("paired GCS component histogram is invalid")
        if (
            sum(size * count for size, count in self.component_size_histogram)
            != self.query_count
        ):
            raise ValueError("paired GCS component histogram does not cover queries")
        if (
            sum(count for _size, count in self.component_size_histogram)
            != self.component_count
        ):
            raise ValueError("paired GCS component histogram count is inconsistent")
        if (
            tuple(item.capability_id for item in self.per_capability)
            != GCS_CAPABILITY_ORDER
        ):
            raise ValueError("paired GCS capability universe/order is not frozen")
        if (self.macro_ci95 is not None) != (
            self.macro_available_replicates == self.replicates
        ):
            raise ValueError("paired GCS macro CI availability is inconsistent")
        if self.macro_delta_pp is None and self.macro_ci95 is not None:
            raise ValueError("paired GCS macro CI lacks a point estimate")
        if not all(
            math.isfinite(value)
            for value in (self.query_micro_delta_pp, self.hard_error_delta_pp)
        ):
            raise ValueError("paired GCS point deltas must be finite")
        if self.precision_warnings != tuple(dict.fromkeys(self.precision_warnings)):
            raise ValueError("paired GCS precision warnings must be unique")
        return self


class PairedGCSBootstrapMetricsV2(PairedGCSBootstrap):
    """Unchanged component-bootstrap schema evaluated under GCS v2 coverage."""

    gcs_policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(
    values: Sequence[float], *, orientation: Literal[-1, 1]
) -> GCSBootstrapInterval | None:
    if len(values) != GCS_BOOTSTRAP_REPLICATES:
        return None
    low = _percentile(values, 0.025)
    high = _percentile(values, 0.975)
    if orientation == -1:
        low, high = -high, -low
    return GCSBootstrapInterval(low_pp=low, high_pp=high)


def _metric_rates(
    query_ids: Sequence[str],
    scores: Mapping[str, GCSQueryScore],
    query_by_id: Mapping[str, Query],
) -> tuple[float | None, float, float, dict[str, float | None]]:
    total = len(query_ids)
    micro = sum(scores[query_id].gcs for query_id in query_ids) / total
    hard_error = sum(scores[query_id].hard_error for query_id in query_ids) / total
    per_capability: dict[str, float | None] = {}
    for capability in GCS_CAPABILITY_ORDER:
        capability_ids = tuple(
            query_id
            for query_id in query_ids
            if query_by_id[query_id].canonical_capability == capability
        )
        per_capability[capability] = (
            None
            if not capability_ids
            else sum(scores[query_id].gcs for query_id in capability_ids)
            / len(capability_ids)
        )
    macro = (
        None
        if any(value is None for value in per_capability.values())
        else sum(value for value in per_capability.values() if value is not None)
        / len(GCS_CAPABILITY_ORDER)
    )
    return macro, micro, hard_error, per_capability


def paired_component_bootstrap(
    baseline: Sequence[GCSQueryScore],
    treatment: Sequence[GCSQueryScore],
    queries: Sequence[Query],
    scope: str,
    *,
    _coverage_semantics: Literal["v1", "v2"] = "v1",
) -> PairedGCSBootstrap:
    """Bootstrap a paired contrast by resampling whole connected components."""

    scope = _nonblank(scope, "scope")
    population = build_gcs_population(queries)
    baseline_config, baseline_by_id = _validate_score_population(
        baseline,
        queries,
        population,
        expected_config=None,
        label="baseline",
    )
    treatment_config, treatment_by_id = _validate_score_population(
        treatment,
        queries,
        population,
        expected_config=None,
        label="treatment",
    )
    if baseline_config == treatment_config:
        raise PortfolioGCSError("paired GCS contrast requires two distinct configs")

    query_by_id = {query.query_id: query for query in queries}
    present_capabilities = {
        query.canonical_capability for query in query_by_id.values()
    }
    require_semantic_resolution = _coverage_semantics == "v1"
    baseline_coverage = present_capabilities == set(GCS_CAPABILITY_ORDER) and all(
        item.oracle_available
        and (item.semantic_claim_support_resolved or not require_semantic_resolution)
        for item in baseline_by_id.values()
    )
    treatment_coverage = present_capabilities == set(GCS_CAPABILITY_ORDER) and all(
        item.oracle_available
        and (item.semantic_claim_support_resolved or not require_semantic_resolution)
        for item in treatment_by_id.values()
    )
    if str(baseline_config) < str(treatment_config):
        left_by_id, right_by_id = baseline_by_id, treatment_by_id
        orientation: Literal[-1, 1] = 1
    else:
        left_by_id, right_by_id = treatment_by_id, baseline_by_id
        orientation = -1

    ordered_ids = tuple(sorted(query_by_id))
    left_point = _metric_rates(ordered_ids, left_by_id, query_by_id)
    right_point = _metric_rates(ordered_ids, right_by_id, query_by_id)
    point_macro = (
        None
        if left_point[0] is None or right_point[0] is None
        else orientation * 100.0 * (right_point[0] - left_point[0])
    )
    point_micro = orientation * 100.0 * (right_point[1] - left_point[1])
    point_hard_error = orientation * 100.0 * (right_point[2] - left_point[2])
    point_per_capability = {
        capability: (
            None
            if left_point[3][capability] is None or right_point[3][capability] is None
            else orientation
            * 100.0
            * (right_point[3][capability] - left_point[3][capability])
        )
        for capability in GCS_CAPABILITY_ORDER
    }

    component_ids = tuple(component for component, _size in population.component_sizes)
    members: dict[str, tuple[str, ...]] = defaultdict(tuple)
    mutable_members: dict[str, list[str]] = defaultdict(list)
    for binding in population.bindings:
        mutable_members[binding.component_id].append(binding.query_id)
    members = {
        component: tuple(sorted(query_ids))
        for component, query_ids in mutable_members.items()
    }
    seed_payload = {
        "policy_version": GCS_BOOTSTRAP_POLICY_VERSION,
        "root_seed": GCS_BOOTSTRAP_ROOT_SEED,
        "scope": scope,
        "population_mapping_sha256": population.population_mapping_sha256,
    }
    seed_digest = hashlib.sha256(canonical_json_bytes(seed_payload)).digest()
    derived_seed = int.from_bytes(seed_digest[:16], "big")
    rng = random.Random(derived_seed)
    draw_hasher = hashlib.sha256()
    macro_deltas: list[float] = []
    micro_deltas: list[float] = []
    hard_error_deltas: list[float] = []
    capability_deltas: dict[str, list[float]] = {
        capability: [] for capability in GCS_CAPABILITY_ORDER
    }
    for _replicate in range(GCS_BOOTSTRAP_REPLICATES):
        draw = tuple(
            component_ids[rng.randrange(len(component_ids))] for _index in component_ids
        )
        draw_hasher.update(canonical_json_bytes(list(draw)))
        sampled_ids = tuple(
            query_id for component in draw for query_id in members[component]
        )
        left_rates = _metric_rates(sampled_ids, left_by_id, query_by_id)
        right_rates = _metric_rates(sampled_ids, right_by_id, query_by_id)
        micro_deltas.append(100.0 * (right_rates[1] - left_rates[1]))
        hard_error_deltas.append(100.0 * (right_rates[2] - left_rates[2]))
        if left_rates[0] is not None and right_rates[0] is not None:
            macro_deltas.append(100.0 * (right_rates[0] - left_rates[0]))
        for capability in GCS_CAPABILITY_ORDER:
            left_rate = left_rates[3][capability]
            right_rate = right_rates[3][capability]
            if left_rate is not None and right_rate is not None:
                capability_deltas[capability].append(100.0 * (right_rate - left_rate))

    per_capability = tuple(
        GCSCapabilityBootstrapDelta(
            capability_id=capability,
            point_delta_pp=point_per_capability[capability],
            ci95=_interval(capability_deltas[capability], orientation=orientation),
            available_replicates=len(capability_deltas[capability]),
        )
        for capability in GCS_CAPABILITY_ORDER
    )
    all_nonempty_series = [micro_deltas, hard_error_deltas] + [
        values for values in capability_deltas.values() if values
    ]
    if macro_deltas:
        all_nonempty_series.append(macro_deltas)
    zero_variance = all(min(series) == max(series) for series in all_nonempty_series)
    warnings: list[str] = []
    if population.component_count == 1:
        warnings.append("single_component_descriptive_only")
    elif population.component_count < 20:
        warnings.append("low_component_count_precision")
    if zero_variance:
        warnings.append("zero_variance_descriptive_only")
    if len(macro_deltas) != GCS_BOOTSTRAP_REPLICATES or any(
        len(values) != GCS_BOOTSTRAP_REPLICATES for values in capability_deltas.values()
    ):
        warnings.append("replicate_capability_missing")

    return PairedGCSBootstrap(
        baseline_config=baseline_config,
        treatment_config=treatment_config,
        scope=scope,
        population_mapping_sha256=population.population_mapping_sha256,
        query_count=len(ordered_ids),
        component_count=population.component_count,
        component_size_histogram=_component_size_histogram(population),
        derived_seed=derived_seed,
        draw_stream_sha256=draw_hasher.hexdigest(),
        macro_delta_pp=point_macro,
        macro_ci95=_interval(macro_deltas, orientation=orientation),
        macro_available_replicates=len(macro_deltas),
        query_micro_delta_pp=point_micro,
        query_micro_ci95=_interval(
            micro_deltas, orientation=orientation
        ),  # always complete
        hard_error_delta_pp=point_hard_error,
        hard_error_ci95=_interval(
            hard_error_deltas, orientation=orientation
        ),  # always complete
        per_capability=per_capability,
        baseline_oracle_coverage_complete=baseline_coverage,
        treatment_oracle_coverage_complete=treatment_coverage,
        zero_variance=zero_variance,
        descriptive_only=population.component_count == 1 or zero_variance,
        precision_warnings=tuple(warnings),
    )


class GCSFiveConfigBootstrapEnvelopeV2(_StrictFrozenModel):
    """Component bootstrap plus the five-config physical-map provenance."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    sidecar_mapping_sha256: Sha256
    score_matrix_sha256: Sha256
    bootstrap: PairedGCSBootstrapMetricsV2
    bootstrap_sha256: Sha256

    @model_validator(mode="after")
    def validate_v2_bootstrap(self) -> Self:
        if self.bootstrap_sha256 != _self_hash(self, "bootstrap_sha256"):
            raise ValueError("GCS v2 five-config bootstrap hash mismatch")
        return self


def _gcs_v2_bootstrap_compat_scores(
    scores: Sequence[GCSQueryScoreV2],
) -> tuple[GCSQueryScore, ...]:
    """Project v2 metric fields into the unchanged v1 bootstrap carrier.

    The component bootstrap reads only the five binary components, hard-error
    flag, population identity, and oracle-coverage flags.  Its v1 carrier is
    retained for byte-compatible confidence intervals; v2-only diagnostics stay
    bound by the enclosing five-config score-matrix hash.
    """

    rows: list[GCSQueryScore] = []
    for score in scores:
        payload = score.model_dump(
            mode="python", exclude={"evaluated_capability", "style_support_status"}
        )
        payload.update(
            policy_version=GCS_POLICY_VERSION,
            policy_sha256=GCS_POLICY_SHA256,
            reason_codes=tuple(
                reason for reason in score.reason_codes if reason in GCS_REASON_CODES
            ),
        )
        rows.append(GCSQueryScore.model_validate(payload, strict=True))
    return tuple(rows)


def paired_component_bootstrap_five_config_v2(
    scores: GCSFiveConfigScoresV2,
    baseline_config: AssistantRunConfig,
    treatment_config: AssistantRunConfig,
    queries: Sequence[Query],
    scope: str,
) -> GCSFiveConfigBootstrapEnvelopeV2:
    """Run the frozen component bootstrap under five-config map provenance."""

    if not _strict_model_revalidates(scores, GCSFiveConfigScoresV2):
        raise PortfolioGCSError("GCS v2 score matrix contract is invalid")
    population = build_gcs_population_v2(queries)
    if scores.population_mapping_sha256 != population.population_mapping_sha256:
        raise PortfolioGCSError("GCS v2 score matrix population is inconsistent")
    legacy_bootstrap = paired_component_bootstrap(
        _gcs_v2_bootstrap_compat_scores(
            gcs_five_config_logical_scores_v2(scores, baseline_config)
        ),
        _gcs_v2_bootstrap_compat_scores(
            gcs_five_config_logical_scores_v2(scores, treatment_config)
        ),
        queries,
        scope,
        _coverage_semantics="v2",
    )
    bootstrap = PairedGCSBootstrapMetricsV2.model_validate(
        {
            **legacy_bootstrap.model_dump(mode="python"),
            "gcs_policy_sha256": GCS_V2_POLICY_SHA256,
        },
        strict=True,
    )
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "sidecar_mapping_sha256": scores.sidecar_mapping_sha256,
        "score_matrix_sha256": scores.scores_sha256,
        "bootstrap": bootstrap,
    }
    return GCSFiveConfigBootstrapEnvelopeV2.model_validate(
        {**unsigned, "bootstrap_sha256": _hash_json(_jsonable(unsigned))},
        strict=True,
    )


class GCSSystemGainCriterion(_StrictFrozenModel):
    name: str
    operator: Literal[">=", "<="]
    threshold_pp: float
    observed_pp: float | None
    passed: bool | None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _nonblank(value, "criterion name")

    @model_validator(mode="after")
    def validate_criterion(self) -> Self:
        values = (self.threshold_pp, self.observed_pp)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("GCS gate criterion values must be finite")
        if self.passed is not None:
            if self.observed_pp is None:
                raise ValueError("resolved GCS criterion lacks an observation")
            expected = (
                self.observed_pp >= self.threshold_pp
                if self.operator == ">="
                else self.observed_pp <= self.threshold_pp
            )
            if self.passed != expected:
                raise ValueError("GCS criterion result differs from its threshold")
        return self


class GCSSystemGainDecision(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-grounded-contract-success-v1"] = (
        GCS_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_POLICY_SHA256] = GCS_POLICY_SHA256
    baseline_config: AssistantRunConfig
    treatment_config: AssistantRunConfig
    scope: str
    population_mapping_sha256: Sha256
    status: Literal["passed", "failed", "unavailable"]
    criteria: tuple[GCSSystemGainCriterion, ...]
    reason_codes: tuple[str, ...]

    @field_validator("criteria", "reason_codes", mode="before")
    @classmethod
    def coerce_decision_tuples(cls, value: object) -> object:
        return _coerce_tuple(value)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.status == "unavailable":
            if any(item.passed is not None for item in self.criteria):
                raise ValueError("unavailable GCS gate cannot claim criterion results")
        elif any(item.passed is None for item in self.criteria):
            raise ValueError("available GCS gate requires every criterion result")
        if self.status == "passed" and not all(item.passed for item in self.criteria):
            raise ValueError("passed GCS gate contains a failed criterion")
        if self.status == "failed" and all(item.passed for item in self.criteria):
            raise ValueError("failed GCS gate lacks a failed criterion")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("GCS gate reason codes must be sorted and unique")
        return self


def evaluate_gcs_system_gain(
    summary: GCSConfigSummary, bootstrap: PairedGCSBootstrap
) -> GCSSystemGainDecision:
    """Apply the frozen Portfolio system-gain thresholds without a Judge."""

    if (
        summary.config != bootstrap.treatment_config
        or summary.scope != bootstrap.scope
        or summary.population_mapping_sha256 != bootstrap.population_mapping_sha256
        or summary.query_count != bootstrap.query_count
        or summary.oracle_coverage_complete
        != bootstrap.treatment_oracle_coverage_complete
    ):
        raise PortfolioGCSError("GCS summary/bootstrap binding mismatch")

    per_capability = {item.capability_id: item for item in bootstrap.per_capability}
    raw_criteria = [
        (
            "macro_delta_pp",
            ">=",
            SYSTEM_GAIN_MACRO_DELTA_PP_MIN,
            bootstrap.macro_delta_pp,
        ),
        (
            "macro_ci95_low_pp",
            ">=",
            SYSTEM_GAIN_MACRO_CI95_LOW_PP_MIN,
            None if bootstrap.macro_ci95 is None else bootstrap.macro_ci95.low_pp,
        ),
        (
            "hard_error_delta_pp",
            "<=",
            SYSTEM_GAIN_HARD_ERROR_DELTA_PP_MAX,
            bootstrap.hard_error_delta_pp,
        ),
        *(
            (
                f"capability_delta_pp:{capability}",
                ">=",
                SYSTEM_GAIN_PER_CAPABILITY_DELTA_PP_MIN,
                per_capability[capability].point_delta_pp,
            )
            for capability in GCS_CAPABILITY_ORDER
        ),
    ]
    unavailable_reasons: set[str] = set()
    if not bootstrap.baseline_oracle_coverage_complete:
        unavailable_reasons.add("baseline_oracle_coverage_incomplete")
    if not bootstrap.treatment_oracle_coverage_complete:
        unavailable_reasons.add("treatment_oracle_coverage_incomplete")
    if not summary.headline_available:
        unavailable_reasons.add("headline_unavailable")
    if bootstrap.macro_ci95 is None:
        unavailable_reasons.add("macro_ci_unavailable")
    if any(value is None for _name, _operator, _threshold, value in raw_criteria):
        unavailable_reasons.add("criterion_unavailable")
    available = not unavailable_reasons

    criteria: list[GCSSystemGainCriterion] = []
    for name, operator, threshold, observed in raw_criteria:
        passed: bool | None = None
        if available:
            assert observed is not None
            passed = (
                observed >= threshold if operator == ">=" else observed <= threshold
            )
        criteria.append(
            GCSSystemGainCriterion(
                name=name,
                operator=operator,  # type: ignore[arg-type]
                threshold_pp=threshold,
                observed_pp=observed,
                passed=passed,
            )
        )
    if not available:
        status: Literal["passed", "failed", "unavailable"] = "unavailable"
        reasons = unavailable_reasons
    elif all(item.passed for item in criteria):
        status = "passed"
        reasons = set()
    else:
        status = "failed"
        reasons = {
            f"threshold_failed:{item.name}" for item in criteria if item.passed is False
        }
    return GCSSystemGainDecision(
        baseline_config=bootstrap.baseline_config,
        treatment_config=bootstrap.treatment_config,
        scope=bootstrap.scope,
        population_mapping_sha256=bootstrap.population_mapping_sha256,
        status=status,
        criteria=tuple(criteria),
        reason_codes=tuple(sorted(reasons)),
    )


class GCSFiveConfigSystemGainDecisionV2(_StrictFrozenModel):
    """System-gain decision bound to one five-config score matrix."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-grounded-contract-success-v2"] = (
        GCS_V2_POLICY_VERSION
    )
    policy_sha256: Literal[GCS_V2_POLICY_SHA256] = GCS_V2_POLICY_SHA256
    mapping_policy_sha256: Literal[GCS_V2_ALIAS_MAPPING_POLICY_SHA256] = (
        GCS_V2_ALIAS_MAPPING_POLICY_SHA256
    )
    sidecar_mapping_sha256: Sha256
    score_matrix_sha256: Sha256
    summary_sha256: Sha256
    bootstrap_sha256: Sha256
    decision: GCSSystemGainDecision
    decision_sha256: Sha256

    @model_validator(mode="after")
    def validate_v2_decision(self) -> Self:
        if self.decision_sha256 != _self_hash(self, "decision_sha256"):
            raise ValueError("GCS v2 five-config system-gain decision hash mismatch")
        return self


def evaluate_gcs_system_gain_five_config_v2(
    summary: GCSFiveConfigSummaryEnvelopeV2,
    bootstrap: GCSFiveConfigBootstrapEnvelopeV2,
) -> GCSFiveConfigSystemGainDecisionV2:
    """Apply the system gate while retaining exact physical-map provenance."""

    if not _strict_model_revalidates(
        summary, GCSFiveConfigSummaryEnvelopeV2
    ) or not _strict_model_revalidates(bootstrap, GCSFiveConfigBootstrapEnvelopeV2):
        raise PortfolioGCSError("GCS v2 summary/bootstrap contract is invalid")
    if (
        summary.sidecar_mapping_sha256 != bootstrap.sidecar_mapping_sha256
        or summary.score_matrix_sha256 != bootstrap.score_matrix_sha256
    ):
        raise PortfolioGCSError("GCS v2 summary/bootstrap provenance differs")
    decision = evaluate_gcs_system_gain(summary.summary, bootstrap.bootstrap)
    unsigned = {
        "schema_version": 2,
        "policy_version": GCS_V2_POLICY_VERSION,
        "policy_sha256": GCS_V2_POLICY_SHA256,
        "mapping_policy_sha256": GCS_V2_ALIAS_MAPPING_POLICY_SHA256,
        "sidecar_mapping_sha256": summary.sidecar_mapping_sha256,
        "score_matrix_sha256": summary.score_matrix_sha256,
        "summary_sha256": summary.summary_sha256,
        "bootstrap_sha256": bootstrap.bootstrap_sha256,
        "decision": decision,
    }
    return GCSFiveConfigSystemGainDecisionV2.model_validate(
        {**unsigned, "decision_sha256": _hash_json(_jsonable(unsigned))},
        strict=True,
    )
