"""Verified Portfolio treatment compilation and stage-lineage contracts.

The historical Portfolio runtime used :class:`StaticBankArtifact` as a carrier
for every skilled configuration.  That carrier is still useful to the
Assistant runtime, but it is not evidence that S1, S2, or S3 actually ran.
This module adds the missing provenance layer:

* a fail-closed bridge from an accepted Codex author draft to the semantic
  ``AuthoringInput`` consumed by the trusted compiler;
* deterministic, edit-scope-limited S2/S3 Bank mutation;
* receipts and a four-record chain manifest that distinguish real model
  stages from deterministic scaffolds.

The module is intentionally independent from launch/execution orchestration.
Those callers can adopt the verified handle without changing historical run
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from typing import Annotated, Literal, Mapping, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.codex_authoring import (
    CodexAuthoringContractError,
    CodexAuthoringInput,
)
from skillchain.codex_authoring_v5 import load_codex_authoring_input
from skillchain.evaluation.packets import VisibleCard
from skillchain.static_authoring import (
    AuthoringContractError,
    AuthoringDraftBundle,
    AuthoringInput,
    StaticBankArtifact,
    StrictSkillArtifact,
    _compile as _compile_static_bank,
    build_authoring_draft_bundle,
    load_authoring_packet,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)
from skillchain.tools.registry import ToolRegistry


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION = "portfolio-real-treatment-chain-v1"
PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION = (
    "portfolio-execution-artifact-alias-v1"
)
PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION = (
    "portfolio-runtime-compatibility-rebind-v1"
)
PORTFOLIO_STAGE_MUTATION_POLICY_VERSION = "portfolio-stage-mutation-v1"
PORTFOLIO_GATE_DECISION_RULE = (
    "causal-pareto-routing-quality-assistant-hard-error-skill-adherence-v3"
)
PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION = "portfolio-stage-gate-evidence-v4"
_LEGACY_PORTFOLIO_GATE_DECISION_RULE = (
    "pareto-routing-quality-hard-error-skill-adherence-v2"
)
_LEGACY_PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION = "portfolio-stage-gate-evidence-v3"
PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION = "portfolio-treatment-optimization25-smoke-v3"
PORTFOLIO_FINAL_RUBRIC_FILE_SHA256 = (
    "3270d3f12ca31a24c78e250437a2f159ac660f6df1fd9f215796720b99636e62"
)
PORTFOLIO_OPTIMIZATION_QUERY_IDS = tuple(f"dm-{index:03d}" for index in range(1, 26))
PORTFOLIO_STAGE_GATE_QUERY_COUNT = len(PORTFOLIO_OPTIMIZATION_QUERY_IDS)

PortfolioTreatmentConfig = Literal["llm_static", "s1", "s1s2", "full"]
PortfolioTreatmentStage = Literal[
    "static_author",
    "s1_creator",
    "s2_route_optimizer",
    "s3_body_refiner",
]
PortfolioGenerationKind = Literal[
    "codex_static_author",
    "codex_s1_creator",
    "llm_route_optimizer",
    "llm_body_refiner",
]
PortfolioEditScope = Literal["create_bank", "description_only", "body_only"]
PortfolioDecision = Literal["accepted", "rolled_back", "inconclusive"]
PortfolioGateMetricSource = Literal["candidate", "reused_parent"]
PortfolioGateReuseBasis = Literal[
    "direct_observation",
    "same_selected_route",
    "same_selected_skill_body_and_operators",
]
PortfolioGateConfig = Literal["s1", "s1s2", "full"]
PortfolioInputArtifactKind = Literal[
    "semantic_authoring_input",
    "codex_authoring_input",
    "pre_review_draft",
    "human_review",
    "engineer_review",
    "trajectory_bundle",
    "s1_feedback_bundle",
    "creator_packet",
    "failure_attribution",
    "route_examples",
    "feedback_batch",
    "body_attribution",
    "gate_report",
    "detailed_invocation_receipt",
    "prior_invocation_receipt",
    "prior_stage_gate",
    "rejected_edit_buffer",
]


@dataclass(frozen=True)
class PortfolioSkillOutputContract:
    """Frozen output contract parsed from one canonical Skill Body."""

    required_sections: tuple[str, ...]
    card_requirement: Literal["required", "forbidden", "optional"]
    card_fields: tuple[str, ...]


def parse_portfolio_skill_output_contract(
    body: str,
) -> PortfolioSkillOutputContract:
    """Read the compiler-owned output-contract block without model judgment."""

    heading = re.search(r"(?m)^## Output contract\s*$", body)
    if heading is None:
        raise PortfolioTreatmentError("Skill Body lacks its frozen output contract")
    remainder = body[heading.end() :]
    next_heading = re.search(r"(?m)^## ", remainder)
    block = remainder[: next_heading.start()] if next_heading else remainder
    values: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        for key in ("required_sections", "card_requirement", "card_fields"):
            prefix = f"- {key}:"
            if not stripped.startswith(prefix):
                continue
            if key in values:
                raise PortfolioTreatmentError(f"Skill output contract repeats {key}")
            values[key] = stripped[len(prefix) :].strip()
    if set(values) != {"required_sections", "card_requirement", "card_fields"}:
        raise PortfolioTreatmentError("Skill output contract is incomplete")
    required_sections = tuple(
        item.strip() for item in values["required_sections"].split(",")
    )
    card_fields = tuple(
        item.strip() for item in values["card_fields"].split(",") if item.strip()
    )
    card_requirement = values["card_requirement"]
    if (
        not required_sections
        or any(not item for item in required_sections)
        or len(set(required_sections)) != len(required_sections)
        or len(set(card_fields)) != len(card_fields)
        or card_requirement not in {"required", "forbidden", "optional"}
        or (card_requirement == "required" and not card_fields)
        or (card_requirement == "forbidden" and card_fields)
    ):
        raise PortfolioTreatmentError("Skill output contract is invalid")
    return PortfolioSkillOutputContract(
        required_sections=required_sections,
        card_requirement=card_requirement,
        card_fields=card_fields,
    )


def calculate_portfolio_skill_adherence(
    *,
    bank: StaticBankArtifact,
    selected_capability: str | None,
    skill_slug: str | None,
    response_text: str,
    visible_cards: tuple[VisibleCard, ...],
    hard_error: bool,
    require_skill_slug_match: bool = True,
) -> float:
    """Score explicit section labels and visible-card contract, independent of J."""

    if hard_error or selected_capability is None or skill_slug is None:
        return 0.0
    skill = next(
        (item for item in bank.skills if item.capability_id == selected_capability),
        None,
    )
    if skill is None or (require_skill_slug_match and skill.slug != skill_slug):
        return 0.0
    try:
        contract = parse_portfolio_skill_output_contract(skill.body)
    except PortfolioTreatmentError:
        return 0.0

    section_hits = sum(
        re.search(
            rf"(?im)^\s*(?:#{{1,6}}\s*)?{re.escape(label)}(?:\s*:|\s*$)",
            response_text,
        )
        is not None
        for label in contract.required_sections
    )
    section_score = section_hits / len(contract.required_sections)

    cards = tuple(visible_cards)
    if contract.card_requirement == "forbidden":
        card_score = 1.0 if not cards else 0.0
    elif not cards:
        card_score = 1.0 if contract.card_requirement == "optional" else 0.0
    elif not contract.card_fields:
        card_score = 1.0
    else:

        def card_has_field(card: VisibleCard, field: str) -> bool:
            if field == "title":
                return bool(card.title.strip())
            visible_fields = dict(card.fields)
            value = visible_fields.get(field)
            return isinstance(value, str) and bool(value.strip())

        field_checks = tuple(
            card_has_field(card, field)
            for card in cards
            for field in contract.card_fields
        )
        card_score = sum(field_checks) / len(field_checks)
    return (section_score + card_score) / 2.0


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_AUTHORING_INPUT_BYTES = 16 * 1024 * 1024
_MAX_DRAFT_BYTES = 4 * 1024 * 1024
_VERIFIED_REBIND_MARKER = object()
_VERIFIED_CHAIN_MARKER = object()

_CONFIG_ORDER: tuple[PortfolioTreatmentConfig, ...] = (
    "llm_static",
    "s1",
    "s1s2",
    "full",
)
_STAGE_BY_CONFIG: dict[PortfolioTreatmentConfig, PortfolioTreatmentStage] = {
    "llm_static": "static_author",
    "s1": "s1_creator",
    "s1s2": "s2_route_optimizer",
    "full": "s3_body_refiner",
}
_GENERATION_BY_CONFIG: dict[PortfolioTreatmentConfig, PortfolioGenerationKind] = {
    "llm_static": "codex_static_author",
    "s1": "codex_s1_creator",
    "s1s2": "llm_route_optimizer",
    "full": "llm_body_refiner",
}
_EDIT_SCOPE_BY_CONFIG: dict[PortfolioTreatmentConfig, PortfolioEditScope] = {
    "llm_static": "create_bank",
    "s1": "create_bank",
    "s1s2": "description_only",
    "full": "body_only",
}
_REQUIRED_INPUT_KIND: dict[PortfolioTreatmentConfig, PortfolioInputArtifactKind] = {
    "llm_static": "pre_review_draft",
    "s1": "creator_packet",
    "s1s2": "route_examples",
    "full": "body_attribution",
}
_PARENT_CONFIG_BY_CONFIG: dict[
    PortfolioTreatmentConfig,
    PortfolioTreatmentConfig,
] = {
    "s1": "llm_static",
    "s1s2": "s1",
    "full": "s1s2",
}


class PortfolioTreatmentError(ValueError):
    """A treatment artifact, transition, or compiler input is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonblank without edge whitespace")
    return value


def _relative_posix_path(value: str, label: str) -> str:
    _nonblank(value, label)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _self_hash(model: BaseModel, field_name: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(field_name, None)
    return sha256_bytes(canonical_json_bytes(payload))


def _validated_model[T: BaseModel](value: T, model_type: type[T], label: str) -> T:
    try:
        return model_type.model_validate(value.model_dump(mode="python"), strict=True)
    except (AttributeError, ValidationError) as error:
        raise PortfolioTreatmentError(f"{label} is invalid") from error


def _read_canonical_model[T: BaseModel](
    path: str | Path,
    *,
    expected_file_sha256: str,
    model_type: type[T],
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes, T]:
    if not _SHA256_RE.fullmatch(expected_file_sha256):
        raise PortfolioTreatmentError(f"{label} expected SHA-256 is invalid")
    resolved = Path(path).absolute()
    try:
        content = read_stable_regular_file(
            resolved,
            label=label,
            max_bytes=max_bytes,
        )
        if sha256_bytes(content) != expected_file_sha256:
            raise PortfolioTreatmentError(f"{label} file digest mismatch")
        raw = parse_canonical_json(content, label=label)
        if not isinstance(raw, dict):
            raise PortfolioTreatmentError(f"{label} must contain an object")
        value = model_type.model_validate(raw, strict=True)
    except PortfolioTreatmentError:
        raise
    except (ArtifactFormatError, OSError, ValidationError) as error:
        raise PortfolioTreatmentError(f"{label} is not canonical and valid") from error
    canonical_method = getattr(value, "canonical_bytes", None)
    if canonical_method is None or canonical_method() != content:
        raise PortfolioTreatmentError(f"{label} bytes are not canonical")
    return resolved, content, value


@dataclass(frozen=True)
class VerifiedCodexDraftRebind:
    """Opaque result of re-deriving the Codex-to-semantic draft binding."""

    codex_input_path: Path
    semantic_input_path: Path
    draft_path: Path
    codex_input_file_sha256: str
    semantic_input_file_sha256: str
    draft_file_sha256: str
    codex_input: CodexAuthoringInput
    semantic_input: AuthoringInput
    source_draft: AuthoringDraftBundle
    rebound_draft: AuthoringDraftBundle
    _source_contents: tuple[bytes, bytes, bytes] = field(repr=False)
    _marker: object = field(repr=False)


def load_verified_codex_draft_rebind(
    *,
    codex_input_path: str | Path,
    expected_codex_input_file_sha256: str,
    semantic_input_path: str | Path,
    expected_semantic_input_file_sha256: str,
    draft_path: str | Path,
    expected_draft_file_sha256: str,
) -> VerifiedCodexDraftRebind:
    """Verify and rebind a Codex draft to its exact semantic AuthoringInput.

    Codex output is deliberately bound to ``CodexAuthoringInput.input_sha256``
    while the trusted Bank compiler consumes the semantic
    ``AuthoringInput.input_sha256``.  Replacing one digest with the other is
    safe only after proving that the Codex input names this exact semantic
    packet and carries byte-identical taxonomy, TaskSpec, ToolSpec, compiler,
    public-source, and reference-skill semantics.
    """

    codex_expected = expected_codex_input_file_sha256
    semantic_expected = expected_semantic_input_file_sha256
    draft_expected = expected_draft_file_sha256
    if any(
        not _SHA256_RE.fullmatch(item)
        for item in (codex_expected, semantic_expected, draft_expected)
    ):
        raise PortfolioTreatmentError("Codex rebind expected SHA-256 is invalid")

    codex_path = Path(codex_input_path).absolute()
    semantic_path = Path(semantic_input_path).absolute()
    try:
        codex_bytes = read_stable_regular_file(
            codex_path,
            label="Codex authoring input",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        )
        if sha256_bytes(codex_bytes) != codex_expected:
            raise PortfolioTreatmentError("Codex authoring input file digest mismatch")
        codex_input = load_codex_authoring_input(
            codex_path,
            expected_file_sha256=codex_expected,
        )
        semantic_bytes = read_stable_regular_file(
            semantic_path,
            label="semantic authoring input",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        )
        if sha256_bytes(semantic_bytes) != semantic_expected:
            raise PortfolioTreatmentError(
                "semantic authoring input file digest mismatch"
            )
        semantic_input = load_authoring_packet(
            semantic_path,
            expected_file_sha256=semantic_expected,
        )
    except PortfolioTreatmentError:
        raise
    except (
        ArtifactFormatError,
        AuthoringContractError,
        CodexAuthoringContractError,
        OSError,
        ValueError,
    ) as error:
        raise PortfolioTreatmentError(
            "Codex or semantic authoring input verification failed"
        ) from error

    draft_resolved, draft_bytes, source_draft = _read_canonical_model(
        draft_path,
        expected_file_sha256=draft_expected,
        model_type=AuthoringDraftBundle,
        label="Codex pre-review draft",
        max_bytes=_MAX_DRAFT_BYTES,
    )
    if codex_bytes != codex_input.canonical_bytes():
        raise PortfolioTreatmentError("Codex authoring input is not canonical")
    if semantic_bytes != semantic_input.canonical_bytes():
        raise PortfolioTreatmentError("semantic authoring input is not canonical")
    if (
        codex_input.semantic_source_packet_file_sha256 != semantic_expected
        or source_draft.authoring_input_sha256 != codex_input.input_sha256
        or codex_input.taxonomy != semantic_input.taxonomy
        or codex_input.task_specification != semantic_input.task_specification
        or codex_input.tool_registry != semantic_input.tool_registry
        or codex_input.tool_registry_runtime_binding
        != semantic_input.tool_registry_runtime_binding
        or codex_input.tool_registry_runtime_sha256
        != semantic_input.tool_registry_runtime_sha256
        or codex_input.compiler != semantic_input.compiler
        or codex_input.public_sources != semantic_input.public_sources
        or codex_input.reference_skill_bundle != semantic_input.reference_skill_bundle
    ):
        raise PortfolioTreatmentError(
            "Codex input, semantic source, or draft binding mismatch"
        )

    rebound = build_authoring_draft_bundle(
        authoring_input_sha256=semantic_input.input_sha256,
        drafts=source_draft.drafts,
    )
    if (
        rebound.drafts != source_draft.drafts
        or rebound.authoring_input_sha256 != semantic_input.input_sha256
    ):
        raise PortfolioTreatmentError("Codex draft semantic rebind changed content")

    source_contents = (codex_bytes, semantic_bytes, draft_bytes)
    final_contents = (
        read_stable_regular_file(
            codex_path,
            label="Codex authoring input final recheck",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        ),
        read_stable_regular_file(
            semantic_path,
            label="semantic authoring input final recheck",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        ),
        read_stable_regular_file(
            draft_resolved,
            label="Codex pre-review draft final recheck",
            max_bytes=_MAX_DRAFT_BYTES,
        ),
    )
    if final_contents != source_contents:
        raise PortfolioTreatmentError(
            "Codex rebind source artifacts changed during verification"
        )
    return VerifiedCodexDraftRebind(
        codex_input_path=codex_path,
        semantic_input_path=semantic_path,
        draft_path=draft_resolved,
        codex_input_file_sha256=codex_expected,
        semantic_input_file_sha256=semantic_expected,
        draft_file_sha256=draft_expected,
        codex_input=codex_input,
        semantic_input=semantic_input,
        source_draft=source_draft,
        rebound_draft=rebound,
        _source_contents=source_contents,
        _marker=_VERIFIED_REBIND_MARKER,
    )


def require_verified_codex_draft_rebind(
    value: object,
) -> VerifiedCodexDraftRebind:
    """Reject forged, stale, or mutated rebind handles."""

    if (
        type(value) is not VerifiedCodexDraftRebind
        or value._marker is not _VERIFIED_REBIND_MARKER
    ):
        raise TypeError("Codex draft compilation requires a verified rebind")
    current = load_verified_codex_draft_rebind(
        codex_input_path=value.codex_input_path,
        expected_codex_input_file_sha256=value.codex_input_file_sha256,
        semantic_input_path=value.semantic_input_path,
        expected_semantic_input_file_sha256=value.semantic_input_file_sha256,
        draft_path=value.draft_path,
        expected_draft_file_sha256=value.draft_file_sha256,
    )
    held = (
        value.codex_input,
        value.semantic_input,
        value.source_draft,
        value.rebound_draft,
        value._source_contents,
    )
    observed = (
        current.codex_input,
        current.semantic_input,
        current.source_draft,
        current.rebound_draft,
        current._source_contents,
    )
    if held != observed:
        raise PortfolioTreatmentError("verified Codex draft rebind drifted")
    return value


def compile_verified_codex_llm_static_bank(
    rebind: VerifiedCodexDraftRebind,
    *,
    tool_registry_runtime_sha256: str,
) -> StaticBankArtifact:
    """Compile a real LLMStatic Bank through the trusted compiler.

    This is the public Portfolio compiler entry point.  It deliberately does
    not reproduce the historical ``prepare_portfolio_runtime._bank`` renderer.
    """

    verified = require_verified_codex_draft_rebind(rebind)
    if not _SHA256_RE.fullmatch(tool_registry_runtime_sha256):
        raise PortfolioTreatmentError("tool runtime SHA-256 is invalid")
    try:
        bank = _compile_static_bank(
            verified.semantic_input,
            verified.rebound_draft,
            baseline_kind="llm_static",
            tool_registry_runtime_sha256=tool_registry_runtime_sha256,
        )
    except AuthoringContractError as error:
        raise PortfolioTreatmentError("trusted LLMStatic compilation failed") from error
    if (
        bank.construction_identity_sha256 != verified.rebound_draft.bundle_sha256
        or bank.tool_registry_sha256
        != verified.semantic_input.tool_registry.identity_sha256
        or bank.tool_registry_runtime_sha256 != tool_registry_runtime_sha256
        or any(skill.parent_skill_sha256 is not None for skill in bank.skills)
    ):
        raise PortfolioTreatmentError(
            "trusted LLMStatic compiler returned an invalid lineage"
        )
    return bank


def compile_portfolio_contract_refreshed_llm_static_bank(
    semantic_input: AuthoringInput,
    refreshed_draft_bundle: AuthoringDraftBundle,
    *,
    tool_registry_runtime_sha256: str,
) -> StaticBankArtifact:
    """Compile a deterministic contract-only refresh of LLMStatic.

    Unlike the S1 entry point, this compiler does not imply a learned or
    model-authored improvement.  The caller is responsible for proving that
    only the capability whose frozen ToolSpec changed was replaced and that
    every other Skill artifact stayed byte-identical.
    """

    semantic = _validated_model(
        semantic_input,
        AuthoringInput,
        "contract-refreshed semantic authoring input",
    )
    draft = _validated_model(
        refreshed_draft_bundle,
        AuthoringDraftBundle,
        "contract-refreshed LLMStatic draft",
    )
    if draft.authoring_input_sha256 != semantic.input_sha256:
        raise PortfolioTreatmentError(
            "contract-refreshed draft is not bound to its semantic input"
        )
    if not _SHA256_RE.fullmatch(tool_registry_runtime_sha256):
        raise PortfolioTreatmentError("tool runtime SHA-256 is invalid")
    try:
        bank = _compile_static_bank(
            semantic,
            draft,
            baseline_kind="llm_static",
            tool_registry_runtime_sha256=tool_registry_runtime_sha256,
        )
    except AuthoringContractError as error:
        raise PortfolioTreatmentError(
            "trusted contract-only LLMStatic compilation failed"
        ) from error
    if (
        bank.construction_identity_sha256 != draft.bundle_sha256
        or bank.tool_registry_sha256 != semantic.tool_registry.identity_sha256
        or bank.tool_registry_runtime_sha256 != tool_registry_runtime_sha256
        or any(skill.parent_skill_sha256 is not None for skill in bank.skills)
    ):
        raise PortfolioTreatmentError(
            "contract-only LLMStatic compiler returned invalid lineage"
        )
    return bank


def compile_portfolio_s1_creator_bank(
    semantic_input: AuthoringInput,
    creator_draft_bundle: AuthoringDraftBundle,
    *,
    tool_registry_runtime_sha256: str,
) -> StaticBankArtifact:
    """Compile one real S1 Creator draft through the same trusted compiler.

    The Creator runner must first bind its normalized content to the common
    semantic ``AuthoringInput``.  This entry point refuses an implicit digest
    rewrite so that a caller cannot pass an unrelated model draft.
    """

    semantic = _validated_model(
        semantic_input,
        AuthoringInput,
        "S1 semantic authoring input",
    )
    draft = _validated_model(
        creator_draft_bundle,
        AuthoringDraftBundle,
        "S1 Creator draft",
    )
    if draft.authoring_input_sha256 != semantic.input_sha256:
        raise PortfolioTreatmentError(
            "S1 Creator draft is not bound to the common semantic input"
        )
    if not _SHA256_RE.fullmatch(tool_registry_runtime_sha256):
        raise PortfolioTreatmentError("tool runtime SHA-256 is invalid")
    try:
        bank = _compile_static_bank(
            semantic,
            draft,
            baseline_kind="llm_static",
            tool_registry_runtime_sha256=tool_registry_runtime_sha256,
        )
    except AuthoringContractError as error:
        raise PortfolioTreatmentError("trusted S1 compilation failed") from error
    if (
        len(bank.skills) != 6
        or len(bank.capability_map) != 6
        or bank.construction_identity_sha256 != draft.bundle_sha256
        or bank.tool_registry_sha256 != semantic.tool_registry.identity_sha256
        or bank.tool_registry_runtime_sha256 != tool_registry_runtime_sha256
        or any(
            skill.version != 1 or skill.parent_skill_sha256 is not None
            for skill in bank.skills
        )
    ):
        raise PortfolioTreatmentError(
            "trusted S1 compiler returned an invalid creation Bank"
        )
    return bank


class PortfolioArtifactBinding(_StrictFrozenModel):
    artifact_kind: PortfolioInputArtifactKind
    artifact_file: str
    artifact_file_sha256: Sha256
    artifact_content_sha256: Sha256

    @field_validator("artifact_file")
    @classmethod
    def validate_artifact_file(cls, value: str) -> str:
        return _relative_posix_path(value, "artifact_file")


class PortfolioSkillMutation(_StrictFrozenModel):
    capability_id: str
    parent_skill_sha256: Sha256
    description: str | None = None
    body: str | None = None

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        return _nonblank(value, "capability_id")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is not None:
            _nonblank(value, "description")
        return value

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or not value.endswith("\n") or value != value.lstrip()
        ):
            raise ValueError("body must be non-empty canonical Markdown")
        return value

    @model_validator(mode="after")
    def validate_exact_change(self) -> Self:
        if (self.description is None) == (self.body is None):
            raise ValueError(
                "a skill mutation must provide exactly one of description or body"
            )
        return self


class PortfolioStageMutation(_StrictFrozenModel):
    """A model-proposed S2 or S3 edit plan before gate disposition."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-stage-mutation"] = "portfolio-stage-mutation"
    policy_version: Literal[PORTFOLIO_STAGE_MUTATION_POLICY_VERSION] = (
        PORTFOLIO_STAGE_MUTATION_POLICY_VERSION
    )
    config: Literal["s1s2", "full"]
    stage: Literal["s2_route_optimizer", "s3_body_refiner"]
    edit_scope: Literal["description_only", "body_only"]
    parent_bank_sha256: Sha256
    implementation_id: str
    implementation_version: str
    implementation_file_sha256: Sha256
    input_artifacts: tuple[PortfolioArtifactBinding, ...] = Field(min_length=1)
    changes: tuple[PortfolioSkillMutation, ...] = Field(min_length=1)
    mutation_sha256: Sha256

    @field_validator("input_artifacts", "changes", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("implementation_id", "implementation_version")
    @classmethod
    def validate_implementation_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_mutation(self) -> Self:
        expected = (
            ("s2_route_optimizer", "description_only")
            if self.config == "s1s2"
            else ("s3_body_refiner", "body_only")
        )
        if (self.stage, self.edit_scope) != expected:
            raise ValueError("mutation config, stage, and edit scope mismatch")
        capabilities = tuple(item.capability_id for item in self.changes)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("mutation capabilities must be sorted and unique")
        if self.edit_scope == "description_only" and any(
            item.description is None or item.body is not None for item in self.changes
        ):
            raise ValueError("S2 mutation may change Description only")
        if self.edit_scope == "body_only" and any(
            item.body is None or item.description is not None for item in self.changes
        ):
            raise ValueError("S3 mutation may change Body only")
        if self.mutation_sha256 != _self_hash(self, "mutation_sha256"):
            raise ValueError("mutation_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_stage_mutation(
    *,
    config: Literal["s1s2", "full"],
    parent_bank_sha256: str,
    implementation_id: str,
    implementation_version: str,
    implementation_file_sha256: str,
    input_artifacts: tuple[PortfolioArtifactBinding, ...],
    changes: tuple[PortfolioSkillMutation, ...],
) -> PortfolioStageMutation:
    """Build one canonical, self-hashed S2 or S3 mutation."""

    stage = _STAGE_BY_CONFIG[config]
    edit_scope = _EDIT_SCOPE_BY_CONFIG[config]
    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-stage-mutation",
        "policy_version": PORTFOLIO_STAGE_MUTATION_POLICY_VERSION,
        "config": config,
        "stage": stage,
        "edit_scope": edit_scope,
        "parent_bank_sha256": parent_bank_sha256,
        "implementation_id": implementation_id,
        "implementation_version": implementation_version,
        "implementation_file_sha256": implementation_file_sha256,
        "input_artifacts": [item.model_dump(mode="json") for item in input_artifacts],
        "changes": [item.model_dump(mode="json") for item in changes],
    }
    try:
        return PortfolioStageMutation.model_validate(
            {
                **payload,
                "mutation_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentError("stage mutation is invalid") from error


def _bank_skills_by_capability(
    bank: StaticBankArtifact,
) -> dict[str, StrictSkillArtifact]:
    return {skill.capability_id: skill for skill in bank.skills}


def _validate_bank_carrier(bank: StaticBankArtifact, label: str) -> StaticBankArtifact:
    validated = _validated_model(bank, StaticBankArtifact, label)
    if validated.canonical_bytes() != bank.canonical_bytes():
        raise PortfolioTreatmentError(f"{label} canonical bytes drifted")
    return validated


def rebind_portfolio_bank_runtime(
    bank: StaticBankArtifact,
    registry: ToolRegistry,
) -> StaticBankArtifact:
    """Rebind an accepted Bank to one verified compatible tool runtime.

    This is a zero-authoring compatibility operation.  It preserves every
    Skill byte, construction identity, compiler identity and capability
    mapping.  Only the two registry commitments and the enclosing Bank hash
    may change, and every referenced operator must still exist.
    """

    source = _validate_bank_carrier(bank, "runtime-rebind source Bank")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("runtime-rebind registry must be a ToolRegistry")
    registered = {item.name for item in registry.specs()}
    missing = sorted(
        {
            operator
            for skill in source.skills
            for operator in skill.operators
            if operator not in registered
        }
    )
    if missing:
        raise PortfolioTreatmentError(
            "runtime rebind removed Bank operators: " + ", ".join(missing)
        )
    payload = source.model_dump(mode="json", exclude={"bank_sha256"})
    payload.update(
        {
            "tool_registry_sha256": registry.registry_sha256,
            "tool_registry_runtime_sha256": registry.registry_runtime_sha256,
        }
    )
    try:
        rebound = StaticBankArtifact.model_validate(
            {
                **payload,
                "bank_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:  # pragma: no cover - source/schema guard it
        raise PortfolioTreatmentError("runtime-rebound Bank is invalid") from error
    invariant_fields = (
        "schema_version",
        "baseline_kind",
        "construction_identity_sha256",
        "construction_identity_policy",
        "runtime_binding_policy",
        "compiler",
        "skills",
        "capability_map",
    )
    if any(
        getattr(rebound, field) != getattr(source, field) for field in invariant_fields
    ):
        raise PortfolioTreatmentError("runtime rebind changed Bank semantics")
    return rebound


def apply_portfolio_stage_mutation(
    parent_bank: StaticBankArtifact,
    mutation: PortfolioStageMutation,
) -> StaticBankArtifact:
    """Apply a strictly scoped S2/S3 mutation and establish per-Skill lineage."""

    parent = _validate_bank_carrier(parent_bank, "parent Bank")
    plan = _validated_model(mutation, PortfolioStageMutation, "stage mutation")
    if plan.parent_bank_sha256 != parent.bank_sha256:
        raise PortfolioTreatmentError("stage mutation parent Bank mismatch")
    changes = {item.capability_id: item for item in plan.changes}
    parent_skills = _bank_skills_by_capability(parent)
    if not set(changes) <= set(parent_skills):
        raise PortfolioTreatmentError("stage mutation names an unknown capability")

    output_skills: list[StrictSkillArtifact] = []
    for source in parent.skills:
        change = changes.get(source.capability_id)
        if change is None:
            output_skills.append(source)
            continue
        if change.parent_skill_sha256 != source.skill_sha256:
            raise PortfolioTreatmentError("stage mutation parent Skill mismatch")
        payload = {
            "slug": source.slug,
            "version": source.version + 1,
            "description": (
                change.description
                if plan.edit_scope == "description_only"
                else source.description
            ),
            "body": (change.body if plan.edit_scope == "body_only" else source.body),
            "static_refs": list(source.static_refs),
            "operators": list(source.operators),
            "capability_id": source.capability_id,
            "parent_skill_sha256": source.skill_sha256,
        }
        output_skills.append(
            StrictSkillArtifact.model_validate(
                {
                    **payload,
                    "skill_sha256": sha256_bytes(canonical_json_bytes(payload)),
                },
                strict=True,
            )
        )

    bank_payload = {
        "schema_version": parent.schema_version,
        "baseline_kind": parent.baseline_kind,
        "construction_identity_sha256": plan.mutation_sha256,
        "construction_identity_policy": parent.construction_identity_policy,
        "runtime_binding_policy": parent.runtime_binding_policy,
        "compiler": parent.compiler.model_dump(mode="json"),
        "tool_registry_sha256": parent.tool_registry_sha256,
        "tool_registry_runtime_sha256": parent.tool_registry_runtime_sha256,
        "skills": [
            item.model_dump(mode="json")
            for item in sorted(output_skills, key=lambda item: item.slug)
        ],
        "capability_map": [
            item.model_dump(mode="json") for item in parent.capability_map
        ],
    }
    try:
        candidate = StaticBankArtifact.model_validate(
            {
                **bank_payload,
                "bank_sha256": sha256_bytes(canonical_json_bytes(bank_payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentError(
            "stage mutation produced an invalid Bank"
        ) from error
    validate_portfolio_bank_transition(parent, candidate, plan.edit_scope)
    return candidate


def validate_portfolio_bank_transition(
    parent_bank: StaticBankArtifact,
    candidate_bank: StaticBankArtifact,
    edit_scope: Literal["description_only", "body_only"],
) -> None:
    """Assert the exact semantic field boundary for S2 or S3."""

    parent = _validate_bank_carrier(parent_bank, "parent Bank")
    candidate = _validate_bank_carrier(candidate_bank, "candidate Bank")
    invariant_bank_fields = (
        "schema_version",
        "baseline_kind",
        "runtime_binding_policy",
        "compiler",
        "tool_registry_sha256",
        "tool_registry_runtime_sha256",
    )
    if any(
        getattr(parent, field_name) != getattr(candidate, field_name)
        for field_name in invariant_bank_fields
    ):
        raise PortfolioTreatmentError("stage mutation changed a Bank invariant")

    parent_by_capability = _bank_skills_by_capability(parent)
    candidate_by_capability = _bank_skills_by_capability(candidate)
    if set(parent_by_capability) != set(candidate_by_capability):
        raise PortfolioTreatmentError("stage mutation changed capability coverage")
    parent_map = {item.capability_id: item.skill_slug for item in parent.capability_map}
    candidate_map = {
        item.capability_id: item.skill_slug for item in candidate.capability_map
    }
    if parent_map != candidate_map:
        raise PortfolioTreatmentError("stage mutation changed capability mapping")

    for capability_id, source in parent_by_capability.items():
        target = candidate_by_capability[capability_id]
        if target.skill_sha256 == source.skill_sha256:
            if target != source:
                raise PortfolioTreatmentError(
                    "unchanged stage Skill differs despite identical hash"
                )
            continue
        invariant_skill_fields = (
            "slug",
            "static_refs",
            "operators",
            "capability_id",
        )
        if any(
            getattr(source, field_name) != getattr(target, field_name)
            for field_name in invariant_skill_fields
        ):
            raise PortfolioTreatmentError("stage mutation changed a Skill invariant")
        if (
            target.version != source.version + 1
            or target.parent_skill_sha256 != source.skill_sha256
        ):
            raise PortfolioTreatmentError("stage mutation lineage is invalid")
        if edit_scope == "description_only":
            if target.body != source.body:
                raise PortfolioTreatmentError("S2 mutation changed Skill Body")
        elif edit_scope == "body_only":
            if target.description != source.description:
                raise PortfolioTreatmentError("S3 mutation changed Skill Description")
        else:  # pragma: no cover - Literal callers plus defensive runtime check
            raise PortfolioTreatmentError("unknown stage edit scope")


class PortfolioModelInvocationReceipt(_StrictFrozenModel):
    """Canonical evidence for exactly one platform-mediated Codex turn."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-model-invocation-receipt"] = (
        "portfolio-model-invocation-receipt"
    )
    policy_version: Literal[PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION] = (
        PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
    )
    stage: PortfolioTreatmentStage
    execution_surface: Literal["codex_exec"] = "codex_exec"
    evidence_tier: Literal["platform-mediated_non-provider-attested"] = (
        "platform-mediated_non-provider-attested"
    )
    requested_model: str
    effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"]
    thread_id: str
    process_exit_code: Literal[0]
    turn_completed_count: Literal[1]
    agent_message_count: Literal[1]
    visible_tool_activity: Literal[False]
    prompt_sha256: Sha256
    output_sha256: Sha256
    event_log_sha256: Sha256
    stderr_sha256: Sha256
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    model_call_count: Literal[1]
    invocation_receipt_sha256: Sha256

    @field_validator("requested_model", "thread_id")
    @classmethod
    def validate_invocation_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        if self.invocation_receipt_sha256 != _self_hash(
            self,
            "invocation_receipt_sha256",
        ):
            raise ValueError("invocation_receipt_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_model_invocation_receipt(
    *,
    stage: PortfolioTreatmentStage,
    requested_model: str,
    effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"],
    thread_id: str,
    prompt_sha256: str,
    output_sha256: str,
    event_log_sha256: str,
    stderr_sha256: str,
    input_tokens: int,
    output_tokens: int,
) -> PortfolioModelInvocationReceipt:
    """Build the only accepted one-call Codex invocation evidence shape."""

    payload = {
        "schema_version": 1,
        "artifact_kind": "portfolio-model-invocation-receipt",
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "stage": stage,
        "execution_surface": "codex_exec",
        "evidence_tier": "platform-mediated_non-provider-attested",
        "requested_model": requested_model,
        "effort": effort,
        "thread_id": thread_id,
        "process_exit_code": 0,
        "turn_completed_count": 1,
        "agent_message_count": 1,
        "visible_tool_activity": False,
        "prompt_sha256": prompt_sha256,
        "output_sha256": output_sha256,
        "event_log_sha256": event_log_sha256,
        "stderr_sha256": stderr_sha256,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_call_count": 1,
    }
    try:
        return PortfolioModelInvocationReceipt.model_validate(
            {
                **payload,
                "invocation_receipt_sha256": sha256_bytes(
                    canonical_json_bytes(payload)
                ),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentError("model invocation receipt is invalid") from error


class PortfolioStageGateRowProvenance(_StrictFrozenModel):
    """Per-query binding for a direct or causally reused gate metric row."""

    schema_version: Literal[1] = 1
    query_id: str
    metric_source: PortfolioGateMetricSource
    reuse_basis: PortfolioGateReuseBasis
    candidate_source_result_sha256: Sha256
    reused_parent_source_result_sha256: Sha256 | None
    selected_source_result_sha256: Sha256

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        return _nonblank(value, "query_id")

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.metric_source == "candidate":
            if (
                self.reuse_basis != "direct_observation"
                or self.reused_parent_source_result_sha256 is not None
                or self.selected_source_result_sha256
                != self.candidate_source_result_sha256
            ):
                raise ValueError("candidate gate row has inconsistent provenance")
        elif (
            self.reuse_basis == "direct_observation"
            or self.reused_parent_source_result_sha256 is None
            or self.selected_source_result_sha256
            != self.reused_parent_source_result_sha256
        ):
            raise ValueError("reused-parent gate row has inconsistent provenance")
        return self


class PortfolioStageGateResultSet(_StrictFrozenModel):
    """Canonical optimization25 metric/pairing projection from one smoke run."""

    schema_version: Literal[3, 4] = 4
    artifact_kind: Literal["portfolio-stage-gate-result-set"] = (
        "portfolio-stage-gate-result-set"
    )
    gate_evidence_policy_version: Literal[
        "portfolio-stage-gate-evidence-v3",
        "portfolio-stage-gate-evidence-v4",
    ] = PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
    source_policy_version: Literal[PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION] = (
        PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION
    )
    source_config: PortfolioTreatmentConfig
    bank_sha256: Sha256
    adherence_contract_bank_sha256: Sha256
    evaluation_query_ids: tuple[str, ...] = Field(
        min_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
        max_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    route_accuracy: float = Field(ge=0.0, le=1.0)
    mean_j: float = Field(ge=0.0, le=100.0)
    mean_skill_adherence: float = Field(ge=0.0, le=1.0)
    # This count is deliberately limited to Assistant/runtime failures.  Judge
    # failures are evaluator anomalies and cannot be treated as algorithm
    # regressions under evidence policy v4.
    hard_error_count: int = Field(
        ge=0,
        le=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    evaluator_anomaly_count: int | None = Field(
        default=None,
        ge=0,
        le=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    causal_reuse_unaffected: bool | None = None
    rubric_file_sha256: Sha256
    shared_route_artifact_sha256s: tuple[Sha256 | None, ...] = Field(
        min_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
        max_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    source_summary_file_sha256: Sha256
    source_summary_sha256: Sha256
    source_results_file_sha256: Sha256
    source_result_sha256s: tuple[Sha256, ...] = Field(
        min_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
        max_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    row_provenance: tuple[PortfolioStageGateRowProvenance, ...] | None = Field(
        default=None,
        min_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
        max_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    result_set_sha256: Sha256

    @field_validator(
        "evaluation_query_ids",
        "shared_route_artifact_sha256s",
        "source_result_sha256s",
        "row_provenance",
        mode="before",
    )
    @classmethod
    def coerce_result_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @model_validator(mode="after")
    def validate_result_set(self) -> Self:
        if self.schema_version == 3:
            if (
                self.gate_evidence_policy_version
                != _LEGACY_PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
                or self.evaluator_anomaly_count is not None
                or self.causal_reuse_unaffected is not None
                or self.row_provenance is not None
            ):
                raise ValueError("legacy gate result set has v4-only fields")
        elif (
            self.gate_evidence_policy_version != PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
            or self.evaluator_anomaly_count is None
            or self.causal_reuse_unaffected is None
            or self.row_provenance is None
        ):
            raise ValueError("v4 gate result set lacks causal evidence fields")
        if self.evaluation_query_ids != tuple(
            sorted(set(self.evaluation_query_ids))
        ) or any(
            not query_id or query_id != query_id.strip()
            for query_id in self.evaluation_query_ids
        ):
            raise ValueError(
                "gate result evaluation query IDs must be sorted, unique, and nonblank"
            )
        if self.evaluation_query_ids != PORTFOLIO_OPTIMIZATION_QUERY_IDS:
            raise ValueError("gate result must use the frozen optimization25 profile")
        if self.rubric_file_sha256 != PORTFOLIO_FINAL_RUBRIC_FILE_SHA256:
            raise ValueError("gate result does not bind the frozen Portfolio rubric")
        if len(set(self.source_result_sha256s)) != len(self.source_result_sha256s):
            raise ValueError("gate source result SHA-256 values must be unique")
        if self.row_provenance is not None:
            if tuple(item.query_id for item in self.row_provenance) != (
                self.evaluation_query_ids
            ):
                raise ValueError("gate row provenance is not query-aligned")
            if (
                tuple(
                    item.candidate_source_result_sha256 for item in self.row_provenance
                )
                != self.source_result_sha256s
            ):
                raise ValueError("gate row provenance is not source-result aligned")
            reused_count = sum(
                item.metric_source == "reused_parent" for item in self.row_provenance
            )
            if self.causal_reuse_unaffected != (reused_count > 0):
                raise ValueError("gate causal-reuse flag does not match row provenance")
        shared_routes = self.shared_route_artifact_sha256s
        if self.source_config in {"s1s2", "full"}:
            if any(item is None for item in shared_routes) or len(
                set(shared_routes)
            ) != len(shared_routes):
                raise ValueError(
                    "Stage-2 result set requires 25 unique shared-route artifacts"
                )
        elif any(item is not None for item in shared_routes):
            raise ValueError("pre-S2 result set must not claim shared-route artifacts")
        if self.result_set_sha256 != _self_hash(self, "result_set_sha256"):
            raise ValueError("result_set_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_dump(self, *args, **kwargs):
        payload = super().model_dump(*args, **kwargs)
        if self.schema_version == 3:
            payload.pop("evaluator_anomaly_count", None)
            payload.pop("causal_reuse_unaffected", None)
            payload.pop("row_provenance", None)
        return payload


def build_portfolio_stage_gate_result_set(
    *,
    source_config: PortfolioTreatmentConfig,
    bank_sha256: str,
    adherence_contract_bank_sha256: str,
    evaluation_query_ids: tuple[str, ...],
    route_accuracy: float,
    mean_j: float,
    mean_skill_adherence: float,
    hard_error_count: int,
    evaluator_anomaly_count: int = 0,
    causal_reuse_unaffected: bool = False,
    rubric_file_sha256: str,
    shared_route_artifact_sha256s: tuple[str | None, ...],
    source_summary_file_sha256: str,
    source_summary_sha256: str,
    source_results_file_sha256: str,
    source_result_sha256s: tuple[str, ...],
    row_provenance: tuple[PortfolioStageGateRowProvenance, ...] | None = None,
) -> PortfolioStageGateResultSet:
    """Build the typed projection consumed by a Portfolio stage gate."""

    if row_provenance is None:
        row_provenance = tuple(
            PortfolioStageGateRowProvenance(
                query_id=query_id,
                metric_source="candidate",
                reuse_basis="direct_observation",
                candidate_source_result_sha256=result_sha256,
                reused_parent_source_result_sha256=None,
                selected_source_result_sha256=result_sha256,
            )
            for query_id, result_sha256 in zip(
                evaluation_query_ids,
                source_result_sha256s,
                strict=True,
            )
        )
    payload = {
        "schema_version": 4,
        "artifact_kind": "portfolio-stage-gate-result-set",
        "gate_evidence_policy_version": PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION,
        "source_policy_version": PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION,
        "source_config": source_config,
        "bank_sha256": bank_sha256,
        "adherence_contract_bank_sha256": adherence_contract_bank_sha256,
        "evaluation_query_ids": list(evaluation_query_ids),
        "route_accuracy": route_accuracy,
        "mean_j": mean_j,
        "mean_skill_adherence": mean_skill_adherence,
        "hard_error_count": hard_error_count,
        "evaluator_anomaly_count": evaluator_anomaly_count,
        "causal_reuse_unaffected": causal_reuse_unaffected,
        "rubric_file_sha256": rubric_file_sha256,
        "shared_route_artifact_sha256s": list(shared_route_artifact_sha256s),
        "source_summary_file_sha256": source_summary_file_sha256,
        "source_summary_sha256": source_summary_sha256,
        "source_results_file_sha256": source_results_file_sha256,
        "source_result_sha256s": list(source_result_sha256s),
        "row_provenance": [item.model_dump(mode="json") for item in row_provenance],
    }
    try:
        return PortfolioStageGateResultSet.model_validate(
            {
                **payload,
                "result_set_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentError("stage gate result set is invalid") from error


class PortfolioStageGateReport(_StrictFrozenModel):
    """Deterministic disposition of one S1/S2/S3 candidate on the opt split."""

    schema_version: Literal[3, 4] = 4
    artifact_kind: Literal["portfolio-stage-gate-report"] = (
        "portfolio-stage-gate-report"
    )
    gate_evidence_policy_version: Literal[
        "portfolio-stage-gate-evidence-v3",
        "portfolio-stage-gate-evidence-v4",
    ] = PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
    policy_version: Literal[PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION] = (
        PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
    )
    config: PortfolioGateConfig
    stage: Literal["s1_creator", "s2_route_optimizer", "s3_body_refiner"]
    parent_bank_sha256: Sha256
    candidate_bank_sha256: Sha256
    adherence_contract_bank_sha256: Sha256
    evaluation_query_ids: tuple[str, ...] = Field(
        min_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
        max_length=PORTFOLIO_STAGE_GATE_QUERY_COUNT,
    )
    rubric_file_sha256: Sha256
    paired_shared_route_artifact_sha256s: tuple[Sha256, ...] | None
    parent_route_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_route_accuracy: float = Field(ge=0.0, le=1.0)
    parent_mean_j: float = Field(ge=0.0, le=100.0)
    candidate_mean_j: float = Field(ge=0.0, le=100.0)
    parent_mean_skill_adherence: float = Field(ge=0.0, le=1.0)
    candidate_mean_skill_adherence: float = Field(ge=0.0, le=1.0)
    parent_hard_error_count: int = Field(ge=0)
    candidate_hard_error_count: int = Field(ge=0)
    parent_evaluator_anomaly_count: int | None = Field(default=None, ge=0)
    candidate_evaluator_anomaly_count: int | None = Field(default=None, ge=0)
    causal_reuse_unaffected: bool | None = None
    decision_rule: Literal[
        "pareto-routing-quality-hard-error-skill-adherence-v2",
        "causal-pareto-routing-quality-assistant-hard-error-skill-adherence-v3",
    ] = PORTFOLIO_GATE_DECISION_RULE
    decision: PortfolioDecision
    parent_result_file: str
    parent_result_file_sha256: Sha256
    candidate_result_file: str
    candidate_result_file_sha256: Sha256
    gate_report_sha256: Sha256

    @field_validator(
        "evaluation_query_ids",
        "paired_shared_route_artifact_sha256s",
        mode="before",
    )
    @classmethod
    def coerce_query_ids(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("parent_result_file", "candidate_result_file")
    @classmethod
    def validate_result_file(cls, value: str, info) -> str:
        return _relative_posix_path(value, info.field_name)

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        if self.schema_version == 3:
            if (
                self.gate_evidence_policy_version
                != _LEGACY_PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
                or self.decision_rule != _LEGACY_PORTFOLIO_GATE_DECISION_RULE
                or self.parent_evaluator_anomaly_count is not None
                or self.candidate_evaluator_anomaly_count is not None
                or self.causal_reuse_unaffected is not None
            ):
                raise ValueError("legacy gate report has v4-only evidence")
        elif (
            self.gate_evidence_policy_version != PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION
            or self.decision_rule != PORTFOLIO_GATE_DECISION_RULE
            or self.parent_evaluator_anomaly_count is None
            or self.candidate_evaluator_anomaly_count is None
            or self.causal_reuse_unaffected is None
        ):
            raise ValueError("v4 gate report lacks causal evidence fields")
        if self.stage != _STAGE_BY_CONFIG[self.config]:
            raise ValueError("gate config and stage mismatch")
        if self.adherence_contract_bank_sha256 != self.parent_bank_sha256:
            raise ValueError("gate adherence contract must come from its parent Bank")
        if self.evaluation_query_ids != tuple(sorted(set(self.evaluation_query_ids))):
            raise ValueError("gate evaluation query IDs must be sorted and unique")
        if self.evaluation_query_ids != PORTFOLIO_OPTIMIZATION_QUERY_IDS:
            raise ValueError("gate must use the frozen optimization25 profile")
        if self.rubric_file_sha256 != PORTFOLIO_FINAL_RUBRIC_FILE_SHA256:
            raise ValueError("gate does not bind the frozen Portfolio rubric")
        if self.config == "full":
            paired_routes = self.paired_shared_route_artifact_sha256s
            if (
                paired_routes is None
                or len(paired_routes) != len(self.evaluation_query_ids)
                or len(set(paired_routes)) != len(paired_routes)
            ):
                raise ValueError(
                    "S3 gate requires one unique common-route artifact per query"
                )
        elif self.paired_shared_route_artifact_sha256s is not None:
            raise ValueError("only S3 gate may bind common-route artifacts")
        if self.config == "s1" and self.causal_reuse_unaffected is True:
            raise ValueError("causal unaffected-row reuse is only defined for S2/S3")
        if self.parent_hard_error_count > len(
            self.evaluation_query_ids
        ) or self.candidate_hard_error_count > len(self.evaluation_query_ids):
            raise ValueError("gate hard-error count exceeds evaluation rows")
        if self.schema_version == 4:
            assert self.parent_evaluator_anomaly_count is not None
            assert self.candidate_evaluator_anomaly_count is not None
            if self.parent_evaluator_anomaly_count > len(
                self.evaluation_query_ids
            ) or self.candidate_evaluator_anomaly_count > len(
                self.evaluation_query_ids
            ):
                raise ValueError("gate evaluator-anomaly count exceeds evaluation rows")
        nonregressed = (
            self.candidate_hard_error_count <= self.parent_hard_error_count
            and self.candidate_route_accuracy >= self.parent_route_accuracy
            and self.candidate_mean_j >= self.parent_mean_j
            and self.candidate_mean_skill_adherence >= self.parent_mean_skill_adherence
        )
        strictly_improved = (
            self.candidate_hard_error_count < self.parent_hard_error_count
            or self.candidate_route_accuracy > self.parent_route_accuracy
            or self.candidate_mean_j > self.parent_mean_j
            or self.candidate_mean_skill_adherence > self.parent_mean_skill_adherence
        )
        if self.schema_version == 4 and (
            self.parent_evaluator_anomaly_count > 0
            or self.candidate_evaluator_anomaly_count > 0
        ):
            expected_decision: PortfolioDecision = "inconclusive"
        else:
            expected_decision = (
                "accepted" if nonregressed and strictly_improved else "rolled_back"
            )
        if self.decision != expected_decision:
            raise ValueError("gate decision does not follow decision_rule")
        if self.gate_report_sha256 != _self_hash(
            self,
            "gate_report_sha256",
        ):
            raise ValueError("gate_report_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_dump(self, *args, **kwargs):
        payload = super().model_dump(*args, **kwargs)
        if self.schema_version == 3:
            payload.pop("parent_evaluator_anomaly_count", None)
            payload.pop("candidate_evaluator_anomaly_count", None)
            payload.pop("causal_reuse_unaffected", None)
        return payload


def verify_portfolio_stage_gate_evidence(
    report: PortfolioStageGateReport,
    parent_result: PortfolioStageGateResultSet,
    candidate_result: PortfolioStageGateResultSet,
) -> None:
    """Require one report to be the exact projection of two typed result sets."""

    gate = _validated_model(report, PortfolioStageGateReport, "stage gate report")
    parent = _validated_model(
        parent_result,
        PortfolioStageGateResultSet,
        "stage gate parent result set",
    )
    candidate = _validated_model(
        candidate_result,
        PortfolioStageGateResultSet,
        "stage gate candidate result set",
    )
    expected_configs: dict[
        PortfolioGateConfig,
        tuple[PortfolioTreatmentConfig, PortfolioTreatmentConfig],
    ] = {
        "s1": ("llm_static", "s1"),
        "s1s2": ("s1", "s1s2"),
        "full": ("s1s2", "full"),
    }
    expected_parent_config, expected_candidate_config = expected_configs[gate.config]
    if (
        parent.schema_version != gate.schema_version
        or candidate.schema_version != gate.schema_version
        or parent.source_config != expected_parent_config
        or candidate.source_config != expected_candidate_config
        or parent.bank_sha256 != gate.parent_bank_sha256
        or candidate.bank_sha256 != gate.candidate_bank_sha256
        or parent.adherence_contract_bank_sha256 != gate.adherence_contract_bank_sha256
        or candidate.adherence_contract_bank_sha256
        != gate.adherence_contract_bank_sha256
        or parent.evaluation_query_ids != gate.evaluation_query_ids
        or candidate.evaluation_query_ids != gate.evaluation_query_ids
        or parent.route_accuracy != gate.parent_route_accuracy
        or candidate.route_accuracy != gate.candidate_route_accuracy
        or parent.mean_j != gate.parent_mean_j
        or candidate.mean_j != gate.candidate_mean_j
        or parent.mean_skill_adherence != gate.parent_mean_skill_adherence
        or candidate.mean_skill_adherence != gate.candidate_mean_skill_adherence
        or parent.hard_error_count != gate.parent_hard_error_count
        or candidate.hard_error_count != gate.candidate_hard_error_count
        or parent.evaluator_anomaly_count != gate.parent_evaluator_anomaly_count
        or candidate.evaluator_anomaly_count != gate.candidate_evaluator_anomaly_count
        or candidate.causal_reuse_unaffected != gate.causal_reuse_unaffected
        or parent.rubric_file_sha256 != gate.rubric_file_sha256
        or candidate.rubric_file_sha256 != gate.rubric_file_sha256
        or gate.parent_result_file_sha256 != sha256_bytes(parent.canonical_bytes())
        or gate.candidate_result_file_sha256
        != sha256_bytes(candidate.canonical_bytes())
    ):
        raise PortfolioTreatmentError(
            "stage gate report differs from its typed parent/candidate results"
        )
    if gate.schema_version == 4:
        assert parent.row_provenance is not None
        assert candidate.row_provenance is not None
        for parent_row, candidate_row in zip(
            parent.row_provenance,
            candidate.row_provenance,
            strict=True,
        ):
            if parent_row.metric_source != "candidate":
                raise PortfolioTreatmentError(
                    "gate parent result must use direct candidate-source rows"
                )
            if candidate_row.metric_source == "reused_parent" and (
                candidate_row.reused_parent_source_result_sha256
                != parent_row.selected_source_result_sha256
            ):
                raise PortfolioTreatmentError(
                    "gate reused-parent row does not bind its paired parent result"
                )
            expected_reuse_basis = {
                "s1s2": "same_selected_route",
                "full": "same_selected_skill_body_and_operators",
            }.get(gate.config)
            if candidate_row.metric_source == "reused_parent" and (
                candidate_row.reuse_basis != expected_reuse_basis
            ):
                raise PortfolioTreatmentError(
                    "gate reused-parent row has the wrong stage causal basis"
                )
    if gate.config == "full" and (
        parent.shared_route_artifact_sha256s != candidate.shared_route_artifact_sha256s
        or gate.paired_shared_route_artifact_sha256s
        != parent.shared_route_artifact_sha256s
    ):
        raise PortfolioTreatmentError(
            "S3 gate parent/candidate do not share exact per-query routes"
        )


def build_portfolio_stage_gate_report(
    *,
    config: PortfolioGateConfig,
    parent_bank_sha256: str,
    candidate_bank_sha256: str,
    adherence_contract_bank_sha256: str,
    evaluation_query_ids: tuple[str, ...],
    rubric_file_sha256: str,
    paired_shared_route_artifact_sha256s: tuple[str, ...] | None,
    parent_route_accuracy: float,
    candidate_route_accuracy: float,
    parent_mean_j: float,
    candidate_mean_j: float,
    parent_mean_skill_adherence: float,
    candidate_mean_skill_adherence: float,
    parent_hard_error_count: int,
    candidate_hard_error_count: int,
    decision: PortfolioDecision,
    parent_result_file: str,
    parent_result_file_sha256: str,
    candidate_result_file: str,
    candidate_result_file_sha256: str,
    parent_evaluator_anomaly_count: int = 0,
    candidate_evaluator_anomaly_count: int = 0,
    causal_reuse_unaffected: bool = False,
) -> PortfolioStageGateReport:
    """Build one canonical stage gate under the frozen Pareto rule."""

    payload = {
        "schema_version": 4,
        "artifact_kind": "portfolio-stage-gate-report",
        "gate_evidence_policy_version": PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION,
        "policy_version": PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION,
        "config": config,
        "stage": _STAGE_BY_CONFIG[config],
        "parent_bank_sha256": parent_bank_sha256,
        "candidate_bank_sha256": candidate_bank_sha256,
        "adherence_contract_bank_sha256": adherence_contract_bank_sha256,
        "evaluation_query_ids": list(evaluation_query_ids),
        "rubric_file_sha256": rubric_file_sha256,
        "paired_shared_route_artifact_sha256s": (
            list(paired_shared_route_artifact_sha256s)
            if paired_shared_route_artifact_sha256s is not None
            else None
        ),
        "parent_route_accuracy": parent_route_accuracy,
        "candidate_route_accuracy": candidate_route_accuracy,
        "parent_mean_j": parent_mean_j,
        "candidate_mean_j": candidate_mean_j,
        "parent_mean_skill_adherence": parent_mean_skill_adherence,
        "candidate_mean_skill_adherence": candidate_mean_skill_adherence,
        "parent_hard_error_count": parent_hard_error_count,
        "candidate_hard_error_count": candidate_hard_error_count,
        "parent_evaluator_anomaly_count": parent_evaluator_anomaly_count,
        "candidate_evaluator_anomaly_count": candidate_evaluator_anomaly_count,
        "causal_reuse_unaffected": causal_reuse_unaffected,
        "decision_rule": PORTFOLIO_GATE_DECISION_RULE,
        "decision": decision,
        "parent_result_file": parent_result_file,
        "parent_result_file_sha256": parent_result_file_sha256,
        "candidate_result_file": candidate_result_file,
        "candidate_result_file_sha256": candidate_result_file_sha256,
    }
    try:
        return PortfolioStageGateReport.model_validate(
            {
                **payload,
                "gate_report_sha256": sha256_bytes(canonical_json_bytes(payload)),
            },
            strict=True,
        )
    except ValidationError as error:
        raise PortfolioTreatmentError("stage gate report is invalid") from error


class PortfolioTreatmentReceipt(_StrictFrozenModel):
    """One final stage disposition with complete upstream model evidence."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-treatment-receipt"] = (
        "portfolio-treatment-receipt"
    )
    policy_version: Literal[PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION] = (
        PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
    )
    config: PortfolioTreatmentConfig
    stage: PortfolioTreatmentStage
    generation_kind: PortfolioGenerationKind
    edit_scope: PortfolioEditScope
    common_authoring_input_sha256: Sha256
    implementation_id: str
    implementation_version: str
    implementation_file_sha256: Sha256
    input_artifacts: tuple[PortfolioArtifactBinding, ...] = Field(min_length=1)
    parent_bank_sha256: Sha256 | None
    mutation_sha256: Sha256 | None
    mutation_file: str | None
    mutation_file_sha256: Sha256 | None
    candidate_bank_sha256: Sha256
    candidate_bank_file: str
    candidate_bank_file_sha256: Sha256
    output_bank_sha256: Sha256
    output_bank_file_sha256: Sha256
    model_call_count: Literal[1]
    invocation_receipt_file: str
    invocation_receipt_file_sha256: Sha256
    raw_model_output_file: str
    raw_model_output_file_sha256: Sha256
    gate_report_file: str
    gate_report_sha256: Sha256
    gate_report_file_sha256: Sha256
    decision: PortfolioDecision
    receipt_sha256: Sha256

    @field_validator("input_artifacts", mode="before")
    @classmethod
    def coerce_inputs(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("implementation_id", "implementation_version")
    @classmethod
    def validate_implementation_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "candidate_bank_file",
        "invocation_receipt_file",
        "raw_model_output_file",
        "gate_report_file",
    )
    @classmethod
    def validate_receipt_file(cls, value: str, info) -> str:
        return _relative_posix_path(value, info.field_name)

    @field_validator("mutation_file")
    @classmethod
    def validate_mutation_file(cls, value: str | None) -> str | None:
        if value is not None:
            return _relative_posix_path(value, "mutation_file")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if (
            self.stage != _STAGE_BY_CONFIG[self.config]
            or self.generation_kind != _GENERATION_BY_CONFIG[self.config]
            or self.edit_scope != _EDIT_SCOPE_BY_CONFIG[self.config]
        ):
            raise ValueError(
                "treatment config, stage, generation kind, or edit scope mismatch"
            )
        kinds = {item.artifact_kind for item in self.input_artifacts}
        if _REQUIRED_INPUT_KIND[self.config] not in kinds:
            raise ValueError("treatment receipt lacks its required stage input")
        if (
            self.config == "s1"
            and not {
                "trajectory_bundle",
                "creator_packet",
                "engineer_review",
            }
            <= kinds
        ):
            raise ValueError("S1 receipt lacks Creator/Engineer-Gate inputs")
        if self.config == "llm_static":
            if (
                self.parent_bank_sha256 is not None
                or self.mutation_sha256 is not None
                or self.mutation_file is not None
                or self.mutation_file_sha256 is not None
                or self.decision != "accepted"
            ):
                raise ValueError("static author must be an accepted creation")
        elif self.config == "s1":
            if (
                self.parent_bank_sha256 is None
                or self.mutation_sha256 is not None
                or self.mutation_file is not None
                or self.mutation_file_sha256 is not None
            ):
                raise ValueError("S1 must bind its baseline parent without a mutation")
        elif (
            self.parent_bank_sha256 is None
            or self.mutation_sha256 is None
            or self.mutation_file is None
            or self.mutation_file_sha256 is None
        ):
            raise ValueError("S2/S3 receipt lacks parent or mutation identity")
        if self.decision == "accepted":
            if (
                self.output_bank_sha256 != self.candidate_bank_sha256
                or self.output_bank_file_sha256 != self.candidate_bank_file_sha256
            ):
                raise ValueError("accepted treatment output is not its candidate")
        elif (
            self.parent_bank_sha256 is None
            or self.output_bank_sha256 != self.parent_bank_sha256
        ):
            raise ValueError("rolled-back treatment output is not its parent")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("receipt_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioExecutionArtifactAlias(_StrictFrozenModel):
    """Fail-closed instruction to reuse one accepted parent execution.

    Candidate-generation evidence remains part of the treatment chain, but a
    rejected stage is not a distinct matrix treatment.  This record makes the
    distinction executable: the target configuration must project the source
    configuration's Assistant and evaluator artifacts and is not allowed to
    reserve or perform another provider call.
    """

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-execution-artifact-alias"] = (
        "portfolio-execution-artifact-alias"
    )
    policy_version: Literal[PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION] = (
        PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION
    )
    target_config: PortfolioTreatmentConfig
    source_config: PortfolioTreatmentConfig
    stage_decision: Literal["rolled_back", "inconclusive"]
    source_bank_sha256: Sha256
    target_bank_sha256: Sha256
    source_bank_file_sha256: Sha256
    target_bank_file_sha256: Sha256
    reuse_scope: Literal["assistant_and_evaluator_query_artifacts"] = (
        "assistant_and_evaluator_query_artifacts"
    )
    provider_model_call_count: Literal[0] = 0
    rejected_candidate_use: Literal["diagnostic_only"] = "diagnostic_only"
    alias_sha256: Sha256

    @model_validator(mode="after")
    def validate_alias(self) -> Self:
        if (
            _PARENT_CONFIG_BY_CONFIG.get(self.target_config) != self.source_config
            or self.source_bank_sha256 != self.target_bank_sha256
            or self.source_bank_file_sha256 != self.target_bank_file_sha256
        ):
            raise ValueError(
                "execution artifact alias is not an exact parent-output reuse"
            )
        if self.alias_sha256 != _self_hash(self, "alias_sha256"):
            raise ValueError("alias_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioRuntimeFileBinding(_StrictFrozenModel):
    """One byte-exact file preserved inside the accepted parent runtime."""

    relative_path: str
    file_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _relative_posix_path(value, "relative_path")


class PortfolioRuntimeRebindBankBinding(_StrictFrozenModel):
    """One accepted Bank carrier before and after a zero-authoring rebind."""

    config: PortfolioTreatmentConfig
    role: Literal["output", "candidate"]
    source_bank_sha256: Sha256
    source_bank_file_sha256: Sha256
    rebound_bank_file: str
    rebound_bank_sha256: Sha256
    rebound_bank_file_sha256: Sha256

    @field_validator("rebound_bank_file")
    @classmethod
    def validate_rebound_bank_file(cls, value: str) -> str:
        return _relative_posix_path(value, "rebound_bank_file")


class PortfolioRuntimeCompatibilityRebind(_StrictFrozenModel):
    """Typed lineage for moving accepted Skills to a compatible tool runtime.

    This is deliberately not an S1/S2/S3 treatment.  It preserves the parent
    treatment chain and its model/gate evidence byte-for-byte and permits only
    the two registry commitments and enclosing Bank hash to change.
    """

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-runtime-compatibility-rebind"] = (
        "portfolio-runtime-compatibility-rebind"
    )
    policy_version: Literal[PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION] = (
        PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION
    )
    source_runtime_dir: Literal["parent-runtime"] = "parent-runtime"
    source_runtime_lock_file_sha256: Sha256
    source_runtime_lock_sha256: Sha256
    source_treatment_manifest_file_sha256: Sha256
    source_treatment_chain_sha256: Sha256
    source_optimization_query_count: int = Field(gt=0)
    source_evaluation_query_count: int = Field(gt=0)
    source_optimization_query_ids_sha256: Sha256
    source_evaluation_query_ids_sha256: Sha256
    source_runtime_files: tuple[PortfolioRuntimeFileBinding, ...] = Field(min_length=1)
    core_runtime_sources_dir: Literal["core-runtime-sources"] = "core-runtime-sources"
    core_runtime_sources_receipt_file_sha256: Sha256
    core_runtime_sources_receipt_sha256: Sha256
    target_tool_registry_sha256: Sha256
    target_tool_registry_runtime_sha256: Sha256
    target_runtime_data_sha256: Sha256
    target_source_sha256s: tuple[Sha256, ...] = Field(min_length=6)
    semantic_fields_preserved: tuple[str, ...]
    compatibility_changes: tuple[str, ...]
    algorithm_decisions_preserved: Literal[True] = True
    model_evidence_bytes_preserved: Literal[True] = True
    provider_model_call_count: Literal[0] = 0
    bank_bindings: tuple[PortfolioRuntimeRebindBankBinding, ...] = Field(
        min_length=8,
        max_length=8,
    )
    rebind_sha256: Sha256

    @field_validator(
        "source_runtime_files",
        "target_source_sha256s",
        "semantic_fields_preserved",
        "compatibility_changes",
        "bank_bindings",
        mode="before",
    )
    @classmethod
    def coerce_rebind_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @model_validator(mode="after")
    def validate_rebind(self) -> Self:
        expected_semantic_fields = (
            "schema_version",
            "baseline_kind",
            "construction_identity_sha256",
            "construction_identity_policy",
            "runtime_binding_policy",
            "compiler",
            "skills",
            "capability_map",
        )
        if self.semantic_fields_preserved != expected_semantic_fields:
            raise ValueError("runtime rebind semantic field allowlist differs")
        if self.compatibility_changes != (
            "tool_registry_sha256",
            "tool_registry_runtime_sha256",
            "bank_sha256",
        ):
            raise ValueError("runtime rebind compatibility change set differs")
        file_paths = tuple(item.relative_path for item in self.source_runtime_files)
        if file_paths != tuple(sorted(set(file_paths))):
            raise ValueError("source runtime files must be sorted and unique")
        expected_roles = tuple(
            (config, role)
            for config in _CONFIG_ORDER
            for role in ("output", "candidate")
        )
        actual_roles = tuple((item.config, item.role) for item in self.bank_bindings)
        if actual_roles != expected_roles:
            raise ValueError("runtime rebind Bank bindings are incomplete or unordered")
        rebound_files = tuple(item.rebound_bank_file for item in self.bank_bindings)
        if len(set(rebound_files)) != len(rebound_files):
            raise ValueError("runtime rebind Bank files must be role-specific")
        if self.rebind_sha256 != _self_hash(self, "rebind_sha256"):
            raise ValueError("rebind_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioBankBinding(_StrictFrozenModel):
    config: PortfolioTreatmentConfig
    bank_file: str
    bank_sha256: Sha256
    bank_file_sha256: Sha256

    @field_validator("bank_file")
    @classmethod
    def validate_bank_file(cls, value: str) -> str:
        return _relative_posix_path(value, "bank_file")


class PortfolioTreatmentChainManifest(_StrictFrozenModel):
    """Matrix-ready commitment to the exact four skilled configurations."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-treatment-chain-manifest"] = (
        "portfolio-treatment-chain-manifest"
    )
    policy_version: Literal[PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION] = (
        PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION
    )
    status: Literal["candidate", "development", "ready_for_matrix"]
    track: Literal["portfolio"] = "portfolio"
    formal_eligible: Literal[False] = False
    optimization_query_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_query_ids: tuple[str, ...] = Field(min_length=1)
    optimization_leakage_group_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_leakage_group_ids: tuple[str, ...] = Field(min_length=1)
    records: tuple[PortfolioTreatmentReceipt, ...] = Field(
        min_length=4,
        max_length=4,
    )
    banks: tuple[PortfolioBankBinding, ...] = Field(min_length=4, max_length=4)
    chain_sha256: Sha256

    @field_validator(
        "optimization_query_ids",
        "evaluation_query_ids",
        "optimization_leakage_group_ids",
        "evaluation_leakage_group_ids",
        "records",
        "banks",
        mode="before",
    )
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        for label, values in (
            ("optimization_query_ids", self.optimization_query_ids),
            ("evaluation_query_ids", self.evaluation_query_ids),
            (
                "optimization_leakage_group_ids",
                self.optimization_leakage_group_ids,
            ),
            ("evaluation_leakage_group_ids", self.evaluation_leakage_group_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
        if set(self.optimization_query_ids) & set(self.evaluation_query_ids):
            raise ValueError("optimization and evaluation query IDs overlap")
        if self.optimization_query_ids != PORTFOLIO_OPTIMIZATION_QUERY_IDS:
            raise ValueError(
                "treatment manifest must bind the frozen optimization25 profile"
            )
        if set(self.optimization_leakage_group_ids) & set(
            self.evaluation_leakage_group_ids
        ):
            raise ValueError("optimization and evaluation leakage groups overlap")
        record_configs = tuple(item.config for item in self.records)
        bank_configs = tuple(item.config for item in self.banks)
        if record_configs != _CONFIG_ORDER or bank_configs != _CONFIG_ORDER:
            raise ValueError("treatment records and Banks must use exact config order")
        common_inputs = {
            item.common_authoring_input_sha256 for item in self.records[:2]
        }
        if len(common_inputs) != 1:
            raise ValueError("LLMStatic and S1 common authoring inputs differ")
        for record, bank in zip(self.records, self.banks, strict=True):
            if (
                record.output_bank_sha256 != bank.bank_sha256
                or record.output_bank_file_sha256 != bank.bank_file_sha256
            ):
                raise ValueError("treatment receipt and Bank binding mismatch")
        if self.records[1].parent_bank_sha256 != self.records[0].output_bank_sha256:
            raise ValueError("S1 gate baseline is not the LLMStatic output Bank")
        if self.records[2].parent_bank_sha256 != self.records[1].output_bank_sha256:
            raise ValueError("S2 parent is not the S1 output Bank")
        if self.records[3].parent_bank_sha256 != self.records[2].output_bank_sha256:
            raise ValueError("S3 parent is not the S1+S2 output Bank")
        records_by_config = {item.config: item for item in self.records}
        for target_config, source_config in _PARENT_CONFIG_BY_CONFIG.items():
            target = records_by_config[target_config]
            source = records_by_config[source_config]
            if target.decision != "accepted" and (
                target.output_bank_sha256 != source.output_bank_sha256
                or target.output_bank_file_sha256 != source.output_bank_file_sha256
            ):
                raise ValueError(
                    f"{target_config} rollback is not an exact parent artifact alias"
                )
        if self.chain_sha256 != _self_hash(self, "chain_sha256"):
            raise ValueError("chain_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_portfolio_execution_artifact_aliases(
    manifest: PortfolioTreatmentChainManifest,
) -> tuple[PortfolioExecutionArtifactAlias, ...]:
    """Derive the only matrix-valid disposition for rejected stage candidates."""

    value = _validated_model(
        manifest,
        PortfolioTreatmentChainManifest,
        "treatment chain manifest",
    )
    records = {item.config: item for item in value.records}
    aliases: list[PortfolioExecutionArtifactAlias] = []
    for target_config in ("s1", "s1s2", "full"):
        target = records[target_config]
        if target.decision == "accepted":
            continue
        source_config = _PARENT_CONFIG_BY_CONFIG[target_config]
        source = records[source_config]
        payload = {
            "schema_version": 1,
            "artifact_kind": "portfolio-execution-artifact-alias",
            "policy_version": PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
            "target_config": target_config,
            "source_config": source_config,
            "stage_decision": target.decision,
            "source_bank_sha256": source.output_bank_sha256,
            "target_bank_sha256": target.output_bank_sha256,
            "source_bank_file_sha256": source.output_bank_file_sha256,
            "target_bank_file_sha256": target.output_bank_file_sha256,
            "reuse_scope": "assistant_and_evaluator_query_artifacts",
            "provider_model_call_count": 0,
            "rejected_candidate_use": "diagnostic_only",
        }
        try:
            aliases.append(
                PortfolioExecutionArtifactAlias.model_validate(
                    {
                        **payload,
                        "alias_sha256": sha256_bytes(canonical_json_bytes(payload)),
                    },
                    strict=True,
                )
            )
        except ValidationError as error:  # pragma: no cover - manifest guards it
            raise PortfolioTreatmentError(
                f"{target_config} execution artifact alias is invalid"
            ) from error
    return tuple(aliases)


@dataclass(frozen=True)
class VerifiedPortfolioTreatmentChain:
    manifest: PortfolioTreatmentChainManifest
    output_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact]
    candidate_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact]
    execution_artifact_aliases: tuple[PortfolioExecutionArtifactAlias, ...]
    mutations: Mapping[PortfolioTreatmentConfig, PortfolioStageMutation]
    invocation_receipts: Mapping[
        PortfolioTreatmentConfig,
        PortfolioModelInvocationReceipt,
    ]
    gate_reports: Mapping[PortfolioGateConfig, PortfolioStageGateReport]
    _marker: object = field(repr=False)
    compatibility_rebind: PortfolioRuntimeCompatibilityRebind | None = None
    source_chain: "VerifiedPortfolioTreatmentChain | None" = None


def _verify_bank_runtime_rebind(
    source: StaticBankArtifact,
    rebound: StaticBankArtifact,
    *,
    target_tool_registry_sha256: str,
    target_tool_registry_runtime_sha256: str,
) -> None:
    source_value = _validate_bank_carrier(source, "runtime-rebind source Bank")
    rebound_value = _validate_bank_carrier(rebound, "runtime-rebind target Bank")
    expected_payload = source_value.model_dump(mode="json", exclude={"bank_sha256"})
    expected_payload.update(
        {
            "tool_registry_sha256": target_tool_registry_sha256,
            "tool_registry_runtime_sha256": target_tool_registry_runtime_sha256,
        }
    )
    expected = StaticBankArtifact.model_validate(
        {
            **expected_payload,
            "bank_sha256": sha256_bytes(canonical_json_bytes(expected_payload)),
        },
        strict=True,
    )
    if rebound_value != expected:
        raise PortfolioTreatmentError(
            "runtime rebind changed fields outside the compatibility allowlist"
        )


def verify_portfolio_runtime_compatibility_rebind_chain(
    *,
    rebind: PortfolioRuntimeCompatibilityRebind,
    source_chain: "VerifiedPortfolioTreatmentChain",
    output_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
    candidate_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
) -> VerifiedPortfolioTreatmentChain:
    """Verify a zero-authoring execution projection of an accepted chain."""

    receipt = _validated_model(
        rebind,
        PortfolioRuntimeCompatibilityRebind,
        "runtime compatibility rebind",
    )
    parent = require_verified_portfolio_treatment_chain(source_chain)
    if parent.compatibility_rebind is not None or parent.source_chain is not None:
        raise PortfolioTreatmentError(
            "nested runtime compatibility rebind is forbidden"
        )
    if (
        len(parent.manifest.optimization_query_ids)
        != receipt.source_optimization_query_count
        or len(parent.manifest.evaluation_query_ids)
        != receipt.source_evaluation_query_count
        or sha256_bytes(
            canonical_json_bytes(list(parent.manifest.optimization_query_ids))
        )
        != receipt.source_optimization_query_ids_sha256
        or sha256_bytes(
            canonical_json_bytes(list(parent.manifest.evaluation_query_ids))
        )
        != receipt.source_evaluation_query_ids_sha256
    ):
        raise PortfolioTreatmentError(
            "runtime rebind changed the accepted treatment split lineage"
        )
    if set(output_banks) != set(_CONFIG_ORDER) or set(candidate_banks) != set(
        _CONFIG_ORDER
    ):
        raise PortfolioTreatmentError("runtime rebind Bank set is incomplete")
    bindings = {(item.config, item.role): item for item in receipt.bank_bindings}
    outputs: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    candidates: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    for config in _CONFIG_ORDER:
        for role, source_map, rebound_map in (
            ("output", parent.output_banks, output_banks),
            ("candidate", parent.candidate_banks, candidate_banks),
        ):
            binding = bindings[(config, role)]
            source = source_map[config]
            rebound = rebound_map[config]
            if (
                binding.source_bank_sha256 != source.bank_sha256
                or binding.source_bank_file_sha256
                != sha256_bytes(source.canonical_bytes())
                or binding.rebound_bank_sha256 != rebound.bank_sha256
                or binding.rebound_bank_file_sha256
                != sha256_bytes(rebound.canonical_bytes())
            ):
                raise PortfolioTreatmentError(
                    f"{config} {role} runtime-rebind binding differs"
                )
            _verify_bank_runtime_rebind(
                source,
                rebound,
                target_tool_registry_sha256=receipt.target_tool_registry_sha256,
                target_tool_registry_runtime_sha256=(
                    receipt.target_tool_registry_runtime_sha256
                ),
            )
        outputs[config] = output_banks[config]
        candidates[config] = candidate_banks[config]

    # Decisions stay exactly those of the accepted source chain.  A rejected
    # candidate remains diagnostic-only; its output must be the rebound parent
    # bytes, yielding the execution alias without another provider call.
    source_records = {item.config: item for item in parent.manifest.records}
    for config, parent_config in _PARENT_CONFIG_BY_CONFIG.items():
        if source_records[config].decision != "accepted" and (
            outputs[config] != outputs[parent_config]
            or sha256_bytes(outputs[config].canonical_bytes())
            != sha256_bytes(outputs[parent_config].canonical_bytes())
        ):
            raise PortfolioTreatmentError(
                f"{config} rebound rollback is not an exact parent artifact alias"
            )

    aliases: list[PortfolioExecutionArtifactAlias] = []
    for target_config in ("s1", "s1s2", "full"):
        record = source_records[target_config]
        if record.decision == "accepted":
            continue
        source_config = _PARENT_CONFIG_BY_CONFIG[target_config]
        source_binding = bindings[(source_config, "output")]
        target_binding = bindings[(target_config, "output")]
        payload = {
            "schema_version": 1,
            "artifact_kind": "portfolio-execution-artifact-alias",
            "policy_version": PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION,
            "target_config": target_config,
            "source_config": source_config,
            "stage_decision": record.decision,
            "source_bank_sha256": outputs[source_config].bank_sha256,
            "target_bank_sha256": outputs[target_config].bank_sha256,
            "source_bank_file_sha256": source_binding.rebound_bank_file_sha256,
            "target_bank_file_sha256": target_binding.rebound_bank_file_sha256,
            "reuse_scope": "assistant_and_evaluator_query_artifacts",
            "provider_model_call_count": 0,
            "rejected_candidate_use": "diagnostic_only",
        }
        aliases.append(
            PortfolioExecutionArtifactAlias.model_validate(
                {
                    **payload,
                    "alias_sha256": sha256_bytes(canonical_json_bytes(payload)),
                },
                strict=True,
            )
        )
    return VerifiedPortfolioTreatmentChain(
        manifest=parent.manifest,
        output_banks=outputs,
        candidate_banks=candidates,
        execution_artifact_aliases=tuple(aliases),
        mutations=parent.mutations,
        invocation_receipts=parent.invocation_receipts,
        gate_reports=parent.gate_reports,
        compatibility_rebind=receipt,
        source_chain=parent,
        _marker=_VERIFIED_CHAIN_MARKER,
    )


def verify_portfolio_treatment_chain(
    manifest: PortfolioTreatmentChainManifest,
    *,
    output_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
    candidate_banks: Mapping[PortfolioTreatmentConfig, StaticBankArtifact],
    mutations: Mapping[PortfolioTreatmentConfig, PortfolioStageMutation],
    invocation_receipts: Mapping[
        PortfolioTreatmentConfig,
        PortfolioModelInvocationReceipt,
    ],
    gate_reports: Mapping[PortfolioGateConfig, PortfolioStageGateReport],
) -> VerifiedPortfolioTreatmentChain:
    """Cross-check a manifest against actual output/candidate Bank objects."""

    value = _validated_model(
        manifest,
        PortfolioTreatmentChainManifest,
        "treatment chain manifest",
    )
    if value.status != "ready_for_matrix":
        raise PortfolioTreatmentError(
            "only a ready_for_matrix treatment chain may be verified"
        )
    if set(output_banks) != set(_CONFIG_ORDER) or set(candidate_banks) != set(
        _CONFIG_ORDER
    ):
        raise PortfolioTreatmentError("treatment chain Bank set is incomplete")
    if set(mutations) != {"s1s2", "full"}:
        raise PortfolioTreatmentError("treatment chain mutation set is incomplete")
    if set(invocation_receipts) != set(_CONFIG_ORDER):
        raise PortfolioTreatmentError(
            "treatment chain invocation receipt set is incomplete"
        )
    if set(gate_reports) != {"s1", "s1s2", "full"}:
        raise PortfolioTreatmentError(
            "treatment chain stage gate report set is incomplete"
        )

    outputs: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    candidates: dict[PortfolioTreatmentConfig, StaticBankArtifact] = {}
    for binding, record in zip(value.banks, value.records, strict=True):
        output = _validate_bank_carrier(
            output_banks[binding.config],
            f"{binding.config} output Bank",
        )
        candidate = _validate_bank_carrier(
            candidate_banks[binding.config],
            f"{binding.config} candidate Bank",
        )
        if (
            output.bank_sha256 != binding.bank_sha256
            or sha256_bytes(output.canonical_bytes()) != binding.bank_file_sha256
            or candidate.bank_sha256 != record.candidate_bank_sha256
            or sha256_bytes(candidate.canonical_bytes())
            != record.candidate_bank_file_sha256
        ):
            raise PortfolioTreatmentError("treatment chain Bank digest mismatch")
        outputs[binding.config] = output
        candidates[binding.config] = candidate

        invocation = _validated_model(
            invocation_receipts[binding.config],
            PortfolioModelInvocationReceipt,
            f"{binding.config} model invocation receipt",
        )
        if (
            invocation.stage != record.stage
            or invocation.model_call_count != record.model_call_count
            or invocation.output_sha256 != record.raw_model_output_file_sha256
            or sha256_bytes(invocation.canonical_bytes())
            != record.invocation_receipt_file_sha256
        ):
            raise PortfolioTreatmentError(
                "treatment model invocation receipt binding mismatch"
            )

    gate_parents: dict[PortfolioGateConfig, PortfolioTreatmentConfig] = {
        "s1": "llm_static",
        "s1s2": "s1",
        "full": "s1s2",
    }
    gate_query_sets = {report.evaluation_query_ids for report in gate_reports.values()}
    if len(gate_query_sets) != 1:
        raise PortfolioTreatmentError(
            "S1/S2/S3 gates must use one shared evaluation query set"
        )
    for config, parent_config in gate_parents.items():
        report = _validated_model(
            gate_reports[config],
            PortfolioStageGateReport,
            f"{config} stage gate report",
        )
        record = next(item for item in value.records if item.config == config)
        if (
            report.evaluation_query_ids != value.optimization_query_ids
            or report.parent_bank_sha256 != outputs[parent_config].bank_sha256
            or report.candidate_bank_sha256 != candidates[config].bank_sha256
            or report.decision != record.decision
            or report.gate_report_sha256 != record.gate_report_sha256
            or sha256_bytes(report.canonical_bytes()) != record.gate_report_file_sha256
        ):
            raise PortfolioTreatmentError(
                "treatment stage gate report binding mismatch"
            )

    s1_record = value.records[1]
    expected_s1_output = (
        candidates["s1"] if s1_record.decision == "accepted" else outputs["llm_static"]
    )
    if outputs["s1"] != expected_s1_output:
        raise PortfolioTreatmentError(
            "S1 gate decision does not reproduce its output Bank"
        )

    for config, parent_config in (("s1s2", "s1"), ("full", "s1s2")):
        typed_config: PortfolioTreatmentConfig = config
        plan = _validated_model(
            mutations[typed_config],
            PortfolioStageMutation,
            f"{config} mutation",
        )
        record = next(item for item in value.records if item.config == config)
        if (
            plan.mutation_sha256 != record.mutation_sha256
            or plan.parent_bank_sha256 != outputs[parent_config].bank_sha256
        ):
            raise PortfolioTreatmentError("stage mutation receipt binding mismatch")
        rebuilt = apply_portfolio_stage_mutation(outputs[parent_config], plan)
        if rebuilt != candidates[typed_config]:
            raise PortfolioTreatmentError(
                "stage candidate does not reproduce from its mutation"
            )
        expected_output = (
            candidates[typed_config]
            if record.decision == "accepted"
            else outputs[parent_config]
        )
        if outputs[typed_config] != expected_output:
            raise PortfolioTreatmentError(
                "stage decision does not reproduce its output Bank"
            )

    for config in ("llm_static", "s1"):
        bank = candidates[config]
        if any(
            item.version != 1 or item.parent_skill_sha256 is not None
            for item in bank.skills
        ):
            raise PortfolioTreatmentError(
                f"{config} creation Bank contains derived Skill lineage"
            )
    return VerifiedPortfolioTreatmentChain(
        manifest=value,
        output_banks=dict(outputs),
        candidate_banks=dict(candidates),
        execution_artifact_aliases=build_portfolio_execution_artifact_aliases(value),
        mutations=dict(mutations),
        invocation_receipts=dict(invocation_receipts),
        gate_reports=dict(gate_reports),
        _marker=_VERIFIED_CHAIN_MARKER,
    )


def require_verified_portfolio_treatment_chain(
    value: object,
) -> VerifiedPortfolioTreatmentChain:
    if (
        type(value) is not VerifiedPortfolioTreatmentChain
        or value._marker is not _VERIFIED_CHAIN_MARKER
    ):
        raise TypeError("Portfolio treatments require a verified chain handle")
    if value.compatibility_rebind is not None:
        if value.source_chain is None:
            raise PortfolioTreatmentError(
                "runtime compatibility rebind lacks its accepted source chain"
            )
        replayed = verify_portfolio_runtime_compatibility_rebind_chain(
            rebind=value.compatibility_rebind,
            source_chain=value.source_chain,
            output_banks=value.output_banks,
            candidate_banks=value.candidate_banks,
        )
        if replayed != value:
            raise PortfolioTreatmentError("runtime compatibility rebind drifted")
        return value
    if value.source_chain is not None:
        raise PortfolioTreatmentError(
            "ordinary treatment chain unexpectedly carries a source chain"
        )
    return verify_portfolio_treatment_chain(
        value.manifest,
        output_banks=value.output_banks,
        candidate_banks=value.candidate_banks,
        mutations=value.mutations,
        invocation_receipts=value.invocation_receipts,
        gate_reports=value.gate_reports,
    )


__all__ = [
    "PORTFOLIO_EXECUTION_ARTIFACT_ALIAS_POLICY_VERSION",
    "PORTFOLIO_FINAL_RUBRIC_FILE_SHA256",
    "PORTFOLIO_GATE_DECISION_RULE",
    "PORTFOLIO_GATE_EVIDENCE_POLICY_VERSION",
    "PORTFOLIO_OPTIMIZATION_QUERY_IDS",
    "PORTFOLIO_RUNTIME_COMPATIBILITY_REBIND_POLICY_VERSION",
    "PORTFOLIO_STAGE_GATE_QUERY_COUNT",
    "PORTFOLIO_STAGE_MUTATION_POLICY_VERSION",
    "PORTFOLIO_TREATMENT_CHAIN_POLICY_VERSION",
    "PORTFOLIO_TREATMENT_SMOKE_POLICY_VERSION",
    "PortfolioArtifactBinding",
    "PortfolioBankBinding",
    "PortfolioExecutionArtifactAlias",
    "PortfolioModelInvocationReceipt",
    "PortfolioRuntimeCompatibilityRebind",
    "PortfolioRuntimeFileBinding",
    "PortfolioRuntimeRebindBankBinding",
    "PortfolioSkillMutation",
    "PortfolioSkillOutputContract",
    "PortfolioStageMutation",
    "PortfolioStageGateReport",
    "PortfolioStageGateRowProvenance",
    "PortfolioStageGateResultSet",
    "PortfolioTreatmentChainManifest",
    "PortfolioTreatmentError",
    "PortfolioTreatmentReceipt",
    "VerifiedCodexDraftRebind",
    "VerifiedPortfolioTreatmentChain",
    "apply_portfolio_stage_mutation",
    "build_portfolio_model_invocation_receipt",
    "build_portfolio_execution_artifact_aliases",
    "build_portfolio_stage_mutation",
    "build_portfolio_stage_gate_report",
    "build_portfolio_stage_gate_result_set",
    "calculate_portfolio_skill_adherence",
    "compile_portfolio_contract_refreshed_llm_static_bank",
    "compile_portfolio_s1_creator_bank",
    "compile_verified_codex_llm_static_bank",
    "load_verified_codex_draft_rebind",
    "parse_portfolio_skill_output_contract",
    "rebind_portfolio_bank_runtime",
    "require_verified_codex_draft_rebind",
    "require_verified_portfolio_treatment_chain",
    "validate_portfolio_bank_transition",
    "verify_portfolio_stage_gate_evidence",
    "verify_portfolio_runtime_compatibility_rebind_chain",
    "verify_portfolio_treatment_chain",
]
