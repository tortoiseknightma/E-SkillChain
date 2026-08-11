"""Create-only Portfolio smoke-matrix checklists without model execution.

This module records the smallest useful Portfolio execution plan: one already
verified ``dev_mini`` query crossed with the five canonical Assistant
configurations.  It deliberately does not manufacture a runtime lock, Skill
Bank identity, or request identity.  Until those inputs exist, every planned
instance remains explicitly blocked and the artifact records zero model calls.

The checklist is a planning artifact, not an Assistant run manifest and not a
Formal Research result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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

from skillchain.data.portfolio_remote_processing import (
    PortfolioProcessor,
    VerifiedPortfolioRemoteProcessingRuntime,
)
from skillchain.evaluation.assistant_runs import (
    MAIN_CONFIG_ORDER,
    AssistantQueryInput,
)
from skillchain.evaluation.packets import AssistantRunConfig
from skillchain.evaluation.portfolio_inputs import (
    PORTFOLIO_PROCESSOR_ORDER,
    PortfolioQueryAssetBinding,
    VerifiedPortfolioDevMiniInputs,
    require_verified_portfolio_dev_mini_inputs,
)
from skillchain.schemas import Query
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

PORTFOLIO_CHECKLIST_POLICY_VERSION = "portfolio-five-config-checklist-v1"
PORTFOLIO_DEFAULT_SELECTION_POLICY = (
    "first-dev-mini-document-reading-single-turn-nonboundary-no-card-v1"
)
PORTFOLIO_EXPLICIT_SELECTION_POLICY = "explicit-verified-query-id-v1"
PORTFOLIO_FORMAL_INELIGIBLE_REASON = "portfolio-pre-execution-checklist-only"

PortfolioSelectionPolicy = Literal[
    "first-dev-mini-document-reading-single-turn-nonboundary-no-card-v1",
    "explicit-verified-query-id-v1",
]
PortfolioChecklistBlocker = Literal[
    "assistant_runtime_lock_missing",
    "llm_static_bank_missing",
    "s1_bank_missing",
    "s1s2_bank_missing",
    "full_bank_missing",
]

_CHECKLIST_MARKER = object()
_MAX_CHECKLIST_BYTES = 1024 * 1024
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXPECTED_CONFIG_BLOCKERS: dict[
    AssistantRunConfig, tuple[PortfolioChecklistBlocker, ...]
] = {
    "noskill": ("assistant_runtime_lock_missing",),
    "llm_static": (
        "assistant_runtime_lock_missing",
        "llm_static_bank_missing",
    ),
    "s1": ("assistant_runtime_lock_missing", "s1_bank_missing"),
    "s1s2": ("assistant_runtime_lock_missing", "s1s2_bank_missing"),
    "full": ("assistant_runtime_lock_missing", "full_bank_missing"),
}


class PortfolioChecklistError(ValueError):
    """A Portfolio planning checklist is malformed, stale, or forged."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(value)))


def _self_hash(model: BaseModel, field_name: str) -> str:
    return _hash_payload(model.model_dump(mode="json", exclude={field_name}))


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


class PortfolioRemoteRuntimeBinding(_StrictFrozenModel):
    """Persisted identity of one already verified upload-authority preflight."""

    processor: PortfolioProcessor
    authorization_id: str
    authorization_file_sha256: Sha256
    receipt_file_sha256: Sha256
    receipt_sha256: Sha256
    selection_manifest_sha256: Sha256
    dataset_assets_sha256: Sha256
    plan_sha256: Sha256
    query_artifact_sha256: Sha256
    base_catalog_sha256: Sha256
    output_catalog_sha256: Sha256
    binding_sha256: Sha256

    @field_validator("authorization_id")
    @classmethod
    def validate_authorization_id(cls, value: str) -> str:
        return _nonblank(value, "authorization_id")

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.binding_sha256 != _self_hash(self, "binding_sha256"):
            raise ValueError("Portfolio remote-runtime binding self hash mismatch")
        return self


class PortfolioPlannedRunInstance(_StrictFrozenModel):
    """One blocked, not-yet-executable query/config pair."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-planned-assistant-run"] = (
        "portfolio-planned-assistant-run"
    )
    matrix_run_id: str
    query_ordinal: Literal[0] = 0
    config_ordinal: int = Field(ge=0, le=4)
    config: AssistantRunConfig
    query_id: str
    query_sha256: Sha256
    public_input_sha256: Sha256
    asset_id: str
    image_path: str
    image_sha256: Sha256
    runtime_catalog_sha256: Sha256
    assistant_runtime_lock_sha256: None = None
    bank_sha256: None = None
    request_sha256: None = None
    execution_ready: Literal[False] = False
    status: Literal["blocked"] = "blocked"
    blockers: tuple[PortfolioChecklistBlocker, ...]
    output_relpath: str
    instance_sha256: Sha256

    @field_validator(
        "matrix_run_id",
        "query_id",
        "asset_id",
        "image_path",
        "output_relpath",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("matrix_run_id contains unsafe characters")
        return value

    @field_validator("output_relpath")
    @classmethod
    def validate_output_relpath(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or value in {"", "."}
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("output_relpath must be a normalized relative POSIX path")
        return value

    @model_validator(mode="after")
    def validate_instance(self) -> Self:
        if self.config_ordinal >= len(MAIN_CONFIG_ORDER):
            raise ValueError("planned config ordinal is outside canonical order")
        if MAIN_CONFIG_ORDER[self.config_ordinal] != self.config:
            raise ValueError("planned config ordinal differs from canonical order")
        if self.blockers != _EXPECTED_CONFIG_BLOCKERS[self.config]:
            raise ValueError("planned blockers differ from the config requirements")
        expected_path = f"configs/{self.config_ordinal:02d}-{self.config}"
        if self.output_relpath != expected_path:
            raise ValueError("planned output path differs from canonical layout")
        if self.instance_sha256 != _self_hash(self, "instance_sha256"):
            raise ValueError("Portfolio planned instance self hash mismatch")
        return self


class PortfolioFiveConfigRunChecklist(_StrictFrozenModel):
    """Canonical one-query/five-config plan that proves no run has started."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-five-config-run-checklist"] = (
        "portfolio-five-config-run-checklist"
    )
    policy_version: Literal["portfolio-five-config-checklist-v1"] = (
        PORTFOLIO_CHECKLIST_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    matrix_run_id: str
    portfolio_plan_sha256: Sha256
    portfolio_plan_manifest_file_sha256: Sha256
    accepted_ledger_sha256: Sha256
    query_artifact_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed_set_sha256: Sha256
    base_catalog_sha256: Sha256
    runtime_catalog_sha256: Sha256
    authorization_file_sha256: Sha256
    receipt_file_sha256: Sha256
    receipt_sha256: Sha256
    selection_manifest_sha256: Sha256
    dataset_assets_sha256: Sha256
    remote_runtimes: tuple[PortfolioRemoteRuntimeBinding, ...]
    selection_policy: PortfolioSelectionPolicy
    selected_query: AssistantQueryInput
    selected_asset: PortfolioQueryAssetBinding
    planned_query_count: Literal[1] = 1
    planned_config_count: Literal[5] = 5
    planned_run_count: Literal[5] = 5
    config_order: tuple[AssistantRunConfig, ...]
    config_order_sha256: Sha256
    coverage_sha256: Sha256
    instances: tuple[PortfolioPlannedRunInstance, ...]
    assistant_runtime_lock_sha256: None = None
    model_calls_performed: Literal[0] = 0
    execution_authorized: Literal[False] = False
    status: Literal["planned_no_model_calls"] = "planned_no_model_calls"
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal[
        "portfolio-pre-execution-checklist-only"
    ] = PORTFOLIO_FORMAL_INELIGIBLE_REASON
    checklist_sha256: Sha256

    @field_validator("matrix_run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        value = _nonblank(value, "matrix_run_id")
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("matrix_run_id contains unsafe characters")
        return value

    @model_validator(mode="after")
    def validate_checklist(self) -> Self:
        if self.config_order != MAIN_CONFIG_ORDER:
            raise ValueError("Portfolio config order must be the canonical five")
        if self.config_order_sha256 != _hash_payload(list(MAIN_CONFIG_ORDER)):
            raise ValueError("Portfolio config-order hash mismatch")
        if (
            self.selected_query.query_id != self.selected_asset.query_id
            or not self.instances
        ):
            raise ValueError("selected query and image binding differ")
        try:
            public_input = parse_canonical_json(
                self.selected_query.public_input_json.encode("utf-8"),
                label="Portfolio selected public input",
            )
        except ArtifactFormatError as error:  # already guarded by nested model
            raise ValueError("selected public input is not canonical") from error
        if (
            not isinstance(public_input, dict)
            or public_input.get("asset_id") != self.selected_asset.asset_id
            or public_input.get("image_path") != self.selected_asset.image_path
        ):
            raise ValueError("selected public input differs from its image binding")

        if tuple(item.processor for item in self.remote_runtimes) != (
            PORTFOLIO_PROCESSOR_ORDER
        ):
            raise ValueError("Portfolio checklist remote processors drifted")
        if len(
            {item.authorization_id for item in self.remote_runtimes}
        ) != 1:
            raise ValueError("Portfolio remote runtimes use different authorities")
        for runtime in self.remote_runtimes:
            if (
                runtime.authorization_file_sha256
                != self.authorization_file_sha256
                or runtime.receipt_file_sha256 != self.receipt_file_sha256
                or runtime.receipt_sha256 != self.receipt_sha256
                or runtime.selection_manifest_sha256
                != self.selection_manifest_sha256
                or runtime.dataset_assets_sha256 != self.dataset_assets_sha256
                or runtime.plan_sha256 != self.portfolio_plan_sha256
                or runtime.query_artifact_sha256
                != self.query_artifact_sha256
                or runtime.base_catalog_sha256 != self.base_catalog_sha256
                or runtime.output_catalog_sha256
                != self.runtime_catalog_sha256
            ):
                raise ValueError("Portfolio remote-runtime commitment drifted")

        if len(self.instances) != 5:
            raise ValueError("Portfolio checklist must contain exactly five runs")
        expected_coverage = tuple(
            (self.selected_query.query_id, config) for config in MAIN_CONFIG_ORDER
        )
        actual_coverage = tuple(
            (item.query_id, item.config) for item in self.instances
        )
        if actual_coverage != expected_coverage:
            raise ValueError("Portfolio checklist coverage or order drifted")
        if self.coverage_sha256 != _hash_payload(
            [
                {"query_id": query_id, "config": config}
                for query_id, config in expected_coverage
            ]
        ):
            raise ValueError("Portfolio checklist coverage hash mismatch")
        for ordinal, instance in enumerate(self.instances):
            if (
                instance.matrix_run_id != self.matrix_run_id
                or instance.query_ordinal != 0
                or instance.config_ordinal != ordinal
                or instance.query_sha256 != self.selected_query.query_sha256
                or instance.public_input_sha256
                != self.selected_query.public_input_sha256
                or instance.asset_id != self.selected_asset.asset_id
                or instance.image_path != self.selected_asset.image_path
                or instance.image_sha256 != self.selected_asset.image_sha256
                or instance.runtime_catalog_sha256
                != self.runtime_catalog_sha256
            ):
                raise ValueError("Portfolio planned instance binding drifted")
        if len({item.output_relpath for item in self.instances}) != 5:
            raise ValueError("Portfolio planned output paths must be unique")
        if self.checklist_sha256 != _self_hash(self, "checklist_sha256"):
            raise ValueError("Portfolio checklist self hash mismatch")
        return self


@dataclass(frozen=True)
class CreatedPortfolioFiveConfigRunChecklist:
    path: Path
    checklist: PortfolioFiveConfigRunChecklist
    file_sha256: str


@dataclass(frozen=True)
class VerifiedPortfolioFiveConfigRunChecklist:
    """A canonical checklist re-derived from currently verified inputs."""

    path: Path
    expected_file_sha256: str
    checklist: PortfolioFiveConfigRunChecklist
    _content: bytes = field(repr=False)
    _marker: object = field(repr=False)


def build_portfolio_five_config_run_checklist(
    inputs: VerifiedPortfolioDevMiniInputs,
    *,
    matrix_run_id: str,
    query_id: str | None = None,
) -> PortfolioFiveConfigRunChecklist:
    """Build the blocked 1x5 plan; this function cannot call a model."""

    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    return _build_checklist(
        verified_inputs,
        matrix_run_id=matrix_run_id,
        query_id=query_id,
    )


def create_portfolio_five_config_run_checklist(
    path: str | Path,
    checklist: PortfolioFiveConfigRunChecklist,
    *,
    inputs: VerifiedPortfolioDevMiniInputs,
) -> CreatedPortfolioFiveConfigRunChecklist:
    """Create one canonical checklist without ever replacing an existing file."""

    if type(checklist) is not PortfolioFiveConfigRunChecklist:
        raise TypeError("checklist must be PortfolioFiveConfigRunChecklist")
    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    rebuilt = _rebuild_checklist(checklist, verified_inputs)
    if rebuilt != checklist:
        raise PortfolioChecklistError(
            "checklist differs from the currently verified Portfolio inputs"
        )
    content = canonical_json_bytes(checklist.model_dump(mode="json"))
    output = Path(path).absolute()
    atomic_create_file(output, content)
    held = read_stable_regular_file(
        output,
        label="Portfolio five-config checklist",
        max_bytes=_MAX_CHECKLIST_BYTES,
    )
    if held != content:
        raise PortfolioChecklistError("created Portfolio checklist changed on disk")
    return CreatedPortfolioFiveConfigRunChecklist(
        path=output,
        checklist=checklist,
        file_sha256=sha256_bytes(content),
    )


def load_verified_portfolio_five_config_run_checklist(
    path: str | Path,
    *,
    expected_file_sha256: str,
    inputs: VerifiedPortfolioDevMiniInputs,
) -> VerifiedPortfolioFiveConfigRunChecklist:
    """Load, re-derive, and stably re-read one externally committed checklist."""

    expected = _require_sha256(
        expected_file_sha256,
        "expected Portfolio checklist file SHA-256",
    )
    checklist_path = Path(path).absolute()
    content = _read_checklist(checklist_path)
    if sha256_bytes(content) != expected:
        raise PortfolioChecklistError("Portfolio checklist external digest mismatch")
    checklist = _parse_checklist(content)
    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    rebuilt = _rebuild_checklist(checklist, verified_inputs)
    if rebuilt != checklist:
        raise PortfolioChecklistError(
            "Portfolio checklist differs from currently verified inputs"
        )
    require_verified_portfolio_dev_mini_inputs(verified_inputs)
    if _read_checklist(checklist_path) != content:
        raise PortfolioChecklistError(
            "Portfolio checklist changed during verified loading"
        )
    return VerifiedPortfolioFiveConfigRunChecklist(
        path=checklist_path,
        expected_file_sha256=expected,
        checklist=checklist,
        _content=content,
        _marker=_CHECKLIST_MARKER,
    )


def require_verified_portfolio_five_config_run_checklist(
    value: object,
    *,
    inputs: VerifiedPortfolioDevMiniInputs,
) -> VerifiedPortfolioFiveConfigRunChecklist:
    """Reject forged, stale, mutated, or no-longer-reproducible handles."""

    if (
        type(value) is not VerifiedPortfolioFiveConfigRunChecklist
        or value._marker is not _CHECKLIST_MARKER
    ):
        raise TypeError(
            "Portfolio checklist requires the external-digest verified loader"
        )
    if _require_sha256(
        value.expected_file_sha256,
        "verified Portfolio checklist file SHA-256",
    ) != sha256_bytes(value._content):
        raise PortfolioChecklistError("verified checklist digest/content mismatch")
    current_content = _read_checklist(value.path)
    if current_content != value._content:
        raise PortfolioChecklistError("verified Portfolio checklist changed on disk")
    current = _parse_checklist(current_content)
    if current != value.checklist:
        raise PortfolioChecklistError("verified Portfolio checklist was mutated")
    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    if _rebuild_checklist(current, verified_inputs) != current:
        raise PortfolioChecklistError(
            "verified Portfolio checklist no longer matches its inputs"
        )
    require_verified_portfolio_dev_mini_inputs(verified_inputs)
    if _read_checklist(value.path) != current_content:
        raise PortfolioChecklistError(
            "Portfolio checklist changed during handle verification"
        )
    return value


def _build_checklist(
    inputs: VerifiedPortfolioDevMiniInputs,
    *,
    matrix_run_id: str,
    query_id: str | None,
) -> PortfolioFiveConfigRunChecklist:
    matrix_run_id = _validate_run_id(matrix_run_id)
    selected_query, selection_policy = _select_query(inputs, query_id=query_id)
    assistant_by_id = {
        item.query_id: item for item in inputs.assistant_queries
    }
    asset_by_id = {item.query_id: item for item in inputs.query_assets}
    if (
        len(assistant_by_id) != len(inputs.assistant_queries)
        or len(asset_by_id) != len(inputs.query_assets)
    ):
        raise PortfolioChecklistError("Portfolio input projections are duplicated")
    try:
        assistant_query = assistant_by_id[selected_query.query_id]
        selected_asset = asset_by_id[selected_query.query_id]
    except KeyError as error:
        raise PortfolioChecklistError(
            "selected query lacks an Assistant or image projection"
        ) from error

    remote_runtimes = tuple(
        _make_remote_binding(runtime) for runtime in inputs.remote_runtimes
    )
    if tuple(item.processor for item in remote_runtimes) != (
        PORTFOLIO_PROCESSOR_ORDER
    ):
        raise PortfolioChecklistError(
            "verified Portfolio runtimes are not in canonical order"
        )
    first_runtime = inputs.remote_runtimes[0]
    receipt = first_runtime.receipt
    instances = tuple(
        _make_instance(
            matrix_run_id=matrix_run_id,
            config_ordinal=ordinal,
            config=config,
            query=assistant_query,
            asset=selected_asset,
            runtime_catalog_sha256=inputs.expected_output_catalog_sha256,
        )
        for ordinal, config in enumerate(MAIN_CONFIG_ORDER)
    )
    coverage = [
        {"query_id": assistant_query.query_id, "config": config}
        for config in MAIN_CONFIG_ORDER
    ]
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-five-config-run-checklist",
        "policy_version": PORTFOLIO_CHECKLIST_POLICY_VERSION,
        "track": "portfolio",
        "matrix_run_id": matrix_run_id,
        "portfolio_plan_sha256": inputs.expected_plan_sha256,
        "portfolio_plan_manifest_file_sha256": (
            inputs.expected_plan_manifest_file_sha256
        ),
        "accepted_ledger_sha256": inputs.expected_accepted_ledger_sha256,
        "query_artifact_sha256": inputs.expected_query_artifact_sha256,
        "capability_assignments_sha256": (
            inputs.expected_capability_assignments_sha256
        ),
        "seed_set_sha256": inputs.expected_seed_set_sha256,
        "base_catalog_sha256": inputs.expected_base_catalog_sha256,
        "runtime_catalog_sha256": inputs.expected_output_catalog_sha256,
        "authorization_file_sha256": (
            first_runtime.authorization_file_sha256
        ),
        "receipt_file_sha256": first_runtime.receipt_file_sha256,
        "receipt_sha256": receipt.receipt_sha256,
        "selection_manifest_sha256": (
            receipt.base_selection_manifest_sha256
        ),
        "dataset_assets_sha256": first_runtime.dataset_assets_sha256,
        "remote_runtimes": remote_runtimes,
        "selection_policy": selection_policy,
        "selected_query": assistant_query,
        "selected_asset": selected_asset,
        "planned_query_count": 1,
        "planned_config_count": 5,
        "planned_run_count": 5,
        "config_order": MAIN_CONFIG_ORDER,
        "config_order_sha256": _hash_payload(list(MAIN_CONFIG_ORDER)),
        "coverage_sha256": _hash_payload(coverage),
        "instances": instances,
        "assistant_runtime_lock_sha256": None,
        "model_calls_performed": 0,
        "execution_authorized": False,
        "status": "planned_no_model_calls",
        "formal_eligible": False,
        "formal_ineligible_reason": PORTFOLIO_FORMAL_INELIGIBLE_REASON,
    }
    return PortfolioFiveConfigRunChecklist.model_validate(
        {**unsigned, "checklist_sha256": _hash_payload(unsigned)},
        strict=True,
    )


def _rebuild_checklist(
    checklist: PortfolioFiveConfigRunChecklist,
    inputs: VerifiedPortfolioDevMiniInputs,
) -> PortfolioFiveConfigRunChecklist:
    query_id = (
        checklist.selected_query.query_id
        if checklist.selection_policy == PORTFOLIO_EXPLICIT_SELECTION_POLICY
        else None
    )
    return _build_checklist(
        inputs,
        matrix_run_id=checklist.matrix_run_id,
        query_id=query_id,
    )


def _select_query(
    inputs: VerifiedPortfolioDevMiniInputs,
    *,
    query_id: str | None,
) -> tuple[Query, PortfolioSelectionPolicy]:
    if query_id is not None:
        query_id = _nonblank(query_id, "query_id")
        matches = tuple(item for item in inputs.queries if item.query_id == query_id)
        if len(matches) != 1:
            raise PortfolioChecklistError(
                f"explicit query_id is not uniquely verified: {query_id}"
            )
        return matches[0], PORTFOLIO_EXPLICIT_SELECTION_POLICY

    candidates = sorted(
        (
            item
            for item in inputs.queries
            if item.split == "dev_mini"
            and item.canonical_capability == "utility.document_reading"
            and item.is_boundary is False
            and len(item.turns) == 1
            and item.requires_card is False
        ),
        key=lambda item: item.query_id,
    )
    if not candidates:
        raise PortfolioChecklistError(
            "no query satisfies the frozen Portfolio smoke-selection policy"
        )
    return candidates[0], PORTFOLIO_DEFAULT_SELECTION_POLICY


def _make_remote_binding(
    runtime: VerifiedPortfolioRemoteProcessingRuntime,
) -> PortfolioRemoteRuntimeBinding:
    unsigned = {
        "processor": runtime.processor,
        "authorization_id": runtime.authorization.authorization_id,
        "authorization_file_sha256": runtime.authorization_file_sha256,
        "receipt_file_sha256": runtime.receipt_file_sha256,
        "receipt_sha256": runtime.receipt.receipt_sha256,
        "selection_manifest_sha256": (
            runtime.receipt.base_selection_manifest_sha256
        ),
        "dataset_assets_sha256": runtime.dataset_assets_sha256,
        "plan_sha256": runtime.plan_sha256,
        "query_artifact_sha256": runtime.query_artifact_sha256,
        "base_catalog_sha256": runtime.receipt.base_catalog_sha256,
        "output_catalog_sha256": runtime.catalog.catalog_sha256,
    }
    return PortfolioRemoteRuntimeBinding.model_validate(
        {**unsigned, "binding_sha256": _hash_payload(unsigned)},
        strict=True,
    )


def _make_instance(
    *,
    matrix_run_id: str,
    config_ordinal: int,
    config: AssistantRunConfig,
    query: AssistantQueryInput,
    asset: PortfolioQueryAssetBinding,
    runtime_catalog_sha256: str,
) -> PortfolioPlannedRunInstance:
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-planned-assistant-run",
        "matrix_run_id": matrix_run_id,
        "query_ordinal": 0,
        "config_ordinal": config_ordinal,
        "config": config,
        "query_id": query.query_id,
        "query_sha256": query.query_sha256,
        "public_input_sha256": query.public_input_sha256,
        "asset_id": asset.asset_id,
        "image_path": asset.image_path,
        "image_sha256": asset.image_sha256,
        "runtime_catalog_sha256": runtime_catalog_sha256,
        "assistant_runtime_lock_sha256": None,
        "bank_sha256": None,
        "request_sha256": None,
        "execution_ready": False,
        "status": "blocked",
        "blockers": _EXPECTED_CONFIG_BLOCKERS[config],
        "output_relpath": f"configs/{config_ordinal:02d}-{config}",
    }
    return PortfolioPlannedRunInstance.model_validate(
        {**unsigned, "instance_sha256": _hash_payload(unsigned)},
        strict=True,
    )


def _parse_checklist(content: bytes) -> PortfolioFiveConfigRunChecklist:
    try:
        parse_canonical_json(content, label="Portfolio five-config checklist")
        return PortfolioFiveConfigRunChecklist.model_validate_json(
            content,
            strict=True,
        )
    except (ArtifactFormatError, ValidationError, ValueError) as error:
        raise PortfolioChecklistError(
            "Portfolio five-config checklist is invalid"
        ) from error


def _read_checklist(path: Path) -> bytes:
    try:
        return read_stable_regular_file(
            path,
            label="Portfolio five-config checklist",
            max_bytes=_MAX_CHECKLIST_BYTES,
        )
    except ArtifactFormatError as error:
        raise PortfolioChecklistError(str(error)) from error


def _validate_run_id(value: object) -> str:
    if type(value) is not str:
        raise TypeError("matrix_run_id must be a string")
    value = _nonblank(value, "matrix_run_id")
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("matrix_run_id contains unsafe characters")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "PORTFOLIO_CHECKLIST_POLICY_VERSION",
    "PORTFOLIO_DEFAULT_SELECTION_POLICY",
    "PORTFOLIO_EXPLICIT_SELECTION_POLICY",
    "CreatedPortfolioFiveConfigRunChecklist",
    "PortfolioChecklistError",
    "PortfolioFiveConfigRunChecklist",
    "PortfolioPlannedRunInstance",
    "PortfolioRemoteRuntimeBinding",
    "VerifiedPortfolioFiveConfigRunChecklist",
    "build_portfolio_five_config_run_checklist",
    "create_portfolio_five_config_run_checklist",
    "load_verified_portfolio_five_config_run_checklist",
    "require_verified_portfolio_five_config_run_checklist",
]
