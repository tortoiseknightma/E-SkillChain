"""Portfolio-only Stage 1 input handoff for the verified ``dev_mini`` corpus.

This module deliberately does not reuse the Formal Research
``TrajectoryBundle`` or ``S1CreatorPacket`` types.  Formal Stage 1 remains
``opt_pool``-only.  The types here prepare a small, deterministic Portfolio
handoff for an engineering smoke run and can never authorize a model call,
claim a generated Bank, or acquire formal eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Annotated, Literal, Self, TypeVar

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
    load_codex_authoring_input,
)
from skillchain.evaluation.portfolio_checklist import (
    VerifiedPortfolioFiveConfigRunChecklist,
    require_verified_portfolio_five_config_run_checklist,
)
from skillchain.evaluation.portfolio_inputs import (
    VerifiedPortfolioDevMiniInputs,
    require_verified_portfolio_dev_mini_inputs,
)
from skillchain.schemas import Intent, Query
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

PORTFOLIO_STAGE1_POLICY_VERSION = "portfolio-dev-mini-s1-handoff-v1"
PORTFOLIO_STAGE1_SELECTION_POLICY = (
    "paper-five-intent-lexical-first-single-turn-non-boundary-"
    "excluding-smoke-leakage-v1"
)
PORTFOLIO_STAGE1_PURPOSE = "input_handoff_smoke"
PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON = (
    "portfolio-dev-mini-is-not-formal-opt-pool"
)
PORTFOLIO_STAGE1_OUTPUT_FILES: tuple[str, ...] = (
    "trajectory-bundle.json",
    "s1-creator-packet.json",
    "handoff-manifest.json",
)
PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS: tuple[str, ...] = (
    "dm-006",
    "dm-009",
    "dm-013",
    "dm-016",
    "dm-020",
)
PORTFOLIO_STAGE1_INTENT_ORDER: tuple[Intent, ...] = (
    "exact_match",
    "multi_product",
    "divergent_rec",
    "encyclopedia",
    "utility",
)

_PORTFOLIO_STAGE1_INTENT_SET = frozenset(PORTFOLIO_STAGE1_INTENT_ORDER)
_MAX_AUTHORING_INPUT_BYTES = 16 * 1024 * 1024
_MAX_HANDOFF_FILE_BYTES = 64 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_HANDOFF_MARKER = object()


class PortfolioStage1Error(ValueError):
    """A Portfolio Stage 1 input, output, or external binding drifted."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _as_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _nonblank(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonblank without edge whitespace")
    return value


def _normalized_relative_posix_path(value: str, label: str) -> str:
    value = _nonblank(value, label)
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hash_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(_jsonable(value)))


def _self_hash(model: BaseModel, field_name: str) -> str:
    payload = model.model_dump(mode="json")
    payload.pop(field_name, None)
    return _hash_payload(payload)


class PortfolioTrajectoryTurn(_StrictFrozenModel):
    """One verbatim user interaction turn visible to the Portfolio Creator."""

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _nonblank(value, "content")


class PortfolioUserTrajectory(_StrictFrozenModel):
    """Allowlisted projection of one accepted Portfolio query."""

    query_id: str
    asset_id: str
    image_path: str
    image_sha256: Sha256
    turns: tuple[PortfolioTrajectoryTurn, ...] = Field(min_length=1)
    canonical_intent: Intent
    canonical_capability: str

    @field_validator("turns", mode="before")
    @classmethod
    def coerce_turns(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator("query_id", "asset_id", "canonical_capability")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        return _normalized_relative_posix_path(value, "image_path")

    @model_validator(mode="after")
    def validate_turn_shape(self) -> Self:
        roles = tuple(turn.role for turn in self.turns)
        if roles not in (("user",), ("user", "assistant", "user")):
            raise ValueError(
                "turns must be exactly [user] or [user, assistant, user]"
            )
        return self


class PortfolioDevMiniTrajectoryBundle(_StrictFrozenModel):
    """Five-intent Portfolio trajectory input, distinct from Formal opt-pool."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-dev-mini-trajectory-bundle"] = (
        "portfolio-dev-mini-trajectory-bundle"
    )
    policy_version: Literal[PORTFOLIO_STAGE1_POLICY_VERSION] = (
        PORTFOLIO_STAGE1_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    purpose: Literal[PORTFOLIO_STAGE1_PURPOSE] = PORTFOLIO_STAGE1_PURPOSE
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    source_split: Literal["dev_mini"] = "dev_mini"
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal[
        PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    ] = PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    portfolio_plan_sha256: Sha256
    accepted_ledger_sha256: Sha256
    query_artifact_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed_set_sha256: Sha256
    base_catalog_sha256: Sha256
    runtime_catalog_sha256: Sha256
    run_checklist_file_sha256: Sha256
    run_checklist_sha256: Sha256
    smoke_query_id: str
    smoke_query_sha256: Sha256
    smoke_public_input_sha256: Sha256
    excluded_smoke_leakage_group_id: str
    selection_policy: Literal[PORTFOLIO_STAGE1_SELECTION_POLICY] = (
        PORTFOLIO_STAGE1_SELECTION_POLICY
    )
    selected_query_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    taxonomy_version: str
    task_spec_version: str
    trajectories: tuple[PortfolioUserTrajectory, ...] = Field(
        min_length=5,
        max_length=5,
    )
    model_calls_performed: Literal[0] = 0
    creator_invoked: Literal[False] = False
    bank_generated: Literal[False] = False
    execution_authorized: Literal[False] = False
    trajectory_bundle_sha256: Sha256

    @field_validator("selected_query_ids", "trajectories", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @field_validator(
        "smoke_query_id",
        "excluded_smoke_leakage_group_id",
        "taxonomy_version",
        "task_spec_version",
    )
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        trajectory_ids = tuple(item.query_id for item in self.trajectories)
        if (
            self.selected_query_ids != PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS
            or trajectory_ids != self.selected_query_ids
            or trajectory_ids != tuple(sorted(set(trajectory_ids)))
            or self.smoke_query_id in trajectory_ids
        ):
            raise ValueError(
                "Portfolio Stage 1 default query selection or order drifted"
            )
        observed_intents = {
            trajectory.canonical_intent for trajectory in self.trajectories
        }
        if observed_intents != _PORTFOLIO_STAGE1_INTENT_SET:
            raise ValueError(
                "Portfolio Stage 1 bundle must cover exactly the paper five intents"
            )
        if self.trajectory_bundle_sha256 != _self_hash(
            self,
            "trajectory_bundle_sha256",
        ):
            raise ValueError("trajectory_bundle_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1CreatorPacket(_StrictFrozenModel):
    """Common Codex input plus a distinct Portfolio-only trajectory bundle."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-creator-packet"] = (
        "portfolio-s1-creator-packet"
    )
    policy_version: Literal[PORTFOLIO_STAGE1_POLICY_VERSION] = (
        PORTFOLIO_STAGE1_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    purpose: Literal[PORTFOLIO_STAGE1_PURPOSE] = PORTFOLIO_STAGE1_PURPOSE
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal[
        PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    ] = PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    authoring_input_file_sha256: Sha256
    authoring_input: CodexAuthoringInput
    trajectory_bundle: PortfolioDevMiniTrajectoryBundle
    model_calls_performed: Literal[0] = 0
    creator_invoked: Literal[False] = False
    bank_generated: Literal[False] = False
    execution_authorized: Literal[False] = False
    packet_sha256: Sha256

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if self.authoring_input_file_sha256 != sha256_bytes(
            self.authoring_input.canonical_bytes()
        ):
            raise ValueError(
                "Portfolio S1 common authoring-input file binding drifted"
            )
        if (
            self.trajectory_bundle.taxonomy_version
            != self.authoring_input.taxonomy.version
        ):
            raise ValueError("Portfolio S1 taxonomy binding drifted")
        if (
            self.trajectory_bundle.task_spec_version
            != self.authoring_input.task_specification.version
        ):
            raise ValueError(
                "Portfolio S1 task-specification binding drifted"
            )
        if self.packet_sha256 != _self_hash(self, "packet_sha256"):
            raise ValueError("packet_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1HandoffManifest(_StrictFrozenModel):
    """Create-only three-file handoff commitment."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["portfolio-s1-handoff-manifest"] = (
        "portfolio-s1-handoff-manifest"
    )
    policy_version: Literal[PORTFOLIO_STAGE1_POLICY_VERSION] = (
        PORTFOLIO_STAGE1_POLICY_VERSION
    )
    track: Literal["portfolio"] = "portfolio"
    purpose: Literal[PORTFOLIO_STAGE1_PURPOSE] = PORTFOLIO_STAGE1_PURPOSE
    status: Literal["prepared_not_invoked"] = "prepared_not_invoked"
    formal_eligible: Literal[False] = False
    formal_ineligible_reason: Literal[
        PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    ] = PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
    artifact_files: tuple[str, ...] = PORTFOLIO_STAGE1_OUTPUT_FILES
    portfolio_plan_sha256: Sha256
    accepted_ledger_sha256: Sha256
    query_artifact_sha256: Sha256
    capability_assignments_sha256: Sha256
    seed_set_sha256: Sha256
    base_catalog_sha256: Sha256
    runtime_catalog_sha256: Sha256
    run_checklist_file_sha256: Sha256
    run_checklist_sha256: Sha256
    authoring_input_file_sha256: Sha256
    authoring_input_sha256: Sha256
    selection_policy: Literal[PORTFOLIO_STAGE1_SELECTION_POLICY] = (
        PORTFOLIO_STAGE1_SELECTION_POLICY
    )
    selected_query_ids: tuple[str, ...] = Field(min_length=5, max_length=5)
    selected_query_count: Literal[5] = 5
    trajectory_bundle_sha256: Sha256
    trajectory_bundle_file_sha256: Sha256
    creator_packet_sha256: Sha256
    creator_packet_file_sha256: Sha256
    model_calls_performed: Literal[0] = 0
    creator_invoked: Literal[False] = False
    bank_generated: Literal[False] = False
    execution_authorized: Literal[False] = False
    manifest_sha256: Sha256

    @field_validator("artifact_files", "selected_query_ids", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return _as_tuple(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if (
            self.artifact_files != PORTFOLIO_STAGE1_OUTPUT_FILES
            or self.selected_query_ids != PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS
        ):
            raise ValueError("Portfolio S1 manifest file set or selection drifted")
        if self.manifest_sha256 != _self_hash(self, "manifest_sha256"):
            raise ValueError("manifest_sha256 mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class PortfolioS1HandoffArtifacts:
    """Deterministically rebuilt handoff values before publication."""

    trajectory_bundle: PortfolioDevMiniTrajectoryBundle
    creator_packet: PortfolioS1CreatorPacket
    manifest: PortfolioS1HandoffManifest


@dataclass(frozen=True)
class CreatedPortfolioS1Handoff:
    output_dir: Path
    trajectory_bundle_path: Path
    creator_packet_path: Path
    manifest_path: Path
    trajectory_bundle_file_sha256: str
    creator_packet_file_sha256: str
    manifest_file_sha256: str
    selected_query_count: int


@dataclass(frozen=True)
class VerifiedPortfolioS1Handoff:
    """A deeply re-derived, exact-file-set Portfolio handoff handle."""

    output_dir: Path
    authoring_input_path: Path
    expected_authoring_input_file_sha256: str
    expected_manifest_file_sha256: str
    trajectory_bundle: PortfolioDevMiniTrajectoryBundle
    creator_packet: PortfolioS1CreatorPacket
    manifest: PortfolioS1HandoffManifest
    _contents: tuple[bytes, bytes, bytes] = field(repr=False)
    _marker: object = field(repr=False)


def build_portfolio_s1_handoff(
    *,
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
    authoring_input_path: str | Path,
    expected_authoring_input_file_sha256: str,
) -> PortfolioS1HandoffArtifacts:
    """Build the deterministic zero-call handoff without writing files."""

    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    verified_checklist = require_verified_portfolio_five_config_run_checklist(
        checklist,
        inputs=verified_inputs,
    )
    authoring_path, authoring_content, authoring_input = (
        _load_common_authoring_input(
            authoring_input_path,
            expected_file_sha256=expected_authoring_input_file_sha256,
        )
    )
    artifacts = _build_artifacts(
        verified_inputs,
        verified_checklist,
        authoring_input=authoring_input,
        authoring_input_file_sha256=sha256_bytes(authoring_content),
    )

    require_verified_portfolio_dev_mini_inputs(verified_inputs)
    require_verified_portfolio_five_config_run_checklist(
        verified_checklist,
        inputs=verified_inputs,
    )
    _, final_content, final_input = _load_common_authoring_input(
        authoring_path,
        expected_file_sha256=expected_authoring_input_file_sha256,
    )
    if final_content != authoring_content or final_input != authoring_input:
        raise PortfolioStage1Error(
            "common Codex authoring input changed during handoff build"
        )
    return artifacts


def create_portfolio_s1_handoff(
    *,
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
    authoring_input_path: str | Path,
    expected_authoring_input_file_sha256: str,
    output_dir: str | Path,
) -> CreatedPortfolioS1Handoff:
    """Create and independently reload the exact three-file handoff directory."""

    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(
            f"Portfolio S1 handoff is create-only; target exists: {destination}"
        )
    artifacts = build_portfolio_s1_handoff(
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=authoring_input_path,
        expected_authoring_input_file_sha256=(
            expected_authoring_input_file_sha256
        ),
    )
    contents = _artifact_contents(artifacts)
    staging = new_staging_directory(destination)
    try:
        for name, content in zip(
            PORTFOLIO_STAGE1_OUTPUT_FILES,
            contents,
            strict=True,
        ):
            atomic_create_file(staging / name, content)
        _read_exact_handoff_files(staging)
        atomic_publish_new_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    manifest_file_sha256 = sha256_bytes(contents[2])
    observed_contents = _read_exact_handoff_files(destination)
    if observed_contents != contents:
        raise PortfolioStage1Error(
            "created Portfolio S1 handoff differs from prepared bytes"
        )
    observed_artifacts = _parse_artifacts(observed_contents)
    if observed_artifacts != artifacts:
        raise PortfolioStage1Error(
            "created Portfolio S1 handoff failed independent parsing"
        )
    verified = load_verified_portfolio_s1_handoff(
        destination,
        expected_manifest_file_sha256=manifest_file_sha256,
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=authoring_input_path,
        expected_authoring_input_file_sha256=(
            expected_authoring_input_file_sha256
        ),
    )
    if (
        verified.trajectory_bundle != artifacts.trajectory_bundle
        or verified.creator_packet != artifacts.creator_packet
        or verified.manifest != artifacts.manifest
        or _read_exact_handoff_files(destination) != observed_contents
    ):
        raise PortfolioStage1Error(
            "created Portfolio S1 handoff or its source inputs changed "
            "during verification"
        )
    return CreatedPortfolioS1Handoff(
        output_dir=destination,
        trajectory_bundle_path=destination / PORTFOLIO_STAGE1_OUTPUT_FILES[0],
        creator_packet_path=destination / PORTFOLIO_STAGE1_OUTPUT_FILES[1],
        manifest_path=destination / PORTFOLIO_STAGE1_OUTPUT_FILES[2],
        trajectory_bundle_file_sha256=(
            observed_artifacts.manifest.trajectory_bundle_file_sha256
        ),
        creator_packet_file_sha256=(
            observed_artifacts.manifest.creator_packet_file_sha256
        ),
        manifest_file_sha256=manifest_file_sha256,
        selected_query_count=observed_artifacts.manifest.selected_query_count,
    )


def load_verified_portfolio_s1_handoff(
    output_dir: str | Path,
    *,
    expected_manifest_file_sha256: str,
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
    authoring_input_path: str | Path,
    expected_authoring_input_file_sha256: str,
) -> VerifiedPortfolioS1Handoff:
    """Load, re-derive, and stably re-read an externally pinned handoff."""

    expected_manifest = _require_sha256(
        expected_manifest_file_sha256,
        "expected Portfolio S1 manifest file SHA-256",
    )
    expected_authoring = _require_sha256(
        expected_authoring_input_file_sha256,
        "expected common authoring-input file SHA-256",
    )
    root = Path(output_dir).absolute()
    verified_inputs = require_verified_portfolio_dev_mini_inputs(inputs)
    verified_checklist = require_verified_portfolio_five_config_run_checklist(
        checklist,
        inputs=verified_inputs,
    )
    authoring_path, authoring_content, authoring_input = (
        _load_common_authoring_input(
            authoring_input_path,
            expected_file_sha256=expected_authoring,
        )
    )
    contents = _read_exact_handoff_files(root)
    if sha256_bytes(contents[2]) != expected_manifest:
        raise PortfolioStage1Error(
            "Portfolio S1 handoff manifest external digest mismatch"
        )
    observed = _parse_artifacts(contents)
    rebuilt = _build_artifacts(
        verified_inputs,
        verified_checklist,
        authoring_input=authoring_input,
        authoring_input_file_sha256=sha256_bytes(authoring_content),
    )
    if observed != rebuilt or _artifact_contents(rebuilt) != contents:
        raise PortfolioStage1Error(
            "Portfolio S1 handoff differs from currently verified inputs"
        )

    require_verified_portfolio_dev_mini_inputs(verified_inputs)
    require_verified_portfolio_five_config_run_checklist(
        verified_checklist,
        inputs=verified_inputs,
    )
    _, final_authoring_content, final_authoring_input = (
        _load_common_authoring_input(
            authoring_path,
            expected_file_sha256=expected_authoring,
        )
    )
    if (
        final_authoring_content != authoring_content
        or final_authoring_input != authoring_input
        or _read_exact_handoff_files(root) != contents
    ):
        raise PortfolioStage1Error(
            "Portfolio S1 handoff inputs or outputs changed during verified load"
        )
    return VerifiedPortfolioS1Handoff(
        output_dir=root,
        authoring_input_path=authoring_path,
        expected_authoring_input_file_sha256=expected_authoring,
        expected_manifest_file_sha256=expected_manifest,
        trajectory_bundle=observed.trajectory_bundle,
        creator_packet=observed.creator_packet,
        manifest=observed.manifest,
        _contents=contents,
        _marker=_VERIFIED_HANDOFF_MARKER,
    )


def require_verified_portfolio_s1_handoff(
    value: object,
    *,
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
) -> VerifiedPortfolioS1Handoff:
    """Reject forged, stale, mutated, or no-longer-reproducible handles."""

    if (
        type(value) is not VerifiedPortfolioS1Handoff
        or value._marker is not _VERIFIED_HANDOFF_MARKER
    ):
        raise TypeError(
            "Portfolio S1 handoff requires the external-digest verified loader"
        )
    current = load_verified_portfolio_s1_handoff(
        value.output_dir,
        expected_manifest_file_sha256=value.expected_manifest_file_sha256,
        inputs=inputs,
        checklist=checklist,
        authoring_input_path=value.authoring_input_path,
        expected_authoring_input_file_sha256=(
            value.expected_authoring_input_file_sha256
        ),
    )
    held = (
        value.output_dir,
        value.authoring_input_path,
        value.expected_authoring_input_file_sha256,
        value.expected_manifest_file_sha256,
        value.trajectory_bundle,
        value.creator_packet,
        value.manifest,
        value._contents,
    )
    observed = (
        current.output_dir,
        current.authoring_input_path,
        current.expected_authoring_input_file_sha256,
        current.expected_manifest_file_sha256,
        current.trajectory_bundle,
        current.creator_packet,
        current.manifest,
        current._contents,
    )
    if observed != held:
        raise PortfolioStage1Error(
            "verified Portfolio S1 handoff handle was mutated or drifted"
        )
    return value


def _build_artifacts(
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
    *,
    authoring_input: CodexAuthoringInput,
    authoring_input_file_sha256: str,
) -> PortfolioS1HandoffArtifacts:
    selected, smoke_query = _select_default_queries(inputs, checklist)
    asset_by_query_id = {item.query_id: item for item in inputs.query_assets}
    if len(asset_by_query_id) != len(inputs.query_assets):
        raise PortfolioStage1Error(
            "verified Portfolio query-asset bindings are duplicated"
        )
    trajectories: list[PortfolioUserTrajectory] = []
    for query in selected:
        try:
            asset = asset_by_query_id[query.query_id]
        except KeyError as error:
            raise PortfolioStage1Error(
                f"selected Portfolio Stage 1 query lacks an image binding: "
                f"{query.query_id}"
            ) from error
        if asset.asset_id != query.asset_id or asset.image_path != query.image_path:
            raise PortfolioStage1Error(
                f"selected Portfolio Stage 1 image binding drifted: "
                f"{query.query_id}"
            )
        trajectories.append(
            _project_trajectory(query, image_sha256=asset.image_sha256)
        )
    trajectory_values = tuple(trajectories)
    selected_query_ids = tuple(item.query_id for item in trajectory_values)
    checklist_value = checklist.checklist
    bundle_unsigned = {
        "schema_version": 1,
        "artifact_kind": "portfolio-dev-mini-trajectory-bundle",
        "policy_version": PORTFOLIO_STAGE1_POLICY_VERSION,
        "track": "portfolio",
        "purpose": PORTFOLIO_STAGE1_PURPOSE,
        "status": "prepared_not_invoked",
        "source_split": "dev_mini",
        "formal_eligible": False,
        "formal_ineligible_reason": (
            PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
        ),
        "portfolio_plan_sha256": inputs.expected_plan_sha256,
        "accepted_ledger_sha256": inputs.expected_accepted_ledger_sha256,
        "query_artifact_sha256": inputs.expected_query_artifact_sha256,
        "capability_assignments_sha256": (
            inputs.expected_capability_assignments_sha256
        ),
        "seed_set_sha256": inputs.expected_seed_set_sha256,
        "base_catalog_sha256": inputs.expected_base_catalog_sha256,
        "runtime_catalog_sha256": inputs.expected_output_catalog_sha256,
        "run_checklist_file_sha256": checklist.expected_file_sha256,
        "run_checklist_sha256": checklist_value.checklist_sha256,
        "smoke_query_id": smoke_query.query_id,
        "smoke_query_sha256": checklist_value.selected_query.query_sha256,
        "smoke_public_input_sha256": (
            checklist_value.selected_query.public_input_sha256
        ),
        "excluded_smoke_leakage_group_id": smoke_query.leakage_group_id,
        "selection_policy": PORTFOLIO_STAGE1_SELECTION_POLICY,
        "selected_query_ids": selected_query_ids,
        "taxonomy_version": selected[0].taxonomy_version,
        "task_spec_version": selected[0].task_spec_version,
        "trajectories": trajectory_values,
        "model_calls_performed": 0,
        "creator_invoked": False,
        "bank_generated": False,
        "execution_authorized": False,
    }
    bundle = PortfolioDevMiniTrajectoryBundle.model_validate(
        {
            **bundle_unsigned,
            "trajectory_bundle_sha256": _hash_payload(bundle_unsigned),
        },
        strict=True,
    )

    packet_unsigned = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-creator-packet",
        "policy_version": PORTFOLIO_STAGE1_POLICY_VERSION,
        "track": "portfolio",
        "purpose": PORTFOLIO_STAGE1_PURPOSE,
        "status": "prepared_not_invoked",
        "formal_eligible": False,
        "formal_ineligible_reason": (
            PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
        ),
        "authoring_input_file_sha256": authoring_input_file_sha256,
        "authoring_input": authoring_input,
        "trajectory_bundle": bundle,
        "model_calls_performed": 0,
        "creator_invoked": False,
        "bank_generated": False,
        "execution_authorized": False,
    }
    packet = PortfolioS1CreatorPacket.model_validate(
        {**packet_unsigned, "packet_sha256": _hash_payload(packet_unsigned)},
        strict=True,
    )
    bundle_bytes = bundle.canonical_bytes()
    packet_bytes = packet.canonical_bytes()
    manifest_unsigned = {
        "schema_version": 1,
        "artifact_kind": "portfolio-s1-handoff-manifest",
        "policy_version": PORTFOLIO_STAGE1_POLICY_VERSION,
        "track": "portfolio",
        "purpose": PORTFOLIO_STAGE1_PURPOSE,
        "status": "prepared_not_invoked",
        "formal_eligible": False,
        "formal_ineligible_reason": (
            PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON
        ),
        "artifact_files": PORTFOLIO_STAGE1_OUTPUT_FILES,
        "portfolio_plan_sha256": bundle.portfolio_plan_sha256,
        "accepted_ledger_sha256": bundle.accepted_ledger_sha256,
        "query_artifact_sha256": bundle.query_artifact_sha256,
        "capability_assignments_sha256": (
            bundle.capability_assignments_sha256
        ),
        "seed_set_sha256": bundle.seed_set_sha256,
        "base_catalog_sha256": bundle.base_catalog_sha256,
        "runtime_catalog_sha256": bundle.runtime_catalog_sha256,
        "run_checklist_file_sha256": bundle.run_checklist_file_sha256,
        "run_checklist_sha256": bundle.run_checklist_sha256,
        "authoring_input_file_sha256": authoring_input_file_sha256,
        "authoring_input_sha256": authoring_input.input_sha256,
        "selection_policy": PORTFOLIO_STAGE1_SELECTION_POLICY,
        "selected_query_ids": bundle.selected_query_ids,
        "selected_query_count": 5,
        "trajectory_bundle_sha256": bundle.trajectory_bundle_sha256,
        "trajectory_bundle_file_sha256": sha256_bytes(bundle_bytes),
        "creator_packet_sha256": packet.packet_sha256,
        "creator_packet_file_sha256": sha256_bytes(packet_bytes),
        "model_calls_performed": 0,
        "creator_invoked": False,
        "bank_generated": False,
        "execution_authorized": False,
    }
    manifest = PortfolioS1HandoffManifest.model_validate(
        {**manifest_unsigned, "manifest_sha256": _hash_payload(manifest_unsigned)},
        strict=True,
    )
    artifacts = PortfolioS1HandoffArtifacts(
        trajectory_bundle=bundle,
        creator_packet=packet,
        manifest=manifest,
    )
    _validate_artifact_bindings(artifacts)
    return artifacts


def _select_default_queries(
    inputs: VerifiedPortfolioDevMiniInputs,
    checklist: VerifiedPortfolioFiveConfigRunChecklist,
) -> tuple[tuple[Query, ...], Query]:
    checklist_value = checklist.checklist
    if (
        checklist_value.track != "portfolio"
        or checklist_value.status != "planned_no_model_calls"
        or checklist_value.model_calls_performed != 0
        or checklist_value.execution_authorized is not False
        or checklist_value.formal_eligible is not False
        or checklist_value.planned_query_count != 1
    ):
        raise PortfolioStage1Error(
            "Portfolio S1 requires the blocked zero-call 1x5 checklist"
        )
    by_id = {item.query_id: item for item in inputs.queries}
    if len(by_id) != len(inputs.queries):
        raise PortfolioStage1Error("verified Portfolio query IDs are duplicated")
    smoke_query_id = checklist_value.selected_query.query_id
    try:
        smoke_query = by_id[smoke_query_id]
    except KeyError as error:
        raise PortfolioStage1Error(
            "smoke checklist query is absent from verified Portfolio inputs"
        ) from error
    if smoke_query.split != "dev_mini":
        raise PortfolioStage1Error("smoke query is not a Portfolio dev_mini query")

    excluded_group = smoke_query.leakage_group_id
    selected: list[Query] = []
    used_groups = {excluded_group}
    for intent in PORTFOLIO_STAGE1_INTENT_ORDER:
        candidates = sorted(
            (
                query
                for query in inputs.queries
                if query.split == "dev_mini"
                and query.canonical_intent == intent
                and query.is_boundary is False
                and len(query.turns) == 1
                and query.leakage_group_id not in used_groups
            ),
            key=lambda query: query.query_id,
        )
        if not candidates:
            raise PortfolioStage1Error(
                f"no safe default Portfolio Stage 1 query for intent: {intent}"
            )
        selected.append(candidates[0])
        used_groups.add(candidates[0].leakage_group_id)

    ordered = tuple(sorted(selected, key=lambda query: query.query_id))
    selected_ids = tuple(query.query_id for query in ordered)
    if selected_ids != PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS:
        raise PortfolioStage1Error(
            "current Portfolio snapshot no longer reproduces the default "
            "five-intent selection"
        )
    if any(
        query.leakage_group_id == excluded_group for query in ordered
    ) or len({query.leakage_group_id for query in ordered}) != len(ordered):
        raise PortfolioStage1Error(
            "Portfolio Stage 1 selection crosses the excluded leakage component"
        )
    versions = {
        (query.taxonomy_version, query.task_spec_version) for query in ordered
    }
    if len(versions) != 1:
        raise PortfolioStage1Error(
            "Portfolio Stage 1 selected query specification versions drifted"
        )
    return ordered, smoke_query


def _project_trajectory(
    query: Query,
    *,
    image_sha256: str,
) -> PortfolioUserTrajectory:
    return PortfolioUserTrajectory(
        query_id=query.query_id,
        asset_id=query.asset_id,
        image_path=query.image_path,
        image_sha256=image_sha256,
        turns=tuple(
            PortfolioTrajectoryTurn(role=turn.role, content=turn.content)
            for turn in query.turns
        ),
        canonical_intent=query.canonical_intent,
        canonical_capability=query.canonical_capability,
    )


def _load_common_authoring_input(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> tuple[Path, bytes, CodexAuthoringInput]:
    expected = _require_sha256(
        expected_file_sha256,
        "expected common authoring-input file SHA-256",
    )
    authoring_path = Path(path).absolute()
    try:
        content = read_stable_regular_file(
            authoring_path,
            label="Portfolio S1 common Codex authoring input",
            max_bytes=_MAX_AUTHORING_INPUT_BYTES,
        )
        if sha256_bytes(content) != expected:
            raise PortfolioStage1Error(
                "common Codex authoring-input external digest mismatch"
            )
        value = load_codex_authoring_input(
            authoring_path,
            expected_file_sha256=expected,
        )
    except PortfolioStage1Error:
        raise
    except (
        ArtifactFormatError,
        CodexAuthoringContractError,
        OSError,
        ValueError,
    ) as error:
        raise PortfolioStage1Error(
            "common Codex authoring input verification failed"
        ) from error
    if content != value.canonical_bytes():
        raise PortfolioStage1Error(
            "common Codex authoring input is not canonical byte-identical"
        )
    if value.consumers != ("llm_static", "s1"):
        raise PortfolioStage1Error(
            "common Codex authoring input consumers must be exactly "
            "llm_static and s1"
        )
    return authoring_path, content, value


def _artifact_contents(
    artifacts: PortfolioS1HandoffArtifacts,
) -> tuple[bytes, bytes, bytes]:
    return (
        artifacts.trajectory_bundle.canonical_bytes(),
        artifacts.creator_packet.canonical_bytes(),
        artifacts.manifest.canonical_bytes(),
    )


def _validate_artifact_bindings(
    artifacts: PortfolioS1HandoffArtifacts,
) -> None:
    bundle = artifacts.trajectory_bundle
    packet = artifacts.creator_packet
    manifest = artifacts.manifest
    bundle_bytes, packet_bytes, _ = _artifact_contents(artifacts)
    if packet.trajectory_bundle != bundle:
        raise PortfolioStage1Error(
            "Portfolio S1 packet embeds a different trajectory bundle"
        )
    if (
        manifest.portfolio_plan_sha256 != bundle.portfolio_plan_sha256
        or manifest.accepted_ledger_sha256 != bundle.accepted_ledger_sha256
        or manifest.query_artifact_sha256 != bundle.query_artifact_sha256
        or manifest.capability_assignments_sha256
        != bundle.capability_assignments_sha256
        or manifest.seed_set_sha256 != bundle.seed_set_sha256
        or manifest.base_catalog_sha256 != bundle.base_catalog_sha256
        or manifest.runtime_catalog_sha256 != bundle.runtime_catalog_sha256
        or manifest.run_checklist_file_sha256
        != bundle.run_checklist_file_sha256
        or manifest.run_checklist_sha256 != bundle.run_checklist_sha256
        or manifest.authoring_input_file_sha256
        != packet.authoring_input_file_sha256
        or manifest.authoring_input_sha256
        != packet.authoring_input.input_sha256
        or manifest.selected_query_ids != bundle.selected_query_ids
        or manifest.trajectory_bundle_sha256
        != bundle.trajectory_bundle_sha256
        or manifest.trajectory_bundle_file_sha256
        != sha256_bytes(bundle_bytes)
        or manifest.creator_packet_sha256 != packet.packet_sha256
        or manifest.creator_packet_file_sha256 != sha256_bytes(packet_bytes)
    ):
        raise PortfolioStage1Error(
            "Portfolio S1 handoff manifest bindings drifted"
        )


ModelT = TypeVar("ModelT", bound=BaseModel)


def _parse_model(content: bytes, model_type: type[ModelT], label: str) -> ModelT:
    try:
        raw = parse_canonical_json(content, label=label)
        if not isinstance(raw, dict):
            raise PortfolioStage1Error(f"{label} must contain an object")
        value = model_type.model_validate(raw, strict=True)
    except PortfolioStage1Error:
        raise
    except (ArtifactFormatError, ValidationError, ValueError) as error:
        raise PortfolioStage1Error(f"{label} is invalid") from error
    canonical = canonical_json_bytes(value.model_dump(mode="json"))
    if canonical != content:
        raise PortfolioStage1Error(f"{label} is not canonical")
    return value


def _parse_artifacts(
    contents: tuple[bytes, bytes, bytes],
) -> PortfolioS1HandoffArtifacts:
    artifacts = PortfolioS1HandoffArtifacts(
        trajectory_bundle=_parse_model(
            contents[0],
            PortfolioDevMiniTrajectoryBundle,
            "Portfolio Stage 1 trajectory bundle",
        ),
        creator_packet=_parse_model(
            contents[1],
            PortfolioS1CreatorPacket,
            "Portfolio S1 Creator packet",
        ),
        manifest=_parse_model(
            contents[2],
            PortfolioS1HandoffManifest,
            "Portfolio S1 handoff manifest",
        ),
    )
    _validate_artifact_bindings(artifacts)
    return artifacts


def _read_exact_handoff_files(
    root: Path,
) -> tuple[bytes, bytes, bytes]:
    _require_real_directory(root, "Portfolio S1 handoff directory")
    try:
        children = tuple(root.iterdir())
    except OSError as error:
        raise PortfolioStage1Error(
            "Portfolio S1 handoff directory cannot be listed"
        ) from error
    observed_names = tuple(sorted(child.name for child in children))
    if observed_names != tuple(sorted(PORTFOLIO_STAGE1_OUTPUT_FILES)):
        raise PortfolioStage1Error(
            "Portfolio S1 handoff must contain exactly the three committed files"
        )
    for child in children:
        try:
            metadata = child.lstat()
        except OSError as error:
            raise PortfolioStage1Error(
                f"Portfolio S1 handoff file cannot be inspected: {child.name}"
            ) from error
        reparse = int(getattr(metadata, "st_file_attributes", 0)) & int(
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or reparse
        ):
            raise PortfolioStage1Error(
                f"Portfolio S1 handoff entry is not a regular file: {child.name}"
            )
    try:
        return tuple(
            read_stable_regular_file(
                root / name,
                label=f"Portfolio S1 handoff {name}",
                max_bytes=_MAX_HANDOFF_FILE_BYTES,
            )
            for name in PORTFOLIO_STAGE1_OUTPUT_FILES
        )
    except ArtifactFormatError as error:
        raise PortfolioStage1Error(
            "Portfolio S1 handoff file verification failed"
        ) from error


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PortfolioStage1Error(f"{label} cannot be inspected: {path}") from error
    reparse = int(getattr(metadata, "st_file_attributes", 0)) & int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or reparse:
        raise PortfolioStage1Error(
            f"{label} must be a real non-reparse directory: {path}"
        )


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


__all__ = [
    "PORTFOLIO_STAGE1_DEFAULT_QUERY_IDS",
    "PORTFOLIO_STAGE1_FORMAL_INELIGIBLE_REASON",
    "PORTFOLIO_STAGE1_OUTPUT_FILES",
    "PORTFOLIO_STAGE1_POLICY_VERSION",
    "PORTFOLIO_STAGE1_PURPOSE",
    "PORTFOLIO_STAGE1_SELECTION_POLICY",
    "CreatedPortfolioS1Handoff",
    "PortfolioDevMiniTrajectoryBundle",
    "PortfolioS1CreatorPacket",
    "PortfolioS1HandoffArtifacts",
    "PortfolioS1HandoffManifest",
    "PortfolioStage1Error",
    "PortfolioTrajectoryTurn",
    "PortfolioUserTrajectory",
    "VerifiedPortfolioS1Handoff",
    "build_portfolio_s1_handoff",
    "create_portfolio_s1_handoff",
    "load_verified_portfolio_s1_handoff",
    "require_verified_portfolio_s1_handoff",
]
