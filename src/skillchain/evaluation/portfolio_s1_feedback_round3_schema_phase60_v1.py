"""Forward-only phase60 authority over the frozen Round3 schema canary.

The canary identity is terminal and immutable: its three retry claims are spent.
This module imports only the twelve final parsed canary artifacts as a verified
prefix, then gives ordinals 13..60 a new call ledger and a separately approved
pool of at most twelve retry claims.  Nothing in this module can publish a
Feedback bundle or authorize the S1 Creator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
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

from skillchain import config
from skillchain.evaluation.packets import FeedbackPacketV3
from skillchain.evaluation.portfolio_s1_feedback import (
    PortfolioS1FeedbackError,
    PortfolioS1FeedbackSelectionV2,
    VerifiedStaticFeedbackSourceV2,
    _hash_payload,
    _model_hash,
    _nested_json_model,
    require_verified_static_feedback_source_v2,
)
from skillchain.evaluation.portfolio_s1_feedback_recovery_v1 import (
    PARENT_SELECTION_FILE_SHA256,
    PARENT_SELECTION_SHA256,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1,
    ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1,
    RecoveryFeedbackEvaluationResultV2,
    redact_recovery_result_for_creator_privacy_v1,
)
from skillchain.evaluation.portfolio_s1_feedback_round3_schema_v1 import (
    BoundRound3SchemaFeedbackArtifactV2,
    PortfolioS1FeedbackBundleV11Identity,
    PortfolioS1FeedbackRound3AuthorizationV2,
    PortfolioS1FeedbackRound3ControlV2,
    PortfolioS1FeedbackRound3LaunchV2,
    PortfolioS1FeedbackRound3SchemaLedgerV2,
    PortfolioS1FeedbackRound3SchemaRunV2,
    PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
    VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    build_round3_schema_run_v2,
    is_round3_schema_retry_eligible_v1,
    load_round3_schema_ledger_v1,
    load_round3_schema_run_v2,
)
from skillchain.evaluation.portfolio_s1_qwen_governance import (
    QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8,
    QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8,
    QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8,
    QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15,
    QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V15,
    QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15,
    QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5,
    QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5,
    QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5,
    QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
    QWEN38_FEEDBACK_MODEL,
    QWEN38_FEEDBACK_PROVIDER,
    Qwen38FeedbackModelSourceLockV5,
    Qwen38FeedbackPricingLockV8,
    Qwen38FeedbackRoleSelectionV15,
    ROUND3_PHASE60_RETRY_POLICY_SHA256_V1,
    ROUND3_PHASE60_RETRY_POLICY_VERSION_V1,
    load_qwen38_feedback_model_source_lock_v5,
    load_qwen38_feedback_pricing_lock_v8,
    load_qwen38_feedback_role_selection_v15,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1 = (
    "specs/authoring/portfolio-s1-feedback-round3-schema-canary12-result-v1.json"
)
# Frozen after the tracked manifest is generated.  A missing trust root fails
# closed, so the module cannot authorize phase60 merely because a local canary
# directory happens to look plausible.
ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1 = (
    "71d7e8805c2e26237ce2f135929b32331faa27ccadcebd0e202f770c0b6f9ee0"
)

ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS = tuple(range(1, 13))
ROUND3_SCHEMA_PHASE60_NEW_ORDINALS = tuple(range(13, 61))
ROUND3_SCHEMA_PHASE60_TARGET_COUNT: Literal[60] = 60
ROUND3_SCHEMA_PHASE60_NEW_PRIMARY_COUNT: Literal[48] = 48
ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING: Literal[12] = 12
ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING: Literal[60] = 60
ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY = "0.461544000000"
ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY = "26.264100000000"
ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY = "27.692640000000"
ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY = "53.956740000000"
ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY = "28.000000000000"
ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY = "54.264100000000"

ROUND3_SCHEMA_PHASE60_PREFIX_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-canary12-result-v1"
)
ROUND3_SCHEMA_PHASE60_OWNER_APPROVAL_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-owner-approval-v1"
)
ROUND3_SCHEMA_PHASE60_AUTHORIZATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-authorization-v1"
)
ROUND3_SCHEMA_PHASE60_CONTROL_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-control-v1"
)
ROUND3_SCHEMA_PHASE60_LAUNCH_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-launch-v1"
)
ROUND3_SCHEMA_PHASE60_CLAIM_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-retry-claim-v1"
)
ROUND3_SCHEMA_PHASE60_RESERVATION_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-reservation-v1"
)
ROUND3_SCHEMA_PHASE60_BOUND_POLICY_VERSION_V1 = (
    "portfolio-s1-bound-feedback-round3-schema-phase60-v1"
)
ROUND3_SCHEMA_PHASE60_RUN_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-schema-phase60-run-v1"
)

ROUND3_SCHEMA_PHASE60_CLAIM_DIR = "global-retry-claims-round3-schema-phase60-v1"
ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR = "provider-attempts-round3-schema-phase60-v1"
ROUND3_SCHEMA_PHASE60_BOUND_DIR = "bound-feedback-round3-schema-phase60-v1"

_CANARY_TOP_FILES = frozenset(
    {
        "authorization-round3-schema-v2.json",
        "bundle-v11-identity.json",
        "control-round3-schema-v2.json",
        "launch-round3-schema-v2.json",
        "predecessors-round3-schema-v2.json",
        "remote-runtime-receipt-round3-schema-v2.json",
        "run-round3-schema-v2.json",
        "selection-v2.json",
    }
)
_CANARY_DYNAMIC_DIRS = frozenset(
    {
        "global-retry-claims-round3-schema-v1",
        "provider-attempts-round3-schema-v1",
        "bound-feedback-round3-schema-v1",
    }
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Phase60Status = Literal["parsed", "parse_error", "provider_error", "timeout"]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _load_canonical_model(
    path: str | Path,
    *,
    model_type: type[BaseModel],
    label: str,
    max_bytes: int = 128 * 1024 * 1024,
) -> BaseModel:
    content = read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    model = model_type.model_validate_json(content, strict=True)
    if canonical_json_bytes(model.model_dump(mode="json")) != content:
        raise PortfolioS1FeedbackError(f"{label} is not canonical")
    return model


def _inventory_sha256(directory: Path, *, label: str) -> str:
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} is missing or unsafe")
    members = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in members
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory is unsafe")
    return _hash_payload(
        [
            {
                "name": item.name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        item, label=f"{label} {item.name}", max_bytes=64 * 1024 * 1024
                    )
                ),
            }
            for item in members
        ]
    )


def _top_file_inventory_sha256(root: Path) -> str:
    return _hash_payload(
        [
            {
                "name": name,
                "file_sha256": sha256_bytes(
                    read_stable_regular_file(
                        root / name,
                        label=f"schema canary prefix {name}",
                        max_bytes=128 * 1024 * 1024,
                    )
                ),
            }
            for name in sorted(_CANARY_TOP_FILES)
        ]
    )


class Round3SchemaCanary12ArtifactReferenceV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=1, le=12)
    selection_entry_sha256: Sha256
    final_attempt_index: Literal[1, 2]
    final_global_call_ordinal: int = Field(ge=1, le=15)
    final_artifact_file_name: str
    final_artifact_file_sha256: Sha256
    artifact_sha256: Sha256
    feedback_result_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reference(self) -> Self:
        expected = (
            f"{self.final_global_call_ordinal:04d}-"
            f"{self.selection_entry_sha256[:16]}-"
            f"attempt-{self.final_attempt_index}.json"
        )
        if self.final_artifact_file_name != expected:
            raise ValueError("canary final artifact filename drifted")
        return self


class PortfolioS1FeedbackRound3SchemaCanary12ResultManifestV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-canary12-result"] = (
        "portfolio-s1-feedback-round3-schema-canary12-result"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-canary12-result-v1"
    ] = ROUND3_SCHEMA_PHASE60_PREFIX_POLICY_VERSION_V1
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    predecessor_receipt_file_sha256: Sha256
    predecessor_receipt_sha256: Sha256
    authorization_file_sha256: Sha256
    authorization_sha256: Sha256
    remote_receipt_file_sha256: Sha256
    remote_receipt_sha256: Sha256
    control_file_sha256: Sha256
    control_sha256: Sha256
    launch_file_sha256: Sha256
    launch_sha256: Sha256
    bundle_identity_file_sha256: Sha256
    bundle_identity_sha256: Sha256
    run_file_sha256: Sha256
    run_sha256: Sha256
    artifact_set_sha256: Sha256
    top_file_inventory_sha256: Sha256
    claim_file_inventory_sha256: Sha256
    reservation_file_inventory_sha256: Sha256
    bound_file_inventory_sha256: Sha256
    canary_root_file_count: Literal[41] = 41
    top_json_count: Literal[8] = 8
    claim_file_count: Literal[3] = 3
    reservation_file_count: Literal[15] = 15
    bound_file_count: Literal[15] = 15
    attempted_count: Literal[12] = 12
    parsed_count: Literal[12] = 12
    provider_calls_reserved: Literal[15] = 15
    retry_count: Literal[3] = 3
    orphan_count: Literal[0] = 0
    new_actual_cost_cny: Literal["1.944600000000"] = "1.944600000000"
    cumulative_actual_cost_cny: Literal[ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY] = (
        ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    )
    retry_claim_sha256s: tuple[Sha256, ...]
    final_artifacts: tuple[Round3SchemaCanary12ArtifactReferenceV1, ...]
    manifest_sha256: Sha256

    @field_validator("retry_claim_sha256s", "final_artifacts", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if (
            len(self.retry_claim_sha256s) != 3
            or len(set(self.retry_claim_sha256s)) != 3
            or tuple(item.selection_ordinal for item in self.final_artifacts)
            != ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
            or len({item.selection_entry_sha256 for item in self.final_artifacts}) != 12
            or len({item.artifact_sha256 for item in self.final_artifacts}) != 12
            or any(
                item.final_attempt_index not in {1, 2} for item in self.final_artifacts
            )
            or self.manifest_sha256 != _model_hash(self, "manifest_sha256")
        ):
            raise ValueError("schema canary12 result manifest drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1:
    repository_root: Path
    canary_root: Path
    manifest: PortfolioS1FeedbackRound3SchemaCanary12ResultManifestV1
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1
    selection: PortfolioS1FeedbackSelectionV2
    authorization: PortfolioS1FeedbackRound3AuthorizationV2
    remote_receipt: PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2
    control: PortfolioS1FeedbackRound3ControlV2
    launch: PortfolioS1FeedbackRound3LaunchV2
    bundle_identity: PortfolioS1FeedbackBundleV11Identity
    run: PortfolioS1FeedbackRound3SchemaRunV2
    ledger: PortfolioS1FeedbackRound3SchemaLedgerV2
    final_artifacts: tuple[BoundRound3SchemaFeedbackArtifactV2, ...]
    _marker: object | None = field(default=None, repr=False, compare=False)


_VERIFIED_CANARY_PREFIX_TOKEN = object()


def load_verified_round3_schema_canary_prefix_v1(
    repository_root: str | Path,
    canary_root: str | Path,
    *,
    predecessors: VerifiedPortfolioS1FeedbackRound3SchemaPredecessorsV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaGovernanceV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1:
    """Deep-verify the tracked result manifest and its external canary bytes."""

    repository = Path(repository_root).resolve(strict=True)
    manifest_path = repository / ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1
    manifest_content = read_stable_regular_file(
        manifest_path,
        label="schema canary12 result manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if (
        manifest_path.is_symlink()
        or sha256_bytes(manifest_content)
        != ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ):
        raise PortfolioS1FeedbackError(
            "schema canary12 result manifest file hash drifted"
        )
    manifest = (
        PortfolioS1FeedbackRound3SchemaCanary12ResultManifestV1.model_validate_json(
            manifest_content, strict=True
        )
    )
    if manifest.canonical_bytes() != manifest_content:
        raise PortfolioS1FeedbackError(
            "schema canary12 result manifest is not canonical"
        )

    supplied = Path(canary_root)
    if supplied.is_symlink():
        raise PortfolioS1FeedbackError("schema canary prefix root is a symlink")
    root = supplied.resolve(strict=True)
    children = tuple(root.iterdir())
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in children if item.is_file()} != _CANARY_TOP_FILES
        or {item.name for item in children if item.is_dir()} != _CANARY_DYNAMIC_DIRS
        or any(item.is_symlink() for item in children)
        or any(not item.is_file() and not item.is_dir() for item in children)
    ):
        raise PortfolioS1FeedbackError("schema canary prefix root inventory drifted")
    all_files = tuple(item for item in root.rglob("*") if item.is_file())
    if len(all_files) != manifest.canary_root_file_count or any(
        item.is_symlink() for item in all_files
    ):
        raise PortfolioS1FeedbackError("schema canary prefix file count drifted")

    selection = _load_canonical_model(
        root / "selection-v2.json",
        model_type=PortfolioS1FeedbackSelectionV2,
        label="schema canary selection",
    )
    authorization = _load_canonical_model(
        root / "authorization-round3-schema-v2.json",
        model_type=PortfolioS1FeedbackRound3AuthorizationV2,
        label="schema canary authorization",
    )
    remote_receipt = _load_canonical_model(
        root / "remote-runtime-receipt-round3-schema-v2.json",
        model_type=PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2,
        label="schema canary remote receipt",
    )
    control = _load_canonical_model(
        root / "control-round3-schema-v2.json",
        model_type=PortfolioS1FeedbackRound3ControlV2,
        label="schema canary control",
    )
    launch = _load_canonical_model(
        root / "launch-round3-schema-v2.json",
        model_type=PortfolioS1FeedbackRound3LaunchV2,
        label="schema canary launch",
    )
    bundle_identity = _load_canonical_model(
        root / "bundle-v11-identity.json",
        model_type=PortfolioS1FeedbackBundleV11Identity,
        label="schema canary reserved BundleV11 identity",
    )
    assert isinstance(selection, PortfolioS1FeedbackSelectionV2)
    assert isinstance(authorization, PortfolioS1FeedbackRound3AuthorizationV2)
    assert isinstance(
        remote_receipt, PortfolioS1Qwen38FeedbackRound3RemoteRuntimeReceiptV2
    )
    assert isinstance(control, PortfolioS1FeedbackRound3ControlV2)
    assert isinstance(launch, PortfolioS1FeedbackRound3LaunchV2)
    assert isinstance(bundle_identity, PortfolioS1FeedbackBundleV11Identity)
    run = load_round3_schema_run_v2(root / "run-round3-schema-v2.json")
    ledger = load_round3_schema_ledger_v1(
        root,
        predecessors=predecessors,
        governance=governance,
        authorization=authorization,
        control=control,
        remote_receipt=remote_receipt,
        sources=sources,
    )
    expected_run = build_round3_schema_run_v2(
        predecessors,
        governance,
        authorization,
        control,
        launch,
        remote_receipt,
        ledger,
        sources,
        terminal_reason=run.terminal_reason,
    )
    if run != expected_run:
        raise PortfolioS1FeedbackError("schema canary run differs from exact ledger")

    top_inventory = _top_file_inventory_sha256(root)
    claim_inventory = _inventory_sha256(
        root / "global-retry-claims-round3-schema-v1",
        label="schema canary claims",
    )
    reservation_inventory = _inventory_sha256(
        root / "provider-attempts-round3-schema-v1",
        label="schema canary reservations",
    )
    bound_inventory = _inventory_sha256(
        root / "bound-feedback-round3-schema-v1",
        label="schema canary bound artifacts",
    )
    final_by_ordinal: dict[int, BoundRound3SchemaFeedbackArtifactV2] = {}
    for artifact in ledger.artifacts:
        prior = final_by_ordinal.get(artifact.selection_ordinal)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_ordinal[artifact.selection_ordinal] = artifact
    final_artifacts = tuple(
        final_by_ordinal[ordinal] for ordinal in ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
    )
    references: list[Round3SchemaCanary12ArtifactReferenceV1] = []
    for artifact in final_artifacts:
        filename = (
            f"{artifact.global_call_ordinal:04d}-"
            f"{artifact.selection_entry_sha256[:16]}-"
            f"attempt-{artifact.attempt_index}.json"
        )
        references.append(
            Round3SchemaCanary12ArtifactReferenceV1(
                selection_ordinal=artifact.selection_ordinal,
                selection_entry_sha256=artifact.selection_entry_sha256,
                final_attempt_index=artifact.attempt_index,
                final_global_call_ordinal=artifact.global_call_ordinal,
                final_artifact_file_name=filename,
                final_artifact_file_sha256=sha256_bytes(
                    read_stable_regular_file(
                        root / "bound-feedback-round3-schema-v1" / filename,
                        label=f"schema canary final artifact {filename}",
                        max_bytes=16 * 1024 * 1024,
                    )
                ),
                artifact_sha256=artifact.artifact_sha256,
                feedback_result_sha256=artifact.feedback_result.result_sha256,
            )
        )

    file_sha = lambda name: sha256_bytes(  # noqa: E731
        read_stable_regular_file(
            root / name,
            label=f"schema canary {name}",
            max_bytes=128 * 1024 * 1024,
        )
    )
    if (
        selection != predecessors.prior.base.selection
        or any(item.status != "parsed" for item in final_artifacts)
        or manifest.selection_file_sha256 != file_sha("selection-v2.json")
        or manifest.predecessor_receipt_file_sha256
        != file_sha("predecessors-round3-schema-v2.json")
        or manifest.predecessor_receipt_sha256 != predecessors.receipt.receipt_sha256
        or manifest.authorization_file_sha256
        != file_sha("authorization-round3-schema-v2.json")
        or manifest.authorization_sha256 != authorization.authorization_sha256
        or manifest.remote_receipt_file_sha256
        != file_sha("remote-runtime-receipt-round3-schema-v2.json")
        or manifest.remote_receipt_sha256 != remote_receipt.receipt_sha256
        or manifest.control_file_sha256 != file_sha("control-round3-schema-v2.json")
        or manifest.control_sha256 != control.control_sha256
        or manifest.launch_file_sha256 != file_sha("launch-round3-schema-v2.json")
        or manifest.launch_sha256 != launch.launch_sha256
        or manifest.bundle_identity_file_sha256 != file_sha("bundle-v11-identity.json")
        or manifest.bundle_identity_sha256 != bundle_identity.identity_sha256
        or manifest.run_file_sha256 != file_sha("run-round3-schema-v2.json")
        or manifest.run_sha256 != run.run_sha256
        or manifest.artifact_set_sha256 != run.artifact_set_sha256
        or manifest.top_file_inventory_sha256 != top_inventory
        or manifest.claim_file_inventory_sha256 != claim_inventory
        or manifest.reservation_file_inventory_sha256 != reservation_inventory
        or manifest.bound_file_inventory_sha256 != bound_inventory
        or manifest.retry_claim_sha256s
        != tuple(item.claim_sha256 for item in ledger.claims)
        or manifest.final_artifacts != tuple(references)
        or run.status != "completed_canary"
        or run.attempted_count != 12
        or run.parsed_count != 12
        or run.provider_calls_reserved != 15
        or run.retry_count != 3
        or run.orphan_count != 0
        or run.new_actual_cost_cny != "1.944600000000"
        or run.cumulative_actual_cost_cny != ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
        or bundle_identity.live_canary_publishable
    ):
        raise PortfolioS1FeedbackError("schema canary prefix facts drifted")
    return VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1(
        repository_root=repository,
        canary_root=root,
        manifest=manifest,
        predecessors=predecessors,
        governance=governance,
        selection=selection,
        authorization=authorization,
        remote_receipt=remote_receipt,
        control=control,
        launch=launch,
        bundle_identity=bundle_identity,
        run=run,
        ledger=ledger,
        final_artifacts=final_artifacts,
        _marker=_VERIFIED_CANARY_PREFIX_TOKEN,
    )


@dataclass(frozen=True)
class VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1:
    repository_root: Path
    source_lock: Qwen38FeedbackModelSourceLockV5
    pricing_lock: Qwen38FeedbackPricingLockV8
    role_selection: Qwen38FeedbackRoleSelectionV15
    _marker: object | None = field(default=None, repr=False, compare=False)


_VERIFIED_PHASE60_GOVERNANCE_TOKEN = object()


def load_verified_round3_schema_phase60_governance_v1(
    repository_root: str | Path,
) -> VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1:
    """Load the exact V5/V8/V15 phase60 triad, including its live-call status."""

    root = Path(repository_root).resolve(strict=True)
    source = load_qwen38_feedback_model_source_lock_v5(
        root / QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5,
        expected_file_sha256=QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5,
    )
    pricing = load_qwen38_feedback_pricing_lock_v8(
        root / QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8,
        expected_file_sha256=QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8,
    )
    role = load_qwen38_feedback_role_selection_v15(
        root / QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V15,
        expected_file_sha256=QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15,
    )
    feedback = role.feedback_evaluator
    if (
        type(source) is not Qwen38FeedbackModelSourceLockV5
        or type(pricing) is not Qwen38FeedbackPricingLockV8
        or type(role) is not Qwen38FeedbackRoleSelectionV15
        or source.source_lock_sha256 != QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
        or pricing.pricing_lock_sha256 != QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
        or role.selection_sha256 != QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
        or source.outer_orchestration_policy_sha256
        != ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
        or pricing.outer_orchestration_policy_sha256
        != ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
        or feedback.get("outer_orchestration_policy_sha256")
        != ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
        or source.transport_policy_sha256
        != ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        or pricing.attempt_transport_policy_sha256
        != ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        or feedback.get("transport_policy_sha256")
        != ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        or source.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
        or pricing.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
        or feedback.get("requested_json_schema_sha256")
        != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    ):
        raise PortfolioS1FeedbackError("phase60 governance triad drifted")
    return VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1(
        repository_root=root,
        source_lock=source,
        pricing_lock=pricing,
        role_selection=role,
        _marker=_VERIFIED_PHASE60_GOVERNANCE_TOKEN,
    )


def require_round3_schema_phase60_live_governance_v1(
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
) -> None:
    """Require the exact owner-approved V5/V8/V15 phase60 live envelope."""

    if (
        type(governance)
        is not VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1
        or governance._marker is not _VERIFIED_PHASE60_GOVERNANCE_TOKEN
    ):
        raise PortfolioS1FeedbackError("phase60 governance is not verified")
    source = governance.source_lock
    pricing = governance.pricing_lock
    feedback = governance.role_selection.feedback_evaluator
    if (
        source.live_call_authority is not True
        or source.live_provider_calls_authorized is not True
        or source.live_call_authority_scope
        != "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
        or source.owner_phase60_budget_authorization_status != "granted"
        or source.owner_phase60_retry_authorization_status != "granted"
        or source.fresh_maximum_reservation_cny != "27.692640000000"
        or source.fresh_technical_hard_cap_cny != "28.000000000000"
        or source.cumulative_technical_hard_cap_cny != "54.264100000000"
        or source.phase60_requires_new_owner_approval is not False
        or source.phase120_requires_new_owner_approval is not True
        or pricing.fresh_run_and_retry_scope_owner_approved is not True
        or pricing.live_provider_calls_authorized is not True
        or pricing.live_authorized_selected_query_count != 60
        or pricing.live_authorized_phase_counts != (60,)
        or pricing.owner_budget_authorization_status
        != "granted_phase60_budget_and_retry_approval"
        or pricing.owner_budget_authorized_cap_cny != "28.000000000000"
        or pricing.maximum_reservation_cny != "27.692640000000"
        or pricing.technical_phase_hard_cap_cny != "28.000000000000"
        or pricing.live_cumulative_hard_cap_cny != "54.264100000000"
        or pricing.owner_phase60_retry_authorization_status != "granted"
        or pricing.phase60_requires_new_owner_approval is not False
        or pricing.phase120_requires_new_owner_approval is not True
        or feedback.get("live_call_authority") is not True
        or feedback.get("live_provider_calls_authorized") is not True
        or feedback.get("fresh_run_and_retry_scope_owner_approved") is not True
        or feedback.get("live_authorized_selected_query_count") != 60
        or feedback.get("live_authorized_phase_counts") != [60]
        or feedback.get("live_call_authority_scope")
        != "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
        or feedback.get("owner_phase60_budget_authorization_status") != "granted"
        or feedback.get("owner_phase60_retry_authorization_status") != "granted"
        or feedback.get("owner_authorized_budget_ceiling_cny") != "28.000000000000"
        or feedback.get("fresh_maximum_reservation_cny") != "27.692640000000"
        or feedback.get("fresh_stage_hard_cap_cny") != "28.000000000000"
        or feedback.get("live_cumulative_hard_cap_cny") != "54.264100000000"
        or feedback.get("phase60_requires_new_owner_approval") is not False
        or feedback.get("phase120_requires_new_owner_approval") is not True
    ):
        raise PortfolioS1FeedbackError(
            "phase60 governance does not match the exact owner-approved live envelope"
        )


class PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1(_StrictFrozenModel):
    """Separate owner grant required after the terminal canary spent its claims."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-owner-approval"] = (
        "portfolio-s1-feedback-round3-schema-phase60-owner-approval"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-phase60-owner-approval-v1"
    ] = ROUND3_SCHEMA_PHASE60_OWNER_APPROVAL_POLICY_VERSION_V1
    status: Literal["owner-approved"] = "owner-approved"
    approval_id: str
    reviewer_id: str
    reviewed_at: str
    owner_statement: Literal[
        "Approve phase60 only: reuse the frozen parsed canary12 prefix; add exactly "
        "48 first attempts for selection ordinals 13 through 60 and at most 12 "
        "new global same-entry retries with CNY27.692640 worst-case reservation "
        "under a CNY28 technical cap; do not publish BundleV11 or start the S1 "
        "Creator."
    ] = (
        "Approve phase60 only: reuse the frozen parsed canary12 prefix; add exactly "
        "48 first attempts for selection ordinals 13 through 60 and at most 12 "
        "new global same-entry retries with CNY27.692640 worst-case reservation "
        "under a CNY28 technical cap; do not publish BundleV11 or start the S1 "
        "Creator."
    )
    canary_manifest_file_sha256: Literal[
        ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ] = ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    canary_manifest_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
    )
    pricing_lock_file_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8
    )
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    imported_prefix_count: Literal[12] = 12
    phase_target_count: Literal[60] = 60
    new_first_attempt_count: Literal[48] = 48
    new_global_retry_ceiling: Literal[12] = 12
    new_provider_call_ceiling: Literal[60] = 60
    cumulative_provider_call_ceiling: Literal[75] = 75
    prior_retry_claims_consumed: Literal[3] = 3
    prior_retry_claims_reusable: Literal[False] = False
    per_call_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    approved_fresh_budget_cny: Literal[ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY] = (
        ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY
    )
    maximum_new_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    ] = ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    cumulative_maximum_cost_cny: Literal[
        ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    ] = ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    cumulative_technical_hard_cap_cny: Literal[
        ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY
    ] = ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY
    retry_policy_version: Literal[
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    ] = ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    response_format: Literal["json_schema"] = "json_schema"
    phase120_requires_new_owner_approval: Literal[True] = True
    bundle_v11_publish_authorized: Literal[False] = False
    s1_creator_start_authorized: Literal[False] = False
    approval_sha256: Sha256

    @model_validator(mode="after")
    def _validate_approval(self) -> Self:
        try:
            parsed = datetime.fromisoformat(self.reviewed_at)
        except ValueError as error:
            raise ValueError("phase60 reviewed_at is not ISO-8601") from error
        if (
            not self.approval_id
            or self.approval_id != self.approval_id.strip()
            or not self.reviewer_id
            or self.reviewer_id != self.reviewer_id.strip()
            or parsed.tzinfo is None
            or parsed.utcoffset() is None
            or parsed.microsecond
            or parsed.isoformat(timespec="seconds") != self.reviewed_at
            or Decimal(self.per_call_reservation_cny) * self.new_provider_call_ceiling
            != Decimal(self.maximum_new_reservation_cny)
            or Decimal(self.maximum_new_reservation_cny)
            >= Decimal(self.approved_fresh_budget_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_new_reservation_cny)
            != Decimal(self.cumulative_maximum_cost_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.approved_fresh_budget_cny)
            != Decimal(self.cumulative_technical_hard_cap_cny)
            or self.approval_sha256 != _model_hash(self, "approval_sha256")
        ):
            raise ValueError("phase60 owner approval drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_schema_phase60_owner_approval_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    *,
    approval_id: str,
    reviewer_id: str,
    reviewed_at: str,
) -> PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1:
    if (
        type(prefix) is not VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1
        or prefix._marker is not _VERIFIED_CANARY_PREFIX_TOKEN
    ):
        raise PortfolioS1FeedbackError("phase60 requires verified canary prefix")
    require_round3_schema_phase60_live_governance_v1(governance)
    draft = PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1.model_construct(
        approval_id=approval_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        approval_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"approval_sha256"})
    return PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1.model_validate(
        {**unsigned, "approval_sha256": _hash_payload(unsigned)}, strict=True
    )


class Round3SchemaPhase60EntryBindingV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=13, le=60)
    selection_entry_sha256: Sha256
    query_id: str
    asset_id: str
    image_sha256: Sha256


class PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-authorization"] = (
        "portfolio-s1-feedback-round3-schema-phase60-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-phase60-authorization-v1"
    ] = ROUND3_SCHEMA_PHASE60_AUTHORIZATION_POLICY_VERSION_V1
    status: Literal["owner-approved-phase60"] = "owner-approved-phase60"
    authorization_id: str
    reviewer_id: str
    reviewed_at: str
    owner_approval_sha256: Sha256
    owner_approval_file_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
    )
    pricing_lock_file_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8
    )
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
    )
    canary_manifest_file_sha256: Literal[
        ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ] = ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    canary_manifest_sha256: Sha256
    canary_run_file_sha256: Sha256
    canary_run_sha256: Sha256
    canary_artifact_set_sha256: Sha256
    selection_file_sha256: Literal[PARENT_SELECTION_FILE_SHA256] = (
        PARENT_SELECTION_FILE_SHA256
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    selected_entry_sha256s: tuple[Sha256, ...]
    selected_query_ids: tuple[str, ...]
    imported_prefix_ordinals: tuple[int, ...] = ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
    imported_prefix_artifact_sha256s: tuple[Sha256, ...]
    phase60_new_ordinals: tuple[int, ...] = ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
    phase60_entries: tuple[Round3SchemaPhase60EntryBindingV1, ...]
    phase60_entry_set_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    endpoint: str
    response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    requested_json_schema_strict: Literal[True] = True
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    wire_kind: Literal["round3_primary_json_schema_v1"] = (
        "round3_primary_json_schema_v1"
    )
    retry_policy_version: Literal[
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    ] = ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    imported_prefix_count: Literal[12] = 12
    phase_target_count: Literal[60] = 60
    new_first_attempt_count: Literal[48] = 48
    new_global_retry_ceiling: Literal[12] = 12
    new_provider_call_ceiling: Literal[60] = 60
    cumulative_provider_call_ceiling: Literal[75] = 75
    concurrency: Literal[2] = 2
    provider_internal_max_attempts: Literal[1] = 1
    max_lifetime_attempts_per_new_entry: Literal[2] = 2
    per_call_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    ] = ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    maximum_new_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    cumulative_maximum_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    ] = ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    membership_authorization_id: str
    membership_authorization_file_sha256: Sha256
    membership_receipt_file_sha256: Sha256
    membership_receipt_sha256: Sha256
    membership_catalog_sha256: Sha256
    canary_retry_claims_reused: Literal[0] = 0
    canary_provider_attempts_replayed: Literal[0] = 0
    phase120_requires_new_owner_approval: Literal[True] = True
    bundle_v11_publish_authorized: Literal[False] = False
    s1_creator_start_authorized: Literal[False] = False
    authorization_sha256: Sha256

    @field_validator(
        "selected_entry_sha256s",
        "selected_query_ids",
        "imported_prefix_ordinals",
        "imported_prefix_artifact_sha256s",
        "phase60_new_ordinals",
        "phase60_entries",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if (
            self.endpoint != config.PROVIDER_ENDPOINTS["qwen"]
            or self.provider != QWEN38_FEEDBACK_PROVIDER
            or self.model != QWEN38_FEEDBACK_MODEL
            or len(self.selected_entry_sha256s) != 240
            or len(set(self.selected_entry_sha256s)) != 240
            or len(self.selected_query_ids) != 240
            or len(set(self.selected_query_ids)) != 240
            or self.imported_prefix_ordinals != ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
            or len(self.imported_prefix_artifact_sha256s) != 12
            or len(set(self.imported_prefix_artifact_sha256s)) != 12
            or self.phase60_new_ordinals != ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
            or tuple(item.selection_ordinal for item in self.phase60_entries)
            != ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
            or len({item.query_id for item in self.phase60_entries}) != 48
            or len({item.asset_id for item in self.phase60_entries}) != 48
            or self.phase60_entry_set_sha256
            != _hash_payload(
                [item.model_dump(mode="json") for item in self.phase60_entries]
            )
            or Decimal(self.per_call_reservation_cny) * self.new_provider_call_ceiling
            != Decimal(self.maximum_new_reservation_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_new_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or self.authorization_sha256 != _model_hash(self, "authorization_sha256")
        ):
            raise ValueError("phase60 authorization drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_schema_phase60_authorization_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    authorization_id: str,
) -> PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1:
    if (
        type(prefix) is not VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1
        or prefix._marker is not _VERIFIED_CANARY_PREFIX_TOKEN
        or approval
        != build_round3_schema_phase60_owner_approval_v1(
            prefix,
            governance,
            approval_id=approval.approval_id,
            reviewer_id=approval.reviewer_id,
            reviewed_at=approval.reviewed_at,
        )
        or len(sources) != 240
        or tuple(item.selection_entry_sha256 for item in sources)
        != tuple(item.entry_sha256 for item in prefix.selection.entries)
    ):
        raise PortfolioS1FeedbackError("phase60 authority inputs drifted")
    for entry, source in zip(prefix.selection.entries, sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, prefix.selection, prefix.predecessors.prior.base.control, entry
        )
    entries = tuple(
        Round3SchemaPhase60EntryBindingV1(
            selection_ordinal=entry.selection_ordinal,
            selection_entry_sha256=entry.entry_sha256,
            query_id=entry.query_id,
            asset_id=entry.asset_id,
            image_sha256=entry.image_sha256,
        )
        for entry in prefix.selection.entries[12:60]
    )
    draft = PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1.model_construct(
        authorization_id=authorization_id,
        reviewer_id=approval.reviewer_id,
        reviewed_at=approval.reviewed_at,
        owner_approval_sha256=approval.approval_sha256,
        owner_approval_file_sha256=sha256_bytes(approval.canonical_bytes()),
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        canary_run_file_sha256=prefix.manifest.run_file_sha256,
        canary_run_sha256=prefix.run.run_sha256,
        canary_artifact_set_sha256=prefix.run.artifact_set_sha256,
        selected_entry_sha256s=tuple(
            item.entry_sha256 for item in prefix.selection.entries
        ),
        selected_query_ids=tuple(item.query_id for item in prefix.selection.entries),
        imported_prefix_artifact_sha256s=tuple(
            item.artifact_sha256 for item in prefix.final_artifacts
        ),
        phase60_entries=entries,
        phase60_entry_set_sha256=_hash_payload(
            [item.model_dump(mode="json") for item in entries]
        ),
        endpoint=config.PROVIDER_ENDPOINTS["qwen"],
        membership_authorization_id=prefix.authorization.authorization_id,
        membership_authorization_file_sha256=prefix.manifest.authorization_file_sha256,
        membership_receipt_file_sha256=prefix.manifest.remote_receipt_file_sha256,
        membership_receipt_sha256=prefix.remote_receipt.receipt_sha256,
        membership_catalog_sha256=prefix.remote_receipt.catalog_sha256,
        authorization_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    return PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1.model_validate(
        {**unsigned, "authorization_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1FeedbackRound3SchemaPhase60ControlV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-control"] = (
        "portfolio-s1-feedback-round3-schema-phase60-control"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-phase60-control-v1"
    ] = ROUND3_SCHEMA_PHASE60_CONTROL_POLICY_VERSION_V1
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    source_control_sha256: Sha256
    canary_manifest_file_sha256: Literal[
        ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ] = ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    canary_manifest_sha256: Sha256
    canary_run_sha256: Sha256
    canary_artifact_set_sha256: Sha256
    owner_approval_sha256: Sha256
    owner_approval_file_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
    )
    pricing_lock_file_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8
    )
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
    )
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    imported_prefix_ordinals: tuple[int, ...] = ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
    new_selection_ordinals: tuple[int, ...] = ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
    imported_final_artifact_sha256s: tuple[Sha256, ...]
    phase_target_count: Literal[60] = 60
    new_first_attempt_count: Literal[48] = 48
    new_global_retry_ceiling: Literal[12] = 12
    new_provider_call_ceiling: Literal[60] = 60
    cumulative_provider_call_ceiling: Literal[75] = 75
    prior_retry_claims_consumed: Literal[3] = 3
    prior_retry_claims_reusable: Literal[False] = False
    concurrency: Literal[2] = 2
    provider_internal_max_attempts: Literal[1] = 1
    maximum_lifetime_attempts_per_new_entry: Literal[2] = 2
    per_call_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    ] = ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    maximum_new_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    ] = ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY
    cumulative_maximum_reservation_cny: Literal[
        ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    ] = ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY
    fresh_technical_hard_cap_cny: Literal[ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY] = (
        ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY
    )
    cumulative_technical_hard_cap_cny: Literal[
        ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY
    ] = ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY
    membership_authorization_id: str
    membership_authorization_file_sha256: Sha256
    membership_receipt_file_sha256: Sha256
    membership_receipt_sha256: Sha256
    membership_catalog_sha256: Sha256
    canary_root_read_only: Literal[True] = True
    phase60_requires_new_output_root: Literal[True] = True
    phase120_requires_new_owner_approval: Literal[True] = True
    bundle_v11_publish_authorized: Literal[False] = False
    s1_creator_start_authorized: Literal[False] = False
    control_sha256: Sha256

    @field_validator(
        "imported_prefix_ordinals",
        "new_selection_ordinals",
        "imported_final_artifact_sha256s",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_control(self) -> Self:
        if (
            self.imported_prefix_ordinals != ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS
            or self.new_selection_ordinals != ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
            or len(self.imported_final_artifact_sha256s) != 12
            or len(set(self.imported_final_artifact_sha256s)) != 12
            or self.new_first_attempt_count + self.new_global_retry_ceiling
            != self.new_provider_call_ceiling
            or Decimal(self.per_call_reservation_cny) * self.new_provider_call_ceiling
            != Decimal(self.maximum_new_reservation_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_new_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.maximum_new_reservation_cny)
            >= Decimal(self.fresh_technical_hard_cap_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.fresh_technical_hard_cap_cny)
            != Decimal(self.cumulative_technical_hard_cap_cny)
            or self.control_sha256 != _model_hash(self, "control_sha256")
        ):
            raise ValueError("phase60 control drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_schema_phase60_control_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> PortfolioS1FeedbackRound3SchemaPhase60ControlV1:
    expected_authorization = build_round3_schema_phase60_authorization_v1(
        prefix,
        governance,
        approval,
        sources,
        authorization_id=authorization.authorization_id,
    )
    if authorization != expected_authorization:
        raise PortfolioS1FeedbackError("phase60 control authorization drifted")
    draft = PortfolioS1FeedbackRound3SchemaPhase60ControlV1.model_construct(
        source_control_sha256=prefix.predecessors.prior.base.control.control_sha256,
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        canary_run_sha256=prefix.run.run_sha256,
        canary_artifact_set_sha256=prefix.run.artifact_set_sha256,
        owner_approval_sha256=approval.approval_sha256,
        owner_approval_file_sha256=sha256_bytes(approval.canonical_bytes()),
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        imported_final_artifact_sha256s=tuple(
            item.artifact_sha256 for item in prefix.final_artifacts
        ),
        membership_authorization_id=authorization.membership_authorization_id,
        membership_authorization_file_sha256=(
            authorization.membership_authorization_file_sha256
        ),
        membership_receipt_file_sha256=authorization.membership_receipt_file_sha256,
        membership_receipt_sha256=authorization.membership_receipt_sha256,
        membership_catalog_sha256=authorization.membership_catalog_sha256,
        control_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"control_sha256"})
    return PortfolioS1FeedbackRound3SchemaPhase60ControlV1.model_validate(
        {**unsigned, "control_sha256": _hash_payload(unsigned)}, strict=True
    )


class PortfolioS1FeedbackRound3SchemaPhase60LaunchV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-launch"] = (
        "portfolio-s1-feedback-round3-schema-phase60-launch"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-schema-phase60-launch-v1"] = (
        ROUND3_SCHEMA_PHASE60_LAUNCH_POLICY_VERSION_V1
    )
    run_id: str
    canary_manifest_file_sha256: Literal[
        ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ] = ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    canary_manifest_sha256: Sha256
    canary_run_sha256: Sha256
    owner_approval_sha256: Sha256
    owner_approval_file_sha256: Sha256
    model_source_lock_file_sha256: Literal[
        QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    ] = QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
    model_source_lock_sha256: Literal[QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5
    )
    pricing_lock_file_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8
    )
    pricing_lock_sha256: Literal[QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8] = (
        QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8
    )
    role_selection_file_sha256: Literal[
        QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    ] = QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15
    role_selection_sha256: Literal[QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15] = (
        QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15
    )
    authorization_sha256: Sha256
    authorization_file_sha256: Sha256
    control_sha256: Sha256
    control_file_sha256: Sha256
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    phase_target_count: Literal[60] = 60
    new_first_attempt_count: Literal[48] = 48
    new_global_retry_ceiling: Literal[12] = 12
    new_provider_call_ceiling: Literal[60] = 60
    cumulative_provider_call_ceiling: Literal[75] = 75
    owner_budget_and_retry_approval_status: Literal["granted"] = "granted"
    canary_retry_claims_reused: Literal[0] = 0
    phase120_requires_new_owner_approval: Literal[True] = True
    bundle_v11_publish_authorized: Literal[False] = False
    s1_creator_start_authorized: Literal[False] = False
    launch_sha256: Sha256

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if (
            not self.run_id
            or self.run_id != self.run_id.strip()
            or self.new_first_attempt_count + self.new_global_retry_ceiling
            != self.new_provider_call_ceiling
            or self.launch_sha256 != _model_hash(self, "launch_sha256")
        ):
            raise ValueError("phase60 launch drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def build_round3_schema_phase60_launch_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    run_id: str,
) -> PortfolioS1FeedbackRound3SchemaPhase60LaunchV1:
    if control != build_round3_schema_phase60_control_v1(
        prefix, governance, approval, authorization, sources
    ):
        raise PortfolioS1FeedbackError("phase60 launch control drifted")
    draft = PortfolioS1FeedbackRound3SchemaPhase60LaunchV1.model_construct(
        run_id=run_id,
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        canary_run_sha256=prefix.run.run_sha256,
        owner_approval_sha256=approval.approval_sha256,
        owner_approval_file_sha256=sha256_bytes(approval.canonical_bytes()),
        authorization_sha256=authorization.authorization_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        control_sha256=control.control_sha256,
        control_file_sha256=sha256_bytes(control.canonical_bytes()),
        launch_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_sha256"})
    return PortfolioS1FeedbackRound3SchemaPhase60LaunchV1.model_validate(
        {**unsigned, "launch_sha256": _hash_payload(unsigned)}, strict=True
    )


class Round3SchemaPhase60GlobalRetryClaimV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-global-retry-claim"] = (
        "portfolio-s1-feedback-round3-schema-phase60-global-retry-claim"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-phase60-retry-claim-v1"
    ] = ROUND3_SCHEMA_PHASE60_CLAIM_POLICY_VERSION_V1
    retry_policy_version: Literal[
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    ] = ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    control_sha256: Sha256
    authorization_sha256: Sha256
    claim_ordinal: int = Field(ge=1, le=12)
    previous_claim_sha256: Sha256 | None = None
    selection_ordinal: int = Field(ge=13, le=60)
    selection_entry_sha256: Sha256
    query_id: str
    first_phase60_call_ordinal: int = Field(ge=1, le=60)
    first_cumulative_global_call_ordinal: int = Field(ge=16, le=75)
    first_artifact_sha256: Sha256
    first_feedback_result_sha256: Sha256
    trigger_finish_reason: Literal["stop", "length"]
    trigger_raw_response_bytes: int = Field(ge=0)
    claim_sha256: Sha256

    @model_validator(mode="after")
    def _validate_claim(self) -> Self:
        if (
            (self.claim_ordinal == 1) != (self.previous_claim_sha256 is None)
            or self.first_cumulative_global_call_ordinal
            != self.first_phase60_call_ordinal + 15
            or self.claim_sha256 != _model_hash(self, "claim_sha256")
        ):
            raise ValueError("phase60 retry claim drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Round3SchemaPhase60CallReservationV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-call-reservation"] = (
        "portfolio-s1-feedback-round3-schema-phase60-call-reservation"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-round3-schema-phase60-reservation-v1"
    ] = ROUND3_SCHEMA_PHASE60_RESERVATION_POLICY_VERSION_V1
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    canary_manifest_sha256: Sha256
    source_control_sha256: Sha256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=13, le=60)
    selection_entry_sha256: Sha256
    query_id: str
    packet_sha256: Sha256
    checkpoint_file_sha256: Sha256
    checkpoint_row_sha256: Sha256
    sidecar_sha256: Sha256
    attempt_index: Literal[1, 2]
    phase60_call_ordinal: int = Field(ge=1, le=60)
    cumulative_global_call_ordinal: int = Field(ge=16, le=75)
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: int | None = Field(default=None, ge=1, le=12)
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.8-max"] = "qwen3.8-max"
    wire_kind: Literal["round3_primary_json_schema_v1"] = (
        "round3_primary_json_schema_v1"
    )
    provider_internal_max_attempts: Literal[1] = 1
    reservation_cny: Literal[ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY] = (
        ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY
    )
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def _validate_reservation(self) -> Self:
        ancestry = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
        )
        if (
            self.cumulative_global_call_ordinal != self.phase60_call_ordinal + 15
            or (self.attempt_index == 1 and any(item is not None for item in ancestry))
            or (self.attempt_index == 2 and any(item is None for item in ancestry))
            or self.reservation_sha256 != _model_hash(self, "reservation_sha256")
        ):
            raise ValueError("phase60 reservation drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class BoundRound3SchemaPhase60FeedbackArtifactV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-bound-feedback-round3-schema-phase60"] = (
        "portfolio-s1-bound-feedback-round3-schema-phase60"
    )
    policy_version: Literal["portfolio-s1-bound-feedback-round3-schema-phase60-v1"] = (
        ROUND3_SCHEMA_PHASE60_BOUND_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    canary_manifest_sha256: Sha256
    source_control_sha256: Sha256
    control_sha256: Sha256
    authorization_sha256: Sha256
    selection_ordinal: int = Field(ge=13, le=60)
    selection_entry_sha256: Sha256
    query_id: str
    attempt_index: Literal[1, 2]
    phase60_call_ordinal: int = Field(ge=1, le=60)
    cumulative_global_call_ordinal: int = Field(ge=16, le=75)
    reservation_sha256: Sha256
    previous_artifact_sha256: Sha256 | None = None
    retry_claim_sha256: Sha256 | None = None
    retry_claim_ordinal: int | None = Field(default=None, ge=1, le=12)
    feedback_packet: FeedbackPacketV3
    feedback_result: RecoveryFeedbackEvaluationResultV2
    status: Phase60Status
    artifact_sha256: Sha256

    @field_validator("feedback_packet", mode="before")
    @classmethod
    def _packet(cls, value: object) -> object:
        return _nested_json_model(value, FeedbackPacketV3)

    @field_validator("feedback_result", mode="before")
    @classmethod
    def _result(cls, value: object) -> object:
        return _nested_json_model(value, RecoveryFeedbackEvaluationResultV2)

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        ancestry = (
            self.previous_artifact_sha256,
            self.retry_claim_sha256,
            self.retry_claim_ordinal,
        )
        if (
            self.cumulative_global_call_ordinal != self.phase60_call_ordinal + 15
            or self.feedback_packet.query_id != self.query_id
            or self.feedback_result.query_id != self.query_id
            or self.feedback_result.packet_sha256 != self.feedback_packet.packet_sha256
            or self.feedback_result.wire_kind != "round3_primary_json_schema_v1"
            or self.feedback_result.status != self.status
            or (self.attempt_index == 1 and any(item is not None for item in ancestry))
            or (self.attempt_index == 2 and any(item is None for item in ancestry))
            or self.artifact_sha256 != _model_hash(self, "artifact_sha256")
        ):
            raise ValueError("phase60 bound artifact drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def validate_round3_schema_phase60_claim_set_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    claims: tuple[Round3SchemaPhase60GlobalRetryClaimV1, ...],
    artifacts: tuple[BoundRound3SchemaPhase60FeedbackArtifactV1, ...],
) -> None:
    if len(claims) > ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING:
        raise PortfolioS1FeedbackError("phase60 exceeds twelve retry claims")
    first_by_sha = {
        item.artifact_sha256: item for item in artifacts if item.attempt_index == 1
    }
    prior: Round3SchemaPhase60GlobalRetryClaimV1 | None = None
    used_entries: set[str] = set()
    for ordinal, claim in enumerate(claims, 1):
        first = first_by_sha.get(claim.first_artifact_sha256)
        entry = selection.entries[claim.selection_ordinal - 1]
        eligible_unclaimed = sorted(
            (
                item
                for item in artifacts
                if item.attempt_index == 1
                and item.status != "parsed"
                and item.selection_entry_sha256 not in used_entries
                and is_round3_schema_retry_eligible_v1(item.feedback_result)
            ),
            key=lambda item: item.phase60_call_ordinal,
        )
        if (
            claim.claim_ordinal != ordinal
            or claim.previous_claim_sha256
            != (None if prior is None else prior.claim_sha256)
            or claim.selection_sha256 != selection.selection_sha256
            or claim.control_sha256 != control.control_sha256
            or claim.authorization_sha256 != control.authorization_sha256
            or claim.selection_entry_sha256 != entry.entry_sha256
            or claim.query_id != entry.query_id
            or claim.selection_entry_sha256 in used_entries
            or first is None
            or first.selection_entry_sha256 != claim.selection_entry_sha256
            or first.phase60_call_ordinal != claim.first_phase60_call_ordinal
            or first.cumulative_global_call_ordinal
            != claim.first_cumulative_global_call_ordinal
            or first.feedback_result.result_sha256 != claim.first_feedback_result_sha256
            or first.feedback_result.finish_reason != claim.trigger_finish_reason
            or first.feedback_result.raw_response_bytes
            != claim.trigger_raw_response_bytes
            or not is_round3_schema_retry_eligible_v1(first.feedback_result)
            or not eligible_unclaimed
            or eligible_unclaimed[0] != first
        ):
            raise PortfolioS1FeedbackError("phase60 retry claim set drifted")
        used_entries.add(claim.selection_entry_sha256)
        prior = claim


def build_round3_schema_phase60_retry_claim_v1(
    selection: PortfolioS1FeedbackSelectionV2,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    first_artifact: BoundRound3SchemaPhase60FeedbackArtifactV1,
    *,
    existing_claims: tuple[Round3SchemaPhase60GlobalRetryClaimV1, ...],
    existing_artifacts: tuple[BoundRound3SchemaPhase60FeedbackArtifactV1, ...],
) -> Round3SchemaPhase60GlobalRetryClaimV1:
    validate_round3_schema_phase60_claim_set_v1(
        selection, control, existing_claims, existing_artifacts
    )
    if (
        len(existing_claims) >= ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING
        or first_artifact.attempt_index != 1
        or first_artifact not in existing_artifacts
        or not is_round3_schema_retry_eligible_v1(first_artifact.feedback_result)
        or any(
            item.selection_entry_sha256 == first_artifact.selection_entry_sha256
            for item in existing_claims
        )
    ):
        raise PortfolioS1FeedbackError("phase60 retry claim is not eligible")
    eligible_unclaimed = sorted(
        (
            item
            for item in existing_artifacts
            if item.attempt_index == 1
            and item.status != "parsed"
            and is_round3_schema_retry_eligible_v1(item.feedback_result)
            and not any(
                claim.selection_entry_sha256 == item.selection_entry_sha256
                for claim in existing_claims
            )
        ),
        key=lambda item: item.phase60_call_ordinal,
    )
    if not eligible_unclaimed or eligible_unclaimed[0] != first_artifact:
        raise PortfolioS1FeedbackError("phase60 skipped an eligible failure")
    entry = selection.entries[first_artifact.selection_ordinal - 1]
    draft = Round3SchemaPhase60GlobalRetryClaimV1.model_construct(
        control_sha256=control.control_sha256,
        authorization_sha256=control.authorization_sha256,
        claim_ordinal=len(existing_claims) + 1,
        previous_claim_sha256=(
            None if not existing_claims else existing_claims[-1].claim_sha256
        ),
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        first_phase60_call_ordinal=first_artifact.phase60_call_ordinal,
        first_cumulative_global_call_ordinal=(
            first_artifact.cumulative_global_call_ordinal
        ),
        first_artifact_sha256=first_artifact.artifact_sha256,
        first_feedback_result_sha256=first_artifact.feedback_result.result_sha256,
        trigger_finish_reason=first_artifact.feedback_result.finish_reason,
        trigger_raw_response_bytes=first_artifact.feedback_result.raw_response_bytes,
        claim_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"claim_sha256"})
    claim = Round3SchemaPhase60GlobalRetryClaimV1.model_validate(
        {**unsigned, "claim_sha256": _hash_payload(unsigned)}, strict=True
    )
    validate_round3_schema_phase60_claim_set_v1(
        selection, control, (*existing_claims, claim), existing_artifacts
    )
    return claim


def build_round3_schema_phase60_reservation_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    source: VerifiedStaticFeedbackSourceV2,
    *,
    attempt_index: Literal[1, 2],
    phase60_call_ordinal: int,
    first_artifact: BoundRound3SchemaPhase60FeedbackArtifactV1 | None = None,
    retry_claim: Round3SchemaPhase60GlobalRetryClaimV1 | None = None,
) -> Round3SchemaPhase60CallReservationV1:
    if control != build_round3_schema_phase60_control_v1(
        prefix, governance, approval, authorization, sources
    ):
        raise PortfolioS1FeedbackError("phase60 reservation control drifted")
    selection = prefix.selection
    entry = next(
        (
            item
            for item in selection.entries
            if item.entry_sha256 == source.selection_entry_sha256
        ),
        None,
    )
    if (
        entry is None
        or entry.selection_ordinal not in ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
    ):
        raise PortfolioS1FeedbackError("phase60 reservation source is out of scope")
    require_verified_static_feedback_source_v2(
        source, selection, prefix.predecessors.prior.base.control, entry
    )
    if attempt_index == 1:
        if first_artifact is not None or retry_claim is not None:
            raise PortfolioS1FeedbackError("phase60 first call has retry ancestry")
    elif (
        first_artifact is None
        or retry_claim is None
        or first_artifact.attempt_index != 1
        or first_artifact.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.selection_entry_sha256 != entry.entry_sha256
        or retry_claim.first_artifact_sha256 != first_artifact.artifact_sha256
        or retry_claim.first_feedback_result_sha256
        != first_artifact.feedback_result.result_sha256
        or phase60_call_ordinal <= first_artifact.phase60_call_ordinal
        or not is_round3_schema_retry_eligible_v1(first_artifact.feedback_result)
    ):
        raise PortfolioS1FeedbackError("phase60 retry ancestry drifted")
    if phase60_call_ordinal not in range(
        1, ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING + 1
    ):
        raise PortfolioS1FeedbackError("phase60 provider call ceiling exceeded")
    draft = Round3SchemaPhase60CallReservationV1.model_construct(
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        source_control_sha256=prefix.predecessors.prior.base.control.control_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=entry.selection_ordinal,
        selection_entry_sha256=entry.entry_sha256,
        query_id=entry.query_id,
        packet_sha256=source.packet.packet_sha256,
        checkpoint_file_sha256=source.row.checkpoint_file_sha256,
        checkpoint_row_sha256=source.row.checkpoint_row_sha256,
        sidecar_sha256=source.row.sidecar.evidence_sha256,
        attempt_index=attempt_index,
        phase60_call_ordinal=phase60_call_ordinal,
        cumulative_global_call_ordinal=phase60_call_ordinal + 15,
        previous_artifact_sha256=(
            None if first_artifact is None else first_artifact.artifact_sha256
        ),
        retry_claim_sha256=None if retry_claim is None else retry_claim.claim_sha256,
        retry_claim_ordinal=(
            None if retry_claim is None else retry_claim.claim_ordinal
        ),
        reservation_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"reservation_sha256"})
    return Round3SchemaPhase60CallReservationV1.model_validate(
        {**unsigned, "reservation_sha256": _hash_payload(unsigned)}, strict=True
    )


def build_bound_round3_schema_phase60_artifact_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    source: VerifiedStaticFeedbackSourceV2,
    result: RecoveryFeedbackEvaluationResultV2,
    *,
    reservation: Round3SchemaPhase60CallReservationV1,
    first_artifact: BoundRound3SchemaPhase60FeedbackArtifactV1 | None = None,
    retry_claim: Round3SchemaPhase60GlobalRetryClaimV1 | None = None,
) -> BoundRound3SchemaPhase60FeedbackArtifactV1:
    expected = build_round3_schema_phase60_reservation_v1(
        prefix,
        governance,
        approval,
        authorization,
        control,
        sources,
        source,
        attempt_index=reservation.attempt_index,
        phase60_call_ordinal=reservation.phase60_call_ordinal,
        first_artifact=first_artifact,
        retry_claim=retry_claim,
    )
    redacted = redact_recovery_result_for_creator_privacy_v1(
        result,
        private_query_ids=tuple(item.query_id for item in prefix.selection.entries),
    )
    if (
        reservation != expected
        or type(redacted) is not RecoveryFeedbackEvaluationResultV2
        or redacted.wire_kind != "round3_primary_json_schema_v1"
        or redacted.query_id != reservation.query_id
        or redacted.packet_sha256 != source.packet.packet_sha256
        or redacted.image_sha256 != source.packet.image.sha256
        or redacted.remote_authorization_id != authorization.membership_authorization_id
        or redacted.remote_authorization_file_sha256
        != authorization.membership_authorization_file_sha256
        or redacted.remote_receipt_file_sha256
        != authorization.membership_receipt_file_sha256
        or redacted.remote_receipt_sha256 != authorization.membership_receipt_sha256
        or redacted.asset_catalog_sha256 != authorization.membership_catalog_sha256
    ):
        raise PortfolioS1FeedbackError("phase60 provider result drifted")
    draft = BoundRound3SchemaPhase60FeedbackArtifactV1.model_construct(
        canary_manifest_sha256=prefix.manifest.manifest_sha256,
        source_control_sha256=prefix.predecessors.prior.base.control.control_sha256,
        control_sha256=control.control_sha256,
        authorization_sha256=authorization.authorization_sha256,
        selection_ordinal=reservation.selection_ordinal,
        selection_entry_sha256=reservation.selection_entry_sha256,
        query_id=reservation.query_id,
        attempt_index=reservation.attempt_index,
        phase60_call_ordinal=reservation.phase60_call_ordinal,
        cumulative_global_call_ordinal=reservation.cumulative_global_call_ordinal,
        reservation_sha256=reservation.reservation_sha256,
        previous_artifact_sha256=reservation.previous_artifact_sha256,
        retry_claim_sha256=reservation.retry_claim_sha256,
        retry_claim_ordinal=reservation.retry_claim_ordinal,
        feedback_packet=source.packet,
        feedback_result=redacted,
        status=redacted.status,
        artifact_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"artifact_sha256"})
    return BoundRound3SchemaPhase60FeedbackArtifactV1.model_validate(
        {**unsigned, "artifact_sha256": _hash_payload(unsigned)}, strict=True
    )


def round3_schema_phase60_attempt_filename_v1(
    phase60_call_ordinal: int,
    selection_entry_sha256: str,
    attempt_index: int,
) -> str:
    if phase60_call_ordinal not in range(
        1, ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING + 1
    ) or attempt_index not in {1, 2}:
        raise PortfolioS1FeedbackError("phase60 attempt identity is invalid")
    return (
        f"{phase60_call_ordinal:04d}-{selection_entry_sha256[:16]}-"
        f"attempt-{attempt_index}.json"
    )


def round3_schema_phase60_claim_filename_v1(claim_ordinal: int) -> str:
    if claim_ordinal not in range(1, ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING + 1):
        raise PortfolioS1FeedbackError("phase60 claim ordinal is invalid")
    return f"claim-{claim_ordinal:02d}.json"


def _write_phase60_model(path: str | Path, model: BaseModel) -> Path:
    target = Path(path)
    atomic_create_file(target, canonical_json_bytes(model.model_dump(mode="json")))
    return target


def write_round3_schema_phase60_claim_v1(
    path: str | Path, claim: Round3SchemaPhase60GlobalRetryClaimV1
) -> Path:
    if Path(path).name != round3_schema_phase60_claim_filename_v1(claim.claim_ordinal):
        raise PortfolioS1FeedbackError("phase60 claim filename drifted")
    return _write_phase60_model(path, claim)


def load_round3_schema_phase60_claim_v1(
    path: str | Path,
) -> Round3SchemaPhase60GlobalRetryClaimV1:
    model = _load_canonical_model(
        path,
        model_type=Round3SchemaPhase60GlobalRetryClaimV1,
        label="phase60 retry claim",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3SchemaPhase60GlobalRetryClaimV1)
    if Path(path).name != round3_schema_phase60_claim_filename_v1(model.claim_ordinal):
        raise PortfolioS1FeedbackError("phase60 claim filename drifted")
    return model


def write_round3_schema_phase60_reservation_v1(
    path: str | Path, reservation: Round3SchemaPhase60CallReservationV1
) -> Path:
    if Path(path).name != round3_schema_phase60_attempt_filename_v1(
        reservation.phase60_call_ordinal,
        reservation.selection_entry_sha256,
        reservation.attempt_index,
    ):
        raise PortfolioS1FeedbackError("phase60 reservation filename drifted")
    return _write_phase60_model(path, reservation)


def load_round3_schema_phase60_reservation_v1(
    path: str | Path,
) -> Round3SchemaPhase60CallReservationV1:
    model = _load_canonical_model(
        path,
        model_type=Round3SchemaPhase60CallReservationV1,
        label="phase60 reservation",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, Round3SchemaPhase60CallReservationV1)
    if Path(path).name != round3_schema_phase60_attempt_filename_v1(
        model.phase60_call_ordinal,
        model.selection_entry_sha256,
        model.attempt_index,
    ):
        raise PortfolioS1FeedbackError("phase60 reservation filename drifted")
    return model


def write_bound_round3_schema_phase60_artifact_v1(
    path: str | Path, artifact: BoundRound3SchemaPhase60FeedbackArtifactV1
) -> Path:
    if Path(path).name != round3_schema_phase60_attempt_filename_v1(
        artifact.phase60_call_ordinal,
        artifact.selection_entry_sha256,
        artifact.attempt_index,
    ):
        raise PortfolioS1FeedbackError("phase60 artifact filename drifted")
    return _write_phase60_model(path, artifact)


def load_bound_round3_schema_phase60_artifact_v1(
    path: str | Path,
) -> BoundRound3SchemaPhase60FeedbackArtifactV1:
    model = _load_canonical_model(
        path,
        model_type=BoundRound3SchemaPhase60FeedbackArtifactV1,
        label="phase60 bound artifact",
    )
    assert isinstance(model, BoundRound3SchemaPhase60FeedbackArtifactV1)
    if Path(path).name != round3_schema_phase60_attempt_filename_v1(
        model.phase60_call_ordinal,
        model.selection_entry_sha256,
        model.attempt_index,
    ):
        raise PortfolioS1FeedbackError("phase60 artifact filename drifted")
    return model


@dataclass(frozen=True)
class PortfolioS1FeedbackRound3SchemaPhase60LedgerV1:
    claims: tuple[Round3SchemaPhase60GlobalRetryClaimV1, ...]
    reservations: tuple[Round3SchemaPhase60CallReservationV1, ...]
    artifacts: tuple[BoundRound3SchemaPhase60FeedbackArtifactV1, ...]
    orphaned_reservations: tuple[Round3SchemaPhase60CallReservationV1, ...]
    pending_claims: tuple[Round3SchemaPhase60GlobalRetryClaimV1, ...]


def _safe_phase60_inventory(directory: Path, *, label: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise PortfolioS1FeedbackError(f"{label} directory is unsafe")
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        not item.is_file() or item.is_symlink() or item.suffix != ".json"
        for item in paths
    ):
        raise PortfolioS1FeedbackError(f"{label} inventory is unsafe")
    return paths


def validate_round3_schema_phase60_ledger_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> None:
    if (
        control
        != build_round3_schema_phase60_control_v1(
            prefix, governance, approval, authorization, sources
        )
        or tuple(item.phase60_call_ordinal for item in ledger.reservations)
        != tuple(range(1, len(ledger.reservations) + 1))
        or tuple(item.cumulative_global_call_ordinal for item in ledger.reservations)
        != tuple(range(16, 16 + len(ledger.reservations)))
        or len(ledger.reservations) > ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING
        or len({item.reservation_sha256 for item in ledger.reservations})
        != len(ledger.reservations)
        or len(
            {
                (item.selection_entry_sha256, item.attempt_index)
                for item in ledger.reservations
            }
        )
        != len(ledger.reservations)
        or len({item.artifact_sha256 for item in ledger.artifacts})
        != len(ledger.artifacts)
    ):
        raise PortfolioS1FeedbackError("phase60 ledger root identity drifted")
    source_by_entry: dict[str, VerifiedStaticFeedbackSourceV2] = {}
    for entry, source in zip(prefix.selection.entries, sources, strict=True):
        require_verified_static_feedback_source_v2(
            source, prefix.selection, prefix.predecessors.prior.base.control, entry
        )
        source_by_entry[entry.entry_sha256] = source
    validate_round3_schema_phase60_claim_set_v1(
        prefix.selection, control, ledger.claims, ledger.artifacts
    )
    reservation_by_sha = {item.reservation_sha256: item for item in ledger.reservations}
    artifact_by_reservation: dict[str, BoundRound3SchemaPhase60FeedbackArtifactV1] = {}
    artifact_by_key: dict[
        tuple[str, int], BoundRound3SchemaPhase60FeedbackArtifactV1
    ] = {}
    for artifact in ledger.artifacts:
        reservation = reservation_by_sha.get(artifact.reservation_sha256)
        key = (artifact.selection_entry_sha256, artifact.attempt_index)
        if (
            reservation is None
            or artifact.reservation_sha256 in artifact_by_reservation
            or key in artifact_by_key
            or artifact.selection_ordinal != reservation.selection_ordinal
            or artifact.selection_entry_sha256 != reservation.selection_entry_sha256
            or artifact.phase60_call_ordinal != reservation.phase60_call_ordinal
            or artifact.cumulative_global_call_ordinal
            != reservation.cumulative_global_call_ordinal
            or artifact.control_sha256 != control.control_sha256
            or artifact.authorization_sha256 != authorization.authorization_sha256
        ):
            raise PortfolioS1FeedbackError(
                "phase60 artifact/reservation binding drifted"
            )
        artifact_by_reservation[artifact.reservation_sha256] = artifact
        artifact_by_key[key] = artifact
    expected_orphans = tuple(
        item
        for item in ledger.reservations
        if item.reservation_sha256 not in artifact_by_reservation
    )
    if ledger.orphaned_reservations != expected_orphans or len(expected_orphans) > 2:
        raise PortfolioS1FeedbackError("phase60 orphan ledger drifted")
    if expected_orphans:
        final_reservation = ledger.reservations[-1]
        allowed_wave = (final_reservation,)
        if len(ledger.reservations) >= 2:
            previous = ledger.reservations[-2]
            if (
                previous.attempt_index == 1
                and final_reservation.attempt_index == 1
                and previous.selection_ordinal + 1
                == final_reservation.selection_ordinal
            ):
                allowed_wave = (previous, final_reservation)
        if any(item not in allowed_wave for item in expected_orphans):
            raise PortfolioS1FeedbackError(
                "phase60 orphan is outside final reserved wave"
            )
    first_reservations = tuple(
        item for item in ledger.reservations if item.attempt_index == 1
    )
    if tuple(item.selection_ordinal for item in first_reservations) != tuple(
        range(13, 13 + len(first_reservations))
    ):
        raise PortfolioS1FeedbackError("phase60 first attempts changed fixed order")

    claim_by_sha = {item.claim_sha256: item for item in ledger.claims}
    first_artifact_by_entry = {
        item.selection_entry_sha256: item
        for item in ledger.artifacts
        if item.attempt_index == 1
    }
    used_claims: list[str] = []
    for reservation in ledger.reservations:
        entry = prefix.selection.entries[reservation.selection_ordinal - 1]
        source = source_by_entry.get(reservation.selection_entry_sha256)
        first_artifact = (
            first_artifact_by_entry.get(reservation.selection_entry_sha256)
            if reservation.attempt_index == 2
            else None
        )
        claim = (
            claim_by_sha.get(reservation.retry_claim_sha256 or "")
            if reservation.attempt_index == 2
            else None
        )
        if source is None or entry.entry_sha256 != reservation.selection_entry_sha256:
            raise PortfolioS1FeedbackError("phase60 reservation source unknown")
        if reservation.attempt_index == 2:
            if (
                claim is None
                or first_artifact is None
                or claim.selection_entry_sha256 != entry.entry_sha256
                or reservation.previous_artifact_sha256
                != first_artifact.artifact_sha256
                or reservation.retry_claim_ordinal != claim.claim_ordinal
            ):
                raise PortfolioS1FeedbackError(
                    "phase60 retry reservation claim drifted"
                )
            used_claims.append(claim.claim_sha256)
        expected = build_round3_schema_phase60_reservation_v1(
            prefix,
            governance,
            approval,
            authorization,
            control,
            sources,
            source,
            attempt_index=reservation.attempt_index,
            phase60_call_ordinal=reservation.phase60_call_ordinal,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if reservation != expected:
            raise PortfolioS1FeedbackError(
                "phase60 reservation differs from verified source"
            )
    for artifact in ledger.artifacts:
        reservation = reservation_by_sha[artifact.reservation_sha256]
        source = source_by_entry[artifact.selection_entry_sha256]
        first_artifact = (
            first_artifact_by_entry.get(artifact.selection_entry_sha256)
            if artifact.attempt_index == 2
            else None
        )
        claim = (
            claim_by_sha.get(artifact.retry_claim_sha256 or "")
            if artifact.attempt_index == 2
            else None
        )
        expected = build_bound_round3_schema_phase60_artifact_v1(
            prefix,
            governance,
            approval,
            authorization,
            control,
            sources,
            source,
            artifact.feedback_result,
            reservation=reservation,
            first_artifact=first_artifact,
            retry_claim=claim,
        )
        if artifact != expected:
            raise PortfolioS1FeedbackError(
                "phase60 artifact differs from verified source/result"
            )
    claim_sha256s = [item.claim_sha256 for item in ledger.claims]
    if used_claims != claim_sha256s[: len(used_claims)]:
        raise PortfolioS1FeedbackError("phase60 retry claims consumed out of order")
    expected_pending = ledger.claims[len(used_claims) :]
    if ledger.pending_claims != expected_pending or len(expected_pending) > 1:
        raise PortfolioS1FeedbackError("phase60 pending claim drifted")


def load_round3_schema_phase60_ledger_v1(
    output_root: str | Path,
    *,
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
) -> PortfolioS1FeedbackRound3SchemaPhase60LedgerV1:
    root = Path(output_root)
    claims = tuple(
        load_round3_schema_phase60_claim_v1(path)
        for path in _safe_phase60_inventory(
            root / ROUND3_SCHEMA_PHASE60_CLAIM_DIR,
            label="phase60 claims",
        )
    )
    reservations = tuple(
        sorted(
            (
                load_round3_schema_phase60_reservation_v1(path)
                for path in _safe_phase60_inventory(
                    root / ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR,
                    label="phase60 reservations",
                )
            ),
            key=lambda item: item.phase60_call_ordinal,
        )
    )
    artifacts = tuple(
        sorted(
            (
                load_bound_round3_schema_phase60_artifact_v1(path)
                for path in _safe_phase60_inventory(
                    root / ROUND3_SCHEMA_PHASE60_BOUND_DIR,
                    label="phase60 bound artifacts",
                )
            ),
            key=lambda item: item.phase60_call_ordinal,
        )
    )
    settled = {item.reservation_sha256 for item in artifacts}
    used_claim_count = sum(item.attempt_index == 2 for item in reservations)
    ledger = PortfolioS1FeedbackRound3SchemaPhase60LedgerV1(
        claims=claims,
        reservations=reservations,
        artifacts=artifacts,
        orphaned_reservations=tuple(
            item for item in reservations if item.reservation_sha256 not in settled
        ),
        pending_claims=claims[used_claim_count:],
    )
    validate_round3_schema_phase60_ledger_v1(
        prefix, governance, approval, authorization, control, ledger, sources
    )
    return ledger


def _phase60_actual_cost(
    result: RecoveryFeedbackEvaluationResultV2,
) -> Decimal | None:
    usage = result.usage
    if usage is None:
        return None
    return (
        Decimal(usage.input_tokens) * Decimal(12)
        + Decimal(usage.output_tokens) * Decimal(36)
    ) / Decimal(1_000_000)


def round3_schema_phase60_fresh_accountable_cost_v1(
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
) -> str:
    actual = Decimal("0")
    unknown = len(ledger.orphaned_reservations)
    for artifact in ledger.artifacts:
        cost = _phase60_actual_cost(artifact.feedback_result)
        if cost is None:
            unknown += 1
        else:
            actual += cost
    accountable = actual + (
        Decimal(ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY) * unknown
    )
    return format(accountable.quantize(Decimal("0.000000000001")), "f")


def require_round3_schema_phase60_pre_reservation_budget_v1(
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
) -> str:
    """Fail closed immediately before the next create-only reservation write."""

    if len(ledger.reservations) >= ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING:
        raise PortfolioS1FeedbackError("phase60 provider call ceiling exceeded")
    if any(
        item.feedback_result.usage is not None
        and (
            item.feedback_result.usage.input_tokens > 20_000
            or item.feedback_result.usage.output_tokens > 6_154
        )
        for item in ledger.artifacts
    ):
        raise PortfolioS1FeedbackError("phase60 per-call usage ceiling exceeded")
    committed = Decimal(round3_schema_phase60_fresh_accountable_cost_v1(ledger))
    after_next = committed + Decimal(ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY)
    if after_next > Decimal(ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY):
        raise PortfolioS1FeedbackError("phase60 approved fresh budget exceeded")
    cumulative = Decimal(ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY) + after_next
    if cumulative > Decimal(ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY):
        raise PortfolioS1FeedbackError("phase60 cumulative budget exceeded")
    return format(cumulative.quantize(Decimal("0.000000000001")), "f")


class Round3SchemaPhase60NextStepV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal[
        "create_claim",
        "reserve_retry",
        "reserve_primary_wave",
        "phase_complete",
        "terminal_nonparsed",
        "terminal_orphan",
        "terminal_budget",
    ]
    selection_ordinals: tuple[int, ...] = ()
    attempt_index: Literal[1, 2] | None = None
    claim_ordinal: int | None = Field(default=None, ge=1, le=12)
    next_phase60_call_ordinal: int | None = Field(default=None, ge=1, le=60)
    next_cumulative_global_call_ordinal: int | None = Field(default=None, ge=16, le=75)
    imported_parsed_count: Literal[12] = 12
    new_parsed_count: int = Field(ge=0, le=48)
    combined_parsed_count: int = Field(ge=12, le=60)
    new_provider_calls_reserved: int = Field(ge=0, le=60)
    new_retry_claim_count: int = Field(ge=0, le=12)
    terminal_error_code: str | None = None

    @field_validator("selection_ordinals", mode="before")
    @classmethod
    def _ordinals(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_step(self) -> Self:
        has_next = self.next_phase60_call_ordinal is not None
        if (
            self.combined_parsed_count
            != self.imported_parsed_count + self.new_parsed_count
            or has_next != (self.next_cumulative_global_call_ordinal is not None)
            or (
                has_next
                and self.next_cumulative_global_call_ordinal
                != self.next_phase60_call_ordinal + 15  # type: ignore[operator]
            )
            or any(
                item not in ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
                for item in self.selection_ordinals
            )
        ):
            raise ValueError("phase60 next-step identity drifted")
        return self


def _phase60_budget_error_for_new_calls(
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    count: int,
) -> (
    Literal[
        "usage_limit_exceeded",
        "accountable_cost_exceeded",
        "provider_call_ceiling_exceeded",
    ]
    | None
):
    if (
        len(ledger.reservations) + count
        > ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING
    ):
        return "provider_call_ceiling_exceeded"
    if any(
        item.feedback_result.usage is not None
        and (
            item.feedback_result.usage.input_tokens > 20_000
            or item.feedback_result.usage.output_tokens > 6_154
        )
        for item in ledger.artifacts
    ):
        return "usage_limit_exceeded"
    after = Decimal(round3_schema_phase60_fresh_accountable_cost_v1(ledger)) + (
        Decimal(ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY) * count
    )
    if after > Decimal(ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY):
        return "accountable_cost_exceeded"
    return None


def next_round3_schema_phase60_step_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    phase_target_count: int = 60,
) -> Round3SchemaPhase60NextStepV1:
    validate_round3_schema_phase60_ledger_v1(
        prefix, governance, approval, authorization, control, ledger, sources
    )
    if phase_target_count != ROUND3_SCHEMA_PHASE60_TARGET_COUNT:
        raise PortfolioS1FeedbackError("phase60 authority cannot continue to phase120")
    artifacts_by_key = {
        (item.selection_entry_sha256, item.attempt_index): item
        for item in ledger.artifacts
    }
    final_by_ordinal: dict[int, BoundRound3SchemaPhase60FeedbackArtifactV1] = {}
    for artifact in ledger.artifacts:
        prior = final_by_ordinal.get(artifact.selection_ordinal)
        if prior is None or artifact.attempt_index > prior.attempt_index:
            final_by_ordinal[artifact.selection_ordinal] = artifact
    new_parsed = sum(item.status == "parsed" for item in final_by_ordinal.values())
    common = {
        "new_parsed_count": new_parsed,
        "combined_parsed_count": 12 + new_parsed,
        "new_provider_calls_reserved": len(ledger.reservations),
        "new_retry_claim_count": len(ledger.claims),
    }
    if ledger.orphaned_reservations:
        return Round3SchemaPhase60NextStepV1(
            kind="terminal_orphan",
            selection_ordinals=tuple(
                item.selection_ordinal for item in ledger.orphaned_reservations
            ),
            terminal_error_code="orphan",
            **common,
        )
    retry_failures = sorted(
        (
            item
            for item in ledger.artifacts
            if item.attempt_index == 2 and item.status != "parsed"
        ),
        key=lambda item: item.phase60_call_ordinal,
    )
    if retry_failures:
        failed = retry_failures[0]
        return Round3SchemaPhase60NextStepV1(
            kind="terminal_nonparsed",
            selection_ordinals=(failed.selection_ordinal,),
            attempt_index=2,
            terminal_error_code=failed.feedback_result.error_code,
            **common,
        )
    first_failures = sorted(
        (
            item
            for item in ledger.artifacts
            if item.attempt_index == 1
            and item.status != "parsed"
            and (item.selection_entry_sha256, 2) not in artifacts_by_key
        ),
        key=lambda item: item.phase60_call_ordinal,
    )
    if first_failures:
        noneligible = tuple(
            item
            for item in first_failures
            if not is_round3_schema_retry_eligible_v1(item.feedback_result)
        )
        if noneligible:
            failed = noneligible[0]
            return Round3SchemaPhase60NextStepV1(
                kind="terminal_nonparsed",
                selection_ordinals=(failed.selection_ordinal,),
                attempt_index=1,
                terminal_error_code=failed.feedback_result.error_code,
                **common,
            )
        claimed_entries = {item.selection_entry_sha256 for item in ledger.claims}
        unclaimed = tuple(
            item
            for item in first_failures
            if item.selection_entry_sha256 not in claimed_entries
        )
        if len(unclaimed) > (
            ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING - len(ledger.claims)
        ):
            return Round3SchemaPhase60NextStepV1(
                kind="terminal_nonparsed",
                selection_ordinals=tuple(item.selection_ordinal for item in unclaimed),
                attempt_index=1,
                terminal_error_code="global_retry_ceiling_exceeded",
                **common,
            )
        first = first_failures[0]
        claim = next(
            (
                item
                for item in ledger.claims
                if item.selection_entry_sha256 == first.selection_entry_sha256
            ),
            None,
        )
        budget_error = _phase60_budget_error_for_new_calls(ledger, 1)
        if budget_error is not None:
            return Round3SchemaPhase60NextStepV1(
                kind="terminal_budget",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=None if claim is None else claim.claim_ordinal,
                terminal_error_code=budget_error,
                **common,
            )
        if claim is None:
            return Round3SchemaPhase60NextStepV1(
                kind="create_claim",
                selection_ordinals=(first.selection_ordinal,),
                attempt_index=2,
                claim_ordinal=len(ledger.claims) + 1,
                **common,
            )
        next_call = len(ledger.reservations) + 1
        return Round3SchemaPhase60NextStepV1(
            kind="reserve_retry",
            selection_ordinals=(first.selection_ordinal,),
            attempt_index=2,
            claim_ordinal=claim.claim_ordinal,
            next_phase60_call_ordinal=next_call,
            next_cumulative_global_call_ordinal=next_call + 15,
            **common,
        )
    if all(
        ordinal in final_by_ordinal and final_by_ordinal[ordinal].status == "parsed"
        for ordinal in ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
    ):
        return Round3SchemaPhase60NextStepV1(kind="phase_complete", **common)
    pending = tuple(
        ordinal
        for ordinal in ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
        if ordinal not in final_by_ordinal
    )
    if not pending:
        raise PortfolioS1FeedbackError("phase60 has unresolved nonparsed output")
    wave = pending[:2]
    budget_error = _phase60_budget_error_for_new_calls(ledger, len(wave))
    if budget_error is not None:
        return Round3SchemaPhase60NextStepV1(
            kind="terminal_budget",
            selection_ordinals=wave,
            attempt_index=1,
            terminal_error_code=budget_error,
            **common,
        )
    next_call = len(ledger.reservations) + 1
    return Round3SchemaPhase60NextStepV1(
        kind="reserve_primary_wave",
        selection_ordinals=wave,
        attempt_index=1,
        next_phase60_call_ordinal=next_call,
        next_cumulative_global_call_ordinal=next_call + 15,
        **common,
    )


class Round3SchemaPhase60RunAttemptV1(_StrictFrozenModel):
    selection_ordinal: int = Field(ge=13, le=60)
    selection_entry_sha256: Sha256
    attempt_index: Literal[1, 2]
    phase60_call_ordinal: int = Field(ge=1, le=60)
    cumulative_global_call_ordinal: int = Field(ge=16, le=75)
    reservation_sha256: Sha256
    artifact_sha256: Sha256 | None = None
    retry_claim_ordinal: int | None = Field(default=None, ge=1, le=12)
    status: Literal["parsed", "parse_error", "provider_error", "timeout", "orphan"]
    error_code: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    raw_response_bytes: int | None = Field(default=None, ge=0)
    tool_call_count: int | None = Field(default=None, ge=0)
    refusal_present: bool | None = None
    response_redaction_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_bytes: int | None = Field(default=None, ge=0)
    actual_cost_cny: str | None = None

    @model_validator(mode="after")
    def _validate_attempt(self) -> Self:
        usage_known = self.input_tokens is not None or self.output_tokens is not None
        if usage_known:
            if self.input_tokens is None or self.output_tokens is None:
                raise ValueError("phase60 run attempt has partial usage")
            expected = (
                Decimal(self.input_tokens) * Decimal(12)
                + Decimal(self.output_tokens) * Decimal(36)
            ) / Decimal(1_000_000)
            if self.actual_cost_cny != format(
                expected.quantize(Decimal("0.000000000001")), "f"
            ):
                raise ValueError("phase60 run attempt cost differs from usage")
        elif self.actual_cost_cny is not None:
            raise ValueError("phase60 run attempt cost exists without usage")
        if self.status == "orphan":
            if self.artifact_sha256 is not None or any(
                item is not None
                for item in (
                    self.request_id,
                    self.finish_reason,
                    self.raw_response_bytes,
                    self.tool_call_count,
                    self.refusal_present,
                    self.input_tokens,
                    self.output_tokens,
                    self.actual_cost_cny,
                )
            ):
                raise ValueError("phase60 run orphan contains response data")
        elif self.artifact_sha256 is None:
            raise ValueError("phase60 settled run attempt lacks artifact")
        if self.status == "parsed" and self.error_code is not None:
            raise ValueError("phase60 parsed attempt claims error")
        if self.status not in {"parsed", "orphan"} and self.error_code is None:
            raise ValueError("phase60 failed attempt lacks error code")
        if (self.attempt_index == 2) != (self.retry_claim_ordinal is not None):
            raise ValueError("phase60 run retry identity drifted")
        if self.cumulative_global_call_ordinal != self.phase60_call_ordinal + 15:
            raise ValueError("phase60 run cumulative call identity drifted")
        return self


class PortfolioS1FeedbackRound3SchemaPhase60RunV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-feedback-round3-schema-phase60-run"] = (
        "portfolio-s1-feedback-round3-schema-phase60-run"
    )
    policy_version: Literal["portfolio-s1-feedback-round3-schema-phase60-run-v1"] = (
        ROUND3_SCHEMA_PHASE60_RUN_POLICY_VERSION_V1
    )
    selection_sha256: Literal[PARENT_SELECTION_SHA256] = PARENT_SELECTION_SHA256
    canary_manifest_file_sha256: Literal[
        ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    ] = ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
    canary_manifest_sha256: Sha256
    canary_run_file_sha256: Sha256
    canary_run_sha256: Sha256
    canary_artifact_set_sha256: Sha256
    owner_approval_sha256: Sha256
    authorization_sha256: Sha256
    control_sha256: Sha256
    launch_sha256: Sha256
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    retry_policy_sha256: Literal[ROUND3_PHASE60_RETRY_POLICY_SHA256_V1] = (
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    )
    imported_prefix_count: Literal[12] = 12
    phase_target_count: Literal[60] = 60
    new_expected_count: Literal[48] = 48
    imported_parsed_count: Literal[12] = 12
    new_attempted_count: int = Field(ge=1, le=48)
    combined_attempted_count: int = Field(ge=13, le=60)
    new_parsed_count: int = Field(ge=0, le=48)
    combined_parsed_count: int = Field(ge=12, le=60)
    new_error_count: int = Field(ge=0, le=48)
    orphan_count: int = Field(ge=0, le=2)
    canary_provider_calls_reserved: Literal[15] = 15
    new_provider_calls_reserved: int = Field(ge=1, le=60)
    cumulative_provider_calls_reserved: int = Field(ge=16, le=75)
    canary_retry_count: Literal[3] = 3
    new_retry_count: int = Field(ge=0, le=12)
    cumulative_retry_count: int = Field(ge=3, le=15)
    new_retry_claim_sha256s: tuple[Sha256, ...]
    status: Literal[
        "stopped_nonparsed",
        "stopped_orphan",
        "stopped_budget",
        "completed_phase60",
    ]
    terminal_reason: (
        Literal[
            "usage_limit_exceeded",
            "accountable_cost_exceeded",
            "provider_call_ceiling_exceeded",
        ]
        | None
    ) = None
    canary_retry_claims_reused: Literal[0] = 0
    canary_provider_attempts_replayed: Literal[0] = 0
    phase120_requires_new_owner_approval: Literal[True] = True
    bundle_v11_publishable_from_phase60: Literal[False] = False
    s1_creator_start_authorized: Literal[False] = False
    usage_known_count: int = Field(ge=0, le=60)
    usage_unknown_count: int = Field(ge=0, le=60)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    prior_cumulative_actual_cost_cny: Literal[
        ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    ] = ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
    new_actual_cost_cny: str
    fresh_accountable_cost_cny: str
    cumulative_actual_cost_cny: str
    cumulative_accountable_cost_cny: str
    artifacts: tuple[Round3SchemaPhase60RunAttemptV1, ...]
    new_artifact_set_sha256: Sha256
    run_sha256: Sha256

    @field_validator("new_retry_claim_sha256s", "artifacts", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        final = {item.selection_entry_sha256: item for item in self.artifacts}
        statuses = Counter(item.status for item in final.values())
        new_actual = sum(
            (
                Decimal(item.actual_cost_cny)
                for item in self.artifacts
                if item.actual_cost_cny is not None
            ),
            Decimal("0"),
        )
        unknown = sum(item.input_tokens is None for item in self.artifacts)
        fresh_accountable = new_actual + (
            Decimal(ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY) * unknown
        )
        cumulative_actual = (
            Decimal(ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY) + new_actual
        )
        cumulative_accountable = (
            Decimal(ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY) + fresh_accountable
        )
        usage_breach = any(
            item.input_tokens is not None
            and (item.input_tokens > 20_000 or (item.output_tokens or 0) > 6_154)
            for item in self.artifacts
        )
        accountable_breach = fresh_accountable > Decimal(
            ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY
        )

        def money(value: Decimal) -> str:
            return format(value.quantize(Decimal("0.000000000001")), "f")

        if (
            tuple(item.phase60_call_ordinal for item in self.artifacts)
            != tuple(range(1, len(self.artifacts) + 1))
            or tuple(item.cumulative_global_call_ordinal for item in self.artifacts)
            != tuple(range(16, 16 + len(self.artifacts)))
            or self.new_provider_calls_reserved != len(self.artifacts)
            or self.cumulative_provider_calls_reserved != 15 + len(self.artifacts)
            or self.new_attempted_count != len(final)
            or self.combined_attempted_count != 12 + len(final)
            or self.new_parsed_count != statuses["parsed"]
            or self.combined_parsed_count != 12 + self.new_parsed_count
            or self.new_error_count != self.new_attempted_count - self.new_parsed_count
            or self.orphan_count
            != sum(item.status == "orphan" for item in self.artifacts)
            or self.new_retry_count
            != sum(item.attempt_index == 2 for item in self.artifacts)
            or self.cumulative_retry_count != 3 + self.new_retry_count
            or self.new_retry_count != len(self.new_retry_claim_sha256s)
            or self.usage_known_count
            != sum(item.input_tokens is not None for item in self.artifacts)
            or self.usage_unknown_count
            != self.new_provider_calls_reserved - self.usage_known_count
            or self.input_tokens
            != sum(item.input_tokens or 0 for item in self.artifacts)
            or self.output_tokens
            != sum(item.output_tokens or 0 for item in self.artifacts)
            or self.new_actual_cost_cny != money(new_actual)
            or self.fresh_accountable_cost_cny != money(fresh_accountable)
            or self.cumulative_actual_cost_cny != money(cumulative_actual)
            or self.cumulative_accountable_cost_cny != money(cumulative_accountable)
            or self.new_artifact_set_sha256
            != _hash_payload([item.model_dump(mode="json") for item in self.artifacts])
        ):
            raise ValueError("phase60 run counts, costs, or hashes drifted")
        if self.status == "completed_phase60":
            if (
                self.terminal_reason is not None
                or usage_breach
                or accountable_breach
                or self.new_attempted_count != 48
                or self.new_parsed_count != 48
                or self.orphan_count
                or tuple(sorted(item.selection_ordinal for item in final.values()))
                != ROUND3_SCHEMA_PHASE60_NEW_ORDINALS
            ):
                raise ValueError("completed phase60 run is not parsed60")
        elif self.status == "stopped_orphan":
            if not self.orphan_count:
                raise ValueError("phase60 orphan status lacks orphan")
        elif self.status == "stopped_budget":
            if self.terminal_reason is None or self.orphan_count:
                raise ValueError("phase60 budget status drifted")
        elif (
            self.new_error_count < 1
            or self.orphan_count
            or self.terminal_reason is not None
            or usage_breach
            or accountable_breach
        ):
            raise ValueError("phase60 nonparsed status drifted")
        if self.status in {"stopped_budget", "stopped_orphan"}:
            if self.terminal_reason == "usage_limit_exceeded" and not usage_breach:
                raise ValueError("phase60 usage reason lacks evidence")
            if (
                self.terminal_reason == "accountable_cost_exceeded"
                and not accountable_breach
            ):
                raise ValueError("phase60 cost reason lacks evidence")
            if (
                self.terminal_reason == "provider_call_ceiling_exceeded"
                and self.new_provider_calls_reserved
                != ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING
            ):
                raise ValueError("phase60 call reason lacks evidence")
        if self.run_sha256 != _model_hash(self, "run_sha256"):
            raise ValueError("phase60 run hash drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _phase60_run_actual_cost(
    result: RecoveryFeedbackEvaluationResultV2,
) -> str | None:
    value = _phase60_actual_cost(result)
    return (
        None
        if value is None
        else format(value.quantize(Decimal("0.000000000001")), "f")
    )


def build_round3_schema_phase60_run_v1(
    prefix: VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1,
    governance: VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
    launch: PortfolioS1FeedbackRound3SchemaPhase60LaunchV1,
    ledger: PortfolioS1FeedbackRound3SchemaPhase60LedgerV1,
    sources: tuple[VerifiedStaticFeedbackSourceV2, ...],
    *,
    terminal_reason: (
        Literal[
            "usage_limit_exceeded",
            "accountable_cost_exceeded",
            "provider_call_ceiling_exceeded",
        ]
        | None
    ) = None,
) -> PortfolioS1FeedbackRound3SchemaPhase60RunV1:
    validate_round3_schema_phase60_ledger_v1(
        prefix, governance, approval, authorization, control, ledger, sources
    )
    if launch != build_round3_schema_phase60_launch_v1(
        prefix,
        governance,
        approval,
        authorization,
        control,
        sources,
        run_id=launch.run_id,
    ):
        raise PortfolioS1FeedbackError("phase60 run launch drifted")
    artifact_by_reservation = {
        item.reservation_sha256: item for item in ledger.artifacts
    }
    rows: list[Round3SchemaPhase60RunAttemptV1] = []
    for reservation in ledger.reservations:
        artifact = artifact_by_reservation.get(reservation.reservation_sha256)
        if artifact is None:
            rows.append(
                Round3SchemaPhase60RunAttemptV1(
                    selection_ordinal=reservation.selection_ordinal,
                    selection_entry_sha256=reservation.selection_entry_sha256,
                    attempt_index=reservation.attempt_index,
                    phase60_call_ordinal=reservation.phase60_call_ordinal,
                    cumulative_global_call_ordinal=(
                        reservation.cumulative_global_call_ordinal
                    ),
                    reservation_sha256=reservation.reservation_sha256,
                    retry_claim_ordinal=reservation.retry_claim_ordinal,
                    status="orphan",
                )
            )
            continue
        result = artifact.feedback_result
        usage = result.usage
        rows.append(
            Round3SchemaPhase60RunAttemptV1(
                selection_ordinal=artifact.selection_ordinal,
                selection_entry_sha256=artifact.selection_entry_sha256,
                attempt_index=artifact.attempt_index,
                phase60_call_ordinal=artifact.phase60_call_ordinal,
                cumulative_global_call_ordinal=(
                    artifact.cumulative_global_call_ordinal
                ),
                reservation_sha256=artifact.reservation_sha256,
                artifact_sha256=artifact.artifact_sha256,
                retry_claim_ordinal=artifact.retry_claim_ordinal,
                status=artifact.status,
                error_code=result.error_code,
                request_id=result.request_id,
                finish_reason=result.finish_reason,
                raw_response_bytes=result.raw_response_bytes,
                tool_call_count=result.tool_call_count,
                refusal_present=result.refusal_present,
                response_redaction_reason=result.response_redaction_reason,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                reasoning_tokens=result.reasoning_tokens,
                reasoning_bytes=result.reasoning_bytes,
                actual_cost_cny=_phase60_run_actual_cost(result),
            )
        )
    if not rows:
        raise PortfolioS1FeedbackError("phase60 run has no new provider attempts")
    final = {item.selection_entry_sha256: item for item in rows}
    new_parsed = sum(item.status == "parsed" for item in final.values())
    usage_limit = any(
        item.input_tokens is not None
        and (item.input_tokens > 20_000 or (item.output_tokens or 0) > 6_154)
        for item in rows
    )
    new_actual = sum(
        (
            Decimal(item.actual_cost_cny)
            for item in rows
            if item.actual_cost_cny is not None
        ),
        Decimal("0"),
    )
    unknown = sum(item.input_tokens is None for item in rows)
    fresh_accountable = new_actual + (
        Decimal(ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY) * unknown
    )
    cumulative_actual = (
        Decimal(ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY) + new_actual
    )
    cumulative_accountable = (
        Decimal(ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY) + fresh_accountable
    )
    if usage_limit:
        terminal_reason = "usage_limit_exceeded"
    elif fresh_accountable > Decimal(ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY):
        terminal_reason = "accountable_cost_exceeded"
    if ledger.orphaned_reservations:
        status = "stopped_orphan"
    elif terminal_reason is not None:
        status = "stopped_budget"
    elif len(final) == 48 and new_parsed == 48:
        status = "completed_phase60"
    elif any(item.status != "parsed" for item in final.values()):
        status = "stopped_nonparsed"
    else:
        raise PortfolioS1FeedbackError("phase60 nonterminal prefix cannot publish run")

    def money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.000000000001")), "f")

    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-s1-feedback-round3-schema-phase60-run",
        "policy_version": ROUND3_SCHEMA_PHASE60_RUN_POLICY_VERSION_V1,
        "selection_sha256": prefix.selection.selection_sha256,
        "canary_manifest_file_sha256": (
            ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1
        ),
        "canary_manifest_sha256": prefix.manifest.manifest_sha256,
        "canary_run_file_sha256": prefix.manifest.run_file_sha256,
        "canary_run_sha256": prefix.run.run_sha256,
        "canary_artifact_set_sha256": prefix.run.artifact_set_sha256,
        "owner_approval_sha256": approval.approval_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "control_sha256": control.control_sha256,
        "launch_sha256": launch.launch_sha256,
        "transport_policy_sha256": (
            ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
        ),
        "retry_policy_sha256": ROUND3_PHASE60_RETRY_POLICY_SHA256_V1,
        "imported_prefix_count": 12,
        "phase_target_count": 60,
        "new_expected_count": 48,
        "imported_parsed_count": 12,
        "new_attempted_count": len(final),
        "combined_attempted_count": 12 + len(final),
        "new_parsed_count": new_parsed,
        "combined_parsed_count": 12 + new_parsed,
        "new_error_count": len(final) - new_parsed,
        "orphan_count": len(ledger.orphaned_reservations),
        "canary_provider_calls_reserved": 15,
        "new_provider_calls_reserved": len(rows),
        "cumulative_provider_calls_reserved": 15 + len(rows),
        "canary_retry_count": 3,
        "new_retry_count": sum(item.attempt_index == 2 for item in rows),
        "cumulative_retry_count": 3 + sum(item.attempt_index == 2 for item in rows),
        "new_retry_claim_sha256s": tuple(item.claim_sha256 for item in ledger.claims),
        "status": status,
        "terminal_reason": terminal_reason,
        "canary_retry_claims_reused": 0,
        "canary_provider_attempts_replayed": 0,
        "phase120_requires_new_owner_approval": True,
        "bundle_v11_publishable_from_phase60": False,
        "s1_creator_start_authorized": False,
        "usage_known_count": len(rows) - unknown,
        "usage_unknown_count": unknown,
        "input_tokens": sum(item.input_tokens or 0 for item in rows),
        "output_tokens": sum(item.output_tokens or 0 for item in rows),
        "prior_cumulative_actual_cost_cny": (
            ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY
        ),
        "new_actual_cost_cny": money(new_actual),
        "fresh_accountable_cost_cny": money(fresh_accountable),
        "cumulative_actual_cost_cny": money(cumulative_actual),
        "cumulative_accountable_cost_cny": money(cumulative_accountable),
        "artifacts": tuple(rows),
        "new_artifact_set_sha256": _hash_payload(
            [item.model_dump(mode="json") for item in rows]
        ),
    }
    return PortfolioS1FeedbackRound3SchemaPhase60RunV1.model_validate(
        {**unsigned, "run_sha256": _hash_payload(unsigned)}, strict=True
    )


def write_round3_schema_phase60_run_v1(
    path: str | Path, run: PortfolioS1FeedbackRound3SchemaPhase60RunV1
) -> Path:
    return _write_phase60_model(path, run)


def load_round3_schema_phase60_run_v1(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaPhase60RunV1:
    model = _load_canonical_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaPhase60RunV1,
        label="phase60 run",
        max_bytes=256 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaPhase60RunV1)
    return model


def write_round3_schema_phase60_owner_approval_v1(
    path: str | Path,
    approval: PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
) -> Path:
    return _write_phase60_model(path, approval)


def load_round3_schema_phase60_owner_approval_v1(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1:
    model = _load_canonical_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1,
        label="phase60 owner approval",
        max_bytes=2 * 1024 * 1024,
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1)
    return model


def write_round3_schema_phase60_authorization_v1(
    path: str | Path,
    authorization: PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
) -> Path:
    return _write_phase60_model(path, authorization)


def load_round3_schema_phase60_authorization_v1(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1:
    model = _load_canonical_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1,
        label="phase60 authorization",
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1)
    return model


def write_round3_schema_phase60_control_v1(
    path: str | Path,
    control: PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
) -> Path:
    return _write_phase60_model(path, control)


def load_round3_schema_phase60_control_v1(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaPhase60ControlV1:
    model = _load_canonical_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaPhase60ControlV1,
        label="phase60 control",
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaPhase60ControlV1)
    return model


def write_round3_schema_phase60_launch_v1(
    path: str | Path,
    launch: PortfolioS1FeedbackRound3SchemaPhase60LaunchV1,
) -> Path:
    return _write_phase60_model(path, launch)


def load_round3_schema_phase60_launch_v1(
    path: str | Path,
) -> PortfolioS1FeedbackRound3SchemaPhase60LaunchV1:
    model = _load_canonical_model(
        path,
        model_type=PortfolioS1FeedbackRound3SchemaPhase60LaunchV1,
        label="phase60 launch",
    )
    assert isinstance(model, PortfolioS1FeedbackRound3SchemaPhase60LaunchV1)
    return model


__all__ = [
    "BoundRound3SchemaPhase60FeedbackArtifactV1",
    "PortfolioS1FeedbackRound3SchemaCanary12ResultManifestV1",
    "PortfolioS1FeedbackRound3SchemaPhase60AuthorizationV1",
    "PortfolioS1FeedbackRound3SchemaPhase60ControlV1",
    "PortfolioS1FeedbackRound3SchemaPhase60LaunchV1",
    "PortfolioS1FeedbackRound3SchemaPhase60LedgerV1",
    "PortfolioS1FeedbackRound3SchemaPhase60OwnerApprovalV1",
    "PortfolioS1FeedbackRound3SchemaPhase60RunV1",
    "ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_FILE_SHA256_V1",
    "ROUND3_SCHEMA_CANARY12_RESULT_MANIFEST_RELATIVE_PATH_V1",
    "ROUND3_SCHEMA_PHASE60_ATTEMPT_DIR",
    "ROUND3_SCHEMA_PHASE60_BOUND_DIR",
    "ROUND3_SCHEMA_PHASE60_CLAIM_DIR",
    "ROUND3_SCHEMA_PHASE60_CUMULATIVE_HARD_CAP_CNY",
    "ROUND3_SCHEMA_PHASE60_CUMULATIVE_MAXIMUM_CNY",
    "ROUND3_SCHEMA_PHASE60_FRESH_HARD_CAP_CNY",
    "ROUND3_SCHEMA_PHASE60_GLOBAL_RETRY_CEILING",
    "ROUND3_SCHEMA_PHASE60_IMPORTED_ORDINALS",
    "ROUND3_SCHEMA_PHASE60_MAXIMUM_NEW_RESERVATION_CNY",
    "ROUND3_SCHEMA_PHASE60_NEW_ORDINALS",
    "ROUND3_SCHEMA_PHASE60_NEW_PRIMARY_COUNT",
    "ROUND3_SCHEMA_PHASE60_NEW_PROVIDER_CALL_CEILING",
    "ROUND3_SCHEMA_PHASE60_PER_CALL_RESERVATION_CNY",
    "ROUND3_SCHEMA_PHASE60_PRIOR_ACTUAL_COST_CNY",
    "ROUND3_SCHEMA_PHASE60_TARGET_COUNT",
    "Round3SchemaCanary12ArtifactReferenceV1",
    "Round3SchemaPhase60CallReservationV1",
    "Round3SchemaPhase60GlobalRetryClaimV1",
    "Round3SchemaPhase60NextStepV1",
    "Round3SchemaPhase60RunAttemptV1",
    "VerifiedPortfolioS1FeedbackRound3SchemaCanaryPrefixV1",
    "VerifiedPortfolioS1FeedbackRound3SchemaPhase60GovernanceV1",
    "build_bound_round3_schema_phase60_artifact_v1",
    "build_round3_schema_phase60_authorization_v1",
    "build_round3_schema_phase60_control_v1",
    "build_round3_schema_phase60_launch_v1",
    "build_round3_schema_phase60_owner_approval_v1",
    "build_round3_schema_phase60_reservation_v1",
    "build_round3_schema_phase60_retry_claim_v1",
    "build_round3_schema_phase60_run_v1",
    "load_bound_round3_schema_phase60_artifact_v1",
    "load_round3_schema_phase60_authorization_v1",
    "load_round3_schema_phase60_claim_v1",
    "load_round3_schema_phase60_control_v1",
    "load_round3_schema_phase60_launch_v1",
    "load_round3_schema_phase60_ledger_v1",
    "load_round3_schema_phase60_owner_approval_v1",
    "load_round3_schema_phase60_reservation_v1",
    "load_round3_schema_phase60_run_v1",
    "load_verified_round3_schema_canary_prefix_v1",
    "load_verified_round3_schema_phase60_governance_v1",
    "next_round3_schema_phase60_step_v1",
    "require_round3_schema_phase60_live_governance_v1",
    "require_round3_schema_phase60_pre_reservation_budget_v1",
    "round3_schema_phase60_attempt_filename_v1",
    "round3_schema_phase60_claim_filename_v1",
    "round3_schema_phase60_fresh_accountable_cost_v1",
    "validate_round3_schema_phase60_claim_set_v1",
    "validate_round3_schema_phase60_ledger_v1",
    "write_bound_round3_schema_phase60_artifact_v1",
    "write_round3_schema_phase60_authorization_v1",
    "write_round3_schema_phase60_claim_v1",
    "write_round3_schema_phase60_control_v1",
    "write_round3_schema_phase60_launch_v1",
    "write_round3_schema_phase60_owner_approval_v1",
    "write_round3_schema_phase60_reservation_v1",
    "write_round3_schema_phase60_run_v1",
]
