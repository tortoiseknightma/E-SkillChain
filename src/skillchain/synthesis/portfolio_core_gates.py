"""Frozen, text-free r2 validation gates and a create-only single-file publisher.

The r2 validation cohort is selected before any corpus turns exist.  This
module intentionally projects only the metadata that the existing group-atomic
split solver needs, then persists the result as one canonical JSON receipt.
It neither opens assets nor creates placeholder ``Query`` bodies, text, or
turns.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from skillchain.synthesis.planning import (
    R2CoreInMemoryPlan,
    audit_r2_core_in_memory_plan,
)
from skillchain.synthesis.portfolio_core_authoring import (
    ReuseAssignment,
    core_r2_realism_quota_spec,
    validate_realism_sidecar,
)
from skillchain.synthesis.portfolio_core_r2 import R2CoreBridge
from skillchain.synthesis.splitting import (
    CORE_R2_VALIDATION_GATE_SPEC,
    R2_CAPABILITY_ORDER,
    GateName,
    ValidationGateAudit,
    ValidationGateInfeasibleError,
    ValidationGateSpec,
    plan_validation_gates,
    validate_validation_gate_audit,
    verify_r2_split_constraint_binding,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GATE_BUNDLE_POLICY_VERSION = "portfolio-core-r2-text-free-validation-gates-v1"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_REQUIRED_INTERACTIONS: tuple[str, ...] = (
    "constraint_correction",
    "direct_request",
    "goal_change_or_multi_query",
    "no_result_relaxation",
    "underspecified_clarification",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def _require_sha256(value: str | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


class CoreR2ValidationGateError(ValueError):
    """The r2 plan, bridge, sidecar, or gate receipt has drifted."""


class CoreR2ValidationGatePublishError(RuntimeError):
    """A validation-gate receipt cannot be safely created at the target path."""


class ValidationGatePublicationConflict(CoreR2ValidationGatePublishError):
    """An existing regular file does not match this canonical gate receipt."""


class ValidationGatePublicationSafetyError(CoreR2ValidationGatePublishError):
    """A parent or target is a symlink, reparse point, or unsuitable entry."""


class R2ValidationGateEntry(_StrictModel):
    """One final validation assignment without user text, turns, or asset paths."""

    schema_version: Literal[1] = 1
    plan_id: str
    gate: GateName
    source_split: Literal["val"] = "val"
    canonical_intent: str
    canonical_capability: str
    is_boundary: bool
    interaction_pattern: str
    leakage_group_id: str
    boundary_group_id: str | None = None
    template_family: str
    generator_batch_id: str

    @field_validator(
        "plan_id",
        "canonical_intent",
        "canonical_capability",
        "interaction_pattern",
        "leakage_group_id",
        "boundary_group_id",
        "template_family",
        "generator_batch_id",
    )
    @classmethod
    def _validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)


class R2ValidationGateManifest(_StrictModel):
    """Digest binding for the complete deterministic 200-row gate receipt."""

    schema_version: Literal[1] = 1
    policy_version: Literal["portfolio-core-r2-text-free-validation-gates-v1"] = (
        GATE_BUNDLE_POLICY_VERSION
    )
    seed: int = Field(ge=0)
    plan_sha256: Sha256
    asset_catalog_sha256: Sha256
    capability_assignments_sha256: Sha256
    realism_manifest_sha256: Sha256
    realism_assignments_sha256: Sha256
    val_interactions_sha256: Sha256
    gate_spec_sha256: Sha256
    gate_rows_sha256: Sha256
    gate_mapping_sha256: Sha256
    gate_audit_sha256: Sha256
    bundle_payload_sha256: Sha256


@dataclass(frozen=True)
class CoreR2ValidationGateBundle:
    """Complete text-free gate mapping for later run setup and provenance checks."""

    entries: tuple[R2ValidationGateEntry, ...]
    audit: ValidationGateAudit
    spec: ValidationGateSpec
    manifest: R2ValidationGateManifest


@dataclass(frozen=True)
class _GateCandidate:
    """Private structural projection that satisfies ``plan_validation_gates``.

    The existing solver only reads these attributes.  Keeping this distinct
    from ``Query`` is intentional: creating a fake text/turn-bearing Query at
    this point would make ungenerated corpus content appear to exist.
    """

    plan_id: str
    canonical_intent: str
    canonical_capability: str
    is_boundary: bool
    leakage_group_id: str
    boundary_group_id: str | None
    template_family: str
    generator_batch_id: str

    @property
    def query_id(self) -> str:
        return self.plan_id

    @property
    def split(self) -> Literal["val"]:
        return "val"


CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC = ValidationGateSpec(
    sizes=CORE_R2_VALIDATION_GATE_SPEC.sizes,
    capability_minimums=CORE_R2_VALIDATION_GATE_SPEC.capability_minimums,
    boundary_minimums=CORE_R2_VALIDATION_GATE_SPEC.boundary_minimums,
    interaction_minimums={
        gate: {pattern: 1 for pattern in _REQUIRED_INTERACTIONS}
        for gate in ("route_gate", "body_gate", "shadow_val")
    },
)


def _plan_sha256(result: R2CoreInMemoryPlan) -> str:
    return sha256_bytes(canonical_json_bytes(result.plan))


def _final_splits(result: R2CoreInMemoryPlan) -> dict[str, str]:
    return {
        row.plan_id: result.final_split_by_plan_id[row.plan_id]
        for row in result.plan.queries
    }


def _reuse_sidecar(result: R2CoreInMemoryPlan) -> dict[str, ReuseAssignment]:
    return {
        row.plan_id: ReuseAssignment(
            plan_id=row.plan_id,
            reuse_variant=result.reuse_variant_by_plan_id[row.plan_id],
            reuse_reason=result.reuse_reason_by_plan_id[row.plan_id],
        )
        for row in result.plan.queries
    }


def _canonical_val_interactions_bytes(interactions: Mapping[str, str]) -> bytes:
    return canonical_jsonl_bytes(
        {
            "query_id": query_id,
            "interaction_pattern": interactions[query_id],
        }
        for query_id in sorted(interactions)
    )


def _validate_inputs(
    result: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
) -> tuple[tuple[_GateCandidate, ...], dict[str, str]]:
    """Validate r2 lineage and return only the val metadata needed by the solver."""

    try:
        trusted = _require_sha256(trusted_plan_sha256, "trusted_plan_sha256")
        actual = _plan_sha256(result)
        if actual != trusted:
            raise ValueError("trusted plan SHA-256 does not match the r2 plan")
        if audit_r2_core_in_memory_plan(result) != result.audit:
            raise ValueError("stored r2 plan audit drifted")
        catalog_sha = _require_sha256(
            result.plan.asset_catalog_sha256,
            "asset_catalog_sha256",
        )
        assignment_sha = _require_sha256(
            result.plan.capability_assignments_sha256,
            "capability_assignments_sha256",
        )
        verify_r2_split_constraint_binding(
            bridge.split_constraints,
            expected_plan_sha256=trusted,
            expected_asset_catalog_sha256=catalog_sha,
            expected_capability_assignments_sha256=assignment_sha,
        )
    except (TypeError, ValueError, AssertionError) as exc:
        raise CoreR2ValidationGateError("r2 plan/constraint binding failed") from exc

    final_splits = _final_splits(result)
    if bridge.split_constraints.plan_id_to_split != final_splits:
        raise CoreR2ValidationGateError("split constraints drifted from r2 final splits")
    realism = bridge.realism_sidecar
    if (
        realism.manifest.plan_sha256 != trusted
        or realism.manifest.asset_catalog_sha256 != catalog_sha
        or realism.manifest.capability_assignments_sha256 != assignment_sha
    ):
        raise CoreR2ValidationGateError("realism sidecar binding drifted from r2 plan")
    try:
        validate_realism_sidecar(
            realism,
            plan_rows=result.plan.queries,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=_reuse_sidecar(result),
            quota_spec=core_r2_realism_quota_spec(),
        )
    except (TypeError, ValueError) as exc:
        raise CoreR2ValidationGateError("realism sidecar validation failed") from exc

    assignments = {item.plan_id: item for item in realism.assignments}
    val_rows = tuple(
        _GateCandidate(
            plan_id=row.plan_id,
            canonical_intent=row.canonical_intent,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            leakage_group_id=row.leakage_group_id,
            boundary_group_id=row.boundary_group_id,
            template_family=row.template_family,
            generator_batch_id=row.generator_batch_id,
        )
        for row in result.plan.queries
        if final_splits[row.plan_id] == "val"
    )
    if len(val_rows) != 200 or len({row.plan_id for row in val_rows}) != 200:
        raise CoreR2ValidationGateError("r2 val projection must contain 200 unique rows")
    if {row.canonical_capability for row in val_rows} != set(R2_CAPABILITY_ORDER):
        raise CoreR2ValidationGateError("r2 val projection does not cover six capabilities")
    interactions = {
        row.plan_id: assignments[row.plan_id].interaction_pattern for row in val_rows
    }
    bridge_interactions = {
        query_id: pattern.strip()
        for query_id, pattern in bridge.val_interaction_by_query_id.items()
    }
    if bridge_interactions != interactions:
        raise CoreR2ValidationGateError("val interaction sidecar drifted from realism")
    if set(interactions.values()) != set(_REQUIRED_INTERACTIONS):
        raise CoreR2ValidationGateError(
            "r2 val realism must contain exactly the five required interactions"
        )
    return val_rows, interactions


def _entries_from_audit(
    candidates: Sequence[_GateCandidate],
    interactions: Mapping[str, str],
    audit: ValidationGateAudit,
) -> tuple[R2ValidationGateEntry, ...]:
    return tuple(
        R2ValidationGateEntry(
            plan_id=row.plan_id,
            gate=audit.query_id_to_gate[row.plan_id],
            canonical_intent=row.canonical_intent,
            canonical_capability=row.canonical_capability,
            is_boundary=row.is_boundary,
            interaction_pattern=interactions[row.plan_id],
            leakage_group_id=row.leakage_group_id,
            boundary_group_id=row.boundary_group_id,
            template_family=row.template_family,
            generator_batch_id=row.generator_batch_id,
        )
        for row in sorted(candidates, key=lambda item: item.plan_id)
    )


def canonical_r2_validation_gate_entries_bytes(
    entries: Sequence[R2ValidationGateEntry],
) -> bytes:
    """Canonical JSONL for exactly one complete 200-row text-free gate mapping."""

    plan_ids = [entry.plan_id for entry in entries]
    if len(entries) != 200 or len(plan_ids) != len(set(plan_ids)):
        raise ValueError("validation gate entries must contain 200 unique plan IDs")
    if plan_ids != sorted(plan_ids):
        raise ValueError("validation gate entries must be sorted by plan_id")
    return canonical_jsonl_bytes(entries)


def _canonical_mapping_bytes(entries: Sequence[R2ValidationGateEntry]) -> bytes:
    return canonical_jsonl_bytes(
        {"plan_id": entry.plan_id, "gate": entry.gate} for entry in entries
    )


def _bundle_payload_bytes(
    *,
    entries: Sequence[R2ValidationGateEntry],
    audit: ValidationGateAudit,
    spec: ValidationGateSpec,
    manifest: R2ValidationGateManifest,
) -> bytes:
    """Stable payload used for the manifest's non-circular self-binding hash."""

    manifest_without_payload_hash = manifest.model_dump(
        mode="json",
        exclude={"bundle_payload_sha256"},
    )
    return canonical_json_bytes(
        {
            "schema_version": 1,
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "audit": audit.model_dump(mode="json"),
            "gate_spec": spec.model_dump(mode="json"),
            "manifest": manifest_without_payload_hash,
        }
    )


def canonical_core_r2_validation_gate_bundle_bytes(
    bundle: CoreR2ValidationGateBundle,
) -> bytes:
    """Canonical single-file contents, including the validated self-binding hash."""

    return canonical_json_bytes(
        {
            "schema_version": 1,
            "entries": [entry.model_dump(mode="json") for entry in bundle.entries],
            "audit": bundle.audit.model_dump(mode="json"),
            "gate_spec": bundle.spec.model_dump(mode="json"),
            "manifest": bundle.manifest.model_dump(mode="json"),
        }
    )


def _build_manifest(
    *,
    result: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    trusted_plan_sha256: str,
    seed: int,
    entries: Sequence[R2ValidationGateEntry],
    audit: ValidationGateAudit,
    spec: ValidationGateSpec,
    interactions: Mapping[str, str],
) -> R2ValidationGateManifest:
    provisional = R2ValidationGateManifest(
        seed=seed,
        plan_sha256=trusted_plan_sha256,
        asset_catalog_sha256=_require_sha256(
            result.plan.asset_catalog_sha256,
            "asset_catalog_sha256",
        ),
        capability_assignments_sha256=_require_sha256(
            result.plan.capability_assignments_sha256,
            "capability_assignments_sha256",
        ),
        realism_manifest_sha256=sha256_bytes(
            canonical_json_bytes(bridge.realism_sidecar.manifest)
        ),
        realism_assignments_sha256=bridge.realism_sidecar.manifest.assignments_sha256,
        val_interactions_sha256=sha256_bytes(
            _canonical_val_interactions_bytes(interactions)
        ),
        gate_spec_sha256=sha256_bytes(canonical_json_bytes(spec)),
        gate_rows_sha256=sha256_bytes(canonical_r2_validation_gate_entries_bytes(entries)),
        gate_mapping_sha256=sha256_bytes(_canonical_mapping_bytes(entries)),
        gate_audit_sha256=sha256_bytes(canonical_json_bytes(audit)),
        bundle_payload_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "bundle_payload_sha256": sha256_bytes(
                _bundle_payload_bytes(
                    entries=entries,
                    audit=audit,
                    spec=spec,
                    manifest=provisional,
                )
            )
        }
    )


def build_core_r2_validation_gate_bundle(
    result: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
    seed: int,
    spec: ValidationGateSpec | None = None,
) -> CoreR2ValidationGateBundle:
    """Build the deterministic 75/75/50 r2 validation gate partition in memory."""

    if seed < 0:
        raise CoreR2ValidationGateError("validation gate seed must be non-negative")
    effective_spec = spec or CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC
    if effective_spec != CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC:
        raise CoreR2ValidationGateError("r2 text-free validation gate spec is frozen")
    candidates, interactions = _validate_inputs(
        result,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    try:
        audit = plan_validation_gates(
            list(candidates),
            seed=seed,
            spec=effective_spec,
            interaction_by_query_id=interactions,
        )
        validate_validation_gate_audit(
            list(candidates),
            audit,
            spec=effective_spec,
            interaction_by_query_id=interactions,
        )
    except (TypeError, ValueError, ValidationGateInfeasibleError) as exc:
        raise CoreR2ValidationGateError("r2 validation gate solver failed") from exc
    entries = _entries_from_audit(candidates, interactions, audit)
    manifest = _build_manifest(
        result=result,
        bridge=bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        seed=seed,
        entries=entries,
        audit=audit,
        spec=effective_spec,
        interactions=interactions,
    )
    bundle = CoreR2ValidationGateBundle(
        entries=entries,
        audit=audit,
        spec=effective_spec,
        manifest=manifest,
    )
    validate_core_r2_validation_gate_bundle(
        bundle,
        result,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    return bundle


def validate_core_r2_validation_gate_bundle(
    bundle: CoreR2ValidationGateBundle,
    result: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    *,
    trusted_plan_sha256: str,
) -> None:
    """Fail closed on source, mapping, manifest, deterministic, or hash drift."""

    if bundle.spec != CORE_R2_TEXT_FREE_VALIDATION_GATE_SPEC:
        raise CoreR2ValidationGateError("r2 text-free validation gate spec drifted")
    candidates, interactions = _validate_inputs(
        result,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    try:
        expected_audit = plan_validation_gates(
            list(candidates),
            seed=bundle.manifest.seed,
            spec=bundle.spec,
            interaction_by_query_id=interactions,
        )
        validate_validation_gate_audit(
            list(candidates),
            bundle.audit,
            spec=bundle.spec,
            interaction_by_query_id=interactions,
        )
    except (TypeError, ValueError, ValidationGateInfeasibleError) as exc:
        raise CoreR2ValidationGateError("validation gate audit is invalid") from exc
    expected_entries = _entries_from_audit(candidates, interactions, expected_audit)
    if bundle.audit != expected_audit or bundle.entries != expected_entries:
        raise CoreR2ValidationGateError("validation gate bundle drifted from deterministic solver")
    manifest = bundle.manifest
    expected_manifest = _build_manifest(
        result=result,
        bridge=bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        seed=manifest.seed,
        entries=expected_entries,
        audit=expected_audit,
        spec=bundle.spec,
        interactions=interactions,
    )
    if manifest != expected_manifest:
        raise CoreR2ValidationGateError("validation gate manifest binding drifted")
    try:
        row_bytes = canonical_r2_validation_gate_entries_bytes(bundle.entries)
    except ValueError as exc:
        raise CoreR2ValidationGateError("validation gate rows are malformed") from exc
    if manifest.gate_rows_sha256 != sha256_bytes(row_bytes):
        raise CoreR2ValidationGateError("validation gate rows digest mismatch")
    if manifest.gate_mapping_sha256 != sha256_bytes(_canonical_mapping_bytes(bundle.entries)):
        raise CoreR2ValidationGateError("validation gate mapping digest mismatch")
    payload_hash = sha256_bytes(
        _bundle_payload_bytes(
            entries=bundle.entries,
            audit=bundle.audit,
            spec=bundle.spec,
            manifest=manifest.model_copy(update={"bundle_payload_sha256": "0" * 64}),
        )
    )
    if manifest.bundle_payload_sha256 != payload_hash:
        raise CoreR2ValidationGateError("validation gate bundle self-hash mismatch")


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _require_real_parent(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidationGatePublicationSafetyError(
            f"unable to inspect output parent: {path}"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValidationGatePublicationSafetyError(
            "validation gate output parent must be a real non-reparse directory"
        )


def _require_safe_existing_target(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidationGatePublicationSafetyError(
            f"unable to inspect validation gate target: {path}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise ValidationGatePublicationSafetyError(
            "validation gate target must not be a link or reparse point"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationGatePublicationSafetyError(
            "validation gate target must be a regular file"
        )
    return True


def publish_core_r2_validation_gates(
    bundle: CoreR2ValidationGateBundle,
    *,
    output_path: str | Path,
    result: R2CoreInMemoryPlan,
    bridge: R2CoreBridge,
    trusted_plan_sha256: str,
) -> Path:
    """Create a single canonical gate file, or verify an identical prior one.

    ``output_path`` is caller-owned. Its parent must already be a real directory;
    the function never creates or overwrites parent trees, active pointers, or
    corpus content.
    """

    validate_core_r2_validation_gate_bundle(
        bundle,
        result,
        bridge,
        trusted_plan_sha256=trusted_plan_sha256,
    )
    content = canonical_core_r2_validation_gate_bundle_bytes(bundle)
    target = Path(output_path)
    _require_real_parent(target.parent)
    if _require_safe_existing_target(target):
        if target.read_bytes() != content:
            raise ValidationGatePublicationConflict(
                "existing validation gate file differs from canonical bytes"
            )
        return target
    try:
        atomic_create_file(target, content)
    except FileExistsError:
        if not _require_safe_existing_target(target):  # pragma: no cover - race guard
            raise ValidationGatePublicationSafetyError(
                "validation gate target disappeared during create-only publish"
            )
        if target.read_bytes() != content:
            raise ValidationGatePublicationConflict(
                "racing validation gate file differs from canonical bytes"
            )
    return target
