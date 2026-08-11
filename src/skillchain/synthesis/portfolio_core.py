"""Resumable Portfolio core-corpus orchestration.

This module deliberately lives beside, rather than inside, the formal
human-acceptance workflow.  A Portfolio core run records that each batch was
mechanically validated and automatically made usable under the owner's chosen
post-hoc sample-audit policy.  It never rewrites the formal dev_mini corpus or
pretends that automatic approval was human review.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.synthesis.batches import (
    AcceptedLedgerEntry,
    accept_generated_batch,
    compute_generation_input_sha256,
    corpus_status,
    stage_generated_batch,
    verify_accepted_corpus,
)
from skillchain.synthesis.models import (
    ActivePlanPointer,
    BatchManifest,
    PlannedQuery,
    SeedManifest,
)
from skillchain.synthesis.planning import (
    build_core_plan_from_resources,
    load_capability_assignments,
    load_plan,
    read_active_plan,
    write_core_plan,
    write_dev_mini_plan,
)
from skillchain.synthesis.seeds import load_accepted_seed_manifest
from skillchain.synthesis.splitting import (
    CORE_SPLIT_SPEC,
    FrozenSplitError,
    freeze_test_split,
    stratified_split,
    verify_frozen_split,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    atomic_publish_new_directory,
    atomic_replace_file,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    new_staging_directory,
    sha256_bytes,
)

_RUN_FILE = "portfolio-core-run.json"
_AUTO_LEDGER = "auto-approved-ledger.jsonl"
_CODEX_JOBS_DIRECTORY = "codex-generation-jobs"
_WORK_ORDER_FILE = "work-order.json"
_CHECKPOINT_FILE = "checkpoint.json"
_JOB_INBOX_DIRECTORY = "inbox"
_JOB_DRAFT_FILE = "draft.jsonl"
_JOB_DRAFT_MANIFEST_FILE = "draft.manifest.json"
_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,79}$")
_AUTO_POLICY = "mechanical_validation_auto_approved_usable_pending_sample_review_v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_identifier(value: str, field_name: str) -> str:
    value = value.strip()
    if not _ID_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_ID_RE.pattern!r}: {value!r}"
        )
    return value


class PortfolioCoreRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    status: Literal[
        "ready",
        "running",
        "auto_approved_usable_pending_sample_review",
        "approved",
        "rejected",
    ]
    approval_policy: Literal[
        "mechanical_validation_auto_approved_usable_pending_sample_review_v1"
    ] = _AUTO_POLICY
    target_query_count: Literal[1500] = 1500
    batch_size: Literal[25] = 25
    expected_batch_count: Literal[60] = 60
    core_plan_sha256: str
    dev_prefix_plan_sha256: str
    asset_catalog_sha256: str
    leakage_policy_version: str
    seed_set_sha256: str
    auto_approved_batch_ids: tuple[str, ...] = ()
    audit_sample_id: str | None = None
    supersedes_run: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("run_id", "audit_sample_id", "supersedes_run")
    @classmethod
    def validate_optional_identifiers(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_identifier(value, info.field_name)

    @field_validator("auto_approved_batch_ids")
    @classmethod
    def validate_batch_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("auto_approved_batch_ids must be unique")
        if tuple(sorted(value)) != value:
            raise ValueError("auto_approved_batch_ids must be sorted")
        return value


class AutoApprovalEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    batch_id: str
    base_batch_id: str
    revision: int = Field(ge=1)
    count: Literal[25] = 25
    results_sha256: str
    mechanically_approved_at: datetime
    approval_policy: Literal[
        "mechanical_validation_auto_approved_usable_pending_sample_review_v1"
    ] = _AUTO_POLICY


class AuditSampleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sample_id: str
    run_id: str
    sample_size: Literal[200] = 200
    seed: int
    source_queries_sha256: str
    sample_queries_sha256: str
    strata_counts: dict[str, int]
    created_at: datetime

    @field_validator("sample_id", "run_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)


class AuditReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sample_id: str
    run_id: str
    decision: Literal["pass", "reject"]
    reviewer_id: str
    review_minutes: int = Field(gt=0)
    reason: str | None = None
    recorded_at: datetime

    @field_validator("sample_id", "run_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reviewer_id must be non-blank")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class CodexImageReference(BaseModel):
    """The exact on-disk image a Codex author must inspect for one plan item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    image_path: str
    resolved_image_path: str


class CodexGenerationWorkOrder(BaseModel):
    """Immutable, deterministic author handoff for one portfolio core batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str
    run_id: str
    base_batch_id: str
    expected_revision: int = Field(ge=1)
    core_plan_sha256: str
    asset_catalog_sha256: str
    leakage_policy_version: str
    seed_set_sha256: str
    generation_input_sha256: str
    asset_root: str
    accepted_seed_examples_path: str
    accepted_seed_manifest_path: str
    plan_items: tuple[PlannedQuery, ...]
    image_references: tuple[CodexImageReference, ...]
    inbox_draft_path: str
    inbox_draft_manifest_path: str
    author_instructions: tuple[str, ...]
    draft_manifest_requirements: dict[str, str | int | None]
    created_at: datetime

    @field_validator("job_id", "run_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_plan_and_image_references(self):
        if len(self.plan_items) != 25:
            raise ValueError("Codex work order must contain exactly 25 plan items")
        if any(item.batch_id != self.base_batch_id for item in self.plan_items):
            raise ValueError("Codex work order plan items have the wrong base batch")
        if [item.position for item in self.plan_items] != list(range(1, 26)):
            raise ValueError("Codex work order plan items must cover positions 1..25")
        if len(self.image_references) != 25:
            raise ValueError("Codex work order must contain exactly 25 image references")
        if [reference.plan_id for reference in self.image_references] != [
            item.plan_id for item in self.plan_items
        ]:
            raise ValueError("Codex work order image references do not match plan items")
        return self


class CodexGenerationCheckpoint(BaseModel):
    """Mutable, monotonic checkpoint for one immutable Codex work order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str
    run_id: str
    base_batch_id: str
    expected_revision: int = Field(ge=1)
    work_order_sha256: str
    generation_input_sha256: str
    state: Literal["issued", "staged", "accepted"]
    staged_batch_id: str | None = None
    accepted_batch_id: str | None = None
    results_sha256: str | None = None
    updated_at: datetime

    @field_validator("job_id", "run_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_state_fields(self):
        if self.state == "issued":
            if any(
                value is not None
                for value in (
                    self.staged_batch_id,
                    self.accepted_batch_id,
                    self.results_sha256,
                )
            ):
                raise ValueError("issued Codex job checkpoint must not name batch output")
        elif self.state == "staged":
            if (
                self.staged_batch_id is None
                or self.results_sha256 is None
                or self.accepted_batch_id is not None
            ):
                raise ValueError("staged Codex job checkpoint is incomplete")
        elif (
            self.staged_batch_id is None
            or self.accepted_batch_id is None
            or self.results_sha256 is None
        ):
            raise ValueError("accepted Codex job checkpoint is incomplete")
        return self


def _run_path(run_root: Path) -> Path:
    return run_root / _RUN_FILE


def _load_run(run_root: str | Path) -> PortfolioCoreRun:
    path = _run_path(Path(run_root))
    try:
        return PortfolioCoreRun.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"portfolio core run is missing: {path}") from None
    except ValidationError as exc:
        raise ValueError(f"portfolio core run manifest is invalid: {exc}") from exc


def _write_run(run_root: str | Path, run: PortfolioCoreRun, *, create: bool = False) -> Path:
    path = _run_path(Path(run_root))
    content = canonical_json_bytes(run)
    return atomic_create_file(path, content) if create else atomic_replace_file(path, content)


def _copy_create(source: Path, destination: Path) -> Path:
    content = source.read_bytes()
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == content:
            return destination
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    return atomic_create_file(destination, content)


def _copy_accepted_seed(seed_source_root: Path, run_root: Path) -> SeedManifest:
    source_root = seed_source_root / "seeds" / "accepted"
    manifest = load_accepted_seed_manifest(seed_source_root)
    for name in ("seed_examples.json", "manifest.json"):
        _copy_create(source_root / name, run_root / "seeds" / "accepted" / name)
    copied = load_accepted_seed_manifest(run_root)
    if copied != manifest:
        raise ValueError("copied accepted seed manifest does not match source")
    return copied


def _codex_jobs_root(run_root: Path) -> Path:
    return run_root / _CODEX_JOBS_DIRECTORY


def _codex_job_root(run_root: Path, job_id: str) -> Path:
    return _codex_jobs_root(run_root) / _require_identifier(job_id, "job_id")


def _canonical_model_from_file(path: Path, model_type, *, label: str):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or is not a regular file: {path}")
    content = path.read_bytes()
    try:
        model = model_type.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    if content != canonical_json_bytes(model):
        raise ValueError(f"{label} is not canonical JSON: {path}")
    return model


def _work_order_identity_payload(order: CodexGenerationWorkOrder) -> dict:
    """Return stable work-order inputs, excluding publication-local fields."""

    payload = order.model_dump(mode="json")
    for field in (
        "job_id",
        "created_at",
        "inbox_draft_path",
        "inbox_draft_manifest_path",
    ):
        payload.pop(field)
    return payload


def _deterministic_codex_job_id(order: CodexGenerationWorkOrder) -> str:
    return f"codex-job-{sha256_bytes(canonical_json_bytes(_work_order_identity_payload(order)))[:24]}"


def _resolve_author_image_references(
    asset_root: str | Path, plan_items: tuple[PlannedQuery, ...]
) -> tuple[str, tuple[CodexImageReference, ...]]:
    root = Path(asset_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"asset root is missing or is not a regular directory: {root}")
    resolved_root = root.resolve(strict=True)
    references: list[CodexImageReference] = []
    for item in plan_items:
        relative = Path(item.image_path)
        if relative.is_absolute():
            raise ValueError(f"planned image path must be relative: {item.image_path}")
        try:
            resolved = (resolved_root / relative).resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"planned image is absent from the registered asset root: {item.image_path}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"planned image is not a regular file: {resolved}")
        references.append(
            CodexImageReference(
                plan_id=item.plan_id,
                image_path=item.image_path,
                resolved_image_path=str(resolved),
            )
        )
    return str(resolved_root), tuple(references)


def _work_order_plan_items(plan, base_batch_id: str) -> tuple[PlannedQuery, ...]:
    items = tuple(
        sorted(
            (item for item in plan.queries if item.batch_id == base_batch_id),
            key=lambda item: item.position,
        )
    )
    if len(items) != 25:
        raise ValueError(f"core work order requires exactly 25 plan items: {base_batch_id}")
    return items


def _build_codex_work_order(
    *,
    run_root: Path,
    run: PortfolioCoreRun,
    plan,
    expected_revision: int,
    base_batch_id: str,
    asset_root: str | Path,
) -> CodexGenerationWorkOrder:
    plan_items = _work_order_plan_items(plan, base_batch_id)
    resolved_asset_root, image_references = _resolve_author_image_references(
        asset_root, plan_items
    )
    generation_input_sha256 = compute_generation_input_sha256(
        base_batch_id=base_batch_id,
        plan_items=plan_items,
        plan_sha256=run.core_plan_sha256,
        asset_catalog_sha256=run.asset_catalog_sha256,
        leakage_policy_version=run.leakage_policy_version,
        seed_set_sha256=run.seed_set_sha256,
    )
    placeholder = CodexGenerationWorkOrder(
        job_id="codex-job-placeholder",
        run_id=run.run_id,
        base_batch_id=base_batch_id,
        expected_revision=expected_revision,
        core_plan_sha256=run.core_plan_sha256,
        asset_catalog_sha256=run.asset_catalog_sha256,
        leakage_policy_version=run.leakage_policy_version,
        seed_set_sha256=run.seed_set_sha256,
        generation_input_sha256=generation_input_sha256,
        asset_root=resolved_asset_root,
        accepted_seed_examples_path=str(
            (run_root / "seeds" / "accepted" / "seed_examples.json").resolve()
        ),
        accepted_seed_manifest_path=str(
            (run_root / "seeds" / "accepted" / "manifest.json").resolve()
        ),
        plan_items=plan_items,
        image_references=image_references,
        inbox_draft_path="pending deterministic job id",
        inbox_draft_manifest_path="pending deterministic job id",
        author_instructions=(
            "This handoff does not call an external LLM API; a Codex author composes the draft.",
            "Before image inspection or draft writing, satisfy the current-session model guard required by the corpus skill.",
            "Use the accepted seed examples only as style anchors; do not copy their text or change plan-owned fields.",
            "Inspect every listed resolved_image_path and write exactly 25 JSONL records containing only plan_id and turns.",
            "Write the companion manifest with the bound provenance below and the SHA-256 of the exact draft bytes.",
            "Submit only through approve-batch with this job_id; it will revalidate the plan, seed, catalog, and draft.",
        ),
        draft_manifest_requirements={
            "schema_version": 2,
            "base_batch_id": base_batch_id,
            "data_origin": "synthetic_derived",
            "provider": "codex",
            "model_display_name": "5.6 Sol Ultra",
            "model_claim_source": "user_confirmation",
            "generated_at": "author supplies an aware current UTC timestamp",
            "plan_sha256": run.core_plan_sha256,
            "asset_catalog_sha256": run.asset_catalog_sha256,
            "leakage_policy_version": run.leakage_policy_version,
            "seed_set_sha256": run.seed_set_sha256,
            "generation_input_sha256": generation_input_sha256,
            "draft_sha256": "SHA-256 of the exact draft.jsonl bytes",
        },
        created_at=_utc_now(),
    )
    job_id = _deterministic_codex_job_id(placeholder)
    inbox = _codex_job_root(run_root, job_id) / _JOB_INBOX_DIRECTORY
    return placeholder.model_copy(
        update={
            "job_id": job_id,
            "inbox_draft_path": str((inbox / _JOB_DRAFT_FILE).resolve()),
            "inbox_draft_manifest_path": str(
                (inbox / _JOB_DRAFT_MANIFEST_FILE).resolve()
            ),
        }
    )


def _load_codex_job(
    run_root: Path, job_id: str
) -> tuple[CodexGenerationWorkOrder, CodexGenerationCheckpoint]:
    root = _codex_job_root(run_root, job_id)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Codex generation job is missing: {root}")
    work_order = _canonical_model_from_file(
        root / _WORK_ORDER_FILE,
        CodexGenerationWorkOrder,
        label="Codex generation work order",
    )
    if work_order.job_id != job_id:
        raise ValueError("Codex generation job directory does not match work-order job_id")
    expected_job_id = _deterministic_codex_job_id(work_order)
    if work_order.job_id != expected_job_id:
        raise ValueError("Codex generation work order has an invalid deterministic job_id")
    checkpoint = _canonical_model_from_file(
        root / _CHECKPOINT_FILE,
        CodexGenerationCheckpoint,
        label="Codex generation checkpoint",
    )
    work_order_sha256 = sha256_bytes(canonical_json_bytes(work_order))
    if (
        checkpoint.job_id != work_order.job_id
        or checkpoint.run_id != work_order.run_id
        or checkpoint.base_batch_id != work_order.base_batch_id
        or checkpoint.expected_revision != work_order.expected_revision
        or checkpoint.generation_input_sha256 != work_order.generation_input_sha256
        or checkpoint.work_order_sha256 != work_order_sha256
    ):
        raise ValueError("Codex generation checkpoint does not bind to its work order")
    return work_order, checkpoint


def _write_codex_checkpoint(
    run_root: Path,
    work_order: CodexGenerationWorkOrder,
    existing: CodexGenerationCheckpoint,
    *,
    state: Literal["issued", "staged", "accepted"],
    staged_batch_id: str | None = None,
    accepted_batch_id: str | None = None,
    results_sha256: str | None = None,
) -> CodexGenerationCheckpoint:
    ranks = {"issued": 0, "staged": 1, "accepted": 2}
    desired = CodexGenerationCheckpoint(
        job_id=work_order.job_id,
        run_id=work_order.run_id,
        base_batch_id=work_order.base_batch_id,
        expected_revision=work_order.expected_revision,
        work_order_sha256=sha256_bytes(canonical_json_bytes(work_order)),
        generation_input_sha256=work_order.generation_input_sha256,
        state=state,
        staged_batch_id=staged_batch_id,
        accepted_batch_id=accepted_batch_id,
        results_sha256=results_sha256,
        updated_at=_utc_now(),
    )
    if ranks[desired.state] < ranks[existing.state]:
        raise ValueError("Codex generation checkpoint would move backward")
    if (
        desired.model_dump(mode="json", exclude={"updated_at"})
        == existing.model_dump(mode="json", exclude={"updated_at"})
    ):
        return existing
    path = _codex_job_root(run_root, work_order.job_id) / _CHECKPOINT_FILE
    atomic_replace_file(path, canonical_json_bytes(desired))
    return desired


def _load_canonical_batch_manifest(path: Path) -> BatchManifest:
    return _canonical_model_from_file(
        path, BatchManifest, label="Codex job staged batch manifest"
    )


def _reconcile_codex_job_checkpoint(
    run_root: Path,
    work_order: CodexGenerationWorkOrder,
    checkpoint: CodexGenerationCheckpoint,
) -> CodexGenerationCheckpoint:
    accepted_matches = [
        entry
        for entry in _accepted_entries(run_root)
        if entry.base_batch_id == work_order.base_batch_id
    ]
    if len(accepted_matches) > 1:
        raise ValueError("multiple accepted revisions match one Codex generation job")
    if accepted_matches:
        entry = accepted_matches[0]
        if entry.revision != work_order.expected_revision:
            raise ValueError("accepted batch revision does not match its Codex work order")
        return _write_codex_checkpoint(
            run_root,
            work_order,
            checkpoint,
            state="accepted",
            staged_batch_id=entry.batch_id,
            accepted_batch_id=entry.batch_id,
            results_sha256=entry.results_sha256,
        )

    staged = _find_active_staging(run_root, work_order.base_batch_id)
    if staged is not None:
        manifest = _load_canonical_batch_manifest(staged / "manifest.json")
        if (
            manifest.base_batch_id != work_order.base_batch_id
            or manifest.revision != work_order.expected_revision
            or manifest.plan_sha256 != work_order.core_plan_sha256
            or manifest.seed_set_sha256 != work_order.seed_set_sha256
        ):
            raise ValueError("staged batch does not match its Codex work order")
        return _write_codex_checkpoint(
            run_root,
            work_order,
            checkpoint,
            state="staged",
            staged_batch_id=staged.name,
            results_sha256=manifest.results_sha256,
        )
    if checkpoint.state != "issued":
        raise ValueError("Codex generation checkpoint names output that is no longer present")
    return checkpoint


def _validate_codex_work_order_binding(
    *,
    run_root: Path,
    run: PortfolioCoreRun,
    plan,
    work_order: CodexGenerationWorkOrder,
    asset_root: str | Path,
) -> None:
    if work_order.run_id != run.run_id:
        raise ValueError("Codex generation work order belongs to a different run")
    expected = _build_codex_work_order(
        run_root=run_root,
        run=run,
        plan=plan,
        expected_revision=work_order.expected_revision,
        base_batch_id=work_order.base_batch_id,
        asset_root=asset_root,
    )
    if (
        work_order.job_id != expected.job_id
        or _work_order_identity_payload(work_order)
        != _work_order_identity_payload(expected)
        or work_order.inbox_draft_path != expected.inbox_draft_path
        or work_order.inbox_draft_manifest_path != expected.inbox_draft_manifest_path
    ):
        raise ValueError("Codex generation work order no longer matches registered inputs")


def _work_order_result(
    run_root: Path,
    work_order: CodexGenerationWorkOrder,
    checkpoint: CodexGenerationCheckpoint,
    *,
    resumed: bool,
) -> dict:
    root = _codex_job_root(run_root, work_order.job_id)
    return {
        "job_id": work_order.job_id,
        "base_batch_id": work_order.base_batch_id,
        "expected_revision": work_order.expected_revision,
        "state": checkpoint.state,
        "work_order_path": str(root / _WORK_ORDER_FILE),
        "checkpoint_path": str(root / _CHECKPOINT_FILE),
        "draft_path": work_order.inbox_draft_path,
        "draft_manifest_path": work_order.inbox_draft_manifest_path,
        "resumed": resumed,
    }


def prepare_core_run(
    *,
    run_root: str | Path,
    run_id: str,
    data_root: str | Path,
    seed_source_root: str | Path,
    asset_catalog: AssetCatalog,
    capability_assignments_path: str | Path,
    expected_capability_assignments_sha256: str,
    seed: int = 20260711,
    supersedes_run: str | None = None,
) -> dict:
    """Create an isolated core run with a fresh catalog-bound 200-query prefix.

    It writes plans and state only.  No natural-language corpus content is
    generated or accepted by this operation.
    """

    run_root = Path(run_root)
    _require_identifier(run_id, "run_id")
    if supersedes_run is not None:
        _require_identifier(supersedes_run, "supersedes_run")
    asset_catalog.require_verified_files()
    source_seed_manifest = load_accepted_seed_manifest(seed_source_root)
    assignments = load_capability_assignments(
        capability_assignments_path,
        expected_sha256=expected_capability_assignments_sha256,
    )
    dev_prefix, core_plan = build_core_plan_from_resources(
        data_root,
        seed=seed,
        asset_catalog=asset_catalog,
        capability_assignments=assignments,
        expected_capability_assignments_sha256=expected_capability_assignments_sha256,
    )
    expected_prefix_sha256 = sha256_bytes(canonical_json_bytes(dev_prefix))
    expected_core_sha256 = sha256_bytes(canonical_json_bytes(core_plan))
    if run_root.exists():
        if not run_root.is_dir():
            raise ValueError(f"run root is not a directory: {run_root}")
        if not _run_path(run_root).is_file():
            raise FileExistsError(
                "run root contains an unpublished or foreign partial run; "
                "preserve it and choose a new run root"
            )
        existing = _load_run(run_root)
        expected = {
            "run_id": run_id,
            "core_plan_sha256": expected_core_sha256,
            "dev_prefix_plan_sha256": expected_prefix_sha256,
            "asset_catalog_sha256": asset_catalog.catalog_sha256,
            "leakage_policy_version": asset_catalog.leakage_policy_version,
            "seed_set_sha256": source_seed_manifest.seed_set_sha256,
            "supersedes_run": supersedes_run,
        }
        mismatches = [
            name
            for name, value in expected.items()
            if getattr(existing, name) != value
        ]
        if mismatches:
            raise FileExistsError(
                "existing portfolio core run has a different immutable identity: "
                + ", ".join(mismatches)
            )
        return core_run_status(run_root, asset_catalog=asset_catalog)

    # Build the run privately first.  The create-only publish below means an
    # interruption never exposes a half-prepared run at ``run_root``.
    staging_root = new_staging_directory(run_root)
    prefix_path, _ = write_dev_mini_plan(
        dev_prefix, staging_root / "plans" / "core-dev-prefix.json"
    )
    _, prefix_manifest = load_plan(prefix_path)
    core_path, _ = write_core_plan(
        core_plan,
        staging_root / "plans" / "core.json",
        parent_plan_sha256=prefix_manifest.plan_sha256,
    )
    _, core_manifest = load_plan(core_path)
    seed_manifest = _copy_accepted_seed(Path(seed_source_root), staging_root)
    pointer = ActivePlanPointer(
        scope="core",
        plan_path=str((run_root / "plans" / "core.json").resolve()),
        plan_sha256=core_manifest.plan_sha256,
        asset_catalog_sha256=core_manifest.asset_catalog_sha256,
        leakage_policy_version=core_manifest.leakage_policy_version,
    )
    atomic_create_file(
        staging_root / "plans" / "active.json", canonical_json_bytes(pointer)
    )
    now = _utc_now()
    run = PortfolioCoreRun(
        run_id=run_id,
        status="ready",
        core_plan_sha256=core_manifest.plan_sha256,
        dev_prefix_plan_sha256=prefix_manifest.plan_sha256,
        asset_catalog_sha256=asset_catalog.catalog_sha256,
        leakage_policy_version=asset_catalog.leakage_policy_version,
        seed_set_sha256=seed_manifest.seed_set_sha256,
        supersedes_run=supersedes_run,
        created_at=now,
        updated_at=now,
    )
    _write_run(staging_root, run, create=True)
    atomic_publish_new_directory(staging_root, run_root)
    return core_run_status(run_root, asset_catalog=asset_catalog)


def _load_auto_ledger(run_root: Path) -> list[AutoApprovalEntry]:
    path = run_root / _AUTO_LEDGER
    if not path.exists():
        return []
    if not path.is_file():
        raise ValueError(f"auto-approval ledger is not a file: {path}")
    entries: list[AutoApprovalEntry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank line in auto-approval ledger: {line_number}")
        try:
            entries.append(AutoApprovalEntry.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(
                f"invalid auto-approval ledger entry {line_number}: {exc}"
            ) from exc
    if len({entry.batch_id for entry in entries}) != len(entries):
        raise ValueError("duplicate batch_id in auto-approval ledger")
    return entries


def _write_auto_ledger(run_root: Path, entries: list[AutoApprovalEntry]) -> Path:
    ordered = sorted(entries, key=lambda entry: entry.base_batch_id)
    content = canonical_jsonl_bytes(ordered)
    path = run_root / _AUTO_LEDGER
    if path.exists() and path.read_bytes() == content:
        return path
    return atomic_replace_file(path, content)


def _accepted_entries(run_root: Path) -> list[AcceptedLedgerEntry]:
    ledger = run_root / "accepted-ledger.jsonl"
    if not ledger.exists():
        return []
    entries: list[AcceptedLedgerEntry] = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        try:
            entries.append(AcceptedLedgerEntry.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"invalid accepted ledger entry {line_number}: {exc}") from exc
    return entries


def _core_plan_batch_ids(run_root: Path) -> set[str]:
    _, plan, manifest, _ = read_active_plan(run_root)
    run = _load_run(run_root)
    if plan.scope != "core" or manifest.plan_sha256 != run.core_plan_sha256:
        raise ValueError("run state is not bound to its registered core plan")
    return {item.batch_id for item in plan.queries}


def _validate_seed_binding(run_root: Path, run: PortfolioCoreRun) -> None:
    seed_manifest = load_accepted_seed_manifest(run_root)
    if seed_manifest.seed_set_sha256 != run.seed_set_sha256:
        raise ValueError("accepted seed set no longer matches portfolio core run")


def _sync_auto_ledger(run_root: Path) -> list[AutoApprovalEntry]:
    """Recover bookkeeping after a process stopped between accept and state write."""

    run = _load_run(run_root)
    _validate_seed_binding(run_root, run)
    entries = _load_auto_ledger(run_root)
    allowed_bases = _core_plan_batch_ids(run_root)
    known = {entry.batch_id for entry in entries}
    accepted = _accepted_entries(run_root)
    accepted_by_batch_id = {entry.batch_id: entry for entry in accepted}
    if len(accepted_by_batch_id) != len(accepted):
        raise ValueError("accepted ledger contains duplicate batch_id")
    for auto_entry in entries:
        accepted_entry = accepted_by_batch_id.get(auto_entry.batch_id)
        if accepted_entry is None:
            raise ValueError(
                "auto-approval ledger references a batch absent from accepted ledger: "
                f"{auto_entry.batch_id}"
            )
        if auto_entry.base_batch_id not in allowed_bases:
            raise ValueError(
                "auto-approval ledger references a batch outside the core plan: "
                f"{auto_entry.base_batch_id}"
            )
        if (
            auto_entry.base_batch_id != accepted_entry.base_batch_id
            or auto_entry.revision != accepted_entry.revision
            or auto_entry.count != accepted_entry.count
            or auto_entry.results_sha256 != accepted_entry.results_sha256
        ):
            raise ValueError(
                "auto-approval ledger does not match accepted ledger: "
                f"{auto_entry.batch_id}"
            )
        if (
            accepted_entry.plan_sha256 != run.core_plan_sha256
            or accepted_entry.seed_set_sha256 != run.seed_set_sha256
        ):
            raise ValueError(
                "accepted ledger does not match the core run plan/seed binding: "
                f"{auto_entry.batch_id}"
            )
    additions: list[AutoApprovalEntry] = []
    for entry in accepted:
        if entry.batch_id in known:
            continue
        if entry.base_batch_id not in allowed_bases:
            raise ValueError(
                "portfolio core run contains an accepted batch outside its plan: "
                f"{entry.base_batch_id}"
            )
        if (
            entry.plan_sha256 != run.core_plan_sha256
            or entry.seed_set_sha256 != run.seed_set_sha256
        ):
            raise ValueError(
                "accepted ledger does not match the core run plan/seed binding: "
                f"{entry.batch_id}"
            )
        additions.append(
            AutoApprovalEntry(
                batch_id=entry.batch_id,
                base_batch_id=entry.base_batch_id,
                revision=entry.revision,
                count=entry.count,
                results_sha256=entry.results_sha256,
                mechanically_approved_at=entry.accepted_at,
            )
        )
    if additions:
        entries = [*entries, *additions]
        _write_auto_ledger(run_root, entries)
    return sorted(entries, key=lambda entry: entry.base_batch_id)


def _update_run_from_workflow(
    run_root: Path, run: PortfolioCoreRun
) -> PortfolioCoreRun:
    workflow = corpus_status(run_root)
    auto_entries = _sync_auto_ledger(run_root)
    if workflow["accepted_queries"] > run.target_query_count:
        raise ValueError("core run accepted more queries than its registered target")
    if workflow["accepted_queries"] != len(auto_entries) * run.batch_size:
        raise ValueError("auto-approval ledger does not match accepted query count")
    if workflow["accepted_batches"] != len(auto_entries):
        raise ValueError("auto-approval ledger does not match accepted batch count")
    # Partial runs remain resumable from their ledger/checkpoint alone.  The
    # stronger, full-corpus verification is intentionally deferred until the
    # moment this wrapper would report the completed auto-approved state.
    # Without it, a missing accepted directory or a tampered derived
    # queries.jsonl could be misreported as ready for the owner's 200-query
    # audit solely because the ledger count reached 1,500.
    if workflow["accepted_queries"] == run.target_query_count:
        try:
            verified = verify_accepted_corpus(run_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "cannot report completed portfolio core run: "
                "accepted corpus integrity verification failed"
            ) from exc
        if len(verified) != run.target_query_count:
            raise ValueError(
                "cannot report completed portfolio core run: accepted corpus "
                "count does not match the registered target"
            )
    if run.status in {"approved", "rejected"}:
        status = run.status
    elif workflow["accepted_queries"] == 0:
        status = "ready"
    elif workflow["accepted_queries"] == run.target_query_count:
        status = "auto_approved_usable_pending_sample_review"
    else:
        status = "running"
    batch_ids = tuple(entry.batch_id for entry in auto_entries)
    if status == run.status and batch_ids == run.auto_approved_batch_ids:
        return run
    updated = run.model_copy(
        update={
            "status": status,
            "auto_approved_batch_ids": batch_ids,
            "updated_at": _utc_now(),
        }
    )
    _write_run(run_root, updated)
    return updated


def core_run_status(
    run_root: str | Path, *, asset_catalog: AssetCatalog | None = None
) -> dict:
    run_root = Path(run_root)
    run = _load_run(run_root)
    plan_path, plan, manifest, pointer = read_active_plan(run_root)
    if plan.scope != "core" or manifest.plan_sha256 != run.core_plan_sha256:
        raise ValueError("run state is not bound to its registered core plan")
    if asset_catalog is not None:
        asset_catalog.require_verified_files()
        if asset_catalog.catalog_sha256 != run.asset_catalog_sha256:
            raise ValueError("asset catalog hash does not match portfolio core run")
    _validate_seed_binding(run_root, run)
    run = _update_run_from_workflow(run_root, run)
    workflow = corpus_status(run_root)
    return {
        "run_id": run.run_id,
        "status": run.status,
        "approval_policy": run.approval_policy,
        "target_queries": run.target_query_count,
        "auto_approved_queries": workflow["accepted_queries"],
        "auto_approved_batches": workflow["accepted_batches"],
        "next_batch_id": workflow["next_batch_id"],
        "next_revision": workflow["next_revision"],
        "audit_sample_id": run.audit_sample_id,
        "core_plan_path": str(plan_path),
        "core_plan_sha256": pointer.plan_sha256,
    }


def emit_or_retrieve_codex_work_order(
    *,
    run_root: str | Path,
    asset_catalog: AssetCatalog,
    asset_root: str | Path,
    job_id: str | None = None,
) -> dict:
    """Create-only publish or safely retrieve the next Codex author handoff.

    The work order contains no synthesized query text.  It is an immutable
    binding of one next batch to the exact plan, seed, catalog, image files,
    and expected revision.  The adjacent checkpoint is reconciled from the
    existing staging/accepted state so a stopped author session can resume
    without inventing a second job or reusing a different batch.
    """

    run_root = Path(run_root)
    run = _load_run(run_root)
    asset_catalog.require_verified_files()
    if asset_catalog.catalog_sha256 != run.asset_catalog_sha256:
        raise ValueError("asset catalog hash does not match portfolio core run")
    _validate_seed_binding(run_root, run)
    _, plan, manifest, _ = read_active_plan(run_root)
    if plan.scope != "core" or manifest.plan_sha256 != run.core_plan_sha256:
        raise ValueError("active plan is not this run's core plan")
    run = _update_run_from_workflow(run_root, run)

    if job_id is not None:
        work_order, checkpoint = _load_codex_job(run_root, job_id)
        _validate_codex_work_order_binding(
            run_root=run_root,
            run=run,
            plan=plan,
            work_order=work_order,
            asset_root=asset_root,
        )
        checkpoint = _reconcile_codex_job_checkpoint(
            run_root, work_order, checkpoint
        )
        return _work_order_result(run_root, work_order, checkpoint, resumed=True)

    if run.status in {"approved", "rejected"}:
        raise ValueError(f"run is terminal and cannot issue a Codex job: {run.status}")
    workflow = corpus_status(run_root)
    base_batch_id = workflow["next_batch_id"]
    expected_revision = workflow["next_revision"]
    if base_batch_id is None:
        raise ValueError("all core batches are already accepted; no next Codex job exists")
    if expected_revision is None:
        raise ValueError("next core batch has no recoverable revision checkpoint")
    work_order = _build_codex_work_order(
        run_root=run_root,
        run=run,
        plan=plan,
        expected_revision=expected_revision,
        base_batch_id=base_batch_id,
        asset_root=asset_root,
    )
    existing_for_next = _matching_codex_job_for_next_batch(
        run_root=run_root,
        run=run,
        plan=plan,
        base_batch_id=base_batch_id,
        expected_revision=expected_revision,
    )
    if existing_for_next is not None:
        existing_order, checkpoint = existing_for_next
        if (
            _work_order_identity_payload(existing_order)
            != _work_order_identity_payload(work_order)
            or existing_order.inbox_draft_path != work_order.inbox_draft_path
            or existing_order.inbox_draft_manifest_path
            != work_order.inbox_draft_manifest_path
        ):
            raise ValueError(
                "an existing Codex generation job binds this batch to different "
                "immutable inputs"
            )
        checkpoint = _reconcile_codex_job_checkpoint(
            run_root, existing_order, checkpoint
        )
        return _work_order_result(run_root, existing_order, checkpoint, resumed=True)
    destination = _codex_job_root(run_root, work_order.job_id)
    if destination.exists():
        existing_order, checkpoint = _load_codex_job(run_root, work_order.job_id)
        if (
            _work_order_identity_payload(existing_order)
            != _work_order_identity_payload(work_order)
            or existing_order.inbox_draft_path != work_order.inbox_draft_path
            or existing_order.inbox_draft_manifest_path
            != work_order.inbox_draft_manifest_path
        ):
            raise ValueError("existing Codex generation job has different immutable inputs")
        checkpoint = _reconcile_codex_job_checkpoint(
            run_root, existing_order, checkpoint
        )
        return _work_order_result(run_root, existing_order, checkpoint, resumed=True)

    staging = new_staging_directory(destination)
    initial_checkpoint = CodexGenerationCheckpoint(
        job_id=work_order.job_id,
        run_id=work_order.run_id,
        base_batch_id=work_order.base_batch_id,
        expected_revision=work_order.expected_revision,
        work_order_sha256=sha256_bytes(canonical_json_bytes(work_order)),
        generation_input_sha256=work_order.generation_input_sha256,
        state="issued",
        updated_at=_utc_now(),
    )
    # Keep an interrupted unpublished staging directory for diagnosis.  The
    # destination itself remains create-only, so no partially written order is
    # ever exposed as an active job.
    (staging / _JOB_INBOX_DIRECTORY).mkdir()
    atomic_create_file(staging / _WORK_ORDER_FILE, canonical_json_bytes(work_order))
    atomic_create_file(staging / _CHECKPOINT_FILE, canonical_json_bytes(initial_checkpoint))
    atomic_publish_new_directory(staging, destination)
    return _work_order_result(run_root, work_order, initial_checkpoint, resumed=False)


def _find_active_staging(run_root: Path, base_batch_id: str) -> Path | None:
    paths = sorted((run_root / "staging").glob(f"{base_batch_id}-r*"))
    if not paths:
        return None
    if len(paths) != 1 or not paths[0].is_dir():
        raise ValueError(f"ambiguous staging for {base_batch_id}")
    return paths[0]


def _matching_codex_job_for_next_batch(
    *,
    run_root: Path,
    run: PortfolioCoreRun,
    plan,
    base_batch_id: str,
    expected_revision: int,
) -> tuple[CodexGenerationWorkOrder, CodexGenerationCheckpoint] | None:
    jobs_root = _codex_jobs_root(run_root)
    if not jobs_root.exists():
        return None
    if jobs_root.is_symlink() or not jobs_root.is_dir():
        raise ValueError(f"Codex generation jobs root is not a regular directory: {jobs_root}")
    matches: list[tuple[CodexGenerationWorkOrder, CodexGenerationCheckpoint]] = []
    for child in sorted(jobs_root.iterdir(), key=lambda path: path.name):
        # Interrupted create-only publication staging directories are preserved
        # for diagnosis and are not active jobs.
        if child.name.startswith("."):
            continue
        if child.is_symlink() or not child.is_dir():
            raise ValueError(f"invalid entry in Codex generation jobs root: {child}")
        work_order, checkpoint = _load_codex_job(run_root, child.name)
        if (
            work_order.run_id == run.run_id
            and work_order.base_batch_id == base_batch_id
            and work_order.expected_revision == expected_revision
        ):
            _validate_codex_work_order_binding(
                run_root=run_root,
                run=run,
                plan=plan,
                work_order=work_order,
                asset_root=work_order.asset_root,
            )
            matches.append((work_order, checkpoint))
    if len(matches) > 1:
        raise ValueError("multiple active Codex generation jobs match the next core batch")
    return matches[0] if matches else None


def auto_approve_core_batch(
    *,
    run_root: str | Path,
    base_batch_id: str,
    asset_catalog: AssetCatalog,
    draft_path: str | Path | None = None,
    draft_manifest_path: str | Path | None = None,
    job_id: str | None = None,
) -> dict:
    """Stage then mechanically approve the next core batch, with restart recovery."""

    run_root = Path(run_root)
    run = _load_run(run_root)
    if run.status in {"approved", "rejected"}:
        raise ValueError(f"run is terminal and cannot accept a batch: {run.status}")
    asset_catalog.require_verified_files()
    if asset_catalog.catalog_sha256 != run.asset_catalog_sha256:
        raise ValueError("asset catalog hash does not match portfolio core run")
    _validate_seed_binding(run_root, run)
    plan_path, plan, manifest, _ = read_active_plan(run_root)
    if plan.scope != "core" or manifest.plan_sha256 != run.core_plan_sha256:
        raise ValueError("active plan is not this run's core plan")
    # A process can stop after the immutable accepted directory/ledger publish
    # but before the wrapper's auto-approval state update.  Reconcile first,
    # then make a retry for that just-finished batch idempotent.
    run = _update_run_from_workflow(run_root, run)
    already_accepted = [
        entry
        for entry in _accepted_entries(run_root)
        if entry.base_batch_id == base_batch_id
    ]
    if already_accepted:
        if len(already_accepted) != 1:
            raise ValueError(f"multiple accepted revisions for {base_batch_id}")
        accepted_entry = already_accepted[0]
        bound_job: CodexGenerationWorkOrder | None = None
        checkpoint: CodexGenerationCheckpoint | None = None
        if job_id is not None:
            bound_job, checkpoint = _load_codex_job(run_root, job_id)
            _validate_codex_work_order_binding(
                run_root=run_root,
                run=run,
                plan=plan,
                work_order=bound_job,
                asset_root=bound_job.asset_root,
            )
            if (
                bound_job.base_batch_id != base_batch_id
                or bound_job.expected_revision != accepted_entry.revision
            ):
                raise ValueError("Codex job does not bind to the accepted core batch")
        accepted = accept_generated_batch(
            run_root,
            accepted_entry.batch_id,
            confirmation="ACCEPT",
            plan_path=plan_path,
            asset_catalog=asset_catalog,
        )
        if bound_job is not None and checkpoint is not None:
            _reconcile_codex_job_checkpoint(run_root, bound_job, checkpoint)
        return {
            **core_run_status(run_root, asset_catalog=asset_catalog),
            "batch_id": accepted.name,
            "accepted_path": str(accepted),
            "resumed": True,
            **({"job_id": bound_job.job_id} if bound_job is not None else {}),
        }

    status = corpus_status(run_root)
    expected = status["next_batch_id"]
    if expected is None:
        run = _update_run_from_workflow(run_root, run)
        return {**core_run_status(run_root, asset_catalog=asset_catalog), "batch_id": None}
    if base_batch_id != expected:
        raise ValueError(f"expected next batch {expected}, got {base_batch_id}")

    expected_revision = status["next_revision"]
    if expected_revision is None:
        raise ValueError("next core batch has no recoverable revision checkpoint")
    bound_job: CodexGenerationWorkOrder | None = None
    checkpoint: CodexGenerationCheckpoint | None = None
    if job_id is not None:
        bound_job, checkpoint = _load_codex_job(run_root, job_id)
        _validate_codex_work_order_binding(
            run_root=run_root,
            run=run,
            plan=plan,
            work_order=bound_job,
            asset_root=bound_job.asset_root,
        )
        if (
            bound_job.base_batch_id != base_batch_id
            or bound_job.expected_revision != expected_revision
        ):
            raise ValueError("Codex job does not bind to the current next core batch")
        checkpoint = _reconcile_codex_job_checkpoint(run_root, bound_job, checkpoint)
    else:
        existing_job = _matching_codex_job_for_next_batch(
            run_root=run_root,
            run=run,
            plan=plan,
            base_batch_id=base_batch_id,
            expected_revision=expected_revision,
        )
        if existing_job is not None:
            raise ValueError(
                "a Codex work order exists for this batch; pass its --job-id "
                "to bind the mechanical approval"
            )

    staged = _find_active_staging(run_root, base_batch_id)
    if bound_job is not None:
        expected_draft = Path(bound_job.inbox_draft_path).resolve()
        expected_manifest = Path(bound_job.inbox_draft_manifest_path).resolve()
        if (draft_path is None) != (draft_manifest_path is None):
            raise ValueError("draft and draft manifest must be provided together")
        if draft_path is None:
            draft_path = expected_draft
            draft_manifest_path = expected_manifest
        elif (
            Path(draft_path).resolve() != expected_draft
            or Path(draft_manifest_path).resolve() != expected_manifest
        ):
            raise ValueError("Codex job approval must use its immutable inbox paths")
    if staged is None:
        if (draft_path is None) != (draft_manifest_path is None):
            raise ValueError("draft and draft manifest must be provided together")
        if draft_path is None:
            raise ValueError(
                "no staged batch to resume; provide the next draft and its manifest"
            )
        staged = stage_generated_batch(
            draft_path=draft_path,
            draft_manifest_path=draft_manifest_path,
            plan_path=plan_path,
            queries_root=run_root,
            base_batch_id=base_batch_id,
            asset_catalog=asset_catalog,
        )
        if bound_job is not None and checkpoint is not None:
            staged_manifest = _load_canonical_batch_manifest(staged / "manifest.json")
            checkpoint = _write_codex_checkpoint(
                run_root,
                bound_job,
                checkpoint,
                state="staged",
                staged_batch_id=staged.name,
                results_sha256=staged_manifest.results_sha256,
            )
    accepted = accept_generated_batch(
        run_root,
        staged.name,
        confirmation="ACCEPT",
        plan_path=plan_path,
        asset_catalog=asset_catalog,
    )
    if bound_job is not None and checkpoint is not None:
        _reconcile_codex_job_checkpoint(run_root, bound_job, checkpoint)
    run = _update_run_from_workflow(run_root, run)
    return {
        **core_run_status(run_root, asset_catalog=asset_catalog),
        "batch_id": accepted.name,
        "accepted_path": str(accepted),
        **({"job_id": bound_job.job_id} if bound_job is not None else {}),
    }


def _stratum_key(query) -> str:
    return (
        f"{query.split}|{query.canonical_intent}|{query.canonical_capability}|"
        f"boundary={str(query.is_boundary).lower()}"
    )


def _allocate_sample_quotas(
    strata: dict[str, list], *, sample_size: int
) -> dict[str, int]:
    total = sum(len(values) for values in strata.values())
    if sample_size != 200:
        raise ValueError("portfolio core audit sample must contain exactly 200 queries")
    if total < sample_size:
        raise ValueError(f"cannot sample {sample_size} from only {total} queries")
    quotas = {
        key: min(len(values), (len(values) * sample_size) // total)
        for key, values in strata.items()
    }
    remaining = sample_size - sum(quotas.values())
    candidates = sorted(
        strata,
        key=lambda key: (
            -((len(strata[key]) * sample_size) % total),
            key,
        ),
    )
    for key in candidates:
        if remaining == 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("sample quota allocator ran out of capacity")
    return quotas


def create_audit_sample(
    *,
    run_root: str | Path,
    sample_id: str,
    seed: int,
    asset_catalog: AssetCatalog,
) -> dict:
    """Publish a deterministic, stratified 200-query owner audit package."""

    run_root = Path(run_root)
    _require_identifier(sample_id, "sample_id")
    run = _load_run(run_root)
    asset_catalog.require_verified_files()
    if asset_catalog.catalog_sha256 != run.asset_catalog_sha256:
        raise ValueError("asset catalog hash does not match portfolio core run")
    run = _update_run_from_workflow(run_root, run)
    if run.status not in {
        "auto_approved_usable_pending_sample_review",
        "approved",
    }:
        raise ValueError("audit sample is available only after all 1,500 queries approve")
    if run.audit_sample_id is not None and run.audit_sample_id != sample_id:
        raise ValueError(
            "this run already has an immutable audit sample: "
            f"{run.audit_sample_id}"
        )
    queries = verify_accepted_corpus(run_root)
    if len(queries) != 1500:
        raise ValueError("accepted core corpus does not contain exactly 1,500 queries")
    source_bytes = (run_root / "queries.jsonl").read_bytes()
    strata: dict[str, list] = defaultdict(list)
    for query in queries:
        strata[_stratum_key(query)].append(query)
    quotas = _allocate_sample_quotas(strata, sample_size=200)
    rng = random.Random(seed)
    selected = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda query: query.query_id)
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[key]])
    selected.sort(key=lambda query: query.query_id)
    if len(selected) != 200:
        raise AssertionError("audit sampler did not select exactly 200 queries")
    sample_bytes = canonical_jsonl_bytes(selected)
    sample_manifest = AuditSampleManifest(
        sample_id=sample_id,
        run_id=run.run_id,
        seed=seed,
        source_queries_sha256=sha256_bytes(source_bytes),
        sample_queries_sha256=sha256_bytes(sample_bytes),
        strata_counts={key: quotas[key] for key in sorted(quotas) if quotas[key]},
        created_at=_utc_now(),
    )
    destination = run_root / "audit-samples" / sample_id
    if destination.exists():
        manifest_path = destination / "manifest.json"
        existing_bytes = manifest_path.read_bytes()
        existing = AuditSampleManifest.model_validate_json(existing_bytes)
        if existing_bytes != canonical_json_bytes(existing):
            raise ValueError("existing audit sample manifest is not canonical JSON")
        if (
            existing.run_id == sample_manifest.run_id
            and existing.seed == sample_manifest.seed
            and existing.source_queries_sha256 == sample_manifest.source_queries_sha256
            and existing.sample_queries_sha256 == sample_manifest.sample_queries_sha256
            and (destination / "queries.jsonl").read_bytes() == sample_bytes
        ):
            if run.audit_sample_id != sample_id:
                _write_run(
                    run_root,
                    run.model_copy(
                        update={
                            "audit_sample_id": sample_id,
                            "updated_at": _utc_now(),
                        }
                    ),
                )
            return {
                "sample_id": sample_id,
                "path": str(destination),
                "count": 200,
                "resumed": True,
            }
        raise FileExistsError(f"audit sample already exists with different content: {destination}")
    staging = new_staging_directory(destination)
    try:
        atomic_create_file(staging / "queries.jsonl", sample_bytes)
        atomic_create_file(staging / "manifest.json", canonical_json_bytes(sample_manifest))
        atomic_publish_new_directory(staging, destination)
    except BaseException:
        # Preserve an unpublished staging directory for diagnosis/recovery;
        # sample publishing is create-only and never overwrites a prior audit.
        raise
    updated = run.model_copy(
        update={
            "status": (
                "approved"
                if run.status == "approved"
                else "auto_approved_usable_pending_sample_review"
            ),
            "audit_sample_id": sample_id,
            "updated_at": _utc_now(),
        }
    )
    _write_run(run_root, updated)
    return {
        "sample_id": sample_id,
        "path": str(destination),
        "count": 200,
        "source_queries_sha256": sample_manifest.source_queries_sha256,
        "sample_queries_sha256": sample_manifest.sample_queries_sha256,
        "strata": sample_manifest.strata_counts,
    }


def record_audit_review(
    *,
    run_root: str | Path,
    sample_id: str,
    decision: Literal["pass", "reject"],
    reviewer_id: str,
    review_minutes: int,
    reason: str | None = None,
) -> dict:
    """Record the owner's 200-query sample decision without mutating corpus data."""

    run_root = Path(run_root)
    run = _load_run(run_root)
    if run.audit_sample_id != sample_id:
        raise ValueError("sample_id is not the active audit sample for this run")
    if run.status not in {
        "auto_approved_usable_pending_sample_review",
        "approved",
        "rejected",
    }:
        raise ValueError("audit review is unavailable until the core run is complete")
    sample_root = run_root / "audit-samples" / sample_id
    manifest_bytes = (sample_root / "manifest.json").read_bytes()
    manifest = AuditSampleManifest.model_validate_json(manifest_bytes)
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ValueError("audit sample manifest is not canonical JSON")
    if manifest.run_id != run.run_id:
        raise ValueError("audit sample belongs to a different core run")
    sample_bytes = (sample_root / "queries.jsonl").read_bytes()
    if sha256_bytes(sample_bytes) != manifest.sample_queries_sha256:
        raise ValueError("audit sample content hash does not match its manifest")
    accepted = verify_accepted_corpus(run_root)
    if len(accepted) != run.target_query_count:
        raise ValueError("current accepted corpus does not contain the registered 1,500")
    source_bytes = (run_root / "queries.jsonl").read_bytes()
    if sha256_bytes(source_bytes) != manifest.source_queries_sha256:
        raise ValueError("audit sample is not bound to the current accepted corpus")
    review = AuditReview(
        sample_id=sample_id,
        run_id=run.run_id,
        decision=decision,
        reviewer_id=reviewer_id,
        review_minutes=review_minutes,
        reason=reason,
        recorded_at=_utc_now(),
    )
    if decision == "reject" and review.reason is None:
        raise ValueError("a rejected audit sample requires a non-blank reason")
    review_path = sample_root / "review.json"
    if review_path.exists():
        existing_bytes = review_path.read_bytes()
        try:
            existing = AuditReview.model_validate_json(existing_bytes)
        except ValidationError as exc:
            raise ValueError(f"existing audit review is invalid: {exc}") from exc
        if existing_bytes != canonical_json_bytes(existing):
            raise ValueError("existing audit review is not canonical JSON")
        if (
            existing.sample_id != sample_id
            or existing.run_id != run.run_id
            or existing.decision != decision
            or existing.reviewer_id != review.reviewer_id
            or existing.review_minutes != review.review_minutes
            or existing.reason != review.reason
        ):
            raise FileExistsError("audit review already exists and cannot be replaced")
        review = existing
    else:
        atomic_create_file(review_path, canonical_json_bytes(review))
    target_status: Literal["approved", "rejected"] = (
        "approved" if review.decision == "pass" else "rejected"
    )
    if run.status in {"approved", "rejected"} and run.status != target_status:
        raise ValueError("terminal run status conflicts with immutable audit review")
    updated = run.model_copy(
        update={
            "status": target_status,
            "updated_at": _utc_now(),
        }
    )
    if updated != run:
        _write_run(run_root, updated)
    return {
        "run_id": run.run_id,
        "status": updated.status,
        "sample_id": sample_id,
        "review_path": str(review_path),
    }


def freeze_portfolio_core_splits(
    *,
    run_root: str | Path,
    seed: int,
    asset_catalog: AssetCatalog,
) -> dict:
    """Freeze the core 200/800/200/300 split after owner audit approval.

    The core workflow intentionally does not masquerade as the formal
    cross-label pipeline.  It nevertheless reuses the exact group-constrained
    solver and immutable frozen-test files, binding them to the audited core
    run and its verified catalog.
    """

    run_root = Path(run_root)
    run = _load_run(run_root)
    if run.status != "approved":
        raise ValueError("core splits may be frozen only after the 200-query audit passes")
    asset_catalog.require_verified_files()
    if asset_catalog.catalog_sha256 != run.asset_catalog_sha256:
        raise ValueError("asset catalog hash does not match portfolio core run")
    _validate_seed_binding(run_root, run)
    plan_path, plan, manifest, _ = read_active_plan(run_root)
    if plan.scope != "core" or manifest.plan_sha256 != run.core_plan_sha256:
        raise ValueError("active plan is not this run's core plan")
    candidates = verify_accepted_corpus(run_root)
    if len(candidates) != run.target_query_count:
        raise ValueError("accepted core corpus does not contain exactly 1,500 queries")
    locked_dev_query_ids = {item.plan_id for item in plan.queries[:200]}
    assigned = stratified_split(
        candidates,
        locked_dev_query_ids=locked_dev_query_ids,
        seed=seed,
        split_spec=CORE_SPLIT_SPEC,
    )
    test_path, manifest_path = freeze_test_split(
        run_root,
        assigned,
        seed=seed,
        full_plan_path=plan_path,
        asset_catalog=asset_catalog,
        split_spec=CORE_SPLIT_SPEC,
    )
    frozen = verify_frozen_split(
        run_root,
        asset_catalog=asset_catalog,
        full_plan_path=plan_path,
        expected_full_plan_sha256=manifest.plan_sha256,
    )
    if frozen is None:  # pragma: no cover - create/verify contract guard
        raise FrozenSplitError("core frozen split manifest was not published")
    return {
        "profile": frozen.profile,
        "split_assignment": str(run_root / "split_assignment.jsonl"),
        "test_frozen": str(test_path),
        "test_frozen_manifest": str(manifest_path),
        "split_sizes": frozen.split_sizes,
        "assignment_sha256": frozen.assignment_sha256,
        "core_plan_sha256": frozen.full_plan_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio core synthesis preparation and recovery workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--seed-source-root", type=Path, required=True)
    prepare.add_argument("--asset-catalog", type=Path, required=True)
    prepare.add_argument("--asset-root", type=Path, required=True)
    prepare.add_argument("--capability-assignments", type=Path, required=True)
    prepare.add_argument("--expected-capability-assignments-sha256", required=True)
    prepare.add_argument("--seed", type=int, default=20260711)
    prepare.add_argument("--supersedes-run")

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.add_argument("--asset-catalog", type=Path)
    status.add_argument("--asset-root", type=Path)

    work_order = subparsers.add_parser("work-order")
    work_order.add_argument("--run-root", type=Path, required=True)
    work_order.add_argument("--asset-catalog", type=Path, required=True)
    work_order.add_argument("--asset-root", type=Path, required=True)
    work_order.add_argument("--job-id")

    approve = subparsers.add_parser("approve-batch")
    approve.add_argument("--run-root", type=Path, required=True)
    approve.add_argument("--base-batch-id", required=True)
    approve.add_argument("--draft", type=Path)
    approve.add_argument("--draft-manifest", type=Path)
    approve.add_argument("--asset-catalog", type=Path, required=True)
    approve.add_argument("--asset-root", type=Path, required=True)
    approve.add_argument("--job-id")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--run-root", type=Path, required=True)
    sample.add_argument("--sample-id", required=True)
    sample.add_argument("--seed", type=int, default=20260804)
    sample.add_argument("--asset-catalog", type=Path, required=True)
    sample.add_argument("--asset-root", type=Path, required=True)

    review = subparsers.add_parser("review-sample")
    review.add_argument("--run-root", type=Path, required=True)
    review.add_argument("--sample-id", required=True)
    review.add_argument("--decision", choices=("pass", "reject"), required=True)
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--review-minutes", type=int, required=True)
    review.add_argument("--reason")

    freeze = subparsers.add_parser("freeze-splits")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20260804)
    freeze.add_argument("--asset-catalog", type=Path, required=True)
    freeze.add_argument("--asset-root", type=Path, required=True)
    return parser


def _load_catalog_from_args(args: argparse.Namespace, *, required: bool) -> AssetCatalog | None:
    catalog_path = getattr(args, "asset_catalog", None)
    asset_root = getattr(args, "asset_root", None)
    if catalog_path is None and asset_root is None and not required:
        return None
    if catalog_path is None or asset_root is None:
        raise ValueError("--asset-catalog and --asset-root must be supplied together")
    return load_asset_catalog(catalog_path, asset_root, verify_files=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            catalog = _load_catalog_from_args(args, required=True)
            assert catalog is not None
            result = prepare_core_run(
                run_root=args.run_root,
                run_id=args.run_id,
                data_root=args.data_root,
                seed_source_root=args.seed_source_root,
                asset_catalog=catalog,
                capability_assignments_path=args.capability_assignments,
                expected_capability_assignments_sha256=(
                    args.expected_capability_assignments_sha256
                ),
                seed=args.seed,
                supersedes_run=args.supersedes_run,
            )
        elif args.command == "status":
            result = core_run_status(
                args.run_root, asset_catalog=_load_catalog_from_args(args, required=False)
            )
        elif args.command == "work-order":
            catalog = _load_catalog_from_args(args, required=True)
            assert catalog is not None
            result = emit_or_retrieve_codex_work_order(
                run_root=args.run_root,
                asset_catalog=catalog,
                asset_root=args.asset_root,
                job_id=args.job_id,
            )
        elif args.command == "approve-batch":
            catalog = _load_catalog_from_args(args, required=True)
            assert catalog is not None
            result = auto_approve_core_batch(
                run_root=args.run_root,
                base_batch_id=args.base_batch_id,
                draft_path=args.draft,
                draft_manifest_path=args.draft_manifest,
                asset_catalog=catalog,
                job_id=args.job_id,
            )
        elif args.command == "sample":
            catalog = _load_catalog_from_args(args, required=True)
            assert catalog is not None
            result = create_audit_sample(
                run_root=args.run_root,
                sample_id=args.sample_id,
                seed=args.seed,
                asset_catalog=catalog,
            )
        elif args.command == "freeze-splits":
            catalog = _load_catalog_from_args(args, required=True)
            assert catalog is not None
            result = freeze_portfolio_core_splits(
                run_root=args.run_root,
                seed=args.seed,
                asset_catalog=catalog,
            )
        else:
            result = record_audit_review(
                run_root=args.run_root,
                sample_id=args.sample_id,
                decision=args.decision,
                reviewer_id=args.reviewer_id,
                review_minutes=args.review_minutes,
                reason=args.reason,
            )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser = build_parser()
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
