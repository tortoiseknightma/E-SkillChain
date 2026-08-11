"""Create-only CLI entry points for Portfolio core r2 authoring execution.

The commands deliberately rebuild the r2 in-memory plan from caller-supplied
audited inputs.  They do not accept raw drafts, invoke a model, or embed
environment-specific paths.  ``prepare-job`` publishes one opaque 25-item
author job; ``approve-published-draft`` delegates revalidation of the existing
create-only author-draft artifact to the mechanical executor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from skillchain.data.asset_catalog import AssetCatalog, load_asset_catalog
from skillchain.synthesis.planning import (
    R2CoreInMemoryPlan,
    build_r2_core_in_memory_plan,
    load_capability_assignments,
    load_plan,
    rebind_r2_dev_prefix,
)
from skillchain.synthesis.portfolio_core_author_jobs import publish_authoring_job
from skillchain.synthesis.portfolio_core_authoring import (
    AuthoringJob,
    build_authoring_job,
    canonical_author_packet_bytes,
)
from skillchain.synthesis.portfolio_core_execution import (
    LegacyTurnExclusionReceipt,
    auto_approve_portfolio_core_r2_batch,
)
from skillchain.synthesis.portfolio_core_r2 import R2CoreBridge, build_r2_core_bridge
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


_R2_LAYOUT_SEED = 20260804
_R2_REALISM_SEED = 20260805
_BATCH_SIZE = 25


@dataclass(frozen=True)
class RebuiltR2Context:
    """The deterministic r2 values shared by both CLI commands."""

    result: R2CoreInMemoryPlan
    bridge: R2CoreBridge
    trusted_plan_sha256: str
    target_catalog: AssetCatalog


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _rebuild_context(arguments: argparse.Namespace) -> RebuiltR2Context:
    """Recreate the audited r2 plan and bridge from locked caller inputs."""

    expected_parent_plan_sha256 = _require_sha256(
        arguments.expected_parent_plan_sha256,
        "expected_parent_plan_sha256",
    )
    expected_r2_plan_sha256 = _require_sha256(
        arguments.expected_r2_plan_sha256,
        "expected_r2_plan_sha256",
    )
    parent_plan, parent_manifest = load_plan(arguments.r1_plan)
    if parent_manifest.plan_sha256 != expected_parent_plan_sha256:
        raise ValueError("r1 plan manifest does not match expected_parent_plan_sha256")

    parent_catalog = load_asset_catalog(
        arguments.parent_catalog,
        arguments.parent_asset_root,
        verify_files=arguments.verify_files,
    )
    target_catalog = load_asset_catalog(
        arguments.target_catalog,
        arguments.target_asset_root,
        verify_files=arguments.verify_files,
    )
    parent_assignments = load_capability_assignments(
        arguments.parent_assignments,
        expected_sha256=arguments.expected_parent_assignments_sha256,
    )
    target_assignments = load_capability_assignments(
        arguments.target_assignments,
        expected_sha256=arguments.expected_target_assignments_sha256,
    )
    rebind = rebind_r2_dev_prefix(
        parent_plan,
        parent_core_plan_sha256=expected_parent_plan_sha256,
        parent_catalog=parent_catalog,
        expected_parent_catalog_sha256=(arguments.expected_parent_catalog_sha256),
        parent_capability_assignments=parent_assignments,
        expected_parent_capability_assignments_sha256=(
            arguments.expected_parent_assignments_sha256
        ),
        target_catalog=target_catalog,
        expected_target_catalog_sha256=arguments.expected_target_catalog_sha256,
        target_capability_assignments=target_assignments,
        expected_target_capability_assignments_sha256=(
            arguments.expected_target_assignments_sha256
        ),
    )
    result = build_r2_core_in_memory_plan(
        rebind,
        target_catalog=target_catalog,
        target_capability_assignments=target_assignments,
        expected_target_catalog_sha256=arguments.expected_target_catalog_sha256,
        expected_target_capability_assignments_sha256=(
            arguments.expected_target_assignments_sha256
        ),
        seed=_R2_LAYOUT_SEED,
    )
    trusted_plan_sha256 = sha256_bytes(canonical_json_bytes(result.plan))
    if trusted_plan_sha256 != expected_r2_plan_sha256:
        raise ValueError("reconstructed r2 plan does not match expected_r2_plan_sha256")
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=trusted_plan_sha256,
        seed=_R2_REALISM_SEED,
    )
    return RebuiltR2Context(
        result=result,
        bridge=bridge,
        trusted_plan_sha256=trusted_plan_sha256,
        target_catalog=target_catalog,
    )


def _batch_rows(
    result: R2CoreInMemoryPlan,
    batch_id: str,
) -> tuple[object, ...]:
    rows = tuple(row for row in result.plan.queries if row.batch_id == batch_id)
    positions = [row.position for row in rows]
    if len(rows) != _BATCH_SIZE or positions != list(range(1, _BATCH_SIZE + 1)):
        raise ValueError("batch_id must identify one complete 25-item r2 batch")
    return rows


def _asset_bindings(
    catalog: AssetCatalog,
    rows: Sequence[object],
) -> dict[str, dict[str, object]]:
    """Build internal-only source bindings for exactly one complete job."""

    bindings: dict[str, dict[str, object]] = {}
    for row in rows:
        asset_id = row.asset_id
        if asset_id in bindings:
            continue
        resolution = catalog.verify_reference(
            asset_id=asset_id,
            image_path=row.image_path,
            leakage_group_id=row.leakage_group_id,
        )
        source_path = (catalog.asset_root / resolution.asset.local_path).resolve()
        bindings[asset_id] = {
            "asset_sha256": resolution.asset.sha256,
            "byte_size": source_path.stat().st_size,
            "canonical_path": str(source_path),
            "source_dataset": resolution.asset.source_dataset,
            "source_record_id": resolution.asset.source_record_id,
        }
    return bindings


def _build_authoring_job_for_batch(
    context: RebuiltR2Context,
    batch_id: str,
) -> AuthoringJob:
    rows = _batch_rows(context.result, batch_id)
    reuse_by_plan_id = {
        plan_id: {
            "plan_id": plan_id,
            "reuse_variant": context.result.reuse_variant_by_plan_id[plan_id],
            "reuse_reason": context.result.reuse_reason_by_plan_id[plan_id],
        }
        for plan_id in context.result.final_split_by_plan_id
    }
    return build_authoring_job(
        context.result.plan.queries,
        final_split_by_plan_id=context.result.final_split_by_plan_id,
        reuse_by_plan_id=reuse_by_plan_id,
        realism_sidecar=context.bridge.realism_sidecar,
        base_batch_id=batch_id,
        asset_bindings=_asset_bindings(context.target_catalog, rows),
    )


def _job_metadata(
    job: AuthoringJob, *, job_dir: Path | None = None
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "job_id": job.work_order.job_id,
        "packet_sha256": sha256_bytes(canonical_author_packet_bytes(job.author_packet)),
        "alias_count": len(job.work_order.aliases),
    }
    if job_dir is not None:
        metadata["job_dir"] = str(job_dir)
    return metadata


def prepare_job(arguments: argparse.Namespace) -> dict[str, object]:
    """Build and create-only publish one complete deterministic author job."""

    context = _rebuild_context(arguments)
    job = _build_authoring_job_for_batch(context, arguments.batch_id)
    published = publish_authoring_job(job, arguments.jobs_root)
    return _job_metadata(job, job_dir=published.job_dir)


def _load_body_free_legacy_receipt(path: str | Path) -> LegacyTurnExclusionReceipt:
    receipt_path = Path(path)
    content = receipt_path.read_bytes()
    receipt = LegacyTurnExclusionReceipt.model_validate_json(content)
    if content != receipt.canonical_bytes():
        raise ValueError("legacy turn exclusion receipt is not canonical JSON")
    return receipt


def approve_published_draft(arguments: argparse.Namespace) -> dict[str, object]:
    """Mechanically approve one reverified, already-published author draft."""

    context = _rebuild_context(arguments)
    job = _build_authoring_job_for_batch(context, arguments.batch_id)
    receipt = _load_body_free_legacy_receipt(arguments.legacy_turn_exclusion_receipt)
    auto_approve_portfolio_core_r2_batch(
        context.result,
        context.bridge,
        trusted_plan_sha256=context.trusted_plan_sha256,
        legacy_turn_exclusion_receipt=receipt,
        authoring_job=job,
        draft_artifact_root=arguments.draft_artifact_root,
        execution_root=arguments.execution_root,
    )
    return _job_metadata(job)


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r1-plan", type=Path, required=True)
    parser.add_argument("--parent-catalog", type=Path, required=True)
    parser.add_argument("--parent-asset-root", type=Path, required=True)
    parser.add_argument("--expected-parent-catalog-sha256", required=True)
    parser.add_argument("--target-catalog", type=Path, required=True)
    parser.add_argument("--target-asset-root", type=Path, required=True)
    parser.add_argument("--expected-target-catalog-sha256", required=True)
    parser.add_argument(
        "--parent-assignments",
        "--parent-capability-assignments",
        dest="parent_assignments",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-parent-assignments-sha256",
        "--expected-parent-capability-assignments-sha256",
        dest="expected_parent_assignments_sha256",
        required=True,
    )
    parser.add_argument(
        "--target-assignments",
        "--target-capability-assignments",
        dest="target_assignments",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-target-assignments-sha256",
        "--expected-target-capability-assignments-sha256",
        dest="expected_target_assignments_sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-parent-plan-sha256",
        "--expected-r1-plan-sha256",
        dest="expected_parent_plan_sha256",
        required=True,
        help="locked SHA-256 for the serialized r1 parent core plan",
    )
    parser.add_argument(
        "--expected-r2-plan-sha256",
        required=True,
        help="locked SHA-256 required from the reconstructed canonical r2 plan",
    )
    parser.add_argument(
        "--verify-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify catalog asset files before rebuilding (default: enabled)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create-only Portfolio core r2 author-job execution commands."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-job",
        help="rebuild r2 and publish one complete opaque 25-item author job",
    )
    _add_context_arguments(prepare)
    prepare.add_argument("--batch-id", "--job-batch-id", dest="batch_id", required=True)
    prepare.add_argument("--jobs-root", type=Path, required=True)

    approve = commands.add_parser(
        "approve-published-draft",
        help="approve one already-published, reverified author draft",
    )
    _add_context_arguments(approve)
    approve.add_argument("--batch-id", "--job-batch-id", dest="batch_id", required=True)
    approve.add_argument("--draft-artifact-root", type=Path, required=True)
    approve.add_argument("--execution-root", type=Path, required=True)
    approve.add_argument("--legacy-turn-exclusion-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare-job":
        metadata = prepare_job(arguments)
    elif arguments.command == "approve-published-draft":
        metadata = approve_published_draft(arguments)
    else:  # pragma: no cover - argparse fixes the command set.
        raise ValueError(f"unknown command: {arguments.command}")
    print(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
