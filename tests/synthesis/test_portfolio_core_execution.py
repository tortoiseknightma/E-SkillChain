"""Focused tmp-path tests for r2 mechanical author-draft execution."""

from __future__ import annotations

import inspect
import runpy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from skillchain.schemas import ConversationTurn, Query
from skillchain.synthesis.models import BatchDraftManifest, GeneratedTrajectory
from skillchain.synthesis.portfolio_core_authoring import build_authoring_job
from skillchain.synthesis.portfolio_core_execution import (
    EXECUTION_POLICY_VERSION,
    LegacyTurnExclusionReceipt,
    PortfolioCoreExecutionError,
    R2AuthorDraftManifest,
    R2ExecutionBatchManifest,
    R2ExecutionCheckpoint,
    auto_approve_portfolio_core_r2_batch,
    build_legacy_turn_exclusion_receipt,
    build_r2_author_draft_manifest,
    canonical_turns_sha256,
    load_published_r2_author_drafts,
    prepare_portfolio_core_r2_execution,
    publish_r2_author_drafts,
)
from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


_R2_TESTS = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
_GENERATED_AT = datetime(2026, 8, 5, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def r2_inputs():
    result = _R2_TESTS["_full_r2_fixture"]()
    plan_sha256 = _R2_TESTS["_plan_sha256"](result.plan)
    return (
        result,
        build_r2_core_bridge(
            result,
            trusted_plan_sha256=plan_sha256,
            seed=20260805,
        ),
        plan_sha256,
    )


def _batch_ids(result) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.batch_id for row in result.plan.queries))


def _job_for_batch(tmp_path: Path, result, bridge, batch_id: str):
    source_root = tmp_path / "synthetic-source-assets" / batch_id
    source_root.mkdir(parents=True)
    bindings: dict[str, dict[str, object]] = {}
    for index, row in enumerate(
        (row for row in result.plan.queries if row.batch_id == batch_id), start=1
    ):
        if row.asset_id in bindings:
            continue
        payload = f"mechanical-test-only-asset-{batch_id}-{index:04d}".encode("ascii")
        source = source_root / f"asset-{index:04d}.jpg"
        source.write_bytes(payload)
        bindings[row.asset_id] = {
            "asset_sha256": sha256_bytes(payload),
            "byte_size": len(payload),
            "canonical_path": str(source.resolve()),
            "source_dataset": "mechanical-test-only",
            "source_record_id": f"record-{index:04d}",
        }
    return build_authoring_job(
        result.plan.queries,
        final_split_by_plan_id=result.final_split_by_plan_id,
        reuse_by_plan_id={
            plan_id: {
                "reuse_variant": result.reuse_variant_by_plan_id[plan_id],
                "reuse_reason": result.reuse_reason_by_plan_id[plan_id],
            }
            for plan_id in result.final_split_by_plan_id
        },
        realism_sidecar=bridge.realism_sidecar,
        base_batch_id=batch_id,
        asset_bindings=bindings,
    )


def _drafts_for_job(
    bridge, job, *, marker: str = "primary"
) -> tuple[GeneratedTrajectory, ...]:
    assignment_by_id = {
        assignment.plan_id: assignment
        for assignment in bridge.realism_sidecar.assignments
    }
    drafts: list[GeneratedTrajectory] = []
    for item in job.work_order.items:
        assignment = assignment_by_id[item.plan_id]
        final_text = f"mechanical-test-only final {marker} {item.plan_id}"
        if assignment.trajectory_shape == "single_turn":
            turns = [ConversationTurn(role="user", content=final_text)]
        else:
            turns = [
                ConversationTurn(
                    role="user",
                    content=f"mechanical-test-only opening {marker} {item.plan_id}",
                ),
                ConversationTurn(
                    role="assistant",
                    content=f"mechanical-test-only reply {marker} {item.plan_id}",
                ),
                ConversationTurn(role="user", content=final_text),
            ]
        drafts.append(GeneratedTrajectory(plan_id=item.plan_id, turns=turns))
    return tuple(drafts)


def _legacy_receipt(
    *,
    matching_draft: GeneratedTrajectory | None = None,
    artifact_marker: str = "primary",
) -> LegacyTurnExclusionReceipt:
    turns_by_plan_id = {
        f"dm-{index:03d}": [
            ConversationTurn(
                role="user",
                content=f"mechanical-test-only legacy {artifact_marker} {index:03d}",
            )
        ]
        for index in range(1, 201)
    }
    if matching_draft is not None:
        turns_by_plan_id[matching_draft.plan_id] = matching_draft.turns
    return build_legacy_turn_exclusion_receipt(
        legacy_artifact_sha256=sha256_bytes(
            f"mechanical-test-only accepted-dev-artifact {artifact_marker}".encode(
                "utf-8"
            )
        ),
        turns_by_plan_id=turns_by_plan_id,
    )


def _draft_manifest(result, bridge, plan_sha256: str, job, drafts):
    return build_r2_author_draft_manifest(
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
        authoring_job=job,
        drafts=drafts,
        generated_at=_GENERATED_AT,
    )


def _publish(
    tmp_path: Path,
    result,
    bridge,
    plan_sha256: str,
    job,
    drafts,
    *,
    artifact_root: Path | None = None,
    manifest: R2AuthorDraftManifest | None = None,
) -> tuple[Path, R2AuthorDraftManifest]:
    artifact_root = artifact_root or tmp_path / "published-author-drafts"
    manifest = manifest or _draft_manifest(result, bridge, plan_sha256, job, drafts)
    publish_r2_author_drafts(
        authoring_job=job,
        drafts=drafts,
        draft_manifest=manifest,
        draft_artifact_root=artifact_root,
    )
    return artifact_root, manifest


def _execute(
    root: Path,
    result,
    bridge,
    plan_sha256: str,
    receipt: LegacyTurnExclusionReceipt,
    job,
    artifact_root: Path,
):
    return auto_approve_portfolio_core_r2_batch(
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
        legacy_turn_exclusion_receipt=receipt,
        authoring_job=job,
        draft_artifact_root=artifact_root,
        execution_root=root,
    )


def _prepare(root: Path, result, bridge, plan_sha256: str, receipt):
    return prepare_portfolio_core_r2_execution(
        result,
        bridge,
        trusted_plan_sha256=plan_sha256,
        legacy_turn_exclusion_receipt=receipt,
        execution_root=root,
    )


def test_mechanical_execution_publishes_final_split_queries_and_receipt_bindings(
    tmp_path: Path,
    r2_inputs,
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"

    ready = _prepare(root, result, bridge, plan_sha256, receipt)
    accepted = _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)

    assert ready.status == "ready"
    assert accepted.created
    assert accepted.checkpoint.status == "running"
    assert accepted.checkpoint.accepted_query_count == 25
    assert accepted.checkpoint.legacy_turn_exclusion_receipt_sha256 == sha256_bytes(
        receipt.canonical_bytes()
    )
    result_lines = (accepted.accepted_dir / "results.jsonl").read_bytes().splitlines()
    assert len(result_lines) == 25
    for line, item in zip(result_lines, job.work_order.items, strict=True):
        query = Query.model_validate_json(line)
        assert query.query_id == item.plan_id
        assert query.split == bridge.split_constraints.plan_id_to_split[item.plan_id]
        assert query.label_status == "auto"
        assert query.turns[-1].content == query.text

    batch_manifest = R2ExecutionBatchManifest.model_validate_json(
        (accepted.accepted_dir / "manifest.json").read_bytes()
    )
    assert batch_manifest.legacy_turn_exclusion_receipt_sha256 == sha256_bytes(
        receipt.canonical_bytes()
    )
    ledger = (root / "accepted-ledger.jsonl").read_bytes()
    assert batch_manifest.author_draft_receipt_sha256 in ledger.decode("utf-8")


def test_public_builder_requires_aware_time_and_autoapprove_has_no_raw_drafts(
    tmp_path: Path, r2_inputs
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)

    with pytest.raises(PortfolioCoreExecutionError, match="generated_at"):
        build_r2_author_draft_manifest(
            result,
            bridge,
            trusted_plan_sha256=plan_sha256,
            authoring_job=job,
            drafts=drafts,
            generated_at=datetime(2026, 8, 5),
        )

    parameters = inspect.signature(auto_approve_portfolio_core_r2_batch).parameters
    assert "drafts" not in parameters
    assert "draft_manifest" not in parameters
    assert "draft_artifact_root" in parameters


def test_rejects_drifted_bridge_shape_id_and_job_hash_bindings(
    tmp_path: Path, r2_inputs
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, manifest = _publish(
        tmp_path, result, bridge, plan_sha256, job, drafts
    )
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"

    changed_splits = dict(bridge.split_constraints.plan_id_to_split)
    changed_splits[job.work_order.items[0].plan_id] = "val"
    bad_bridge = replace(
        bridge,
        split_constraints=bridge.split_constraints.model_copy(
            update={"plan_id_to_split": changed_splits}
        ),
    )
    with pytest.raises(PortfolioCoreExecutionError, match="split constraints drifted"):
        _execute(root, result, bad_bridge, plan_sha256, receipt, job, artifact_root)

    three_turn_index = next(
        index for index, draft in enumerate(drafts) if len(draft.turns) == 3
    )
    malformed_shape = list(drafts)
    malformed_shape[three_turn_index] = GeneratedTrajectory(
        plan_id=drafts[three_turn_index].plan_id,
        turns=[ConversationTurn(role="user", content="mechanical malformed")],
    )
    malformed_shape = tuple(malformed_shape)
    malformed_root, _ = _publish(
        tmp_path,
        result,
        bridge,
        plan_sha256,
        job,
        malformed_shape,
        artifact_root=tmp_path / "malformed-drafts",
    )
    with pytest.raises(PortfolioCoreExecutionError, match="turn roles"):
        _execute(
            root,
            result,
            bridge,
            plan_sha256,
            receipt,
            job,
            malformed_root,
        )

    malformed_ids = list(drafts)
    malformed_ids[0] = GeneratedTrajectory(
        plan_id=malformed_ids[1].plan_id,
        turns=malformed_ids[0].turns,
    )
    with pytest.raises(PortfolioCoreExecutionError, match="plan IDs"):
        _draft_manifest(result, bridge, plan_sha256, job, tuple(malformed_ids))

    bad_hash_manifest = manifest.model_copy(update={"work_order_sha256": "0" * 64})
    with pytest.raises(PortfolioCoreExecutionError, match="work_order_sha256"):
        publish_r2_author_drafts(
            authoring_job=job,
            drafts=drafts,
            draft_manifest=bad_hash_manifest,
            draft_artifact_root=tmp_path / "bad-manifest-drafts",
        )


def test_rejects_legacy_dev_mini_manifest_without_author_job_bindings(
    tmp_path: Path,
    r2_inputs,
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    legacy = BatchDraftManifest(
        base_batch_id=batch_id,
        data_origin="synthetic_derived",
        provider="codex",
        model_display_name="5.6 Sol Ultra",
        model_claim_source="user_confirmation",
        generated_at=_GENERATED_AT,
        plan_sha256=plan_sha256,
        asset_catalog_sha256=result.plan.asset_catalog_sha256,
        leakage_policy_version=result.plan.leakage_policy_version,
        seed_set_sha256="0" * 64,
        draft_sha256=sha256_bytes(canonical_jsonl_bytes(drafts)),
        generation_input_sha256=job.work_order.generation_input_sha256,
    )

    with pytest.raises(PortfolioCoreExecutionError, match="R2AuthorDraftManifest"):
        publish_r2_author_drafts(
            authoring_job=job,
            drafts=drafts,
            draft_manifest=legacy,  # type: ignore[arg-type]
            draft_artifact_root=tmp_path / "legacy-drafts",
        )


def test_legacy_turn_exclusion_is_exact_and_rejects_same_plan_turns(
    tmp_path: Path, r2_inputs
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    matching = next(draft for draft in drafts if draft.plan_id == "dm-001")
    receipt = _legacy_receipt(matching_draft=matching)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)

    assert receipt.plan_ids == tuple(f"dm-{index:03d}" for index in range(1, 201))
    assert receipt.turn_digests[0].canonical_turns_sha256 == canonical_turns_sha256(
        matching.turns
    )
    with pytest.raises(PortfolioCoreExecutionError, match="reuses excluded legacy"):
        _execute(
            tmp_path / "execution-root",
            result,
            bridge,
            plan_sha256,
            receipt,
            job,
            artifact_root,
        )

    incomplete = {
        f"dm-{index:03d}": [ConversationTurn(role="user", content="legacy")]
        for index in range(1, 200)
    }
    with pytest.raises(ValueError, match="exactly dm-001"):
        build_legacy_turn_exclusion_receipt(
            legacy_artifact_sha256="a" * 64,
            turns_by_plan_id=incomplete,
        )


def test_published_draft_is_create_only_idempotent_and_fails_closed_on_tamper(
    tmp_path: Path, r2_inputs
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, manifest = _publish(
        tmp_path, result, bridge, plan_sha256, job, drafts
    )
    first = load_published_r2_author_drafts(
        authoring_job=job, draft_artifact_root=artifact_root
    )
    second = publish_r2_author_drafts(
        authoring_job=job,
        drafts=drafts,
        draft_manifest=manifest,
        draft_artifact_root=artifact_root,
    )
    assert second.receipt == first.receipt
    directory = artifact_root / batch_id
    assert {path.name for path in directory.iterdir()} == {
        "drafts.jsonl",
        "manifest.json",
        "receipt.json",
    }

    extra = directory / "extra.txt"
    extra.write_text("mechanical-test-only", encoding="utf-8")
    with pytest.raises(PortfolioCoreExecutionError, match="unexpected or missing"):
        load_published_r2_author_drafts(
            authoring_job=job, draft_artifact_root=artifact_root
        )
    assert extra.read_text(encoding="utf-8") == "mechanical-test-only"


def test_rejects_out_of_order_batch_and_cross_batch_duplicate_text(
    tmp_path: Path, r2_inputs
):
    result, bridge, plan_sha256 = r2_inputs
    first_batch, second_batch = _batch_ids(result)[:2]
    first_job = _job_for_batch(tmp_path, result, bridge, first_batch)
    first_drafts = _drafts_for_job(bridge, first_job, marker="first")
    second_job = _job_for_batch(tmp_path, result, bridge, second_batch)
    second_drafts = _drafts_for_job(bridge, second_job, marker="second")
    artifact_root = tmp_path / "published-author-drafts"
    _publish(
        tmp_path,
        result,
        bridge,
        plan_sha256,
        first_job,
        first_drafts,
        artifact_root=artifact_root,
    )
    _publish(
        tmp_path,
        result,
        bridge,
        plan_sha256,
        second_job,
        second_drafts,
        artifact_root=artifact_root,
    )
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"

    with pytest.raises(PortfolioCoreExecutionError, match="complete r2 plan order"):
        _execute(
            root,
            result,
            bridge,
            plan_sha256,
            receipt,
            second_job,
            artifact_root,
        )
    _execute(
        root,
        result,
        bridge,
        plan_sha256,
        receipt,
        first_job,
        artifact_root,
    )

    duplicate_root = tmp_path / "duplicate-author-drafts"
    duplicated = list(second_drafts)
    duplicate_turns = list(duplicated[0].turns)
    duplicate_turns[-1] = ConversationTurn(
        role="user", content=first_drafts[0].turns[-1].content
    )
    duplicated[0] = GeneratedTrajectory(
        plan_id=duplicated[0].plan_id,
        turns=duplicate_turns,
    )
    _publish(
        tmp_path,
        result,
        bridge,
        plan_sha256,
        second_job,
        tuple(duplicated),
        artifact_root=duplicate_root,
    )
    with pytest.raises(PortfolioCoreExecutionError, match="duplicates normalized text"):
        _execute(
            root,
            result,
            bridge,
            plan_sha256,
            receipt,
            second_job,
            duplicate_root,
        )


def test_prepare_recovers_ledger_before_promotion_without_draft_payload(
    tmp_path: Path,
    r2_inputs,
    monkeypatch: pytest.MonkeyPatch,
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"
    import skillchain.synthesis.portfolio_core_execution as execution

    original_publish = execution.atomic_publish_new_directory

    def interrupt_after_ledger(staging, destination):
        if Path(destination).parent.name == "accepted":
            assert (root / "accepted-ledger.jsonl").read_bytes().strip()
            raise OSError("mechanical test interruption")
        return original_publish(staging, destination)

    monkeypatch.setattr(
        execution, "atomic_publish_new_directory", interrupt_after_ledger
    )
    with pytest.raises(PortfolioCoreExecutionError, match="promote"):
        _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert not (root / "accepted" / batch_id).exists()
    assert (root / "staging" / batch_id).is_dir()
    assert len((root / "accepted-ledger.jsonl").read_bytes().splitlines()) == 1

    monkeypatch.setattr(execution, "atomic_publish_new_directory", original_publish)
    repaired = _prepare(root, result, bridge, plan_sha256, receipt)
    assert repaired.accepted_query_count == 25
    assert (root / "accepted" / batch_id).is_dir()
    assert not (root / "staging" / batch_id).exists()


def test_prepare_recovers_promote_before_checkpoint(
    tmp_path: Path,
    r2_inputs,
    monkeypatch: pytest.MonkeyPatch,
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"
    import skillchain.synthesis.portfolio_core_execution as execution

    original_replace = execution.atomic_replace_file

    def interrupt_checkpoint(path, content):
        if Path(path).name == "runtime-checkpoint.json":
            raise OSError("mechanical test interruption")
        return original_replace(path, content)

    monkeypatch.setattr(execution, "atomic_replace_file", interrupt_checkpoint)
    with pytest.raises(PortfolioCoreExecutionError, match="runtime checkpoint"):
        _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert (root / "accepted" / batch_id).is_dir()
    assert len((root / "accepted-ledger.jsonl").read_bytes().splitlines()) == 1

    monkeypatch.setattr(execution, "atomic_replace_file", original_replace)
    repaired = _prepare(root, result, bridge, plan_sha256, receipt)
    assert repaired.accepted_query_count == 25


def test_uncommitted_staging_waits_for_same_published_draft_retry(
    tmp_path: Path,
    r2_inputs,
    monkeypatch: pytest.MonkeyPatch,
):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"
    import skillchain.synthesis.portfolio_core_execution as execution

    original_replace = execution.atomic_replace_file

    def interrupt_ledger(path, content):
        if Path(path).name == "accepted-ledger.jsonl":
            raise OSError("mechanical test interruption")
        return original_replace(path, content)

    monkeypatch.setattr(execution, "atomic_replace_file", interrupt_ledger)
    with pytest.raises(PortfolioCoreExecutionError, match="accepted ledger"):
        _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert (root / "staging" / batch_id).is_dir()
    assert not (root / "accepted-ledger.jsonl").read_bytes()
    with pytest.raises(PortfolioCoreExecutionError, match="same published draft retry"):
        _prepare(root, result, bridge, plan_sha256, receipt)

    monkeypatch.setattr(execution, "atomic_replace_file", original_replace)
    recovered = _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert recovered.created
    assert recovered.checkpoint.accepted_query_count == 25


def test_tampered_accepted_batch_or_receipt_fails_closed(tmp_path: Path, r2_inputs):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"
    accepted = _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    results_path = accepted.accepted_dir / "results.jsonl"
    results_path.write_bytes(b"{}\n")

    with pytest.raises(PortfolioCoreExecutionError, match="results SHA-256"):
        _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert results_path.read_bytes() == b"{}\n"

    changed_receipt = _legacy_receipt(artifact_marker="different")
    with pytest.raises(PortfolioCoreExecutionError, match="legacy turn exclusion"):
        _prepare(root, result, bridge, plan_sha256, changed_receipt)


def test_extra_runtime_entry_fails_closed_without_cleanup(tmp_path: Path, r2_inputs):
    result, bridge, plan_sha256 = r2_inputs
    batch_id = _batch_ids(result)[0]
    job = _job_for_batch(tmp_path, result, bridge, batch_id)
    drafts = _drafts_for_job(bridge, job)
    artifact_root, _ = _publish(tmp_path, result, bridge, plan_sha256, job, drafts)
    receipt = _legacy_receipt()
    root = tmp_path / "execution-root"
    _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    extra = root / "unexpected.txt"
    extra.write_text("mechanical-test-only", encoding="utf-8")

    with pytest.raises(PortfolioCoreExecutionError, match="unexpected or missing"):
        _execute(root, result, bridge, plan_sha256, receipt, job, artifact_root)
    assert extra.read_text(encoding="utf-8") == "mechanical-test-only"


def test_runtime_checkpoint_statuses_have_exact_count_boundaries():
    common = {
        "plan_sha256": "a" * 64,
        "asset_catalog_sha256": "b" * 64,
        "capability_assignments_sha256": "c" * 64,
        "realism_manifest_sha256": "d" * 64,
        "realism_assignments_sha256": "e" * 64,
        "final_split_sidecar_sha256": "f" * 64,
        "legacy_turn_exclusion_receipt_sha256": "1" * 64,
        "accepted_ledger_sha256": "0" * 64,
    }

    def checkpoint_for(count: int, status: str) -> R2ExecutionCheckpoint:
        payload = {
            **common,
            "schema_version": 1,
            "execution_policy_version": EXECUTION_POLICY_VERSION,
            "target_query_count": 1500,
            "batch_size": 25,
            "accepted_batch_ids": tuple(
                f"batch-{index:03d}" for index in range(count // 25)
            ),
            "accepted_query_count": count,
            "status": status,
        }
        return R2ExecutionCheckpoint.model_validate(
            {
                **payload,
                "checkpoint_sha256": sha256_bytes(canonical_json_bytes(payload)),
            }
        )

    assert checkpoint_for(0, "ready").status == "ready"
    assert checkpoint_for(25, "running").status == "running"
    assert (
        checkpoint_for(1500, "auto_approved_usable_pending_sample_review").status
        == "auto_approved_usable_pending_sample_review"
    )
    with pytest.raises(ValueError, match="runtime status"):
        checkpoint_for(1500, "running")
