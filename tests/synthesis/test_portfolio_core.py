"""Regression tests for the isolated Portfolio core workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import skillchain.synthesis.portfolio_core as portfolio_core
from skillchain.synthesis.batches import AcceptedLedgerEntry
from skillchain.synthesis.planning import (
    build_dev_mini_plan,
    extend_core_plan,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


class _Catalog:
    catalog_sha256 = "c" * 64
    leakage_policy_version = "dataset-asset-components-v1"

    def require_verified_files(self) -> None:
        return None


def _core_plans(image_root: Path):
    dev = build_dev_mini_plan(
        image_root, seed=20260711, allow_provisional_asset_groups=True
    )
    pools = {
        intent: [
            Path(f"query_images/{intent}/core-{index:04d}.jpg")
            for index in range(1, 401)
        ]
        for intent in (
            "exact_match",
            "multi_product",
            "divergent_rec",
            "encyclopedia",
            "utility",
        )
    }
    core = extend_core_plan(
        dev,
        pools,
        seed=20260711,
        allow_provisional_asset_groups=True,
    )
    return dev, core


def test_prepare_core_run_publishes_once_and_is_idempotent(
    monkeypatch, corpus_fixture, fake_image_root, tmp_path
) -> None:
    dev, core = _core_plans(fake_image_root)
    monkeypatch.setattr(
        portfolio_core,
        "load_capability_assignments",
        lambda path, *, expected_sha256: (),
    )
    monkeypatch.setattr(
        portfolio_core,
        "build_core_plan_from_resources",
        lambda *args, **kwargs: (dev, core),
    )
    run_root = tmp_path / "portfolio-core-run"
    catalog = _Catalog()

    first = portfolio_core.prepare_core_run(
        run_root=run_root,
        run_id="core-run-a",
        data_root=fake_image_root,
        seed_source_root=corpus_fixture.root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        capability_assignments_path=tmp_path / "assignments.jsonl",
        expected_capability_assignments_sha256="a" * 64,
    )
    before = (run_root / "portfolio-core-run.json").read_bytes()
    second = portfolio_core.prepare_core_run(
        run_root=run_root,
        run_id="core-run-a",
        data_root=fake_image_root,
        seed_source_root=corpus_fixture.root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        capability_assignments_path=tmp_path / "assignments.jsonl",
        expected_capability_assignments_sha256="a" * 64,
    )

    assert first["status"] == "ready"
    assert second == first
    assert (run_root / "portfolio-core-run.json").read_bytes() == before
    assert (run_root / "plans" / "core.json").is_file()
    assert (run_root / "seeds" / "accepted" / "manifest.json").is_file()


def test_sync_auto_ledger_accepts_the_fresh_dev_prefix(
    monkeypatch, corpus_fixture, fake_image_root, tmp_path
) -> None:
    dev, core = _core_plans(fake_image_root)
    monkeypatch.setattr(
        portfolio_core,
        "load_capability_assignments",
        lambda path, *, expected_sha256: (),
    )
    monkeypatch.setattr(
        portfolio_core,
        "build_core_plan_from_resources",
        lambda *args, **kwargs: (dev, core),
    )
    run_root = tmp_path / "portfolio-core-run"
    catalog = _Catalog()
    portfolio_core.prepare_core_run(
        run_root=run_root,
        run_id="core-run-a",
        data_root=fake_image_root,
        seed_source_root=corpus_fixture.root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        capability_assignments_path=tmp_path / "assignments.jsonl",
        expected_capability_assignments_sha256="a" * 64,
    )
    run = portfolio_core._load_run(run_root)
    entry = AcceptedLedgerEntry(
        batch_id="dev-mini-001-r1",
        base_batch_id="dev-mini-001",
        revision=1,
        count=25,
        results_sha256="b" * 64,
        plan_sha256=run.core_plan_sha256,
        seed_set_sha256=run.seed_set_sha256,
        accepted_at=datetime.now(timezone.utc),
    )
    (run_root / "accepted-ledger.jsonl").write_bytes(canonical_jsonl_bytes([entry]))

    synchronized = portfolio_core._sync_auto_ledger(run_root)

    assert [item.base_batch_id for item in synchronized] == ["dev-mini-001"]
    assert (run_root / "auto-approved-ledger.jsonl").is_file()


def test_record_audit_review_is_idempotent_after_state_write_interruption(
    monkeypatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "portfolio-core-run"
    run_root.mkdir()
    now = datetime.now(timezone.utc)
    run = portfolio_core.PortfolioCoreRun(
        run_id="core-run-a",
        status="auto_approved_usable_pending_sample_review",
        core_plan_sha256="a" * 64,
        dev_prefix_plan_sha256="b" * 64,
        asset_catalog_sha256="c" * 64,
        leakage_policy_version="dataset-asset-components-v1",
        seed_set_sha256="d" * 64,
        audit_sample_id="audit-200",
        created_at=now,
        updated_at=now,
    )
    portfolio_core._write_run(run_root, run, create=True)
    sample_root = run_root / "audit-samples" / "audit-200"
    sample_root.mkdir(parents=True)
    source_bytes = b'{"mechanical":"accepted"}\n'
    sample_bytes = b'{"mechanical":"sample"}\n'
    manifest = portfolio_core.AuditSampleManifest(
        sample_id="audit-200",
        run_id=run.run_id,
        seed=20260804,
        source_queries_sha256=sha256_bytes(source_bytes),
        sample_queries_sha256=sha256_bytes(sample_bytes),
        strata_counts={"opt_pool|utility.recipe_guidance|boundary=false": 200},
        created_at=now,
    )
    (run_root / "queries.jsonl").write_bytes(source_bytes)
    (sample_root / "queries.jsonl").write_bytes(sample_bytes)
    (sample_root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    monkeypatch.setattr(
        portfolio_core,
        "verify_accepted_corpus",
        lambda root: [object()] * 1500,
    )

    first = portfolio_core.record_audit_review(
        run_root=run_root,
        sample_id="audit-200",
        decision="pass",
        reviewer_id="owner",
        review_minutes=10,
    )
    review_bytes = (sample_root / "review.json").read_bytes()
    second = portfolio_core.record_audit_review(
        run_root=run_root,
        sample_id="audit-200",
        decision="pass",
        reviewer_id="owner",
        review_minutes=10,
    )

    assert first["status"] == "approved"
    assert second["status"] == "approved"
    assert (sample_root / "review.json").read_bytes() == review_bytes


def test_audit_quota_remainder_is_distributed_across_strata() -> None:
    strata = {
        "a": [object()] * 100,
        "b": [object()] * 101,
        "c": [object()] * 102,
        "d": [object()] * 103,
    }

    quotas = portfolio_core._allocate_sample_quotas(strata, sample_size=200)

    assert sum(quotas.values()) == 200
    assert all(quotas[key] <= len(values) for key, values in strata.items())
    expected_floors = {
        key: len(values) * 200 // 406 for key, values in strata.items()
    }
    assert all(quotas[key] - expected_floors[key] in {0, 1} for key in strata)


def test_completed_status_fails_closed_when_accepted_output_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "portfolio-core-run"
    run_root.mkdir()
    now = datetime.now(timezone.utc)
    run = portfolio_core.PortfolioCoreRun(
        run_id="core-run-a",
        status="ready",
        core_plan_sha256="a" * 64,
        dev_prefix_plan_sha256="b" * 64,
        asset_catalog_sha256="c" * 64,
        leakage_policy_version="dataset-asset-components-v1",
        seed_set_sha256="d" * 64,
        created_at=now,
        updated_at=now,
    )
    portfolio_core._write_run(run_root, run, create=True)
    entries = [
        portfolio_core.AutoApprovalEntry(
            batch_id=f"core-{index:03d}-r1",
            base_batch_id=f"core-{index:03d}",
            revision=1,
            results_sha256="e" * 64,
            mechanically_approved_at=now,
        )
        for index in range(1, 61)
    ]
    monkeypatch.setattr(
        portfolio_core,
        "read_active_plan",
        lambda root: (
            root / "plans" / "core.json",
            SimpleNamespace(scope="core"),
            SimpleNamespace(plan_sha256=run.core_plan_sha256),
            SimpleNamespace(plan_sha256=run.core_plan_sha256),
        ),
    )
    monkeypatch.setattr(portfolio_core, "_validate_seed_binding", lambda *_args: None)
    monkeypatch.setattr(portfolio_core, "_sync_auto_ledger", lambda _root: entries)
    monkeypatch.setattr(
        portfolio_core,
        "corpus_status",
        lambda _root: {
            "accepted_queries": 1500,
            "accepted_batches": 60,
            "next_batch_id": None,
            "next_revision": None,
        },
    )
    with pytest.raises(ValueError, match="cannot report completed portfolio core run"):
        portfolio_core.core_run_status(run_root)

    assert portfolio_core._load_run(run_root).status == "ready"


def test_codex_work_order_is_create_only_resumable_and_binds_approval(
    monkeypatch, corpus_fixture, fake_image_root, tmp_path: Path
) -> None:
    dev, core = _core_plans(fake_image_root)
    monkeypatch.setattr(
        portfolio_core,
        "load_capability_assignments",
        lambda path, *, expected_sha256: (),
    )
    monkeypatch.setattr(
        portfolio_core,
        "build_core_plan_from_resources",
        lambda *args, **kwargs: (dev, core),
    )
    # Plans keep paths relative to the clean root as query_images/<intent>/... .
    # Mirror only the first batch's existing fixture files because this test
    # emits a work order but deliberately does not synthesize a query draft.
    for source in fake_image_root.glob("*/*.jpg"):
        destination = fake_image_root / "query_images" / source.relative_to(fake_image_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    run_root = tmp_path / "portfolio-core-run"
    catalog = _Catalog()
    portfolio_core.prepare_core_run(
        run_root=run_root,
        run_id="core-run-a",
        data_root=fake_image_root,
        seed_source_root=corpus_fixture.root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        capability_assignments_path=tmp_path / "assignments.jsonl",
        expected_capability_assignments_sha256="a" * 64,
    )

    first = portfolio_core.emit_or_retrieve_codex_work_order(
        run_root=run_root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        asset_root=fake_image_root,
    )
    second = portfolio_core.emit_or_retrieve_codex_work_order(
        run_root=run_root,
        asset_catalog=catalog,  # type: ignore[arg-type]
        asset_root=fake_image_root,
    )
    order = portfolio_core.CodexGenerationWorkOrder.model_validate_json(
        Path(first["work_order_path"]).read_bytes()
    )
    checkpoint = portfolio_core.CodexGenerationCheckpoint.model_validate_json(
        Path(first["checkpoint_path"]).read_bytes()
    )

    assert first["resumed"] is False
    assert second["resumed"] is True
    assert first["job_id"] == second["job_id"]
    assert order.base_batch_id == "dev-mini-001"
    assert len(order.plan_items) == len(order.image_references) == 25
    assert checkpoint.state == "issued"
    assert Path(first["draft_path"]).parent.name == "inbox"

    with pytest.raises(ValueError, match="pass its --job-id"):
        portfolio_core.auto_approve_core_batch(
            run_root=run_root,
            base_batch_id="dev-mini-001",
            draft_path=tmp_path / "outside.jsonl",
            draft_manifest_path=tmp_path / "outside.manifest.json",
            asset_catalog=catalog,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="immutable inbox paths"):
        portfolio_core.auto_approve_core_batch(
            run_root=run_root,
            base_batch_id="dev-mini-001",
            draft_path=tmp_path / "outside.jsonl",
            draft_manifest_path=tmp_path / "outside.manifest.json",
            asset_catalog=catalog,  # type: ignore[arg-type]
            job_id=first["job_id"],
        )
