"""Focused tmp-path tests for the create-only r2 author-job publisher."""

from __future__ import annotations

import os
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis.portfolio_core_author_jobs import (
    AuthorJobPublishError,
    AuthorJobPublishManifest,
    publish_authoring_job,
)
from skillchain.synthesis.portfolio_core_authoring import (
    AuthoringJob,
    build_authoring_job,
)
from skillchain.synthesis.store import sha256_bytes


_AUTHORING_TESTS = runpy.run_path(
    str(Path(__file__).with_name("test_portfolio_core_authoring.py"))
)


def _authoring_job(tmp_path: Path) -> tuple[AuthoringJob, dict[str, Path]]:
    sidecar, rows, final_splits, reuse = _AUTHORING_TESTS["_sidecar"]()
    source_root = tmp_path / "canonical-source-assets"
    source_root.mkdir()
    selected_rows = [row for row in rows if row["batch_id"] == "r2-batch-001"]
    bindings: dict[str, dict[str, object]] = {}
    sources: dict[str, Path] = {}
    for index, row in enumerate(selected_rows, start=1):
        payload = f"synthetic-jpeg-bytes-{index:04d}".encode("ascii")
        source = source_root / f"private-original-{index:04d}.jpg"
        source.write_bytes(payload)
        sources[row["asset_id"]] = source
        bindings[row["asset_id"]] = {
            "asset_sha256": sha256_bytes(payload),
            "byte_size": len(payload),
            "canonical_path": str(source.resolve()),
            "source_dataset": "synthetic-private-catalog",
            "source_record_id": f"record-{index:04d}",
        }
    return (
        build_authoring_job(
            rows,
            final_split_by_plan_id=final_splits,
            reuse_by_plan_id=reuse,
            realism_sidecar=sidecar,
            base_batch_id="r2-batch-001",
            asset_bindings=bindings,
        ),
        sources,
    )


def _jobs_root(tmp_path: Path) -> Path:
    root = tmp_path / "published-author-jobs"
    root.mkdir()
    return root


def test_publishes_opaque_assets_and_is_byte_exact_idempotent(tmp_path: Path):
    job, sources = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)

    published = publish_authoring_job(job, jobs_root)
    repeated = publish_authoring_job(job, jobs_root)

    assert repeated == published
    assert published.job_dir == jobs_root / job.work_order.job_id
    assert {path.name for path in published.job_dir.iterdir()} == {
        "author-assets",
        "author-packet.json",
        "checkpoint.json",
        "manifest.json",
        "work-order.json",
    }
    manifest = AuthorJobPublishManifest.model_validate_json(
        (published.job_dir / "manifest.json").read_bytes()
    )
    assert manifest == published.manifest
    assert manifest.asset_count == 25
    assert {path.name for path in (published.job_dir / "author-assets").iterdir()} == {
        binding.opaque_alias for binding in job.work_order.aliases
    }
    assert all(
        asset.relative_path == f"author-assets/{asset.opaque_alias}"
        for asset in manifest.assets
    )
    packet_bytes = (published.job_dir / "author-packet.json").read_bytes()
    assert str(next(iter(sources.values())).parent).encode() not in packet_bytes
    assert b"canonical_path" not in packet_bytes
    assert b"synthetic-private-catalog" not in packet_bytes


@pytest.mark.parametrize(
    "drift_payload", (b"x" * 25, b"size-drifted-source-bytes-extended")
)
def test_rejects_source_hash_or_size_drift_without_final_or_staging_residue(
    tmp_path: Path,
    drift_payload: bytes,
):
    job, sources = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)
    source = sources[job.work_order.aliases[0].asset_id]
    source.write_bytes(drift_payload)

    with pytest.raises(AuthorJobPublishError, match="size or SHA-256 drifted"):
        publish_authoring_job(job, jobs_root)

    assert not os.path.lexists(jobs_root / job.work_order.job_id)
    assert not list(jobs_root.glob(f".{job.work_order.job_id}.staging-*"))


def test_rejects_model_visible_packet_leak_without_writing(tmp_path: Path):
    job, _ = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)
    leaked_job = replace(
        job,
        author_packet=job.author_packet.model_copy(
            update={"job_id": r"C:\private-source\leaked-job"}
        ),
    )

    with pytest.raises(AuthorJobPublishError, match="leak scanner"):
        publish_authoring_job(leaked_job, jobs_root)

    assert not os.path.lexists(jobs_root / job.work_order.job_id)
    assert not list(jobs_root.iterdir())


def test_existing_target_with_extra_file_fails_closed(tmp_path: Path):
    job, _ = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)
    published = publish_authoring_job(job, jobs_root)
    extra = published.job_dir / "unexpected.txt"
    extra.write_text("tamper", encoding="utf-8")

    with pytest.raises(AuthorJobPublishError, match="unexpected or missing"):
        publish_authoring_job(job, jobs_root)

    assert extra.read_text(encoding="utf-8") == "tamper"


def test_existing_target_metadata_tamper_fails_closed(tmp_path: Path):
    job, _ = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)
    published = publish_authoring_job(job, jobs_root)
    packet_path = published.job_dir / "author-packet.json"
    packet_path.write_bytes(b"{}\n")

    with pytest.raises(AuthorJobPublishError, match="bytes drifted"):
        publish_authoring_job(job, jobs_root)

    assert packet_path.read_bytes() == b"{}\n"


def test_existing_target_symlink_asset_fails_closed(tmp_path: Path):
    job, sources = _authoring_job(tmp_path)
    jobs_root = _jobs_root(tmp_path)
    published = publish_authoring_job(job, jobs_root)
    binding = job.work_order.aliases[0]
    destination = published.job_dir / binding.materialization_relative_path
    destination.unlink()
    source = sources[binding.asset_id]
    try:
        os.symlink(source, destination)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this test host: {exc}")

    with pytest.raises(AuthorJobPublishError, match="symlink or reparse"):
        publish_authoring_job(job, jobs_root)

    assert destination.is_symlink()
    assert source.is_file()
