from __future__ import annotations

from pathlib import Path

import pytest

from skillchain.data.source_lock import (
    AcquisitionIdentity,
    RequiredSourceLock,
    SourceLockError,
    build_artifact_scope,
    load_and_verify_required_source_lock,
    load_required_source_lock,
    stable_file_digest,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _fixture_lock(tmp_path: Path) -> tuple[Path, Path, str]:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "archive.bin").write_bytes(b"archive")
    tree = raw / "images"
    tree.mkdir()
    (tree / "a.jpg").write_bytes(b"image-a")
    (tree / "b.jpg").write_bytes(b"image-b")

    explicit = build_artifact_scope(
        raw,
        scope_id="archive",
        mode="explicit_files",
        paths=("archive.bin",),
    )
    recursive = build_artifact_scope(
        raw,
        scope_id="images",
        mode="recursive_tree",
        root="images",
    )
    archive_sha256, archive_bytes = stable_file_digest(
        raw / "archive.bin", label="archive"
    )
    lock = RequiredSourceLock(
        source_id="fixture",
        source_revision="fixture-revision",
        lock_plan_sha256="1" * 64,
        artifact_scopes=(explicit, recursive),
        acquisition_identities=(
            AcquisitionIdentity(
                logical_path="archive.bin",
                url="https://example.test/archive.bin",
                bytes=archive_bytes,
                local_sha256=archive_sha256,
            ),
        ),
    )
    content = canonical_json_bytes(lock.model_dump(mode="json"))
    path = tmp_path / "fixture.source-lock.json"
    path.write_bytes(content)
    return raw, path, sha256_bytes(content)


def test_source_lock_recomputes_explicit_and_recursive_content(tmp_path: Path) -> None:
    raw, path, digest = _fixture_lock(tmp_path)
    verified = load_and_verify_required_source_lock(
        path,
        raw,
        expected_lock_file_sha256=digest,
    )
    assert verified.lock.source_id == "fixture"
    assert sum(scope.file_count for scope in verified.lock.artifact_scopes) == 3


def test_source_lock_can_preflight_canonical_bytes_without_raw_walk(
    tmp_path: Path,
) -> None:
    raw, path, digest = _fixture_lock(tmp_path)
    (raw / "images/a.jpg").write_bytes(b"changed-after-lock")

    lock = load_required_source_lock(
        path,
        expected_lock_file_sha256=digest,
    )

    assert lock.source_id == "fixture"
    with pytest.raises(SourceLockError, match="RAW scope differs from lock"):
        load_and_verify_required_source_lock(
            path,
            raw,
            expected_lock_file_sha256=digest,
        )


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_source_lock_rejects_tree_drift(tmp_path: Path, change: str) -> None:
    raw, path, digest = _fixture_lock(tmp_path)
    if change == "modify":
        (raw / "images/a.jpg").write_bytes(b"changed")
    elif change == "add":
        (raw / "images/c.jpg").write_bytes(b"added")
    else:
        (raw / "images/a.jpg").unlink()

    with pytest.raises(SourceLockError, match="RAW scope differs from lock"):
        load_and_verify_required_source_lock(
            path,
            raw,
            expected_lock_file_sha256=digest,
        )


def test_source_lock_requires_external_lock_digest(tmp_path: Path) -> None:
    raw, path, _ = _fixture_lock(tmp_path)
    with pytest.raises(SourceLockError, match="external digest mismatch"):
        load_and_verify_required_source_lock(
            path,
            raw,
            expected_lock_file_sha256="f" * 64,
        )
