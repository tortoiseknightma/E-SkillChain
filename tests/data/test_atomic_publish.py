import os

import pytest

from skillchain.data import (
    publish_staged_directory,
    publish_staged_directory_and_file,
    recover_directory_and_file_publish,
)


def _backup_paths(destination_dir, destination_file):
    return (
        destination_dir.with_name(f".{destination_dir.name}.backup"),
        destination_file.with_name(f".{destination_file.name}.backup"),
    )


def test_recovery_restores_directory_when_crash_happens_between_backups(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    directory_backup, _ = _backup_paths(destination_dir, destination_file)
    directory_backup.mkdir()
    (directory_backup / "old.jpg").write_bytes(b"old-image")
    destination_file.write_bytes(b"old-parquet")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert (destination_dir / "old.jpg").read_bytes() == b"old-image"
    assert destination_file.read_bytes() == b"old-parquet"
    assert not directory_backup.exists()


def test_recovery_restores_pair_when_both_destinations_are_missing(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    directory_backup, file_backup = _backup_paths(destination_dir, destination_file)
    directory_backup.mkdir()
    (directory_backup / "old.jpg").write_bytes(b"old-image")
    file_backup.write_bytes(b"old-parquet")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert (destination_dir / "old.jpg").read_bytes() == b"old-image"
    assert destination_file.read_bytes() == b"old-parquet"
    assert not directory_backup.exists()
    assert not file_backup.exists()


def test_recovery_removes_new_directory_and_restores_old_pair(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    directory_backup, file_backup = _backup_paths(destination_dir, destination_file)
    directory_backup.mkdir()
    (directory_backup / "old.jpg").write_bytes(b"old-image")
    file_backup.write_bytes(b"old-parquet")
    destination_dir.mkdir()
    (destination_dir / "new.jpg").write_bytes(b"new-image")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert (destination_dir / "old.jpg").read_bytes() == b"old-image"
    assert not (destination_dir / "new.jpg").exists()
    assert destination_file.read_bytes() == b"old-parquet"


def test_recovery_keeps_committed_pair_and_cleans_stale_backups(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    directory_backup, file_backup = _backup_paths(destination_dir, destination_file)
    directory_backup.mkdir()
    (directory_backup / "old.jpg").write_bytes(b"old-image")
    file_backup.write_bytes(b"old-parquet")
    destination_dir.mkdir()
    (destination_dir / "new.jpg").write_bytes(b"new-image")
    destination_file.write_bytes(b"new-parquet")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert (destination_dir / "new.jpg").read_bytes() == b"new-image"
    assert destination_file.read_bytes() == b"new-parquet"
    assert not directory_backup.exists()
    assert not file_backup.exists()


def test_recovery_removes_new_directory_when_only_old_file_backup_exists(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    _, file_backup = _backup_paths(destination_dir, destination_file)
    destination_dir.mkdir()
    (destination_dir / "new.jpg").write_bytes(b"new-image")
    file_backup.write_bytes(b"old-parquet")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert not destination_dir.exists()
    assert destination_file.read_bytes() == b"old-parquet"
    assert not file_backup.exists()


def test_recovery_keeps_completed_directory_and_file_with_only_file_backup(tmp_path):
    destination_dir = tmp_path / "images"
    destination_file = tmp_path / "products.parquet"
    _, file_backup = _backup_paths(destination_dir, destination_file)
    destination_dir.mkdir()
    (destination_dir / "new.jpg").write_bytes(b"new-image")
    destination_file.write_bytes(b"new-parquet")
    file_backup.write_bytes(b"old-parquet")

    recover_directory_and_file_publish(destination_dir, destination_file)

    assert (destination_dir / "new.jpg").read_bytes() == b"new-image"
    assert destination_file.read_bytes() == b"new-parquet"
    assert not file_backup.exists()


def test_joint_publish_rolls_back_directory_when_file_replace_fails(
    tmp_path, monkeypatch
):
    destination_dir = tmp_path / "images"
    destination_dir.mkdir()
    (destination_dir / "old.jpg").write_bytes(b"old-image")
    destination_file = tmp_path / "products.parquet"
    destination_file.write_bytes(b"old-parquet")
    staging_dir = tmp_path / ".images.staging"
    staging_dir.mkdir()
    (staging_dir / "new.jpg").write_bytes(b"new-image")
    staging_file = tmp_path / "products.parquet.tmp"
    staging_file.write_bytes(b"new-parquet")
    real_replace = os.replace

    def fail_final_file_replace(source, destination):
        if str(source) == str(staging_file) and str(destination) == str(
            destination_file
        ):
            raise PermissionError("simulated Windows file lock")
        return real_replace(source, destination)

    monkeypatch.setattr("skillchain.data.os.replace", fail_final_file_replace)

    with pytest.raises(PermissionError, match="file lock"):
        publish_staged_directory_and_file(
            staging_dir, destination_dir, staging_file, destination_file
        )

    assert (destination_dir / "old.jpg").read_bytes() == b"old-image"
    assert not (destination_dir / "new.jpg").exists()
    assert destination_file.read_bytes() == b"old-parquet"


def test_directory_publish_keeps_new_destination_when_backup_cleanup_fails(
    tmp_path, monkeypatch
):
    destination = tmp_path / "index"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".index.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    backup = tmp_path / ".index.backup"
    real_rmtree = __import__("shutil").rmtree

    def fail_only_backup_cleanup(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(backup):
            raise PermissionError("backup cleanup locked")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("skillchain.data.shutil.rmtree", fail_only_backup_cleanup)

    publish_staged_directory(staging, destination)

    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (destination / "old.txt").exists()
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
