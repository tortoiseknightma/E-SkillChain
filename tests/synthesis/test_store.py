import os
from pathlib import Path

import pytest

import skillchain.synthesis.store as store
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)


def test_directory_publish_is_create_only_for_existing_empty_directory(tmp_path: Path):
    staging = new_staging_directory(tmp_path / "published")
    (staging / "payload.txt").write_text("ours", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        atomic_publish_new_directory(staging, destination)

    assert staging.is_dir()
    assert list(destination.iterdir()) == []


def test_directory_publish_refuses_dangling_symlink_destination(tmp_path: Path):
    destination = tmp_path / "published"
    try:
        os.symlink(tmp_path / "missing-target", destination, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")

    with pytest.raises(FileExistsError):
        new_staging_directory(destination)


def test_directory_publish_race_cannot_replace_winner(tmp_path: Path, monkeypatch):
    destination = tmp_path / "published"
    staging = new_staging_directory(destination)
    (staging / "payload.txt").write_text("ours", encoding="utf-8")
    original = store._rename_directory_noreplace

    def racing_publish(source: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("theirs", encoding="utf-8")
        original(source, target)

    monkeypatch.setattr(store, "_rename_directory_noreplace", racing_publish)
    with pytest.raises(FileExistsError):
        atomic_publish_new_directory(staging, destination)

    assert (destination / "winner.txt").read_text(encoding="utf-8") == "theirs"
    assert (staging / "payload.txt").read_text(encoding="utf-8") == "ours"


def test_directory_publish_moves_complete_staging_tree(tmp_path: Path):
    destination = tmp_path / "published"
    staging = new_staging_directory(destination)
    (staging / "payload.txt").write_text("complete", encoding="utf-8")

    assert atomic_publish_new_directory(staging, destination) == destination
    assert not staging.exists()
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "complete"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing-lock behavior")
def test_directory_publish_retries_transient_windows_sharing_lock(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "published"
    staging = new_staging_directory(destination)
    (staging / "payload.txt").write_text("complete", encoding="utf-8")
    real_rename = store.os.rename
    attempts = 0
    delays: list[float] = []

    def transient_then_success(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "simulated Windows sharing lock")
        real_rename(source, target)

    monkeypatch.setattr(store.os, "rename", transient_then_success)
    monkeypatch.setattr(store.time, "sleep", delays.append)

    assert atomic_publish_new_directory(staging, destination) == destination
    assert attempts == 3
    assert delays == [0.05, 0.1]
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "complete"
