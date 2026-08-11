from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from skillchain.data.raw_download import (
    Artifact,
    RawDownloadError,
    State,
    download_artifact,
    download_profile,
    print_status,
    state_lock,
)


class _Response:
    def __init__(
        self,
        body: bytes = b"",
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class _Session:
    def __init__(
        self,
        *,
        body: bytes,
        status_code: int,
        etag: str = '"fixture-v1"',
        content_range: str | None = None,
        remote_bytes: int | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.etag = etag
        self.content_range = content_range
        self.remote_bytes = remote_bytes
        self.get_headers: dict[str, str] | None = None

    def head(self, *_args, **_kwargs) -> _Response:
        headers = {
            "Content-Length": str(
                self.remote_bytes if self.remote_bytes is not None else len(self.body)
            ),
            "ETag": self.etag,
            "Last-Modified": "Wed, 23 Jul 2026 00:00:00 GMT",
        }
        return _Response(headers=headers)

    def get(self, *_args, headers=None, **_kwargs) -> _Response:
        self.get_headers = dict(headers or {})
        response_headers = {}
        if self.content_range is not None:
            response_headers["Content-Range"] = self.content_range
        return _Response(
            self.body,
            status_code=self.status_code,
            headers=response_headers,
        )


def _state(tmp_path: Path) -> State:
    return State(tmp_path / "state.json", "mvp", "fixture-manifest")


def _artifact(
    content: bytes, *, url: str = "https://example.test/data.bin"
) -> Artifact:
    return Artifact(
        source_id="fixture",
        path="fixture/data.bin",
        url=url,
        expected_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_status_ignores_stale_items_from_sources_removed_from_profile(
    tmp_path: Path, capsys
) -> None:
    state = _state(tmp_path)
    state.item("active:data.bin").update({"status": "complete", "downloaded_bytes": 3})
    state.item("retired:__manual__").update({"status": "blocked"})
    state.save()

    assert (
        print_status(
            state.path,
            active_source_ids={"active"},
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "complete   active:data.bin 3 bytes" in output
    assert "retired:__manual__" not in output
    assert 'summary={"complete": 1}' in output
    assert "ignored_stale_items=1" in output


def test_download_resumes_partial_with_identity_and_content_range(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    artifact = _artifact(complete)
    destination = tmp_path / artifact.path
    destination.parent.mkdir(parents=True)
    destination.with_name(destination.name + ".part").write_bytes(b"abc")
    state = _state(tmp_path)
    state.item("fixture:fixture/data.bin")["remote"] = {
        "url": artifact.url,
        "bytes": len(complete),
        "etag": '"fixture-v1"',
        "last_modified": "Wed, 23 Jul 2026 00:00:00 GMT",
    }
    state.save()
    session = _Session(
        body=b"def",
        status_code=206,
        content_range="bytes 3-5/6",
        remote_bytes=6,
    )

    download_artifact(
        artifact,
        raw_root=tmp_path,
        state=state,
        session=session,
    )

    assert destination.read_bytes() == complete
    assert session.get_headers == {
        "Range": "bytes=3-",
        "If-Range": '"fixture-v1"',
    }
    assert not destination.with_name(destination.name + ".part").exists()
    assert state.item("fixture:fixture/data.bin")["status"] == "complete"


def test_download_restarts_partial_when_server_ignores_range(tmp_path: Path) -> None:
    complete = b"new-content"
    artifact = _artifact(complete)
    destination = tmp_path / artifact.path
    destination.parent.mkdir(parents=True)
    destination.with_name(destination.name + ".part").write_bytes(b"old")
    state = _state(tmp_path)
    session = _Session(body=complete, status_code=200)

    download_artifact(
        artifact,
        raw_root=tmp_path,
        state=state,
        session=session,
    )

    assert destination.read_bytes() == complete
    assert session.get_headers["Range"] == "bytes=3-"


def test_download_rejects_changed_remote_identity_with_partial(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    artifact = _artifact(complete)
    destination = tmp_path / artifact.path
    destination.parent.mkdir(parents=True)
    destination.with_name(destination.name + ".part").write_bytes(b"abc")
    state = _state(tmp_path)
    state.item("fixture:fixture/data.bin")["remote"] = {
        "url": artifact.url,
        "bytes": len(complete),
        "etag": '"old"',
        "last_modified": "Wed, 23 Jul 2026 00:00:00 GMT",
    }
    state.save()
    session = _Session(
        body=b"def",
        status_code=206,
        etag='"new"',
        content_range="bytes 3-5/6",
        remote_bytes=6,
    )

    with pytest.raises(RawDownloadError, match="remote identity changed"):
        download_artifact(
            artifact,
            raw_root=tmp_path,
            state=state,
            session=session,
        )

    assert destination.with_name(destination.name + ".part").read_bytes() == b"abc"


def test_download_rejects_aria2_owned_partial_even_at_expected_length(
    tmp_path: Path,
) -> None:
    complete = b"abcdef"
    artifact = _artifact(complete)
    destination = tmp_path / artifact.path
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(b"\0" * len(complete))
    partial.with_name(partial.name + ".aria2").write_bytes(b"piece bitmap")

    with pytest.raises(RawDownloadError, match="aria2 control file exists"):
        download_artifact(
            artifact,
            raw_root=tmp_path,
            state=_state(tmp_path),
            session=_Session(body=complete, status_code=200),
        )

    assert partial.exists()
    assert not destination.exists()


def test_download_state_redacts_url_query_and_fragment(tmp_path: Path) -> None:
    content = b"secret-safe-state"
    artifact = _artifact(
        content,
        url="https://user:password@example.test/data.bin?token=secret#fragment",
    )
    state = _state(tmp_path)
    session = _Session(body=content, status_code=200)

    download_artifact(
        artifact,
        raw_root=tmp_path,
        state=state,
        session=session,
    )

    raw_state = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "secret" not in raw_state
    assert "password" not in raw_state
    saved = json.loads(raw_state)
    remote = saved["items"]["fixture:fixture/data.bin"]["remote"]
    assert remote["url"] == "https://example.test/data.bin"


def test_state_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    lock = tmp_path / "download.lock"
    with state_lock(lock):
        with pytest.raises(RawDownloadError, match="another downloader owns"):
            with state_lock(lock):
                pass
    assert not lock.exists()


def test_local_artifacts_are_verified_without_network_and_clear_manual_state(
    tmp_path: Path,
) -> None:
    content = b"browser-acquired"
    destination = tmp_path / "sroie" / "official.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)
    state_root = tmp_path / ".download-state"
    state = State(state_root / "mvp" / "state.json", "mvp", "fixture-manifest")
    state.item("sroie:__manual__").update({"kind": "manual", "status": "blocked"})
    state.save()
    manifest = {
        "manifest_id": "fixture-manifest",
        "profiles": {"mvp": {"sources": ["sroie"]}},
        "sources": {
            "sroie": {
                "kind": "local_artifacts",
                "artifacts": [
                    {
                        "path": "sroie/official.zip",
                        "url": "https://example.test/authorized-download",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        },
    }

    result = download_profile(
        manifest,
        profile_id="mvp",
        raw_root=tmp_path,
        state_root=state_root,
        only=["sroie"],
        break_lock=False,
        retries=1,
    )

    saved = json.loads((state_root / "mvp" / "state.json").read_text("utf-8"))
    assert result == 0
    assert "sroie:__manual__" not in saved["items"]
    assert saved["items"]["sroie:sroie/official.zip"]["status"] == "complete"


def test_local_artifact_promotes_a_verified_partial_without_network(
    tmp_path: Path,
) -> None:
    content = b"locally-downloaded"
    destination = tmp_path / "rpc" / "archive.zip"
    destination.parent.mkdir(parents=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(content)
    state_root = tmp_path / ".download-state"
    manifest = {
        "manifest_id": "fixture-manifest",
        "profiles": {"mvp": {"sources": ["rpc"]}},
        "sources": {
            "rpc": {
                "kind": "local_artifacts",
                "artifacts": [
                    {
                        "path": "rpc/archive.zip",
                        "url": "https://example.test/authorized-download",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        },
    }

    result = download_profile(
        manifest,
        profile_id="mvp",
        raw_root=tmp_path,
        state_root=state_root,
        only=["rpc"],
        break_lock=False,
        retries=1,
    )

    assert result == 0
    assert destination.read_bytes() == content
    assert not partial.exists()


def test_failed_local_artifact_keeps_manual_blocker(tmp_path: Path) -> None:
    state_root = tmp_path / ".download-state"
    state = State(state_root / "mvp" / "state.json", "mvp", "fixture-manifest")
    state.item("sroie:__manual__").update({"kind": "manual", "status": "blocked"})
    state.save()
    manifest = {
        "manifest_id": "fixture-manifest",
        "profiles": {"mvp": {"sources": ["sroie"]}},
        "sources": {
            "sroie": {
                "kind": "local_artifacts",
                "artifacts": [
                    {
                        "path": "sroie/missing.zip",
                        "url": "https://example.test/authorized-download",
                        "bytes": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                    }
                ],
            }
        },
    }

    result = download_profile(
        manifest,
        profile_id="mvp",
        raw_root=tmp_path,
        state_root=state_root,
        only=["sroie"],
        break_lock=False,
        retries=1,
    )

    saved = json.loads((state_root / "mvp" / "state.json").read_text("utf-8"))
    assert result == 1
    assert saved["items"]["sroie:__manual__"]["status"] == "blocked"
