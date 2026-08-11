"""Cross-session raw dataset download orchestration.

The downloader deliberately separates public HTTP artifacts, idempotent project
commands, and gated/manual sources.  Runtime state lives beside ``data/raw`` and
never stores URL queries, fragments, or embedded credentials.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from skillchain import config

DEFAULT_MANIFEST = (
    config.ROOT / "specs" / "data_sources" / "raw-download-profiles-v1.json"
)
STATE_SCHEMA_VERSION = 1
CHUNK_BYTES = 8 << 20
CHECKPOINT_BYTES = 64 << 20
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RawDownloadError(RuntimeError):
    """Raised when acquisition cannot continue without risking corrupt bytes."""


@dataclass(frozen=True)
class Artifact:
    source_id: str
    path: str
    url: str
    expected_bytes: int | None = None
    sha256: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RawDownloadError(f"JSON root must be an object: {path}")
    return value


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawDownloadError(f"unsafe artifact path: {value!r}")
    return path.as_posix()


def _redacted_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RawDownloadError(
            f"artifact URL must be absolute HTTP(S): {_redacted_url(url)!r}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise RawDownloadError("artifact URLs must not contain embedded credentials")


def _validate_artifact(source_id: str, raw: Mapping[str, Any]) -> Artifact:
    path = _safe_relative_path(str(raw.get("path", "")))
    url = str(raw.get("url", ""))
    _validate_url(url)
    expected = raw.get("bytes")
    if expected is not None and (not isinstance(expected, int) or expected <= 0):
        raise RawDownloadError(f"{source_id}/{path}: bytes must be a positive integer")
    digest = raw.get("sha256")
    if digest is not None and (
        not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise RawDownloadError(f"{source_id}/{path}: invalid sha256")
    return Artifact(source_id, path, url, expected, digest)


def load_manifest(
    manifest_path: Path,
    *,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise RawDownloadError("unsupported raw-download manifest schema")
    if overrides_path is not None:
        overrides = _read_json(overrides_path)
        if overrides.get("schema_version") != 1:
            raise RawDownloadError("unsupported overrides schema")
        for source_id, replacement in overrides.get("sources", {}).items():
            if not isinstance(replacement, dict):
                raise RawDownloadError(f"override for {source_id} must be an object")
            current = dict(manifest.get("sources", {}).get(source_id, {}))
            current.update(replacement)
            manifest.setdefault("sources", {})[source_id] = current
    return manifest


def resolve_profile(manifest: Mapping[str, Any], profile_id: str) -> list[str]:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        choices = ", ".join(sorted(profiles or {}))
        raise RawDownloadError(
            f"unknown profile {profile_id!r}; choose from: {choices}"
        )
    visiting: set[str] = set()

    def visit(current: str) -> list[str]:
        if current in visiting:
            raise RawDownloadError(f"profile inheritance cycle at {current}")
        visiting.add(current)
        raw = profiles.get(current)
        if not isinstance(raw, dict):
            raise RawDownloadError(f"profile {current} must be an object")
        result: list[str] = []
        parent = raw.get("extends")
        if parent is not None:
            result.extend(visit(str(parent)))
        values = raw.get("sources", [])
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise RawDownloadError(f"profile {current}.sources must be strings")
        result.extend(values)
        visiting.remove(current)
        return list(dict.fromkeys(result))

    source_ids = visit(profile_id)
    known = manifest.get("sources", {})
    missing = sorted(set(source_ids).difference(known))
    if missing:
        raise RawDownloadError(f"profile references unknown sources: {missing}")
    return source_ids


def _selected_sources(
    manifest: Mapping[str, Any],
    profile_id: str,
    only: Iterable[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    ids = resolve_profile(manifest, profile_id)
    if only:
        requested = set(only)
        unknown = requested.difference(ids)
        if unknown:
            raise RawDownloadError(
                f"--source values are not in profile {profile_id}: {sorted(unknown)}"
            )
        ids = [source_id for source_id in ids if source_id in requested]
    return [(source_id, dict(manifest["sources"][source_id])) for source_id in ids]


class State:
    def __init__(self, path: Path, profile_id: str, manifest_id: str) -> None:
        self.path = path
        if path.is_file():
            self.value = _read_json(path)
            expected = (profile_id, manifest_id)
            actual = (
                self.value.get("profile_id"),
                self.value.get("manifest_id"),
            )
            if actual != expected:
                raise RawDownloadError(
                    f"state belongs to profile/manifest {actual}, expected {expected}"
                )
        else:
            self.value = {
                "schema_version": STATE_SCHEMA_VERSION,
                "profile_id": profile_id,
                "manifest_id": manifest_id,
                "created_at": _now(),
                "updated_at": _now(),
                "items": {},
            }
            self.save()

    def item(self, key: str) -> dict[str, Any]:
        return self.value.setdefault("items", {}).setdefault(key, {})

    def save(self) -> None:
        self.value["updated_at"] = _now()
        _atomic_json(self.path, self.value)


@contextmanager
def state_lock(path: Path, *, break_lock: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if break_lock and path.exists():
        path.unlink()
    payload = _canonical_bytes(
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _now(),
        }
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        owner = path.read_text(encoding="utf-8", errors="replace").strip()
        raise RawDownloadError(
            f"another downloader owns {path}: {owner}; verify it is stale, "
            "then rerun with --break-lock"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
        yield
    finally:
        path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_metadata(session: requests.Session, artifact: Artifact) -> dict[str, Any]:
    response = session.head(artifact.url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return {
        "url": _redacted_url(artifact.url),
        "bytes": int(length)
        if length and length.isdigit()
        else artifact.expected_bytes,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }


def _identity(metadata: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        metadata.get("bytes"),
        metadata.get("etag"),
        metadata.get("last_modified"),
    )


def _verify_complete(
    path: Path, artifact: Artifact, expected_bytes: int | None
) -> None:
    actual = path.stat().st_size
    if expected_bytes is not None and actual != expected_bytes:
        raise RawDownloadError(
            f"{path} has {actual} bytes, expected {expected_bytes}; "
            "move it aside before retrying"
        )
    if artifact.sha256 is not None:
        actual_digest = _sha256(path)
        if actual_digest != artifact.sha256:
            raise RawDownloadError(
                f"{path} SHA-256 mismatch: expected {artifact.sha256}, "
                f"got {actual_digest}; move the file aside before retrying"
            )


def _finalize_local_artifact(path: Path, artifact: Artifact) -> None:
    """Verify and atomically promote a completed local ``.part`` artifact."""
    partial = path.with_name(path.name + ".part")
    aria2_control = partial.with_name(partial.name + ".aria2")
    if path.exists():
        _verify_complete(path, artifact, artifact.expected_bytes)
        return
    if aria2_control.exists():
        raise RawDownloadError(
            f"{aria2_control} exists; the local downloader may still be writing"
        )
    if not partial.exists():
        raise RawDownloadError(
            f"{path} is missing and no completed local partial is available"
        )
    _verify_complete(partial, artifact, artifact.expected_bytes)
    os.replace(partial, path)


def _safe_error(error: BaseException) -> str:
    """Return a log-safe exception summary without request URLs or credentials."""
    if isinstance(error, requests.RequestException):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        suffix = f" (HTTP {status})" if status is not None else ""
        return f"{type(error).__name__}{suffix}"
    return str(error)


def download_artifact(
    artifact: Artifact,
    *,
    raw_root: Path,
    state: State,
    session: requests.Session,
) -> None:
    key = f"{artifact.source_id}:{artifact.path}"
    record = state.item(key)
    destination = raw_root / artifact.path
    partial = destination.with_name(destination.name + ".part")
    aria2_control = partial.with_name(partial.name + ".aria2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if aria2_control.exists():
        raise RawDownloadError(
            f"{key}: aria2 control file exists beside the partial; let the "
            "single aria2 process finish or inspect its piece bitmap before "
            "running the project downloader"
        )
    metadata = _remote_metadata(session, artifact)
    expected_bytes = artifact.expected_bytes or metadata.get("bytes")
    if (
        artifact.expected_bytes is not None
        and metadata.get("bytes") is not None
        and artifact.expected_bytes != metadata["bytes"]
    ):
        raise RawDownloadError(
            f"{key}: manifest length {artifact.expected_bytes} differs from "
            f"remote length {metadata['bytes']}"
        )
    previous = record.get("remote")
    if partial.exists() and previous and _identity(previous) != _identity(metadata):
        raise RawDownloadError(
            f"{key}: remote identity changed while .part exists; move the partial "
            "file aside and restart after reviewing the source revision"
        )
    record.update(
        {
            "kind": "artifact",
            "status": "running",
            "destination": artifact.path,
            "remote": metadata,
            "expected_sha256": artifact.sha256,
            "updated_at": _now(),
        }
    )
    state.save()

    if destination.is_file():
        _verify_complete(destination, artifact, expected_bytes)
        record.update(
            {
                "status": "complete",
                "downloaded_bytes": destination.stat().st_size,
                "completed_at": _now(),
            }
        )
        state.save()
        print(f"[skip] {key} ({destination.stat().st_size:,} bytes)")
        return

    have = partial.stat().st_size if partial.exists() else 0
    if expected_bytes is not None and have > expected_bytes:
        raise RawDownloadError(f"{partial} is larger than the expected remote object")
    if expected_bytes is not None and have == expected_bytes:
        _verify_complete(partial, artifact, expected_bytes)
        os.replace(partial, destination)
        record.update(
            {
                "status": "complete",
                "downloaded_bytes": destination.stat().st_size,
                "completed_at": _now(),
            }
        )
        state.save()
        print(f"[done] {key} ({destination.stat().st_size:,} bytes)")
        return
    headers: dict[str, str] = {}
    if have:
        etag = metadata.get("etag")
        validator = (
            etag
            if isinstance(etag, str) and not etag.strip().startswith("W/")
            else metadata.get("last_modified")
        )
        if validator:
            headers["Range"] = f"bytes={have}-"
            headers["If-Range"] = str(validator)
        else:
            print(
                f"[info] {key}: remote supplies no strong resume validator; "
                "restarting the .part file"
            )
            have = 0
    print(
        f"[get ] {key} {have:,}/"
        f"{expected_bytes if expected_bytes is not None else '?'} bytes"
    )
    with session.get(
        artifact.url,
        headers=headers,
        stream=True,
        timeout=(30, 180),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        append = have > 0 and response.status_code == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {have}-"):
                raise RawDownloadError(
                    f"{key}: invalid resume Content-Range {content_range!r}"
                )
        mode = "ab" if append else "wb"
        if have and not append:
            print(f"[info] {key}: server ignored Range; restarting the .part file")
        checkpoint = have if append else 0
        with partial.open(mode) as target:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                target.write(chunk)
                current = target.tell()
                if current - checkpoint >= CHECKPOINT_BYTES:
                    target.flush()
                    os.fsync(target.fileno())
                    record["downloaded_bytes"] = current
                    record["updated_at"] = _now()
                    state.save()
                    checkpoint = current
            target.flush()
            os.fsync(target.fileno())

    _verify_complete(partial, artifact, expected_bytes)
    os.replace(partial, destination)
    record.update(
        {
            "status": "complete",
            "downloaded_bytes": destination.stat().st_size,
            "completed_at": _now(),
        }
    )
    state.save()
    print(f"[done] {key} ({destination.stat().st_size:,} bytes)")


def _run_command(
    source_id: str,
    source: Mapping[str, Any],
    *,
    state: State,
    repository_root: Path,
) -> None:
    raw = source.get("command")
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise RawDownloadError(f"{source_id}: command must be a string array")
    command = [
        value.replace("{python}", sys.executable).replace(
            "{repository_root}", str(repository_root)
        )
        for value in raw
    ]
    key = f"{source_id}:__command__"
    record = state.item(key)
    if record.get("status") == "complete":
        print(f"[skip] {source_id}: command already completed")
        return
    record.update(
        {
            "kind": "command",
            "status": "running",
            "command": [
                "{python}" if value == sys.executable else value for value in command
            ],
            "updated_at": _now(),
        }
    )
    state.save()
    print(f"[exec] {source_id}: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=repository_root, check=True)
    except subprocess.CalledProcessError as error:
        record.update(
            {"status": "failed", "returncode": error.returncode, "updated_at": _now()}
        )
        state.save()
        raise RawDownloadError(
            f"{source_id}: delegated command failed with {error.returncode}; "
            "rerun the profile to resume"
        ) from error
    record.update({"status": "complete", "completed_at": _now()})
    state.save()


def _artifacts(source_id: str, source: Mapping[str, Any]) -> list[Artifact]:
    raw = source.get("artifacts")
    if not isinstance(raw, list):
        raise RawDownloadError(f"{source_id}: artifacts must be an array")
    return [_validate_artifact(source_id, item) for item in raw]


def _known_bytes(source_id: str, source: Mapping[str, Any]) -> tuple[int, int]:
    if source.get("kind") not in {"artifacts", "local_artifacts"}:
        return 0, 0
    artifacts = _artifacts(source_id, source)
    return (
        sum(item.expected_bytes or 0 for item in artifacts),
        sum(item.expected_bytes is None for item in artifacts),
    )


def print_plan(
    selected: list[tuple[str, dict[str, Any]]],
    *,
    raw_root: Path,
) -> None:
    total = 0
    unknown = 0
    manual = 0
    print(f"raw_root={raw_root}")
    for source_id, source in selected:
        kind = source.get("kind")
        known, missing = _known_bytes(source_id, source)
        total += known
        unknown += missing
        if kind == "manual":
            manual += 1
        label = (
            f"{known / (1024**3):.2f} GiB"
            if known
            else ("runtime-sized" if kind != "manual" else "manual/gated")
        )
        print(f"{source_id:24} {kind:10} {label}")
    print(f"known_download_bytes={total:,} ({total / (1024**3):.2f} GiB)")
    print(f"unknown_size_artifacts={unknown}")
    print(f"manual_or_gated_sources={manual}")


def _remaining_known_bytes(
    selected: list[tuple[str, dict[str, Any]]], raw_root: Path
) -> int:
    remaining = 0
    for source_id, source in selected:
        if source.get("kind") != "artifacts":
            continue
        for artifact in _artifacts(source_id, source):
            if artifact.expected_bytes is None:
                continue
            destination = raw_root / artifact.path
            partial = destination.with_name(destination.name + ".part")
            have = (
                destination.stat().st_size
                if destination.exists()
                else partial.stat().st_size
                if partial.exists()
                else 0
            )
            remaining += max(0, artifact.expected_bytes - have)
    return remaining


def download_profile(
    manifest: Mapping[str, Any],
    *,
    profile_id: str,
    raw_root: Path,
    state_root: Path,
    only: Iterable[str] | None,
    break_lock: bool,
    retries: int,
) -> int:
    selected = _selected_sources(manifest, profile_id, only)
    raw_root.mkdir(parents=True, exist_ok=True)
    remaining = _remaining_known_bytes(selected, raw_root)
    free = shutil.disk_usage(raw_root).free
    if free < remaining:
        raise RawDownloadError(
            f"known remaining downloads need {remaining:,} bytes but only "
            f"{free:,} bytes are free; unknown-size/manual sources are additional"
        )
    state_path = state_root / profile_id / "state.json"
    lock_path = state_root / profile_id / "download.lock"
    state = State(state_path, profile_id, str(manifest["manifest_id"]))
    blocked: list[str] = []
    failures: list[str] = []
    with state_lock(lock_path, break_lock=break_lock):
        with requests.Session() as session:
            for source_id, source in selected:
                kind = source.get("kind")
                if kind == "manual":
                    blocked.append(source_id)
                    record = state.item(f"{source_id}:__manual__")
                    record.update(
                        {
                            "kind": "manual",
                            "status": "blocked",
                            "reason": source.get("reason"),
                            "instructions": source.get("instructions"),
                            "updated_at": _now(),
                        }
                    )
                    state.save()
                    print(f"[BLOCKED] {source_id}: {source.get('reason')}")
                    continue
                try:
                    if kind == "command":
                        _run_command(
                            source_id,
                            source,
                            state=state,
                            repository_root=config.ROOT,
                        )
                    elif kind == "artifacts":
                        for artifact in _artifacts(source_id, source):
                            for attempt in range(1, retries + 1):
                                try:
                                    download_artifact(
                                        artifact,
                                        raw_root=raw_root,
                                        state=state,
                                        session=session,
                                    )
                                    break
                                except (OSError, requests.RequestException) as error:
                                    if attempt == retries:
                                        raise
                                    delay = min(30, 2 ** (attempt - 1))
                                    print(
                                        f"[retry] {source_id}/{artifact.path}: "
                                        f"{_safe_error(error)}; waiting {delay}s"
                                    )
                                    time.sleep(delay)
                    elif kind == "local_artifacts":
                        for artifact in _artifacts(source_id, source):
                            path = raw_root / artifact.path
                            _finalize_local_artifact(path, artifact)
                            record = state.item(f"{artifact.source_id}:{artifact.path}")
                            record.update(
                                {
                                    "kind": "local_artifact",
                                    "status": "complete",
                                    "downloaded_bytes": path.stat().st_size,
                                    "source": _redacted_url(artifact.url),
                                    "sha256": artifact.sha256,
                                    "completed_at": _now(),
                                }
                            )
                            state.save()
                            print(
                                f"[local] {artifact.source_id}:{artifact.path} "
                                f"{path.stat().st_size:,} bytes"
                            )
                    else:
                        raise RawDownloadError(
                            f"{source_id}: unsupported source kind {kind!r}"
                        )
                    obsolete_manual_key = f"{source_id}:__manual__"
                    if obsolete_manual_key in state.value.get("items", {}):
                        del state.value["items"][obsolete_manual_key]
                        state.save()
                except (OSError, requests.RequestException, RawDownloadError) as error:
                    failures.append(source_id)
                    print(
                        f"[FAILED] {source_id}: {_safe_error(error)}",
                        file=sys.stderr,
                    )
    print(f"state={state_path}")
    if blocked:
        print(f"blocked={','.join(blocked)}")
    if failures:
        print(f"failed={','.join(dict.fromkeys(failures))}", file=sys.stderr)
    return 1 if failures else 2 if blocked else 0


def verify_profile(
    manifest: Mapping[str, Any],
    *,
    profile_id: str,
    raw_root: Path,
    only: Iterable[str] | None,
) -> int:
    incomplete: list[str] = []
    for source_id, source in _selected_sources(manifest, profile_id, only):
        if source.get("kind") not in {"artifacts", "local_artifacts"}:
            print(f"[skip] {source_id}: verification is {source.get('kind')}-managed")
            continue
        for artifact in _artifacts(source_id, source):
            path = raw_root / artifact.path
            if not path.is_file():
                incomplete.append(f"{source_id}:{artifact.path}")
                print(f"[MISS] {source_id}:{artifact.path}")
                continue
            try:
                _verify_complete(path, artifact, artifact.expected_bytes)
            except RawDownloadError as error:
                incomplete.append(f"{source_id}:{artifact.path}")
                print(f"[BAD ] {error}")
            else:
                print(f"[ OK ] {source_id}:{artifact.path}")
    return 1 if incomplete else 0


def print_status(
    state_path: Path,
    *,
    active_source_ids: set[str] | None = None,
) -> int:
    if not state_path.is_file():
        print(f"no state: {state_path}")
        return 1
    state = _read_json(state_path)
    counts: dict[str, int] = {}
    ignored_stale_items = 0
    for key, item in sorted(state.get("items", {}).items()):
        source_id = key.partition(":")[0]
        if active_source_ids is not None and source_id not in active_source_ids:
            ignored_stale_items += 1
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        downloaded = item.get("downloaded_bytes")
        suffix = f" {downloaded:,} bytes" if isinstance(downloaded, int) else ""
        print(f"{status:10} {key}{suffix}")
    print(f"summary={json.dumps(counts, sort_keys=True)}")
    if ignored_stale_items:
        print(f"ignored_stale_items={ignored_stale_items}")
    print(f"updated_at={state.get('updated_at')}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--overrides",
        type=Path,
        help="local JSON that replaces gated/manual source definitions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "download", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--profile", required=True)
        child.add_argument("--root", type=Path, default=config.DATA_DIR / "raw")
        child.add_argument("--source", action="append", dest="sources")
        if name == "download":
            child.add_argument("--state-root", type=Path)
            child.add_argument("--break-lock", action="store_true")
            child.add_argument("--retries", type=int, default=4)
    status = subparsers.add_parser("status")
    status.add_argument("--profile", required=True)
    status.add_argument(
        "--state-root",
        type=Path,
        default=config.DATA_DIR / "raw" / ".download-state",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        manifest = load_manifest(args.manifest, overrides_path=args.overrides)
        if args.command == "status":
            raise SystemExit(
                print_status(
                    args.state_root / args.profile / "state.json",
                    active_source_ids=set(resolve_profile(manifest, args.profile)),
                )
            )
        raw_root = args.root.resolve()
        if args.command == "plan":
            print_plan(
                _selected_sources(manifest, args.profile, args.sources),
                raw_root=raw_root,
            )
            return
        if args.command == "verify":
            raise SystemExit(
                verify_profile(
                    manifest,
                    profile_id=args.profile,
                    raw_root=raw_root,
                    only=args.sources,
                )
            )
        if args.retries < 1:
            raise RawDownloadError("--retries must be at least 1")
        state_root = (
            args.state_root.resolve()
            if args.state_root
            else raw_root / ".download-state"
        )
        raise SystemExit(
            download_profile(
                manifest,
                profile_id=args.profile,
                raw_root=raw_root,
                state_root=state_root,
                only=args.sources,
                break_lock=args.break_lock,
                retries=args.retries,
            )
        )
    except RawDownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
