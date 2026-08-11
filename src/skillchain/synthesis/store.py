"""Phase 3 文件状态机使用的规范序列化与不可覆盖原语。"""

from __future__ import annotations

import hashlib
import json
import errno
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

_WINDOWS_RENAME_ATTEMPTS = 6
_WINDOWS_RENAME_INITIAL_DELAY_SECONDS = 0.05
_WINDOWS_TRANSIENT_RENAME_WINERRORS = {5, 32}


def canonical_json_bytes(value: BaseModel | dict | list) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(values: Iterable[BaseModel | dict]) -> bytes:
    return b"".join(canonical_json_bytes(value) for value in values)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_create_file(path: str | Path, content: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"目标已存在，拒绝覆盖: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def atomic_replace_file(path: str | Path, content: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def new_staging_directory(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"目标目录已存在，拒绝覆盖: {destination}")
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )


def atomic_publish_new_directory(staging: str | Path, destination: str | Path) -> Path:
    """Publish a directory without replacing a path created by a racing writer."""

    staging = Path(staging)
    destination = Path(destination)
    _require_real_directory(staging, "staging directory")
    if os.path.lexists(destination):
        raise FileExistsError(f"目标目录已存在，拒绝覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(destination.parent, "destination parent")
    _rename_directory_noreplace(staging, destination)
    return destination


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"unable to inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a real directory: {path}")


def _rename_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically rename while refusing every existing destination.

    POSIX ``rename`` is replace-capable, so it is not a safe fallback.  Linux
    uses ``renameat2(RENAME_NOREPLACE)``; Windows ``os.rename`` already refuses
    an existing destination.  Other platforms fail closed.
    """

    if os.name == "nt":
        for attempt in range(_WINDOWS_RENAME_ATTEMPTS):
            try:
                os.rename(staging, destination)
                return
            except OSError as exc:
                if os.path.lexists(destination):
                    raise FileExistsError(
                        "target directory already exists; refusing overwrite: "
                        f"{destination}"
                    ) from None
                transient = (
                    exc.errno in {errno.EACCES, errno.EBUSY}
                    or getattr(exc, "winerror", None)
                    in _WINDOWS_TRANSIENT_RENAME_WINERRORS
                )
                if not transient or attempt == _WINDOWS_RENAME_ATTEMPTS - 1:
                    raise
                time.sleep(
                    _WINDOWS_RENAME_INITIAL_DELAY_SECONDS * (2**attempt)
                )
        raise AssertionError("unreachable Windows rename retry state")

    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "atomic create-only directory publish requires Linux renameat2"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"target directory already exists; refusing overwrite: {destination}"
            ) from None
        if error_number in {errno.ENOSYS, errno.EINVAL}:
            raise RuntimeError(
                "atomic create-only directory publish is unavailable on this filesystem"
            ) from None
        raise OSError(error_number, os.strerror(error_number), str(destination))

    raise RuntimeError(
        "atomic create-only directory publish is unsupported on this platform"
    )


def next_revision(root: str | Path, base_batch_id: str) -> int:
    root = Path(root)
    pattern = re.compile(rf"^{re.escape(base_batch_id)}-r(\d+)$")
    accepted = [
        path
        for path in (root / "accepted").glob(f"{base_batch_id}-r*")
        if pattern.fullmatch(path.name)
    ]
    if accepted:
        raise ValueError(f"base batch 已 accepted，禁止新 revision: {base_batch_id}")
    staging = [
        path
        for path in (root / "staging").glob(f"{base_batch_id}-r*")
        if pattern.fullmatch(path.name)
    ]
    if staging:
        raise ValueError(f"base batch 已有 active staging revision: {base_batch_id}")
    rejected_revisions = [
        int(match.group(1))
        for path in (root / "rejected").glob(f"{base_batch_id}-r*")
        if (match := pattern.fullmatch(path.name))
    ]
    return max(rejected_revisions, default=0) + 1
