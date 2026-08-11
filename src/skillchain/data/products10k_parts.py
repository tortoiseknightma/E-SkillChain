"""Safely assemble a Products-10K split archive downloaded from JD Pan."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Iterable


class ProductPartsError(RuntimeError):
    """Raised when a split archive cannot be assembled without ambiguity."""


CHUNK_BYTES = 8 << 20
MD5_PATTERN = re.compile(r"^([0-9a-fA-F]{32})\s+\*?(.+)$")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(manifest_path: Path, archive_name: str) -> str:
    """Read one expected MD5 from the JD-supplied manifest without guessing."""

    try:
        lines = Path(manifest_path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProductPartsError(f"cannot read MD5 manifest: {manifest_path}") from error
    matches: list[str] = []
    for line in lines:
        match = MD5_PATTERN.fullmatch(line.strip())
        if match and match.group(2) == archive_name:
            matches.append(match.group(1).lower())
    if len(matches) != 1:
        raise ProductPartsError(
            f"MD5 manifest must contain {archive_name} exactly once, found {len(matches)}"
        )
    return matches[0]


def split_parts(parts_directory: Path, stem: str) -> tuple[Path, ...]:
    """Return strict .z01..zNN followed by .zip order for one archive stem."""

    directory = Path(parts_directory)
    final_part = directory / f"{stem}.zip"
    numbered: list[tuple[int, Path]] = []
    expression = re.compile(rf"^{re.escape(stem)}\.z(\d{{2}})$", re.IGNORECASE)
    for path in directory.iterdir() if directory.is_dir() else ():
        if not path.is_file():
            continue
        match = expression.fullmatch(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    if not final_part.is_file() or not numbered:
        raise ProductPartsError(f"incomplete split archive for {stem}")
    expected_numbers = list(range(1, len(numbered) + 1))
    actual_numbers = [number for number, _ in numbered]
    if actual_numbers != expected_numbers:
        raise ProductPartsError(f"split archive {stem} has missing or duplicate volume numbers")
    paths = tuple(path for _, path in numbered) + (final_part,)
    if any(path.stat().st_size == 0 for path in paths):
        raise ProductPartsError(f"split archive {stem} contains an empty volume")
    return paths


def _copy_from(paths: Iterable[Path], destination: Path, have: int) -> int:
    written = have
    offset = 0
    with destination.open("ab") as target:
        for path in paths:
            size = path.stat().st_size
            end = offset + size
            if written >= end:
                offset = end
                continue
            source_offset = max(0, written - offset)
            with path.open("rb") as source:
                source.seek(source_offset)
                while chunk := source.read(CHUNK_BYTES):
                    target.write(chunk)
                    written += len(chunk)
            offset = end
    return written


def assemble_split_archive(
    *,
    parts_directory: Path,
    stem: str,
    destination: Path,
    expected_md5_hex: str,
) -> int:
    """Resume a stream-concatenation and atomically promote only on MD5 match."""

    if re.fullmatch(r"[0-9a-f]{32}", expected_md5_hex) is None:
        raise ProductPartsError("expected MD5 must be 32 lowercase hexadecimal characters")
    parts = split_parts(parts_directory, stem)
    expected_bytes = sum(path.stat().st_size for path in parts)
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _md5(destination)
        if actual != expected_md5_hex:
            raise ProductPartsError(f"existing destination MD5 mismatch: {destination}")
        return destination.stat().st_size
    have = partial.stat().st_size if partial.exists() else 0
    if have > expected_bytes:
        raise ProductPartsError("assembled partial is larger than supplied volumes")
    final_bytes = _copy_from(parts, partial, have)
    if final_bytes != expected_bytes:
        raise ProductPartsError("assembled partial size does not match supplied volumes")
    actual = _md5(partial)
    if actual != expected_md5_hex:
        raise ProductPartsError("assembled partial MD5 does not match the official manifest")
    os.replace(partial, destination)
    return final_bytes
