from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATHS = (
    ROOT / "docs" / "reproduction-contract.md",
    ROOT / "docs" / "evaluation-protocol.md",
)
PROTOCOL_HASH_LINE = re.compile(
    r"^> \*\*规范化内容 SHA-256：\*\* `(?P<digest>[0-9a-f]{64})`$",
    re.MULTILINE,
)


def _declared_and_computed_hash(content: bytes) -> tuple[str, str]:
    text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    matches = tuple(PROTOCOL_HASH_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError("protocol must contain exactly one hash metadata line")

    match = matches[0]
    first_section = text.find("\n## ")
    if first_section != -1 and match.start() > first_section:
        raise ValueError("protocol hash metadata must precede the first section")

    line_end = match.end()
    if line_end < len(text) and text[line_end] == "\n":
        line_end += 1
    normalized = text[: match.start()] + text[line_end:]
    computed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return match.group("digest"), computed


@pytest.mark.parametrize("path", PROTOCOL_PATHS, ids=lambda path: path.name)
def test_protocol_declares_exact_normalized_content_hash(path: Path) -> None:
    declared, computed = _declared_and_computed_hash(path.read_bytes())
    assert declared == computed


def test_protocol_hash_rejects_duplicate_metadata() -> None:
    metadata = f"> **规范化内容 SHA-256：** `{'0' * 64}`"
    content = f"# Protocol\n\n{metadata}\n{metadata}\n\n## Section\nBody\n"

    with pytest.raises(ValueError, match="exactly one"):
        _declared_and_computed_hash(content.encode("utf-8"))


def test_protocol_hash_rejects_missing_metadata() -> None:
    content = "# Protocol\n\n## Section\n正文提及 `规范化内容 SHA-256` 也不是元数据。\n"

    with pytest.raises(ValueError, match="exactly one"):
        _declared_and_computed_hash(content.encode("utf-8"))


def test_protocol_hash_rejects_metadata_outside_header() -> None:
    metadata = f"> **规范化内容 SHA-256：** `{'0' * 64}`"
    content = f"# Protocol\n\n## Section\n{metadata}\n"

    with pytest.raises(ValueError, match="precede the first section"):
        _declared_and_computed_hash(content.encode("utf-8"))
