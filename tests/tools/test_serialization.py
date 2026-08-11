from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from skillchain.tools.serialization import (
    ArtifactFormatError,
    artifact_descriptor,
    canonical_json_bytes,
    parse_canonical_json,
    parse_canonical_jsonl,
    parse_strict_json,
    read_stable_regular_file,
)


def test_canonical_json_is_sorted_utf8_and_rejects_non_json_values():
    assert canonical_json_bytes({"z": 1, "中文": [True, None]}) == (
        '{"z":1,"中文":[true,null]}\n'.encode()
    )

    with pytest.raises(ArtifactFormatError, match="finite"):
        canonical_json_bytes({"score": math.nan})
    with pytest.raises(ArtifactFormatError, match="unsupported"):
        canonical_json_bytes({"bad": object()})


def test_canonical_json_rejects_cycles_and_excessive_depth():
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ArtifactFormatError, match="cycles"):
        canonical_json_bytes(cycle)

    value: object = None
    for _ in range(34):
        value = [value]
    with pytest.raises(ArtifactFormatError, match="nesting"):
        canonical_json_bytes(value)


def test_parser_rejects_duplicates_noncanonical_and_blank_jsonl():
    with pytest.raises(ArtifactFormatError, match="duplicate key"):
        parse_canonical_json(b'{"a":1,"a":2}\n', label="value")
    with pytest.raises(ArtifactFormatError, match="canonical"):
        parse_canonical_json(b'{"b": 2, "a": 1}\n', label="value")
    with pytest.raises(ArtifactFormatError, match="blank line"):
        parse_canonical_jsonl(b'{"a":1}\n\n', label="rows")
    with pytest.raises(ArtifactFormatError, match="newline"):
        parse_canonical_jsonl(b'{"a":1}', label="rows")


def test_strict_provider_json_accepts_formatting_but_not_unsafe_values():
    assert parse_strict_json(
        b'{\n  "b": 2,\n  "a": 1\n}',
        label="provider arguments",
    ) == {"a": 1, "b": 2}
    with pytest.raises(ArtifactFormatError, match="duplicate key"):
        parse_strict_json(
            b'{"a":1,"a":2}',
            label="provider arguments",
        )
    with pytest.raises(ArtifactFormatError, match="non-finite"):
        parse_strict_json(
            b'{"score":NaN}',
            label="provider arguments",
        )


def test_regular_file_reader_and_descriptor(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"payload")

    assert read_stable_regular_file(artifact, label="artifact") == b"payload"
    assert artifact_descriptor(artifact) == {
        "path": "artifact.json",
        "bytes": 7,
        "sha256": "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
    }
    with pytest.raises(ArtifactFormatError, match="exceeds"):
        read_stable_regular_file(artifact, label="artifact", max_bytes=6)
    with pytest.raises(ArtifactFormatError, match="regular"):
        read_stable_regular_file(tmp_path, label="directory")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unavailable")
def test_regular_file_reader_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.write_text("payload", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    with pytest.raises(ArtifactFormatError, match="non-symlink"):
        read_stable_regular_file(link, label="link")


def test_parse_roundtrip_preserves_exact_canonical_value():
    value = {"rows": [{"id": "一"}, {"id": "二"}]}
    content = canonical_json_bytes(value)
    assert parse_canonical_json(content, label="value") == value
    assert json.loads(content) == value
