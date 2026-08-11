import hashlib

import pytest

from skillchain.data.products10k_parts import (
    ProductPartsError,
    assemble_split_archive,
    expected_md5,
    split_parts,
)


def _write_parts(directory, stem, payload):
    directory.mkdir()
    (directory / f"{stem}.z01").write_bytes(payload[:3])
    (directory / f"{stem}.z02").write_bytes(payload[3:6])
    (directory / f"{stem}.zip").write_bytes(payload[6:])


def test_assemble_split_archive_promotes_only_official_md5_match(tmp_path):
    payload = b"products-ten-k-fixture"
    parts = tmp_path / "parts"
    _write_parts(parts, "train_part", payload)
    manifest = tmp_path / "md5.txt"
    manifest.write_text(
        f"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}  train.zip\n",
        encoding="utf-8",
    )
    destination = tmp_path / "archives" / "train.zip"

    actual = assemble_split_archive(
        parts_directory=parts,
        stem="train_part",
        destination=destination,
        expected_md5_hex=expected_md5(manifest, "train.zip"),
    )

    assert actual == len(payload)
    assert destination.read_bytes() == payload
    assert not destination.with_name("train.zip.part").exists()
    assert all(path.exists() for path in split_parts(parts, "train_part"))


def test_assemble_split_archive_retains_mismatching_partial(tmp_path):
    payload = b"products-ten-k-fixture"
    parts = tmp_path / "parts"
    _write_parts(parts, "train_part", payload)
    destination = tmp_path / "archives" / "train.zip"

    with pytest.raises(ProductPartsError, match="MD5"):
        assemble_split_archive(
            parts_directory=parts,
            stem="train_part",
            destination=destination,
            expected_md5_hex="0" * 32,
        )

    assert destination.with_name("train.zip.part").read_bytes() == payload
    assert not destination.exists()
