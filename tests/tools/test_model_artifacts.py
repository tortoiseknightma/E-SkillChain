import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillchain.tools.model_artifacts import (
    ArtifactDescriptor,
    ModelArtifactError,
    canonical_json_bytes,
    load_model_artifact_manifest,
    manifest_digest,
    publish_model_artifact_manifest,
    read_regular_file_snapshot,
    verify_file_snapshot,
)


def test_artifact_descriptor_rejects_unsafe_or_ambiguous_paths():
    for path in ("../weights.bin", "weights.bin:stream", "dir\\weights.bin", "a//b"):
        with pytest.raises(ValidationError):
            ArtifactDescriptor(
                role="weights",
                path=path,
                bytes=1,
                sha256="0" * 64,
            )


def _detector_artifacts(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "weights": root / "detector.bin",
        "labels": root / "labels.json",
        "config": root / "config.json",
    }
    values["weights"].write_bytes(b"pinned-detector-weights")
    values["labels"].write_bytes(b'{"classes":[]}\n')
    values["config"].write_bytes(b'{"device":"cpu"}\n')
    return values


def _publish_detector(root: Path):
    artifacts = _detector_artifacts(root)
    verified = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="object_detector",
        model_id="detector-test-v1",
        backend_name="fake-detector",
        backend_version="1.0.0",
        artifacts=artifacts,
    )
    return verified, artifacts


def test_publish_and_load_model_manifest_is_canonical_create_only_and_bound(tmp_path):
    root = tmp_path / "model"
    verified, artifacts = _publish_detector(root)

    content = (root / "manifest.json").read_bytes()
    assert content == canonical_json_bytes(verified.manifest)
    assert b"detector-test-v1" in content
    assert str(tmp_path).encode() not in content
    assert [item.role for item in verified.manifest.artifacts] == [
        "config",
        "labels",
        "weights",
    ]
    by_role = {item.role: item for item in verified.manifest.artifacts}
    assert by_role["weights"].bytes == len(artifacts["weights"].read_bytes())
    assert by_role["weights"].sha256 != by_role["labels"].sha256
    assert verified.runtime_binding.manifest_sha256 == verified.manifest.manifest_sha256
    assert verified.external_sha256_verified is False

    loaded = load_model_artifact_manifest(
        root / "manifest.json",
        expected_kind="object_detector",
        expected_manifest_sha256=verified.manifest.manifest_sha256,
    )
    assert loaded.manifest == verified.manifest
    assert loaded.external_sha256_verified is True
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        publish_model_artifact_manifest(
            root / "manifest.json",
            artifact_kind="object_detector",
            model_id="other",
            backend_name="fake-detector",
            backend_version="1.0.0",
            artifacts=artifacts,
        )


def test_manifest_binds_model_label_and_config_bytes(tmp_path):
    root = tmp_path / "model"
    verified, artifacts = _publish_detector(root)
    descriptors = {item.role: item for item in verified.manifest.artifacts}

    assert descriptors["weights"].sha256
    assert descriptors["labels"].sha256
    assert descriptors["config"].sha256
    artifacts["labels"].write_bytes(b"changed")

    with pytest.raises(ModelArtifactError, match="does not match"):
        verified.verify_files()
    with pytest.raises(ModelArtifactError, match="does not match"):
        load_model_artifact_manifest(
            root / "manifest.json",
            expected_manifest_sha256=verified.manifest.manifest_sha256,
        )


def test_document_ocr_manifest_requires_model_labels_and_config(tmp_path):
    root = tmp_path / "ocr"
    root.mkdir()
    files = {}
    for role in ("recognizer_model", "labels", "config"):
        files[role] = root / f"{role}.bin"
        files[role].write_bytes(role.encode())

    verified = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="document_ocr",
        model_id="ocr-test-v1",
        backend_name="fake-ocr",
        backend_version="1.0.0",
        artifacts=files,
    )
    assert verified.manifest.artifact_kind == "document_ocr"

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    incomplete_labels = incomplete / "labels.bin"
    incomplete_config = incomplete / "config.bin"
    incomplete_labels.write_bytes(b"labels")
    incomplete_config.write_bytes(b"config")
    with pytest.raises(ValueError, match="requires model, labels, and config"):
        publish_model_artifact_manifest(
            incomplete / "manifest.json",
            artifact_kind="document_ocr",
            model_id="ocr-test-v1",
            backend_name="fake-ocr",
            backend_version="1.0.0",
            artifacts={
                "labels": incomplete_labels,
                "config": incomplete_config,
            },
        )


def test_multimodal_embedding_manifest_requires_locked_local_runtime_files(tmp_path):
    root = tmp_path / "embedding"
    root.mkdir()
    files = {role: root / f"{role}.bin" for role in ("weights", "config", "tokenizer")}
    for role, path in files.items():
        path.write_bytes(role.encode())

    verified = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="multimodal_embedding",
        model_id="open-clip:RN50:openai",
        backend_name="open-clip-torch",
        backend_version="3.2.0",
        artifacts=files,
    )
    assert verified.manifest.artifact_kind == "multimodal_embedding"

    missing = tmp_path / "embedding-missing"
    missing.mkdir()
    weights = missing / "weights.bin"
    config = missing / "config.json"
    weights.write_bytes(b"weights")
    config.write_bytes(b"config")
    with pytest.raises(
        ValueError,
        match="requires weights, config, and tokenizer",
    ):
        publish_model_artifact_manifest(
            missing / "manifest.json",
            artifact_kind="multimodal_embedding",
            model_id="open-clip:RN50:openai",
            backend_name="open-clip-torch",
            backend_version="3.2.0",
            artifacts={"weights": weights, "config": config},
        )


def test_manifest_rejects_noncanonical_duplicate_and_tampered_json(tmp_path):
    root = tmp_path / "model"
    verified, _ = _publish_detector(root)
    original = verified.manifest.model_dump(mode="json")

    (root / "manifest.json").unlink()
    (root / "manifest.json").write_text(
        json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(ModelArtifactError, match="canonical JSON"):
        load_model_artifact_manifest(
            root / "manifest.json",
            expected_manifest_sha256=verified.manifest.manifest_sha256,
        )

    (root / "manifest.json").write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(ModelArtifactError, match="duplicate key"):
        load_model_artifact_manifest(
            root / "manifest.json",
            expected_manifest_sha256=verified.manifest.manifest_sha256,
        )

    tampered = dict(original)
    tampered["model_id"] = "tampered"
    (root / "manifest.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ModelArtifactError, match="self hash mismatch"):
        load_model_artifact_manifest(
            root / "manifest.json",
            expected_manifest_sha256=verified.manifest.manifest_sha256,
        )


def test_artifacts_outside_root_and_nonregular_files_are_rejected(tmp_path):
    root = tmp_path / "model"
    files = _detector_artifacts(root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    files["weights"] = outside
    with pytest.raises(ModelArtifactError, match="below the manifest directory"):
        publish_model_artifact_manifest(
            root / "manifest.json",
            artifact_kind="object_detector",
            model_id="detector",
            backend_name="fake",
            backend_version="1",
            artifacts=files,
        )

    files["weights"] = root
    with pytest.raises(ModelArtifactError, match="regular file"):
        publish_model_artifact_manifest(
            root / "manifest.json",
            artifact_kind="object_detector",
            model_id="detector",
            backend_name="fake",
            backend_version="1",
            artifacts=files,
        )


def test_artifact_and_manifest_symlinks_are_rejected(tmp_path):
    root = tmp_path / "model"
    files = _detector_artifacts(root)
    link = root / "weights-link.bin"
    try:
        link.symlink_to(files["weights"])
    except OSError:
        pytest.skip("symlink creation is not available on this Windows host")
    files["weights"] = link
    with pytest.raises(ModelArtifactError, match="symbolic link"):
        publish_model_artifact_manifest(
            root / "manifest.json",
            artifact_kind="object_detector",
            model_id="detector",
            backend_name="fake",
            backend_version="1",
            artifacts=files,
        )

    verified, _ = _publish_detector(tmp_path / "other")
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(verified.manifest_path)
    with pytest.raises(ModelArtifactError, match="symbolic link"):
        load_model_artifact_manifest(
            manifest_link,
            expected_manifest_sha256=verified.manifest.manifest_sha256,
        )


def test_formal_model_load_requires_and_matches_external_manifest_lock(tmp_path):
    verified, _ = _publish_detector(tmp_path / "model")
    with pytest.raises(ModelArtifactError, match="external expected"):
        load_model_artifact_manifest(verified.manifest_path)
    with pytest.raises(ModelArtifactError, match="external expected lock"):
        load_model_artifact_manifest(
            verified.manifest_path,
            expected_manifest_sha256="0" * 64,
        )


def test_external_lock_rejects_coordinated_model_manifest_rewrite(tmp_path):
    verified, _ = _publish_detector(tmp_path / "model")
    original_lock = verified.manifest.manifest_sha256
    rewritten = verified.manifest.model_dump(mode="json")
    rewritten["model_id"] = "attacker-replacement"
    rewritten["manifest_sha256"] = manifest_digest(rewritten)
    verified.manifest_path.write_bytes(canonical_json_bytes(rewritten))

    with pytest.raises(ModelArtifactError, match="external expected lock"):
        load_model_artifact_manifest(
            verified.manifest_path,
            expected_manifest_sha256=original_lock,
        )


def test_regular_file_snapshot_detects_in_place_and_atomic_replacement(tmp_path):
    path = tmp_path / "image.bin"
    path.write_bytes(b"original")
    snapshot = read_regular_file_snapshot(path, "test input")
    verify_file_snapshot(snapshot, "test input")

    path.write_bytes(b"modified")
    with pytest.raises(ModelArtifactError, match="changed"):
        verify_file_snapshot(snapshot, "test input")

    path.write_bytes(b"original")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"original")
    os.replace(replacement, path)
    with pytest.raises(ModelArtifactError, match="changed"):
        verify_file_snapshot(snapshot, "test input")
