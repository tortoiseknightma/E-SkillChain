from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from skillchain.tools.model_artifacts import load_model_artifact_manifest
from skillchain.tools.serialization import sha256_bytes


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_rapidocr_artifact",
    ROOT / "scripts" / "prepare_rapidocr_artifact.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_package(root: Path) -> None:
    models = root / "models"
    models.mkdir(parents=True)
    for filename in MODULE._MODEL_FILES.values():
        (models / filename).write_bytes(f"model:{filename}".encode())


def test_prepare_rapidocr_artifact_is_create_only_and_externally_loadable(
    tmp_path: Path,
    monkeypatch,
):
    package = tmp_path / "package"
    _fake_package(package)
    labels = tmp_path / "labels.txt"
    labels.write_text("a\n中\n", encoding="utf-8", newline="\n")
    labels_sha256 = sha256_bytes(labels.read_bytes())
    monkeypatch.setattr(
        MODULE.metadata,
        "version",
        lambda name: (
            MODULE._BACKEND_VERSION
            if name == "rapidocr-onnxruntime"
            else pytest.fail(name)
        ),
    )
    destination = tmp_path / "ocr"

    result = MODULE.prepare_rapidocr_artifact(
        destination,
        labels_path=labels,
        expected_labels_sha256=labels_sha256,
        package_root=package,
        verify_runtime=False,
    )

    assert result["runtime_loaded"] is False
    verified = load_model_artifact_manifest(
        destination / "manifest.json",
        expected_kind="document_ocr",
        expected_manifest_sha256=result["manifest_sha256"],
        verify_files=True,
    )
    assert verified.manifest.model_id == MODULE._MODEL_ID
    with pytest.raises(FileExistsError):
        MODULE.prepare_rapidocr_artifact(
            destination,
            labels_path=labels,
            expected_labels_sha256=labels_sha256,
            package_root=package,
            verify_runtime=False,
        )


def test_prepare_rapidocr_rejects_self_asserted_or_changed_labels(
    tmp_path: Path,
    monkeypatch,
):
    package = tmp_path / "package"
    _fake_package(package)
    labels = tmp_path / "labels.txt"
    labels.write_text("a\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE.metadata,
        "version",
        lambda _name: MODULE._BACKEND_VERSION,
    )
    with pytest.raises(ValueError, match="external digest"):
        MODULE.prepare_rapidocr_artifact(
            tmp_path / "ocr",
            labels_path=labels,
            expected_labels_sha256="0" * 64,
            package_root=package,
            verify_runtime=False,
        )
