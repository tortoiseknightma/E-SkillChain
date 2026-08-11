"""Create and load-check a content-locked RapidOCR model artifact.

The script copies model files from the exact installed rapidocr-onnxruntime
wheel.  The recognition label file is supplied separately because it is legal
and model evidence, not a dependency that the runtime may fetch implicitly.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Sequence

from skillchain.tools.document_ocr import (
    DocumentOCRService,
    RapidOCROnnxBackend,
)
from skillchain.tools.model_artifacts import (
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


_MODEL_FILES = {
    "classifier_model": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "detector_model": "ch_PP-OCRv4_det_infer.onnx",
    "recognizer_model": "ch_PP-OCRv4_rec_infer.onnx",
}
_MODEL_ID = "rapidocr:ch_PP-OCRv4-mobile"
_BACKEND_VERSION = "1.4.4"


def prepare_rapidocr_artifact(
    destination: Path,
    *,
    labels_path: Path,
    expected_labels_sha256: str,
    package_root: Path | None = None,
    verify_runtime: bool = True,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_labels_sha256) is None:
        raise ValueError("expected labels SHA-256 must be lowercase hexadecimal")
    labels = read_stable_regular_file(
        labels_path,
        label="RapidOCR recognition labels",
        max_bytes=4 * 1024 * 1024,
    )
    if sha256_bytes(labels) != expected_labels_sha256:
        raise ValueError("RapidOCR labels do not match the external digest")
    try:
        labels.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("RapidOCR labels must be UTF-8") from error

    if metadata.version("rapidocr-onnxruntime") != _BACKEND_VERSION:
        raise ValueError("installed rapidocr-onnxruntime version is not locked")
    if package_root is None:
        import rapidocr_onnxruntime

        package_root = Path(rapidocr_onnxruntime.__file__).resolve().parent
    package_root = Path(package_root).resolve()
    models_root = package_root / "models"
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"RapidOCR artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        artifacts: dict[str, Path] = {}
        for role, filename in _MODEL_FILES.items():
            source = models_root / filename
            content = read_stable_regular_file(
                source,
                label=f"RapidOCR wheel {role}",
                max_bytes=128 * 1024 * 1024,
            )
            target = staging / filename
            target.write_bytes(content)
            artifacts[role] = target
        labels_target = staging / "ppocr_keys_v1.txt"
        labels_target.write_bytes(labels)
        artifacts["labels"] = labels_target
        config_target = staging / "ocr-config.json"
        config_target.write_bytes(
            canonical_json_bytes(
                {
                    "coordinate_decimals": 6,
                    "device": "cpu",
                    "languages": ["en", "zh"],
                    "max_characters": 200000,
                    "max_lines": 5000,
                    "schema_version": 1,
                    "use_classifier": True,
                }
            )
        )
        artifacts["config"] = config_target
        published = publish_model_artifact_manifest(
            staging / "manifest.json",
            artifact_kind="document_ocr",
            model_id=_MODEL_ID,
            backend_name="rapidocr-onnxruntime",
            backend_version=_BACKEND_VERSION,
            artifacts=artifacts,
        )
        manifest_sha256 = published.manifest.manifest_sha256
        os.replace(staging, destination)
        verified = load_model_artifact_manifest(
            destination / "manifest.json",
            expected_kind="document_ocr",
            expected_manifest_sha256=manifest_sha256,
            verify_files=True,
        )
        runtime_sha256 = None
        if verify_runtime:
            backend = RapidOCROnnxBackend(verified)
            service = DocumentOCRService(verified, backend)
            backend._load_engine(service.config)
            runtime_sha256 = backend.formal_runtime_binding_sha256
        return {
            "artifact_kind": "document_ocr",
            "backend_version": _BACKEND_VERSION,
            "manifest_path": str(destination / "manifest.json"),
            "manifest_sha256": manifest_sha256,
            "model_id": _MODEL_ID,
            "runtime_binding_sha256": runtime_sha256,
            "runtime_loaded": verify_runtime,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        result = prepare_rapidocr_artifact(
            arguments.destination,
            labels_path=arguments.labels,
            expected_labels_sha256=arguments.expected_labels_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"prepare-rapidocr-artifact: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
