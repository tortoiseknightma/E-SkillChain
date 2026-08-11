from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from skillchain.tools.model_artifacts import (
    canonical_json_bytes,
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.object_detect import (
    DetectedObject,
    DetectorClass,
    DetectorConfig,
    DetectorLabels,
    ObjectDetectionError,
    ObjectDetectionResult,
    ObjectDetectionService,
    RawDetection,
    UltralyticsDetectorBackend,
)


class FakeDetector:
    def __init__(self, values=(), callback=None):
        self.values = tuple(values)
        self.callback = callback
        self.calls = []

    def detect(self, image, *, config):
        self.calls.append((image.size, config))
        if self.callback is not None:
            self.callback()
        return self.values


def _write_image(path: Path, size=(100, 80)) -> Path:
    Image.new("RGB", size, (20, 40, 60)).save(path, format="PNG")
    return path


def _detector_service(
    tmp_path: Path,
    backend: FakeDetector,
    *,
    backend_name: str = "fake-detector",
):
    root = tmp_path / "detector"
    root.mkdir()
    labels = DetectorLabels(
        classes=(
            DetectorClass(class_id=0, label="person", label_zh="人"),
            DetectorClass(class_id=1, label="bag", label_zh="包"),
        )
    )
    config = DetectorConfig(
        image_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        max_detections=100,
    )
    files = {
        "weights": root / "weights.bin",
        "labels": root / "labels.json",
        "config": root / "config.json",
    }
    files["weights"].write_bytes(b"fixed-detector-weights")
    files["labels"].write_bytes(canonical_json_bytes(labels))
    files["config"].write_bytes(canonical_json_bytes(config))
    published = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="object_detector",
        model_id="fake-detector-v1",
        backend_name=backend_name,
        backend_version="1.0.0",
        artifacts=files,
    )
    artifact = load_model_artifact_manifest(
        published.manifest_path,
        expected_manifest_sha256=published.manifest.manifest_sha256,
    )
    return ObjectDetectionService(artifact, backend), artifact, files


def test_detection_clips_boxes_sorts_stably_and_binds_runtime_and_input(tmp_path):
    backend = FakeDetector(
        [
            RawDetection(
                class_id=1,
                bbox_xyxy=(40.0, 10.0, 120.0, 70.0),
                confidence=0.8,
            ),
            RawDetection(
                class_id=0,
                bbox_xyxy=(-2.0, -1.0, 20.1234567, 30.7654321),
                confidence=0.9,
            ),
            RawDetection(
                class_id=0,
                bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
                confidence=0.2,
            ),
        ]
    )
    service, artifact, _ = _detector_service(tmp_path, backend)
    image = _write_image(tmp_path / "query.png")

    first = service.detect(image, asset_id="asset-query")
    second = service.detect(image, asset_id="asset-query")

    assert first == second
    assert first.input_binding.asset_id == "asset-query"
    assert first.input_binding.width == 100
    assert first.input_binding.height == 80
    assert first.runtime_binding.manifest_sha256 == artifact.manifest.manifest_sha256
    assert [item.class_id for item in first.detections] == [0, 1]
    assert first.detections[0].bbox_xyxy == (0.0, 0.0, 20.123457, 30.765432)
    assert first.detections[1].bbox_xyxy == (40.0, 10.0, 100.0, 70.0)
    assert len({item.detection_id for item in first.detections}) == 2
    assert len(backend.calls) == 2
    assert ObjectDetectionResult.model_validate_json(first.model_dump_json()) == first


def test_detector_service_rejects_artifact_without_external_lock(tmp_path):
    _, artifact, _ = _detector_service(tmp_path, FakeDetector())
    forged = replace(artifact, external_sha256_verified=False)
    with pytest.raises(ObjectDetectionError, match="externally locked"):
        ObjectDetectionService(forged, FakeDetector())


def test_empty_detection_still_carries_input_and_runtime_bindings(tmp_path):
    service, artifact, _ = _detector_service(tmp_path, FakeDetector())
    image = _write_image(tmp_path / "query.png")

    result = service.detect(image, asset_id="asset-empty")

    assert result.detections == ()
    assert result.input_binding.image_sha256
    assert result.runtime_binding.manifest_sha256 == artifact.manifest.manifest_sha256
    assert {item.role for item in result.runtime_binding.artifacts} == {
        "weights",
        "labels",
        "config",
    }


@pytest.mark.parametrize(
    "bbox",
    [
        (1.0, 1.0, 1.0, 5.0),
        (1.0, 1.0, 5.0, 1.0),
        (-10.0, -10.0, -1.0, -1.0),
        (101.0, 2.0, 120.0, 4.0),
    ],
)
def test_detection_rejects_boxes_without_positive_in_bounds_area(tmp_path, bbox):
    service, _, _ = _detector_service(
        tmp_path,
        FakeDetector([RawDetection(class_id=0, bbox_xyxy=bbox, confidence=0.9)]),
    )
    image = _write_image(tmp_path / "query.png")

    with pytest.raises(ObjectDetectionError, match="positive in-bounds area"):
        service.detect(image, asset_id="asset-query")


def test_detection_rejects_unmapped_classes(tmp_path):
    service, _, _ = _detector_service(
        tmp_path,
        FakeDetector(
            [
                RawDetection(
                    class_id=99,
                    bbox_xyxy=(1.0, 1.0, 5.0, 5.0),
                    confidence=0.9,
                )
            ]
        ),
    )
    image = _write_image(tmp_path / "query.png")

    with pytest.raises(ObjectDetectionError, match="unmapped class_id"):
        service.detect(image, asset_id="asset-query")


def test_detection_rechecks_input_bytes_after_backend_execution(tmp_path):
    image = _write_image(tmp_path / "query.png")
    backend = FakeDetector(callback=lambda: image.write_bytes(b"changed-during-call"))
    service, _, _ = _detector_service(tmp_path, backend)

    with pytest.raises(ObjectDetectionError, match="changed"):
        service.detect(image, asset_id="asset-query")


def test_detection_rechecks_model_artifacts_after_backend_execution(tmp_path):
    holder = {}
    backend = FakeDetector(callback=lambda: holder["weights"].write_bytes(b"tampered"))
    service, _, files = _detector_service(tmp_path, backend)
    holder["weights"] = files["weights"]
    image = _write_image(tmp_path / "query.png")

    with pytest.raises(ObjectDetectionError, match="does not match"):
        service.detect(image, asset_id="asset-query")


def test_detection_rejects_symlink_and_non_image_inputs(tmp_path):
    service, _, _ = _detector_service(tmp_path, FakeDetector())
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not an image", encoding="utf-8")
    with pytest.raises(ObjectDetectionError, match="cannot be decoded"):
        service.detect(invalid, asset_id="asset-query")

    target = _write_image(tmp_path / "target.png")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available on this Windows host")
    with pytest.raises(ObjectDetectionError, match="symbolic link"):
        service.detect(link, asset_id="asset-query")


def test_detector_contracts_are_strict_and_positive_area():
    with pytest.raises(ValidationError):
        DetectorConfig.model_validate(
            {
                "image_size": "640",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "max_detections": 100,
            }
        )
    with pytest.raises(ValidationError):
        DetectedObject(
            detection_id="1" * 64,
            class_id=0,
            label="person",
            label_zh="人",
            bbox_xyxy=(0.0, 0.0, 0.0, 1.0),
            confidence=0.9,
        )
    with pytest.raises(ValidationError):
        RawDetection.model_validate(
            {
                "class_id": 0,
                "bbox_xyxy": (0.0, 0.0, 1.0, 1.0),
                "confidence": 0.9,
                "unexpected": True,
            }
        )


def test_ultralytics_backend_construction_is_lazy(tmp_path, monkeypatch):
    service, artifact, _ = _detector_service(
        tmp_path, FakeDetector(), backend_name="ultralytics"
    )
    imported = []
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics":
            imported.append(name)
            raise AssertionError("Ultralytics must not be imported during construction")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    backend = UltralyticsDetectorBackend(artifact)

    assert backend._model is None
    assert len(backend.formal_runtime_binding_sha256) == 64
    assert imported == []
    assert service.artifact is artifact


def test_ultralytics_backend_rejects_private_model_injection(tmp_path):
    _, artifact, _ = _detector_service(
        tmp_path, FakeDetector(), backend_name="ultralytics"
    )
    backend = UltralyticsDetectorBackend(artifact)

    with pytest.raises(AttributeError):
        backend._model = object()

    backend.__dict__["_model"] = object()
    with pytest.raises(ObjectDetectionError, match="injected outside"):
        _ = backend.formal_runtime_binding_sha256
