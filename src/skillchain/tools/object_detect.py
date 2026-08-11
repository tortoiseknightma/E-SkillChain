"""Auditable local object detection with injectable and lazy backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib import metadata
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, Protocol, Self
from weakref import WeakKeyDictionary

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.tools.model_artifacts import (
    ModelArtifactError,
    ModelRuntimeBinding,
    VerifiedModelArtifact,
    canonical_json_bytes,
    read_regular_file_snapshot,
    verify_file_snapshot,
)

DETECTOR_CONTRACT_VERSION = 1
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 40_000_000
_MODEL_METADATA_BYTES = 2 * 1024 * 1024

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ObjectDetectionError(ValueError):
    """Base error for invalid detector inputs, runtime artifacts, or outputs."""


class DetectorRuntimeUnavailableError(ObjectDetectionError):
    """Raised when the optional local detector runtime is not installed."""


class _LoadedDetectorModelAuthority:
    """Hold loader-issued model receipts outside writable backend state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: WeakKeyDictionary[object, tuple[object, object]] = (
            WeakKeyDictionary()
        )

    def loaded_model(self, backend: object, artifact: object) -> object | None:
        with self._lock:
            receipt = self._records.get(backend)
        if receipt is None:
            return None
        model, issued_artifact = receipt
        if issued_artifact is not artifact:
            raise ObjectDetectionError(
                "loaded detector model does not belong to the current artifact"
            )
        return model

    def issue(self, backend: object, artifact: object, model: object) -> None:
        if model is None:
            raise ObjectDetectionError("detector loader returned no model")
        with self._lock:
            if backend in self._records:
                raise ObjectDetectionError("detector model receipt was already issued")
            self._records[backend] = (model, artifact)


_LOADED_DETECTOR_MODELS = _LoadedDetectorModelAuthority()


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DetectorClass(_StrictFrozenModel):
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    label_zh: str = Field(min_length=1)

    @field_validator("label", "label_zh")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("detector labels must not have surrounding whitespace")
        return value


class DetectorLabels(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    classes: tuple[DetectorClass, ...]

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        if not self.classes:
            raise ValueError("detector class mapping must not be empty")
        if tuple(sorted(self.classes, key=lambda item: item.class_id)) != self.classes:
            raise ValueError("detector classes must be ordered by class_id")
        class_ids = [item.class_id for item in self.classes]
        labels = [item.label for item in self.classes]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("detector class_id values must be unique")
        if len(labels) != len({label.casefold() for label in labels}):
            raise ValueError("detector labels must be unique")
        return self


class DetectorConfig(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    image_size: int = Field(ge=32)
    confidence_threshold: FiniteFloat = Field(ge=0.0, le=1.0)
    iou_threshold: FiniteFloat = Field(ge=0.0, le=1.0)
    max_detections: int = Field(ge=1, le=1000)
    device: Literal["cpu"] = "cpu"
    verbose: Literal[False] = False
    coordinate_decimals: Literal[6] = 6


class RawDetection(_StrictFrozenModel):
    """Backend-neutral detection before clipping and stable normalization."""

    class_id: int = Field(ge=0)
    bbox_xyxy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)


class DetectionInputBinding(_StrictFrozenModel):
    asset_id: str = Field(min_length=1)
    image_sha256: Sha256
    image_bytes: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("asset_id must not have surrounding whitespace")
        return value


class DetectedObject(_StrictFrozenModel):
    detection_id: Sha256
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    label_zh: str = Field(min_length=1)
    bbox_xyxy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @field_validator("label", "label_zh")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("detection labels must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_positive_box(self) -> Self:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("detection bbox must have positive area inside the image")
        return self


class ObjectDetectionResult(_StrictFrozenModel):
    schema_version: Literal[1] = DETECTOR_CONTRACT_VERSION
    input_binding: DetectionInputBinding
    runtime_binding: ModelRuntimeBinding
    detections: tuple[DetectedObject, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.runtime_binding.artifact_kind != "object_detector":
            raise ValueError("object detection result requires a detector runtime")
        ids = [item.detection_id for item in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("detection_id values must be unique")
        for item in self.detections:
            _, _, x2, y2 = item.bbox_xyxy
            if x2 > self.input_binding.width or y2 > self.input_binding.height:
                raise ValueError("detection bbox exceeds image bounds")
        expected = tuple(sorted(self.detections, key=_detection_sort_key))
        if expected != self.detections:
            raise ValueError("detections must use the canonical stable order")
        return self


class DetectorBackend(Protocol):
    """Minimal protocol used by fake backends and optional real runtimes."""

    def detect(
        self,
        image: Image.Image,
        *,
        config: DetectorConfig,
    ) -> Sequence[RawDetection]: ...


class UltralyticsDetectorBackend:
    """Lazy optional Ultralytics runtime; constructing it performs no import."""

    def __init__(self, artifact: VerifiedModelArtifact) -> None:
        if not artifact.external_sha256_verified:
            raise ObjectDetectionError(
                "detector runtime requires an externally locked model manifest"
            )
        if artifact.manifest.artifact_kind != "object_detector":
            raise ObjectDetectionError(
                "Ultralytics backend requires detector artifacts"
            )
        if artifact.manifest.backend_name != "ultralytics":
            raise ObjectDetectionError(
                "Ultralytics backend requires an ultralytics model manifest"
            )
        self._artifact = artifact
        self._lock = Lock()

    @property
    def _model(self) -> Any | None:
        """Expose loader state read-only for diagnostics and legacy observability."""

        if "_model" in vars(self):
            raise ObjectDetectionError(
                "detector model state was injected outside the verified loader"
            )
        return _LOADED_DETECTOR_MODELS.loaded_model(self, self._artifact)

    @property
    def formal_runtime_binding_sha256(self) -> str:
        self._artifact.verify_files()
        # Resolve the authority-held receipt, or prove the lazy backend is pristine.
        self._model
        payload = {
            "artifact_runtime": self._artifact.runtime_binding.model_dump(mode="json"),
            "backend_policy_version": "ultralytics-detector-backend-v2",
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def detect(
        self,
        image: Image.Image,
        *,
        config: DetectorConfig,
    ) -> Sequence[RawDetection]:
        model = self._load_model()
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - NumPy is a base dependency.
            raise DetectorRuntimeUnavailableError(
                "NumPy is required for detection"
            ) from exc

        results = model.predict(
            source=np.asarray(image, dtype=np.uint8),
            imgsz=config.image_size,
            conf=float(config.confidence_threshold),
            iou=float(config.iou_threshold),
            max_det=config.max_detections,
            device=config.device,
            verbose=config.verbose,
        )
        if len(results) != 1:
            raise ObjectDetectionError("detector runtime returned an unexpected batch")
        boxes = results[0].boxes
        if boxes is None:
            return ()
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        if not (len(coordinates) == len(confidences) == len(classes)):
            raise ObjectDetectionError("detector runtime returned misaligned arrays")
        return tuple(
            RawDetection(
                class_id=int(class_id),
                bbox_xyxy=tuple(float(value) for value in bbox),
                confidence=float(confidence),
            )
            for bbox, confidence, class_id in zip(
                coordinates, confidences, classes, strict=True
            )
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise DetectorRuntimeUnavailableError(
                    "optional Ultralytics runtime is not installed"
                ) from exc
            try:
                installed_version = metadata.version("ultralytics")
            except metadata.PackageNotFoundError as exc:
                raise DetectorRuntimeUnavailableError(
                    "Ultralytics package metadata is unavailable"
                ) from exc
            if installed_version != self._artifact.manifest.backend_version:
                raise ObjectDetectionError(
                    "installed Ultralytics version does not match the model manifest"
                )
            self._artifact.verify_files()
            model = YOLO(str(self._artifact.artifact_path("weights")))
            self._artifact.verify_files()
            _LOADED_DETECTOR_MODELS.issue(self, self._artifact, model)
            return model


class ObjectDetectionService:
    """Validate one image and normalize detector output into an auditable result."""

    def __init__(
        self,
        artifact: VerifiedModelArtifact,
        backend: DetectorBackend,
    ) -> None:
        if not artifact.external_sha256_verified:
            raise ObjectDetectionError(
                "object detector service requires an externally locked model manifest"
            )
        if artifact.manifest.artifact_kind != "object_detector":
            raise ObjectDetectionError(
                "object detector service requires detector artifacts"
            )
        self.artifact = artifact
        self.backend = backend
        self.labels = _load_canonical_model(
            artifact.read_artifact_bytes("labels", max_bytes=_MODEL_METADATA_BYTES),
            DetectorLabels,
            "detector labels",
        )
        self.config = _load_canonical_model(
            artifact.read_artifact_bytes("config", max_bytes=_MODEL_METADATA_BYTES),
            DetectorConfig,
            "detector config",
        )
        self._labels_by_id = {item.class_id: item for item in self.labels.classes}

    @property
    def formal_runtime_binding_sha256(self) -> str:
        self.artifact.verify_files()
        binding = getattr(self.backend, "formal_runtime_binding_sha256", None)
        if (
            not isinstance(binding, str)
            or len(binding) != 64
            or any(character not in "0123456789abcdef" for character in binding)
        ):
            raise ObjectDetectionError(
                "formal detector runtime requires an explicitly bound backend"
            )
        payload = {
            "artifact_runtime": self.artifact.runtime_binding.model_dump(mode="json"),
            "backend_runtime_sha256": binding,
            "service_policy_version": "object-detection-service-v1",
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def detect(self, image_path: str | Path, *, asset_id: str) -> ObjectDetectionResult:
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ObjectDetectionError("asset_id must be a non-blank string")
        if asset_id != asset_id.strip():
            raise ObjectDetectionError("asset_id must not have surrounding whitespace")
        try:
            snapshot = read_regular_file_snapshot(
                image_path,
                "detector input image",
                max_bytes=_MAX_IMAGE_BYTES,
            )
            image = _decode_image(snapshot.content)
            self.artifact.verify_files()
        except ModelArtifactError as exc:
            raise ObjectDetectionError(str(exc)) from exc

        input_binding = DetectionInputBinding(
            asset_id=asset_id,
            image_sha256=snapshot.sha256,
            image_bytes=len(snapshot.content),
            width=image.width,
            height=image.height,
        )
        try:
            raw_results = tuple(self.backend.detect(image.copy(), config=self.config))
        finally:
            _reverify_after_inference(snapshot, self.artifact)

        normalized = self._normalize(
            raw_results,
            input_binding=input_binding,
        )
        return ObjectDetectionResult(
            input_binding=input_binding,
            runtime_binding=self.artifact.runtime_binding,
            detections=normalized,
        )

    def _normalize(
        self,
        values: Sequence[RawDetection],
        *,
        input_binding: DetectionInputBinding,
    ) -> tuple[DetectedObject, ...]:
        candidates: list[dict[str, Any]] = []
        for value in values:
            try:
                raw = RawDetection.model_validate(value)
            except ValueError as exc:
                raise ObjectDetectionError(
                    f"invalid detector backend output: {exc}"
                ) from exc
            if raw.confidence < self.config.confidence_threshold:
                continue
            try:
                mapped = self._labels_by_id[raw.class_id]
            except KeyError:
                raise ObjectDetectionError(
                    f"detector returned unmapped class_id {raw.class_id}"
                ) from None
            bbox = _clip_and_round_bbox(
                raw.bbox_xyxy,
                width=input_binding.width,
                height=input_binding.height,
                decimals=self.config.coordinate_decimals,
            )
            confidence = round(float(raw.confidence), self.config.coordinate_decimals)
            candidates.append(
                {
                    "class_id": raw.class_id,
                    "label": mapped.label,
                    "label_zh": mapped.label_zh,
                    "bbox_xyxy": bbox,
                    "confidence": confidence,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["confidence"],
                item["class_id"],
                *item["bbox_xyxy"],
                item["label"],
            )
        )
        candidates = candidates[: self.config.max_detections]
        detections = []
        for ordinal, candidate in enumerate(candidates, start=1):
            identity = {
                "contract_version": DETECTOR_CONTRACT_VERSION,
                "input_sha256": input_binding.image_sha256,
                "runtime_sha256": self.artifact.manifest.manifest_sha256,
                "ordinal": ordinal,
                **candidate,
            }
            detections.append(
                DetectedObject(
                    detection_id=hashlib.sha256(
                        canonical_json_bytes(identity)
                    ).hexdigest(),
                    **candidate,
                )
            )
        return tuple(detections)


def object_detect(
    image_path: str | Path,
    *,
    asset_id: str,
    service: ObjectDetectionService,
) -> ObjectDetectionResult:
    """Explicit dependency-injected convenience entry point."""

    return service.detect(image_path, asset_id=asset_id)


def _load_canonical_model(content: bytes, model_type, label: str):
    try:
        raw = json.loads(
            content.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
        value = model_type.model_validate_json(canonical_json_bytes(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ObjectDetectionError(f"{label} is invalid: {exc}") from exc
    if content != canonical_json_bytes(value):
        raise ObjectDetectionError(f"{label} must use canonical JSON")
    return value


def _decode_image(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.width * opened.height > _MAX_IMAGE_PIXELS:
                raise ObjectDetectionError(
                    "detector input image exceeds the pixel limit"
                )
            if getattr(opened, "n_frames", 1) != 1:
                raise ObjectDetectionError("detector input image must have one frame")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except ObjectDetectionError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ObjectDetectionError("detector input image cannot be decoded") from exc
    return image


def _clip_and_round_bbox(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    decimals: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    clipped = (
        round(min(max(x1, 0.0), float(width)), decimals),
        round(min(max(y1, 0.0), float(height)), decimals),
        round(min(max(x2, 0.0), float(width)), decimals),
        round(min(max(y2, 0.0), float(height)), decimals),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ObjectDetectionError("detector bbox has no positive in-bounds area")
    return clipped


def _detection_sort_key(item: DetectedObject) -> tuple[Any, ...]:
    return (-item.confidence, item.class_id, *item.bbox_xyxy, item.label)


def _reverify_after_inference(snapshot, artifact: VerifiedModelArtifact) -> None:
    input_error: Exception | None = None
    model_error: Exception | None = None
    try:
        verify_file_snapshot(snapshot, "detector input image")
    except Exception as exc:  # Preserve both checks even if the first fails.
        input_error = exc
    try:
        artifact.verify_files()
    except Exception as exc:
        model_error = exc
    if input_error is not None:
        raise ObjectDetectionError(str(input_error)) from input_error
    if model_error is not None:
        raise ObjectDetectionError(str(model_error)) from model_error


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ObjectDetectionError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


__all__ = [
    "DETECTOR_CONTRACT_VERSION",
    "DetectedObject",
    "DetectionInputBinding",
    "DetectorBackend",
    "DetectorClass",
    "DetectorConfig",
    "DetectorLabels",
    "DetectorRuntimeUnavailableError",
    "ObjectDetectionError",
    "ObjectDetectionResult",
    "ObjectDetectionService",
    "RawDetection",
    "UltralyticsDetectorBackend",
    "object_detect",
]
