"""Auditable ``detect -> crop -> product search`` composition.

The detector and product index remain independently testable services.  This
module composes them without persisting crops or allowing a caller to replace
the parent image between stages.  Every per-object result binds the detector
identity, exact deterministic crop bytes, query vector, and retrieval index.
"""

from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, Protocol, Self

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

from skillchain.tools.contracts import (
    ProductHit,
    ProductSearchTrace,
    RetrievalArtifactBinding,
)
from skillchain.tools.model_artifacts import (
    ModelArtifactError,
    ModelRuntimeBinding,
    read_regular_file_snapshot,
    verify_file_snapshot,
)
from skillchain.tools.object_detect import (
    DetectedObject,
    DetectionInputBinding,
    ObjectDetectionResult,
)
from skillchain.tools.serialization import (
    ArtifactFormatError,
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)

MULTI_PRODUCT_CONTRACT_VERSION = 1
_MAX_PARENT_BYTES = 20 * 1024 * 1024
_MAX_PARENT_PIXELS = 40_000_000

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class MultiProductError(ValueError):
    """The composition input or transitive tool evidence is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MultiProductObjectResult(_StrictFrozenModel):
    """One canonical detection, deterministic crop, and retrieval trace."""

    detection_id: Sha256
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    label_zh: str = Field(min_length=1)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    crop_box_xyxy: tuple[int, int, int, int]
    crop_mode: Literal["RGB"] = "RGB"
    crop_format: Literal["PNG"] = "PNG"
    crop_policy_version: Literal["floor-ceil-clamp-rgb-png-v1"] = (
        "floor-ceil-clamp-rgb-png-v1"
    )
    crop_sha256: Sha256
    crop_bytes: int = Field(gt=0)
    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)
    query_vector_sha256: Sha256
    artifact_binding: RetrievalArtifactBinding
    hits: tuple[ProductHit, ...]

    @field_validator("label", "label_zh")
    @classmethod
    def validate_labels(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("detection labels must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_crop_and_hits(self) -> Self:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("detection bbox must have positive non-negative area")
        left, top, right, bottom = self.crop_box_xyxy
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("crop box must have positive non-negative area")
        if right - left != self.crop_width or bottom - top != self.crop_height:
            raise ValueError("crop dimensions must match crop_box_xyxy")
        if any(hit.artifact_binding != self.artifact_binding for hit in self.hits):
            raise ValueError("every product hit must use the crop artifact binding")
        if [hit.rank for hit in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("product hit ranks must be contiguous")
        return self


class MultiProductResult(_StrictFrozenModel):
    """Complete evidence for one parent image, including an empty detection."""

    schema_version: Literal[1] = MULTI_PRODUCT_CONTRACT_VERSION
    input_binding: DetectionInputBinding
    detection_runtime_binding: ModelRuntimeBinding
    artifact_binding: RetrievalArtifactBinding
    objects: tuple[MultiProductObjectResult, ...]

    @model_validator(mode="after")
    def validate_bindings_and_order(self) -> Self:
        if self.detection_runtime_binding.artifact_kind != "object_detector":
            raise ValueError("multi-product results require an object detector runtime")
        ids = [item.detection_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("multi-product detection_id values must be unique")
        if any(item.artifact_binding != self.artifact_binding for item in self.objects):
            raise ValueError("all objects must use the result artifact binding")
        for item in self.objects:
            expected_crop = _canonical_crop_box(
                item.bbox_xyxy,
                width=self.input_binding.width,
                height=self.input_binding.height,
            )
            if item.crop_box_xyxy != expected_crop:
                raise ValueError("object crop box does not match its detection bbox")
        expected = tuple(sorted(self.objects, key=_object_sort_key))
        if expected != self.objects:
            raise ValueError("multi-product objects must use canonical detection order")
        return self


class DetectionService(Protocol):
    def detect(
        self,
        image_path: str | Path,
        *,
        asset_id: str,
    ) -> ObjectDetectionResult: ...


class TraceProductSearchService(Protocol):
    artifact_binding: RetrievalArtifactBinding

    def trace_image_product_search(
        self,
        image: str | Path,
    ) -> ProductSearchTrace: ...


class MultiProductSearchService:
    """Compose explicit detector and trace-search dependencies."""

    def __init__(
        self,
        detector: DetectionService,
        product_search: TraceProductSearchService,
    ) -> None:
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide detect()")
        if not callable(getattr(product_search, "trace_image_product_search", None)):
            raise TypeError("product_search must provide trace_image_product_search()")
        binding = getattr(product_search, "artifact_binding", None)
        if not isinstance(binding, RetrievalArtifactBinding):
            raise TypeError("product_search must expose a retrieval artifact binding")
        self.detector = detector
        self.product_search = product_search

    @property
    def execution_location(self) -> Literal["local", "remote"]:
        location = getattr(self.product_search, "execution_location", None)
        if location not in {"local", "remote"}:
            raise MultiProductError("product search execution location is unavailable")
        return location

    @property
    def formal_runtime_binding_sha256(self) -> str:
        """Bind both transitive runtimes and the private crop contract."""

        try:
            detector_binding = self.detector.artifact.runtime_binding
            detector_runtime = self.detector.formal_runtime_binding_sha256
            product_runtime = self.product_search.formal_runtime_binding_sha256
        except (AttributeError, TypeError, ValueError) as error:
            raise MultiProductError(
                "multi-product service is not formal-runtime ready"
            ) from error
        if not isinstance(detector_binding, BaseModel):
            raise MultiProductError(
                "multi-product detector runtime binding is not typed"
            )
        for label, value in (
            ("detector runtime", detector_runtime),
            ("product runtime", product_runtime),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise MultiProductError(f"{label} is not a lowercase sha256")
        payload = {
            "contract_version": MULTI_PRODUCT_CONTRACT_VERSION,
            "crop_policy_version": "floor-ceil-clamp-rgb-png-v1",
            "detector_artifact_runtime": detector_binding.model_dump(mode="json"),
            "detector_runtime_sha256": detector_runtime,
            "policy_version": "authority-multi-product-runtime-v2",
            "product_runtime_sha256": product_runtime,
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def search(
        self,
        image_path: str | Path,
        *,
        asset_id: str,
    ) -> MultiProductResult:
        """Detect and retrieve every object while retaining transitive evidence."""

        asset_id = _require_asset_id(asset_id)
        try:
            parent_snapshot = read_regular_file_snapshot(
                image_path,
                "multi-product parent image",
                max_bytes=_MAX_PARENT_BYTES,
            )
            parent_image = _decode_parent(parent_snapshot.content)
        except ModelArtifactError as exc:
            raise MultiProductError(str(exc)) from exc

        try:
            detection_result = self.detector.detect(
                parent_snapshot.path,
                asset_id=asset_id,
            )
            detection_result = _validated_detection_result(
                detection_result,
                asset_id=asset_id,
                parent_sha256=parent_snapshot.sha256,
                parent_bytes=len(parent_snapshot.content),
                width=parent_image.width,
                height=parent_image.height,
            )
            binding = self.product_search.artifact_binding
            if not isinstance(binding, RetrievalArtifactBinding):
                raise MultiProductError(
                    "product search artifact binding changed to an invalid value"
                )
            objects = self._search_detections(
                parent_image,
                detection_result.detections,
                binding=binding,
            )
            if self.product_search.artifact_binding != binding:
                raise MultiProductError(
                    "product search artifact binding changed during composition"
                )
            result = MultiProductResult(
                input_binding=detection_result.input_binding,
                detection_runtime_binding=detection_result.runtime_binding,
                artifact_binding=binding,
                objects=objects,
            )
        finally:
            try:
                verify_file_snapshot(
                    parent_snapshot,
                    "multi-product parent image",
                )
            except ModelArtifactError as exc:
                raise MultiProductError(str(exc)) from exc
        return result

    def _search_detections(
        self,
        parent_image: Image.Image,
        detections: tuple[DetectedObject, ...],
        *,
        binding: RetrievalArtifactBinding,
    ) -> tuple[MultiProductObjectResult, ...]:
        objects: list[MultiProductObjectResult] = []
        with TemporaryDirectory(prefix="skillchain-multi-product-") as directory:
            temporary_root = Path(directory)
            for ordinal, detection in enumerate(detections):
                crop_box = _canonical_crop_box(
                    detection.bbox_xyxy,
                    width=parent_image.width,
                    height=parent_image.height,
                )
                crop_image = parent_image.crop(crop_box).convert("RGB")
                crop_bytes = _canonical_png_bytes(crop_image)
                crop_sha256 = sha256_bytes(crop_bytes)
                crop_path = temporary_root / (
                    f"crop-{ordinal:04d}-{detection.detection_id[:12]}.png"
                )
                crop_path.write_bytes(crop_bytes)
                try:
                    trace = self.product_search.trace_image_product_search(crop_path)
                    trace = _validated_search_trace(
                        trace,
                        crop_sha256=crop_sha256,
                        artifact_binding=binding,
                    )
                    try:
                        current_crop = read_stable_regular_file(
                            crop_path,
                            label="multi-product temporary crop",
                            max_bytes=len(crop_bytes),
                        )
                    except ArtifactFormatError as exc:
                        raise MultiProductError(str(exc)) from exc
                    if current_crop != crop_bytes:
                        raise MultiProductError(
                            "temporary crop changed during product search"
                        )
                finally:
                    crop_path.unlink(missing_ok=True)
                objects.append(
                    MultiProductObjectResult(
                        detection_id=detection.detection_id,
                        class_id=detection.class_id,
                        label=detection.label,
                        label_zh=detection.label_zh,
                        confidence=detection.confidence,
                        bbox_xyxy=detection.bbox_xyxy,
                        crop_box_xyxy=crop_box,
                        crop_sha256=crop_sha256,
                        crop_bytes=len(crop_bytes),
                        crop_width=crop_image.width,
                        crop_height=crop_image.height,
                        query_vector_sha256=trace.query_vector_sha256,
                        artifact_binding=trace.artifact_binding,
                        hits=trace.hits,
                    )
                )
        return tuple(objects)


def multi_product_search(
    image_path: str | Path,
    *,
    asset_id: str,
    detector: DetectionService,
    product_search: TraceProductSearchService,
) -> MultiProductResult:
    """Dependency-injected convenience entry point."""

    return MultiProductSearchService(detector, product_search).search(
        image_path,
        asset_id=asset_id,
    )


def _validated_detection_result(
    value: object,
    *,
    asset_id: str,
    parent_sha256: str,
    parent_bytes: int,
    width: int,
    height: int,
) -> ObjectDetectionResult:
    if not isinstance(value, ObjectDetectionResult):
        raise MultiProductError("detector must return ObjectDetectionResult")
    input_binding = value.input_binding
    expected = (asset_id, parent_sha256, parent_bytes, width, height)
    actual = (
        input_binding.asset_id,
        input_binding.image_sha256,
        input_binding.image_bytes,
        input_binding.width,
        input_binding.height,
    )
    if actual != expected:
        raise MultiProductError("detector input binding does not match parent image")
    if value.runtime_binding.artifact_kind != "object_detector":
        raise MultiProductError("detector returned a non-detector runtime binding")
    detections = value.detections
    ids = [item.detection_id for item in detections]
    if len(ids) != len(set(ids)):
        raise MultiProductError("detector returned duplicate detection_id")
    if tuple(sorted(detections, key=_detection_sort_key)) != detections:
        raise MultiProductError("detector results are not in canonical order")
    for detection in detections:
        _validate_detection_bounds(detection, width=width, height=height)
    return value


def _validated_search_trace(
    value: object,
    *,
    crop_sha256: str,
    artifact_binding: RetrievalArtifactBinding,
) -> ProductSearchTrace:
    if not isinstance(value, ProductSearchTrace):
        raise MultiProductError("product search must return ProductSearchTrace")
    if value.query_image_sha256 != crop_sha256:
        raise MultiProductError("product search trace does not match crop bytes")
    if value.artifact_binding != artifact_binding:
        raise MultiProductError("product search trace artifact binding mismatch")
    if any(hit.artifact_binding != artifact_binding for hit in value.hits):
        raise MultiProductError("product hit artifact binding mismatch")
    if [hit.rank for hit in value.hits] != list(range(1, len(value.hits) + 1)):
        raise MultiProductError("product search trace ranks are not contiguous")
    return value


def _canonical_crop_box(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or any(not math.isfinite(float(value)) for value in bbox):
        raise MultiProductError("detection bbox must contain four finite values")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise MultiProductError("detection bbox exceeds parent image bounds")
    if x2 <= x1 or y2 <= y1:
        raise MultiProductError("detection bbox has no positive area")
    crop_box = (
        min(max(math.floor(x1), 0), width),
        min(max(math.floor(y1), 0), height),
        min(max(math.ceil(x2), 0), width),
        min(max(math.ceil(y2), 0), height),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise MultiProductError("canonical crop has no positive area")
    return crop_box


def _validate_detection_bounds(
    detection: DetectedObject,
    *,
    width: int,
    height: int,
) -> None:
    if not isinstance(detection, DetectedObject):
        raise MultiProductError("detector returned an invalid detection object")
    _canonical_crop_box(detection.bbox_xyxy, width=width, height=height)


def _canonical_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    content = buffer.getvalue()
    if not content:
        raise MultiProductError("canonical crop encoding produced no bytes")
    return content


def _decode_parent(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.width * opened.height > _MAX_PARENT_PIXELS:
                raise MultiProductError("parent image exceeds the pixel limit")
            if getattr(opened, "n_frames", 1) != 1:
                raise MultiProductError("parent image must contain one frame")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except MultiProductError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise MultiProductError("parent image cannot be decoded") from exc


def _require_asset_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiProductError("asset_id must be a non-blank string")
    if value != value.strip():
        raise MultiProductError("asset_id must not have surrounding whitespace")
    return value


def _detection_sort_key(item: DetectedObject) -> tuple[Any, ...]:
    return (-item.confidence, item.class_id, *item.bbox_xyxy, item.label)


def _object_sort_key(item: MultiProductObjectResult) -> tuple[Any, ...]:
    return (-item.confidence, item.class_id, *item.bbox_xyxy, item.label)


__all__ = [
    "MULTI_PRODUCT_CONTRACT_VERSION",
    "DetectionService",
    "MultiProductError",
    "MultiProductObjectResult",
    "MultiProductResult",
    "MultiProductSearchService",
    "TraceProductSearchService",
    "multi_product_search",
]
