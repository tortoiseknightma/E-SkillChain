"""Local document OCR with evidence, provenance, and fail-closed PII approval."""

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

OCR_CONTRACT_VERSION = 1
DOCUMENT_SAFETY_SCHEMA_VERSION = 1
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 40_000_000
_MODEL_CONFIG_BYTES = 2 * 1024 * 1024

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Point = tuple[FiniteFloat, FiniteFloat]
Polygon = tuple[Point, Point, Point, Point]


class DocumentOCRError(ValueError):
    """Base error for OCR input, safety, artifact, and output violations."""


class DocumentSafetyError(DocumentOCRError):
    """Raised when explicit PII review approval is absent or invalid."""


class OCRRuntimeUnavailableError(DocumentOCRError):
    """Raised when the optional local OCR runtime is unavailable."""


class _LoadedOCREngineAuthority:
    """Hold loader-issued engine receipts outside writable backend state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: WeakKeyDictionary[object, tuple[object, object, str]] = (
            WeakKeyDictionary()
        )

    def loaded_engine(self, backend: object, artifact: object) -> object | None:
        with self._lock:
            receipt = self._records.get(backend)
        if receipt is None:
            return None
        engine, issued_artifact, _config_sha256 = receipt
        if issued_artifact is not artifact:
            raise DocumentOCRError(
                "loaded OCR engine does not belong to the current artifact"
            )
        return engine

    def validate_config(
        self,
        backend: object,
        artifact: object,
        config: "OCRConfig",
    ) -> None:
        with self._lock:
            receipt = self._records.get(backend)
        if receipt is None:
            return
        _engine, issued_artifact, issued_config_sha256 = receipt
        if issued_artifact is not artifact:
            raise DocumentOCRError(
                "loaded OCR engine does not belong to the current artifact"
            )
        if issued_config_sha256 != _ocr_config_sha256(config):
            raise DocumentOCRError(
                "loaded OCR engine does not match the authoritative config"
            )

    def issue(
        self,
        backend: object,
        artifact: object,
        engine: object,
        config: "OCRConfig",
    ) -> None:
        if engine is None:
            raise DocumentOCRError("OCR loader returned no engine")
        with self._lock:
            if backend in self._records:
                raise DocumentOCRError("OCR engine receipt was already issued")
            self._records[backend] = (engine, artifact, _ocr_config_sha256(config))


_LOADED_OCR_ENGINES = _LoadedOCREngineAuthority()


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OCRConfig(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    languages: tuple[str, ...]
    max_lines: int = Field(ge=1, le=5000)
    max_characters: int = Field(ge=1, le=200_000)
    use_classifier: bool
    device: Literal["cpu"] = "cpu"
    coordinate_decimals: Literal[6] = 6

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("OCR languages must not be empty")
        if any(not item or item != item.strip() for item in value):
            raise ValueError("OCR languages must be non-blank canonical strings")
        if tuple(sorted(value)) != value or len(value) != len(set(value)):
            raise ValueError("OCR languages must be unique and sorted")
        return value


class RawOCRLine(_StrictFrozenModel):
    """Backend-neutral line before canonical ordering and evidence IDs."""

    text: str = Field(min_length=1)
    polygon: Polygon
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OCR line text must not be blank")
        return value


class OCRLine(RawOCRLine):
    line_id: Sha256


class RawOCRField(_StrictFrozenModel):
    """A deterministic structured field tied to final OCR line IDs."""

    field_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: str = Field(min_length=1)
    evidence_line_ids: tuple[Sha256, ...] = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OCR field value must not be blank")
        return value

    @field_validator("evidence_line_ids")
    @classmethod
    def validate_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("OCR field evidence_line_ids must not repeat")
        return value


class OCRField(RawOCRField):
    field_id: Sha256


class DocumentSafetyApproval(_StrictFrozenModel):
    """Explicit approval bound to exactly one reviewed image byte sequence."""

    schema_version: Literal[1] = DOCUMENT_SAFETY_SCHEMA_VERSION
    asset_id: str = Field(min_length=1)
    image_sha256: Sha256
    decision: Literal["approved_no_pii", "approved_redacted"]
    policy_version: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    redaction_parent_sha256: Sha256 | None = None
    approval_sha256: Sha256

    @field_validator("asset_id", "policy_version", "reviewer_id")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(
                "safety approval strings must not have surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_decision_and_hash(self) -> Self:
        if self.decision == "approved_redacted":
            if self.redaction_parent_sha256 is None:
                raise ValueError("redacted approval requires the parent image hash")
            if self.redaction_parent_sha256 == self.image_sha256:
                raise ValueError("redacted image must differ from its parent bytes")
        elif self.redaction_parent_sha256 is not None:
            raise ValueError("no-PII approval must not claim a redaction parent")
        if self.approval_sha256 != document_safety_approval_digest(self):
            raise ValueError("document safety approval self hash mismatch")
        return self


class OCRInputBinding(_StrictFrozenModel):
    asset_id: str = Field(min_length=1)
    image_sha256: Sha256
    image_bytes: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    safety_decision: Literal["approved_no_pii", "approved_redacted"]
    safety_approval_sha256: Sha256

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("asset_id must not have surrounding whitespace")
        return value


class DocumentOCRResult(_StrictFrozenModel):
    schema_version: Literal[1] = OCR_CONTRACT_VERSION
    input_binding: OCRInputBinding
    runtime_binding: ModelRuntimeBinding
    content_trust: Literal["untrusted_document_text"] = "untrusted_document_text"
    languages: tuple[str, ...]
    full_text: str
    lines: tuple[OCRLine, ...]
    fields: tuple[OCRField, ...]
    truncated: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.runtime_binding.artifact_kind != "document_ocr":
            raise ValueError("OCR result requires a document OCR runtime")
        line_ids = [line.line_id for line in self.lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("OCR line_id values must be unique")
        if self.full_text != "\n".join(line.text for line in self.lines):
            raise ValueError("OCR full_text must be derived from ordered lines")
        expected_lines = tuple(sorted(self.lines, key=_ocr_line_sort_key))
        if expected_lines != self.lines:
            raise ValueError("OCR lines must use canonical reading order")
        for line in self.lines:
            if any(
                x < 0
                or y < 0
                or x > self.input_binding.width
                or y > self.input_binding.height
                for x, y in line.polygon
            ):
                raise ValueError("OCR polygon exceeds image bounds")
            if _polygon_area(line.polygon) <= 0:
                raise ValueError("OCR polygon must have positive area")
        known_line_ids = set(line_ids)
        field_ids = [field.field_id for field in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("OCR field_id values must be unique")
        for field in self.fields:
            if not set(field.evidence_line_ids).issubset(known_line_ids):
                raise ValueError("OCR fields must cite existing line evidence")
        if tuple(sorted(self.fields, key=_ocr_field_sort_key)) != self.fields:
            raise ValueError("OCR fields must use canonical order")
        return self


class OCRBackend(Protocol):
    def recognize(
        self,
        image: Image.Image,
        *,
        config: OCRConfig,
    ) -> Sequence[RawOCRLine]: ...


class OCRFieldExtractor(Protocol):
    def extract(self, lines: tuple[OCRLine, ...]) -> Sequence[RawOCRField]: ...


class RapidOCROnnxBackend:
    """Lazy RapidOCR ONNX runtime; imports and model loading occur on first call."""

    def __init__(self, artifact: VerifiedModelArtifact) -> None:
        if not artifact.external_sha256_verified:
            raise DocumentOCRError(
                "OCR runtime requires an externally locked model manifest"
            )
        if artifact.manifest.artifact_kind != "document_ocr":
            raise DocumentOCRError("RapidOCR backend requires OCR artifacts")
        if artifact.manifest.backend_name != "rapidocr-onnxruntime":
            raise DocumentOCRError(
                "RapidOCR backend requires a rapidocr-onnxruntime model manifest"
            )
        self._artifact = artifact
        self._lock = Lock()

    @property
    def _engine(self) -> Any | None:
        """Expose loader state read-only for diagnostics and legacy observability."""

        if "_engine" in vars(self):
            raise DocumentOCRError(
                "OCR engine state was injected outside the verified loader"
            )
        return _LOADED_OCR_ENGINES.loaded_engine(self, self._artifact)

    @property
    def formal_runtime_binding_sha256(self) -> str:
        self._artifact.verify_files()
        # Resolve the authority-held receipt, or prove the lazy backend is pristine.
        _ = self._engine
        _LOADED_OCR_ENGINES.validate_config(
            self,
            self._artifact,
            _load_canonical_config(
                self._artifact.read_artifact_bytes(
                    "config", max_bytes=_MODEL_CONFIG_BYTES
                )
            ),
        )
        payload = {
            "artifact_runtime": self._artifact.runtime_binding.model_dump(mode="json"),
            "backend_policy_version": "rapidocr-onnx-backend-v2",
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def recognize(
        self,
        image: Image.Image,
        *,
        config: OCRConfig,
    ) -> Sequence[RawOCRLine]:
        engine = self._load_engine(config)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - NumPy is a base dependency.
            raise OCRRuntimeUnavailableError("NumPy is required for OCR") from exc
        result, _elapsed = engine(
            np.asarray(image, dtype=np.uint8), use_cls=config.use_classifier
        )
        if result is None:
            return ()
        lines: list[RawOCRLine] = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise DocumentOCRError("OCR runtime returned an invalid line")
            polygon, text, confidence = item
            try:
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
            except (IndexError, TypeError, ValueError) as exc:
                raise DocumentOCRError(
                    "OCR runtime returned an invalid polygon"
                ) from exc
            lines.append(
                RawOCRLine(
                    text=str(text),
                    polygon=points,
                    confidence=float(confidence),
                )
            )
        return tuple(lines)

    def _load_engine(self, config: OCRConfig) -> Any:
        if self._engine is not None:
            _LOADED_OCR_ENGINES.validate_config(self, self._artifact, config)
            return self._engine
        with self._lock:
            if self._engine is not None:
                _LOADED_OCR_ENGINES.validate_config(self, self._artifact, config)
                return self._engine
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OCRRuntimeUnavailableError(
                    "optional rapidocr-onnxruntime package is not installed"
                ) from exc
            try:
                installed_version = metadata.version("rapidocr-onnxruntime")
            except metadata.PackageNotFoundError as exc:
                raise OCRRuntimeUnavailableError(
                    "RapidOCR package metadata is unavailable"
                ) from exc
            if installed_version != self._artifact.manifest.backend_version:
                raise DocumentOCRError(
                    "installed RapidOCR version does not match the model manifest"
                )
            required_roles = ("detector_model", "recognizer_model")
            for role in required_roles:
                self._artifact.descriptor(role)
            arguments: dict[str, str] = {
                "det_model_path": str(self._artifact.artifact_path("detector_model")),
                "rec_model_path": str(self._artifact.artifact_path("recognizer_model")),
                "rec_keys_path": str(self._artifact.artifact_path("labels")),
            }
            try:
                self._artifact.descriptor("classifier_model")
            except ModelArtifactError:
                if config.use_classifier:
                    raise DocumentOCRError(
                        "OCR config requires a classifier model artifact"
                    ) from None
            else:
                arguments["cls_model_path"] = str(
                    self._artifact.artifact_path("classifier_model")
                )
            self._artifact.verify_files()
            engine = RapidOCR(**arguments)
            self._artifact.verify_files()
            _LOADED_OCR_ENGINES.issue(self, self._artifact, engine, config)
            return engine


class DocumentOCRService:
    """Execute OCR only for bytes covered by an explicit safety approval."""

    def __init__(
        self,
        artifact: VerifiedModelArtifact,
        backend: OCRBackend,
        *,
        field_extractor: OCRFieldExtractor | None = None,
    ) -> None:
        if not artifact.external_sha256_verified:
            raise DocumentOCRError(
                "document OCR service requires an externally locked model manifest"
            )
        if artifact.manifest.artifact_kind != "document_ocr":
            raise DocumentOCRError("document OCR service requires OCR artifacts")
        self.artifact = artifact
        self.backend = backend
        self.field_extractor = field_extractor
        self.config = _load_canonical_config(
            artifact.read_artifact_bytes("config", max_bytes=_MODEL_CONFIG_BYTES)
        )

    @property
    def formal_runtime_binding_sha256(self) -> str:
        self.artifact.verify_files()
        backend_binding = getattr(self.backend, "formal_runtime_binding_sha256", None)
        if (
            not isinstance(backend_binding, str)
            or len(backend_binding) != 64
            or any(character not in "0123456789abcdef" for character in backend_binding)
        ):
            raise DocumentOCRError(
                "formal OCR runtime requires an explicitly bound backend"
            )
        if self.field_extractor is None:
            extractor_binding = hashlib.sha256(
                b"document-ocr-no-field-extractor-v1"
            ).hexdigest()
        else:
            extractor_binding = getattr(
                self.field_extractor, "formal_runtime_binding_sha256", None
            )
            if (
                not isinstance(extractor_binding, str)
                or len(extractor_binding) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in extractor_binding
                )
            ):
                raise DocumentOCRError(
                    "formal OCR runtime requires an explicitly bound field extractor"
                )
        payload = {
            "artifact_runtime": self.artifact.runtime_binding.model_dump(mode="json"),
            "backend_runtime_sha256": backend_binding,
            "field_extractor_runtime_sha256": extractor_binding,
            "service_policy_version": "document-ocr-service-v1",
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def ocr(
        self,
        image_path: str | Path,
        *,
        asset_id: str,
        safety_approval: DocumentSafetyApproval | None,
    ) -> DocumentOCRResult:
        if safety_approval is None:
            raise DocumentSafetyError(
                "document OCR requires explicit PII safety approval"
            )
        if not isinstance(safety_approval, DocumentSafetyApproval):
            raise DocumentSafetyError(
                "document safety approval must use the strict contract"
            )
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise DocumentOCRError("asset_id must be a non-blank string")
        if asset_id != asset_id.strip():
            raise DocumentOCRError("asset_id must not have surrounding whitespace")
        try:
            snapshot = read_regular_file_snapshot(
                image_path,
                "OCR input image",
                max_bytes=_MAX_IMAGE_BYTES,
            )
            image = _decode_image(snapshot.content)
            self.artifact.verify_files()
        except ModelArtifactError as exc:
            raise DocumentOCRError(str(exc)) from exc
        _validate_safety_approval(
            safety_approval,
            asset_id=asset_id,
            image_sha256=snapshot.sha256,
        )

        input_binding = OCRInputBinding(
            asset_id=asset_id,
            image_sha256=snapshot.sha256,
            image_bytes=len(snapshot.content),
            width=image.width,
            height=image.height,
            safety_decision=safety_approval.decision,
            safety_approval_sha256=safety_approval.approval_sha256,
        )
        try:
            raw_lines = tuple(self.backend.recognize(image.copy(), config=self.config))
        finally:
            _reverify_after_inference(snapshot, self.artifact)

        lines, truncated = self._normalize_lines(raw_lines, input_binding=input_binding)
        fields = self._extract_fields(lines, input_binding=input_binding)
        return DocumentOCRResult(
            input_binding=input_binding,
            runtime_binding=self.artifact.runtime_binding,
            languages=self.config.languages,
            full_text="\n".join(line.text for line in lines),
            lines=lines,
            fields=fields,
            truncated=truncated,
        )

    def _normalize_lines(
        self,
        values: Sequence[RawOCRLine],
        *,
        input_binding: OCRInputBinding,
    ) -> tuple[tuple[OCRLine, ...], bool]:
        candidates: list[dict[str, Any]] = []
        for value in values:
            try:
                raw = RawOCRLine.model_validate(value)
            except ValueError as exc:
                raise DocumentOCRError(f"invalid OCR backend output: {exc}") from exc
            polygon = _clip_and_round_polygon(
                raw.polygon,
                width=input_binding.width,
                height=input_binding.height,
                decimals=self.config.coordinate_decimals,
            )
            candidates.append(
                {
                    "text": raw.text,
                    "polygon": polygon,
                    "confidence": round(
                        float(raw.confidence), self.config.coordinate_decimals
                    ),
                }
            )
        candidates.sort(key=_raw_line_sort_key)

        truncated = len(candidates) > self.config.max_lines
        candidates = candidates[: self.config.max_lines]
        remaining = self.config.max_characters
        selected: list[dict[str, Any]] = []
        for candidate in candidates:
            separator_cost = 1 if selected else 0
            if remaining <= separator_cost:
                truncated = True
                break
            available = remaining - separator_cost
            text = candidate["text"]
            if len(text) > available:
                text = text[:available]
                truncated = True
            if not text:
                break
            selected.append({**candidate, "text": text})
            remaining -= separator_cost + len(text)
            if text != candidate["text"]:
                break

        selected.sort(key=_raw_line_sort_key)

        lines: list[OCRLine] = []
        for ordinal, candidate in enumerate(selected, start=1):
            identity = {
                "contract_version": OCR_CONTRACT_VERSION,
                "input_sha256": input_binding.image_sha256,
                "runtime_sha256": self.artifact.manifest.manifest_sha256,
                "ordinal": ordinal,
                **candidate,
            }
            lines.append(
                OCRLine(
                    line_id=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
                    **candidate,
                )
            )
        return tuple(lines), truncated

    def _extract_fields(
        self,
        lines: tuple[OCRLine, ...],
        *,
        input_binding: OCRInputBinding,
    ) -> tuple[OCRField, ...]:
        if self.field_extractor is None:
            return ()
        known_line_ids = {line.line_id for line in lines}
        raw_fields: list[RawOCRField] = []
        for value in self.field_extractor.extract(lines):
            try:
                field = RawOCRField.model_validate(value)
            except ValueError as exc:
                raise DocumentOCRError(f"invalid OCR field output: {exc}") from exc
            if not set(field.evidence_line_ids).issubset(known_line_ids):
                raise DocumentOCRError("OCR field cites unknown line evidence")
            raw_fields.append(field)
        raw_fields.sort(
            key=lambda item: (item.field_name, item.value, item.evidence_line_ids)
        )
        fields: list[OCRField] = []
        for ordinal, field in enumerate(raw_fields, start=1):
            payload = field.model_dump(mode="python")
            identity = {
                "contract_version": OCR_CONTRACT_VERSION,
                "input_sha256": input_binding.image_sha256,
                "runtime_sha256": self.artifact.manifest.manifest_sha256,
                "ordinal": ordinal,
                **payload,
            }
            fields.append(
                OCRField(
                    field_id=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
                    **payload,
                )
            )
        return tuple(fields)


def build_document_safety_approval(
    *,
    asset_id: str,
    image_sha256: str,
    decision: Literal["approved_no_pii", "approved_redacted"],
    policy_version: str,
    reviewer_id: str,
    redaction_parent_sha256: str | None = None,
) -> DocumentSafetyApproval:
    """Build a self-hashed approval after an external human/privacy review."""

    unsigned = {
        "schema_version": DOCUMENT_SAFETY_SCHEMA_VERSION,
        "asset_id": asset_id,
        "image_sha256": image_sha256,
        "decision": decision,
        "policy_version": policy_version,
        "reviewer_id": reviewer_id,
        "redaction_parent_sha256": redaction_parent_sha256,
    }
    return DocumentSafetyApproval.model_validate(
        {**unsigned, "approval_sha256": document_safety_approval_digest(unsigned)}
    )


def document_safety_approval_digest(
    approval: DocumentSafetyApproval | dict[str, Any],
) -> str:
    if isinstance(approval, BaseModel):
        unsigned = approval.model_dump(mode="json", exclude={"approval_sha256"})
    else:
        unsigned = dict(approval)
        unsigned.pop("approval_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def document_ocr(
    image_path: str | Path,
    *,
    asset_id: str,
    safety_approval: DocumentSafetyApproval | None,
    service: DocumentOCRService,
) -> DocumentOCRResult:
    """Explicit dependency-injected convenience entry point."""

    return service.ocr(
        image_path,
        asset_id=asset_id,
        safety_approval=safety_approval,
    )


def _validate_safety_approval(
    approval: DocumentSafetyApproval,
    *,
    asset_id: str,
    image_sha256: str,
) -> None:
    if approval.asset_id != asset_id:
        raise DocumentSafetyError("document safety approval asset_id mismatch")
    if approval.image_sha256 != image_sha256:
        raise DocumentSafetyError("document safety approval image hash mismatch")
    if approval.approval_sha256 != document_safety_approval_digest(approval):
        raise DocumentSafetyError("document safety approval hash mismatch")


def _ocr_config_sha256(config: OCRConfig) -> str:
    return hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="json"))
    ).hexdigest()


def _load_canonical_config(content: bytes) -> OCRConfig:
    try:
        raw = json.loads(
            content.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
        value = OCRConfig.model_validate_json(canonical_json_bytes(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DocumentOCRError(f"OCR config is invalid: {exc}") from exc
    if content != canonical_json_bytes(value):
        raise DocumentOCRError("OCR config must use canonical JSON")
    return value


def _decode_image(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.width * opened.height > _MAX_IMAGE_PIXELS:
                raise DocumentOCRError("OCR input image exceeds the pixel limit")
            if getattr(opened, "n_frames", 1) != 1:
                raise DocumentOCRError("OCR input image must have one frame")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except DocumentOCRError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise DocumentOCRError("OCR input image cannot be decoded") from exc
    return image


def _clip_and_round_polygon(
    polygon: Polygon,
    *,
    width: int,
    height: int,
    decimals: int,
) -> Polygon:
    points = tuple(
        (
            round(min(max(float(point[0]), 0.0), float(width)), decimals),
            round(min(max(float(point[1]), 0.0), float(height)), decimals),
        )
        for point in polygon
    )
    if _polygon_area(points) <= 0:
        raise DocumentOCRError("OCR polygon has no positive in-bounds area")
    return points


def _polygon_area(polygon: Polygon) -> float:
    return (
        abs(
            sum(
                polygon[index][0] * polygon[(index + 1) % 4][1]
                - polygon[(index + 1) % 4][0] * polygon[index][1]
                for index in range(4)
            )
        )
        / 2.0
    )


def _raw_line_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    polygon = item["polygon"]
    return (
        min(point[1] for point in polygon),
        min(point[0] for point in polygon),
        item["text"],
        -item["confidence"],
        polygon,
    )


def _ocr_line_sort_key(item: OCRLine) -> tuple[Any, ...]:
    return _raw_line_sort_key(
        {
            "text": item.text,
            "polygon": item.polygon,
            "confidence": item.confidence,
        }
    )


def _ocr_field_sort_key(item: OCRField) -> tuple[Any, ...]:
    return (item.field_name, item.value, item.evidence_line_ids)


def _reverify_after_inference(snapshot, artifact: VerifiedModelArtifact) -> None:
    input_error: Exception | None = None
    model_error: Exception | None = None
    try:
        verify_file_snapshot(snapshot, "OCR input image")
    except Exception as exc:
        input_error = exc
    try:
        artifact.verify_files()
    except Exception as exc:
        model_error = exc
    if input_error is not None:
        raise DocumentOCRError(str(input_error)) from input_error
    if model_error is not None:
        raise DocumentOCRError(str(model_error)) from model_error


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DocumentOCRError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


__all__ = [
    "DOCUMENT_SAFETY_SCHEMA_VERSION",
    "OCR_CONTRACT_VERSION",
    "DocumentOCRError",
    "DocumentOCRResult",
    "DocumentOCRService",
    "DocumentSafetyApproval",
    "DocumentSafetyError",
    "OCRBackend",
    "OCRConfig",
    "OCRField",
    "OCRFieldExtractor",
    "OCRInputBinding",
    "OCRLine",
    "OCRRuntimeUnavailableError",
    "RapidOCROnnxBackend",
    "RawOCRField",
    "RawOCRLine",
    "build_document_safety_approval",
    "document_ocr",
    "document_safety_approval_digest",
]
