from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from pydantic import ValidationError
import pytest

from skillchain.schemas import Product
from skillchain.tools.contracts import (
    ProductHit,
    ProductSearchTrace,
    RetrievalArtifactBinding,
)
from skillchain.tools.model_artifacts import ModelRuntimeBinding
from skillchain.tools.multi_product import (
    MultiProductError,
    MultiProductObjectResult,
    MultiProductResult,
    MultiProductSearchService,
    multi_product_search,
)
from skillchain.tools.object_detect import (
    DetectedObject,
    DetectionInputBinding,
    ObjectDetectionResult,
)
from skillchain.tools.serialization import sha256_bytes


def _binding(character: str = "1") -> RetrievalArtifactBinding:
    return RetrievalArtifactBinding(
        mode="verified",
        index_integrity_sha256=character * 64,
        eligibility_sha256="2" * 64,
        query_artifact_sha256="3" * 64,
        gallery_artifact_sha256="4" * 64,
        asset_catalog_sha256="5" * 64,
        products_parquet_sha256="6" * 64,
        leakage_policy_version="fixture-policy-v1",
    )


RUNTIME_BINDING = ModelRuntimeBinding(
    artifact_kind="object_detector",
    model_id="fixture-detector-v1",
    backend_name="fixture",
    backend_version="1.0.0",
    manifest_sha256="7" * 64,
    artifacts=(),
)


def _write_parent(path: Path) -> Path:
    image = Image.new("RGB", (10, 9), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x * 20, y * 20, (x + y) * 10)
    image.save(path, format="PNG")
    return path


def _detections() -> tuple[DetectedObject, ...]:
    return (
        DetectedObject(
            detection_id="8" * 64,
            class_id=0,
            label="person",
            label_zh="人",
            bbox_xyxy=(1.8, 2.2, 6.1, 7.01),
            confidence=0.9,
        ),
        DetectedObject(
            detection_id="9" * 64,
            class_id=1,
            label="bag",
            label_zh="包",
            bbox_xyxy=(0.0, 0.0, 3.0, 4.0),
            confidence=0.8,
        ),
    )


def _result_for(
    path: Path,
    asset_id: str,
    *,
    detections: tuple[DetectedObject, ...] | None = None,
    bound_asset_id: str | None = None,
    image_sha256: str | None = None,
) -> ObjectDetectionResult:
    content = path.read_bytes()
    with Image.open(BytesIO(content)) as image:
        width, height = image.size
    return ObjectDetectionResult(
        input_binding=DetectionInputBinding(
            asset_id=bound_asset_id or asset_id,
            image_sha256=image_sha256 or sha256_bytes(content),
            image_bytes=len(content),
            width=width,
            height=height,
        ),
        runtime_binding=RUNTIME_BINDING,
        detections=_detections() if detections is None else detections,
    )


class FakeDetector:
    def __init__(self, factory=None, callback=None):
        self.factory = factory or (lambda path, asset_id: _result_for(path, asset_id))
        self.callback = callback
        self.calls: list[tuple[Path, str]] = []

    def detect(self, image_path, *, asset_id):
        path = Path(image_path)
        self.calls.append((path, asset_id))
        result = self.factory(path, asset_id)
        if self.callback is not None:
            self.callback()
        return result


class FakeProductSearch:
    def __init__(
        self,
        binding: RetrievalArtifactBinding,
        *,
        trace_binding: RetrievalArtifactBinding | None = None,
        hit_binding: RetrievalArtifactBinding | None = None,
        image_sha256: str | None = None,
        callback=None,
        raises: Exception | None = None,
        execution_location: str = "local",
    ) -> None:
        self.artifact_binding = binding
        self.trace_binding = trace_binding or binding
        self.hit_binding = hit_binding or self.trace_binding
        self.image_sha256 = image_sha256
        self.callback = callback
        self.raises = raises
        self.execution_location = execution_location
        self.calls: list[dict] = []

    def trace_image_product_search(self, image):
        path = Path(image)
        content = path.read_bytes()
        with Image.open(BytesIO(content)) as crop:
            self.calls.append(
                {
                    "bytes": content,
                    "mode": crop.mode,
                    "path": path,
                    "size": crop.size,
                }
            )
        if self.callback is not None:
            self.callback(path)
        if self.raises is not None:
            raise self.raises
        crop_sha256 = sha256_bytes(content)
        product = Product(
            product_id=f"product-{crop_sha256[:12]}",
            title="fixture product",
            category_l1="fixture",
            image_path="gallery/fixture.png",
            source="muge",
        )
        return ProductSearchTrace(
            query_image_sha256=self.image_sha256 or crop_sha256,
            query_vector_sha256=sha256_bytes(b"vector:" + content),
            artifact_binding=self.trace_binding,
            hits=(
                ProductHit(
                    rank=1,
                    score=0.5,
                    product=product,
                    artifact_binding=self.hit_binding,
                ),
            ),
        )


def test_composition_exposes_actual_embedding_execution_location():
    local = MultiProductSearchService(FakeDetector(), FakeProductSearch(_binding()))
    remote = MultiProductSearchService(
        FakeDetector(),
        FakeProductSearch(_binding(), execution_location="remote"),
    )

    assert local.execution_location == "local"
    assert remote.execution_location == "remote"
    remote.product_search.execution_location = "unknown"
    with pytest.raises(MultiProductError, match="execution location"):
        _ = remote.execution_location


def test_composition_crops_in_canonical_order_and_retains_all_bindings(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")
    parent_before = parent.read_bytes()
    detector = FakeDetector()
    search = FakeProductSearch(_binding())
    service = MultiProductSearchService(detector, search)

    first = service.search(parent, asset_id="asset-parent")
    second = service.search(parent, asset_id="asset-parent")

    assert first == second
    assert isinstance(first, MultiProductResult)
    assert first.input_binding.asset_id == "asset-parent"
    assert first.input_binding.image_sha256 == sha256_bytes(parent_before)
    assert first.detection_runtime_binding == RUNTIME_BINDING
    assert first.artifact_binding == search.artifact_binding
    assert [item.detection_id for item in first.objects] == ["8" * 64, "9" * 64]
    assert first.objects[0].bbox_xyxy == (1.8, 2.2, 6.1, 7.01)
    assert first.objects[0].crop_box_xyxy == (1, 2, 7, 8)
    assert first.objects[0].crop_mode == "RGB"
    assert first.objects[0].crop_format == "PNG"
    assert first.objects[0].crop_policy_version == "floor-ceil-clamp-rgb-png-v1"
    assert (first.objects[0].crop_width, first.objects[0].crop_height) == (6, 6)
    assert first.objects[1].crop_box_xyxy == (0, 0, 3, 4)
    assert (first.objects[1].crop_width, first.objects[1].crop_height) == (3, 4)
    assert all(item.crop_bytes > 0 for item in first.objects)
    assert all(item.query_vector_sha256 for item in first.objects)
    assert all(item.artifact_binding == _binding() for item in first.objects)
    assert all(item.hits[0].artifact_binding == _binding() for item in first.objects)
    assert all(call["mode"] == "RGB" for call in search.calls)
    assert all(not call["path"].exists() for call in search.calls)
    assert parent.read_bytes() == parent_before
    assert MultiProductResult.model_validate_json(first.model_dump_json()) == first


def test_crop_png_bytes_use_fixed_floor_ceil_clamp_encoding(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")
    search = FakeProductSearch(_binding())

    result = multi_product_search(
        parent,
        asset_id="asset-parent",
        detector=FakeDetector(),
        product_search=search,
    )

    with Image.open(parent) as opened:
        expected_crop = opened.convert("RGB").crop((1, 2, 7, 8))
    buffer = BytesIO()
    expected_crop.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    expected = buffer.getvalue()
    assert search.calls[0]["bytes"] == expected
    assert result.objects[0].crop_sha256 == sha256_bytes(expected)
    assert result.objects[0].crop_bytes == len(expected)


def test_empty_detection_is_a_valid_bound_result_without_search_calls(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")
    detector = FakeDetector(
        lambda path, asset_id: _result_for(path, asset_id, detections=())
    )
    search = FakeProductSearch(_binding())

    result = MultiProductSearchService(detector, search).search(
        parent,
        asset_id="asset-empty",
    )

    assert result.objects == ()
    assert result.input_binding.asset_id == "asset-empty"
    assert result.detection_runtime_binding == RUNTIME_BINDING
    assert result.artifact_binding == search.artifact_binding
    assert search.calls == []


@pytest.mark.parametrize("mismatch", ["asset", "hash"])
def test_rejects_detector_input_binding_mismatch(tmp_path, mismatch):
    parent = _write_parent(tmp_path / "parent.png")

    def factory(path, asset_id):
        return _result_for(
            path,
            asset_id,
            bound_asset_id="asset-other" if mismatch == "asset" else None,
            image_sha256="0" * 64 if mismatch == "hash" else None,
        )

    with pytest.raises(MultiProductError, match="input binding"):
        MultiProductSearchService(
            FakeDetector(factory), FakeProductSearch(_binding())
        ).search(parent, asset_id="asset-parent")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate detection_id"),
        ("order", "canonical order"),
        ("empty", "positive area"),
        ("bounds", "exceeds parent image bounds"),
    ],
)
def test_rejects_duplicate_noncanonical_empty_or_out_of_bounds_detections(
    tmp_path, mutation, message
):
    parent = _write_parent(tmp_path / "parent.png")

    def factory(path, asset_id):
        valid = _result_for(path, asset_id)
        detections = valid.detections
        if mutation == "duplicate":
            malicious = (detections[0], detections[0])
        elif mutation == "order":
            malicious = tuple(reversed(detections))
        else:
            bbox = (
                (1.0, 1.0, 1.0, 4.0) if mutation == "empty" else (1.0, 1.0, 11.0, 4.0)
            )
            bad = DetectedObject.model_construct(
                detection_id="a" * 64,
                class_id=0,
                label="person",
                label_zh="人",
                bbox_xyxy=bbox,
                confidence=0.9,
            )
            malicious = (bad,)
        return ObjectDetectionResult.model_construct(
            schema_version=1,
            input_binding=valid.input_binding,
            runtime_binding=valid.runtime_binding,
            detections=malicious,
        )

    service = MultiProductSearchService(
        FakeDetector(factory), FakeProductSearch(_binding())
    )
    with pytest.raises(MultiProductError, match=message):
        service.search(parent, asset_id="asset-parent")


@pytest.mark.parametrize("mismatch", ["crop", "trace-binding", "hit-binding"])
def test_rejects_incoherent_product_search_trace(tmp_path, mismatch):
    parent = _write_parent(tmp_path / "parent.png")
    binding = _binding()
    wrong = _binding("a")
    search = FakeProductSearch(
        binding,
        image_sha256="0" * 64 if mismatch == "crop" else None,
        trace_binding=wrong if mismatch == "trace-binding" else None,
        hit_binding=wrong if mismatch == "hit-binding" else None,
    )

    with pytest.raises(
        (MultiProductError, ValidationError), match="crop bytes|artifact binding"
    ):
        MultiProductSearchService(FakeDetector(), search).search(
            parent,
            asset_id="asset-parent",
        )
    assert all(not call["path"].exists() for call in search.calls)


@pytest.mark.parametrize("stage", ["detector", "search"])
def test_parent_bytes_are_rechecked_after_every_composed_stage(tmp_path, stage):
    parent = _write_parent(tmp_path / "parent.png")

    def mutate_parent(*_):
        parent.write_bytes(b"mutated")

    detector = FakeDetector(callback=mutate_parent if stage == "detector" else None)
    search = FakeProductSearch(
        _binding(), callback=mutate_parent if stage == "search" else None
    )

    with pytest.raises(MultiProductError, match="changed"):
        MultiProductSearchService(detector, search).search(
            parent,
            asset_id="asset-parent",
        )


def test_rejects_symlink_parent_and_cleans_crop_after_search_failure(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(parent)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(MultiProductError, match="symbolic link"):
        MultiProductSearchService(FakeDetector(), FakeProductSearch(_binding())).search(
            link, asset_id="asset-parent"
        )

    search = FakeProductSearch(_binding(), raises=RuntimeError("search failed"))
    with pytest.raises(RuntimeError, match="search failed"):
        MultiProductSearchService(FakeDetector(), search).search(
            parent,
            asset_id="asset-parent",
        )
    assert all(not call["path"].exists() for call in search.calls)


def test_temporary_crop_mutation_is_rejected_and_removed(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")

    def mutate_crop(path: Path):
        path.write_bytes(path.read_bytes() + b"tampered")

    search = FakeProductSearch(_binding(), callback=mutate_crop)
    with pytest.raises(ValueError, match="crop|changed|exceeds"):
        MultiProductSearchService(FakeDetector(), search).search(
            parent,
            asset_id="asset-parent",
        )
    assert all(not call["path"].exists() for call in search.calls)


def test_result_contracts_are_strict_and_frozen(tmp_path):
    parent = _write_parent(tmp_path / "parent.png")
    result = MultiProductSearchService(
        FakeDetector(), FakeProductSearch(_binding())
    ).search(parent, asset_id="asset-parent")

    with pytest.raises(ValidationError):
        result.objects[0].crop_width = 99
    payload = result.objects[0].model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        MultiProductObjectResult.model_validate(payload)

    object_payload = result.objects[0].model_dump(mode="python")
    object_payload["crop_box_xyxy"] = (0, 0, 1, 1)
    object_payload["crop_width"] = 1
    object_payload["crop_height"] = 1
    inconsistent = MultiProductObjectResult(**object_payload)
    result_payload = result.model_dump(mode="python")
    result_payload["objects"] = (inconsistent, *result.objects[1:])
    with pytest.raises(ValidationError, match="crop box"):
        MultiProductResult(**result_payload)
