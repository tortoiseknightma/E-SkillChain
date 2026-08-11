from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from skillchain.data.rpc import (
    DATASET_ASSETS_FILE,
    MANIFEST_FILE,
    RPCArchiveError,
    RPCBundleError,
    SCENES_FILE,
    build_rpc_val_query_adapter,
    load_verified_rpc_val_adapter,
)
from skillchain.data.source_lock import (
    AcquisitionIdentity,
    RequiredSourceLock,
    build_artifact_scope,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes

_ARCHIVE_LOGICAL_PATH = "rpc/kaggle-v5/archive.zip"
_MIRROR_ROOT = "retail_product_checkout"


def _jpeg_bytes(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 12), color=color).save(output, format="JPEG")
    return output.getvalue()


def _annotations() -> dict[str, object]:
    return {
        "images": [
            {"id": 10, "file_name": "scene-10.jpg", "width": 16, "height": 12},
            {"id": 20, "file_name": "scene-20.jpg", "width": 16, "height": 12},
            {"id": 30, "file_name": "scene-30.jpg", "width": 16, "height": 12},
            {"id": 40, "file_name": "scene-40.jpg", "width": 16, "height": 12},
        ],
        "categories": [
            {"id": 1, "name": "apple"},
            {"id": 2, "name": "bottle"},
            {"id": 3, "name": "carton"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 10,
                "category_id": 1,
                "bbox": [0, 0, 4, 5],
            },
            {
                "id": 102,
                "image_id": 10,
                "category_id": 2,
                "bbox": [5, 1, 4, 5],
            },
            {
                "id": 201,
                "image_id": 20,
                "category_id": 1,
                "bbox": [0, 0, 3, 4],
            },
            {
                "id": 202,
                "image_id": 20,
                "category_id": 1,
                "bbox": [4, 0, 3, 4],
            },
            {
                "id": 203,
                "image_id": 20,
                "category_id": 3,
                "bbox": [8, 1, 4, 5],
            },
            {
                "id": 301,
                "image_id": 30,
                "category_id": 2,
                "bbox": [1, 1, 5, 6],
            },
            {
                "id": 302,
                "image_id": 30,
                "category_id": 3,
                "bbox": [7, 2, 5, 6],
            },
            {
                "id": 401,
                "image_id": 40,
                "category_id": 1,
                "bbox": [1, 1, 5, 6],
            },
        ],
    }


def _write_archive(
    raw_root: Path,
    *,
    unsafe_member: str | None = None,
    divergent_annotations: bool = False,
    divergent_image_id: int | None = None,
) -> tuple[Path, dict[int, bytes]]:
    archive_path = raw_root / Path(*_ARCHIVE_LOGICAL_PATH.split("/"))
    archive_path.parent.mkdir(parents=True)
    annotation_bytes = json.dumps(
        _annotations(), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    mirror_annotation_bytes = (
        annotation_bytes + b" "
        if divergent_annotations
        else annotation_bytes
    )
    images = {
        10: _jpeg_bytes("red"),
        20: _jpeg_bytes("green"),
        30: _jpeg_bytes("blue"),
        40: _jpeg_bytes("yellow"),
    }
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("instances_val2019.json", annotation_bytes)
        archive.writestr(
            f"{_MIRROR_ROOT}/instances_val2019.json",
            mirror_annotation_bytes,
        )
        for image_id, image_bytes in images.items():
            file_name = f"scene-{image_id}.jpg"
            archive.writestr(f"val2019/{file_name}", image_bytes)
            mirror_bytes = (
                _jpeg_bytes("purple")
                if image_id == divergent_image_id
                else image_bytes
            )
            archive.writestr(
                f"{_MIRROR_ROOT}/val2019/{file_name}",
                mirror_bytes,
            )
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")
    return archive_path, images


def _write_lock(raw_root: Path, tmp_path: Path) -> tuple[Path, str]:
    archive_path = raw_root / Path(*_ARCHIVE_LOGICAL_PATH.split("/"))
    archive_bytes = archive_path.stat().st_size
    archive_sha256 = sha256_bytes(archive_path.read_bytes())
    scope = build_artifact_scope(
        raw_root,
        scope_id="kaggle_archive",
        mode="explicit_files",
        paths=(_ARCHIVE_LOGICAL_PATH,),
    )
    lock = RequiredSourceLock(
        source_id="rpc",
        source_revision="kaggle-v5-fixture",
        lock_plan_sha256="a" * 64,
        artifact_scopes=(scope,),
        acquisition_identities=(
            AcquisitionIdentity(
                logical_path=_ARCHIVE_LOGICAL_PATH,
                url="https://example.invalid/rpc/archive.zip",
                bytes=archive_bytes,
                local_sha256=archive_sha256,
            ),
        ),
    )
    lock_path = tmp_path / "rpc.source-lock.json"
    lock_bytes = canonical_json_bytes(lock.model_dump(mode="json"))
    lock_path.write_bytes(lock_bytes)
    return lock_path, sha256_bytes(lock_bytes)


def _fixture_inputs(
    tmp_path: Path,
    **archive_kwargs,
) -> tuple[Path, Path, str, dict[int, bytes]]:
    raw_root = tmp_path / "raw"
    _, images = _write_archive(raw_root, **archive_kwargs)
    lock_path, lock_sha256 = _write_lock(raw_root, tmp_path)
    return raw_root, lock_path, lock_sha256, images


def test_rpc_adapter_selects_deterministically_and_preserves_exact_bytes(
    tmp_path,
):
    raw_root, lock_path, lock_sha256, source_images = _fixture_inputs(tmp_path)
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first_output = first_parent / "rpc-adapter"
    second_output = second_parent / "rpc-adapter"

    first = build_rpc_val_query_adapter(
        raw_root=raw_root,
        source_lock_path=lock_path,
        expected_source_lock_sha256=lock_sha256,
        output_dir=first_output,
        count=2,
    )
    second = build_rpc_val_query_adapter(
        raw_root=raw_root,
        source_lock_path=lock_path,
        expected_source_lock_sha256=lock_sha256,
        output_dir=second_output,
        count=2,
    )

    assert [scene.image_id for scene in first.scenes] == [
        scene.image_id for scene in second.scenes
    ]
    assert [scene.selection_key_sha256 for scene in first.scenes] == sorted(
        scene.selection_key_sha256 for scene in first.scenes
    )
    assert first.manifest.candidate_scene_count == 3
    assert first.manifest.selected_scene_count == 2
    assert first.manifest.cloud_upload_allowed is True
    assert first.manifest.public_demo_allowed is True
    assert first.manifest.annotation_member_paths == (
        "instances_val2019.json",
        f"{_MIRROR_ROOT}/instances_val2019.json",
    )
    assert first.manifest.canonical_root == _MIRROR_ROOT
    assert {
        DATASET_ASSETS_FILE,
        MANIFEST_FILE,
        SCENES_FILE,
    }.issubset(path.name for path in first_output.iterdir())

    for scene, draft in zip(first.scenes, first.drafts, strict=True):
        assert scene.instance_count >= 2
        assert scene.bbox_count == scene.instance_count
        assert scene.image_member_path.startswith(f"{_MIRROR_ROOT}/val2019/")
        assert len(scene.replica_image_member_paths) == 2
        assert scene.dataset_asset_local_path == draft.local_path
        assert draft.local_path.startswith("rpc-adapter/images/")
        assert draft.cloud_upload_allowed is True
        assert draft.public_demo_allowed is True
        output_image = first_parent / draft.local_path
        assert output_image.read_bytes() == source_images[scene.image_id]
        assert sha256_bytes(output_image.read_bytes()) == scene.image_sha256

    verified = load_verified_rpc_val_adapter(
        first_output,
        expected_manifest_file_sha256=first.manifest_file_sha256,
    )
    assert verified.drafts == first.drafts
    assert verified.scenes == first.scenes
    assert verified.manifest == first.manifest

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        build_rpc_val_query_adapter(
            raw_root=raw_root,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            output_dir=first_output,
            count=2,
        )


@pytest.mark.parametrize(
    ("archive_kwargs", "match"),
    [
        ({"unsafe_member": "../escape.txt"}, "unsafe RPC ZIP member path"),
        ({"divergent_annotations": True}, "different annotations"),
    ],
)
def test_rpc_adapter_rejects_unsafe_or_ambiguous_archives(
    tmp_path,
    archive_kwargs,
    match,
):
    raw_root, lock_path, lock_sha256, _ = _fixture_inputs(
        tmp_path, **archive_kwargs
    )
    output = tmp_path / "adapter"

    with pytest.raises(RPCArchiveError, match=match):
        build_rpc_val_query_adapter(
            raw_root=raw_root,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            output_dir=output,
            count=2,
        )

    assert not output.exists()


def test_rpc_adapter_rejects_selected_image_replica_ambiguity(tmp_path):
    raw_root, lock_path, lock_sha256, _ = _fixture_inputs(
        tmp_path,
        divergent_image_id=10,
    )
    output = tmp_path / "adapter"

    with pytest.raises(
        RPCArchiveError,
        match="duplicate roots differ for image 10",
    ):
        build_rpc_val_query_adapter(
            raw_root=raw_root,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            output_dir=output,
            count=3,
        )
    assert not output.exists()


def test_rpc_adapter_rejects_archive_hash_drift_and_bundle_tampering(tmp_path):
    raw_root, lock_path, lock_sha256, _ = _fixture_inputs(tmp_path)
    archive_path = raw_root / Path(*_ARCHIVE_LOGICAL_PATH.split("/"))
    archive_path.write_bytes(archive_path.read_bytes() + b"drift")
    drifted_output = tmp_path / "drifted-adapter"

    with pytest.raises(RPCArchiveError, match="differs from lock"):
        build_rpc_val_query_adapter(
            raw_root=raw_root,
            source_lock_path=lock_path,
            expected_source_lock_sha256=lock_sha256,
            output_dir=drifted_output,
            count=2,
        )
    assert not drifted_output.exists()

    clean_root, clean_lock, clean_lock_sha256, _ = _fixture_inputs(
        tmp_path / "clean"
    )
    result = build_rpc_val_query_adapter(
        raw_root=clean_root,
        source_lock_path=clean_lock,
        expected_source_lock_sha256=clean_lock_sha256,
        output_dir=tmp_path / "clean-adapter",
        count=2,
    )
    image_descriptor = next(
        item for item in result.manifest.files if item.path.startswith("images/")
    )
    image_path = result.output_dir / Path(*image_descriptor.path.split("/"))
    image_path.write_bytes(image_path.read_bytes() + b"tamper")

    with pytest.raises(RPCBundleError, match="payload drifted"):
        load_verified_rpc_val_adapter(
            result.output_dir,
            expected_manifest_file_sha256=result.manifest_file_sha256,
        )
