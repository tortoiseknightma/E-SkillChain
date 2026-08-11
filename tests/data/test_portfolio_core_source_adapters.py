from __future__ import annotations

from io import BytesIO
import gzip
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image
import pytest

from skillchain.data.portfolio_core_assets import (
    CORE_POOL_ASSET_FLOORS,
    PortfolioCoreSourcePolicy,
    PortfolioCoreSourcePolicyDocument,
    load_candidate_inventory,
    preflight_portfolio_core_assets,
)
from skillchain.data.portfolio_core_source_adapters import (
    DEFAULT_ABO_CROSS_INTENT_COUNT,
    DEFAULT_ABO_INCLUDE_COUNT,
    DEFAULT_ABO_RESERVE_COUNT,
    DEFAULT_RESERVE_COUNT,
    DEFAULT_RPC_INCLUDE_COUNT,
    PortfolioCoreSourceAdapterError,
    build_abo_portfolio_core_candidates,
    build_parser,
    build_rpc_portfolio_core_candidates,
)
from skillchain.data.source_lock import (
    AcquisitionIdentity,
    RequiredSourceLock,
    build_artifact_scope,
)
from skillchain.synthesis.store import canonical_json_bytes, sha256_bytes


_ABO_IMAGES = "abo/archives/abo-images-small.tar"
_ABO_LISTINGS = "abo/archives/abo-listings.tar"
_RPC_ARCHIVE = "rpc/kaggle-v5/archive.zip"
_RPC_MIRROR = "retail_product_checkout"


def test_default_new_source_counts_fill_fresh_core_floors() -> None:
    assert CORE_POOL_ASSET_FLOORS["exact_match"] == 157
    assert DEFAULT_ABO_INCLUDE_COUNT == 340
    assert DEFAULT_ABO_RESERVE_COUNT == 8
    assert DEFAULT_RPC_INCLUDE_COUNT == CORE_POOL_ASSET_FLOORS["multi_product"] == 105
    assert DEFAULT_RESERVE_COUNT == 4
    assert DEFAULT_ABO_CROSS_INTENT_COUNT == 25


def test_cli_keeps_source_specific_defaults_and_explicit_overrides() -> None:
    common = [
        "--raw-root",
        "raw",
        "--source-policy",
        "policy.json",
        "--dev-mini-selection-manifest",
        "dev.json",
        "--output-root",
        "output",
    ]
    abo = build_parser().parse_args(["abo", *common])
    assert (abo.include_count, abo.reserve_count, abo.cross_intent_count) == (
        340,
        8,
        25,
    )
    rpc = build_parser().parse_args(["rpc", *common])
    assert (rpc.include_count, rpc.reserve_count) == (105, 4)
    overridden = build_parser().parse_args(
        [
            "abo",
            *common,
            "--include-count",
            "17",
            "--reserve-count",
            "3",
            "--cross-intent-count",
            "9",
        ]
    )
    assert (
        overridden.include_count,
        overridden.reserve_count,
        overridden.cross_intent_count,
    ) == (17, 3, 9)


def _image_bytes(seed: int, image_format: str = "PNG") -> bytes:
    output = BytesIO()
    image = Image.new(
        "RGB",
        (18, 14),
        color=((seed * 29) % 251, (seed * 47) % 251, (seed * 71) % 251),
    )
    for index in range(14):
        image.putpixel(
            ((seed + index * 3) % 18, index),
            ((seed * 13) % 251, (index * 17) % 251, (seed * 19) % 251),
        )
    image.save(output, format=image_format)
    return output.getvalue()


def _add_tar_bytes(
    archive: tarfile.TarFile,
    name: str,
    content: bytes,
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mtime = 0
    archive.addfile(member, BytesIO(content))


def _write_abo_archives(raw_root: Path) -> dict[str, bytes]:
    image_path = raw_root / Path(*_ABO_IMAGES.split("/"))
    listing_path = raw_root / Path(*_ABO_LISTINGS.split("/"))
    image_path.parent.mkdir(parents=True)
    images: dict[str, bytes] = {}
    metadata_rows: list[tuple[str, str]] = []
    listings: list[dict[str, object]] = []
    for item_index in range(8):
        image_ids: list[str] = []
        for role_index in range(2):
            image_id = f"IMG{item_index:02d}{role_index}"
            relative = f"{item_index:02x}/{item_index * 2 + role_index:08x}.png"
            content = _image_bytes(item_index * 2 + role_index + 1)
            images[image_id] = content
            image_ids.append(image_id)
            metadata_rows.append((image_id, relative))
        listings.append(
            {
                "item_id": f"ITEM_{item_index:02d}",
                "main_image_id": image_ids[0],
                "other_image_id": [image_ids[1]],
            }
        )
    with tarfile.open(image_path, "w") as archive:
        for image_id, relative in metadata_rows:
            _add_tar_bytes(
                archive,
                f"images/small/{relative}",
                images[image_id],
            )
        metadata = "image_id,height,width,path\n" + "".join(
            f"{image_id},14,18,{relative}\n"
            for image_id, relative in metadata_rows
        )
        _add_tar_bytes(
            archive,
            "images/metadata/images.csv.gz",
            gzip.compress(metadata.encode(), mtime=0),
        )
    listing_bytes = b"".join(
        json.dumps(row, sort_keys=True).encode() + b"\n" for row in listings
    )
    with tarfile.open(listing_path, "w") as archive:
        _add_tar_bytes(
            archive,
            "listings/metadata/listings_0.json.gz",
            gzip.compress(listing_bytes, mtime=0),
        )
    return images


def _write_rpc_archive(raw_root: Path) -> dict[int, bytes]:
    archive_path = raw_root / Path(*_RPC_ARCHIVE.split("/"))
    archive_path.parent.mkdir(parents=True)
    images = {image_id: _image_bytes(image_id + 100) for image_id in range(10, 20)}
    annotation = {
        "images": [
            {
                "id": image_id,
                "file_name": f"scene-{image_id}.png",
                "width": 18,
                "height": 14,
            }
            for image_id in images
        ],
        "categories": [
            {"id": 1, "name": "bottle"},
            {"id": 2, "name": "carton"},
        ],
        "annotations": [
            row
            for image_id in images
            for row in (
                {
                    "id": image_id * 10 + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [0, 0, 5, 6],
                },
                {
                    "id": image_id * 10 + 2,
                    "image_id": image_id,
                    "category_id": 2,
                    "bbox": [7, 2, 6, 7],
                },
            )
        ],
    }
    annotation_bytes = json.dumps(annotation, sort_keys=True).encode()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("instances_val2019.json", annotation_bytes)
        archive.writestr(
            f"{_RPC_MIRROR}/instances_val2019.json", annotation_bytes
        )
        for image_id, content in images.items():
            member = f"val2019/scene-{image_id}.png"
            archive.writestr(member, content)
            archive.writestr(f"{_RPC_MIRROR}/{member}", content)
    return images


def _write_lock(
    *,
    raw_root: Path,
    tmp_path: Path,
    source_id: str,
    source_revision: str,
    scope_id: str,
    logical_paths: tuple[str, ...],
) -> tuple[Path, str]:
    scope = build_artifact_scope(
        raw_root,
        scope_id=scope_id,
        mode="explicit_files",
        paths=logical_paths,
    )
    identities = []
    for logical_path in logical_paths:
        path = raw_root / Path(*logical_path.split("/"))
        content = path.read_bytes()
        identities.append(
            AcquisitionIdentity(
                logical_path=logical_path,
                url=f"https://example.invalid/{logical_path}",
                bytes=len(content),
                local_sha256=sha256_bytes(content),
            )
        )
    lock = RequiredSourceLock(
        source_id=source_id,
        source_revision=source_revision,
        lock_plan_sha256="a" * 64,
        artifact_scopes=(scope,),
        acquisition_identities=tuple(identities),
    )
    path = tmp_path / f"{source_id}.source-lock.json"
    content = canonical_json_bytes(lock.model_dump(mode="json"))
    path.write_bytes(content)
    return path, sha256_bytes(content)


def _write_source_policy(path: Path) -> None:
    policy = PortfolioCoreSourcePolicyDocument(
        sources=(
            PortfolioCoreSourcePolicy(
                source_id="abo",
                availability="ready",
                allowed_pools=("exact_match",),
                note="Fixture ABO Portfolio policy.",
            ),
            PortfolioCoreSourcePolicy(
                source_id="rpc",
                availability="ready",
                allowed_pools=("multi_product",),
                note="Fixture RPC Portfolio policy.",
            ),
        )
    )
    path.write_bytes(canonical_json_bytes(policy))


def _write_dev_manifest(
    path: Path,
    *,
    abo_images: dict[str, bytes],
    rpc_images: dict[int, bytes],
) -> dict[str, object]:
    rows = [
        {
            "source_dataset": "abo",
            "source_record_id": "listing:ITEM_00/image:IMG000",
            "product_id": "abo:ITEM_00",
            "image_sha256": sha256_bytes(abo_images["IMG000"]),
        },
        {
            "source_dataset": "abo",
            "source_record_id": "ITEM_01:IMG010",
            "product_id": None,
            "image_sha256": sha256_bytes(abo_images["IMG010"]),
        },
        {
            "source_dataset": "rpc",
            "source_record_id": "val2019:10",
            "product_id": None,
            "image_sha256": sha256_bytes(rpc_images[10]),
        },
    ]
    manifest = {
        "asset_count": len(rows),
        "formal_eligible": False,
        "formal_status": "non_formal",
        "schema_version": 1,
        "selections": rows,
        "track": "portfolio",
    }
    path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def _fixture_inputs(tmp_path: Path) -> dict[str, object]:
    raw_root = tmp_path / "raw"
    abo_images = _write_abo_archives(raw_root)
    rpc_images = _write_rpc_archive(raw_root)
    abo_lock, abo_lock_sha256 = _write_lock(
        raw_root=raw_root,
        tmp_path=tmp_path,
        source_id="abo",
        source_revision="abo-fixture-v1",
        scope_id="compact_archives",
        logical_paths=(_ABO_IMAGES, _ABO_LISTINGS),
    )
    rpc_lock, rpc_lock_sha256 = _write_lock(
        raw_root=raw_root,
        tmp_path=tmp_path,
        source_id="rpc",
        source_revision="rpc-fixture-v1",
        scope_id="kaggle_archive",
        logical_paths=(_RPC_ARCHIVE,),
    )
    policy = tmp_path / "source-policy.json"
    _write_source_policy(policy)
    dev_manifest = tmp_path / "dev-selection-manifest.json"
    _write_dev_manifest(
        dev_manifest,
        abo_images=abo_images,
        rpc_images=rpc_images,
    )
    return {
        "abo_images": abo_images,
        "abo_lock": abo_lock,
        "abo_lock_sha256": abo_lock_sha256,
        "dev_manifest": dev_manifest,
        "policy": policy,
        "raw_root": raw_root,
        "rpc_images": rpc_images,
        "rpc_lock": rpc_lock,
        "rpc_lock_sha256": rpc_lock_sha256,
    }


def test_abo_selected_adapter_resumes_excludes_dev_and_binds_cross_intents(
    tmp_path: Path,
) -> None:
    fixture = _fixture_inputs(tmp_path)
    output = tmp_path / "abo-selected"
    first = build_abo_portfolio_core_candidates(
        raw_root=fixture["raw_root"],
        source_lock_path=fixture["abo_lock"],
        expected_source_lock_sha256=fixture["abo_lock_sha256"],
        source_policy_path=fixture["policy"],
        dev_mini_selection_manifest=fixture["dev_manifest"],
        output_root=output,
        include_count=3,
        reserve_count=1,
        cross_intent_count=2,
        max_items=2,
    )
    assert not first.complete
    assert first.checkpoint_count == 2
    assert not (output / "candidates.jsonl").exists()

    resumed = build_abo_portfolio_core_candidates(
        raw_root=fixture["raw_root"],
        source_lock_path=fixture["abo_lock"],
        expected_source_lock_sha256=fixture["abo_lock_sha256"],
        source_policy_path=fixture["policy"],
        dev_mini_selection_manifest=fixture["dev_manifest"],
        output_root=output,
        include_count=3,
        reserve_count=1,
        cross_intent_count=2,
    )
    assert resumed.complete
    assert resumed.published_this_call == 2
    candidates, _ = load_candidate_inventory(resumed.candidates_path)
    assert [item.selection for item in candidates] == [
        "include",
        "include",
        "include",
        "reserve",
    ]
    assert len({item.expected_sha256 for item in candidates}) == 4
    assert all(item.draft.product_id != "abo:ITEM_00" for item in candidates)
    assert sha256_bytes(fixture["abo_images"]["IMG010"]) not in {
        item.expected_sha256 for item in candidates
    }
    assert [len(item.capability_bindings) for item in candidates] == [3, 3, 1, 1]
    preflight = preflight_portfolio_core_assets(
        inventory_path=resumed.candidates_path,
        source_policy_path=fixture["policy"],
        source_roots={"abo": output},
    )
    assert preflight.ready_candidate_count == 3
    assert preflight.blocked_candidates == {}
    assert preflight.selected_unique_content_counts == {"exact_match": 3}
    for candidate in candidates:
        content = (output / candidate.source_local_path).read_bytes()
        assert len(content) == candidate.expected_bytes
        assert sha256_bytes(content) == candidate.expected_sha256
        with Image.open(BytesIO(content)) as image:
            image.verify()

    repeated = build_abo_portfolio_core_candidates(
        raw_root=fixture["raw_root"],
        source_lock_path=fixture["abo_lock"],
        expected_source_lock_sha256=fixture["abo_lock_sha256"],
        source_policy_path=fixture["policy"],
        dev_mini_selection_manifest=fixture["dev_manifest"],
        output_root=output,
        include_count=3,
        reserve_count=1,
        cross_intent_count=2,
    )
    assert repeated.complete
    assert repeated.published_this_call == 0


def test_selected_adapter_rejects_output_beneath_raw_root(tmp_path: Path) -> None:
    fixture = _fixture_inputs(tmp_path)
    output = fixture["raw_root"] / "derived-output"
    with pytest.raises(
        PortfolioCoreSourceAdapterError,
        match="must be disjoint from RAW/source root",
    ):
        build_abo_portfolio_core_candidates(
            raw_root=fixture["raw_root"],
            source_lock_path=fixture["abo_lock"],
            expected_source_lock_sha256=fixture["abo_lock_sha256"],
            source_policy_path=fixture["policy"],
            dev_mini_selection_manifest=fixture["dev_manifest"],
            output_root=output,
            include_count=3,
            reserve_count=1,
            cross_intent_count=2,
        )
    assert not output.exists()


def test_rpc_selected_adapter_resumes_and_rejects_published_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture_inputs(tmp_path)
    output = tmp_path / "rpc-selected"
    first = build_rpc_portfolio_core_candidates(
        raw_root=fixture["raw_root"],
        source_lock_path=fixture["rpc_lock"],
        expected_source_lock_sha256=fixture["rpc_lock_sha256"],
        source_policy_path=fixture["policy"],
        dev_mini_selection_manifest=fixture["dev_manifest"],
        output_root=output,
        include_count=3,
        reserve_count=1,
        max_items=1,
    )
    assert not first.complete
    assert first.checkpoint_count == 1

    resumed = build_rpc_portfolio_core_candidates(
        raw_root=fixture["raw_root"],
        source_lock_path=fixture["rpc_lock"],
        expected_source_lock_sha256=fixture["rpc_lock_sha256"],
        source_policy_path=fixture["policy"],
        dev_mini_selection_manifest=fixture["dev_manifest"],
        output_root=output,
        include_count=3,
        reserve_count=1,
    )
    assert resumed.complete
    candidates, _ = load_candidate_inventory(resumed.candidates_path)
    assert len(candidates) == 4
    assert all(item.pool == "multi_product" for item in candidates)
    assert all(
        item.capability_bindings[0].canonical_capability == "product.multi_search"
        for item in candidates
    )
    assert all(item.draft.source_record_id != "val2019:10" for item in candidates)
    assert sha256_bytes(fixture["rpc_images"][10]) not in {
        item.expected_sha256 for item in candidates
    }
    preflight = preflight_portfolio_core_assets(
        inventory_path=resumed.candidates_path,
        source_policy_path=fixture["policy"],
        source_roots={"rpc": output},
    )
    assert preflight.ready_candidate_count == 3
    assert preflight.blocked_candidates == {}
    assert preflight.selected_unique_content_counts == {"multi_product": 3}

    damaged = output / candidates[0].source_local_path
    damaged.write_bytes(b"not-an-image")
    with pytest.raises(
        PortfolioCoreSourceAdapterError,
        match="published candidate bytes conflict",
    ):
        build_rpc_portfolio_core_candidates(
            raw_root=fixture["raw_root"],
            source_lock_path=fixture["rpc_lock"],
            expected_source_lock_sha256=fixture["rpc_lock_sha256"],
            source_policy_path=fixture["policy"],
            dev_mini_selection_manifest=fixture["dev_manifest"],
            output_root=output,
            include_count=3,
            reserve_count=1,
        )
