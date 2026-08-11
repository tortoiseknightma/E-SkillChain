from __future__ import annotations

import json
from pathlib import Path

from skillchain import config
from skillchain.data.raw_download import load_manifest, resolve_profile


DOWNLOAD_MANIFEST = (
    config.ROOT / "specs" / "data_sources" / "raw-download-profiles-v1.json"
)
PORTFOLIO = (
    config.ROOT / "specs" / "data_sources" / "ecommerce-mvp-source-portfolio-v1.json"
)
RPC_ACQUISITION = (
    config.ROOT / "specs" / "data_sources" / "rpc-kaggle-acquisition-v1.json"
)


def _dataset_ids(manifest: dict, profile: str) -> set[str]:
    return {
        manifest["sources"][source_id]["dataset_id"]
        for source_id in resolve_profile(manifest, profile)
    }


def test_mvp_profile_covers_non_core_portfolio_sources() -> None:
    manifest = load_manifest(DOWNLOAD_MANIFEST)
    portfolio = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    required = {
        source["source_id"]
        for source in portfolio["sources"]
        if source["tier"] in {"mvp", "mvp_secondary", "mvp_language", "challenge"}
    }

    assert required <= _dataset_ids(manifest, "mvp")
    assert "jddc_2_0" not in _dataset_ids(manifest, "mvp")


def test_full_profile_covers_the_complete_project_portfolio() -> None:
    manifest = load_manifest(DOWNLOAD_MANIFEST)
    portfolio = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    required = {
        source["source_id"]
        for source in portfolio["sources"]
        if source["tier"] not in {"retired", "mvp_derived"}
    }

    assert required <= _dataset_ids(manifest, "full")
    assert set(resolve_profile(manifest, "mvp")) < set(
        resolve_profile(manifest, "full")
    )
    assert set(resolve_profile(manifest, "full")) < set(
        resolve_profile(manifest, "full-upstream")
    )


def test_retired_sources_are_recorded_and_never_resolved() -> None:
    manifest = load_manifest(DOWNLOAD_MANIFEST)
    retired = {item["source_id"]: item for item in manifest["retired_sources"]}

    assert retired["jddc_2_0"]["disposition"] == "permanently_abandoned"
    assert retired["u_need"]["disposition"] == "permanently_abandoned"
    for profile in ("mvp", "full", "full-upstream"):
        assert "jddc_2_0" not in resolve_profile(manifest, profile)
        assert "u_need" not in resolve_profile(manifest, profile)


def test_every_source_is_explicitly_downloadable_or_blocked() -> None:
    manifest = load_manifest(DOWNLOAD_MANIFEST)
    for source_id, source in manifest["sources"].items():
        assert source["kind"] in {"artifacts", "command", "manual"}, source_id
        if source["kind"] == "artifacts":
            assert source["artifacts"], source_id
            for artifact in source["artifacts"]:
                assert artifact["path"]
                assert artifact["url"].startswith(("http://", "https://"))
        elif source["kind"] == "command":
            assert source["command"][0] == "{python}", source_id
        else:
            assert source["reason"], source_id
            assert source["instructions"], source_id


def test_local_overrides_replace_a_manual_source(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "rpc": {
                        "kind": "artifacts",
                        "artifacts": [
                            {
                                "path": "rpc/official.zip",
                                "url": "https://example.test/rpc.zip",
                                "bytes": 123,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(DOWNLOAD_MANIFEST, overrides_path=overrides)

    assert manifest["sources"]["rpc"]["kind"] == "artifacts"
    assert manifest["sources"]["rpc"]["artifacts"][0]["bytes"] == 123


def test_rpc_acquisition_receipt_is_pinned_but_not_formal_ready() -> None:
    receipt = json.loads(RPC_ACQUISITION.read_text(encoding="utf-8"))

    assert receipt["dataset_id"] == "rpc"
    assert receipt["source"]["dataset_version"] == 5
    assert receipt["acquisition"]["bytes"] == 27_205_167_166
    assert (
        receipt["acquisition"]["sha256"]
        == "3b7db198a3f36ad86842f754f9845e8907cae8d1b553ec6f734b3ddff087dc79"
    )
    assert receipt["acquisition"]["archive_validation"] == {
        "format": "zip",
        "central_directory_readable": True,
        "entry_count": 167_484,
        "uncompressed_bytes": 31_810_311_622,
        "jpg_entry_count": 167_478,
        "json_entry_count": 6,
        "unsafe_path_count": 0,
    }
    assert receipt["status"] == "locally_verified_raw_not_formal_ready"
    assert not any(receipt["remaining_gates"].values())
