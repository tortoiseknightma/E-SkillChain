"""Focused tmp-path tests for r2 create-only pre-generation publication."""

from __future__ import annotations

import json
import runpy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from skillchain.synthesis.portfolio_core_audit import build_core_r2_audit_sample
from skillchain.synthesis import portfolio_core_publication as publication
from skillchain.synthesis.portfolio_core_publication import (
    CoreR2PublicationBundle,
    PublicationConflictError,
    PublicationSafetyError,
    build_core_r2_publication_payload,
    publish_core_r2_pre_generation_bundle,
)
from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.portfolio_core_selection import build_core_r2_creator_selection


@pytest.fixture(scope="module")
def publication_bundle() -> CoreR2PublicationBundle:
    """Use the complete synthetic r2 layout only; no asset paths are opened."""

    namespace = runpy.run_path(str(Path(__file__).with_name("test_r2_planning.py")))
    result = namespace["_full_r2_fixture"]()
    plan_sha256 = namespace["_plan_sha256"](result.plan)
    bridge = build_r2_core_bridge(
        result,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    audit_sample = build_core_r2_audit_sample(
        result.plan.queries,
        final_split_by_plan_id=result.final_split_by_plan_id,
        realism_sidecar=bridge.realism_sidecar,
        seed=20260805,
    )
    creator_selection = build_core_r2_creator_selection(
        result,
        bridge.realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    return CoreR2PublicationBundle(
        r2_plan=result,
        bridge=bridge,
        audit_sample=audit_sample,
        creator_selection=creator_selection,
        trusted_parent_r1_sha256="d" * 64,
        supersedes_run="core-r1-frozen",
        run_id="core-r2-pre-generation",
    )


def test_payload_has_exact_text_free_pre_generation_paths_and_root_binding(
    publication_bundle: CoreR2PublicationBundle,
) -> None:
    payload = build_core_r2_publication_payload(publication_bundle)
    paths = {item.relative_path for item in payload.artifacts}

    assert paths == {
        "plan/core.json",
        "sidecars/final-splits.jsonl",
        "sidecars/reuse.jsonl",
        "sidecars/split-constraints.json",
        "sidecars/val-interactions.jsonl",
        "authoring/realism.jsonl",
        "authoring/realism-manifest.json",
        "authoring/prompt-recipes.jsonl",
        "authoring/prompt-recipes-manifest.json",
        "audit/index.jsonl",
        "audit/manifest.json",
        "audit/audit.json",
        "stage1/index.jsonl",
        "stage1/manifest.json",
        "stage1/audit.json",
        "pre-generation-manifest.json",
    }
    assert payload.manifest.file_count == 15
    assert {item.relative_path for item in payload.manifest.files} == paths - {
        "pre-generation-manifest.json"
    }
    assert all(
        "author-packet" not in path and "author_packet" not in path
        for path in paths
    )
    assert all("work-order" not in path and "work_order" not in path for path in paths)

    root_bytes = next(
        item.content
        for item in payload.artifacts
        if item.relative_path == "pre-generation-manifest.json"
    )
    root = json.loads(root_bytes)
    assert root["run_id"] == publication_bundle.run_id
    assert root["supersedes_run"] == publication_bundle.supersedes_run
    assert root["trusted_parent_r1_sha256"] == "d" * 64
    assert "pre-generation-manifest.json" not in {
        item["relative_path"] for item in root["files"]
    }


def test_first_publish_is_atomic_and_repeat_is_byte_identical_idempotent(
    tmp_path: Path,
    publication_bundle: CoreR2PublicationBundle,
) -> None:
    payload = build_core_r2_publication_payload(publication_bundle)
    first = publish_core_r2_pre_generation_bundle(
        publication_bundle,
        run_parent=tmp_path,
    )
    second = publish_core_r2_pre_generation_bundle(
        publication_bundle,
        run_parent=tmp_path,
    )

    assert first.created is True
    assert second.created is False
    assert first.run_root == second.run_root == tmp_path / publication_bundle.run_id
    assert not (tmp_path / "active").exists()
    for artifact in payload.artifacts:
        assert (first.run_root / artifact.relative_path).read_bytes() == artifact.content


def test_modified_or_extra_existing_content_is_rejected_without_overwrite(
    tmp_path: Path,
    publication_bundle: CoreR2PublicationBundle,
) -> None:
    published = publish_core_r2_pre_generation_bundle(
        publication_bundle,
        run_parent=tmp_path,
    )
    core_path = published.run_root / "plan/core.json"
    tampered = core_path.read_bytes()[:-1] + b" "
    core_path.write_bytes(tampered)
    with pytest.raises(PublicationConflictError, match="content differs"):
        publish_core_r2_pre_generation_bundle(publication_bundle, run_parent=tmp_path)
    assert core_path.read_bytes() == tampered

    extra_bundle = replace(publication_bundle, run_id="core-r2-extra-file")
    extra_published = publish_core_r2_pre_generation_bundle(
        extra_bundle,
        run_parent=tmp_path,
    )
    extra_path = extra_published.run_root / "unexpected.txt"
    extra_path.write_text("no overwrite", encoding="utf-8")
    with pytest.raises(PublicationConflictError, match="file collection"):
        publish_core_r2_pre_generation_bundle(extra_bundle, run_parent=tmp_path)
    assert extra_path.read_text(encoding="utf-8") == "no overwrite"


def test_link_target_is_rejected(
    tmp_path: Path,
    publication_bundle: CoreR2PublicationBundle,
) -> None:
    linked_target = tmp_path / publication_bundle.run_id
    link_destination = tmp_path / "link-destination"
    link_destination.mkdir()
    try:
        linked_target.symlink_to(link_destination, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows privilege dependent
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(PublicationSafetyError, match="link or reparse"):
        publish_core_r2_pre_generation_bundle(publication_bundle, run_parent=tmp_path)


def test_reparse_detection_is_fail_closed(
    tmp_path: Path,
    publication_bundle: CoreR2PublicationBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "_is_reparse", lambda _: True)
    with pytest.raises(PublicationSafetyError, match="non-reparse real directory"):
        publish_core_r2_pre_generation_bundle(publication_bundle, run_parent=tmp_path)


def test_concurrent_calls_have_one_creator(
    tmp_path: Path,
    publication_bundle: CoreR2PublicationBundle,
) -> None:
    concurrent_bundle = replace(publication_bundle, run_id="core-r2-concurrent")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: publish_core_r2_pre_generation_bundle(
                    concurrent_bundle,
                    run_parent=tmp_path,
                ),
                range(2),
            )
        )
    assert sum(result.created for result in results) == 1
    assert sum(not result.created for result in results) == 1
    assert all(result.run_root == tmp_path / concurrent_bundle.run_id for result in results)
