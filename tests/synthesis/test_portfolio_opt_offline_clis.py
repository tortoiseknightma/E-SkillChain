from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import scripts.build_portfolio_hybrid_router as hybrid_cli
import scripts.build_portfolio_opt_folds as folds_cli
from skillchain.evolution.hybrid_router import (
    AUTHORIZATION_STATUS,
    load_hybrid_router_bundle,
)
from skillchain.synthesis.portfolio_core_audit import build_core_r2_audit_sample
from skillchain.synthesis.portfolio_core_publication import (
    CoreR2PublicationBundle,
    build_core_r2_publication_payload,
)
from skillchain.synthesis.portfolio_core_r2 import build_r2_core_bridge
from skillchain.synthesis.portfolio_core_selection import (
    build_core_r2_creator_selection,
)
from skillchain.synthesis.portfolio_opt_folds import (
    LegacySelectionConflictAudit,
    OptFoldAssignment,
    OptFoldManifest,
    canonical_opt_fold_mapping_bytes,
    canonical_opt_projection_bytes,
)
from skillchain.synthesis.store import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
)


CAPABILITIES = (
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "knowledge.visual_encyclopedia",
    "utility.document_reading",
    "utility.recipe_guidance",
)


@pytest.fixture(scope="module")
def published_r2_artifacts() -> dict[str, bytes]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_r2_planning.py"))
    )
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
    legacy = build_core_r2_creator_selection(
        result,
        bridge.realism_sidecar,
        trusted_plan_sha256=plan_sha256,
        seed=20260805,
    )
    payload = build_core_r2_publication_payload(
        CoreR2PublicationBundle(
            r2_plan=result,
            bridge=bridge,
            audit_sample=audit_sample,
            creator_selection=legacy,
            trusted_parent_r1_sha256="d" * 64,
            supersedes_run="core-r1-frozen",
            run_id="core-r2-offline-cli-test",
        )
    )
    return {item.relative_path: item.content for item in payload.artifacts}


def _query_row(index: int) -> dict[str, object]:
    batch_number = index // 25
    capability = CAPABILITIES[index % len(CAPABILITIES)]
    query_id = f"opt-{index + 1:04d}"
    text = f"Please route {capability} request {index + 1}"
    return {
        "schema_version": 2,
        "taxonomy_version": "ecommerce-mvp-taxonomy-v0",
        "task_spec_version": "ecommerce-task-spec-v1",
        "query_id": query_id,
        "asset_id": f"asset-{index + 1:04d}",
        "image_path": f"query_images/exact_match/{query_id}.jpg",
        "leakage_group_id": f"leakage-{index + 1:04d}",
        "boundary_group_id": None,
        "template_family": f"template-{batch_number:02d}",
        "generator_batch_id": f"batch-{batch_number:02d}",
        "text": text,
        "turns": [{"role": "user", "content": text}],
        "canonical_intent": "exact_match",
        "canonical_capability": capability,
        "acceptable_capabilities": [capability],
        "is_boundary": False,
        "boundary_strategy": None,
        "requires_card": False,
        "episode": "t0",
        "split": "opt_pool",
        "label_status": "auto",
        "label_provenance": [
            {
                "decision_type": "constructed",
                "annotator_kind": "planner",
                "annotator_id": "offline-cli-test",
                "canonical_intent": "exact_match",
                "canonical_capability": capability,
                "acceptable_capabilities": [capability],
                "reason": None,
                "legacy_skill_slug": None,
                "source_artifact_sha256": None,
                "decided_at": None,
            }
        ],
        "synth_provider": None,
        "synth_model": None,
        "synthesis_batch_id": None,
        "synthesis_prompt_id": None,
        "seed_set_sha256": None,
    }


def _write_query_artifact(path: Path) -> tuple[bytes, str]:
    # This intentionally malformed non-opt row proves that neither CLI typed-
    # loads val/test content.  Only its split discriminator is inspected.
    rows = [
        {"split": "val", "private_field_that_is_not_a_Query": "ignored"},
        *(_query_row(index) for index in range(800)),
    ]
    content = canonical_jsonl_bytes(rows)
    path.write_bytes(content)
    return content, sha256_bytes(content)


def _write_fold_bundle(
    root: Path,
    *,
    source_queries_sha256: str,
    queries,
) -> tuple[str, str]:
    root.mkdir()
    assignments: list[OptFoldAssignment] = []
    for index, query in enumerate(queries):
        batch_number = index // 25
        fold_number = batch_number // 8
        assignments.append(
            OptFoldAssignment(
                query_id=query.query_id,
                atomic_batch_id=query.generator_batch_id,
                fold_id=f"fold-{fold_number:02d}",
                role="replay" if fold_number == 0 else "discovery",
            )
        )
    assignments_tuple = tuple(assignments)
    mapping_bytes = canonical_opt_fold_mapping_bytes(assignments_tuple)
    projection = hybrid_cli._query_fold_projection(queries)
    discovery = [item.query_id for item in assignments if item.role == "discovery"]
    replay = [item.query_id for item in assignments if item.role == "replay"]
    manifest = OptFoldManifest(
        seed=20260808,
        source_queries_sha256=source_queries_sha256,
        plan_sha256="a" * 64,
        opt_projection_sha256=sha256_bytes(
            canonical_opt_projection_bytes(projection)
        ),
        policy_sha256="b" * 64,
        mapping_sha256=sha256_bytes(mapping_bytes),
        discovery_query_ids_sha256=hybrid_cli._ids_sha256(discovery),
        replay_query_ids_sha256=hybrid_cli._ids_sha256(replay),
        legacy_conflict_audit=LegacySelectionConflictAudit(
            selection_bytes_sha256="c" * 64,
            selected_query_count=240,
            touched_atomic_batch_count=32,
            closure_query_count=800,
            remaining_replay_count=0,
            incompatible_with_nonempty_replay=True,
        ),
    )
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "fold-mapping.jsonl").write_bytes(mapping_bytes)
    (root / "fold-manifest.json").write_bytes(manifest_bytes)
    return sha256_bytes(manifest_bytes), sha256_bytes(mapping_bytes)


def test_fold_cli_projects_only_opt_metadata_and_binds_source_hash(tmp_path: Path):
    query_path = tmp_path / "queries.jsonl"
    _, digest = _write_query_artifact(query_path)

    projection = folds_cli.load_opt_projection(
        query_path,
        expected_file_sha256=digest,
    )

    assert len(projection) == 800
    assert projection[0].query_id == "opt-0001"
    assert not hasattr(projection[0], "turns")
    with pytest.raises(ValueError, match="caller-supplied SHA"):
        folds_cli.load_opt_projection(
            query_path,
            expected_file_sha256="f" * 64,
        )


def test_published_r2_loader_is_independent_of_current_taskspec_and_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_r2_artifacts: dict[str, bytes],
):
    publication_root = tmp_path / "published-r2"
    for relative_path, content in published_r2_artifacts.items():
        destination = publication_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    import skillchain.synthesis.planning as planning
    import skillchain.task_spec as task_spec

    monkeypatch.setattr(task_spec, "MVP_TASK_SPEC_V1_SHA256", "f" * 64)
    monkeypatch.setattr(planning, "MVP_TASK_SPEC_V1_SHA256", "f" * 64)
    monkeypatch.setattr(planning, "PHASE3_TASK_SPEC_SHA256", "f" * 64)

    def forbidden_assignment_loader(*args, **kwargs):
        raise AssertionError("published replay must not typed-load assignments")

    monkeypatch.setattr(
        planning, "load_capability_assignments", forbidden_assignment_loader
    )
    loaded = folds_cli.load_published_r2_selection_context(
        publication_root,
        expected_manifest_file_sha256=sha256_bytes(
            published_r2_artifacts["pre-generation-manifest.json"]
        ),
    )

    assert len(loaded.context.plan.queries) == 1500
    assert len(loaded.legacy_selection.entries) == 240
    parser_dests = {action.dest for action in folds_cli.build_parser()._actions}
    assert "parent_assignments" not in parser_dests
    assert "target_assignments" not in parser_dests


def test_fold_cli_main_publishes_create_only_without_rebuilding_in_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        folds_cli,
        "_bundle_files",
        lambda arguments: {
            "fold-manifest.json": b'{"complete":true}\n',
            "fold-mapping.jsonl": b'{"query_id":"q"}\n',
            "creator-v2-index.jsonl": b'{"plan_id":"q"}\n',
        },
    )
    output = tmp_path / "folds"
    placeholder = tmp_path / "placeholder"
    hash_value = "a" * 64
    argv = [
        "--r2-publication-root",
        str(placeholder),
        "--expected-r2-publication-manifest-sha256",
        hash_value,
        "--queries",
        str(placeholder),
        "--expected-queries-sha256",
        hash_value,
        "--output-dir",
        str(output),
    ]

    assert folds_cli.main(argv) == 0
    original = (output / "fold-manifest.json").read_bytes()
    assert folds_cli.main(argv) == 2
    assert (output / "fold-manifest.json").read_bytes() == original


def test_fold_cli_staging_is_removed_when_atomic_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "folds"

    def fail_publish(staging, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(folds_cli, "atomic_publish_new_directory", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        folds_cli.publish_create_only(output, {"one.json": b"{}\n"})

    assert not output.exists()
    assert not list(tmp_path.glob(".folds.staging-*"))


def test_hybrid_cli_runs_opt800_offline_and_refuses_existing_output(
    tmp_path: Path,
):
    query_path = tmp_path / "queries.jsonl"
    _, source_sha256 = _write_query_artifact(query_path)
    queries = hybrid_cli.load_opt_queries(
        query_path,
        expected_file_sha256=source_sha256,
    )
    folds = tmp_path / "folds"
    manifest_sha256, mapping_sha256 = _write_fold_bundle(
        folds,
        source_queries_sha256=source_sha256,
        queries=queries,
    )
    output = tmp_path / "router"
    argv = [
        "--queries",
        str(query_path),
        "--expected-queries-sha256",
        source_sha256,
        "--fold-dir",
        str(folds),
        "--expected-fold-manifest-sha256",
        manifest_sha256,
        "--expected-fold-mapping-sha256",
        mapping_sha256,
        "--output-dir",
        str(output),
    ]

    assert hybrid_cli.main(argv) == 0
    loaded = load_hybrid_router_bundle(
        output,
        expected_manifest_file_sha256=sha256_bytes(
            (output / "model-manifest.json").read_bytes()
        ),
        expected_source_queries_sha256=source_sha256,
        expected_fold_plan_sha256=mapping_sha256,
    )
    assert loaded.manifest.authorization_status == AUTHORIZATION_STATUS
    assert len(loaded.predictions) == 800
    original_manifest = (output / "model-manifest.json").read_bytes()
    assert hybrid_cli.main(argv) == 2
    assert (output / "model-manifest.json").read_bytes() == original_manifest


def test_hybrid_cli_rejects_fold_mapping_hash_drift(tmp_path: Path):
    query_path = tmp_path / "queries.jsonl"
    _, source_sha256 = _write_query_artifact(query_path)
    queries = hybrid_cli.load_opt_queries(
        query_path,
        expected_file_sha256=source_sha256,
    )
    folds = tmp_path / "folds"
    manifest_sha256, _ = _write_fold_bundle(
        folds,
        source_queries_sha256=source_sha256,
        queries=queries,
    )

    with pytest.raises(ValueError, match="caller-supplied SHA"):
        hybrid_cli.load_verified_fold_mapping(
            folds,
            expected_manifest_sha256=manifest_sha256,
            expected_mapping_sha256="f" * 64,
            source_queries_sha256=source_sha256,
            queries=queries,
        )
