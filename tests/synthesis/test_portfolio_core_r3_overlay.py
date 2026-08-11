"""Focused synthetic tests for the independent Portfolio Core r3 overlay."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillchain.schemas import ConversationTurn, LabelDecision, Query
from skillchain.synthesis.portfolio_core_r3_overlay import (
    PortfolioCoreR3OverlayError,
    R3OverlayConflictError,
    R3OverlayPublicationError,
    R3RepairDraftBatch,
    R3RepairManifest,
    R3ReplacementDraft,
    build_portfolio_core_r3_overlay_payload,
    build_r3_repair_plan,
    publish_portfolio_core_r3_overlay,
)
from skillchain.synthesis.store import canonical_jsonl_bytes, sha256_bytes


def _query(index: int) -> Query:
    text = f"查询{chr(0x4E00 + index)}"
    return Query(
        schema_version=2,
        taxonomy_version="taxonomy-v1",
        task_spec_version="task-v1",
        query_id=f"q-{index + 1:04d}",
        asset_id=f"asset-{index + 1:04d}",
        image_path=f"synthetic/{index + 1:04d}.jpg",
        leakage_group_id=f"group-{index + 1:04d}",
        template_family="synthetic",
        generator_batch_id=f"parent-{index // 25 + 1:03d}",
        text=text,
        turns=[ConversationTurn(role="user", content=text)],
        canonical_intent="exact_match",
        canonical_capability="product.exact_match",
        acceptable_capabilities=["product.exact_match"],
        requires_card=True,
        split="opt_pool",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="synthetic-fixture",
                canonical_intent="exact_match",
                canonical_capability="product.exact_match",
                acceptable_capabilities=["product.exact_match"],
                reason="合成测试夹具",
            )
        ],
    )


@pytest.fixture(scope="module")
def source_queries() -> tuple[Query, ...]:
    """A complete typed-only fixture; no disk corpus or image is opened."""

    return tuple(_query(index) for index in range(1500))


@pytest.fixture(scope="module")
def repair_inputs(source_queries: tuple[Query, ...]):
    replacement_source = source_queries[:250]
    batches = tuple(
        (
            f"repair-{batch_index + 1:03d}",
            tuple(
                query.query_id
                for query in replacement_source[
                    batch_index * 25 : (batch_index + 1) * 25
                ]
            ),
        )
        for batch_index in range(10)
    )
    plan = build_r3_repair_plan(batches)
    drafts = tuple(
        R3RepairDraftBatch(
            batch_id=batch_id,
            drafts=tuple(
                R3ReplacementDraft(
                    query_id=query_id,
                    turns=(
                        ConversationTurn(
                            role="user",
                            content=f"修订{chr(0x7000 + ordinal * 25 + position)}",
                        ),
                    ),
                )
                for position, query_id in enumerate(query_ids)
            ),
        )
        for ordinal, (batch_id, query_ids) in enumerate(batches)
    )
    return plan, drafts


def _payload(source_queries, repair_inputs, *, run_id: str = "core-r3-overlay"):
    plan, drafts = repair_inputs
    source_bytes = canonical_jsonl_bytes(source_queries)
    return build_portfolio_core_r3_overlay_payload(
        source_queries_bytes=source_bytes,
        expected_r2_source_sha256=sha256_bytes(source_bytes),
        repair_plan=plan,
        draft_batches=drafts,
        run_id=run_id,
    )


def _replace_first_draft(
    repair_inputs,
    *,
    content: str,
):
    plan, drafts = repair_inputs
    first_batch = drafts[0]
    first_draft = first_batch.drafts[0]
    changed_draft = first_draft.model_copy(
        update={"turns": (ConversationTurn(role="user", content=content),)}
    )
    changed_batch = first_batch.model_copy(
        update={"drafts": (changed_draft, *first_batch.drafts[1:])}
    )
    return plan, (changed_batch, *drafts[1:])


def test_success_materializes_and_publishes_a_complete_create_only_overlay(
    tmp_path: Path,
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    payload = _payload(source_queries, repair_inputs)

    assert len(payload.final_queries) == 1500
    assert payload.final_queries[0].query_id == source_queries[0].query_id
    assert payload.final_queries[249].text != source_queries[249].text
    assert payload.final_queries[250] == source_queries[250]
    for source, repaired in zip(
        source_queries[:250], payload.final_queries[:250], strict=True
    ):
        for field_name, source_value in source.model_dump(mode="json").items():
            if field_name not in {"text", "turns"}:
                assert repaired.model_dump(mode="json")[field_name] == source_value

    published = publish_portfolio_core_r3_overlay(payload, run_parent=tmp_path)
    assert published.created is True
    assert published.run_root == tmp_path / "core-r3-overlay"
    assert len(list((published.run_root / "parent-batches").glob("parent-*.jsonl"))) == 60
    assert len(list((published.run_root / "repair-batches").glob("repair-*"))) == 10
    assert len((published.run_root / "queries.jsonl").read_bytes().splitlines()) == 1500
    source_lines = canonical_jsonl_bytes(source_queries).splitlines(keepends=True)
    final_lines = (published.run_root / "queries.jsonl").read_bytes().splitlines(
        keepends=True
    )
    assert final_lines[250:] == source_lines[250:]  # 1,250 canonical byte carry-forwards

    language_rows = [
        json.loads(line)
        for line in (published.run_root / "language-validation.jsonl").read_bytes().splitlines()
    ]
    assert len(language_rows) == 1500
    assert language_rows[0]["disposition"] == "replacement"
    assert language_rows[250]["disposition"] == "source_carry_forward"
    assert "text" not in language_rows[0]
    manifest = json.loads((published.run_root / "repair-manifest.json").read_bytes())
    assert manifest["replacement_query_count"] == 250
    assert manifest["carry_forward_query_count"] == 1250
    assert manifest["parent_batch_count"] == 60
    assert manifest["repair_batch_count"] == 10


@pytest.mark.parametrize("bad_content", ["修订A", "修订Ａ"])
def test_rejects_ascii_and_nfkc_ascii_in_replacement_turns(
    source_queries: tuple[Query, ...],
    repair_inputs,
    bad_content: str,
) -> None:
    changed_inputs = _replace_first_draft(repair_inputs, content=bad_content)
    with pytest.raises(PortfolioCoreR3OverlayError, match="ASCII letter"):
        _payload(source_queries, changed_inputs)


def test_rejects_global_normalized_final_text_duplicates(
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    changed_inputs = _replace_first_draft(repair_inputs, content=source_queries[500].text)
    with pytest.raises(PortfolioCoreR3OverlayError, match="duplicate normalized text"):
        _payload(source_queries, changed_inputs)


def test_drafts_are_body_only_and_replacement_metadata_remains_source_bound(
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        R3ReplacementDraft.model_validate(
            {
                "query_id": source_queries[0].query_id,
                "turns": [{"role": "user", "content": "修订甲"}],
                "asset_id": "metadata-drift",
            }
        )

    payload = _payload(source_queries, repair_inputs)
    source = source_queries[0].model_dump(mode="json")
    repaired = payload.final_queries[0].model_dump(mode="json")
    assert {key: repaired[key] for key in repaired if key not in {"text", "turns"}} == {
        key: source[key] for key in source if key not in {"text", "turns"}
    }


def test_rejects_draft_that_changes_the_r2_turn_role_shape(
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    plan, drafts = repair_inputs
    first_batch = drafts[0]
    first_draft = first_batch.drafts[0]
    changed_draft = first_draft.model_copy(
        update={
            "turns": (
                ConversationTurn(role="user", content="改写甲"),
                ConversationTurn(role="assistant", content="补充乙"),
                ConversationTurn(role="user", content="改写丙"),
            )
        }
    )
    changed_batch = first_batch.model_copy(
        update={"drafts": (changed_draft, *first_batch.drafts[1:])}
    )

    with pytest.raises(PortfolioCoreR3OverlayError, match="role shape"):
        _payload(source_queries, (plan, (changed_batch, *drafts[1:])))


def test_fails_closed_when_caller_source_rows_drift_from_frozen_r2_hash(
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    original_bytes = canonical_jsonl_bytes(source_queries)
    drifted = list(source_queries)
    drifted[777] = drifted[777].model_copy(update={"asset_id": "asset-drifted"})
    plan, drafts = repair_inputs

    with pytest.raises(PortfolioCoreR3OverlayError, match="source SHA-256 drifted"):
        build_portfolio_core_r3_overlay_payload(
            source_queries=drifted,
            expected_r2_source_sha256=sha256_bytes(original_bytes),
            repair_plan=plan,
            draft_batches=drafts,
            run_id="core-r3-drift",
        )


def test_existing_target_is_idempotent_only_when_every_byte_is_identical(
    tmp_path: Path,
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    payload = _payload(source_queries, repair_inputs, run_id="core-r3-conflict")
    first = publish_portfolio_core_r3_overlay(payload, run_parent=tmp_path)
    second = publish_portfolio_core_r3_overlay(payload, run_parent=tmp_path)

    assert first.created is True
    assert second.created is False
    queries_path = first.run_root / "queries.jsonl"
    tampered = queries_path.read_bytes() + b" "
    queries_path.write_bytes(tampered)
    with pytest.raises(R3OverlayConflictError, match="content differs"):
        publish_portfolio_core_r3_overlay(payload, run_parent=tmp_path)
    assert queries_path.read_bytes() == tampered


def test_rejects_root_manifest_that_omits_an_emitted_artifact(
    tmp_path: Path,
    source_queries: tuple[Query, ...],
    repair_inputs,
) -> None:
    payload = _payload(source_queries, repair_inputs, run_id="core-r3-root-binding")
    unsigned_manifest = payload.manifest.model_dump(mode="json")
    unsigned_manifest.pop("manifest_sha256")
    unsigned_manifest["files"] = unsigned_manifest["files"][:-1]
    unsigned_manifest["file_count"] = len(unsigned_manifest["files"])
    incomplete_manifest = R3RepairManifest.create(**unsigned_manifest)
    incomplete_artifacts = tuple(
        replace(artifact, content=incomplete_manifest.canonical_bytes())
        if artifact.relative_path == "repair-manifest.json"
        else artifact
        for artifact in payload.artifacts
    )
    incomplete_payload = replace(
        payload,
        manifest=incomplete_manifest,
        artifacts=incomplete_artifacts,
    )

    with pytest.raises(R3OverlayPublicationError, match="root binding drifted"):
        publish_portfolio_core_r3_overlay(incomplete_payload, run_parent=tmp_path)
    assert not (tmp_path / payload.run_id).exists()
