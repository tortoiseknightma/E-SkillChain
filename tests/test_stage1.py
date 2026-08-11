from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import scripts.prepare_stage1 as prepare_cli
import skillchain.stage1 as stage1
from skillchain.schemas import LabelDecision, Query
from skillchain.stage1 import (
    PAPER_INTENTS,
    S1CreatorPacket,
    Stage1ContractError,
    TrajectoryBundle,
    build_query_selection,
    build_trajectory_bundle,
    load_query_selection,
    load_s1_creator_packet,
    load_trajectory_bundle,
    prepare_stage1,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    parse_canonical_json,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_INPUT = ROOT / "specs" / "authoring" / "authoring-packet-codex-high-v5.json"
AUTHORING_INPUT_FILE_SHA256 = (
    "b389da568575e7b1502923db0a4667e70954233924623856b4f96d6a4fdf9cf1"
)
INTENT_CAPABILITIES = {
    "exact_match": "product.exact_match",
    "multi_product": "product.multi_search",
    "divergent_rec": "product.style_recommendation",
    "encyclopedia": "knowledge.visual_encyclopedia",
    "utility": "utility.document_reading",
}


def _query(
    index: int,
    intent: str,
    *,
    split: str = "opt_pool",
    assistant_marker: str | None = None,
) -> Query:
    capability = INTENT_CAPABILITIES[intent]
    final_text = f"mechanical final user request {index}"
    turns: list[dict[str, str]]
    if assistant_marker is None:
        turns = [{"role": "user", "content": final_text}]
    else:
        turns = [
            {"role": "user", "content": f"mechanical first request {index}"},
            {"role": "assistant", "content": assistant_marker},
            {"role": "user", "content": final_text},
        ]
    return Query(
        schema_version=2,
        taxonomy_version="ecommerce-mvp-taxonomy-v0",
        task_spec_version="ecommerce-task-spec-v1",
        query_id=f"opt-{index:02d}",
        asset_id=f"asset-{index:02d}",
        image_path=f"images/query-{index:02d}.jpg",
        leakage_group_id=f"leakage-{index:02d}",
        template_family=f"family-{index:02d}",
        generator_batch_id="mechanical-stage1-test",
        text=final_text,
        turns=turns,
        canonical_intent=intent,
        canonical_capability=capability,
        acceptable_capabilities=[capability],
        is_boundary=False,
        requires_card=True,
        split=split,
        label_status="auto",
        label_provenance=[
            LabelDecision(
                decision_type="constructed",
                annotator_kind="planner",
                annotator_id="mechanical-stage1-test",
                canonical_intent=intent,
                canonical_capability=capability,
                acceptable_capabilities=[capability],
            )
        ],
    )


def _five_queries() -> tuple[Query, ...]:
    return tuple(
        _query(
            index,
            intent,
            assistant_marker=(
                "ASSISTANT_CLARIFICATION_MUST_BE_PRESERVED" if index == 1 else None
            ),
        )
        for index, intent in enumerate(PAPER_INTENTS, start=1)
    )


def _queries_root(tmp_path: Path, queries: tuple[Query, ...]) -> Path:
    root = tmp_path / "queries"
    root.mkdir()
    (root / "queries.jsonl").write_bytes(
        canonical_jsonl_bytes(tuple(query.model_dump(mode="json") for query in queries))
    )
    return root


def _install_verified_corpus(
    monkeypatch: pytest.MonkeyPatch,
    queries_root: Path,
    queries: tuple[Query, ...],
) -> list[Path]:
    calls: list[Path] = []

    def verify(path: str | Path) -> list[Query]:
        calls.append(Path(path))
        assert Path(path) == queries_root
        return list(queries)

    monkeypatch.setattr(stage1, "verify_accepted_corpus", verify)
    return calls


def test_bundle_preserves_five_intent_trajectories_and_excludes_later_signals() -> None:
    queries = _five_queries()
    bundle = build_trajectory_bundle(
        queries, reversed([query.query_id for query in queries])
    )

    assert tuple(item.query_id for item in bundle.trajectories) == tuple(
        sorted(query.query_id for query in queries)
    )
    assert {item.canonical_intent for item in bundle.trajectories} == set(PAPER_INTENTS)
    raw = parse_canonical_json(bundle.canonical_bytes(), label="bundle")
    assert set(raw) == {
        "schema_version",
        "status",
        "accepted_corpus_sha256",
        "taxonomy_version",
        "task_spec_version",
        "trajectories",
        "trajectory_bundle_sha256",
    }
    for trajectory in raw["trajectories"]:
        assert set(trajectory) == {
            "query_id",
            "asset_id",
            "image_path",
            "turns",
            "canonical_intent",
            "canonical_capability",
        }
    assert raw["trajectories"][0]["turns"] == [
        turn.model_dump(mode="json") for turn in queries[0].turns
    ]
    assert [turn["role"] for turn in raw["trajectories"][0]["turns"]] == [
        "user",
        "assistant",
        "user",
    ]
    projected_bytes = canonical_json_bytes(raw["trajectories"])
    assert b"ASSISTANT_CLARIFICATION_MUST_BE_PRESERVED" in projected_bytes
    for forbidden in (
        b"label_provenance",
        b"judge",
        b"failure",
        b"attribution",
        b"tool_trace",
        b"gate",
        b"test_frozen",
    ):
        assert forbidden not in projected_bytes


@pytest.mark.parametrize("split", ["dev_mini", "val", "test_frozen"])
def test_bundle_rejects_every_non_opt_pool_split(split: str) -> None:
    queries = list(_five_queries())
    queries[0] = queries[0].model_copy(update={"split": split})
    with pytest.raises(Stage1ContractError, match="only opt_pool"):
        build_trajectory_bundle(queries, [query.query_id for query in queries])


def test_bundle_requires_explicit_unique_selection_and_five_intent_coverage() -> None:
    queries = _five_queries()
    query_ids = [query.query_id for query in queries]
    with pytest.raises(Stage1ContractError, match="unique"):
        build_trajectory_bundle(queries, [*query_ids, query_ids[0]])
    with pytest.raises(Stage1ContractError, match="paper five intents"):
        build_trajectory_bundle(queries, query_ids[:-1])

    missing_capability = queries[0].model_copy(update={"canonical_capability": None})
    with pytest.raises(Stage1ContractError, match="lacks canonical capability"):
        build_trajectory_bundle(
            (missing_capability, *queries[1:]),
            query_ids,
        )


def test_bundle_and_creator_packet_reject_specification_version_drift() -> None:
    queries = _five_queries()
    query_ids = [query.query_id for query in queries]
    mixed = list(queries)
    mixed[0] = mixed[0].model_copy(
        update={"task_spec_version": "ecommerce-task-spec-v0"}
    )
    with pytest.raises(Stage1ContractError, match="must share one"):
        build_trajectory_bundle(mixed, query_ids)

    authoring_input = stage1.load_codex_authoring_input(
        AUTHORING_INPUT,
        expected_file_sha256=AUTHORING_INPUT_FILE_SHA256,
    )
    for field_name, drifted_version in (
        ("taxonomy_version", "drifted-taxonomy-v9"),
        ("task_spec_version", "ecommerce-task-spec-v0"),
    ):
        drifted_queries = tuple(
            query.model_copy(update={field_name: drifted_version}) for query in queries
        )
        bundle = build_trajectory_bundle(drifted_queries, query_ids)
        with pytest.raises(Stage1ContractError, match="specification versions differ"):
            stage1.build_s1_creator_packet(authoring_input, bundle)


def test_strict_models_reject_pollution_even_if_payload_is_rehashed() -> None:
    bundle = build_trajectory_bundle(
        _five_queries(), [query.query_id for query in _five_queries()]
    )
    raw_bundle = bundle.model_dump(mode="json")
    unsigned_bundle = {
        key: value
        for key, value in raw_bundle.items()
        if key != "trajectory_bundle_sha256"
    }
    unsigned_bundle["judge_scores"] = {"opt-01": 10}
    raw_bundle = {
        **unsigned_bundle,
        "trajectory_bundle_sha256": sha256_bytes(canonical_json_bytes(unsigned_bundle)),
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrajectoryBundle.model_validate(raw_bundle, strict=True)

    raw_packet = {
        "schema_version": 1,
        "status": "prepared_not_invoked",
        "authoring_input": parse_canonical_json(
            AUTHORING_INPUT.read_bytes(), label="authoring input"
        ),
        "trajectory_bundle": bundle.model_dump(mode="json"),
        "route_gate": {"score": 1},
    }
    raw_packet["packet_sha256"] = sha256_bytes(canonical_json_bytes(raw_packet))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        S1CreatorPacket.model_validate(raw_packet, strict=True)


def test_prepare_embeds_byte_identical_common_input_and_is_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = _five_queries()
    queries_root = _queries_root(tmp_path, queries)
    calls = _install_verified_corpus(monkeypatch, queries_root, queries)
    output = tmp_path / "prepared-stage1"

    prepared = prepare_stage1(
        authoring_input_path=AUTHORING_INPUT,
        expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
        queries_root=queries_root,
        selected_query_ids=reversed([query.query_id for query in queries]),
        output_dir=output,
    )

    assert calls == [queries_root]
    assert set(path.name for path in output.iterdir()) == {
        "trajectory-bundle.json",
        "s1-creator-packet.json",
    }
    bundle = load_trajectory_bundle(
        prepared.trajectory_bundle_path,
        expected_file_sha256=prepared.trajectory_bundle_file_sha256,
    )
    packet = load_s1_creator_packet(
        prepared.s1_creator_packet_path,
        expected_file_sha256=prepared.s1_creator_packet_file_sha256,
    )
    raw_packet = parse_canonical_json(
        prepared.s1_creator_packet_path.read_bytes(), label="packet"
    )
    assert set(raw_packet) == {
        "schema_version",
        "status",
        "authoring_input",
        "trajectory_bundle",
        "packet_sha256",
    }
    assert (
        canonical_json_bytes(raw_packet["authoring_input"])
        == AUTHORING_INPUT.read_bytes()
    )
    assert (
        canonical_json_bytes(raw_packet["trajectory_bundle"])
        == prepared.trajectory_bundle_path.read_bytes()
    )
    assert packet.authoring_input.canonical_bytes() == AUTHORING_INPUT.read_bytes()
    assert packet.trajectory_bundle == bundle
    assert prepared.selected_query_count == 5

    with pytest.raises(FileExistsError, match="create-only"):
        prepare_stage1(
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
            queries_root=queries_root,
            selected_query_ids=[query.query_id for query in queries],
            output_dir=output,
        )
    assert calls == [queries_root]


def test_prepare_calls_authoritative_verifier_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = _five_queries()
    queries_root = _queries_root(tmp_path, queries)
    output = tmp_path / "must-not-exist"

    def reject(_path: str | Path) -> list[Query]:
        raise ValueError("accepted ledger mismatch")

    monkeypatch.setattr(stage1, "verify_accepted_corpus", reject)
    with pytest.raises(
        Stage1ContractError, match="accepted corpus verification failed"
    ):
        prepare_stage1(
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
            queries_root=queries_root,
            selected_query_ids=[query.query_id for query in queries],
            output_dir=output,
        )
    assert not os.path.lexists(output)


def test_prepare_rejects_authoring_hash_drift_and_noncanonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = _five_queries()
    queries_root = _queries_root(tmp_path, queries)
    _install_verified_corpus(monkeypatch, queries_root, queries)
    with pytest.raises(
        Stage1ContractError, match="CodexAuthoringInput verification failed"
    ):
        prepare_stage1(
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256="0" * 64,
            queries_root=queries_root,
            selected_query_ids=[query.query_id for query in queries],
            output_dir=tmp_path / "hash-drift",
        )

    raw = json.loads(AUTHORING_INPUT.read_text(encoding="utf-8"))
    noncanonical = tmp_path / "noncanonical-authoring.json"
    noncanonical.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(
        Stage1ContractError, match="CodexAuthoringInput verification failed"
    ):
        prepare_stage1(
            authoring_input_path=noncanonical,
            expected_authoring_input_file_sha256=sha256_bytes(
                noncanonical.read_bytes()
            ),
            queries_root=queries_root,
            selected_query_ids=[query.query_id for query in queries],
            output_dir=tmp_path / "noncanonical-output",
        )


def test_prepare_rejects_symlink_queries_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = _five_queries()
    real_root = _queries_root(tmp_path, queries)
    linked_root = tmp_path / "linked-queries"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    monkeypatch.setattr(
        stage1,
        "verify_accepted_corpus",
        lambda _path: pytest.fail("verifier must not receive a symlink root"),
    )
    with pytest.raises(Stage1ContractError, match="non-symlink directory"):
        prepare_stage1(
            authoring_input_path=AUTHORING_INPUT,
            expected_authoring_input_file_sha256=AUTHORING_INPUT_FILE_SHA256,
            queries_root=linked_root,
            selected_query_ids=[query.query_id for query in queries],
            output_dir=tmp_path / "symlink-output",
        )


def test_queries_root_rejects_windows_junction_reparse_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = tmp_path.lstat()
    reparse_metadata = SimpleNamespace(
        st_mode=metadata.st_mode,
        st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )
    monkeypatch.setattr(Path, "lstat", lambda _path: reparse_metadata)

    with pytest.raises(Stage1ContractError, match="junction or reparse point"):
        stage1._require_real_directory(tmp_path, "queries root")


def test_selection_loader_requires_canonical_externally_hashed_bytes(
    tmp_path: Path,
) -> None:
    selection = build_query_selection(["opt-03", "opt-01", "opt-02"])
    path = tmp_path / "selection.json"
    path.write_bytes(selection.canonical_bytes())
    assert (
        load_query_selection(path, expected_file_sha256=sha256_bytes(path.read_bytes()))
        == selection
    )

    path.write_text(
        json.dumps(selection.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    with pytest.raises(Stage1ContractError, match="not canonical and valid"):
        load_query_selection(path, expected_file_sha256=sha256_bytes(path.read_bytes()))


def test_prepare_cli_reports_preparation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queries = _five_queries()
    queries_root = _queries_root(tmp_path, queries)
    _install_verified_corpus(monkeypatch, queries_root, queries)
    output = tmp_path / "cli-output"
    query_arguments = [
        argument
        for query in reversed(queries)
        for argument in ("--query-id", query.query_id)
    ]

    assert (
        prepare_cli.main(
            [
                "--authoring-input",
                str(AUTHORING_INPUT),
                "--expected-authoring-input-sha256",
                AUTHORING_INPUT_FILE_SHA256,
                "--queries-root",
                str(queries_root),
                *query_arguments,
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "prepared_not_invoked"
    assert report["model_invoked"] is False
    assert report["bank_generated"] is False
    assert report["selected_query_count"] == 5
    assert not any("bank" in path.name.casefold() for path in output.iterdir())
