from __future__ import annotations

import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from scripts import smoke_portfolio_gcs_v2 as smoke
from skillchain import llm as llm_module
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _legacy_row() -> bytes:
    unsigned = {
        "schema_version": 1,
        "kind": "portfolio-assistant-checkpoint",
        "instance_sha256": "1" * 64,
        "query_ordinal": 0,
        # The v2 reader must reject on checkpoint generation before attempting
        # to interpret any legacy response/receipt payload.
        "request": {},
        "response": {},
        "receipt": {},
    }
    return canonical_json_bytes({**unsigned, "row_sha256": smoke._hash(unsigned)})


def test_legacy_checkpoint_is_rejected_without_changing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v5.json"
    content = _legacy_row()
    path.write_bytes(content)

    audit = smoke.verify_legacy_checkpoint_rejected(path)

    assert audit["status"] == "passed_immutable_rejection"
    assert audit["reason_code"] == "legacy_schema_v1_checkpoint_rejected"
    assert audit["before_sha256"] == sha256_bytes(content)
    assert audit["after_sha256"] == sha256_bytes(content)
    assert path.read_bytes() == content


def test_legacy_checkpoint_with_bad_hash_is_not_mislabeled_as_v2_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "damaged-v7.json"
    path.write_bytes(b'{"schema_version":1,"row_sha256":"bad"}')

    with pytest.raises(smoke.OfflineSmokeError, match="failed for a different reason"):
        smoke.verify_legacy_checkpoint_rejected(path)


def test_legacy_runtime_is_rejected_without_changing_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "treatment-runtime-v7"
    root.mkdir()
    (root / "runtime-lock.json").write_bytes(b'{"schema_version":1}')
    (root / "bank.json").write_bytes(b"legacy-bank")
    before = smoke.artifact_tree_bytes(root)

    def reject(*_args, **_kwargs):
        raise ValueError("not a one-Bank Static GCS-v2 runtime")

    monkeypatch.setattr(smoke, "load_verified_portfolio_static_opt_runtime", reject)
    audit = smoke.verify_legacy_runtime_rejected(root)

    assert audit["status"] == "passed_immutable_rejection"
    assert audit["reason_code"] == ("legacy_runtime_rejected_by_static_gcs_v2_loader")
    assert audit["artifact_count"] == 2
    assert smoke.artifact_tree_bytes(root) == before


def test_deterministic_offline_chat_emits_route_tool_and_final() -> None:
    chat = smoke.DeterministicOfflineChat()
    chat.configure(
        capability_id="product.style_recommendation",
        tool_name="style_similar_search",
    )

    route = chat("qwen", [], model="frozen-model", json_mode=True, tools=None)
    tool = chat(
        "qwen",
        [],
        model="frozen-model",
        json_mode=False,
        tools=[
            {
                "type": "function",
                "function": {"name": "style_similar_search"},
            }
        ],
    )
    final = chat(
        "qwen",
        [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "final_response_contract": {
                            "policy_version": (
                                smoke.GCS_V2_MODEL_RESPONSE_CONTRACT_VERSION
                            )
                        }
                    }
                ),
            }
        ],
        model="frozen-model",
        json_mode=False,
        tools=[
            {
                "type": "function",
                "function": {"name": "style_similar_search"},
            }
        ],
    )

    assert json.loads(route.text) == {
        "selected_capability": "product.style_recommendation"
    }
    assert route.finish_reason == "stop"
    assert tool.finish_reason == "tool_calls"
    assert tool.tool_calls[0].name == "style_similar_search"
    assert json.loads(tool.tool_calls[0].arguments_json) == {"asset_id": "query_asset"}
    assert final.finish_reason == "stop"
    assert final.text
    assert chat.fake_model_calls == 3
    assert chat.response_contract_observations == 1
    assert chat.provider_calls == 0


def test_offline_guard_replaces_chat_and_denies_socket_then_restores() -> None:
    chat = smoke.DeterministicOfflineChat()
    original_chat = llm_module.chat

    with smoke.offline_chat_and_network_guard(chat):
        assert llm_module.chat is chat
        with pytest.raises(smoke.OfflineNetworkAccessError):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    assert llm_module.chat is original_chat


def _checkpoint_row(*, marker: str = "original") -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "kind": "portfolio-assistant-checkpoint",
        "instance_sha256": "2" * 64,
        "query_ordinal": 1,
        "request": {"marker": marker},
        "response": {},
        "receipt": {},
        "public_scorer_evidence": {"schema_version": 2},
    }
    return {**unsigned, "row_sha256": smoke._hash(unsigned)}


def test_checkpoint_publish_is_create_only_and_exact_resume(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoint.json"
    verified: list[Path] = []

    def verifier(candidate: Path) -> None:
        verified.append(candidate)
        assert json.loads(candidate.read_bytes())["schema_version"] == 2

    row = _checkpoint_row()
    first = smoke.publish_verified_checkpoint(path, row, verifier=verifier)
    second = smoke.publish_verified_checkpoint(path, row, verifier=verifier)

    assert first == second == path.read_bytes()
    assert verified == [path, path]
    with pytest.raises(smoke.OfflineSmokeError, match="conflicting bytes"):
        smoke.publish_verified_checkpoint(
            path, _checkpoint_row(marker="conflict"), verifier=verifier
        )


def _query(
    capability: str,
    *,
    split: str,
    query_id: str,
    text: str = "普通查询",
) -> SimpleNamespace:
    return SimpleNamespace(
        canonical_capability=capability,
        split=split,
        query_id=query_id,
        text=text,
    )


def test_probe_selection_covers_six_capabilities_dev_opt_and_style_modes() -> None:
    queries = [
        _query(
            capability,
            split="dev_mini" if index % 2 == 0 else "opt_pool",
            query_id=f"q-{index}",
        )
        for index, capability in enumerate(smoke.GCS_CAPABILITY_ORDER)
    ]
    # Replace the selected Style query text with a same-category request and
    # add a cross-category coordination request.
    queries[3].text = "给我推荐相似款"
    queries.append(
        _query(
            "product.style_recommendation",
            split="opt_pool",
            query_id="style-cross",
            text="帮我搭配一双鞋",
        )
    )

    selected = smoke.select_capability_probes(queries)
    same, cross = smoke.style_probe_pools(queries)

    assert tuple(item.canonical_capability for item in selected) == (
        smoke.GCS_CAPABILITY_ORDER
    )
    assert {item.split for item in selected} == {"dev_mini", "opt_pool"}
    assert [item.query_id for item in same] == ["q-3"]
    assert [item.query_id for item in cross] == ["style-cross"]


def _record(
    capability: str,
    *,
    label: str,
    submode: str | None = None,
    status: str | None = None,
) -> smoke.ProbeRecord:
    return smoke.ProbeRecord(
        label=label,
        query_id=label,
        split="opt_pool",
        capability_id=capability,
        checkpoint_file_sha256="3" * 64,
        checkpoint_row_sha256="4" * 64,
        scorer_evidence_sha256="5" * 64,
        style_submode=submode,
        style_support_status=status,
    )


def test_probe_coverage_requires_same_mapped_and_unsupported_style() -> None:
    records = [
        _record(capability, label=f"cap-{index}")
        for index, capability in enumerate(smoke.GCS_CAPABILITY_ORDER)
        if capability != "product.style_recommendation"
    ]
    records.extend(
        [
            _record(
                "product.style_recommendation",
                label="same",
                submode="same_category_alternative",
                status="candidates",
            ),
            _record(
                "product.style_recommendation",
                label="mapped",
                submode="cross_category_coordination",
                status="candidates",
            ),
            _record(
                "product.style_recommendation",
                label="unsupported",
                submode="cross_category_coordination",
                status="unsupported",
            ),
        ]
    )

    smoke.require_probe_coverage(records)
    with pytest.raises(smoke.OfflineSmokeError, match="unsupported outcomes"):
        smoke.require_probe_coverage(records[:-1])


def test_analyzer_replay_requires_all_output_bytes_to_match(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir()

    def deterministic_analyzer(
        _execution_root: Path, *, batch_id: str, output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "analysis.json").write_bytes(
            canonical_json_bytes({"batch_id": batch_id})
        )
        (output_dir / "gcs-summary.json").write_bytes(b"stable")

    hashes = smoke.run_deterministic_analysis_replay(
        execution_root,
        batch_id="opt-001",
        output_parent=tmp_path / "outputs",
        analyzer=deterministic_analyzer,
    )

    assert hashes == {
        "analysis.json": sha256_bytes(canonical_json_bytes({"batch_id": "opt-001"})),
        "gcs-summary.json": sha256_bytes(b"stable"),
    }


def test_analyzer_replay_fails_on_one_byte_drift(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    calls = 0

    def drifting_analyzer(
        _execution_root: Path, *, batch_id: str, output_dir: Path
    ) -> None:
        nonlocal calls
        calls += 1
        output_dir.mkdir(parents=True)
        (output_dir / "analysis.json").write_bytes(f"{batch_id}:{calls}".encode())

    with pytest.raises(smoke.OfflineSmokeError, match="bytes drifted"):
        smoke.run_deterministic_analysis_replay(
            execution_root,
            batch_id="opt-001",
            output_parent=tmp_path / "outputs",
            analyzer=drifting_analyzer,
        )


def test_execution_control_clone_keeps_source_bytes_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    key = b"k" * 32
    unsigned = {
        "schema_version": 1,
        "blinding_key_sha256": sha256_bytes(key),
        "marker": "source-control",
    }
    control = {**unsigned, "control_sha256": smoke._hash(unsigned)}
    control_bytes = canonical_json_bytes(control)
    (source / "execution-control.json").write_bytes(control_bytes)
    (source / "blinding-key.bin").write_bytes(key)

    loaded = smoke._clone_execution_control(source, target)

    assert loaded == control
    assert (source / "execution-control.json").read_bytes() == control_bytes
    assert (source / "blinding-key.bin").read_bytes() == key
    assert (target / "execution-control.json").read_bytes() == control_bytes
    assert (target / "blinding-key.bin").read_bytes() == key


def test_preauthorization_control_is_explicitly_non_executable() -> None:
    runtime_lock_sha = "6" * 64
    launch = SimpleNamespace(
        plan=SimpleNamespace(
            kind="portfolio-core-static-opt-800x1-launch-plan",
            execution_mode="static_opt_rollout",
            execution_ready=False,
            blockers=("missing_env:DASHSCOPE_API_KEY",),
            query_count=800,
            instance_count=800,
            shard_count=32,
            config_order=("llm_static",),
            selected_splits=("opt_pool",),
            matrix_run_id="preauth-matrix",
            launch_plan_sha256="7" * 64,
            artifacts=(
                SimpleNamespace(
                    artifact_id="assistant_runtime_lock",
                    status="verified",
                    file_sha256="8" * 64,
                    content_sha256=runtime_lock_sha,
                ),
            ),
        )
    )
    runtime = SimpleNamespace(
        runtime_lock={
            "runtime_lock_sha256": runtime_lock_sha,
            "runtime_data_sha256": "9" * 64,
            "task_spec_version": "task-v1",
            "task_spec_sha256": "a" * 64,
            "task_spec_file_sha256": "b" * 64,
        },
        bank_file_sha256="c" * 64,
        bank=SimpleNamespace(bank_sha256="d" * 64),
        refresh_receipt_file_sha256="e" * 64,
        refresh_receipt={"receipt_sha256": "f" * 64},
        semantic_authoring_input_file_sha256="1" * 64,
        semantic_authoring_input=SimpleNamespace(input_sha256="2" * 64),
        core_source_receipt_file_sha256="3" * 64,
        core_source_receipt={"receipt_sha256": "4" * 64},
    )

    control = smoke.preauthorization_smoke_control_payload(
        launch=launch,
        launch_root=Path("launch"),
        launch_plan_file_sha256="5" * 64,
        runtime=runtime,
        runtime_root=Path("runtime"),
        runtime_lock_file_sha256="8" * 64,
        blinding_key=b"x" * 32,
    )

    assert control["preauthorization_zero_provider"] is True
    assert control["execution_authorized"] is False
    assert control["provider_calls_permitted"] is False
    assert control["budget_ledger_permitted"] is False
    assert control["authorized_shard_ids"] == []
    assert control["evaluation_stages"] == ["preauthorization_zero_provider_smoke"]
    assert control["evaluation_stages"] != ["assistant", "gcs_v2"]
    unsigned = dict(control)
    supplied = unsigned.pop("control_sha256")
    assert supplied == smoke._hash(unsigned)
