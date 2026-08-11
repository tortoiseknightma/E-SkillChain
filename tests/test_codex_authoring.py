from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from skillchain import config
from skillchain.codex_authoring import (
    CODEX_AUTHOR_RUN_ID,
    CODEX_COMMAND_SHAPE,
    CodexAuthoringContractError,
    load_codex_authoring_input,
    project_codex_cli_output_schema,
    validate_codex_cli_output_schema,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes
import scripts.approve_codex_authoring as approval_script
import scripts.build_codex_authoring_approval_package as package_builder
from scripts.run_codex_authoring import (
    _event_audit,
    _run_capped_process,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    ROOT
    / "specs"
    / "authoring"
    / "authoring-freeze-lock-codex-high-v2.json"
)


def _script_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
    }


def _clean_events(message: str = "{}") -> bytes:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "reasoning-1", "type": "reasoning", "text": ""},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "message-1",
                "type": "agent_message",
                "text": message,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
            },
        },
    ]
    return b"".join(
        json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
        for event in events
    )


def test_model_roles_are_explicit_and_historical_qwen_is_separate() -> None:
    assert (
        config.ASSISTANT_PROVIDER,
        config.ASSISTANT_MODEL,
        config.ASSISTANT_MODEL_REVISION,
    ) == ("qwen", "qwen3-vl-flash-2026-01-22", "2026-01-22")
    assert (
        config.AUTHOR_PROVIDER,
        config.AUTHOR_MODEL,
        config.AUTHOR_REASONING_EFFORT,
    ) == ("codex_internal", "gpt-5.6-sol", "high")
    assert (
        config.JUDGE_PROVIDER,
        config.JUDGE_MODEL,
        config.JUDGE_REASONING_EFFORT,
        config.JUDGE_TEMPERATURE,
        config.JUDGE_TOP_P,
    ) == ("kimi", "kimi/kimi-k3", "max", 1.0, 0.95)
    assert config.LEGACY_AUTHOR_MODEL == "qwen3-vl-flash-2026-01-22"
    assert config.LABEL_VISION_SYNTH_MODEL != config.ASSISTANT_MODEL


def test_codex_schema_projection_uses_frozen_supported_subset() -> None:
    semantic = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Example",
        "type": "object",
        "properties": {
            "kind": {"const": "ok", "title": "Kind"},
            "items": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "oneOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "integer"},
                    ]
                },
            },
        },
        "required": ["kind", "items"],
        "additionalProperties": False,
    }
    projected = project_codex_cli_output_schema(semantic)
    validate_codex_cli_output_schema(projected)
    encoded = canonical_json_bytes(projected)
    for forbidden in (
        b'"$schema"',
        b'"title"',
        b'"uniqueItems"',
        b'"minLength"',
        b'"oneOf"',
        b'"const"',
    ):
        assert forbidden not in encoded
    assert projected["properties"]["kind"] == {
        "enum": ["ok"],
        "type": "string",
    }
    assert "anyOf" in projected["properties"]["items"]["items"]


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"x": {"type": "string", "pattern": "^x$"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": True,
        },
    ],
)
def test_codex_schema_subset_rejects_unsupported_or_open_shapes(
    schema: dict[str, object],
) -> None:
    with pytest.raises(CodexAuthoringContractError):
        validate_codex_cli_output_schema(schema)


def test_codex_v2_approved_preclaim_history_remains_byte_bound() -> None:
    freeze_bytes = FREEZE_PATH.read_bytes()
    freeze = json.loads(freeze_bytes)
    unsigned = dict(freeze)
    payload_sha256 = unsigned.pop("freeze_payload_sha256")
    assert payload_sha256 == (
        "6aa283f7749dc58cb6461853849c702acd0b545dcf12fb0c622d83954e0d6c1a"
    )
    assert sha256_bytes(freeze_bytes) == (
        "63c15d4461d15472fe713f7b0ff446370124b31d4d2a912e5bf70dc1e5dfdf39"
    )
    assert payload_sha256 == sha256_bytes(canonical_json_bytes(unsigned))
    assert freeze["model"] == {
        "reasoning_effort": "high",
        "requested_model": "gpt-5.6-sol",
        "served_model": None,
        "served_revision": None,
    }
    assert freeze["evidence_boundary"]["formal_provider_call_eligible"] is False
    assert (
        freeze["evidence_boundary"]["input_isolation"]
        == "behaviorally_constrained_not_mechanically_proven"
    )
    paths = freeze["authorization"]["paths"]
    output_directory = f"runs/formal-authoring/{CODEX_AUTHOR_RUN_ID}"
    assert paths == {
        "approval_record_file": (
            "specs/authoring/"
            f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"
        ),
        "claim_file": (
            "specs/authoring/"
            f"{CODEX_AUTHOR_RUN_ID}-attempt-claim.json"
        ),
        "output_directory": output_directory,
        "receipt_file": f"{output_directory}/invocation-receipt.json",
        "run_id": CODEX_AUTHOR_RUN_ID,
    }
    approval_path = ROOT / paths["approval_record_file"]
    approval_bytes = approval_path.read_bytes()
    approval = json.loads(approval_bytes)
    assert sha256_bytes(approval_bytes) == (
        "410aadbe56513a484fc52f50455211655ed6c41cb4ef5e5f7ffd9b6acf1489ac"
    )
    assert approval["approval_payload_sha256"] == (
        "55fb240de77e7c1f7956cb5e6694b11adb2cd996dbd11aa2b4481bd875f1e654"
    )
    claim_path = ROOT / paths["claim_file"]
    if claim_path.exists():
        retirement = json.loads(claim_path.read_text(encoding="utf-8"))
        assert retirement["status"] == (
            "authorization_retired_without_codex_process_launch"
        )
        assert retirement["codex_exec_process_spawned"] is False
        assert retirement["inference_requested"] is False
        assert retirement["v2_authorization_reusable"] is False
    assert not (ROOT / paths["output_directory"]).exists()


def test_codex_packet_and_runtime_do_not_claim_api_provider_evidence() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    packet_binding = freeze["bindings"]["authoring_input"]
    packet = load_codex_authoring_input(
        ROOT / packet_binding["file"],
        expected_file_sha256=packet_binding["file_sha256"],
    )
    assert packet.model.requested_model == "gpt-5.6-sol"
    assert packet.model.reasoning_effort == "high"
    assert packet.model.served_revision is None
    assert packet.formal_provider_call_eligible is False
    assert packet.session_budget.max_exec_sessions == 1
    assert packet.session_budget.max_repository_retries == 0
    assert packet.session_budget.max_followup_sessions == 0
    assert packet.session_budget.token_and_cost_enforcement == (
        "unavailable_on_codex_cli"
    )

    runtime_binding = freeze["bindings"]["runtime_lock"]
    runtime_path = ROOT / runtime_binding["file"]
    runtime_bytes = runtime_path.read_bytes()
    assert sha256_bytes(runtime_bytes) == runtime_binding["file_sha256"]
    runtime = json.loads(runtime_bytes)
    assert runtime["command_shape"] == list(CODEX_COMMAND_SHAPE)
    assert runtime["binary"]["sha256"] == (
        "83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c"
    )
    assert runtime["inference_connectivity_verified"] is False
    assert runtime["sandbox_policy"]["network_disabled"] is False

    source_manifest_binding = freeze["bindings"]["source_manifest"]
    source_manifest = json.loads(
        (ROOT / source_manifest_binding["file"]).read_text(encoding="utf-8")
    )
    assert source_manifest["schema_version"] == 2
    distributions = source_manifest["distributions"]
    assert {
        item["requested_name"] for item in distributions
    } >= {"pydantic", "pydantic_core", "python-dotenv"}
    assert all(item["files"] and len(item["tree_sha256"]) == 64 for item in distributions)


def test_codex_event_audit_accepts_exact_clean_turn() -> None:
    audit = _event_audit(_clean_events())
    audit.require_formal_success()
    assert audit.thread_id == "thread-1"
    assert audit.agent_messages == ("{}",)
    assert (audit.input_tokens, audit.cached_input_tokens, audit.output_tokens) == (
        100,
        20,
        10,
    )
    assert audit.visible_tool_activity is False


@pytest.mark.parametrize(
    "content",
    [
        (
            b'{"type":"turn.started"}\n'
            b'{"type":"thread.started","thread_id":"thread-1"}\n'
        ),
        b'{"type":"thread.started","thread_id":"a","thread_id":"b"}\n',
        (
            b'{"type":"thread.started","thread_id":"thread-1"}\n'
            b'{"type":"turn.started"}\n'
            b'{"item":{"id":"x","type":"reasoning"},'
            b'"type":"item.started"}\n'
            b'{"item":{"id":"x","type":"reasoning"},'
            b'"type":"item.started"}\n'
        ),
    ],
)
def test_codex_event_audit_rejects_non_strict_or_out_of_order_jsonl(
    content: bytes,
) -> None:
    with pytest.raises(CodexAuthoringContractError):
        _event_audit(content)


def test_codex_event_audit_rejects_tool_activity_and_failed_turn() -> None:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "dir",
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "message-1", "type": "agent_message", "text": "{}"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "output_tokens": 1,
            },
        },
    ]
    content = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n"
        for event in events
    )
    audit = _event_audit(content)
    assert audit.visible_tool_activity is True
    with pytest.raises(CodexAuthoringContractError):
        audit.require_formal_success()

    failed = (
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.started"}\n'
        b'{"type":"turn.failed"}\n'
    )
    failed_audit = _event_audit(failed)
    with pytest.raises(CodexAuthoringContractError):
        failed_audit.require_formal_success()


def test_capped_process_supervises_large_stdin_timeout() -> None:
    spawned = threading.Event()
    started = time.monotonic()
    result = _run_capped_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=ROOT,
        environment=os.environ.copy(),
        stdin=b"x" * 1_000_000,
        timeout_seconds=1,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=spawned,
    )
    elapsed = time.monotonic() - started
    assert spawned.is_set()
    assert result.timed_out is True
    assert result.stdin_completed is False
    assert elapsed < 6


def test_capped_process_completes_stdin_and_bounds_stdout() -> None:
    spawned = threading.Event()
    payload = b"x" * 100_000
    result = _run_capped_process(
        [
            sys.executable,
            "-c",
            (
                "import sys; data=sys.stdin.buffer.read(); "
                "sys.stdout.buffer.write(data)"
            ),
        ],
        cwd=ROOT,
        environment=os.environ.copy(),
        stdin=payload,
        timeout_seconds=10,
        stdout_limit=len(payload),
        stderr_limit=1024,
        spawned=spawned,
    )
    assert spawned.is_set()
    assert result.returncode == 0
    assert result.stdin_completed is True
    assert result.supervision_error is None
    assert result.stdout == payload
    assert result.stdout_limit_exceeded is False

    overflow = _run_capped_process(
        [sys.executable, "-c", "import sys; sys.stdout.write('x'*100000)"],
        cwd=ROOT,
        environment=os.environ.copy(),
        stdin=b"",
        timeout_seconds=10,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=threading.Event(),
    )
    assert overflow.stdout_limit_exceeded is True
    assert len(overflow.stdout) == 1024


def test_capped_process_reaps_child_when_thread_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = threading.Event()

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("synthetic thread start failure")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    started = time.monotonic()
    result = _run_capped_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=ROOT,
        environment=os.environ.copy(),
        stdin=b"x" * 100_000,
        timeout_seconds=30,
        stdout_limit=1024,
        stderr_limit=1024,
        spawned=spawned,
    )
    assert spawned.is_set()
    assert result.returncode != 0
    assert result.supervision_error is not None
    assert "synthetic thread start failure" in result.supervision_error
    assert time.monotonic() - started < 6


def _write_test_freeze(root: Path) -> tuple[Path, str, str]:
    freeze_path = (
        root
        / "specs"
        / "authoring"
        / "authoring-freeze-lock-codex-high-v2.json"
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_candidate_awaiting_owner_confirmation",
        "candidate_id": "authoring-codex-high-20260724-v2",
        "invocation_authorized": False,
        "authorization": {
            "paths": {
                "run_id": CODEX_AUTHOR_RUN_ID,
                "approval_record_file": (
                    "specs/authoring/"
                    f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"
                ),
            }
        },
    }
    payload_sha256 = sha256_bytes(canonical_json_bytes(payload))
    content = canonical_json_bytes(
        {**payload, "freeze_payload_sha256": payload_sha256}
    )
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(content)
    return freeze_path, sha256_bytes(content), payload_sha256


def test_owner_approval_requires_both_freeze_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path, file_sha256, payload_sha256 = _write_test_freeze(tmp_path)
    monkeypatch.setattr(approval_script, "ROOT", tmp_path)
    monkeypatch.setattr(approval_script, "FREEZE_PATH", freeze_path)
    common = [
        "approve_codex_authoring.py",
        "--accept-freeze-file-sha256",
        file_sha256,
        "--accept-freeze-payload-sha256",
        payload_sha256,
        "--accept-platform-mediated-non-provider-attested",
        "--accept-one-session-and-permanent-consumption",
        "--accept-input-isolation-limit",
        "--accept-s1-comparability-envelope",
    ]
    monkeypatch.setattr(sys, "argv", common)
    assert approval_script.main() == 0
    approval_path = (
        tmp_path
        / "specs"
        / "authoring"
        / f"{CODEX_AUTHOR_RUN_ID}-owner-approval.json"
    )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    assert approval["freeze_lock_file_sha256"] == file_sha256
    assert approval["freeze_payload_sha256"] == payload_sha256

    second_root = tmp_path / "wrong"
    second_freeze, second_file, second_payload = _write_test_freeze(second_root)
    monkeypatch.setattr(approval_script, "ROOT", second_root)
    monkeypatch.setattr(approval_script, "FREEZE_PATH", second_freeze)
    wrong = list(common)
    wrong[2] = second_file
    wrong[4] = "0" * 64
    monkeypatch.setattr(sys, "argv", wrong)
    with pytest.raises(ValueError, match="accepted Codex freeze identity"):
        approval_script.main()
    assert second_payload != "0" * 64


def test_historical_qwen_freezes_remain_reproducible() -> None:
    for script in (
        "scripts/freeze_authoring_packet.py",
        "scripts/freeze_authoring_packet_v4.py",
        "scripts/build_authoring_packet_v5_candidate.py",
    ):
        subprocess.run(
            [sys.executable, script, "--check"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=_script_env(),
        )
