from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

import scripts.approve_codex_authoring_v3 as approval_v3
import scripts.build_codex_authoring_approval_package_v3 as builder_v3
import scripts.run_codex_authoring_v3 as runner_v3
import skillchain.codex_authoring_v3 as contract_v3
from skillchain.codex_authoring_v3 import (
    CODEX_AUTHOR_CANDIDATE_ID,
    CODEX_AUTHOR_RUN_ID,
    CodexAuthoringContractError,
    build_codex_v3_environment_policy,
    construct_codex_v3_environment,
    resolve_frozen_codex_binary,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
V2_RUN_ID = "llm-static-codex-primary-20260724-high-v2"
FREEZE_PATH = (
    ROOT
    / "specs"
    / "authoring"
    / "authoring-freeze-lock-codex-high-v3.json"
)


def _script_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
    }


def _signed(payload: dict[str, object], field: str) -> bytes:
    return canonical_json_bytes(
        {**payload, field: sha256_bytes(canonical_json_bytes(payload))}
    )


def _fake_binary(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "path_policy": "frozen_absolute_path_no_path_lookup",
        "bytes": 1,
        "sha256": "0" * 64,
        "cli_version": "0.145.0",
    }


def test_v3_environment_is_invariant_to_dynamic_parent_path(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path / "codex.exe")
    stable = {
        "APPDATA": "appdata",
        "SYSTEMROOT": r"C:\Windows",
        "USERPROFILE": "profile",
        "TEMP": "parent-temp-must-not-be-inherited",
        "PATHEXT": ".EXE",
        "OPENAI_API_KEY": "must-not-be-inherited",
        "PYTHONPATH": "must-not-be-inherited",
    }
    first = {
        **stable,
        "PATH": r"C:\Users\u\.codex\tmp\arg0\codex-arg0-first",
    }
    second = {
        **stable,
        "PATH": r"C:\Users\u\.codex\tmp\arg0\codex-arg0-second",
    }
    first_policy = build_codex_v3_environment_policy(
        binary=binary,
        parent_environment=first,
    )
    second_policy = build_codex_v3_environment_policy(
        binary=binary,
        parent_environment=second,
    )
    assert first_policy == second_policy
    assert first_policy["parent_path_inherited"] is False
    assert first_policy["parent_pathext_inherited"] is False
    assert first_policy["parent_temp_inherited"] is False

    invocation_temp = tmp_path / "invocation-temp"
    invocation_temp.mkdir()
    child, policy_sha, instance_sha = construct_codex_v3_environment(
        runtime={"binary": binary, "environment_policy": first_policy},
        executable=Path(binary["path"]),
        invocation_temp=invocation_temp,
        parent_environment=second,
    )
    assert child["PATH"] == str(Path(binary["path"]).parent)
    assert child["TEMP"] == child["TMP"] == str(invocation_temp.resolve())
    assert "PATHEXT" not in child
    assert "OPENAI_API_KEY" not in child
    assert "PYTHONPATH" not in child
    assert child["APPDATA"] == stable["APPDATA"]
    assert len(policy_sha) == len(instance_sha) == 64


def test_v3_environment_rejects_case_aliases_and_inherited_drift(
    tmp_path: Path,
) -> None:
    binary = _fake_binary(tmp_path / "codex.exe")
    with pytest.raises(
        CodexAuthoringContractError,
        match="case-insensitive duplicate",
    ):
        build_codex_v3_environment_policy(
            binary=binary,
            parent_environment={
                "HTTP_PROXY": "one",
                "http_proxy": "two",
            },
        )

    frozen_parent = {"APPDATA": "one", "SYSTEMROOT": r"C:\Windows"}
    runtime = {
        "binary": binary,
        "environment_policy": build_codex_v3_environment_policy(
            binary=binary,
            parent_environment=frozen_parent,
        ),
    }
    invocation_temp = tmp_path / "invocation-temp"
    invocation_temp.mkdir()
    with pytest.raises(CodexAuthoringContractError, match="differs"):
        construct_codex_v3_environment(
            runtime=runtime,
            executable=Path(binary["path"]),
            invocation_temp=invocation_temp,
            parent_environment={
                "APPDATA": "two",
                "SYSTEMROOT": r"C:\Windows",
            },
        )


def test_v3_binary_resolution_never_uses_path_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"codex")
    runtime = {
        "binary": {
            "path": str(executable.resolve()),
            "path_policy": "frozen_absolute_path_no_path_lookup",
            "bytes": 5,
            "sha256": hashlib.sha256(b"codex").hexdigest(),
            "cli_version": "0.145.0",
        }
    }

    def forbidden_which(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("PATH lookup must not be used")

    monkeypatch.setattr(shutil, "which", forbidden_which)
    assert resolve_frozen_codex_binary(runtime) == executable.resolve()

    runtime["binary"] = {**runtime["binary"], "sha256": "0" * 64}
    with pytest.raises(CodexAuthoringContractError, match="differs"):
        resolve_frozen_codex_binary(runtime)
    runtime["binary"] = {
        **runtime["binary"],
        "sha256": hashlib.sha256(b"codex").hexdigest(),
        "cli_version": "0.144.0",
    }
    with pytest.raises(CodexAuthoringContractError, match="differs"):
        resolve_frozen_codex_binary(runtime)


def test_v3_claim_atomicity_allows_only_one_launch(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    barrier = threading.Barrier(2)
    launched: list[int] = []
    failures: list[Exception] = []

    def contender(index: int) -> None:
        barrier.wait()
        try:
            runner_v3._consume_claim(claim, b"claim")
        except Exception as error:
            failures.append(error)
            return
        assert claim.read_bytes() == b"claim"
        launched.append(index)

    threads = [
        threading.Thread(target=contender, args=(index,))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(launched) == 1
    assert len(failures) == 1
    assert claim.read_bytes() == b"claim"


def test_v3_package_and_history_bindings_are_reproducible() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_codex_authoring_approval_package_v3.py",
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "frozen_candidate_awaiting_owner_confirmation"
    assert report["candidate_id"] == CODEX_AUTHOR_CANDIDATE_ID
    assert report["run_id"] == CODEX_AUTHOR_RUN_ID
    assert report["invocation_authorized"] is False

    freeze_bytes = FREEZE_PATH.read_bytes()
    freeze = json.loads(freeze_bytes)
    unsigned = dict(freeze)
    payload_sha = unsigned.pop("freeze_payload_sha256")
    assert payload_sha == report["freeze_payload_sha256"]
    assert sha256_bytes(freeze_bytes) == report["freeze_file_sha256"]
    assert payload_sha == sha256_bytes(canonical_json_bytes(unsigned))

    paths = freeze["authorization"]["paths"]
    assert paths["run_id"] == CODEX_AUTHOR_RUN_ID
    assert paths["output_directory"].endswith(CODEX_AUTHOR_RUN_ID)
    assert CODEX_AUTHOR_RUN_ID in paths["claim_file"]
    assert CODEX_AUTHOR_RUN_ID in paths["approval_record_file"]
    assert V2_RUN_ID not in canonical_json_bytes(paths).decode("utf-8")

    runtime_binding = freeze["bindings"]["runtime_lock"]
    runtime = json.loads(
        (ROOT / runtime_binding["file"]).read_text(encoding="utf-8")
    )
    assert runtime["binary"]["path_policy"] == (
        "frozen_absolute_path_no_path_lookup"
    )
    assert runtime["environment_policy"]["parent_path_inherited"] is False
    assert runtime["environment_policy"]["parent_pathext_inherited"] is False
    assert runtime["environment_policy"]["parent_temp_inherited"] is False
    assert "PATH" not in runtime["environment_policy"]["inherited_names"]
    assert "PATHEXT" not in runtime["environment_policy"]["inherited_names"]
    assert "TEMP" not in runtime["environment_policy"]["inherited_names"]
    assert "TMP" not in runtime["environment_policy"]["inherited_names"]

    manifest_binding = freeze["bindings"]["source_manifest"]
    manifest = json.loads(
        (ROOT / manifest_binding["file"]).read_text(encoding="utf-8")
    )
    trusted = {item["file"] for item in manifest["trusted_sources"]}
    assert {
        "scripts/run_codex_authoring.py",
        "scripts/run_codex_authoring_v3.py",
        "src/skillchain/codex_authoring.py",
        "src/skillchain/codex_authoring_v3.py",
    } <= trusted

    incident_binding = freeze["bindings"]["v2_preflight_incident"]
    incident = json.loads(
        (ROOT / incident_binding["file"]).read_text(encoding="utf-8")
    )
    assert incident["claim_created"] is False
    assert incident["codex_exec_process_spawned"] is False
    assert incident["inference_requested"] is False
    assert incident["sensitive_environment_values_recorded"] is False
    assert freeze["prior_authorization_retirement"][
        "retirement_claim_expected_file_sha256"
    ] == sha256_bytes(builder_v3.planned_v2_retirement_claim_bytes())


def test_v2_frozen_authoring_sources_remain_byte_exact() -> None:
    expected = {
        "scripts/run_codex_authoring.py": (
            "7face7a874d19dccd714c5be098f28546878a8326e65c1b833368af380f79a88"
        ),
        "src/skillchain/codex_authoring.py": (
            "3f340bddfe59c4d8b194fd7546a296c79d6eafe88a4dceb34d60a96930644f0a"
        ),
        "scripts/build_codex_authoring_approval_package.py": (
            "fcffe459b3c751eca8660a9464498e49cf6a0a8c697a0f9f5ba6a06545413723"
        ),
        "scripts/approve_codex_authoring.py": (
            "8e5bf2681b775086980e000e7996b1bf4038f2f6fbbba0891154ef825e6693be"
        ),
    }
    for relative, digest in expected.items():
        assert sha256_bytes((ROOT / relative).read_bytes()) == digest


def test_v3_approval_retires_v2_before_creating_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    authoring = root / "specs" / "authoring"
    authoring.mkdir(parents=True)
    runs = root / "runs" / "formal-authoring"
    runs.mkdir(parents=True)

    v2_freeze_path = authoring / "v2-freeze.json"
    v2_freeze_bytes = canonical_json_bytes(
        {"freeze_payload_sha256": "f" * 64}
    )
    v2_freeze_path.write_bytes(v2_freeze_bytes)
    v2_approval_path = authoring / "v2-approval.json"
    v2_approval_bytes = canonical_json_bytes(
        {
            "status": "owner_approved_for_one_codex_author_session",
            "approval_payload_sha256": "a" * 64,
        }
    )
    v2_approval_path.write_bytes(v2_approval_bytes)
    incident_path = authoring / "incident.json"
    incident_path.write_bytes(b"{}")
    retirement_path = authoring / "v2-claim.json"
    v3_approval_path = authoring / "v3-approval.json"

    retirement_bytes = _signed(
        {
            "status": "authorization_retired_without_codex_process_launch",
            "run_id": V2_RUN_ID,
            "codex_exec_process_spawned": False,
            "inference_requested": False,
            "v2_authorization_reusable": False,
        },
        "retirement_payload_sha256",
    )
    freeze_payload = {
        "schema_version": 1,
        "status": "frozen_candidate_awaiting_owner_confirmation",
        "candidate_id": CODEX_AUTHOR_CANDIDATE_ID,
        "invocation_authorized": False,
        "authorization": {
            "paths": {
                "run_id": CODEX_AUTHOR_RUN_ID,
                "approval_record_file": (
                    v3_approval_path.relative_to(root).as_posix()
                ),
            }
        },
        "prior_authorization_retirement": {
            "retirement_claim_file": retirement_path.relative_to(root).as_posix(),
            "retirement_claim_expected_file_sha256": sha256_bytes(
                retirement_bytes
            ),
            "retirement_must_precede_v3_approval": True,
            "v2_approval_reusable": False,
        },
    }
    freeze_payload_sha = sha256_bytes(canonical_json_bytes(freeze_payload))
    freeze_bytes = canonical_json_bytes(
        {**freeze_payload, "freeze_payload_sha256": freeze_payload_sha}
    )
    freeze_path = authoring / "v3-freeze.json"
    freeze_path.write_bytes(freeze_bytes)

    monkeypatch.setattr(approval_v3, "ROOT", root)
    monkeypatch.setattr(approval_v3, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(approval_v3, "APPROVAL_PATH", v3_approval_path)
    monkeypatch.setattr(approval_v3, "V2_FREEZE_PATH", v2_freeze_path)
    monkeypatch.setattr(
        approval_v3,
        "V2_FREEZE_FILE_SHA256",
        sha256_bytes(v2_freeze_bytes),
    )
    monkeypatch.setattr(approval_v3, "V2_FREEZE_PAYLOAD_SHA256", "f" * 64)
    monkeypatch.setattr(approval_v3, "V2_APPROVAL_PATH", v2_approval_path)
    monkeypatch.setattr(
        approval_v3,
        "V2_APPROVAL_FILE_SHA256",
        sha256_bytes(v2_approval_bytes),
    )
    monkeypatch.setattr(approval_v3, "V2_APPROVAL_PAYLOAD_SHA256", "a" * 64)
    monkeypatch.setattr(approval_v3, "V2_PREFLIGHT_INCIDENT_PATH", incident_path)
    monkeypatch.setattr(approval_v3, "V2_RETIREMENT_CLAIM_PATH", retirement_path)
    monkeypatch.setattr(approval_v3, "V2_OUTPUT_DIR", runs / "v2")
    monkeypatch.setattr(
        approval_v3,
        "V2_RECEIPT_PATH",
        runs / "v2" / "invocation-receipt.json",
    )
    monkeypatch.setattr(
        approval_v3,
        "planned_v2_retirement_claim_bytes",
        lambda: retirement_bytes,
    )
    order: list[str] = []
    original = approval_v3._create_or_verify

    def record(path: Path, content: bytes, label: str) -> None:
        original(path, content, label)
        order.append(path.name)

    monkeypatch.setattr(approval_v3, "_create_or_verify", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "approve_codex_authoring_v3.py",
            "--accept-freeze-file-sha256",
            sha256_bytes(freeze_bytes),
            "--accept-freeze-payload-sha256",
            freeze_payload_sha,
            "--accept-platform-mediated-non-provider-attested",
            "--accept-one-session-and-permanent-consumption",
            "--accept-input-isolation-limit",
            "--accept-s1-comparability-envelope",
            "--accept-deterministic-environment-v3",
            "--accept-v2-non-inference-retirement",
        ],
    )
    assert approval_v3.main() == 0
    assert order == [retirement_path.name, v3_approval_path.name]
    assert retirement_path.read_bytes() == retirement_bytes
    approval = json.loads(v3_approval_path.read_text(encoding="utf-8"))
    assert approval["v2_authorization_retired_without_inference"] is True
    assert approval["candidate_id"] == CODEX_AUTHOR_CANDIDATE_ID
