from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import skillchain.offline_fixture as offline_fixture
from skillchain.offline_fixture import verify_offline_engineering_fixture
from skillchain.tools.formal_evaluation import FormalEvaluationError
from skillchain.tools import formal_evaluation as formal_evaluation_module
from skillchain.tools.registry import MVPToolServices, build_mvp_registry


def _run_fixture(clean_cwd: Path, output: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), *(filter(None, [existing]))]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "skillchain.offline_fixture",
            "--output",
            str(output),
        ],
        cwd=clean_cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return json.loads(completed.stdout)


def test_default_pytest_profile_blocks_python_network() -> None:
    with pytest.raises(RuntimeError, match="disabled for non-integration"):
        socket.getaddrinfo("example.com", 443)


def test_offline_fixture_python_socket_guard_rejects_dns_and_connect(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    program = """
import socket
from skillchain.offline_fixture import _install_network_guard

_install_network_guard()
blocked = []
for operation in (
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.create_connection(("127.0.0.1", 9), timeout=0.01),
):
    try:
        operation()
    except RuntimeError as error:
        blocked.append(str(error))
if len(blocked) != 2:
    raise SystemExit(f"expected two blocked socket operations, got {blocked!r}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_offline_fixture_registry_cannot_enter_formal_evaluator() -> None:
    registry = build_mvp_registry(
        MVPToolServices(
            product_search=offline_fixture._DiagnosticProductService(),  # noqa: SLF001
            kb_lookup=offline_fixture._DiagnosticKBService(),  # noqa: SLF001
            object_detection=offline_fixture._DiagnosticDetectionService(),  # noqa: SLF001
            document_ocr=offline_fixture._DiagnosticOCRService(),  # noqa: SLF001
            safety_approval_for=offline_fixture._DiagnosticSafetyResolver({}),  # noqa: SLF001
        )
    )

    assert registry.formal_runtime_ready is False
    with pytest.raises(FormalEvaluationError, match="explicit runtime bindings"):
        formal_evaluation_module._verify_registry_identity(  # noqa: SLF001
            registry,
            registry.registry_sha256,
            registry.registry_runtime_sha256,
        )


def test_offline_fixture_rebuilds_deterministically_from_clean_directory(
    tmp_path: Path,
) -> None:
    clean_cwd = tmp_path / "empty-working-directory"
    clean_cwd.mkdir()
    first_root = clean_cwd / "first"
    second_root = clean_cwd / "second"

    first = _run_fixture(clean_cwd, first_root)
    second = _run_fixture(clean_cwd, second_root)

    assert first["formal_eligible"] is False
    assert second["formal_eligible"] is False
    assert first["artifact_tree_sha256"] == second["artifact_tree_sha256"]
    assert first["manifest_file_sha256"] == second["manifest_file_sha256"]

    manifest = verify_offline_engineering_fixture(
        first_root,
        expected_manifest_file_sha256=str(first["manifest_file_sha256"]),
    )
    assert manifest["formal_eligible"] is False
    assert manifest["diagnostic_registry_kind"] == "typed-provisional"
    assert manifest["coverage"]["query_schema_version"] == 2
    assert manifest["coverage"]["task_spec_version"] == "ecommerce-task-spec-v1"
    assert manifest["coverage"]["group_leakage_violation_count"] == 0
    assert manifest["coverage"]["query_gallery_violation_count"] == 0
    assert manifest["coverage"]["assistant_configurations"] == [
        "noskill",
        "llm_static",
        "s1",
        "s1s2",
        "full",
    ]
    assert len(manifest["coverage"]["tool_interfaces"]) == 8
    assert len(manifest["limitations"]) >= 5

    for path in first_root.rglob("*"):
        if path.is_file():
            assert b'"formal_eligible":true' not in path.read_bytes()


def test_offline_fixture_command_is_create_only(tmp_path: Path) -> None:
    clean_cwd = tmp_path / "empty-working-directory"
    clean_cwd.mkdir()
    output = clean_cwd / "fixture"
    first = _run_fixture(clean_cwd, output)
    manifest_before = (output / "fixture-manifest.json").read_bytes()

    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    repeated = subprocess.run(
        [
            sys.executable,
            "-m",
            "skillchain.offline_fixture",
            "--output",
            str(output),
        ],
        cwd=clean_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert repeated.returncode != 0
    assert "拒绝覆盖" in repeated.stderr or "exists" in repeated.stderr
    assert (output / "fixture-manifest.json").read_bytes() == manifest_before
    verify_offline_engineering_fixture(
        output,
        expected_manifest_file_sha256=str(first["manifest_file_sha256"]),
    )
