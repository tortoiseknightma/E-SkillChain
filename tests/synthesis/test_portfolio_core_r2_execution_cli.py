"""Focused tests for the body-free Portfolio core r2 execution CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "portfolio_core_r2_execution.py"
)


@pytest.fixture()
def execution_cli():
    module_name = "portfolio_core_r2_execution_cli_test"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _sha(character: str) -> str:
    return character * 64


def _common_arguments(
    tmp_path: Path,
    *,
    expected_r2_plan_sha256: str = _sha("4"),
) -> list[str]:
    return [
        "--r1-plan",
        str(tmp_path / "r1-plan.json"),
        "--parent-catalog",
        str(tmp_path / "parent-catalog"),
        "--parent-asset-root",
        str(tmp_path / "parent-assets"),
        "--expected-parent-catalog-sha256",
        _sha("a"),
        "--target-catalog",
        str(tmp_path / "target-catalog"),
        "--target-asset-root",
        str(tmp_path / "target-assets"),
        "--expected-target-catalog-sha256",
        _sha("b"),
        "--parent-assignments",
        str(tmp_path / "parent-assignments.jsonl"),
        "--expected-parent-assignments-sha256",
        _sha("c"),
        "--target-assignments",
        str(tmp_path / "target-assignments.jsonl"),
        "--expected-target-assignments-sha256",
        _sha("d"),
        "--expected-parent-plan-sha256",
        _sha("e"),
        "--expected-r2-plan-sha256",
        expected_r2_plan_sha256,
    ]


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        work_order=SimpleNamespace(
            job_id="r2-author-" + "f" * 24,
            aliases=(object(), object(), object()),
        ),
        author_packet=object(),
    )


def test_prepare_job_rebuilds_then_publishes_one_job_and_emits_only_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    execution_cli,
) -> None:
    context = object()
    job = _job()
    calls: dict[str, object] = {}
    jobs_root = tmp_path / "published-jobs"
    job_dir = jobs_root / job.work_order.job_id

    monkeypatch.setattr(execution_cli, "_rebuild_context", lambda arguments: context)
    monkeypatch.setattr(
        execution_cli,
        "_build_authoring_job_for_batch",
        lambda rebuilt, batch_id: (
            calls.update({"context": rebuilt, "batch_id": batch_id}) or job
        ),
    )
    monkeypatch.setattr(
        execution_cli,
        "publish_authoring_job",
        lambda candidate, root: (
            calls.update({"job": candidate, "jobs_root": root})
            or SimpleNamespace(
                job_dir=job_dir,
                manifest=SimpleNamespace(author_packet_sha256=_sha("1")),
            )
        ),
    )
    monkeypatch.setattr(execution_cli, "canonical_author_packet_bytes", lambda _: b"x")
    monkeypatch.setattr(execution_cli, "sha256_bytes", lambda _: _sha("1"))

    assert (
        execution_cli.main(
            [
                "prepare-job",
                *_common_arguments(tmp_path),
                "--batch-id",
                "core-001",
                "--jobs-root",
                str(jobs_root),
            ]
        )
        == 0
    )

    assert calls == {
        "context": context,
        "batch_id": "core-001",
        "job": job,
        "jobs_root": jobs_root,
    }
    assert json.loads(capsys.readouterr().out) == {
        "alias_count": 3,
        "job_dir": str(job_dir),
        "job_id": job.work_order.job_id,
        "packet_sha256": _sha("1"),
    }


def test_approve_uses_only_the_published_draft_root_and_body_free_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    execution_cli,
) -> None:
    context = SimpleNamespace(
        result=object(),
        bridge=object(),
        trusted_plan_sha256=_sha("2"),
    )
    job = _job()
    receipt = object()
    calls: dict[str, object] = {}
    draft_root = tmp_path / "published-drafts"
    execution_root = tmp_path / "execution"
    receipt_path = tmp_path / "legacy-receipt.json"

    monkeypatch.setattr(execution_cli, "_rebuild_context", lambda arguments: context)
    monkeypatch.setattr(
        execution_cli,
        "_build_authoring_job_for_batch",
        lambda rebuilt, batch_id: (
            calls.update({"context": rebuilt, "batch_id": batch_id}) or job
        ),
    )
    monkeypatch.setattr(
        execution_cli,
        "_load_body_free_legacy_receipt",
        lambda path: calls.update({"receipt_path": path}) or receipt,
    )
    monkeypatch.setattr(
        execution_cli,
        "auto_approve_portfolio_core_r2_batch",
        lambda result, bridge, **kwargs: calls.update(
            {"result": result, "bridge": bridge, **kwargs}
        ),
    )
    monkeypatch.setattr(execution_cli, "canonical_author_packet_bytes", lambda _: b"x")
    monkeypatch.setattr(execution_cli, "sha256_bytes", lambda _: _sha("3"))

    assert (
        execution_cli.main(
            [
                "approve-published-draft",
                *_common_arguments(tmp_path),
                "--job-batch-id",
                "core-001",
                "--draft-artifact-root",
                str(draft_root),
                "--execution-root",
                str(execution_root),
                "--legacy-turn-exclusion-receipt",
                str(receipt_path),
            ]
        )
        == 0
    )

    assert calls == {
        "context": context,
        "batch_id": "core-001",
        "receipt_path": receipt_path,
        "result": context.result,
        "bridge": context.bridge,
        "trusted_plan_sha256": context.trusted_plan_sha256,
        "legacy_turn_exclusion_receipt": receipt,
        "authoring_job": job,
        "draft_artifact_root": draft_root,
        "execution_root": execution_root,
    }
    assert json.loads(capsys.readouterr().out) == {
        "alias_count": 3,
        "job_id": job.work_order.job_id,
        "packet_sha256": _sha("3"),
    }


def test_rebuild_context_uses_verified_catalogs_and_locked_r2_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_cli,
) -> None:
    arguments = execution_cli.build_parser().parse_args(
        [
            "prepare-job",
            *_common_arguments(tmp_path),
            "--batch-id",
            "core-001",
            "--jobs-root",
            str(tmp_path / "jobs"),
        ]
    )
    parent_plan = object()
    target_catalog = object()
    result = SimpleNamespace(plan={"reconstructed": "r2"})
    calls: dict[str, list[object]] = {"catalogs": [], "assignments": []}

    monkeypatch.setattr(
        execution_cli,
        "load_plan",
        lambda _: (parent_plan, SimpleNamespace(plan_sha256=_sha("e"))),
    )
    monkeypatch.setattr(
        execution_cli,
        "load_asset_catalog",
        lambda catalog, asset_root, *, verify_files: (
            calls["catalogs"].append((catalog, asset_root, verify_files))
            or (target_catalog if catalog == arguments.target_catalog else object())
        ),
    )
    monkeypatch.setattr(
        execution_cli,
        "load_capability_assignments",
        lambda path, *, expected_sha256: (
            calls["assignments"].append((path, expected_sha256))
            or ("target" if path == arguments.target_assignments else "parent")
        ),
    )
    monkeypatch.setattr(
        execution_cli, "rebind_r2_dev_prefix", lambda *_, **__: "rebind"
    )
    monkeypatch.setattr(
        execution_cli,
        "build_r2_core_in_memory_plan",
        lambda *_, **kwargs: calls.setdefault("layout", []).append(kwargs) or result,
    )
    monkeypatch.setattr(execution_cli, "canonical_json_bytes", lambda _: b"r2-plan")
    monkeypatch.setattr(execution_cli, "sha256_bytes", lambda _: _sha("4"))
    monkeypatch.setattr(
        execution_cli,
        "build_r2_core_bridge",
        lambda built, *, trusted_plan_sha256, seed: (
            calls.setdefault("bridge", []).append((built, trusted_plan_sha256, seed))
            or "bridge"
        ),
    )

    rebuilt = execution_cli._rebuild_context(arguments)

    assert calls["catalogs"] == [
        (arguments.parent_catalog, arguments.parent_asset_root, True),
        (arguments.target_catalog, arguments.target_asset_root, True),
    ]
    assert calls["assignments"] == [
        (arguments.parent_assignments, _sha("c")),
        (arguments.target_assignments, _sha("d")),
    ]
    assert calls["layout"] == [
        {
            "target_catalog": target_catalog,
            "target_capability_assignments": "target",
            "expected_target_catalog_sha256": _sha("b"),
            "expected_target_capability_assignments_sha256": _sha("d"),
            "seed": 20260804,
        }
    ]
    assert calls["bridge"] == [(result, _sha("4"), 20260805)]
    assert rebuilt.result is result
    assert rebuilt.bridge == "bridge"
    assert rebuilt.trusted_plan_sha256 == _sha("4")
    assert rebuilt.target_catalog is target_catalog


def test_rebuild_context_rejects_a_reconstructed_r2_plan_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_cli,
) -> None:
    arguments = execution_cli.build_parser().parse_args(
        [
            "prepare-job",
            *_common_arguments(tmp_path, expected_r2_plan_sha256=_sha("f")),
            "--batch-id",
            "core-001",
            "--jobs-root",
            str(tmp_path / "jobs"),
        ]
    )
    target_catalog = object()
    result = SimpleNamespace(plan={"reconstructed": "r2"})

    monkeypatch.setattr(
        execution_cli,
        "load_plan",
        lambda _: (object(), SimpleNamespace(plan_sha256=_sha("e"))),
    )
    monkeypatch.setattr(
        execution_cli,
        "load_asset_catalog",
        lambda catalog, asset_root, *, verify_files: (
            target_catalog if catalog == arguments.target_catalog else object()
        ),
    )
    monkeypatch.setattr(
        execution_cli,
        "load_capability_assignments",
        lambda path, *, expected_sha256: (
            "target" if path == arguments.target_assignments else "parent"
        ),
    )
    monkeypatch.setattr(
        execution_cli, "rebind_r2_dev_prefix", lambda *_, **__: "rebind"
    )
    monkeypatch.setattr(
        execution_cli,
        "build_r2_core_in_memory_plan",
        lambda *_, **__: result,
    )
    monkeypatch.setattr(execution_cli, "canonical_json_bytes", lambda _: b"r2-plan")
    monkeypatch.setattr(execution_cli, "sha256_bytes", lambda _: _sha("4"))
    monkeypatch.setattr(
        execution_cli,
        "build_r2_core_bridge",
        lambda *_, **__: pytest.fail("bridge must not be built after r2 hash drift"),
    )

    with pytest.raises(ValueError, match="expected_r2_plan_sha256"):
        execution_cli._rebuild_context(arguments)
