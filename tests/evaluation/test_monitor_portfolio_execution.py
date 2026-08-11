from __future__ import annotations

from decimal import Decimal
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.monitor_portfolio_execution import (
    _active_worker_processes,
    _execution_status,
    _load_control,
    _no_progress_recovery_required,
    _progress_key,
    _require_empty_initial_parallel_gate,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _status(
    *,
    completed: int,
    settlements: int,
    events: int | None = None,
) -> dict[str, object]:
    return {
        "completed_shards": completed,
        "ledger_settlements": settlements,
        **({} if events is None else {"ledger_events": events}),
    }


def _core_shards(count: int = 5) -> tuple[SimpleNamespace, ...]:
    configs = ("noskill", "llm_static", "s1", "s1s2", "full")
    return tuple(
        SimpleNamespace(
            shard_id=f"shard-{index}",
            accepted_batch_id=("dev-mini-001" if index < 5 else f"batch-{index}"),
            config=(configs[index] if index < 5 else "noskill"),
            query_ids=(tuple(f"q-{ordinal}" for ordinal in range(25))),
        )
        for index in range(count)
    )


def _initial_launch(*, status: str = "not_started") -> SimpleNamespace:
    return SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id="core-canary",
            kind="portfolio-core-split-x5-launch-plan",
            selected_splits=("dev_mini",),
            shards=_core_shards(),
        ),
        state=SimpleNamespace(
            completed_shard_ids=(),
            failed_shard_ids=(),
            status=status,
            model_calls_performed=0,
            dashscope_observed_cost_cny=None,
            aifast_observed_cost_cny=None,
        ),
    )


def _initial_control() -> dict[str, object]:
    return {
        "schema_version": 4,
        "status": "prepared_not_started",
        "model_calls_performed": 0,
        "execution_scope": "core_canary",
        "matrix_run_id": "core-canary",
        "launch_root": "launch",
        "launch_plan_file_sha256": "a" * 64,
        "launch_shard_count": 5,
        "authorized_shard_ids": [f"shard-{index}" for index in range(5)],
        "external_frozen_shard_count": 0,
        "budget_ledger_relpath": "budget-ledger",
        "budget_authority_relpath": "budget-ledger/budget-authority.json",
        "budget_authority_creation_policy": "create_only_on_first_execute",
        "approved_dashscope_budget_cny": "100.000000000000",
        "prior_dashscope_observed_cost_cny": "0.000000000000",
        "phase_cumulative_cap_cny": "30.000000000000",
        "incremental_authorized_dashscope_budget_cny": "30.000000000000",
    }


def _write_initial_execution_root(
    root: Path,
    control: dict[str, object] | None = None,
    *,
    key: bytes = b"x" * 32,
) -> dict[str, object]:
    unsigned = dict(_initial_control() if control is None else control)
    unsigned.pop("control_sha256", None)
    unsigned["blinding_key_sha256"] = sha256_bytes(key)
    bound = {
        **unsigned,
        "control_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    (root / "execution-control.json").write_bytes(canonical_json_bytes(bound))
    (root / "blinding-key.bin").write_bytes(key)
    return bound


def _write_parallel_gate(
    root: Path,
    *,
    with_start: bool = False,
    call_level: bool = False,
) -> None:
    connection = sqlite3.connect(root / "parallel-gate.sqlite3")
    try:
        connection.execute(
            "CREATE TABLE provider_permits ("
            "owner TEXT PRIMARY KEY, stage TEXT NOT NULL, pid INTEGER NOT NULL, "
            "acquired_at REAL NOT NULL)"
        )
        if call_level:
            connection.execute(
                "CREATE TABLE provider_call_starts ("
                "start_id TEXT PRIMARY KEY, label TEXT NOT NULL, "
                "pid INTEGER NOT NULL, started_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX provider_call_starts_started_at "
                "ON provider_call_starts(started_at)"
            )
        else:
            connection.execute(
                "CREATE TABLE provider_starts ("
                "start_id TEXT PRIMARY KEY, stage TEXT NOT NULL, "
                "started_at REAL NOT NULL)"
            )
        if with_start:
            if call_level:
                connection.execute(
                    "INSERT INTO provider_call_starts VALUES ('start', 'label', 1, 1.0)"
                )
            else:
                connection.execute(
                    "INSERT INTO provider_starts VALUES ('start', 'assistant', 1.0)"
                )
        connection.commit()
    finally:
        connection.close()


def test_initial_parallel_gate_accepts_empty_call_level_schema(tmp_path: Path) -> None:
    _write_parallel_gate(tmp_path, call_level=True)

    _require_empty_initial_parallel_gate(tmp_path)


def _write_bound_control(root: Path) -> dict[str, object]:
    key = b"k" * 32
    (root / "blinding-key.bin").write_bytes(key)
    unsigned = {
        "kind": "portfolio-matrix-execution-control",
        "blinding_key_sha256": sha256_bytes(key),
    }
    control = {
        **unsigned,
        "control_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }
    (root / "execution-control.json").write_bytes(canonical_json_bytes(control))
    return control


def test_monitor_loads_only_stable_self_hashed_control_and_bound_key(
    tmp_path: Path,
) -> None:
    expected = _write_bound_control(tmp_path)

    assert _load_control(tmp_path) == expected


def test_monitor_rejects_execution_control_with_invalid_self_hash(
    tmp_path: Path,
) -> None:
    control = _write_bound_control(tmp_path)
    control["kind"] = "tampered"
    (tmp_path / "execution-control.json").write_bytes(canonical_json_bytes(control))

    with pytest.raises(ValueError, match="self hash mismatch"):
        _load_control(tmp_path)


def test_monitor_rejects_control_with_unbound_blinding_key(tmp_path: Path) -> None:
    _write_bound_control(tmp_path)
    (tmp_path / "blinding-key.bin").write_bytes(b"z" * 32)

    with pytest.raises(ValueError, match="blinding key mismatch"):
        _load_control(tmp_path)


@pytest.mark.parametrize("artifact", ["execution-control.json", "blinding-key.bin"])
def test_monitor_rejects_symlinked_control_or_key(
    tmp_path: Path,
    artifact: str,
) -> None:
    _write_bound_control(tmp_path)
    path = tmp_path / artifact
    backing = tmp_path / f"{artifact}.backing"
    path.replace(backing)
    try:
        path.symlink_to(backing)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    with pytest.raises(ValueError, match="regular non-symlink"):
        _load_control(tmp_path)


@pytest.mark.parametrize(
    "sidecar",
    ["parallel-gate.sqlite3-wal", "parallel-gate.sqlite3-shm"],
)
def test_initial_parallel_gate_rejects_any_sidecar(
    tmp_path: Path,
    sidecar: str,
) -> None:
    _write_parallel_gate(tmp_path)
    (tmp_path / sidecar).write_bytes(b"sidecar")

    with pytest.raises(ValueError, match="must not have WAL/SHM sidecars"):
        _require_empty_initial_parallel_gate(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_view",
        "extra_index",
        "extra_trigger",
        "extra_column",
        "provider_row",
    ],
)
def test_initial_parallel_gate_rejects_schema_or_activity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_parallel_gate(tmp_path)
    with sqlite3.connect(tmp_path / "parallel-gate.sqlite3") as connection:
        if mutation == "extra_view":
            connection.execute("CREATE VIEW leaked_activity AS SELECT 1")
        elif mutation == "extra_index":
            connection.execute(
                "CREATE INDEX leaked_index ON provider_starts(started_at)"
            )
        elif mutation == "extra_trigger":
            connection.execute(
                "CREATE TRIGGER leaked_trigger AFTER INSERT ON provider_starts "
                "BEGIN SELECT 1; END"
            )
        elif mutation == "extra_column":
            connection.execute("ALTER TABLE provider_starts ADD COLUMN extra TEXT")
        else:
            connection.execute(
                "INSERT INTO provider_starts VALUES ('start', 'assistant', 1.0)"
            )

    expected = {
        "extra_view": "unexpected objects",
        "extra_index": "unexpected objects",
        "extra_trigger": "unexpected objects",
        "extra_column": "schema drifted",
        "provider_row": "provider activity",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        _require_empty_initial_parallel_gate(tmp_path)


def test_initial_parallel_gate_rejects_corrupt_database(tmp_path: Path) -> None:
    (tmp_path / "parallel-gate.sqlite3").write_bytes(b"not a sqlite database")

    with pytest.raises(ValueError, match="cannot be verified"):
        _require_empty_initial_parallel_gate(tmp_path)


def test_initial_parallel_gate_rejects_symlink(tmp_path: Path) -> None:
    _write_parallel_gate(tmp_path)
    gate = tmp_path / "parallel-gate.sqlite3"
    backing = tmp_path / "gate.backing.sqlite3"
    gate.replace(backing)
    try:
        gate.symlink_to(backing)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    with pytest.raises(ValueError, match="regular non-symlink"):
        _require_empty_initial_parallel_gate(tmp_path)


def test_initial_parallel_gate_rejects_stat_drift(monkeypatch, tmp_path: Path) -> None:
    import scripts.monitor_portfolio_execution as monitor

    _write_parallel_gate(tmp_path)
    gate = tmp_path / "parallel-gate.sqlite3"
    original = monitor.read_stable_regular_file
    gate_reads = 0

    def mutate_after_first_gate_read(path, **kwargs):
        nonlocal gate_reads
        content = original(path, **kwargs)
        if Path(path) == gate:
            gate_reads += 1
            if gate_reads == 1:
                metadata = gate.stat()
                os.utime(
                    gate,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000),
                )
        return content

    monkeypatch.setattr(
        monitor, "read_stable_regular_file", mutate_after_first_gate_read
    )

    with pytest.raises(ValueError, match="changed during verification"):
        _require_empty_initial_parallel_gate(tmp_path)


def test_monitor_stops_restarting_when_no_accountable_progress_occurs() -> None:
    status = _status(completed=5, settlements=11, events=23)
    assert _progress_key(status) == (5, 23)
    assert _no_progress_recovery_required(
        restarts=1,
        last_restart_progress=(5, 23),
        status=status,
    )
    assert not _no_progress_recovery_required(
        restarts=1,
        last_restart_progress=(4, 23),
        status=status,
    )
    assert not _no_progress_recovery_required(
        restarts=0,
        last_restart_progress=(5, 23),
        status=status,
    )


def test_monitor_progress_counts_reservation_or_forfeit_events() -> None:
    before = _status(completed=2, settlements=7, events=15)
    after_reservation = _status(completed=2, settlements=7, events=16)
    after_forfeit = _status(completed=2, settlements=7, events=17)

    assert _progress_key(before) == (2, 15)
    assert _progress_key(after_reservation) == (2, 16)
    assert _progress_key(after_forfeit) == (2, 17)


def test_monitor_progress_remains_backward_compatible_without_event_count() -> None:
    status = _status(completed=5, settlements=11)
    assert _progress_key(status) == (5, 11)
    assert _no_progress_recovery_required(
        restarts=1,
        last_restart_progress=(5, 11),
        status=status,
    )
    assert not _no_progress_recovery_required(
        restarts=1,
        last_restart_progress=(4, 11),
        status=status,
    )
    assert not _no_progress_recovery_required(
        restarts=0,
        last_restart_progress=(5, 11),
        status=status,
    )


def test_active_worker_process_scan_is_exposed_for_supervision_guard() -> None:
    # The helper is intentionally callable with a path that has no matching
    # process; the monitor uses this result to avoid a duplicate matrix start
    # when a Windows matrix wrapper disappears before its shard child exits.
    assert _active_worker_processes(Path(__file__)) == ()


def test_monitor_reports_zero_before_create_on_first_execute_ledger(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    control = _write_initial_execution_root(tmp_path)
    _write_parallel_gate(tmp_path)
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: pytest.fail("strict initial state must not load a ledger"),
    )

    status = _execution_status(tmp_path, control)

    assert status["ledger_events"] == 0
    assert status["ledger_reservations"] == 0
    assert status["ledger_settlements"] == 0
    assert status["ledger_unresolved"] == 0
    assert status["accountable_cost_cny"] == "0.000000000000"
    assert status["phase_cap_cny"] == "30.000000000000"
    assert status["model_calls_performed"] == 0
    assert status["launch_state_model_calls_performed"] == 0
    assert status["complete"] is False


def test_monitor_carries_nonzero_prior_before_create_on_first_execute_ledger(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    initial = _initial_control()
    initial["prior_dashscope_observed_cost_cny"] = "11.000000000000"
    initial["incremental_authorized_dashscope_budget_cny"] = "19.000000000000"
    control = _write_initial_execution_root(tmp_path, initial)
    _write_parallel_gate(tmp_path)
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: pytest.fail("strict initial state must not load a ledger"),
    )

    status = _execution_status(tmp_path, control)

    assert status["ledger_events"] == 0
    assert status["ledger_reservations"] == 0
    assert status["ledger_settlements"] == 0
    assert status["ledger_unresolved"] == 0
    assert status["accountable_cost_cny"] == "11.000000000000"
    assert status["phase_cap_cny"] == "30.000000000000"
    assert status["model_calls_performed"] == 0
    assert status["launch_state_model_calls_performed"] == 0
    assert status["complete"] is False


@pytest.mark.parametrize(
    "sidecars",
    [
        ("parallel-gate.sqlite3-wal",),
        ("parallel-gate.sqlite3-shm", "parallel-gate.sqlite3-wal"),
    ],
)
def test_monitor_defers_live_parallel_gate_verification_during_active_startup(
    monkeypatch,
    tmp_path: Path,
    sidecars: tuple[str, ...],
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    control = _write_initial_execution_root(tmp_path)
    _write_parallel_gate(tmp_path)
    for sidecar in sidecars:
        (tmp_path / sidecar).write_bytes(b"live mutable sqlite sidecar")
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )

    status = _execution_status(
        tmp_path,
        control,
        allow_active_startup_transition=True,
    )

    assert status["ledger_startup_transition"] is True
    assert status["ledger_events"] == 0
    assert status["ledger_reservations"] == 0
    assert status["ledger_settlements"] == 0
    assert status["ledger_forfeits"] == 0
    assert status["ledger_unresolved"] == 0
    assert status["model_calls_performed"] == 0
    assert status["complete"] is False


def test_monitor_still_rejects_live_gate_sidecars_without_matching_activity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    control = _write_initial_execution_root(tmp_path)
    _write_parallel_gate(tmp_path)
    (tmp_path / "parallel-gate.sqlite3-wal").write_bytes(b"orphaned sidecar")
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )

    with pytest.raises(ValueError, match="must not have WAL/SHM sidecars"):
        _execution_status(tmp_path, control)


@pytest.mark.parametrize("with_ledger", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "prior_equals_phase_cap",
        "phase_cap_exceeds_approved",
        "incremental_mismatch",
    ],
)
def test_monitor_rejects_inconsistent_cumulative_budget_with_or_without_ledger(
    monkeypatch,
    tmp_path: Path,
    with_ledger: bool,
    mutation: str,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    initial = _initial_control()
    if mutation == "prior_equals_phase_cap":
        initial["prior_dashscope_observed_cost_cny"] = "30.000000000000"
        initial["incremental_authorized_dashscope_budget_cny"] = "0.000000000000"
    elif mutation == "phase_cap_exceeds_approved":
        initial["phase_cumulative_cap_cny"] = "101.000000000000"
        initial["incremental_authorized_dashscope_budget_cny"] = "101.000000000000"
    else:
        initial["incremental_authorized_dashscope_budget_cny"] = "29.000000000000"
    control = _write_initial_execution_root(tmp_path, initial)
    if with_ledger:
        (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )

    with pytest.raises(ValueError, match="cumulative budget fields are inconsistent"):
        _execution_status(tmp_path, control)


def test_monitor_reloads_control_and_key_during_missing_ledger_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    startup_control = _write_initial_execution_root(tmp_path)
    _write_initial_execution_root(tmp_path, key=b"z" * 32)
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )

    with pytest.raises(ValueError, match="changed after monitor startup"):
        _execution_status(tmp_path, startup_control)


def test_monitor_rejects_partial_ledger_instead_of_treating_it_as_initial(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    control = _write_initial_execution_root(tmp_path)
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: _initial_launch(),
    )

    def reject_partial(*_args, **_kwargs):
        raise ValueError("partial ledger")

    monkeypatch.setattr(monitor, "load_portfolio_budget_ledger", reject_partial)

    with pytest.raises(ValueError, match="partial ledger"):
        _execution_status(tmp_path, control)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("noninitial_launch", "strict initial state"),
        ("control_calls", "strict initial state"),
        ("wrong_scope", "execution_scope=core_canary"),
        ("wrong_matrix", "matrix run differs from launch"),
        ("inconsistent_prior", "cumulative budget fields are inconsistent"),
        ("run_artifact", "execution artifacts appeared"),
        ("provider_start", "provider activity"),
        ("control_symlink", "regular non-symlink"),
    ],
)
def test_monitor_rejects_missing_ledger_after_any_execution_evidence(
    monkeypatch, tmp_path: Path, mutation: str, expected: str
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    control = _initial_control()
    launch = _initial_launch(
        status="running" if mutation == "noninitial_launch" else "not_started"
    )
    if mutation == "control_calls":
        control["model_calls_performed"] = 1
    elif mutation == "wrong_scope":
        control["execution_scope"] = "full_matrix"
    elif mutation == "wrong_matrix":
        control["matrix_run_id"] = "another-run"
    elif mutation == "inconsistent_prior":
        control["prior_dashscope_observed_cost_cny"] = "1.000000000000"
    control = _write_initial_execution_root(tmp_path, control)
    if mutation == "run_artifact":
        (tmp_path / "shards").mkdir()
    elif mutation == "provider_start":
        _write_parallel_gate(tmp_path, with_start=True)
    elif mutation == "control_symlink":
        control_path = tmp_path / "execution-control.json"
        backing = tmp_path.parent / f"{tmp_path.name}-control.backing"
        control_path.replace(backing)
        try:
            control_path.symlink_to(backing)
        except OSError:
            pytest.skip("symlink creation is not permitted")
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )

    with pytest.raises(ValueError, match=expected):
        _execution_status(tmp_path, control)


def test_monitor_completion_uses_authorized_subset_not_full_launch(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = SimpleNamespace(
        plan=SimpleNamespace(
            matrix_run_id="core-canary",
            kind="portfolio-core-split-x5-launch-plan",
            selected_splits=("dev_mini",),
            shards=_core_shards(40),
        ),
        state=SimpleNamespace(
            completed_shard_ids=tuple(f"shard-{index}" for index in range(5)),
            status="running",
            model_calls_performed=17,
        ),
    )
    ledger = SimpleNamespace(
        last_event_index=17,
        reservations=tuple(range(17)),
        settlements=tuple(range(17)),
        unresolved_reservations=(),
        accountable_cost_cny=Decimal("1.25"),
        authority=SimpleNamespace(
            matrix_run_id="core-canary",
            phase_cap_cny=Decimal("30"),
            prior_observed_cost_cny=Decimal("0"),
        ),
    )
    monkeypatch.setattr(
        monitor, "load_portfolio_launch_package", lambda *_a, **_k: launch
    )
    monkeypatch.setattr(
        monitor, "load_portfolio_budget_ledger", lambda *_a, **_k: ledger
    )
    (tmp_path / "budget-ledger").mkdir()
    control = {
        "execution_scope": "core_canary",
        "matrix_run_id": "core-canary",
        "launch_root": "launch",
        "launch_plan_file_sha256": "a" * 64,
        "launch_shard_count": 40,
        "authorized_shard_ids": [f"shard-{index}" for index in range(5)],
        "external_frozen_shard_count": 0,
        "approved_dashscope_budget_cny": "100.000000000000",
        "phase_cumulative_cap_cny": "30.000000000000",
        "incremental_authorized_dashscope_budget_cny": "30.000000000000",
        "prior_dashscope_observed_cost_cny": "0.000000000000",
    }

    status = _execution_status(tmp_path, control)

    assert status["complete"] is True
    assert status["authorized_completed_shards"] == 5
    assert status["authorized_shards"] == 5
    assert status["launch_shards"] == 40
    assert status["model_calls_performed"] == 17
    assert status["launch_state_model_calls_performed"] == 17


def test_monitor_accepts_complete_static_opt_rollout(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    shards = tuple(
        SimpleNamespace(shard_id=f"static-{index:02d}") for index in range(32)
    )
    launch = SimpleNamespace(
        plan=SimpleNamespace(matrix_run_id="static-opt", shards=shards),
        state=SimpleNamespace(
            completed_shard_ids=tuple(item.shard_id for item in shards),
            status="complete",
            model_calls_performed=1600,
        ),
    )
    ledger = SimpleNamespace(
        last_event_index=3200,
        reservations=tuple(range(1600)),
        settlements=tuple(range(1600)),
        forfeits=(),
        unresolved_reservations=(),
        forfeited_reserved_cost_cny=Decimal("0"),
        accountable_cost_cny=Decimal("9"),
        authority=SimpleNamespace(
            matrix_run_id="static-opt",
            phase_cap_cny=Decimal("20"),
            prior_observed_cost_cny=Decimal("0"),
        ),
    )
    control = {
        "execution_scope": "static_opt_rollout",
        "matrix_run_id": "static-opt",
        "launch_root": "launch",
        "launch_plan_file_sha256": "a" * 64,
        "launch_shard_count": 32,
        "approved_dashscope_budget_cny": "20.000000000000",
        "phase_cumulative_cap_cny": "20.000000000000",
        "incremental_authorized_dashscope_budget_cny": "20.000000000000",
        "prior_dashscope_observed_cost_cny": "0.000000000000",
    }
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor, "load_portfolio_launch_package", lambda *_a, **_k: launch
    )
    monkeypatch.setattr(
        monitor, "load_portfolio_budget_ledger", lambda *_a, **_k: ledger
    )
    monkeypatch.setattr(monitor, "_ordered_authorized_shards", lambda *_a, **_k: shards)
    monkeypatch.setattr(
        monitor,
        "load_verified_portfolio_static_opt_runtime",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        monitor,
        "validate_portfolio_static_opt_execution_control",
        lambda *_a, **_k: None,
    )
    control["runtime_root"] = "static-runtime"
    control["runtime_lock_file_sha256"] = "b" * 64

    status = _execution_status(tmp_path, control)

    assert status["complete"] is True
    assert status["authorized_shards"] == 32
    assert status["authorized_completed_shards"] == 32
    assert status["ledger_unresolved"] == 0


def test_monitor_reports_live_ledger_calls_separately_from_launch_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = _initial_launch(status="running")
    launch.state.model_calls_performed = 17
    control = _initial_control()
    control["prior_dashscope_observed_cost_cny"] = "11.000000000000"
    control["incremental_authorized_dashscope_budget_cny"] = "19.000000000000"
    ledger = SimpleNamespace(
        last_event_index=45,
        reservations=tuple(range(23)),
        settlements=tuple(range(22)),
        unresolved_reservations=(object(),),
        accountable_cost_cny=Decimal("12.5"),
        authority=SimpleNamespace(
            matrix_run_id="core-canary",
            phase_cap_cny=Decimal("30"),
            prior_observed_cost_cny=Decimal("11"),
        ),
    )
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: ledger,
    )

    status = _execution_status(tmp_path, control)

    assert status["model_calls_performed"] == 23
    assert status["launch_state_model_calls_performed"] == 17
    assert status["ledger_settlements"] == 22
    assert status["ledger_unresolved"] == 1
    assert status["accountable_cost_cny"] == "12.500000000000"
    assert status["complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "narrow_ids",
        "reordered_ids",
        "wrong_config",
        "wrong_batch",
        "wrong_queries",
        "wrong_kind",
        "wrong_split",
    ],
)
def test_monitor_revalidates_exact_core_canary_authorization_before_completion(
    monkeypatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = _initial_launch(status="running")
    launch.state.completed_shard_ids = tuple(f"shard-{index}" for index in range(5))
    control = _initial_control()
    if mutation == "narrow_ids":
        control["authorized_shard_ids"] = control["authorized_shard_ids"][:4]
    elif mutation == "reordered_ids":
        control["authorized_shard_ids"] = [
            control["authorized_shard_ids"][1],
            control["authorized_shard_ids"][0],
            *control["authorized_shard_ids"][2:],
        ]
    elif mutation == "wrong_config":
        launch.plan.shards[0].config = "s1"
    elif mutation == "wrong_batch":
        launch.plan.shards[0].accepted_batch_id = "another-batch"
    elif mutation == "wrong_queries":
        launch.plan.shards[0].query_ids = ("another-query",)
    elif mutation == "wrong_kind":
        launch.plan.kind = "portfolio-launch-plan"
    else:
        launch.plan.selected_splits = ("opt",)
    ledger = SimpleNamespace(
        last_event_index=10,
        reservations=tuple(range(5)),
        settlements=tuple(range(5)),
        unresolved_reservations=(),
        accountable_cost_cny=Decimal("1"),
        authority=SimpleNamespace(
            matrix_run_id="core-canary",
            phase_cap_cny=Decimal("30"),
            prior_observed_cost_cny=Decimal("0"),
        ),
    )
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: ledger,
    )

    with pytest.raises(ValueError, match="Core canary authorization"):
        _execution_status(tmp_path, control)


def test_monitor_does_not_complete_with_unresolved_budget_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = _initial_launch(status="running")
    launch.state.completed_shard_ids = tuple(f"shard-{index}" for index in range(5))
    ledger = SimpleNamespace(
        last_event_index=9,
        reservations=tuple(range(5)),
        settlements=tuple(range(4)),
        unresolved_reservations=(object(),),
        accountable_cost_cny=Decimal("2"),
        authority=SimpleNamespace(
            matrix_run_id="core-canary",
            phase_cap_cny=Decimal("30"),
            prior_observed_cost_cny=Decimal("0"),
        ),
    )
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: ledger,
    )

    status = _execution_status(tmp_path, _initial_control())

    assert status["authorized_completed_shards"] == 5
    assert status["ledger_unresolved"] == 1
    assert status["complete"] is False


def test_monitor_rejects_completed_shard_outside_exact_core_canary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = _initial_launch(status="running")
    launch.plan.shards = _core_shards(6)
    launch.state.completed_shard_ids = (
        *(f"shard-{index}" for index in range(5)),
        "shard-5",
    )
    ledger = SimpleNamespace(
        last_event_index=10,
        reservations=tuple(range(5)),
        settlements=tuple(range(5)),
        unresolved_reservations=(),
        accountable_cost_cny=Decimal("1"),
        authority=SimpleNamespace(
            matrix_run_id="core-canary",
            phase_cap_cny=Decimal("30"),
            prior_observed_cost_cny=Decimal("0"),
        ),
    )
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: ledger,
    )

    with pytest.raises(ValueError, match="outside exact Core canary"):
        _execution_status(tmp_path, _initial_control())


@pytest.mark.parametrize("mismatch", ["matrix_run_id", "phase_cap_cny", "prior_cost"])
def test_monitor_rejects_ledger_authority_that_differs_from_control(
    monkeypatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    import scripts.monitor_portfolio_execution as monitor

    launch = _initial_launch(status="running")
    authority = {
        "matrix_run_id": "core-canary",
        "phase_cap_cny": Decimal("30"),
        "prior_observed_cost_cny": Decimal("0"),
    }
    if mismatch == "matrix_run_id":
        authority["matrix_run_id"] = "another-run"
    elif mismatch == "phase_cap_cny":
        authority["phase_cap_cny"] = Decimal("29")
    else:
        authority["prior_observed_cost_cny"] = Decimal("1")
    ledger = SimpleNamespace(
        last_event_index=0,
        reservations=(),
        settlements=(),
        unresolved_reservations=(),
        accountable_cost_cny=Decimal("0"),
        authority=SimpleNamespace(**authority),
    )
    (tmp_path / "budget-ledger").mkdir()
    monkeypatch.setattr(
        monitor,
        "load_portfolio_launch_package",
        lambda *_a, **_k: launch,
    )
    monkeypatch.setattr(
        monitor,
        "load_portfolio_budget_ledger",
        lambda *_a, **_k: ledger,
    )

    with pytest.raises(ValueError, match="authority differs"):
        _execution_status(tmp_path, _initial_control())
