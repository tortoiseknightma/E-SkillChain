"""Safely supervise and recover one Portfolio matrix execution root.

The monitor never creates a new launch or changes an experiment input.  It
only restarts the same matrix command after a recoverable worker/matrix exit;
the execution code itself remains responsible for orphan handling, budget
reservations, checkpoints, and all fail-closed decisions.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.portfolio_execution import (  # noqa: E402
    load_portfolio_budget_ledger,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
)
from scripts.run_portfolio_matrix import (  # noqa: E402
    _ordered_authorized_shards,
    validate_runtime_binding,
)
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    parse_canonical_json,
    read_stable_regular_file,
    sha256_bytes,
)


_CONTROL_MAX_BYTES = 256 * 1024
_PARALLEL_GATE_MAX_BYTES = 4 * 1024 * 1024
_LEGACY_PARALLEL_GATE_TABLE_SCHEMAS = {
    "provider_permits": (
        (0, "owner", "TEXT", 0, None, 1),
        (1, "stage", "TEXT", 1, None, 0),
        (2, "pid", "INTEGER", 1, None, 0),
        (3, "acquired_at", "REAL", 1, None, 0),
    ),
    "provider_starts": (
        (0, "start_id", "TEXT", 0, None, 1),
        (1, "stage", "TEXT", 1, None, 0),
        (2, "started_at", "REAL", 1, None, 0),
    ),
}
_LEGACY_PARALLEL_GATE_SQLITE_OBJECTS = {
    ("table", "provider_permits", "provider_permits"),
    ("table", "provider_starts", "provider_starts"),
    ("index", "sqlite_autoindex_provider_permits_1", "provider_permits"),
    ("index", "sqlite_autoindex_provider_starts_1", "provider_starts"),
}
_CALL_LEVEL_PARALLEL_GATE_TABLE_SCHEMAS = {
    "provider_permits": _LEGACY_PARALLEL_GATE_TABLE_SCHEMAS["provider_permits"],
    "provider_call_starts": (
        (0, "start_id", "TEXT", 0, None, 1),
        (1, "label", "TEXT", 1, None, 0),
        (2, "pid", "INTEGER", 1, None, 0),
        (3, "started_at", "REAL", 1, None, 0),
    ),
}
_CALL_LEVEL_PARALLEL_GATE_SQLITE_OBJECTS = {
    ("table", "provider_permits", "provider_permits"),
    ("table", "provider_call_starts", "provider_call_starts"),
    ("index", "sqlite_autoindex_provider_permits_1", "provider_permits"),
    (
        "index",
        "sqlite_autoindex_provider_call_starts_1",
        "provider_call_starts",
    ),
    (
        "index",
        "provider_call_starts_started_at",
        "provider_call_starts",
    ),
}
_CALL_LEVEL_START_INDEX_SQL = (
    "CREATE INDEX provider_call_starts_started_at ON provider_call_starts(started_at)"
)
_PARALLEL_GATE_SCHEMA_VARIANTS = (
    (
        _LEGACY_PARALLEL_GATE_SQLITE_OBJECTS,
        _LEGACY_PARALLEL_GATE_TABLE_SCHEMAS,
    ),
    (
        _CALL_LEVEL_PARALLEL_GATE_SQLITE_OBJECTS,
        _CALL_LEVEL_PARALLEL_GATE_TABLE_SCHEMAS,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--parallel-profile", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--matrix-log", type=Path, required=True)
    return parser


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(message.rstrip() + "\n")


def _load_control(execution_root: Path) -> dict:
    control_bytes = read_stable_regular_file(
        execution_root / "execution-control.json",
        label="Portfolio monitor execution control",
        max_bytes=_CONTROL_MAX_BYTES,
    )
    control = parse_canonical_json(
        control_bytes,
        label="Portfolio monitor execution control",
    )
    if not isinstance(control, dict):
        raise ValueError("Portfolio monitor execution control must be an object")
    supplied_sha256 = control.get("control_sha256")
    unsigned = dict(control)
    unsigned.pop("control_sha256", None)
    if supplied_sha256 != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("Portfolio monitor execution control self hash mismatch")
    key = read_stable_regular_file(
        execution_root / "blinding-key.bin",
        label="Portfolio monitor blinding key",
        max_bytes=32,
    )
    if len(key) != 32 or sha256_bytes(key) != control.get("blinding_key_sha256"):
        raise ValueError("Portfolio monitor blinding key mismatch")
    return control


def _matrix_processes(execution_root: Path) -> tuple[psutil.Process, ...]:
    root_token = str(execution_root.resolve()).casefold()
    found: list[psutil.Process] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command_line = " ".join(process.info.get("cmdline") or []).casefold()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if (
            "run_portfolio_matrix.py" in command_line
            and root_token in command_line
            and process.pid != 0
        ):
            found.append(process)
    return tuple(found)


def _active_worker_processes(execution_root: Path) -> tuple[psutil.Process, ...]:
    """Return shard/finalizer workers still attached to this execution root.

    On Windows the matrix wrapper can disappear before its child worker exits.
    Restarting in that window creates a second worker and reuses an unresolved
    budget identity, which is classified as an orphaned provider call. Keep
    supervision paused while an execution worker is still alive.
    """

    root_token = str(execution_root.resolve()).casefold()
    found: list[psutil.Process] = []
    worker_tokens = (
        "run_portfolio_shard.py",
        "finalize_portfolio_shard.py",
    )
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command_line = " ".join(process.info.get("cmdline") or []).casefold()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if (
            root_token in command_line
            and any(token in command_line for token in worker_tokens)
            and process.pid != 0
        ):
            found.append(process)
    return tuple(found)


def _canonical_nonnegative_cny(control: dict, field: str) -> Decimal:
    raw = control.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"execution control {field} must be a CNY string")
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"execution control {field} is not decimal CNY") from error
    if not value.is_finite() or value < 0 or raw != format(value, ".12f"):
        raise ValueError(f"execution control {field} is not canonical non-negative CNY")
    return value


def _validated_cumulative_budget(control: dict) -> tuple[Decimal, Decimal]:
    """Return prior/cap after validating one cumulative phase authority."""

    prior = _canonical_nonnegative_cny(
        control,
        "prior_dashscope_observed_cost_cny",
    )
    phase_cap = _canonical_nonnegative_cny(
        control,
        "phase_cumulative_cap_cny",
    )
    approved = _canonical_nonnegative_cny(
        control,
        "approved_dashscope_budget_cny",
    )
    incremental = _canonical_nonnegative_cny(
        control,
        "incremental_authorized_dashscope_budget_cny",
    )
    if prior >= phase_cap or phase_cap > approved or incremental != phase_cap - prior:
        raise ValueError("execution cumulative budget fields are inconsistent")
    return prior, phase_cap


def _require_empty_initial_parallel_gate(execution_root: Path) -> None:
    """Prove that an optional dry-run-created provider gate has no starts."""

    gate_path = execution_root / "parallel-gate.sqlite3"
    sidecars = tuple(
        execution_root / name
        for name in ("parallel-gate.sqlite3-shm", "parallel-gate.sqlite3-wal")
    )
    if any(path.exists() or path.is_symlink() for path in sidecars):
        raise ValueError("initial parallel gate must not have WAL/SHM sidecars")
    if not gate_path.exists():
        if gate_path.is_symlink():
            raise ValueError("initial parallel gate must not be a symlink")
        return
    before = gate_path.lstat()
    before_snapshot = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    first_bytes = read_stable_regular_file(
        gate_path,
        label="initial parallel gate",
        max_bytes=_PARALLEL_GATE_MAX_BYTES,
    )
    uri = f"{gate_path.resolve().as_uri()}?mode=ro&immutable=1"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("initial parallel gate quick_check failed")
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        object_projection = {
            (str(object_type), str(name), str(table_name))
            for object_type, name, table_name, _sql in objects
        }
        matching_schemas = tuple(
            table_schemas
            for expected_objects, table_schemas in _PARALLEL_GATE_SCHEMA_VARIANTS
            if object_projection == expected_objects
            and len(objects) == len(expected_objects)
        )
        if len(matching_schemas) != 1:
            raise ValueError("initial parallel gate contains unexpected objects")
        for object_type, name, _table_name, sql in objects:
            if object_type == "table" and not isinstance(sql, str):
                raise ValueError("initial parallel gate object definition drifted")
            if object_type == "index":
                expected_sql = (
                    _CALL_LEVEL_START_INDEX_SQL
                    if name == "provider_call_starts_started_at"
                    else None
                )
                if sql != expected_sql:
                    raise ValueError("initial parallel gate object definition drifted")
        for table, expected_schema in matching_schemas[0].items():
            actual_schema = tuple(
                tuple(row)
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if actual_schema != expected_schema:
                raise ValueError(f"initial parallel gate {table} schema drifted")
            if (
                int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                != 0
            ):
                raise ValueError("initial parallel gate contains provider activity")
    except sqlite3.Error as error:
        raise ValueError("initial parallel gate cannot be verified") from error
    finally:
        if connection is not None:
            connection.close()
    if any(path.exists() or path.is_symlink() for path in sidecars):
        raise ValueError("initial parallel gate created WAL/SHM sidecars during read")
    second_bytes = read_stable_regular_file(
        gate_path,
        label="initial parallel gate",
        max_bytes=_PARALLEL_GATE_MAX_BYTES,
    )
    after = gate_path.lstat()
    after_snapshot = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_snapshot != after_snapshot or first_bytes != second_bytes:
        raise ValueError("initial parallel gate changed during verification")


def _missing_ledger_initial_status(
    execution_root: Path,
    control: dict,
    launch: object,
    *,
    allow_active_startup_transition: bool = False,
) -> dict[str, object]:
    """Allow an absent ledger only for the exact create-on-first-execute state.

    A matrix opens the SQLite parallel gate before its first provider call and
    before it creates the budget authority.  While that *same* matrix (or one
    of its workers) is demonstrably alive, WAL/SHM files are therefore mutable
    startup state rather than evidence of an unaccounted provider call.  Do
    not inspect the live database in immutable mode in that narrow window;
    retain all control/launch/artifact checks and re-enter the normal ledger
    path as soon as the authority appears.
    """

    state = launch.state
    if (
        control.get("schema_version") != 4
        or control.get("status") != "prepared_not_started"
        or control.get("execution_scope") not in {"core_canary", "static_opt_rollout"}
        or type(control.get("model_calls_performed")) is not int
        or control["model_calls_performed"] != 0
        or control.get("budget_ledger_relpath") != "budget-ledger"
        or control.get("budget_authority_relpath")
        != "budget-ledger/budget-authority.json"
        or control.get("budget_authority_creation_policy")
        != "create_only_on_first_execute"
        or state.status != "not_started"
        or state.completed_shard_ids
        or state.failed_shard_ids
        or state.model_calls_performed != 0
        or state.dashscope_observed_cost_cny is not None
        or state.aifast_observed_cost_cny is not None
    ):
        raise ValueError("budget ledger is missing outside the strict initial state")
    prior, phase_cap = _validated_cumulative_budget(control)

    allowed_names = {
        "blinding-key.bin",
        "execution-control.json",
        "parallel-gate.sqlite3",
        "parallel-gate.sqlite3-shm",
        "parallel-gate.sqlite3-wal",
    }
    unexpected = sorted(
        path.name for path in execution_root.iterdir() if path.name not in allowed_names
    )
    if unexpected:
        raise ValueError(
            "budget ledger is missing after execution artifacts appeared: "
            + ", ".join(unexpected)
        )
    if _load_control(execution_root) != control:
        raise ValueError("execution control changed after monitor startup")
    if allow_active_startup_transition:
        for name in (
            "parallel-gate.sqlite3",
            "parallel-gate.sqlite3-shm",
            "parallel-gate.sqlite3-wal",
        ):
            path = execution_root / name
            if path.is_symlink():
                raise ValueError("active startup parallel gate must not use symlinks")
            if path.exists() and not path.is_file():
                raise ValueError("active startup parallel gate paths must be files")
    else:
        _require_empty_initial_parallel_gate(execution_root)
    return {
        "ledger_events": 0,
        "ledger_reservations": 0,
        "ledger_settlements": 0,
        "ledger_forfeits": 0,
        "ledger_unresolved": 0,
        "forfeited_reserved_cost_cny": "0.000000000000",
        "accountable_cost_cny": format(prior, ".12f"),
        "phase_cap_cny": format(phase_cap, ".12f"),
        "ledger_startup_transition": allow_active_startup_transition,
    }


def _execution_status(
    execution_root: Path,
    control: dict,
    *,
    allow_active_startup_transition: bool = False,
) -> dict[str, object]:
    launch = load_portfolio_launch_package(
        control["launch_root"],
        expected_plan_file_sha256=control["launch_plan_file_sha256"],
    )
    if control.get("execution_scope") not in {
        "core_canary",
        "static_opt_rollout",
    }:
        raise ValueError(
            "monitor requires execution_scope=core_canary or static_opt_rollout"
        )
    if control.get("execution_scope") == "static_opt_rollout":
        verified_static_runtime = load_verified_portfolio_static_opt_runtime(
            control["runtime_root"],
            expected_runtime_lock_file_sha256=control["runtime_lock_file_sha256"],
        )
        validate_portfolio_static_opt_execution_control(
            control,
            verified_static_runtime,
        )
    if control.get("matrix_run_id") != launch.plan.matrix_run_id:
        raise ValueError("execution control matrix run differs from launch")
    control_prior, control_phase_cap = _validated_cumulative_budget(control)
    ledger_root = execution_root / "budget-ledger"
    if ledger_root.exists() or ledger_root.is_symlink():
        ledger = load_portfolio_budget_ledger(ledger_root)
        if (
            ledger.authority.matrix_run_id != control["matrix_run_id"]
            or ledger.authority.phase_cap_cny != control_phase_cap
            or ledger.authority.prior_observed_cost_cny != control_prior
        ):
            raise ValueError("budget ledger authority differs from execution control")
        ledger_status = {
            "ledger_events": ledger.last_event_index,
            "ledger_reservations": len(ledger.reservations),
            "ledger_settlements": len(ledger.settlements),
            "ledger_forfeits": len(getattr(ledger, "forfeits", ())),
            "ledger_unresolved": len(ledger.unresolved_reservations),
            "forfeited_reserved_cost_cny": format(
                getattr(ledger, "forfeited_reserved_cost_cny", Decimal("0")),
                ".12f",
            ),
            "accountable_cost_cny": format(ledger.accountable_cost_cny, ".12f"),
            "phase_cap_cny": format(ledger.authority.phase_cap_cny, ".12f"),
            "ledger_startup_transition": False,
        }
    else:
        ledger_status = _missing_ledger_initial_status(
            execution_root,
            control,
            launch,
            allow_active_startup_transition=allow_active_startup_transition,
        )
    authorized_shards = _ordered_authorized_shards(control, launch)
    authorized = tuple(item.shard_id for item in authorized_shards)
    completed = set(launch.state.completed_shard_ids)
    if not completed <= set(authorized):
        scope_label = (
            "Core canary"
            if control.get("execution_scope") == "core_canary"
            else "static opt"
        )
        raise ValueError(
            f"launch completed shards fall outside exact {scope_label} authorization"
        )
    authorized_completed = sum(shard_id in completed for shard_id in authorized)
    complete = (
        authorized_completed == len(authorized)
        and int(ledger_status["ledger_unresolved"]) == 0
    )
    return {
        "completed_shards": len(launch.state.completed_shard_ids),
        "authorized_completed_shards": authorized_completed,
        "authorized_shards": len(authorized),
        "launch_shards": control["launch_shard_count"],
        "launch_status": launch.state.status,
        "model_calls_performed": int(ledger_status["ledger_reservations"]),
        "launch_state_model_calls_performed": launch.state.model_calls_performed,
        **ledger_status,
        "complete": complete,
    }


def _progress_key(status: dict[str, object]) -> tuple[int, int]:
    """Return state that proves a restart made accountable progress."""

    return (
        int(status.get("authorized_completed_shards", status["completed_shards"])),
        int(status.get("ledger_events", status["ledger_settlements"])),
    )


def _no_progress_recovery_required(
    *,
    restarts: int,
    last_restart_progress: tuple[int, int] | None,
    status: dict[str, object],
) -> bool:
    return restarts > 0 and _progress_key(status) == last_restart_progress


def _start_matrix(
    *,
    execution_root: Path,
    parallel_profile: Path,
    matrix_log: Path,
) -> int:
    matrix_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = matrix_log.open("a", encoding="utf-8", newline="\n")
    stderr = matrix_log.open("a", encoding="utf-8", newline="\n")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py"),
            "--execution-root",
            str(execution_root),
            "--parallel-profile",
            str(parallel_profile),
            "--execute",
        ],
        cwd=str(REPOSITORY_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        creationflags=flags,
    )
    stdout.close()
    stderr.close()
    return process.pid


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds < 5 or args.max_restarts < 1:
        raise SystemExit("poll-seconds must be >=5 and max-restarts must be positive")
    execution_root = args.execution_root.resolve()
    parallel_profile = args.parallel_profile.resolve()
    try:
        control = _load_control(execution_root)
        validate_runtime_binding(control)
    except (OSError, TypeError, ValueError) as error:
        _log(
            args.status_log,
            json.dumps(
                {
                    "status": "monitor_runtime_binding_mismatch",
                    "error": str(error),
                },
                sort_keys=True,
            ),
        )
        return 2
    restarts = 0
    last_start = 0.0
    last_restart_progress: tuple[int, int] | None = None
    _log(args.status_log, json.dumps({"status": "monitor_started"}, sort_keys=True))
    while True:
        processes = _matrix_processes(execution_root)
        workers = _active_worker_processes(execution_root)
        status = _execution_status(
            execution_root,
            control,
            allow_active_startup_transition=bool(processes or workers),
        )
        status.update(
            {
                "matrix_processes": len(processes),
                "worker_processes": len(workers),
                "restarts": restarts,
            }
        )
        _log(args.status_log, json.dumps(status, sort_keys=True))
        if bool(status["complete"]):
            _log(args.status_log, json.dumps({"status": "monitor_complete"}))
            return 0
        if float(status["accountable_cost_cny"]) >= float(status["phase_cap_cny"]):
            _log(args.status_log, json.dumps({"status": "monitor_budget_cap_reached"}))
            return 0
        if not processes:
            if workers:
                _log(
                    args.status_log,
                    json.dumps(
                        {
                            "status": "matrix_supervision_deferred_active_workers",
                            "worker_processes": len(workers),
                            "completed_shards": status["completed_shards"],
                            "ledger_settlements": status["ledger_settlements"],
                            "restarts": restarts,
                        },
                        sort_keys=True,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            progress_key = _progress_key(status)
            if _no_progress_recovery_required(
                restarts=restarts,
                last_restart_progress=last_restart_progress,
                status=status,
            ):
                _log(
                    args.status_log,
                    json.dumps(
                        {
                            "status": "monitor_no_progress_recovery_required",
                            "completed_shards": status["completed_shards"],
                            "ledger_settlements": status["ledger_settlements"],
                            "restarts": restarts,
                        },
                        sort_keys=True,
                    ),
                )
                return 2
            if restarts >= args.max_restarts:
                _log(
                    args.status_log,
                    json.dumps({"status": "monitor_restart_limit_reached"}),
                )
                return 2
            now = time.monotonic()
            if now - last_start >= args.poll_seconds:
                last_restart_progress = progress_key
                pid = _start_matrix(
                    execution_root=execution_root,
                    parallel_profile=parallel_profile,
                    matrix_log=args.matrix_log,
                )
                restarts += 1
                last_start = now
                _log(
                    args.status_log,
                    json.dumps(
                        {"status": "matrix_restarted", "pid": pid, "restart": restarts},
                        sort_keys=True,
                    ),
                )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
