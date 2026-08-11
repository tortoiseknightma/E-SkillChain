"""Create-only execution overlay and aggregate provider gates.

The launch package remains the source of truth for inputs and scoring.  This
module adds an immutable, zero-call execution overlay so multiple shard
workers share one aggregate Qwen/Kimi concurrency budget instead of each
process interpreting a local semaphore as the global limit.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Iterator, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ParallelStage = Literal["assistant", "final_judge"]
ParallelExecutionScope = Literal[
    "partial_shard_repair",
    "core_canary",
    "full_matrix",
    "static_opt_rollout",
]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LEGACY_QWEN_RATE_LIMIT_POLICY = "smooth_start_v1"
QWEN_RATE_LIMIT_POLICY = "smooth_provider_call_start_v2"
QwenRateLimitPolicy = Literal[
    "smooth_start_v1",
    "smooth_provider_call_start_v2",
]
MAX_SAFE_ASSISTANT_CONCURRENCY = 2
_QWEN_RATE_WINDOW_SECONDS = 60.0
_SQLITE_BUSY_TIMEOUT_SECONDS = 30.0
_SQLITE_INITIALIZATION_ATTEMPT_TIMEOUT_SECONDS = 1.0
_SQLITE_INITIALIZATION_RETRY_SECONDS = 0.05


def _file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class PortfolioParallelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = 1
    kind: Literal["portfolio-parallel-execution-profile-v1"] = (
        "portfolio-parallel-execution-profile-v1"
    )
    matrix_run_id: str
    launch_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_scope: ParallelExecutionScope = "full_matrix"
    worker_count: int = Field(ge=1, le=16)
    assistant_concurrency: int = Field(ge=1, le=MAX_SAFE_ASSISTANT_CONCURRENCY)
    final_judge_concurrency: int = Field(ge=0, le=16)
    qwen_requests_per_minute_cap: int = Field(ge=1, le=50)
    qwen_rate_limit_policy: QwenRateLimitPolicy = QWEN_RATE_LIMIT_POLICY
    scheduler_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parallel_module_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scheduling_policy: Literal["balanced_paired_batch_v1"] = "balanced_paired_batch_v1"
    gate_db_relpath: str = "parallel-gate.sqlite3"
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_profile(self) -> "PortfolioParallelProfile":
        if not self.matrix_run_id.strip():
            raise ValueError("matrix_run_id must be non-blank")
        if (
            self.gate_db_relpath.startswith("/")
            or ".." in Path(self.gate_db_relpath).parts
            or Path(self.gate_db_relpath).as_posix() != self.gate_db_relpath
        ):
            raise ValueError("gate_db_relpath must be a safe relative POSIX path")
        if self.execution_scope == "static_opt_rollout":
            if self.final_judge_concurrency != 0:
                raise ValueError(
                    "static_opt_rollout must disable Final Judge concurrency"
                )
        elif self.final_judge_concurrency < 1:
            raise ValueError("Final Judge concurrency must be positive")
        unsigned = self.model_dump(mode="json")
        unsigned.pop("profile_sha256")
        # Preserve the self hash of profiles created before the policy field
        # existed.  Pydantic exposes the current call-level policy as their
        # read-time default without changing the bytes they originally bound.
        if "qwen_rate_limit_policy" not in self.model_fields_set:
            unsigned.pop("qwen_rate_limit_policy")
        expected = sha256_bytes(canonical_json_bytes(unsigned))
        if expected != self.profile_sha256:
            raise ValueError("parallel profile self hash mismatch")
        return self


def build_parallel_profile(
    *,
    matrix_run_id: str,
    launch_plan_sha256: str,
    worker_count: int = 4,
    assistant_concurrency: int = MAX_SAFE_ASSISTANT_CONCURRENCY,
    final_judge_concurrency: int = 4,
    qwen_requests_per_minute_cap: int = 40,
    execution_scope: ParallelExecutionScope = "full_matrix",
) -> PortfolioParallelProfile:
    payload = {
        "schema_version": 1,
        "kind": "portfolio-parallel-execution-profile-v1",
        "matrix_run_id": matrix_run_id,
        "launch_plan_sha256": launch_plan_sha256,
        "execution_scope": execution_scope,
        "worker_count": worker_count,
        "assistant_concurrency": assistant_concurrency,
        "final_judge_concurrency": final_judge_concurrency,
        "qwen_requests_per_minute_cap": qwen_requests_per_minute_cap,
        "qwen_rate_limit_policy": QWEN_RATE_LIMIT_POLICY,
        "scheduler_file_sha256": _file_sha256(
            _REPOSITORY_ROOT / "scripts" / "run_portfolio_matrix.py"
        ),
        "parallel_module_file_sha256": _file_sha256(Path(__file__)),
        "scheduling_policy": "balanced_paired_batch_v1",
        "gate_db_relpath": "parallel-gate.sqlite3",
    }
    return PortfolioParallelProfile.model_validate(
        {
            **payload,
            "profile_sha256": sha256_bytes(canonical_json_bytes(payload)),
        },
        strict=True,
    )


def load_parallel_profile(path: str | Path) -> PortfolioParallelProfile:
    raw = json.loads(Path(path).read_bytes())
    return PortfolioParallelProfile.model_validate(raw, strict=True)


def validate_parallel_profile_sources(
    profile: PortfolioParallelProfile,
    *,
    repository_root: Path | None = None,
) -> None:
    """Reject a profile before a provider call if scheduler sources drifted."""

    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    scheduler_sha = _file_sha256(root / "scripts" / "run_portfolio_matrix.py")
    module_sha = _file_sha256(
        root / "src" / "skillchain" / "evaluation" / "portfolio_parallel.py"
    )
    if scheduler_sha != profile.scheduler_file_sha256:
        raise ValueError("parallel profile scheduler source drifted")
    if module_sha != profile.parallel_module_file_sha256:
        raise ValueError("parallel profile parallel-module source drifted")


def write_parallel_profile(path: str | Path, profile: PortfolioParallelProfile) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_file(
        target,
        canonical_json_bytes(profile.model_dump(mode="json", exclude_unset=True)),
    )


def qwen_minimum_start_interval_seconds(requests_per_minute: int) -> float:
    """Return the even-spacing interval required by an aggregate RPM cap."""

    if requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive")
    return _QWEN_RATE_WINDOW_SECONDS / requests_per_minute


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a liveness probe on Windows: the CRT
        # can map it to a real process termination request.  Use the read-only
        # kernel query API so gate cleanup can never kill a worker.
        try:
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,
                ctypes.c_bool,
                ctypes.c_ulong,
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel32.GetExitCodeProcess.restype = ctypes.c_bool
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_bool
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class PortfolioAggregateProviderGate:
    """SQLite-backed cross-process query and provider-call gates.

    ``acquire()`` controls only the number of in-flight query executions.  A
    query can make more than one provider request, so every actual Qwen HTTP
    start must independently call ``wait_for_qwen_call_start()``.  SQLite
    serializes both short allocation transactions; no provider network call
    holds the database transaction open.  Dead query permits are removed only
    when the owning PID is no longer alive, preventing a crashed worker from
    blocking resume.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        assistant_concurrency: int,
        final_judge_concurrency: int,
        qwen_requests_per_minute_cap: int = 40,
        qwen_rate_limit_policy: QwenRateLimitPolicy = QWEN_RATE_LIMIT_POLICY,
    ) -> None:
        if (
            assistant_concurrency <= 0
            or assistant_concurrency > MAX_SAFE_ASSISTANT_CONCURRENCY
            or final_judge_concurrency < 0
        ):
            raise ValueError("provider concurrency limits are invalid")
        if qwen_requests_per_minute_cap <= 0:
            raise ValueError("qwen_requests_per_minute_cap must be positive")
        if qwen_rate_limit_policy != QWEN_RATE_LIMIT_POLICY:
            raise ValueError(
                f"unsupported Qwen rate-limit policy: {qwen_rate_limit_policy}"
            )
        self.db_path = Path(db_path)
        self.limits = {
            "assistant": assistant_concurrency,
            "final_judge": final_judge_concurrency,
        }
        self.qwen_requests_per_minute_cap = qwen_requests_per_minute_cap
        self.qwen_rate_limit_policy = qwen_rate_limit_policy
        self.qwen_minimum_start_interval_seconds = qwen_minimum_start_interval_seconds(
            qwen_requests_per_minute_cap
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=_SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        connection.execute(
            f"PRAGMA busy_timeout={int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1_000)}"
        )
        return connection

    def _initialize_database(self) -> None:
        deadline = time.monotonic() + _SQLITE_BUSY_TIMEOUT_SECONDS
        while True:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connect()
                connection.execute(
                    "PRAGMA busy_timeout="
                    f"{int(_SQLITE_INITIALIZATION_ATTEMPT_TIMEOUT_SECONDS * 1_000)}"
                )
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
                if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                    configured_mode = connection.execute(
                        "PRAGMA journal_mode=WAL"
                    ).fetchone()
                    if (
                        configured_mode is None
                        or str(configured_mode[0]).lower() != "wal"
                    ):
                        raise RuntimeError(
                            "SQLite provider gate requires WAL journal mode"
                        )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS provider_permits (
                        owner TEXT PRIMARY KEY,
                        stage TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        acquired_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS provider_call_starts (
                        start_id TEXT PRIMARY KEY,
                        label TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        started_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS provider_call_starts_started_at "
                    "ON provider_call_starts(started_at)"
                )
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise
            finally:
                if connection is not None:
                    connection.close()
            time.sleep(min(_SQLITE_INITIALIZATION_RETRY_SECONDS, remaining_seconds))

    def _cleanup_dead(
        self, connection: sqlite3.Connection, *, now: float | None = None
    ) -> None:
        rows = connection.execute("SELECT owner, pid FROM provider_permits").fetchall()
        dead = [owner for owner, pid in rows if not _pid_is_alive(int(pid))]
        for owner in dead:
            connection.execute("DELETE FROM provider_permits WHERE owner = ?", (owner,))
        connection.execute(
            "DELETE FROM provider_call_starts WHERE started_at <= ?",
            ((time.time() if now is None else now) - _QWEN_RATE_WINDOW_SECONDS,),
        )

    def wait_for_qwen_call_start(
        self,
        label: str,
        *,
        poll_seconds: float = 0.1,
    ) -> float:
        """Reserve and return the start time for one real Qwen HTTP call.

        Callers invoke this immediately before each HTTP attempt, including
        retries.  The reservation is durable rather than a held permit because
        both even spacing and the rolling 60-second count concern starts, not
        requests that remain in flight.
        """

        if not label.strip():
            raise ValueError("Qwen provider-call label must be non-blank")
        start_id = f"{os.getpid()}:{uuid.uuid4().hex}:{label}"
        while True:
            wait_seconds = poll_seconds
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                self._cleanup_dead(connection, now=now)
                rate_count, oldest_start, latest_start = connection.execute(
                    "SELECT COUNT(*), MIN(started_at), MAX(started_at) "
                    "FROM provider_call_starts"
                ).fetchone()
                rolling_window_available = (
                    int(rate_count) < self.qwen_requests_per_minute_cap
                )
                smooth_start_available = (
                    latest_start is None
                    or now - float(latest_start)
                    >= self.qwen_minimum_start_interval_seconds
                )
                if rolling_window_available and smooth_start_available:
                    connection.execute(
                        "INSERT INTO provider_call_starts"
                        "(start_id, label, pid, started_at) VALUES (?, ?, ?, ?)",
                        (start_id, label, os.getpid(), now),
                    )
                    connection.execute("COMMIT")
                    return now
                if not smooth_start_available:
                    wait_seconds = max(
                        wait_seconds,
                        self.qwen_minimum_start_interval_seconds
                        - (now - float(latest_start)),
                    )
                if not rolling_window_available and oldest_start is not None:
                    wait_seconds = max(
                        wait_seconds,
                        _QWEN_RATE_WINDOW_SECONDS - (now - float(oldest_start)),
                    )
                connection.execute("ROLLBACK")
            time.sleep(wait_seconds)

    @contextmanager
    def acquire(
        self,
        stage: ParallelStage,
        *,
        label: str,
        poll_seconds: float = 0.1,
    ) -> Iterator[None]:
        if stage not in self.limits:
            raise ValueError(f"unsupported provider gate stage: {stage}")
        if self.limits[stage] == 0:
            raise RuntimeError(f"provider gate stage is disabled: {stage}")
        owner = f"{os.getpid()}:{uuid.uuid4().hex}:{label}"
        acquired = False
        try:
            while not acquired:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    now = time.time()
                    self._cleanup_dead(connection, now=now)
                    count = connection.execute(
                        "SELECT COUNT(*) FROM provider_permits WHERE stage = ?",
                        (stage,),
                    ).fetchone()[0]
                    if int(count) < self.limits[stage]:
                        connection.execute(
                            "INSERT INTO provider_permits(owner, stage, pid, acquired_at) "
                            "VALUES (?, ?, ?, ?)",
                            (owner, stage, os.getpid(), now),
                        )
                        connection.execute("COMMIT")
                        acquired = True
                    else:
                        connection.execute("ROLLBACK")
                if not acquired:
                    time.sleep(poll_seconds)
            yield
        finally:
            if acquired:
                with self._connect() as connection:
                    connection.execute(
                        "DELETE FROM provider_permits WHERE owner = ?", (owner,)
                    )


__all__ = [
    "MAX_SAFE_ASSISTANT_CONCURRENCY",
    "QWEN_RATE_LIMIT_POLICY",
    "PortfolioAggregateProviderGate",
    "PortfolioParallelProfile",
    "build_parallel_profile",
    "load_parallel_profile",
    "qwen_minimum_start_interval_seconds",
    "validate_parallel_profile_sources",
    "write_parallel_profile",
]
