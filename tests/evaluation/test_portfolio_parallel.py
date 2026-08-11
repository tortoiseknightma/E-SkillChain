from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sqlite3
import threading
import time

import pytest
from pydantic import ValidationError

from skillchain.evaluation.portfolio_parallel import (
    MAX_SAFE_ASSISTANT_CONCURRENCY,
    PortfolioAggregateProviderGate,
    build_parallel_profile,
    load_parallel_profile,
    qwen_minimum_start_interval_seconds,
    write_parallel_profile,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


def _record_qwen_start_in_subprocess(db_path: str) -> None:
    gate = PortfolioAggregateProviderGate(
        db_path,
        assistant_concurrency=1,
        final_judge_concurrency=1,
        qwen_requests_per_minute_cap=600,
    )
    gate.wait_for_qwen_call_start(
        label=f"subprocess-{multiprocessing.current_process().pid}"
    )


def test_parallel_profile_is_create_only_and_self_hashed(tmp_path: Path) -> None:
    profile = build_parallel_profile(
        matrix_run_id="portfolio-dev-mini-200x5-real-treatment-v14",
        launch_plan_sha256="a" * 64,
    )
    path = tmp_path / "parallel-profile.json"
    write_parallel_profile(path, profile)
    assert load_parallel_profile(path) == profile
    with pytest.raises(FileExistsError):
        write_parallel_profile(path, profile)

    payload = json.loads(path.read_bytes())
    payload["final_judge_concurrency"] = 8
    with pytest.raises(ValidationError, match="self hash"):
        type(profile).model_validate(payload, strict=True)

    assert profile.qwen_rate_limit_policy == "smooth_provider_call_start_v2"
    assert profile.assistant_concurrency == MAX_SAFE_ASSISTANT_CONCURRENCY == 2
    assert qwen_minimum_start_interval_seconds(
        profile.qwen_requests_per_minute_cap
    ) == pytest.approx(1.5)


def test_parallel_profile_loads_pre_pacing_profile_with_safe_default(
    tmp_path: Path,
) -> None:
    profile = build_parallel_profile(
        matrix_run_id="portfolio-dev-mini-200x5-legacy",
        launch_plan_sha256="b" * 64,
    )
    payload = profile.model_dump(mode="json")
    payload.pop("qwen_rate_limit_policy")
    payload.pop("profile_sha256")
    payload["profile_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    path = tmp_path / "legacy-parallel-profile.json"
    path.write_bytes(canonical_json_bytes(payload))

    loaded = load_parallel_profile(path)
    assert loaded.qwen_rate_limit_policy == "smooth_provider_call_start_v2"


def test_parallel_profile_retains_explicit_legacy_policy_for_reading(
    tmp_path: Path,
) -> None:
    profile = build_parallel_profile(
        matrix_run_id="portfolio-dev-mini-200x5-legacy-explicit",
        launch_plan_sha256="f" * 64,
    )
    payload = profile.model_dump(mode="json")
    payload["qwen_rate_limit_policy"] = "smooth_start_v1"
    payload.pop("profile_sha256")
    payload["profile_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    path = tmp_path / "explicit-legacy-parallel-profile.json"
    path.write_bytes(canonical_json_bytes(payload))

    loaded = load_parallel_profile(path)
    assert loaded.qwen_rate_limit_policy == "smooth_start_v1"

    with pytest.raises(ValueError, match="unsupported Qwen rate-limit policy"):
        PortfolioAggregateProviderGate(
            tmp_path / "legacy-policy-gate.sqlite3",
            assistant_concurrency=1,
            final_judge_concurrency=1,
            qwen_rate_limit_policy=loaded.qwen_rate_limit_policy,
        )


def test_parallel_profile_binds_core_canary_scope() -> None:
    profile = build_parallel_profile(
        matrix_run_id="portfolio-core-gate0-canary",
        launch_plan_sha256="c" * 64,
        execution_scope="core_canary",
        assistant_concurrency=2,
        final_judge_concurrency=8,
    )

    assert profile.execution_scope == "core_canary"
    assert profile.assistant_concurrency == 2
    assert profile.final_judge_concurrency == 8


def test_static_opt_parallel_profile_disables_final_judge_gate(tmp_path: Path) -> None:
    profile = build_parallel_profile(
        matrix_run_id="portfolio-core-static-opt",
        launch_plan_sha256="d" * 64,
        execution_scope="static_opt_rollout",
        assistant_concurrency=MAX_SAFE_ASSISTANT_CONCURRENCY,
        final_judge_concurrency=0,
    )
    assert profile.final_judge_concurrency == 0

    with pytest.raises(ValidationError, match="must disable Final Judge"):
        build_parallel_profile(
            matrix_run_id="portfolio-core-static-opt",
            launch_plan_sha256="d" * 64,
            execution_scope="static_opt_rollout",
            final_judge_concurrency=1,
        )

    gate = PortfolioAggregateProviderGate(
        tmp_path / "static-gate.sqlite3",
        assistant_concurrency=2,
        final_judge_concurrency=0,
    )
    with pytest.raises(RuntimeError, match="stage is disabled"):
        with gate.acquire("final_judge", label="must-not-run"):
            pass


def test_parallel_profile_and_gate_reject_assistant_concurrency_above_safe_cap(
    tmp_path: Path,
) -> None:
    assert MAX_SAFE_ASSISTANT_CONCURRENCY == 2
    with pytest.raises(ValidationError, match="less than or equal to 2"):
        build_parallel_profile(
            matrix_run_id="portfolio-core-static-opt",
            launch_plan_sha256="e" * 64,
            execution_scope="static_opt_rollout",
            assistant_concurrency=3,
            final_judge_concurrency=0,
        )

    with pytest.raises(ValueError, match="provider concurrency limits are invalid"):
        PortfolioAggregateProviderGate(
            tmp_path / "unsafe-gate.sqlite3",
            assistant_concurrency=3,
            final_judge_concurrency=0,
        )


def test_aggregate_gate_limits_threads_to_stage_cap(tmp_path: Path) -> None:
    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=2,
        final_judge_concurrency=1,
        qwen_requests_per_minute_cap=60_000,
    )
    active = 0
    maximum = 0
    lock = threading.Lock()

    def work(index: int) -> None:
        nonlocal active, maximum
        with gate.acquire("assistant", label=f"test-{index}"):
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(work, range(5)))
    assert maximum == 2


def test_aggregate_gate_retries_locked_wal_initialization_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_connect = sqlite3.connect
    wal_attempts: list[str] = []
    sleeps: list[float] = []

    class LockOnceConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(
            self, sql: str, parameters: tuple[object, ...] = ()
        ) -> sqlite3.Cursor:
            if sql.strip().upper() == "PRAGMA JOURNAL_MODE=WAL":
                wal_attempts.append(sql)
                if len(wal_attempts) == 1:
                    raise sqlite3.OperationalError("database is locked")
            return self.connection.execute(sql, parameters)

        def close(self) -> None:
            self.connection.close()

    def flaky_connect(*args: object, **kwargs: object) -> LockOnceConnection:
        return LockOnceConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", flaky_connect)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=1,
        final_judge_concurrency=1,
    )

    assert len(wal_attempts) == 2
    assert sleeps == pytest.approx([0.05])

    connection = gate._connect()
    assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    connection.close()
    assert len(wal_attempts) == 2


def test_outer_assistant_acquire_does_not_consume_provider_call_starts(
    tmp_path: Path,
) -> None:
    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=MAX_SAFE_ASSISTANT_CONCURRENCY,
        final_judge_concurrency=1,
        qwen_requests_per_minute_cap=1,
    )

    with gate.acquire("assistant", label="first-query"):
        pass
    with gate.acquire("assistant", label="second-query"):
        pass

    with gate._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM provider_call_starts"
        ).fetchone()[0]
    assert count == 0


def test_aggregate_gate_evenly_spaces_true_qwen_call_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000.0]
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_parallel.time.time", lambda: now[0]
    )
    monkeypatch.setattr("skillchain.evaluation.portfolio_parallel.time.sleep", advance)
    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=MAX_SAFE_ASSISTANT_CONCURRENCY,
        final_judge_concurrency=1,
        qwen_requests_per_minute_cap=40,
    )

    first = gate.wait_for_qwen_call_start("first-http-call")
    second = gate.wait_for_qwen_call_start(label="second-http-call")

    with gate._connect() as connection:
        starts = [
            float(row[0])
            for row in connection.execute(
                "SELECT started_at FROM provider_call_starts ORDER BY started_at"
            ).fetchall()
        ]
    assert first == pytest.approx(1_000.0)
    assert second == pytest.approx(1_001.5)
    assert starts == pytest.approx([1_000.0, 1_001.5])
    assert sum(sleeps) == pytest.approx(1.5)


def test_qwen_call_start_gate_enforces_rolling_sixty_second_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000.0]
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(
        "skillchain.evaluation.portfolio_parallel.time.time", lambda: now[0]
    )
    monkeypatch.setattr("skillchain.evaluation.portfolio_parallel.time.sleep", advance)
    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=1,
        final_judge_concurrency=1,
        qwen_requests_per_minute_cap=2,
    )
    with gate._connect() as connection:
        connection.executemany(
            "INSERT INTO provider_call_starts"
            "(start_id, label, pid, started_at) VALUES (?, ?, ?, ?)",
            [
                ("oldest", "seed-oldest", 1, 941.0),
                ("latest", "seed-latest", 1, 970.0),
            ],
        )

    started_at = gate.wait_for_qwen_call_start(label="rolling-window-http-call")

    assert started_at == pytest.approx(1_001.0)
    assert sleeps == pytest.approx([1.0])
    with gate._connect() as connection:
        starts = connection.execute(
            "SELECT started_at FROM provider_call_starts ORDER BY started_at"
        ).fetchall()
    assert starts == pytest.approx([(970.0,), (1_001.0,)])


def test_qwen_call_start_gate_serializes_concurrent_processes_and_initialization(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cross-process-gate.sqlite3"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_record_qwen_start_in_subprocess,
            args=(str(db_path),),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    with sqlite3.connect(db_path) as connection:
        starts = [
            float(row[0])
            for row in connection.execute(
                "SELECT started_at FROM provider_call_starts ORDER BY started_at"
            ).fetchall()
        ]
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 0.09


def test_gate_rejects_unknown_stage(tmp_path: Path) -> None:
    gate = PortfolioAggregateProviderGate(
        tmp_path / "gate.sqlite3",
        assistant_concurrency=1,
        final_judge_concurrency=1,
    )
    with pytest.raises(ValueError, match="unsupported provider gate stage"):
        with gate.acquire("other", label="bad"):  # type: ignore[arg-type]
            pass
