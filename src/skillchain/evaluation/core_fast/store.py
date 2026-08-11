from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from pydantic import ValidationError

from skillchain.tools.serialization import canonical_json_bytes

from .models import CallIntent, CallResult, CallRole, CoreFastSpec


class FastStoreError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def atomic_write_jsonl(
    path: Path, rows: Iterable[object], *, overwrite: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    content = b"".join(canonical_json_bytes(row) for row in rows)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class CallStore:
    """Two-file call journal with conservative crash recovery.

    A completed result is reused.  An intent without a result is materialized
    as ``interrupted_unknown`` and is never silently replayed.
    """

    def __init__(self, root: Path, spec: CoreFastSpec) -> None:
        self.root = root
        self.spec = spec
        self._lock = threading.Lock()
        self._observed_cost_cache: float | None = None
        self._pending_cost_estimate = 0.0

    def _paths(self, role: CallRole, call_id: str) -> tuple[Path, Path]:
        if not call_id or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for char in call_id
        ):
            raise ValueError(f"unsafe call ID: {call_id!r}")
        directory = self.root / "calls" / role
        return (
            directory / f"{call_id}.intent.json",
            directory / f"{call_id}.result.json",
        )

    def _load_observed_cost_unlocked(self) -> float:
        if self._observed_cost_cache is not None:
            return self._observed_cost_cache
        total = 0.0
        calls = self.root / "calls"
        if calls.exists():
            for path in calls.glob("*/*.result.json"):
                try:
                    value = CallResult.model_validate(load_json(path), strict=True)
                except (OSError, ValueError, ValidationError) as error:
                    raise FastStoreError(f"invalid call result: {path}") from error
                total += value.cost_cny
        self._observed_cost_cache = total
        return total

    def observed_cost(self) -> float:
        with self._lock:
            return self._load_observed_cost_unlocked()

    def role_count(self, role: CallRole) -> int:
        directory = self.root / "calls" / role
        return (
            0
            if not directory.exists()
            else sum(1 for _ in directory.glob("*.intent.json"))
        )

    def _interrupted(self, intent: CallIntent, result_path: Path) -> CallResult:
        result = CallResult(
            call_id=intent.call_id,
            role=intent.role,
            status="interrupted_unknown",
            requested_model=intent.requested_model,
            schema_valid=False,
            failure_reason="intent existed without a completed result; automatic replay is disabled",
        )
        atomic_write_json(result_path, result.model_dump(mode="json"))
        return result

    def get(self, role: CallRole, call_id: str) -> CallResult | None:
        intent_path, result_path = self._paths(role, call_id)
        if result_path.exists():
            try:
                return CallResult.model_validate(load_json(result_path), strict=True)
            except (OSError, ValueError, ValidationError) as error:
                raise FastStoreError(
                    f"invalid completed result: {result_path}"
                ) from error
        if intent_path.exists():
            try:
                intent = CallIntent.model_validate(load_json(intent_path), strict=True)
            except (OSError, ValueError, ValidationError) as error:
                raise FastStoreError(f"invalid call intent: {intent_path}") from error
            return self._interrupted(intent, result_path)
        return None

    def ensure_budget(self, role: CallRole, count: int = 1) -> None:
        estimate = self.spec.models[role].estimated_call_cost_cny * count
        with self._lock:
            projected = (
                self._load_observed_cost_unlocked()
                + self._pending_cost_estimate
                + estimate
            )
            if projected > self.spec.limits.external_cost_cny:
                raise FastStoreError(
                    "next call estimate would exceed the CNY "
                    f"{self.spec.limits.external_cost_cny:g} hard cap"
                )

    def invoke(
        self,
        intent: CallIntent,
        invoke_fn: Callable[[CallIntent], CallResult],
    ) -> CallResult:
        intent_path, result_path = self._paths(intent.role, intent.call_id)
        estimate = self.spec.models[intent.role].estimated_call_cost_cny
        with self._lock:
            existing = self.get(intent.role, intent.call_id)
            if existing is not None:
                return existing
            projected = (
                self._load_observed_cost_unlocked()
                + self._pending_cost_estimate
                + estimate
            )
            if projected > self.spec.limits.external_cost_cny:
                raise FastStoreError(
                    "next call estimate would exceed the CNY "
                    f"{self.spec.limits.external_cost_cny:g} hard cap"
                )
            # The intent is the only durable pre-call record.  The in-memory
            # reservation merely prevents concurrent workers overshooting the
            # cap; it is deliberately not another governance artifact.
            atomic_write_json(intent_path, intent.model_dump(mode="json"))
            self._pending_cost_estimate += estimate
        try:
            started = time.perf_counter()
            try:
                result = invoke_fn(intent)
            except Exception as error:
                result = CallResult(
                    call_id=intent.call_id,
                    role=intent.role,
                    status="provider_error",
                    requested_model=intent.requested_model,
                    schema_valid=False,
                    latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
                    failure_reason=f"{type(error).__name__}: {error}",
                )
            if result.call_id != intent.call_id or result.role != intent.role:
                raise FastStoreError("adapter result identity differs from call intent")
            if result.requested_model != intent.requested_model:
                raise FastStoreError(
                    "adapter result requested_model differs from intent"
                )
            atomic_write_json(result_path, result.model_dump(mode="json"))
            with self._lock:
                self._observed_cost_cache = (
                    self._load_observed_cost_unlocked() + result.cost_cny
                )
            return result
        finally:
            with self._lock:
                self._pending_cost_estimate -= estimate


__all__ = [
    "CallStore",
    "FastStoreError",
    "atomic_write_json",
    "atomic_write_jsonl",
    "load_json",
]
