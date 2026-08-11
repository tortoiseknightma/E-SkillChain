from __future__ import annotations

import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from skillchain.tools.serialization import canonical_json_bytes

from .models import CallIntent, CallResult, CoreFastSpec


@runtime_checkable
class CoreFastAdapter(Protocol):
    def invoke(self, intent: CallIntent) -> CallResult:
        """Perform exactly one provider/execution attempt."""


class CommandAdapter:
    """Thin JSON-over-stdin adapter for the existing pure execution commands.

    Commands receive one canonical ``CallIntent`` on stdin and must emit one
    ``CallResult`` JSON object on stdout.  The wrapper itself never retries.
    """

    def __init__(self, spec: CoreFastSpec, *, cwd: Path) -> None:
        self.spec = spec
        self.cwd = cwd

    def invoke(self, intent: CallIntent) -> CallResult:
        command = self.spec.runtime.commands.for_role(intent.role)
        if not command:
            raise RuntimeError(f"no command configured for role {intent.role}")
        started = time.perf_counter()
        process = subprocess.run(
            list(command),
            input=canonical_json_bytes(intent.model_dump(mode="json")),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            timeout=self.spec.runtime.command_timeout_seconds,
            check=False,
        )
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="replace")[-2000:]
            return CallResult(
                call_id=intent.call_id,
                role=intent.role,
                status="provider_error",
                requested_model=intent.requested_model,
                latency_ms=latency_ms,
                failure_reason=f"command exited {process.returncode}: {stderr}",
            )
        try:
            raw = json.loads(process.stdout.decode("utf-8", errors="strict"))
            result = CallResult.model_validate(raw, strict=True)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            return CallResult(
                call_id=intent.call_id,
                role=intent.role,
                status="schema_error",
                requested_model=intent.requested_model,
                raw_output=process.stdout.decode("utf-8", errors="replace"),
                latency_ms=latency_ms,
                failure_reason=f"invalid command result: {error}",
            )
        return result.model_copy(update={"latency_ms": result.latency_ms or latency_ms})


def load_adapter(spec: CoreFastSpec, *, cwd: Path, spec_path: Path) -> CoreFastAdapter:
    if spec.runtime.adapter == "command":
        return CommandAdapter(spec, cwd=cwd)
    assert spec.runtime.python_factory is not None
    module_name, separator, attribute = spec.runtime.python_factory.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("python_factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory(spec=spec, cwd=cwd, base_dir=spec_path.resolve().parent)
    if not isinstance(adapter, CoreFastAdapter):
        raise TypeError("python adapter factory did not return CoreFastAdapter")
    return adapter


__all__ = ["CommandAdapter", "CoreFastAdapter", "load_adapter"]
