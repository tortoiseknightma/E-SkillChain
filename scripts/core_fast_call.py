#!/usr/bin/env python3
"""JSON call bridge for a project-specific real Core Fast provider.

The orchestration path is provider-neutral.  Set ``CORE_FAST_CALL_FACTORY`` to
``module:function``; the factory is called with no arguments and must return an
object implementing ``invoke(CallIntent) -> CallResult``.  Keeping the bridge
small prevents provider SDK retry/ledger behavior from leaking into the Fast
Path.  For deterministic local validation, set the factory to
``skillchain.evaluation.core_fast.fake_provider:create_adapter_for_bridge``.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillchain.evaluation.core_fast.models import CallIntent, CallResult  # noqa: E402


def main() -> int:
    raw = json.loads(sys.stdin.buffer.read())
    intent = CallIntent.model_validate(raw, strict=True)
    target = os.environ.get("CORE_FAST_CALL_FACTORY")
    if not target:
        result = CallResult(
            call_id=intent.call_id,
            role=intent.role,
            status="provider_error",
            requested_model=intent.requested_model,
            failure_reason=(
                "CORE_FAST_CALL_FACTORY is unset; configure the real pure-execution "
                "adapter or select runtime.adapter=python for an in-process adapter"
            ),
        )
    else:
        module_name, separator, attribute = target.partition(":")
        if not separator:
            raise ValueError("CORE_FAST_CALL_FACTORY must be module:function")
        factory = getattr(importlib.import_module(module_name), attribute)
        adapter = factory()
        result = adapter.invoke(intent)
    sys.stdout.buffer.write(result.model_dump_json().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
