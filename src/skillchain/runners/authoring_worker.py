"""Minimal worker executed inside the pinned LLMStatic authoring image.

The container command deliberately exposes no configurable input paths.  A
formal parent runner mounts exactly one canonical request at ``/input`` and a
fresh output directory at ``/output``.
"""

from __future__ import annotations

from pathlib import Path

from skillchain.static_authoring import UnifiedChatTransport
from skillchain.synthesis.store import atomic_create_file
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
)


def main() -> int:
    request = read_stable_regular_file(
        Path("/input/authoring-request.json"),
        label="isolated authoring request",
    )
    response = UnifiedChatTransport().complete(request)
    atomic_create_file(
        Path("/output/authoring-response.json"),
        canonical_json_bytes(response.model_dump(mode="json")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
