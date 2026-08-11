"""Finalize repeated FashionIQ failures as an explicit local exclusion ledger."""

from pathlib import Path
import json
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.fashioniq import finalize_repeated_failures  # noqa: E402


if __name__ == "__main__":
    print(
        json.dumps(
            finalize_repeated_failures(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
