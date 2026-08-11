"""Resume and verify the owner-approved Hugging Face Polyvore replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from huggingface_hub import HfApi, snapshot_download


DATASET_ID = "Marqo/polyvore"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _parquet_has_magic(path: Path) -> bool:
    if path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        leading = handle.read(4)
        handle.seek(-4, os.SEEK_END)
        trailing = handle.read(4)
    return leading == b"PAR1" and trailing == b"PAR1"


def _get_inventory(api: HfApi) -> tuple[str, dict[str, int]]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            info = api.dataset_info(DATASET_ID)
            files = [
                entry
                for entry in api.list_repo_tree(
                    DATASET_ID,
                    repo_type="dataset",
                    revision=info.sha,
                    recursive=True,
                    expand=True,
                )
                if entry.path.startswith("data/") and entry.path.endswith(".parquet")
            ]
            expected = {entry.path: entry.size for entry in files}
            if len(expected) != 6 or any(size is None for size in expected.values()):
                raise RuntimeError("unexpected Polyvore replacement inventory")
            return info.sha, {path: int(size) for path, size in expected.items()}
        except Exception as error:  # Network proxies can reset an initial TLS handshake.
            last_error = error
            if attempt == 3:
                raise
            time.sleep(attempt * 10)
    raise RuntimeError("unreachable") from last_error


def _download_snapshot(
    *, revision: str, staging_root: Path, cache_root: Path, max_workers: int
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                DATASET_ID,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["README.md", "data/*.parquet"],
                local_dir=staging_root,
                cache_dir=cache_root,
                max_workers=max_workers,
            )
            return
        except Exception as error:  # Resume partial cache entries after a proxy TLS reset.
            last_error = error
            if attempt == 3:
                raise
            time.sleep(attempt * 15)
    raise RuntimeError("unreachable") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    final_root = root / "snapshot"
    marker = final_root / "manifest.json"
    if marker.is_file():
        print(f"already_verified={final_root}")
        return

    staging_root = root / ".staging"
    cache_root = root / ".download-state" / "hf-cache"
    staging_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision, expected = _get_inventory(api)

    _download_snapshot(
        revision=revision,
        staging_root=staging_root,
        cache_root=cache_root,
        max_workers=args.max_workers,
    )

    verified: dict[str, int] = {}
    for relative_path, expected_size in sorted(expected.items()):
        path = staging_root / relative_path
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"size verification failed: {relative_path}")
        if not _parquet_has_magic(path):
            raise RuntimeError(f"Parquet signature verification failed: {relative_path}")
        verified[relative_path] = expected_size

    _atomic_json(
        staging_root / "manifest.json",
        {
            "dataset_id": DATASET_ID,
            "revision": revision,
            "files": verified,
            "total_bytes": sum(verified.values()),
        },
    )
    if final_root.exists():
        raise RuntimeError(f"final destination already exists without a verified marker: {final_root}")
    os.replace(staging_root, final_root)
    print(f"verified_files={len(verified)} total_bytes={sum(verified.values())}")


if __name__ == "__main__":
    main()
