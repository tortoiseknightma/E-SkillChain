"""Download the approved Hugging Face Polyvore replacement with curl resume."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi

from download_polyvore_hf import DATASET_ID, _atomic_json, _get_inventory, _parquet_has_magic


def _download_file(*, revision: str, relative_path: str, expected_size: int, root: Path) -> None:
    destination = root / relative_path
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == expected_size and _parquet_has_magic(destination):
            print(f"already_verified={relative_path}", flush=True)
            return
        raise RuntimeError(f"invalid completed shard: {relative_path}")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"partial exceeds expected size: {relative_path}")

    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{relative_path}"
    for attempt in range(1, 9):
        command = [
            "curl.exe",
            "--fail",
            "--location",
            "--connect-timeout",
            "20",
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "--output",
            str(partial),
        ]
        if partial.exists() and partial.stat().st_size:
            command.extend(["--continue-at", "-"])
        command.append(url)
        print(f"downloading={relative_path} attempt={attempt}", flush=True)
        result = subprocess.run(command, check=False)
        if (
            result.returncode == 0
            and partial.is_file()
            and partial.stat().st_size == expected_size
            and _parquet_has_magic(partial)
        ):
            break
        if attempt == 8:
            raise RuntimeError(f"curl resume attempts exhausted: {relative_path}")
        time.sleep(10)

    if partial.stat().st_size != expected_size or not _parquet_has_magic(partial):
        raise RuntimeError(f"verification failed: {relative_path}")
    os.replace(partial, destination)
    print(f"verified={relative_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    final_root = root / "snapshot"
    marker = final_root / "manifest.json"
    if marker.is_file():
        print(f"already_verified={final_root}")
        return

    staging_root = root / ".curl-staging"
    revision, expected = _get_inventory(HfApi())
    for relative_path, expected_size in sorted(expected.items()):
        _download_file(
            revision=revision,
            relative_path=relative_path,
            expected_size=expected_size,
            root=staging_root,
        )

    _atomic_json(
        staging_root / "manifest.json",
        {
            "dataset_id": DATASET_ID,
            "revision": revision,
            "files": expected,
            "total_bytes": sum(expected.values()),
            "transport": "curl-resume",
        },
    )
    if final_root.exists():
        raise RuntimeError(f"final destination already exists without a verified marker: {final_root}")
    os.replace(staging_root, final_root)
    print(f"verified_files={len(expected)} total_bytes={sum(expected.values())}")


if __name__ == "__main__":
    main()
