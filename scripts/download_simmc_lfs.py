#!/usr/bin/env python3
"""Resume SIMMC 2.1's public Git LFS objects with size/hash verification.

Git LFS can leave a checkout blocked behind an indefinite transfer.  This helper
operates on a detached, already-fetched official repository revision, requests
fresh short-lived download actions from the public LFS batch endpoint, and keeps
resumable bytes in ``.git/lfs/incomplete``.  It never checks out the files;
run ``git lfs checkout`` only after every object has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LFS_BATCH_ACCEPT = "application/vnd.git-lfs+json"
LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"
OID_RE = re.compile(r"^([0-9a-f]{10,64})\s+[-*]\s+(.+)$")
SIZE_RE = re.compile(r"^size\s+(\d+)$", re.MULTILINE)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def object_inventory(repo: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for line in run_git(repo, "lfs", "ls-files", "--all").splitlines():
        match = OID_RE.match(line)
        if not match:
            continue
        prefix, path = match.groups()
        pointer = run_git(repo, "show", f"HEAD:{path}")
        oid = next(
            (item.split(":", 1)[1] for item in pointer.splitlines() if item.startswith("oid sha256:")),
            None,
        )
        size_match = SIZE_RE.search(pointer)
        if not oid or not size_match or not oid.startswith(prefix):
            raise RuntimeError(f"invalid LFS pointer: {path}")
        inventory.append({"oid": oid, "size": int(size_match.group(1)), "path": path})
    if not inventory:
        raise RuntimeError("no Git LFS pointers found at HEAD")
    return inventory


def verified_file(path: Path, oid: str, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == oid


def lfs_object_path(repo: Path, oid: str) -> Path:
    return repo / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid


def partial_path(repo: Path, oid: str, expected_size: int) -> Path:
    incomplete = repo / ".git" / "lfs" / "incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    target = incomplete / f"{oid}.partial"
    if target.exists():
        return target
    candidates = [
        item
        for item in incomplete.glob(f"{oid}*")
        if item.is_file() and item.stat().st_size <= expected_size
    ]
    if candidates:
        candidate = max(candidates, key=lambda item: item.stat().st_size)
        os.replace(candidate, target)
    return target


def batch_actions(repo: Path, objects: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    remote = run_git(repo, "config", "--get", "remote.origin.url").strip()
    if not remote.startswith("https://") or not remote.endswith(".git"):
        raise RuntimeError(f"unsupported public HTTPS remote: {remote}")
    endpoint = f"{remote}/info/lfs/objects/batch"
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": item["oid"], "size": item["size"]} for item in objects],
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        method="POST",
        headers={"Accept": LFS_BATCH_ACCEPT, "Content-Type": LFS_CONTENT_TYPE},
    )
    with urlopen(request, timeout=60) as response:
        body = json.load(response)
    result: dict[str, dict[str, object]] = {}
    for item in body.get("objects", []):
        if "error" in item:
            raise RuntimeError(f"LFS server rejected {item.get('oid')}: {item['error']}")
        result[item["oid"]] = item["actions"]["download"]
    return result


def download_one(repo: Path, item: dict[str, object], action: dict[str, object]) -> None:
    oid, expected_size, source_path = item["oid"], item["size"], item["path"]
    assert isinstance(oid, str) and isinstance(expected_size, int) and isinstance(source_path, str)
    destination = lfs_object_path(repo, oid)
    if verified_file(destination, oid, expected_size):
        print(f"[OK] {source_path}: existing verified object")
        return

    partial = partial_path(repo, oid, expected_size)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0

    headers = {str(key): str(value) for key, value in dict(action.get("header", {})).items()}
    headers["Accept-Encoding"] = "identity"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(str(action["href"]), headers=headers)
    try:
        with urlopen(request, timeout=90) as response:
            status = response.status
            content_range = response.headers.get("Content-Range", "")
            if offset and (status != 206 or not content_range.startswith(f"bytes {offset}-")):
                offset = 0
            if not offset and status not in (200, 206):
                raise RuntimeError(f"unexpected HTTP {status}")
            mode = "ab" if offset and status == 206 else "wb"
            with partial.open(mode) as target:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"download interrupted for {source_path}: {error}") from error

    if partial.stat().st_size != expected_size or not verified_file(partial, oid, expected_size):
        raise RuntimeError(f"verification failed for {source_path}; preserved {partial.name} for resume")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    print(f"[OK] {source_path}: {expected_size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--retry-wait", type=float, default=10.0)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means retry until complete")
    args = parser.parse_args()
    repo = args.repo.resolve()
    inventory = object_inventory(repo)
    rounds = 0
    while True:
        missing = [item for item in inventory if not verified_file(lfs_object_path(repo, str(item["oid"])), str(item["oid"]), int(item["size"]))]
        if not missing:
            print(f"[OK] verified {len(inventory)} LFS objects")
            return 0
        rounds += 1
        if args.max_rounds and rounds > args.max_rounds:
            print(f"[PENDING] {len(missing)} LFS objects remain", file=sys.stderr)
            return 2
        try:
            actions = batch_actions(repo, missing)
            for item in missing:
                download_one(repo, item, actions[str(item["oid"])])
        except (RuntimeError, HTTPError, URLError, TimeoutError) as error:
            print(f"[RETRY] {error}", file=sys.stderr)
            time.sleep(args.retry_wait)


if __name__ == "__main__":
    raise SystemExit(main())
