"""Build compact content locks for the ten required MVP RAW sources.

This script is read-only with respect to RAW.  It writes only the requested
repository output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.source_lock import (  # noqa: E402
    AcquisitionIdentity,
    ArtifactScope,
    RequiredSourceLock,
    build_artifact_scope,
    stable_file_digest,
)
from skillchain.data.source_review import load_source_review_policy  # noqa: E402
from skillchain.tools.serialization import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/required-source-lock-plan-v1.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/mvp-source-review-policy-v2.json",
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/ecommerce-mvp-source-portfolio-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "specs/data_sources/c2/source-locks",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    plan_bytes = arguments.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("source lock plan is invalid")
    raw_root = arguments.raw_root.resolve(strict=True)
    portfolio_sha256 = sha256_bytes(arguments.portfolio.read_bytes())
    policy_bytes = arguments.policy.read_bytes()
    policy = load_source_review_policy(
        arguments.policy,
        expected_policy_file_sha256=sha256_bytes(policy_bytes),
        expected_portfolio_sha256=portfolio_sha256,
    )
    required_sources = sorted(
        item.source_id for item in policy.requirements if item.required
    )
    sources = plan.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source lock plan sources must be an array")
    planned_sources = [item.get("source_id") for item in sources]
    if planned_sources != required_sources:
        raise ValueError(
            "source lock plan must contain exactly the sorted required sources"
        )

    arguments.output.mkdir(parents=True, exist_ok=False)
    plan_sha256 = sha256_bytes(plan_bytes)
    manifest_rows: list[dict[str, Any]] = []
    for source in sources:
        lock = _build_source(raw_root, source, lock_plan_sha256=plan_sha256)
        content = canonical_json_bytes(lock.model_dump(mode="json"))
        relative_path = f"{lock.source_id}.source-lock.json"
        output_path = arguments.output / relative_path
        output_path.write_bytes(content)
        manifest_rows.append(
            {
                "artifact_scope_count": len(lock.artifact_scopes),
                "file_count": sum(item.file_count for item in lock.artifact_scopes),
                "lock_path": relative_path,
                "source_id": lock.source_id,
                "source_lock_sha256": sha256_bytes(content),
                "source_revision": lock.source_revision,
                "total_bytes": sum(item.total_bytes for item in lock.artifact_scopes),
            }
        )
        print(
            json.dumps(
                {
                    "file_count": manifest_rows[-1]["file_count"],
                    "source_id": lock.source_id,
                    "source_lock_sha256": manifest_rows[-1][
                        "source_lock_sha256"
                    ],
                    "total_bytes": manifest_rows[-1]["total_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "lock_plan_path": "../../required-source-lock-plan-v1.json",
        "lock_plan_sha256": plan_sha256,
        "policy_id": policy.policy_id,
        "policy_sha256": sha256_bytes(policy_bytes),
        "portfolio_sha256": portfolio_sha256,
        "raw_mutation_performed": False,
        "schema_version": 1,
        "sources": manifest_rows,
    }
    manifest_content = canonical_json_bytes(manifest)
    manifest_path = arguments.output / "required-source-lock-manifest.json"
    manifest_path.write_bytes(manifest_content)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_bytes(manifest_content),
                "source_count": len(manifest_rows),
                "status": "source-locks-built",
            },
            sort_keys=True,
        )
    )
    return 0


def _build_source(
    raw_root: Path,
    source: dict[str, Any],
    *,
    lock_plan_sha256: str,
) -> RequiredSourceLock:
    digest_cache: dict[str, tuple[str, int]] = {}
    scopes: list[ArtifactScope] = []
    for scope in source["scopes"]:
        mode = scope["mode"]
        if mode == "recursive_tree":
            built = build_artifact_scope(
                raw_root,
                scope_id=scope["scope_id"],
                mode=mode,
                root=scope["root"],
                exclude_prefixes=tuple(scope.get("exclude_prefixes", [])),
            )
        elif mode == "explicit_files":
            paths = tuple(scope["paths"])
            digest = hashlib.sha256()
            total_bytes = 0
            for logical_path in paths:
                file_sha256, file_bytes = stable_file_digest(
                    raw_root / Path(*logical_path.split("/")),
                    label=f"{source['source_id']}:{logical_path}",
                )
                digest_cache[logical_path] = (file_sha256, file_bytes)
                digest.update(
                    canonical_json_bytes(
                        {
                            "bytes": file_bytes,
                            "path": logical_path,
                            "sha256": file_sha256,
                        }
                    )
                )
                total_bytes += file_bytes
            built = ArtifactScope(
                scope_id=scope["scope_id"],
                mode=mode,
                paths=paths,
                file_count=len(paths),
                total_bytes=total_bytes,
                manifest_sha256=digest.hexdigest(),
            )
        else:
            raise ValueError(f"unsupported artifact scope mode: {mode}")
        scopes.append(built)

    identities: list[AcquisitionIdentity] = []
    for identity in source["acquisition_identities"]:
        logical_path = identity["logical_path"]
        local_sha256, local_bytes = digest_cache[logical_path]
        if identity["bytes"] != local_bytes:
            raise ValueError(
                f"{source['source_id']}:{logical_path}: byte count differs from plan"
            )
        identities.append(
            AcquisitionIdentity(
                **identity,
                local_sha256=local_sha256,
            )
        )
    return RequiredSourceLock(
        source_id=source["source_id"],
        source_revision=source["source_revision"],
        lock_plan_sha256=lock_plan_sha256,
        artifact_scopes=tuple(scopes),
        acquisition_identities=tuple(identities),
    )


if __name__ == "__main__":
    raise SystemExit(main())
