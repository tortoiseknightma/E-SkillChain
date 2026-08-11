"""Run and finalize every authorized Portfolio shard in launch order.

The command is resumable: completed shards are skipped, create-only query
checkpoints are reused by the shard runner, and any recoverable provider or
budget stop is returned immediately without inventing a score.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.run_portfolio_shard import _load_control  # noqa: E402
from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
)
from skillchain.evaluation.portfolio_parallel import (  # noqa: E402
    PortfolioParallelProfile,
    load_parallel_profile,
    validate_parallel_profile_sources,
)
from skillchain.evaluation.portfolio_static_opt_runtime import (  # noqa: E402
    load_verified_portfolio_static_opt_runtime,
    validate_portfolio_static_opt_execution_control,
)
from skillchain.tools.serialization import sha256_bytes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform real model calls; default validates every shard as a dry run.",
    )
    parser.add_argument(
        "--stop-after-shards",
        type=int,
        default=None,
        help="Optional positive bound for one invocation; resume with the same command.",
    )
    parser.add_argument(
        "--parallel-profile",
        type=Path,
        help="Run the dependency-aware aggregate-concurrency scheduler.",
    )
    return parser


def _ordered_authorized_shards(control: dict, launch) -> tuple:
    authorized_ids = tuple(control.get("authorized_shard_ids", ()))
    if not authorized_ids or len(authorized_ids) != len(set(authorized_ids)):
        raise ValueError("execution control lacks unique authorized shard IDs")
    by_id = {item.shard_id: item for item in launch.plan.shards}
    if not set(authorized_ids) <= set(by_id):
        raise ValueError("execution control authorizes a shard outside the launch")
    ordered = tuple(
        item for item in launch.plan.shards if item.shard_id in authorized_ids
    )
    scope = control.get("execution_scope")
    if scope == "full_matrix":
        if (
            len(ordered) != launch.plan.shard_count
            or tuple(item.shard_id for item in ordered) != authorized_ids
            or control.get("external_frozen_shard_count") != 0
        ):
            raise ValueError("full-matrix authorization is not the exact launch")
    elif scope == "partial_shard_repair":
        if len(ordered) != 4 or control.get("external_frozen_shard_count") != 1:
            raise ValueError("partial repair authorization is incomplete")
    elif scope == "core_canary":
        if (
            launch.plan.kind != "portfolio-core-split-x5-launch-plan"
            or launch.plan.selected_splits != ("dev_mini",)
            or len(ordered) != 5
            or len({item.accepted_batch_id for item in ordered}) != 1
            or {item.config for item in ordered}
            != {"noskill", "llm_static", "s1", "s1s2", "full"}
            or len({item.query_ids for item in ordered}) != 1
            or tuple(item.shard_id for item in ordered) != authorized_ids
            or control.get("external_frozen_shard_count") != 0
        ):
            raise ValueError("Core canary authorization is not one 25-query x5 batch")
    elif scope == "static_opt_rollout":
        if (
            launch.plan.kind != "portfolio-core-static-opt-800x1-launch-plan"
            or launch.plan.execution_mode != "static_opt_rollout"
            or launch.plan.selected_splits != ("opt_pool",)
            or launch.plan.config_order != ("llm_static",)
            or launch.plan.query_count != 800
            or launch.plan.instance_count != 800
            or launch.plan.shard_count != 32
            or len(ordered) != 32
            or any(
                item.config != "llm_static" or item.query_count != 25
                for item in ordered
            )
            or len({query_id for item in ordered for query_id in item.query_ids}) != 800
            or tuple(item.shard_id for item in ordered) != authorized_ids
            or control.get("external_frozen_shard_count") != 0
            or control.get("assistant_checkpoint_schema_version") != 2
            or control.get("gcs_policy_version")
            != "portfolio-grounded-contract-success-v2"
            or control.get("gcs_scorer_evidence_policy_version")
            != "portfolio-gcs-scorer-evidence-v2"
        ):
            raise ValueError(
                "static opt authorization is not exact opt_pool × llm_static 800x1"
            )
    else:
        raise ValueError("execution control has an unsupported scope")
    return ordered


def _artifact_alias_source_shard(control: dict, shard: object, shards: tuple):
    aliases = control.get("execution_artifact_aliases", [])
    if not isinstance(aliases, list) or any(
        not isinstance(item, dict) for item in aliases
    ):
        raise ValueError("execution artifact aliases must be a list of objects")
    matches = tuple(
        item for item in aliases if item.get("target_config") == shard.config
    )
    if not matches:
        return None
    if len(matches) != 1 or matches[0].get("provider_model_call_count") != 0:
        raise ValueError("execution artifact alias is ambiguous or call-bearing")
    source_config = matches[0].get("source_config")
    candidates = tuple(
        item
        for item in shards
        if item.accepted_batch_id == shard.accepted_batch_id
        and item.config == source_config
    )
    if len(candidates) != 1 or candidates[0].query_ids != shard.query_ids:
        raise ValueError("execution artifact alias lacks its paired source shard")
    return candidates[0]


def _dependency_ordered_shards(control: dict, shards: tuple) -> tuple:
    """Keep launch batches stable while placing zero-call aliases after sources."""

    by_batch: dict[str, list] = {}
    for shard in shards:
        by_batch.setdefault(shard.accepted_batch_id, []).append(shard)
    ordered = []
    for batch_shards in by_batch.values():
        direct = [
            item
            for item in batch_shards
            if _artifact_alias_source_shard(control, item, shards) is None
        ]
        aliases = [
            item
            for item in batch_shards
            if _artifact_alias_source_shard(control, item, shards) is not None
        ]
        ordered.extend((*direct, *aliases))
    return tuple(ordered)


def validate_runtime_binding(
    control: dict,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Fail closed when a launch points at a runtime built from other code.

    The matrix launcher is the last safe place to detect source drift before
    it creates provider reservations.  In particular, an old launch must not
    be resumed against a working tree that contains a newer runner contract.
    """

    runtime_root = Path(control["runtime_root"])
    if not runtime_root.is_absolute():
        runtime_root = repository_root / runtime_root
    runtime_lock_path = runtime_root / "runtime-lock.json"
    runtime_lock_bytes = runtime_lock_path.read_bytes()
    expected_file_sha = control["runtime_lock_file_sha256"]
    actual_file_sha = sha256_bytes(runtime_lock_bytes)
    if actual_file_sha != expected_file_sha:
        raise ValueError(
            "runtime lock file hash mismatch: "
            f"expected {expected_file_sha}, got {actual_file_sha}"
        )
    runtime_lock = json.loads(runtime_lock_bytes.decode("utf-8"))
    source_files = {
        "runner_file_sha256": repository_root
        / "src"
        / "skillchain"
        / "runners"
        / "assistant.py",
        "shard_runner_file_sha256": repository_root
        / "scripts"
        / "run_portfolio_shard.py",
        "portfolio_execution_file_sha256": repository_root
        / "src"
        / "skillchain"
        / "evaluation"
        / "portfolio_execution.py",
        "llm_adapter_file_sha256": repository_root / "src" / "skillchain" / "llm.py",
    }
    mismatches = []
    for lock_key, source_path in source_files.items():
        expected = runtime_lock.get(lock_key)
        if expected is None:
            continue
        actual = sha256_bytes(source_path.read_bytes())
        if actual != expected:
            mismatches.append(f"{lock_key}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError("runtime binding drifted: " + "; ".join(mismatches))
    if control.get("execution_scope") == "static_opt_rollout":
        verified = load_verified_portfolio_static_opt_runtime(
            runtime_root,
            expected_runtime_lock_file_sha256=expected_file_sha,
        )
        validate_portfolio_static_opt_execution_control(control, verified)


def _run(command: tuple[str, ...]) -> int:
    environment = os.environ.copy()
    # Each shard imports numerical libraries during startup.  Without an
    # explicit one-thread BLAS setting, four short-lived workers can exhaust
    # the Windows process address space before any model call begins.
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(command, check=False, env=environment)
    return completed.returncode


def _shard_command(
    *,
    execution_root: Path,
    shard_id: str,
    execute: bool,
    parallel_profile: Path | None,
    legacy_launch_compat: bool = False,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_portfolio_shard.py"),
        "--execution-root",
        str(execution_root),
        "--shard-id",
        shard_id,
    ]
    if execute:
        command.append("--execute")
    if parallel_profile is not None:
        command.extend(("--parallel-profile", str(parallel_profile)))
    if legacy_launch_compat:
        command.append("--legacy-launch-compat")
    return tuple(command)


def _finalizer_command(
    *,
    execution_root: Path,
    shard_id: str,
    legacy_launch_compat: bool = False,
    artifact_alias_source_shard_id: str | None = None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "finalize_portfolio_shard.py"),
        "--execution-root",
        str(execution_root),
        "--shard-id",
        shard_id,
    ]
    if legacy_launch_compat:
        command.append("--legacy-launch-compat")
    if artifact_alias_source_shard_id is not None:
        command.extend(
            (
                "--artifact-alias-source-shard-id",
                artifact_alias_source_shard_id,
            )
        )
    return tuple(command)


def _run_parallel_stage(
    *,
    execution_root: Path,
    shards: tuple,
    profile: PortfolioParallelProfile,
    parallel_profile_path: Path,
    execute: bool,
    legacy_launch_compat: bool = False,
) -> dict[str, int]:
    """Run a ready DAG level concurrently, returning deterministic results."""

    if not shards:
        return {}
    results: dict[str, int] = {}
    worker_count = min(profile.worker_count, len(shards))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="portfolio-shard",
    ) as executor:
        futures = {
            executor.submit(
                _run,
                _shard_command(
                    execution_root=execution_root,
                    shard_id=shard.shard_id,
                    execute=execute,
                    parallel_profile=parallel_profile_path,
                    legacy_launch_compat=legacy_launch_compat,
                ),
            ): shard.shard_id
            for shard in shards
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return {shard.shard_id: results[shard.shard_id] for shard in shards}


def _finalize_parallel_stage(
    *,
    args: argparse.Namespace,
    control: dict,
    launch,
    stage: tuple,
    results: dict[str, int],
) -> tuple[object, int | None]:
    """Commit successful siblings before returning a stage failure.

    A worker failure is local to its shard.  Finalizing the successful siblings
    first makes the next invocation resume at the failed shard instead of
    replaying an already completed batch (and consuming another restart).
    """

    finalized: list[str] = []
    if args.execute:
        for shard in stage:
            if results[shard.shard_id] != 0:
                continue
            returncode = _run(
                _finalizer_command(
                    execution_root=args.execution_root,
                    shard_id=shard.shard_id,
                    legacy_launch_compat=bool(control.get("legacy_launch_compat")),
                )
            )
            if returncode != 0:
                return launch, returncode
            finalized.append(shard.shard_id)
        launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )

    failed = [
        {"shard_id": shard.shard_id, "returncode": results[shard.shard_id]}
        for shard in stage
        if results[shard.shard_id] != 0
    ]
    if failed:
        print(
            json.dumps(
                {
                    "status": "parallel_stage_incomplete",
                    "failed_shards": failed,
                    "finalized_shards": finalized,
                    "resume_policy": "retry_failed_shards_only",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return launch, int(failed[0]["returncode"])
    return launch, None


def _run_parallel_matrix(
    *,
    args: argparse.Namespace,
    control: dict,
    launch,
    shards: tuple,
    profile: PortfolioParallelProfile,
) -> int:
    """Run one accepted batch at a time with Full gated on S1+S2."""

    if control.get("execution_scope") == "static_opt_rollout":
        return _run_parallel_static_opt(
            args=args,
            control=control,
            launch=launch,
            shards=shards,
            profile=profile,
        )

    parallel_profile_path = args.parallel_profile.absolute()
    by_batch: dict[str, list] = {}
    for shard in shards:
        by_batch.setdefault(shard.accepted_batch_id, []).append(shard)
    processed = 0
    for batch_id, batch_items in by_batch.items():
        pending = tuple(
            item
            for item in batch_items
            if item.shard_id not in launch.state.completed_shard_ids
        )
        if not pending:
            continue
        if args.stop_after_shards is not None and processed >= args.stop_after_shards:
            break

        # Full consumes the S1+S2 shared-route artifacts.  The other four
        # configurations are independent and form the first ready DAG level.
        stage_one = tuple(item for item in pending if item.config != "full")
        stage_two = tuple(item for item in pending if item.config == "full")
        for stage in (stage_one, stage_two):
            if not stage:
                continue
            alias_pairs = tuple(
                (item, _artifact_alias_source_shard(control, item, shards))
                for item in stage
            )
            alias_pairs = tuple(
                (target, source) for target, source in alias_pairs if source is not None
            )
            if alias_pairs:
                if len(alias_pairs) != len(stage):
                    raise ValueError(
                        "parallel stage mixes direct and aliased execution shards"
                    )
                if args.execute:
                    for target, source in alias_pairs:
                        if source.shard_id not in launch.state.completed_shard_ids:
                            raise ValueError(
                                "artifact alias source shard is not complete"
                            )
                        returncode = _run(
                            _finalizer_command(
                                execution_root=args.execution_root,
                                shard_id=target.shard_id,
                                legacy_launch_compat=bool(
                                    control.get("legacy_launch_compat")
                                ),
                                artifact_alias_source_shard_id=source.shard_id,
                            )
                        )
                        if returncode != 0:
                            return returncode
                    launch = load_portfolio_launch_package(
                        control["launch_root"],
                        expected_plan_file_sha256=control["launch_plan_file_sha256"],
                    )
                processed += len(alias_pairs)
                continue
            results = _run_parallel_stage(
                execution_root=args.execution_root,
                shards=stage,
                profile=profile,
                parallel_profile_path=parallel_profile_path,
                execute=args.execute,
                legacy_launch_compat=bool(control.get("legacy_launch_compat")),
            )
            before_completed = len(launch.state.completed_shard_ids)
            launch, stage_error = _finalize_parallel_stage(
                args=args,
                control=control,
                launch=launch,
                stage=stage,
                results=results,
            )
            processed += max(
                0, len(launch.state.completed_shard_ids) - before_completed
            )
            if stage_error is not None:
                return stage_error
        print(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "status": "parallel_batch_committed",
                    "processed_shards": processed,
                    "parallel_profile_sha256": profile.profile_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    launch = load_portfolio_launch_package(
        control["launch_root"],
        expected_plan_file_sha256=control["launch_plan_file_sha256"],
    )
    print(
        json.dumps(
            {
                "execution_scope": control["execution_scope"],
                "launch_status": launch.state.status,
                "model_calls_performed": launch.state.model_calls_performed,
                "processed_shards": processed,
                "remaining_authorized_shards": sum(
                    item.shard_id not in launch.state.completed_shard_ids
                    for item in shards
                ),
                "parallel_profile_sha256": profile.profile_sha256,
                "status": (
                    "parallel_execution_progress"
                    if args.execute
                    else "parallel_dry_run_passed"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_parallel_static_opt(
    *,
    args: argparse.Namespace,
    control: dict,
    launch,
    shards: tuple,
    profile: PortfolioParallelProfile,
) -> int:
    """Run independent Static opt shards concurrently across generator batches."""

    pending = tuple(
        item for item in shards if item.shard_id not in launch.state.completed_shard_ids
    )
    if args.stop_after_shards is not None:
        pending = pending[: args.stop_after_shards]
    processed = 0
    profile_path = args.parallel_profile.absolute()
    for offset in range(0, len(pending), profile.worker_count):
        stage = pending[offset : offset + profile.worker_count]
        results = _run_parallel_stage(
            execution_root=args.execution_root,
            shards=stage,
            profile=profile,
            parallel_profile_path=profile_path,
            execute=args.execute,
            legacy_launch_compat=False,
        )
        before_completed = len(launch.state.completed_shard_ids)
        launch, stage_error = _finalize_parallel_stage(
            args=args,
            control=control,
            launch=launch,
            stage=stage,
            results=results,
        )
        processed += (
            max(0, len(launch.state.completed_shard_ids) - before_completed)
            if args.execute
            else len(stage)
        )
        if stage_error is not None:
            return stage_error
        print(
            json.dumps(
                {
                    "status": "parallel_static_stage_committed",
                    "processed_shards": processed,
                    "stage_shard_ids": [item.shard_id for item in stage],
                    "parallel_profile_sha256": profile.profile_sha256,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    launch = load_portfolio_launch_package(
        control["launch_root"],
        expected_plan_file_sha256=control["launch_plan_file_sha256"],
    )
    print(
        json.dumps(
            {
                "execution_scope": "static_opt_rollout",
                "launch_status": launch.state.status,
                "model_calls_performed": launch.state.model_calls_performed,
                "processed_shards": processed,
                "remaining_authorized_shards": sum(
                    item.shard_id not in launch.state.completed_shard_ids
                    for item in shards
                ),
                "parallel_profile_sha256": profile.profile_sha256,
                "status": (
                    "parallel_execution_progress"
                    if args.execute
                    else "parallel_dry_run_passed"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stop_after_shards is not None and args.stop_after_shards <= 0:
        print("portfolio-matrix: stop-after-shards must be positive", file=sys.stderr)
        return 2
    try:
        control = _load_control(args.execution_root)
        validate_runtime_binding(control)
        launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        shards = _ordered_authorized_shards(control, launch)
        profile = None
        if args.parallel_profile is not None:
            profile = load_parallel_profile(args.parallel_profile)
            validate_parallel_profile_sources(profile)
            if (
                profile.matrix_run_id != launch.plan.matrix_run_id
                or profile.launch_plan_sha256 != launch.plan.launch_plan_sha256
                or profile.execution_scope != control.get("execution_scope")
            ):
                raise ValueError(
                    "parallel profile does not bind this launch and execution scope"
                )
    except (OSError, TypeError, ValueError) as error:
        print(f"portfolio-matrix: {error}", file=sys.stderr)
        return 2

    if profile is not None:
        return _run_parallel_matrix(
            args=args,
            control=control,
            launch=launch,
            shards=shards,
            profile=profile,
        )

    processed = 0
    execution_shards = _dependency_ordered_shards(control, shards)
    for shard in execution_shards:
        launch = load_portfolio_launch_package(
            control["launch_root"],
            expected_plan_file_sha256=control["launch_plan_file_sha256"],
        )
        if shard.shard_id in launch.state.completed_shard_ids:
            continue
        if args.stop_after_shards is not None and processed >= args.stop_after_shards:
            break
        alias_source = _artifact_alias_source_shard(control, shard, shards)
        if alias_source is not None:
            if args.execute:
                if alias_source.shard_id not in launch.state.completed_shard_ids:
                    raise ValueError("artifact alias source shard is not complete")
                returncode = _run(
                    _finalizer_command(
                        execution_root=args.execution_root,
                        shard_id=shard.shard_id,
                        legacy_launch_compat=bool(control.get("legacy_launch_compat")),
                        artifact_alias_source_shard_id=alias_source.shard_id,
                    )
                )
                if returncode != 0:
                    return returncode
            processed += 1
            continue
        run_command = _shard_command(
            execution_root=args.execution_root,
            shard_id=shard.shard_id,
            execute=args.execute,
            parallel_profile=None,
            legacy_launch_compat=bool(control.get("legacy_launch_compat")),
        )
        returncode = _run(run_command)
        if returncode != 0:
            return returncode
        if args.execute:
            returncode = _run(
                (
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "finalize_portfolio_shard.py"),
                    "--execution-root",
                    str(args.execution_root),
                    "--shard-id",
                    shard.shard_id,
                    *(
                        ("--legacy-launch-compat",)
                        if control.get("legacy_launch_compat")
                        else ()
                    ),
                )
            )
            if returncode != 0:
                return returncode
        processed += 1

    launch = load_portfolio_launch_package(
        control["launch_root"],
        expected_plan_file_sha256=control["launch_plan_file_sha256"],
    )
    print(
        json.dumps(
            {
                "execution_scope": control["execution_scope"],
                "launch_status": launch.state.status,
                "model_calls_performed": launch.state.model_calls_performed,
                "processed_shards": processed,
                "remaining_authorized_shards": sum(
                    item.shard_id not in launch.state.completed_shard_ids
                    for item in shards
                ),
                "status": "execution_progress" if args.execute else "dry_run_passed",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
