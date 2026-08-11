"""Create a zero-call aggregate-concurrency overlay for a Portfolio execution root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.portfolio_parallel import (  # noqa: E402
    MAX_SAFE_ASSISTANT_CONCURRENCY,
    build_parallel_profile,
    qwen_minimum_start_interval_seconds,
    write_parallel_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument(
        "--assistant-concurrency",
        type=int,
        default=MAX_SAFE_ASSISTANT_CONCURRENCY,
    )
    parser.add_argument(
        "--final-judge-concurrency",
        type=int,
        default=None,
        help=("Defaults to 0 for static_opt_rollout and 8 for Judge-enabled scopes."),
    )
    parser.add_argument(
        "--qwen-rpm",
        type=int,
        default=40,
        help="Aggregate Qwen RPM cap, enforced with even request-start spacing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control_path = args.execution_root / "execution-control.json"
    control = json.loads(control_path.read_bytes())
    launch_root = Path(control["launch_root"])
    launch_plan = json.loads((launch_root / "launch-plan.json").read_bytes())
    final_judge_concurrency = args.final_judge_concurrency
    if final_judge_concurrency is None:
        final_judge_concurrency = (
            0 if control["execution_scope"] == "static_opt_rollout" else 8
        )
    profile = build_parallel_profile(
        matrix_run_id=launch_plan["matrix_run_id"],
        launch_plan_sha256=control["launch_plan_sha256"],
        worker_count=args.worker_count,
        assistant_concurrency=args.assistant_concurrency,
        final_judge_concurrency=final_judge_concurrency,
        qwen_requests_per_minute_cap=args.qwen_rpm,
        execution_scope=control["execution_scope"],
    )
    write_parallel_profile(args.output, profile)
    print(
        json.dumps(
            {
                "profile": str(args.output),
                "profile_sha256": profile.profile_sha256,
                "launch_plan_sha256": profile.launch_plan_sha256,
                "assistant_concurrency": profile.assistant_concurrency,
                "final_judge_concurrency": profile.final_judge_concurrency,
                "worker_count": profile.worker_count,
                "qwen_requests_per_minute_cap": profile.qwen_requests_per_minute_cap,
                "qwen_rate_limit_policy": profile.qwen_rate_limit_policy,
                "qwen_minimum_start_interval_seconds": (
                    qwen_minimum_start_interval_seconds(
                        profile.qwen_requests_per_minute_cap
                    )
                ),
                "status": "prepared_zero_call",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
