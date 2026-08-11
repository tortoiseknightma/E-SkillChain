"""Prepare the zero-provider S1 replay/body-gate runtime and populations."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for candidate in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from skillchain.evaluation.portfolio_launch import (  # noqa: E402
    load_portfolio_launch_package,
    reconstruct_verified_portfolio_core_inputs,
)
from skillchain.evaluation.portfolio_s1_experiment_runtime import (  # noqa: E402
    S1_BODY_GATE_SCOPE,
    S1_OPT_REPLAY_SCOPE,
    create_portfolio_s1_experiment_execution_control,
    create_portfolio_s1_experiment_launch_package,
    create_portfolio_s1_experiment_runtime,
    load_verified_portfolio_s1_experiment_runtime,
)


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("value must be an exact decimal") from error
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    runtime = commands.add_parser("runtime", help="derive the independent two-Bank runtime")
    runtime.add_argument("--parent-static-runtime-root", type=Path, required=True)
    runtime.add_argument("--parent-runtime-lock-file-sha256", required=True)
    runtime.add_argument("--creator-output-root", type=Path, required=True)
    runtime.add_argument(
        "--creator-invocation-receipt-file-sha256", required=True
    )
    runtime.add_argument(
        "--creator-model-invocation-receipt-file-sha256", required=True
    )
    runtime.add_argument("--candidate-bank-file-sha256", required=True)
    runtime.add_argument("--output-dir", type=Path, required=True)

    population = commands.add_parser(
        "population", help="prepare one exact GCS-only launch and execution control"
    )
    population.add_argument(
        "--scope", choices=(S1_OPT_REPLAY_SCOPE, S1_BODY_GATE_SCOPE), required=True
    )
    population.add_argument("--matrix-run-id", required=True)
    population.add_argument("--core-input-binding-launch-root", type=Path, required=True)
    population.add_argument(
        "--core-input-binding-launch-plan-file-sha256", required=True
    )
    population.add_argument("--runtime-root", type=Path, required=True)
    population.add_argument("--runtime-lock-file-sha256", required=True)
    population.add_argument("--replay-manifest", type=Path)
    population.add_argument("--replay-manifest-file-sha256")
    population.add_argument("--replay-mapping", type=Path)
    population.add_argument("--replay-mapping-file-sha256")
    population.add_argument("--validation-gates", type=Path)
    population.add_argument("--validation-gates-file-sha256")
    population.add_argument("--launch-output-dir", type=Path, required=True)
    population.add_argument("--execution-output-dir", type=Path, required=True)
    population.add_argument(
        "--approved-dashscope-budget-cny", type=_decimal, required=True
    )
    population.add_argument("--phase-cumulative-cap-cny", type=_decimal, required=True)
    population.add_argument(
        "--prior-dashscope-observed-cost-cny",
        type=_decimal,
        default=Decimal("0"),
    )
    return parser


def _prepare_runtime(args: argparse.Namespace) -> dict[str, object]:
    runtime = create_portfolio_s1_experiment_runtime(
        parent_static_runtime_root=args.parent_static_runtime_root,
        expected_parent_runtime_lock_file_sha256=(
            args.parent_runtime_lock_file_sha256
        ),
        creator_output_root=args.creator_output_root,
        expected_creator_invocation_receipt_file_sha256=(
            args.creator_invocation_receipt_file_sha256
        ),
        expected_creator_model_invocation_receipt_file_sha256=(
            args.creator_model_invocation_receipt_file_sha256
        ),
        expected_candidate_bank_file_sha256=args.candidate_bank_file_sha256,
        output_dir=args.output_dir,
    )
    return {
        "status": "verified_ready",
        "output_dir": str(runtime.root),
        "runtime_lock_file_sha256": runtime.runtime_lock_file_sha256,
        "runtime_lock_sha256": runtime.runtime_lock["runtime_lock_sha256"],
        "bank_sha256s": dict(runtime.runtime_lock["bank_sha256s"]),
        "provider_calls": 0,
    }


def _prepare_population(args: argparse.Namespace) -> dict[str, object]:
    source = load_portfolio_launch_package(
        args.core_input_binding_launch_root,
        expected_plan_file_sha256=(
            args.core_input_binding_launch_plan_file_sha256
        ),
    )
    inputs = reconstruct_verified_portfolio_core_inputs(source.plan)
    runtime = load_verified_portfolio_s1_experiment_runtime(
        args.runtime_root,
        expected_runtime_lock_file_sha256=args.runtime_lock_file_sha256,
    )
    launch = create_portfolio_s1_experiment_launch_package(
        inputs,
        runtime,
        execution_scope=args.scope,
        matrix_run_id=args.matrix_run_id,
        output_dir=args.launch_output_dir,
        replay_manifest_path=args.replay_manifest,
        expected_replay_manifest_file_sha256=args.replay_manifest_file_sha256,
        replay_mapping_path=args.replay_mapping,
        expected_replay_mapping_file_sha256=args.replay_mapping_file_sha256,
        validation_gate_path=args.validation_gates,
        expected_validation_gate_file_sha256=args.validation_gates_file_sha256,
        core_input_binding_launch_root=args.core_input_binding_launch_root,
        core_input_binding_launch_plan_file_sha256=(
            args.core_input_binding_launch_plan_file_sha256
        ),
    )
    control = create_portfolio_s1_experiment_execution_control(
        launch,
        runtime,
        output_dir=args.execution_output_dir,
        approved_dashscope_budget_cny=args.approved_dashscope_budget_cny,
        phase_cumulative_cap_cny=args.phase_cumulative_cap_cny,
        prior_dashscope_observed_cost_cny=args.prior_dashscope_observed_cost_cny,
    )
    return {
        "status": "prepared_not_started",
        "scope": args.scope,
        "launch_output_dir": str(launch.root),
        "launch_plan_file_sha256": launch.plan_file_sha256,
        "launch_plan_sha256": launch.plan["launch_plan_sha256"],
        "execution_output_dir": str(args.execution_output_dir.absolute()),
        "control_sha256": control["control_sha256"],
        "query_count": launch.plan["query_count"],
        "instance_count": len(launch.instances),
        "shard_count": len(launch.shards),
        "assistant_concurrency": control["assistant_concurrency"],
        "provider_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            _prepare_runtime(args)
            if args.command == "runtime"
            else _prepare_population(args)
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"prepare-portfolio-s1-experiment: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
