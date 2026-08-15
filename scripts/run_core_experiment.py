#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.evaluation.core_fast import (  # noqa: E402
    CoreFastEngine,
    FastPathError,
    load_core_fast_spec,
)
from skillchain.evaluation.core_fast.adapters import load_adapter  # noqa: E402


DEFAULT_SPEC = REPOSITORY_ROOT / "specs" / "core-experiment-fast-v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal-governance Core 1,500 experiment path."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="validate frozen inputs and runtime"
    )
    validate.add_argument(
        "--inputs-only",
        action="store_true",
        help="skip credential/command readiness checks",
    )
    commands.add_parser(
        "static-opt800",
        help="run a fresh create-only Static opt800 baseline and emit bootstrap metadata",
    )
    commands.add_parser(
        "s1-parent-opt800",
        help="run create-only opt800 under a SHA-bound accepted S1 parent",
    )
    commands.add_parser(
        "s2-parent-route800",
        help="run create-only route-only opt800 under a SHA-bound S2 parent",
    )
    commands.add_parser(
        "prepare-feedback-selection",
        help="freeze or verify discovery600 Feedback summary/selection without provider calls",
    )
    commands.add_parser(
        "prepare-s2-round",
        help="freeze or verify the 3-failure/3-success/3-regression S2 packet",
    )
    commands.add_parser(
        "s2-readiness",
        help="verify one frozen adaptive S2 round without provider calls",
    )
    run = commands.add_parser("run", help="run or conservatively resume the experiment")
    run.add_argument("--through", choices=("s1", "s2", "full", "test"), required=True)
    commands.add_parser("report", help="rebuild metrics/cases from completed results")
    return parser


def _root(args: argparse.Namespace, experiment_id: str) -> Path:
    return (
        args.output_root
        if args.output_root is not None
        else REPOSITORY_ROOT / "runs" / "core-fast" / experiment_id
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec_path = args.spec.resolve()
        spec = load_core_fast_spec(spec_path)
        adapter = load_adapter(spec, cwd=REPOSITORY_ROOT, spec_path=spec_path)
        engine = CoreFastEngine(
            spec=spec,
            spec_path=spec_path,
            output_root=_root(args, spec.experiment_id),
            adapter=adapter,
        )
        if args.command == "validate":
            result = engine.validate(require_runtime=not args.inputs_only)
            print(
                json.dumps(
                    result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "run":
            engine.run(through=args.through)
            print(
                json.dumps(
                    {
                        "status": "complete_through_stage",
                        "through": args.through,
                        "output_root": str(engine.output_root),
                        "observed_cost_cny": engine.calls.observed_cost(),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "static-opt800":
            engine.initialize_static_opt800()
            result = engine.run_static_opt800()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "s1-parent-opt800":
            engine.initialize_s1_parent_opt800()
            result = engine.run_s1_parent_opt800()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "s2-parent-route800":
            engine.initialize_s2_parent_route800()
            result = engine.run_s2_parent_route800()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-feedback-selection":
            result = engine.prepare_feedback_selection()
            manifest = result["selection_manifest"]
            population_count = manifest.get(
                "discovery_population_count", manifest.get("population_count")
            )
            if not isinstance(population_count, int):
                raise FastPathError(
                    "Feedback selection manifest lacks its population count"
                )
            print(
                json.dumps(
                    {
                        "status": "feedback_selection_prepared",
                        "output_root": str(engine.output_root),
                        "discovery_count": population_count,
                        "replay_count": 200,
                        "feedback_count": manifest["effective_count"],
                        "selection_manifest_sha256": manifest["manifest_sha256"],
                        "provider_calls": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "prepare-s2-round":
            engine.initialize()
            result = engine.prepare_s2_round()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "s2-readiness":
            result = engine.s2_readiness()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        report = engine.report()
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (FastPathError, ValueError, RuntimeError) as error:
        print(f"Core Fast Path error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
